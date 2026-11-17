"""Official CPU FP32 golden for the OneJev1.6-0.8B multimodal graph.

The independent reference stage: loads the SAME weights folder
(ONEJEV_MODEL_DIR) as an official ``Qwen3_5ForConditionalGeneration`` float32
CPU model and scores every case of ``work/inputs``.  Per case it reads exactly
three tensors from the packed pool -- ``input_ids``, ``pixel_values``
(bf16 -> fp32) and ``image_grid_thw`` -- and lets the official model do the
rest itself: vision interpolation / position ids / cu_seqlens, MRoPE, the
embedding scatter and the tied output head.  The other six tensors of the
9-input graph contract are the graph-side precomputations of those same
quantities; the reference deliberately does not read them, so its result stays
independent of the input generator's math.  ``mm_token_type_ids`` is derived
from ``input_ids`` here (it is part of the official preprocessing, not one of
the packed inputs).

``scores`` is ``logits[0, -1, slot_token_ids]``: the final-token hidden state
times the 26 tied lm_head rows for the A..Z answer slots (slot ids from
``records.json``, re-derived with the tokenizer and checked).  B=1 is pinned by
the graph contract: the laid-out width IS the row's true length.

Output (the ge-compare/1 layout ``tools/compare.py`` reads): ``work/golden/
outputs.bin`` (concatenated [1, 26] float32 score tensors) + ``work/golden/
result.json`` with the outputs index.

This stage needs no NPU, no LD_PRELOAD and no model-package import; its only
inputs are the files written by generate_inputs.py.  Paths default to
work/inputs and work/golden; ``--inputs-dir`` / ``--output-dir`` override them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The CPU thread pins must be set BEFORE torch imports: the OpenMP pool is
# sized at library init from OMP_NUM_THREADS, and an unpinned pool (one
# thread per core) makes every small parallel region pay pool-sized sync
# overhead.  setdefault respects a user override.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

ADAPTER_DIR = Path(__file__).resolve().parents[1]
DEPS_DIR = ADAPTER_DIR / "work" / "deps" / "py"
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

import transformers

if transformers.__version__ != "5.17.0":
    raise RuntimeError(f"OneJev needs the task-local transformers 5.17.0 from {DEPS_DIR}, "
                       f"got {transformers.__version__} from {transformers.__file__}")

import numpy as np
import torch

OUTPUT_NAME = "scores"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def weights_dir():
    """ONEJEV_MODEL_DIR resolved once, loudly."""
    value = os.environ.get("ONEJEV_MODEL_DIR")
    if not value or not value.strip():
        raise ValueError("ONEJEV_MODEL_DIR is not set or is empty")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"ONEJEV_MODEL_DIR {path} is not a directory")
    return path


def ensure_weights(folder):
    """The checkpoint must exist and its safetensors header must parse."""
    # A download.log is an artifact of the materials fetcher, not part of the
    # HF snapshot; honor it when present (an unfinished fetch must not load),
    # but never require it -- the folder may come from the user's own download.
    log = folder / "download.log"
    if log.is_file() and "DONE" not in log.read_text(encoding="utf-8"):
        raise SystemExit(f"golden: {log} has no DONE line; the checkpoint is still "
                         "downloading. Wait for the download to finish, then rerun.")
    weights = folder / "model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"golden: {weights} does not exist; the checkpoint is missing "
                         "or still downloading")
    from safetensors import safe_open
    try:
        with safe_open(str(weights), framework="pt") as handle:
            if not handle.keys():
                raise ValueError("the safetensors header carries no tensors")
    except Exception as error:
        raise SystemExit(f"golden: {weights} has an unreadable safetensors header "
                         f"({error}); the download is incomplete or corrupt") from error


def letter_ids(tokenizer):
    """The 26 A..Z slot ids, re-derived from the tokenizer (single round-trip each)."""
    ids = []
    for label in LETTERS:
        enc = tokenizer.encode(label, add_special_tokens=False)
        if len(enc) != 1 or tokenizer.decode(enc) != label:
            raise SystemExit(f"golden: answer slot {label!r} is not a single round-trip token")
        ids.append(enc[0])
    if len(set(ids)) != 26:
        raise SystemExit("golden: the A..Z answer slot tokens collide")
    return ids


def _tensor_ref(case, name):
    for ref in case["tensors"]:
        if ref["name"] == name:
            return ref
    raise SystemExit(f"golden: case {case['id']} misses tensor {name}")


def _load_cases(inputs):
    """records.json + inputs.bin -> [(case_id, ids, pixel_values, grid)], loudly.

    Only the three official-forward inputs are materialised; the six graph-side
    precomputations stay unread on purpose.
    """
    records = json.loads((inputs / "records.json").read_text())
    if records.get("schema") != "ge-records/3":
        raise SystemExit(f"golden: {inputs / 'records.json'} is not the graph record index "
                         "(ge-records/3); run generate_inputs.py first")
    blob_path = Path(records.get("file", inputs / "inputs.bin"))
    if not blob_path.is_absolute():
        blob_path = inputs / blob_path
    if not blob_path.is_file():
        raise SystemExit(f"golden: {blob_path} does not exist; regenerate the inputs dir")
    blob = blob_path.read_bytes()
    cases = []
    for case in records["cases"]:
        refs = {name: _tensor_ref(case, name)
                for name in ("input_ids", "pixel_values", "image_grid_thw")}
        for ref in refs.values():
            if ref["offset"] < 0 or ref["bytes"] < 0 or ref["offset"] + ref["bytes"] > len(blob):
                raise SystemExit(f"golden: case {case['id']} tensor {ref['name']} exceeds "
                                 f"{blob_path}; regenerate the inputs dir")
        ids_ref, pix_ref, grid_ref = (refs["input_ids"], refs["pixel_values"],
                                      refs["image_grid_thw"])
        if ids_ref["dtype"] != "int64" or tuple(ids_ref["shape"][:1]) != (1,) \
                or len(ids_ref["shape"]) != 2:
            raise SystemExit(f"golden: case {case['id']} input_ids dtype/shape "
                             f"{ids_ref['dtype']} {ids_ref['shape']} is not int64 [1, S]")
        if pix_ref["dtype"] != "bfloat16" or len(pix_ref["shape"]) != 2 \
                or pix_ref["shape"][1] != 1536:
            raise SystemExit(f"golden: case {case['id']} pixel_values dtype/shape "
                             f"{pix_ref['dtype']} {pix_ref['shape']} is not bfloat16 [P, 1536]")
        if grid_ref["dtype"] != "int64" or tuple(grid_ref["shape"]) != (1, 3):
            raise SystemExit(f"golden: case {case['id']} image_grid_thw dtype/shape "
                             f"{grid_ref['dtype']} {grid_ref['shape']} is not int64 [1, 3]")
        length = int(ids_ref["shape"][1])
        patches = int(pix_ref["shape"][0])
        ids = np.frombuffer(blob, np.int64, count=length, offset=ids_ref["offset"]).copy()
        payload = blob[pix_ref["offset"]:pix_ref["offset"] + pix_ref["bytes"]]
        raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        pixels = raw.view(torch.bfloat16).reshape(pix_ref["shape"]).to(torch.float32)
        grid = np.frombuffer(blob, np.int64, count=3, offset=grid_ref["offset"]).copy()
        if int(grid.prod()) != patches or int(grid[0]) != 1:
            raise SystemExit(f"golden: case {case['id']} grid {grid.tolist()} does not cover "
                             f"{patches} pixel patches")
        cases.append((case["id"], torch.from_numpy(ids.reshape(1, length)), pixels,
                      torch.from_numpy(grid.reshape(1, 3))))
    if not cases:
        raise SystemExit(f"golden: no cases in {inputs / 'records.json'}")
    return cases, records


def main():
    parser = argparse.ArgumentParser(description="Official CPU FP32 golden "
                                                 "(work/golden)")
    parser.add_argument("--inputs-dir", type=Path,
                        default=ADAPTER_DIR / "work" / "inputs",
                        help="directory holding inputs.bin and records.json "
                             "(default: %(default)s)")
    parser.add_argument("--output-dir", type=Path,
                        default=ADAPTER_DIR / "work" / "golden",
                        help="directory for outputs.bin and result.json "
                             "(default: %(default)s)")
    args = parser.parse_args()

    folder = weights_dir()
    ensure_weights(folder)
    cases, records = _load_cases(args.inputs_dir)
    slots = records.get("slot_token_ids")
    if not isinstance(slots, list) or len(slots) != 26 \
            or any(isinstance(i, bool) or not isinstance(i, int) for i in slots):
        raise SystemExit("golden: records.json slot_token_ids must be 26 integer ids")

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    derived = letter_ids(tokenizer)
    if derived != slots:
        raise SystemExit(f"golden: tokenizer slot ids {derived} disagree with records.json "
                         f"slot_token_ids {slots}")

    source = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(folder), dtype=torch.float32, local_files_only=True)
    source = source.to(device="cpu", dtype=torch.float32).eval()
    image_token_id = source.config.image_token_id
    slot_index = torch.tensor(slots, dtype=torch.long)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"status": "success", "mode": "infer", "backend": "cpu_golden",
              "outputs": {}}
    with (args.output_dir / "outputs.bin").open("wb") as stream:
        for case_id, ids, pixels, grid in cases:
            started = time.perf_counter()
            mm_token_type_ids = (ids == image_token_id).to(torch.int32)
            with torch.inference_mode():
                output = source(input_ids=ids, pixel_values=pixels,
                                image_grid_thw=grid,
                                mm_token_type_ids=mm_token_type_ids,
                                use_cache=False)
                scores = output.logits[0, -1].index_select(0, slot_index).float()
            row = np.ascontiguousarray(scores.detach().numpy(), dtype=np.float32)
            tensor = row.reshape(1, -1)   # one captured row per case ([1, 26])
            data = tensor.tobytes(order="C")
            result["outputs"][case_id] = [{"instance": 0, "tensors": [{
                "name": OUTPUT_NAME, "dtype": "float32", "shape": list(tensor.shape),
                "offset": stream.tell(), "bytes": len(data)}]}]
            stream.write(data)
            elapsed = time.perf_counter() - started
            print(f"golden: {case_id} S={int(ids.shape[1])} scores {list(tensor.shape)} "
                  f"{elapsed:.1f}s")

    (args.output_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"golden: {len(result['outputs'])} cases; output: {args.output_dir}")


if __name__ == "__main__":
    main()
