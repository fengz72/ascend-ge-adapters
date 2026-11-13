"""Minimal official CPU FP32 golden for the text pipeline.

The independent reference stage: loads the SAME weights folder
(STARTLUX_MODEL_DIR) as an official ``Qwen3_5ForConditionalGeneration`` float32
CPU model, applies the vendored official bidirectional attention patch (the
text trunk's 6 full-attention layers are bidirectional), and scores every case
of ``work/inputs`` -- ``input_ids`` rows are read straight from the packed
pool, no re-tokenization.  B=1 is pinned by the graph contract: the laid-out
width IS the row's true length (the records format cannot express padding),
so the reference runs each row at its full laid-out length, exactly the
official unpadded semantics.  The 26 letter columns come from the folder's
official ``decision_config.json``; the head rows are the original (tied)
output embedding.

Output (the ge-compare/1 layout ``tools/compare.py`` reads): ``work/golden/
outputs.bin`` (concatenated [B, 26] float32 score tensors) + ``work/golden/
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
from pathlib import Path

# The CPU thread pins must be set BEFORE torch imports: the OpenMP pool is
# sized at library init from OMP_NUM_THREADS, and an unpinned pool (one
# thread per core) makes every small parallel region pay pool-sized sync
# overhead -- on a many-core host that alone slows the reference ~40x.  The
# parent flow pins this via its env bootstrap; this minimal flow keeps the
# behavior self-contained in-script (setdefault respects a user override).
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import torch

ADAPTER_DIR = Path(__file__).resolve().parents[1]

OUTPUT_NAME = "scores"


def weights_dir():
    value = os.environ.get("STARTLUX_MODEL_DIR")
    if not value or not value.strip():
        raise ValueError("STARTLUX_MODEL_DIR is not set or is empty")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"STARTLUX_MODEL_DIR {path} is not a directory")
    return path


def letter_ids(folder):
    letters = json.loads((Path(folder) / "decision_config.json").read_text())["letter_token_ids"]
    if len(letters) != 26 or len(set(letters)) != 26:
        raise ValueError("decision_config letter_token_ids must be 26 unique ids")
    return letters


def _load_cases(inputs):
    """records.json + inputs.bin -> [(case_id, input_ids int64 [1, S])], loudly."""
    records = json.loads((inputs / "records.json").read_text())
    if records.get("schema") != "startlux-text-records/2":
        raise SystemExit(f"golden: {inputs / 'records.json'} is not the text record "
                         "index (startlux-text-records/2); run generate_inputs.py first")
    blob = (inputs / "inputs.bin").read_bytes()
    cases = []
    for case in records["cases"]:
        tensors = {t["name"]: t for t in case["text_tensors"]}
        try:
            ids_ref, pos_ref = (tensors[name] for name in
                                ("input_ids", "position_ids"))
        except KeyError as error:
            raise SystemExit(f"golden: case {case['id']} misses tensor {error}")
        for ref in (ids_ref, pos_ref):
            if ref["offset"] + ref["bytes"] > len(blob):
                raise SystemExit(f"golden: case {case['id']} tensor {ref['name']} "
                                 "exceeds inputs.bin; regenerate the inputs dir")
        if ids_ref["dtype"] != "int64" or pos_ref["dtype"] != "int64":
            raise SystemExit(f"golden: case {case['id']} carries unexpected dtypes "
                             f"({ids_ref['dtype']}/{pos_ref['dtype']})")
        # B=1 pinned by the graph contract: the laid-out width IS the row's
        # true length -- the records format cannot express padding.
        shape = tuple(ids_ref["shape"])
        if len(shape) != 2 or shape[0] != 1 or shape[1] < 1 \
                or tuple(pos_ref["shape"]) != (3, 1, shape[1]):
            raise SystemExit(f"golden: case {case['id']} shapes violate the B=1 "
                             f"true-length contract (ids {ids_ref['shape']}, "
                             f"positions {pos_ref['shape']})")
        length = int(shape[1])
        ids = np.frombuffer(blob, np.int64, count=length, offset=ids_ref["offset"])
        cases.append((case["id"], torch.from_numpy(ids.reshape(1, length))))
    if not cases:
        raise SystemExit(f"golden: no cases in {inputs / 'records.json'}")
    return cases


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
    letters = letter_ids(folder)
    cases = _load_cases(args.inputs_dir)

    from transformers import Qwen3_5ForConditionalGeneration

    source = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(folder), dtype=torch.float32, local_files_only=True)
    source = source.to(device="cpu", dtype=torch.float32).eval()
    body = source.model.language_model
    head = source.get_output_embeddings().weight

    if str(ADAPTER_DIR / "vendor") not in sys.path:
        sys.path.insert(0, str(ADAPTER_DIR / "vendor"))
    import startlux_decide.backends.bidirectional as bidirectional

    bidirectional.enable(body)
    head_rows = head.index_select(0, torch.tensor(letters)).float()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {"status": "success", "mode": "infer", "backend": "cpu_golden",
              "outputs": {}}
    with (args.output_dir / "outputs.bin").open("wb") as stream:
        for case_id, ids in cases:
            with torch.inference_mode():
                hidden = bidirectional.text_forward(body, ids, attention_mask=None)
                last = hidden[0, -1].float()
                scores = last @ head_rows.transpose(0, 1)
            row = np.ascontiguousarray(scores.detach().numpy(), dtype=np.float32)
            tensor = row.reshape(1, -1)   # one captured row per case ([1, 26])
            data = tensor.tobytes(order="C")
            result["outputs"][case_id] = [{"instance": 0, "tensors": [{
                "name": OUTPUT_NAME, "dtype": "float32", "shape": list(tensor.shape),
                "offset": stream.tell(), "bytes": len(data)}]}]
            stream.write(data)
            print(f"golden: {case_id} S={int(ids.shape[1])} scores {list(tensor.shape)}")

    (args.output_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"golden: {len(result['outputs'])} cases; output: {args.output_dir}")


if __name__ == "__main__":
    main()
