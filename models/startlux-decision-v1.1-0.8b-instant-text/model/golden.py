"""Minimal official CPU FP32 golden for the text pipeline.

The independent reference stage: loads the SAME weights folder
(STARTLUX_MODEL_DIR) as an official ``Qwen3_5ForConditionalGeneration`` float32
CPU model, applies the vendored official bidirectional attention patch (the
text trunk's 6 full-attention layers are bidirectional), and scores every case
of ``work/inputs`` -- ``input_ids`` rows are read straight from the npy pool,
no re-tokenization.  B=1 is pinned by the graph contract: the laid-out
width IS the row's true length (the case format cannot express padding),
so the reference runs each row at its full laid-out length, exactly the
official unpadded semantics.  The 26 letter columns come from the folder's
official ``decision_config.json``; the head rows are the original (tied)
output embedding.

Output (the per-case npy layout ``tools/compare.py`` reads): ``work/golden/
<case_id>/output_0.npy`` ([1, 26] float32) + ``work/golden/result.json`` whose
``cases`` index mirrors the ``ge_execute`` infer result, so compare.py
dispatches the layout from the ``cases``/``outputs`` keys.

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

# The runtime names outputs by position (runtime/main.cpp:245: output_<i>; the
# GE semantic name is not propagated) and tools/compare.py aligns by name.
OUTPUT_NAME = "output_0"


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
    """cases.json + <case>/input_ids.npy -> [(case_id, input_ids int64 [1, S])]."""
    index = json.loads((inputs / "cases.json").read_text())
    cases = []
    for case in index["cases"]:
        path = {item["name"]: item["path"] for item in case["inputs"]}["input_ids"]
        cases.append((case["id"], torch.from_numpy(np.load(inputs / path))))
    return cases


def main():
    parser = argparse.ArgumentParser(description="Official CPU FP32 golden "
                                                 "(work/golden)")
    parser.add_argument("--inputs-dir", type=Path,
                        default=ADAPTER_DIR / "work" / "inputs",
                        help="directory holding cases.json and the per-case npy dirs "
                             "(default: %(default)s)")
    parser.add_argument("--output-dir", type=Path,
                        default=ADAPTER_DIR / "work" / "golden",
                        help="directory for the per-case npy dirs and result.json "
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
    (args.output_dir / "outputs.bin").unlink(missing_ok=True)   # 旧 ge-compare/1 blob
    result = {"status": "success", "cases": []}
    for case_id, ids in cases:
        with torch.inference_mode():
            hidden = bidirectional.text_forward(body, ids, attention_mask=None)
            last = hidden[0, -1].float()
            scores = last @ head_rows.transpose(0, 1)
        row = np.ascontiguousarray(scores.detach().numpy(), dtype=np.float32)
        tensor = row.reshape(1, -1)   # one captured row per case ([1, 26])
        relative = f"{case_id}/{OUTPUT_NAME}.npy"
        (args.output_dir / case_id).mkdir(parents=True, exist_ok=True)   # np.save 不建父目录
        np.save(args.output_dir / relative, tensor)
        result["cases"].append({"id": case_id, "outputs": [{
            "name": OUTPUT_NAME, "dtype": str(tensor.dtype),
            "shape": list(tensor.shape), "path": relative,
            "bytes": tensor.nbytes}]})
        print(f"golden: {case_id} S={int(ids.shape[1])} scores {list(tensor.shape)}")

    (args.output_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"golden: {len(result['cases'])} cases; output: {args.output_dir}")


if __name__ == "__main__":
    main()
