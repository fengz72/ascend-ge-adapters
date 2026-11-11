"""Standalone official CPU FP32 golden for the dev.txt bypass route.

The reference stage of the dev bypass flow: mirrors decider/golden.py over
the dev case pool (work/devcase/inputs) with ZERO code duplication of the
official forward -- it imports the official flow's OfficialReference
unchanged -- and writes the same compare.py-compatible result layout under
work/devcase/golden.  Only the input/output locations differ; the official
flow's own work/golden is untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root
DEV_WORK = MODEL_DIR / "work" / "devcase"
INPUTS = DEV_WORK / "inputs"
GOLDEN = DEV_WORK / "golden"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from model import modeling_multimodal as multimodal                  # noqa: E402
from decider.golden import OfficialReference                          # noqa: E402

OUTPUT_NAME = "scores"


def _load_saved_rows(case_id, samples):
    """The data stage's saved per-row official encodings, as torch rows."""
    path = INPUTS / "rows" / f"{case_id}.npz"
    if not path.is_file():
        raise ValueError(f"saved preprocessing for case {case_id!r} is missing: {path}; "
                         "run generate_dev_inputs.py first")
    rows = []
    with np.load(path, allow_pickle=False) as archive:
        for index in range(samples):
            if f"row{index}_input_ids" not in archive.files:
                raise ValueError(f"{path} misses row {index} input_ids")
            row = {"input_ids": torch.from_numpy(archive[f"row{index}_input_ids"])}
            for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
                name = f"row{index}_{key}"
                if name in archive.files:
                    row[key] = torch.from_numpy(archive[name])
            rows.append(row)
    return rows


def main():
    folder = multimodal.source_folder()
    index = json.loads((INPUTS / "cases.json").read_text())
    if index.get("schema") != "startlux-dev-cases/1":
        raise SystemExit("golden_dev: work/devcase/inputs/cases.json is not the dev case "
                         "index; run generate_dev_inputs.py first")
    precision_ids = index["pools"]["precision"]
    cases = index["cases"]

    from transformers import Qwen3_5ForConditionalGeneration

    source = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(folder), dtype=torch.float32, local_files_only=True)
    source = source.to(device="cpu", dtype=torch.float32).eval()
    reference = OfficialReference(source, folder)

    GOLDEN.mkdir(parents=True, exist_ok=True)
    result = {"status": "success", "mode": "infer", "backend": "cpu_golden_dev",
              "outputs": {}}
    total_rows = 0
    with (GOLDEN / "outputs.bin").open("wb") as stream:
        for case_id in precision_ids:
            samples = cases[case_id]["samples"]
            rows = _load_saved_rows(case_id, samples)
            scores = [reference.row_scores(row) for row in rows]
            tensor = torch.stack(scores, dim=0).detach().cpu().numpy()
            tensor = np.ascontiguousarray(tensor, dtype=np.float32)
            data = tensor.tobytes(order="C")
            result["outputs"][case_id] = [{"instance": 0, "tensors": [{
                "name": OUTPUT_NAME, "dtype": "float32", "shape": list(tensor.shape),
                "offset": stream.tell(), "bytes": len(data)}]}]
            stream.write(data)
            total_rows += samples
            print(f"golden_dev: {case_id} S={int(rows[0]['input_ids'].shape[1])} "
                  f"scores {list(tensor.shape)}")

    (GOLDEN / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"golden_dev: {len(result['outputs'])} cases, {total_rows} rows; output: {GOLDEN}")


if __name__ == "__main__":
    main()
