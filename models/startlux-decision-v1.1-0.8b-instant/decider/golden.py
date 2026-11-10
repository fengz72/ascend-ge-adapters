"""Standalone official CPU FP32 golden for the StartLux unified ONE-AIR route.

The reference stage of the standalone shell flow (docs/standalone-runtime-plan.md):
no ``core``, no ``model.py`` adaptation code, no NPU, no LD_PRELOAD.  It loads
the SAME weights folder (STARTLUX_MODEL_DIR) as an independent CPU float32
official model and consumes the SAVED official preprocessing results of the
data stage (work/inputs/rows/<case>.npz, the per-row TRUE-length encodings), so
nothing is re-tokenized or re-preprocessed here.

Official recipe per row: the original HF get_image_features -> official
get_placeholder_mask + masked_scatter injection -> the official get_rope_index
planes of the REAL loaded model -> the vendored official bidirectional
text_forward -> the ORIGINAL embedding head (get_output_embeddings().weight,
tied).  Every batch row runs UNPADDED at its true length; image features keep
their official order and usage positions.  The 26 letter columns come from the
model folder's official decision_config.json, never from fixture provenance.

Output (isomorphic to the runtime infer result, so the comparison tool treats
both sides uniformly): work/golden/outputs.bin (the concatenated [B, 26]
float32 score tensors) + work/golden/result.json with the top-level
status/mode/backend fields and the outputs index
case -> [{instance: 0, tensors: [{name, dtype, shape, offset, bytes}]}].
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root
WORK = MODEL_DIR / "work"
INPUTS = WORK / "inputs"
GOLDEN = WORK / "golden"

OUTPUT_NAME = "scores"


def _load_module(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(MODEL_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


base = _load_module("startlux_npu_adapter_for_native", "model/startlux_npu_adapter.py")


def _source_ref():
    value = os.environ.get("STARTLUX_MODEL_DIR")
    if not value or not value.strip():
        raise SystemExit("golden: STARTLUX_MODEL_DIR is not set or is empty")
    path = Path(value).resolve()
    if not path.is_dir():
        raise SystemExit(f"golden: STARTLUX_MODEL_DIR {path} is not a directory")
    return path


class OfficialReference:
    """The independent official CPU FP32 oracle."""

    def __init__(self, source, folder):
        self.letters26 = base._decision_config(folder)["letter_token_ids"]
        self.inner = source.model
        self.body = base._model_body(source)
        self.head = source.get_output_embeddings().weight
        if str(MODEL_DIR / "decider" / "vendor") not in sys.path:
            sys.path.insert(0, str(MODEL_DIR / "decider" / "vendor"))
        import startlux_decide.backends.bidirectional as bidirectional

        self.bidirectional = bidirectional
        self.bidirectional.enable(self.body)

    def row_scores(self, row):
        """One official request row (TRUE length, never padded) -> 26 letter logits."""
        ids = row["input_ids"]
        with torch.inference_mode():
            embeds = self.inner.get_input_embeddings()(ids)
            grid = row.get("image_grid_thw")
            if grid is not None:
                feats = self.inner.get_image_features(
                    row["pixel_values"], grid, return_dict=True).pooler_output
                feats = torch.cat(list(feats), 0).to(embeds.device, embeds.dtype)
                mask, _ = self.inner.get_placeholder_mask(ids, inputs_embeds=embeds,
                                                          image_features=feats)
                embeds = embeds.masked_scatter(mask, feats)
                positions, _deltas = self.inner.get_rope_index(
                    ids, image_grid_thw=grid, attention_mask=None,
                    mm_token_type_ids=row.get("mm_token_type_ids"))
                hidden = self.bidirectional.text_forward(self.body, inputs_embeds=embeds,
                                                         position_ids=positions)
            else:
                hidden = self.bidirectional.text_forward(self.body, ids, attention_mask=None)
            last = hidden[0, -1].float()
            head_rows = self.head.index_select(0, torch.tensor(self.letters26)).float()
            return last @ head_rows.transpose(0, 1)


def _load_saved_rows(case_id, samples):
    """The data stage's saved per-row official encodings, as torch rows."""
    path = INPUTS / "rows" / f"{case_id}.npz"
    if not path.is_file():
        raise ValueError(f"saved preprocessing for case {case_id!r} is missing: {path}; "
                         "run generate_inputs.py first")
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
    folder = _source_ref()
    index = json.loads((INPUTS / "cases.json").read_text())
    if index.get("schema") != "startlux-standalone-cases/1":
        raise SystemExit("golden: work/inputs/cases.json is not the standalone case index; "
                         "run generate_inputs.py first")
    precision_ids = index["pools"]["precision"]
    cases = index["cases"]

    from transformers import Qwen3_5ForConditionalGeneration

    source = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(folder), dtype=torch.float32, local_files_only=True)
    source = source.to(device="cpu", dtype=torch.float32).eval()
    reference = OfficialReference(source, folder)

    GOLDEN.mkdir(parents=True, exist_ok=True)
    result = {"status": "success", "mode": "infer", "backend": "ge_session",
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

    (GOLDEN / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"golden: {len(result['outputs'])} cases, {total_rows} rows, "
          f"letters {reference.letters26[:3]}...; output: {GOLDEN}")


if __name__ == "__main__":
    main()
