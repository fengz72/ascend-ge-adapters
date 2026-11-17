"""Cross-check the 9-input graph contract against one official whole-model forward.

For one case, run the official CPU model twice:

A. the reference path -- ``Qwen3_5ForConditionalGeneration`` forward with
   input_ids / pixel_values + the grid metadata (the same call golden.py makes);
B. the graph path -- the vision tower fed the precomputed ``interp_indices``,
   ``interp_weights`` and ``vision_position_ids`` from
   inputs.bin, then ``pooler_output`` scattered into the token embeddings by
   ``feature_index``, then ``language_model(inputs_embeds, position_ids)`` with
   the packed ``position_ids``, then the last token against the A..Z tied
   lm_head rows.

A cosine of at least 0.999 on the [1, 26] scores means the nine packed tensors
are semantically equivalent to the official internal computation.

Usage:
    ONEJEV_MODEL_DIR=... python3 scripts/verify/contract_crosscheck.py [case-id]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ADAPTER / "work" / "deps" / "py"))

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import torch
from transformers import Qwen3_5ForConditionalGeneration

INPUTS = Path(os.environ.get("ONEJEV_INPUTS_DIR", ADAPTER / "work" / "inputs"))
GOLDEN = Path(os.environ.get("ONEJEV_GOLDEN_DIR", ADAPTER / "work" / "golden"))
CASE = sys.argv[1] if len(sys.argv) > 1 else "onejev-00"


def main():
    folder = os.environ.get("ONEJEV_MODEL_DIR")
    if not folder:
        raise SystemExit("contract_crosscheck: set ONEJEV_MODEL_DIR")
    records = json.loads((INPUTS / "records.json").read_text())
    blob = Path(records["file"]).read_bytes()
    golden = json.loads((GOLDEN / "result.json").read_text())

    case = next(c for c in records["cases"] if c["id"] == CASE)
    refs = {t["name"]: t for t in case["tensors"]}

    def np_tensor(name, dtype):
        ref = refs[name]
        return np.frombuffer(blob, dtype, count=int(np.prod(ref["shape"])),
                             offset=ref["offset"]).reshape(ref["shape"])

    ids = torch.from_numpy(np_tensor("input_ids", np.int64).copy())
    grid = torch.tensor([case["grid"]], dtype=torch.int64)   # records metadata
    position_ids = torch.from_numpy(np_tensor("position_ids", np.int64).copy())
    interp_indices = torch.from_numpy(np_tensor("interp_indices", np.int64).copy())
    interp_weights = torch.from_numpy(np_tensor("interp_weights", np.float32).copy())
    vision_position_ids = torch.from_numpy(np_tensor("vision_position_ids", np.int64).copy())
    feature_index = np_tensor("feature_index", np.int64)
    raw = blob[refs["pixel_values"]["offset"]:refs["pixel_values"]["offset"] + refs["pixel_values"]["bytes"]]
    pixels = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(torch.bfloat16).reshape(
        refs["pixel_values"]["shape"]).to(torch.float32)

    source = Qwen3_5ForConditionalGeneration.from_pretrained(
        folder, dtype=torch.float32, local_files_only=True).to("cpu").eval()
    slots = torch.tensor(records["slot_token_ids"], dtype=torch.long)
    image_token_id = source.config.image_token_id

    with torch.inference_mode():
        logits = source(input_ids=ids, pixel_values=pixels, image_grid_thw=grid,
                        mm_token_type_ids=(ids == image_token_id).to(torch.int32),
                        use_cache=False).logits
        scores_a = logits[0, -1].index_select(0, slots).float()

        vision = source.model.visual
        vision_out = vision(pixels, grid_thw=grid, interp_indices=interp_indices,
                            interp_weights=interp_weights, position_ids=vision_position_ids,
                            return_dict=True)
        image_embeds = vision_out.pooler_output
        embeds = source.model.get_input_embeddings()(ids)
        mask = (ids == image_token_id)
        if int(mask.sum()) != image_embeds.shape[0]:
            raise SystemExit(f"contract_crosscheck: {int(mask.sum())} image tokens vs "
                             f"{image_embeds.shape[0]} feature rows")
        if feature_index[0][feature_index[0] >= 0].tolist() != list(range(image_embeds.shape[0])):
            raise SystemExit("contract_crosscheck: feature_index is not 0..P/4-1")
        embeds = embeds.masked_scatter(mask.unsqueeze(-1), image_embeds.to(embeds.dtype))
        hidden = source.model.language_model(inputs_embeds=embeds,
                                             position_ids=position_ids).last_hidden_state
        scores_b = (hidden[0, -1].float()
                    @ source.get_output_embeddings().weight.index_select(0, slots).float().T)

    golden_ref = golden["outputs"][CASE][0]["tensors"][0]
    gblob = (GOLDEN / "outputs.bin").read_bytes()
    scores_g = torch.from_numpy(np.frombuffer(
        gblob, np.float32, count=26, offset=golden_ref["offset"]).copy()).reshape(1, 26)
    for name, value in (("a", scores_a), ("b", scores_b)):
        cosine = torch.nn.functional.cosine_similarity(
            value.reshape(1, -1), scores_g.reshape(1, -1)).item()
        maxdiff = (value - scores_g).abs().max().item()
        print(f"contract_crosscheck: {CASE} path {name} vs golden cosine={cosine:.9f} "
              f"maxabs={maxdiff:.3e}")
        if cosine < 0.999:
            raise SystemExit(f"contract_crosscheck: path {name} cosine {cosine} < 0.999")
    cosine_ab = torch.nn.functional.cosine_similarity(
        scores_a.reshape(1, -1), scores_b.reshape(1, -1)).item()
    print(f"contract_crosscheck: graph path vs whole-model cosine={cosine_ab:.9f}")
    print("contract_crosscheck: OK")


if __name__ == "__main__":
    main()
