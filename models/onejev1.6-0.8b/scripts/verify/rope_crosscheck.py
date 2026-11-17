"""Cross-check the meta-device get_rope_index output against a literal transcription
of the 5.17.0 source loop, plus the pure-text degenerate case.

The transcription follows transformers/models/qwen3_5/modeling_qwen3_5.py
``Qwen3_5Model.get_rope_index`` exactly: group mm_token_type_ids into runs,
text runs get ``arange + current_pos`` on all three axes, an image run gets the
model's ``get_vision_position_ids(current_pos, grid, 1, merge)`` (t=1 here) and
advances current_pos by ``max(h, w) // merge``.

Usage:
    python3 scripts/verify/rope_crosscheck.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ADAPTER / "work" / "deps" / "py"))

import numpy as np
import torch
from transformers import AutoConfig

INPUTS = ADAPTER / "work" / "inputs"
MODEL = Path(os.environ["ONEJEV_MODEL_DIR"]) if os.environ.get("ONEJEV_MODEL_DIR") else None


def manual_rope(ids, grid, merge):
    """Literal transcription of the 5.17.0 get_rope_index (one batch row, no mask)."""
    token_type = (np.asarray(ids) == 248056).astype(np.int64)
    groups = []
    start = 0
    for i in range(1, len(token_type) + 1):
        if i == len(token_type) or token_type[i] != token_type[start]:
            groups.append((int(token_type[start]), start, i))
            start = i
    current_pos = 0
    planes = []
    for modality, start_idx, end_idx in groups:
        if modality == 0:
            text_len = end_idx - start_idx
            planes.append(np.tile(np.arange(text_len, dtype=np.int64) + current_pos, (3, 1)))
            current_pos += text_len
        else:
            t, h, w = (int(x) for x in grid)
            assert t == 1
            llm_h, llm_w = h // merge, w // merge
            tpos = np.arange(1, dtype=np.int64)
            hpos = np.arange(llm_h, dtype=np.int64) + current_pos
            wpos = np.arange(llm_w, dtype=np.int64) + current_pos
            tg, hg, wg = np.meshgrid(tpos, hpos, wpos, indexing="ij")
            vision = np.stack([tg.reshape(-1), hg.reshape(-1), wg.reshape(-1)], axis=0)
            vision[0] += current_pos
            planes.append(vision)
            current_pos += max(h, w) // merge
    return np.concatenate(planes, axis=1).reshape(3, -1)


def main():
    if MODEL is None:
        raise SystemExit("rope_crosscheck: set ONEJEV_MODEL_DIR")
    config = AutoConfig.from_pretrained(str(MODEL), local_files_only=True)
    merge = config.vision_config.spatial_merge_size

    records = json.loads((INPUTS / "records.json").read_text())
    blob = Path(records["file"]).read_bytes()
    checked = 0
    for case in records["cases"]:
        refs = {t["name"]: t for t in case["tensors"]}
        ids_ref = refs["input_ids"]
        pos_ref = refs["position_ids"]
        length = ids_ref["shape"][1]
        ids = np.frombuffer(blob, np.int64, count=length, offset=ids_ref["offset"])
        grid = np.asarray(case["grid"], dtype=np.int64)   # records metadata
        got = np.frombuffer(blob, np.int64, count=3 * length,
                            offset=pos_ref["offset"]).reshape(3, 1, length)
        want = manual_rope(ids, grid, merge)[:, None, :]
        if not np.array_equal(got, want):
            bad = int((got != want).sum())
            raise SystemExit(f"rope_crosscheck: case {case['id']} differs in {bad} positions")
        checked += 1
    print(f"rope_crosscheck: {checked} cases match the literal get_rope_index loop")

    # Pure-text degenerate: no image run -> three arange planes.
    import torch
    from transformers import Qwen3_5Model
    with torch.device("meta"):
        model = Qwen3_5Model(config)
    text = torch.tensor([[32, 33, 34, 248044, 57]], dtype=torch.int64)
    positions, _ = model.get_rope_index(text, torch.zeros_like(text, dtype=torch.int32))
    arange = torch.arange(text.shape[1], dtype=torch.int64)[None, None, :].expand(3, 1, -1)
    if not torch.equal(positions, arange):
        raise SystemExit("rope_crosscheck: pure-text position_ids are not the arange planes")
    print("rope_crosscheck: pure-text position_ids == three arange planes")
    print("rope_crosscheck: OK")


if __name__ == "__main__":
    main()
