"""Audit the generated work/inputs pool against the shared 7-tensor graph contract.

Checks, per case: tensor order, dtype, shape, offset/bytes packing, the image
expansion (P/4 merged rows), feature_index contiguity, the position_ids shape,
and the readout metadata.  Also prints the case table.

Usage:
    python3 scripts/verify/inputs_audit.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ADAPTER = Path(__file__).resolve().parents[2]
INPUTS = ADAPTER / "work" / "inputs"
CASES = ADAPTER / "fixtures" / "cases.jsonl"

ORDER = ["input_ids", "pixel_values", "position_ids",
         "interp_indices", "interp_weights", "vision_position_ids",
         "feature_index"]
ITEMSIZE = {"int64": 8, "float32": 4, "int32": 4, "bfloat16": 2}
IMAGE_TOKEN_ID = 248056


def main():
    records = json.loads((INPUTS / "records.json").read_text())
    readout = json.loads((INPUTS / "readout.json").read_text())
    blob = Path(records["file"]).read_bytes()
    rows = [json.loads(x) for x in CASES.read_text().splitlines() if x.strip()]
    kinds = [next(iter(r["questions"].values()))["type"] for r in rows]
    labels_by_id = {c["id"]: c for c in readout["cases"]}

    assert records["schema"] == "ge-records/3" and records["output_name"] == "scores"
    assert len(records["slot_token_ids"]) == 26 and len(set(records["slot_token_ids"])) == 26
    assert records["slot_token_ids"] == list(range(32, 58)), "expected the A..Z token ids 32..57"

    print(f"{'case':10} {'type':7} {'S':>5} {'P':>5} {'grid':>12} {'labels':>22}")
    cursor = 0
    for i, case in enumerate(records["cases"]):
        refs = case["tensors"]
        assert [t["name"] for t in refs] == ORDER, f"{case['id']}: tensor order drifted"
        shapes = {t["name"]: tuple(t["shape"]) for t in refs}
        for t in refs:
            assert t["dtype"] in ITEMSIZE, f"{case['id']}: unknown dtype {t['dtype']}"
            assert math.prod(t["shape"]) * ITEMSIZE[t["dtype"]] == t["bytes"], \
                f"{case['id']}/{t['name']}: bytes mismatch"
            assert t["offset"] == cursor, f"{case['id']}/{t['name']}: gap in payload"
            cursor += t["bytes"]
        length = shapes["input_ids"][1]
        patches = shapes["pixel_values"][0]
        assert shapes["input_ids"] == (1, length)
        assert shapes["pixel_values"] == (patches, 1536)
        assert shapes["position_ids"] == (3, 1, length)
        assert shapes["interp_indices"] == (patches, 4)
        assert shapes["interp_weights"] == (patches, 4)
        assert shapes["vision_position_ids"] == (patches, 2)
        assert shapes["feature_index"] == (1, length)
        merged = patches // 4
        ref = {t["name"]: t for t in refs}
        ids = np.frombuffer(blob, np.int64, count=length, offset=ref["input_ids"]["offset"])
        assert int((ids == IMAGE_TOKEN_ID).sum()) == merged, f"{case['id']}: image pads != P/4"
        f = np.frombuffer(blob, np.int64, count=length, offset=ref["feature_index"]["offset"])
        assert int((f >= 0).sum()) == merged
        assert np.array_equal(f[f >= 0], np.arange(merged))
        g = case.get("grid")
        assert isinstance(g, list) and len(g) == 3 and g[0] == 1, f"{case['id']}: grid metadata"
        assert g[1] * g[2] == patches, f"{case['id']}: grid does not cover P"
        ro = labels_by_id[case["id"]]
        assert ro["letters_asked"] == len(ro["labels"]) <= 26 and ro["temperature"] == 1.0
        print(f"{case['id']:10} {kinds[i]:7} {length:5d} {patches:5d} "
              f"{'x'.join(map(str, g)):>12} {','.join(ro['labels'])[:22]:>22}")

    expected = sum(t["bytes"] for c in records["cases"] for t in c["tensors"])
    assert expected == len(blob), "inputs.bin has trailing or missing bytes"
    print(f"inputs_audit: {len(records['cases'])} cases, {len(blob)} bytes, contract OK")


if __name__ == "__main__":
    main()
