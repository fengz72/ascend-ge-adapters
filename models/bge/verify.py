"""Compare GE rankings to the supplied business reference, independently of CPU golden."""
import json
from pathlib import Path

import numpy as np


def compare(config, outputs):
    params = config["verification"]["params"]
    path = Path(params["expected_outputs"])
    if not path.is_absolute():
        path = Path(config["_root"]) / path
    atol = params["atol"]
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not np.isfinite(atol) or atol < 0:
        raise ValueError("BGE Top-10 atol must be a finite nonnegative number")
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("expected_outputs must contain a nonempty list")
    expected = {}
    for record in records:
        if not isinstance(record, dict) or type(record.get("case_id")) is not int or record["case_id"] < 0:
            raise ValueError("expected case_id must be a nonnegative integer")
        case_id = f"case-{record['case_id']:04d}"
        if case_id in expected:
            raise ValueError(f"duplicate expected case_id: {case_id}")
        value = record.get("expected", {})
        ids = np.asarray(value.get("top10_dense_id", []))
        scores = np.asarray(value.get("top10_score", []), dtype=np.float64)
        if (type(value.get("scores_len")) is not int or value["scores_len"] != 10112
                or ids.shape != (10,) or ids.dtype.kind not in "iu"
                or len(set(ids.tolist())) != 10 or np.any((ids < 1) | (ids > 10112))
                or scores.shape != (10,) or not np.isfinite(scores).all()
                or np.any(scores[:-1] < scores[1:])):
            raise ValueError(f"{case_id}: invalid expected Top-10 IDs, scores or scores_len")
        expected[case_id] = (ids, scores)
    if config["verification"]["case_ids"] == "all" and set(expected) != set(outputs):
        raise ValueError("expected/output case IDs differ")
    if set(outputs) - set(expected):
        raise ValueError(f"missing expected cases: {sorted(set(outputs) - set(expected))}")
    reports = {}
    for case_id, row in outputs.items():
        scores = np.asarray(row["outputs"]["scores"])
        if scores.shape != (1, 10112) or scores.dtype != np.float32:
            raise ValueError(f"{case_id}: BGE scores must be float32 [1,10112]")
        if not np.isfinite(scores).all():
            reports[case_id] = {"passed": False, "reason": "nonfinite"}
            continue
        indices = np.argsort(-scores[0], kind="stable")[:10]
        ids = indices + 1  # Business dense IDs start at 1; NumPy indices start at 0.
        actual_scores = scores[0, indices].astype(np.float64)
        expected_ids, expected_scores = expected[case_id]
        errors = np.abs(actual_scores - expected_scores)
        ids_match = set(ids.tolist()) == set(expected_ids.tolist())
        order_match = np.array_equal(ids, expected_ids)
        score_match = bool(np.all(errors <= atol))
        reports[case_id] = {
            "passed": bool(order_match and score_match),
            "top10_ids_match": ids_match,
            "top10_order_match": bool(order_match),
            "top10_scores_match": score_match,
            "atol": atol, "rtol": 0,
            "max_abs_error": float(errors.max()),
            "mean_abs_error": float(errors.mean()),
            "ranks": [{"rank": rank + 1, "expected_dense_id": int(expected_ids[rank]),
                       "actual_dense_id": int(ids[rank]),
                       "expected_score": float(expected_scores[rank]),
                       "actual_score": float(actual_scores[rank]),
                       "abs_error": float(errors[rank])} for rank in range(10)],
        }
    return reports
