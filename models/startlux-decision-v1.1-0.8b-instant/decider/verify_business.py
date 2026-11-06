"""Standalone business probability gate for the StartLux unified route.

The business stage of the standalone shell flow
(docs/standalone-runtime-plan.md): the migration of the validated unified
business verifier (unified_verify.py, the P7-accepted gate with the same
thresholds as the text-route verifier).  Pure CPU NumPy: no weights, no HF
forward, no NPU, no ``core``.  It runs AFTER the numeric comparison and reads
exactly four independent sources:

* ``STARTLUX_MODEL_DIR/decision_config.json`` -- the REAL model folder's 26
  letter ids and per-type temperatures (never fixture provenance);
* ``work/inputs/cases.json`` -- the data stage's per-row business metadata
  (payload.rows_meta of the business cases: question_type and the letter_ids
  of the K ACTIVE options);
* ``work/golden`` -- the official CPU FP32 baseline (result.json +
  outputs.bin);
* ``work/infer`` -- the GE candidate result (result.json + outputs.bin).

Gate per business row, with the accepted thresholds (softmax over the K ACTIVE
columns only, selected by token id, at the question type's official
temperature):

  * max |p_cand - p_golden| <= 0.01;
  * when the reference top1-top2 probability margin > 0.02, the candidate
    top1 must agree;
  * all 26 scores finite.

Cases flagged ``business: false`` in the case index (the b4-long boundary
case: a shape-coverage row without business semantics) are listed explicitly
in the report as numeric-gate-only -- never silently passed here; they are
fully gated by the numeric comparison stage that must run first.  Failures
are loud: unknown case ids, missing golden rows, missing or misaligned
rows_meta, invalid temperatures, unknown letter token ids and nonfinite
values FAIL with an explicit reason.

Writes work/infer/business.json; exit code 0 only when every business row
passes.  A failure blocks the performance stage (run.sh runs under set -e).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root
WORK = MODEL_DIR / "work"
INPUTS = WORK / "inputs"
INFER = WORK / "infer"
GOLDEN = WORK / "golden"

OUTPUT_NAME = "scores"
N_COLUMNS = 26
PROB_MAX_ABS_MAX = 0.01                     # the accepted policy
TOP1_MARGIN_TRIGGER = 0.02                   # the accepted policy


def _softmax(values):
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def _fail(reason):
    return {"passed": False, "reason": reason}


def _load_scores(directory):
    """One result directory (result.json + outputs.bin) -> {case: [B, 26] f64}."""
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if result.get("status") != "success":
        raise SystemExit(f"verify_business: {directory}/result.json status is "
                         f"{result.get('status')!r}; run the producing stage first")
    blob = (directory / "outputs.bin").read_bytes()
    scores = {}
    for case_id, rows in result["outputs"].items():
        if len(rows) != 1 or rows[0]["instance"] != 0:
            raise SystemExit(f"verify_business: case {case_id!r} must carry exactly one "
                             f"instance-0 row in {directory}")
        arrays = {}
        for tensor in rows[0]["tensors"]:
            if tensor["dtype"] != "float32":
                raise SystemExit(f"verify_business: unsupported dtype "
                                 f"{tensor['dtype']!r} for {tensor['name']!r}")
            shape = tensor["shape"]
            expected = int(np.prod(shape, dtype=np.int64)) * 4
            offset, size = tensor["offset"], tensor["bytes"]
            if size != expected or offset + size > len(blob):
                raise SystemExit(f"verify_business: {case_id}/{tensor['name']} byte range "
                                 f"is inconsistent with outputs.bin in {directory}")
            arrays[tensor["name"]] = np.frombuffer(
                blob, dtype=np.float32, count=expected // 4,
                offset=offset).reshape(shape).astype(np.float64)
        if set(arrays) != {OUTPUT_NAME}:
            raise SystemExit(f"verify_business: case {case_id!r} in {directory} must carry "
                             f"exactly the {OUTPUT_NAME!r} output")
        scores[case_id] = arrays[OUTPUT_NAME]
    return scores


def _row_report(case_id, row_index, candidate, reference, meta, letters, temperatures):
    if not isinstance(meta, dict):
        return _fail(f"row {row_index} of {case_id!r} has no rows_meta entry: the "
                     "business gate needs question_type and the active letter ids")
    question_type = meta.get("question_type")
    temperature = temperatures.get(question_type) if isinstance(question_type, str) else None
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
            or not np.isfinite(temperature) or temperature <= 0:
        return _fail(f"row {row_index} of {case_id!r}: invalid temperature for "
                    f"question_type {question_type!r} in source decision_config "
                    f"temperature_by_type: {temperature!r}")
    letter_ids = meta.get("letter_ids")
    if not isinstance(letter_ids, list) or not letter_ids \
            or any(type(t) is not int for t in letter_ids):
        return _fail(f"row {row_index} of {case_id!r}: letter_ids must be a nonempty "
                    "list of true ints (the K active options)")
    if len(set(letter_ids)) != len(letter_ids):
        return _fail(f"row {row_index} of {case_id!r}: duplicated letter ids "
                    f"{letter_ids!r}")
    unknown = [t for t in letter_ids if t not in letters]
    if unknown:
        return _fail(f"row {row_index} of {case_id!r}: unknown letter token id(s) "
                     f"{unknown}: not in source decision_config letter_token_ids")
    active = [letters.index(t) for t in letter_ids]
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        return _fail(f"row {row_index} of {case_id!r}: scores contain nonfinite values")
    reference_probs = _softmax(reference[active] / temperature)
    candidate_probs = _softmax(candidate[active] / temperature)
    prob_max_abs = float(np.abs(candidate_probs - reference_probs).max())
    order = np.argsort(-reference_probs, kind="stable")
    margin = 1.0 if len(active) == 1 else float(reference_probs[order[0]]
                                                - reference_probs[order[1]])
    top1_consistent = True if len(active) == 1 else None
    if top1_consistent is None and margin > TOP1_MARGIN_TRIGGER:
        top1_consistent = bool(np.argmax(candidate_probs) == np.argmax(reference_probs))
    failures = []
    if prob_max_abs > PROB_MAX_ABS_MAX:
        failures.append(f"active-option probability max abs {prob_max_abs:.6g} > "
                        f"{PROB_MAX_ABS_MAX}")
    if top1_consistent is False:
        failures.append(f"top1 letter inconsistent while reference margin "
                        f"{margin:.6g} > {TOP1_MARGIN_TRIGGER}")
    report = {"passed": not failures,
              "question_type": question_type,
              "temperature": float(temperature),
              "active_K": len(active),
              "active_columns": active,
              "prob_max_abs": prob_max_abs,
              "margin": margin,
              "top1_consistent": top1_consistent}
    if failures:
        report["reason"] = "; ".join(failures)
    return report


def _case_report(case_id, case, candidate, reference, letters, temperatures):
    meta_entry = (case.get("payload") or {}).get("rows_meta")
    if not isinstance(meta_entry, list) or len(meta_entry) != len(reference):
        return _fail(f"rows_meta for {case_id!r} must carry exactly {len(reference)} "
                     "per-row entries (one per batch row)")
    if candidate.shape != reference.shape or candidate.shape[1] != N_COLUMNS:
        return _fail(f"golden/candidate shapes disagree for {case_id!r}: "
                     f"{reference.shape} vs {candidate.shape}")
    rows = [_row_report(case_id, index, candidate[index], reference[index],
                        meta_entry[index], letters, temperatures)
            for index in range(len(reference))]
    failures = [f"row {index}: {row['reason']}" for index, row in enumerate(rows)
                if not row["passed"]]
    report = {"passed": not failures, "rows": rows}
    metric_rows = [row for row in rows if "prob_max_abs" in row]
    if metric_rows:
        report.update({"prob_max_abs": max(row["prob_max_abs"] for row in metric_rows),
                       "margin": min(row["margin"] for row in metric_rows)})
    if failures:
        report["reason"] = "; ".join(failures)
    return report


def main():
    value = os.environ.get("STARTLUX_MODEL_DIR")
    if not value or not value.strip():
        raise SystemExit("verify_business: STARTLUX_MODEL_DIR is not set or is empty")
    folder = Path(value).resolve()
    decision = json.loads((folder / "decision_config.json").read_text())
    letters = decision["letter_token_ids"]
    temperatures = decision["temperature_by_type"]
    if not isinstance(letters, list) or len(letters) != N_COLUMNS:
        raise SystemExit("verify_business: decision_config letter_token_ids must carry "
                         f"{N_COLUMNS} ids")

    index = json.loads((INPUTS / "cases.json").read_text())
    if index.get("schema") != "startlux-standalone-cases/1":
        raise SystemExit("verify_business: work/inputs/cases.json is not the standalone "
                         "case index; run generate_inputs.py first")
    cases = index["cases"]
    golden_scores = _load_scores(GOLDEN)
    infer_scores = _load_scores(INFER)
    if set(golden_scores) != set(infer_scores):
        raise SystemExit(f"verify_business: golden and infer case sets differ "
                         f"(golden-only {sorted(set(golden_scores) - set(infer_scores))}, "
                         f"infer-only {sorted(set(infer_scores) - set(golden_scores))})")

    reports, skipped = {}, {}
    for case_id, case in cases.items():
        if not case.get("business", False):
            skipped[case_id] = "numeric-gate-only (no business metadata; gated by the " \
                                "numeric comparison stage)"
            continue
        if case_id not in infer_scores:
            reports[case_id] = _fail(f"no infer row for {case_id!r}")
            continue
        reports[case_id] = _case_report(case_id, case, infer_scores[case_id],
                                        golden_scores[case_id], letters, temperatures)

    failed = {case_id: report for case_id, report in reports.items()
              if not report["passed"]}
    report = {"schema": "startlux-business-gate/1",
              "thresholds": {"prob_max_abs_max": PROB_MAX_ABS_MAX,
                             "top1_margin_trigger": TOP1_MARGIN_TRIGGER},
              "passed": not failed,
              "cases": len(reports),
              "skipped": skipped,
              "case_reports": reports}
    if failed:
        report["failed_cases"] = failed
    target = INFER / "business.json"
    target.write_text(json.dumps(report, indent=1) + "\n")

    if reports:
        worst = max((report.get("prob_max_abs", 0.0) for report in reports.values()),
                    default=0.0)
        print(f"verify_business: {len(reports)} business cases, "
              f"worst active-probability max abs {worst:.6f} "
              f"(max {PROB_MAX_ABS_MAX})")
    for case_id, entry in failed.items():
        print(f"FAIL {case_id}: {entry.get('reason')}", file=sys.stderr)
    if failed:
        raise SystemExit(f"verify_business: {len(failed)} case(s) failed; see {target}")
    print(f"PASS ({len(skipped)} numeric-gate-only case(s) skipped by design); "
          f"report: {target}")


if __name__ == "__main__":
    main()
