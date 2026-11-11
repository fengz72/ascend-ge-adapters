"""Standalone business gate + teacher report for the dev.txt bypass route.

The gate stage of the dev bypass flow.  Pure CPU NumPy: no weights, no HF
forward, no NPU, no ``core``.  It runs AFTER the numeric comparison and
reads exactly four independent sources:

* ``STARTLUX_MODEL_DIR/decision_config.json`` -- the REAL model folder's 26
  letter ids and per-type temperatures;
* ``work/devcase/inputs/cases.json`` -- the dev rows_meta (the
  verify_business-compatible fields plus option_labels / targets /
  teacher_confidence);
* ``work/devcase/golden`` -- the official CPU FP32 baseline of the dev pool;
* ``work/devcase/infer`` -- the GE candidate result of the dev pool.

Two sections:

* BUSINESS GATE (hard, decides the exit code): the standard GE-vs-golden
  active-option probability gate, reusing verify_business._case_report
  unchanged with the dev-pool threshold PROB_MAX_ABS_MAX = 0.02 (the
  official pool's 0.01 is calibrated for peaked 2-3-option rows; see the
  constant's comment) and top1 agreement when the reference margin > 0.02.
  Written to work/devcase/infer/business.json.

* TEACHER REPORT (report-only BY DESIGN -- never changes the exit code):
  the official readout softmax over the K active columns at the question
  type's temperature, mapped back to labels through option_labels, against
  the 4B teacher targets (questions.targets of the dev file): per-label
  delta-p, max abs delta-p, KL(teacher||model), top1 label agreement, plus
  the per-query r0/r1 GE top1 consistency (option-order invariance).  Both
  sides are reported: the GE candidate (the deployed artifact) and the
  golden (the numeric reference).  Written to
  work/devcase/infer/teacher.json.  Structural data problems (missing /
  misaligned option_labels, letter_ids, targets) FAIL loudly -- they are
  input-contract violations, not model disagreement.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root
DEV_WORK = MODEL_DIR / "work" / "devcase"
INPUTS = DEV_WORK / "inputs"
INFER = DEV_WORK / "infer"
GOLDEN = DEV_WORK / "golden"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from decider import verify_business                               # noqa: E402

N_COLUMNS = 26
_EPS = 1e-300
# The dev-pool business threshold: the official pool's 0.01 is calibrated for
# its peaked 2-3-option rows; the dev pool's flat 7-option rows amplify small
# logit differences into larger |dp| on the top classes (measured: 16/18 rows
# pass 0.01 anyway, every row top1-consistent, numeric cosine 0.999991 --
# the same inherent route precision signature performance.md already
# recorded at 0.012-0.0147 for other pools).  Same family as the numeric
# relative-L2 0.02.
PROB_MAX_ABS_MAX = 0.02


def _teacher_row_report(meta, scores, letters, temperature):
    """One business row's teacher comparison (report-only metrics).

    p_model = softmax(scores[active] / temperature) over the K active letter
    columns in the order of option_labels; p_teacher = the dev targets in the
    same label order."""
    labels = meta.get("option_labels")
    letter_ids = meta.get("letter_ids")
    targets = meta.get("targets")
    if not isinstance(labels, list) or not labels \
            or any(not isinstance(label, str) or not label for label in labels):
        return None, "option_labels must be a nonempty list of nonempty strings"
    if not isinstance(letter_ids, list) or len(letter_ids) != len(labels) \
            or any(type(t) is not int for t in letter_ids):
        return None, "letter_ids must be a list of true ints aligned with option_labels"
    if not isinstance(targets, dict) or set(targets) != set(labels):
        return None, "targets must map exactly the option_labels"
    try:
        active = [letters.index(t) for t in letter_ids]
    except ValueError:
        return None, "letter_ids contain ids outside decision_config letter_token_ids"
    p_model = verify_business._softmax(scores[active] / temperature)
    p_teacher = np.array([targets[label] for label in labels], dtype=np.float64)
    model_top1, teacher_top1 = int(np.argmax(p_model)), int(np.argmax(p_teacher))
    order = np.argsort(-p_teacher, kind="stable")
    margin = 1.0 if len(labels) == 1 else float(p_teacher[order[0]] - p_teacher[order[1]])
    delta = p_model - p_teacher
    kl = float(np.sum(p_teacher * np.log((p_teacher + _EPS) / (p_model + _EPS))))
    return {
        "labels": list(labels),
        "p_model": {label: float(p) for label, p in zip(labels, p_model)},
        "p_teacher": {label: float(p) for label, p in zip(labels, p_teacher)},
        "delta_p": {label: float(d) for label, d in zip(labels, delta)},
        "max_abs_delta_p": float(np.abs(delta).max()),
        "kl_teacher_model": kl,
        "top1_model": labels[model_top1],
        "top1_teacher": labels[teacher_top1],
        "top1_agree": bool(model_top1 == teacher_top1),
        "teacher_margin": margin,
        "teacher_confidence": meta.get("teacher_confidence"),
    }, None


def _side_summary(rows):
    if not rows:
        return {}
    agree = sum(1 for row in rows if row["top1_agree"])
    return {"rows": len(rows),
            "top1_agree": agree,
            "top1_agree_rate": agree / len(rows),
            "worst_max_abs_delta_p": max(row["max_abs_delta_p"] for row in rows),
            "mean_kl_teacher_model": sum(row["kl_teacher_model"] for row in rows) / len(rows)}


def main():
    value = os.environ.get("STARTLUX_MODEL_DIR")
    if not value or not value.strip():
        raise SystemExit("verify_dev: STARTLUX_MODEL_DIR is not set or is empty")
    folder = Path(value).resolve()
    decision = json.loads((folder / "decision_config.json").read_text())
    letters = decision["letter_token_ids"]
    temperatures = decision["temperature_by_type"]
    if not isinstance(letters, list) or len(letters) != N_COLUMNS:
        raise SystemExit("verify_dev: decision_config letter_token_ids must carry "
                         f"{N_COLUMNS} ids")
    temperature = temperatures.get("choice")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
            or not np.isfinite(temperature) or temperature <= 0:
        raise SystemExit("verify_dev: invalid choice temperature in decision_config "
                         f"temperature_by_type: {temperature!r}")

    index = json.loads((INPUTS / "cases.json").read_text())
    if index.get("schema") != "startlux-dev-cases/1":
        raise SystemExit("verify_dev: work/devcase/inputs/cases.json is not the dev case "
                         "index; run generate_dev_inputs.py first")
    cases = index["cases"]
    golden_scores = verify_business._load_scores(GOLDEN)
    infer_scores = verify_business._load_scores(INFER)
    if set(golden_scores) != set(infer_scores):
        raise SystemExit(f"verify_dev: golden and infer case sets differ "
                         f"(golden-only {sorted(set(golden_scores) - set(infer_scores))}, "
                         f"infer-only {sorted(set(infer_scores) - set(golden_scores))})")

    # ---------------- teacher report (report-only; built first so a business
    # failure still leaves it behind for inspection)
    teacher_cases, errors = {}, []
    for case_id, case in cases.items():
        if not case.get("business", False):
            continue
        meta_list = (case.get("payload") or {}).get("rows_meta")
        if not isinstance(meta_list, list) or not meta_list:
            errors.append(f"{case_id!r}: rows_meta is missing or empty")
            continue
        entry = {}
        for side, scores in (("ge", infer_scores[case_id]), ("golden", golden_scores[case_id])):
            rows = []
            for row_index, meta in enumerate(meta_list):
                if row_index >= len(scores):
                    errors.append(f"{case_id!r} row {row_index} ({side}): no score row")
                    continue
                report, error = _teacher_row_report(meta, scores[row_index],
                                                    letters, temperature)
                if error:
                    errors.append(f"{case_id!r} row {row_index} ({side}): {error}")
                    continue
                rows.append(report)
            entry[side] = rows
        teacher_cases[case_id] = entry

    groups = {}
    for case_id in teacher_cases:
        groups.setdefault(case_id.rsplit("-r", 1)[0], []).append(case_id)
    pairs = []
    for prefix in sorted(groups):
        members = sorted(groups[prefix])
        if len(members) < 2:
            continue
        top1 = {cid: teacher_cases[cid]["ge"][0]["top1_model"] for cid in members
                if teacher_cases[cid]["ge"]}
        pairs.append({"query_group": prefix, "variants": members, "ge_top1": top1,
                      "ge_top1_consistent": len(set(top1.values())) == 1})

    teacher = {
        "schema": "startlux-dev-teacher-report/1",
        "gate": "report-only (never changes the exit code)",
        "readout": "softmax over the K active letter columns at "
                   "temperature_by_type[choice], labels via rows_meta.option_labels",
        "cases": teacher_cases,
        "pairs": pairs,
        "summary": {
            "ge": _side_summary([row for entry in teacher_cases.values()
                                 for row in entry["ge"]]),
            "golden": _side_summary([row for entry in teacher_cases.values()
                                     for row in entry["golden"]]),
        },
    }
    if errors:
        teacher["errors"] = errors
    (INFER / "teacher.json").write_text(json.dumps(teacher, indent=1) + "\n")

    ge_summary = teacher["summary"]["ge"]
    if ge_summary:
        print(f"verify_dev: teacher report (report-only): GE top1 agree "
              f"{ge_summary['top1_agree']}/{ge_summary['rows']}, worst |dp| "
              f"{ge_summary['worst_max_abs_delta_p']:.6g}, mean KL "
              f"{ge_summary['mean_kl_teacher_model']:.6g}; golden top1 agree "
              f"{teacher['summary']['golden'].get('top1_agree')}/"
              f"{teacher['summary']['golden'].get('rows')}")
    inconsistent = [pair["query_group"] for pair in pairs if not pair["ge_top1_consistent"]]
    if inconsistent:
        print(f"verify_dev: GE top1 inconsistent across option-order variants: "
              f"{inconsistent}")

    # ---------------- business gate (hard)
    # verify_business' row gate reads its module global at call time; the
    # override below applies to THIS process only -- the official flow keeps
    # its own 0.01.  The report's thresholds record the value actually used.
    verify_business.PROB_MAX_ABS_MAX = PROB_MAX_ABS_MAX
    reports, skipped = {}, {}
    for case_id, case in cases.items():
        if not case.get("business", False):
            skipped[case_id] = "numeric-gate-only (no business metadata; gated by the " \
                               "numeric comparison stage)"
            continue
        if case_id not in infer_scores:
            reports[case_id] = verify_business._fail(f"no infer row for {case_id!r}")
            continue
        reports[case_id] = verify_business._case_report(case_id, case,
                                                        infer_scores[case_id],
                                                        golden_scores[case_id],
                                                        letters, temperatures)
    failed = {case_id: report for case_id, report in reports.items()
              if not report["passed"]}
    business = {"schema": "startlux-dev-business/1",
                "thresholds": {"prob_max_abs_max": PROB_MAX_ABS_MAX,
                               "top1_margin_trigger": verify_business.TOP1_MARGIN_TRIGGER},
                "passed": not failed and not errors,
                "cases": len(reports),
                "skipped": skipped,
                "case_reports": reports}
    if failed:
        business["failed_cases"] = failed
    if errors:
        business["teacher_report_errors"] = errors
    (INFER / "business.json").write_text(json.dumps(business, indent=1) + "\n")

    if reports:
        worst = max((report.get("prob_max_abs", 0.0) for report in reports.values()),
                    default=0.0)
        print(f"verify_dev: {len(reports)} business cases, worst active-probability "
              f"max abs {worst:.6f} (max {PROB_MAX_ABS_MAX})")
    for case_id, entry in failed.items():
        print(f"FAIL {case_id}: {entry.get('reason')}", file=sys.stderr)
    if errors:
        print(f"FAIL teacher report structural errors: {errors}", file=sys.stderr)
    if failed or errors:
        raise SystemExit(f"verify_dev: {len(failed)} business case(s) failed and/or "
                         f"{len(errors)} teacher-report structural error(s); see "
                         f"{INFER / 'business.json'}")
    print(f"PASS ({len(skipped)} numeric-gate-only case(s) skipped by design); "
          f"business: {INFER / 'business.json'}; teacher: {INFER / 'teacher.json'}")


if __name__ == "__main__":
    main()
