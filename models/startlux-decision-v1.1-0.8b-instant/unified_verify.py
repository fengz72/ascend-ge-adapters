"""Business probability verification for the StartLux UNIFIED multimodal route.

Loaded by core.verify.compare_business as the model-owned verification script
(config.verification.script of the unified work YAML).  Pure CPU numpy: no
weights are loaded, no HF forward is called, no NPU.  It reads exactly three
independent sources:

* ``source.ref/decision_config.json`` -- the REAL model folder's 26 letter ids
  and per-type temperatures (the same authority verify.py uses; never fixture
  provenance);
* the case business metadata -- ``payload.rows_meta`` of the unified case pool
  (generate_unified_cases.py): one entry per batch row carrying question_type
  and the letter_ids of the K ACTIVE options;
* ``<work_dir>/golden.npz`` -- the CORE REAL CPU BASELINE produced by the
  unified adapter's independent official reference handle (the original HF
  vision + masked_scatter + official rope + vendored bidirectional forward +
  original embedding head on CPU FP32).  Candidate-derived numbers are never
  accepted as references here.

core's structural numeric policy (verification.outputs.scores,
cosine_min 0.999 / relative_l2_max 0.02, per_sample scope) already gates every
row of the 26 letter logits.  This script adds the BUSINESS probability gate
per row that the structural policy cannot express, with the SAME thresholds as
the accepted text-route verifier (verify.py -- reused, not redefined):

  * softmax over the K ACTIVE columns only, selected by token id, at the
    question type's official temperature;
  * max |p_cand - p_ref| <= 0.01;
  * when the reference top1-top2 probability margin > 0.02, the candidate
    top1 must agree;
  * all 26 scores finite.

Row shape: scores [B, 26] with B = the case's samples (the unified route
serves real batches); EVERY row is checked against its own rows_meta entry.
Failures are loud, never silent passes: unknown case ids, missing or invalid
golden rows, missing/misaligned rows_meta, invalid temperatures, unknown
letter token ids and nonfinite values FAIL the case with an explicit reason.
The runtime accepting arbitrary prompts does not mean such cases verify
without a reference: they fail.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]


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


verify = _load_module("startlux_verify_for_unified", "verify.py")

OUTPUT_NAME = "scores"
N_COLUMNS = 26
PROB_MAX_ABS_MAX = verify.PROB_MAX_ABS_MAX          # 0.01 -- the accepted policy
TOP1_MARGIN_TRIGGER = verify.TOP1_MARGIN_TRIGGER    # 0.02 -- the accepted policy
_softmax = verify._softmax


def _fail(reason):
    return {"passed": False, "reason": reason}


def _read_row(archive, case_id):
    """One saved row (core.worker.save_rows layout) from an open npz archive."""
    prefix = case_id + "/outputs/"
    alignment_key = case_id + "/metadata/alignment"
    if alignment_key not in archive.files:
        return None
    return {"alignment": json.loads(str(archive[alignment_key])),
            "outputs": {key[len(prefix):]: archive[key] for key in archive.files
                        if key.startswith(prefix)}}


def _case_metadata(config):
    """case_id -> payload rows_meta, read through the model's own data script."""
    script = (config.get("data") or {}).get("script")
    if not script:
        raise ValueError("the unified business verifier needs data.script (the case "
                         "pool with payload.rows_meta); a directory-mode pool cannot "
                         "carry per-row business metadata")
    path = Path(script)
    if not path.is_absolute():
        path = ROOT / path
    spec = importlib.util.spec_from_file_location("_ge_unified_case_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = module.load(config)
    metadata = {}
    for case in cases:
        rows_meta = (case.get("payload") or {}).get("rows_meta")
        if case["id"] in metadata:
            raise ValueError(f"duplicate case id {case['id']!r} in the case pool")
        metadata[case["id"]] = rows_meta
    return metadata


def _row_report(case_id, row_index, scores, reference, meta, letters, temperatures):
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
    if not np.isfinite(scores).all():
        return _fail(f"row {row_index} of {case_id!r}: scores contain nonfinite values")
    if not np.isfinite(reference).all():
        return _fail(f"row {row_index} of {case_id!r}: the golden baseline row "
                     "contains nonfinite values")
    reference_probs = _softmax(reference[active] / temperature)
    candidate_probs = _softmax(scores[active] / temperature)
    prob_max_abs = float(np.abs(candidate_probs - reference_probs).max())
    order = np.argsort(-reference_probs, kind="stable")
    margin = 1.0 if len(active) == 1 else float(reference_probs[order[0]]
                                                - reference_probs[order[1]])
    top1_consistent = True if len(active) == 1 else None
    if top1_consistent is None and margin > TOP1_MARGIN_TRIGGER:
        top1_consistent = bool(np.argmax(candidate_probs) == np.argmax(reference_probs))
    norm_reference = float(np.linalg.norm(reference))
    norm_candidate = float(np.linalg.norm(scores))
    cosine = float(reference @ scores / (norm_reference * norm_candidate)) \
        if norm_reference and norm_candidate else 0.0
    relative_l2 = float(np.linalg.norm(reference - scores) / norm_reference) \
        if norm_reference else float("inf")
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
              "top1_consistent": top1_consistent,
              "cosine": cosine,
              "relative_l2": relative_l2}
    if failures:
        report["reason"] = "; ".join(failures)
    return report


def _case_report(case_id, row, metadata_entry, golden_row, letters, temperatures):
    if metadata_entry is None:
        return _fail(f"no case metadata for {case_id!r}: the case pool carries no "
                     "rows_meta for this id")
    if golden_row is None:
        return _fail(f"no golden baseline row for {case_id!r}: the business gate reads "
                     "the core CPU baseline golden.npz; a case without a reference "
                     "fails verification")
    if not isinstance(row, dict):
        return _fail(f"business row for {case_id!r} must be a mapping with 'outputs' "
                     "and 'alignment'")
    outputs = row.get("outputs")
    if not isinstance(outputs, dict) or OUTPUT_NAME not in outputs:
        return _fail(f"business row outputs for {case_id!r} must contain "
                     f"{OUTPUT_NAME!r}")
    alignment = row.get("alignment")
    if not isinstance(alignment, dict):
        return _fail(f"business row alignment for {case_id!r} must be a mapping")
    scores = np.asarray(outputs[OUTPUT_NAME])
    if scores.dtype.kind not in "fiu":
        return _fail(f"scores for {case_id!r} must be numeric, got dtype {scores.dtype}")
    if scores.ndim != 2 or scores.shape[1] != N_COLUMNS:
        return _fail(f"scores for {case_id!r} must be [B, {N_COLUMNS}] (the unified "
                     f"batch row), got shape {scores.shape}")
    batch = scores.shape[0]
    scores = scores.astype(np.float64)
    if not np.isfinite(scores).all():
        return _fail(f"scores for {case_id!r} contain nonfinite values")
    if alignment.get("rows") != [case_id] * batch:
        return _fail(f"alignment.rows for {case_id!r} must identify all {batch} row(s): "
                     f"['{case_id}'] * {batch}; got {alignment.get('rows')!r}")
    if alignment.get("columns") != letters:
        return _fail(f"alignment.columns for {case_id!r} must equal the {N_COLUMNS} "
                     f"letter token ids of source decision_config.json (order "
                     f"included); got {alignment.get('columns')!r}")
    if not isinstance(metadata_entry, list) or len(metadata_entry) != batch:
        return _fail(f"rows_meta for {case_id!r} must carry exactly {batch} per-row "
                     "entries (one per batch row)")
    reference = np.asarray(golden_row["outputs"].get(OUTPUT_NAME))
    if reference.shape != scores.shape:
        return _fail(f"golden baseline shape {reference.shape} != candidate "
                     f"{scores.shape} for {case_id!r}")
    reference = reference.astype(np.float64)
    rows = [_row_report(case_id, i, scores[i], reference[i], metadata_entry[i],
                        letters, temperatures) for i in range(batch)]
    failures = [f"row {i}: {r['reason']}" for i, r in enumerate(rows) if not r["passed"]]
    report = {"passed": not failures, "rows": rows}
    metric_rows = [r for r in rows if "prob_max_abs" in r]
    if metric_rows:
        report.update({"prob_max_abs": max(r["prob_max_abs"] for r in metric_rows),
                       "margin": min(r["margin"] for r in metric_rows),
                       "cosine": min(r["cosine"] for r in metric_rows),
                       "relative_l2": max(r["relative_l2"] for r in metric_rows)})
    if failures:
        report["reason"] = "; ".join(failures)
    return report


def compare(config, outputs):
    """Business gate for every selected case; exact cover of the given rows."""
    letters, temperatures = verify._decision_config(config)
    metadata = _case_metadata(config)
    golden_path = Path(config["work_dir"]) / "golden.npz"
    if not golden_path.is_file():
        raise ValueError(f"the core CPU baseline is missing: {golden_path}; run the "
                         "baseline stage before compare")
    with np.load(golden_path, allow_pickle=False) as golden:
        reports = {}
        for case_id, row in outputs.items():
            golden_row = _read_row(golden, case_id)
            if golden_row is not None and golden_row["alignment"].get("rows") \
                    != (row.get("alignment") or {}).get("rows"):
                reports[case_id] = _fail(
                    f"golden/candidate alignment rows disagree for {case_id!r}: "
                    f"{golden_row['alignment'].get('rows')!r} vs "
                    f"{(row.get('alignment') or {}).get('rows')!r}")
                continue
            reports[case_id] = _case_report(case_id, row, metadata.get(case_id),
                                            golden_row, letters, temperatures)
    return reports
