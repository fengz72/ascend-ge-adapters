"""Business probability verification for the StartLux decision read-outs.

Loaded by core.verify.compare_business as the model-owned verification script
(config.verification.script).  Pure CPU numpy: no weights are loaded and no HF
forward is called; from the source folder only the lightweight
decision_config.json is read.

core's structural numeric policy (verification.outputs.scores) already gates the
26 letter logits with cosine/relative-L2.  This script adds the business
probability gate that the structural policy cannot express:

  * reference : the fixed canonical read-out fixture (default
                model.params.fixtures_file, i.e. fixtures/p2b-readouts.json,
                the P2b CPU FP32 goldens, one B=1 case per read-out) provides
                scores26_golden_fp32, letter_ids (the K active options) and
                question_type per case id.
  * letters26 and temperature ALWAYS come from source.ref/decision_config.json
                (the real model folder), never from the fixture provenance.
  * gate      : softmax over the K ACTIVE columns only, selected by token id;
                max |p_cand - p_ref| <= 0.01; when the reference top1-top2
                probability margin > 0.02 the candidate top1 must agree; the 26
                scores must be finite.  A softmax over all 26 columns would
                dilute per-option probability error with the inactive columns
                and is deliberately NOT used.

Row shape (what core.worker infer/bench actually saves): a B=1 read-out row is
scores with shape (1, 26); the unique row is selected.  A flat (26,) vector is
accepted for direct single-read-out use.  Batched rows (B > 1) are NOT
supported and fail with a clear reason.

Alignment contract: the current RuntimeHandle.decode emits
{"rows": [case_id]*B, "columns": letters26} with no business metadata (see
work/startlux-cannbot/p4-io-contract.json).  rows must equal [case_id] for
B=1.  question_type comes from the reference fixture; an optional
alignment.question_type is cross-checked only when present.  Legacy P3 outputs
whose alignment carries readout (or id) instead of rows are accepted with the
same case-id check; golden_file is ignored either way.

source.ref resolution mirrors the framework: ${VAR} placeholders are expanded
via core.config.expand_source_path (an unset or empty variable raises), and a
relative source.ref resolves against the project ROOT, never against the
current working directory.  load_authoring(source_paths=False) deliberately
leaves source.ref unexpanded, so this script resolves it itself.

Failures are loud, never silent passes: unknown case ids, missing or invalid
references, invalid temperatures, unknown letter token ids, empty or
misaligned columns, wrong shapes and nonfinite values FAIL the case with an
explicit reason.  Fixture- or decision-config-level errors raise.  The runtime
accepting arbitrary prompts does not mean such cases verify without a
reference: they fail.

Path rules for verification.params.fixtures_file (explicit override for a new
dataset): absolute paths are used as-is, relative paths resolve against the
project ROOT (the same convention as verification.script).  Without the
override the fixture is model.params.fixtures_file resolved against this model
directory, exactly like the adapter's own fixture loading.

Limitations (deliberate): B=1 single-read-out rows only.  A fixture schema for
batched rows (a readout_ids mapping) is deliberately NOT invented until a real
requirement defines it; a batched scores row fails with a clear reason.
"""
import json
from pathlib import Path

import numpy as np

from core.config import expand_source_path

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]

OUTPUT_NAME = "scores"
N_COLUMNS = 26
PROB_MAX_ABS_MAX = 0.01
TOP1_MARGIN_TRIGGER = 0.02
DEFAULT_FIXTURES_FILE = "fixtures/p2b-readouts.json"


def _fail(reason):
    return {"passed": False, "reason": reason}


def _read_json(path, what):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"{what} is missing: {path}") from None
    except (OSError, ValueError) as error:
        raise ValueError(f"{what} at {path} is unreadable or invalid JSON: {error}") from None


def _source_folder(config):
    ref = (config.get("source") or {}).get("ref")
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("verification requires source.ref: letter ids and temperatures are "
                         "read from the model folder's decision_config.json")
    path = Path(expand_source_path(ref, "source.ref"))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _fixture_path(config):
    override = config.get("verification", {}).get("params", {}).get("fixtures_file")
    if override:
        if not isinstance(override, str) or not override.strip():
            raise ValueError("verification.params.fixtures_file must be a nonempty path")
        path = Path(override)
        return path if path.is_absolute() else ROOT / path
    name = config.get("model", {}).get("params", {}).get("fixtures_file", DEFAULT_FIXTURES_FILE)
    path = Path(name)
    return path if path.is_absolute() else MODEL_DIR / path


def _decision_config(config):
    data = _read_json(_source_folder(config) / "decision_config.json",
                      "source decision_config.json")
    letters = data.get("letter_token_ids")
    if not isinstance(letters, list) or len(letters) != N_COLUMNS \
            or any(not isinstance(token, int) or isinstance(token, bool) for token in letters) \
            or len(set(letters)) != N_COLUMNS:
        raise ValueError(f"decision_config letter_token_ids must be {N_COLUMNS} unique integers")
    temperatures = data.get("temperature_by_type")
    if not isinstance(temperatures, dict):
        raise ValueError("decision_config temperature_by_type must be a mapping")
    return letters, temperatures


def _references(config):
    data = _read_json(_fixture_path(config), "read-out fixture")
    readouts = data.get("readouts")
    if not isinstance(readouts, list) or not readouts:
        raise ValueError("fixture must contain a nonempty 'readouts' list")
    references = {}
    for readout in readouts:
        case_id = readout.get("case_id") if isinstance(readout, dict) else None
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"fixture readout case_id must be a nonempty string: {case_id!r}")
        if case_id in references:
            raise ValueError(f"fixture contains duplicate case_id {case_id!r}")
        references[case_id] = readout
    return references


def _softmax(values):
    weights = np.exp(values - values.max())
    return weights / weights.sum()


def _check_alignment(case_id, alignment, letters):
    if not isinstance(alignment, dict):
        return f"business row alignment for {case_id!r} must be a mapping"
    identifier = None
    for field, single in (("rows", False), ("readout", True), ("id", True)):
        if field not in alignment:
            continue
        value = alignment.get(field)
        expected = [case_id] if single is False else case_id
        if value != expected:
            return (f"alignment.{field} must identify the B=1 case "
                    f"({expected!r}), got {value!r}")
        identifier = field
    if identifier is None:
        return ("alignment must identify its rows: 'rows': [case_id] (current RuntimeHandle "
                "decode contract) or legacy 'readout'/'id' equal to the case id")
    columns = alignment.get("columns")
    if columns != letters:
        return (f"alignment.columns for {case_id!r} must equal the {N_COLUMNS} letter token "
                f"ids of source decision_config.json (order included); got {columns!r}")
    return None


def _case_report(case_id, row, readout, letters, temperatures):
    if readout is None:
        return _fail(f"no fixture reference for case id {case_id!r}: the runtime accepts any "
                     "prompt, but a case without a reference read-out fails verification")
    if not isinstance(row, dict):
        return _fail(f"business row for {case_id!r} must be a mapping with 'outputs' and 'alignment'")
    outputs = row.get("outputs")
    if not isinstance(outputs, dict) or OUTPUT_NAME not in outputs:
        return _fail(f"business row outputs for {case_id!r} must contain {OUTPUT_NAME!r}")
    problem = _check_alignment(case_id, row.get("alignment"), letters)
    if problem:
        return _fail(problem)
    question_type = readout.get("question_type")
    temperature = temperatures.get(question_type) if isinstance(question_type, str) else None
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
            or not np.isfinite(temperature) or temperature <= 0:
        return _fail(f"invalid temperature for question_type {question_type!r} in source "
                     f"decision_config temperature_by_type: {temperature!r}")
    row_type = (row.get("alignment") or {}).get("question_type")
    if row_type is not None and row_type != question_type:
        return _fail(f"alignment.question_type {row_type!r} does not match the fixture "
                     f"reference question_type {question_type!r} for {case_id!r}")
    letter_ids = readout.get("letter_ids")
    if not isinstance(letter_ids, list) or not letter_ids:
        return _fail(f"reference letter_ids for {case_id!r} must be a nonempty list of "
                     "active token ids")
    if len(set(letter_ids)) != len(letter_ids):
        return _fail(f"reference letter_ids for {case_id!r} contain duplicates: {letter_ids!r}")
    unknown = [token for token in letter_ids if token not in letters]
    if unknown:
        return _fail(f"unknown letter token id(s) {unknown} for {case_id!r}: not in source "
                     "decision_config letter_token_ids")
    active = [letters.index(token) for token in letter_ids]
    try:
        reference = np.asarray(readout.get("scores26_golden_fp32"), dtype=np.float64)
    except (TypeError, ValueError):
        reference = None
    if reference is None or reference.shape != (N_COLUMNS,) or not np.isfinite(reference).all():
        return _fail(f"reference scores26_golden_fp32 for {case_id!r} must be {N_COLUMNS} "
                     "finite values")
    scores = np.asarray(outputs[OUTPUT_NAME])
    if scores.dtype.kind not in "fiu":
        return _fail(f"scores for {case_id!r} must be numeric, got dtype {scores.dtype}")
    if scores.shape == (1, N_COLUMNS):
        scores = scores[0]
    elif scores.shape != (N_COLUMNS,):
        return _fail(f"scores for {case_id!r} must have shape (1, {N_COLUMNS}) (the pipeline "
                     f"row for B=1) or ({N_COLUMNS},) flat single read-out; batched B>1 rows "
                     f"are not supported by this verifier, got shape {scores.shape}")
    scores = scores.astype(np.float64)
    if not np.isfinite(scores).all():
        return _fail(f"scores for {case_id!r} contain nonfinite values")
    reference_probs = _softmax(reference[active] / temperature)
    candidate_probs = _softmax(scores[active] / temperature)
    prob_max_abs = float(np.abs(candidate_probs - reference_probs).max())
    order = np.argsort(-reference_probs, kind="stable")
    margin = 1.0 if len(active) == 1 else float(reference_probs[order[0]] - reference_probs[order[1]])
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
        failures.append(f"active-option probability max abs {prob_max_abs:.6g} > {PROB_MAX_ABS_MAX}")
    if top1_consistent is False:
        failures.append(f"top1 letter inconsistent while reference margin {margin:.6g} "
                        f"> {TOP1_MARGIN_TRIGGER}")
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


def compare(config, outputs):
    """Business gate for every selected case; exact cover of the given rows."""
    references = _references(config)
    letters, temperatures = _decision_config(config)
    return {case_id: _case_report(case_id, row, references.get(case_id), letters, temperatures)
            for case_id, row in outputs.items()}
