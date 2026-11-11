"""Standalone input generation for the dev.txt bypass route.

The data stage of the dev-case bypass flow (README "怎么用" -> run_dev.sh):
pure CPU, no weights forward, no NPU, no ``core``.  It consumes an external
dev.txt JSONL case file (business-provided e-commerce query-intent
classification cases, teacher-annotated by StartLux-Decision-v1.1-4B-AutoThink)
and persists everything the later bypass stages consume, under
``work/devcase/`` (generated, never committed):

* ``work/devcase/inputs/inputs.bin`` -- the packed graph-input tensors of the
  whole dev pool (the runtime request pool format);
* ``work/devcase/inputs/records.json`` -- the per-case tensor index in the
  C++ runtime's hard contract schema (startlux-split-records/1);
* ``work/devcase/inputs/cases.json`` -- the dev case index
  (startlux-dev-cases/1) with per-row business metadata: rows_meta carries
  the verify_business-compatible fields (question_type / letter_ids /
  letters_asked) PLUS the teacher-comparison fields (option_labels, targets,
  teacher_confidence, query);
* ``work/devcase/inputs/rows/<case>.npz`` -- the per-row OFFICIAL encodings at
  the TRUE lengths for decider/golden_dev.py.

Supported dev row shape (the only one this bypass serves, validated loudly):
ONE question per request of type ``choice`` with 2..26 criteria, state a
nonempty mapping (rendered by the official state_text as compact JSON),
targets a probability distribution over exactly the criteria keys.  Each
JSONL row becomes one B=1 pure-text case; case ids
``dev-intent-q{qi:02d}-r{ri}`` group rows by state content (qi =
first-appearance order, ri = variant order), because the dev file lists the
same query with permuted option orders (an option-order invariance pair).

The OFFICIAL case pool of generate_inputs.py is untouched: this is a
parallel, independent flow that shares only the official prompt pipeline
(vendored startlux_decide) and the graph-input packing path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
WORK = MODEL_DIR / "work"
DEV_WORK = WORK / "devcase"
INPUTS = DEV_WORK / "inputs"

DEFAULT_CASES = DEV_WORK / "source" / "dev.txt"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from model import modeling_multimodal as multimodal, unified_preprocess  # noqa: E402

OUTPUT_NAME = "scores"
QUESTION_TYPE = "choice"


def _load_dev_rows(path):
    """The dev.txt JSONL rows, strictly validated.

    Each row must carry exactly state / questions / targets / meta, one
    choice question whose criteria keys are nonempty strings (descriptions
    null or strings), and targets that sum to ~1 over exactly the criteria
    keys with finite nonnegative values.
    """
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path}: no JSONL rows")
    rows = []
    for lineno, line in enumerate(lines, 1):
        where = f"{path.name}:{lineno}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}: not valid JSON: {error}") from error
        if not isinstance(row, dict) or set(row) != {"state", "questions", "targets", "meta"}:
            raise ValueError(f"{where}: each row must carry exactly "
                             "state/questions/targets/meta")
        state = row["state"]
        if not isinstance(state, dict) or not state:
            raise ValueError(f"{where}: state must be a nonempty mapping")
        questions = row["questions"]
        if not isinstance(questions, dict) or len(questions) != 1:
            raise ValueError(f"{where}: exactly one question per row is supported")
        name, spec = next(iter(questions.items()))
        if not isinstance(spec, dict) or spec.get("type") != QUESTION_TYPE:
            raise ValueError(f"{where}: question {name!r} must be type {QUESTION_TYPE!r}")
        criteria = spec.get("criteria")
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 26:
            raise ValueError(f"{where}: question {name!r} needs 2..26 criteria")
        for key, value in criteria.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{where}: criterion ids must be nonempty strings")
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{where}: criterion {key!r} description must be "
                                 "a string or null")
        if not isinstance(spec.get("instructions"), str) or not spec["instructions"].strip():
            raise ValueError(f"{where}: instructions are required")
        targets = row["targets"]
        if not isinstance(targets, dict) or set(targets) != {name}:
            raise ValueError(f"{where}: targets must carry exactly question {name!r}")
        scores = targets[name]
        if not isinstance(scores, dict) or set(scores) != set(criteria):
            raise ValueError(f"{where}: targets must map exactly the criteria of {name!r}")
        total = 0.0
        for key, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not np.isfinite(value) or value < 0:
                raise ValueError(f"{where}: target {key!r} must be a finite "
                                 "nonnegative number")
            total += float(value)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"{where}: targets sum to {total!r}, not ~1")
        meta = row["meta"]
        if not isinstance(meta, dict):
            raise ValueError(f"{where}: meta must be a mapping")
        confidence = meta.get("teacher_confidence")
        if confidence is not None and (isinstance(confidence, bool)
                                        or not isinstance(confidence, (int, float))
                                        or not np.isfinite(confidence)):
            raise ValueError(f"{where}: teacher_confidence must be a finite number or null")
        rows.append({"state": state, "question": name, "spec": spec,
                     "criteria": criteria,
                     "targets": {key: float(value) for key, value in scores.items()},
                     "teacher_confidence": confidence, "where": where})
    return rows, hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_dev_pool(folder, letters, rows):
    """The dev case pool: one B=1 pure-text case per dev.txt row, rendered
    through the SAME official prompt pipeline as any request
    (from_systemone -> messages -> plain_prompt)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    if str(MODEL_DIR / "decider" / "vendor") not in sys.path:
        sys.path.insert(0, str(MODEL_DIR / "decider" / "vendor"))
    from startlux_decide.prompt import from_systemone, messages, plain_prompt

    def meta_entry(case_id, labels, targets, confidence, query):
        active = [int(t) for t in letters[:len(labels)]]
        return {"case_id": case_id, "question_type": QUESTION_TYPE,
                "letter_ids": active, "letters_asked": len(active),
                "option_labels": list(labels), "targets": dict(targets),
                "teacher_confidence": confidence, "query": query}

    query_index, variant_count, cases = {}, {}, []
    for row in rows:
        key = json.dumps(row["state"], ensure_ascii=False, sort_keys=True)
        if key not in query_index:
            query_index[key] = len(query_index)
        qi = query_index[key]
        ri = variant_count.get(qi, 0)
        variant_count[qi] = ri + 1
        case_id = f"dev-intent-q{qi:02d}-r{ri}"
        rendered = from_systemone(row["state"], row["spec"])
        prompt = plain_prompt(tokenizer, messages(rendered))
        cases.append({"id": case_id, "samples": 1, "business": True, "payload": {
            "rows": [{"prompt": prompt}],
            "rows_meta": [meta_entry(case_id, list(row["criteria"]), row["targets"],
                                      row["teacher_confidence"],
                                      row["state"].get("query"))]}})
    return cases


def _save_row_encodings(case_id, encoded):
    """Persist the per-row OFFICIAL encodings (true lengths) for golden_dev.py."""
    arrays = {}
    for index, row in enumerate(encoded):
        arrays[f"row{index}_input_ids"] = row["input_ids"].numpy()
        for key in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
            if key in row:
                arrays[f"row{index}_{key}"] = row[key].numpy()
    path = INPUTS / "rows" / f"{case_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez(stream, **arrays)


def main():
    parser = argparse.ArgumentParser(description="Generate the dev.txt bypass inputs "
                                                 "(work/devcase/inputs)")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES,
                        help="the dev JSONL case file (default: %(default)s)")
    args = parser.parse_args()
    if not args.cases.is_file():
        raise SystemExit(f"generate_dev_inputs: case file {args.cases} does not exist; "
                         "place dev.txt there or pass --cases (run_dev.sh downloads it "
                         "from the business URL)")

    folder = multimodal.source_folder()
    decision = multimodal.decision_config(folder)
    letters = decision["letter_token_ids"]
    vision_order = [node["name"] for node in multimodal.vision_io_spec()["inputs"]]
    text_order = [node["name"] for node in multimodal.text_io_spec()["inputs"]
                  if node["name"] != "image_features"]

    rows, digest = _load_dev_rows(args.cases)
    preprocessor = unified_preprocess.OfficialPreprocessor(folder)
    cases = build_dev_pool(folder, letters, rows)

    INPUTS.mkdir(parents=True, exist_ok=True)
    records, packed_cases = [], {}
    with (INPUTS / "inputs.bin").open("wb") as stream:
        for case in cases:
            payload = case["payload"]
            encoded = preprocessor.encode_payload_rows(payload["rows"])
            _save_row_encodings(case["id"], encoded)
            has_images = any("image_grid_thw" in row for row in encoded)
            arrays = preprocessor.build_graph_inputs(
                encoded, vocab_size=decision["vocab_size"])
            vision_tensors, text_tensors = [], []
            for name in vision_order:
                array = arrays[name]
                data = array.tobytes(order="C")
                vision_tensors.append({"name": name, "dtype": str(array.dtype),
                                       "shape": list(array.shape),
                                       "offset": stream.tell(), "bytes": len(data)})
                stream.write(data)
            for name in text_order:
                array = arrays[name]
                data = array.tobytes(order="C")
                text_tensors.append({"name": name, "dtype": str(array.dtype),
                                     "shape": list(array.shape),
                                     "offset": stream.tell(), "bytes": len(data)})
                stream.write(data)
            records.append({"id": case["id"], "samples": case["samples"],
                            "has_images": has_images,
                            "vision_tensors": vision_tensors,
                            "text_tensors": text_tensors})
            packed_cases[case["id"]] = case
            print(f"generate_dev_inputs: {case['id']} S="
                  f"{int(encoded[0]['input_ids'].shape[1])} options="
                  f"{case['payload']['rows_meta'][0]['letters_asked']}")

    precision_ids = [case["id"] for case in cases]
    bench_ids = [case["id"] for case in cases if case["business"]]
    (INPUTS / "cases.json").write_text(json.dumps({
        "schema": "startlux-dev-cases/1",
        "source_ref": str(folder),
        "source": {"path": str(args.cases.resolve()), "sha256": digest,
                   "lines": len(rows)},
        "pools": {"precision": precision_ids, "bench": bench_ids},
        "cases": packed_cases,
    }, indent=1) + "\n")
    (INPUTS / "records.json").write_text(json.dumps({
        "schema": "startlux-split-records/1",
        "file": str((INPUTS / "inputs.bin").resolve()),
        "output_name": OUTPUT_NAME,
        "cases": records,
    }, indent=1) + "\n")

    rows_total = sum(record["samples"] for record in records)
    queries = len({case_id.rsplit("-r", 1)[0] for case_id in packed_cases})
    print(f"generate_dev_inputs: {len(records)} cases, {rows_total} rows, "
          f"{queries} distinct queries; source sha256 {digest[:12]}; inputs: {INPUTS}")


if __name__ == "__main__":
    main()
