"""Minimal text-only input generation for the StartLux-Decision text pipeline.

Consumes the business dev.txt JSONL case file (one teacher-annotated choice
question per row) and writes the input pool under work/inputs/:

* ``cases.json``   the per-case index (schema ``ge-cases/1``: the ``inputs``
  array is bound positionally to the graph inputs, dtype/shape are read by the
  runtime from the npy headers, never from the JSON);
* ``<case_id>/``   one directory per case holding the two graph-input tensors
  (``input_ids.npy``, ``position_ids.npy``);
* ``readout.json`` the per-case read-out metadata for downstream read-out
  (schema ``startlux-text-readout/1``: labels = the criteria keys in prompt
  order, letters_asked = K, temperature = the official choice temperature).

Each JSONL row becomes ONE B=1 case at its TRUE length -- the B=1 contract is
pinned by the exported graph itself (io-spec axis 0 static) and the case
format cannot express padding: the laid-out width IS the row's true length,
there is no actual_seq_lengths tensor.  The official prompt rendering
(vendored ``startlux_decide.prompt``: from_systemone -> messages ->
plain_prompt, chat template with thinking off) and the official tokenization
recipe (``tokenizer.encode(prompt, add_special_tokens=False)``) produce
``input_ids``; the text-only MRoPE ``position_ids`` are the arange planes on
all three axes (verified equal to the official ``get_rope_index`` output of
the pinned checkpoint on the dev pool).  Steps of this pipeline
(generate_inputs -> golden -> export_air -> ge_execute) never import each
other; they interact only through the files under work/.  Paths default to
fixtures/dev.txt and work/inputs; ``--cases`` / ``--output-dir`` override them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parents[1]

if str(ADAPTER_DIR / "vendor") not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR / "vendor"))

QUESTION_TYPE = "choice"
INPUT_NAMES = ("input_ids", "position_ids")   # 顺序 = 图输入顺序


def weights_dir():
    """STARTLUX_MODEL_DIR resolved once, loudly."""
    value = os.environ.get("STARTLUX_MODEL_DIR")
    if not value or not value.strip():
        raise ValueError("STARTLUX_MODEL_DIR is not set or is empty")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"STARTLUX_MODEL_DIR {path} is not a directory")
    return path


def text_bounds(folder):
    """vocab_size / max_position_embeddings of the text trunk from config.json."""
    model_config = json.loads((Path(folder) / "config.json").read_text())
    text_config = model_config.get("text_config") or {}
    vocab = text_config.get("vocab_size", model_config.get("vocab_size"))
    max_position_embeddings = text_config.get(
        "max_position_embeddings", model_config.get("max_position_embeddings"))
    if not isinstance(vocab, int) or vocab < 1 \
            or not isinstance(max_position_embeddings, int) or max_position_embeddings < 1:
        raise ValueError(f"cannot read vocab_size / max_position_embeddings from "
                         f"{folder / 'config.json'}")
    return vocab, max_position_embeddings


def choice_temperature(folder):
    """The official choice read-out temperature from decision_config.json."""
    decision = json.loads((Path(folder) / "decision_config.json").read_text())
    value = (decision.get("temperature_by_type") or {}).get(QUESTION_TYPE)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not np.isfinite(value) or value <= 0:
        raise ValueError(f"cannot read a positive finite {QUESTION_TYPE!r} temperature "
                         f"from {folder / 'decision_config.json'} temperature_by_type")
    return float(value)


def _load_dev_rows(path):
    """The dev.txt JSONL rows, strictly validated (the parent flow's contract).

    Each row must carry exactly state / questions / targets / meta, one choice
    question whose criteria keys are nonempty strings (descriptions null or
    strings), and targets that sum to ~1 over exactly the criteria keys with
    finite nonnegative values.  targets/meta are validated for file integrity
    only -- this minimal flow has no business gate stage consuming them.
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
        if not isinstance(row["meta"], dict):
            raise ValueError(f"{where}: meta must be a mapping")
        rows.append({"state": state, "spec": spec, "where": where})
    return rows, hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generate the text pipeline inputs "
                                                 "(work/inputs)")
    parser.add_argument("--cases", type=Path,
                        default=ADAPTER_DIR / "fixtures" / "dev.txt",
                        help="the dev JSONL case file (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path,
                        default=ADAPTER_DIR / "work" / "inputs",
                        help="directory for cases.json, the per-case npy directories "
                             "and readout.json (default: %(default)s)")
    args = parser.parse_args()
    if not args.cases.is_file():
        raise SystemExit(f"generate_inputs: case file {args.cases} does not exist; place "
                         "dev.txt there or pass --cases")

    from transformers import AutoTokenizer
    from startlux_decide.prompt import from_systemone, messages, plain_prompt

    folder = weights_dir()
    vocab, max_position_embeddings = text_bounds(folder)
    temperature = choice_temperature(folder)
    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    rows, digest = _load_dev_rows(args.cases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("inputs.bin", "records.json"):
        (args.output_dir / stale).unlink(missing_ok=True)
    cases, readout_cases, query_index, variant_count = [], [], {}, {}
    for row in rows:
        key = json.dumps(row["state"], ensure_ascii=False, sort_keys=True)
        qi = query_index.setdefault(key, len(query_index))
        ri = variant_count.get(qi, 0)
        variant_count[qi] = ri + 1
        case_id = f"dev-intent-q{qi:02d}-r{ri}"

        rendered = from_systemone(row["state"], row["spec"])
        prompt = plain_prompt(tokenizer, messages(rendered))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError(f"{row['where']}: the prompt tokenized to 0 tokens")
        if min(ids) < 0 or max(ids) >= vocab:
            raise ValueError(f"{row['where']}: input_ids outside the vocabulary "
                             f"bounds [0, {vocab})")
        length = len(ids)
        if length > max_position_embeddings:
            raise ValueError(f"{row['where']}: prompt length {length} exceeds the "
                             f"model's max_position_embeddings {max_position_embeddings}")

        arrays = [np.asarray([ids], dtype=np.int64),
                  np.tile(np.arange(length, dtype=np.int64)[None, None, :], (3, 1, 1))]
        case_dir = args.output_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)      # np.save 不建父目录
        inputs = []
        for name, array in zip(INPUT_NAMES, arrays):
            np.save(case_dir / f"{name}.npy", array)
            inputs.append({"name": name, "path": f"{case_id}/{name}.npy"})
        cases.append({"id": case_id, "samples": 1, "inputs": inputs})
        labels = list(row["spec"]["criteria"].keys())
        readout_cases.append({"id": case_id, "labels": labels,
                              "letters_asked": len(labels),
                              "temperature": temperature})
        print(f"generate_inputs: {case_id} S={length}")

    (args.output_dir / "cases.json").write_text(json.dumps({"cases": cases}, indent=1) + "\n")
    (args.output_dir / "readout.json").write_text(json.dumps({
        "schema": "startlux-text-readout/1",
        "temperature_source": "decision_config.json temperature_by_type[choice]",
        "cases": readout_cases,
    }, indent=1) + "\n")
    queries = len(query_index)
    print(f"generate_inputs: {len(cases)} cases, {queries} distinct queries; source "
          f"sha256 {digest[:12]}; inputs: {args.output_dir}")


if __name__ == "__main__":
    main()
