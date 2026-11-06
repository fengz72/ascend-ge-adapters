"""Standalone input generation for the StartLux unified ONE-AIR route.

The data stage of the standalone shell flow (docs/standalone-runtime-plan.md):
pure CPU, no weights, no NPU, no ``core``.  It runs the OFFICIAL preprocessing
exactly once and persists everything the later stages consume:

* ``work/inputs/inputs.bin``  -- the packed graph-input tensors of the whole
  pool (the runtime request pool format);
* ``work/inputs/records.json`` -- the per-case tensor index (id, samples,
  tensors with name/dtype/shape/offset/bytes, output capacities) that
  export_air.py embeds into infer/bench request.json;
* ``work/inputs/cases.json`` -- the case pool definition (payload rows and the
  per-row business metadata rows_meta the business gate reads) with the
  precision / bench pool split;
* ``work/inputs/rows/<case>.npz`` -- the per-row OFFICIAL encodings at the
  TRUE lengths (input_ids / pixel_values / image_grid_thw /
  mm_token_type_ids): golden.py consumes these saved results and never
  re-tokenizes or re-preprocesses;
* ``work/fixtures/*.ppm`` -- the self-regenerating deterministic 256x256
  fixtures (byte-identical to the validated P6 production fixtures).

Case pool (fixed, no configuration):

* the SEVEN canonical pure-text read-outs of decider/fixtures/p2b-readouts.json
  (pre-encoded official rows, one B=1 case each);
* FOUR real image cases on the self-regenerating fixtures with
  official-rendered prompts (image-red, image-blue, image-two,
  mixed-text-image B2);
* the B4 mixed-effective-length boundary case (rows of true lengths
  1/65/129/2048, decider/fixtures/b4-long-rows.json) -- an ordinary member of the
  PRECISION pool (it participates in golden and the per-sample comparison like
  every other case); it carries no business metadata and is excluded from the
  bench pool, matching the P8 formal-11 benchmark input set.

The 11 business cases form the bench pool (the P8 formal 11); the precision
pool is those 11 plus b4-long.  Graph bounds (B<=4, S<=2048, minimum physical
width 256, at most 2 global image slots) are enforced here at encode time by
the shared validated validator (multimodal.build_multimodal_batch /
validate_batch_arrays) -- over-limit requests are rejected, never truncated.
The boundary constants come from the modules' single source
(multimodal.MAX_BATCH / MAX_SEQ_LEN / startlux_npu_adapter._params defaults);
this file invents no copies.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
WORK = MODEL_DIR / "work"
INPUTS = WORK / "inputs"
FIXTURES = MODEL_DIR / "decider" / "fixtures"
WORK_FIXTURES = WORK / "fixtures"

SEED_RED = 20261031
SEED_BLUE = 20261032
IMAGE_SIZE = (256, 256)                 # official processing -> grid [1, 16, 16]
QUESTIONS = {
    "dominant": {
        "instructions": "Which color occupies the largest area of the image?",
        "criteria": {"red": "the red areas are the largest",
                     "green": "the green areas are the largest",
                     "blue": "the blue areas are the largest"},
    }
}
STATE_ONE = ("A synthetic 256x256 RGB test pattern generated for pipeline "
             "validation.\n<image>")
STATE_AB = ("Two synthetic 256x256 RGB test patterns: the first is red-dominant, "
            "the second is blue-dominant.\n<image>\n<image>")
MIXED_TEXT_STATE = "A short plain-text evidence block for the mixed batch case."


def _load_module(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(MODEL_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    # register before exec: dynamo resolves traced functions by __module__
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


base = _load_module("startlux_npu_adapter_for_native", "model/startlux_npu_adapter.py")
unified_preprocess = _load_module("startlux_unified_preprocess", "model/unified_preprocess.py")
multimodal = _load_module("startlux_multimodal_for_unified", "model/multimodal.py")

OUTPUT_NAME = "scores"
N_COLUMNS = 26


def _config(source_ref):
    """The minimal framework-free config dict the reused modules expect.

    Model params stay EMPTY on purpose: the single source of the accepted
    values (max_seq_len 2048 / compiled_seq_len 256 / chunk_size 64 /
    max_batch 4 / gdn_route native / fixtures_file) is the
    startlux_npu_adapter._params default table; no constants are copied here.
    """
    return {"model": {"name": "startlux-decision-v1.1-0.8b-instant", "params": {}},
            "source": {"kind": "transformers",
                       "loader": "Qwen3_5ForConditionalGeneration",
                       "ref": str(Path(source_ref).resolve())}}


def _source_ref():
    value = os.environ.get("STARTLUX_MODEL_DIR")
    if not value or not value.strip():
        raise SystemExit("generate_inputs: STARTLUX_MODEL_DIR is not set or is empty")
    path = Path(value).resolve()
    if not path.is_dir():
        raise SystemExit(f"generate_inputs: STARTLUX_MODEL_DIR {path} is not a directory")
    return path


# ------------------------------------------------------- self-regenerating PPMs

RED_IMAGE = dict(dominant=(210, 40, 40),
                 patches=[((36, 30, 70, 196), (240, 235, 225)),
                          ((150, 90, 60, 80), (35, 35, 215))],
                 seed=SEED_RED)
BLUE_IMAGE = dict(dominant=(35, 45, 215),
                  patches=[((40, 40, 90, 60), (230, 230, 240)),
                           ((160, 130, 56, 90), (215, 40, 40))],
                  seed=SEED_BLUE)


def fixture_rgb(dominant, patches, seed):
    """A deterministic 256x256 solid-dominant RGB pattern (pure function of seed)."""
    rng = np.random.default_rng(seed)
    img = np.full((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dominant, dtype=np.int16)
    for (top, left, height, width), color in patches:
        img[top:top + height, left:left + width] = color
    img = img + rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def _ppm_bytes(spec):
    import io

    buffer = io.BytesIO()
    image = fixture_rgb(**spec)
    height, width, _ = image.shape
    buffer.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
    buffer.write(np.ascontiguousarray(image).tobytes())
    return buffer.getvalue()


def ensure_fixtures(directory=WORK_FIXTURES):
    """(re)generate the deterministic fixtures; returns {name: absolute path}.

    The pictures are pure functions of their seeds, so regeneration is
    byte-identical to the validated P6 production fixtures; an existing file
    with different bytes is overwritten back to the canonical content.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, spec in (("fixture-red-256.ppm", RED_IMAGE),
                       ("fixture-blue-256.ppm", BLUE_IMAGE)):
        path = directory / name
        canonical = _ppm_bytes(spec)
        if not path.is_file() or path.read_bytes() != canonical:
            path.write_bytes(canonical)
        out[name.removesuffix(".ppm")] = path
    return out


# --------------------------------------------------------------- case pool

def _text_cases():
    """The seven canonical pure-text read-outs as B1 pre-encoded rows."""
    data = json.loads((FIXTURES / "p2b-readouts.json").read_text())
    readouts = data["readouts"]
    if not readouts:
        raise ValueError("fixture has no readouts")
    if len({r["case_id"] for r in readouts}) != len(readouts):
        raise ValueError("duplicate readout case ids in fixture")
    cases = []
    for readout in readouts:
        length = len(readout["input_ids"])
        if readout["seq_len"] != length or readout["position_ids"] != list(range(length)) \
                or readout["actual_seq_lengths"] != [length] \
                or len(readout["scores26_golden_fp32"]) != 26:
            raise ValueError(f"fixture readout {readout['case_id']} is inconsistent")
        cases.append({"id": f"text-{readout['case_id']}", "samples": 1, "business": True,
                      "payload": {
                          "rows": [{"input_ids": [int(t) for t in readout["input_ids"]]}],
                          "rows_meta": [{
                              "case_id": f"text-{readout['case_id']}",
                              "question_type": readout["question_type"],
                              "letter_ids": [int(t) for t in readout["letter_ids"]],
                              "letters_asked": int(readout["letters_asked"]),
                          }]}})
    return cases


def _render_official_prompts(folder, tokenizer, states):
    """The official one-pass prompts (official renderer, real tokenizer, no weights)."""
    if str(MODEL_DIR / "decider" / "vendor") not in sys.path:
        sys.path.insert(0, str(MODEL_DIR / "decider" / "vendor"))
    from startlux_decide.images import IMAGE_PLACEHOLDER, place_images
    from startlux_decide.prompt import from_systemone, messages, plain_prompt

    rendered = {}
    for key, (state, n_images) in states.items():
        row = from_systemone(place_images(state, n_images), QUESTIONS["dominant"])
        prompt = plain_prompt(tokenizer, messages(row))
        if prompt.count(IMAGE_PLACEHOLDER) != n_images:
            raise ValueError(f"the official renderer wrote {prompt.count(IMAGE_PLACEHOLDER)} "
                             f"placeholders for {n_images} images")
        rendered[key] = prompt
    return rendered


def _image_cases(folder):
    """The real image cases on the self-regenerating fixtures (needs the tokenizer)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    letters = json.loads((folder / "decision_config.json").read_text())["letter_token_ids"]
    fixtures = ensure_fixtures()
    prompts = _render_official_prompts(folder, tokenizer, {
        "one": (STATE_ONE, 1), "ab": (STATE_AB, 2),
        "text": (MIXED_TEXT_STATE, 0)})
    active = [int(t) for t in letters[:3]]          # red / green / blue options

    def meta(case_id):
        return {"case_id": case_id, "question_type": "choice",
                "letter_ids": list(active), "letters_asked": len(active)}

    def rel(path):
        return str(Path(path).relative_to(ROOT))

    return [
        {"id": "image-red-r0", "samples": 1, "business": True, "payload": {
            "rows": [{"prompt": prompts["one"], "images": [rel(fixtures["fixture-red-256"])]}],
            "rows_meta": [meta("image-red-r0")]}},
        {"id": "image-blue-r0", "samples": 1, "business": True, "payload": {
            "rows": [{"prompt": prompts["one"], "images": [rel(fixtures["fixture-blue-256"])]}],
            "rows_meta": [meta("image-blue-r0")]}},
        {"id": "image-two-a-b-r0", "samples": 1, "business": True, "payload": {
            "rows": [{"prompt": prompts["ab"], "images": [rel(fixtures["fixture-red-256"]),
                                                          rel(fixtures["fixture-blue-256"])]}],
            "rows_meta": [meta("image-two-a-b-r0")]}},
        {"id": "mixed-text-image-b2-r0", "samples": 2, "business": True, "payload": {
            "rows": [{"prompt": prompts["text"]},
                     {"prompt": prompts["one"],
                      "images": [rel(fixtures["fixture-red-256"])]}],
            "rows_meta": [meta("mixed-text-image-b2-r0"), meta("mixed-text-image-b2-r0")]}},
    ]


def _b4_long_case():
    """The B4 mixed-effective-length boundary case (pure official text rows).

    Pre-encoded rows of true lengths [1, 65, 129, 2048] taken from the P8
    boundary fixture.  A shape-coverage member of the precision pool, not a
    business case: no rows_meta, excluded from the bench pool.
    """
    data = json.loads((FIXTURES / "b4-long-rows.json").read_text())
    rows = data["rows"]
    if [len(row) for row in rows] != [1, 65, 129, 2048]:
        raise ValueError(f"b4-long-rows.json row lengths {[len(r) for r in rows]} "
                         "!= [1, 65, 129, 2048]")
    return {"id": data["case_id"], "samples": len(rows), "business": False,
            "payload": {"rows": [{"input_ids": [int(t) for t in row]} for row in rows]}}


def build_pool(folder):
    """The unified case pool: pre-encoded text cases FIRST, then the image
    cases (the P8 formal 11 order), then the b4-long boundary case."""
    cases = _text_cases()
    cases.extend(_image_cases(folder))
    cases.append(_b4_long_case())
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate case ids in the pool: {ids}")
    return cases


# ------------------------------------------------------- encode + persist

def _save_row_encodings(case_id, encoded):
    """Persist the per-row OFFICIAL encodings (true lengths) for golden.py."""
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
    return path


def _validate_against_io(arrays, io_inputs):
    expected = {node["name"]: node for node in io_inputs}
    if set(arrays) != set(expected):
        raise ValueError(f"encoded input names {sorted(arrays)} differ from the graph "
                         f"inputs {sorted(expected)}")
    for name, node in expected.items():
        array = arrays[name]
        dtype = str(array.dtype)
        if dtype != node["dtype"] or array.ndim != len(node["shape"]):
            raise ValueError(f"input contract mismatch: {name} {dtype} "
                             f"{array.shape} vs {node['dtype']} {node['shape']}")
        for axis, (size, declared) in enumerate(zip(array.shape, node["shape"])):
            if declared >= 0 and declared != size:
                raise ValueError(f"input contract mismatch: {name} axis {axis} "
                                 f"{size} != static {declared}")
        if any(size < 1 for size in array.shape):
            raise ValueError(f"empty input dimension in {name}: {array.shape}")


def main():
    folder = _source_ref()
    config = _config(folder)
    params = base._params(config)
    decision = base._decision_config(config)
    if len(decision["letter_token_ids"]) != N_COLUMNS:
        raise ValueError("decision_config letter_token_ids must carry 26 ids")
    caps = {"max_batch": params["max_batch"], "max_seq_len": params["max_seq_len"],
            "min_width": params["compiled_seq_len"]}
    io_inputs = multimodal.describe(config)["io"]["inputs"]
    order = [node["name"] for node in io_inputs]

    preprocessor = unified_preprocess.OfficialPreprocessor(folder)
    cases = build_pool(folder)

    INPUTS.mkdir(parents=True, exist_ok=True)
    records, packed_cases = [], {}
    with (INPUTS / "inputs.bin").open("wb") as stream:
        for case in cases:
            payload = case["payload"]
            samples = case["samples"]
            if len(payload["rows"]) != samples:
                raise ValueError(f"case {case['id']} samples={samples} but the payload "
                                 f"carries {len(payload['rows'])} row(s)")
            encoded = preprocessor.encode_payload_rows(payload["rows"])
            _save_row_encodings(case["id"], encoded)
            arrays = preprocessor.build_graph_inputs(
                encoded, vocab_size=decision["vocab_size"],
                max_batch=caps["max_batch"], max_seq_len=caps["max_seq_len"],
                min_width=caps["min_width"])
            _validate_against_io(arrays, io_inputs)
            tensors = []
            for name in order:
                array = arrays[name]
                data = array.tobytes(order="C")
                tensors.append({"name": name, "dtype": str(array.dtype),
                                "shape": list(array.shape),
                                "offset": stream.tell(), "bytes": len(data)})
                stream.write(data)
            records.append({"id": case["id"], "samples": samples, "tensors": tensors,
                            "output_capacities": {OUTPUT_NAME: samples * N_COLUMNS * 4}})
            packed_cases[case["id"]] = case

    bench_ids = [case["id"] for case in cases if case["business"]]
    precision_ids = [case["id"] for case in cases]
    (INPUTS / "cases.json").write_text(json.dumps({
        "schema": "startlux-standalone-cases/1",
        "source_ref": str(folder),
        "pools": {"precision": precision_ids, "bench": bench_ids},
        "cases": packed_cases,
    }, indent=1) + "\n")
    (INPUTS / "records.json").write_text(json.dumps({
        "schema": "startlux-standalone-records/1",
        "file": str((INPUTS / "inputs.bin").resolve()),
        "output_name": OUTPUT_NAME,
        "cases": records,
    }, indent=1) + "\n")

    rows = sum(record["samples"] for record in records)
    print(f"generate_inputs: {len(records)} cases, {rows} rows, "
          f"precision pool {len(precision_ids)} / bench pool {len(bench_ids)}; "
          f"inputs: {INPUTS}")


if __name__ == "__main__":
    main()
