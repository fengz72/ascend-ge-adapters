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
  fixture image (the official example image is not distributed).

Case pool (fixed, no configuration): ONLY the five official published
example cases of StartLuxLabs/StartLux-Decision (examples @ main):

* official-ticket-b3-r0: the client.py / request.sh support ticket asked its
  three typed questions in ONE request (B3, one row per question);
* official-ticket-{team,urgent,severity}-r0: the same three questions as
  single-row requests;
* official-image-damaged-r0: the image_request.py photo noul question
  ("Photo taken at delivery: <image>") on the self-regenerating red fixture
  image (the official example image is not distributed with the repo).

All five are business cases (bench pool = precision pool); the golden,
per-row comparison and business gate treat them like any request.  Graph
bounds (B<=4, S<=2048, at most 2 global image slots, grid exactly [1, 16, 16])
are enforced at encode time by multimodal.build_multimodal_batch -- over-limit
requests are rejected, never truncated, and every case runs at its TRUE length
(width = max row length, no padding floor).  The S bound is the model's own
config.text_config.max_position_embeddings; the image-slot and grid constants
come from their single source (multimodal.SLOTS / vision.GRID_THW); this file
invents no copies.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]
WORK = MODEL_DIR / "work"
INPUTS = WORK / "inputs"
WORK_FIXTURES = WORK / "fixtures"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from model import modeling_multimodal as multimodal, unified_preprocess

SEED_RED = 20261031
IMAGE_SIZE = (256, 256)                 # official processing -> grid [1, 16, 16]

# The official published inference examples (StartLuxLabs/StartLux-Decision
# examples/client.py, request.sh, image_request.py @ main): the canonical
# business-case shapes this model is served with.
OFFICIAL_TICKET_STATE = {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."}
OFFICIAL_TICKET_QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "Payments and refunds", "shipping": "Delivery",
                          "technical": "App and login"}},
    "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
    "severity": {"type": "score", "instructions": "How severe is the impact?",
                 "criteria": ["cosmetic", "annoying", "blocks the customer"]},
}
OFFICIAL_IMAGE_STATE = "Photo taken at delivery: <image>"
OFFICIAL_IMAGE_QUESTION = {"type": "noul", "instructions": "Is the parcel damaged?"}


def _active_letters(letters, question_type, count):
    """rows_meta active columns: N options -> the first N letters (the official
    renderer maps criteria/score levels to letters in order; noul is always
    true/false = 2)."""
    if question_type == "noul":
        count = 2
    return [int(t) for t in letters[:count]]


OUTPUT_NAME = "scores"
N_COLUMNS = 26



# ------------------------------------------------------- self-regenerating PPMs

RED_IMAGE = dict(dominant=(210, 40, 40),
                 patches=[((36, 30, 70, 196), (240, 235, 225)),
                          ((150, 90, 60, 80), (35, 35, 215))],
                 seed=SEED_RED)


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


def ensure_fixtures():
    """(Re)generate the deterministic red fixture and return its path.

    The picture is a pure function of its seed, so an existing file with
    different bytes is overwritten back to the canonical content.
    """
    WORK_FIXTURES.mkdir(parents=True, exist_ok=True)
    path = WORK_FIXTURES / "fixture-red-256.ppm"
    canonical = _ppm_bytes(RED_IMAGE)
    if not path.is_file() or path.read_bytes() != canonical:
        path.write_bytes(canonical)
    return path


# --------------------------------------------------------------- case pool

def build_pool(folder, letters):
    """The case pool: ONLY the five official published examples
    (examples/{client,request,image_request}.py of StartLuxLabs/StartLux-Decision
    @ main), rendered through the same official prompt pipeline as any request:

    * the support-ticket request of client.py / request.sh: one state asked
      three typed questions (team choice / urgent noul / severity score) --
      delivered as the official single-request form (one B3 case, one row per
      question) and as three B1 cases (one question per request);
    * the image_request.py photo question: state "Photo taken at delivery:
      <image>" with the noul damaged question on the self-regenerating red
      fixture image (the official example image is not distributed).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    fixture = ensure_fixtures()
    if str(MODEL_DIR / "decider" / "vendor") not in sys.path:
        sys.path.insert(0, str(MODEL_DIR / "decider" / "vendor"))
    from startlux_decide.images import place_images
    from startlux_decide.prompt import from_systemone, messages, plain_prompt

    def render(state, spec, n_images=0):
        row = from_systemone(place_images(state, n_images) if n_images else state, spec)
        return plain_prompt(tokenizer, messages(row))

    def meta(case_id, question_type, options):
        active = _active_letters(letters, question_type, options)
        return {"case_id": case_id, "question_type": question_type,
                "letter_ids": active, "letters_asked": len(active)}

    ticket_prompts = {
        name: render(OFFICIAL_TICKET_STATE, spec)
        for name, spec in OFFICIAL_TICKET_QUESTIONS.items()}
    image_prompt = render(OFFICIAL_IMAGE_STATE, OFFICIAL_IMAGE_QUESTION, n_images=1)

    def rel(path):
        return str(Path(path).relative_to(ROOT))

    option_counts = {"team": 3, "urgent": 0, "severity": 3}
    cases = [
        {"id": "official-ticket-b3-r0", "samples": 3, "business": True, "payload": {
            "rows": [{"prompt": ticket_prompts[name]} for name in OFFICIAL_TICKET_QUESTIONS],
            "rows_meta": [meta("official-ticket-b3-r0", spec["type"], option_counts[name])
                          for name, spec in OFFICIAL_TICKET_QUESTIONS.items()]}},
    ]
    for name, spec in OFFICIAL_TICKET_QUESTIONS.items():
        case_id = f"official-ticket-{name}-r0"
        cases.append({"id": case_id, "samples": 1, "business": True, "payload": {
            "rows": [{"prompt": ticket_prompts[name]}],
            "rows_meta": [meta(case_id, spec["type"], option_counts[name])]}})
    cases.append({"id": "official-image-damaged-r0", "samples": 1, "business": True, "payload": {
        "rows": [{"prompt": image_prompt, "images": [rel(fixture)]}],
        "rows_meta": [meta("official-image-damaged-r0", "noul", 0)]}})
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


def main():
    folder = multimodal.source_folder()
    decision = multimodal.decision_config(folder)
    order = [node["name"] for node in multimodal.io_spec()["inputs"]]

    preprocessor = unified_preprocess.OfficialPreprocessor(folder)
    cases = build_pool(folder, decision["letter_token_ids"])

    INPUTS.mkdir(parents=True, exist_ok=True)
    records, packed_cases = [], {}
    with (INPUTS / "inputs.bin").open("wb") as stream:
        for case in cases:
            payload = case["payload"]
            samples = case["samples"]
            encoded = preprocessor.encode_payload_rows(payload["rows"])
            _save_row_encodings(case["id"], encoded)
            arrays = preprocessor.build_graph_inputs(
                encoded, vocab_size=decision["vocab_size"])
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
