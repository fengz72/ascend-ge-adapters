"""Case generation for the StartLux unified multimodal ONE-AIR route (P6 production).

Pure CPU, no model execution, no NPU.  Two pools:

* the SEVEN canonical pure-text read-outs of the P2b fixture
  (model.params.fixtures_file): each becomes one B=1 case whose payload row is
  the fixture's official input_ids (a pre-encoded official text row -- the
  graph encode needs no tokenizer and no Pillow for these);
* REAL image cases built from SELF-REGENERATING fixtures: deterministic
  256x256 solid-color-dominant PPM pictures written by this module into
  data.params.fixtures_directory (default
  work/startlux-validation/p6-production/fixtures) -- red-dominant and
  blue-dominant -- plus the official image-question prompts.  The prompts are
  rendered by the OFFICIAL package (startlux_decide.prompt.from_systemone +
  place_images + plain_prompt on the folder's real AutoTokenizer; no weights),
  so every marker, layout and thinking-off suffix is the trained format.  The
  payload carries the prompt TEXT and the local image PATHS only -- the pixel
  tensors are produced at encode time by the official preprocessing, so no
  giant pixel JSON and no duplicated preprocessing source ride in the case.

Cases: image-red (single red, B1), image-blue (single blue, B1), image-two
(red + blue in one request, B1), mixed-text-image (one text choice row + one
single-red row, B2).  The FIRST case of the pool is always a pre-encoded text
case so a plain CLI build (which encodes the first case before the exporter
prefers the deterministic trace example) never needs Pillow.

``payload.rows_meta`` carries the OPTIONAL per-row business metadata (case_id,
question_type, letter_ids of the active options, letters_asked) that the
business verifier (unified_verify.py) reads; the graph encode never reads it.
No golden predictions live anywhere here: goldens come ONLY from the official
CPU FP32 baseline of the unified adapter.

Fixture determinism: the images are pure functions of fixed seeds
(solid-dominant base color + small deterministic patches + +-3 noise from
np.random.default_rng); regenerating always yields byte-identical PPMs, and a
mismatching existing file is overwritten back to the canonical bytes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]

if str(MODEL_DIR / "vendor") not in sys.path:
    sys.path.insert(0, str(MODEL_DIR / "vendor"))

FIXTURES_DEFAULT = ROOT / "work" / "startlux-validation" / "p6-production" / "fixtures"
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
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


base = _load_module("startlux_npu_adapter_for_unified_cases", "startlux_npu_adapter.py")


def _fixtures_directory(config):
    override = (config.get("data") or {}).get("params", {}).get("fixtures_directory")
    if override:
        path = Path(override)
        return path if path.is_absolute() else ROOT / path
    return FIXTURES_DEFAULT


def fixture_rgb(dominant, patches, seed):
    """A deterministic 256x256 solid-dominant RGB pattern (pure function of seed)."""
    rng = np.random.default_rng(seed)
    img = np.full((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dominant, dtype=np.int16)
    for (top, left, height, width), color in patches:
        img[top:top + height, left:left + width] = color
    img = img + rng.integers(-3, 4, img.shape, dtype=np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_ppm(path, image):
    height, width, _ = image.shape
    with open(path, "wb") as stream:
        stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        stream.write(np.ascontiguousarray(image).tobytes())


RED_IMAGE = dict(dominant=(210, 40, 40),
                 patches=[((36, 30, 70, 196), (240, 235, 225)),
                          ((150, 90, 60, 80), (35, 35, 215))],
                 seed=SEED_RED)
BLUE_IMAGE = dict(dominant=(35, 45, 215),
                  patches=[((40, 40, 90, 60), (230, 230, 240)),
                           ((160, 130, 56, 90), (215, 40, 40))],
                  seed=SEED_BLUE)


def _canonical_bytes(spec):
    import io

    buffer = io.BytesIO()
    image = fixture_rgb(**spec)
    height, width, _ = image.shape
    buffer.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
    buffer.write(np.ascontiguousarray(image).tobytes())
    return buffer.getvalue()


def ensure_fixtures(directory):
    """(re)generate the deterministic fixtures; returns {name: absolute path}.

    The pictures are pure functions of their seeds, so regeneration is
    byte-identical; an existing file with different bytes is overwritten back
    to the canonical content (never trusted as-is).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, spec in (("fixture-red-256.ppm", RED_IMAGE),
                       ("fixture-blue-256.ppm", BLUE_IMAGE)):
        path = directory / name
        canonical = _canonical_bytes(spec)
        if not path.is_file() or path.read_bytes() != canonical:
            path.write_bytes(canonical)
        out[name.removesuffix(".ppm")] = path
    return out


def _fixture_readouts(config):
    params = base._params(config)
    path = Path(params["fixtures_file"])
    if not path.is_absolute():
        path = MODEL_DIR / path
    data = json.loads(path.read_text())
    readouts = data["readouts"]
    if not readouts:
        raise ValueError("fixture has no readouts")
    if len({r["case_id"] for r in readouts}) != len(readouts):
        raise ValueError("duplicate readout case ids in fixture")
    return readouts


def _text_cases(config):
    """The seven canonical pure-text read-outs as B1 pre-encoded rows."""
    cases = []
    for readout in _fixture_readouts(config):
        length = len(readout["input_ids"])
        if readout["seq_len"] != length or readout["position_ids"] != list(range(length)) \
                or readout["actual_seq_lengths"] != [length] \
                or len(readout["scores26_golden_fp32"]) != 26:
            raise ValueError(f"fixture readout {readout['case_id']} is inconsistent")
        cases.append({"id": f"text-{readout['case_id']}", "samples": 1,
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


def _image_cases(config):
    """The real image cases on the self-regenerating fixtures (needs the tokenizer)."""
    folder = base._source_ref(config)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    letters = json.loads((folder / "decision_config.json").read_text())["letter_token_ids"]
    fixtures = ensure_fixtures(_fixtures_directory(config))
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
        {"id": "image-red-r0", "samples": 1, "payload": {
            "rows": [{"prompt": prompts["one"], "images": [rel(fixtures["fixture-red-256"])]}],
            "rows_meta": [meta("image-red-r0")]}},
        {"id": "image-blue-r0", "samples": 1, "payload": {
            "rows": [{"prompt": prompts["one"], "images": [rel(fixtures["fixture-blue-256"])]}],
            "rows_meta": [meta("image-blue-r0")]}},
        {"id": "image-two-a-b-r0", "samples": 1, "payload": {
            "rows": [{"prompt": prompts["ab"], "images": [rel(fixtures["fixture-red-256"]),
                                                          rel(fixtures["fixture-blue-256"])]}],
            "rows_meta": [meta("image-two-a-b-r0")]}},
        {"id": "mixed-text-image-b2-r0", "samples": 2, "payload": {
            "rows": [{"prompt": prompts["text"]},
                     {"prompt": prompts["one"],
                      "images": [rel(fixtures["fixture-red-256"])]}],
            "rows_meta": [meta("mixed-text-image-b2-r0"), meta("mixed-text-image-b2-r0")]}},
    ]


def generate(config):
    """The unified case pool: pre-encoded text cases FIRST, then the image cases."""
    cases = _text_cases(config)
    cases.extend(_image_cases(config))
    if not cases:
        raise ValueError("unified case generation produced no cases")
    return cases


def load(config):
    """Script-mode reload: generation is deterministic, generate == load."""
    return generate(config)
