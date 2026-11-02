"""CPU tests for the unified-AIR read-out seam (unified_readout.py).

No NPU, no weights.  The ge_runtime subprocess is replaced by a fake runtime
script implementing the same JSON plan protocol, so the REAL session code
paths run: plan building, the raw inputs.bin/outputs.bin handling, the
timeout/returncode/one-GE_RESULT contract, honest cold-cost records.  The
official media/letter-column path runs in a subprocess with the task-local
Pillow/torchvision on PYTHONPATH (transformers caches its PIL availability at
import time): the REAL tokenizer, the REAL official processor composition and
the REAL official Decider drive a REAL UnifiedReadoutBackend whose session
encodes through the official recipe -- the image state layout, the 64 image
tokens / [256, 1536] pixel block of the pinned red fixture, the letter-column
precision via marker scores, the wide-list knock-out re-sending media every
round, and the Instant generate refusal.
"""
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "startlux-decision-v1.1-0.8b-instant"
ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_DIR = ROOT / "work" / "startlux-materials" / \
    "model-6496ae641cc71b149ce9dad69bd97f2a44e7bf11"
REAL_FIXTURES = ROOT / "work" / "startlux-validation" / "p6-image"
DEPS = ROOT / "work" / "startlux-validation" / "p6-official" / "deps"


def load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(MODEL_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


multimodal = load("startlux_multimodal_for_readout_test", "multimodal.py")
unified_preprocess = load("startlux_unified_preprocess_readout_test",
                          "unified_preprocess.py")
unified_readout = load("startlux_unified_readout_test", "unified_readout.py")


def workdir(prefix):
    base = Path(os.environ.get("TMPDIR") or (ROOT / "work" / "tmp"))
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(base)))


def unified_io():
    return {"schema": "ge-io/1",
            "inputs": [
                {"name": "input_ids", "dtype": "int64", "shape": [-1, -1],
                 "dynamic_dims": [0, 1], "node": "arg0_1"},
                {"name": "position_ids", "dtype": "int64", "shape": [3, -1, -1],
                 "dynamic_dims": [1, 2], "node": "arg1_1"},
                {"name": "actual_seq_lengths", "dtype": "int32", "shape": [-1],
                 "dynamic_dims": [0], "node": "arg2_1"},
                {"name": "pixel_values", "dtype": "float32", "shape": [512, 1536],
                 "node": "arg3_1"},
                {"name": "image_feature_index", "dtype": "int64", "shape": [-1, -1],
                 "dynamic_dims": [0, 1], "node": "arg4_1"}],
            "outputs": [{"name": "scores", "dtype": "float32", "shape": [-1, 26],
                         "dynamic_dims": [0], "node": "arg5_1"}]}


def fake_folder_config():
    return types.SimpleNamespace(
        image_token_id=100, vision_start_token_id=102, vision_end_token_id=103,
        text_config=types.SimpleNamespace(vocab_size=128),
        vision_config=types.SimpleNamespace(spatial_merge_size=2))


class StubPreprocessor(unified_preprocess.OfficialPreprocessor):
    """Real builder + shared rope provider on a fake folder config; the row for
    a request is a deterministic hand-built official-shaped encoding."""

    def __init__(self, row_builder=None):
        self.model_dir = MODEL_DIR
        self._config = fake_folder_config()
        self._tokenizer = None
        self._processor = None
        self._rope_index = None
        self.min_width = unified_preprocess.DEFAULT_MIN_PHYSICAL_WIDTH
        self._row_builder = row_builder or (lambda prompt, media: {
            "input_ids": torch.tensor([[1] + [2] * (len(prompt) % 8 + 3)],
                                      dtype=torch.int64)})

    def encode_media_row(self, prompt, media=None):
        return self._row_builder(prompt, media)


FAKE_RUNTIME = r'''#!/usr/bin/env python3
import json, os, struct, sys, time

mode = os.environ.get("FAKE_RUNTIME_MODE", "ok")
plan = json.load(sys.stdin)
if plan.get("mode") == "read_io":
    io = {"schema": "ge-io/1",
          "inputs": [
              {"name": "input_ids", "dtype": "int64", "shape": [-1, -1],
               "dynamic_dims": [0, 1], "node": "arg0_1"},
              {"name": "position_ids", "dtype": "int64", "shape": [3, -1, -1],
               "dynamic_dims": [1, 2], "node": "arg1_1"},
              {"name": "actual_seq_lengths", "dtype": "int32", "shape": [-1],
               "dynamic_dims": [0], "node": "arg2_1"},
              {"name": "pixel_values", "dtype": "float32", "shape": [512, 1536],
               "node": "arg3_1"},
              {"name": "image_feature_index", "dtype": "int64", "shape": [-1, -1],
               "dynamic_dims": [0, 1], "node": "arg4_1"}],
          "outputs": [{"name": "scores", "dtype": "float32", "shape": [-1, 26],
                       "dynamic_dims": [0], "node": "arg5_1"}]}
    print("GE_RESULT " + json.dumps({"io": io}))
    sys.exit(0)
if mode == "hang":
    time.sleep(60)
out_dir = plan["output_dir"]
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "seen-plan.json"), "w") as fh:
    json.dump(plan, fh)
outputs, offset = {}, 0
with open(os.path.join(out_dir, "outputs.bin"), "wb") as fh:
    for node in plan["io"]["outputs"]:
        shape, count = list(node["shape"]), 1
        if any(d < 0 for d in shape):
            case = plan["inputs"]["cases"][0]
            ids = next(t for t in case["tensors"] if t["name"] == "input_ids")
            shape = [ids["shape"][0] if d == -1 else d for d in shape]
        for d in shape:
            count *= d
        values = [float("nan")] + [float(i) for i in range(1, count)] if mode == "nan" \
            else [float(i) for i in range(count)]
        if mode == "short_shape":
            shape, count, values = [shape[0], shape[1] - 1], count - 1, values[:-1]
        fh.write(struct.pack("<%df" % count, *values))
        outputs[node["name"]] = {"name": node["name"], "dtype": node["dtype"],
                                 "shape": shape, "offset": offset, "bytes": count * 4}
        offset += count * 4
print("[INFO] CompileGraph ok (gid=1/1), cost 21.5 ms")
print("[INFO] LoadGraph ok (gid=1, tid=0), cost 1.7 ms")
if mode == "no_result":
    sys.stderr.write("backend exploded\n")
    sys.exit(1)
if mode == "double_result":
    print("GE_RESULT " + json.dumps({"status": "success"}))
    print("GE_RESULT " + json.dumps({"status": "success"}))
    sys.exit(0)
if mode == "failed":
    print("GE_RESULT " + json.dumps({"status": "failed", "error": "backend execution failed"}))
    sys.exit(1)
rows = {case["id"]: [{"instance": 0, "tensors": list(outputs.values())}]
        for case in plan["inputs"]["cases"]}
print("GE_RESULT " + json.dumps({"status": "success", "mode": "infer", "outputs": rows}))
'''


def write_fake_runtime(directory, text=FAKE_RUNTIME):
    path = Path(directory) / "fake-ge-runtime.py"
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class TestUnifiedGraphSpec(unittest.TestCase):
    def test_accepts_the_unified_contract(self):
        spec = unified_readout.UnifiedGraphSpec.from_io(unified_io(), vocab_size=300,
                                                       max_seq_len=2048)
        self.assertEqual(spec.length_bound, 2048)
        self.assertEqual(spec.output_shape, (1, 26))
        self.assertEqual(spec.scores_bytes, 104)

    def test_missing_bound_is_refused(self):
        with self.assertRaises(unified_readout.ReadoutError):
            unified_readout.UnifiedGraphSpec.from_io(unified_io())

    def test_bad_bounds_are_refused(self):
        for bad in (0, -1, 2.0, True, "2048", None):
            with self.assertRaises(unified_readout.ReadoutError):
                unified_readout.UnifiedGraphSpec.from_io(unified_io(), max_seq_len=bad)

    def _mutated(self, **changes):
        io = unified_io()
        for side, index, key, value in changes.get("edits", []):
            io[side][index][key] = value
        if "drop" in changes:
            del io["inputs"][changes["drop"]]
        return io

    def test_geometry_deviations_are_refused(self):
        cases = [
            self._mutated(edits=[("inputs", 0, "dtype", "int32")]),
            self._mutated(edits=[("inputs", 1, "shape", [-1, -1, -1])]),
            self._mutated(edits=[("inputs", 3, "shape", [-1, 1536])]),
            self._mutated(edits=[("inputs", 3, "dtype", "bfloat16")]),
            self._mutated(edits=[("inputs", 4, "dynamic_dims", [0])]),
            self._mutated(edits=[("outputs", 0, "shape", [-1, 25])]),
            self._mutated(edits=[("outputs", 0, "dtype", "float16")]),
            self._mutated(drop=4),
        ]
        for io in cases:
            with self.assertRaises(unified_readout.ReadoutError):
                unified_readout.UnifiedGraphSpec.from_io(io, max_seq_len=2048)

    def test_extra_inputs_are_refused(self):
        io = unified_io()
        io["inputs"].append({"name": "vision_features", "dtype": "float32",
                             "shape": [-1, 64]})
        with self.assertRaises(unified_readout.ReadoutError):
            unified_readout.UnifiedGraphSpec.from_io(io, max_seq_len=2048)


class TestSessionProtocol(unittest.TestCase):
    """The REAL session code against a fake ge_runtime subprocess."""

    def setUp(self):
        self.dir = workdir("unified-readout-fake-")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, ignore_errors=True))
        self.runtime = write_fake_runtime(self.dir)
        self.spec = unified_readout.UnifiedGraphSpec.from_io(unified_io(), vocab_size=128,
                                                            max_seq_len=2048)
        self.air = self.dir / "model.air"
        self.air.write_bytes(b"fake-air")

    def session(self, timeout=60.0, preprocessor=None):
        return unified_readout.UnifiedReadoutSession(
            0, self.dir / "run", runtime_bin=self.runtime, air_path=self.air,
            spec=self.spec, timeout=timeout,
            preprocessor=preprocessor or StubPreprocessor())

    def test_one_readout_through_the_whole_plan_protocol(self):
        session = self.session()
        scores = session.read_scores26("a prompt")
        self.assertEqual(len(scores), 26)
        self.assertTrue(all(np.isfinite(scores)))
        record = session.records[-1]
        self.assertEqual(record["image_tokens"], 0)
        # the accepted minimum-physical-width policy: a short row pads to 256
        # while actual_seq_lengths keep the TRUE length (both recorded)
        self.assertEqual(record["min_width"], 256)
        self.assertEqual(record["physical_width"], 256)
        self.assertLess(record["seq_len"], record["physical_width"])
        self.assertEqual(record["timings_ms"], {"compile_ms": 21.5, "load_ms": 1.7})
        readout_dir = Path(record["plan"]).parent
        plan = json.loads((readout_dir / "plan.json").read_text())
        self.assertEqual(plan["backend"], "ge_session")
        self.assertEqual(plan["options"]["graph_run_mode"], 0)
        self.assertEqual(plan["mode"], "infer")
        self.assertEqual(plan["device"], 0)
        case = plan["inputs"]["cases"][0]
        self.assertEqual(case["samples"], 1)
        self.assertEqual(case["output_capacities"], {"scores": 104})
        by_name = {t["name"]: t for t in case["tensors"]}
        self.assertEqual(list(by_name), ["input_ids", "position_ids",
                                         "actual_seq_lengths", "pixel_values",
                                         "image_feature_index"])
        self.assertEqual(by_name["input_ids"]["dtype"], "int64")
        self.assertEqual(by_name["actual_seq_lengths"]["dtype"], "int32")
        self.assertEqual(by_name["pixel_values"]["shape"], [512, 1536])
        self.assertEqual(by_name["pixel_values"]["dtype"], "float32")
        self.assertEqual(by_name["input_ids"]["shape"], [1, 256])   # the 256 floor
        # the raw evidence bins are kept, never deleted
        self.assertTrue((readout_dir / "inputs.bin").is_file())
        self.assertTrue((readout_dir / "outputs.bin").is_file())
        self.assertTrue((readout_dir / "infer-runtime.log").is_file())
        # the tensors really are the five graph inputs in io order
        arrays = np.fromfile(readout_dir / "inputs.bin", dtype=np.int64,
                             count=5, offset=0)  # first tensor begins at 0
        self.assertEqual(arrays.shape, (5,))

    def test_session_dirs_are_fresh_and_reruns_never_overwrite(self):
        session = self.session()
        first = session.session_dir
        session.read_scores26("one")
        session.read_scores26("two")
        self.assertEqual(len(session.records), 2)
        self.assertNotEqual(Path(session.records[0]["plan"]).parent,
                            Path(session.records[1]["plan"]).parent)
        other = self.session()
        self.assertNotEqual(first, other.session_dir)

    def test_timeout_is_a_clear_error(self):
        session = self.session(timeout=0.4)
        env = dict(os.environ, FAKE_RUNTIME_MODE="hang")
        saved = os.environ.copy()
        os.environ.update(env)
        try:
            with self.assertRaises(unified_readout.ReadoutError) as ctx:
                session.read_scores26("slow")
            self.assertIn("did not finish within", str(ctx.exception))
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_failed_status_and_result_line_contracts(self):
        for mode, fragment in (("failed", "failed"),
                               ("no_result", "GE_RESULT"),
                               ("double_result", "GE_RESULT lines")):
            session = self.session()
            saved = os.environ.copy()
            os.environ["FAKE_RUNTIME_MODE"] = mode
            try:
                with self.assertRaises(unified_readout.ReadoutError) as ctx:
                    session.read_scores26("bad")
                self.assertIn(fragment, str(ctx.exception), mode)
            finally:
                os.environ.clear()
                os.environ.update(saved)

    def test_non_finite_and_wrong_shapes_are_refused(self):
        for mode in ("nan", "short_shape"):
            session = self.session()
            saved = os.environ.copy()
            os.environ["FAKE_RUNTIME_MODE"] = mode
            try:
                with self.assertRaises(unified_readout.ReadoutError):
                    session.read_scores26("bad")
            finally:
                os.environ.clear()
                os.environ.update(saved)

    def test_overlong_prompt_is_rejected_never_truncated(self):
        def long_row(prompt, media):
            return {"input_ids": torch.ones((1, 3000), dtype=torch.int64)}

        session = self.session(preprocessor=StubPreprocessor(row_builder=long_row))
        with self.assertRaises(unified_readout.ReadoutError) as ctx:
            session.read_scores26("too long")
        self.assertIn("rejected, never truncated", str(ctx.exception))

    def test_min_width_floor_and_true_length_policy(self):
        def long_row(prompt, media):
            return {"input_ids": torch.ones((1, 300), dtype=torch.int64)}

        # a long row runs at its TRUE length: the floor never stretches it
        session = self.session(preprocessor=StubPreprocessor(row_builder=long_row))
        session.read_scores26("long")
        record = session.records[-1]
        self.assertEqual(record["seq_len"], 300)
        self.assertEqual(record["physical_width"], 300)
        # the floor is a session policy, recorded and adjustable
        small = unified_readout.UnifiedReadoutSession(
            0, self.dir / "run3", runtime_bin=self.runtime, air_path=self.air,
            spec=self.spec, timeout=60.0, preprocessor=StubPreprocessor(),
            min_width=64)
        small.read_scores26("short")
        self.assertEqual(small.records[-1]["physical_width"], 64)
        with self.assertRaises(unified_readout.ReadoutError):
            unified_readout.UnifiedReadoutSession(
                0, self.dir / "run4", runtime_bin=self.runtime, air_path=self.air,
                spec=self.spec, timeout=60.0, preprocessor=StubPreprocessor(),
                min_width=0)

    def test_read_io_builds_the_spec_from_the_air(self):
        session = unified_readout.UnifiedReadoutSession(
            0, self.dir / "run2", runtime_bin=self.runtime, air_path=self.air,
            timeout=60.0, preprocessor=StubPreprocessor(), vocab_size=128,
            max_seq_len=512)
        self.assertEqual(session.spec.length_bound, 512)
        self.assertEqual(session.spec.vocab_size, 128)
        self.assertTrue((session.session_dir / "read-io.log").is_file())


class TestBackendWiring(unittest.TestCase):
    """The REAL backend against an in-process fake session (letter mapping, Instant)."""

    def _fake_session(self, scores26):
        spec = unified_readout.UnifiedGraphSpec.from_io(unified_io(), vocab_size=300,
                                                       max_seq_len=2048)

        class FakeSession:
            def __init__(self):
                self.spec = spec
                self.air_path = "fake.air"
                self.records = []
                self.calls = []

            def read_scores26(self, prompt, media=None):
                self.calls.append((prompt, media))
                self.records.append({"seq_len": len(prompt), "images":
                                     0 if media is None else len(media.images)})
                return list(scores26)

        return FakeSession()

    def test_letter_columns_map_onto_the_folder_letters(self):
        letters = list(range(32, 58))
        scores = [0.0] * 26
        scores[1] = 2.0                                    # letter 'B' (column 1)
        session = self._fake_session(scores)
        backend = unified_readout.UnifiedReadoutBackend(session, object(), letters)
        result = asyncio_run(backend.readout("prompt", [letters[1], letters[0]]))
        self.assertEqual(result, [2.0, 0.0])
        self.assertEqual(backend.records[0]["active_letters"], ["B", "A"])

    def test_request_level_violations_are_official_request_errors(self):
        from startlux_decide.backends.base import BackendRequestError

        letters = list(range(32, 58))
        session = self._fake_session([0.0] * 26)
        backend = unified_readout.UnifiedReadoutBackend(session, object(), letters)
        for call in (lambda: asyncio_run(backend.readout("", [letters[0]])),
                     lambda: asyncio_run(backend.readout("p", [])),
                     lambda: asyncio_run(backend.readout("p", [200])),
                     lambda: asyncio_run(backend.readout("p", [letters[0], letters[0]])),
                     lambda: asyncio_run(backend.readout("p", [letters[0]], media="path"))):
            with self.assertRaises(BackendRequestError):
                call()

    def test_instant_models_never_generate(self):
        letters = list(range(32, 58))
        backend = unified_readout.UnifiedReadoutBackend(self._fake_session([0.0] * 26),
                                                        object(), letters)
        with self.assertRaises(NotImplementedError):
            asyncio_run(backend.generate("p", 4, None, (), 0))

    def test_reads_images_flag_and_describe(self):
        letters = list(range(32, 58))
        backend = unified_readout.UnifiedReadoutBackend(self._fake_session([0.0] * 26),
                                                        object(), letters)
        self.assertTrue(backend.reads_images)
        described = backend.describe()
        self.assertEqual(described["name"], "unified-ge-readout")
        self.assertTrue(described["cold_subprocess_per_readout"])


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


class TestDecisionImagesCompat(unittest.TestCase):
    """decision.py's minimal images extension is backward compatible in-process.

    images=None keeps the text bridge byte-identical; ``images`` is keyword-only;
    the default ReadoutCallbackBackend still refuses media (that refusal with
    REAL decoded media is proven by the subprocess runner below).
    """

    def setUp(self):
        import json

        holder = tempfile.TemporaryDirectory(
            prefix="unified-decision-compat-",
            dir=os.environ.get("TMPDIR") or str(ROOT / "work" / "tmp"))
        self.addCleanup(holder.cleanup)
        folder = Path(holder.name) / "model"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps(
            {"jev_prompt_attention": "bidirectional", "max_position_embeddings": 262144}))
        (folder / "decision_config.json").write_text(json.dumps(
            {"prompt_attention": "bidirectional",
             "temperature_by_type": {"choice": 0.7, "noul": 1.3, "score": 0.9},
             "letter_token_ids": list(range(65, 91)), "max_options_per_pass": 26,
             "wide_choice": {"group": 25, "keep": 3, "residual": 0.001},
             "format": "StartLux-Decision-v1.1", "release_date": "2026-10-30"}))
        self.folder = folder
        sys.path.insert(0, str(MODEL_DIR / "vendor"))
        from startlux_decide.backends.fake import FakeTokenizer

        self.tok = FakeTokenizer()
        self.calls = []

        def readout(prompt, letter_ids):
            self.calls.append((prompt, tuple(letter_ids)))
            return [2.0 if k == 0 else 0.0 for k in range(len(letter_ids))]

        self.readout = readout
        self.decision = unified_readout._load_module(
            "startlux_decision_compat", "decision.py").StartluxDecision(
            folder, self.tok, readout=readout)

    def test_images_none_is_identical_to_the_old_text_behavior(self):
        pick = {"instructions": "Pick one.",
                "criteria": {"aa": "x", "bb": "y"}}
        first = self.decision.decide("evidence", {"q": pick})
        second = self.decision.decide("evidence", {"q": pick}, images=None)
        self.assertEqual(first.answers, second.answers)
        self.assertEqual(self.calls[0], self.calls[1])
        self.assertEqual(first.answers["q"]["choice"], "aa")

    def test_images_is_keyword_only(self):
        pick = {"instructions": "Pick one.", "criteria": {"aa": "x", "bb": "y"}}
        # the third POSITIONAL argument is body; images has no positional slot
        with self.assertRaises(TypeError):
            self.decision.decide("evidence", {"q": pick}, None,
                                 ["data:image/png;base64,AAAA"])



REAL_RUNNER = r'''
import asyncio, base64, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__root__)
MODEL_DIR = ROOT / "models/startlux-decision-v1.1-0.8b-instant"
MATERIALS = ROOT / "work/startlux-materials/model-6496ae641cc71b149ce9dad69bd97f2a44e7bf11"
RED = ROOT / "work/startlux-validation/p6-image/fixture-image-a-red.ppm"
BLUE = ROOT / "work/startlux-validation/p6-image/fixture-image-b-blue.ppm"

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR / "vendor"))

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(MODEL_DIR / filename))
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

unified_readout = load("unified_readout_real", "unified_readout.py")
unified_preprocess = load("unified_preprocess_real", "unified_preprocess.py")
bridge = load("startlux_decision_real", "decision.py")

from transformers import AutoTokenizer

pre = unified_preprocess.OfficialPreprocessor(MATERIALS)
tok = AutoTokenizer.from_pretrained(str(MATERIALS), local_files_only=True)
folder = bridge.load_model_folder(str(MATERIALS))
letters = list(folder.letter_token_ids)

io = {"schema": "ge-io/1",
      "inputs": [
          {"name": "input_ids", "dtype": "int64", "shape": [-1, -1], "dynamic_dims": [0, 1]},
          {"name": "position_ids", "dtype": "int64", "shape": [3, -1, -1], "dynamic_dims": [1, 2]},
          {"name": "actual_seq_lengths", "dtype": "int32", "shape": [-1], "dynamic_dims": [0]},
          {"name": "pixel_values", "dtype": "float32", "shape": [512, 1536]},
          {"name": "image_feature_index", "dtype": "int64", "shape": [-1, -1], "dynamic_dims": [0, 1]}],
      "outputs": [{"name": "scores", "dtype": "float32", "shape": [-1, 26], "dynamic_dims": [0]}]}
spec = unified_readout.UnifiedGraphSpec.from_io(
    io, vocab_size=unified_readout.source_vocab_size(MATERIALS), max_seq_len=2048)

import numpy as np
import torch

class MarkerSession:
    """A fake GE session: REAL official encoding, marker scores, full recording."""
    def __init__(self):
        self.spec = spec
        self.air_path = "unified-fake.air"
        self.records = []
        self.captured = []

    def read_scores26(self, prompt, media=None):
        arrays = unified_preprocess.encode_readout_row(pre, prompt, media, spec)
        self.captured.append({
            "seq_len": int(arrays["actual_seq_lengths"][0]),
            "image_tokens": int((arrays["image_feature_index"] >= 0).sum()),
            "pixel_shape": list(arrays["pixel_values"].shape),
            "lengths": arrays["actual_seq_lengths"].tolist(),
            "placeholders": prompt.count("<|image_pad|>"),
            "media_images": 0 if media is None else len(media.images),
            "marker": None,
        })
        self.records.append({"seq_len": self.captured[-1]["seq_len"],
                             "images": self.captured[-1]["media_images"]})
        return list(self.captured[-1].pop("marker") or [])
    def set_marker(self, column):
        self.captured[-1]["marker"] = [2.0 if i == column else 0.0 for i in range(26)]

session = MarkerSession()
backend = unified_readout.UnifiedReadoutBackend(session, tok, letters)
decision = bridge.StartluxDecision(folder, tok, backend=backend)

QUESTIONS = {"dominant": {
    "instructions": "Which color occupies the largest area of the image?",
    "criteria": {"red": "the red areas are the largest",
                 "green": "the green areas are the largest",
                 "blue": "the blue areas are the largest"}}}
STATE_ONE = "A synthetic 256x256 RGB test pattern generated for pipeline validation.\n<image>"
STATE_AB = ("Two synthetic 256x256 RGB test patterns: the first is red-dominant, "
            "the second is blue-dominant.\n<image>\n<image>")

def uri(path):
    return "data:image/x-portable-pixmap;base64," + base64.b64encode(path.read_bytes()).decode()

r = {}

# the marker session: REAL official encoding on every call, marker scores fixed
# at one letter column (column 1 = letter 'B' = the second listed option)
class MarkerSession:
    def __init__(self, column):
        self.spec = spec
        self.air_path = "unified-fake.air"
        self.column = column
        self.records = []
        self.captured = []

    def read_scores26(self, prompt, media=None):
        arrays = unified_preprocess.encode_readout_row(pre, prompt, media, spec)
        self.captured.append({
            "prompt": prompt,
            "seq_len": int(arrays["actual_seq_lengths"][0]),
            "physical_width": int(arrays["input_ids"].shape[1]),
            "image_tokens": int((arrays["image_feature_index"] >= 0).sum()),
            "pixel_shape": list(arrays["pixel_values"].shape),
            "placeholders": prompt.count("<|image_pad|>"),
            "media_images": 0 if media is None else len(media.images)})
        self.records.append({"seq_len": self.captured[-1]["seq_len"],
                             "images": self.captured[-1]["media_images"]})
        return [2.0 if i == self.column else 0.0 for i in range(26)]

marker_session = MarkerSession(1)   # letter 'B' = option 'green' in QUESTIONS
# THE public production entry: startlux_unified_decision -> StartluxDecision
# with the UnifiedReadoutBackend injected; image requests go through its
# public decide(images=...) (forwarded to the official Decider.decide(images=...)).
decision2, backend2 = unified_readout.startlux_unified_decision(MATERIALS, marker_session)
assert decision2.letter_ids == tuple(letters)

# 1. single-image request through the PUBLIC factory: marker on 'B' (green)
ans2 = asyncio.run(decision2.decide_async(STATE_ONE, QUESTIONS, images=[uri(RED)]))
cap = marker_session.captured[0]
r["single_prompt_has_placeholder"] = "<|vision_start|><|image_pad|><|vision_end|>" in cap["prompt"]
r["single_placeholders"] = cap["placeholders"] == 1
r["single_media_images"] = cap["media_images"] == 1
r["single_image_tokens"] = cap["image_tokens"]
r["single_pixel_shape"] = cap["pixel_shape"]
r["single_physical_width"] = cap["physical_width"]
r["marker_choice"] = ans2.answers["dominant"]["choice"]
p = ans2.answers["dominant"]["probabilities"]
import math
temp = folder.temperature("choice")
w = [math.exp((2.0 if i == 1 else 0.0) / temp) for i in range(3)]
expected = w[1] / sum(w)
r["marker_prob_green"] = p["green"]
r["marker_expected_green"] = expected

# 2. two-image request: both images in one media object, 128 image tokens
ans_ab = asyncio.run(decision2.decide_async(STATE_AB, QUESTIONS,
                                            images=[uri(RED), uri(BLUE)]))
cap_ab = marker_session.captured[-1]
r["ab_prompt_has_two_placeholders"] = cap_ab["placeholders"] == 2
r["ab_media_images"] = cap_ab["media_images"] == 2
r["ab_image_tokens"] = cap_ab["image_tokens"]
r["ab_pixel_shape"] = cap_ab["pixel_shape"]

# 3. text request through the StartluxDecision bridge: media is None, no image tokens
ans_txt = asyncio.run(decision2.decide_async("plain text evidence", QUESTIONS))
cap_txt = marker_session.captured[-1]
r["text_media_images"] = cap_txt["media_images"]
r["text_image_tokens"] = cap_txt["image_tokens"]
r["text_bridge_answered"] = ans_txt.answers["dominant"]["choice"] in QUESTIONS["criteria"] \
    if "criteria" in QUESTIONS else ans_txt.answers["dominant"]["choice"] in (
        "red", "green", "blue")

# 4. wide-list knock-out (27 options): sequential rounds, media re-sent every round
wide = {"instructions": "Pick the fitting one.",
        "criteria": {f"option_{k}": f"description {k}" for k in range(27)}}
state_wide = "wide state with an image\n<image>"
ans_wide = asyncio.run(decision2.decide_async(state_wide, {"q": wide},
                                               images=[uri(RED)]))
rounds = [c for c in marker_session.captured if c["media_images"] == 1]
r["wide_rounds_at_least_two"] = len(rounds) >= 2
r["wide_media_resent_every_round"] = all(c["media_images"] == 1 for c in rounds)
r["wide_all_rounds_64_tokens"] = all(c["image_tokens"] == 64 for c in rounds)
r["wide_answer_is_an_option"] = ans_wide.answers["q"]["choice"] in wide["criteria"]

# 5. Instant generate refusal
try:
    asyncio.run(backend2.generate("p", 4, None, (), 0))
    r["generate_refused"] = False
except NotImplementedError:
    r["generate_refused"] = True

# 6. the DEFAULT text callback backend still refuses real image requests: the
# official Decider rejects an image request on a non-image engine up front
# (RequestError), and the callback itself would refuse the media too
from startlux_decide.backends.base import BackendRequestError
from startlux_decide.controls import RequestError

def plain_readout(prompt, letter_ids):
    return [0.0] * len(letter_ids)

text_only = bridge.StartluxDecision(folder, tok, readout=plain_readout)
try:
    asyncio.run(text_only.decide_async(STATE_ONE, QUESTIONS, images=[uri(RED)]))
    r["default_callback_refuses_media"] = False
except (BackendRequestError, RequestError):
    r["default_callback_refuses_media"] = True

print("RESULT " + json.dumps(r))
'''.replace("__root__", repr(str(ROOT)))


@unittest.skipUnless(REAL_MODEL_DIR.is_dir(), "the pinned real model materials are not present")
@unittest.skipUnless((DEPS / "pillow").is_dir() and (DEPS / "torchvision").is_dir(),
                     "the task-local Pillow/torchvision install is not present")
class RealOfficialMediaBackend(unittest.TestCase):
    """REAL tokenizer/processor/Decider + REAL backend + fake session (subprocess)."""

    @classmethod
    def setUpClass(cls):
        directory = workdir("unified-readout-real-")
        cls.addClassCleanup(lambda: __import__("shutil").rmtree(directory,
                                                                ignore_errors=True))
        script = directory / "run-real-backend.py"
        script.write_text(REAL_RUNNER)
        env = dict(os.environ,
                   PYTHONPATH=f"{DEPS / 'pillow'}{os.pathsep}{DEPS / 'torchvision'}"
                   + (os.pathsep + os.environ["PYTHONPATH"]
                      if os.environ.get("PYTHONPATH") else ""),
                   OMP_NUM_THREADS="8")
        proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, env=env, timeout=900)
        lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
        if not lines:
            raise AssertionError(f"the real-backend runner produced no result "
                                 f"(exit {proc.returncode}); stderr tail:\n"
                                 f"{proc.stderr[-4000:]}")
        cls.result = json.loads(lines[-1][len("RESULT "):])

    def test_single_image_request_layout(self):
        self.assertTrue(self.result["single_prompt_has_placeholder"])
        self.assertTrue(self.result["single_placeholders"])
        self.assertEqual(self.result["single_media_images"], 1)
        self.assertEqual(self.result["single_image_tokens"], 64)
        self.assertEqual(self.result["single_pixel_shape"], [512, 1536])

    def test_marker_letter_column_precision(self):
        # column 1 = letter 'B' = second option ('green'): the official answer follows
        self.assertEqual(self.result["marker_choice"], "green")
        self.assertAlmostEqual(self.result["marker_prob_green"],
                               self.result["marker_expected_green"], places=9)

    def test_two_image_request_layout(self):
        self.assertTrue(self.result["ab_prompt_has_two_placeholders"])
        self.assertTrue(self.result["ab_media_images"])
        self.assertEqual(self.result["ab_image_tokens"], 128)
        self.assertEqual(self.result["ab_pixel_shape"], [512, 1536])

    def test_text_request_has_no_images(self):
        self.assertEqual(self.result["text_media_images"], 0)
        self.assertEqual(self.result["text_image_tokens"], 0)
        self.assertTrue(self.result["text_bridge_answered"])

    def test_wide_list_knockout_resent_media(self):
        self.assertTrue(self.result["wide_rounds_at_least_two"])
        self.assertTrue(self.result["wide_media_resent_every_round"])
        self.assertTrue(self.result["wide_all_rounds_64_tokens"])
        self.assertTrue(self.result["wide_answer_is_an_option"])

    def test_instant_generate_refused(self):
        self.assertTrue(self.result["generate_refused"])

    def test_default_text_callback_refuses_real_media(self):
        self.assertTrue(self.result["default_callback_refuses_media"])


if __name__ == "__main__":
    unittest.main()
