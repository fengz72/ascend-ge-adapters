"""CPU tests for the unified-route official preprocessing seam (unified_preprocess.py).

No weights, no NPU.  The payload-validation and shared-rope-provider paths run
on plain torch/numpy fakes.  The real-component tests (guarded on the pinned
materials and the task-local Pillow/torchvision install) prove the REAL official
processor composition -- AutoImageProcessor(backend='pil') + AutoTokenizer +
AutoVideoProcessor, because Qwen3VLProcessor REJECTS video_processor=None
(the real TypeError is asserted) -- and the official numbers: 64 image tokens
per officially processed 256x256 image, a [256, 1536] pixel block per image,
grid [1, 16, 16], the strict cap/grid rejections, using the pinned red/blue
PPM fixtures.
"""
import importlib.util
import json
import os
import shutil
import sys
import types
import unittest
from pathlib import Path

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


multimodal = load("startlux_multimodal_for_unified_test", "model/multimodal.py")
unified_preprocess = load("startlux_unified_preprocess_test", "model/unified_preprocess.py")


def fake_folder_config(image_token=100, vision_start=102, vision_end=103, vocab=128):
    return types.SimpleNamespace(
        image_token_id=image_token, vision_start_token_id=vision_start,
        vision_end_token_id=vision_end,
        text_config=types.SimpleNamespace(vocab_size=vocab),
        vision_config=types.SimpleNamespace(spatial_merge_size=2))


class FakePreprocessor(unified_preprocess.OfficialPreprocessor):
    """A real OfficialPreprocessor whose folder config is a fake namespace.

    All encode/validate/build logic is the production one; only the AutoConfig
    load is bypassed (the fake config feeds the shared official rope provider
    and the real builder).  prompt rows would need a real tokenizer and are
    never encoded here; the payload tests only exercise rejection paths that
    fire before any encoding.
    """

    def __init__(self, config=None):
        self.model_dir = MODEL_DIR
        self._config = config or fake_folder_config()
        self._tokenizer = None
        self._processor = None
        self._rope_index = None


def official_image_deps():
    """Make the task-local Pillow/torchvision importable, or skip the test."""
    for sub in ("pillow", "torchvision"):
        path = str(DEPS / sub)
        if Path(path).is_dir() and path not in sys.path:
            sys.path.insert(0, path)
    try:
        import PIL          # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as error:
        raise unittest.SkipTest(f"task-local Pillow/torchvision not importable: {error}")


DEPS_PYPATH = f"{DEPS / 'pillow'}{os.pathsep}{DEPS / 'torchvision'}"

RUNNER = r"""
import importlib.util, json, sys
from pathlib import Path

ROOT = Path(__root__)
MODEL_DIR = ROOT / "models/startlux-decision-v1.1-0.8b-instant"
MATERIALS = ROOT / "work/startlux-materials/model-6496ae641cc71b149ce9dad69bd97f2a44e7bf11"
FIXTURES = ROOT / "work/startlux-validation/p6-image"
RED = str(FIXTURES / "fixture-image-a-red.ppm")
BLUE = str(FIXTURES / "fixture-image-b-blue.ppm")

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(MODEL_DIR / "decider" / "vendor"))
spec = importlib.util.spec_from_file_location("unified_preprocess", MODEL_DIR / "model" / "unified_preprocess.py")
up = importlib.util.module_from_spec(spec); sys.modules["unified_preprocess"] = up
spec.loader.exec_module(up)

from startlux_decide.images import IMAGE_PLACEHOLDER, place_images
from startlux_decide.prompt import from_systemone, messages, plain_prompt

QUESTIONS = {"instructions": "Which color occupies the largest area of the image?",
             "criteria": {"red": "the red areas are the largest",
                          "green": "the green areas are the largest",
                          "blue": "the blue areas are the largest"}}
STATE_ONE = "A synthetic 256x256 RGB test pattern generated for pipeline validation.\n<image>"
STATE_AB = ("Two synthetic 256x256 RGB test patterns: the first is red-dominant, "
            "the second is blue-dominant.\n<image>\n<image>")
# no <image> marks: place_images prepends the 3 slots, so the prompt stays
# well-formed and the per-request cap is what rejects it (not the token count)
STATE_THREE = "Three synthetic 256x256 RGB test patterns for the per-request image cap."

pre = up.OfficialPreprocessor(MATERIALS)

def render(state, n):
    row = from_systemone(place_images(state, n), QUESTIONS)
    prompt = plain_prompt(pre.tokenizer, messages(row))
    assert prompt.count(IMAGE_PLACEHOLDER) == n
    return prompt

r = {}
try:
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    processor = pre.processor
    r["processor_class"] = type(processor).__name__
    r["video_processor_instance"] = processor.video_processor is not None
    from transformers.models.auto.image_processing_auto import AutoImageProcessor
    ip = AutoImageProcessor.from_pretrained(str(MATERIALS), local_files_only=True, backend="pil")
    try:
        Qwen3VLProcessor(image_processor=ip, tokenizer=pre.tokenizer, video_processor=None)
        r["none_video_processor_rejected"] = False
    except TypeError:
        r["none_video_processor_rejected"] = True

    p_one, p_ab = render(STATE_ONE, 1), render(STATE_AB, 2)
    p_three = render(STATE_THREE, 3)
    row = pre.encode_row(p_one, [RED])
    r["single_ids_shape"] = list(row["input_ids"].shape)
    r["single_image_tokens"] = int((row["input_ids"][0] == pre.image_token).sum())
    r["single_pixel_shape"] = list(row["pixel_values"].shape)
    r["single_grid"] = row["image_grid_thw"].tolist()
    row_ab = pre.encode_row(p_ab, [RED, BLUE])
    r["double_image_tokens"] = int((row_ab["input_ids"][0] == pre.image_token).sum())
    r["double_pixel_shape"] = list(row_ab["pixel_values"].shape)

    def build(rows, **caps):
        return pre.build_graph_inputs(pre.encode_payload_rows(rows), **caps)

    arrays = build([{"prompt": p_one, "images": [RED]},
                    {"prompt": "a plain text second row"}])
    r["b2_shapes"] = {k: list(v.shape) for k, v in arrays.items()}
    r["b2_lengths"] = arrays["actual_seq_lengths"].tolist()
    r["b2_row1_index_all_minus1"] = bool((arrays["image_feature_index"][1] == -1).all())
    row1 = arrays["position_ids"][:, 1, :]
    r["b2_row1_positions_arange"] = bool(
        (row1[:, :5] == np.arange(5)[None, :]).all() and (row1[:, 5:] == 0).all())
    image_at = [i for i, v in enumerate(arrays["image_feature_index"][0]) if v >= 0]
    r["b2_t_plane_flat_over_image"] = len(set(arrays["position_ids"][0, 0][image_at].tolist())) == 1

    caps = {}
    for name, fn in (
            ("max_batch", lambda: build(
                [{"prompt": p_one, "images": [RED]}] * 3, max_batch=2)),
            ("max_seq_len", lambda: build(
                [{"prompt": p_one, "images": [RED]}], max_seq_len=100)),
            ("three_per_request", lambda: build(
                [{"prompt": p_three, "images": [RED, BLUE, RED]}])),
            ("three_slots", lambda: build(
                [{"prompt": p_one, "images": [RED]},
                 {"prompt": p_one, "images": [BLUE]},
                 {"prompt": p_one, "images": [RED]}])),
            ("four_slots_b2", lambda: build(
                [{"prompt": p_ab, "images": [RED, BLUE]},
                 {"prompt": p_one, "images": [BLUE]}]))):
        try:
            fn(); caps[name] = False
        except ValueError:
            caps[name] = True
    r["rejections"] = caps

    ids = pre.encode_row("hello official text row")["input_ids"][0].tolist()
    r["text_row_matches_tokenizer"] = ids == pre.tokenizer.encode(
        "hello official text row", add_special_tokens=False)
except Exception as error:
    r["runner_error"] = f"{type(error).__name__}: {error}"
print("RESULT " + json.dumps(r))
""".replace("__root__", repr(str(ROOT)))


@unittest.skipUnless(REAL_MODEL_DIR.is_dir(), "the pinned real model materials are not present")
@unittest.skipUnless((DEPS / "pillow").is_dir() and (DEPS / "torchvision").is_dir(),
                     "the task-local Pillow/torchvision install is not present")
class RealOfficialComponents(unittest.TestCase):
    """The REAL official components on the pinned folder, run in a subprocess.

    transformers caches its PIL/torchvision availability flags at import time,
    so the real composition is exercised exactly the way the validated runners
    do it: a fresh interpreter with the task-local deps on PYTHONPATH.  No
    weights are loaded and no NPU context is created anywhere.
    """

    @classmethod
    def setUpClass(cls):
        import os
        import subprocess
        import tempfile

        base = Path(os.environ.get("TMPDIR") or (ROOT / "work" / "tmp"))
        base.mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(prefix="unified-preprocess-real-", dir=str(base)))
        cls.addClassCleanup(lambda: shutil.rmtree(workdir, ignore_errors=True))
        script = workdir / "run-real-components.py"
        script.write_text("import numpy as np\n" + RUNNER)
        env = dict(os.environ, PYTHONPATH=f"{DEPS / 'pillow'}{os.pathsep}{DEPS / 'torchvision'}"
                   + (os.pathsep + os.environ["PYTHONPATH"]
                      if os.environ.get("PYTHONPATH") else ""),
                   OMP_NUM_THREADS="8")
        cls.proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                                  text=True, env=env, timeout=600)
        lines = [line for line in cls.proc.stdout.splitlines() if line.startswith("RESULT ")]
        if not lines:
            raise AssertionError(f"the real-component runner produced no result "
                                 f"(exit {cls.proc.returncode}); stderr tail:\n"
                                 f"{cls.proc.stderr[-3000:]}")
        cls.result = json.loads(lines[-1][len("RESULT "):])
        if "runner_error" in cls.result:
            raise AssertionError(f"the real-component runner failed: "
                                 f"{cls.result['runner_error']}")

    def test_real_processor_composition(self):
        self.assertEqual(self.result["processor_class"], "Qwen3VLProcessor")
        self.assertTrue(self.result["video_processor_instance"],
                        "the verified composition needs a real AutoVideoProcessor "
                        "instance; video_processor=None is rejected by the constructor")

    def test_none_video_processor_is_really_rejected(self):
        self.assertTrue(self.result["none_video_processor_rejected"],
                        "Qwen3VLProcessor(video_processor=None) must raise TypeError")

    def test_single_image_official_numbers(self):
        self.assertEqual(self.result["single_ids_shape"], [1, 173])
        self.assertEqual(self.result["single_image_tokens"], 64)
        self.assertEqual(self.result["single_pixel_shape"], [256, 1536])
        self.assertEqual(self.result["single_grid"], [[1, 16, 16]])

    def test_double_image_official_numbers(self):
        self.assertEqual(self.result["double_image_tokens"], 128)
        self.assertEqual(self.result["double_pixel_shape"], [512, 1536])

    def test_build_graph_inputs_mixed_b2(self):
        shapes = self.result["b2_shapes"]
        self.assertEqual(shapes["input_ids"], [2, 173])
        self.assertEqual(shapes["position_ids"], [3, 2, 173])
        self.assertEqual(shapes["pixel_values"], [512, 1536])
        self.assertEqual(shapes["image_feature_index"], [2, 173])
        self.assertEqual(self.result["b2_lengths"], [173, 5])
        self.assertTrue(self.result["b2_row1_index_all_minus1"])
        self.assertTrue(self.result["b2_row1_positions_arange"])
        self.assertTrue(self.result["b2_t_plane_flat_over_image"])

    def test_caps_are_the_callers(self):
        self.assertTrue(self.result["rejections"]["max_batch"])
        self.assertTrue(self.result["rejections"]["max_seq_len"])

    def test_rejections(self):
        for name in ("three_per_request", "three_slots", "four_slots_b2"):
            self.assertTrue(self.result["rejections"][name], name)

    def test_text_row_matches_the_official_tokenizer(self):
        self.assertTrue(self.result["text_row_matches_tokenizer"])


@unittest.skipUnless(REAL_MODEL_DIR.is_dir(), "the pinned real model materials are not present")
class MissingDepsGuidance(unittest.TestCase):
    """Without the image deps the processor property names them (no system install).

    The guidance contract is independent of the real environment: the formal
    environment has the task-local Pillow/torchvision importable (env.sh
    PYTHONPATH) and transformers resolves their availability at import time,
    so whether a plain in-process ``pre.processor`` raises depends on where
    and when it runs.  Instead the EXACT import boundary that loads the
    official image-processing dependency chain -- the first ``from
    transformers.models.auto.image_processing_auto import AutoImageProcessor``
    of the processor property -- is simulated as missing: builtins.__import__
    denies exactly that module name and forwards every other import to the
    real machinery (a ``from ... import`` calls ``__import__`` every time,
    even for an already-cached module, so no cache surgery or subprocess is
    needed).  The production property must turn that ImportError into the
    task-local guidance with the original error preserved; production code is
    not touched.
    """

    def test_hint_names_the_task_local_install(self):
        import builtins
        from unittest import mock

        pre = unified_preprocess.OfficialPreprocessor(REAL_MODEL_DIR)
        original_import = builtins.__import__

        def deny_image_processing_auto(name, *args, **kwargs):
            if name == "transformers.models.auto.image_processing_auto":
                raise ImportError("simulated missing image dependency")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=deny_image_processing_auto):
            with self.assertRaises(ImportError) as ctx:
                pre.processor
        text = str(ctx.exception)
        self.assertIn("task-local", text)
        self.assertIn("deps/pillow", text)
        self.assertIn("deps/torchvision", text)
        self.assertIn("simulated missing image dependency", text)


class TestPayloadValidation(unittest.TestCase):
    """The strict payload contract needs no official components at all."""

    def setUp(self):
        self.pre = FakePreprocessor()

    def encode(self, rows, **caps):
        return self.pre.build_graph_inputs(self.pre.encode_payload_rows(rows), **caps)

    def test_input_ids_rows_encode(self):
        arrays = self.encode([{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6, 7]}])
        self.assertEqual(arrays["input_ids"].shape, (2, 4))
        self.assertEqual(arrays["actual_seq_lengths"].tolist(), [3, 4])
        self.assertTrue((arrays["image_feature_index"] == -1).all())

    def test_rows_run_at_their_true_length(self):
        arrays = self.encode([{"input_ids": [1, 2, 3]},
                              {"input_ids": [1 + (i % 90) for i in range(300)]}])
        self.assertEqual(arrays["input_ids"].shape, (2, 300))
        self.assertEqual(arrays["actual_seq_lengths"].tolist(), [3, 300])

    def test_float_and_bool_token_ids_rejected(self):
        for bad in ([1.0, 2], [True, 2], ["1", 2]):
            with self.assertRaises(ValueError):
                self.pre.encode_payload_rows([{"input_ids": bad}])

    def test_mixed_and_missing_row_fields_rejected(self):
        for bad in ({"input_ids": [1], "prompt": "p"},
                    {"images": ["a.ppm"]},
                    {"prompt": None},
                    "not-a-dict"):
            with self.assertRaises(ValueError):
                self.pre.encode_payload_rows([bad])

    def test_bad_images_field_rejected(self):
        for bad in ("a.ppm", [1, 2], [True]):
            with self.assertRaises(ValueError):
                self.pre.encode_payload_rows([{"prompt": "p", "images": bad}])

    def test_row_beyond_the_cap_rejected(self):
        with self.assertRaises(ValueError):
            self.encode([{"input_ids": [1] * 2049}])


class TestSharedRopeProvider(unittest.TestCase):
    """The preprocessor uses the SHARED official provider; no local duplicate."""

    def test_provider_is_the_multimodal_one(self):
        pre = unified_preprocess.OfficialPreprocessor.__new__(
            unified_preprocess.OfficialPreprocessor)
        pre._config = fake_folder_config()
        pre._rope_index = None
        self.assertTrue(callable(pre.rope_index))

    def test_planes_match_the_real_official_model(self):
        from transformers.models.qwen3_5 import configuration_qwen3_5 as c
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m

        tc = dict(hidden_size=64, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                  intermediate_size=128, num_hidden_layers=2, linear_num_key_heads=2,
                  linear_num_value_heads=2, linear_key_head_dim=16,
                  linear_value_head_dim=16, linear_conv_kernel_dim=4, vocab_size=128,
                  rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                   "partial_rotary_factor": 0.5,
                                   "mrope_section": [3, 3, 2]})
        vc = dict(depth=1, hidden_size=64, num_heads=4, patch_size=16,
                  spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
                  num_position_embeddings=64, out_hidden_size=64, intermediate_size=128)
        cfg = c.Qwen3_5Config(text_config=tc, vision_config=vc)
        torch.manual_seed(3)
        model = m.Qwen3_5Model(cfg).eval()
        provider = multimodal.official_rope_index_provider(cfg)

        ids = torch.tensor([[1, 2] + [cfg.image_token_id] * 16 + [3, 4]])
        types = torch.tensor([[0, 0] + [1] * 16 + [0, 0]])
        grid = torch.tensor([[1, 8, 8]])
        official, _ = model.get_rope_index(ids, mm_token_type_ids=types, image_grid_thw=grid,
                                           attention_mask=None)
        mine = provider(ids, types, grid)
        self.assertTrue(torch.equal(official, mine))

        ids2 = torch.tensor([[5, 6, 7, 8]])
        types2 = torch.zeros_like(ids2)
        official2, _ = model.get_rope_index(ids2, mm_token_type_ids=types2, attention_mask=None)
        self.assertTrue(torch.equal(official2, provider(ids2, types2, None)))


if __name__ == "__main__":
    unittest.main()
