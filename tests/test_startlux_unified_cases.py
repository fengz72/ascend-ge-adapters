"""CPU tests for the unified-route case generator (generate_unified_cases.py).

No weights, no NPU.  The fixture-determinism and text-case tests need nothing
beyond the model directory; the image-case tests (guarded on the pinned real
materials) prove the OFFICIAL prompt rendering (placeholders, layout, thinking-
off suffix via the real tokenizer) and the payload contract (prompt text +
local fixture paths, no pixel JSON, no golden predictions anywhere).
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "startlux-decision-v1.1-0.8b-instant"
ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_DIR = ROOT / "work" / "startlux-materials" / \
    "model-6496ae641cc71b149ce9dad69bd97f2a44e7bf11"


def load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(MODEL_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = load("startlux_generate_unified_cases_test", "generate_unified_cases.py")


def workdir(prefix):
    import os

    base = Path(os.environ.get("TMPDIR") or (ROOT / "work" / "tmp"))
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(base)))


def config(**data_params):
    cfg = {"model": {"params": {"max_seq_len": 2048, "compiled_seq_len": 256,
                                "chunk_size": 64, "max_batch": 4,
                                "fixtures_file": "fixtures/p2b-readouts.json"}},
           "source": {"ref": str(REAL_MODEL_DIR)},
           "data": {"params": dict(data_params)}}
    return cfg


class TestFixtureDeterminism(unittest.TestCase):
    def test_regeneration_is_byte_identical(self):
        directory = workdir("unified-fixtures-")
        self.addCleanup(lambda: __import__("shutil").rmtree(directory,
                                                            ignore_errors=True))
        first = generator.ensure_fixtures(directory)
        red = first["fixture-red-256"].read_bytes()
        blue = first["fixture-blue-256"].read_bytes()
        second = generator.ensure_fixtures(directory)
        self.assertEqual(red, second["fixture-red-256"].read_bytes())
        self.assertEqual(blue, second["fixture-blue-256"].read_bytes())
        # PPM P6, 256x256, RGB
        self.assertTrue(red.startswith(b"P6\n256 256\n255\n"))
        self.assertEqual(len(red), len(b"P6\n256 256\n255\n") + 256 * 256 * 3)
        # the two fixtures are genuinely different pictures
        self.assertNotEqual(red, blue)
        # byte-identical to the canonical production fixtures directory
        canonical = generator.ensure_fixtures(generator.FIXTURES_DEFAULT)
        self.assertEqual(red, canonical["fixture-red-256"].read_bytes())
        self.assertEqual(blue, canonical["fixture-blue-256"].read_bytes())

    def test_corrupted_fixture_is_regenerated(self):
        directory = workdir("unified-fixtures-corrupt-")
        self.addCleanup(lambda: __import__("shutil").rmtree(directory,
                                                            ignore_errors=True))
        fixtures = generator.ensure_fixtures(directory)
        fixtures["fixture-red-256"].write_bytes(b"garbage")
        restored = generator.ensure_fixtures(directory)
        self.assertNotEqual(restored["fixture-red-256"].read_bytes(), b"garbage")
        canonical = generator.ensure_fixtures(generator.FIXTURES_DEFAULT)
        self.assertEqual(restored["fixture-red-256"].read_bytes(),
                         canonical["fixture-red-256"].read_bytes())

    def test_directory_override(self):
        directory = workdir("unified-fixtures-alt-")
        self.addCleanup(lambda: __import__("shutil").rmtree(directory,
                                                            ignore_errors=True))
        fixtures = generator.ensure_fixtures(directory)
        for path in fixtures.values():
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent, directory)


class TestTextCases(unittest.TestCase):
    """The seven canonical pure-text read-outs (no materials needed)."""

    def setUp(self):
        holder = tempfile.TemporaryDirectory(
            prefix="unified-cases-decision-",
            dir=str(ROOT / "work" / "tmp"))
        self.addCleanup(holder.cleanup)
        self.folder = Path(holder.name)
        (self.folder / "decision_config.json").write_text(json.dumps(
            {"letter_token_ids": list(range(32, 58)),
             "temperature_by_type": {"choice": 1.4364, "noul": 1.0411,
                                     "score": 1.4021}}))
        (self.folder / "config.json").write_text(json.dumps(
            {"text_config": {"vocab_size": 300}, "image_token_id": 100,
             "vision_start_token_id": 102, "vision_end_token_id": 103}))

    def config(self):
        cfg = config()
        cfg["source"]["ref"] = str(self.folder)      # _text_cases reads only params
        return cfg

    def test_seven_preencoded_text_cases(self):
        cases = generator._text_cases(self.config())
        self.assertEqual(len(cases), 7)
        fixture_ids = {r["case_id"] for r in json.loads(
            (MODEL_DIR / "fixtures" / "p2b-readouts.json").read_text())["readouts"]}
        self.assertEqual({c["id"] for c in cases},
                         {f"text-{cid}" for cid in fixture_ids})
        for case in cases:
            self.assertEqual(case["samples"], 1)
            row = case["payload"]["rows"][0]
            self.assertIn("input_ids", row)
            self.assertNotIn("prompt", row)
            self.assertTrue(all(type(t) is int for t in row["input_ids"]))
            meta = case["payload"]["rows_meta"][0]
            self.assertEqual(meta["case_id"], case["id"])
            self.assertIn(meta["question_type"], ("choice", "score", "noul"))
            self.assertTrue(meta["letter_ids"])


@unittest.skipUnless(REAL_MODEL_DIR.is_dir(), "the pinned real model materials are not present")
class TestImageCases(unittest.TestCase):
    def setUp(self):
        self.directory = workdir("unified-cases-images-")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.directory,
                                                            ignore_errors=True))

    def config(self):
        return config(fixtures_directory=str(self.directory))

    def test_full_pool_shape_and_official_prompts(self):
        cases = generator.generate(self.config())
        self.assertEqual(len(cases), 11)
        # the FIRST case is a pre-encoded text case (a plain CLI build encodes it
        # before the exporter prefers the deterministic trace example)
        self.assertIn("input_ids", cases[0]["payload"]["rows"][0])
        by_id = {c["id"]: c for c in cases}
        for case_id, images, placeholders, samples in (
                ("image-red-r0", 1, 1, 1), ("image-blue-r0", 1, 1, 1),
                ("image-two-a-b-r0", 2, 2, 1), ("mixed-text-image-b2-r0", 1, 1, 2)):
            case = by_id[case_id]
            rows = case["payload"]["rows"]
            self.assertEqual(case["samples"], samples)
            self.assertEqual(len(rows), samples)
            self.assertEqual(len(case["payload"]["rows_meta"]), samples)
            self.assertEqual(sum(len(r.get("images", [])) for r in rows), images)
            image_rows = [r for r in rows if "images" in r]
            for row in image_rows:
                self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", row["prompt"])
                # one placeholder per image of the row, in official order
                self.assertEqual(row["prompt"].count("<|image_pad|>"),
                                 len(row["images"]))
            for row in rows:
                if "images" not in row:
                    self.assertNotIn("<|image_pad|>", row["prompt"])
            for row in image_rows:
                for path in row["images"]:
                    self.assertTrue((ROOT / path).is_file(), path)
        # no golden predictions anywhere in the pool
        text = json.dumps(cases)
        for banned in ("scores26_golden", "golden_scores26", "golden_file"):
            self.assertNotIn(banned, text)

    def test_official_prompt_is_the_trained_format(self):
        sys.path.insert(0, str(MODEL_DIR / "vendor"))
        from startlux_decide.prompt import SYSTEM, THINK_OFF_SUFFIX

        cases = generator.generate(self.config())
        row = next(c for c in cases if c["id"] == "image-red-r0")["payload"]["rows"][0]
        prompt = row["prompt"]
        self.assertTrue(prompt.startswith(f"<|im_start|>system\n{SYSTEM}<|im_end|>"))
        self.assertTrue(prompt.endswith(THINK_OFF_SUFFIX))
        self.assertIn("Question: Which color occupies the largest area of the image?",
                      prompt)
        self.assertIn("A) red:", prompt)

    def test_active_letters_are_the_folder_letters(self):
        letters = json.loads((REAL_MODEL_DIR / "decision_config.json").read_text())\
            ["letter_token_ids"]
        cases = generator.generate(self.config())
        for case in cases:
            if not case["id"].startswith("image-") and case["id"] != "mixed-text-image-b2-r0":
                continue
            for meta in case["payload"]["rows_meta"]:
                self.assertEqual(meta["letter_ids"], letters[:3])
                self.assertEqual(meta["letters_asked"], 3)
                self.assertEqual(meta["question_type"], "choice")

    def test_load_equals_generate(self):
        cfg = self.config()
        self.assertEqual([c["id"] for c in generator.load(cfg)],
                         [c["id"] for c in generator.generate(cfg)])


if __name__ == "__main__":
    unittest.main()
