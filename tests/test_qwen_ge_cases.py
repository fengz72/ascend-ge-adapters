"""ge-cases/1 contract of the Qwen2.5-0.5B input pool writer.

CPU-only: the historical RNG sequence, the packed input layout, the pool
fingerprint and the on-disk cases.json + npy layout are all checked without
torch, HF weights or an NPU.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen2.5-0.5b" / "model"
VOCAB_SIZE = 151936  # Qwen2.5-0.5B
POOL_SHA256 = "cfeed98ec4a6"  # historical work/inputs pool fingerprint


def _load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_inputs = _load_module(MODEL_DIR / "generate_inputs.py")
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class GenerateCases(unittest.TestCase):
    def test_keeps_the_historical_case_ids_and_batch(self):
        cases = list(generate_inputs.generate_cases(VOCAB_SIZE))
        self.assertEqual(len(cases), generate_inputs.COUNT)
        self.assertEqual([case[0] for case in cases],
                         [f"case-{index:04d}" for index in range(generate_inputs.COUNT)])
        self.assertTrue(all(len(requests) == generate_inputs.BATCH_SIZE
                            for _, _, requests in cases))


class PackedInputs(unittest.TestCase):
    def test_layout_and_platform_bounds(self):
        _, prefix, requests = next(iter(generate_inputs.generate_cases(VOCAB_SIZE)))
        for platform, dtype in (("a3", np.int64), ("a5", np.int32)):
            with self.subTest(platform=platform):
                ids, positions, bounds = generate_inputs.packed_inputs(
                    prefix, requests, generate_inputs.BOUNDS_DTYPE[platform])
                self.assertEqual(ids.dtype, np.dtype("int64"))
                self.assertEqual(positions.dtype, np.dtype("int64"))
                self.assertEqual(len(ids), len(positions))
                self.assertEqual(bounds.dtype, np.dtype(dtype))
                self.assertEqual(len(bounds), len(requests) + 1)
                self.assertTrue(np.all(np.diff(bounds.astype(np.int64)) > 0))
                self.assertEqual(int(bounds[-1]), len(ids))
                self.assertEqual(int(bounds[0]), len(prefix))


class PoolFingerprint(unittest.TestCase):
    def test_sha256_matches_the_historical_records(self):
        digest = hashlib.sha256()
        total = 0
        for _, prefix, requests in generate_inputs.generate_cases(VOCAB_SIZE):
            arrays = generate_inputs.packed_inputs(
                prefix, requests, generate_inputs.BOUNDS_DTYPE["a3"])
            for array in arrays:
                digest.update(array.tobytes(order="C"))
            total += len(requests)
        self.assertEqual(total, generate_inputs.COUNT * generate_inputs.BATCH_SIZE)
        self.assertEqual(digest.hexdigest()[:12], POOL_SHA256)


class WritePool(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = self.directory / "inputs"

    def test_emits_cases_json_and_per_case_npy(self):
        self.out.mkdir()
        for stale in ("inputs.bin", "records.json"):
            (self.out / stale).write_bytes(b"stale")

        with contextlib.redirect_stdout(io.StringIO()):
            count, total, sha256 = generate_inputs.write_pool(
                self.out, VOCAB_SIZE, "a3")

        self.assertEqual(count, generate_inputs.COUNT)
        self.assertEqual(total, generate_inputs.COUNT * generate_inputs.BATCH_SIZE)
        self.assertEqual(sha256[:12], POOL_SHA256)
        for stale in ("inputs.bin", "records.json"):
            self.assertFalse((self.out / stale).exists())

        index = json.loads((self.out / "cases.json").read_text())
        cases = index["cases"]
        self.assertEqual(len(cases), generate_inputs.COUNT)

        expected = generate_inputs.generate_cases(VOCAB_SIZE)
        for case, (case_id, prefix, requests) in zip(cases, expected):
            self.assertEqual(case["id"], case_id)
            self.assertEqual(case["samples"], generate_inputs.BATCH_SIZE)
            self.assertEqual([item["name"] for item in case["inputs"]],
                             list(generate_inputs.INPUT_NAMES))
            arrays = generate_inputs.packed_inputs(
                prefix, requests, generate_inputs.BOUNDS_DTYPE["a3"])
            for item, array in zip(case["inputs"], arrays):
                self.assertEqual(set(item), {"name", "path"})
                self.assertFalse(Path(item["path"]).is_absolute())
                self.assertTrue((self.out / item["path"]).is_file())
                loaded = np.load(self.out / item["path"])
                self.assertEqual(loaded.dtype, array.dtype)
                self.assertEqual(loaded.shape, array.shape)
                np.testing.assert_array_equal(loaded, array)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed")
class WriteCaseScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = _load_module(MODEL_DIR / "golden.py")

    def setUp(self):
        # A fresh, empty directory: the case subdirectory does not exist yet.
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = self.directory / "golden"

    def test_creates_the_missing_case_directory_and_entry(self):
        tensor = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)

        entry = self.golden.write_case_scores(self.out, "case-0000", tensor)

        path = self.out / "case-0000" / "output_0.npy"
        self.assertTrue(path.is_file())
        loaded = np.load(path)
        np.testing.assert_array_equal(loaded, tensor)
        self.assertEqual(loaded.dtype, np.dtype("float32"))
        self.assertEqual(entry, {"name": "output_0", "dtype": "float32",
                                 "shape": [16, 8], "path": "case-0000/output_0.npy",
                                 "bytes": 16 * 8 * 4})

    def test_two_cases_get_separate_directories_in_order(self):
        zeros = np.zeros((16, 8), dtype=np.float32)
        entries = [self.golden.write_case_scores(self.out, case_id, zeros)
                   for case_id in ("case-0000", "case-0001")]

        for case_id, entry in zip(("case-0000", "case-0001"), entries):
            self.assertTrue((self.out / case_id / "output_0.npy").is_file())
            self.assertEqual(entry["path"], f"{case_id}/output_0.npy")
        self.assertEqual([entry["path"] for entry in entries],
                         ["case-0000/output_0.npy", "case-0001/output_0.npy"])


if __name__ == "__main__":
    unittest.main()
