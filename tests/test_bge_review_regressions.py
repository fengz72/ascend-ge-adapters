"""Regressions for importing supplied BGE samples into the persistent input pool."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core import data
from core.config import ROOT


class BgeInputPoolReviewRegressions(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.samples_file = self.root / "sample_inputs.json"
        self.config = {"_root": str(self.root), "data": {
            "script": str(ROOT / "models/bge/generate_cases.py"),
            "directory": str(self.root / "inputs"),
            "params": {"samples_file": "sample_inputs.json"}}}
        self.samples = [{"case_id": i, "input": {
            "query_token_ids": [101, i + 200, 102], "attention_mask": [1, 1, 1],
            "long_cid3_raw": [123, 0], "long_action": [0, -1],
            "long_time_bucket": [1, 0], "stat_dense_raw": [123] + [0] * 19,
            "stat_ratio": [0.75] + [0.0] * 19, "stat_count": 1}} for i in range(5)]
        self.write_samples()

    def write_samples(self):
        self.samples_file.write_text(json.dumps(self.samples))

    def test_imports_all_five_samples_renames_tokens_and_adds_batch_dimension(self):
        cases = data.generate(self.config)
        self.assertEqual([case["id"] for case in cases], [f"case-{i:04d}" for i in range(5)])
        self.assertEqual(len(list((self.root / "inputs").rglob("*.npy"))), 40)
        for case, sample in zip(cases, self.samples):
            self.assertEqual(case["samples"], 1)
            self.assertNotIn("query_token_ids", case["payload"])
            for key, value in sample["input"].items():
                name = "input_ids" if key == "query_token_ids" else key
                array = np.load(self.root / "inputs" / case["id"] / f"{name}.npy")
                expected = np.asarray([value], dtype=np.float32 if name == "stat_ratio" else np.int64)
                np.testing.assert_array_equal(array, expected)
                self.assertEqual(array.dtype, expected.dtype)
        self.assertEqual(cases, data.load(self.config))

    def test_shrinking_samples_removes_stale_cases_and_overwrites_retained_case(self):
        data.generate(self.config)
        self.samples = self.samples[:1]
        self.samples[0]["input"]["query_token_ids"] = [101, 777, 888, 102]
        self.samples[0]["input"]["attention_mask"] = [1] * 4
        self.write_samples()
        cases = data.generate(self.config)
        self.assertEqual([case["id"] for case in cases], ["case-0000"])
        self.assertEqual(cases[0]["payload"]["input_ids"], [[101, 777, 888, 102]])
        self.assertEqual(len(list((self.root / "inputs").rglob("*.npy"))), 8)
        self.assertEqual(cases, data.load(self.config))

    def test_invalid_sample_does_not_partially_replace_existing_pool(self):
        original = data.generate(self.config)
        fixtures = []
        duplicate = copy.deepcopy(self.samples)
        duplicate[-1]["case_id"] = 0
        fixtures.append(duplicate)
        for name, value in (("attention_mask", [1]), ("stat_count", 21),
                            ("long_action", [0.5, 1]), ("stat_ratio", [float("nan")] * 20)):
            malformed = copy.deepcopy(self.samples)
            malformed[0]["input"]["query_token_ids"][1] = 999
            malformed[-1]["input"][name] = value
            fixtures.append(malformed)
        for samples in fixtures:
            with self.subTest(samples=samples[-1]):
                self.samples_file.write_text(json.dumps(samples))
                with self.assertRaises(ValueError):
                    data.generate(self.config)
                self.assertEqual(data.load(self.config), original)

    def test_reading_persisted_inputs_does_not_require_json_or_original_model(self):
        original = data.generate(self.config)
        self.samples_file.unlink()
        self.assertEqual(data.load(self.config), original)
