"""Business Top-10 checks must gate normal compare independently of CPU golden."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.config import ROOT, load_authoring
from core.pipeline import execute
from core.verify import compare_business
from core.worker import compare_outputs, save_rows


class BgeBusinessVerification(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(dir=ROOT / "work")))
        self.expected_file = self.root / "expected_outputs.json"
        self.config = load_authoring(ROOT / "models/bge/config.yaml", source_paths=False)
        self.config["work_dir"] = str(self.root)
        self.config["verification"]["params"]["expected_outputs"] = str(self.expected_file)
        self.ids = [10112, 1, 5, 7, 10, 20, 30, 40, 50, 60]
        self.expected = [{"case_id": i, "expected": {"scores_len": 10112,
            "top10_dense_id": self.ids, "top10_score": [round(0.99 - rank * 0.05, 6) for rank in range(10)],
            "scores_sha256_16": "unused-summary"}} for i in range(2)]
        self.write_expected()
        self.rows = {}
        for i, item in enumerate(self.expected):
            scores = np.zeros((1, 10112), dtype=np.float32)
            scores[0, np.asarray(self.ids) - 1] = item["expected"]["top10_score"]
            self.rows[f"case-{i:04d}"] = {"outputs": {"scores": scores}, "alignment": {"rows": [0]}}

    def write_expected(self):
        self.expected_file.write_text(json.dumps(self.expected))

    def compare(self):
        # Identical CPU and GE arrays isolate the external-reference gate.
        for name in ("golden.npz", "outputs.npz"):
            save_rows(self.root / name, self.rows)
        return compare_outputs(self.config)

    def test_normal_compare_checks_all_ten_ids_in_order_with_one_based_dense_ids(self):
        result = self.compare()
        self.assertEqual(result["status"], "success")
        for report in result["accuracy"]["cases"].values():
            business = report["business"]
            self.assertTrue(business["top10_ids_match"])
            self.assertTrue(business["top10_order_match"])
            self.assertTrue(business["top10_scores_match"])
            self.assertEqual([rank["actual_dense_id"] for rank in business["ranks"]], self.ids)
            self.assertEqual(len(business["ranks"]), 10)

    def test_wrong_candidate_id_fails_even_when_cpu_comparison_passes(self):
        self.rows["case-0000"]["outputs"]["scores"][0, 100] = 1.0
        result = self.compare()
        report = result["accuracy"]["cases"]["case-0000"]
        self.assertEqual(result["status"], "failed")
        self.assertTrue(report["outputs"]["scores"]["passed"])
        self.assertFalse(report["business"]["top10_ids_match"])
        self.assertEqual(result["accuracy"]["failed_cases"], ["case-0000"])

    def test_same_ids_in_wrong_order_fail(self):
        scores = self.rows["case-0000"]["outputs"]["scores"]
        scores[0, [10111, 0]] = scores[0, [0, 10111]]
        report = self.compare()["accuracy"]["cases"]["case-0000"]
        self.assertTrue(report["business"]["top10_ids_match"])
        self.assertFalse(report["business"]["top10_order_match"])
        self.assertFalse(report["passed"])

    def test_score_tolerance_gates_compare_and_keeps_per_rank_errors(self):
        scores = self.rows["case-0000"]["outputs"]["scores"]
        scores[0, 10111] += 0.0005
        self.assertEqual(self.compare()["status"], "success")
        scores[0, 10111] += 0.002
        result = self.compare()
        report = result["accuracy"]["cases"]["case-0000"]["business"]
        self.assertFalse(report["top10_scores_match"])
        self.assertTrue(report["top10_order_match"])
        self.assertGreater(report["max_abs_error"], 0.001)
        self.assertGreater(report["ranks"][0]["abs_error"], 0.001)
        self.assertEqual(result["status"], "failed")

    def test_compare_reads_current_expected_file_and_subset_without_inputs_or_model(self):
        self.compare()
        self.expected[1]["expected"]["top10_score"][0] += 0.01
        self.write_expected()
        self.assertEqual(compare_outputs(self.config)["status"], "failed")
        self.config["verification"]["case_ids"] = ["case-0000"]
        self.assertEqual(compare_outputs(self.config)["status"], "success")
        self.assertFalse((self.root / "inputs").exists())

    def test_missing_expected_case_cannot_silently_skip_verification(self):
        self.expected = self.expected[:1]
        self.write_expected()
        with self.assertRaisesRegex(ValueError, "case IDs differ"):
            self.compare()
        self.config["verification"]["case_ids"] = ["case-0001"]
        with self.assertRaisesRegex(ValueError, "missing expected cases"):
            self.compare()

    def test_invalid_reference_and_nonfinite_outputs_fail(self):
        original = copy.deepcopy(self.expected)
        for value in ([0] + self.ids[1:], [10113] + self.ids[1:], [1] * 10):
            with self.subTest(ids=value):
                self.expected = copy.deepcopy(original)
                self.expected[0]["expected"]["top10_dense_id"] = value
                self.write_expected()
                with self.assertRaisesRegex(ValueError, "invalid expected Top-10"):
                    self.compare()
        self.expected = original
        self.write_expected()
        self.rows["case-0000"]["outputs"]["scores"][0, 0] = np.nan
        result = self.compare()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["accuracy"]["cases"]["case-0000"]["business"]["reason"], "nonfinite")

    def test_pipeline_compare_fails_and_records_business_mismatch(self):
        self.rows["case-0000"]["outputs"]["scores"][0, 100] = 1.0
        self.compare()
        path = self.root / "config.yaml"
        path.write_text(f"""model:
  name: bge-business-test
source:
  kind: onnx
  ref: ${{MISSING_MODEL_FOR_COMPARE}}
target: Ascend910_9382
work_dir: {self.root}
verification:
  script: models/bge/verify.py
  params:
    expected_outputs: {self.expected_file}
    atol: 0.001
  outputs:
    scores:
      policy: numeric
""")
        with self.assertRaisesRegex(ValueError, "accuracy failed.*case-0000"):
            execute("compare", path)
        record = json.loads(next((self.root / "results").rglob("result.json")).read_text())
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["accuracy"]["cases"]["case-0000"]["business"]["top10_ids_match"])

    def test_business_hook_must_report_every_selected_case(self):
        script = self.root / "invalid_verify.py"
        script.write_text("def compare(config, outputs): return {}\n")
        self.config["verification"]["script"] = str(script)
        with self.assertRaisesRegex(ValueError, "every selected case"):
            compare_business(self.config, self.rows)
