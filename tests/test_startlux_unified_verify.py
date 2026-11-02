"""CPU tests for the unified-route business verifier (unified_verify.py).

The verifier reads three independent sources: the real decision_config
(letters + temperatures), the case pool's payload.rows_meta (per-row active
options) and the core CPU baseline golden.npz -- never candidate-derived
numbers.  These tests build a small golden + case pool + outputs entirely in a
temp work dir and drive every gate branch: exact passes, probability
deviations, top1 flips behind a reference margin > 0.02, missing golden or
metadata, misaligned columns, nonfinite scores, and per-row reports of real B2
batches.
"""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "startlux-decision-v1.1-0.8b-instant"
ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("startlux_unified_verify_test",
                                             str(MODEL_DIR / "unified_verify.py"))
unified_verify = importlib.util.module_from_spec(spec)
sys.modules["startlux_unified_verify_test"] = unified_verify
spec.loader.exec_module(unified_verify)

LETTERS = list(range(32, 58))
TEMPS = {"choice": 1.4364, "noul": 1.0411, "score": 1.4021}
ACTIVE = LETTERS[:3]                       # red / green / blue style: 3 options

DATA_SCRIPT = '''
import json
from pathlib import Path

POOL = json.loads(Path(__pool__).read_text())


def load(config):
    return POOL


def generate(config):
    return load(config)
'''


def workdir(prefix):
    import os

    base = Path(os.environ.get("TMPDIR") or (ROOT / "work" / "tmp"))
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(base)))


def softmax(values):
    weights = np.exp(values - values.max())
    return weights / weights.sum()


class VerifierCase(unittest.TestCase):
    def setUp(self):
        self.work = workdir("unified-verify-")
        self.addCleanup(lambda: __import__("shutil").rmtree(self.work,
                                                            ignore_errors=True))
        self.decision = self.work / "decision"
        self.decision.mkdir()
        (self.decision / "decision_config.json").write_text(json.dumps(
            {"letter_token_ids": LETTERS, "temperature_by_type": TEMPS}))
        (self.decision / "config.json").write_text(json.dumps(
            {"text_config": {"vocab_size": 300}}))
        self.work_dir = self.work / "model-work"
        self.work_dir.mkdir()

    def config(self, pool):
        pool_path = self.work / "pool.json"
        pool_path.write_text(json.dumps(pool))
        script = self.work / "data_script.py"
        script.write_text(DATA_SCRIPT.replace("__pool__", repr(str(pool_path))))
        return {"source": {"ref": str(self.decision)},
                "data": {"script": str(script)},
                "work_dir": str(self.work_dir),
                "verification": {"params": {}}}

    def pool_case(self, case_id, rows_meta):
        return {"id": case_id, "samples": len(rows_meta),
                "payload": {"rows": [{"input_ids": [1, 2]}] * len(rows_meta),
                            "rows_meta": rows_meta}}

    def meta(self, case_id="c1", question_type="choice", letter_ids=None):
        return {"case_id": case_id, "question_type": question_type,
                "letter_ids": list(letter_ids or ACTIVE), "letters_asked": 3}

    def save_golden(self, rows):
        """rows: {case_id: scores [B, 26]} in the core.worker.save_rows layout."""
        arrays = {}
        for case_id, scores in rows.items():
            arrays[f"{case_id}/metadata/alignment"] = np.asarray(json.dumps(
                {"rows": [case_id] * scores.shape[0], "columns": LETTERS}))
            arrays[f"{case_id}/outputs/scores"] = scores
        np.savez(self.work_dir / "golden.npz", **arrays)

    def outputs(self, rows):
        return {case_id: {"outputs": {"scores": scores},
                          "alignment": {"rows": [case_id] * scores.shape[0],
                                        "columns": list(LETTERS)}}
                for case_id, scores in rows.items()}

    def scores(self, vector, batch=1):
        base = np.zeros(26, dtype=np.float64)
        base[:len(vector)] = vector
        return np.tile(base, (batch, 1)).astype(np.float32)


class TestBusinessGate(VerifierCase):
    def test_exact_match_passes_with_per_row_reports(self):
        golden = self.scores([2.0, 0.0, 0.0], batch=2)
        self.save_golden({"c1": golden})
        pool = [self.pool_case("c1", [self.meta(), self.meta()])]
        reports = unified_verify.compare(self.config(pool), self.outputs({"c1": golden}))
        self.assertTrue(reports["c1"]["passed"], reports["c1"])
        self.assertEqual(len(reports["c1"]["rows"]), 2)
        self.assertEqual(reports["c1"]["rows"][0]["active_K"], 3)
        self.assertEqual(reports["c1"]["rows"][0]["temperature"], TEMPS["choice"])
        self.assertLessEqual(reports["c1"]["prob_max_abs"], 1e-6)

    def test_probability_deviation_fails(self):
        golden = self.scores([2.0, 0.0, 0.0])
        shifted = self.scores([1.35, 0.0, 0.0])       # p shift far over 0.01 at T=1.4364
        self.save_golden({"c1": golden})
        pool = [self.pool_case("c1", [self.meta()])]
        reports = unified_verify.compare(self.config(pool), self.outputs({"c1": shifted}))
        self.assertFalse(reports["c1"]["passed"])
        self.assertIn("probability max abs", reports["c1"]["reason"])
        self.assertGreater(reports["c1"]["prob_max_abs"], unified_verify.PROB_MAX_ABS_MAX)

    def test_top1_flip_behind_reference_margin_fails(self):
        golden = self.scores([4.0, 0.0, 0.0])          # big reference margin
        flipped = self.scores([0.0, 4.0, 0.0])         # candidate top1 flips
        self.save_golden({"c1": golden})
        pool = [self.pool_case("c1", [self.meta()])]
        reports = unified_verify.compare(self.config(pool), self.outputs({"c1": flipped}))
        self.assertFalse(reports["c1"]["passed"])
        self.assertIn("top1 letter inconsistent", reports["c1"]["reason"])

    def test_missing_golden_row_fails_not_passes(self):
        self.save_golden({"other": self.scores([1.0, 0.0, 0.0])})
        pool = [self.pool_case("c1", [self.meta()])]
        reports = unified_verify.compare(self.config(pool), self.outputs(
            {"c1": self.scores([1.0, 0.0, 0.0])}))
        self.assertFalse(reports["c1"]["passed"])
        self.assertIn("no golden baseline row", reports["c1"]["reason"])

    def test_missing_rows_meta_fails(self):
        self.save_golden({"c1": self.scores([1.0, 0.0, 0.0])})
        pool = [self.pool_case("other", [self.meta("other")])]
        reports = unified_verify.compare(self.config(pool), self.outputs(
            {"c1": self.scores([1.0, 0.0, 0.0])}))
        self.assertFalse(reports["c1"]["passed"])
        self.assertIn("no case metadata", reports["c1"]["reason"])

    def test_rows_meta_count_mismatch_fails(self):
        golden = self.scores([1.0, 0.0, 0.0], batch=2)
        self.save_golden({"c1": golden})
        pool = [self.pool_case("c1", [self.meta()])]   # 1 meta entry for 2 rows
        reports = unified_verify.compare(self.config(pool), self.outputs({"c1": golden}))
        self.assertFalse(reports["c1"]["passed"])
        self.assertIn("exactly 2 per-row entries", reports["c1"]["reason"])

    def test_misaligned_columns_fail(self):
        golden = self.scores([2.0, 0.0, 0.0])
        self.save_golden({"c1": golden})
        pool = [self.pool_case("c1", [self.meta()])]
        outputs = self.outputs({"c1": golden})
        outputs["c1"]["alignment"]["columns"] = list(reversed(LETTERS))
        reports = unified_verify.compare(self.config(pool), outputs)
        self.assertFalse(reports["c1"]["passed"])
        self.assertIn("alignment.columns", reports["c1"]["reason"])

    def test_nonfinite_candidate_fails(self):
        golden = self.scores([1.0, 0.0, 0.0])
        bad = self.scores([1.0, 0.0, 0.0])
        bad[0, 7] = np.inf
        self.save_golden({"c1": golden})
        pool = [self.pool_case("c1", [self.meta()])]
        reports = unified_verify.compare(self.config(pool), self.outputs({"c1": bad}))
        self.assertFalse(reports["c1"]["passed"])

    def test_unknown_active_letter_fails(self):
        golden = self.scores([1.0, 0.0, 0.0])
        self.save_golden({"c1": golden})
        pool = [self.pool_case("c1", [self.meta(letter_ids=[999, 998, 997])])]
        reports = unified_verify.compare(self.config(pool), self.outputs({"c1": golden}))
        self.assertFalse(reports["c1"]["passed"])
        self.assertIn("unknown letter token id", reports["c1"]["reason"])

    def test_invalid_temperature_fails(self):
        golden = self.scores([1.0, 0.0, 0.0])
        self.save_golden({"c1": golden})
        pool = [self.pool_case("c1", [self.meta(question_type="mystery")])]
        reports = unified_verify.compare(self.config(pool), self.outputs({"c1": golden}))
        self.assertFalse(reports["c1"]["passed"])
        self.assertIn("invalid temperature", reports["c1"]["reason"])

    def test_b2_one_bad_row_fails_the_case(self):
        golden = np.stack([self.scores([2.0, 0.0, 0.0])[0],
                           self.scores([2.0, 0.0, 0.0])[0]])
        candidate = golden.copy()
        candidate[1, 0] = 0.0                          # row 1 flips top1
        candidate[1, 1] = 4.0
        self.save_golden({"c1": golden})
        pool = [self.pool_case("c1", [self.meta(), self.meta()])]
        reports = unified_verify.compare(self.config(pool), self.outputs({"c1": candidate}))
        self.assertFalse(reports["c1"]["passed"])
        self.assertTrue(reports["c1"]["rows"][0]["passed"])
        self.assertFalse(reports["c1"]["rows"][1]["passed"])
        self.assertIn("row 1", reports["c1"]["reason"])

    def test_per_row_question_types_use_their_own_temperatures(self):
        golden = np.stack([self.scores([2.0, 0.0, 0.0])[0],
                           self.scores([2.0, 0.0])[0]])
        self.save_golden({"c1": golden})
        pool = [self.pool_case("c1", [self.meta(question_type="choice"),
                                      self.meta(question_type="noul",
                                                letter_ids=LETTERS[:2])])]
        reports = unified_verify.compare(self.config(pool), self.outputs({"c1": golden}))
        self.assertTrue(reports["c1"]["passed"], reports["c1"])
        self.assertEqual(reports["c1"]["rows"][0]["temperature"], TEMPS["choice"])
        self.assertEqual(reports["c1"]["rows"][1]["temperature"], TEMPS["noul"])

    def test_missing_golden_file_is_a_loud_error(self):
        pool = [self.pool_case("c1", [self.meta()])]
        with self.assertRaises(ValueError) as ctx:
            unified_verify.compare(self.config(pool), self.outputs(
                {"c1": self.scores([1.0, 0.0, 0.0])}))
        self.assertIn("golden.npz", str(ctx.exception))

    def test_missing_data_script_is_a_loud_error(self):
        self.save_golden({"c1": self.scores([1.0, 0.0, 0.0])})
        config = self.config([self.pool_case("c1", [self.meta()])])
        del config["data"]["script"]
        with self.assertRaises(ValueError) as ctx:
            unified_verify.compare(config, self.outputs({"c1": self.scores(
                [1.0, 0.0, 0.0])}))
        self.assertIn("data.script", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
