"""Behavioral tests for the StartLux business verification script (verify.py).

Everything runs against the REAL contracts, not invented schemas:

  * rows come from the adapter's actual RuntimeHandle.decode output --
    scores (1, 26) with alignment {"rows": [case_id], "columns": letters26}
    (work/startlux-cannbot/p4-io-contract.json); malformed rows for the
    rejection tests are explicit variants of that schema, never a second one;
  * configs come from core.config.load_authoring(..., source_paths=False),
    so source.ref reaches the verifier exactly as in compare/infer: an
    unexpanded ${VAR} placeholder;
  * one integration test drives core.worker.compare_outputs over the copied
    real pipeline outputs.npz/golden.npz (work/startlux-validation/
    p4-verifier/it-work).

Expected gate outcomes are derived in-test from the fixture reference numbers
with the documented official softmax/margin math, re-derived here
independently of the implementation.

The scratch model folder carries only decision_config.json + config.json --
no weights -- proving the verifier is light; the adapter's RuntimeHandle is
array-only as well.  One test additionally uses the real pinned model folder
when present.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "startlux-decision-v1.1-0.8b-instant"
VERIFY = MODEL_DIR / "verify.py"
ADAPTER = MODEL_DIR / "startlux_npu_adapter.py"
FIXTURE = MODEL_DIR / "fixtures" / "p2b-readouts.json"
SCRATCH = ROOT / "work" / "startlux-validation" / "p4-verifier" / "test-scratch"
IT_WORK = ROOT / "work" / "startlux-validation" / "p4-verifier" / "it-work"
REAL_SOURCE = ROOT / "work" / "startlux-materials" / "model-6496ae641cc71b149ce9dad69bd97f2a44e7bf11"

LETTERS = list(range(32, 58))
TEMPERATURES = {"choice": 1.4364, "noul": 1.0411, "score": 1.4021}
PROB_GATE = 0.01
MARGIN_TRIGGER = 0.02
CASE_IDS = ["choice_short-r0", "score_short-r0", "noul_short-r0", "allow_none-r0",
            "wide_27-r0", "wide_27-r1", "wide_27-r2"]
ENV_NAME = "STARTLUX_P4_VERIFY_TEST_MODEL"
INLINE_TARGET = {"id": "p4-verify-test", "family": "p4-verify-test", "soc": "p4-verify-test"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    weights = np.exp(values - values.max())
    return weights / weights.sum()


class ScratchCase(unittest.TestCase):
    """Hermetic scratch model folder + real decode/authoring machinery."""

    @classmethod
    def setUpClass(cls):
        cls.verifier = load_module(f"verify_under_{cls.__name__.lower()}", VERIFY)
        cls.adapter = load_module(f"adapter_for_{cls.__name__.lower()}", ADAPTER)
        cls.dir = SCRATCH / cls.__name__.lower()
        if cls.dir.exists():
            shutil.rmtree(cls.dir)
        cls.dir.mkdir(parents=True)
        cls.source = cls.dir / "source"
        cls.source.mkdir()
        (cls.source / "decision_config.json").write_text(json.dumps(
            {"letter_token_ids": LETTERS, "temperature_by_type": TEMPERATURES}))
        (cls.source / "config.json").write_text(json.dumps(
            {"text_config": {"vocab_size": 151936}}))
        cls.fixture = json.loads(FIXTURE.read_text())
        cls.readouts = {r["case_id"]: r for r in cls.fixture["readouts"]}
        cls.runtime = cls.adapter.prepare_runtime(
            {"model": {"params": {"fixtures_file": "fixtures/p2b-readouts.json"}},
             "source": {"ref": str(cls.source)}},
            {"schema": "ge-io/1",
             "outputs": [{"name": "scores", "dtype": "float32", "shape": [-1, 26]}]})
        cls.previous_env = os.environ.get(ENV_NAME)
        os.environ[ENV_NAME] = str(cls.source)

    @classmethod
    def tearDownClass(cls):
        if cls.previous_env is None:
            os.environ.pop(ENV_NAME, None)
        else:
            os.environ[ENV_NAME] = cls.previous_env
        shutil.rmtree(cls.dir, ignore_errors=True)

    def authored(self, ref=None, verification_params=None, work_dir=None):
        from core.config import load_authoring
        text = (f"model:\n"
                f"  name: startlux-decision-v1.1-0.8b-instant\n"
                f"  adapter_file: {ADAPTER}\n"
                f"  params:\n    fixtures_file: fixtures/p2b-readouts.json\n"
                f"source:\n  kind: transformers\n"
                f"  loader: Qwen3_5ForConditionalGeneration\n"
                f"  ref: {ref if ref is not None else '${' + ENV_NAME + '}'}\n"
                f"verification:\n  case_ids: all\n"
                f"  outputs:\n    scores: {{policy: numeric, cosine_min: 0.999, "
                f"relative_l2_max: 0.02}}\n"
                f"  script: {VERIFY}\n"
                f"  params: {json.dumps(dict(verification_params or {}))}\n"
                f"work_dir: {work_dir if work_dir is not None else self.dir}\n")
        path = self.dir / "authored.yaml"
        path.write_text(text)
        return load_authoring([path], target=dict(INLINE_TARGET), source_paths=False)

    def config(self, verification_params=None):
        return self.authored(verification_params=verification_params)

    def decode_row(self, case_id, scores=None, samples=1):
        """A row EXACTLY as the adapter's RuntimeHandle.decode produces it."""
        if scores is None:
            scores = np.asarray(self.readouts[case_id]["scores26_golden_fp32"],
                                dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(samples, 26)
        return self.runtime.decode({"scores": scores}, {"id": case_id, "samples": samples})

    def oracle_rows(self, case_ids=CASE_IDS):
        return {case_id: self.decode_row(case_id) for case_id in case_ids}

    def reference(self, case_id):
        return np.asarray(self.readouts[case_id]["scores26_golden_fp32"], dtype=np.float64)

    def active_columns(self, case_id):
        return [LETTERS.index(token) for token in self.readouts[case_id]["letter_ids"]]

    def perturbed(self, case_id, column, delta):
        scores = self.reference(case_id).copy()
        scores[column] += delta
        return scores


class OraclePass(ScratchCase):
    def test_oracle_identical_passes_all_seven(self):
        rows = self.oracle_rows()
        reports = self.verifier.compare(self.config(), rows)
        self.assertEqual(set(reports), set(rows))
        for case_id, report in reports.items():
            self.assertTrue(report["passed"], report.get("reason"))
            self.assertEqual(report["prob_max_abs"], 0.0)
            self.assertEqual(report["temperature"],
                             TEMPERATURES[self.readouts[case_id]["question_type"]])
            self.assertEqual(report["active_K"], len(self.active_columns(case_id)))
        self.assertNotIn("question_type", rows["choice_short-r0"]["alignment"])

    def test_margin_and_prob_metrics_match_independent_spec_math(self):
        for case_id in CASE_IDS:
            readout = self.readouts[case_id]
            active = self.active_columns(case_id)
            temperature = TEMPERATURES[readout["question_type"]]
            probs = softmax(self.reference(case_id)[active] / temperature)
            ordered = np.sort(probs)[::-1]
            expected_margin = float(ordered[0] - ordered[1]) if len(active) > 1 else 1.0
            report = self.verifier.compare(self.config(),
                                           {case_id: self.decode_row(case_id)})[case_id]
            self.assertAlmostEqual(report["margin"], expected_margin, places=12)
            self.assertAlmostEqual(report["prob_max_abs"], 0.0, places=12)

    @unittest.skipUnless(REAL_SOURCE.is_dir(), "the pinned real model materials are not present")
    def test_real_source_materials_pass(self):
        previous = os.environ[ENV_NAME]
        os.environ[ENV_NAME] = str(REAL_SOURCE)
        try:
            config = self.authored()
            reports = self.verifier.compare(config, self.oracle_rows())
        finally:
            os.environ[ENV_NAME] = previous
        self.assertTrue(all(report["passed"] for report in reports.values()))


class ActiveKGate(ScratchCase):
    def test_gate_uses_active_k_not_diluted_26_softmax(self):
        case_id = "wide_27-r2"
        active = self.active_columns(case_id)
        temperature = TEMPERATURES[self.readouts[case_id]["question_type"]]
        column = active[int(np.argmin(self.reference(case_id)[active]))]
        rows = {case_id: self.decode_row(case_id, scores=self.perturbed(case_id, column, 0.3))}
        report = self.verifier.compare(self.config(), rows)[case_id]
        active_delta = float(np.abs(softmax(self.perturbed(case_id, column, 0.3)[active] / temperature)
                                    - softmax(self.reference(case_id)[active] / temperature)).max())
        full26_delta = float(np.abs(softmax(self.perturbed(case_id, column, 0.3) / temperature)
                                    - softmax(self.reference(case_id) / temperature)).max())
        self.assertGreater(active_delta, PROB_GATE)
        self.assertLess(full26_delta, PROB_GATE)
        self.assertFalse(report["passed"])
        self.assertGreater(report["prob_max_abs"], PROB_GATE)
        self.assertIn("probability", report["reason"])

    def test_top1_flip_with_stable_margin_is_rejected(self):
        case_id = "allow_none-r0"
        active = self.active_columns(case_id)
        temperature = TEMPERATURES[self.readouts[case_id]["question_type"]]
        probs = softmax(self.reference(case_id)[active] / temperature)
        ordered = np.sort(probs)[::-1]
        self.assertGreater(float(ordered[0] - ordered[1]), MARGIN_TRIGGER)
        runner_up = active[int(np.argsort(-probs)[1])]
        rows = {case_id: self.decode_row(case_id, scores=self.perturbed(case_id, runner_up, 5.0))}
        report = self.verifier.compare(self.config(), rows)[case_id]
        self.assertFalse(report["passed"])
        self.assertFalse(report["top1_consistent"])
        self.assertIn("top1", report["reason"])

    def test_perturbation_within_gate_still_passes(self):
        case_id = "wide_27-r2"
        active = self.active_columns(case_id)
        column = active[int(np.argmin(self.reference(case_id)[active]))]
        rows = {case_id: self.decode_row(case_id, scores=self.perturbed(case_id, column, 0.01))}
        report = self.verifier.compare(self.config(), rows)[case_id]
        self.assertTrue(report["passed"], report.get("reason"))
        self.assertLess(report["prob_max_abs"], PROB_GATE)
        self.assertGreater(report["prob_max_abs"], 0.0)


class ShapeAndAlignment(ScratchCase):
    def test_flat_26_compat_passes(self):
        row = self.decode_row("noul_short-r0")
        row["outputs"]["scores"] = row["outputs"]["scores"][0]
        report = self.verifier.compare(self.config(), {"noul_short-r0": row})["noul_short-r0"]
        self.assertTrue(report["passed"], report.get("reason"))

    def test_batched_and_wrong_shapes_rejected(self):
        for scores in (np.zeros((2, 26), dtype=np.float32),
                       np.zeros((1, 25), dtype=np.float32),
                       np.zeros(25, dtype=np.float32),
                       np.zeros((), dtype=np.float32)):
            row = self.decode_row("noul_short-r0", scores=np.zeros(26, dtype=np.float32))
            row["outputs"]["scores"] = scores
            report = self.verifier.compare(self.config(), {"noul_short-r0": row})["noul_short-r0"]
            self.assertFalse(report["passed"])
            self.assertIn("shape", report["reason"])

    def test_nonfinite_scores_rejected(self):
        for bad in (np.inf, -np.inf, np.nan):
            scores = self.reference("score_short-r0").astype(np.float32).reshape(1, 26)
            scores[0, 3] = bad
            row = self.decode_row("score_short-r0", scores=np.zeros(26, dtype=np.float32))
            row["outputs"]["scores"] = scores
            report = self.verifier.compare(self.config(), {"score_short-r0": row})["score_short-r0"]
            self.assertFalse(report["passed"])
            self.assertIn("nonfinite", report["reason"])

    def test_wrong_alignment_columns_rejected(self):
        for columns in (LETTERS[:-1], LETTERS[::-1], LETTERS[:-1] + [999], "ABCDEFGHIJ"):
            row = self.decode_row("choice_short-r0")
            row["alignment"]["columns"] = columns
            report = self.verifier.compare(self.config(), {"choice_short-r0": row})["choice_short-r0"]
            self.assertFalse(report["passed"])
            self.assertIn("columns", report["reason"])

    def test_alignment_identifier_contract(self):
        wrong_rows = self.decode_row("choice_short-r0")
        wrong_rows["alignment"]["rows"] = ["some-other-case"]
        missing = self.decode_row("choice_short-r0")
        del missing["alignment"]["rows"]
        for row in (wrong_rows, missing):
            report = self.verifier.compare(self.config(), {"choice_short-r0": row})["choice_short-r0"]
            self.assertFalse(report["passed"])
            self.assertIn("alignment", report["reason"])
        for legacy_field in ("readout", "id"):
            legacy = self.decode_row("choice_short-r0")
            del legacy["alignment"]["rows"]
            legacy["alignment"][legacy_field] = "choice_short-r0"
            report = self.verifier.compare(self.config(), {"choice_short-r0": legacy})["choice_short-r0"]
            self.assertTrue(report["passed"], report.get("reason"))

    def test_question_type_mismatch_rejected_but_absence_is_legal(self):
        mismatched = self.decode_row("choice_short-r0")
        mismatched["alignment"]["question_type"] = "score"
        report = self.verifier.compare(self.config(), {"choice_short-r0": mismatched})["choice_short-r0"]
        self.assertFalse(report["passed"])
        self.assertIn("question_type", report["reason"])
        absent = self.decode_row("choice_short-r0")
        report = self.verifier.compare(self.config(), {"choice_short-r0": absent})["choice_short-r0"]
        self.assertTrue(report["passed"], report.get("reason"))

    def test_missing_scores_or_alignment_rejected(self):
        for row in ({"outputs": {}, "alignment": self.decode_row("noul_short-r0")["alignment"]},
                    {"outputs": {"scores": np.zeros((1, 26), dtype=np.float32)}}):
            report = self.verifier.compare(self.config(), {"noul_short-r0": row})["noul_short-r0"]
            self.assertFalse(report["passed"])


class ReferenceAndTemperature(ScratchCase):
    def test_temperature_comes_from_source_not_fixture_provenance(self):
        lying = json.loads(json.dumps(self.fixture))
        lying["provenance"]["temperature_by_type"]["choice"] = 50.0
        fixture_path = self.dir / "fixture-lying-provenance.json"
        fixture_path.write_text(json.dumps(lying))
        case_id = "choice_short-r0"
        active = self.active_columns(case_id)
        reference = self.reference(case_id)
        top = active[int(np.argmax(reference[active]))]
        rows = {case_id: self.decode_row(case_id, scores=self.perturbed(case_id, top, 0.5))}
        config = self.config(verification_params={
            "fixtures_file": os.path.relpath(fixture_path, ROOT)})
        report = self.verifier.compare(config, rows)[case_id]
        under_source = float(np.abs(softmax(self.perturbed(case_id, top, 0.5)[active] / TEMPERATURES["choice"])
                                    - softmax(reference[active] / TEMPERATURES["choice"])).max())
        under_lie = float(np.abs(softmax(self.perturbed(case_id, top, 0.5)[active] / 50.0)
                                 - softmax(reference[active] / 50.0)).max())
        self.assertGreater(under_source, PROB_GATE)
        self.assertLess(under_lie, PROB_GATE)
        self.assertFalse(report["passed"])
        self.assertGreater(report["prob_max_abs"], PROB_GATE)

    def test_invalid_temperature_rejected(self):
        partial = self.dir / "source-partial"
        partial.mkdir(exist_ok=True)
        (partial / "decision_config.json").write_text(json.dumps(
            {"letter_token_ids": LETTERS,
             "temperature_by_type": {"noul": TEMPERATURES["noul"], "score": TEMPERATURES["score"]}}))
        previous = os.environ[ENV_NAME]
        os.environ[ENV_NAME] = str(partial)
        try:
            config = self.authored()
            reports = self.verifier.compare(config, self.oracle_rows(
                ["choice_short-r0", "noul_short-r0"]))
        finally:
            os.environ[ENV_NAME] = previous
        self.assertFalse(reports["choice_short-r0"]["passed"])
        self.assertIn("invalid temperature", reports["choice_short-r0"]["reason"])
        self.assertTrue(reports["noul_short-r0"]["passed"])

    def test_reference_letter_ids_validated(self):
        for broken_ids in ([32, 999], []):
            broken = json.loads(json.dumps(self.fixture))
            for readout in broken["readouts"]:
                if readout["case_id"] == "noul_short-r0":
                    readout["letter_ids"] = broken_ids
            fixture_path = self.dir / "fixture-broken-letters.json"
            fixture_path.write_text(json.dumps(broken))
            config = self.config(verification_params={
                "fixtures_file": os.path.relpath(fixture_path, ROOT)})
            report = self.verifier.compare(config, {"noul_short-r0": self.decode_row("noul_short-r0")})["noul_short-r0"]
            self.assertFalse(report["passed"])
            self.assertIn("nonempty" if not broken_ids else "unknown letter", report["reason"])

    def test_unset_source_env_var_raises_clearly(self):
        config = self.authored(ref="${STARTLUX_P4_VERIFY_UNSET_VAR}")
        with self.assertRaises(ValueError) as raised:
            self.verifier.compare(config, self.oracle_rows(["noul_short-r0"]))
        self.assertIn("STARTLUX_P4_VERIFY_UNSET_VAR", str(raised.exception))
        self.assertIn("is not set or is empty", str(raised.exception))

    def test_relative_source_ref_resolves_against_root_not_cwd(self):
        config = self.authored(ref=os.path.relpath(self.source, ROOT))
        original = os.getcwd()
        os.chdir(ROOT / "models")
        try:
            report = self.verifier.compare(config, {"noul_short-r0": self.decode_row("noul_short-r0")})["noul_short-r0"]
        finally:
            os.chdir(original)
        self.assertTrue(report["passed"], report.get("reason"))


class FrameworkWiring(ScratchCase):
    def test_core_compare_business_loads_script_and_exact_covers(self):
        from core.verify import compare_business
        rows = self.oracle_rows()
        reports = compare_business(self.config(), rows)
        self.assertEqual(set(reports), set(rows))
        self.assertTrue(all(report["passed"] for report in reports.values()))

    @unittest.skipUnless((IT_WORK / "golden.npz").is_file() and (IT_WORK / "outputs.npz").is_file(),
                         "the copied real pipeline outputs are not present")
    def test_core_compare_outputs_integration(self):
        from core.worker import compare_outputs
        config = self.authored(work_dir=IT_WORK)
        result = compare_outputs(config)
        self.assertEqual(result["status"], "success")
        cases = result["accuracy"]["cases"]
        self.assertEqual(sorted(cases), sorted(CASE_IDS))
        for case_id, report in cases.items():
            self.assertTrue(report["passed"], report.get("reason"))
            self.assertIn("business", report)
            self.assertTrue(report["business"]["passed"], report["business"].get("reason"))
            self.assertEqual(report["business"]["active_K"], len(self.active_columns(case_id)))


class Purity(unittest.TestCase):
    def test_imports_no_torch_or_transformers(self):
        code = ("import importlib.util, sys\n"
                f"spec = importlib.util.spec_from_file_location('v', {str(VERIFY)!r})\n"
                "module = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(module)\n"
                "assert 'torch' not in sys.modules, 'verify.py must not import torch'\n"
                "assert 'transformers' not in sys.modules, 'verify.py must not import transformers'\n")
        result = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
