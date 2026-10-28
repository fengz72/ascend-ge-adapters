"""CPU-only scenario/input regressions; synthetic events are never NPU evidence."""
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import workload_identity as w
import verify_outputs as v


class WorkloadIdentity(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.ref = w.reference_dir()
        self.original = json.loads((self.ref.parent / "original-plan.json").read_text())
        self.raw = (self.ref.parent / "original-inputs/inputs.bin").read_bytes()

    def plan(self, fixed=None, legacy=False):
        plan = copy.deepcopy(self.original)
        plan.update(instances=1, seed=42, device=10)
        cases = plan["inputs"]["cases"]
        if fixed:
            cases = [c for c in cases if c["id"] == fixed]
        blob = bytearray()
        for case in cases:
            for tensor in case["tensors"]:
                start, size = tensor["offset"], tensor["bytes"]
                tensor["offset"] = len(blob)
                blob.extend(self.raw[start:start + size])
        file = self.directory / (fixed or "random")
        file.write_bytes(blob)
        plan["inputs"] = {"file": str(file), "cases": cases}
        if not legacy:
            plan["workload"] = {"kind": "fixed_case" if fixed else "random_cases",
                                "case_ids": [c["id"] for c in cases],
                                "sampling": "fixed" if fixed else "uniform_with_replacement"}
        return plan

    def timing(self, plan):
        ids = [c["id"] for c in plan["inputs"]["cases"]]
        warm = []
        for i in range(10 * len(ids)):
            cid = ids[i % len(ids)]
            size = sum(t["bytes"] for t in next(c for c in plan["inputs"]["cases"] if c["id"] == cid)["tensors"])
            warm.append({"run_id": "cpu-test", "request_id": f"warmup-0-{i}", "instance": 0,
                         "case_id": cid, "phase": "warmup", "status": "success",
                         "started_ns": i * 1000000, "completed_ns": (i + 1) * 1000000,
                         "h2d_ms": .1, "desc_ms": .1, "execute_ms": .7, "d2h_ms": .1,
                         "h2d_bytes": size, "d2h_bytes": 40448})
        measure = []
        for i in range(100):
            row = dict(warm[i % len(warm)], phase="measure", request_id=i,
                       started_ns=100000000 + i * 1000000, completed_ns=101000000 + i * 1000000)
            measure.append(row)
        performance = {"window_start_ns": 100000000, "window_end_ns": 200000000,
                       "window_successes": 100, "cohort_successes": 100, "drain_ms": 0,
                       "sampling": {"seed": 42, "pool_size": len(ids)}}
        pp = self.directory / "plan.json"; pp.write_text(json.dumps(plan))
        ep = self.directory / "events.jsonl"; ep.write_text("".join(json.dumps(e) + "\n" for e in warm + measure))
        fp = self.directory / "performance.json"; fp.write_text(json.dumps(performance))
        return pp, ep, fp

    def test_all_five_legal_fixed_ids_match_their_own_original_bytes(self):
        for cid in w.CASE_IDS:
            result = w.check_plan_inputs(self.plan(cid))
            self.assertEqual(result["workload"]["case_ids"], [cid])
        fixed = w.check_plan_inputs(self.plan("case-0002"))
        self.assertEqual(fixed["input_bytes"], 3448)
        self.assertEqual(fixed["input_binary_sha256"], "27ce13c73cb35762ee7d0396d4425829d7e25a3b8ae2a12b37ed7f96b8690aa5")

    def test_random_and_legacy_five_case_plans_pass(self):
        for legacy in (False, True):
            self.assertEqual(w.check_plan_inputs(self.plan(legacy=legacy))["input_bytes"], 17304)

    def test_fixed_label_on_five_entry_pool_is_rejected(self):
        plan = self.plan(); plan["workload"].update(kind="fixed_case", sampling="fixed")
        with self.assertRaises(AssertionError): w.check_plan_inputs(plan)

    def test_unknown_case_and_metadata_pool_mismatch_are_rejected(self):
        for unknown in (True, False):
            plan = self.plan("case-0002")
            plan["workload"]["case_ids"] = ["case-9999" if unknown else "case-0004"]
            if unknown: plan["inputs"]["cases"][0]["id"] = "case-9999"
            with self.assertRaises(AssertionError): w.check_plan_inputs(plan)

    def test_missing_random_case_and_undeclared_fixed_pool_are_rejected(self):
        for plan in (self.plan(), self.plan("case-0002", legacy=True)):
            if "workload" in plan:
                plan["inputs"]["cases"].pop()
                plan["workload"]["case_ids"].pop()
            with self.assertRaises(AssertionError): w.check_plan_inputs(plan)

    def test_tampered_bytes_and_same_shape_wrong_case_are_rejected(self):
        plan = self.plan("case-0002"); file = Path(plan["inputs"]["file"])
        raw = bytearray(file.read_bytes()); raw[0] ^= 1; file.write_bytes(raw)
        with self.assertRaises(AssertionError): w.check_plan_inputs(plan)
        plan = self.plan("case-0002")
        other = self.plan("case-0004")
        file.write_bytes(Path(other["inputs"]["file"]).read_bytes())
        with self.assertRaises(AssertionError): w.check_plan_inputs(plan)

    def test_fixed_and_random_performance_events_pass(self):
        for selected in ("case-0002", None):
            pp, ep, fp = self.timing(self.plan(selected))
            result = v.perf_check(fp, ep, 1, 0, 100, 10, pp)
            self.assertTrue(result["passed"])
            self.assertEqual(result["qps"], 1000)

    def test_fixed_events_cannot_pass_without_explicit_plan(self):
        pp, ep, fp = self.timing(self.plan("case-0002"))
        with self.assertRaises(AssertionError): v.perf_check(fp, ep, 1, 0, 100, 10)

    def test_fixed_performance_rejects_runtime_pool_or_event_case_mismatch(self):
        for bad_pool in (True, False):
            pp, ep, fp = self.timing(self.plan("case-0002"))
            if bad_pool:
                perf = json.loads(fp.read_text()); perf["sampling"]["pool_size"] = 5; fp.write_text(json.dumps(perf))
            else:
                rows = v.read_jsonl(ep); rows[-1]["case_id"] = "case-0004"
                ep.write_text("".join(json.dumps(e) + "\n" for e in rows))
            with self.assertRaises(AssertionError): v.perf_check(fp, ep, 1, 0, 100, 10, pp)

    def test_fixed_plan_cli_passes(self):
        plan = self.plan("case-0002"); pp = self.directory / "fixed-plan.json"; pp.write_text(json.dumps(plan))
        p = subprocess.run([sys.executable, str(Path(__file__).with_name("verify_plan_inputs.py")),
                            "--plan", str(pp), "--report", str(self.directory / "identity.json")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_completed_fixed_identity_requires_actual_runtime_pool_one(self):
        plan = self.plan("case-0002")
        name = "cpu-test"; folder = self.directory / name; folder.mkdir()
        jobs = self.directory / "jobs" / name; jobs.mkdir(parents=True)
        binary = self.directory / "fake-binary"; binary.write_bytes(b"CPU identity fixture")
        graph = self.directory / "fake-graph"; graph.write_bytes(b"CPU graph fixture")
        plan["graph"]["path"] = str(graph)
        plan["output_dir"] = str(folder)
        plan["replay"] = {"run_id": name, "capture_outputs": False}
        pp = self.directory / "completed-plan.json"; pp.write_text(json.dumps(plan))
        (jobs / "status.json").write_text(json.dumps({"state": "finished", "exit_code": 0, "process_group_stopped": True}))
        (jobs / "request.json").write_text(json.dumps({"argv": [str(pp), str(binary)], "devices": ["A03-R40-I191-99-4100006.JD.LOCAL/0000:89:00.0"]}))
        (folder / "events.jsonl").write_text(json.dumps({"run_id": name}) + "\n")
        result = {"returncode": 0, "plan": str(pp), "binary": str(binary), "runtime": {
            "status": "success", "backend": "ge_session", "mode": "bench", "replay": plan["replay"],
            "performance": {"run_id": name, "sampling": {"seed": 42, "pool_size": 1}, "capture_enabled": False, "instances": [{}]}}}
        real_sha = v.sha
        fake_hashes = {str(binary): "8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d",
                       str(graph): "5e641362d99c9a1afb6dfce70f47b4ac07f1a7c88272fdfc791ea04ae84f4822"}
        argv = ["validate_run_identity.py", "--runs", name, "--root", str(self.directory), "--report", str(self.directory / "run-identity.json")]
        for pool in (1, 5):
            result["runtime"]["performance"]["sampling"]["pool_size"] = pool
            (folder / "result.json").write_text(json.dumps(result))
            # Only binary/graph digest reads are synthetic; actual input-byte checks are real.
            with patch.object(v, "sha", side_effect=lambda p: fake_hashes.get(str(p)) or real_sha(p)), patch.object(sys, "argv", argv):
                if pool == 1:
                    runpy.run_path(str(Path(__file__).with_name("validate_run_identity.py")), run_name="__main__")
                    self.assertTrue(json.loads((self.directory / "run-identity.json").read_text())["passed"])
                else:
                    with self.assertRaises(AssertionError): runpy.run_path(str(Path(__file__).with_name("validate_run_identity.py")), run_name="__main__")

    def test_full_five_case_pressure_stays_strict(self):
        meta = w.metadata()
        with np.load(self.ref / "cpu-golden.npz", allow_pickle=False) as z:
            gold = {cid: z[cid + "/outputs/scores"].copy() for cid in w.CASE_IDS}
        archived = json.loads((self.ref / "multi8-pressure-correctness.json").read_text())["first_result_per_case"]
        expected = {cid: {"top10_dense_id": row["expected_top10_dense_id"], "top10_score": row["expected_top10_score"]} for cid, row in archived.items()}
        for fixed in (False, True):
            blob = bytearray(); manifest = []; events = []
            for i in range(100):
                cid = "case-0002" if fixed else w.CASE_IDS[i % 5]
                manifest.append({"run_id": "pressure-cpu-test", "request_id": i, "instance": 0, "case_id": cid, "phase": "measure",
                                 "tensors": [{"name": "scores", "dtype": "float32", "shape": [1, 10112], "offset": len(blob), "bytes": 40448}]})
                events.append({"run_id": "pressure-cpu-test", "request_id": i, "instance": 0, "case_id": cid, "phase": "measure", "status": "success"})
                blob.extend(gold[cid].tobytes())
            binary = self.directory / "outputs.bin"; binary.write_bytes(blob)
            mp = self.directory / "manifest.jsonl"; mp.write_text("".join(json.dumps(e) + "\n" for e in manifest))
            ep = self.directory / "events.jsonl"; ep.write_text("".join(json.dumps(e) + "\n" for e in events))
            with patch.object(v, "reference", return_value=(meta, gold, expected)):
                if fixed:
                    with self.assertRaises(AssertionError): v.pressure_check(mp, binary, ep, 1, 100, 20)
                else:
                    self.assertTrue(v.pressure_check(mp, binary, ep, 1, 100, 20)["passed"])

    def test_pressure_cli_cannot_lower_frozen_request_or_coverage_minimum(self):
        for flag, value in (("--min-requests", "1999"), ("--min-per-case", "19")):
            report = self.directory / "rejected-pressure.json"
            argv = ["verify_outputs.py", "pressure", "--report", str(report), "--instances", "1",
                    "--manifest", "not-read", "--binary", "not-read", "--events", "not-read", flag, value]
            with patch.object(sys, "argv", argv):
                self.assertEqual(v.main(), 1)
            self.assertIn("cannot lower", json.loads(report.read_text())["exception"])


if __name__ == "__main__":
    unittest.main()
