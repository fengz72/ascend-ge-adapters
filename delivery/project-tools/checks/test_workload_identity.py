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
import validate_run_identity as identity


class WorkloadIdentity(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.ref = w.reference_dir()
        self.original = json.loads((self.ref.parent / "original-plan.json").read_text())
        self.raw = (self.ref.parent / "original-inputs/inputs.bin").read_bytes()

    def plan(self, fixed=None, legacy=False):
        plan = copy.deepcopy(self.original)
        plan.update(instances=1, seed=42, device=14)
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
        performance = {"latency_scope": "h2d_desc_execute_sync_d2h", "arrival_model": "closed_loop",
                       "capture_enabled": False, "profiling_enabled": False,
                       "window_start_ns": 100000000, "window_end_ns": 200000000,
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

    def test_expected_and_golden_overrides_keep_digest_guards(self):
        bad_expected = self.directory / "expected.json"; bad_expected.write_text("[]")
        with patch.object(v, "EXPECTED_OVERRIDE", bad_expected):
            with self.assertRaisesRegex(AssertionError, "expected hash differs"): v.reference()
        bad_golden = self.directory / "golden.npz"; bad_golden.write_bytes(b"changed reference")
        with patch.object(v, "GOLDEN_OVERRIDE", bad_golden):
            with self.assertRaisesRegex(AssertionError, "golden hash differs"): v.reference()

    def test_manifest_override_keeps_frozen_manifest_digest(self):
        (self.directory / "cpu-golden-manifest.json").write_text("{}")
        with self.assertRaisesRegex(AssertionError, "manifest differs"): w.metadata(self.directory)

    def test_runtime_semantic_labels_must_be_present_and_match(self):
        fields = {"latency_scope": "execute_only", "arrival_model": "open_loop", "capture_enabled": True, "profiling_enabled": True}
        for key, wrong in fields.items():
            for missing in (False, True):
                pp, ep, fp = self.timing(self.plan("case-0002"))
                perf = json.loads(fp.read_text())
                if missing: perf.pop(key)
                else: perf[key] = wrong
                fp.write_text(json.dumps(perf))
                with self.assertRaises(AssertionError): v.perf_check(fp, ep, 1, 0, 100, 10, pp)

    def test_capture_or_profiling_cannot_establish_performance(self):
        for kind in ("capture", "profiling", "dump"):
            plan = self.plan("case-0002")
            if kind == "capture": plan["replay"] = {"capture_outputs": True}
            else: plan["observability"][kind] = True
            pp, ep, fp = self.timing(plan)
            with self.assertRaisesRegex(AssertionError, "cannot establish performance"):
                v.perf_check(fp, ep, 1, 0, 100, 10, pp)


class MigratedRunIdentity(unittest.TestCase):
    setUp = WorkloadIdentity.setUp
    plan = WorkloadIdentity.plan
    # Reuse the real original input bytes; executable/graph bytes are CPU fixtures.
    def completed(self):
        plan = self.plan("case-0002")
        name = "cpu-100-test"
        folder = self.directory / "work" / name; folder.mkdir(parents=True)
        jobs = self.directory / "work/jobs" / name; jobs.mkdir(parents=True)
        binary = self.directory / "fake-binary"; binary.write_bytes(b"CPU executable fixture")
        graph = self.directory / "fake-graph"; graph.write_bytes(b"CPU graph fixture")
        plugin = self.directory / "fusion.so"; plugin.write_bytes(b"CPU plugin fixture")
        plan["graph"]["path"] = str(graph)
        plan["output_dir"] = str(folder)
        plan["replay"] = {"run_id": name, "capture_outputs": False}
        pp = self.directory / "plan.json"; pp.write_text(json.dumps(plan))
        devices = ["100/0000:81:00.0", "100/0000:83:00.0"]
        job = {"argv": [str(binary), str(pp)], "devices": devices}
        (jobs / "status.json").write_text(json.dumps({"state": "finished", "exit_code": 0, "process_group_stopped": True, "task_returncode": 0}))
        (jobs / "request.json").write_text(json.dumps(job))
        (folder / "events.jsonl").write_text(json.dumps({"run_id": name}) + "\n")
        result = {"returncode": 0, "plan": str(pp), "binary": str(binary), "runtime": {
            "status": "success", "backend": "ge_session", "mode": "bench", "replay": plan["replay"],
            "performance": {"run_id": name, "sampling": {"seed": 42, "pool_size": 1}, "capture_enabled": False, "instances": [{}]}}}
        (folder / "result.json").write_text(json.dumps(result))
        manifest = {"project_root": str(self.directory), "logical_device": 14, "physical_devices": devices,
                    "binary": {"path": str(binary), "sha256": v.sha(binary)},
                    "graph": {"path": str(graph), "sha256": v.sha(graph)},
                    "plugins": {str(plugin): v.sha(plugin)}, "required_preloads": ["/allocator.so"]}
        return name, plan, pp, result, manifest, folder, jobs, plugin

    def checked(self, name, manifest):
        return identity.check_run(name, manifest, self.directory / "work", self.directory / "work/jobs")

    def test_new_host_and_new_binary_digest_are_accepted(self):
        name, _, _, _, manifest, _, _, _ = self.completed()
        self.assertTrue(self.checked(name, manifest)["passed"])

    def test_changed_physical_device_or_rebuilt_binary_are_rejected(self):
        name, _, _, _, manifest, _, _, _ = self.completed()
        changed = copy.deepcopy(manifest); changed["physical_devices"].pop()
        with self.assertRaises(AssertionError): self.checked(name, changed)
        Path(manifest["binary"]["path"]).write_bytes(b"tampered executable")
        with self.assertRaises(AssertionError): self.checked(name, manifest)

    def test_actual_fixed_runtime_pool_must_be_one(self):
        name, _, _, result, manifest, folder, _, _ = self.completed()
        result["runtime"]["performance"]["sampling"]["pool_size"] = 5
        (folder / "result.json").write_text(json.dumps(result))
        with self.assertRaises(AssertionError): self.checked(name, manifest)

    def test_fusion_preload_must_preserve_allocator_and_stay_native_only(self):
        name, plan, pp, _, manifest, _, jobs, plugin = self.completed()
        plan["runtime_env"] = {"LD_PRELOAD": "/allocator.so:" + str(plugin)}
        pp.write_text(json.dumps(plan)); self.assertTrue(self.checked(name, manifest)["passed"])
        plan["runtime_env"]["LD_PRELOAD"] = str(plugin); pp.write_text(json.dumps(plan))
        with self.assertRaises(AssertionError): self.checked(name, manifest)
        plan["runtime_env"]["LD_PRELOAD"] = "/allocator.so:" + str(plugin); pp.write_text(json.dumps(plan))
        job = json.loads((jobs / "request.json").read_text()); job["argv"].append("LD_PRELOAD=" + str(plugin))
        (jobs / "request.json").write_text(json.dumps(job))
        with self.assertRaises(AssertionError): self.checked(name, manifest)

    def test_unfrozen_plugin_or_tampered_graph_is_rejected(self):
        name, plan, pp, _, manifest, _, _, _ = self.completed()
        plan["runtime_env"] = {"LD_PRELOAD": "/unfrozen-fusion.so"}; pp.write_text(json.dumps(plan))
        with self.assertRaises(AssertionError): self.checked(name, manifest)
        plan.pop("runtime_env"); pp.write_text(json.dumps(plan))
        Path(manifest["graph"]["path"]).write_bytes(b"tampered graph")
        with self.assertRaises(AssertionError): self.checked(name, manifest)

    def test_perf_cli_cannot_lower_formal_or_search_minima(self):
        for extra in ([], ["--search"]):
            report = self.directory / "bad-perf.json"
            argv = ["verify_outputs.py", "perf", "--report", str(report), "--instances", "1",
                    "--performance", "not-read", "--events", "not-read", "--min-seconds", "1", "--min-requests", "100"] + extra
            with patch.object(sys, "argv", argv): self.assertEqual(v.main(), 1)
            self.assertIn("cannot lower", json.loads(report.read_text())["exception"])

    def test_identity_cli_records_the_actual_manifest(self):
        name, _, _, _, manifest, _, _, _ = self.completed()
        mp = self.directory / "identity.json"; mp.write_text(json.dumps(manifest))
        report = self.directory / "run-identity.json"
        cmd = [sys.executable, str(Path(__file__).with_name("validate_run_identity.py")), "--runs", name,
               "--identity-manifest", str(mp), "--report", str(report)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(json.loads(report.read_text())["identity_manifest_sha256"], v.sha(mp))


if __name__ == "__main__":
    unittest.main()
