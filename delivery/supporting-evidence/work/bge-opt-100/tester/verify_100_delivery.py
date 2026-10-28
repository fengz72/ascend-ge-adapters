#!/usr/bin/env python3
"""Offline delivery evidence audit. Does not invoke or initialize any NPU API."""
import argparse
import collections
import datetime
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import traceback


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def slope(points):
    xs, ys = zip(*points)
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - xm) ** 2 for x in xs)
    return 60 * sum((x - xm) * (y - ym) for x, y in points) / denominator if denominator else 0.0


def ten_second_windows(events, start, end):
    buckets = collections.Counter()
    for event in events:
        if event["phase"] == "measure" and start <= event["completed_ns"] <= end:
            i = min((event["completed_ns"] - start) // 10_000_000_000,
                    max(0, math.ceil((end - start) / 10_000_000_000) - 1))
            buckets[i] += 1
    duration = (end - start) / 1e9
    return [{"start_seconds": i * 10, "seconds": min(10.0, duration - i * 10),
             "successes": buckets[i], "qps": buckets[i] / min(10.0, duration - i * 10)}
            for i in range(math.ceil(duration / 10))]


def lifecycle(root, names, identity):
    assert len(names) >= 3 and len(set(names)) == len(names), "at least three distinct process cycles required"
    output = {}
    task_ids = set()
    parse = datetime.datetime.fromisoformat
    for name in names:
        status_path = root / "work/jobs" / name / "status.json"
        status = load(status_path)
        assert status["state"] == "finished" and status["exit_code"] == status["task_returncode"] == 0
        assert status["reason"] == "exited" and status["process_group_stopped"]
        assert sorted(status["devices"]) == sorted(identity["physical_devices"])
        assert status["task_id"] not in task_ids, "process cycle task_id repeated"
        task_ids.add(status["task_id"])
        before_path = root / "work" / (name + "-before-gate.json")
        after_path = root / "work" / (name + "-after-gate.json")
        before, after = load(before_path), load(after_path)
        for gate in (before, after):
            assert gate["clear"] and gate["logical_device"] == identity["logical_device"]
            assert ("100/" + gate["pci"]).lower() == identity["physical_devices"][0].lower(), "gate BDF differs from selected device"
            assert {row["chip"] for row in gate["rows"]} == {0, 1}, "both chips must be inspected"
            assert all(row["returncode"] == 0 and row["stdout"].strip() == "No process in device." for row in gate["rows"])
        # The gate is now inside the owned physical-device locks.
        assert parse(status["started_at"]) <= parse(before["at"]) <= parse(after["at"]) <= parse(status["finished_at"])
        result = load(root / "work" / name / "result.json")
        assert result["returncode"] == 0 and result["runtime"]["status"] == "success"
        output[name] = {"task_id": status["task_id"], "passed": True,
                        "status_sha256": digest(status_path), "before_gate_sha256": digest(before_path),
                        "after_gate_sha256": digest(after_path), "started_at": status["started_at"],
                        "finished_at": status["finished_at"], "device_release_checked": "both chips proc-mem clear after task"}
    return {"passed": True, "independent_process_cycles": len(task_ids), "runs": output}


def resource_analysis(rows, start, end, pci, npu_id, chip_id):
    data = []
    for row in rows:
        if not start <= row["monotonic_ns"] <= end:
            continue
        assert row["npu_smi_code"] == 0, "resource npu-smi failed"
        matches = [line for line in row["npu_smi_stdout"].splitlines() if pci.lower() in line.lower()]
        assert len(matches) == 1, "resource snapshot does not identify selected BDF exactly once"
        capacities = re.findall(r"(\d+)\s*/\s*(\d+)", matches[0])
        assert capacities and int(capacities[-1][1]) > 0, "HBM capacity not present"
        pids = {int(match.group(1)) for line in row["npu_smi_stdout"].splitlines()
                if (match := re.match(r"^\|\s*" + str(npu_id) + r"\s+" + str(chip_id) + r"\s*\|\s*(\d+)\s*\|", line))}
        assert isinstance(row.get("rss_kib"), (int, float)) and row["rss_kib"] >= 0, "missing RSS sample"
        data.append({"seconds": (row["monotonic_ns"] - start) / 1e9, "rss_kib": row["rss_kib"],
                     "hbm_mb": int(capacities[-1][0]), "host_pids": sorted(pids),
                     "runtime_container_pid": row.get("runtime_pid")})
    assert len(data) >= 12, "fewer than legacy minimum 12 in-window resource samples"
    duration = (end - start) / 1e9
    head = [row for row in data if row["seconds"] <= 120]
    tail = [row for row in data if row["seconds"] >= duration - 120]
    assert len(head) >= 3 and len(tail) >= 3, "missing first/last two-minute evidence"
    gaps = [b["seconds"] - a["seconds"] for a, b in zip(data, data[1:])]
    assert all(gap > 0 for gap in gaps), "resource clock sequence is not increasing"
    metrics = {}
    for key, minimum, fraction, unit in (("rss_kib", 128 * 1024, .05, "KiB"), ("hbm_mb", 64, .01, "MB")):
        initial = statistics.median(row[key] for row in head)
        final = statistics.median(row[key] for row in tail)
        limit = max(minimum, initial * fraction)
        steps = [b[key] - a[key] for a, b in zip(data, data[1:])]
        metrics[key] = {"unit": unit, "initial_2min_median": initial, "final_2min_median": final,
                        "increment": final - initial, "allowed_increment": limit,
                        "within_increment_limit": final - initial <= limit,
                        "linear_slope_per_minute": slope([(row["seconds"], row[key]) for row in data]),
                        "positive_steps": sum(x > 0 for x in steps), "negative_steps": sum(x < 0 for x in steps),
                        "zero_steps": sum(x == 0 for x in steps), "min": min(row[key] for row in data),
                        "max": max(row[key] for row in data)}
    return {"passed_increment_contract": all(row["within_increment_limit"] for row in metrics.values()),
            "duration_seconds": duration, "sample_count": len(data), "head_samples": len(head), "tail_samples": len(tail),
            "first_sample_seconds": data[0]["seconds"], "last_sample_seconds": data[-1]["seconds"],
            "sample_gap_median_seconds": statistics.median(gaps), "sample_gap_max_seconds": max(gaps),
            "sampling_review": "Review approximate 10-second cadence and missing stretches; legacy design has no new gap threshold.",
            "trend_review": "Required: inspect retained series for sustained growth; no invented slope cutoff.",
            "host_pid_review": "Map observed host PIDs to own runtime before claiming absence of external device load.",
            "metrics": metrics, "series": data}


def aggregate(root, names, kind, identity, v, identity_checker):
    assert len(names) == 3 and len(set(names)) == 3, "three distinct formal rounds required for each scenario"
    rounds, merged, signatures = [], [], []
    for name in names:
        folder = root / "work" / name
        bound = identity_checker.check_run(name, identity, root / "work", root / "work/jobs")
        result = load(folder / "result.json")
        plan = load(result["plan"])
        check = v.perf_check(folder / "result.json", folder / "events.jsonl", plan["instances"], 60, 10000, 10, result["plan"])
        assert check["passed"] and check["workload"]["kind"] == kind, "formal statistics/scenario failed"
        if kind == "fixed_case":
            assert check["workload"]["case_ids"] == ["case-0002"]
        events = v.read_jsonl(folder / "events.jsonl")
        merged.extend((event["completed_ns"] - event["started_ns"]) / 1e6 for event in events if event["phase"] == "measure")
        perf = result["runtime"]["performance"]
        signatures.append({"binary": bound["binary_sha256"], "graph": bound["packed_graph_sha256"],
                           "plugins": bound["plugins"], "options": plan["options"], "instances": plan["instances"]})
        check.update(run_id=name, seed=plan["seed"], identity=bound,
                     ten_second_windows=ten_second_windows(events, perf["window_start_ns"], perf["window_end_ns"]))
        rounds.append(check)
    assert all(signature == signatures[0] for signature in signatures), "configuration changed across formal rounds"
    if kind == "random_cases":
        assert {row["seed"] for row in rounds} == {2027, 2028, 2029}, "formal random runs need the frozen three independent replay seeds"
    qps = [row["qps"] for row in rounds]
    median = statistics.median(qps)
    return {"passed_formal_rounds": True, "workload_kind": kind, "rounds": rounds,
            "configuration": signatures[0], "median_qps": median, "min_qps": min(qps), "max_qps": max(qps),
            "qps_relative_range_percent": 100 * (max(qps) - min(qps)) / median,
            "merged_request_count": len(merged), "merged_p50_ms": v.quantile(merged, .5),
            "merged_p99_ms": v.quantile(merged, .99), "p99_limit": None, "SLO_acceptance": "not_configured"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checks-dir", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--fixed-runs", nargs=3, required=True)
    parser.add_argument("--random-runs", nargs=3, required=True)
    parser.add_argument("--stability-run", required=True)
    parser.add_argument("--lifecycle-runs", nargs="+", required=True)
    parser.add_argument("--pci", required=True)
    parser.add_argument("--npu-id", type=int, required=True)
    parser.add_argument("--chip-id", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(args.checks_dir or root / "tools/checks"))
    v, w, checker = (importlib.import_module(name) for name in ("verify_outputs", "workload_identity", "validate_run_identity"))
    w._REFERENCE_OVERRIDE = root / "materials"
    w._ORIGINAL_OVERRIDE = root / "materials/original-inputs/inputs.bin"
    v.GOLDEN_OVERRIDE = root / "materials/cpu-golden.npz"
    v.EXPECTED_OVERRIDE = root / "materials/expected_outputs.json"
    # Frozen large weights may be referenced by every round; hash once per unchanged stat identity.
    cache = {}
    original_sha = v.sha
    def cached_sha(path):
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if key not in cache:
            cache[key] = original_sha(path)
        return cache[key]
    v.sha = cached_sha
    try:
        identity_path = args.identity or root / "tools/identity.json"
        identity = load(identity_path)
        assert "100/" + args.pci in identity["physical_devices"]
        if "npu_id" in identity:
            assert identity["npu_id"] == args.npu_id, "resource NPU ID differs from frozen device mapping"
        fixed = aggregate(root, args.fixed_runs, "fixed_case", identity, v, checker)
        random = aggregate(root, args.random_runs, "random_cases", identity, v, checker)
        assert fixed["configuration"] == random["configuration"], "two scenarios do not share one frozen recommendation"
        cycle_names = list(dict.fromkeys(args.lifecycle_runs + [args.stability_run]))
        cycles = lifecycle(root, cycle_names, identity)
        for cycle in cycle_names:
            cycle_bound = checker.check_run(cycle, identity, root / "work", root / "work/jobs")
            cycle_result = load(root / "work" / cycle / "result.json")
            cycle_plan = load(cycle_result["plan"])
            cycle_configuration = {"binary": cycle_bound["binary_sha256"], "graph": cycle_bound["packed_graph_sha256"],
                                   "plugins": cycle_bound["plugins"], "options": cycle_plan["options"], "instances": cycle_plan["instances"]}
            assert cycle_configuration == fixed["configuration"], "lifecycle configuration differs from frozen recommendation"
            assert cycle_bound["workload"]["kind"] == "random_cases", "lifecycle cycle must contain mixed-case requests"
            measured = [event for event in v.read_jsonl(root / "work" / cycle / "events.jsonl") if event["phase"] == "measure"]
            assert {event["case_id"] for event in measured} == set(w.CASE_IDS), "lifecycle does not cover all five cases"
            assert all(event["status"] in ("success", "ok") for event in measured), "lifecycle request error"
        name = args.stability_run
        bound = checker.check_run(name, identity, root / "work", root / "work/jobs")
        folder = root / "work" / name
        result = load(folder / "result.json")
        plan = load(result["plan"])
        performance = v.perf_check(folder / "result.json", folder / "events.jsonl", plan["instances"], 600, 1, 10, result["plan"])
        assert performance["passed"] and performance["workload"]["kind"] == "random_cases"
        perf = result["runtime"]["performance"]
        current = {"binary": bound["binary_sha256"], "graph": bound["packed_graph_sha256"],
                   "plugins": bound["plugins"], "options": plan["options"], "instances": plan["instances"]}
        assert current == fixed["configuration"], "stability configuration differs from frozen formal runs"
        rows = v.read_jsonl(folder / "resources.jsonl")
        resources = resource_analysis(rows, perf["window_start_ns"], perf["window_end_ns"], args.pci, args.npu_id, args.chip_id)
        events = v.read_jsonl(folder / "events.jsonl")
        numeric_ok = resources["passed_increment_contract"]
        report = {"schema": "bge100-independent-delivery-audit/1", "passed": None if numeric_ok else False,
                  "status": "requires_trend_sampling_and_exclusivity_review" if numeric_ok else "resource_increment_contract_failed",
                  "npu_task_started": False, "fixed": fixed, "random": random, "lifecycle": cycles,
                  "stability": {"run_id": name, "identity": bound, "performance": performance, "resources": resources,
                                "ten_second_windows": ten_second_windows(events, perf["window_start_ns"], perf["window_end_ns"]),
                                "resources_sha256": digest(folder / "resources.jsonl")},
                  "numeric_contracts_passed": resources["passed_increment_contract"],
                  "identity_manifest_sha256": digest(identity_path),
                  "audit_script_sha256": digest(__file__),
                  "resource_target": {"pci": args.pci, "npu_id": args.npu_id, "chip_id": args.chip_id},
                  "limitations": ["Trend, sampling gaps, per-window throughput and host-PID ownership require evidence review.",
                                  "A final pressure correctness report for this exact configuration must be bound separately.",
                                  "No p99 SLO is configured; no latency SLA PASS is asserted."]}
    except Exception as error:
        report = {"schema": "bge100-independent-delivery-audit/1", "passed": False, "status": "evidence_contract_failed",
                  "exception": str(error), "traceback": traceback.format_exc(), "npu_task_started": False}
    finally:
        v.sha = original_sha
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report.get(key) for key in ("passed", "status", "numeric_contracts_passed", "exception")}))
    return 1 if report["passed"] is False else 0


if __name__ == "__main__":
    sys.exit(main())
