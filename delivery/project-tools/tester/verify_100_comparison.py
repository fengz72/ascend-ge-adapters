#!/usr/bin/env python3
"""Offline BGE final baseline/candidate comparison and budget/record coverage."""
import argparse
import datetime
import importlib
import json
import math
from pathlib import Path
import re
import sys
import traceback

import verify_100_delivery as audit

EXPECTED_OPTS = tuple(f"OPT-{index:03d}" for index in range(1, 12))


def selected_formal_names(best_fixed_r3="final-best-fixed-r3"):
    assert Path(best_fixed_r3).name == best_fixed_r3 and best_fixed_r3 not in (".", ".."), "run name must be a directory basename"
    fixed_best = ["final-best-fixed-r1", "final-best-fixed-r2", best_fixed_r3]
    random_best = [f"final-best-random-r{i}" for i in (1, 2, 3)]
    baseline = [f"final-base-{scenario}-r{i}" for scenario in ("fixed", "random") for i in (1, 2, 3)]
    return fixed_best, random_best, baseline + fixed_best + random_best


def compare_rounds(baseline, candidate):
    assert baseline["workload_kind"] == candidate["workload_kind"], "cross-scenario comparison prohibited"
    change = 100 * (candidate["median_qps"] / baseline["median_qps"] - 1)
    spread = max(baseline["qps_relative_range_percent"], candidate["qps_relative_range_percent"])
    exceeds = abs(change) + 1e-10 >= 3 and abs(change) > spread + 1e-10
    direction = "improvement" if change > 0 else "regression" if change < 0 else "unchanged"
    return {"baseline_median_qps": baseline["median_qps"], "candidate_median_qps": candidate["median_qps"],
            "qps_change_percent": change, "baseline_round_range_percent": baseline["qps_relative_range_percent"],
            "candidate_round_range_percent": candidate["qps_relative_range_percent"],
            "comparison_threshold_percent": max(3, spread), "exceeds_frozen_change_rule": exceeds,
            "interpretation": direction if exceeds else "no_significant_change_under_frozen_rule",
            "baseline_merged_p99_ms": baseline["merged_p99_ms"], "candidate_merged_p99_ms": candidate["merged_p99_ms"],
            "p99_change_percent": 100 * (candidate["merged_p99_ms"] / baseline["merged_p99_ms"] - 1),
            "p99_limit": None, "SLO_acceptance": "not_configured"}


def budget_audit(root, historical_path):
    historical = audit.load(historical_path)
    assert historical["config_count"] == len(historical["unique_configs"]) == 9
    assert math.isclose(historical["charged_seconds"], 1616.67991, abs_tol=1e-6)
    assert historical["max_unique_configs"] == 12 and historical["max_seconds"] == 7200
    v1 = [entry for entry in historical["unique_configs"]
          if any("libbge_add_layernorm_pass.so" in token for token in entry["signature"].get("ld_preload", []))]
    assert len(v1) == 1, "historical invalid V1 must retain its real configuration slot"
    registry_path = root / "tools/config-registry.json"
    registry = audit.load(registry_path)
    assert len(registry["new"]) == 3, "frozen 100 search used exactly three new configurations"
    assert len(registry["known"]) == 8, "reusable registry excludes the historical invalid V1"
    actual_configs = historical["config_count"] + len(registry["new"])
    assert actual_configs == 12, "do not equate reusable-key count with historical actual configurations"
    signatures = [json.dumps(signature, sort_keys=True) for signature in registry["new"].values()]
    assert len(set(signatures)) == len(signatures), "new registry aliases duplicate actual configurations"
    jobs = []
    for folder in sorted((root / "work/jobs").iterdir()):
        if not folder.is_dir():
            continue
        status_path = folder / "status.json"
        assert status_path.is_file(), (folder.name, "unresolved task without status")
        status = audit.load(status_path)
        assert status["state"] == "finished" and status["process_group_stopped"], (folder.name, "unresolved task")
        seconds = (datetime.datetime.fromisoformat(status["finished_at"]) - datetime.datetime.fromisoformat(status["started_at"])).total_seconds()
        assert seconds >= 0, "negative device charge"
        jobs.append({"name": folder.name, "seconds": seconds, "exit_code": status["exit_code"], "status_sha256": audit.digest(status_path)})
    charged = historical["charged_seconds"] + sum(row["seconds"] for row in jobs)
    assert charged <= 7200, "device time budget exceeded"
    ledger_path = root / "work/budget.json"
    ledger = audit.load(ledger_path)
    assert math.isclose(ledger["historical_seconds"], historical["charged_seconds"], abs_tol=1e-6)
    assert ledger["max_seconds"] == 7200
    assert math.isclose(ledger["charged_seconds"], charged, abs_tol=1e-5), "budget ledger does not account for all finished tasks"
    assert math.isclose(ledger["remaining_seconds"], 7200 - charged, abs_tol=1e-5)
    return {"passed": True, "historical_config_count": 9, "reusable_known_key_count": 8,
            "historical_nonreusable_configuration": v1, "new_config_count": 3,
            "actual_unique_config_count": actual_configs, "max_unique_configs": 12,
            "new_config_keys": sorted(registry["new"]), "historical_seconds": historical["charged_seconds"],
            "charged_seconds": charged, "remaining_seconds": 7200 - charged, "max_seconds": 7200, "jobs": jobs,
            "failed_jobs_charged": [row for row in jobs if row["exit_code"] != 0],
            "historical_budget_sha256": audit.digest(historical_path), "registry_sha256": audit.digest(registry_path),
            "budget_ledger_sha256": audit.digest(ledger_path)}


def record_coverage(docs_root, delivery_doc=None):
    files = {path.stem: path for path in (docs_root / "experiments").glob("OPT-*.md")}
    assert set(files) == set(EXPECTED_OPTS), "missing/unexpected OPT records"
    records = {}
    for name, path in sorted(files.items()):
        text = path.read_text()
        headings = re.findall(r"^##\s+(\d+)\.\s+(.+)$", text, flags=re.M)
        assert [number for number, _ in headings] == [str(i) for i in range(1, 8)], (name, "fixed seven-section structure differs")
        records[name] = {"sha256": audit.digest(path), "sections": [title for _, title in headings]}
    summary = Path(delivery_doc) if delivery_doc is not None else docs_root / "acceptance.md"
    rows = {}
    for line in summary.read_text().splitlines():
        match = re.match(r"^\s*\|\s*(?:\[(OPT-\d{3})\]\([^)]*\)|(OPT-\d{3}))\s*\|", line)
        if not match:
            continue
        name = match.group(1) or match.group(2)
        assert name not in rows, (name, "duplicate final summary row")
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
        assert len(cells) == 7 and all(cells), (name, "summary needs seven populated columns")
        rows[name] = {"cells": cells, "source_line": line}
    assert set(rows) == set(EXPECTED_OPTS), "final summary must retain all 11 OPT rows, including failed and rejected attempts"
    return {"passed_structure": True, "expected_opt_count": 11, "records": records, "summary_rows": rows,
            "summary_path": str(summary.resolve()), "summary_sha256": audit.digest(summary),
            "semantic_review": "Check numeric provenance and decisions manually; structural coverage is not evidence that every summary value is correct."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--docs-root", type=Path, required=True)
    parser.add_argument("--delivery-doc", type=Path, help="Final delivery report; defaults to docs-root/acceptance.md")
    parser.add_argument("--historical-budget", type=Path, required=True)
    parser.add_argument("--checks-dir", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--pressure-run", default="half8-pressure")
    parser.add_argument("--stability-run", default="final-best-stability")
    parser.add_argument("--best-fixed-r3", default="final-best-fixed-r3", help="Valid third candidate fixed round, including an explicitly named retry")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(args.checks_dir or root / "tools/checks"))
    v, checker = (importlib.import_module(name) for name in ("verify_outputs", "validate_run_identity"))
    original_sha = v.sha
    cache = {}
    def cached_sha(path):
        path = Path(path); stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if key not in cache:
            cache[key] = original_sha(path)
        return cache[key]
    v.sha = cached_sha
    report = {"schema": "bge100-independent-final-comparison/1", "passed": None, "npu_task_started": False}
    try:
        identity_path = args.identity or root / "tools/identity.json"
        identity = audit.load(identity_path)
        fixed_best, random_best, formal_names = selected_formal_names(args.best_fixed_r3)
        report["selected_formal_runs"] = formal_names
        candidate_path = args.report.with_name(args.report.stem + "-candidate-details.json")
        argv = [str(Path(audit.__file__)), "--project-root", str(root), "--identity", str(identity_path),
                "--fixed-runs", *fixed_best, "--random-runs", *random_best, "--stability-run", args.stability_run,
                "--lifecycle-runs", random_best[0], random_best[1], args.stability_run,
                "--pci", "0000:99:00.0", "--npu-id", "1", "--chip-id", "0", "--report", str(candidate_path)]
        if args.checks_dir:
            argv += ["--checks-dir", str(args.checks_dir)]
        original_argv = sys.argv
        try:
            sys.argv = argv
            code = audit.main()
        finally:
            sys.argv = original_argv
        candidate = audit.load(candidate_path)
        report["candidate_audit"] = candidate
        assert code == 0 and candidate["passed"] is not False, "candidate numerical/lifecycle/stability contract failed"
        expected = candidate["fixed"]["configuration"]
        assert expected["instances"] == 8 and expected["options"].get("aicore_num") == "12|24"
        assert expected["options"]["graph_run_mode"] == 0 and expected["options"]["precision_mode"] == "must_keep_origin_dtype"
        assert expected["plugins"] == {}, "frozen recommendation uses original graph without custom fusion"
        baseline = {kind: audit.aggregate(root, [f"final-base-{label}-r{i}" for i in (1, 2, 3)], kind, identity, v, checker)
                    for kind, label in (("fixed_case", "fixed"), ("random_cases", "random"))}
        assert baseline["fixed_case"]["configuration"] == baseline["random_cases"]["configuration"]
        base_signature = baseline["fixed_case"]["configuration"]
        assert base_signature["instances"] == 1 and not base_signature["options"].get("aicore_num")
        assert base_signature["options"]["precision_mode"] == "must_keep_origin_dtype" and base_signature["plugins"] == {}
        assert base_signature["graph"] == expected["graph"] and base_signature["binary"] == expected["binary"]
        report["baseline"] = baseline
        report["comparison"] = {"fixed_case": compare_rounds(baseline["fixed_case"], candidate["fixed"]),
                                "random_cases": compare_rounds(baseline["random_cases"], candidate["random"])}
        report["all_formal_job_gates"] = audit.lifecycle(root, formal_names, identity)
        name = args.pressure_run
        bound = checker.check_run(name, identity, root / "work", root / "work/jobs")
        folder = root / "work" / name
        result = audit.load(folder / "result.json"); plan = audit.load(result["plan"])
        signature = {"binary": bound["binary_sha256"], "graph": bound["packed_graph_sha256"],
                     "plugins": bound["plugins"], "options": plan["options"], "instances": plan["instances"]}
        assert signature == expected and bound["capture_enabled"]
        pressure = v.pressure_check(folder / "manifest.jsonl", folder / "outputs.bin", folder / "events.jsonl", 8, 2000, 20)
        assert pressure["passed"]
        events = [event for event in v.read_jsonl(folder / "events.jsonl") if event["phase"] == "measure"]
        assert len(events) == pressure["requests"] == result["runtime"]["performance"]["cohort_successes"]
        assert {event["request_id"] for event in events} == set(range(len(events)))
        report["final_pressure"] = pressure
        report["pressure_job_gates"] = audit.lifecycle(root, [name, random_best[0], random_best[1]], identity)
        report["budget"] = budget_audit(root, args.historical_budget)
        report["record_coverage"] = record_coverage(args.docs_root, args.delivery_doc)
        report["status"] = "requires_resource_trend_exclusivity_and_summary_semantic_review"
        report["audit_script_sha256"] = audit.digest(__file__)
    except Exception as error:
        report.update(passed=False, status="evidence_contract_failed", exception=str(error), traceback=traceback.format_exc())
    finally:
        v.sha = original_sha
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report.get(key) for key in ("passed", "status", "comparison", "exception")}))
    return 1 if report["passed"] is False else 0


if __name__ == "__main__":
    sys.exit(main())
