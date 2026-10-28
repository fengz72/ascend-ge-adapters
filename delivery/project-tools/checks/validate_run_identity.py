#!/usr/bin/env python3
"""Bind a completed BGE run to a frozen host, executable, graph and plugin manifest."""
import argparse
import json
from pathlib import Path
import re
import sys
import traceback

import verify_outputs as v
import workload_identity as w


def artifact(entry, actual_path=None):
    path = Path(entry["path"]).resolve()
    if actual_path is not None:
        assert Path(actual_path).resolve() == path, "artifact path differs from frozen identity"
    digest = v.sha(path)
    assert digest == entry["sha256"], (str(path), "artifact digest differs")
    return digest


def check_run(name, identity, run_root, job_root):
    assert Path(name).name == name and name not in (".", ".."), "run name must be a directory basename"
    folder = run_root / name
    status_path = job_root / name / "status.json"
    status = json.loads(status_path.read_text())
    assert status["state"] == "finished" and status["exit_code"] == 0 and status["process_group_stopped"], "outer device job failed"
    if "task_returncode" in status:
        assert status["task_returncode"] == 0, "native task group failed"
    result = json.loads((folder / "result.json").read_text())
    assert result["returncode"] == 0
    runtime = result["runtime"]
    assert runtime["status"] == "success" and runtime["backend"] == "ge_session" and runtime["mode"] == "bench"
    plan_path = Path(result["plan"])
    plan = json.loads(plan_path.read_text())
    job_path = job_root / name / "request.json"
    job = json.loads(job_path.read_text())
    assert plan["device"] == identity["logical_device"] and plan["options"]["graph_run_mode"] == 0
    assert plan["replay"]["run_id"] == name == runtime["performance"]["run_id"]
    assert plan["replay"] == runtime["replay"]
    assert Path(plan["output_dir"]).resolve() == folder.resolve(), "result directory differs from actual plan"
    input_identity = w.check_plan_inputs(plan)
    perf = runtime["performance"]
    assert perf["sampling"]["seed"] == plan["seed"] and perf["sampling"]["pool_size"] == len(input_identity["workload"]["case_ids"])
    assert perf["capture_enabled"] == plan["replay"]["capture_outputs"]
    assert len(perf["instances"]) == plan["instances"]
    binary_sha = artifact(identity["binary"], result["binary"])
    graph_sha = artifact(identity["graph"], plan["graph"]["path"])
    if "weights" in identity:
        artifact(identity["weights"])
    assert str(plan_path) in job["argv"] and result["binary"] in job["argv"], "job does not invoke recorded binary/plan"
    assert sorted(job["devices"]) == sorted(identity["physical_devices"]), "physical device lock identity differs"
    assert len(job["devices"]) == len(set(job["devices"])), "duplicate physical device identity"
    if "devices" in status:
        assert sorted(status["devices"]) == sorted(identity["physical_devices"])
    events = v.read_jsonl(folder / "events.jsonl")
    assert events and {e["run_id"] for e in events} == {name}
    if plan["replay"]["capture_outputs"]:
        rows = v.read_jsonl(folder / "manifest.jsonl")
        assert rows and {row["run_id"] for row in rows} == {name}
    # The candidate belongs only to the native child. Python compiler workers
    # retain the existing allocator, but must not inherit the fusion plugin.
    allowed = identity.get("plugins", {})
    preserved = identity.get("required_preloads", [])
    tokens = [t for t in re.split(r"[:\s]+", plan.get("runtime_env", {}).get("LD_PRELOAD", "")) if t]
    plugins = {}
    for token in tokens:
        assert token in allowed or token in preserved, (token, "unfrozen preload library")
        if token in allowed:
            plugins[token] = v.sha(token)
            assert plugins[token] == allowed[token], "plugin digest differs"
            assert not any(token in arg for arg in job["argv"]), "plugin must not preload Python launcher"
    if plugins:
        assert set(preserved) <= set(tokens), "fusion preload lost required inherited allocator"
    for arg in job["argv"]:
        if arg.startswith("LD_PRELOAD="):
            assert not any(token in allowed for token in re.split(r"[:\s]+", arg.split("=", 1)[1])), "outer preload must not load candidate plugin"
    return {"passed": True, "result_sha256": v.sha(folder / "result.json"), "plan_sha256": v.sha(plan_path),
            "job_status_sha256": v.sha(status_path), "job_request_sha256": v.sha(job_path),
            "binary_sha256": binary_sha, "packed_graph_sha256": graph_sha,
            "input_binary_sha256": input_identity["input_binary_sha256"], "workload": input_identity["workload"],
            "precision_mode": plan["options"]["precision_mode"], "plugins": plugins, "seed": plan["seed"],
            "physical_devices": identity["physical_devices"], "capture_enabled": plan["replay"]["capture_outputs"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--identity-manifest", type=Path, default=w.PROJECT_ROOT / "tools/identity.json")
    p.add_argument("--run-root", type=Path)
    p.add_argument("--job-root", type=Path)
    p.add_argument("--report", type=Path, required=True)
    w.add_reference_arguments(p)
    args = p.parse_args()
    w.configure(args)
    try:
        identity = json.loads(args.identity_manifest.read_text())
        assert identity["physical_devices"] and len(identity["physical_devices"]) == len(set(identity["physical_devices"]))
        root = Path(identity["project_root"])
        checks = {name: check_run(name, identity, args.run_root or root / "work", args.job_root or root / "work/jobs") for name in args.runs}
        report = {"schema": "bge-run-artifact-identity/2", "passed": True,
                  "identity_manifest_sha256": v.sha(args.identity_manifest), "runs": checks}
    except Exception as error:
        report = {"schema": "bge-run-artifact-identity/2", "passed": False,
                  "exception": str(error), "traceback": traceback.format_exc()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: val for k, val in report.items() if k != "traceback"}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
