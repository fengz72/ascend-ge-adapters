"""Prepare fixed-case or five-sample BGE replay; never submits a device job."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config import ROOT, load_authoring
from core.data import load as load_cases
from core.model_api import load_adapter
from core.worker import encode, packed
from core.backends.runtime import graph_files, read_io

INPUT_DTYPES = {name: "int64" for name in (
    "input_ids", "attention_mask", "long_cid3_raw", "long_action",
    "long_time_bucket", "stat_dense_raw", "stat_count")}
INPUT_DTYPES["stat_ratio"] = "float32"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=ROOT / "models/bge/config.yaml")
    result.add_argument("--device", type=int, required=True, help="Logical device for the managed job; no device is queried here")
    result.add_argument("--output-dir", type=Path, required=True, help="New preparation root; runtime results will use its runtime/ child")
    result.add_argument("--graph", type=Path, help="Already exported ONNX with embedded IO and external weights; defaults to the config artifact")
    result.add_argument("--instances", type=int, default=None, help="Defaults to benchmark.instances in YAML")
    result.add_argument("--case-id", help="Repeat one supplied case; omitted means random replay of all five cases")
    result.add_argument("--precision-mode", default=None, help="Defaults to build.options.precision_mode in YAML")
    result.add_argument("--aicore-num", help="Positive AIC|AIV pair; defaults to build.options.aicore_num in YAML")
    result.add_argument("--min-seconds", type=float, default=60)
    result.add_argument("--max-seconds", type=float, default=300)
    result.add_argument("--min-requests", type=int, default=10000)
    result.add_argument("--max-requests", type=int, default=2000000)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--warmup-per-case", type=int, default=10)
    result.add_argument("--run-id", help="Defaults to the preparation directory name")
    result.add_argument("--capture", action="store_true", help="Capture every measured output for independent correctness checking")
    result.add_argument("--preload", type=Path, action="append", default=[], help="Append this shared library to native-only LD_PRELOAD; may repeat")
    result.add_argument("--prepare-only", action="store_true", help="Explicit preparation mode (the only mode and the default)")
    return result


def validate_args(args):
    if args.device < 0 or (args.instances is not None and args.instances < 1):
        raise ValueError("device must be nonnegative and instances must be positive")
    if (not math.isfinite(args.min_seconds) or not math.isfinite(args.max_seconds)
            or args.min_seconds <= 0 or args.max_seconds < args.min_seconds):
        raise ValueError("require finite 0 < min-seconds <= max-seconds")
    if not 1 <= args.min_requests <= args.max_requests <= 10000000:
        raise ValueError("require 1 <= min-requests <= max-requests <= 10000000")
    if args.warmup_per_case < 1 or not 0 <= args.seed <= 2**64 - 1:
        raise ValueError("warmup-per-case must be positive and seed must be uint64")
    if args.precision_mode is not None and not re.fullmatch(r"[A-Za-z0-9_]+", args.precision_mode):
        raise ValueError("precision-mode must name a concrete CANN precision mode")
    if args.aicore_num is not None:
        if not re.fullmatch(r"[1-9][0-9]*\|[1-9][0-9]*", args.aicore_num):
            raise ValueError("aicore-num must be a positive AIC|AIV pair")
    if args.run_id is not None and not args.run_id.strip():
        raise ValueError("run-id must be nonempty")
    if args.case_id is not None and not args.case_id.strip():
        raise ValueError("case-id must be nonempty")
    for library in args.preload:
        path = library.expanduser().resolve()
        if not path.is_file() or re.search(r"[\s:]", str(path)):
            raise ValueError(f"preload must be an existing file without loader separators: {path}")
    if args.output_dir.expanduser().resolve().exists():
        raise FileExistsError(f"output-dir must be new: {args.output_dir}")


def validate_bge_io(io):
    inputs = io["inputs"]
    if len(inputs) != 8 or {node["name"]: node["dtype"] for node in inputs} != INPUT_DTYPES:
        raise ValueError("BGE replay requires the original eight INT64/FP32 input types")
    outputs = io["outputs"]
    if (len(outputs) != 1 or outputs[0]["name"] != "scores"
            or outputs[0]["dtype"] != "float32" or outputs[0]["shape"] not in ([-1, 10112], [1, 10112])):
        raise ValueError("BGE replay requires FP32 scores[1,10112] output")


def prepare(args):
    """Read existing small artifacts/NPYs, then write a plan and packed inputs."""
    validate_args(args)
    # Do not expand/read source.ref: runtime encoding does not need the original weights.
    config = load_authoring(args.config, source_paths=False)
    instances = args.instances if args.instances is not None else config.get("benchmark", {}).get("instances", 1)
    precision_mode = args.precision_mode or config["build"]["options"].get("precision_mode", "must_keep_origin_dtype")
    if config["source"]["kind"] != "onnx" or config["build"]["backend"] != "ge_session":
        raise ValueError("BGE replay supports only the selected ONNX -> GE Session route")
    if config["build"]["options"].get("graph_run_mode", 0) != 0:
        raise ValueError("BGE replay requires GE prediction mode=0")
    graph = ({"path": str(args.graph.expanduser().resolve()), "format": "onnx", "om_path": None}
             if args.graph else graph_files(config))
    graph_path = Path(graph["path"])
    if graph_path.suffix.lower() != ".onnx" or not graph_path.is_file():
        raise ValueError("graph must be an existing exported ONNX artifact")
    # The original 1.35 GB protobuf must not be loaded for a runtime-only replay.
    if graph_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("use the exported external-weight ONNX artifact, rather than an inline-weight source model")
    io = read_io(graph)
    validate_bge_io(io)
    cases = load_cases(config)
    if (len(cases) != 5 or len({case["id"] for case in cases}) != 5
            or any(case["samples"] != 1 or set(case["payload"]) != set(INPUT_DTYPES) for case in cases)):
        raise ValueError("BGE replay requires all five supplied batch=1 cases with eight features")
    if args.case_id is not None:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            raise ValueError(f"unknown BGE case-id: {args.case_id}")
    handle = load_adapter(config["model"]["adapter_file"]).prepare_runtime(config, io)
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    _, records = encode(handle, cases, io, directory)
    options = copy.deepcopy(config["build"]["options"])
    options.update(precision_mode=precision_mode, graph_run_mode=0)
    if args.aicore_num is not None:
        options["aicore_num"] = args.aicore_num
    plan = {"mode": "bench", "backend": "ge_session", "graph": graph, "io": io,
            "inputs": packed(directory, records), "device": args.device,
            "instances": instances, "warmup": 0, "iterations": 1,
            "seed": args.seed, "options": options,
            "workload": {"kind": "fixed_case" if args.case_id is not None else "random_cases",
                         "case_ids": [case["id"] for case in cases],
                         "sampling": "fixed" if args.case_id is not None else "uniform_with_replacement"},
            "observability": {"profiling": False, "dump": False},
            "output_dir": str(directory / "runtime"),
            "replay": {"run_id": args.run_id or directory.name,
                       "min_seconds": args.min_seconds, "max_seconds": args.max_seconds,
                       "min_requests": args.min_requests, "max_requests": args.max_requests,
                       "warmup_per_case": args.warmup_per_case, "capture_outputs": args.capture}}
    if args.preload:
        inherited = os.environ.get("LD_PRELOAD", "")
        added = ":".join(str(path.expanduser().resolve()) for path in args.preload)
        plan["runtime_env"] = {"LD_PRELOAD": ":".join(part for part in (inherited, added) if part)}
    plan_path = directory / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan_path


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        path = prepare(args)
    except (ValueError, OSError) as error:
        cli.error(str(error))
    print(json.dumps({"status": "prepared", "plan": str(path), "device_job_submitted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
