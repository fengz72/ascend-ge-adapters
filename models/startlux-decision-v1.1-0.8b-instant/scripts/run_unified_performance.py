#!/usr/bin/env python3
"""P8 performance worker: the FORMAL unified 5-IO AIR on a warm GE pool.

Runs INSIDE the container (repo-readable, numpy/official processor env) as the
job launched by models/startlux-decision-v1.1-0.8b-instant/scripts/
monitor_idle_benchmark.py on host 100.  Phases (default ``gate,bench``):

  gate   the B4 long [1,65,129,2048] fresh-GE-session functional compare that
         was never accepted on a shared device: the pre-encoded 5-IO fixture
         work/startlux-validation/p7-boundary/fixtures/cases.npz (b4-long-
         1-65-129-2048::*) against the INDEPENDENT original CPU goldens in
         work/startlux-validation/boundary-oracles/boundary-fixtures.npz
         (scores_L1 / scores_L65 / scores_L129 / scores_L2048).  One fresh
         ge_runtime infer subprocess, per-row finite / cosine>=0.999 /
         relative L2<=0.02.  Evidence goes to a NEW p8 directory; the old
         phase-B OOM history is never overwritten.  Bench runs only when
         every row passes; a failure stops honestly with status kept.
  bench  the FORMAL 11-case pool (7 text read-outs + single red + single blue
         + red+blue + the B2 mixed case = 11 cases / 12 rows) encoded by the
         production unified route (core.worker.encode over the unified
         adapter's official CPU preprocessing; no weights are loaded, the
         vision tower runs INSIDE the graph), then ONE ge_runtime bench run:
         instances=1, warmup=32 (all distinct shapes first), iterations=2048,
         seed=42, closed loop.  requests.csv is the raw runtime artifact; the
         worker adds result.json / summary.json / report.md (no new CSV
         summaries).

Measurement boundaries (runtime/bench.cpp semantics, never paraphrased as
business QPS): e2e = h2d + desc + execute + sync, NO D2H; the official CPU
preprocessing is NOT inside the timed window; graph QPS = graph executions/s,
readout rows/s = B-weighted rows; warmup/compile/load are separate columns;
HBM is a two-point stage sample, not a peak.

The worker polls the shared abort file in the host job dir (bind-mounted at
the same path); when the monitor writes it the worker kills ONLY its own
ge_runtime child (process group) and exits with status ``contaminated`` /
``aborted`` -- it never kills or even signals any other PID.  Status
transitions land in worker-status.json (chmod 0644) in the job dir; the
numeric evidence stays in the repo run directory.

Terminal protocol: ``bench-done`` is written ONLY when the bench record
passed (no runtime errors, no first-hit pollution, and every
publish-integrity gate consistent: runtime summary vs raw requests.csv vs
the planned std::mt19937_64 draw counts, finite metrics, wall_ms > 0);
a failed record writes ``bench-failed`` and the process exits 1 -- the
monitor additionally requires its docker-exec client exit code to be 0
before accepting any ``bench-done`` claim.  Raw artifacts (requests.csv,
runtime log, plan) always stay on disk for investigation.

Exit codes: 0 done / 1 failed or gate-failed / 2 error / 3 aborted.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
ROOT = MODEL_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from core.backends.runtime import execute as ge_execute  # noqa: E402

MODEL_YAML = MODEL_DIR / "config" / "model.yaml"
MODEL_FOLDER = ROOT / "work" / "startlux-materials" / \
    "model-6496ae641cc71b149ce9dad69bd97f2a44e7bf11"
AIR = ROOT / "work" / "startlux-decision" / "build" / "model.air"
AIR_BYTES = 1907534423
AIR_SHA256 = "c1f8816ee25e03119b2522e55adb49ebf42a88692791b179920f5f60221edba1"
RUNTIME_BIN = ROOT / "runtime" / "build" / "ge_runtime"
RUNTIME_SHA256 = "692996b5df8a7ed8fdebdc365cdaa9383b6b538868ecf3e019303930d178eecc"
PROTO_LIB = "libchunk_gated_delta_rule_compat_proto.so"
OFFLINE_TARGET = "Ascend910_9382"
BOUNDRY_FIXTURES = ROOT / "work" / "startlux-validation" / "p7-boundary" / "fixtures"
ORACLE_NPZ = ROOT / "work" / "startlux-validation" / "boundary-oracles" / \
    "boundary-fixtures.npz"
B4_LONG = "b4-long-1-65-129-2048"
B4_GOLDENS = ["scores_L1", "scores_L65", "scores_L129", "scores_L2048"]
INPUT_ORDER = ["input_ids", "position_ids", "actual_seq_lengths", "pixel_values",
               "image_feature_index"]
OUTPUT_NAME = "scores"
LETTER_COLUMNS = 26
GATES = {"cosine_min": 0.999, "relative_l2_max": 0.02}
MIN_WIDTH = 256
MAX_BATCH = 4
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 248320
FORMAL11 = ["text-choice_short-r0", "text-score_short-r0", "text-noul_short-r0",
            "text-allow_none-r0", "text-wide_27-r0", "text-wide_27-r1",
            "text-wide_27-r2", "image-red-r0", "image-blue-r0",
            "image-two-a-b-r0", "mixed-text-image-b2-r0"]
OPTIONS = {"graph_run_mode": 0, "precision_mode": "must_keep_origin_dtype"}
DEFAULTS = {"instances": 1, "warmup": 32, "iterations": 2048, "seed": 42,
            "gate_timeout": 900.0, "timeout": 5400.0, "poll_interval": 2.0}


class Aborted(Exception):
    """The monitor wrote the abort marker; only our own child was killed."""


class WorkerError(RuntimeError):
    """A fail-closed worker error."""


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_id():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S") \
        + "-" + uuid.uuid4().hex[:8]


def sha256_of(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(Path(path))


def load_module(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


ge_readout = load_module("p8_ge_readout", MODEL_DIR / "ge_readout.py")


# ------------------------------------------------------------ abort polling

class AbortWatch:
    """Polls the shared abort file written by the host monitor (if any)."""

    def __init__(self, job_dir, poll_interval=DEFAULTS["poll_interval"]):
        self.path = Path(job_dir) / "abort" if job_dir else None
        self.interval = float(poll_interval)
        self.reason = None

    def triggered(self):
        if self.path is None or not self.path.is_file():
            return False
        if self.reason is None:
            try:
                self.reason = self.path.read_text(errors="replace")[:500]
            except OSError:
                self.reason = "(unreadable abort marker)"
        return True


def ge_call_polling(runtime_bin, plan, log_path, timeout, watch):
    """One ge_runtime subprocess with abort polling.

    Delegates the raw plan protocol (own process group, timeout/abort kill of
    exactly that group, single-GE_RESULT parsing) to the shared core
    implementation core.backends.runtime.execute; only the P8 abort reason
    message stays local.  Returns (result, log_text, wall_seconds).
    """
    try:
        result, text, wall = ge_execute(runtime_bin, plan, log_path=log_path,
                                        timeout=timeout, should_abort=watch.triggered,
                                        poll_interval=watch.interval, error=WorkerError,
                                        aborted_error=Aborted)
    except Aborted as aborted:
        raise Aborted("abort marker present: %s" % watch.reason) from aborted
    return result, text, wall


# ---------------------------------------------------------------- job status

class StatusWriter:
    """worker-status.json (0644) in the host job dir for the host monitor."""

    def __init__(self, job_dir, job_id):
        self.path = Path(job_dir) / "worker-status.json" if job_dir else None
        self.job_id = job_id
        self.started = time.perf_counter()

    def update(self, phase, status, **fields):
        if self.path is None:
            return
        record = {"schema": "startlux-p8-worker-status/1",
                  "job_id": self.job_id, "phase": phase, "status": status,
                  "updated_at": now_iso(),
                  "elapsed_seconds": round(time.perf_counter() - self.started, 1)}
        record.update(fields)
        path = Path(self.path)
        path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        os.chmod(path, 0o644)
        print(f"[worker {phase}] {status}"
              + (f": {fields}" if fields else ""), flush=True)


# ------------------------------------------------------------ pool previews

class Mt19937_64:
    """std::mt19937_64 replica (identical stream) -- the verified helper."""

    NN, MM = 312, 156
    MATRIX_A = 0xB5026F5AA96619E9
    UM, LM = 0xFFFFFFFF80000000, 0x7FFFFFFF

    def __init__(self, seed):
        self.mt = [0] * self.NN
        self.mt[0] = seed & 0xFFFFFFFFFFFFFFFF
        for i in range(1, self.NN):
            self.mt[i] = (6364136223846793005 *
                          (self.mt[i - 1] ^ (self.mt[i - 1] >> 62)) + i) \
                & 0xFFFFFFFFFFFFFFFF
        self.mti = self.NN

    def next(self):
        if self.mti >= self.NN:
            for i in range(self.NN - self.MM):
                x = (self.mt[i] & self.UM) | (self.mt[i + 1] & self.LM)
                self.mt[i] = self.mt[i + self.MM] ^ (x >> 1) ^ ((x & 1) * self.MATRIX_A)
            for i in range(self.NN - self.MM, self.NN - 1):
                x = (self.mt[i] & self.UM) | (self.mt[i + 1] & self.LM)
                self.mt[i] = self.mt[i + (self.MM - self.NN)] ^ (x >> 1) \
                    ^ ((x & 1) * self.MATRIX_A)
            x = (self.mt[self.NN - 1] & self.UM) | (self.mt[0] & self.LM)
            self.mt[self.NN - 1] = self.mt[self.MM - 1] ^ (x >> 1) \
                ^ ((x & 1) * self.MATRIX_A)
            self.mti = 0
        x = self.mt[self.mti]
        self.mti += 1
        x ^= (x >> 29) & 0x5555555555555555
        x ^= (x << 17) & 0x71D67FFFEDA60000
        x ^= (x << 37) & 0xFFF7EEE000000000
        x ^= x >> 43
        return x & 0xFFFFFFFFFFFFFFFF


def sampler_next(rng, pool_size):
    if pool_size <= 1:
        return 0
    return rng.next() % pool_size


def nearest_rank(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


def shape_to_string(shape):
    return "[" + ", ".join(str(int(axis)) for axis in shape) + "]"


def shape_key_of(record):
    return ";".join(shape_to_string(t["shape"]) for t in record["tensors"]) + ";"


def warmup_preview(stats, instances, warmup, seed):
    reps, seen = [], set()
    for index, case in enumerate(stats):
        if case["shape_key"] not in seen:
            seen.add(case["shape_key"])
            reps.append(index)
    sequences = [list(reps) for _ in range(instances)]
    planned = len(reps) * instances
    rng = Mt19937_64(seed)
    fill = 0
    while planned < max(0, warmup) * instances:
        sequences[planned % instances].append(sampler_next(rng, len(stats)))
        planned += 1
        fill += 1
    return {"shape_representative_indices": reps,
            "distinct_shapes": len(reps),
            "planned_runs": planned, "random_fill_runs": fill,
            "per_instance_sequences": sequences,
            "note": "every distinct (B,S) shape executes inside warmup; a "
                    "first_hit inside the measurement window marks the run "
                    "polluted/failed"}


def measurement_preview(stats, instances, requests, seed):
    rng = Mt19937_64(seed)
    sequence = [sampler_next(rng, len(stats)) for _ in range(requests)]
    draws = {}
    for index in sequence:
        draws[stats[index]["id"]] = draws.get(stats[index]["id"], 0) + 1
    return {"requests": requests, "case_draw_counts": draws,
            "note": "exact draws (std::mt19937_64 replica), closed loop with "
                    "replacement; instances=%d" % instances}


# ----------------------------------------------------------------- preflight

def preflight():
    preload = [entry for entry in os.environ.get("LD_PRELOAD", "").split(":")
               if entry]
    canonical = str(ROOT / "work/startlux-decision/dependencies/"
                    "chunk_gated_delta_rule_proto/install/lib" / PROTO_LIB)
    foreign = [entry for entry in preload
               if "chunk_gated_delta_rule" in os.path.basename(entry)
               and entry != canonical]
    if foreign:
        raise WorkerError("non-canonical ChunkGatedDeltaRule preload(s) were "
                          f"already set BEFORE python started (the launcher "
                          f"must reject them in Bash; old+canonical double "
                          f"load can SIGILL in the loader): {foreign}; "
                          f"canonical: {canonical}")
    if canonical not in preload:
        raise WorkerError(f"{PROTO_LIB} missing from LD_PRELOAD; source "
                          "models/startlux-decision-v1.1-0.8b-instant/scripts/"
                          "env_chunk_proto.sh work/startlux-decision first (the "
                          "canonical work/startlux-decision proto only)")
    if not AIR.is_file():
        raise WorkerError(f"AIR missing: {AIR}")
    if AIR.stat().st_size != AIR_BYTES:
        raise WorkerError(f"AIR bytes {AIR.stat().st_size} != pinned {AIR_BYTES}")
    sha = sha256_of(AIR)
    if sha != AIR_SHA256:
        raise WorkerError(f"AIR sha256 {sha} != pinned {AIR_SHA256}")
    rsha = sha256_of(RUNTIME_BIN)
    if rsha != RUNTIME_SHA256:
        raise WorkerError(f"ge_runtime sha256 {rsha} != pinned {RUNTIME_SHA256}")
    os.environ.setdefault("STARTLUX_MODEL_DIR", str(MODEL_FOLDER))
    return {"air_bytes": AIR_BYTES, "air_sha256": sha, "runtime_sha256": rsha,
            "ld_preload": os.environ.get("LD_PRELOAD", ""),
            "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES")}


def read_air_io(directory, timeout):
    log_path = Path(directory) / "read-io.log"
    plan = {"mode": "read_io",
            "graph": {"format": "air", "path": str(AIR), "om_path": None}}
    class _NoAbort:
        path = None
        interval = 5.0
        reason = None
        def triggered(self):
            return False
    result, _text, _wall = ge_call_polling(RUNTIME_BIN, plan, log_path, timeout,
                                           _NoAbort())
    io = result.get("io")
    if not io:
        raise WorkerError("read_io returned no io")
    return io


def validate_unified_io(io):
    unified_readout = load_module("p8_unified_readout", MODEL_DIR / "unified_readout.py")
    spec = unified_readout.UnifiedGraphSpec.from_io(io, vocab_size=VOCAB_SIZE,
                                                    max_seq_len=MAX_SEQ_LEN)
    return spec


# ------------------------------------------------------------- pool encoding

def load_formal_cases():
    from core.config import load_authoring
    from core import data
    config = load_authoring(MODEL_YAML, target=OFFLINE_TARGET, device=None,
                             source_paths=False)
    cases = data.load(config)
    ids = [case["id"] for case in cases]
    if ids != FORMAL11:
        raise WorkerError(f"the generated unified pool is not the formal 11: {ids}")
    return config, cases


def build_bench_plan(io, records, directory, args):
    inputs = {"file": str(Path(directory) / "inputs.bin"),
              "cases": [{"id": r["id"], "samples": r["samples"],
                         "tensors": r["tensors"],
                         "output_capacities": r["output_capacities"]}
                        for r in records]}
    plan = {"mode": "bench", "backend": "ge_session",
            "graph": {"format": "air", "path": str(AIR), "om_path": None},
            "io": io, "inputs": inputs, "device": int(args.device),
            "instances": int(args.instances), "warmup": int(args.warmup),
            "iterations": int(args.iterations), "seed": int(args.seed),
            "options": dict(OPTIONS),
            "observability": {"profiling": False, "dump": False},
            "output_dir": str(directory)}
    return plan, inputs


def summarize_requests(csv_path, stats_by_id, wall_ms):
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ok = [row for row in rows if row["error"] == "0"]
    accounting = {"requests_total": len(rows), "requests_ok": len(ok),
                  "readout_rows": sum(int(row["samples"]) for row in ok),
                  "true_tokens": sum(stats_by_id[row["input_set"]]["true_tokens"]
                                     for row in ok),
                  "padded_tokens": sum(stats_by_id[row["input_set"]]
                                        ["padded_tokens"] for row in ok),
                  "first_hit_requests": sum(1 for row in ok
                                             if row["first_hit"] == "1")}
    if wall_ms > 0 and ok:
        seconds = wall_ms / 1000.0
        accounting["readout_rows_per_second"] = accounting["readout_rows"] / seconds
        accounting["graph_executions_per_second"] = len(ok) / seconds
        accounting["true_tokens_per_second"] = accounting["true_tokens"] / seconds
        accounting["padded_tokens_per_second"] = accounting["padded_tokens"] / seconds
        accounting["token_efficiency"] = (accounting["true_tokens"] /
                                          accounting["padded_tokens"]) \
            if accounting["padded_tokens"] else None

    def group(key):
        grouped = {}
        for row in ok:
            grouped.setdefault(row[key], []).append(float(row["e2e_ms"]))
        table = {}
        for name, values in sorted(grouped.items()):
            table[name] = {"n": len(values),
                           "e2e_p50_ms": nearest_rank(values, 0.50),
                           "e2e_p95_ms": nearest_rank(values, 0.95),
                           "e2e_p99_ms": nearest_rank(values, 0.99),
                           "e2e_avg_ms": sum(values) / len(values)}
        return table

    e2e = [float(row["e2e_ms"]) for row in ok]
    exec_ms = [float(row["exec_ms"]) for row in ok]
    h2d = [float(row["h2d_ms"]) for row in ok]
    desc = [float(row["desc_ms"]) for row in ok]
    accounting["percentiles"] = {
        "e2e_ms": {"n": len(e2e), "p50": nearest_rank(e2e, 0.50),
                   "p95": nearest_rank(e2e, 0.95), "p99": nearest_rank(e2e, 0.99)},
        "exec_ms": {"n": len(exec_ms), "p50": nearest_rank(exec_ms, 0.50),
                    "p95": nearest_rank(exec_ms, 0.95),
                    "p99": nearest_rank(exec_ms, 0.99)},
        "by_case": group("input_set"), "by_shape": group("shape_key")}
    accounting["component_ms"] = {
        "h2d_avg_ms": sum(h2d) / len(h2d) if h2d else None,
        "desc_avg_ms": sum(desc) / len(desc) if desc else None}
    return accounting


# --------------------------------------------------------------- gate phase

def gate_arrays():
    import numpy as np
    meta = json.loads((BOUNDRY_FIXTURES / "cases-meta.json").read_text())
    if meta.get("air", {}).get("sha256") != AIR_SHA256 \
            or meta.get("air", {}).get("bytes") != AIR_BYTES:
        raise WorkerError("p7-boundary cases-meta.json does not bind the formal "
                          "AIR (sha/bytes mismatch)")
    arrays = np.load(BOUNDRY_FIXTURES / "cases.npz")
    goldens = np.load(ORACLE_NPZ)
    needed = [f"{B4_LONG}::{name}" for name in INPUT_ORDER]
    missing = [key for key in needed + B4_GOLDENS
               if key not in set(arrays.files) | set(goldens.files)]
    if missing:
        raise WorkerError(f"gate fixtures miss keys {missing}")
    out = {name: np.ascontiguousarray(arrays[key]) for name, key
           in zip(INPUT_ORDER, needed)}
    return meta, out, {key: np.asarray(goldens[key], dtype=np.float32).reshape(-1)
                       for key in B4_GOLDENS}


def row_gates(scores, golden):
    import numpy as np
    a = np.asarray(golden, dtype=np.float64)
    b = np.asarray(scores, dtype=np.float64)
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel_l2 = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    finite = bool(np.isfinite(b).all())
    return {"cosine": cosine, "relative_l2": rel_l2,
            "max_abs_error": float(np.abs(a - b).max()), "finite": finite,
            "passed": bool(finite and cosine >= GATES["cosine_min"]
                           and rel_l2 <= GATES["relative_l2_max"])}


def run_gate(io, directory, args, watch, status):
    import numpy as np
    status.update("gate", "gate-running")
    gate_dir = Path(directory) / f"gate-{run_id()}"
    gate_dir.mkdir(parents=True, exist_ok=True)
    meta, arrays, goldens = gate_arrays()
    lengths = [int(v) for v in arrays["actual_seq_lengths"]]
    if lengths != [1, 65, 129, 2048]:
        raise WorkerError(f"{B4_LONG} fixture lengths {lengths} != [1, 65, 129, 2048]")
    if int(arrays["input_ids"].shape[0]) != 4:
        raise WorkerError("B4 gate fixture is not batch 4")
    pool_path = gate_dir / "inputs.bin"
    tensors, offset = [], 0
    with pool_path.open("wb") as stream:
        for name in INPUT_ORDER:
            data = arrays[name].tobytes(order="C")
            tensors.append({"name": name, "dtype": None, "shape":
                            list(arrays[name].shape), "offset": offset,
                            "bytes": len(data)})
            tensors[-1]["dtype"] = {"input_ids": "int64",
                                    "position_ids": "int64",
                                    "actual_seq_lengths": "int32",
                                    "pixel_values": "float32",
                                    "image_feature_index": "int64"}[name]
            offset += stream.write(data)
    plan_cases = [{"id": B4_LONG, "samples": 4, "tensors": tensors,
                   "output_capacities": {OUTPUT_NAME: 4 * LETTER_COLUMNS * 4}}]
    plan = {"mode": "infer", "backend": "ge_session",
            "graph": {"format": "air", "path": str(AIR), "om_path": None},
            "io": io,
            "inputs": {"file": str(pool_path), "cases": plan_cases},
            "device": int(args.device), "instances": 1, "warmup": 0,
            "iterations": 1, "seed": int(args.seed), "options": dict(OPTIONS),
            "observability": {"profiling": False, "dump": False},
            "output_dir": str(gate_dir)}
    (gate_dir / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    report = {"schema": "startlux-p8-b4-gate/1", "generated_at": now_iso(),
              "case": B4_LONG, "rows": lengths,
              "goldens": B4_GOLDENS, "gates": GATES,
              "air": {"path": rel(AIR), "bytes": AIR_BYTES, "sha256": AIR_SHA256},
              "runtime_sha256": sha256_of(RUNTIME_BIN),
              "fixture_meta_air": meta["air"],
              "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
              "fresh_ge_session": True,
              "note": "fresh GE subprocess, one CompileGraph+LoadGraph+run; "
                      "gate passes only when EVERY row is finite "
                      "cos>=0.999 relL2<=0.02 vs the independent original "
                      "CPU goldens; failure stops before any bench"}
    try:
        result, text, wall = ge_call_polling(RUNTIME_BIN, plan,
                                            gate_dir / "infer-runtime.log",
                                            args.gate_timeout, watch)
    except Aborted:
        raise
    except (WorkerError, Exception) as error:
        message = str(error)
        oom = ("Memory_Allocation_Failure" in message
               or "out of memory" in message.lower()
               or "memory allocation" in message.lower())
        report.update({"status": "gate-failed", "oom": oom,
                       "error": message[-4000:]})
        (gate_dir / "gate-report.json").write_text(json.dumps(report, indent=1) + "\n")
        status.update("gate", "gate-failed", error=message[-300:], oom=oom,
                      gate_report=rel(gate_dir / "gate-report.json"))
        return False, report, gate_dir
    report["wall_seconds"] = round(wall, 3)
    report["timings_ms"] = ge_readout.parse_timings(text)
    report["ge_status"] = result.get("status")
    if result.get("status") != "success":
        report.update({"status": "gate-failed",
                       "error": "ge_runtime infer status %r" % result.get("status")})
        (gate_dir / "gate-report.json").write_text(json.dumps(report, indent=1) + "\n")
        status.update("gate", "gate-failed", error=str(report["error"]),
                      gate_report=rel(gate_dir / "gate-report.json"))
        return False, report, gate_dir
    rows = (result.get("outputs") or {}).get(B4_LONG)
    if not rows or len(rows) != 1 or rows[0].get("instance") != 0:
        report.update({"status": "gate-failed", "error": "no captured output row"})
        (gate_dir / "gate-report.json").write_text(json.dumps(report, indent=1) + "\n")
        status.update("gate", "gate-failed", error="no captured output row",
                      gate_report=rel(gate_dir / "gate-report.json"))
        return False, report, gate_dir
    tensor = next((t for t in rows[0].get("tensors", [])
                   if t.get("name") == OUTPUT_NAME), None)
    if tensor is None or tensor.get("shape") != [4, LETTER_COLUMNS] \
            or tensor.get("bytes") != 4 * LETTER_COLUMNS * 4:
        report.update({"status": "gate-failed",
                       "error": f"bad output tensor {tensor}"})
        (gate_dir / "gate-report.json").write_text(json.dumps(report, indent=1) + "\n")
        status.update("gate", "gate-failed", error=f"bad output tensor {tensor}",
                      gate_report=rel(gate_dir / "gate-report.json"))
        return False, report, gate_dir
    blob = (gate_dir / "outputs.bin").read_bytes()
    scores = np.frombuffer(blob, dtype=np.float32,
                           count=4 * LETTER_COLUMNS,
                           offset=tensor["offset"]).reshape(4, LETTER_COLUMNS)
    comparison = {}
    for index, key in enumerate(B4_GOLDENS):
        comparison[f"row{index}"] = row_gates(scores[index], goldens[key])
    report["comparison"] = comparison
    passed = all(row["passed"] for row in comparison.values())
    report["status"] = "gate-passed" if passed else "gate-failed"
    report["summary"] = {
        "rows": len(comparison),
        "rows_passed": sum(1 for row in comparison.values() if row["passed"]),
        "min_cosine": min(row["cosine"] for row in comparison.values()),
        "max_relative_l2": max(row["relative_l2"] for row in comparison.values())}
    report["artifacts"] = {"gate_dir": rel(gate_dir),
                           "inputs_bin": rel(pool_path),
                           "outputs_bin": rel(gate_dir / "outputs.bin"),
                           "plan": rel(gate_dir / "plan.json"),
                           "log": rel(gate_dir / "infer-runtime.log")}
    (gate_dir / "gate-report.json").write_text(json.dumps(report, indent=1) + "\n")
    if passed:
        status.update("gate", "gate-passed",
                      gate_report=rel(gate_dir / "gate-report.json"),
                      min_cosine=report["summary"]["min_cosine"],
                      max_relative_l2=report["summary"]["max_relative_l2"])
    else:
        status.update("gate", "gate-failed",
                      gate_report=rel(gate_dir / "gate-report.json"),
                      rows=report["summary"])
    return passed, report, gate_dir


# -------------------------------------------------------------- bench phase

def bench_report_md(record):
    perf = record["performance"]
    acc = record["token_and_percentile_accounting"]
    inst = (perf.get("instances") or [{}])[0]
    lines = [
        "# P8 unified warm-pool benchmark (%s)" % record["label"],
        "",
        "- status: **%s** (contaminated: %s)" % (record["status"],
                                                  record.get("contaminated")),
        "- scope: e2e = h2d + desc + execute + sync; **no D2H**; the official",
        "  CPU preprocessing is outside the timed window; this is NOT a full",
        "  business QPS number.",
        "- graph QPS = graph executions/s; readout rows/s = B-weighted rows;",
        "  true/padded token rates are pool-mix weighted.",
        "",
        "## Compile / load / warmup (outside the steady-state window)",
        "",
        "| metric | value |",
        "|---|---|",
        "| compile_ms | %s |" % (record["native_timings"].get("compile_ms"),),
        "| load_ms | %s |" % (record["native_timings"].get("load_ms"),),
        "| warmup runs | %s |" % (perf.get("warmup", {}).get("runs"),),
        "| warmup ms | %s |" % (perf.get("warmup", {}).get("ms"),),
        "| distinct shapes | %s |" % (perf.get("distinct_shapes"),),
        "| specialize_ms (in warmup) | %s |" % (inst.get("specialize_ms"),),
        "| HBM base->after_setup MB | %s -> %s |" % (
            perf.get("hbm_mb", {}).get("base"), perf.get("hbm_mb", {}).get("after_setup")),
        "",
        "## Steady state",
        "",
        "| wall_ms | requests | errors | graph exec/s | rows/s | true tok/s | "
        "padded tok/s | token eff |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| %.1f | %d | %d | %.3f | %.3f | %.1f | %.1f | %.4f |" % (
            perf.get("wall_ms", 0.0), perf.get("requests", 0), perf.get("errors", 0),
            perf.get("qps", 0.0), acc.get("readout_rows_per_second", 0.0),
            acc.get("true_tokens_per_second", 0.0),
            acc.get("padded_tokens_per_second", 0.0),
            acc.get("token_efficiency") or 0.0),
        "",
        "## Latency percentiles (nearest-rank, n mandatory)",
        "",
        "| scope | n | P50 ms | P95 ms | P99 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope in ("e2e_ms", "exec_ms"):
        block = acc["percentiles"][scope]
        lines.append("| %s | %d | %.3f | %.3f | %.3f |" % (
            scope, block["n"], block["p50"], block["p95"], block["p99"]))
    lines += [
        "",
        "## Per-case latency (e2e)",
        "",
        "| case | n | P50 ms | P95 ms | P99 ms | avg ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case, block in acc["percentiles"]["by_case"].items():
        lines.append("| %s | %d | %.3f | %.3f | %.3f | %.3f |" % (
            case, block["n"], block["e2e_p50_ms"], block["e2e_p95_ms"],
            block["e2e_p99_ms"], block["e2e_avg_ms"]))
    lines += [
        "",
        "## Not measured (never fabricated)",
        "",
        "- D2H latency (bench mode leaves scores on device).",
        "- Full business QPS / original questions/s (needs a real request mix).",
        "- Continuous HBM peak (two point-in-time stage samples only).",
        "- Multi-instance scaling (explicit separate task; default is 1).",
        "",
    ]
    return "\n".join(lines)


def encode_pool(io, directory):
    """The production encode of the formal 11-case pool -> (records, stats).

    No weights are loaded; the vision tower runs inside the graph; the
    official CPU preprocessing runs here (outside any timed window).
    """
    config, cases = load_formal_cases()
    from core.model_api import load_adapter
    from core.worker import encode as worker_encode
    handle = load_adapter(config["model"]["adapter_file"]).prepare_runtime(config, io)
    _example, records = worker_encode(handle, cases, io, Path(directory))
    stats = []
    for case, record in zip(cases, records):
        arrays = handle.encode(case)
        lengths = [int(v) for v in arrays["actual_seq_lengths"]]
        width = int(arrays["input_ids"].shape[1])
        samples = int(case["samples"])
        stats.append({"id": case["id"], "samples": samples, "lengths": lengths,
                      "padded_width": width, "true_tokens": sum(lengths),
                      "padded_tokens": samples * width,
                      "input_bytes": sum(t["bytes"] for t in record["tensors"]),
                      "output_capacity_bytes":
                          record["output_capacities"][OUTPUT_NAME],
                      "shapes": [t["shape"] for t in record["tensors"]],
                      "shape_key": shape_key_of(record)})
    return records, stats


def pool_summary(stats):
    return {"cases": len(stats),
            "readout_rows": sum(case["samples"] for case in stats),
            "distinct_shapes": len({case["shape_key"] for case in stats}),
            "true_tokens": sum(case["true_tokens"] for case in stats),
            "padded_tokens": sum(case["padded_tokens"] for case in stats),
            "input_bytes": sum(case["input_bytes"] for case in stats),
            "output_capacity_bytes": sum(case["output_capacity_bytes"]
                                         for case in stats),
            "case_ids": [case["id"] for case in stats]}


def validate_bench_integrity(performance, accounting, previews, plan, stats):
    """Publish-integrity gates over the REAL runtime artifacts.

    Cross-checks the runtime's own summary (requests/errors), the raw
    requests.csv accounting and the planned mt19937_64 draw preview; every
    timing/throughput metric must be finite with wall_ms > 0.  Any violation
    means the run MUST NOT be published (status failed), raw artifacts stay
    on disk for investigation.  iterations semantics follow runtime/bench.cpp:
    plan['iterations'] is the TOTAL request count split across instances
    (never multiplied by instances here).
    """
    import math
    violations = []
    stats_by_id = {case["id"]: case for case in stats}
    draws = previews["measurement_preview"]["case_draw_counts"]
    expected_requests = int(plan["iterations"])
    expected_rows = sum(count * stats_by_id[case_id]["samples"]
                        for case_id, count in draws.items())
    if accounting["requests_total"] != expected_requests:
        violations.append("requests.csv has %d rows != the planned %d "
                          "measurement requests (truncated/extra rows)"
                          % (accounting["requests_total"], expected_requests))
    runtime_requests = performance.get("requests")
    if not isinstance(runtime_requests, int) \
            or runtime_requests != accounting["requests_ok"]:
        violations.append("runtime reports requests %r != %d ok CSV rows"
                          % (runtime_requests, accounting["requests_ok"]))
    runtime_errors = performance.get("errors")
    csv_errors = accounting["requests_total"] - accounting["requests_ok"]
    if not isinstance(runtime_errors, int) or runtime_errors != csv_errors:
        violations.append("runtime reports errors %r != %d error CSV rows"
                          % (runtime_errors, csv_errors))
    if accounting["readout_rows"] != expected_rows:
        violations.append("accounting readout rows %d != %d rows expected "
                          "from the measurement preview draw counts"
                          % (accounting["readout_rows"], expected_rows))
    by_case = {name: block["n"] for name, block
               in accounting["percentiles"]["by_case"].items()}
    if by_case != draws:
        violations.append("per-case request counts %r != the planned draw "
                          "counts %r" % (by_case, draws))
    wall_ms = performance.get("wall_ms")
    if not isinstance(wall_ms, (int, float)) or not math.isfinite(wall_ms) \
            or wall_ms <= 0:
        violations.append("wall_ms %r is not a positive finite number"
                          % (wall_ms,))

    def finite(value, path):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            violations.append("%s is not numeric: %r" % (path, value))
        elif not math.isfinite(value):
            violations.append("%s is not finite: %r" % (path, value))

    for key in ("qps", "samples_per_second"):
        finite(performance.get(key), "performance.%s" % key)
    for scope in ("e2e_ms", "exec_ms"):
        block = performance.get(scope) or {}
        for stat, value in block.items():
            finite(value, "performance.%s.%s" % (scope, stat))
    samples_block = performance.get("samples") or {}
    for stat, value in samples_block.items():
        finite(value, "performance.samples.%s" % stat)
    warmup_block = performance.get("warmup") or {}
    finite(warmup_block.get("ms"), "performance.warmup.ms")
    hbm_block = performance.get("hbm_mb") or {}
    for stat in ("base", "after_setup"):
        finite(hbm_block.get(stat), "performance.hbm_mb.%s" % stat)
    for index, instance in enumerate(performance.get("instances") or []):
        finite(instance.get("specialize_ms"),
               "performance.instances[%d].specialize_ms" % index)
        finite(instance.get("load_ms"),
               "performance.instances[%d].load_ms" % index)
    for scope in ("e2e_ms", "exec_ms"):
        block = accounting["percentiles"][scope]
        for stat in ("p50", "p95", "p99"):
            if block["n"] and block[stat] is None:
                violations.append("accounting.%s.%s is None with n=%d"
                                  % (scope, stat, block["n"]))
            else:
                finite(block[stat], "accounting.%s.%s" % (scope, stat))
    for key in ("readout_rows_per_second", "graph_executions_per_second",
                "true_tokens_per_second", "padded_tokens_per_second"):
        if key in accounting:
            finite(accounting[key], "accounting.%s" % key)
    if accounting["requests_ok"] and not accounting["percentiles"]["e2e_ms"]["n"]:
        violations.append("no ok requests reached the percentile accounting")
    return violations


def run_bench(io, directory, args, watch, status):
    status.update("bench", "bench-encoding")
    bench_dir = Path(directory) / f"bench-{run_id()}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    records, stats = encode_pool(io, bench_dir)
    pool = pool_summary(stats)
    warmup = args.warmup if args.warmup is not None \
        else len({case["shape_key"] for case in stats}) + 8
    plan, inputs = build_bench_plan(io, records, bench_dir, args)
    plan["warmup"] = warmup
    previews = {"warmup_preview": warmup_preview(stats, args.instances, warmup,
                                                 args.seed),
                "measurement_preview": measurement_preview(stats, args.instances,
                                                           args.iterations,
                                                           args.seed)}
    status.update("bench", "bench-running", pool=pool, distinct_shapes=pool["distinct_shapes"])
    result, text, wall = ge_call_polling(RUNTIME_BIN, plan,
                                         bench_dir / "bench-runtime.log",
                                         args.timeout, watch)
    if result.get("status") != "success":
        raise WorkerError("ge_runtime bench failed: %s"
                          % json.dumps(result)[:3000])
    performance = result["performance"]
    stats_by_id = {case["id"]: case for case in stats}
    accounting = summarize_requests(bench_dir / "requests.csv", stats_by_id,
                                    performance["wall_ms"])
    violations = validate_bench_integrity(performance, accounting, previews,
                                          plan, stats)
    first_hits = accounting["first_hit_requests"]
    errors = performance.get("errors", 0)
    passed = bool(errors == 0 and not first_hits and not violations)
    record = {
        "schema": "startlux-p8-bench-run/1", "label": args.label,
        "status": "passed" if passed else "failed",
        "generated_at": now_iso(),
        "air": {"path": rel(AIR), "bytes": AIR_BYTES, "sha256": AIR_SHA256},
        "runtime": {"path": rel(RUNTIME_BIN), "sha256": sha256_of(RUNTIME_BIN)},
        "device": {"logical": int(args.device),
                   "ascend_rt_visible_devices":
                       os.environ.get("ASCEND_RT_VISIBLE_DEVICES")},
        "ld_preload": os.environ.get("LD_PRELOAD", ""),
        "benchmark": {"instances": args.instances, "warmup": warmup,
                      "iterations": args.iterations, "seed": args.seed,
                      "profiling": False, "dump": False,
                      "sampling": "std::mt19937_64(seed) %% pool_size, closed "
                                  "loop, replacement",
                      "latency_scope": "h2d_desc_execute_sync (bench mode "
                                       "performs no D2H; official CPU "
                                       "preprocessing outside the window)"},
        "io": io, "pool": pool, "pool_case_stats": stats,
        "previews": previews, "plan": plan,
        "performance": performance,
        "native_timings": ge_readout.parse_timings(text),
        "token_and_percentile_accounting": accounting,
        "artifacts": {"run_dir": rel(bench_dir),
                      "requests_csv": rel(bench_dir / "requests.csv"),
                      "runtime_log": rel(bench_dir / "bench-runtime.log"),
                      "inputs_bin": rel(bench_dir / "inputs.bin"),
                      "plan_json": rel(bench_dir / "plan.json")}}
    record["cross_checks"] = {
        "profiling_enabled": performance.get("profiling_enabled"),
        "runtime_requests": performance.get("requests"),
        "accounting_requests_ok": accounting["requests_ok"],
        "csv_rows": accounting["requests_total"],
        "planned_iterations": int(plan["iterations"]),
        "rows_match": (accounting["readout_rows"]
                       == sum(count * stats_by_id[case_id]["samples"]
                              for case_id, count
                              in previews["measurement_preview"]
                              ["case_draw_counts"].items())),
        "draw_counts_match": (
            {name: block["n"] for name, block
             in accounting["percentiles"]["by_case"].items()}
            == previews["measurement_preview"]["case_draw_counts"]),
        "first_hit_requests": first_hits,
        "runtime_errors": errors,
        "integrity_violations": violations,
        "note": "graph QPS is not business QPS; rows/s is B-weighted; token "
                "rates are pool-mix weighted, not a production distribution; "
                "'samples' in the runtime performance JSON is a per-request "
                "stat block, the row totals come from requests.csv"}
    (bench_dir / "result.json").write_text(json.dumps(record, indent=1) + "\n")
    summary = {"schema": "startlux-p8-bench-summary/1", "label": args.label,
               "status": record["status"],
               "benchmark": record["benchmark"], "pool": pool,
               "performance": performance,
               "accounting": accounting,
               "native_timings": record["native_timings"],
               "integrity_violations": violations,
               "artifacts": record["artifacts"]}
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (bench_dir / "report.md").write_text(bench_report_md(record), encoding="utf-8")
    status.update("bench",
                  "bench-done" if passed else "bench-failed",
                  benchmark_status=record["status"],
                  result_json=rel(bench_dir / "result.json"),
                  summary_json=rel(bench_dir / "summary.json"),
                  report_md=rel(bench_dir / "report.md"),
                  integrity_violations=violations,
                  qps=performance.get("qps") if passed else None,
                  rows_per_second=(accounting.get("readout_rows_per_second")
                                   if passed else None))
    return record, bench_dir


# --------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default="gate,bench",
                        help="comma list of gate,bench (default gate,bench)")
    parser.add_argument("--device", type=int, default=0,
                        help="logical device index under ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--run-root",
                        default="work/startlux-validation/p8-idle-performance/runs")
    parser.add_argument("--job-id", default=None,
                        help="unique job id the host monitor's docker-top "
                             "attribution matches on")
    parser.add_argument("--job-dir", default=None,
                        help="host monitor job dir (bind-mounted at the same "
                             "path); worker-status.json + abort live there")
    parser.add_argument("--label", default="p8")
    parser.add_argument("--instances", type=int, default=DEFAULTS["instances"])
    parser.add_argument("--warmup", type=int, default=DEFAULTS["warmup"])
    parser.add_argument("--iterations", type=int, default=DEFAULTS["iterations"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--timeout", type=float, default=DEFAULTS["timeout"],
                        help="bench ge_runtime wall timeout (s)")
    parser.add_argument("--gate-timeout", type=float,
                        default=DEFAULTS["gate_timeout"])
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU-only: preflight + io + pool encode + plan + "
                             "previews; NEVER launches a GE bench subprocess")
    args = parser.parse_args(argv)
    phases = [phase.strip() for phase in args.phase.split(",") if phase.strip()]
    if not phases or any(phase not in ("gate", "bench") for phase in phases):
        parser.error("--phase accepts only gate and bench entries")
    if args.instances < 1 or args.iterations < 1 or args.warmup < 0 \
            or not 0 <= args.seed <= 2 ** 64 - 1:
        parser.error("invalid instances/warmup/iterations/seed")
    if args.job_dir and not args.job_id:
        parser.error("--job-dir requires --job-id (the monitor's attribution "
                     "marker)")
    run_root = (ROOT / args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    directory = run_root / run_id()
    directory.mkdir(parents=True)
    status = StatusWriter(args.job_dir, args.job_id)
    watch = AbortWatch(args.job_dir, DEFAULTS["poll_interval"])
    exit_code = 2
    try:
        status.update("preflight", "starting", phases=phases,
                      run_dir=rel(directory), device=args.device,
                      dry_run=bool(args.dry_run))
        pins = preflight()
        status.update("preflight", "preflight-ok", **pins)
        io = read_air_io(directory, args.gate_timeout)
        spec = validate_unified_io(io)
        if args.dry_run:
            records, stats = encode_pool(io, directory)
            warmup = args.warmup if args.warmup is not None \
                else len({case["shape_key"] for case in stats}) + 8
            plan, _inputs = build_bench_plan(io, records, directory, args)
            plan["warmup"] = warmup
            dry = {"schema": "startlux-p8-worker-dryrun/1",
                   "generated_at": now_iso(), "phases": phases,
                   "pins": pins,
                   "spec": {"max_seq_len": spec.max_seq_len,
                            "vocab_size": spec.vocab_size,
                            "batch": spec.batch},
                   "pool": pool_summary(stats),
                   "warmup_preview": warmup_preview(stats, args.instances, warmup,
                                                    args.seed),
                   "measurement_preview": measurement_preview(stats,
                                                              args.instances,
                                                              args.iterations,
                                                              args.seed),
                   "plan": plan,
                   "note": "dry run: no GE bench subprocess was launched, no "
                           "NPU context was created, no weights were loaded"}
            (directory / "dryrun.json").write_text(json.dumps(dry, indent=1) + "\n")
            status.update("dryrun", "dryrun-done", dryrun=rel(directory / "dryrun.json"))
            print(json.dumps({"status": "dryrun-done",
                              "pool": dry["pool"],
                              "warmup_preview": {k: dry["warmup_preview"][k]
                                                 for k in ("distinct_shapes",
                                                           "planned_runs",
                                                           "random_fill_runs")}},
                             ensure_ascii=False))
            return 0
        if "gate" in phases:
            passed, _report, gate_dir = run_gate(io, directory, args, watch, status)
            if not passed:
                print(json.dumps({"status": "gate-failed",
                                  "gate_report": rel(gate_dir / "gate-report.json")}))
                return 1
        if "bench" in phases:
            record, bench_dir = run_bench(io, directory, args, watch, status)
            print(json.dumps({"status": record["status"],
                              "result_json": rel(bench_dir / "result.json"),
                              "qps": record["performance"].get("qps")},
                             ensure_ascii=False))
            return 0 if record["status"] == "passed" else 1
        return 0
    except Aborted as error:
        status.update("abort", "contaminated", reason=str(error)[:300])
        print(json.dumps({"status": "contaminated", "reason": str(error)[:500]}))
        return 3
    except WorkerError as error:
        status.update("error", "failed", error=str(error)[:300])
        print(json.dumps({"status": "failed", "error": str(error)[-2000:]}))
        return 1
    except SystemExit:
        raise
    except Exception:
        import traceback
        detail = traceback.format_exc()[-4000:]
        status.update("error", "failed", error=detail[-300:])
        print(json.dumps({"status": "error", "error": detail}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
