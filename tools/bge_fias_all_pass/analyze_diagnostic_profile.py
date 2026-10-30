#!/usr/bin/env python3
"""Attribute one fixed-case BGE FIAS profile to verified GE iteration windows.

The per-position groups below describe the exact 392-task sequence observed
in this diagnostic capture. Assertions reject a changed sequence rather than
silently applying these labels to another model or build.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def one_file(directory, pattern):
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {directory}: {matches}")
    return matches[0]


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def duration(row):
    return float(row["Task Duration(us)"].strip())


def timestamp(row):
    return float(row["Task Start Time(us)"].strip())


def summarize(group, requests, total):
    time = sum(duration(row) for row in group)
    return {"kernels_per_request": len(group) / requests,
            "sum_us_per_request": time / requests,
            "fraction_of_summed_kernel_time": time / total}


def main(diag):
    profile_dir = diag / "profile-runtime" / "bench-profiling"
    session = one_file(profile_dir, "PROF_*")
    exported = session / "mindstudio_profiler_output"
    op_csv = one_file(exported, "op_summary_*.csv")
    step_csv = one_file(exported, "step_trace_*.csv")
    ops = rows(op_csv)
    steps = rows(step_csv)
    if not ops or not steps:
        raise ValueError("empty exported profile")
    events = [json.loads(line) for line in (diag / "profile-runtime/events.jsonl").open()]
    event_counts = Counter((event["instance"], event["phase"]) for event in events)
    if any(event["status"] != "success" for event in events):
        raise ValueError("failed replay event")
    if sorted(instance for instance, phase in event_counts if phase == "warmup") != list(range(10)):
        raise ValueError("expected ten instances")
    if any(event_counts[(instance, "warmup")] != 10 for instance in range(10)):
        raise ValueError("expected ten warmups per instance")
    measured_requests = sum(count for (_, phase), count in event_counts.items() if phase == "measure")
    if measured_requests != 100:
        raise ValueError(f"expected 100 measured requests, got {measured_requests}")

    windows = defaultdict(list)
    for step in steps:
        model = int(step["Model ID"])
        iteration = int(step["Iteration ID"])
        end = float(step["Iteration End(us)"].strip())
        start = end - float(step["Iteration Time(us)"].strip())
        windows[model].append((iteration, start, end))
    if set(windows) != set(range(1, 11)):
        raise ValueError(f"unexpected model IDs: {sorted(windows)}")
    for model, model_windows in windows.items():
        model_windows.sort()
        if [iteration for iteration, _, _ in model_windows] != list(range(1, len(model_windows) + 1)):
            raise ValueError(f"nonconsecutive iterations for model {model}")
        if len(model_windows) != 10 + event_counts[(model - 1, "measure")]:
            raise ValueError(f"step/event mismatch for model {model}")

    per_step = defaultdict(list)
    for row in ops:
        model = int(row["Model ID"])
        start = timestamp(row)
        end = start + duration(row)
        matches = [iteration for iteration, left, right in windows[model]
                   if left <= start and end <= right + 0.001]
        if len(matches) != 1:
            raise ValueError(f"kernel has {len(matches)} iteration matches: {model}, {row['Op Name']}")
        per_step[(model, matches[0])].append(row)
    if len(per_step) != len(steps) or any(len(group) != 392 for group in per_step.values()):
        raise ValueError("expected 392 uniquely attributed kernels in every step")
    measured = []
    sequences = set()
    for (model, iteration), group in sorted(per_step.items()):
        if iteration <= 10:
            continue
        group.sort(key=timestamp)
        sequences.add(tuple(row["OP Type"] for row in group))
        measured.append(group)
    if len(measured) != measured_requests or len(sequences) != 1:
        raise ValueError("measurement count or operator sequence changed")
    template = measured[0]
    anchors = {1: ("BatchMatMulV2", "/cid_projection/MatMul"),
               6: ("BatchMatMulV2", "/MatMul"),
               72: ("LayerNorm", None),
               84: ("FusedInferAttentionScore", None),
               368: ("LayerNorm", None),
               390: ("MatMulV2", None)}
    for index, (kind, name) in anchors.items():
        if template[index]["OP Type"] != kind or (name and template[index]["Op Name"] != name):
            raise ValueError(f"unexpected anchor at task position {index}")
    if Counter(row["OP Type"] for row in template)["FusedInferAttentionScore"] != 24:
        raise ValueError("expected 24 FIAS kernels")

    # Each attention is preceded by its three Q/K/V projections. This works
    # even when aclnn assigns identical Op Name strings to all 144 encoder MM.
    fias_positions = [index for index, row in enumerate(template)
                      if row["OP Type"] == "FusedInferAttentionScore"]
    qkv_positions = set()
    output_positions = set()
    for index in fias_positions:
        before = [j for j in range(index) if template[j]["OP Type"] == "MatMulV2"]
        after = [j for j in range(index + 1, 369)
                 if template[j]["OP Type"] == "MatMulV2"]
        qkv_positions.update(before[-3:])
        output_positions.add(after[0])
    if len(qkv_positions) != 72 or len(output_positions) != 24 or qkv_positions & output_positions:
        raise ValueError("cannot identify 72 QKV and 24 attention-output projections")
    if any("1024,1024" not in template[i]["Input Shapes"] for i in qkv_positions):
        raise ValueError("QKV projection shapes changed")
    if any(template[i + 1]["OP Type"] != "Add" or
           template[i + 2]["OP Type"] != "LayerNorm" for i in output_positions):
        raise ValueError("attention-output projection structure changed")
    ffn_up = {i for i, row in enumerate(template)
              if row["OP Type"] == "MatMulV2" and "1024,4096" in row["Input Shapes"]}
    ffn_down = {i for i, row in enumerate(template)
                if row["OP Type"] == "MatMulV2" and "4096,1024" in row["Input Shapes"]}
    if len(ffn_up) != 24 or len(ffn_down) != 24:
        raise ValueError("cannot identify FFN projections")
    norm_positions = {i for i, row in enumerate(template) if row["OP Type"] == "LayerNorm"}
    if len(norm_positions) != 49:
        raise ValueError("expected 49 fused LayerNorm kernels")
    l2_chains = {"candidate_l2": range(2, 6), "stat_l2": range(47, 51),
                 "long_l2": range(60, 64), "query_l2": range(372, 376),
                 "fused_l2": range(386, 390)}
    for key, indices in l2_chains.items():
        if [template[i]["OP Type"] for i in indices] != ["LpNormV2", "Maximum", "Expand", "RealDiv"]:
            raise ValueError(f"unexpected {key} structure")

    positions = {"candidate_constant_chain": set(range(1, 7)),
                 "long_behavior_branch": set(range(7, 27)) | set(range(51, 64)),
                 "stat_branch": set(range(27, 51)),
                 "embedding": set(range(64, 73)),
                 "encoder": set(range(73, 369)),
                 "query_gate_score": set(range(369, 392)),
                 "qkv_projection": qkv_positions,
                 "attention_fias": set(fias_positions),
                 "attention_output_projection": output_positions,
                 "ffn_up_projection": ffn_up,
                 "ffn_down_projection": ffn_down,
                 "layernorm": norm_positions,
                 "gelu": {i for i, row in enumerate(template) if row["OP Type"] == "Gelu"},
                 "gate_weighting": set(range(376, 386)),
                 "final_score_matmul_cast": {390, 391}}
    positions.update({key: set(indices) for key, indices in l2_chains.items()})
    phase_names = ("candidate_constant_chain", "long_behavior_branch", "stat_branch",
                   "embedding", "encoder", "query_gate_score")
    if len(set().union(*(positions[k] for k in phase_names))) != 391 or \
            sum(len(positions[k]) for k in phase_names) != 391:
        raise ValueError("phase partitions do not cover all non-Data tasks")
    total = sum(duration(row) for group in measured for row in group)
    result = {"session": str(session), "op_csv": str(op_csv), "step_csv": str(step_csv),
              "op_csv_sha256": hashlib.sha256(op_csv.read_bytes()).hexdigest(),
              "step_csv_sha256": hashlib.sha256(step_csv.read_bytes()).hexdigest(),
              "model_ids": sorted(windows), "warmup_steps": 100,
              "measured_steps": len(measured), "kernels_per_step": 392,
              "all_op_rows_uniquely_assigned": len(ops),
              "type_sequence_sha256": hashlib.sha256(repr(tuple(row["OP Type"] for row in template)).encode()).hexdigest(),
              "total_summed_kernel_us_per_request": total / measured_requests,
              "groups": {}, "op_types": {},
              "limitations": ["Kernel durations are additive work across ten concurrent streams, not request wall time.",
                              "The 392-position labels apply only to this verified graph and fixed case.",
                              "Profiling adds overhead and does not establish throughput or p99 acceptance."]}
    for name, indices in positions.items():
        result["groups"][name] = summarize([group[index] for group in measured for index in sorted(indices)],
                                             measured_requests, total)
    for kind in sorted({row["OP Type"] for group in measured for row in group}):
        result["op_types"][kind] = summarize([row for group in measured for row in group if row["OP Type"] == kind],
                                              measured_requests, total)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("diag", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = main(args.diag)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"measured_steps": result["measured_steps"],
                      "kernels_per_step": result["kernels_per_step"],
                      "total_summed_kernel_us_per_request": result["total_summed_kernel_us_per_request"],
                      "groups": result["groups"]}, indent=2))
