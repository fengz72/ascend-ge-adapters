"""Strict per-row tensor comparison between two ge-runtime result directories.

Reads the runtime infer/golden layout -- ``result.json`` (the GE_RESULT object
with an ``outputs`` index of case -> rows -> tensors: name, dtype, shape,
offset, bytes) plus the raw ``outputs.bin`` both sides index into -- and gates
every row of every output tensor against cosine / relative-L2 thresholds.

First-version scope (deliberate): float32 outputs only, one captured row per
case (the infer contract), NumPy as the only dependency -- no model modules, no
Torch, no TorchNPU, no ``core``.  NaN/Inf are hard failures.  Rows are the
first-axis slices of each output (for the StartLux ``[B, 26]`` scores that is
one row per batch sample, so a single bad row fails the comparison); zero
reference rows are judged by absolute closeness (atol 1e-5) instead of an
epsilon-patched cosine, and a zero candidate row against a nonzero reference
scores cosine 0.0 -- no silent passes.

Exit code 0 only when every row of every case passes both thresholds; the
report is written to ``<output>/comparison.json``.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SUPPORTED_DTYPES = {"float32": np.float32}
ZERO_ATOL = 1e-5


def _fail(message):
    raise SystemExit(f"compare: {message}")


def _tensor_dtype(name):
    dtype = SUPPORTED_DTYPES.get(name)
    if dtype is None:
        _fail(f"unsupported output dtype {name!r} (first version supports "
              f"{sorted(SUPPORTED_DTYPES)}); refusing a silent conversion")
    return dtype


def load_directory(directory):
    """One result directory -> {case: {output name: float32 ndarray}}.

    The layout is the runtime infer contract the golden stage mirrors exactly:
    result.json with an ``outputs`` mapping of case -> captured rows (one row,
    instance 0) and each row's tensors carrying name/dtype/shape/offset/bytes
    into the directory's outputs.bin.  Structural violations are loud errors.
    """
    directory = Path(directory)
    result_path = directory / "result.json"
    binary_path = directory / "outputs.bin"
    if not result_path.is_file() or not binary_path.is_file():
        _fail(f"{directory} must contain result.json and outputs.bin")
    result = json.loads(result_path.read_text())
    if result.get("status") != "success":
        _fail(f"{result_path}: status is {result.get('status')!r}, not 'success'")
    outputs = result.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        _fail(f"{result_path}: 'outputs' must be a nonempty object")
    blob = binary_path.read_bytes()
    parsed = {}
    for case_id, rows in outputs.items():
        if not isinstance(rows, list) or len(rows) != 1:
            _fail(f"{result_path}: case {case_id!r} must carry exactly one captured "
                  f"row (got {len(rows) if isinstance(rows, list) else type(rows).__name__})")
        row = rows[0]
        if row.get("instance") != 0:
            _fail(f"{result_path}: case {case_id!r} row is not instance 0")
        if case_id in parsed:
            _fail(f"{result_path}: duplicate case {case_id!r}")
        tensors = {}
        for tensor in row.get("tensors", []):
            name = tensor["name"]
            if name in tensors:
                _fail(f"{result_path}: duplicate output {name!r} in case {case_id!r}")
            dtype = _tensor_dtype(tensor["dtype"])
            shape = [int(size) for size in tensor["shape"]]
            expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
            offset, size = int(tensor["offset"]), int(tensor["bytes"])
            if size != expected:
                _fail(f"{result_path}: case {case_id!r} output {name!r} declares "
                      f"{size} bytes but shape {shape} of {tensor['dtype']} needs {expected}")
            if offset < 0 or offset + size > len(blob):
                _fail(f"{result_path}: case {case_id!r} output {name!r} byte range "
                      f"[{offset}, {offset + size}) exceeds outputs.bin ({len(blob)} bytes)")
            array = np.frombuffer(blob, dtype=dtype, count=int(np.prod(shape, dtype=np.int64)),
                                  offset=offset).reshape(shape)
            tensors[name] = np.array(array, dtype=np.float64, copy=True)
        if not tensors:
            _fail(f"{result_path}: case {case_id!r} carries no output tensors")
        parsed[case_id] = tensors
    return parsed


def _row_metrics(reference, candidate):
    """Per-row cosine / relative L2 with the accepted zero-row semantics."""
    norm_ref = float(np.linalg.norm(reference))
    norm_cand = float(np.linalg.norm(candidate))
    diff = np.abs(reference - candidate)
    if norm_ref == 0.0:
        close = bool(np.all(diff <= ZERO_ATOL))
        return (1.0, 0.0, close) if close else (0.0, None, False)
    cosine = float(np.dot(reference, candidate) / (norm_ref * norm_cand)) if norm_cand else 0.0
    relative_l2 = float(np.linalg.norm(reference - candidate) / norm_ref)
    return cosine, relative_l2, None


def compare_case(reference, candidate, case_id, thresholds):
    """One case: every output name must exist on both sides, shapes must
    agree, all rows must pass both thresholds."""
    if set(reference) != set(candidate):
        _fail(f"case {case_id!r}: output names differ between golden "
              f"{sorted(reference)} and candidate {sorted(candidate)}")
    outputs = {}
    worst_cosine, worst_l2 = 1.0, 0.0
    for name in reference:
        gold, cand = reference[name], candidate[name]
        if gold.shape != cand.shape:
            _fail(f"case {case_id!r} output {name!r}: shape mismatch "
                  f"{gold.shape} != {cand.shape}")
        if gold.ndim < 1:
            _fail(f"case {case_id!r} output {name!r}: empty dimension in shape {gold.shape}")
        if not np.isfinite(gold).all() or not np.isfinite(cand).all():
            outputs[name] = {"passed": False, "rows": 0, "reason": "nonfinite values"}
            continue
        rows = [(gold[i].reshape(-1), cand[i].reshape(-1))
                for i in range(gold.shape[0])] if gold.ndim else [(gold.reshape(-1), cand.reshape(-1))]
        samples, failed_rows = [], []
        for index, (x, y) in enumerate(rows):
            cosine, relative_l2, forced = _row_metrics(x, y)
            if forced is None:
                passed = (cosine >= thresholds["cosine_min"]
                          and relative_l2 <= thresholds["relative_l2_max"])
            else:
                passed = forced
            samples.append({"passed": bool(passed), "cosine": cosine,
                            "relative_l2": relative_l2})
            if not passed:
                failed_rows.append(index)
            worst_cosine = min(worst_cosine, cosine)
            if relative_l2 is not None:
                worst_l2 = max(worst_l2, relative_l2)
        outputs[name] = {"passed": not failed_rows, "rows": len(samples),
                         "failed_rows": failed_rows, "samples": samples}
    return {"passed": all(value["passed"] for value in outputs.values()),
            "outputs": outputs,
            "worst_cosine": worst_cosine, "worst_relative_l2": worst_l2}


def main():
    parser = argparse.ArgumentParser(
        description="Compare golden and candidate ge-runtime result directories")
    parser.add_argument("--golden", required=True, help="golden result directory")
    parser.add_argument("--output", required=True, help="candidate result directory")
    parser.add_argument("--cosine-min", type=float, default=0.999)
    parser.add_argument("--relative-l2-max", type=float, default=0.02)
    args = parser.parse_args()
    thresholds = {"cosine_min": args.cosine_min, "relative_l2_max": args.relative_l2_max}

    golden = load_directory(args.golden)
    candidate = load_directory(args.output)
    if set(golden) != set(candidate):
        _fail(f"case sets differ: missing in candidate "
              f"{sorted(set(golden) - set(candidate))}, missing in golden "
              f"{sorted(set(candidate) - set(golden))}")

    cases = {case_id: compare_case(golden[case_id], candidate[case_id], case_id, thresholds)
             for case_id in golden}
    worst_cosine = min(case_["worst_cosine"] for case_ in cases.values())
    worst_l2 = max(case_["worst_relative_l2"] for case_ in cases.values())
    failed = {case_id: {name: value for name, value in case_["outputs"].items()
                        if not value["passed"]}
              for case_id, case_ in cases.items() if not case_["passed"]}
    report = {"schema": "ge-compare/1",
              "golden": str(Path(args.golden).resolve()),
              "output": str(Path(args.output).resolve()),
              "thresholds": thresholds,
              "passed": not failed,
              "cases": len(cases),
              "rows": sum(case_["outputs"][name]["rows"]
                          for case_ in cases.values() for name in case_["outputs"]),
              "worst": {"cosine": worst_cosine, "relative_l2": worst_l2},
              "case_reports": cases}
    if failed:
        report["failed_cases"] = failed
    target = Path(args.output) / "comparison.json"
    target.write_text(json.dumps(report, indent=1) + "\n")

    print(f"compare: {len(cases)} cases, worst cosine {worst_cosine:.6f} "
          f"(min {thresholds['cosine_min']}), worst relative L2 {worst_l2:.6f} "
          f"(max {thresholds['relative_l2_max']})")
    for case_id, outputs in failed.items():
        for name, value in outputs.items():
            reason = value.get("reason") or f"failed rows {value.get('failed_rows')}"
            print(f"FAIL {case_id}/{name}: {reason}", file=sys.stderr)
    if failed:
        _fail(f"{len(failed)} case(s) failed; see {target}")
    print(f"PASS; report: {target}")


if __name__ == "__main__":
    main()
