"""Aggregate the two BGE accuracy gates: CPU-golden numerics and business Top-10.

Standalone accuracy gate for the standalone BGE flow.  The two sides are read
in their own formats: the golden is ``fixtures/cpu-golden.npz`` in the
``<case>/outputs/scores`` layout (see ``fixtures/SOURCE.md``), and the candidate
is ``ge_execute``'s infer output directory -- ``result.json`` plus
``<case_id>/output_<i>.npy``.  The runtime names
outputs positionally (``output_0``), so ``OUTPUT_NAMES`` below maps them onto the
graph's own names by position; ``verify.py`` hardcodes the same name.

    python3 model/compare.py --golden fixtures/cpu-golden.npz \\
        --expected fixtures/expected_outputs.json --infer-dir work/infer \\
        --output-dir work/compare \\
        --cosine-min 0.9999 --relative-l2-max 0.001 --max-abs-error 0.001 \\
        --atol 0.001

Exit code 0 only when every case passes both gates; the per-case breakdown is
written to ``<output-dir>/comparison.json``.

The two gates are **not** interchangeable.  The numeric gate bounds the error on
all 10112 scores; the business gate checks the ranking of the 10 the business
actually consumes.  Measured over the five fixtures, the tightest adjacent
Top-10 gap is 5.26e-4, so flipping one pair needs a one-sided error above
2.63e-4 -- while this FP16 build's measured max absolute error is 2.36e-4 to
3.38e-4, i.e. already past that bound on three of the five cases.  No
``--max-abs-error`` value can therefore both admit this build and guarantee the
ranking; see models/bge/README.md.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

# The business gate is verify.py next to this file, imported directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify as business  # noqa: E402

# An all-zero golden row is judged by absolute closeness.
COSINE_ZERO_ATOL = 1e-5

# The graph's output names, in graph order.  ge_execute reports outputs only
# positionally (output_0, output_1, ...), so this is what turns them back into
# the names the golden npz and verify.py use.  BGE has exactly one output;
# _load_infer asserts the count against result.json, so a graph that grows an
# output fails here rather than being compared against the wrong reference.
OUTPUT_NAMES = ("scores",)


def _load(path):
    """Load a save_rows npz as ``{case: {output name: array}}``."""
    rows = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        for key in archive.files:
            case_id, sep, tensor = key.partition("/outputs/")
            if not sep:
                continue
            rows.setdefault(case_id, {})[tensor] = np.asarray(archive[key])
    if not rows:
        raise ValueError(f"{path}: no <case>/outputs/<tensor> entries")
    return rows


def _resolve(infer_dir, case_id, recorded):
    """Locate an output NPY **inside infer_dir**, which is the authority.

    ``result.json`` records ``path`` relative to ge_execute's own cwd when
    ``--output-dir`` was relative, and absolutely when it was not.  Following it
    blindly would let ``--infer-dir X`` silently read outputs from somewhere
    other than X -- a moved or copied result tree whose original location still
    holds stale NPY files would gate the stale data.  So the documented
    ``<output-dir>/<case_id>/output_<i>.npy`` layout (runtime/README.md) is
    rebuilt from ``infer_dir`` and only the filename is taken from the record.
    """
    expected = Path(infer_dir) / case_id / Path(recorded).name
    if expected.is_file():
        return expected
    raise FileNotFoundError(
        f"{case_id}: no {expected} (result.json indexed {recorded!r}); either the runtime's "
        f"output layout changed or --infer-dir does not point at this run's output directory")


def _load_infer(infer_dir):
    """Read ge_execute's infer output as ``{case: {output name: array}}``.

    Only bytes are copied through; the shape and dtype of every array are checked
    against what ``result.json`` declares, so a truncated or stale NPY is a loud
    error rather than a wrong comparison.
    """
    infer_dir = Path(infer_dir)
    result_path = infer_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"runtime produced no {result_path}")
    result = json.loads(result_path.read_text())
    if result.get("status") != "success":
        raise ValueError(f"{result_path}: status is {result.get('status')!r}, not 'success'")
    if result.get("mode") != "infer":
        raise ValueError(f"{result_path}: mode is {result.get('mode')!r}; a bench run carries "
                         f"no per-case outputs to gate")
    names = list(OUTPUT_NAMES)
    cases = result.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{result_path}: 'cases' must be a nonempty array")
    rows = {}
    for entry in cases:
        case_id = entry["id"]
        if case_id in rows:
            raise ValueError(f"{result_path}: duplicate case {case_id!r}")
        outputs = entry["outputs"]
        if len(outputs) != len(names):
            raise ValueError(f"{case_id}: runtime returned {len(outputs)} outputs but "
                             f"OUTPUT_NAMES declares {len(names)}: {names}")
        tensors = {}
        for index, output in enumerate(outputs):
            source = _resolve(infer_dir, case_id, output["path"])
            array = np.load(source, allow_pickle=False)
            if list(array.shape) != [int(dim) for dim in output["shape"]]:
                raise ValueError(f"{case_id}/{names[index]}: {source} is {array.shape} but "
                                 f"result.json declares {output['shape']}")
            if str(array.dtype) != output["dtype"]:
                raise ValueError(f"{case_id}/{names[index]}: {source} is {array.dtype} but "
                                 f"result.json declares {output['dtype']}")
            tensors[names[index]] = np.asarray(array)
        rows[case_id] = tensors
    return rows


def _numeric(reference, candidate, cosine_min, relative_l2_max, max_abs_error):
    """Per-sample numeric comparison (the ``per_sample`` scope).

    Rows are the first-axis slices of the output (BGE scores are ``[1, 10112]``,
    so one row per captured sample).  A zero golden row is judged by absolute
    closeness instead of an epsilon-patched cosine; a zero candidate row against
    a nonzero golden row scores cosine 0.0.  Non-finite values fail outright.
    """
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return {"passed": False, "reason": "nonfinite", "cosine": None,
                "relative_l2": None, "max_abs_error": None, "samples": []}
    a = reference.astype(np.float64)
    b = candidate.astype(np.float64)
    rows = [(a.reshape(-1), b.reshape(-1))]
    if a.ndim:
        rows = [(x.reshape(-1), y.reshape(-1)) for x, y in zip(a, b)]
    samples = []
    for x, y in rows:
        if x.size == 0:
            raise ValueError("empty numeric output")
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        diff = np.abs(x - y)
        if nx == 0:
            # No relative metric for an all-zero golden row.
            close = bool(np.all(diff <= COSINE_ZERO_ATOL))
            cosine, l2 = (1.0, 0.0) if close else (0.0, None)
            passed = close
        else:
            cosine = float(np.dot(x, y) / (nx * ny)) if ny else 0.0
            l2 = float(np.linalg.norm(x - y) / nx)
            passed = cosine >= cosine_min and l2 <= relative_l2_max
        passed = passed and bool(diff.max() <= max_abs_error)
        samples.append({"passed": bool(passed), "cosine": cosine,
                        "relative_l2": l2, "max_abs_error": float(diff.max())})
    return {"passed": all(sample["passed"] for sample in samples),
            "cosine": min(sample["cosine"] for sample in samples),
            "relative_l2": max(sample["relative_l2"] if sample["relative_l2"] is not None else 0.0
                               for sample in samples),
            "max_abs_error": max(sample["max_abs_error"] for sample in samples),
            "samples": samples}


def run(golden_path, expected_path, infer_dir, output_dir, cosine_min,
        relative_l2_max, max_abs_error, atol):
    golden = _load(golden_path)
    infer = _load_infer(infer_dir)
    if set(golden) != set(infer):
        raise ValueError("golden/infer case IDs differ: "
                         f"missing outputs={sorted(set(golden) - set(infer))}, "
                         f"missing golden={sorted(set(infer) - set(golden))}")
    # No intersection with the golden's key set is needed here: _load_infer keys
    # every case by OUTPUT_NAMES and asserts the count against result.json, and
    # the per-case check below already fails loudly if the golden carries
    # different names.
    names = list(OUTPUT_NAMES)
    reports = {}
    for case_id in sorted(golden):
        if set(golden[case_id]) != set(infer[case_id]):
            raise ValueError(f"{case_id}: output names differ")
        outputs = {}
        for name in names:
            outputs[name] = _numeric(golden[case_id][name], infer[case_id][name],
                                     cosine_min, relative_l2_max, max_abs_error)
        reports[case_id] = {"numeric": {"passed": all(x["passed"] for x in outputs.values()),
                                       "outputs": outputs}}
    # Business gate: verify.py:compare over the same case rows.
    config = {"_root": str(Path.cwd()),
              "verification": {"case_ids": "all",
                               "params": {"expected_outputs": str(Path(expected_path).resolve()),
                                          "atol": atol}}}
    business_reports = business.compare(config, {case_id: {"outputs": infer[case_id]}
                                                 for case_id in sorted(infer)})
    for case_id, report in business_reports.items():
        reports[case_id]["business"] = report
        reports[case_id]["passed"] = (reports[case_id]["numeric"]["passed"]
                                      and report["passed"])
    failed = sorted(case_id for case_id, report in reports.items() if not report["passed"])
    report = {
        "status": "passed" if not failed else "failed",
        "passed": not failed,
        "thresholds": {"cosine_min": cosine_min, "relative_l2_max": relative_l2_max,
                       "max_abs_error": max_abs_error, "atol": atol},
        "golden": str(Path(golden_path).resolve()),
        "infer": str(Path(infer_dir).resolve()),
        "expected": str(Path(expected_path).resolve()),
        "outputs": names,
        "case_ids": sorted(reports),
        "failed_cases": failed,
        "cases": {case_id: reports[case_id] for case_id in sorted(reports)},
    }
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    comparison = path / "comparison.json"
    comparison.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report, comparison.resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate the BGE CPU-golden and business Top-10 accuracy gates."
    )
    parser.add_argument("--golden", required=True,
                        help="CPU FP32 reference npz (fixtures/cpu-golden.npz)")
    parser.add_argument("--expected", required=True, help="business reference JSON")
    parser.add_argument("--infer-dir", required=True,
                        help="ge_execute infer output directory (result.json + <case_id>/output_<i>.npy)")
    parser.add_argument("--output-dir", required=True,
                        help="directory for comparison.json")
    parser.add_argument("--cosine-min", type=float, default=0.9999)
    parser.add_argument("--relative-l2-max", type=float, default=0.001)
    parser.add_argument("--max-abs-error", type=float, default=0.001)
    parser.add_argument("--atol", type=float, default=0.001)
    args = parser.parse_args(argv)
    try:
        report, comparison = run(args.golden, args.expected, args.infer_dir,
                                 args.output_dir, args.cosine_min, args.relative_l2_max,
                                 args.max_abs_error, args.atol)
    except Exception as error:  # noqa: BLE001 - one-line failure report, nonzero exit
        print(f"compare: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "cases": len(report["case_ids"]),
                      "failed_cases": report["failed_cases"],
                      "comparison": str(comparison)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
