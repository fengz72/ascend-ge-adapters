"""Import supplied BGE business samples into the persistent NPY input pool.

Two entry points share the same code:

* the standalone CLI -- ``python3 model/generate_cases.py --samples <json>
  --output-dir <dir>`` -- invoked by ``run.sh`` step 2.  Besides the NPY pool it
  writes ``cases.json`` (schema ``ge-cases/1``) for ``ge_execute``;
* the module-level ``generate(config)`` / ``load(config)`` API over the
  ``config["data"]`` dict, retained with unchanged signatures and return shapes;
  the CLI below calls ``generate`` directly.  This path writes no ``cases.json``:
  that file is a ``ge_execute`` artifact.

``cases.json`` lists inputs in ``DTYPES`` order, which **is** the ONNX graph's
input order -- ``ge_execute`` binds positionally and validates nothing
(runtime/README.md).  The order is deliberately not re-derived from the graph:
the ONNX is a frozen delivery (fixtures/SOURCE.md), the eight names, dtypes and
static shapes are already asserted below and by ``validate_arrays``, and reading
the graph header costs ~3 GB RSS because the weights are inline in the protobuf
(``load_external_data=False`` cannot skip them).  A wrong order is not silent
end to end: it shifts the outputs and the cosine gate against
``fixtures/cpu-golden.npz`` fails.  Changing the model means regenerating the
golden, and that is the point to re-check the interface.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


# Insertion order is load-bearing: it is the graph's input order, reproduced
# positionally in cases.json.  Do not sort or reorder without re-checking the
# ONNX graph's own input list.
DTYPES = {
    "input_ids": np.int64,
    "attention_mask": np.int64,
    "long_cid3_raw": np.int64,
    "long_action": np.int64,
    "long_time_bucket": np.int64,
    "stat_dense_raw": np.int64,
    "stat_ratio": np.float32,
    "stat_count": np.int64,
}

# The business samples name the text feature ``query_token_ids``; the ONNX graph
# calls it ``input_ids``.  Every other feature name is identical on both sides.
SAMPLE_KEYS = {"input_ids": "query_token_ids"}


def load(config):
    cases = []
    for folder in sorted(Path(config["data"]["directory"]).glob("case-*")):
        arrays = {name: np.load(folder / f"{name}.npy", allow_pickle=False)
                  for name in DTYPES}
        validate_arrays(arrays, folder)
        batch = arrays["input_ids"].shape[0]
        cases.append({"id": folder.name, "samples": batch,
                      "payload": {name: value.tolist() for name, value in arrays.items()}})
    return cases


def validate_arrays(arrays, label):
    if arrays["input_ids"].ndim != 2 or arrays["long_cid3_raw"].ndim != 2:
        raise ValueError(f"{label}: text and history inputs must have batch and sequence dimensions")
    batch, sequence = arrays["input_ids"].shape
    long_length = arrays["long_cid3_raw"].shape[1]
    if min(batch, sequence, long_length) < 1:
        raise ValueError(f"{label}: empty input dimension")
    shapes = {"input_ids": (batch, sequence), "attention_mask": (batch, sequence),
              "long_cid3_raw": (batch, long_length), "long_action": (batch, long_length),
              "long_time_bucket": (batch, long_length), "stat_dense_raw": (batch, 20),
              "stat_ratio": (batch, 20), "stat_count": (batch,)}
    for name, value in arrays.items():
        if value.dtype != DTYPES[name] or value.shape != shapes[name]:
            raise ValueError(f"{label}/{name}: invalid dtype or shape")
        if not np.isfinite(value).all():
            raise ValueError(f"{label}/{name}: nonfinite input")
    if np.any((arrays["attention_mask"] != 0) & (arrays["attention_mask"] != 1)):
        raise ValueError(f"{label}: attention_mask must contain only 0 and 1")
    if np.any((arrays["stat_count"] < 0) | (arrays["stat_count"] > 20)):
        raise ValueError(f"{label}: stat_count must be between 0 and 20")


def generate(config):
    path = Path(config["data"]["params"]["samples_file"])
    if not path.is_absolute():
        path = Path(config["_root"]) / path
    samples = json.loads(path.read_text())
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples_file must contain a nonempty list")
    prepared = {}
    names = {SAMPLE_KEYS.get(name, name) for name in DTYPES}
    for sample in samples:
        if not isinstance(sample, dict) or type(sample.get("case_id")) is not int or sample["case_id"] < 0:
            raise ValueError("sample case_id must be a nonnegative integer")
        case_id = f"case-{sample['case_id']:04d}"
        if case_id in prepared:
            raise ValueError(f"duplicate sample case_id: {case_id}")
        values = sample.get("input")
        if not isinstance(values, dict) or set(values) != names:
            raise ValueError(f"{case_id}: input names must match the eight BGE features")
        arrays = {}
        for name, dtype in DTYPES.items():
            value = np.asarray(values[SAMPLE_KEYS.get(name, name)])
            if dtype == np.int64 and value.dtype.kind not in "iu":
                raise ValueError(f"{case_id}/{name}: integer values are required")
            arrays[name] = np.asarray([value], dtype=dtype)
        validate_arrays(arrays, case_id)
        prepared[case_id] = arrays
    # Validate every sample before replacing the previous input pool.
    root = Path(config["data"]["directory"])
    root.mkdir(parents=True, exist_ok=True)
    for case_id, arrays in prepared.items():
        folder = root / case_id
        folder.mkdir(exist_ok=True)
        for name, value in arrays.items():
            np.save(folder / f"{name}.npy", value, allow_pickle=False)
        for path in folder.glob("*.npy"):
            if path.stem not in DTYPES:
                path.unlink()
    for folder in root.glob("case-*"):
        if folder.is_dir() and folder.name not in prepared:
            for path in folder.glob("*.npy"):
                path.unlink()
            folder.rmdir()
    return load(config)


def write_cases_json(directory, cases):
    """Emit ge-cases/1 next to the NPY pool, inputs in DTYPES (= graph) order.

    ``path`` is relative to cases.json's own directory, which is what
    runtime/README.md specifies; dtype and shape are deliberately absent --
    ge_execute reads them from each NPY header, so they cannot drift.
    """
    payload = {"cases": [
        {"id": case["id"], "samples": case["samples"],
         "inputs": [{"name": name, "path": f"{case['id']}/{name}.npy"} for name in DTYPES]}
        for case in cases
    ]}
    path = Path(directory) / "cases.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return str(path.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Import the supplied BGE business samples into a case-NNNN NPY pool."
    )
    parser.add_argument("--samples", required=True,
                        help="business sample JSON (a nonempty list of case_id/input objects)")
    parser.add_argument("--output-dir", required=True,
                        help="input pool directory; case-NNNN/ folders are replaced")
    args = parser.parse_args(argv)
    samples = Path(args.samples).resolve()
    if not samples.is_file():
        print(f"generate_cases: samples file is missing: {samples}", file=sys.stderr)
        return 1
    # The plugin reads its inputs from the config dict the CLI builds here;
    # _root is the CLI cwd, and samples_file is already absolute.
    config = {
        "_root": str(Path.cwd()),
        "data": {"directory": args.output_dir,
                 "params": {"samples_file": str(samples)}},
    }
    try:
        cases = generate(config)
        cases_json = write_cases_json(args.output_dir, cases)
    except Exception as error:  # noqa: BLE001 - one-line failure report, nonzero exit
        print(f"generate_cases: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "ok", "cases": len(cases),
                      "directory": str(Path(args.output_dir).resolve()),
                      "cases_json": cases_json}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
