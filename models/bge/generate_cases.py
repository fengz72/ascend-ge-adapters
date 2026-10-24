"""Import supplied BGE business samples into the persistent NPY input pool."""
import json
from pathlib import Path
import numpy as np


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
    names = {"query_token_ids" if name == "input_ids" else name for name in DTYPES}
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
            value = np.asarray(values["query_token_ids" if name == "input_ids" else name])
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
