"""Small data-validation, JSON result and tensor conversion helpers."""
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return str(path.resolve())


def read_json(path, schema=None):
    with open(path) as stream:
        value = json.load(stream)
    if schema and value.get("schema") != schema:
        raise ValueError(f"{path}: expected schema {schema!r}, got {value.get('schema')!r}")
    return value


def exact_keys(value, allowed, where, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected object")
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown or missing:
        raise ValueError(f"{where}: unknown={sorted(unknown)}, missing={sorted(missing)}")


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise ValueError(f"unsafe logical name: {name!r}")
    return name


def tensor_name(name):
    if not isinstance(name, str) or not name or '\0' in name:
        raise ValueError(f"invalid tensor name: {name!r}")
    return name


def resolve(base, path):
    p = Path(path)
    return str((p if p.is_absolute() else Path(base) / p).resolve())


def tensor_array(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous()
        if str(value.dtype) == "torch.bfloat16":
            import torch
            return value.view(torch.uint16).numpy(), "bfloat16"
        value = value.numpy()
    result = np.asarray(value)
    if result.dtype.hasobject:
        raise ValueError("object tensors are not supported")
    return result, str(result.dtype)


def validate_io(io):
    exact_keys(io, ("schema", "inputs", "outputs"), "IOSpec", ("schema", "inputs", "outputs"))
    if io["schema"] != "ge-io/1":
        raise ValueError("unsupported IOSpec schema")
    for side in ("inputs", "outputs"):
        names = []
        if not io[side]:
            raise ValueError(f"IOSpec {side} is empty")
        for item in io[side]:
            exact_keys(item, ("name", "dtype", "shape", "dynamic_dims", "node", "format", "max_bytes"),
                       f"IOSpec.{side}", ("name", "dtype", "shape"))
            tensor_name(item["name"])
            names.append(item["name"])
            dynamic = item.get("dynamic_dims", [])
            if sorted(set(dynamic)) != dynamic or any(i < 0 or i >= len(item["shape"]) for i in dynamic):
                raise ValueError(f"invalid dynamic axes: {item}")
            if any(d == 0 or d < -1 for d in item["shape"]):
                raise ValueError(f"invalid shape: {item}")
            if {i for i, d in enumerate(item["shape"]) if d == -1} != set(dynamic):
                raise ValueError(f"shape and dynamic_dims disagree: {item}")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate {side}: {names}")
    return io
