"""Structural model protocol; customer modules never import core."""
import importlib.util
import sys
from core.contracts import file_digest, digest, tensor_name
from pathlib import Path


def load_adapter(path):
    name = "_ge_model_" + digest({"path": str(Path(path).resolve()), "sha256": file_digest(path)})[:16]
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    for fn in ("prepare_reference", "prepare_candidate", "prepare_runtime"):
        if not callable(getattr(module, fn, None)):
            raise TypeError(f"adapter must implement {fn}")
    return module


def validate_handle(handle, candidate=False):
    for method in (("encode", "export_spec") if candidate else ("encode", "run", "decode")):
        if not callable(getattr(handle, method, None)):
            raise TypeError(f"model handle must implement {method}")
    return handle


def canonical(value):
    if set(value) != {"outputs", "alignment"} or not value["outputs"]:
        raise ValueError("decode must return nonempty outputs and explicit alignment")
    for name in value['outputs']:tensor_name(name)
    return value
