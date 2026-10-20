"""Source loading only. No model-specific forward or output assumptions."""
import importlib
from pathlib import Path


def torch_device(config, reference=False, device=None):
    target = config["reference"]["device"] if reference else "npu"
    if target == "npu":
        from core.backends.runtime import require_device
        target = f"npu:{require_device(device)}"
        import torch_npu
        torch_npu.npu.set_device(target)
    return target


def load_source(config, reference=False, device=None):
    source = config["source"]
    if source["kind"] == "onnx":
        path = Path(source["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    import torch
    target = torch_device(config, reference, device)
    dtype = getattr(torch, config["reference"]["dtype"] if reference else config["build"]["dtype"])
    if source["kind"] == "transformers":
        import transformers
        cls = getattr(transformers, source["loader"])
        model = cls.from_pretrained(source["ref"], dtype=dtype, local_files_only=True)
    else:
        module, function = source["factory"].split(":")
        model = getattr(importlib.import_module(module), function)(**source.get("params", {}))
    return model.to(device=target, dtype=dtype).eval()
