"""Standalone single-graph model entry for the StartLux unified ONE-AIR route.

The model composition entry of the standalone shell flow
(docs/standalone-runtime-plan.md): the migration of the validated
unified_adapter candidate composition (prepare_candidate /
build_unified_candidate, P6/P7 accepted) onto the retained production modules
-- no ``core``, no pipeline adapter, no handle protocol, no YAML.  All decoder
and vision arithmetic stays in the already-validated modules this file merely
composes:

* ``load(device)``: direct weight loading from STARTLUX_MODEL_DIR as
  bfloat16 on the NPU (the build.dtype=bfloat16 CLI load of the framework
  route), then multimodal.build_unified_model composes the official FP32
  vision tower (constant buffers precomputed by the official methods on a CPU
  float32 copy), the BF16 language trunk with the bit-exact F32 A_log restore
  and the independent F32 head26, and the wrapper is moved with a device-ONLY
  ``.to`` (the registered vision buffers start on CPU when the source came
  from the NPU).  The mixed-dtype placement contract (vision float32, trunk
  bfloat16 except the 18 A_log float32 parameters, head26 float32, everything
  on the target device) is verified before the wrapper is returned -- an
  unknown parameter is an error, not a pass.
* ``io_contract(report)``: the 5-input/1-output unified graph IO
  (input_ids [B,S] i64, position_ids [3,B,S] i64, actual_seq_lengths [B] i32,
  pixel_values [512,1536] f32 STATIC, image_feature_index [B,S] i64 ->
  scores [B,26] f32; dynamic B/S with the static 2-slot global pixel pool).
* ``export_example(wrapper, report, device)``: the deterministic B2/S65 trace
  example (multimodal.example_inputs) that keeps both dynamic axes symbolic.

Model params come from startlux_npu_adapter._params defaults (the single
source: max_seq_len 2048 / compiled_seq_len 256 / chunk_size 64 / max_batch 4 /
native GDR); this file invents no copies.  Consumed by export_air.py.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root
ROOT = MODEL_DIR.parents[1]


def _load_module(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(MODEL_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    # register before exec: dynamo resolves traced functions by __module__
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


base = _load_module("startlux_npu_adapter_for_native", "model/startlux_npu_adapter.py")
multimodal = _load_module("startlux_multimodal_for_unified", "model/multimodal.py")


def _config(folder):
    return {"model": {"name": "startlux-decision-v1.1-0.8b-instant", "params": {}},
            "source": {"kind": "transformers",
                       "loader": "Qwen3_5ForConditionalGeneration",
                       "ref": str(Path(folder).resolve())}}


def source_folder():
    value = os.environ.get("STARTLUX_MODEL_DIR")
    if not value or not value.strip():
        raise ValueError("STARTLUX_MODEL_DIR is not set or is empty")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"STARTLUX_MODEL_DIR {path} is not a directory")
    return path


def _expected_param_dtype(name):
    """The per-name dtype contract of one unified-wrapper parameter.

    vision.* Linear/Conv params      -> float16   (the P2-5 pre-quantized fp16
                                                   vision route; values are
                                                   bit-identical to the removed
                                                   per-forward casts)
    other vision.* params (norms)    -> float32   (the official FP32 vision
                                                   reduction domain)
    text.head_rows                   -> float32   (the independent F32 head26 clone)
    text.body.*.linear_attn.A_log    -> float32   (the deliberate bit-exact F32 restore
                                                   of the 18 GDR decay logs; the checkpoint
                                                   stores them F32 and the BF16 load would
                                                   round them)
    every other text.body.* param    -> bfloat16  (the validated native trunk)
    anything else                    -> None      (an unknown parameter: rejected, the
                                                   contract is exhaustive, not permissive)
    """
    if name.startswith("vision."):
        for module_prefix in ("attn.qkv.", "attn.proj.", "mlp.linear_fc1.",
                              "mlp.linear_fc2.", "merger.linear_fc1.",
                              "merger.linear_fc2.", "patch_embed.conv."):
            if f".{module_prefix}" in name:
                return torch.float16
        return torch.float32
    if name == "text.head_rows":
        return torch.float32
    if name.startswith("text.body."):
        return torch.float32 if name.endswith(".linear_attn.A_log") else torch.bfloat16
    return None


def _verify_candidate_placement(unified, device):
    """The mixed-dtype device contract of the composed wrapper, checked per name."""
    tensors = dict(unified.named_parameters())
    tensors.update(dict(unified.named_buffers()))
    stray = {name: str(t.device) for name, t in tensors.items() if t.device != device}
    if stray:
        raise RuntimeError(f"the unified wrapper still carries off-device tensors "
                           f"after .to({device}): {stray}")
    wrong, a_log = [], []
    for name, param in unified.named_parameters():
        expected = _expected_param_dtype(name)
        if expected is None:
            wrong.append(f"{name}: unknown parameter (no dtype rule)")
        elif param.dtype != expected:
            wrong.append(f"{name}: {param.dtype} != {expected}")
        if name.endswith(".linear_attn.A_log"):
            a_log.append(name)
    if wrong:
        raise RuntimeError("the composed wrapper violates the mixed-dtype contract: "
                           f"{wrong}")
    if not a_log:
        raise RuntimeError("no linear_attn.A_log parameters found: the composed wrapper "
                           "is not the GDR language trunk this contract describes")
    return {"device": str(device), "tensors": len(tensors),
            "vision": "float32", "language_trunk": "bfloat16", "head26": "float32",
            "a_log_float32": sorted(a_log),
            "buffers": [n for n, _ in unified.vision.named_buffers()]}


def load(device):
    """Load the checkpoint as BF16 on the NPU and compose the unified wrapper.

    ``device`` is the in-process device index (with ASCEND_RT_VISIBLE_DEVICES
    pinning one chip, 0).  Returns (unified MultimodalDecoder, report).
    """
    import torch_npu

    folder = source_folder()
    target = torch.device(f"npu:{int(device)}")
    torch_npu.npu.set_device(target)
    from transformers import Qwen3_5ForConditionalGeneration

    source = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(folder), dtype=torch.bfloat16, local_files_only=True)
    source = source.to(device=target, dtype=torch.bfloat16).eval()
    if source.dtype != torch.bfloat16:
        raise ValueError("the unified candidate expects the bfloat16 load "
                         f"(the source is {source.dtype})")
    unified, report = multimodal.build_unified_model(source, _config(folder))
    unified.to(target)                  # device-only move: no dtype cast anywhere
    report["placement"] = _verify_candidate_placement(unified, target)
    return unified, report


def io_contract(report):
    """The graph IO spec of the single AIR (schema, inputs, outputs)."""
    return {"schema": "ge-io/1", "inputs": report["io"]["inputs"],
            "outputs": report["io"]["outputs"]}


def export_example(wrapper, report, device):
    """The deterministic export trace example (B2/S65, both dynamic axes >= 2)."""
    io = io_contract(report)
    spec = {"format": "torch", "model": wrapper, "io": io}
    spec["example_inputs"] = {name: torch.from_numpy(value).to(f"npu:{int(device)}")
                              for name, value in multimodal.example_inputs().items()}
    return spec


if __name__ == "__main__":
    device = int(os.environ.get("STARTLUX_DEVICE", "0"))
    unified, report = load(device)
    placement = report["placement"]
    print(f"model: composed unified wrapper on {placement['device']} "
          f"({placement['tensors']} tensors, vision {placement['vision']}, "
          f"trunk {placement['language_trunk']}, head26 {placement['head26']}, "
          f"{len(placement['a_log_float32'])} A_log f32); io: "
          f"{len(report['io']['inputs'])} inputs -> {len(report['io']['outputs'])} output")
