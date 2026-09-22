"""源摄取 — 3 种客户模型形态 (docs §1) 的原始加载。

    name  — transformers hub id / 本地权重目录 → from_pretrained → nn.Module
    torch — 权重目录 (同 name); 或 PyTorch 源码 (module + class_name [+ weights]) → nn.Module
    onnx  — .onnx 文件 → Graph (原样, 不转换不优化)

source 只负责加载原始模型, 不做 patch — 适配是 adapter.adapt 的事 (docs §8)。
"""

import importlib.util
import os

import torch
import torch_npu
from transformers import AutoModelForCausalLM

from .graph import Graph


def load_source(cfg, model_dir, dtype=torch.float16, device=None):
    """按 cfg.source.type 加载源模型。

    返回: name/torch → torch.nn.Module (已在 NPU、eval); onnx → Graph。
    device **必填** (torch 形态): 由调用方从 CLI --device 传下来, 本函数负责
    torch_npu.npu.set_device(device); onnx 形态不碰设备 (可为 None)。
    dtype 仅 torch 形态生效。
    """
    src = cfg.source
    if src.type == "onnx":
        return Graph.from_onnx(_resolve(src.ref, model_dir))

    if src.type not in ("name", "torch"):
        raise ValueError(f"未知 source.type: {src.type} (期望 name | torch | onnx)")
    if device is None:
        raise ValueError("load_source 需要 device (torch 形态; 由 pipeline 的 --device 传入)")

    torch_npu.npu.set_device(device)
    if src.type == "torch" and src.module and src.class_name:
        return _from_source_code(src, model_dir)
    return _from_pretrained(src.ref, dtype)


def _from_pretrained(ref, dtype):
    """form ① (及 form ② 的权重目录子情况): hub id / 权重目录 → nn.Module。"""
    return AutoModelForCausalLM.from_pretrained(ref, dtype=dtype).npu().eval()


def _from_source_code(src, model_dir):
    """form ② PyTorch 源码: importlib 载入 module → 实例化 class_name → 载权重。

    目前无真实实例, 按合理约定实现 (待实例细化):
      - 类无参构造 (构造参数从何而来待定)
      - weights 是单个 state_dict 文件 (torch.load); 分片/safetensors 目录待定
      - dtype 不在此转换 (源码形态的 dtype 约定待实例细化)
    """
    path = _resolve(src.module, model_dir)
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = getattr(module, src.class_name)()
    if src.weights:
        state = torch.load(_resolve(src.weights, model_dir), map_location="npu")
        model.load_state_dict(state)
    return model.npu().eval()


def _resolve(path, model_dir):
    """相对路径优先按 model_dir 解析 (存在才用), 否则原样 (hub id / cwd 相对)。"""
    if not path or os.path.isabs(path):
        return path
    candidate = os.path.join(model_dir, path)
    return candidate if os.path.exists(candidate) else path
