"""源摄取 — torch-only (docs §1): 来路① `ref` → from_pretrained。

目前只实现来路①: transformers 权重 (hub id / 本地权重目录, ref) → from_pretrained
→ nn.Module。来路② (客户 PyTorch 源码 module+class) 与 ONNX (形态③ .onnx) 均**未实现**
— 按 YAGNI 删除, 待真实需求出现再从头实现 (docs §10/§15)。

source 只负责加载原始模型, 不做 patch — 适配是 adapter.adapt 的事 (docs §8)。
"""

import torch
import torch_npu
from transformers import AutoModelForCausalLM


def load_source(cfg, model_dir, dtype=torch.float16, device=None):
    """加载源模型 (torch-only: 来路① cfg.source.ref → from_pretrained)。

    返回: torch.nn.Module (已在 NPU、eval)。
    device **必填**: 由调用方从 CLI --device 传下来, 本函数负责
    torch_npu.npu.set_device(device)。dtype 生效。
    来路② (客户源码 module+class) 与 ONNX (形态③) 未实现 (见模块 docstring / docs §10)。
    """
    if device is None:
        raise ValueError("load_source 需要 device (torch 形态; 由 pipeline 的 --device 传入)")

    torch_npu.npu.set_device(device)
    return _from_pretrained(cfg.source.ref, dtype)           # 来路①: transformers 权重


def _from_pretrained(ref, dtype):
    """来路① (torch 形态): hub id / 权重目录 → nn.Module。"""
    return AutoModelForCausalLM.from_pretrained(ref, dtype=dtype).npu().eval()
