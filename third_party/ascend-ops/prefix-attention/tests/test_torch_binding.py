#!/usr/bin/env python3
"""torch_binding 冒烟测试 — 不依赖 ops-transformer, 仅 import npu_prefix_infer_attention_score。

覆盖:
  1. eager .default vs CPU golden (fp32 参考实现)
  2. eager .tensor  vs .default (bit 级)
  3. torchair GE 整图 (max-autotune, 含图复用) vs eager (bit 级)

运行前置: CANN 算子包已安装到默认路径 (PIA_VENDOR_PATH 可覆盖), source set_env.sh。
"""
import logging
import math
import os
import sys

CANN_HOME = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
VENDOR_PATH = os.environ.get(
    "PIA_VENDOR_PATH", os.path.join(CANN_HOME, "opp/vendors/custom_prefix_attn")
)
os.environ["ASCEND_CUSTOM_OPP_PATH"] = VENDOR_PATH
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "torch_binding"))

import numpy as np
import torch
import torch_npu  # noqa

import npu_prefix_infer_attention_score  # noqa: F401  独立包: 注册schema+eager+converter

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

P, OWN = 20, 130
B = 10
N1, N2, D = 14, 2, 64
T = P + B * OWN
OP = torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score


def cpu_golden(q_tnd, k_tnd, v_tnd, prefix_len, own_lens, n1, n2, scale):
    """CPU fp32 参考实现: 逐 batch 将 QKV 还原为 [prefix, batch-i] 独立计算标准 causal
    attention, 输出拼接为 TND: [prefix_out, b0_out, ..., bn_out] (prefix 段各 batch 结果
    相同, 只保留一份)"""
    g = n1 // n2
    d = q_tnd.shape[-1]
    outs = []
    off = prefix_len
    for L in own_lens:
        q_i = torch.cat([q_tnd[:prefix_len], q_tnd[off:off + L]], 0).to(torch.float32)
        k_i = torch.cat([k_tnd[:prefix_len], k_tnd[off:off + L]], 0).to(torch.float32)
        v_i = torch.cat([v_tnd[:prefix_len], v_tnd[off:off + L]], 0).to(torch.float32)
        s = prefix_len + L
        q_g = q_i.view(s, n2, g, d).permute(1, 2, 0, 3)  # [n2, g, s, d]
        k_h = k_i.permute(1, 0, 2)                       # [n2, s, d]
        v_h = v_i.permute(1, 0, 2)
        scores = torch.einsum("ngqd,nkd->ngqk", q_g, k_h) * scale
        causal = torch.arange(s)[None, :] <= torch.arange(s)[:, None]
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        o = torch.einsum("ngqk,nkd->ngqd", weights, v_h)  # [n2, g, s, d]
        outs.append(o.permute(2, 0, 1, 3).reshape(s, n1, d))
        off += L
    return torch.cat([outs[0][:prefix_len]] + [o[prefix_len:] for o in outs], 0)


class PiaModule(torch.nn.Module):
    def __init__(self, act_q, act_kv, scale):
        super().__init__()
        self.act_q = act_q
        self.act_kv = act_kv
        self.scale = scale

    def forward(self, query, key, value, mask):
        return OP(query, key, value,
                  atten_mask=mask, actual_seq_lengths=self.act_q,
                  actual_seq_lengths_kv=self.act_kv,
                  num_heads=N1, scale=self.scale, pre_tokens=2147483647, next_tokens=0,
                  num_key_value_heads=N2, sparse_mode=2)


def main():
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    torch.npu.set_device(0)
    dev = "npu:0"
    torch.manual_seed(42)

    # 输入布局: q/k/v全长T行, 前P行为prefix段; prefix KV位于key/value头部(P=act[0])
    q_full = (torch.randn(T, N1, D) * 0.1).to(torch.float16)
    k_full = (torch.randn(T, N2, D) * 0.1).to(torch.float16)
    v_full = (torch.randn(T, N2, D) * 0.1).to(torch.float16)
    q = q_full.to(dev)
    k = k_full.to(dev)
    v = v_full.to(dev)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).to(dev)

    act = np.cumsum([P] + [OWN] * B).tolist()  # act_q与act_kv相同: cumsum([P, L0, ...])
    act_q = act
    act_kv = act
    scale = 1.0 / math.sqrt(D)
    m = PiaModule(act_q, act_kv, scale).to(dev)

    ok = True

    # 1. eager .default vs cpu golden
    golden = cpu_golden(q_full, k_full, v_full, P, [OWN] * B, N1, N2, scale).to(dev)
    with torch.no_grad():
        out_eager = m(q, k, v, mask)
    d1 = (out_eager.float() - golden.float()).abs().max().item()
    logger.info(f"[1] eager .default vs golden : max_diff={d1:.6e}")
    ok &= d1 < 2e-3

    # 2. eager .tensor vs .default (bit级)
    with torch.no_grad():
        out_tensor = OP.tensor(q, k, v, atten_mask=mask,
                               actual_seq_lengths=torch.tensor(act_q, dtype=torch.int64, device=dev),
                               actual_seq_lengths_kv=torch.tensor(act_kv, dtype=torch.int64, device=dev),
                               num_heads=N1, scale=scale, pre_tokens=2147483647, next_tokens=0,
                               num_key_value_heads=N2, sparse_mode=2)
    d2 = (out_tensor.float() - out_eager.float()).abs().max().item()
    logger.info(f"[2] eager .tensor  vs .default: max_diff={d2:.6e}")
    ok &= d2 == 0.0

    # 3. GE整图(max-autotune) 多次执行(图复用) vs eager
    cfg2 = CompilerConfig()
    cfg2.mode = "max-autotune"
    backend2 = torchair.get_npu_backend(compiler_config=cfg2)
    m_ge = PiaModule(act_q, act_kv, scale).to(dev)
    cm2 = torch.compile(m_ge, backend=backend2, dynamic=False)
    with torch.no_grad():
        # GE(max-autotune)输出为GE静态内存张量, 后续copy型操作(.float()等)会触发
        # torch层ERR00007 overlap检查报错 — 取出后先clone()转普通torch张量
        out_ge = cm2(q, k, v, mask).clone()
        out_ge2 = cm2(q, k, v, mask).clone()  # 图复用
    d3 = (out_ge.float() - out_eager.float()).abs().max().item()
    d3b = (out_ge2.float() - out_eager.float()).abs().max().item()
    logger.info(f"[3] GE整图max-autotune(含图复用) vs eager : max_diff={d3:.6e}/{d3b:.6e}")
    ok &= d3 == 0.0 and d3b == 0.0

    logger.info("===== 总体结果: " + ("PASS" if ok else "FAIL") + " =====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
