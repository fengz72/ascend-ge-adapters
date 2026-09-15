"""
自定义 NPU 推理 attention 函数 — npu_fused_infer_attention_score (varlen)

通过 ALL_ATTENTION_FUNCTIONS.register 注入, 不修改 transformers 模型代码。
推理算子替代训练算子 npu_fusion_attention, 支持 GQA + block-diagonal causal mask。
"""

import torch
import torch_npu
import npu_prefix_infer_attention_score  # noqa: F401  注册 schema + eager + torchair converter
from torchair.ops import npu_fused_infer_attention_score as _torchair_fia
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

_PREFIX_IA_OP = torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score


def npu_fia_varlen_forward(module, query, key, value, attention_mask,
                           dropout=0.0, scaling=1.0, sliding_window=None, **kwargs):
    """使用 npu_fused_infer_attention_score (推理算子) 的 varlen attention。

    从 Qwen2Attention.forward 接收 BNSD [B, N, S, D] 格式的 q/k/v,
    转换为 TND [S, N, D] 调用推理算子, 返回 BSND [B, S, N, D]。

    GQA 通过 num_key_value_heads 参数原生支持, 无需手动 repeat_kv。
    因果掩码通过 sparse_mode=2 (压缩 causal) + 2048x2048 优化 mask 实现,
    算子根据 actual_seq_lengths 自动生成 block-diagonal causal mask。

    注: sparse_mode=3 与 2 语义等价 (逐 batch 全左对齐 causal), 但 GE 图编译下
    mode=3 会选到慢的 MIX tiling 分支 (attention kernel ~2x 耗时), mode=2 落到
    纯 AIC 快分支 — 见 docs/prefix-attention/benchmark.md 公平基线复测。

    动态 shape 策略:
    - 图编译模式: torchair.ops.npu_fused_infer_attention_score (Tensor),
      actual_seq_lengths 以 tensor 传入, 避免 dynamo specialize, 配合 dynamic=True
    - eager 模式: torch_npu.npu_fused_infer_attention_score (支持 eager 执行)
    """
    n = int(query.shape[1])
    n_kv = int(key.shape[1])
    if query.dim() == 4:
        q_t = query.permute(0, 2, 1, 3).squeeze(0).contiguous()
        k_t = key.permute(0, 2, 1, 3).squeeze(0).contiguous()
        v_t = value.permute(0, 2, 1, 3).squeeze(0).contiguous()
    else:
        q_t = query.contiguous()
        k_t = key.contiguous()
        v_t = value.contiguous()

    fia_kwargs = dict(
        num_heads=n,
        input_layout="TND",
        scale=scaling,
        actual_seq_lengths=module.actual_seq_lengths_tensor,
        actual_seq_lengths_kv=module.actual_seq_lengths_tensor,
        num_key_value_heads=n_kv,
        atten_mask=module.atten_mask,
        sparse_mode=2,
    )

    if torch.compiler.is_compiling():
        out, _ = _torchair_fia(q_t, k_t, v_t, **fia_kwargs)
    else:
        out, _ = torch_npu.npu_fused_infer_attention_score(q_t, k_t, v_t, **fia_kwargs)

    if query.dim() == 4:
        out = out.unsqueeze(0).contiguous()
    return out, None


def register_npu_fia():
    """注册 npu_fia attention 函数到 transformers 全局注册表。"""
    ALL_ATTENTION_FUNCTIONS.register("npu_fia", npu_fia_varlen_forward)


def prefix_ia_varlen_forward(module, query, key, value, attention_mask,
                             dropout=0.0, scaling=1.0, sliding_window=None, **kwargs):
    """packed prefix-in-Q varlen attention — npu_prefix_infer_attention_score (KV 内嵌版)。

    q/k/v 均为全长 TND [T', N, D], T' = P + sum(L_i), 行布局 [prefix, req0, req1, ...]，
    prefix KV 位于 key/value 头部 (由算子内部处理, 无需切分)。
    act 契约 (新算子): act_q ≡ act_kv = cumsum([P, L0, L1, ...]) — prefix 独立成
    batch 0, P = act[0]; 请求 i 为 batch i+1。

    数学: 与"每请求展开为 [prefix + own] 的独立 causal attention"逐位等价
    (causal 下各 batch 的 prefix 段输出相同, 只算一份), 供与 FIA 基线互证。
    """
    if query.dim() != 3:
        raise ValueError(f"prefix_ia 需要 3D TND 输入, got {query.dim()}D")

    out = _PREFIX_IA_OP.tensor(
        query, key, value,
        atten_mask=module.atten_mask,
        actual_seq_lengths=module.act_tensor,
        actual_seq_lengths_kv=module.act_tensor,
        num_heads=int(query.shape[1]),
        scale=scaling,
        num_key_value_heads=int(key.shape[1]),
        pre_tokens=2147483647,
        next_tokens=0,
        sparse_mode=2,
    )
    return out, None


def register_prefix_ia():
    """注册 prefix_ia attention 函数到 transformers 全局注册表。"""
    ALL_ATTENTION_FUNCTIONS.register("prefix_ia", prefix_ia_varlen_forward)


def build_causal_mask_2048(device):
    """构建 2048x2048 上三角因果掩码 (bool), 用于 sparse_mode=2 压缩 causal。"""
    return torch.triu(
        torch.ones(2048, 2048, dtype=torch.bool, device=device), diagonal=1
    )
