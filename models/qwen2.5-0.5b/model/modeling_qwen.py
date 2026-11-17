"""NPU-executable Qwen2.5-0.5B packed trunk and 8-column score readout.

``QwenPackedModel.forward(input_ids, position_ids, actual_seq_lengths)`` runs
embed -> 24 decoder layers -> the per-request last-token readout and returns
the float32 [N, 8] scores.  The inputs are the shared-prefix packed layout
(``segments = [prefix] + requests``, ``actual_seq_lengths = cumsum(segment
lengths)`` with N+1 entries), so the prefix Q/KV is stored and computed once
and the business rows end at ``actual_seq_lengths[1:] - 1``.  The platform
prefix operator is bound at construction: ``a3`` (AscendC
PrefixInferAttentionScore with the compressed causal mask, int64 bounds) or
``a5`` (CANNBotDSL PrefixAttention, int32 bounds).  Every host op is torch_npu;
the independent CPU reference is model/golden.py.
"""

import torch
import torch_npu
from torch import nn
from torch.nn.functional import silu

# RoPE table depth and the compressed causal mask width; the A3 operator
# requires prefix + longest request <= this.
MAX_SEQ_LEN = 2048


class QwenRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # npu_rms_norm already returns the input dtype (only rstd is float32).
        return torch.ops.npu.npu_rms_norm(x, self.weight, self.eps)[0]


class QwenMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(silu(self.gate_proj(x)) * self.up_proj(x))


class QwenAttention(nn.Module):
    def __init__(self, config, head_dim, attention):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = config["num_attention_heads"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.scaling = head_dim**-0.5
        hidden_size = config["hidden_size"]
        # Qwen2 projects q/k/v with bias and o without.
        self.q_proj = nn.Linear(hidden_size, self.num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * head_dim, hidden_size, bias=False)
        self.attention = attention

    def forward(self, hidden, cos, sin, bounds):
        # -1 for the token axis: an explicit symbolic size makes dynamo lift a
        # SymInt chain into extra graph inputs and emit Pack nodes for the
        # reshape, which GE cannot infer under dynamic shapes.
        q = self.q_proj(hidden).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(-1, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden).view(-1, self.num_key_value_heads, self.head_dim)
        q, k = torch_npu.npu_apply_rotary_pos_emb(
            q, k, cos, sin, layout="TND", rotary_mode="half"
        )
        out = self.attention(q, k, v, bounds, self.scaling)
        return self.o_proj(out.reshape(-1, self.num_heads * self.head_dim))


class QwenDecoderLayer(nn.Module):
    def __init__(self, config, head_dim, attention):
        super().__init__()
        self.self_attn = QwenAttention(config, head_dim, attention)
        self.mlp = QwenMLP(config["hidden_size"], config["intermediate_size"])
        eps = config["rms_norm_eps"]
        self.input_layernorm = QwenRMSNorm(config["hidden_size"], eps)
        self.post_attention_layernorm = QwenRMSNorm(config["hidden_size"], eps)

    def forward(self, hidden, cos, sin, bounds):
        residual = hidden
        hidden = residual + self.self_attn(self.input_layernorm(hidden), cos, sin, bounds)
        residual = hidden
        return residual + self.mlp(self.post_attention_layernorm(hidden))


def prefix_910_attention(config, model):
    """A3 AscendC PrefixInferAttentionScore over the compressed causal mask.

    ``sparse_mode=2`` reads the fixed [2048, 2048] upper-triangular template;
    ``pre_tokens=INT_MAX / next_tokens=0`` makes it causal.  The ``.tensor``
    overload takes the segment bounds as a graph input, so they stay dynamic.
    """
    import npu_prefix_infer_attention_score  # noqa: F401  schema + torchair converter

    op = torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score.tensor
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    # Only this recipe reads the template, so only this platform builds it.
    model.register_buffer(
        "compressed_mask",
        torch.triu(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool), diagonal=1),
        persistent=False,
    )

    def attention(q, k, v, bounds, scale):
        # model.compressed_mask is read LAZILY on purpose: nn.Module.to()
        # rebinds _buffers to fresh tensors, so a reference captured here at
        # construction time would keep pointing at the pre-move CPU copy.
        return op(
            q,
            k,
            v,
            atten_mask=model.compressed_mask,
            actual_seq_lengths=bounds,
            actual_seq_lengths_kv=bounds,
            num_heads=heads,
            scale=scale,
            num_key_value_heads=kv_heads,
            pre_tokens=2147483647,
            next_tokens=0,
            sparse_mode=2,
        )

    return attention


def prefix_a5_attention(config, model):
    """A5 CANNBotDSL PrefixAttention (fp16 GE deliverable, int32 bounds)."""
    import prefix_attn_torch  # noqa: F401  schema + torchair converter

    op = torch.ops.ascendc_ops.prefix_attention
    return lambda q, k, v, bounds, scale: op(q, k, v, bounds, scale)


ATTENTION = {"a3": prefix_910_attention, "a5": prefix_a5_attention}


class QwenPackedModel(nn.Module):
    def __init__(self, config, platform):
        super().__init__()
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        inv_freq = 1.0 / (
            config["rope_theta"]
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        freqs = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)[:, None] * inv_freq[None, :]
        freqs = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", freqs.cos().to(torch.float16), persistent=False)
        self.register_buffer("sin", freqs.sin().to(torch.float16), persistent=False)
        # Bound once for the whole stack; the a3 factory also registers the
        # compressed causal mask that only its recipe reads.
        attention = ATTENTION[platform](config, self)
        self.layers = nn.ModuleList(
            [
                QwenDecoderLayer(config, head_dim, attention)
                for _ in range(config["num_hidden_layers"])
            ]
        )
        self.norm = QwenRMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def install_score_head(self, token_ids):
        """Derive the score columns from the tied embedding table.

        Call once, after ``load_state_dict`` and ``requires_grad_(False)``:
        Qwen2.5-0.5B ties lm_head to embed_tokens, so the checkpoint carries no
        head tensor.
        """
        rows = self.embed_tokens.weight.index_select(
            0,
            torch.tensor(token_ids, device=self.embed_tokens.weight.device),
        ).to(torch.float32)
        self.register_buffer("score_head", rows, persistent=False)

    def forward(self, input_ids, position_ids, actual_seq_lengths):
        hidden = self.embed_tokens(input_ids)
        cos = self.cos.index_select(0, position_ids).unsqueeze(1)
        sin = self.sin.index_select(0, position_ids).unsqueeze(1)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, actual_seq_lengths)
        ends = actual_seq_lengths[1:].to(torch.int64)
        hidden = self.norm(hidden.index_select(0, ends - 1))
        return hidden.float() @ self.score_head.transpose(0, 1)
