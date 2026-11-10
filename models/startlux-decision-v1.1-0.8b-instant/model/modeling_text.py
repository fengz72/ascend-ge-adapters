"""StartLux text decoder: self-owned nn.Module tree whose parameter names mirror
the checkpoint (model.language_model.*, zero-mapping load_state_dict), single-leg
NPU fused ops.  dtype is declared per site like vLLM's qwen_gdn_linear_attn: A_log
is explicitly float32, everything else follows the construction default (bf16
under from_folder).  Numerical why-notes live in validation.md 2026-11-10.
"""
from __future__ import annotations

import torch
import torch_npu
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN


def _softplus(x):
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(torch.clamp(x, max=20.0))))


def causal_conv4(x, weight):
    channels, seq = x.shape[1], x.shape[2]
    w = weight.float().reshape(channels, 1, 1, 4)
    y = F.conv2d(x.float().unsqueeze(2), w, groups=channels, padding=(0, 3))
    return F.silu(y[..., :seq].squeeze(2).to(x.dtype))


def _mrope_interleave_index(mrope_section, half, device=None):
    source = [0] * half
    for dim in (1, 2):
        length = mrope_section[dim] * 3
        for j in range(dim, min(length, half), 3):
            source[j] = dim
    return torch.tensor([source[j] * half + j for j in range(half)],
                        dtype=torch.int64, device=device)


class StartLuxRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        gamma = 1.0 + self.weight
        y, _ = torch.ops.npu.npu_rms_norm(x, gamma, float(self.eps))
        return y.to(x.dtype)


class StartLuxRMSNormGated(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x, gate):
        ones = torch.ones_like(self.weight)
        n, _ = torch.ops.npu.npu_rms_norm(x, ones, float(self.variance_epsilon))
        return (self.weight * n.to(x.dtype) * F.silu(gate.float())).to(x.dtype)


class StartLuxMLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class StartLuxGatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.layer_idx = layer_idx
        if self.num_v_heads != self.num_k_heads:
            raise NotImplementedError(
                "native chunk GDR requires num_k_heads == num_v_heads "
                f"(got {self.num_k_heads}/{self.num_v_heads})")
        # vLLM's declaration: A_log explicitly float32, dt_bias follows the
        # model dtype (the checkpoint stores dt_bias BF16 and A_log F32).
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.norm = StartLuxRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size,
                                     self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(self.key_dim * 2 + self.value_dim,
                                self.key_dim * 2 + self.value_dim, bias=False,
                                kernel_size=config.linear_conv_kernel_dim,
                                groups=self.key_dim * 2 + self.value_dim,
                                padding=config.linear_conv_kernel_dim - 1)
        # Constant so it constant-folds once in the exported graph and follows
        # .to(device) like every buffer; bf16 is the kernel's gamma dtype.
        self.register_buffer("l2_gamma", torch.full(
            (self.head_k_dim,), self.head_k_dim ** -0.5, dtype=torch.bfloat16),
            persistent=False)

    def forward(self, hidden_states):
        batch, width, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch, width, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        mixed = causal_conv4(mixed_qkv, self.conv1d.weight).transpose(1, 2)
        q = mixed[..., :self.key_dim].reshape(batch, width, -1, self.head_k_dim)
        k = mixed[..., self.key_dim:2 * self.key_dim].reshape(
            batch, width, -1, self.head_k_dim)
        v = mixed[..., 2 * self.key_dim:].reshape(batch, width, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * _softplus(a.float() + self.dt_bias)
        # L2-norm as ONE fused RmsNorm per side (the native kernel has no
        # use_qk_l2norm_in_kernel flag): rms_norm(x, gamma=1/sqrt(D), eps/D)
        # == l2norm(x, eps) EXACTLY.  The kernel scale stays Dk**-0.5 UNCHANGED
        # -- the sqrt(D) folding belongs in gamma, never in scale.
        q, _ = torch.ops.npu.npu_rms_norm(q, self.l2_gamma, 1e-6 / self.head_k_dim)
        k, _ = torch.ops.npu.npu_rms_norm(k, self.l2_gamma, 1e-6 / self.head_k_dim)
        q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)
        beta, g = beta.flatten(0, 1), g.flatten(0, 1)
        # Zero initial state, per-row PHYSICAL width S as actual_seq_lengths
        # (never cumulative ends, never the true L); final state discarded.
        state = torch.zeros(batch, self.num_v_heads, self.head_v_dim, self.head_k_dim,
                            dtype=torch.bfloat16, device=hidden_states.device)
        row_widths = torch.full((batch,), width, dtype=torch.int32,
                                device=hidden_states.device)
        out, _final_state = torch.ops.npu.npu_chunk_gated_delta_rule(
            q, k, v, beta=beta, initial_state=state,
            actual_seq_lengths=row_widths,
            scale=float(self.head_k_dim) ** -0.5, g=g)
        out = out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm(out, z).reshape(batch, width, -1)
        return self.out_proj(out)


class StartLuxAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.head_dim = getattr(config, "head_dim", None) \
            or config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.rotary_dim = int(self.head_dim
                               * config.rope_parameters["partial_rotary_factor"])
        self.q_proj = nn.Linear(config.hidden_size,
                                self.num_heads * self.head_dim * 2,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size,
                                self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size,
                                self.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size,
                                bias=config.attention_bias)
        self.q_norm = StartLuxRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = StartLuxRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden, cos, sin, pad_mask):
        batch, width, _ = hidden.shape
        qkv = self.q_proj(hidden).view(batch, width, -1, self.head_dim * 2)
        # explicit slices: aten.chunk / aten.split.sizes have no GE converter
        q, gate = qkv[..., :self.head_dim], qkv[..., self.head_dim:]
        gate = gate.reshape(batch, width, -1)
        q2d = self.q_norm(q.reshape(batch, width, -1, self.head_dim))
        k2d = self.k_norm(self.k_proj(hidden).view(batch, width, -1, self.head_dim))
        v = self.v_proj(hidden).view(batch, width, -1, self.head_dim).transpose(1, 2)
        # Partial RoPE via npu_apply_rotary_pos_emb (BSND, "half" == exact
        # rotate-half math); both legs verified bit-identical vs pure math
        # (2026-11-10 probe).
        q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(
            q2d[..., :self.rotary_dim], k2d[..., :self.rotary_dim],
            cos.unsqueeze(2), sin.unsqueeze(2),
            layout="BSND", rotary_mode="half")
        q2d = torch.cat((q_rot, q2d[..., self.rotary_dim:]), dim=-1)
        k2d = torch.cat((k_rot, k2d[..., self.rotary_dim:]), dim=-1)
        # One fused attention: bf16 BNSD, bool [B, 1, S, S] pad mask (True =
        # blocked); FA runs the GQA expansion natively.
        out = torch_npu.npu_fusion_attention(
            q2d.transpose(1, 2), k2d.transpose(1, 2), v, self.num_heads, "BNSD",
            atten_mask=pad_mask, scale=float(self.scaling), keep_prob=1.0,
            pre_tockens=2147483647, next_tockens=2147483647, sparse_mode=0)[0]
        out = out.transpose(1, 2).contiguous().reshape(batch, width, -1)
        return self.o_proj(out * torch.sigmoid(gate))


class StartLuxDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = StartLuxGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = StartLuxAttention(config, layer_idx)
        else:
            raise ValueError(f"unexpected layer type: {self.layer_type!r}")
        self.mlp = StartLuxMLP(config, config.intermediate_size)
        self.input_layernorm = StartLuxRMSNorm(config.hidden_size,
                                               eps=config.rms_norm_eps)
        self.post_attention_layernorm = StartLuxRMSNorm(config.hidden_size,
                                                        eps=config.rms_norm_eps)

    def forward(self, hidden, cos, sin, pad_mask):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        if self.layer_type == "linear_attention":
            hidden = self.linear_attn(hidden)
        else:
            hidden = self.self_attn(hidden, cos, sin, pad_mask)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        return residual + self.mlp(hidden)


class StartLuxTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         getattr(config, "pad_token_id", None))
        self.layers = nn.ModuleList(
            [StartLuxDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = StartLuxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rope = config.rope_parameters
        if rope.get("rope_type") != "default":
            raise ValueError("only default rope is supported")
        head_dim = getattr(config, "head_dim", None) \
            or config.hidden_size // config.num_attention_heads
        self.rotary_dim = int(head_dim * rope["partial_rotary_factor"])
        self.mrope_section = [int(v) for v in
                              rope.get("mrope_section", [self.rotary_dim // 2])]
        if sum(self.mrope_section) != self.rotary_dim // 2:
            raise ValueError("mrope_section does not match the rotary dim")
        inv_freq = 1.0 / (float(rope["rope_theta"]) ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("mrope_gather", _mrope_interleave_index(
            self.mrope_section, self.rotary_dim // 2), persistent=False)

    def mrope_cos_sin(self, position_ids, dtype):
        # bit-exact against Qwen3_5TextRotaryEmbedding
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq
        freqs = freqs.permute(1, 2, 0, 3).flatten(2)
        freqs = freqs.index_select(-1, self.mrope_gather)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, inputs_embeds, position_ids, actual_seq_lengths):
        batch, width = inputs_embeds.shape[0], inputs_embeds.shape[1]
        cos, sin = self.mrope_cos_sin(position_ids, inputs_embeds.dtype)
        valid = torch.arange(width, device=inputs_embeds.device).unsqueeze(0) \
            < actual_seq_lengths.unsqueeze(1)
        pad_mask = (~valid).view(batch, 1, 1, width).expand(batch, 1, width, width)
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, pad_mask)
        return self.norm(hidden)
