"""NPU-executable StartLux-Decision text trunk and 26-column readout.

``StartLuxTextModel.forward(input_ids, position_ids)`` runs embed -> hybrid
decoder stack (Gated DeltaNet + bidirectional full attention) -> last-token
readout and returns the float32 [B, 26] scores.  Every host op is torch_npu;
the independent CPU reference is model/golden.py.
"""

import torch
import torch.nn.functional as F
import torch_npu
import torchair
from torch import nn
from transformers.activations import ACT2FN


def _softplus(x):
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(torch.clamp(x, max=20.0))))


def _causal_depthwise_conv(x, weight):
    _, width, channels = x.shape
    kernel = weight.shape[2]
    w = weight.float().reshape(channels, 1, 1, kernel)
    y = F.conv2d(
        x.transpose(1, 2).float().unsqueeze(2),
        w,
        groups=channels,
        padding=(0, kernel - 1),
    )
    return F.silu(y[..., :width].squeeze(2).to(x.dtype).transpose(1, 2))


def _mrope_interleave_index(mrope_section, half):
    source = [0] * half
    for dim in (1, 2):
        stop = mrope_section[dim] * 3
        for j in range(dim, min(stop, half), 3):
            source[j] = dim
    return torch.tensor([source[j] * half + j for j in range(half)], dtype=torch.int64)


class StartLuxRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.register_load_state_dict_post_hook(StartLuxRMSNorm._fold_gamma)

    def forward(self, x):
        y, _ = torch.ops.npu.npu_rms_norm(x, self.weight, self.eps)
        return y.to(x.dtype)

    @staticmethod
    def _fold_gamma(module, incompatible_keys):
        # The checkpoint stores a multiplicative gamma as a zero-based offset
        # (weight=0 means identity); fold it into the nn.Parameter convention
        # (weight=1 means identity) right after state loading.
        with torch.no_grad():
            module.weight.data.add_(1.0)


class StartLuxRMSNormGated(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        n, _ = torch.ops.npu.npu_rms_norm(x, self.weight, self.eps)
        return (n * F.silu(gate.float())).to(x.dtype)


class StartLuxMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class StartLuxGatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.scaling = self.head_k_dim**-0.5
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.norm = StartLuxRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        # Checkpoint-shaped only: the real computation is
        # _causal_depthwise_conv(F.conv2d); only conv1d.weight's name/shape is
        # consumed, and groups fixes that depthwise weight shape.
        self.conv1d = nn.Conv1d(
            self.key_dim * 2 + self.value_dim,
            self.key_dim * 2 + self.value_dim,
            bias=False,
            kernel_size=config.linear_conv_kernel_dim,
            groups=self.key_dim * 2 + self.value_dim,
        )
        self.register_buffer(
            "l2_gamma",
            torch.full((self.head_k_dim,), self.head_k_dim**-0.5, dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(self, hidden, zero_state, row_widths):
        batch, width, _ = hidden.shape
        mixed_qkv = self.in_proj_qkv(hidden)
        z = self.in_proj_z(hidden).reshape(batch, width, -1, self.head_v_dim)
        b = self.in_proj_b(hidden)
        a = self.in_proj_a(hidden)
        mixed = _causal_depthwise_conv(mixed_qkv, self.conv1d.weight)
        q = mixed[..., : self.key_dim].reshape(batch, width, -1, self.head_k_dim)
        k = mixed[..., self.key_dim : 2 * self.key_dim].reshape(
            batch, width, -1, self.head_k_dim
        )
        v = mixed[..., 2 * self.key_dim :].reshape(batch, width, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * _softplus(a.float() + self.dt_bias)
        q, _ = torch.ops.npu.npu_rms_norm(q, self.l2_gamma, 1e-6 / self.head_k_dim)
        k, _ = torch.ops.npu.npu_rms_norm(k, self.l2_gamma, 1e-6 / self.head_k_dim)
        q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)
        beta, g = beta.flatten(0, 1), g.flatten(0, 1)
        out, _final_state = torch.ops.npu.npu_chunk_gated_delta_rule(
            q,
            k,
            v,
            beta=beta,
            initial_state=zero_state,
            actual_seq_lengths=row_widths,
            scale=self.scaling,
            g=g,
        )
        out = out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm(out, z).reshape(batch, width, -1)
        return self.out_proj(out)


class StartLuxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rotary_dim = int(
            self.head_dim * config.rope_parameters["partial_rotary_factor"]
        )
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = StartLuxRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = StartLuxRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden, cos, sin):
        batch, width, _ = hidden.shape
        q_gate = self.q_proj(hidden).view(batch, width, -1, self.head_dim * 2)
        q, gate = q_gate[..., : self.head_dim], q_gate[..., self.head_dim :]
        gate = gate.reshape(batch, width, -1)
        q_heads = self.q_norm(q.reshape(batch, width, -1, self.head_dim))
        k_heads = self.k_norm(self.k_proj(hidden).view(batch, width, -1, self.head_dim))
        v = self.v_proj(hidden).view(batch, width, -1, self.head_dim)
        q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(
            q_heads[..., : self.rotary_dim],
            k_heads[..., : self.rotary_dim],
            cos.unsqueeze(2),
            sin.unsqueeze(2),
            layout="BSND",
            rotary_mode="half",
        )
        q_heads = torch.cat((q_rot, q_heads[..., self.rotary_dim :]), dim=-1)
        k_heads = torch.cat((k_rot, k_heads[..., self.rotary_dim :]), dim=-1)
        fia = (
            torchair.ops.npu_fused_infer_attention_score
            if torch.compiler.is_compiling()
            else torch_npu.npu_fused_infer_attention_score
        )
        out = fia(
            q_heads,
            k_heads,
            v,
            num_heads=self.num_heads,
            scale=float(self.scaling),
            pre_tokens=2147483647,
            next_tokens=2147483647,
            input_layout="BSND",
            num_key_value_heads=self.num_key_value_heads,
            sparse_mode=0,
        )[0]
        out = out.reshape(batch, width, -1)
        return self.o_proj(out * torch.sigmoid(gate))


class StartLuxDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = StartLuxGatedDeltaNet(config)
        elif self.layer_type == "full_attention":
            self.self_attn = StartLuxAttention(config)
        self.mlp = StartLuxMLP(config)
        self.input_layernorm = StartLuxRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = StartLuxRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden, cos, sin, row_widths, zero_state):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        if self.layer_type == "linear_attention":
            hidden = self.linear_attn(hidden, zero_state, row_widths)
        else:
            hidden = self.self_attn(hidden, cos, sin)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        return residual + self.mlp(hidden)


class StartLuxTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, getattr(config, "pad_token_id", None)
        )
        self.layers = nn.ModuleList(
            [StartLuxDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = StartLuxRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rope = config.rope_parameters
        rotary_dim = int(config.head_dim * rope["partial_rotary_factor"])
        mrope_section = [int(v) for v in rope["mrope_section"]]
        inv_freq = 1.0 / (
            float(rope["rope_theta"])
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer(
            "mrope_gather",
            _mrope_interleave_index(mrope_section, rotary_dim // 2),
            persistent=False,
        )
        self.gdn_state_shape = (
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            config.linear_key_head_dim,
        )

    def mrope_cos_sin(self, position_ids, dtype):
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq
        freqs = freqs.permute(1, 2, 0, 3).flatten(2)
        freqs = freqs.index_select(-1, self.mrope_gather)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def install_decision_head(self, letter_token_ids):
        """Derive the 26-column readout rows from the tied embedding table.

        Call once, after ``load_state_dict`` and before ``forward``: it
        registers ``decision_head`` from the loaded ``embed_tokens`` weight.
        """
        with torch.no_grad():
            rows = self.embed_tokens.weight.index_select(
                0,
                torch.tensor(letter_token_ids, device=self.embed_tokens.weight.device),
            ).to(torch.float32)
        self.register_buffer("decision_head", rows, persistent=False)

    def forward(self, input_ids, position_ids):
        hidden = self.embed_tokens(input_ids)
        batch, width, hidden_dim = hidden.shape
        # ChunkGatedDeltaRule requires actual_seq_lengths (int32); under the
        # B=1 true-length contract the laid-out width is the true length.
        row_widths = torch.full(
            (batch,), width, dtype=torch.int32, device=hidden.device
        )
        cos, sin = self.mrope_cos_sin(position_ids, hidden.dtype)
        zero_state = torch.zeros(
            batch,
            *self.gdn_state_shape,
            dtype=torch.bfloat16,
            device=hidden.device,
        )
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, row_widths, zero_state)
        hidden = self.norm(hidden)
        last_index = torch.full(
            (batch, 1, 1), width - 1, dtype=torch.int64, device=hidden.device
        )
        last = hidden.gather(1, last_index.expand(batch, 1, hidden_dim)).squeeze(1)
        return last.float() @ self.decision_head.transpose(0, 1)
