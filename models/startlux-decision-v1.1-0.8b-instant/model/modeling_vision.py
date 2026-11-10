"""StartLux vision tower: self-owned nn.Module tree (checkpoint-mirroring names,
model.visual.*) behind fixed same-grid image slots.  FP32 residual stream with
FP16 Linear/Conv inner products; the shape-derived pos_embeds / rotary tables are
precomputed by the OFFICIAL fast_pos_embed_interpolate / rot_pos_emb methods after
the weight load (build_static_buffers, called by from_folder).  Full record:
validation.md 2026-11-10.
"""
from __future__ import annotations

import types

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionModel, Qwen3_5VisionRotaryEmbedding, apply_rotary_pos_emb_vision,
    eager_attention_forward)

GRID_THW = (1, 16, 16)        # t, h, w of every image slot (256x256 input)
PATCH_ROWS = GRID_THW[1] * GRID_THW[2]            # 256 patch rows per slot
MERGED_ROWS = (GRID_THW[1] // 2) * (GRID_THW[2] // 2)   # 64 merged rows per slot


def _fp16_linear(module, x):
    return F.linear(x.to(torch.float16), module.weight, module.bias).to(x.dtype)


class MatMulPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(config.in_channels, config.hidden_size,
                              kernel_size=kernel, stride=kernel, bias=True,
                              dtype=torch.float16)
        self.embed_dim = int(config.hidden_size)
        self.patch_volume = int(config.in_channels * kernel[0] * kernel[1] * kernel[2])

    def forward(self, x):
        # stride == kernel -> one per-patch MatMul is exact: out[p, o] =
        # x_flat[p, :] . W_flat[:, o] + b[o]
        w = self.proj.weight.permute(1, 2, 3, 4, 0).reshape(self.patch_volume, -1)
        y = x.to(torch.float16).view(-1, self.patch_volume) @ w
        if self.proj.bias is not None:
            y = y + self.proj.bias
        return y.to(x.dtype).view(-1, self.embed_dim)


class StartLuxVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_groups = 1     # required by eager_attention_forward
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = 0.0
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True,
                            dtype=torch.float16)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True,
                             dtype=torch.float16)

    def forward(self, x, rotary_cos, rotary_sin, lengths):
        seq_length = x.shape[0]
        qkv = _fp16_linear(self.qkv, x)
        query_states, key_states, value_states = (
            qkv.reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0))
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, rotary_cos, rotary_sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        # one attention chunk per image (fixed lengths), official eager math
        splits = [torch.split(t, lengths, dim=2)
                  for t in (query_states, key_states, value_states)]
        interface = ALL_ATTENTION_FUNCTIONS.get_interface("eager",
                                                          eager_attention_forward)
        attn_outputs = [interface(self, q, k, v, attention_mask=None,
                                  scaling=self.scaling, dropout=0.0, is_causal=False)[0]
                        for q, k, v in zip(*splits)]
        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return _fp16_linear(self.proj, attn_output)


class StartLuxVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size,
                                    bias=True, dtype=torch.float16)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size,
                                    bias=True, dtype=torch.float16)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return _fp16_linear(self.linear_fc2,
                            self.act_fn(_fp16_linear(self.linear_fc1, x)))


class StartLuxVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = StartLuxVisionAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = StartLuxVisionMLP(config)

    def forward(self, hidden, rotary_cos, rotary_sin, lengths):
        hidden = hidden + self.attn(self.norm1(hidden), rotary_cos, rotary_sin,
                                    lengths)
        return hidden + self.mlp(self.norm2(hidden))


class StartLuxVisionPatchMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True,
                                    dtype=torch.float16)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size,
                                    bias=True, dtype=torch.float16)
        self.act_fn = nn.GELU()     # the official merger uses PLAIN GELU

    def forward(self, x):
        # use_postshuffle_norm=False: norm at patch resolution, then the view IS
        # the 2x2 row-major spatial merge
        y = self.norm(x).view(-1, self.hidden_size)
        return _fp16_linear(self.linear_fc2,
                            self.act_fn(_fp16_linear(self.linear_fc1, y)))


class StartLuxVisionTower(nn.Module):
    def __init__(self, config, grid_thw=GRID_THW, slots=2):
        super().__init__()
        self.config = config
        self.patch_embed = MatMulPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings,
                                       config.hidden_size)
        self.blocks = nn.ModuleList([StartLuxVisionBlock(config)
                                      for _ in range(config.depth)])
        self.merger = StartLuxVisionPatchMerger(config)
        self.grid_thw = tuple(int(v) for v in grid_thw)
        self.slots = int(slots)
        t, h, w = self.grid_thw
        self.lengths = [h * w] * self.slots     # one attention chunk per image
        self.head_dim = config.hidden_size // config.num_heads

    def build_static_buffers(self):
        # inv_freq is NOT stored in the checkpoint and a .to(dtype) pass
        # quantizes it irrecoverably, so it is always regenerated with the
        # official constructor and never cast.
        merge = self.config.spatial_merge_size
        shim = types.SimpleNamespace(
            pos_embed=self.pos_embed,
            num_grid_per_side=int(self.config.num_position_embeddings ** 0.5),
            spatial_merge_size=merge,
            config=types.SimpleNamespace(spatial_merge_size=merge),
            rotary_pos_emb=Qwen3_5VisionRotaryEmbedding(self.head_dim // 2))
        with torch.no_grad():
            grid = torch.tensor([list(self.grid_thw)] * self.slots, dtype=torch.long)
            pos = Qwen3_5VisionModel.fast_pos_embed_interpolate.__get__(shim)(grid)
            rot = Qwen3_5VisionModel.rot_pos_emb.__get__(shim)(grid)
            rot = rot.reshape(-1, rot.shape[-1])            # official reshape
            emb = torch.cat((rot, rot), dim=-1)
            self.register_buffer("pos_embeds", pos.contiguous())
            self.register_buffer("rotary_cos", emb.cos().contiguous())
            self.register_buffer("rotary_sin", emb.sin().contiguous())

    def forward(self, pixel_values):
        hidden = self.patch_embed(pixel_values) + self.pos_embeds
        for block in self.blocks:
            hidden = block(hidden, self.rotary_cos, self.rotary_sin, self.lengths)
        return self.merger(hidden)
