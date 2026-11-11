"""StartLux dynamic vision tower: self-owned nn.Module tree (checkpoint-mirroring
names, model.visual.*) for the two-stage split AIR (vision stage + text stage).

forward(pixel_values [T_v, 1536] f32, vision_pos_embeds [T_v, 768] f32,
        vision_rotary_cos [T_v, 64] f32, vision_rotary_sin [T_v, 64] f32,
        vision_cu_seqlens [n] i64) -> merged features [T_m, 1024] f32

T_v is the sum of every image's t*h*w patch rows and n the image count of the
call -- ANY count and ANY even grid inside the official smart_resize bounds
(PATCH_ROW_BOUNDS below).  The shape-derived tables (pos-embed interpolation,
rotary cos/sin, per-image boundaries) are computed per call on the CPU by the
OFFICIAL fast_pos_embed_interpolate / rot_pos_emb methods and enter the graph
as INPUTS (multimodal.vision_tables_provider) -- the same pattern as the text
side's position_ids; the graph carries no interpolation math and no per-slot
buffers.

Attention: the official per-image non-causal full attention is ONE
npu_fused_infer_attention_score varlen call (TND layout, cumulative
cu_seqlens on both q/kv sides, no mask, no S^2).  FIA takes fp16/bf16 only, so
the attention inner math runs fp16 (q/k/v cast after the OFFICIAL fp32 rotary;
output cast back to the fp32 residual stream) -- the one accepted precision
deviation, gated against the CPU FP32 official reference.  Graph leg via
torchair's `air` op, eager leg via the torch_npu host op (is_compiling
dispatch); FIA rejects non-contiguous inputs (v is an unbind view).

The merger's view performs the 2x2 spatial merge in row-major order, so the
output rows follow the input order exactly: image k of the packed pixel rows
maps to its own [T_v_k/4, out_hidden] feature block.
"""
from __future__ import annotations

import torch
import torch_npu
import torchair
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

# Official image policy (vendored startlux_decide/images.py: MAX_IMAGES=16,
# MIN_PIXELS=65536, MAX_PIXELS=1<<20 with 16x16 patches => 256..4096 patch
# rows per image, h/w even for the 2x2 spatial merge).
MAX_IMAGES_PER_REQUEST = 16
MIN_PATCH_ROWS = 256
MAX_PATCH_ROWS = 4096


def _fp16_linear(module, x):
    # fp32 residual stream cast to fp16 for the matmul, cast back -- only the
    # inner product is reduced
    return F.linear(x.to(torch.float16), module.weight, module.bias).to(x.dtype)


class MatMulPatchEmbed(nn.Module):
    # The official patch Conv3d (stride == kernel, no receptive-field overlap)
    # as ONE per-patch MatMul: out[p, o] = x_flat[p, :] . W_flat[:, o] + b[o]
    # exactly; proj keeps the checkpoint's parameter names.
    def __init__(self, config):
        super().__init__()
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(config.in_channels, config.hidden_size,
                              kernel_size=kernel, stride=kernel, bias=True,
                              dtype=torch.float16)
        self.embed_dim = int(config.hidden_size)
        self.patch_volume = int(config.in_channels * kernel[0] * kernel[1] * kernel[2])

    def forward(self, x):
        w = self.proj.weight.permute(1, 2, 3, 4, 0).reshape(self.patch_volume, -1)
        y = x.to(torch.float16).view(-1, self.patch_volume) @ w
        if self.proj.bias is not None:
            y = y + self.proj.bias
        return y.to(x.dtype).view(-1, self.embed_dim)


class StartLuxVisionAttention(nn.Module):
    # Qwen3_5VisionAttention: per-image non-causal full attention as ONE FIA
    # varlen call; num_key_value_groups kept for nothing (no GQA: 12/12 heads).
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True,
                            dtype=torch.float16)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True,
                             dtype=torch.float16)

    def forward(self, x, rotary_cos, rotary_sin, cu_seqlens):
        seq_length = x.shape[0]
        qkv = _fp16_linear(self.qkv, x)
        q, k, v = (qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
                   .permute(1, 0, 2, 3).unbind(0))
        # official rotary in fp32 on the TND views (q/k outputs are fresh
        # contiguous tensors; v is an unbind view -> contiguous for FIA)
        q, k = apply_rotary_pos_emb_vision(q, k, rotary_cos, rotary_sin)
        v = v.contiguous()
        fia = (torchair.ops.npu_fused_infer_attention_score
               if torch.compiler.is_compiling()
               else torch_npu.npu_fused_infer_attention_score)
        out = fia(q.to(torch.float16), k.to(torch.float16), v.to(torch.float16),
                  num_heads=self.num_heads, scale=self.scaling,
                  pre_tokens=2147483647, next_tokens=2147483647,
                  input_layout="TND", num_key_value_heads=self.num_heads,
                  sparse_mode=0,
                  actual_seq_lengths=cu_seqlens,
                  actual_seq_lengths_kv=cu_seqlens)[0].to(x.dtype)
        out = out.reshape(seq_length, self.num_heads * self.head_dim)
        return _fp16_linear(self.proj, out)


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

    def forward(self, hidden, rotary_cos, rotary_sin, cu_seqlens):
        hidden = hidden + self.attn(self.norm1(hidden), rotary_cos, rotary_sin,
                                    cu_seqlens)
        return hidden + self.mlp(self.norm2(hidden))


class StartLuxVisionPatchMerger(nn.Module):
    # use_postshuffle_norm=False: norm at patch resolution, then the view IS
    # the 2x2 row-major spatial merge; the act is the official PLAIN nn.GELU
    # (exact, unlike the MLP's gelu_pytorch_tanh).
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True,
                                    dtype=torch.float16)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size,
                                    bias=True, dtype=torch.float16)
        self.act_fn = nn.GELU()

    def forward(self, x):
        y = self.norm(x).view(-1, self.hidden_size)
        return _fp16_linear(self.linear_fc2,
                            self.act_fn(_fp16_linear(self.linear_fc1, y)))


class StartLuxVisionTower(nn.Module):
    # The pos_embed parameter stays for checkpoint compatibility (load_state_dict
    # expects model.visual.pos_embed.weight); the forward never touches it -- the
    # interpolated tables arrive as inputs -- so dynamo drops it from the graph
    # and the CPU bridge reads the tensor straight from the checkpoint.
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = MatMulPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings,
                                       config.hidden_size)
        self.blocks = nn.ModuleList([StartLuxVisionBlock(config)
                                      for _ in range(config.depth)])
        self.merger = StartLuxVisionPatchMerger(config)

    def forward(self, pixel_values, pos_embeds, rotary_cos, rotary_sin, cu_seqlens):
        hidden = self.patch_embed(pixel_values) + pos_embeds
        for block in self.blocks:
            hidden = block(hidden, rotary_cos, rotary_sin, cu_seqlens)
        return self.merger(hidden)
