"""Fixed-grid vision tower wrapper for the StartLux unified multimodal AIR.

Holds the ORIGINAL HF visual modules (blocks, merger) of the pinned model
folder behind a fixed number of same-grid image slots:

- the patch_embed conv is replaced by the bit-exact equivalent
  MatMulPatchEmbed (stride == kernel, see its docstring): the Conv3DV2 node is
  MTE1-saturated and MAC-starved at this shape, while the equivalent MatMulV2
  carries the same bytes with proper MAC overlap;
- every shape-derived input of Qwen3_5VisionModel.forward (the interpolated
  pos_embeds, the rotary cos/sin tables, the per-image attention chunk
  lengths) is precomputed ONCE on CPU by the model's own official methods for
  the fixed grid and cached as wrapper buffers -- no global patching, no
  reimplementation of the math;
- the attention glue (Qwen3_5VisionAttention.forward, non-flash branch) is
  replayed op-for-op with the chunk lengths as a fixed Python list, through the
  OFFICIAL eager fallback (eager_attention_forward): plain matmul/softmax, i.e.
  registered standard operators only -- this CANN install has no AscendIR
  converter for SDPA under dynamo export.

Precision: the residual stream, norms, rotary tables, softmax and the residual
adds are FP32; only the Linear/Conv inner products run in FP16 (``_linear``).
The checkpoint stores the 153 vision tensors in BF16, so the FP32 upcast is
exact and BF16 -> FP16 is lossless for these values (8-bit mantissa is a subset
of FP16's 10, weights far below the FP16 range).  A BF16 compute domain is NOT
used: it fails the vision gate against the CPU FP32 reference at real depth.

The merger's view performs the 2x2 spatial merge in row-major order, so the
output rows of slot i are exactly the official merged features of image i:
[slots * (h//2) * (w//2), out_hidden_size] float32.
"""
from __future__ import annotations

import torch

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionRotaryEmbedding, apply_rotary_pos_emb_vision, eager_attention_forward)

GRID_THW = (1, 16, 16)        # t, h, w of every image slot (256x256 input)
PATCH_ROWS = GRID_THW[1] * GRID_THW[2]            # 256 patch rows per slot
MERGED_ROWS = (GRID_THW[1] // 2) * (GRID_THW[2] // 2)   # 64 merged rows per slot


class MatMulPatchEmbed(torch.nn.Module):
    """The official patch conv with stride == kernel as ONE per-patch MatMul.

    Qwen3_5VisionPatchEmbed's Conv3d (3 -> embed_dim, kernel (2,16,16),
    stride == kernel, bias) has no receptive-field overlap, so
    ``out[p, o] = x_flat[p, :] . W_flat[:, o] + b[o]`` EXACTLY: one
    ``[P, 3*2*16*16] @ [3*2*16*16, embed_dim]`` matmul with the conv weight
    flattened [C, d, h, w, out] (row-major order of the official input
    view).  The conv module is held BY REFERENCE, so a later ``.to(dtype)`` on
    the shared module reaches this wrapper too and no weight storage is
    duplicated; the permute/reshape of the frozen weight is constant-folded in
    the exported graph.

    Accumulation order differs from conv at ulp level, which the full-model
    numeric gates cover.
    """

    def __init__(self, patch_embed, linear_dtype=torch.float32):
        super().__init__()
        conv = patch_embed.proj
        if list(conv.stride) != list(conv.kernel_size) or conv.padding != (0, 0, 0) \
                or conv.dilation != (1, 1, 1) or conv.groups != 1:
            raise NotImplementedError("MatMulPatchEmbed requires the official"
                                      " stride==kernel, unpadded, dense conv3d")
        self.conv = conv
        self.embed_dim = int(conv.weight.shape[0])
        self.patch_volume = int(conv.weight.shape[1:].numel())
        self.linear_dtype = linear_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv
        w = conv.weight.permute(1, 2, 3, 4, 0).reshape(self.patch_volume, -1)
        dtype = x.dtype if self.linear_dtype == torch.float32 else self.linear_dtype
        y = x.to(dtype).view(-1, self.patch_volume) @ w.to(dtype)
        if conv.bias is not None:
            y = y + conv.bias.to(dtype)
        return y.to(x.dtype).view(-1, self.embed_dim)


class VisionTowerWrapper(torch.nn.Module):
    """Original Qwen3_5VisionModel modules behind a fixed multi-slot fixed-grid IO.

    forward(pixel_values [slots*PATCH_ROWS, 1536] float32)
        -> merged image features [slots*MERGED_ROWS, out_hidden] float32.
    """

    def __init__(self, visual, grid_thw=GRID_THW, slots=2, buffer_source=None,
                 linear_dtype=torch.float32):
        super().__init__()
        self.linear_dtype = linear_dtype
        self.patch_embed = MatMulPatchEmbed(visual.patch_embed, linear_dtype)
        self.blocks = visual.blocks
        self.merger = visual.merger
        # Pre-quantize the Linear/Conv weights IN PLACE: replacing the parameter
        # data keeps the exported graph's Const COUNT unchanged (the same weight
        # constants, now fp16) so torchair's positional weight bake stays
        # aligned.  bf16-stored vision weights convert to fp16 losslessly (an
        # 8-bit mantissa is a subset of fp16's 10, and the weights sit far below
        # the fp16 range), so the values are bit-identical to a per-forward
        # .to(fp16) cast.  Norms and the constant buffers stay fp32, and every
        # consumer goes through the wrapper's replayed paths (_linear / _attn /
        # _mlp / _merger / MatMulPatchEmbed), so no official forward ever sees a
        # mixed-dtype Linear.
        if linear_dtype != torch.float32:
            for module in self.modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv3d)):
                    with torch.no_grad():
                        for param_name in ("weight", "bias"):
                            param = getattr(module, param_name, None)
                            if param is not None and param.dtype != linear_dtype:
                                param.data = param.data.to(linear_dtype)
        self.attn_implementation = "eager"
        self.grid_thw = tuple(int(v) for v in grid_thw)
        self.slots = int(slots)
        t, h, w = self.grid_thw
        self.lengths = [h * w] * self.slots          # one attention chunk per image
        # The official constant buffers (pos_embeds interpolation, rotary
        # cos/sin) are computed at CPU float32 precision by the model's own
        # official methods -- on ``visual`` itself when it already is a CPU
        # float32 module, otherwise on the caller-supplied CPU float32
        # ``buffer_source`` (the deep-copied source visual of an NPU load;
        # checkpoint-stored BF16 weights upcast to F32 exactly).
        #
        # EXCEPT the non-persistent rotary frequency: it is NOT stored in the
        # checkpoint, and a load path ending in model.to(device, dtype)
        # quantizes Qwen3_5VisionRotaryEmbedding.inv_freq irrecoverably -- a
        # later .float() cannot restore the official F32 value, which is small
        # enough (inv_freq max_abs ~1.8e-4) that the cast is pure loss.  The
        # official value is therefore REGENERATED with the official constructor
        # exactly as Qwen3_5VisionModel.__init__ does (head_dim // 2, factory
        # theta), compared first so a pristine CPU float32 source is left
        # untouched and only a quantized one is replaced.
        source = visual if buffer_source is None else buffer_source
        head_dim = source.config.hidden_size // source.config.num_heads
        official_rotary = Qwen3_5VisionRotaryEmbedding(head_dim // 2)
        if not torch.equal(source.rotary_pos_emb.inv_freq, official_rotary.inv_freq):
            source.rotary_pos_emb = official_rotary
        with torch.no_grad():
            grid = torch.tensor([[t, h, w]] * self.slots, dtype=torch.long)
            pos = source.fast_pos_embed_interpolate(grid)          # official method
            rot = source.rot_pos_emb(grid)                          # official method
            rot = rot.reshape(-1, rot.shape[-1])                    # official reshape
            emb = torch.cat((rot, rot), dim=-1)
            self.register_buffer("pos_embeds", pos.contiguous())
            self.register_buffer("rotary_cos", emb.cos().contiguous())
            self.register_buffer("rotary_sin", emb.sin().contiguous())

    def _linear(self, module, x):
        """The official Linear with a reduced-precision inner product.

        linear_dtype=float32 keeps the official module call bit-exact; float16
        runs the matmul in fp16 (NPU fp16 matmul runs at 2x the fp32 rate) and
        returns to the fp32 residual stream, so only the inner product is
        reduced -- norms, rotary tables and softmax stay fp32.
        """
        if self.linear_dtype == x.dtype:
            return module(x)
        dtype = self.linear_dtype
        weight = module.weight.to(dtype)
        bias = module.bias.to(dtype) if module.bias is not None else None
        return torch.nn.functional.linear(x.to(dtype), weight, bias).to(x.dtype)

    def _mlp(self, mlp, x):
        """Qwen3_5VisionMLP.forward op-for-op: fc2(act(fc1(x)))."""
        return self._linear(mlp.linear_fc2, mlp.act_fn(self._linear(mlp.linear_fc1, x)))

    def _merger(self, x):
        """Qwen3_5VisionPatchMerger.forward op-for-op (both norm branches)."""
        merger = self.merger
        y = merger.norm(x.view(-1, merger.hidden_size)
                        if merger.use_postshuffle_norm else x).view(-1, merger.hidden_size)
        return self._linear(merger.linear_fc2,
                            merger.act_fn(self._linear(merger.linear_fc1, y)))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(pixel_values) + self.pos_embeds
        for blk in self.blocks:
            hidden_states = hidden_states + self._attn(blk, hidden_states)
            hidden_states = hidden_states + self._mlp(blk.mlp, blk.norm2(hidden_states))
        return self._merger(hidden_states)

    def _attn(self, blk, hidden_states):
        """Qwen3_5VisionAttention.forward (non-flash branch) with the per-image
        chunk lengths fixed by the wrapper grid; identical ops in identical order."""
        attn = blk.attn
        x = blk.norm1(hidden_states)
        seq_length = x.shape[0]
        qkv = self._linear(attn.qkv, x)
        query_states, key_states, value_states = (
            qkv.reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0))
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, self.rotary_cos, self.rotary_sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        splits = [torch.split(t, self.lengths, dim=2)
                  for t in (query_states, key_states, value_states)]
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.attn_implementation, eager_attention_forward)
        attn_outputs = [interface(attn, q, k, v, attention_mask=None,
                                  scaling=attn.scaling, dropout=0.0, is_causal=False)[0]
                        for q, k, v in zip(*splits)]
        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self._linear(attn.proj, attn_output)
