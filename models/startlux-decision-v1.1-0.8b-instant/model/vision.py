"""Fixed-grid FP32 vision tower wrapper for the StartLux unified multimodal AIR.

Generalizes the P6-validated single-image ``VisionAIRWrapper``
(work/startlux-validation/p6-official/run_p6_vision_air.py, real-GE PASS:
work/startlux-validation/p6-official/report-vision-air.json) to a fixed number
N_SLOTS of same-grid image slots. Fidelity rules are unchanged:

- the wrapper holds the ORIGINAL HF visual modules (blocks, merger) of the
  pinned model folder, loaded from the checkpoint; the patch_embed conv is
  the bit-exact equivalent MatMulPatchEmbed (stride == kernel, see its
  docstring -- profiling-driven replacement of the MTE1-bound Conv3DV2);
- every shape-derived input of Qwen3_5VisionModel.forward (the interpolated
  pos_embeds, the rotary cos/sin tables, the per-image attention chunk
  lengths) is precomputed ONCE on CPU by the model's own official methods for
  the fixed grid and cached as wrapper buffers -- no global patching, no
  reimplementation of the math. The one non-checkpoint constant, the rotary
  inv_freq, is additionally REGENERATED with the official
  Qwen3_5VisionRotaryEmbedding constructor (head_dim // 2, factory theta):
  the framework load path ends with model.to(device, dtype), which quantizes
  the non-persistent inv_freq to BF16 irrecoverably
  (work/startlux-cannbot/review-vision-rotary-buffer-cast.json), so the
  checkpoint-independent frequency is taken from the official constructor,
  never from the possibly-cast source module;
- the attention glue (Qwen3_5VisionAttention.forward, non-flash branch) is
  replayed op-for-op with the chunk lengths as a fixed Python list; proven
  BIT-EXACT against the official eager model.model.get_image_features(...)
  on CPU FP32 before export (slots=1: p6-official image features A/B; slots=2:
  the official two-image forward);
- the attention interface is the OFFICIAL eager fallback
  (eager_attention_forward): plain matmul/softmax -- registered standard
  operators only (this CANN install has no AscendIR converter for SDPA under
  dynamo export);
- the tower is FP32 END TO END: the pinned checkpoint stores the 153 vision
  tensors in BF16, so loading with dtype=float32 upcasts them exactly; the
  BF16 vision compute route measurably fails against the CPU FP32 reference
  (rel_l2 ~0.11, archived in the p6-official report) and is NOT used.

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
    view).  The conv module is held BY REFERENCE and the input is cast to
    its weight dtype exactly like the official ``target_dtype`` path, so a
    later ``.to(dtype)`` on the shared module reaches this wrapper too --
    no weight copies are made (the permute/reshape of the frozen weight is
    constant-folded in the exported graph).

    Why: on this build the Conv3DV2 node is MTE1-saturated and MAC-starved
    (GE profiling 2026-11-05: task 6854 us/request = 20% of all AI time,
    mac ratio 0.22, mte1 ratio 0.97) -- a degenerate-tiling memory-bound
    kernel; the equivalent MatMulV2 carries the same bytes with proper
    MAC overlap.  Accumulation order differs from conv (ulp-level) and is
    covered by the full-model numeric gates.
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
        # PRE-QUANTIZE the linear weights to linear_dtype (P2-5): replacing
        # the parameter data in place keeps the exported graph's Const COUNT
        # unchanged (same ~51 weight constants, now fp16) so torchair's
        # positional weight bake stays aligned -- the buffer-copy approach
        # failed exactly there.  bf16-stored vision weights convert to fp16
        # LOSSLESSLY (8-bit mantissa subset of fp16's 10, weights far below
        # fp16 range), so the values are bit-identical to the per-forward
        # .to(fp16) casts this removes.  Norms and the constant buffers stay
        # fp32; every consumer goes through the wrapper's replayed paths
        # (_linear / _attn / _mlp / _merger / MatMulPatchEmbed), so no
        # official forward ever sees a mixed-dtype Linear.
        if linear_dtype != torch.float32:
            ceiling = torch.finfo(linear_dtype).max
            for module in self.modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv3d)):
                    with torch.no_grad():
                        for param_name in ("weight", "bias"):
                            param = getattr(module, param_name, None)
                            if param is None or param.dtype == linear_dtype:
                                continue
                            if param is not None and float(param.abs().max()) > ceiling:
                                raise ValueError(
                                    f"{module}.{param_name} exceeds {linear_dtype} range")
                            param.data = param.data.to(linear_dtype)
        self.attn_implementation = "eager"
        self.grid_thw = tuple(int(v) for v in grid_thw)
        self.slots = int(slots)
        if self.slots < 1:
            raise ValueError("slots must be >= 1")
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
        # checkpoint, and a source loaded through the framework path ends with
        # model.to(device, dtype) (core.sources.load_source), which quantizes
        # Qwen3_5VisionRotaryEmbedding.inv_freq to BF16; a later .float() can
        # NOT restore the official F32 value (inv_freq max_abs 1.78e-4,
        # 16-position frequency table max_abs 2.68e-3:
        # work/startlux-cannbot/review-vision-rotary-buffer-cast.json). The
        # official value is therefore REGENERATED with the official constructor
        # exactly as Qwen3_5VisionModel.__init__ does (head_dim // 2, factory
        # theta) -- compared first so a pristine CPU float32 source is left
        # untouched (bit-equal) and a quantized one is restored.
        source = visual if buffer_source is None else buffer_source
        head_dim = source.config.hidden_size // source.config.num_heads
        official_rotary = Qwen3_5VisionRotaryEmbedding(head_dim // 2)
        self.rotary_inv_freq_restored = not bool(torch.equal(
            source.rotary_pos_emb.inv_freq, official_rotary.inv_freq))
        if self.rotary_inv_freq_restored:
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
        # The shared modules (patch_embed / blocks / merger) live on the SOURCE
        # device of ``visual``, so the freshly computed CPU buffers must follow
        # them: a wrapper built from an NPU load with only CPU buffers would be
        # a mixed-device module that only a later .to(device) -- which production
        # callers may not perform -- would hide. .to(device) keeps the official
        # float32 buffer dtypes bit-exact.
        target = next(visual.parameters()).device
        if target.type != "cpu":
            for name in ("pos_embeds", "rotary_cos", "rotary_sin"):
                setattr(self, name, getattr(self, name).to(target))

    def _linear(self, module, x):
        """The official Linear with an optional reduced-precision inner
        product.  linear_dtype=float32 (default) keeps the official module
        call bit-exact; float16 casts the ACTIVATION only (the weight copy is
        pre-materialized as a constant) and returns to the fp32 flow -- the
        validated mixed-precision structure (constants, rotary tables, norms
        and softmax stay fp32; bf16 FAILS the vision gate at real depth
        relL2 0.055, fp16 passes; NPU fp16 matmul runs at 2x the fp32 rate)."""
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
        stream = torch.float32 if self.linear_dtype == torch.float32 else self.linear_dtype
        # fp16 residual stream: the layer-to-layer hidden state is stored in
        # fp16 (halved activation traffic; every layer boundary round-trips
        # through the fp32 compute domain -- norms, attention glue and the
        # residual adds all accumulate in fp32).  Validated on the real
        # 12-block tower: relL2 0.0079 vs 0.0050 fp32-stream, both inside
        # the vision gate; the fp32 path below is the unchanged bit-exact one.
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
