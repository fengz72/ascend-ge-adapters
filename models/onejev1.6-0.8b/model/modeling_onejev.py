"""NPU-executable OneJev1.6-0.8B decision wrapper and its export patches.

``Decision.forward`` takes the seven graph-contract inputs (see
docs/adaptation-plan.md §3.2) and returns the float32 ``[1, 26]`` A..Z slot
scores: vision tower (single image) -> ``pooler_output`` scattered into the
token embeddings by ``feature_index`` -> text trunk -> last-token readout
against the tied lm_head rows of the 26 answer slots.  (``image_grid_thw``
and ``cu_seqlens`` are dead under the v1 patches and stay out of the graph;
see the Decision docstring.)

The trunk is the OFFICIAL transformers 5.17.0 ``Qwen3_5`` code; this module
only installs the minimal, audited patches that make it GE-exportable (each
one mirrors the validated prototype ``model/model.py``):

1. ``Qwen3_5VisionAttention.forward`` -> single-image full eager attention.
   The official eager path splits by ``cu_seqlens`` with ``lengths.tolist()``,
   a data-dependent Python list dynamo cannot symbolize; under the v1
   single-image contract ``cu_seqlens == [0, P]``, so one full-attention
   segment over all patches is exactly equivalent.
2. ``torch_chunk_gated_delta_rule`` -> the ``onejev::chunk`` custom op backed
   by ``npu_chunk_gated_delta_rule``, with a torchair GE converter emitting
   ``ChunkGatedDeltaRule`` (the compat proto .so must be LD_PRELOADed; see
   native_proto/README.md).  The official caller passes
   ``use_qk_l2norm_in_kernel=True``, ``initial_state=None``,
   ``output_final_state=False`` and no packed ``cu_seqlens`` -- all guarded.
3. ``F.softplus`` -> the overflow-safe export form (``t > 20`` passthrough,
   else ``log1p(exp(clamp(t, max=20)))``), identical to the -text delivery.
4. ``causal_conv1d_fn`` -> the bias-free four-tap SiLU causal depthwise
   convolution expressed as Conv2D (the standard Pad -> Conv2D -> SiLU GE
   route; no causal-conv1d vendor op).  Computed in fp32 then cast back,
   closer to the CPU golden than the official bf16 conv1d.

Unlike the prototype, vision position ids travel through the NATIVE 5.17.0
kwargs channel (``position_ids=`` is popped by ``get_vision_position_ids``),
so no ``get_vision_position_ids`` patch is needed.

This module is imported by ``model/export_air.py`` only; the pipeline stages
(generate_inputs / golden / export_air / executor) never import each other.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parents[1]
DEPS_DIR = ADAPTER_DIR / "work" / "deps" / "py"
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

import transformers

if transformers.__version__ != "5.17.0":
    raise RuntimeError(f"OneJev needs the task-local transformers 5.17.0 from {DEPS_DIR}, "
                       f"got {transformers.__version__} from {transformers.__file__}")

import torch
import torch_npu  # noqa: F401  (registers the npu device + npu_* ops)
import torchair
from torchair import ge, register_fx_node_ge_converter

from transformers.models.qwen3_5 import modeling_qwen3_5 as m

# The GDN chunk op scale is head_k_dim**-0.5; the checkpoint pins
# linear_key_head_dim=128 (asserted in load_decision_model).
CHUNK_SCALE = 128 ** -0.5

_PATCHED = False
_CHUNK_REGISTERED = False


def _single_image_attention(self, hidden_states, cu_seqlens, position_embeddings=None,
                            **kwargs):
    """Full eager attention over all patches of the one image (v1 contract)."""
    n = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(n, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    q, k = m.apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    q, k, v = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
    out = m.eager_attention_forward(
        self, q, k, v, attention_mask=None, scaling=self.scaling, dropout=0.0
    )[0]
    return self.proj(out.reshape(n, -1).contiguous())


@torch.library.custom_op("onejev::chunk", mutates_args=())
def _chunk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: torch.Tensor,
           s: torch.Tensor, l: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return torch.ops.npu.npu_chunk_gated_delta_rule(
        q, k, v, beta=b, initial_state=s, actual_seq_lengths=l, g=g, scale=CHUNK_SCALE
    )[0]


@_chunk.register_fake
def _chunk_fake(q, k, v, b, s, l, g):
    return torch.empty_like(v)


def _register_chunk_converter():
    global _CHUNK_REGISTERED
    if _CHUNK_REGISTERED:
        return

    @register_fx_node_ge_converter(torch.ops.onejev.chunk.default)
    def _chunk_converter(q, k, v, b, s, l, g, meta_outputs=None):
        return ge.custom_op(
            "ChunkGatedDeltaRule",
            inputs=dict(query=q, key=k, value=v, beta=b, initial_state=s,
                        actual_seq_lengths=l, g=g),
            attrs={"scale_value": ge.attr.Float(CHUNK_SCALE)},
            outputs=["out", "final_state"],
        )[0]

    _CHUNK_REGISTERED = True


def _delta(query, key, value, g, beta, **kwargs):
    """torch_chunk_gated_delta_rule replacement for the export path.

    Official call site semantics (5.17.0 Qwen3_5GatedDeltaNet, no cache):
    use_qk_l2norm_in_kernel=True, initial_state=None, output_final_state=False,
    cu_seqlens=None, B=1 true length -> the packed npu op with a zero initial
    state and actual_seq_lengths=S is exact.
    """
    if kwargs.get("use_qk_l2norm_in_kernel") is not True:
        raise ValueError("the export path expects use_qk_l2norm_in_kernel=True")
    if kwargs.get("initial_state") is not None or kwargs.get("cu_seqlens") is not None:
        raise ValueError("the export path is cache-free and unpacked (B=1 true length)")
    batch, width, heads, dim = value.shape

    def norm(t):
        f = t.float()
        return (
            (f * torch.rsqrt(f.square().sum(-1, keepdim=True) + 1e-6))
            .to(value.dtype)
            .flatten(0, 1)
        )

    out = _chunk(
        norm(query),
        norm(key),
        value.flatten(0, 1),
        beta.to(value.dtype).flatten(0, 1),
        value.new_zeros((batch, heads, dim, dim)),
        torch.full((batch,), width, device=value.device, dtype=torch.int32),
        g.float().flatten(0, 1),
    )
    return out.reshape(batch, width, heads, dim), None


def _export_softplus(t):
    return torch.where(t > 20, t, torch.log1p(torch.exp(torch.clamp(t, max=20))))


def _ge_causal_conv4(hidden_states, weight, bias=None, activation=None, **kwargs):
    if bias is not None or activation not in ("silu", "swish") or weight.shape[-1] != 4:
        raise ValueError("Expected bias-free four-tap SiLU causal convolution")
    channels = weight.shape[0]
    y = torch.nn.functional.conv2d(
        hidden_states.float().unsqueeze(2),
        weight.float().reshape(channels, 1, 1, 4),
        groups=channels,
        padding=(0, 3),
    )
    y = y[..., : hidden_states.shape[-1]].squeeze(2).to(hidden_states.dtype)
    return torch.nn.functional.silu(y)


def install_export_patches():
    """Install the four modeling patches (idempotent, process-wide).

    NOTE: ``m.F.softplus`` patches torch.nn.functional for the whole process
    (the modeling module's ``F`` IS that module); this is the export process,
    nothing else here consumes softplus.
    """
    global _PATCHED
    if _PATCHED:
        return
    _register_chunk_converter()
    m.Qwen3_5VisionAttention.forward = _single_image_attention
    m.torch_chunk_gated_delta_rule = _delta
    m.F.softplus = _export_softplus
    m.causal_conv1d_fn = _ge_causal_conv4
    _PATCHED = True


class Decision(torch.nn.Module):
    """The multimodal decision graph: 7 contract inputs -> scores [1, 26].

    ``image_grid_thw`` and ``cu_seqlens`` are NOT graph inputs: under the v1
    patches every grid consumer is fed a precomputed contract tensor
    (interp_indices / interp_weights / vision_position_ids) and the
    single-image attention ignores cu_seqlens, so both are dead to the graph
    and dynamo DCEs them out of the graph (a 9-input export silently lost the
    two Data nodes; export_air's binding check caught it).  ``visual()`` still
    requires the positional grid_thw argument and pops cu_seqlens from kwargs,
    so forward passes constant dummies for both: their only consumers are the
    ignored attention segmentation and max_seqlen (None under eager), so no
    value can reach the output.  Deriving cu from the dummy grid INSIDE the
    graph is not an option -- repeat_interleave with tensor repeats creates an
    unbacked symint (GuardOnDataDependentSymNode).  The real per-case grid
    travels as records.json metadata for the golden reference.
    """

    GRID_DUMMY = ((1, 2, 2),)
    # cu_seqlens matching GRID_DUMMY ([0, h*w] for the single dummy frame).
    # It MUST be passed precomputed: letting get_vision_cu_seqlens derive it
    # runs repeat_interleave with tensor repeats -> an unbacked symint ->
    # GuardOnDataDependentSymNode at export.
    CU_DUMMY = (0, 4)

    def __init__(self, model, slot_token_ids):
        super().__init__()
        self.backbone = model.model
        weight = model.get_output_embeddings().weight.detach()
        rows = weight.index_select(
            0, torch.tensor(list(slot_token_ids), device=weight.device)
        ).float()
        self.register_buffer("rows", rows)
        # Dead-argument dummies as BUFFERS (real tensors before tracing, baked
        # as Const by frozen_parameter): creating them with torch.tensor
        # inside forward made torchair fail the vision qkv permute fake call
        # (GE ERR03001 GRAPH invalid parameter).
        self.register_buffer(
            "grid_dummy", torch.tensor(self.GRID_DUMMY, dtype=torch.int64),
            persistent=False)
        self.register_buffer(
            "cu_dummy", torch.tensor(self.CU_DUMMY, dtype=torch.int32),
            persistent=False)

    def forward(self, input_ids, pixel_values, position_ids,
                interp_indices, interp_weights, vision_position_ids,
                feature_index):
        # 5.17.0 vision forward natively pops interp_indices / interp_weights /
        # position_ids / cu_seqlens from kwargs (transformers.vision_utils).
        feats = self.backbone.visual(
            pixel_values,
            grid_thw=self.grid_dummy,
            interp_indices=interp_indices,
            interp_weights=interp_weights,
            position_ids=vision_position_ids,
            cu_seqlens=self.cu_dummy,
        ).pooler_output
        text = self.backbone.get_input_embeddings()(input_ids)
        selected = feats.index_select(
            0, feature_index.clamp_min(0).flatten()
        ).reshape_as(text)
        embeds = torch.where(
            (feature_index >= 0).unsqueeze(-1), selected.to(text.dtype), text
        )
        hidden = self.backbone.language_model(
            inputs_embeds=embeds, position_ids=position_ids, use_cache=False
        ).last_hidden_state
        scores = hidden[:, -1].float() @ self.rows.T
        return scores


def slot_token_ids(folder):
    """The 26 A..Z answer-slot ids via the vendored official verification."""
    if str(ADAPTER_DIR / "vendor") not in sys.path:
        sys.path.insert(0, str(ADAPTER_DIR / "vendor"))
    from flide_onejev.slots import verify_slot_token_ids

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(folder),
                                                           local_files_only=True)
    ids = verify_slot_token_ids(tokenizer)[:26]
    if len(ids) != 26 or len(set(ids)) != 26:
        raise ValueError("the 26 A..Z answer slots are not distinct single tokens")
    return ids


def load_decision_model(folder, device):
    """Patched official checkpoint -> the Decision wrapper on ``device``."""
    install_export_patches()
    config = transformers.AutoConfig.from_pretrained(str(folder), local_files_only=True)
    text = config.text_config
    if text.linear_key_head_dim != 128:
        raise ValueError(f"CHUNK_SCALE pins linear_key_head_dim=128, config says "
                         f"{text.linear_key_head_dim}")
    if not getattr(text, "tie_word_embeddings", False):
        raise ValueError("the slot rows are derived from the tied output embedding; "
                         "this checkpoint must have tie_word_embeddings=True")
    ids = slot_token_ids(folder)
    model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        str(folder), dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True)
    model = Decision(model, ids).eval().to(device).requires_grad_(False)
    return model, ids
