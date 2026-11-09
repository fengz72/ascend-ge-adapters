"""StartLux-Decision v1.1 0.8B Instant text adapter: instance wrappers only, no process-wide patches.

Candidate path (the single native route; the former pure-Torch chunk-GDR
standard route was removed by decision 2026-11-02 after the native route's
formal acceptance): a model-specific ``StartLuxTextDecoder`` nn.Module reusing
the loaded HF modules (embeddings, projections, MLP, norms, conv) with
- 18 causal Gated DeltaNet layers via the native
  ``torch.ops.npu.npu_chunk_gated_delta_rule`` (GE op ChunkGatedDeltaRule,
  aclnn host implementation in eager),
- 6 bidirectional full-attention layers via eager math with a padding-only
  additive mask (real Q=8/KV=2/D=256, Q/K RMSNorm 1+weight, partial RoPE 64
  dims in-graph from position_ids, sigmoid output gate),
- final RMSNorm, readout at the last valid position, 26-row FP32 head copy.

Native GDR semantics:
- q/k are L2-normalized ONCE as a fused RmsNorm per side
  (``rms_norm(x, gamma=1/sqrt(D), eps=1e-6/D)`` == ``x * rsqrt(sum(x^2) +
  1e-6)`` exactly, kernel-internal fp32 reduction, bf16 output; the bf16
  gamma carries ~1e-4 relative rounding, single-layer NPU-probed relL2
  0.0029 vs the explicit chain); the kernel ``scale = Dk**-0.5`` is
  UNCHANGED -- the delta update couples q/k magnitudes nonlinearly, so
  magnitude folding must stay in gamma, not in the scale;
- v/beta stay BF16, g stays FP32
  (``-exp(A_log_f32) * softplus(a.float() + dt_bias_bf16)``);
- every row starts from a zero initial_state ``[B, Hv, Dv, Dk]`` BF16; there is
  no state cache and the final state is never reused (Instant readout only);
- the wrapper runs on the padded ``[B, S]`` layout, so the flattened ``T = B*S``
  rows pass each row's PHYSICAL width S as ``actual_seq_lengths`` (the current
  CANN 9.1.0 kernel expects per-row lengths, never cumulative ends, never the
  true L); positions beyond the true length are NOT masked: the chunk
  recurrence is strictly causal, so padded tokens only ever influence outputs
  at their own and later positions and the final state -- the readout
  gathers the true L-1 and the state is discarded, so per-row padding never
  leaks.  The former
  q/k/v/beta zeroing (~90 Mul per request) was removed 2026-11-09 after the
  full-model gates re-passed bit-stably;
- num_k_heads must equal num_v_heads (the kernel layout has no GQA repeat).

Export requires the ``chunk_gated_delta_rule_proto`` dependency (project-built
GE AscendIR prototype registration): ``core.dependencies.prepare`` plus
``source scripts/env_chunk_proto.sh <work_dir>`` so the exporting process and
the native ge_runtime subprocesses load the prototype library explicitly
(auto vendor loading is rejected). Eager NPU execution dispatches to the CANN
aclnn host implementation and needs no prototype library. CANN upgrades or
SoC changes require re-running the prototype gate: the former pure-Torch
standard route no longer exists as a fallback.

A_log is restored bit-exact F32 from the checkpoint after the BF16 load
(core load_source casts the whole model); dt_bias stays its original BF16.
Reference path: the pinned official vendored bidirectional text_forward on the
unpatched FP32 CPU model, exactly the P2b golden recipe.

Known GE-converter-driven deviations (numerically justified):
- F.softplus -> where(x > 20, x, log1p(exp(clamp(x, max=20)))): aten.softplus
  has no GE converter; the expression is the exact PyTorch definition
  (beta=1, threshold=20) on aten ops GE covers (ge.Log1p), verified equal to
  F.softplus in tests.
- torch.split(list) -> explicit slices: aten.split.sizes has no GE converter.
- interleaved MRoPE cos/sin computed from position_ids: the shared
  forward_embeddings loop accepts [B, S] text positions (three identical
  planes; bit-exact with the previous single-plane formula, verified against
  HF rotary in tests) or the official [3, B, S] get_rope_index planes with the
  official interleaved mrope_section layout (mrope_cos_sin), used by the
  unified multimodal AIR (multimodal.py).
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root
ROOT = MODEL_DIR.parents[1]
VENDOR = MODEL_DIR / "decider" / "vendor"


def _params(config):
    params = {"max_seq_len": 2048, "compiled_seq_len": 256, "chunk_size": 64,
              "max_batch": 4, "vision_linear_dtype": "fp16",
              **config["model"]["params"]}
    allowed = {"max_seq_len", "compiled_seq_len", "chunk_size", "max_batch",
               "vision_linear_dtype"}
    if params.get("vision_linear_dtype") not in ("fp32", "fp16"):
        raise ValueError("vision_linear_dtype must be 'fp32' or 'fp16'")
    if set(params) - allowed:
        raise ValueError(f"unknown StartLux model.params: {set(params) - allowed}")
    for key in ("max_seq_len", "compiled_seq_len", "chunk_size", "max_batch"):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if params["compiled_seq_len"] > params["max_seq_len"]:
        raise ValueError("compiled_seq_len cannot exceed max_seq_len")
    return params


def _source_ref(config):
    """Resolve config source.ref the way the framework does: ${VAR} from the
    environment, relative paths from the project root; no core imports."""
    ref = (config.get("source") or {}).get("ref")
    if not isinstance(ref, str) or not ref.strip():
        raise ValueError("source.ref is required: letter ids and temperatures are read from "
                         "the model folder's decision_config.json, not from fixture provenance")
    def replace(match):
        content = os.environ.get(match.group(1))
        if content is None or not content.strip():
            raise ValueError(f"source.ref: environment variable {match.group(1)} is not set or is empty")
        return content

    ref = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, ref.strip())
    path = Path(ref)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _decision_config(config):
    """Letter ids / temperatures / vocab bound read from the model folder itself."""
    folder = _source_ref(config)
    decision = json.loads((folder / "decision_config.json").read_text())
    letters = decision.get("letter_token_ids")
    if not isinstance(letters, list) or len(letters) != 26 or len(set(letters)) != 26 \
            or any(type(t) is not int or t < 0 for t in letters):
        raise ValueError("decision_config letter_token_ids must be 26 unique nonnegative ids")
    table = decision.get("temperature_by_type")
    if not isinstance(table, dict):
        raise ValueError("decision_config temperature_by_type must be a mapping")
    for qtype in ("choice", "score", "noul"):
        value = table.get(qtype)
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"decision_config temperature_by_type needs a positive finite {qtype!r} entry")
    model_config = json.loads((folder / "config.json").read_text())
    vocab = (model_config.get("text_config") or {}).get("vocab_size",
                                                        model_config.get("vocab_size"))
    if type(vocab) is not int or vocab < 1:
        raise ValueError("model config.json must carry a positive text_config.vocab_size")
    token_ids = {name: model_config.get(name) for name in
                 ("image_token_id", "vision_start_token_id", "vision_end_token_id")}
    # optional at this shared layer (text-only fixture folders carry none);
    # any entry that IS present must be an in-vocab nonnegative int. The
    # unified multimodal route enforces completeness in its own describe.
    if any(value is not None and (type(value) is not int or value < 0 or value >= vocab)
           for value in token_ids.values()):
        raise ValueError("model config.json image_token_id / vision_start_token_id / "
                         "vision_end_token_id, when present, must be in-vocab integers")
    return {"folder": folder, "letter_token_ids": letters,
            "temperature_by_type": table, "vocab_size": vocab,
            "token_ids": token_ids}


def _dynamic_axes(params):
    """IO axes honestly derived from the configured caps.

    An axis stays dynamic only when its cap admits a >= 2 trace example; a
    max_batch/max_seq_len of 1 is declared STATIC instead of pretending the
    graph is fully dynamic (dynamo would specialize that size-1 dim anyway).
    """
    batch = -1 if params["max_batch"] >= 2 else 1
    seq = -1 if params["max_seq_len"] >= 2 else 1
    return {"input_ids": [batch, seq], "position_ids": [batch, seq],
            "actual_seq_lengths": [batch], "scores": [batch, 26]}


def _example_geometry(params):
    """The deterministic export trace geometry: B2/S65 whenever caps allow.

    S=65 keeps the example past the 64-token chunk boundary so the chunked GDR
    path traces more than one chunk; both dims >= 2 keep them symbolic.
    """
    return min(2, params["max_batch"]), min(65, params["max_seq_len"])


def _dynamic_io(params):
    """The dynamic-route IOSpec nodes (inputs, output) for these params."""
    axes = _dynamic_axes(params)

    def node(name, dtype, shape):
        return {"name": name, "dtype": dtype, "shape": shape,
                "dynamic_dims": [i for i, dim in enumerate(shape) if dim == -1]}

    inputs = [node("input_ids", "int64", axes["input_ids"]),
              node("position_ids", "int64", axes["position_ids"]),
              node("actual_seq_lengths", "int32", axes["actual_seq_lengths"])]
    return inputs, [node("scores", "float32", axes["scores"])]


def describe(config):
    params = _params(config)
    decision = _decision_config(config)
    example_batch, example_width = _example_geometry(params)
    return {"version": 1, "global_patch": False, "in_place": False,
            "patches": [],
            "referenced_files": [str(decision["folder"] / "decision_config.json")],
            "reference_params": {"letter_token_ids": decision["letter_token_ids"],
                                 "temperature_by_type": decision["temperature_by_type"],
                                 "vocab_size": decision["vocab_size"],
                                 "token_ids": decision["token_ids"],
                                 "decision_config": str(decision["folder"] / "decision_config.json")},
            "readout_contract": {
                "samples": "B independent readout rows (1..max_batch); one readout per row",
                "rows": "each payload row is one full prompt: row length == its true length",
                "positions": "row i position_ids == arange(Li)",
                "lengths": "actual_seq_lengths are per-row true lengths, never cumulative ends",
                "bounds": {"max_batch": params["max_batch"], "max_seq_len": params["max_seq_len"],
                           "vocab_size": decision["vocab_size"]}},
            "changes": ["instance wrapper decoder",
                        "native ChunkGatedDeltaRule GDR (torch.ops.npu.npu_chunk_gated_delta_rule)",
                        "L2norm as fused RmsNorm(gamma=1/sqrt(Dk), eps/1e-6/Dk) per side == exact l2norm (fp32 internal reduction, bf16 out; gamma bf16 ~1e-4); kernel scale Dk**-0.5 unchanged; no pad masking",
                        "v/beta BF16, g FP32; zero initial_state [B,Hv,Dv,Dk] BF16 per layer, no state cache or reuse",
                        "padded B*S flatten passes per-row physical width S as actual_seq_lengths; readout gathers true L-1",
                        "causal short conv as one fp32 depthwise Conv2D node (causal_conv4), replaces Conv1D",
                        "eager-math bidirectional full attention",
                        "vision tower linear compute in fp16 (validated mixed precision: fp16 "
                        "inner products, fp32 constants/norms/softmax; bf16 fails the vision "
                        "gate at real depth, fp16 passes with 4.5x margin)",
                        "in-graph partial RoPE from position_ids (text [B,S] or official [3,B,S] MRoPE planes)",
                        "shared forward_embeddings decoder loop (unified multimodal injection entry)",
                        "26-row FP32 head copy",
                        "A_log restored F32 from checkpoint",
                        "softplus and split(list) replaced by GE-covered equivalents"],
            "dynamic": {
                "axes": _dynamic_axes(params),
                "max_batch": params["max_batch"], "max_seq_len": params["max_seq_len"],
                "example": (f"B{example_batch}/S{example_width} deterministic trace example carried by "
                            "CandidateHandle.export_spec['example_inputs'] (zeros ids, arange "
                            "positions, width lengths): no fixture case, no golden; both dynamic dims "
                            "trace >= 2 so dynamo keeps them symbolic"),
                "length_semantics": "per-row true lengths in actual_seq_lengths; the native GDR call "
                                    "uses the physical padded width S per row (never cumulative ends, "
                                    "never the true L)",
                "one_graph": (f"one AIR for B1..B{params['max_batch']} and "
                              f"S1..{params['max_seq_len']} without per-shape recompilation; "
                              "per-shape kernel specialization at first execution is a runtime cache "
                              "effect, not a new graph"),
            },
            "required_dependencies": ["chunk_gated_delta_rule_proto"],
            "native_gdr": {
                "op": "ChunkGatedDeltaRule (torch.ops.npu.npu_chunk_gated_delta_rule)",
                "fusion": "shared causal_conv4 + native chunk op outside the kernel",
                "qk_l2norm": "fused RmsNorm(gamma=1/sqrt(Dk), eps/1e-6/Dk) == exact l2norm; bf16 out, fp32 internal reduction",
                "scale": "kernel-applied head_k_dim**-0.5 exactly once (adapter never pre-scales)",
                "dtypes": {"query": "bfloat16", "key": "bfloat16", "value": "bfloat16",
                           "beta": "bfloat16", "g": "float32",
                           "initial_state": "bfloat16 zeros [B,Hv,Dv,Dk]", "out": "bfloat16"},
                "length_semantics": ("per-row lengths; padded B*S layout passes the physical width S "
                                     "per row (not cumulative ends, not the true L)"),
                "readout": "gather true L-1 (decoder forward unchanged)",
                "num_matrices": None,
                "short_conv": "causal_conv4 (one fp32 depthwise Conv2D node; SiLU after the bf16 cast)",
                "dependency_loading": "prepare.sh -> scripts/build_chunk_proto.sh + source "
                                      "scripts/env_chunk_proto.sh <work_dir> (explicit local "
                                      "dlopen/LD_PRELOAD; auto vendor loading rejected)",
                "eager": "CANN aclnn host implementation; no prototype library needed",
            }}


def _mrope_interleave_index(mrope_section, half, device=None):
    """Constant gather index of the official interleaved MRoPE layout.

    Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope reorganizes the
    chunked [T... H... W...] frequency layout into interleaved
    [T0 H0 W0 T1 H1 W1 ...]; the output slot j takes plane ``source[j]``
    where source follows the official slices (plane dim at indices
    dim, dim+3, ... below section[dim]*3). The returned index maps the
    flattened [3, half] product to the interleaved [half] layout:
    out[..., j] = freqs_flat[..., source[j] * half + j].
    """
    source = [0] * half
    for dim in (1, 2):
        length = mrope_section[dim] * 3
        for j in range(dim, min(length, half), 3):
            source[j] = dim
    return torch.tensor([source[j] * half + j for j in range(half)],
                        dtype=torch.int64, device=device)


def _softplus(x):
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(torch.clamp(x, max=20.0))))


def _rms_norm(norm, x):
    """HF Qwen3_5RMSNorm.forward fused into the GE NpuRmsNorm node.

    Graph leg: ``y = rsqrt(mean(x^2) + eps) * x * (1 + weight)`` -- the
    official compute semantics.  x and the (1 + weight) gamma are both fed
    DIRECTLY in bf16: the kernel's reduction runs in fp32 internally
    (NPU-verified 2026-11-06), and the pbtxt diff probe showed an fp32 gamma
    gets compiler-inserted [1024] fp32->bf16 Casts at GE tiling anyway (the
    delivered kernel already consumes bf16 gamma), so bf16 keeps the math
    identical and drops the 48 per-request Casts.  Eager leg keeps the
    official module itself.
    """
    if not torch.compiler.is_compiling():
        return norm(x)
    gamma = 1.0 + norm.weight
    y, _ = torch.ops.npu.npu_rms_norm(x, gamma, float(norm.eps))
    return y.to(x.dtype)


def _rms_norm_gated(norm, x, gate):
    """Qwen3_5RMSNormGated.forward with its RMS core fused into NpuRmsNorm.

    Official order (transformers eager fallback, the pinned path):
    ``x32 = x.float(); n = x32 * rsqrt(mean(x32^2) + eps);
    out = (weight * n.to(x.dtype)) * silu(gate.float()); out.to(x.dtype)``.
    The graph leg keeps that exact domain/order; only the
    square/mean/rsqrt/mul core becomes one NpuRmsNorm (gamma = ones).
    """
    if not torch.compiler.is_compiling():
        return norm(x, gate)
    ones = torch.ones_like(norm.weight)
    n, _ = torch.ops.npu.npu_rms_norm(x, ones, float(norm.variance_epsilon))
    return (norm.weight * n.to(x.dtype) * F.silu(gate.float())).to(x.dtype)


def causal_conv4(x, weight):
    """Shared causal 4-tap depthwise short conv for the GDR path: ONE fp32
    depthwise Conv2D node.

    Replaces ``F.silu(conv1d(x, weight, padding=3)[:, :, :S])`` with a
    [B,C,1,S] x [C,1,1,4] depthwise Conv2D (groups=C, padding=(0,3), sliced
    to the first S columns): FP32 multiply/accumulate, cast to the input
    dtype, then SiLU - the exact HF compute order (conv -> BF16 -> SiLU).
    The HF conv is bias-free, so no bias term. Accepts non-contiguous
    [B, C, S] transposed views; returns the SiLU-activated [B, C, S].

    Route evidence (2026-11-05, GE graph-mode small-graph gates + eager, at
    C=6144/groups=C/kernel=4/S256 and S2048): the bf16-input Conv1D AND
    Conv2D nodes both miscompile this configuration (relL2 1.41 vs an fp64
    golden -- the P3 finding, work/startlux-validation/p3/debug-qkvconv;
    torchair routes aten.conv1d to the same Conv2D node), while the fp32-input
    Conv2D node is numerically correct (relL2 2.41e-3 = the expected bf16
    output quantization, matching the previous shift form and eager exactly)
    and 27-45% faster in GE than the explicit 4-tap shift chain. Kernel size
    is fixed at 4 per the StartLux linear_conv_kernel_dim.
    """
    channels, seq = x.shape[1], x.shape[2]
    w = weight.float().reshape(channels, 1, 1, 4)
    y = F.conv2d(x.float().unsqueeze(2), w, groups=channels, padding=(0, 3))
    return F.silu(y[..., :seq].squeeze(2).to(x.dtype))


def rope_cos_sin(position_ids, inv_freq, dtype):
    """Text-only interleaved-MRoPE cos/sin: identical T/H/W sections reduce the
    official interleave to the plain outer product (verified against HF)."""
    positions = position_ids.to(torch.float32)
    freqs = positions[:, :, None] * inv_freq[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class StartLuxTextDecoder(nn.Module):
    """Bidirectional text forward of the StartLux decision decoder, one readout per row."""

    def __init__(self, body, letters26, chunk_size=64):
        super().__init__()
        self.body = body
        self.letters26 = list(letters26)
        self.chunk_size = chunk_size
        cfg = body.config
        self.layer_types = list(cfg.layer_types[:cfg.num_hidden_layers])
        if set(self.layer_types) - {"linear_attention", "full_attention"}:
            raise ValueError(f"unexpected layer types: {sorted(set(self.layer_types))}")
        rope = cfg.rope_parameters
        if rope.get("rope_type") != "default":
            raise ValueError("only default rope is supported")
        full_idx = self.layer_types.index("full_attention")
        rotary_dim = int(body.layers[full_idx].self_attn.head_dim * rope["partial_rotary_factor"])
        self.mrope_section = [int(v) for v in rope.get("mrope_section", [rotary_dim // 2])]
        if sum(self.mrope_section) != rotary_dim // 2:
            raise ValueError("mrope_section does not match the rotary dim")
        inv_freq = 1.0 / (float(rope["rope_theta"]) ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32,
                         device=body.embed_tokens.weight.device) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.rotary_dim = rotary_dim
        self.register_buffer("mrope_gather", _mrope_interleave_index(
            self.mrope_section, rotary_dim // 2,
            device=body.embed_tokens.weight.device), persistent=False)
        # The GDR q/k L2-norm gamma of the fused RmsNorm form (P0-2): a
        # module-level constant so it constant-folds once in the exported
        # graph (no per-layer torch.full node) and follows .to(device) like
        # every buffer.  bf16 (the kernel's gamma dtype; ~1e-4 rounding).
        first_linear = body.layers[self.layer_types.index("linear_attention")].linear_attn
        self.register_buffer("gdr_l2_gamma", torch.full(
            (first_linear.head_k_dim,), first_linear.head_k_dim ** -0.5,
            dtype=torch.bfloat16, device=body.embed_tokens.weight.device),
            persistent=False)
        embedding = body.embed_tokens.weight
        with torch.no_grad():
            rows = embedding.index_select(
                0, torch.tensor(self.letters26, device=embedding.device)).to(torch.float32).clone()
        self.head_rows = nn.Parameter(rows, requires_grad=False)
        self.min_score = float(torch.finfo(torch.bfloat16).min)
        self.embedding_shape = list(embedding.shape)


    def forward(self, input_ids, position_ids, actual_seq_lengths):
        # B/length validation is encode-side (Python, array level): the traced
        # graph must stay symbolic in B and S for the dynamic native route.
        return self.forward_embeddings(self.body.embed_tokens(input_ids),
                                       position_ids, actual_seq_lengths)

    def forward_embeddings(self, inputs_embeds, position_ids, actual_seq_lengths):
        """Shared decoder loop over precomputed input embeddings.

        ``position_ids`` is either [B, S] text positions or the [3, B, S]
        multimodal MRoPE planes of the official get_rope_index; the interleaved
        mrope_section cos/sin (mrope_cos_sin below) is bit-exact with the
        former 2D path on identical text planes. The unified multimodal AIR
        injects image features into ``inputs_embeds`` (equivalent to the
        official masked_scatter) and calls this loop.
        """
        lengths = actual_seq_lengths
        batch = inputs_embeds.shape[0]
        width = inputs_embeds.shape[1]
        hidden = inputs_embeds
        cos, sin = self.mrope_cos_sin(position_ids, hidden.dtype)
        valid = torch.arange(width, device=inputs_embeds.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_mask = ((~valid).to(hidden.dtype) * self.min_score)[:, None, None, :]
        for layer, layer_type in zip(self.body.layers, self.layer_types):
            residual = hidden
            mixed = _rms_norm(layer.input_layernorm, hidden)
            if layer_type == "linear_attention":
                mixed = self._gated_delta(layer.linear_attn, mixed, lengths)
            else:
                mixed = self._full_attention(layer.self_attn, mixed, cos, sin, attn_mask)
            hidden = residual + mixed
            residual = hidden
            mixed = _rms_norm(layer.post_attention_layernorm, hidden)
            hidden = residual + layer.mlp(mixed)
        hidden = _rms_norm(self.body.norm, hidden)
        last_index = (lengths.to(torch.int64) - 1).view(batch, 1, 1)
        last = hidden.gather(1, last_index.expand(batch, 1, hidden.shape[-1])).squeeze(1)
        return last.float() @ self.head_rows.float().transpose(0, 1)

    def mrope_cos_sin(self, position_ids, dtype):
        """Official interleaved MRoPE cos/sin (Qwen3_5TextRotaryEmbedding).

        position_ids [B, S] (text; expanded to three identical planes -- bit-exact
        with the module-level rope_cos_sin) or [3, B, S] (the multimodal planes of
        get_rope_index). Returns (cos, sin) [B, S, rotary_dim].
        """
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq      # [3,B,S,half]
        freqs = freqs.permute(1, 2, 0, 3).flatten(2)                              # [B,S,3*half]
        freqs = freqs.index_select(-1, self.mrope_gather)                         # [B,S,half]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _rope(self, position_ids, dtype):
        return rope_cos_sin(position_ids, self.inv_freq, dtype)

    def _gated_delta(self, attn, hidden, lengths):
        return self._gated_delta_native(attn, hidden, lengths)

    def _gated_delta_native(self, attn, hidden, lengths):
        batch, width, _ = hidden.shape
        mixed_qkv = attn.in_proj_qkv(hidden).transpose(1, 2)
        z = attn.in_proj_z(hidden).reshape(batch, width, -1, attn.head_v_dim)
        b = attn.in_proj_b(hidden)
        a = attn.in_proj_a(hidden)
        mixed = causal_conv4(mixed_qkv, attn.conv1d.weight).transpose(1, 2)
        key_dim, value_dim = attn.key_dim, attn.value_dim
        q = mixed[..., :key_dim].reshape(batch, width, -1, attn.head_k_dim)
        k = mixed[..., key_dim:2 * key_dim].reshape(batch, width, -1, attn.head_k_dim)
        v = mixed[..., 2 * key_dim:].reshape(batch, width, -1, attn.head_v_dim)
        beta = b.sigmoid()
        g = -attn.A_log.float().exp() * _softplus(a.float() + attn.dt_bias)
        if attn.num_v_heads != attn.num_k_heads:
            raise NotImplementedError(
                "native chunk GDR requires num_k_heads == num_v_heads "
                f"(got {attn.num_k_heads}/{attn.num_v_heads})")
        # L2-norm as ONE fused RmsNorm per side: rms_norm(x, gamma=1/sqrt(D),
        # eps/D) == x * rsqrt(sum(x^2)+eps) == l2norm(x, eps) EXACTLY (the
        # sqrt(D) of mean-domain folding cancels in gamma; kernel-internal
        # fp32 reduction, bf16 output; the bf16 gamma carries ~1e-4 relative
        # rounding, NPU-probed at relL2 0.0029 vs the explicit chain on the
        # real single-layer op -- the kernel scale stays Dk**-0.5 UNCHANGED:
        # the delta update couples q/k magnitudes nonlinearly, so folding
        # sqrt(D) into the scale instead was probed WRONG at relL2 0.85).
        q, _ = torch.ops.npu.npu_rms_norm(q, self.gdr_l2_gamma,
                                          1e-6 / attn.head_k_dim)
        k, _ = torch.ops.npu.npu_rms_norm(k, self.gdr_l2_gamma,
                                          1e-6 / attn.head_k_dim)
        q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)
        beta, g = beta.flatten(0, 1), g.flatten(0, 1)
        state = torch.zeros(batch, attn.num_v_heads, attn.head_v_dim, attn.head_k_dim,
                            dtype=torch.bfloat16, device=hidden.device)
        row_widths = torch.full((batch,), width, dtype=torch.int32, device=hidden.device)
        out, _final_state = torch.ops.npu.npu_chunk_gated_delta_rule(
            q, k, v, beta=beta, initial_state=state,
            actual_seq_lengths=row_widths,
            scale=float(attn.head_k_dim) ** -0.5, g=g)
        out = out.reshape(-1, attn.head_v_dim)
        z = z.reshape(-1, attn.head_v_dim)
        out = _rms_norm_gated(attn.norm, out, z).reshape(batch, width, -1)
        return attn.out_proj(out)

    def _full_attention(self, attn, hidden, cos, sin, mask):
        batch, width, _ = hidden.shape
        qkv = attn.q_proj(hidden).view(batch, width, -1, attn.head_dim * 2)
        q, gate = qkv[..., :attn.head_dim], qkv[..., attn.head_dim:]
        gate = gate.reshape(batch, width, -1)
        q2d = _rms_norm(attn.q_norm, q.reshape(batch, width, -1, attn.head_dim))
        k2d = _rms_norm(attn.k_norm, attn.k_proj(hidden).view(batch, width, -1, attn.head_dim))
        v = attn.v_proj(hidden).view(batch, width, -1, attn.head_dim).transpose(1, 2)
        # RoPE runs in the NATIVE [B,S,N,D] domain (before the single
        # transpose to BNSD).
        if torch.compiler.is_compiling():
            # P2-7 fused RoPE: the partial-rotary dims go through
            # npu_apply_rotary_pos_emb in the native BSND layout (official
            # layout=1: q/k [B,S,N,rot], cos/sin [B,S,1,rot], zero
            # reshapes; rotary_mode half == the exact rotate-half math).
            # GE-graph-verified against pure math at relL2 0.0002 for every
            # B/S of the case pool.  The aclnn EAGER leg's layout=1 is NOT
            # BSND (reads [B,S,N,D] as BSH), so the eager branch keeps
            # explicit math.
            import torch_npu
            q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(
                q2d[..., :self.rotary_dim], k2d[..., :self.rotary_dim],
                cos.unsqueeze(2), sin.unsqueeze(2),   # [B,S,1,rot]
                layout="BSND", rotary_mode="half")
            q2d = torch.cat((q_rot, q2d[..., self.rotary_dim:]), dim=-1)
            k2d = torch.cat((k_rot, k2d[..., self.rotary_dim:]), dim=-1)
        else:
            cos4 = cos.unsqueeze(2)
            sin4 = sin.unsqueeze(2)
            q_rot, q_pass = q2d[..., :self.rotary_dim], q2d[..., self.rotary_dim:]
            k_rot, k_pass = k2d[..., :self.rotary_dim], k2d[..., self.rotary_dim:]
            x1, x2 = q_rot[..., :q_rot.shape[-1] // 2], q_rot[..., q_rot.shape[-1] // 2:]
            q2d = torch.cat((q_rot * cos4 + torch.cat((-x2, x1), dim=-1) * sin4, q_pass), dim=-1)
            x1, x2 = k_rot[..., :k_rot.shape[-1] // 2], k_rot[..., k_rot.shape[-1] // 2:]
            k2d = torch.cat((k_rot * cos4 + torch.cat((-x2, x1), dim=-1) * sin4, k_pass), dim=-1)
        q_embed = q2d.transpose(1, 2)
        k_embed = k2d.transpose(1, 2)
        if torch.compiler.is_compiling():
            # P2-8 fused attention: one npu_fusion_attention (FA) per layer
            # replaces the eager chain (GQA k/v expand, scores matmul +
            # additive mask, fp32 softmax, probs matmul) -- bf16 BNSD with
            # a bool [B,1,S,S] pad mask (True = blocked, the exact additive
            # min-score-mask equivalent); FA runs the GQA expansion natively.
            # Pure-math probe: relL2 0.0023 at every B/S of the case pool;
            # eager-chain probe x2.05 at B1/S99.
            import torch_npu
            bool_mask = (mask < 0).expand(batch, 1, width, width)
            out = torch_npu.npu_fusion_attention(
                q_embed, k_embed, v, q_embed.shape[1], "BNSD",
                atten_mask=bool_mask, scale=float(attn.scaling), keep_prob=1.0,
                pre_tockens=2147483647, next_tockens=2147483647,
                sparse_mode=0)[0]
        else:
            groups = attn.num_key_value_groups
            k_embed = k_embed.unsqueeze(2).expand(batch, -1, groups, -1, -1).reshape(batch, -1, width, attn.head_dim)
            v = v.unsqueeze(2).expand(batch, -1, groups, -1, -1).reshape(batch, -1, width, attn.head_dim)
            scores = torch.matmul(q_embed, k_embed.transpose(-1, -2)) * attn.scaling
            scores = scores + mask
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().reshape(batch, width, -1)
        out = out * torch.sigmoid(gate)
        return attn.o_proj(out)


def _model_body(model):
    inner = getattr(model, "model", model)
    return getattr(inner, "language_model", inner)


def _letters_from_model(source):
    folder = Path(getattr(source, "name_or_path", ""))
    decision = json.loads((folder / "decision_config.json").read_text())
    letters = decision["letter_token_ids"]
    if len(letters) != 26 or len(set(letters)) != 26:
        raise ValueError("decision_config letter_token_ids must be 26 unique ids")
    return letters


def adapt(source, config):
    """Build the candidate wrapper on the supplied NPU BF16 model in place-shared fashion."""
    params = _params(config)
    if source.dtype != torch.bfloat16:
        raise ValueError("candidate source must be loaded as bfloat16")
    body = _model_body(source)
    if source.get_output_embeddings().weight.data_ptr() != \
            body.embed_tokens.weight.data_ptr():
        raise ValueError("candidate requires the tied output head: "
                         "source.get_output_embeddings().weight must share "
                         "storage with body.embed_tokens.weight; untied "
                         "checkpoints are not supported")
    report = describe(config)
    letters = report["reference_params"]["letter_token_ids"]
    if letters != _letters_from_model(source):
        raise ValueError("decision_config letter_token_ids from source.ref and from the loaded "
                         "model folder disagree; refusing ambiguous head columns")
    wrapper = StartLuxTextDecoder(body, letters, params["chunk_size"])
    wrapper.width = params["compiled_seq_len"]
    wrapper.requires_grad_(False).eval()
    report["restore"] = _restore_and_verify(body, wrapper, source, params)
    report["weight_storage"] = "original HF modules shared by the wrapper; head26 fp32 clone"
    return wrapper, report


def _restore_and_verify(body, wrapper, source, params):
    from safetensors import safe_open

    folder = Path(source.name_or_path)
    shard = next(folder.glob("*.safetensors"))
    a_log_report, dt_report = {}, {}
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for index, layer in enumerate(body.layers):
            if layer.layer_type != "linear_attention":
                continue
            prefix = f"model.language_model.layers.{index}.linear_attn."
            raw = handle.get_tensor(prefix + "A_log")
            layer.linear_attn.A_log.data = raw.to(layer.linear_attn.A_log.device)
            exact = bool(torch.equal(layer.linear_attn.A_log.detach().cpu(), raw))
            a_log_report[f"layers.{index}"] = {
                "checkpoint_dtype": str(raw.dtype), "restored_dtype": str(layer.linear_attn.A_log.dtype),
                "bit_exact": exact}
            if not exact:
                raise RuntimeError(f"layers.{index}.A_log restore is not bit exact")
            dt_raw = handle.get_tensor(prefix + "dt_bias")
            dt = layer.linear_attn.dt_bias.detach().cpu()
            dt_report[f"layers.{index}"] = {
                "checkpoint_dtype": str(dt_raw.dtype), "loaded_dtype": str(dt.dtype),
                "bit_exact_upcast": bool(torch.equal(dt.float(), dt_raw.float()))}
        embed_raw = handle.get_tensor("model.language_model.embed_tokens.weight")
    head_reference = embed_raw.index_select(0, torch.tensor(wrapper.letters26)).float()
    head_exact = bool(torch.equal(wrapper.head_rows.detach().cpu(), head_reference))
    if not head_exact:
        raise RuntimeError("head_rows are not the exact checkpoint letter rows")
    tied = bool(source.get_output_embeddings().weight.data_ptr()
                == body.embed_tokens.weight.data_ptr())
    if not tied:
        raise ValueError("source output head must be tied to the input embedding")
    return {
        "a_log_restore": {"layers": len(a_log_report), "per_layer": a_log_report,
                          "all_bit_exact_f32": all(v["bit_exact"] for v in a_log_report.values())},
        "dt_bias": {"per_layer": dt_report,
                    "stored_bf16_upcast_only": all(v["checkpoint_dtype"] == "torch.bfloat16"
                                                   and v["bit_exact_upcast"] for v in dt_report.values())},
        "head26": {"source": "model.language_model.embed_tokens.weight rows (tied head)",
                   "rows": len(wrapper.letters26), "dtype": "float32 clone",
                   "bit_exact_vs_checkpoint": head_exact,
                   "embedding_shape": wrapper.embedding_shape,
                   "embedding_truncated": False,
                   "head_tied_in_source": tied},
        "g": "fp32 (-exp(A_log_f32) * softplus(a.float() + dt_bias_bf16)); softplus via where(x>20, x, log1p(exp(clamp(x,max=20))))",
        "rope": {"theta": float(body.config.rope_parameters["rope_theta"]),
                 "rotary_dim": wrapper.rotary_dim,
                 "text_positions_only": True},
        "compiled_width": params["compiled_seq_len"],
        "max_seq_len": params["max_seq_len"],
    }


def _rows_of(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"payload {name} must be a nonempty list")
    if isinstance(value[0], list):
        return value
    return [value]


def _payload_rows(case, params, vocab_size=None):
    """Strict per-row readout validation; returns (rows, lengths).

    Each row is one full prompt: row length == its true length, positions are
    arange(Li), token ids are true ints inside the vocabulary, and
    actual_seq_lengths are per-row true lengths (never cumulative ends).
    """
    payload = case.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"readout {case['id']} payload must be a mapping")
    for name in ("input_ids", "position_ids", "actual_seq_lengths"):
        if name not in payload:
            raise ValueError(f"readout {case['id']} payload misses {name}")
    batch = case["samples"]
    if type(batch) is not int or batch < 1:
        raise ValueError(f"readout {case['id']} samples must be a positive integer")
    rows = _rows_of(payload["input_ids"], "input_ids")
    positions = _rows_of(payload["position_ids"], "position_ids")
    lengths = payload["actual_seq_lengths"]
    if not isinstance(lengths, list) or len(lengths) != batch:
        raise ValueError(f"readout {case['id']} actual_seq_lengths must list {batch} per-row true lengths")
    if len(rows) != batch:
        raise ValueError(f"readout {case['id']} input_ids must hold {batch} row(s) (got {len(rows)})")
    if len(positions) != batch:
        raise ValueError(f"readout {case['id']} position_ids must hold {batch} row(s) (got {len(positions)})")
    for row in rows:
        if not isinstance(row, list) or not row:
            raise ValueError(f"readout {case['id']} has an empty input_ids row")
    for index in range(batch):
        length = lengths[index]
        if type(length) is not int or length < 1:
            raise ValueError(f"readout {case['id']} row {index} needs a positive integer true length")
        if length > params["max_seq_len"]:
            raise ValueError(f"readout {case['id']} row {index} length {length} exceeds "
                             f"max_seq_len {params['max_seq_len']}")
        if len(rows[index]) != length:
            raise ValueError(f"readout {case['id']} row {index} must carry its full prompt: "
                             f"{len(rows[index])} tokens != true length {length}")
        row_positions = positions[index]
        if len(row_positions) != length or any(type(p) is not int for p in row_positions):
            raise ValueError(f"readout {case['id']} row {index} position_ids must be true integers")
        if row_positions != list(range(length)):
            raise ValueError(f"readout {case['id']} row {index} position_ids must be arange({length})")
        for token in rows[index]:
            if type(token) is not int:
                raise ValueError(f"readout {case['id']} row {index} has a non-integer token id")
            if token < 0:
                raise ValueError(f"readout {case['id']} row {index} has a negative token id")
            if vocab_size is not None and token >= vocab_size:
                raise ValueError(f"readout {case['id']} row {index} token {token} is out of "
                                 f"vocabulary bounds [0, {vocab_size})")
    return rows, [int(x) for x in lengths]


def _pad_batch(rows, lengths, width, dtype):
    batch = len(rows)
    arrays = np.zeros((batch, width), dtype=dtype)
    positions = np.zeros((batch, width), dtype=dtype)
    for index, (row, length) in enumerate(zip(rows, lengths)):
        arrays[index, :length] = row
        positions[index, :length] = np.arange(length, dtype=dtype)
    return arrays, positions


def _encode_dynamic(case, params, vocab_size=None):
    """Dynamic B/S route: samples rows right-padded to S = max(true lengths)."""
    rows, lengths = _payload_rows(case, params, vocab_size)
    batch = case["samples"]
    if batch > params["max_batch"]:
        raise ValueError(f"case {case['id']} has samples={batch} > max_batch {params['max_batch']}")
    width = max(lengths)
    ids, positions = _pad_batch(rows, lengths, width, np.int64)
    return {"input_ids": ids, "position_ids": positions,
            "actual_seq_lengths": np.asarray(lengths, dtype=np.int32)}


def _alignment(case, letters26):
    return {"rows": [case["id"]] * case["samples"], "columns": list(letters26)}


class RuntimeHandle:
    """Array-only execution recipe: no weights, no NPU context.

    The alignment contract is provenance-free: rows + letter columns only.
    Business metadata (question_type, golden_file, golden scores) may ride in
    the payload but the graph recipe never reads it.
    """

    def __init__(self, config, io, report=None):
        self.params, self.io = _params(config), io
        self.report = report if report is not None else describe(config)
        self.letters26 = self.report["reference_params"]["letter_token_ids"]
        self.vocab_size = self.report["reference_params"]["vocab_size"]

    def encode(self, case):
        return _encode_dynamic(case, self.params, self.vocab_size)

    def decode(self, outputs, case):
        scores = outputs["scores"]
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (case["samples"], len(self.letters26)):
            raise ValueError(f"readout {case['id']} scores must be "
                             f"({case['samples']}, {len(self.letters26)}), got {scores.shape}")
        return {"outputs": {"scores": scores},
                "alignment": _alignment(case, self.letters26)}

    def output_capacities(self, case):
        node = self.io["outputs"][0]
        return {"scores": case["samples"] * node["shape"][-1] * 4}


class CandidateHandle(RuntimeHandle):
    def __init__(self, wrapper, config, report):
        inputs, outputs = _dynamic_io(_params(config))
        super().__init__(config, {"schema": "ge-io/1", "outputs": outputs}, report)
        self.wrapper = wrapper
        self.device = next(wrapper.parameters()).device

    def encode(self, case):
        return {name: torch.from_numpy(value).to(self.device)
                for name, value in _encode_dynamic(case, self.params, self.vocab_size).items()}

    def run(self, inputs):
        with torch.inference_mode():
            return {"scores": self.wrapper(**inputs)}

    def export_spec(self):
        # NOTE: dynamo_export(dynamic=True) lifts every Python float module
        # attribute (norm eps, attn scaling, min_score, ...) into implicit f64
        # graph inputs unless specialize_float is enabled for the export; the
        # local compile-call context for that lives in the core exporter (see
        # core/exporters/torchair.py), not here.
        inputs, _ = _dynamic_io(self.params)
        # The deterministic B2/S65 trace example: dynamo specializes size-1
        # dimensions, so the first fixture case (always B1 in the default pool)
        # could never keep the batch axis symbolic. This example is pure
        # geometry -- zeros ids, arange positions, per-row width lengths --
        # independent of any fixture case or golden; the core exporter
        # strictly validates it against the declared io and prefers it over
        # the passed inputs, so the regular CLI build traces the dynamic
        # graph. Caps below 2 make the corresponding axis static instead
        # (see _dynamic_axes); the exporter then rejects an example that
        # would specialize a declared-dynamic axis.
        batch, width = _example_geometry(self.params)
        example = {"input_ids": torch.zeros((batch, width), dtype=torch.int64, device=self.device),
                   "position_ids": torch.arange(width, dtype=torch.int64, device=self.device)
                                    .repeat(batch, 1),
                   "actual_seq_lengths": torch.full((batch,), width, dtype=torch.int32,
                                                    device=self.device)}
        return {"format": "torch", "model": self.wrapper,
                "io": {**self.io, "inputs": inputs},
                "example_inputs": example}

    def output_capacities(self, case):
        return {"scores": case["samples"] * 26 * 4}


class ReferenceHandle:
    """Official vendored bidirectional forward on the clean FP32 CPU model (P2b recipe)."""

    def __init__(self, source, config):
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
        import startlux_decide.backends.bidirectional as bidirectional

        self.params = _params(config)
        self.letters26 = _letters_from_model(source)
        self.body = _model_body(source)
        self.head = source.get_output_embeddings().weight
        self.bidirectional = bidirectional
        self.check = bidirectional.enable(self.body)
        decision = _decision_config(config)
        if decision["letter_token_ids"] != self.letters26:
            raise ValueError("source.ref decision_config letters disagree with the loaded model")
        self.vocab_size = decision["vocab_size"]

    def encode(self, case):
        rows, _lengths = _payload_rows(case, self.params, self.vocab_size)
        return {"input_ids": [torch.tensor([row], dtype=torch.int64) for row in rows]}

    def run(self, inputs):
        rows26 = self.head.index_select(0, torch.tensor(self.letters26, device=self.head.device))
        scores = []
        with torch.inference_mode():
            for row in inputs["input_ids"]:
                hidden = self.bidirectional.text_forward(self.body, row, attention_mask=None)
                scores.append(hidden[:, -1].float() @ rows26.float().transpose(0, 1))
        return {"scores": torch.cat(scores, dim=0)}

    def decode(self, outputs, case):
        scores = outputs["scores"]
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (case["samples"], len(self.letters26)):
            raise ValueError(f"readout {case['id']} scores must be "
                             f"({case['samples']}, {len(self.letters26)}), got {scores.shape}")
        return {"outputs": {"scores": scores},
                "alignment": _alignment(case, self.letters26)}

    def output_capacities(self, case):
        return {"scores": case["samples"] * 26 * 4}


def prepare_reference(source, config):
    return ReferenceHandle(source, config)


def prepare_candidate(source, config):
    wrapper, report = adapt(source, config)
    return CandidateHandle(wrapper, config, report)


def prepare_runtime(config, io):
    return RuntimeHandle(config, io)
