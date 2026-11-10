"""StartLux-Decision v1.1 0.8B Instant text decoder: instance wrappers only, no process-wide patches.

A model-specific ``StartLuxTextDecoder`` nn.Module reusing the loaded HF modules
(embeddings, projections, MLP, norms, conv) with
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
  1e-6)`` exactly, kernel-internal fp32 reduction, bf16 output); the kernel
  ``scale = Dk**-0.5`` is UNCHANGED -- the delta update couples q/k magnitudes
  nonlinearly, so magnitude folding must stay in gamma, not in the scale;
- v/beta stay BF16, g stays FP32
  (``-exp(A_log_f32) * softplus(a.float() + dt_bias_bf16)``);
- every row starts from a zero initial_state ``[B, Hv, Dv, Dk]`` BF16; there is
  no state cache and the final state is never reused (Instant readout only);
- the wrapper runs on the padded ``[B, S]`` layout, so the flattened ``T = B*S``
  rows pass each row's PHYSICAL width S as ``actual_seq_lengths`` (the CANN
  9.1.0 kernel expects per-row lengths, never cumulative ends, never the true
  L); positions beyond the true length are NOT masked: the chunk recurrence is
  strictly causal, so padded tokens only ever influence outputs at their own
  and later positions and the final state -- the readout gathers the true L-1
  and the state is discarded, so per-row padding never leaks;
- num_k_heads must equal num_v_heads (the kernel layout has no GQA repeat).

Export requires the ``chunk_gated_delta_rule_proto`` dependency (project-built
GE AscendIR prototype registration), loaded by a single LD_PRELOAD contract:
run.sh injects the .so into the export / infer / bench subprocesses and
export_air.py asserts it at startup; auto vendor loading is rejected.  Eager NPU
execution dispatches to the CANN aclnn host implementation and needs no
prototype library.  CANN upgrades or SoC changes require re-running the
prototype gate: the native route is the only one, there is no standard-op
fallback.

A_log is restored bit-exact F32 from the checkpoint after the BF16 load;
dt_bias stays its original BF16.

Known GE-converter-driven deviations (numerically justified):
- F.softplus -> where(x > 20, x, log1p(exp(clamp(x, max=20)))): aten.softplus
  has no GE converter; the expression is the exact PyTorch definition
  (beta=1, threshold=20) on aten ops GE covers (ge.Log1p), verified equal to
  F.softplus in tests.
- torch.split(list) -> explicit slices: aten.split.sizes has no GE converter.
- interleaved MRoPE cos/sin computed in-graph from the official [3, B, S]
  get_rope_index planes with the official mrope_section layout
  (``mrope_cos_sin``), consumed by the unified multimodal AIR (multimodal.py).
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _decision_config(folder):
    """The 26 letter ids and the vocab bound, read from the model folder itself
    (decision_config.json + config.json) -- never from fixture provenance."""
    folder = Path(folder)
    letters = json.loads((folder / "decision_config.json").read_text())["letter_token_ids"]
    if len(letters) != 26 or len(set(letters)) != 26:
        raise ValueError("decision_config letter_token_ids must be 26 unique ids")
    model_config = json.loads((folder / "config.json").read_text())
    vocab = (model_config.get("text_config") or {}).get("vocab_size",
                                                        model_config.get("vocab_size"))
    return {"letter_token_ids": letters, "vocab_size": vocab}


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
    DIRECTLY in bf16: the kernel's reduction runs in fp32 internally, and an
    fp32 gamma would get compiler-inserted [1024] fp32->bf16 Casts at GE
    tiling anyway (the delivered kernel consumes bf16 gamma), so bf16 keeps the
    math identical and drops 48 Casts per request.  Eager leg keeps the
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

    The FP32 input is required, not a precision preference: on this build the
    bf16-input Conv1D and Conv2D nodes both miscompile this configuration
    (C=6144 / groups=C / kernel=4), while the fp32-input Conv2D node is
    numerically correct up to the expected bf16 output quantization.  It is
    also faster in GE than an explicit 4-tap shift chain.  Kernel size is fixed
    at 4 per the StartLux linear_conv_kernel_dim.
    """
    channels, seq = x.shape[1], x.shape[2]
    w = weight.float().reshape(channels, 1, 1, 4)
    y = F.conv2d(x.float().unsqueeze(2), w, groups=channels, padding=(0, 3))
    return F.silu(y[..., :seq].squeeze(2).to(x.dtype))


class StartLuxTextDecoder(nn.Module):
    """Bidirectional text forward of the StartLux decision decoder, one readout per row."""

    def __init__(self, body, letters26):
        super().__init__()
        self.body = body
        self.letters26 = list(letters26)
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
        # A module-level constant so it constant-folds once in the exported
        # graph instead of emitting a per-layer torch.full node, and follows
        # .to(device) like every buffer.  bf16 is the kernel's gamma dtype.
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

    def forward_embeddings(self, inputs_embeds, position_ids, actual_seq_lengths):
        """Shared decoder loop over precomputed input embeddings.

        ``position_ids`` carries the [3, B, S] multimodal MRoPE planes of the
        official get_rope_index.  The unified multimodal AIR injects image
        features into ``inputs_embeds`` (equivalent to the official
        masked_scatter) and calls this loop.
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
        """Official interleaved MRoPE cos/sin (Qwen3_5TextRotaryEmbedding) from
        the [3, B, S] get_rope_index planes.  Returns (cos, sin) [B, S, rotary_dim].
        """
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq      # [3,B,S,half]
        freqs = freqs.permute(1, 2, 0, 3).flatten(2)                              # [B,S,3*half]
        freqs = freqs.index_select(-1, self.mrope_gather)                         # [B,S,half]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _gated_delta(self, attn, hidden, lengths):
        batch, width, _ = hidden.shape
        mixed_qkv = attn.in_proj_qkv(hidden).transpose(1, 2)
        z = attn.in_proj_z(hidden).reshape(batch, width, -1, attn.head_v_dim)
        b = attn.in_proj_b(hidden)
        a = attn.in_proj_a(hidden)
        mixed = causal_conv4(mixed_qkv, attn.conv1d.weight).transpose(1, 2)
        key_dim = attn.key_dim
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
        # eps/D) == x * rsqrt(sum(x^2)+eps) == l2norm(x, eps) EXACTLY -- the
        # sqrt(D) of the mean-domain folding cancels in gamma (kernel-internal
        # fp32 reduction, bf16 output; the bf16 gamma carries ~1e-4 relative
        # rounding).  The kernel scale stays Dk**-0.5 UNCHANGED: the delta
        # update couples q/k magnitudes nonlinearly, so folding sqrt(D) into
        # the scale instead of gamma is not equivalent.
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
            # Fused RoPE: the partial-rotary dims go through
            # npu_apply_rotary_pos_emb in the native BSND layout (q/k
            # [B,S,N,rot], cos/sin [B,S,1,rot], zero reshapes; rotary_mode
            # "half" is the exact rotate-half math).  The aclnn EAGER leg's
            # layout=1 is NOT BSND (it reads [B,S,N,D] as BSH), so the eager
            # branch keeps explicit math.
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
            # Fused attention: one npu_fusion_attention (FA) per layer replaces
            # the eager chain (GQA k/v expand, scores matmul + additive mask,
            # fp32 softmax, probs matmul) -- bf16 BNSD with a bool [B,1,S,S]
            # pad mask (True = blocked, the exact additive min-score-mask
            # equivalent); FA runs the GQA expansion natively.
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


def _restore_and_verify(body, wrapper, source):
    """Restore the checkpoint's F32 A_log over the BF16 load and prove the
    numeric contracts the decoder depends on: A_log bit-exact F32, head26 the
    exact checkpoint letter rows, and the output head tied to the input
    embedding (head26 is cloned from embed_tokens, so an untied checkpoint
    would silently project the wrong rows).
    """
    from safetensors import safe_open

    folder = Path(source.name_or_path)
    shard = next(folder.glob("*.safetensors"))
    with safe_open(shard, framework="pt", device="cpu") as handle:
        for index, layer in enumerate(body.layers):
            if layer.layer_type != "linear_attention":
                continue
            raw = handle.get_tensor(f"model.language_model.layers.{index}.linear_attn.A_log")
            layer.linear_attn.A_log.data = raw.to(layer.linear_attn.A_log.device)
            if not torch.equal(layer.linear_attn.A_log.detach().cpu(), raw):
                raise RuntimeError(f"layers.{index}.A_log restore is not bit exact")
        embed_raw = handle.get_tensor("model.language_model.embed_tokens.weight")
    head_reference = embed_raw.index_select(0, torch.tensor(wrapper.letters26)).float()
    if not torch.equal(wrapper.head_rows.detach().cpu(), head_reference):
        raise RuntimeError("head_rows are not the exact checkpoint letter rows")
    if source.get_output_embeddings().weight.data_ptr() != \
            body.embed_tokens.weight.data_ptr():
        raise ValueError("source output head must be tied to the input embedding")
