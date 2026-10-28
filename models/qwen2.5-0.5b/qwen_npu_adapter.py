"""Qwen2 HF method patches for isolated NPU inference/export workers.

adapt() modifies the supplied model in place and patches HF methods process-wide.
Generate original golden outputs in a separate clean process before adaptation.
DecoderLayer, MLP, embeddings and projections remain the original HF modules.
Deployment loads the exported AIR and does not execute these Python patches.
No project imports, KV-cache, training or sliding-window attention are supported.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers.models.qwen2 import modeling_qwen2 as hf


def _params(config):
    params = {"attention": "fia", "prune_output": False,
              **config["model"]["params"]}
    allowed = {"score_token_ids_file", "attention", "prune_output", "max_seq_len"}
    if set(params) - allowed:
        raise ValueError(f"unknown Qwen model.params: {set(params)-allowed}")
    if params["attention"] not in ("fia", "prefix_910", "prefix_a5"):
        raise ValueError(f"unknown attention implementation: {params['attention']}")
    if type(params.get("max_seq_len")) is not int or params["max_seq_len"] < 1:
        raise ValueError("max_seq_len must be a positive integer")
    return params


def _bounds_dtype(params):
    return np.int32 if params["attention"] == "prefix_a5" else np.int64


def _tokens(params):
    path = Path(params["score_token_ids_file"])
    if not path.is_absolute():
        path = Path(__file__).parent / path
    value = json.loads(path.read_text())
    ids = value["token_ids"] if isinstance(value, dict) else value
    if not ids or any(type(x) is not int or x < 0 for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("score token IDs must be nonempty, unique, nonnegative integers")
    return ids, str(path.resolve())


def describe(config):
    params = _params(config)
    ids, path = _tokens(params)
    kind = params["attention"]
    requirements = [] if kind == "fia" else ["prefix_attention_a5" if kind == "prefix_a5" else "prefix_attention_910"]
    return {"version": 2, "global_patch": True, "in_place": True,
            "patches": list(PATCH_TARGETS), "required_dependencies": requirements,
            "referenced_files": [path], "reference_params": {"score_token_ids": ids},
            "changes": ["HF method patches", "RMSNorm/RoPE/attention fusion", "packed input and selected output head"]}


def rms_norm_forward(self, hidden_states):
    import torch_npu
    return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]


def rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    import torch_npu
    return torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin, layout="TND", rotary_mode="half")


def _init_attention(model, kind, family):
    """Resolve the platform operator once before patching/exporting the model."""
    weight = next(model.parameters())
    if weight.device.type != "npu":
        raise ValueError("move the original model to the selected NPU before adapt")
    if kind == "prefix_910" and family != "a3":
        raise ValueError("this prefix_910 recipe requires the configured A3 target")
    cfg = model.config
    heads, kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
    if kind == "prefix_a5":
        dims = (heads, kv_heads, model.model.layers[0].self_attn.head_dim)
        if weight.dtype != torch.float16:
            raise ValueError("current A5 prefix delivery supports float16 only")
        if dims != (14, 2, 64):
            raise ValueError(f"current A5 kernel requires (14,2,64), got {dims}")
        if family != "a5":
            raise ValueError("prefix_a5 requires an A5 target")
        import prefix_attn_torch
        op = torch.ops.ascendc_ops.prefix_attention
        return lambda q, k, v, bounds, mask, scale: op(q, k, v, bounds, scale)
    if kind == "prefix_910":
        import npu_prefix_infer_attention_score
        prefix_op = torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score.tensor
    else:
        import torch_npu
        from torchair.ops import npu_fused_infer_attention_score as graph_fia
        eager_fia = torch_npu.npu_fused_infer_attention_score

    def attention(q, k, v, bounds, mask, scale):
        arguments = dict(num_heads=heads, scale=scale, num_key_value_heads=kv_heads,
                         actual_seq_lengths=bounds, actual_seq_lengths_kv=bounds,
                         atten_mask=mask, sparse_mode=2)
        if kind == "prefix_910":
            return prefix_op(q, k, v, pre_tokens=2147483647, next_tokens=0, **arguments)
        op = graph_fia if torch.compiler.is_compiling() else eager_fia
        return op(q, k, v, input_layout="TND", **arguments)[0]
    return attention


def attention_forward(self, hidden_states, position_embeddings, attention_mask,
                      actual_seq_lengths, **kwargs):
    """Keep the HF projections and call the operator bound during adaptation."""
    heads, kv_heads = self.config.num_attention_heads, self.config.num_key_value_heads
    q = self.q_proj(hidden_states).reshape(-1, heads, self.head_dim)
    k = self.k_proj(hidden_states).reshape(-1, kv_heads, self.head_dim)
    v = self.v_proj(hidden_states).reshape(-1, kv_heads, self.head_dim)
    q, k = hf.apply_rotary_pos_emb(q, k, *position_embeddings)
    q = q.reshape(-1, heads, self.head_dim)
    k = k.reshape(-1, kv_heads, self.head_dim)
    out = self._ge_attention_op(q, k, v, actual_seq_lengths, attention_mask, self.scaling)
    return self.o_proj(out.reshape(-1, self.config.hidden_size).contiguous()), None


def packed_model_forward(self, input_ids, position_ids, actual_seq_lengths):
    """Patch Qwen2Model.forward for packed input; reuse HF decoder/MLP calls."""
    x = self.embed_tokens(input_ids)
    position_embeddings = (self._ge_cos.index_select(0, position_ids).unsqueeze(1),
                           self._ge_sin.index_select(0, position_ids).unsqueeze(1))
    bounds = actual_seq_lengths
    for layer in self.layers:
        x = layer(x, attention_mask=self._ge_mask, position_embeddings=position_embeddings,
                  actual_seq_lengths=bounds)
    ends = actual_seq_lengths[1:] if self._ge_attention != "fia" else actual_seq_lengths
    return self.norm(x.index_select(0, ends - 1))


def score_forward(self, input_ids, position_ids, actual_seq_lengths):
    """Patch Qwen2ForCausalLM.forward: one score row per business request."""
    return self.lm_head(self.model(input_ids, position_ids, actual_seq_lengths))


packed_model_forward._ge_qwen_patch = True


PATCHES = (
    (hf.Qwen2RMSNorm, "forward", rms_norm_forward),
    (hf, "apply_rotary_pos_emb", rotary_pos_emb),
    (hf.Qwen2Attention, "forward", attention_forward),
    (hf.Qwen2Model, "forward", packed_model_forward),
    (hf.Qwen2ForCausalLM, "forward", score_forward),
)
PATCH_TARGETS = tuple(name if owner is hf else f"{owner.__name__}.{name}"
                      for owner, name, _ in PATCHES)


def apply_patches():
    """Assign methods from the same table used to describe the patches."""
    for owner, name, implementation in PATCHES:
        setattr(owner, name, implementation)


def _require_clean_reference():
    if getattr(hf.Qwen2Model.forward, "_ge_qwen_patch", False):
        raise RuntimeError("original golden requires a clean process before applying Qwen patches")


def adapt(raw_model, config):
    """Prepare the supplied HF model in place, then install the process-wide patches."""
    if getattr(raw_model, "_ge_qwen_adapted", False):
        raise ValueError("model is already adapted; load an original model for a new candidate")
    if not isinstance(raw_model, hf.Qwen2ForCausalLM):
        raise ValueError("this patch recipe requires the HF Qwen2ForCausalLM model")
    params = _params(config)
    cfg = raw_model.config
    if cfg.model_type != "qwen2" or cfg.hidden_act != "silu" or any(layer.self_attn.sliding_window is not None for layer in raw_model.model.layers):
        raise ValueError("recipe supports standard full-attention Qwen2 with SiLU only")
    rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("dynamic/scaled RoPE requires a separate validated recipe")
    report = describe(config)
    ids = report["reference_params"]["score_token_ids"]
    if max(ids) >= raw_model.lm_head.out_features:
        raise ValueError("score token outside model vocabulary")
    kind = params["attention"]
    attention_op = _init_attention(raw_model, kind, config["target"]["family"])
    for layer in raw_model.model.layers:
        layer.self_attn._ge_attention_op = attention_op
    model, rope = raw_model.model, raw_model.model.rotary_emb
    model._ge_attention = kind
    positions = torch.arange(params["max_seq_len"], device=rope.inv_freq.device).float()
    frequencies = positions[:, None] * rope.inv_freq.detach().float()[None, :]
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    weight = model.embed_tokens.weight
    model.register_buffer("_ge_cos", (frequencies.cos() * rope.attention_scaling).to(weight), persistent=False)
    model.register_buffer("_ge_sin", (frequencies.sin() * rope.attention_scaling).to(weight), persistent=False)
    # sparse_mode=2 uses a fixed compressed causal template on NPU.
    mask_size = 2048
    model.register_buffer("_ge_mask", torch.triu(torch.ones(mask_size, mask_size, dtype=torch.bool,
                                                           device=weight.device), diagonal=1), persistent=False)
    if params["prune_output"]:
        head = raw_model.lm_head
        index = torch.tensor(ids, device=head.weight.device)
        head.weight = nn.Parameter(head.weight.index_select(0, index).clone(), requires_grad=False)
        if head.bias is not None:
            head.bias = nn.Parameter(head.bias.index_select(0, index).clone(), requires_grad=False)
        head.out_features = len(ids)
    raw_model.requires_grad_(False).eval()
    apply_patches()
    raw_model._ge_qwen_adapted = True
    return raw_model, {**report, "weight_storage": "original HF modules; selected head copied", "attention": kind}


def _encode_inputs(case, params, padded=False):
    """Shared NumPy encoding for reference, export and AIR execution."""
    prefix, requests = case["payload"]["prefix"], case["payload"]["requests"]
    if not requests or any(not request["tokens"] for request in requests):
        raise ValueError("every Qwen request needs its own nonempty token sequence")
    if case["samples"] != len(requests):
        raise ValueError("business sample count differs from request count")
    if len({request["id"] for request in requests}) != len(requests):
        raise ValueError("Qwen request IDs must be unique within a case")
    rows = [prefix + request["tokens"] for request in requests]
    if max(map(len, rows)) > params["max_seq_len"]:
        raise ValueError("request exceeds max_seq_len")
    if padded:
        width = max(map(len, rows))
        return {"input_ids": np.asarray([row + [0]*(width-len(row)) for row in rows], dtype=np.int64),
                "attention_mask": np.asarray([[1]*len(row) + [0]*(width-len(row)) for row in rows], dtype=np.int64),
                "position_ids": np.asarray([list(range(len(row))) + [0]*(width-len(row)) for row in rows], dtype=np.int64)}
    if params["attention"] == "fia":
        segments, positions = rows, [list(range(len(row))) for row in rows]
    else:
        if not prefix:
            raise ValueError("prefix recipe needs a nonempty prefix")
        segments = [prefix] + [request["tokens"] for request in requests]
        positions = [list(range(len(prefix)))] + [list(range(len(prefix), len(prefix)+len(row))) for row in segments[1:]]
    bounds = np.cumsum([len(row) for row in segments], dtype=np.int64)
    dtype = _bounds_dtype(params)
    if dtype == np.int32 and bounds[-1] > np.iinfo(np.int32).max:
        raise ValueError("packed bounds exceed int32 range")
    return {"input_ids": np.asarray(sum(segments, []), dtype=np.int64),
            "position_ids": np.asarray(sum(positions, []), dtype=np.int64),
            "actual_seq_lengths": bounds.astype(dtype, copy=False)}


class RuntimeHandle:
    """Array-only execution recipe: no weights, NPU context or HF patch needed."""
    def __init__(self, config, io, report=None):
        self.params, self.io = _params(config), io
        self.report = report if report is not None else describe(config)
        self.ids = self.report["reference_params"]["score_token_ids"]

    def encode(self, case):
        return _encode_inputs(case, self.params)

    def decode(self, outputs, case):
        scores = outputs["scores"]
        if not self.params["prune_output"]:
            scores = scores[..., self.ids]
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        return {"outputs": {"scores": scores},
                "alignment": {"rows": [r["id"] for r in case["payload"]["requests"]], "columns": self.ids}}

    def output_capacities(self, case):
        node = self.io["outputs"][0]
        return {"scores": case["samples"] * node["shape"][-1] * np.dtype(node["dtype"]).itemsize}


class Handle(RuntimeHandle):
    def __init__(self, model, config, candidate, report=None):
        head = model.lm_head.weight
        output = {"name": "scores", "dtype": str(head.dtype).removeprefix("torch."),
                  "shape": [-1, head.shape[0]], "dynamic_dims": [0]}
        super().__init__(config, {"schema": "ge-io/1", "outputs": [output]}, report)
        self.model, self.candidate = model, candidate
        self.device = next(model.parameters()).device

    def encode(self, case):
        values = _encode_inputs(case, self.params, padded=not self.candidate)
        return {name: torch.tensor(value, device=self.device) for name, value in values.items()}

    def run(self, inputs):
        with torch.inference_mode():
            if not self.candidate:
                _require_clean_reference()
                # Select only needed positions without materializing batch*seq*vocab logits.
                positions = (inputs["attention_mask"].sum(dim=1) - 1).unique(sorted=True)
                return {"logits": self.model(**inputs, use_cache=False, logits_to_keep=positions).logits,
                        "positions": positions.detach().cpu().tolist()}
            return {"scores": self.model(**inputs)}

    def decode(self, outputs, case):
        if self.candidate:
            return super().decode(outputs, case)
        logits = outputs["logits"]
        if hasattr(logits, "detach"):
            logits = logits.detach().float().cpu().numpy()
        requests, prefix = case["payload"]["requests"], case["payload"]["prefix"]
        positions = {value: index for index, value in enumerate(outputs["positions"])}
        scores = np.stack([logits[i, positions[len(prefix)+len(r["tokens"])-1], self.ids] for i, r in enumerate(requests)])
        return {"outputs": {"scores": scores},
                "alignment": {"rows": [r["id"] for r in requests], "columns": self.ids}}

    def output_capacities(self, case):
        head = self.model.lm_head.weight
        return {"scores": case["samples"] * head.shape[0] * head.element_size()}

    def export_spec(self):
        inputs = [{"name": name, "dtype": (np.dtype(_bounds_dtype(self.params)).name
                                          if name == "actual_seq_lengths" else "int64"),
                   "shape": [-1], "dynamic_dims": [0]}
                  for name in ("input_ids", "position_ids", "actual_seq_lengths")]
        return {"format": "torch", "model": self.model, "io": {**self.io, "inputs": inputs}}


def prepare_reference(source, config):
    _require_clean_reference()
    return Handle(source, config, False)


def prepare_candidate(source, config):
    model, report = adapt(source, config)
    return Handle(model, config, True, report)


def prepare_runtime(config, io):
    return RuntimeHandle(config, io)
