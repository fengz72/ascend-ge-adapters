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
import torch.nn.functional as F
from transformers.models.qwen2 import modeling_qwen2 as hf

PATCH_TARGETS = (
    "Qwen2RMSNorm.forward", "apply_rotary_pos_emb", "Qwen2Attention.forward",
    "Qwen2Model.forward", "Qwen2ForCausalLM.forward",
)


def _params(config):
    params = config["model"]["params"]
    allowed = {"score_token_ids_file", "attention", "operators", "prune_output", "max_seq_len"}
    if set(params) - allowed:
        raise ValueError(f"unknown Qwen model.params: {set(params)-allowed}")
    return params


def _tokens(config):
    path = Path(_params(config)["score_token_ids_file"])
    if not path.is_absolute():
        path = Path(__file__).parent / path
    value = json.loads(path.read_text())
    ids = value["token_ids"] if isinstance(value, dict) else value
    if not ids or any(type(x) is not int or x < 0 for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("score token IDs must be nonempty, unique, nonnegative integers")
    return ids, str(path.resolve())


def describe(config):
    ids, path = _tokens(config)
    params = _params(config)
    kind = params.get("attention", "fia")
    if kind not in ("fia", "prefix_910", "prefix_a5"):
        raise ValueError(f"unknown attention implementation: {kind}")
    requirements = [] if params.get("operators", "npu") == "torch" or kind == "fia" else ["prefix_attention_a5" if kind == "prefix_a5" else "prefix_attention_910"]
    return {"version": 2, "global_patch": True, "in_place": True,
            "patches": list(PATCH_TARGETS), "required_dependencies": requirements,
            "referenced_files": [path], "reference_params": {"score_token_ids": ids},
            "changes": ["HF method patches", "RMSNorm/RoPE/attention fusion", "packed input and selected output head"]}


def rms_norm_forward(self, hidden_states):
    """Patch Qwen2RMSNorm.forward: select the fused NPU RMSNorm."""
    if self._ge_npu:
        import torch_npu
        return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
    dtype = hidden_states.dtype
    x = hidden_states.float()
    return self.weight * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)).to(dtype)


def rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Patch HF RoPE for packed TND inputs; CPU path is a semantic oracle."""
    if q.device.type == "npu":
        import torch_npu
        return torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin, layout="TND", rotary_mode="half")
    def rotate(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


def _torch_attention(q, k, v, bounds, prefix):
    # CPU semantic oracle for adapter isolation tests; not the NPU export path.
    ends = bounds.tolist()
    outputs, start = [], 0
    prefix_end = ends[0] if prefix else 0
    for index, end in enumerate(ends):
        query = q[start:end].transpose(0, 1)
        keys, values = k[start:end], v[start:end]
        shared = prefix_end if prefix and index > 0 else 0
        if shared:
            keys = torch.cat((k[:shared], keys), dim=0)
            values = torch.cat((v[:shared], values), dim=0)
        repeat = q.shape[1] // k.shape[1]
        keys = keys.repeat_interleave(repeat, dim=1).transpose(0, 1)
        values = values.repeat_interleave(repeat, dim=1).transpose(0, 1)
        causal = torch.arange(keys.shape[1], device=q.device)[None, :] <= (torch.arange(end-start, device=q.device)[:, None] + shared)
        outputs.append(F.scaled_dot_product_attention(query, keys, values, attn_mask=causal).transpose(0, 1))
        start = end
    return torch.cat(outputs, dim=0)


def attention_forward(self, hidden_states, position_embeddings, attention_mask,
                      actual_seq_lengths, **kwargs):
    """Patch Qwen2Attention.forward; keep the HF Q/K/V/O projection modules."""
    heads, kv_heads = self.config.num_attention_heads, self.config.num_key_value_heads
    q = self.q_proj(hidden_states).reshape(-1, heads, self.head_dim)
    k = self.k_proj(hidden_states).reshape(-1, kv_heads, self.head_dim)
    v = self.v_proj(hidden_states).reshape(-1, kv_heads, self.head_dim)
    q, k = hf.apply_rotary_pos_emb(q, k, *position_embeddings)
    q = q.reshape(-1, heads, self.head_dim)
    k = k.reshape(-1, kv_heads, self.head_dim)
    bounds, kind = actual_seq_lengths, self._ge_attention
    if self._ge_operators == "torch":
        out = _torch_attention(q, k, v, bounds, kind != "fia")
    elif kind == "prefix_a5":
        out = self._ge_prefix_op(q, k, v, bounds, self.scaling)
    else:
        arguments = dict(num_heads=heads, scale=self.scaling,
                         actual_seq_lengths=bounds, actual_seq_lengths_kv=bounds,
                         num_key_value_heads=kv_heads, atten_mask=attention_mask, sparse_mode=2)
        if kind == "prefix_910":
            out = self._ge_prefix_op(q, k, v, pre_tokens=2147483647, next_tokens=0, **arguments)
        else:
            op = self._ge_graph_fia if torch.compiler.is_compiling() else self._ge_eager_fia
            out, _ = op(q, k, v, input_layout="TND", **arguments)
    return self.o_proj(out.reshape(-1, self.config.hidden_size).contiguous()), None


def packed_model_forward(self, input_ids, position_ids, actual_seq_lengths):
    """Patch Qwen2Model.forward for packed input; reuse HF decoder/MLP calls."""
    x = self.embed_tokens(input_ids)
    position_embeddings = (self._ge_cos.index_select(0, position_ids).unsqueeze(1),
                           self._ge_sin.index_select(0, position_ids).unsqueeze(1))
    bounds = actual_seq_lengths.to(torch.int32) if self._ge_attention == "prefix_a5" else actual_seq_lengths
    for layer in self.layers:
        x = layer(x, attention_mask=self._ge_mask, position_embeddings=position_embeddings,
                  actual_seq_lengths=bounds)
    ends = actual_seq_lengths[1:] if self._ge_attention != "fia" else actual_seq_lengths
    return self.norm(x.index_select(0, ends - 1))


def score_forward(self, input_ids, position_ids, actual_seq_lengths):
    """Patch Qwen2ForCausalLM.forward: one score row per business request."""
    return self.lm_head(self.model(input_ids, position_ids, actual_seq_lengths))


packed_model_forward._ge_qwen_patch = True


def apply_patches():
    """The complete patch list. Reapplying assigns methods without wrapping them."""
    hf.Qwen2RMSNorm.forward = rms_norm_forward
    hf.apply_rotary_pos_emb = rotary_pos_emb
    hf.Qwen2Attention.forward = attention_forward
    hf.Qwen2Model.forward = packed_model_forward
    hf.Qwen2ForCausalLM.forward = score_forward


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
    ids, _ = _tokens(config)
    if max(ids) >= raw_model.lm_head.out_features:
        raise ValueError("score token outside model vocabulary")
    if params.get("operators", "npu") not in ("torch", "npu"):
        raise ValueError("operators must be torch or npu")
    kind = params.get("attention", "fia")
    describe(config)
    if type(params.get("max_seq_len")) is not int or params["max_seq_len"] < 1:
        raise ValueError("max_seq_len must be a positive integer")
    if params.get("operators", "npu") == "npu":
        if next(raw_model.parameters()).device.type != "npu":
            raise ValueError("move the original model to the selected NPU before adapt")
        if kind == "prefix_910" and config["target"]["family"] != "a3":
            raise ValueError("this prefix_910 recipe requires the configured A3 target")
    if params.get("operators", "npu") == "npu" and kind == "prefix_a5":
        dims = (cfg.num_attention_heads, cfg.num_key_value_heads, raw_model.model.layers[0].self_attn.head_dim)
        if next(raw_model.parameters()).dtype != torch.float16:
            raise ValueError("current A5 prefix delivery supports float16 only")
        if dims != (14, 2, 64):
            raise ValueError(f"current A5 kernel requires (14,2,64), got {dims}")
        if config["target"]["family"] != "a5":
            raise ValueError("prefix_a5 requires an A5 target")
    operators = params.get("operators", "npu")
    if operators == "npu":
        import torch_npu
        if kind == "fia":
            from torchair.ops import npu_fused_infer_attention_score
        elif kind == "prefix_a5":
            import prefix_attn_torch
        else:
            import npu_prefix_infer_attention_score
    for norm in [raw_model.model.norm] + [norm for layer in raw_model.model.layers
                                          for norm in (layer.input_layernorm, layer.post_attention_layernorm)]:
        norm._ge_npu = operators == "npu"
    for layer in raw_model.model.layers:
        attention = layer.self_attn
        attention._ge_attention, attention._ge_operators = kind, operators
        if operators == "npu":
            if kind == "fia":
                attention._ge_graph_fia = npu_fused_infer_attention_score
                attention._ge_eager_fia = torch_npu.npu_fused_infer_attention_score
            elif kind == "prefix_a5":
                attention._ge_prefix_op = torch.ops.ascendc_ops.prefix_attention
            else:
                attention._ge_prefix_op = torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score.tensor
    model, rope = raw_model.model, raw_model.model.rotary_emb
    model._ge_attention = kind
    positions = torch.arange(params["max_seq_len"], device=rope.inv_freq.device).float()
    frequencies = positions[:, None] * rope.inv_freq.detach().float()[None, :]
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    weight = model.embed_tokens.weight
    model.register_buffer("_ge_cos", (frequencies.cos() * rope.attention_scaling).to(weight), persistent=False)
    model.register_buffer("_ge_sin", (frequencies.sin() * rope.attention_scaling).to(weight), persistent=False)
    # sparse_mode=2 uses a fixed compressed causal template on NPU.
    mask_size = 2048 if operators == "npu" else params["max_seq_len"]
    model.register_buffer("_ge_mask", torch.triu(torch.ones(mask_size, mask_size, dtype=torch.bool,
                                                           device=weight.device), diagonal=1), persistent=False)
    if params.get("prune_output", False):
        head = raw_model.lm_head
        index = torch.tensor(ids, device=head.weight.device)
        head.weight = nn.Parameter(head.weight.index_select(0, index).clone(), requires_grad=False)
        if head.bias is not None:
            head.bias = nn.Parameter(head.bias.index_select(0, index).clone(), requires_grad=False)
        head.out_features = len(ids)
    raw_model.requires_grad_(False).eval()
    apply_patches()
    raw_model._ge_qwen_adapted = True
    return raw_model, {**describe(config), "weight_storage": "original HF modules; selected head copied", "attention": kind}


class Handle:
    def __init__(self, model, config, candidate):
        self.model, self.config, self.candidate = model, config, candidate
        self.device = next(model.parameters()).device
        self.ids, _ = _tokens(config)
        self.prefix = config["model"]["params"].get("attention", "fia") != "fia"

    def encode(self, case):
        prefix, requests = case["payload"]["prefix"], case["payload"]["requests"]
        if not requests or any(not request["tokens"] for request in requests):
            raise ValueError("every Qwen request needs its own nonempty token sequence")
        ids = [request["id"] for request in requests]
        if len(set(ids)) != len(ids):
            raise ValueError("Qwen request IDs must be unique within a case")
        rows = [prefix + request["tokens"] for request in requests]
        if case["samples"] != len(requests):
            raise ValueError("business sample count differs from request count")
        if not rows or any(not row for row in rows):
            raise ValueError("empty Qwen request")
        if max(map(len, rows)) > self.config["model"]["params"]["max_seq_len"]:
            raise ValueError("request exceeds max_seq_len")
        if not self.candidate:
            width = max(map(len, rows))
            ids = [row + [0] * (width-len(row)) for row in rows]
            mask = [[1]*len(row) + [0]*(width-len(row)) for row in rows]
            pos = [list(range(len(row))) + [0]*(width-len(row)) for row in rows]
            return {key: torch.tensor(value, dtype=torch.int64, device=self.device) for key, value in [("input_ids", ids), ("attention_mask", mask), ("position_ids", pos)]}
        if self.prefix:
            if not prefix:
                raise ValueError("prefix recipe needs a nonempty prefix")
            segments = [prefix] + [request["tokens"] for request in requests]
            positions = [list(range(len(prefix)))] + [list(range(len(prefix), len(prefix)+len(request["tokens"]))) for request in requests]
        else:
            segments, positions = rows, [list(range(len(row))) for row in rows]
        values = {"input_ids": sum(segments, []), "position_ids": sum(positions, []), "actual_seq_lengths": np.cumsum([len(x) for x in segments]).tolist()}
        if min(values["input_ids"]) < 0 or max(values["input_ids"]) >= self.model.model.embed_tokens.num_embeddings:
            raise ValueError("input token outside vocabulary")
        return {name: torch.tensor(value, dtype=torch.int64, device=self.device) for name, value in values.items()}

    def run(self, inputs):
        with torch.inference_mode():
            if not self.candidate:
                _require_clean_reference()
                # Ask the unmodified HF API for the positions needed by this
                # task. Avoid materializing batch*seq*vocab logits for the
                # batch=10 pool; all requested vocabulary logits remain intact.
                positions = (inputs["attention_mask"].sum(dim=1) - 1).unique(sorted=True)
                return {"logits": self.model(**inputs, use_cache=False, logits_to_keep=positions).logits,
                        "positions": positions.detach().cpu().tolist()}
            return {"scores": self.model(**inputs)}

    def decode(self, outputs, case):
        requests, prefix = case["payload"]["requests"], case["payload"]["prefix"]
        if self.candidate:
            scores = outputs["scores"]
            if not self.config["model"]["params"].get("prune_output", False):
                scores = scores[..., self.ids]
        else:
            logits = outputs["logits"]
            if hasattr(logits, "detach"):
                logits = logits.detach().float().cpu().numpy()
            positions = {value: index for index, value in enumerate(outputs["positions"])}
            scores = np.stack([logits[i, positions[len(prefix)+len(r["tokens"])-1], self.ids] for i, r in enumerate(requests)])
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        return {"outputs": {"scores": scores}, "alignment": {"rows": [r["id"] for r in requests], "columns": self.ids}}

    def output_capacities(self, case):
        width = self.model.lm_head.weight.shape[0]
        return {"scores": len(case["payload"]["requests"])*width*self.model.lm_head.weight.element_size()}

    def export_spec(self):
        if self.config["model"]["params"].get("operators", "npu") != "npu":
            raise ValueError("Qwen torch operator path is a CPU semantic test oracle, not an NPU exporter")
        inputs = [{"name": name, "dtype": "int64", "shape": [-1], "dynamic_dims": [0]} for name in ("input_ids", "position_ids", "actual_seq_lengths")]
        width = self.model.lm_head.weight.shape[0]
        return {"format": "torch", "model": self.model, "io": {"schema": "ge-io/1", "inputs": inputs, "outputs": [{"name": "scores", "dtype": str(self.model.lm_head.weight.dtype).removeprefix("torch."), "shape": [-1, width], "dynamic_dims": [0]}]}}


def prepare_reference(source, config):
    _require_clean_reference()
    return Handle(source, config, False)


def prepare_candidate(source, config):
    model, report = adapt(source, config)
    result = Handle(model, config, True)
    result.report = report
    return result


class RuntimeHandle:
    """Array-only execution recipe: no weights, NPU context or HF patch needed."""
    def __init__(self, config, io):
        self.config, self.io = config, io
        self.ids, _ = _tokens(config)

    def encode(self, case):
        prefix, requests = case['payload']['prefix'], case['payload']['requests']
        if len(requests) != case['samples'] or not requests or any(not r['tokens'] for r in requests):
            raise ValueError('invalid Qwen case request count/tokens')
        if len({r['id'] for r in requests}) != len(requests):
            raise ValueError('duplicate Qwen request IDs')
        if _params(self.config).get('attention', 'fia') == 'fia':
            segments = [prefix + r['tokens'] for r in requests]
            positions = [list(range(len(row))) for row in segments]
        else:
            if not prefix:raise ValueError('prefix recipe needs a nonempty prefix')
            segments = [prefix] + [r['tokens'] for r in requests]
            positions = [list(range(len(prefix)))] + [list(range(len(prefix), len(prefix)+len(row))) for row in segments[1:]]
        return {'input_ids': np.asarray(sum(segments, []), dtype=np.int64),
                'position_ids': np.asarray(sum(positions, []), dtype=np.int64),
                'actual_seq_lengths': np.cumsum([len(row) for row in segments], dtype=np.int64)}

    def decode(self, outputs, case):
        scores = outputs['scores']
        if not _params(self.config).get('prune_output', False):scores = scores[..., self.ids]
        return {'outputs': {'scores': scores},
                'alignment': {'rows': [r['id'] for r in case['payload']['requests']], 'columns': self.ids}}

    def output_capacities(self, case):
        node = self.io['outputs'][0]
        return {'scores': case['samples'] * node['shape'][-1] * np.dtype(node['dtype']).itemsize}


def prepare_runtime(config, io):
    return RuntimeHandle(config, io)
