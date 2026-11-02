"""Independent CPU FP32 HF reference for StartLux-Decision v1.1 0.8B Instant (P2b).

The oracle is the pinned official code, vendored unmodified under vendor/ (upstream
commit a99977d9332decc9fb27a2488b4a83b533b6cd5f): the official in-process reference
recipe of vendor/startlux_decide/backends/transformers.py loads the real model folder
(Qwen3_5ForConditionalGeneration), takes model.model.language_model as the text decoder
and model.get_output_embeddings().weight as the output head (tied to the input embedding;
never modified, never truncated), and enables the official Instant prompt-attention patch
of vendor/startlux_decide/backends/bidirectional.py: full-attention layers read every
one-pass prompt bidirectionally within its row, while the linear Gated DeltaNet layers
and their short causal convolution stay exactly on the mask the stock forward gives them.

Read-outs are the letter logits at the last valid token of a prompt:

    scores26 = last_hidden_state[0, -1].float() @ head.index_select(0, letter_ids26).float().T

Only the selected 26 letter rows of the tied head are projected: no S x vocab logits are
ever materialized.  The reference runs one prompt per forward pass, use_cache=False, CPU
FP32 (the whole checkpoint upcast), the HF PyTorch fallback for the gated-delta layers
(flash-linear-attention and causal-conv1d are deliberately not installed), 8 OpenMP/MKL/
OpenBLAS threads, and no NPU adapter patches.  Checkpoint precision is preserved honestly:
A_log is stored F32 in the safetensors and stays bit-identical; dt_bias is stored BF16 and
is upcast losslessly by the FP32 load (it is NOT originally F32 -- verify_gdr_dtypes()
checks both against the checkpoint bytes and reports the stored dtypes).

This module never imports the candidate bridge (decision.py): goldens are produced by the
official Decider plus this backend alone, so they can judge any candidate implementation.
The single deliberate deviation from the official transformers backend is in _load():
device_map is not passed, because accelerate is not installed in this environment and the
reference targets plain CPU anyway.

Set OMP/MKL/OPENBLAS to 8 before anything imports torch, then use as:

    folder, backend = reference.load_reference(MODEL_DIR)
    asyncio.run(run(folder, backend))    # await backend.start(); await decider.decide(...)
"""
from __future__ import annotations

import os

THREADS = 8
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(THREADS))

import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR) not in sys.path:               # make the pinned official package importable
    sys.path.insert(0, str(VENDOR))

from startlux_decide.backends.base import BackendRequestError          # noqa: E402
from startlux_decide.backends.transformers import TransformersBackend  # noqa: E402
from startlux_decide.modelcfg import load_model_folder                 # noqa: E402

MODEL_REVISION = "6496ae641cc71b149ce9dad69bd97f2a44e7bf11"
UPSTREAM_COMMIT = "a99977d9332decc9fb27a2488b4a83b533b6cd5f"
POSITION_IDS_NOTE = ("text positions are arange(L); the decoder also consumes the 3 mrope "
                     "sections (3, 1, L), identical to arange(L) for this text-only prompt "
                     "(bidirectional.text_forward default)")
EXPECTED_WEIGHTS = {
    "model-00001-of-00001.safetensors": {
        "bytes": 1746942632,
        "sha256": "f788c8e01fd453dfbf2effb23a06148d38238b9e8a872ab8faacbb49a03cecd1",
    }
}
"""The pinned revision's single weight shard (main-agent-verified); weight_identity()
refuses any folder whose shards, sizes or SHA256 do not match exactly."""


class GoldenTransformersBackend(TransformersBackend):
    """The official transformers backend on CPU, extended only by recording every
    read-out (prompt, input ids, positions, FP32 scores of all 26 letters) and by
    computing the full 26-letter projection in the same forward pass.  Text only.
    `dtype` defaults to float32 (the FP32 oracle); the BF16 variant below passes
    "bfloat16"."""

    name = "cpu-fp32-golden"

    def __init__(self, path: str, tokenizer, letters26: list[int], dtype: str = "float32"):
        super().__init__(path, tokenizer, bidirectional=True, device="cpu", dtype=dtype)
        self.letters26 = list(letters26)
        self.records: list[dict] = []

    def _load(self) -> None:
        import torch
        import transformers

        from startlux_decide.backends import bidirectional

        config = transformers.AutoConfig.from_pretrained(self.path, local_files_only=True)
        cls = getattr(transformers, config.architectures[0])
        torch.set_num_threads(THREADS)
        self.device = torch.device(self.device_name or "cpu")
        self.model = cls.from_pretrained(self.path, dtype=getattr(torch, self.dtype_name),
                                         local_files_only=True).to(self.device).eval()
        inner = self.model.model
        self.body = getattr(inner, "language_model", inner)
        self.head = self.model.get_output_embeddings().weight
        self.check = bidirectional.enable(self.body) if self.bidirectional else None

    def _readout(self, prompt: str, letter_ids: list[int], media=None) -> list[float]:
        import time

        import torch

        from startlux_decide.backends import bidirectional

        if media is not None:
            raise BackendRequestError("the golden reference reads text only")
        ids = self._inputs(prompt, None)["input_ids"]
        input_ids = ids[0].tolist()
        t0 = time.perf_counter()
        with torch.inference_mode():
            hidden = bidirectional.text_forward(self.body, ids, attention_mask=None)
            last = hidden[0, -1].float()
            rows = self.head.index_select(0, torch.tensor(self.letters26, device=self.head.device))
            scores26 = (last @ rows.float().T).tolist()
        seconds = time.perf_counter() - t0
        self.records.append({
            "prompt": prompt, "input_ids": input_ids, "seq_len": len(input_ids),
            "position_ids": list(range(len(input_ids))), "actual_seq_lengths": [len(input_ids)],
            "position_ids_note": POSITION_IDS_NOTE, "letters_asked": len(letter_ids),
            "letter_ids": list(letter_ids), "scores26": scores26, "readout_seconds": seconds,
        })
        index = {t: i for i, t in enumerate(self.letters26)}
        return [scores26[index[t]] for t in letter_ids]


class BF16GoldenTransformersBackend(GoldenTransformersBackend):
    """The same official recipe with a BF16 body: quantization evidence, not the oracle.

    The body, projections, causal conv and norms run BF16; the delta-rule core is FP32
    inside HF's fallback by construction (it casts q/k/v/beta/g to float32); the gate g
    reads A_log with .float(), so A_log must hold the checkpoint's original F32 values
    (a plain BF16 load rounds them, and g would be computed from prematurely rounded
    values).  _load therefore verifies what the load kept and, when HF rounded A_log,
    restores the checkpoint's F32 tensor into the parameter (recorded honestly in
    gdr_precision -- the source is the pinned checkpoint, never candidate logic).
    dt_bias keeps its checkpoint BF16 (upcast only inside the softplus sum).  The
    selected 26 head rows and the last hidden are projected in FP32, as the deployment
    plan requires.  Nothing here casts FP32 scores to BF16: the forward runs BF16.
    """

    name = "cpu-bf16-golden"

    def __init__(self, path: str, tokenizer, letters26: list[int]):
        super().__init__(path, tokenizer, letters26, dtype="bfloat16")
        self.gdr_precision: dict = {}

    def _load(self) -> None:
        super()._load()
        self.gdr_precision = self._fix_gdr_precision()

    def _fix_gdr_precision(self) -> dict:
        import torch
        from safetensors import safe_open

        body = self.body
        shard = next(Path(self.path).glob("*.safetensors"))
        a_log_loaded, a_log_restored = None, 0
        dt_loaded, dt_ok = None, True
        with safe_open(shard, framework="pt", device="cpu") as f:
            for i, layer in enumerate(body.layers[:body.config.num_hidden_layers]):
                if body.config.layer_types[i] != "linear_attention":
                    continue
                attn = layer.linear_attn
                ckpt = f.get_tensor(f"model.language_model.layers.{i}.linear_attn.A_log")
                loaded = attn.get_parameter("A_log").detach()
                a_log_loaded = a_log_loaded or str(loaded.dtype)
                if loaded.dtype != torch.float32 or not torch.equal(loaded.float(), ckpt.float()):
                    attn.A_log.data = ckpt.to(loaded.device).clone()
                    a_log_restored += 1
                if not torch.equal(attn.get_parameter("A_log").detach(), ckpt):
                    raise RuntimeError(f"layer {i}: A_log is not the checkpoint's exact F32 values")
                dt = attn.get_parameter("dt_bias").detach()
                dt_loaded = dt_loaded or str(dt.dtype)
                dt_ok = dt_ok and torch.equal(dt, f.get_tensor(
                    f"model.language_model.layers.{i}.linear_attn.dt_bias"))
        if not dt_ok:
            raise RuntimeError("dt_bias differs from the checkpoint's BF16 values")
        return {
            "a_log": {"checkpoint_dtype": "torch.float32", "loaded_dtype": a_log_loaded,
                      "restored_layers": a_log_restored,
                      "action": ("restored_from_checkpoint_f32" if a_log_restored
                                 else "hf_kept_checkpoint_f32"),
                      "bit_exact_after": True},
            "dt_bias": {"checkpoint_dtype": "torch.bfloat16", "loaded_dtype": dt_loaded,
                        "action": "kept as loaded (checkpoint BF16, upcast in the softplus sum)",
                        "bit_exact": True},
            "delta_rule_core": "FP32 inside the HF PyTorch fallback (casts q/k/v/beta/g to float32)",
            "head_projection": "float32 (selected 26 rows; the forward itself runs BF16)",
        }


def load_reference(model_dir: str | Path):
    """The official model folder, tokenizer and the FP32 backend, ready for the Decider."""
    from transformers import AutoTokenizer

    path = str(model_dir)
    folder = load_model_folder(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    backend = GoldenTransformersBackend(path, tokenizer, list(folder.letter_token_ids))
    return folder, backend


def load_reference_bf16(model_dir: str | Path):
    """The same folder on the BF16 evidence backend (quantization comparison only)."""
    from transformers import AutoTokenizer

    path = str(model_dir)
    folder = load_model_folder(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    backend = BF16GoldenTransformersBackend(path, tokenizer, list(folder.letter_token_ids))
    return folder, backend


def weight_identity(model_dir: str | Path) -> dict:
    """Size and SHA256 of the pinned model folder's single weight shard, validated
    against EXPECTED_WEIGHTS: a folder without weights, or with shards whose sizes or
    hashes differ, is refused -- it is never reported as the pinned revision."""
    import hashlib

    d = Path(model_dir)
    names = sorted(p.name for p in d.glob("*.safetensors"))
    if not names:
        raise FileNotFoundError(f"{d}: no .safetensors weights, no weight identity")
    if names != sorted(EXPECTED_WEIGHTS):
        raise ValueError(f"{d}: weight shards {names} are not the pinned revision's "
                         f"{sorted(EXPECTED_WEIGHTS)}")
    out = {"revision": MODEL_REVISION, "files": {}}
    for name in names:
        expected = EXPECTED_WEIGHTS[name]
        h = hashlib.sha256()
        n = 0
        with open(d / name, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
                n += len(chunk)
        if n != expected["bytes"] or h.hexdigest() != expected["sha256"]:
            raise ValueError(f"{d}/{name}: {n} bytes, sha256 {h.hexdigest()}; the pinned "
                             f"revision is {expected['bytes']} bytes, sha256 {expected['sha256']}")
        out["files"][name] = {"bytes": n, "sha256": h.hexdigest(), "verified": True}
    return out


def checkpoint_dtype_report(model_dir: str | Path) -> dict:
    """Stored dtypes of representative tensors straight from the safetensors headers."""
    import json
    import struct

    d = Path(model_dir)
    report: dict[str, dict] = {}
    for path in sorted(d.glob("*.safetensors")):
        with open(path, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            tail = name.split(".")[-1]
            if tail in ("A_log", "dt_bias") or name.endswith("linear_attn.conv1d.weight") \
                    or name.endswith("language_model.embed_tokens.weight") \
                    or name.endswith("layers.3.self_attn.q_proj.weight"):
                report[name] = {"dtype": entry["dtype"], "shape": entry["shape"]}
    return report


def verify_gdr_dtypes(model, model_dir: str | Path) -> dict:
    """The loaded GDR parameters against the checkpoint bytes: A_log must stay the
    checkpoint's exact F32 values; dt_bias is stored BF16 and must be the lossless
    FP32 upcast of them (it was never F32 in the checkpoint)."""
    import torch
    from safetensors import safe_open

    body = getattr(model.model, "language_model", model.model)
    d = Path(model_dir)
    shard = next(d.glob("*.safetensors"))
    out = {}
    with safe_open(shard, framework="pt", device="cpu") as f:
        for layer in range(body.config.num_hidden_layers):
            if body.config.layer_types[layer] != "linear_attention":
                continue
            prefix = f"model.language_model.layers.{layer}.linear_attn."
            for param, cast in (("A_log", "f32"), ("dt_bias", "bf16")):
                raw = f.get_tensor(prefix + param)
                loaded = body.layers[layer].linear_attn.get_parameter(param).detach().cpu()
                exact = bool(torch.equal(loaded.float(), raw.float()))
                out[f"layers.{layer}.{param}"] = {
                    "checkpoint_dtype": str(raw.dtype), "loaded_dtype": str(loaded.dtype),
                    "checkpoint_is": cast, "bit_exact_upcast": exact}
                if not exact or str(loaded.dtype) != "torch.float32":
                    raise RuntimeError(f"layers.{layer}.{param}: loaded {loaded.dtype} is not the "
                                       f"exact {raw.dtype} checkpoint value")
    return out


def runtime_meta(model, backend: GoldenTransformersBackend, model_dir: str | Path,
                 weights: dict | None = None) -> dict:
    """Everything a golden needs to be reproducible and auditable.  `weights` is a
    precomputed weight_identity() result (the hash pass reads the whole checkpoint).

    runtime.dtype is the backend's actual dtype (float32 for the oracle, bfloat16 for
    the BF16 evidence backend); the FP32 oracle additionally reports verify_gdr_dtypes,
    the BF16 backend its gdr_precision (A_log restored to the checkpoint F32, dt_bias
    kept BF16), so the two remain distinguishable under the same API."""
    import platform

    import torch
    import transformers

    folder = load_model_folder(str(model_dir))
    tied = model.get_output_embeddings().weight.data_ptr() == \
        getattr(model.model, "language_model", model.model).embed_tokens.weight.data_ptr()
    if isinstance(backend, BF16GoldenTransformersBackend):
        gdr_check = dict(backend.gdr_precision)
    else:
        gdr_check = verify_gdr_dtypes(model, model_dir)
    return {
        "model": {"revision": MODEL_REVISION,
                  "weight_identity": weights if weights is not None else weight_identity(model_dir),
                  "checkpoint_dtype_report": checkpoint_dtype_report(model_dir),
                  "gdr_dtype_check": gdr_check},
        "code": {"upstream_commit": UPSTREAM_COMMIT,
                 "vendored": "models/startlux-decision-v1.1-0.8b-instant/vendor",
                 "candidate_imported": False},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "device": "cpu",
                    "dtype": str(getattr(backend, "dtype_name", "float32")),
                    "torch_threads": torch.get_num_threads(),
                    "gdr": "HF PyTorch fallback (flash-linear-attention / causal-conv1d not installed)",
                    "head": f"model.get_output_embeddings().weight (tied to embed_tokens: {tied})"},
        "folder": {"kind": folder.kind, "format": folder.format,
                   "prompt_attention": folder.prompt_attention,
                   "temperature_by_type": dict(folder.temperature_by_type),
                   "letter_token_ids": list(folder.letter_token_ids),
                   "wide_choice": {"group": folder.wide_group, "keep": folder.wide_keep,
                                   "residual": folder.wide_residual}},
        "bidirectional_check": backend.check,
    }
