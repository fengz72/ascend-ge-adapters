"""Production framework adapter for the StartLux unified multimodal ONE-AIR route.

A thin bridge over the validated production modules -- multimodal.py (the
unified graph wrapper, owned by the multimodal route) and
startlux_npu_adapter.py (the native chunk-GDR text route, 27/27 dynamic GE) --
plus unified_preprocess.py (the official CPU preprocessing
seam).  This file adds NO decoder or vision arithmetic of its own; it supplies
the production framework entry points around those public interfaces:

* ``describe(config)`` IS multimodal.describe(config): the framework-visible
  contract is exactly the production 5-input/1-output unified schema
  (input_ids [B,S] i64, position_ids [3,B,S] i64, actual_seq_lengths [B] i32,
  pixel_values [512,1536] f32 STATIC, image_feature_index [B,S] i64 ->
  scores [B,26] f32; dynamic B/S with the static 2-slot global pixel pool).
* ``prepare_candidate(source, config)``: the production composition for the
  build.dtype=bfloat16 CLI load (the whole checkpoint on the NPU as BF16).
  multimodal.build_unified_model composes the official FP32 vision tower
  (buffers precomputed by the official methods on a CPU float32 copy), the
  BF16 language trunk with bit-exact F32 A_log restore and the independent
  F32 head26.  The composed wrapper then moves explicitly with
  ``unified.to(original_source_device)`` -- a device-ONLY move -- because the
  vision buffers registered from the CPU copy stay CPU while the original
  visual/text weights already sit on the NPU; without this move the wrapper
  is a mixed-device module and the first forward fails.  The move never
  re-casts: FP32 vision / BF16 trunk / F32 head26 survive exactly as
  composed, and prepare verifies those mixed-dtype device attributes before
  handing the handle back (every parameter and buffer on the target device).
  No second weight load, no process-global patch.
* ``prepare_reference(source, config)``: the INDEPENDENT official CPU FP32
  baseline (the P2b text oracle + the p6-official image recipe): original HF
  get_image_features -> official get_placeholder_mask + masked_scatter ->
  official get_rope_index planes -> the vendored official bidirectional
  text_forward -> the ORIGINAL embedding head.  It never touches the
  candidate wrapper, forward_embeddings or the VisionTowerWrapper, so its
  goldens can judge the candidate.  Every batch row runs unpadded at its
  true length; image features keep their official order and the official
  usage positions.
* ``prepare_runtime(config, io)``: the array-only execution recipe.  encode =
  official CPU preprocessing (unified_preprocess.OfficialPreprocessor:
  official tokenizer / processor composition / load_media / the shared
  official_rope_index_provider; no weights, no NPU context) feeding
  multimodal.build_multimodal_batch with the model params' caps; decode =
  the provenance-free rows + letter columns alignment (inherited from the
  production runtime handle).
* the export trace example comes from multimodal.example_inputs (B2/S65, one
  image row, pure geometry) via the production candidate handle's
  export_spec: both dynamic axes trace >= 2, so the default all-B1 fixture
  pool cannot specialize them (core's exporter strictly validates and
  prefers the example over the first fixture case).

Payload contract of this route (generate_unified_cases.py): one case carries
``payload.rows`` -- one entry per batch row, {'prompt': str, 'images': [paths]?}
(an official request) or {'input_ids': [...]} (a pre-encoded official text row) --
plus optional ``payload.rows_meta`` business metadata the graph encode never
reads.  ``samples`` must be a true int equal to the row count; the caps
(max_batch / max_seq_len) and the vocab bound are the model params' and are
enforced by the builder, never relaxed here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]


def _load_module(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(MODEL_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


base = _load_module("startlux_npu_adapter_for_native", "startlux_npu_adapter.py")
multimodal = _load_module("startlux_multimodal_for_unified", "multimodal.py")
unified_preprocess = _load_module("startlux_unified_preprocess", "unified_preprocess.py")

SLOTS = multimodal.SLOTS
MAX_BATCH = multimodal.MAX_BATCH
MAX_SEQ_LEN = multimodal.MAX_SEQ_LEN
GRID_THW = multimodal.GRID_THW
PATCH_ROWS = multimodal.PATCH_ROWS
MERGED_ROWS = multimodal.MERGED_ROWS
OUTPUT_NAME = "scores"
N_COLUMNS = 26


def describe(config):
    """The production unified IO contract (multimodal.describe) plus this
    route's accepted encode policy (the minimum physical width)."""
    report = multimodal.describe(config)
    params = base._params(config)
    report["encode_policy"] = {
        "min_physical_width": params["compiled_seq_len"],
        "rule": ("width = max(min_physical_width, max(true row lengths)): the official "
                 "per-row encode and position planes are produced at the TRUE lengths "
                 "first, then input_ids / position_ids / image_feature_index are "
                 "right-padded (token 0 / position 0 / index -1) up to the floor; "
                 "actual_seq_lengths keep the true per-row lengths (readout gathers "
                 "L-1) and pixel_values are untouched"),
        "dynamic_s_note": ("the dynamic S axis is NOT fixed at the floor: a row with "
                           "L >= min_physical_width runs at its true length; rows "
                           "beyond max_seq_len are still rejected"),
        "source": "model.params.compiled_seq_len (the existing param; no new param invented)",
        "evidence": ("work/startlux-cannbot/review-p4-pad256-independent.json "
                     "(main-agent-accepted: bit-identical GE input bins vs the "
                     "accepted fixed B1/S256 graph; same-precision full-27 probability "
                     "max 0.0033874 where unpadded short-width execution was unstable) "
                     "+ the formal P7 acceptance on the unified ONE-AIR: "
                     "work/startlux-cannbot/review-p7-cli-independent.json and "
                     "work/startlux-validation/p7-final/compare-report.json "
                     "(13 business requests / 17 real GE read-outs)"),
        "real_ge_validated_here": None,
    }
    return report


def _caps(params):
    """The graph-side caps of this route: the model params, never above the
    module bounds of the single AIR (B<=4, S<=2048, 2 global image slots).
    ``min_width`` is the accepted minimum physical width floor, derived from
    the EXISTING model.params.compiled_seq_len (no new param invented)."""
    return {"max_batch": min(int(params["max_batch"]), MAX_BATCH),
            "max_seq_len": min(int(params["max_seq_len"]), MAX_SEQ_LEN),
            "min_width": min(int(params["compiled_seq_len"]),
                             min(int(params["max_seq_len"]), MAX_SEQ_LEN))}


def _preprocessor(config):
    """The official no-weight preprocessor bound to source.ref (lazy per handle)."""
    folder = base._source_ref(config)
    return unified_preprocess.OfficialPreprocessor(folder)


def _encode_case(preprocessor, case, params, vocab_size):
    caps = _caps(params)
    return unified_preprocess.encode_case_rows(
        preprocessor, case, vocab_size=vocab_size,
        max_batch=caps["max_batch"], max_seq_len=caps["max_seq_len"],
        min_width=caps["min_width"])


class UnifiedRuntimeHandle(multimodal.MultimodalRuntimeHandle):
    """Array-only execution recipe of the unified AIR (no weights, no NPU ctx).

    encode is the production payload.rows path: official CPU preprocessing
    into the validated 5 graph inputs.  decode / output_capacities are the
    production runtime handle's (provenance-free alignment, per-case caps).
    """

    def __init__(self, config, io=None, report=None, preprocessor=None):
        report = report if report is not None else describe(config)
        if io is None:
            io = {"schema": "ge-io/1", "inputs": report["io"]["inputs"],
                  "outputs": report["io"]["outputs"]}
        super().__init__(config, io, report)
        self.config_dict = config
        self._preprocessor = preprocessor

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = _preprocessor(self.config_dict)
        return self._preprocessor

    def encode(self, case):
        return _encode_case(self.preprocessor, case, self.params, self.vocab_size)


class UnifiedCandidateHandle(multimodal.MultimodalCandidateHandle):
    """The candidate wrapper handle: official-preprocess encode to device
    tensors, the wrapper run, the production export spec with its trace
    example, per-case caps."""

    def __init__(self, wrapper, config, report, preprocessor=None):
        super().__init__(wrapper, config, report)
        self.config_dict = config
        self._preprocessor = preprocessor

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = _preprocessor(self.config_dict)
        return self._preprocessor

    def encode(self, case):
        arrays = _encode_case(self.preprocessor, case, self.params, self.vocab_size)
        return {name: torch.from_numpy(value).to(self.device)
                for name, value in arrays.items()}


class UnifiedReferenceHandle:
    """The INDEPENDENT official CPU FP32 baseline of the unified route.

    Official recipe per row (the P2b text oracle + the p6-official image path,
    both pinned materials): the original HF get_image_features, the official
    get_placeholder_mask + masked_scatter injection, the official get_rope_index
    planes of the REAL loaded model, the vendored official bidirectional
    text_forward, and the ORIGINAL embedding head
    (source.get_output_embeddings().weight, tied to the input embedding).
    Every batch row runs unpadded at its true length, image features keep
    their official order and the official usage positions.  Candidate
    modules are importable from this file (the adapter file shares them), but
    this handle never calls candidate computation: no forward_embeddings, no
    VisionTowerWrapper, no wrapper forward -- only the official recipe above.
    """

    def __init__(self, source, config):
        if str(base.VENDOR) not in sys.path:
            sys.path.insert(0, str(base.VENDOR))
        import startlux_decide.backends.bidirectional as bidirectional

        if source.dtype != torch.float32:
            raise ValueError("the unified reference is the CPU float32 official baseline; "
                             f"the source was loaded as {source.dtype}")
        self.params = base._params(config)
        self.letters26 = base._letters_from_model(source)
        decision = base._decision_config(config)
        if decision["letter_token_ids"] != self.letters26:
            raise ValueError("source.ref decision_config letters disagree with the "
                             "loaded model")
        self.vocab_size = decision["vocab_size"]
        self.source = source
        self.inner = source.model
        self.body = base._model_body(source)
        self.head = source.get_output_embeddings().weight
        self.bidirectional = bidirectional
        self.check = bidirectional.enable(self.body)
        self.report = describe(config)
        self.config_dict = config
        self._preprocessor = None

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = _preprocessor(self.config_dict)
        return self._preprocessor

    def encode(self, case):
        """The official per-row encodings of one case (the reference inputs)."""
        samples = case.get("samples")
        if type(samples) is not int or samples < 1:
            raise ValueError(f"unified case {case['id']} samples must be a true "
                             f"positive integer, got {samples!r}")
        payload = case.get("payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list) \
                or not payload["rows"]:
            raise ValueError(f"unified case {case['id']} payload must carry a nonempty "
                             "'rows' list")
        if len(payload["rows"]) != samples:
            raise ValueError(f"unified case {case['id']} samples={samples} but the "
                             f"payload carries {len(payload['rows'])} row(s)")
        return {"rows": self.preprocessor.encode_payload_rows(payload["rows"])}

    def _row_scores(self, row):
        ids = row["input_ids"]
        with torch.inference_mode():
            embeds = self.inner.get_input_embeddings()(ids)
            grid = row.get("image_grid_thw")
            if grid is not None:
                feats = self.inner.get_image_features(
                    row["pixel_values"], grid, return_dict=True).pooler_output
                feats = torch.cat(list(feats), 0).to(embeds.device, embeds.dtype)
                mask, _ = self.inner.get_placeholder_mask(ids, inputs_embeds=embeds,
                                                          image_features=feats)
                embeds = embeds.masked_scatter(mask, feats)
                positions, _deltas = self.inner.get_rope_index(
                    ids, image_grid_thw=grid, attention_mask=None,
                    mm_token_type_ids=row.get("mm_token_type_ids"))
                hidden = self.bidirectional.text_forward(self.body, inputs_embeds=embeds,
                                                         position_ids=positions)
            else:
                hidden = self.bidirectional.text_forward(self.body, ids, attention_mask=None)
            last = hidden[0, -1].float()
            head_rows = self.head.index_select(0, torch.tensor(self.letters26)).float()
            return last @ head_rows.transpose(0, 1)

    def run(self, inputs):
        scores = [self._row_scores(row) for row in inputs["rows"]]
        return {OUTPUT_NAME: torch.stack(scores, dim=0)}

    def decode(self, outputs, case):
        scores = outputs[OUTPUT_NAME]
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (case["samples"], len(self.letters26)):
            raise ValueError(f"unified case {case['id']} scores must be "
                             f"({case['samples']}, {len(self.letters26)}), got "
                             f"{scores.shape}")
        return {"outputs": {OUTPUT_NAME: scores},
                "alignment": {"rows": [case["id"]] * case["samples"],
                              "columns": list(self.letters26)}}

    def output_capacities(self, case):
        return {OUTPUT_NAME: case["samples"] * N_COLUMNS * 4}


def _expected_param_dtype(name):
    """The per-name dtype contract of one unified-wrapper parameter.

    vision.*                         -> float32   (the official FP32 vision route)
    text.head_rows                   -> float32   (the independent F32 head26 clone)
    text.body.*.linear_attn.A_log    -> float32   (the deliberate bit-exact F32 restore
                                                   of the 18 GDR decay logs; the checkpoint
                                                   stores them F32 and the BF16 load would
                                                   round them)
    every other text.body.* param    -> bfloat16  (the validated native trunk)
    anything else                    -> None      (an unknown parameter: rejected, the
                                                   contract is exhaustive, not permissive)
    """
    if name.startswith("vision."):
        return torch.float32
    if name == "text.head_rows":
        return torch.float32
    if name.startswith("text.body."):
        return torch.float32 if name.endswith(".linear_attn.A_log") else torch.bfloat16
    return None


def _verify_candidate_placement(unified, device):
    """The mixed-dtype device contract of the composed wrapper, checked per name.

    After the explicit device move every parameter AND buffer of the unified
    wrapper must sit on the target device.  The dtype contract is exact and
    exhaustive: the vision tower is float32 (the official FP32 route), head26
    is float32, every linear_attn.A_log inside the language trunk is float32
    (the restored checkpoint F32 -- a blanket "body == bfloat16" check would
    wrongly reject those 18 parameters), and every OTHER body parameter is
    bfloat16.  A parameter the contract does not know is an error, not a pass
    -- this verifier must not become a permissive anything-goes mixed check.
    A violation raises before the handle is handed out.
    """
    tensors = dict(unified.named_parameters())
    tensors.update(dict(unified.named_buffers()))
    stray = {name: str(t.device) for name, t in tensors.items() if t.device != device}
    if stray:
        raise RuntimeError(f"the unified wrapper still carries off-device tensors "
                           f"after .to({device}): {stray}")
    wrong, a_log = [], []
    for name, param in unified.named_parameters():
        expected = _expected_param_dtype(name)
        if expected is None:
            wrong.append(f"{name}: unknown parameter (no dtype rule)")
        elif param.dtype != expected:
            wrong.append(f"{name}: {param.dtype} != {expected}")
        if name.endswith(".linear_attn.A_log"):
            a_log.append(name)
    if wrong:
        raise RuntimeError("the composed wrapper violates the mixed-dtype contract: "
                           f"{wrong}")
    if not a_log:
        raise RuntimeError("no linear_attn.A_log parameters found: the composed wrapper "
                           "is not the GDR language trunk this contract describes")
    return {"device": str(device), "tensors": len(tensors),
            "vision": "float32", "language_trunk": "bfloat16", "head26": "float32",
            "a_log_float32": sorted(a_log),
            "buffers": [n for n, _ in unified.vision.named_buffers()]}


def build_unified_candidate(source, config):
    """The production candidate composition on the build.dtype=bfloat16 CLI load.

    multimodal.build_unified_model composes the official FP32 vision tower
    (buffers from the official methods on a CPU float32 copy when the source
    visual is not already CPU float32), casts the language trunk to BF16 in
    place, restores A_log bit-exact F32 and clones head26 as F32.  The
    composed wrapper is then moved explicitly to the ORIGINAL source device
    with .to(device) only -- the registered vision buffers start on CPU when
    the source came from the NPU, so without this move the wrapper is a
    mixed-device module.  The move never re-casts anything; the placement is
    verified before returning.  Returns (unified wrapper, report).
    """
    if source.dtype != torch.bfloat16:
        raise ValueError("the unified candidate expects the build.dtype=bfloat16 load "
                         f"(the source is {source.dtype}); the vision tower is upcast "
                         "to FP32 in place by build_unified_model, the language trunk "
                         "stays BF16")
    device = next(base._model_body(source).parameters()).device
    unified, report = multimodal.build_unified_model(source, config)
    unified.to(device)                     # device-only move: no dtype cast anywhere
    report["placement"] = _verify_candidate_placement(unified, device)
    return unified, report


def prepare_candidate(source, config):
    wrapper, report = build_unified_candidate(source, config)
    return UnifiedCandidateHandle(wrapper, config, report)


def prepare_reference(source, config):
    return UnifiedReferenceHandle(source, config)


def prepare_runtime(config, io):
    return UnifiedRuntimeHandle(config, io, report=describe(config))
