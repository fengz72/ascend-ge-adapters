"""StartLux unified multimodal AIR adapter: vision tower + language trunk in ONE graph.

The final StartLux-Decision v1.1 0.8B Instant delivery is a SINGLE AIR executed by
ONE native GESession call that performs, inside the graph: official fixed-grid
vision feature extraction -> image-feature injection into the token embeddings
(equivalent of the official get_placeholder_mask + masked_scatter) -> the
bidirectional hybrid decoder (18 native chunk-GDR + 6 full-attention layers) with
the official interleaved 3-plane MRoPE -> readout at each row's last valid position
-> 26-letter FP32 head projection. Split vision/decoder graphs are a debugging
convenience only, not the delivery.

Graph IO contract (the official image preprocessing and the official
get_rope_index run on CPU; no vision forward ever runs outside the graph):

    input_ids             [B, S]            int64    dynamic (B<=4, S<=2048)
    position_ids          [3, B, S]         int64    dynamic (official get_rope_index
                                                     planes; padding positions are 0,
                                                     the official attention_mask
                                                     semantics)
    actual_seq_lengths    [B]               int32    dynamic (per-row true lengths,
                                                     never cumulative ends)
    pixel_values          [SLOTS*256, 1536] float32  STATIC (SLOTS=2 fixed GLOBAL image
                                                     slots, each grid [1, 16, 16],
                                                     i.e. official-processed 256x256
                                                     images; unused slots are zeros and
                                                     are still computed -- no dynamic
                                                     control flow, no recompilation)
    image_feature_index   [B, S]            int64    dynamic (-1 on text/padding
                                                     tokens; otherwise the merged
                                                     feature row 0..SLOTS*64-1)
    -> scores             [B, 26]           float32 dynamic

Slot semantics (documented boundary, not hidden): the 2 slots are GLOBAL to one
graph call. Images are assigned to slots in (row, image-within-row) order and the
image_feature_index entries, read in batch-row-major token order, are EXACTLY
0, 1, 2, ... n_images*64-1 -- the official masked_scatter fill order. A call may
carry at most 2 images IN TOTAL (and at most 2 per request); a B=4 batch with 2
images per row is NOT supported by this AIR and is rejected at encode time.

Precision plan: the vision tower keeps an FP32 residual stream with FP16
Linear/Conv inner products (the checkpoint stores the 153 vision tensors in
BF16, so the FP32 upcast is exact; a BF16 vision compute domain fails the gate
against the CPU FP32 reference and is not used), language trunk BF16, A_log
restored bit-exact F32 from the checkpoint, dt_bias at its checkpoint BF16,
head26 an independent FP32 clone of the tied embedding's 26 letter rows, full
input embedding kept (the 26-column readout is a projection of the ORIGINAL
tied head, not a retrained model).

MRoPE: full-attention layers consume the official interleaved
mrope_section [11, 11, 10] layout computed in-graph from the [3, B, S] planes
(StartLuxTextDecoder.mrope_cos_sin); image positions are the official
get_rope_index planes -- never arange text positions.

Every bound is strictly enforced at encode time (build_multimodal_batch): wrong
dtypes/ranks/devices, non-CPU tensors, out-of-vocabulary ids, marker structure
violations, grids other than [1, 16, 16] and the batch/length/image caps are all
rejected with ValueError; nothing is silently resized, cast, cropped or
truncated.  The arrays it returns are correct by construction -- the feature
index runs 0, 1, 2, ... in batch-row-major order because the slot counter does,
and the padding regions keep their zero / -1 initialization.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root

SLOTS = 2                                    # fixed GLOBAL image slots of the single AIR
MAX_BATCH = 4
MAX_SEQ_LEN = 2048


def _load_module(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(MODEL_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    # register before exec: dynamo resolves traced functions by __module__
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


base = _load_module("startlux_npu_adapter_for_native", "model/startlux_npu_adapter.py")
vision_mod = _load_module("startlux_vision_tower", "model/vision.py")

VisionTowerWrapper = vision_mod.VisionTowerWrapper
GRID_THW = vision_mod.GRID_THW
PATCH_ROWS = vision_mod.PATCH_ROWS          # 256 patch rows per slot
MERGED_ROWS = vision_mod.MERGED_ROWS       # 64 merged feature rows per slot


class MultimodalDecoder(nn.Module):
    """One-graph composition: vision tower + injection + shared decoder loop.

    ``text`` is a StartLuxTextDecoder instance providing
    ``forward_embeddings`` (the shared decoder loop with real 3-plane MRoPE) and
    the 26-row FP32 head; ``vision`` is the fixed-slot VisionTowerWrapper.
    """

    def __init__(self, text_decoder, vision_tower):
        super().__init__()
        self.text = text_decoder
        self.vision = vision_tower

    def injected_embeds(self, input_ids, image_feature_index, pixel_values):
        """Token embeddings with the merged image features at the image-pad
        positions: the in-graph equivalent of the official
        get_placeholder_mask + masked_scatter pair (a boolean select, no
        arithmetic on kept text embeddings, so kept rows stay bit-exact)."""
        feats = self.vision(pixel_values)                       # [slots*64, out_hidden] f32
        embeds = self.text.body.embed_tokens(input_ids)         # [B, S, hidden]
        flat = image_feature_index.reshape(-1).clamp(min=0)
        gathered = feats.index_select(0, flat)
        gathered = gathered.reshape(input_ids.shape[0], input_ids.shape[1],
                                    feats.shape[-1]).to(embeds.dtype)
        is_image = (image_feature_index >= 0).unsqueeze(-1)
        return torch.where(is_image, gathered, embeds)

    def forward(self, input_ids, position_ids, actual_seq_lengths, pixel_values,
                image_feature_index):
        embeds = self.injected_embeds(input_ids, image_feature_index, pixel_values)
        return self.text.forward_embeddings(embeds, position_ids, actual_seq_lengths)


# ----------------------------------------------------- official rope-index binding

def official_rope_index_provider(config):
    """The official Qwen3_5Model.get_rope_index bound to a config-only namespace.

    get_rope_index and get_vision_position_ids read nothing but ``self.config``
    (vision_config.spatial_merge_size) and their arguments, so binding the REAL
    unbound official methods to a lightweight object carrying the folder config
    reproduces the official positions without loading any model weights.
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    namespace = types.SimpleNamespace(config=config)
    rope_index = modeling.Qwen3_5Model.get_rope_index.__get__(namespace)
    namespace.get_vision_position_ids = \
        modeling.Qwen3_5Model.get_vision_position_ids.__get__(namespace)

    def provider(input_ids, mm_token_type_ids, image_grid_thw=None):
        planes, _deltas = rope_index(input_ids, mm_token_type_ids=mm_token_type_ids,
                                     image_grid_thw=image_grid_thw, attention_mask=None)
        return planes

    return provider


# --------------------------------------------------------------- CPU-side bridge

def _require_cpu_tensor(value, name, dtype):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype is not dtype:
        raise ValueError(f"{name} must be torch.{str(dtype).removeprefix('torch.')}, "
                         f"got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor (pass .detach().cpu() explicitly); "
                         f"got device {value.device}")
    return value


def _row_groups(ids_row, image_token, merged_per_image):
    """Contiguous image-pad groups of one row: [(start, end)) ranges, each of
    exactly merged_per_image image tokens."""
    groups, t, length = [], 0, len(ids_row)
    while t < length:
        if ids_row[t] != image_token:
            t += 1
            continue
        end = t
        while end < length and ids_row[end] == image_token:
            end += 1
        if end - t != merged_per_image:
            raise ValueError(f"image placeholder group at {t} has {end - t} tokens; "
                             f"grid [1,16,16] requires exactly {merged_per_image}")
        groups.append((t, end))
        t = end
    return groups


def _check_row_structure(ids_row, cfg, n_images, merged_per_image):
    """Strict placeholder structure of one row: exactly n_images vision_start /
    vision_end markers, each group of exactly merged_per_image image tokens
    bracketed as [vision_start, pads..., vision_end].

    Returns the [(start, end)) ranges of the image-pad groups.
    """
    vs, ve, it = cfg.vision_start_token_id, cfg.vision_end_token_id, cfg.image_token_id
    starts = [i for i, v in enumerate(ids_row) if v == vs]
    ends = [i for i, v in enumerate(ids_row) if v == ve]
    pads = [i for i, v in enumerate(ids_row) if v == it]
    if len(starts) != n_images or len(ends) != n_images \
            or len(pads) != n_images * merged_per_image:
        raise ValueError(f"placeholder structure mismatch: {len(starts)} vision_start, "
                         f"{len(ends)} vision_end, {len(pads)} image_pad for {n_images} image(s)")
    groups = _row_groups(ids_row, it, merged_per_image)
    for k, (s, e) in enumerate(groups):
        if s < 1 or ids_row[s - 1] != vs:
            raise ValueError(f"image group {k} is not preceded by vision_start")
        if e >= len(ids_row) or ids_row[e] != ve:
            raise ValueError(f"image group {k} is not followed by vision_end")
    return groups


class _TokenIds:
    def __init__(self, image_token, vision_start, vision_end):
        self.image_token_id = image_token
        self.vision_start_token_id = vision_start
        self.vision_end_token_id = vision_end


def build_multimodal_batch(rows, rope_index, config, *, vocab_size=None, slots=SLOTS,
                           grid_thw=GRID_THW, max_batch=MAX_BATCH,
                           max_seq_len=MAX_SEQ_LEN, pad_token_id=0):
    """Build the 5 graph inputs of one unified-AIR call from per-row official
    processor encodings.

    ``rows``: one dict per batch row, each carrying the OFFICIAL encoding of one
    request (exactly what the official TransformersBackend consumes): CPU
    torch tensors -- {"input_ids": int64 [1, S], "pixel_values": float32
    [256 * n, 1536] and "image_grid_thw": int64 [n, 3] (both absent for a text
    row), "mm_token_type_ids": int32/int64 [1, S] (optional; falls back to
    ids == image_token_id)}.

    ``rope_index``: the official position provider
    (official_rope_index_provider on the folder config).

    ``config``: the folder's transformers config (token ids, vocab).

    Returns a dict of the 5 numpy graph inputs, strictly validated on the way
    (see the module docstring). Every rejection raises ValueError.
    """
    batch = len(rows)
    if not 1 <= batch <= max_batch:
        raise ValueError(f"batch {batch} outside 1..{max_batch}")
    merged_per_image = (grid_thw[1] // 2) * (grid_thw[2] // 2)
    image_token = config.image_token_id
    cfg = _TokenIds(image_token, config.vision_start_token_id, config.vision_end_token_id)
    if vocab_size is None:
        vocab_size = config.text_config.vocab_size if getattr(config, "text_config", None) \
            else config.vocab_size
    lengths, ids_rows, type_rows, pixel_blocks, grid_rows = [], [], [], [], []
    row_groups = []
    for row in rows:
        ids = _require_cpu_tensor(row["input_ids"], "input_ids", torch.int64)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"input_ids must be [1, S], got {tuple(ids.shape)}")
        if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
            raise ValueError("input_ids outside the vocabulary bounds "
                             f"[0, {vocab_size})")
        ids_row = ids[0].tolist()
        expected_types = [1 if v == image_token else 0 for v in ids_row]
        types = row.get("mm_token_type_ids")
        if types is None:
            types_row = expected_types
        else:
            if not isinstance(types, torch.Tensor) or types.device.type != "cpu":
                raise ValueError(f"mm_token_type_ids must be a CPU torch.Tensor, "
                                 f"got {type(types).__name__}")
            if types.dtype not in (torch.int64, torch.int32):
                raise ValueError(f"mm_token_type_ids must be int64/int32, got {types.dtype}")
            if tuple(types.shape) != (1, len(ids_row)):
                raise ValueError("mm_token_type_ids shape mismatch")
            types_row = types[0].long().tolist()
            if types_row != expected_types:
                raise ValueError("mm_token_type_ids disagree with the image tokens")
        grid = row.get("image_grid_thw")
        pixels = row.get("pixel_values")
        n_images = 0 if grid is None else int(grid.shape[0])
        if n_images:
            grid = _require_cpu_tensor(grid, "image_grid_thw", torch.int64)
            if grid.dim() != 2 or grid.shape[1] != 3:
                raise ValueError(f"image_grid_thw must be [n, 3], got {tuple(grid.shape)}")
            for entry in grid.tolist():
                if tuple(int(v) for v in entry) != tuple(grid_thw):
                    raise ValueError(f"image grid {list(entry)} is not the supported "
                                     f"{list(grid_thw)} (256x256 after the official "
                                     "preprocessing); refusing to stretch/crop")
            pixels = _require_cpu_tensor(pixels, "pixel_values", torch.float32)
            if tuple(pixels.shape) != (n_images * grid_thw[1] * grid_thw[2], 1536):
                raise ValueError(f"pixel_values shape {tuple(pixels.shape)} does not "
                                 f"match {n_images} image(s) of grid {list(grid_thw)}")
        elif pixels is not None:
            raise ValueError("pixel_values present without image_grid_thw")
        row_groups.append(_check_row_structure(ids_row, cfg, n_images, merged_per_image))
        lengths.append(len(ids_row))
        ids_rows.append(ids_row)
        type_rows.append(types_row)
        grid_rows.append(n_images)
        pixel_blocks.append(pixels)
    total_images = sum(grid_rows)
    if any(n > slots for n in grid_rows):
        raise ValueError(f"at most {slots} images per request")
    if total_images > slots:
        raise ValueError(f"{total_images} images exceed the {slots} fixed global slots "
                         "of one call (a B>1 batch shares the slot pool)")
    width = max(lengths)
    if width > max_seq_len:
        raise ValueError(f"padded width {width} exceeds max_seq_len {max_seq_len}")

    input_ids = np.full((batch, width), pad_token_id, dtype=np.int64)
    index = np.full((batch, width), -1, dtype=np.int64)
    positions = np.zeros((3, batch, width), dtype=np.int64)
    pixel_values = np.zeros((slots * grid_thw[1] * grid_thw[2], 1536), dtype=np.float32)

    slot = 0
    for b in range(batch):
        length = lengths[b]
        input_ids[b, :length] = ids_rows[b]
        n_images = grid_rows[b]
        if n_images:
            for k, (s, e) in enumerate(row_groups[b]):
                index[b, s:e] = np.arange(slot * merged_per_image,
                                          (slot + 1) * merged_per_image, dtype=np.int64)
                pixel_values[slot * PATCH_ROWS:(slot + 1) * PATCH_ROWS] = \
                    pixel_blocks[b][k * PATCH_ROWS:(k + 1) * PATCH_ROWS].numpy()
                slot += 1
        planes = rope_index(torch.tensor([ids_rows[b]], dtype=torch.long),
                            torch.tensor([type_rows[b]], dtype=torch.long),
                            None if not n_images else torch.tensor(
                                [list(grid_thw)] * n_images, dtype=torch.long))
        if tuple(planes.shape) != (3, 1, length):
            raise ValueError(f"official get_rope_index returned {tuple(planes.shape)}, "
                             f"expected (3, 1, {length})")
        # official attention_mask semantics: padded positions keep the zero
        # initialization of get_rope_index
        positions[:, b, :length] = planes[:, 0, :].numpy()

    return {
        "input_ids": input_ids,
        "position_ids": positions,
        "actual_seq_lengths": np.asarray(lengths, dtype=np.int32),
        "pixel_values": pixel_values,
        "image_feature_index": index,
    }


# --------------------------------------------------------------- adapter surface

def io_spec():
    """The graph IO spec of the single AIR (schema, inputs, outputs).

    Static: the shapes depend only on the module-level boundary constants, so
    the spec needs no config and is the single source both the exporter and
    generate_inputs.py read.
    """
    return {
        "schema": "ge-io/1",
        "inputs": [
            {"name": "input_ids", "dtype": "int64", "shape": [-1, -1], "dynamic_dims": [0, 1]},
            {"name": "position_ids", "dtype": "int64", "shape": [3, -1, -1], "dynamic_dims": [1, 2]},
            {"name": "actual_seq_lengths", "dtype": "int32", "shape": [-1], "dynamic_dims": [0]},
            {"name": "pixel_values", "dtype": "float32", "shape": [SLOTS * PATCH_ROWS, 1536]},
            {"name": "image_feature_index", "dtype": "int64", "shape": [-1, -1], "dynamic_dims": [0, 1]},
        ],
        "outputs": [
            {"name": "scores", "dtype": "float32", "shape": [-1, 26], "dynamic_dims": [0]},
        ],
    }


def build_unified_model(model):
    """Compose the unified one-graph wrapper on an official model load.

    ``model``: transformers.Qwen3_5ForConditionalGeneration, bfloat16 on the
    NPU (the delivery path) or float32 on CPU (a validation oracle) -- both
    are supported:

    - the language trunk is cast to bfloat16 in place (a no-op for a BF16 load;
      for an FP32 load, BF16(F32(x)) == x for every BF16-stored checkpoint
      value), then A_log is restored bit-exact F32 from the checkpoint and
      head26 is an independent F32 clone;
    - the vision tower is cast to float32 in place (exact upcast of the BF16
      checkpoint values) and its official constant buffers (pos_embeds
      interpolation, rotary cos/sin) are computed ON CPU AT FLOAT32 by the
      model's own official methods -- from the module itself when it already is
      CPU float32, otherwise from a CPU float32 deep copy.

    Returns the composed MultimodalDecoder.
    """
    letters = base._decision_config(model.name_or_path)["letter_token_ids"]
    inner = model.model
    visual = inner.visual
    on_cpu_f32 = all(p.device.type == "cpu" and p.dtype == torch.float32
                     for p in visual.parameters())
    buffer_source = visual if on_cpu_f32 else copy.deepcopy(visual).to("cpu", torch.float32)
    # Upcast FIRST (exact bf16->fp32): the wrapper constructor then
    # PRE-QUANTIZES the Linear/Conv weights to fp16 in place (lossless for
    # bf16-stored values), so a later float() would silently undo it.
    visual.float()                                # exact upcast; no-op when already f32
    vision_tower = VisionTowerWrapper(visual, GRID_THW, SLOTS, buffer_source=buffer_source,
                                      linear_dtype=torch.float16)
    body = base._model_body(model)
    body.to(torch.bfloat16)
    decoder = base.StartLuxTextDecoder(body, letters)
    base._restore_and_verify(body, decoder, model)
    unified = MultimodalDecoder(decoder, vision_tower)
    return unified.requires_grad_(False).eval()


def example_inputs():
    """Deterministic, non-predicting export trace example: B2/S65 with one
    image row (feature order 0..63) and one text row. B and S are both >= 2 so
    dynamo keeps both dynamic axes symbolic; the values only need to be legal
    for tracing (token 0, zero pixels, arange positions)."""
    batch, width = 2, 65
    ids = np.zeros((batch, width), dtype=np.int64)
    index = np.full((batch, width), -1, dtype=np.int64)
    index[0, :MERGED_ROWS] = np.arange(MERGED_ROWS)
    positions = np.tile(np.arange(width, dtype=np.int64)[None, None, :], (3, batch, 1))
    return {
        "input_ids": ids,
        "position_ids": positions,
        "actual_seq_lengths": np.asarray([64, 65], dtype=np.int32),
        "pixel_values": np.zeros((SLOTS * PATCH_ROWS, 1536), dtype=np.float32),
        "image_feature_index": index,
    }
