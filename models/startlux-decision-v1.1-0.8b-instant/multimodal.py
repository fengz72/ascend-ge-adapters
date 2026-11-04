"""StartLux unified multimodal AIR adapter: vision tower + language trunk in ONE graph.

The final StartLux-Decision v1.1 0.8B Instant delivery is a SINGLE AIR executed by
ONE native GESession call that performs, inside the graph: official fixed-grid FP32
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

Precision plan (validated in p6-official: BF16 vision compute fails the gate at
rel_l2 ~0.11 and is not used): vision tower FP32 (the checkpoint stores the 153
vision tensors in BF16; upcast is exact), language trunk BF16, A_log restored
bit-exact F32 from the checkpoint, dt_bias at its checkpoint BF16, head26 an
independent FP32 clone of the tied embedding's 26 letter rows, full input
embedding kept (the 26-column readout is a projection of the ORIGINAL tied head,
not a retrained model).

MRoPE: full-attention layers consume the official interleaved
mrope_section [11, 11, 10] layout computed in-graph from the [3, B, S] planes
(StartLuxTextDecoder.mrope_cos_sin); image positions are the official
get_rope_index planes -- never arange text positions.

Every bound is strictly enforced at encode time (build_multimodal_batch /
validate_batch_arrays): wrong dtypes/ranks/devices, non-CPU tensors, out-of-
vocabulary ids, out-of-pool feature indices, duplicated/misordered/holed feature
runs, image tokens inside padding, marker structure violations, grids other than
[1, 16, 16] and batch/length caps are all rejected with ValueError; nothing is
silently resized, cast, cropped or truncated.
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

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parents[1]

SLOTS = 2                                    # fixed GLOBAL image slots of the single AIR
MAX_BATCH = 4
MAX_SEQ_LEN = 2048
IMAGE_TOKENS = 64                            # merged tokens per [1, 16, 16] image


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


base = _load_module("startlux_npu_adapter_for_native", "startlux_npu_adapter.py")
vision_mod = _load_module("startlux_vision_tower", "vision.py")

VisionTowerWrapper = vision_mod.VisionTowerWrapper
GRID_THW = vision_mod.GRID_THW
PATCH_ROWS = vision_mod.PATCH_ROWS          # 256 patch rows per slot
MERGED_ROWS = vision_mod.MERGED_ROWS       # 64 merged feature rows per slot


class MultimodalDecoder(nn.Module):
    """One-graph composition: official FP32 vision tower + injection + shared decoder loop.

    ``text`` is a StartLuxTextDecoder instance providing
    ``forward_embeddings`` (the shared decoder loop with real 3-plane MRoPE) and
    the 26-row FP32 head; ``vision`` is the fixed-slot VisionTowerWrapper.
    """

    def __init__(self, text_decoder, vision_tower, slots=SLOTS):
        super().__init__()
        self.text = text_decoder
        self.vision = vision_tower
        self.slots = int(slots)

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
    reproduces the official positions without loading any model weights -- the
    runtime position computer of the unified route. The validation runners
    cross-check this provider bit-exact against the loaded model's own method.
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


def model_rope_index_provider(model):
    """The loaded official model's own get_rope_index (validation oracle path)."""
    inner = getattr(model, "model", model)

    def provider(input_ids, mm_token_type_ids, image_grid_thw=None):
        planes, _deltas = inner.get_rope_index(input_ids,
                                               mm_token_type_ids=mm_token_type_ids,
                                               image_grid_thw=image_grid_thw,
                                               attention_mask=None)
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
    bracketed as [vision_start, pads..., vision_end], pads contiguous."""
    vs, ve, it = cfg.vision_start_token_id, cfg.vision_end_token_id, cfg.image_token_id
    starts = [i for i, v in enumerate(ids_row) if v == vs]
    ends = [i for i, v in enumerate(ids_row) if v == ve]
    pads = [i for i, v in enumerate(ids_row) if v == it]
    if len(starts) != n_images or len(ends) != n_images \
            or len(pads) != n_images * merged_per_image:
        raise ValueError(f"placeholder structure mismatch: {len(starts)} vision_start, "
                         f"{len(ends)} vision_end, {len(pads)} image_pad for {n_images} image(s)")
    groups = _row_groups(ids_row, it, merged_per_image)
    if len(groups) != n_images:
        raise ValueError(f"expected {n_images} image-pad group(s), found {len(groups)}")
    for k, (s, e) in enumerate(groups):
        if s < 1 or ids_row[s - 1] != vs:
            raise ValueError(f"image group {k} is not preceded by vision_start")
        if e >= len(ids_row) or ids_row[e] != ve:
            raise ValueError(f"image group {k} is not followed by vision_end")
        if pads[k * merged_per_image:(k + 1) * merged_per_image] != list(range(s, e)):
            raise ValueError(f"image group {k} is not one contiguous placeholder run")
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

    ``rope_index``: the official position provider (official_rope_index_provider
    on the folder config, or model_rope_index_provider on a loaded model).

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
    for row in rows:
        ids = _require_cpu_tensor(row["input_ids"], "input_ids", torch.int64)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"input_ids must be [1, S], got {tuple(ids.shape)}")
        if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
            raise ValueError("input_ids outside the vocabulary bounds "
                             f"[0, {vocab_size})")
        ids_row = ids[0].tolist()
        types = row.get("mm_token_type_ids")
        if types is not None:
            types = _require_cpu_tensor(types, "mm_token_type_ids", types.dtype)
            if types.dtype not in (torch.int64, torch.int32):
                raise ValueError(f"mm_token_type_ids must be int64/int32, got {types.dtype}")
            if tuple(types.shape) != (1, len(ids_row)):
                raise ValueError("mm_token_type_ids shape mismatch")
            types_row = types[0].long().tolist()
        else:
            types_row = [1 if v == image_token else 0 for v in ids_row]
        if types_row != [1 if v == image_token else 0 for v in ids_row]:
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
        _check_row_structure(ids_row, cfg, n_images, merged_per_image)
        lengths.append(len(ids_row))
        ids_rows.append(ids_row)
        type_rows.append(types_row)
        grid_rows.append(n_images)
        pixel_blocks.append(None if pixels is None else pixels)
    total_images = sum(grid_rows)
    if any(n > 2 for n in grid_rows):
        raise ValueError("at most 2 images per request")
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
            groups = _row_groups(ids_rows[b], image_token, merged_per_image)
            for k, (s, e) in enumerate(groups):
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

    arrays = {
        "input_ids": input_ids,
        "position_ids": positions,
        "actual_seq_lengths": np.asarray(lengths, dtype=np.int32),
        "pixel_values": pixel_values,
        "image_feature_index": index,
    }
    validate_batch_arrays(arrays, slots=slots, max_batch=max_batch,
                          max_seq_len=max_seq_len, vocab_size=vocab_size,
                          image_token=image_token,
                          vision_start=cfg.vision_start_token_id,
                          vision_end=cfg.vision_end_token_id,
                          merged_per_image=merged_per_image)
    return arrays


def validate_batch_arrays(arrays, *, slots=SLOTS, max_batch=MAX_BATCH,
                          max_seq_len=MAX_SEQ_LEN, vocab_size=None, image_token=None,
                          vision_start=None, vision_end=None,
                          merged_per_image=MERGED_ROWS):
    """Strict numpy-level validation of the 5 graph inputs (the model-free
    encode-side gate). Exact dtypes/ranks/shapes (no coercion), value bounds,
    the exact official masked_scatter row-major feature order, and -- when the
    token ids are supplied -- the placeholder marker structure. Raises
    ValueError on every violation; returns True when everything holds."""
    for name in ("input_ids", "position_ids", "actual_seq_lengths", "pixel_values",
                 "image_feature_index"):
        if name not in arrays:
            raise ValueError(f"missing graph input {name}")
        if not isinstance(arrays[name], np.ndarray):
            raise ValueError(f"{name} must be a numpy ndarray, got {type(arrays[name]).__name__}")
    input_ids, positions = arrays["input_ids"], arrays["position_ids"]
    lengths, pixel_values = arrays["actual_seq_lengths"], arrays["pixel_values"]
    index = arrays["image_feature_index"]
    if input_ids.dtype != np.int64 or input_ids.ndim != 2:
        raise ValueError(f"input_ids must be an int64 2-D ndarray (dtype {input_ids.dtype}, "
                         f"ndim {input_ids.ndim})")
    if positions.dtype != np.int64 or positions.ndim != 3:
        raise ValueError(f"position_ids must be an int64 3-D ndarray (dtype {positions.dtype}, "
                         f"ndim {positions.ndim})")
    if lengths.dtype != np.int32 or lengths.ndim != 1:
        raise ValueError(f"actual_seq_lengths must be an int32 1-D ndarray (dtype "
                         f"{lengths.dtype}, ndim {lengths.ndim})")
    if pixel_values.dtype != np.float32 or pixel_values.ndim != 2:
        raise ValueError(f"pixel_values must be a float32 2-D ndarray (dtype "
                         f"{pixel_values.dtype}, ndim {pixel_values.ndim})")
    if index.dtype != np.int64 or index.ndim != 2:
        raise ValueError(f"image_feature_index must be an int64 2-D ndarray (dtype "
                         f"{index.dtype}, ndim {index.ndim})")
    batch, width = input_ids.shape
    if not 1 <= batch <= max_batch:
        raise ValueError(f"batch {batch} outside 1..{max_batch}")
    if not 1 <= width <= max_seq_len:
        raise ValueError(f"padded width {width} outside 1..{max_seq_len}")
    if positions.shape != (3, batch, width):
        raise ValueError(f"position_ids shape {positions.shape} != (3, {batch}, {width})")
    if lengths.shape != (batch,):
        raise ValueError(f"actual_seq_lengths shape {lengths.shape} != ({batch},)")
    if index.shape != (batch, width):
        raise ValueError(f"image_feature_index shape {index.shape} != ({batch}, {width})")
    if pixel_values.shape != (slots * PATCH_ROWS, 1536):
        raise ValueError(f"pixel_values shape {pixel_values.shape} != "
                         f"({slots * PATCH_ROWS}, 1536)")
    if not np.isfinite(pixel_values).all():
        raise ValueError("pixel_values contain non-finite values")
    if (lengths < 1).any() or (lengths > width).any():
        raise ValueError("actual_seq_lengths must be within 1..S per row")
    if (positions < 0).any():
        raise ValueError("position_ids must be non-negative")
    if vocab_size is not None:
        if (input_ids < 0).any() or (input_ids >= vocab_size).any():
            raise ValueError(f"input_ids outside the vocabulary bounds [0, {vocab_size})")
    if (index < -1).any() or (index >= slots * merged_per_image).any():
        raise ValueError(f"image_feature_index outside [-1, {slots * merged_per_image})")
    flat = index.reshape(-1)
    selected = flat[flat >= 0]
    if not np.array_equal(selected, np.arange(len(selected), dtype=np.int64)):
        raise ValueError("image_feature_index entries, read in batch-row-major token "
                         "order, must be exactly 0, 1, 2, ... (the official "
                         "masked_scatter fill order): no duplicates, holes or "
                         "misordered runs")
    for b in range(batch):
        length = int(lengths[b])
        if (index[b, length:] >= 0).any():
            raise ValueError(f"row {b}: image_feature_index marks padding positions "
                             "(every position at or past the row's true length must "
                             "be -1; image features cannot be scattered into padding)")
        if (positions[:, b, length:] != 0).any():
            raise ValueError(f"row {b}: position_ids padding positions must be 0 "
                             "(the official attention_mask semantics; the valid "
                             "prefix carries the official get_rope_index planes)")
        if image_token is not None:
            row = input_ids[b]
            valid_is_image = row[:length] == image_token
            indexed = index[b, :length] >= 0
            if not np.array_equal(valid_is_image, indexed):
                raise ValueError(f"row {b}: image_feature_index does not mark exactly "
                                 "the image tokens")
            if (row[length:] == image_token).any():
                raise ValueError(f"row {b} has image tokens inside padding")
            _check_row_structure(row[:length].tolist(),
                                 _TokenIds(image_token, vision_start, vision_end),
                                 int(valid_is_image.sum()) // merged_per_image,
                                 merged_per_image)
    return True


# --------------------------------------------------------------- adapter surface

def describe(config):
    params = base._params(config)
    decision = base._decision_config(config)
    if any(value is None for value in decision["token_ids"].values()):
        raise ValueError("the unified multimodal route requires image_token_id / "
                         "vision_start_token_id / vision_end_token_id in the model "
                         "folder's config.json (strict marker validation)")
    text_report = base.describe(config)
    return {
        "version": 1,
        "route": "unified-multimodal-single-air",
        "global_patch": False,
        "in_place": False,
        "referenced_files": text_report["referenced_files"],
        "reference_params": text_report["reference_params"],
        "io": {
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
        },
        "bounds": {
            "max_batch": min(params["max_batch"], MAX_BATCH),
            "max_seq_len": params["max_seq_len"],
            "images_per_request": 2,
            "images_per_call": SLOTS,
            "slot_scope": "the 2 image slots are GLOBAL to one graph call; images are "
                          "assigned in (row, image-within-row) order; a B>1 batch "
                          "shares the pool -- B rows x 2 images each is NOT supported",
            "image_grid_thw": list(GRID_THW),
            "image_tokens_per_image": MERGED_ROWS,
            "grid_policy": "strict rejection of any grid other than [1, 16, 16]; "
                           "no silent resize/crop/truncation",
            "feature_order": "image_feature_index read batch-row-major must be exactly "
                             "arange(n_images*64) -- the official masked_scatter order",
        },
        "precision": {
            "vision": "float32 (checkpoint BF16 upcast exactly at load)",
            "language_trunk": "bfloat16",
            "a_log": "float32 restored bit-exact from the checkpoint",
            "dt_bias": "checkpoint bfloat16, upcast in the g formula only",
            "head26": "float32 clone of the tied embedding letter rows",
            "ge_options": {"graph_run_mode": 0, "precision_mode": "must_keep_origin_dtype"},
        },
        "cpu_side": {
            "image_preprocessing": "official Qwen3VLProcessor image path (PIL backend "
                                   "composition; the full AutoProcessor factory is not "
                                   "usable on this folder and is not claimed)",
            "positions": "official Qwen3_5Model.get_rope_index bound to the folder "
                         "config (official_rope_index_provider: no weights needed); "
                         "padded positions 0, the official attention_mask semantics",
            "feature_index": "masked_scatter row-major order with global (row, image) "
                             "slot assignment",
            "vision_forward_outside_graph": False,
        },
        "native_gdr": text_report["native_gdr"],
        "gdn_route": text_report["gdn_route"],
        "required_dependencies": text_report["required_dependencies"],
        "changes": [
            "unified single-AIR graph: official FP32 vision tower + injection + decoder + head26",
            "in-graph image-feature injection equivalent to official masked_scatter",
            "official interleaved 3-plane MRoPE [11,11,10] from get_rope_index planes",
            "dynamic B/S with static 2-slot global pixel_values (no dynamic control flow)",
            "strict encode-side validation (dtypes/ranks/devices/bounds/order/markers)",
        ],
    }


def build_unified_model(model, config):
    """Compose the unified one-graph wrapper on an official model load.

    ``model``: transformers.Qwen3_5ForConditionalGeneration loaded by the
    framework source (bfloat16, already on NPU) or by a validation runner
    (float32 on CPU) -- both are supported:

    - the language trunk is cast to bfloat16 in place (a no-op for a BF16 load;
      for an FP32 load, BF16(F32(x)) == x for every BF16-stored checkpoint
      value), then A_log is restored bit-exact F32 from the checkpoint and
      head26 is an independent F32 clone -- exactly the validated native text
      route (work/startlux-validation/p4-dynamic, 27/27 GE);
    - the vision tower is cast to float32 in place (exact upcast of the BF16
      checkpoint values) and its official constant buffers (pos_embeds
      interpolation, rotary cos/sin) are computed ON CPU AT FLOAT32 by the
      model's own official methods -- from the module itself when it already is
      CPU float32, otherwise from a CPU float32 deep copy. The one constant
      NOT derived from the checkpoint, the non-persistent rotary inv_freq, is
      regenerated by VisionTowerWrapper with the official constructor: the
      framework load path (core.sources.load_source) ends with
      model.to(device, dtype), which quantizes it to BF16 irrecoverably
      (work/startlux-cannbot/review-vision-rotary-buffer-cast.json), so the
      buffer provenance is "checkpoint-derived values upcast exactly +
      constructor-official frequency", NOT "identical for every load path".

    Returns (unified MultimodalDecoder, report).
    """
    params = base._params(config)
    letters = base._letters_from_model(model)
    decision = base._decision_config(config)
    if decision["letter_token_ids"] != letters:
        raise ValueError("source.ref decision_config letters disagree with the loaded model")
    inner = model.model
    visual = inner.visual
    on_cpu_f32 = all(p.device.type == "cpu" and p.dtype == torch.float32
                     for p in visual.parameters())
    buffer_source = visual if on_cpu_f32 else copy.deepcopy(visual).to("cpu", torch.float32)
    vision_tower = VisionTowerWrapper(visual, GRID_THW, SLOTS, buffer_source=buffer_source)
    visual.float()                                # exact upcast; no-op when already f32
    body = base._model_body(model)
    body.to(torch.bfloat16)
    decoder = base.StartLuxTextDecoder(body, letters, params["chunk_size"],
                                        gdn_route=params["gdn_route"])
    restore = base._restore_and_verify(body, decoder, model, params)
    unified = MultimodalDecoder(decoder, vision_tower, SLOTS)
    unified.requires_grad_(False).eval()
    report = describe(config)
    report["restore"] = restore
    report["weight_storage"] = ("original HF modules shared by the wrapper; "
                                "vision f32, language bf16, head26 fp32 clone")
    report["vision_buffers"] = {
        "source": ("the CPU float32 module" if on_cpu_f32
                   else "a CPU float32 deep copy of the source visual"),
        "checkpoint_derived": "pos_embeds interpolation (official method on the "
                              "checkpoint-stored pos_embed table; BF16->F32 exact)",
        "rotary_inv_freq": "official constructor Qwen3_5VisionRotaryEmbedding("
                           "head_dim // 2) -- regenerated, never the dtype-cast "
                           "source value (core load path model.to(dtype) quantizes "
                           "it irrecoverably)",
        "rotary_inv_freq_restored": bool(vision_tower.rotary_inv_freq_restored),
        "buffers_on_source_device": True,
    }
    return unified, report


def example_inputs(device=None):
    """Deterministic, non-predicting export trace example: B2/S65 with one
    image row (feature order 0..63) and one text row. B and S are both >= 2 so
    dynamo keeps both dynamic axes symbolic; the values only need to be legal
    for tracing (token 0, zero pixels, arange positions)."""
    batch, width = 2, 65
    ids = np.zeros((batch, width), dtype=np.int64)
    index = np.full((batch, width), -1, dtype=np.int64)
    index[0, :IMAGE_TOKENS] = np.arange(IMAGE_TOKENS)
    positions = np.tile(np.arange(width, dtype=np.int64)[None, None, :], (3, batch, 1))
    arrays = {
        "input_ids": ids,
        "position_ids": positions,
        "actual_seq_lengths": np.asarray([64, 65], dtype=np.int32),
        "pixel_values": np.zeros((SLOTS * PATCH_ROWS, 1536), dtype=np.float32),
        "image_feature_index": index,
    }
    if device is None:
        return arrays
    return {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}


class MultimodalRuntimeHandle:
    """Array-only execution recipe of the unified AIR (no weights, no NPU)."""

    def __init__(self, config, io, report=None):
        self.params = base._params(config)
        self.io = io
        self.report = report if report is not None else describe(config)
        self.letters26 = self.report["reference_params"]["letter_token_ids"]
        self.vocab_size = self.report["reference_params"]["vocab_size"]
        tokens = self.report["reference_params"].get("token_ids")
        if not tokens or set(tokens) != {"image_token_id", "vision_start_token_id",
                                         "vision_end_token_id"}:
            raise ValueError("MultimodalRuntimeHandle needs the report's token_ids "
                             "(image/vision marker ids) for strict validation")
        self.token_ids = dict(tokens)

    def encode(self, case):
        payload = case.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"readout {case['id']} payload must be a mapping")
        arrays = {}
        for name in ("input_ids", "position_ids", "actual_seq_lengths",
                     "pixel_values", "image_feature_index"):
            if name not in payload:
                raise ValueError(f"readout {case['id']} payload misses {name}")
            value = payload[name]
            if not isinstance(value, list):
                raise ValueError(f"readout {case['id']} payload {name} must be nested lists")
            dtype = {"input_ids": np.int64, "position_ids": np.int64,
                     "actual_seq_lengths": np.int32, "pixel_values": np.float32,
                     "image_feature_index": np.int64}[name]
            # strict raw typing: a nested list of floats/bools must NEVER be
            # silently truncated into an integer array (np.asarray(dtype=int)
            # floors floats and maps bools to 0/1); ragged lists arrive as
            # object dtype and are rejected the same way.
            raw = np.asarray(value)
            if name == "pixel_values":
                if raw.dtype.kind != "f":
                    raise ValueError(f"readout {case['id']} payload {name} must be nested "
                                     f"float lists, got array dtype {raw.dtype}")
            else:
                if raw.dtype.kind not in "iu":
                    raise ValueError(f"readout {case['id']} payload {name} must be nested "
                                     f"integer lists (float/bool lists are rejected, not "
                                     f"truncated), got array dtype {raw.dtype}")
                if raw.dtype != dtype:
                    bounds = np.iinfo(dtype)
                    if raw.size and (int(raw.min()) < bounds.min
                                     or int(raw.max()) > bounds.max):
                        raise ValueError(f"readout {case['id']} payload {name} values do "
                                         f"not fit {np.dtype(dtype).name}")
            arrays[name] = raw.astype(dtype)
        if case["samples"] != arrays["input_ids"].shape[0]:
            raise ValueError(f"readout {case['id']} samples must equal the batch of "
                             "the carried arrays")
        validate_batch_arrays(arrays, max_batch=self.params["max_batch"],
                              max_seq_len=self.params["max_seq_len"],
                              vocab_size=self.vocab_size,
                              image_token=self.token_ids["image_token_id"],
                              vision_start=self.token_ids["vision_start_token_id"],
                              vision_end=self.token_ids["vision_end_token_id"])
        return arrays

    def decode(self, outputs, case):
        scores = outputs["scores"]
        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (case["samples"], len(self.letters26)):
            raise ValueError(f"readout {case['id']} scores must be "
                             f"({case['samples']}, {len(self.letters26)}), got {scores.shape}")
        return {"outputs": {"scores": scores},
                "alignment": {"rows": [case["id"]] * case["samples"],
                              "columns": list(self.letters26)}}

    def output_capacities(self, case):
        return {"scores": case["samples"] * 26 * 4}


class MultimodalCandidateHandle(MultimodalRuntimeHandle):
    """Eager candidate handle: validated arrays in, wrapper scores out."""

    def __init__(self, wrapper, config, report):
        super().__init__(config, {"schema": "ge-io/1",
                                  "outputs": [{"name": "scores", "dtype": "float32",
                                               "shape": [-1, len(report["reference_params"]["letter_token_ids"])],
                                               "dynamic_dims": [0]}]},
                         report)
        self.wrapper = wrapper
        self.device = next(wrapper.parameters()).device

    def encode(self, case):
        arrays = super().encode(case)
        return {name: torch.from_numpy(value).to(self.device)
                for name, value in arrays.items()}

    def run(self, inputs):
        with torch.inference_mode():
            return {"scores": self.wrapper(**inputs)}

    def export_spec(self):
        io = {"schema": "ge-io/1",
              "inputs": self.report["io"]["inputs"],
              "outputs": self.report["io"]["outputs"]}
        spec = {"format": "torch", "model": self.wrapper, "io": io}
        if self.params["max_batch"] >= 2 and self.params["max_seq_len"] >= 2:
            spec["example_inputs"] = {name: torch.from_numpy(value).to(self.device)
                                      for name, value in example_inputs().items()}
        return spec

    def output_capacities(self, case):
        return {"scores": case["samples"] * 26 * 4}


def prepare_candidate(source, config):
    unified, report = build_unified_model(source, config)
    return MultimodalCandidateHandle(unified, config, report)


def prepare_runtime(config, io):
    return MultimodalRuntimeHandle(config, io)
