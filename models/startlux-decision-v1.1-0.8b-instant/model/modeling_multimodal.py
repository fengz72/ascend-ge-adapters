"""StartLux two-stage split model: dynamic vision AIR + text AIR.

The delivery is TWO dynamic AIRs run by one pipeline (the C++ pipeline_runtime
or the eager two-stage path): the vision stage turns any count of officially
processed images (any even grid inside the official smart_resize bounds) into
merged features; the text stage injects those features into the token
embeddings and runs the bidirectional hybrid decoder + 26-letter FP32 head,
one readout per row, no generation.

Stage contracts (the CPU bridge produces every tensor; the official image
preprocessing and the official get_rope_index run on the CPU side):

  VISION  input  pixel_values        [T_v, 1536]  f32  dynamic (T_v = sum of
                vision_pos_embeds    [T_v, 768]   f32  dynamic (OFFICIAL
                vision_rotary_cos    [T_v, 64]    f32  dynamic  fast_pos_
                vision_rotary_sin    [T_v, 64]    f32  dynamic  embed_
                vision_cu_seqlens    [n]          i64  dynamic  interpolate /
                                                                   rot_pos_emb)
         output image_features       [T_m, 1024]  f32  dynamic
  TEXT   input  input_ids            [B, S]       i64  dynamic (B<=HBM, S <=
                position_ids         [3, B, S]    i64  dynamic  the model's
                actual_seq_lengths   [B]          i32  dynamic  own
                image_features       [T_m, 1024]  f32  dynamic  max_position_
                image_feature_index  [B, S]       i64  dynamic  embeddings)
         output scores               [B, 26]      f32  dynamic

The features stay ON DEVICE between the stages in the pipeline runtime.  A
text-only call carries a 4-row zero dummy in the packed vision tensors (grid
[1,2,2]) so the record format stays uniform and no dynamic axis is ever
zero-sized -- but the PIPELINE RUNTIME skips the vision execution entirely
for such calls (records carry has_images=false) and feeds a pre-allocated
1x1024 zero features tensor to the text stage instead: the injection's
where(is_image=false) discards it elementwise, the scores are bit-identical,
and the [1,1024] shape matches the dummy vision output's exactly.  Image policy (per request <= 16 images, per image 256..4096 patch
rows) follows the official vendored startlux_decide/images.py constants via
modeling_vision; everything over-limit is rejected at encode time, never
truncated.  modeling_text is loaded lazily (it imports torch_npu) so this
module stays importable on CPU for golden.py / generate_inputs.py.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from model.modeling_vision import (MAX_IMAGES_PER_REQUEST, MAX_PATCH_ROWS,
                                    MIN_PATCH_ROWS, StartLuxVisionTower)

DUMMY_GRID = (1, 2, 2)                       # 4 patch rows: the no-image vision dummy


def decision_config(folder):
    folder = Path(folder)
    letters = json.loads((folder / "decision_config.json").read_text())["letter_token_ids"]
    if len(letters) != 26 or len(set(letters)) != 26:
        raise ValueError("decision_config letter_token_ids must be 26 unique ids")
    model_config = json.loads((folder / "config.json").read_text())
    vocab = (model_config.get("text_config") or {}).get("vocab_size",
                                                        model_config.get("vocab_size"))
    return {"letter_token_ids": letters, "vocab_size": vocab}


class StartLuxModel(nn.Module):
    # == Qwen3_5Model: the multimodal body (visual tower + language trunk).
    def __init__(self, visual, language_model):
        super().__init__()
        self.visual = visual
        self.language_model = language_model


class StartLuxForDecision(nn.Module):
    # == Qwen3_5ForConditionalGeneration for the one-pass decision readout.
    # The forward is the TEXT STAGE: image_features arrive as an input (the
    # vision stage's output); self.model.visual is never called here.
    def __init__(self, model, letters26):
        super().__init__()
        self.model = model
        self.letters26 = list(letters26)

    def derive_decision_head(self):
        # tied head: the checkpoint has no lm_head.weight (tie_word_embeddings
        # =True, asserted in from_folder)
        embedding = self.model.language_model.embed_tokens.weight
        with torch.no_grad():
            rows = embedding.index_select(
                0, torch.tensor(self.letters26, device=embedding.device)
            ).to(torch.float32).clone()
        self.register_buffer("decision_head", rows, persistent=False)

    def forward(self, input_ids, position_ids, actual_seq_lengths, image_features,
                image_feature_index):
        embeds = self.model.language_model.embed_tokens(input_ids)   # [B, S, hidden]
        flat = image_feature_index.reshape(-1).clamp(min=0)
        gathered = image_features.index_select(0, flat)
        gathered = gathered.reshape(input_ids.shape[0], input_ids.shape[1],
                                    image_features.shape[-1]).to(embeds.dtype)
        is_image = (image_feature_index >= 0).unsqueeze(-1)
        injected = torch.where(is_image, gathered, embeds)
        hidden = self.model.language_model(injected, position_ids, actual_seq_lengths)
        batch = hidden.shape[0]
        last_index = (actual_seq_lengths.to(torch.int64) - 1).view(batch, 1, 1)
        last = hidden.gather(1, last_index.expand(batch, 1, hidden.shape[-1])).squeeze(1)
        return last.float() @ self.decision_head.float().transpose(0, 1)


# ----------------------------------------------------- official rope-index binding

def official_rope_index_provider(config):
    # binding the REAL unbound official methods to a config-only namespace
    # reproduces the official positions without loading any model weights
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


# ------------------------------------------------- official vision table provider

def vision_tables_provider(config):
    """Per-call OFFICIAL vision position tables (the vision analogue of
    official_rope_index_provider): binds the unbound official
    fast_pos_embed_interpolate / rot_pos_emb to a shim carrying the
    checkpoint's pos_embed table (model.visual.pos_embed.weight, BF16-stored,
    upcast to F32 exactly), and returns compute(grid_thw) -> the four vision
    table inputs for the call's image order."""
    from safetensors import safe_open
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    # from_pretrained keeps name_or_path; fall back to STARTLUX_MODEL_DIR for
    # configs constructed any other way
    folder = Path(getattr(config, "name_or_path", "") or source_folder())
    table = None
    key = "model.visual.pos_embed.weight"
    for shard in sorted(folder.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if key in handle.keys():
                table = handle.get_tensor(key).to(torch.float32)
                break
    if table is None:
        raise ValueError(f"{key} not found in the checkpoint at {folder}")
    vision = config.vision_config
    merge = vision.spatial_merge_size
    shim = types.SimpleNamespace(
        pos_embed=nn.Embedding.from_pretrained(table, freeze=True),
        num_grid_per_side=int(vision.num_position_embeddings ** 0.5),
        spatial_merge_size=merge,
        config=types.SimpleNamespace(spatial_merge_size=merge),
        rotary_pos_emb=modeling.Qwen3_5VisionRotaryEmbedding(
            (vision.hidden_size // vision.num_heads) // 2))
    interpolate = modeling.Qwen3_5VisionModel.fast_pos_embed_interpolate.__get__(shim)
    rot_pos_emb = modeling.Qwen3_5VisionModel.rot_pos_emb.__get__(shim)

    def compute(grid_thw):
        # grid_thw: CPU int64 tensor [n, 3] in the call's image order; returns
        # the official tables in that same row order
        with torch.no_grad():
            pos = interpolate(grid_thw)
            rot = rot_pos_emb(grid_thw)
            rot = rot.reshape(-1, rot.shape[-1])            # official reshape
            emb = torch.cat((rot, rot), dim=-1)
            cu = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]) \
                .cumsum(0).to(torch.int64)
        return {"vision_pos_embeds": pos.contiguous().numpy(),
                "vision_rotary_cos": emb.cos().contiguous().numpy(),
                "vision_rotary_sin": emb.sin().contiguous().numpy(),
                "vision_cu_seqlens": cu.numpy()}
    return compute


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


def _validate_grid(entry):
    t, h, w = (int(v) for v in entry)
    if t < 1 or h < 2 or w < 2 or h % 2 or w % 2:
        raise ValueError(f"image grid {list(entry)} is invalid: t>=1 and even h,w>=2 "
                         "are required (2x2 spatial merge)")
    if not MIN_PATCH_ROWS <= h * w <= MAX_PATCH_ROWS:
        raise ValueError(f"image grid {list(entry)} has {h * w} patch rows outside the "
                         f"official smart_resize bounds "
                         f"[{MIN_PATCH_ROWS}, {MAX_PATCH_ROWS}] "
                         f"({MIN_PATCH_ROWS * 256}..{MAX_PATCH_ROWS * 256} pixels); "
                         "refusing to stretch/crop")
    return t, h, w


def _row_groups(ids_row, image_token, expected_counts):
    """Contiguous image-pad groups of one row: [(start, end)) ranges, the k-th
    of exactly expected_counts[k] image tokens (per-image merged counts)."""
    groups, t, length = [], 0, len(ids_row)
    while t < length:
        if ids_row[t] != image_token:
            t += 1
            continue
        end = t
        while end < length and ids_row[end] == image_token:
            end += 1
        if len(groups) >= len(expected_counts) or end - t != expected_counts[len(groups)]:
            raise ValueError(f"image placeholder group at {t} has {end - t} tokens; "
                             f"the grids require {expected_counts[len(groups)] if len(groups) < len(expected_counts) else 'no more'}")
        groups.append((t, end))
        t = end
    return groups


def _check_row_structure(ids_row, cfg, n_images, expected_counts):
    # exactly n_images vision_start/end markers, each image-pad group of the
    # per-image merged count, bracketed as [vision_start, pads..., vision_end]
    vs, ve, it = cfg.vision_start_token_id, cfg.vision_end_token_id, cfg.image_token_id
    starts = [i for i, v in enumerate(ids_row) if v == vs]
    ends = [i for i, v in enumerate(ids_row) if v == ve]
    pads = [i for i, v in enumerate(ids_row) if v == it]
    if len(starts) != n_images or len(ends) != n_images \
            or len(pads) != sum(expected_counts):
        raise ValueError(f"placeholder structure mismatch: {len(starts)} vision_start, "
                         f"{len(ends)} vision_end, {len(pads)} image_pad for {n_images} "
                         f"image(s) needing {sum(expected_counts)} pads")
    groups = _row_groups(ids_row, it, expected_counts)
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


def build_multimodal_batch(rows, rope_index, config, vision_tables, *,
                           vocab_size=None, max_batch=None, max_seq_len=None,
                           pad_token_id=0):
    """Build the 9 pipeline inputs (5 vision-stage + 4 text-stage) of one call
    from per-row official processor encodings.

    rows: one dict per batch row carrying the OFFICIAL encoding of one request
    (CPU torch tensors: input_ids int64 [1, S]; image rows also pixel_values
    float32 [sum t*h*w, 1536] and image_grid_thw int64 [n, 3]; optional
    mm_token_type_ids).  rope_index: official_rope_index_provider(config).
    vision_tables: vision_tables_provider(config) (the official position
    tables).  Any image count / any official grid is accepted; a call without
    images carries the 4-row dummy through the vision stage.  Every bound /
    dtype / structure / marker violation raises ValueError.
    """
    batch = len(rows)
    if batch < 1:
        raise ValueError("batch must be at least 1 row")
    if max_batch is not None and batch > max_batch:
        raise ValueError(f"batch {batch} exceeds max_batch {max_batch}")
    if max_seq_len is None:
        # the model's own positional limit is the only S bound (the graphs are
        # dynamic on S; nothing kernel- or graph-side caps it below this)
        max_seq_len = config.text_config.max_position_embeddings
    image_token = config.image_token_id
    cfg = _TokenIds(image_token, config.vision_start_token_id, config.vision_end_token_id)
    if vocab_size is None:
        vocab_size = config.text_config.vocab_size if getattr(config, "text_config", None) \
            else config.vocab_size

    lengths, ids_rows, type_rows = [], [], []
    pixel_blocks, row_grids = [], []
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
            if n_images > MAX_IMAGES_PER_REQUEST:
                raise ValueError(f"{n_images} images exceed the official per-request "
                                 f"limit of {MAX_IMAGES_PER_REQUEST}")
            grid = _require_cpu_tensor(grid, "image_grid_thw", torch.int64)
            if grid.dim() != 2 or grid.shape[1] != 3:
                raise ValueError(f"image_grid_thw must be [n, 3], got {tuple(grid.shape)}")
            validated = [_validate_grid(entry) for entry in grid.tolist()]
            pixels = _require_cpu_tensor(pixels, "pixel_values", torch.float32)
            expected = pixels.shape[0]
            if expected != sum(t * h * w for t, h, w in validated):
                raise ValueError(f"pixel_values has {expected} patch rows; the grids "
                                 f"require {sum(t * h * w for t, h, w in validated)}")
            merged = [(h // 2) * (w // 2) for _t, h, w in validated]
        elif pixels is not None:
            raise ValueError("pixel_values present without image_grid_thw")
        else:
            validated, merged = [], []
        _check_row_structure(ids_row, cfg, n_images, merged)
        lengths.append(len(ids_row))
        ids_rows.append(ids_row)
        type_rows.append(types_row)
        row_grids.append(validated)
        pixel_blocks.append(pixels)

    # the call's image order: (row, image-within-row) -- the official
    # masked_scatter fill order; features come out of the vision stage in it
    all_grids = [g for grids in row_grids for g in grids]
    if all_grids:
        grid_thw = torch.tensor(all_grids, dtype=torch.int64)
        pixel_values = torch.cat([p for p in pixel_blocks if p is not None], dim=0)
    else:
        grid_thw = torch.tensor([list(DUMMY_GRID)], dtype=torch.int64)
        pixel_values = torch.zeros(4, 1536, dtype=torch.float32)
    tables = vision_tables(grid_thw)

    width = max(lengths)
    if width > max_seq_len:
        raise ValueError(f"padded width {width} exceeds max_seq_len {max_seq_len}")
    input_ids = np.full((batch, width), pad_token_id, dtype=np.int64)
    index = np.full((batch, width), -1, dtype=np.int64)
    positions = np.zeros((3, batch, width), dtype=np.int64)

    merged_cursor = 0
    for b in range(batch):
        length = lengths[b]
        input_ids[b, :length] = ids_rows[b]
        n_images = len(row_grids[b])
        if n_images:
            groups = _row_groups(ids_rows[b], image_token,
                                 [(h // 2) * (w // 2) for _t, h, w in row_grids[b]])
            for k, (s, e) in enumerate(groups):
                index[b, s:e] = np.arange(merged_cursor,
                                          merged_cursor + (e - s), dtype=np.int64)
                merged_cursor += e - s
        planes = rope_index(torch.tensor([ids_rows[b]], dtype=torch.long),
                            torch.tensor([type_rows[b]], dtype=torch.long),
                            None if not n_images else torch.tensor(
                                row_grids[b], dtype=torch.long))
        if tuple(planes.shape) != (3, 1, length):
            raise ValueError(f"official get_rope_index returned {tuple(planes.shape)}, "
                             f"expected (3, 1, {length})")
        # official attention_mask semantics: padded positions keep the zero
        # initialization of get_rope_index
        positions[:, b, :length] = planes[:, 0, :].numpy()

    return {
        "pixel_values": pixel_values.numpy(),
        **tables,
        "input_ids": input_ids,
        "position_ids": positions,
        "actual_seq_lengths": np.asarray(lengths, dtype=np.int32),
        "image_feature_index": index,
    }


# --------------------------------------------------------------- adapter surface

def vision_io_spec():
    return {
        "schema": "ge-io/1",
        "inputs": [
            {"name": "pixel_values", "dtype": "float32", "shape": [-1, 1536], "dynamic_dims": [0]},
            {"name": "vision_pos_embeds", "dtype": "float32", "shape": [-1, 768], "dynamic_dims": [0]},
            {"name": "vision_rotary_cos", "dtype": "float32", "shape": [-1, 64], "dynamic_dims": [0]},
            {"name": "vision_rotary_sin", "dtype": "float32", "shape": [-1, 64], "dynamic_dims": [0]},
            {"name": "vision_cu_seqlens", "dtype": "int64", "shape": [-1], "dynamic_dims": [0]},
        ],
        "outputs": [
            {"name": "image_features", "dtype": "float32", "shape": [-1, 1024], "dynamic_dims": [0]},
        ],
    }


def text_io_spec():
    return {
        "schema": "ge-io/1",
        "inputs": [
            {"name": "input_ids", "dtype": "int64", "shape": [-1, -1], "dynamic_dims": [0, 1]},
            {"name": "position_ids", "dtype": "int64", "shape": [3, -1, -1], "dynamic_dims": [1, 2]},
            {"name": "actual_seq_lengths", "dtype": "int32", "shape": [-1], "dynamic_dims": [0]},
            {"name": "image_features", "dtype": "float32", "shape": [-1, 1024], "dynamic_dims": [0]},
            {"name": "image_feature_index", "dtype": "int64", "shape": [-1, -1], "dynamic_dims": [0, 1]},
        ],
        "outputs": [
            {"name": "scores", "dtype": "float32", "shape": [-1, 26], "dynamic_dims": [0]},
        ],
    }


def vision_example_inputs():
    # deterministic non-predicting trace example: 2 dummy [1,16,16] images
    # (T_v=512, n=2); values only need to be legal for tracing
    return {
        "pixel_values": np.zeros((512, 1536), dtype=np.float32),
        "vision_pos_embeds": np.zeros((512, 768), dtype=np.float32),
        "vision_rotary_cos": np.zeros((512, 64), dtype=np.float32),
        "vision_rotary_sin": np.zeros((512, 64), dtype=np.float32),
        "vision_cu_seqlens": np.asarray([256, 512], dtype=np.int64),
    }


def text_example_inputs():
    # B2/S65, one image row (feature order 0..63) and one text row; both text
    # dynamic axes and the feature axis stay symbolic (>= 2)
    batch, width = 2, 65
    ids = np.zeros((batch, width), dtype=np.int64)
    index = np.full((batch, width), -1, dtype=np.int64)
    index[0, :64] = np.arange(64)
    positions = np.tile(np.arange(width, dtype=np.int64)[None, None, :], (3, batch, 1))
    return {
        "input_ids": ids,
        "position_ids": positions,
        "actual_seq_lengths": np.asarray([64, 65], dtype=np.int32),
        "image_features": np.zeros((64, 1024), dtype=np.float32),
        "image_feature_index": index,
    }


# ------------------------------------------------------------ loading and export

def source_folder():
    import os

    value = os.environ.get("STARTLUX_MODEL_DIR")
    if not value or not value.strip():
        raise ValueError("STARTLUX_MODEL_DIR is not set or is empty")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"STARTLUX_MODEL_DIR {path} is not a directory")
    return path


def _verify_a_log_float32(model, text_config):
    expected = sum(1 for kind in text_config.layer_types[:text_config.num_hidden_layers]
                   if kind == "linear_attention")
    a_log = {name: param for name, param in model.named_parameters()
             if name.endswith(".linear_attn.A_log")}
    wrong = [name for name, param in a_log.items() if param.dtype != torch.float32]
    if len(a_log) != expected or wrong:
        raise RuntimeError(f"A_log dtype contract violated: {len(a_log)}/{expected} "
                           f"A_log parameters, non-float32: {wrong}")


def from_folder(folder, device):
    # per-site dtype declarations decide every tensor's dtype; load_state_dict's
    # copy_ lands each checkpoint tensor in its declared dtype and the
    # device-only .to moves the whole tree -- never a global cast
    from transformers import Qwen3_5Config
    from safetensors.torch import load_file

    folder = Path(folder)
    config = Qwen3_5Config.from_pretrained(str(folder))
    text_config = config.text_config
    if not getattr(text_config, "tie_word_embeddings", False):
        raise ValueError("decision_head is derived from embed_tokens; this "
                         "checkpoint must have tie_word_embeddings=True")
    letters = decision_config(folder)["letter_token_ids"]

    from model import modeling_text
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        language_model = modeling_text.StartLuxTextModel(text_config)
        torch.set_default_dtype(torch.float32)
        visual = StartLuxVisionTower(config.vision_config)
    finally:
        torch.set_default_dtype(previous)

    model = StartLuxForDecision(StartLuxModel(visual, language_model), letters)

    state = {}
    for shard in sorted(folder.glob("*.safetensors")):
        state.update(load_file(str(shard)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Qwen3_5PreTrainedModel officially ignores ^mtp.* (the multi-token
    # prediction head this delivery never touches); ANYTHING else is a name
    # mismatch and must fail loudly -- strict=False would silently skip it.
    if missing:
        raise RuntimeError(f"the model is missing checkpoint tensors: {sorted(missing)}")
    if not unexpected or any(not key.startswith("mtp.") for key in unexpected):
        raise RuntimeError(f"unexpected checkpoint tensors (only the MTP head may be "
                            f"skipped, e.g. mtp.fc.weight): {sorted(unexpected)}")

    model.derive_decision_head()
    _verify_a_log_float32(model, text_config)
    return model.to(device).requires_grad_(False).eval()


def load(device):
    # device is the in-process index (with ASCEND_RT_VISIBLE_DEVICES pinning
    # one chip, 0); returns (StartLuxForDecision, summary)
    import torch_npu

    target = torch.device(f"npu:{int(device)}")
    torch_npu.npu.set_device(target)
    model = from_folder(source_folder(), target)
    tensors = dict(model.named_parameters())
    tensors.update(dict(model.named_buffers()))
    stray = {name: str(tensor.device) for name, tensor in tensors.items()
             if tensor.device != target}
    if stray:
        raise RuntimeError(f"the composed wrapper still carries off-device tensors "
                            f"after .to({target}): {stray}")
    return model, {"device": str(target), "tensors": len(tensors),
                   "a_log_float32": [name for name in tensors
                                     if name.endswith(".linear_attn.A_log")]}
