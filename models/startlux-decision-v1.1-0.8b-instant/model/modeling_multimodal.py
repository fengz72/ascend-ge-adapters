"""StartLux unified multimodal model (== Qwen3_5ForConditionalGeneration for the
one-pass decision readout): vision tower + in-graph image injection + bidirectional
hybrid decoder + 26-letter FP32 head, ONE AIR, one readout per row, no generation.
Graph IO / slot semantics / bounds live in io_spec and build_multimodal_batch;
the loading contract (zero key mapping, mtp.* skip, tied-head + A_log assertions)
in from_folder.  modeling_text is loaded lazily (it imports torch_npu) so this
module stays importable on CPU for golden.py / generate_inputs.py.  Full record:
validation.md 2026-11-10.
"""
from __future__ import annotations

import json
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from model.modeling_vision import (GRID_THW, MERGED_ROWS, PATCH_ROWS,
                                    StartLuxVisionTower)

SLOTS = 2                                    # fixed GLOBAL image slots of the single AIR
MAX_BATCH = 4
MAX_SEQ_LEN = 2048


def decision_config(folder):
    # The 26 letter ids and the vocab bound, read from the model folder itself
    # (decision_config.json + config.json) -- never from fixture provenance.
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
    def __init__(self, model, letters26):
        super().__init__()
        self.model = model
        self.letters26 = list(letters26)

    def derive_decision_head(self):
        # The 26-row FP32 readout head: a clone of the tied embedding's letter
        # rows.  The checkpoint carries no lm_head.weight (tie_word_embeddings
        # =True, asserted in from_folder) so an untied folder fails loudly there.
        embedding = self.model.language_model.embed_tokens.weight
        with torch.no_grad():
            rows = embedding.index_select(
                0, torch.tensor(self.letters26, device=embedding.device)
            ).to(torch.float32).clone()
        self.register_buffer("decision_head", rows, persistent=False)

    def forward(self, input_ids, position_ids, actual_seq_lengths, pixel_values,
                image_feature_index):
        feats = self.model.visual(pixel_values)                  # [slots*64, out_hidden] f32
        embeds = self.model.language_model.embed_tokens(input_ids)   # [B, S, hidden]
        flat = image_feature_index.reshape(-1).clamp(min=0)
        gathered = feats.index_select(0, flat)
        gathered = gathered.reshape(input_ids.shape[0], input_ids.shape[1],
                                    feats.shape[-1]).to(embeds.dtype)
        is_image = (image_feature_index >= 0).unsqueeze(-1)
        injected = torch.where(is_image, gathered, embeds)
        hidden = self.model.language_model(injected, position_ids, actual_seq_lengths)
        batch = hidden.shape[0]
        last_index = (actual_seq_lengths.to(torch.int64) - 1).view(batch, 1, 1)
        last = hidden.gather(1, last_index.expand(batch, 1, hidden.shape[-1])).squeeze(1)
        return last.float() @ self.decision_head.float().transpose(0, 1)


# ----------------------------------------------------- official rope-index binding

def official_rope_index_provider(config):
    # get_rope_index reads nothing but self.config, so binding the REAL unbound
    # official methods to a config-only namespace reproduces the official
    # positions without loading any model weights.
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
    # Contiguous image-pad groups of one row: [(start, end)) ranges, each of
    # exactly merged_per_image image tokens.
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
    # Strict placeholder structure of one row: exactly n_images vision_start /
    # vision_end markers, each group of exactly merged_per_image image tokens
    # bracketed as [vision_start, pads..., vision_end].  Returns the groups.
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


def build_multimodal_batch(rows, rope_index, config, *, vocab_size=None, slots=SLOTS,
                           grid_thw=GRID_THW, max_batch=MAX_BATCH,
                           max_seq_len=MAX_SEQ_LEN, pad_token_id=0):
    # rows: one dict per batch row carrying the OFFICIAL processor encoding of
    # one request (CPU torch tensors: input_ids int64 [1, S]; for image rows also
    # pixel_values float32 [256*n, 1536] and image_grid_thw int64 [n, 3]; optional
    # mm_token_type_ids int32/int64 [1, S], falling back to ids == image_token).
    # rope_index: official_rope_index_provider(config); config: the folder config.
    # Returns the 5 numpy graph inputs; every bound / dtype / structure / marker
    # violation raises ValueError -- nothing is silently resized or truncated.
    batch = len(rows)
    if not 1 <= batch <= max_batch:
        raise ValueError(f"batch {batch} outside 1..{max_batch}")
    merged_per_image = (grid_thw[1] // 2) * (grid_thw[2] // 2)
    image_token = config.image_token_id
    cfg = SimpleNamespace(image_token_id=image_token,
                          vision_start_token_id=config.vision_start_token_id,
                          vision_end_token_id=config.vision_end_token_id)
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
    # The graph IO spec of the single AIR; static (depends only on the
    # module-level boundary constants), so it needs no config and is the single
    # source both the exporter and generate_inputs.py read.
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


def example_inputs():
    # Deterministic non-predicting trace example: B2/S65, one image row; both
    # dynamic axes symbolic (B, S >= 2).
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


# ------------------------------------------------------------ loading and export

def source_folder():
    # STARTLUX_MODEL_DIR as a resolved Path (the single weight source).
    import os

    value = os.environ.get("STARTLUX_MODEL_DIR")
    if not value or not value.strip():
        raise ValueError("STARTLUX_MODEL_DIR is not set or is empty")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"STARTLUX_MODEL_DIR {path} is not a directory")
    return path


def _verify_a_log_float32(model, text_config):
    # A_log must survive the load as float32 (a bf16 A_log would silently drift
    # the chunk decay).
    expected = sum(1 for kind in text_config.layer_types[:text_config.num_hidden_layers]
                   if kind == "linear_attention")
    a_log = {name: param for name, param in model.named_parameters()
             if name.endswith(".linear_attn.A_log")}
    wrong = [name for name, param in a_log.items() if param.dtype != torch.float32]
    if len(a_log) != expected or wrong:
        raise RuntimeError(f"A_log dtype contract violated: {len(a_log)}/{expected} "
                           f"A_log parameters, non-float32: {wrong}")


def from_folder(folder, device):
    # The per-site dtype declarations of the module tree decide every tensor's
    # dtype (bf16 default for the language trunk, fp32 for the vision tower);
    # load_state_dict's copy_ lands each checkpoint tensor in its declared
    # dtype and the device-only .to moves the whole tree -- never a global cast.
    from transformers import Qwen3_5Config
    from safetensors.torch import load_file

    folder = Path(folder)
    config = Qwen3_5Config.from_pretrained(str(folder))
    text_config = config.text_config
    if not getattr(text_config, "tie_word_embeddings", False):
        raise ValueError("decision_head is derived from embed_tokens; this "
                         "checkpoint must have tie_word_embeddings=True")
    letters = decision_config(folder)["letter_token_ids"]

    # lazy: modeling_text imports torch_npu, this stays importable on CPU
    from model import modeling_text
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        language_model = modeling_text.StartLuxTextModel(text_config)
        torch.set_default_dtype(torch.float32)
        visual = StartLuxVisionTower(config.vision_config, GRID_THW, SLOTS)
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
    visual.build_static_buffers()
    return model.to(device).requires_grad_(False).eval()


def load(device):
    # Load STARTLUX_MODEL_DIR on the NPU; device is the in-process index (with
    # ASCEND_RT_VISIBLE_DEVICES pinning one chip, 0).  Returns (model, summary).
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


def export_example(wrapper, device):
    # The export spec: the unified IO contract plus the deterministic B2/S65
    # trace example that keeps both dynamic axes symbolic.
    return {"format": "torch", "model": wrapper, "io": io_spec(),
            "example_inputs": {name: torch.from_numpy(value).to(f"npu:{int(device)}")
                               for name, value in example_inputs().items()}}
