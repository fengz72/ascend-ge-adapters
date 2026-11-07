"""Official CPU preprocessing for the StartLux unified multimodal AIR (P6 production).

The unified single-AIR contract (multimodal.py) keeps the OFFICIAL image
preprocessing and the OFFICIAL get_rope_index on the CPU side: the graph inputs
pixel_values / image_feature_index / position_ids are produced from what the
official processor and the official position method emit, nothing else.  This
module is that CPU side, as a reusable component for the production seam:

* ``OfficialPreprocessor`` loads ONLY the official no-weight components of the
  pinned model folder: AutoTokenizer, AutoConfig and the official image-capable
  processor, composed exactly the way the REAL validation runners verified on
  this folder (work/startlux-validation/p6-official/run_p6_official.py,
  work/startlux-validation/p6-unified/run_p6_unified.py):
  Qwen3VLProcessor(image_processor=AutoImageProcessor.from_pretrained(...,
  backend="pil"), tokenizer=AutoTokenizer.from_pretrained(...),
  video_processor=AutoVideoProcessor.from_pretrained(...)).  The full
  AutoProcessor factory cannot take backend='pil' on this folder and
  Qwen3VLProcessor REJECTS video_processor=None (real TypeError: "Received a
  NoneType for argument video_processor, but a BaseVideoProcessor was
  expected") -- neither route is used nor claimed.  No model weights are ever
  loaded here and no NPU context is created.  Every component is lazy: a pool
  of pre-encoded text rows runs with no tokenizer and no Pillow at all.
* positions come from the SHARED official provider
  ``multimodal.official_rope_index_provider(config)`` -- the official
  Qwen3_5Model.get_rope_index bound to the folder config; no local duplicate of
  that binding lives in this module.
* ``encode_media_row(prompt, media)`` is the single official row recipe (the
  official TransformersBackend._inputs of the pinned vendor package): text
  prompts use tokenizer.encode(prompt, add_special_tokens=False); a media
  object (the official startlux_decide.images.load_media result, real PIL
  images + smart_resize processor bounds) goes through the official processor
  call with the official image-token-count check.  ``encode_row`` only turns
  local image paths into a media object via the official load_media (data
  URIs) and then calls encode_media_row -- there is exactly one copy of the
  preprocessing arithmetic.  The returned row dicts are exactly what
  multimodal.build_multimodal_batch consumes.
* ``build_graph_inputs(rows, ...)`` runs multimodal.build_multimodal_batch on
  the official per-row encodings with the CALLER'S caps (max_batch /
  max_seq_len / vocab_size are always passed in by the adapter from the model
  params; the builder defaults are never assumed): fixed grid [1, 16, 16] =
  official-processed 256x256, at most 2 images per request, at most 2 global
  image slots per call, B<=4, S<=2048.  Every other grid or overlong row is
  REJECTED, never stretched, cropped or truncated.

* ``pad_graph_inputs`` applies the CALLER'S minimum physical width (the
  standalone flow passes min_width=1: every case runs at its true length;
  the P4-era 256 floor -- review-p4-pad256-independent.json, a transition
  that kept the dynamic AIR's inputs bit-identical to the fixed-width era --
  was retired 2026-11-06 by the short-width gate on the delivered AIR:
  true-length execution is bit-stable over repeats and within bf16
  quantization of both the CPU golden and the accepted pad-256 outputs).  The official
  per-row encode and the official position planes are ALWAYS produced at the
  TRUE lengths first (the CPU oracle rows stay unpadded); the padding only
  extends the already-validated arrays to the minimum physical width, keeps
  actual_seq_lengths at the true per-row lengths (the readout gathers L-1),
  keeps pixel_values untouched, and the padded set is re-validated by the same
  shared strict validator.  The width rule is deterministic per call -- never
  a per-case pick: width = max(min_width, max(true lengths)), dynamic S stays
  dynamic (a row with L >= min_width runs at its true length; S is NOT fixed
  at 256), and rows beyond max_seq_len are still rejected.  No prediction,
  option, threshold or official algorithm changes anywhere.

Pillow 13 / torchvision 0.27 are TASK-LOCAL dependencies (the validated install
is work/startlux-validation/p6-official/deps/{pillow,torchvision}, wheel
sha256 files next to them, PYTHONPATH wiring in
work/startlux-validation/p6-unified/env.sh).  When they are missing the error
points there; nothing here installs or patches system packages.
"""
from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root
ROOT = MODEL_DIR.parents[1]

DEPS_HINT = ("the official image preprocessing needs Pillow (and torchvision for the "
             "official video processor instance); both are task-local dependencies. "
             "The validated task-local install is "
             "work/startlux-validation/p6-official/deps/pillow and "
             "work/startlux-validation/p6-official/deps/torchvision (wheel sha256 "
             "files next to them); prepend them to PYTHONPATH for this process only "
             "(see work/startlux-validation/p6-unified/env.sh). Do not install them "
             "system-wide.")


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


multimodal = _load_module("startlux_multimodal_for_unified", "model/multimodal.py")

GRID_THW = multimodal.GRID_THW                    # (1, 16, 16): 256x256 official processing
PATCH_ROWS = multimodal.PATCH_ROWS                # 256 patch rows per slot
MERGED_ROWS = multimodal.MERGED_ROWS              # 64 merged feature rows per slot
SLOTS = multimodal.SLOTS                          # 2 fixed global image slots per call
MAX_BATCH = multimodal.MAX_BATCH                  # 4
MAX_SEQ_LEN = multimodal.MAX_SEQ_LEN              # 2048
IMAGE_TOKENS_PER_IMAGE = MERGED_ROWS              # 64 merged tokens per 256x256 image


def _vendor_path():
    if str(MODEL_DIR / "decider" / "vendor") not in sys.path:
        sys.path.insert(0, str(MODEL_DIR / "decider" / "vendor"))


def image_data_uri(path):
    """A local image file as the data URI the official load_media reads."""
    path = Path(path)
    mime = {"ppm": "image/x-portable-pixmap", "png": "image/png", "jpg": "image/jpeg",
            "jpeg": "image/jpeg", "webp": "image/webp"}.get(
                path.suffix.lstrip(".").lower(), "image/png")
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def resolve_path(path):
    """Repo convention: relative payload paths resolve against the project ROOT."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_media_from_paths(image_paths):
    """Local image files -> the official load_media result (real PIL decode)."""
    _vendor_path()
    from startlux_decide import images as official_images

    uris = [image_data_uri(resolve_path(p)) for p in image_paths]
    return official_images.load_media(uris)


DEFAULT_MIN_PHYSICAL_WIDTH = 1       # no padding unless a caller asks for it
READOUT_MIN_PHYSICAL_WIDTH = 256     # the accepted production stability floor


def pad_graph_inputs(arrays, config, *, min_width, vocab_size=None,
                     max_batch=MAX_BATCH, max_seq_len=MAX_SEQ_LEN):
    """Right-pad the validated 5 graph inputs to the minimum physical width.

    The accepted unified-graph input strategy (see the module docstring and
    work/startlux-cannbot/review-p4-pad256-independent.json): the arrays come
    from multimodal.build_multimodal_batch -- the official per-row encode and
    the official position planes at the TRUE lengths, already strictly
    validated -- and this helper only extends the padded region: input_ids
    right-padded with token id 0, position_ids with 0 (the official
    attention_mask semantics build_multimodal_batch itself uses for padding),
    image_feature_index with -1; actual_seq_lengths (the true per-row lengths,
    the readout gathers L-1) and pixel_values are carried over UNTOUCHED.  The
    width rule is width = max(min_width, current width) -- one deterministic
    floor per call, never a per-case pick; the effective floor never exceeds
    max_seq_len.  The padded set is re-validated by the same shared strict
    validator before returning.
    """
    if type(min_width) is not int or min_width < 1:
        raise ValueError(f"min_width must be a true positive int, got {min_width!r}")
    batch, width = arrays["input_ids"].shape
    target = min(max(min_width, width), max_seq_len)
    if target == width:
        return {name: arrays[name] for name in arrays}
    pad = target - width
    padded = {
        "input_ids": np.pad(arrays["input_ids"], ((0, 0), (0, pad)),
                            constant_values=0),
        "position_ids": np.pad(arrays["position_ids"], ((0, 0), (0, 0), (0, pad)),
                               constant_values=0),
        "actual_seq_lengths": arrays["actual_seq_lengths"].copy(),
        "pixel_values": arrays["pixel_values"].copy(),
        "image_feature_index": np.pad(arrays["image_feature_index"],
                                      ((0, 0), (0, pad)), constant_values=-1),
    }
    multimodal.validate_batch_arrays(
        padded, max_batch=max_batch, max_seq_len=max_seq_len,
        vocab_size=vocab_size, image_token=config.image_token_id,
        vision_start=config.vision_start_token_id,
        vision_end=config.vision_end_token_id)
    return padded


class OfficialPreprocessor:
    """The official no-weight preprocessing chain of the pinned model folder.

    ``model_dir``: the official model folder (tokenizer + configs + official
    image processor settings; never the checkpoint weights).  ``min_width``:
    the default minimum physical width of build_graph_inputs (1 = the true
    batch width; the production callers pass the accepted 256 floor).
    Everything is lazy: a pool of pre-encoded text rows never touches the
    tokenizer, and text-only prompts never construct the image-capable
    processor (which needs the task-local Pillow/torchvision).
    """

    def __init__(self, model_dir, *, min_width=DEFAULT_MIN_PHYSICAL_WIDTH):
        self.model_dir = Path(model_dir)
        if type(min_width) is not int or min_width < 1:
            raise ValueError(f"min_width must be a true positive int, got {min_width!r}")
        self.min_width = min_width
        self._config = None
        self._tokenizer = None
        self._processor = None
        self._rope_index = None

    @property
    def config(self):
        if self._config is None:
            from transformers import AutoConfig

            self._config = AutoConfig.from_pretrained(str(self.model_dir), local_files_only=True)
        return self._config

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir),
                                                            local_files_only=True)
        return self._tokenizer

    @property
    def image_token(self):
        return int(self.config.image_token_id)

    @property
    def rope_index(self):
        """The SHARED official position provider on the folder config."""
        if self._rope_index is None:
            self._rope_index = multimodal.official_rope_index_provider(self.config)
        return self._rope_index

    @property
    def processor(self):
        """The official image-capable processor: the REAL verified composition.

        The full AutoProcessor.from_pretrained(..., backend='pil') factory fails
        on this pinned folder, and Qwen3VLProcessor rejects video_processor=None
        -- the composition that actually works on this folder (verified by
        work/startlux-validation/p6-official and p6-unified runners with the
        task-local Pillow/torchvision) is the three real official components:
        AutoImageProcessor(backend='pil') + AutoTokenizer +
        AutoVideoProcessor.  Image-only calls never feed the video processor;
        it must still be a real official instance because the constructor
        type-checks it.
        """
        if self._processor is None:
            try:
                from transformers.models.auto.image_processing_auto import AutoImageProcessor
                from transformers.models.auto.video_processing_auto import AutoVideoProcessor
                from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
            except ImportError as error:
                raise ImportError(f"{DEPS_HINT} (original error: {error})") from None
            try:
                image_processor = AutoImageProcessor.from_pretrained(
                    str(self.model_dir), local_files_only=True, backend="pil")
                video_processor = AutoVideoProcessor.from_pretrained(
                    str(self.model_dir), local_files_only=True)
            except ImportError as error:
                raise ImportError(f"{DEPS_HINT} (original error: {error})") from None
            self._processor = Qwen3VLProcessor(image_processor=image_processor,
                                                tokenizer=self.tokenizer,
                                                video_processor=video_processor)
        return self._processor

    def encode_media_row(self, prompt, media):
        """The official TransformersBackend._inputs recipe for one request row.

        ``media``: the official startlux_decide.images.load_media result of
        this row's request (or None for a text row).  Returns the official
        encoding dict of ONE row, exactly what multimodal.build_multimodal_batch
        consumes: text rows carry input_ids [1, S] only; media rows add
        pixel_values / image_grid_thw / mm_token_type_ids, with the official
        image-token-count check enforced.
        """
        import torch

        if not isinstance(prompt, str) or not prompt:
            raise ValueError("a row prompt must be a nonempty string")
        if media is None:
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if not ids:
                raise ValueError("the prompt tokenized to 0 tokens; refusing an empty row")
            return {"input_ids": torch.tensor([ids], dtype=torch.int64)}
        if not getattr(media, "images", None):
            raise ValueError("a media row must carry at least one decoded image")
        if len(media.images) > 2:
            raise ValueError(f"a unified-route request carries at most 2 images; this "
                             f"media object has {len(media.images)} (rejected, never "
                             "silently dropped)")
        enc = self.processor(text=[prompt], images=list(media.images),
                             size=media.processor_size, return_tensors="pt")
        ids = enc["input_ids"]
        if int((ids == self.image_token).sum()) != sum(media.tokens):
            raise RuntimeError(f"the official processor gave {int((ids == self.image_token).sum())} "
                               f"image tokens, the official package counted {sum(media.tokens)}")
        row = {key: enc[key] for key in ("input_ids", "pixel_values", "image_grid_thw")
               if key in enc}
        types = enc.get("mm_token_type_ids")
        row["mm_token_type_ids"] = types if types is not None else (ids == self.image_token).int()
        return row

    def encode_row(self, prompt, images=None):
        """One payload row (prompt + optional local image paths) -> the official encoding.

        Paths become the official media object via load_media (data URIs, real
        PIL decode, official smart_resize bounds) and then the SINGLE official
        recipe encode_media_row runs -- no second copy of the arithmetic.
        """
        if images:
            media = load_media_from_paths(images)
            return self.encode_media_row(prompt, media)
        return self.encode_media_row(prompt, None)

    def encode_payload_rows(self, rows):
        """Payload rows -> official per-row encodings (see build_graph_inputs).

        A payload row is {'prompt': str, 'images': [paths]?} (an official
        request, encoded by encode_row) or {'input_ids': [...]} (a pre-encoded
        official text row, e.g. a P2b fixture readout; wrapped as a [1, S]
        int64 tensor with no tokenizer involvement).  input_ids entries must be
        true ints -- bools and floats are rejected, never coerced.
        """
        import torch

        encoded = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"row {index} must be a mapping")
            if "input_ids" in row:
                if "prompt" in row or "images" in row:
                    raise ValueError(f"row {index} mixes input_ids with prompt/images")
                ids = row["input_ids"]
                if not isinstance(ids, list) or not ids \
                        or any(type(t) is not int for t in ids):
                    raise ValueError(f"row {index} input_ids must be a nonempty list of "
                                     "true ints (bools and floats are rejected)")
                encoded.append({"input_ids": torch.tensor([ids], dtype=torch.int64)})
                continue
            if "prompt" not in row:
                raise ValueError(f"row {index} needs either 'prompt' or 'input_ids'")
            images = row.get("images") or []
            if not isinstance(images, list) or any(not isinstance(p, (str, Path))
                                                    for p in images):
                raise ValueError(f"row {index} images must be a list of paths")
            if len(images) > 2:
                raise ValueError(f"row {index} carries {len(images)} images; at most 2 "
                                 "per request")
            encoded.append(self.encode_row(row["prompt"], images))
        return encoded

    def build_graph_inputs(self, rows, *, vocab_size=None, max_batch=MAX_BATCH,
                           max_seq_len=MAX_SEQ_LEN, min_width=None):
        """Official per-row encodings (or payload rows) -> the validated 5 graph inputs.

        The caps are ALWAYS the caller's (the adapter passes the model params;
        the multimodal defaults are only fallbacks for direct tool use).  Every
        structural bound (B<=max_batch, S<=max_seq_len, <=2 images per request,
        <=2 global image slots per call, grid exactly [1, 16, 16]) is enforced
        by multimodal.build_multimodal_batch / validate_batch_arrays;
        violations raise, nothing is silently resized or truncated.

        ``min_width`` (default: this preprocessor's min_width) applies the
        accepted minimum-physical-width strategy: the official encode runs at
        the TRUE lengths first, then pad_graph_inputs extends the validated
        arrays to width = max(min_width, max(true lengths)) -- one floor per
        call, never a per-case pick.  The strict validator runs again on the
        padded set.
        """
        if rows and all(isinstance(row, dict) and "input_ids" in row
                        and hasattr(row["input_ids"], "dim") for row in rows):
            encoded = rows
        else:
            encoded = self.encode_payload_rows(rows)
        arrays = multimodal.build_multimodal_batch(
            encoded, self.rope_index, self.config, vocab_size=vocab_size,
            max_batch=max_batch, max_seq_len=max_seq_len)
        if min_width is None:
            min_width = self.min_width
        return pad_graph_inputs(arrays, self.config, min_width=min_width,
                                vocab_size=vocab_size, max_batch=max_batch,
                                max_seq_len=max_seq_len)


def encode_case_rows(preprocessor, case, *, vocab_size=None, max_batch=MAX_BATCH,
                     max_seq_len=MAX_SEQ_LEN, min_width=None):
    """One unified case payload -> the 5 graph input arrays (the adapter encode path).

    Payload contract of the unified route (generate_unified_cases.py):
    ``payload.rows``: one entry per batch row, {'prompt': str, 'images': [paths]?}
    (an image request) or {'input_ids': [...]} (a pre-encoded official text
    row); ``payload.rows_meta`` rides along as OPTIONAL business metadata --
    the graph encode never reads it.  ``samples`` must be a true int equal to
    the row count; the caller's caps and vocab bound are enforced by the
    builder, never relaxed here.  ``min_width`` defaults to the
    preprocessor's floor (the adapter passes the accepted production floor).
    """
    samples = case.get("samples")
    if type(samples) is not int or samples < 1:
        raise ValueError(f"unified case {case['id']} samples must be a true positive "
                         f"integer, got {samples!r}")
    payload = case.get("payload")
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list) \
            or not payload["rows"]:
        raise ValueError(f"unified case {case['id']} payload must carry a nonempty "
                         "'rows' list")
    if len(payload["rows"]) != samples:
        raise ValueError(f"unified case {case['id']} samples={samples} but the payload "
                         f"carries {len(payload['rows'])} row(s)")
    return preprocessor.build_graph_inputs(payload["rows"], vocab_size=vocab_size,
                                           max_batch=max_batch,
                                           max_seq_len=max_seq_len,
                                           min_width=min_width)


def encode_readout_row(preprocessor, prompt, media, spec, *,
                       min_width=READOUT_MIN_PHYSICAL_WIDTH):
    """One business read-out row -> the validated 5 graph inputs (shared entry).

    THE single encode entry of the official business read-out path (the
    UnifiedReadoutSession and its Backend): the official encode_media_row recipe
    on the official media object (or None), then preprocessor.build_graph_inputs
    with the session's caps -- the SAME strategy and the SAME shared validator
    as the case-encode path, including the accepted minimum-physical-width
    floor (default 256; see the module docstring).  Overlong prompts and
    over-sized media are rejected, never truncated.
    """
    row = preprocessor.encode_media_row(prompt, media)
    return preprocessor.build_graph_inputs(
        [row], vocab_size=spec.vocab_size, max_batch=1,
        max_seq_len=spec.length_bound, min_width=min_width)
