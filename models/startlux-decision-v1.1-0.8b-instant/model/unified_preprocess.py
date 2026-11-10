"""Official CPU preprocessing for the StartLux unified multimodal AIR.

The unified single-AIR contract (modeling_multimodal.py) keeps the OFFICIAL image
preprocessing and the OFFICIAL get_rope_index on the CPU side: the graph inputs
pixel_values / image_feature_index / position_ids are produced from what the
official processor and the official position method emit, nothing else.  This
module is that CPU side:

* ``OfficialPreprocessor`` loads ONLY the official no-weight components of the
  pinned model folder: AutoTokenizer, AutoConfig and the official image-capable
  processor.  No model weights are ever loaded here and no NPU context is
  created.  Every component is lazy: a pool of pre-encoded text rows runs with
  no tokenizer and no Pillow at all.
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
  max_seq_len / vocab_size are always passed in from the model params; the
  builder defaults are never assumed).  Every structural bound -- fixed grid
  [1, 16, 16], images per request, global image slots per call, B and S caps --
  is enforced there; nothing is stretched, cropped or truncated here.

Every case runs at its TRUE length: width = max(true row lengths), so dynamic S
stays dynamic and actual_seq_lengths carries the per-row true lengths (the
readout gathers L-1).

Pillow / torchvision are TASK-LOCAL dependencies of the official image path
(see DEPS_HINT); nothing here installs or patches system packages.
"""
from __future__ import annotations

import base64
import mimetypes
import sys
from pathlib import Path

from model import modeling_multimodal as multimodal

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root
ROOT = MODEL_DIR.parents[1]

DEPS_HINT = ("the official image preprocessing needs Pillow (and torchvision for the "
             "official video processor instance); both are task-local dependencies. "
             "The validated task-local install is "
             "work/startlux-validation/p6-official/deps/pillow and "
             "work/startlux-validation/p6-official/deps/torchvision; run.sh prepends "
             "them to PYTHONPATH for its python stages only. Do not install them "
             "system-wide.")

MAX_BATCH = multimodal.MAX_BATCH                  # 4
MAX_SEQ_LEN = multimodal.MAX_SEQ_LEN              # 2048


def _vendor_path():
    if str(MODEL_DIR / "decider" / "vendor") not in sys.path:
        sys.path.insert(0, str(MODEL_DIR / "decider" / "vendor"))


def image_data_uri(path):
    """A local image file as the data URI the official load_media reads."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
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


class OfficialPreprocessor:
    """The official no-weight preprocessing chain of the pinned model folder.

    ``model_dir``: the official model folder (tokenizer + configs + official
    image processor settings; never the checkpoint weights).  Everything is
    lazy: a pool of pre-encoded text rows never touches the tokenizer, and
    text-only prompts never construct the image-capable processor (which needs
    the task-local Pillow/torchvision).
    """

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
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
        """The official image-capable processor, composed from its three official
        components.

        The full AutoProcessor.from_pretrained(..., backend='pil') factory fails
        on this pinned folder, and Qwen3VLProcessor rejects video_processor=None,
        so the image processor, tokenizer and video processor are built
        separately.  Image-only calls never feed the video processor; it must
        still be a real official instance because the constructor type-checks it.
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
            encoded.append(self.encode_row(row["prompt"], images))
        return encoded

    def build_graph_inputs(self, rows, *, vocab_size=None, max_batch=MAX_BATCH,
                           max_seq_len=MAX_SEQ_LEN):
        """Official per-row encodings -> the 5 validated graph inputs.

        The caps are ALWAYS the caller's (the model params; the multimodal
        defaults are only fallbacks for direct tool use).  Every structural
        bound (B<=max_batch, S<=max_seq_len, images per request, global image
        slots per call, grid exactly [1, 16, 16]) is enforced by
        multimodal.build_multimodal_batch; violations raise, nothing is
        silently resized or truncated.  Rows run at their TRUE lengths, so
        width = max(true row lengths) and dynamic S stays dynamic.
        """
        return multimodal.build_multimodal_batch(
            rows, self.rope_index, self.config, vocab_size=vocab_size,
            max_batch=max_batch, max_seq_len=max_seq_len)
