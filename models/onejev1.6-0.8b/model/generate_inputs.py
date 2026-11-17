"""Graph-input generation for the OneJev1.6-0.8B multimodal decision model.

Consumes ``fixtures/cases.jsonl`` (one System One request per row, exactly one
typed question and exactly one image) and writes the executor input pool under
``work/inputs/``:

* ``inputs.bin``   the packed per-case graph-input tensors (9 per case, in the
                   shared order below);
* ``records.json`` the per-case tensor index (schema ``ge-records/3``: the
  ``tensors`` array is bound positionally to the graph inputs, ``samples`` is
  the case's business row count) plus the derived A..Z answer-slot token ids;
* ``readout.json`` the per-case read-out metadata (schema
  ``startlux-text-readout/1``: slot labels, letters_asked = K, temperature).

The 9-tensor contract, in order and without padding:

1. ``input_ids``            int64     [1,S]   prompt tokens, image pad expanded
2. ``pixel_values``         bfloat16  [P,1536] official image processor output
3. ``image_grid_thw``       int64     [1,3]   t=1, h, w patch grid
4. ``position_ids``         int64     [3,1,S] official Qwen3_5Model.get_rope_index
5. ``interp_indices``       int64     [P,4]   bilinear pos-embed taps
6. ``interp_weights``       float32   [P,4]   the matching weights
7. ``vision_position_ids``  int64     [P,2]   (h, w) axial ids
8. ``cu_seqlens``           int32     [2]     [0, P]
9. ``feature_index``        int64     [1,S]   merged feature row per image token

Each row becomes ONE B=1 case at its TRUE length.  The prompt is rendered by the
vendored official ``onejev-labels-v2`` renderer (``build_messages`` +
``render_question``), the media placeholder is replaced by the chat template's
vision block, and the single ``<|image_pad|>`` token is expanded to
``grid.prod(-1) // merge_size**2`` copies exactly as the official
``OneJev._prefix_and_suffix_media`` Qwen path does.  Tokens are encoded with
``tokenizer.encode(prompt, add_special_tokens=False)``.  ``position_ids`` come
from the official ``get_rope_index`` running on a meta-device model (it reads
only the config and its input tensors), so no checkpoint weights are loaded.

The image processor is transformers' ``Qwen2VLImageProcessorPil`` -- the PIL
backend of the same 5.17.0 ``Qwen2VLImageProcessor`` (identical
smart_resize/patchify/rescale/normalize, no torchvision dependency, which this
host does not have).  The processor's fp32 ``pixel_values`` are cast to bf16 on
the host, as the graph contract requires.  The fixtures' source sizes are all
factor-32 aligned, so the PIL and torchvision backends agree bit for bit after
that bf16 cast (checked at vendor time); the backend choice does not change the
graph inputs.

Steps of this pipeline (generate_inputs -> golden -> export_air -> executor)
never import each other; they interact only through the files under ``work/``.
Paths default to ``fixtures/cases.jsonl`` and ``work/inputs``; ``--cases`` /
``--output-dir`` override them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parents[1]
DEPS_DIR = ADAPTER_DIR / "work" / "deps" / "py"
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

import transformers

if transformers.__version__ != "5.17.0":
    raise RuntimeError(f"OneJev needs the task-local transformers 5.17.0 from {DEPS_DIR}, "
                       f"got {transformers.__version__} from {transformers.__file__}")

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, Qwen3_5Model
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

if str(ADAPTER_DIR / "vendor") not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR / "vendor"))

from flide.api.schema import ChoiceQuestion, NoulCriteria, NoulQuestion, ScoreQuestion
from flide_onejev.calibrate import Calibration
from flide_onejev.media import MediaError, check_items, render_prompt
from flide_onejev.prompt import build_messages, render_question
from flide_onejev.slots import verify_slot_token_ids

OUTPUT_NAME = "scores"
TENSOR_ORDER = ("input_ids", "pixel_values", "image_grid_thw", "position_ids",
                "interp_indices", "interp_weights", "vision_position_ids",
                "cu_seqlens", "feature_index")


def weights_dir():
    """ONEJEV_MODEL_DIR resolved once, loudly."""
    value = os.environ.get("ONEJEV_MODEL_DIR")
    if not value or not value.strip():
        raise ValueError("ONEJEV_MODEL_DIR is not set or is empty")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"ONEJEV_MODEL_DIR {path} is not a directory")
    return path


def model_bounds(folder):
    """The config values this flow pins: text bounds and the vision grid facts."""
    config = AutoConfig.from_pretrained(str(folder), local_files_only=True)
    text = config.text_config
    vocab = text.vocab_size
    max_position_embeddings = text.max_position_embeddings
    if not isinstance(vocab, int) or vocab < 1 \
            or not isinstance(max_position_embeddings, int) or max_position_embeddings < 1:
        raise ValueError(f"cannot read vocab_size / max_position_embeddings from "
                         f"{folder / 'config.json'}")
    vision = config.vision_config
    merge = vision.spatial_merge_size
    patch = vision.patch_size
    side = int(vision.num_position_embeddings ** 0.5)
    if side * side != vision.num_position_embeddings:
        raise ValueError(f"num_position_embeddings {vision.num_position_embeddings} is not a square")
    expected = {"image_token_id": 248056, "vision_start_token_id": 248053,
                "vision_end_token_id": 248054, "patch_size": 16, "spatial_merge_size": 2,
                "num_grid_per_side": 48}
    actual = {"image_token_id": config.image_token_id,
              "vision_start_token_id": config.vision_start_token_id,
              "vision_end_token_id": config.vision_end_token_id,
              "patch_size": patch, "spatial_merge_size": merge, "num_grid_per_side": side}
    if actual != expected:
        raise ValueError(f"{folder / 'config.json'} vision facts changed: {actual} != {expected}")
    return config, vocab, max_position_embeddings, side, merge


def _entry(value, where, name):
    """One criteria/instructions entry (string, object, array or null), strictly."""
    if value is not None and not isinstance(value, (str, dict, list)):
        raise ValueError(f"{where}: {name} must be a string, object, array or null")
    return value


def _question(where, qid, spec):
    """The typed question object of one row, with the v1 option-count rules."""
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError(f"{where}: question ids must be nonempty strings")
    if not isinstance(spec, dict):
        raise ValueError(f"{where}: question {qid!r} must be an object")
    kind = spec.get("type")
    instructions = _entry(spec.get("instructions"), where, "instructions")
    if kind == "choice":
        criteria = spec.get("criteria")
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 26:
            raise ValueError(f"{where}: question {qid!r} needs 2..26 criteria")
        for key, value in criteria.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{where}: criterion ids must be nonempty strings")
            _entry(value, where, f"criterion {key!r} description")
        return ChoiceQuestion(type="choice", instructions=instructions, criteria=criteria), list(criteria)
    if kind == "noul":
        criteria = spec.get("criteria")
        if criteria is None:
            return NoulQuestion(type="noul", instructions=instructions), ["yes", "no"]
        if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
            raise ValueError(f"{where}: noul criteria may only carry true/false descriptions")
        for key, value in criteria.items():
            _entry(value, where, f"noul criterion {key!r}")
        return NoulQuestion(type="noul", instructions=instructions,
                            criteria=NoulCriteria(**criteria)), ["yes", "no"]
    if kind == "score":
        criteria = spec.get("criteria")
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 26:
            raise ValueError(f"{where}: question {qid!r} needs 2..26 score levels")
        for i, value in enumerate(criteria):
            _entry(value, where, f"score level {i}")
        return ScoreQuestion(type="score", instructions=instructions,
                             criteria=criteria), [str(i) for i in range(len(criteria))]
    raise ValueError(f"{where}: question {qid!r} type must be choice, noul or score, got {kind!r}")


def _load_rows(path):
    """The fixtures JSONL rows, strictly validated (the shared case contract)."""
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path}: no JSONL rows")
    rows = []
    for lineno, line in enumerate(lines, 1):
        where = f"{path.name}:{lineno}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}: not valid JSON: {error}") from error
        if not isinstance(row, dict) or set(row) != {"state", "media", "questions"}:
            raise ValueError(f"{where}: each row must carry exactly state/media/questions")
        state = row["state"]
        if not isinstance(state, dict) or not state:
            raise ValueError(f"{where}: state must be a nonempty mapping")
        try:
            items = check_items(row["media"])
        except MediaError as error:
            raise ValueError(f"{where}: {error}") from error
        if len(items) != 1 or items[0].get("type") != "image":
            raise ValueError(f"{where}: exactly one image in media is supported")
        ref = items[0].get("path")
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError(f"{where}: image media needs a nonempty relative path")
        media_path = Path(ref)
        if not media_path.is_absolute():
            media_path = path.parent / media_path
        media_path = media_path.resolve()
        if not media_path.is_file():
            raise ValueError(f"{where}: image {media_path} does not exist")
        questions = row["questions"]
        if not isinstance(questions, dict) or len(questions) != 1:
            raise ValueError(f"{where}: exactly one question per row is supported")
        qid, spec = next(iter(questions.items()))
        question, labels = _question(where, qid, spec)
        rows.append({"where": where, "state": state, "items": items, "qid": qid,
                     "question": question, "labels": labels, "image": media_path})
    return rows


def _image_tensors(image_processor, image_path, where):
    """Official image-processor output: fp32 pixel_values [P,1536] and grid [1,3]."""
    try:
        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
    except Exception as error:  # unreadable file: the fixture's fault, report the row
        raise ValueError(f"{where}: cannot read image {image_path}: {error}") from error
    out = image_processor(images=[image], return_tensors="pt")
    pixel_values = out["pixel_values"]
    grid = out["image_grid_thw"]
    if pixel_values.dtype != torch.float32 or pixel_values.dim() != 2 \
            or pixel_values.shape[1] != 1536:
        raise ValueError(f"{where}: image processor gave {pixel_values.dtype} "
                         f"{tuple(pixel_values.shape)}, expected float32 [P, 1536]")
    if grid.dtype != torch.int64 or tuple(grid.shape) != (1, 3) or int(grid[0, 0]) != 1:
        raise ValueError(f"{where}: image processor gave grid {grid.dtype} "
                         f"{tuple(grid.shape)}, expected int64 [1, 3] with t=1")
    return pixel_values, grid


def _bf16_bytes(tensor):
    """bf16 payload as raw little-endian bytes (numpy has no bfloat16 dtype here)."""
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _append(stream, tensors, name, dtype, shape, data):
    tensors.append({"name": name, "dtype": dtype, "shape": list(shape),
                    "offset": stream.tell(), "bytes": len(data)})
    stream.write(data)


def main():
    parser = argparse.ArgumentParser(description="Generate the OneJev multimodal graph inputs "
                                                 "(work/inputs)")
    parser.add_argument("--cases", type=Path,
                        default=ADAPTER_DIR / "fixtures" / "cases.jsonl",
                        help="the fixture JSONL case file (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path,
                        default=ADAPTER_DIR / "work" / "inputs",
                        help="directory for inputs.bin, records.json and readout.json "
                             "(default: %(default)s)")
    args = parser.parse_args()
    if not args.cases.is_file():
        raise SystemExit(f"generate_inputs: case file {args.cases} does not exist; place "
                         "cases.jsonl there or pass --cases")

    folder = weights_dir()
    config, vocab, max_position_embeddings, side, merge = model_bounds(folder)
    image_token_id = config.image_token_id

    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    image_processor = Qwen2VLImageProcessorPil.from_pretrained(str(folder), local_files_only=True)
    if tokenizer.convert_tokens_to_ids("<|image_pad|>") != image_token_id:
        raise ValueError(f"{folder}: tokenizer <|image_pad|> id does not match config "
                         f"image_token_id {image_token_id}")

    slot_token_ids = verify_slot_token_ids(tokenizer)[:26]
    if len(slot_token_ids) != 26 or len(set(slot_token_ids)) != 26:
        raise ValueError("the 26 A..Z answer slots are not distinct single round-trip tokens")
    calibration_path = folder / "calibration.json"
    calibration = Calibration.load(calibration_path) if calibration_path.is_file() else Calibration()

    rows = _load_rows(args.cases)
    with torch.device("meta"):
        rope_model = Qwen3_5Model(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, readout_cases = [], []
    with (args.output_dir / "inputs.bin").open("wb") as stream:
        for index, row in enumerate(rows):
            case_id = f"onejev-{index:02d}"
            where = row["where"]

            rendered = render_question(row["qid"], row["question"])
            if rendered.labels != row["labels"]:
                raise ValueError(f"{where}: rendered slot labels {rendered.labels} disagree "
                                 f"with the question labels {row['labels']}")
            messages = build_messages(row["state"], rendered.suffix)
            try:
                prompt = render_prompt(tokenizer, {"enable_thinking": False}, messages, row["items"])
            except MediaError as error:
                raise ValueError(f"{where}: {error}") from error
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not ids:
                raise ValueError(f"{where}: the prompt tokenized to 0 tokens")
            pads = [i for i, token in enumerate(ids) if token == image_token_id]
            if len(pads) != 1:
                raise ValueError(f"{where}: expected exactly one <|image_pad|> token in the "
                                 f"prompt, found {len(pads)}")
            pad_at = pads[0]

            pixel_values, grid = _image_tensors(image_processor, row["image"], where)
            count = int(grid.prod(-1)[0]) // merge ** 2
            patches = int(pixel_values.shape[0])
            if int(grid.prod(-1)[0]) != patches:
                raise ValueError(f"{where}: grid {grid.tolist()} does not cover {patches} patches")
            if count < 1:
                raise ValueError(f"{where}: the image merges to {count} feature rows")

            expanded = ids[:pad_at] + [image_token_id] * count + ids[pad_at + 1:]
            length = len(expanded)
            if length > max_position_embeddings:
                raise ValueError(f"{where}: prompt length {length} exceeds the model's "
                                 f"max_position_embeddings {max_position_embeddings}")
            if min(expanded) < 0 or max(expanded) >= vocab:
                raise ValueError(f"{where}: input_ids outside the vocabulary bounds [0, {vocab})")

            feature_index = np.full((1, length), -1, dtype=np.int64)
            feature_index[0, pad_at:pad_at + count] = np.arange(count, dtype=np.int64)
            if int((feature_index >= 0).sum()) != count \
                    or not np.array_equal(feature_index[0, pad_at:pad_at + count],
                                          np.arange(count, dtype=np.int64)):
                raise ValueError(f"{where}: feature_index is not the contiguous image span")
            if int((np.asarray(expanded) == image_token_id).sum()) != count:
                raise ValueError(f"{where}: image tokens {count} != merged rows {patches // merge ** 2}")
            if patches % (merge ** 2) or count != patches // merge ** 2:
                raise ValueError(f"{where}: P={patches} is not {merge ** 2} * merged rows")

            input_ids = torch.tensor([expanded], dtype=torch.int64)
            mm_token_type_ids = (input_ids == image_token_id).to(torch.int32)
            position_ids, _ = rope_model.get_rope_index(
                input_ids, mm_token_type_ids, image_grid_thw=grid)
            if tuple(position_ids.shape) != (3, 1, length) or position_ids.dtype != torch.int64:
                raise ValueError(f"{where}: get_rope_index gave {position_ids.dtype} "
                                 f"{tuple(position_ids.shape)}, expected int64 [3, 1, {length}]")

            interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
                grid, num_grid_per_side=side, mode="bilinear", align_corners=True,
                spatial_merge_size=merge)
            vision_position_ids = get_vision_position_ids(grid, merge)
            cu_seqlens = get_vision_cu_seqlens(grid)
            if tuple(interp_indices.shape) != (patches, 4) or interp_indices.dtype != torch.int64:
                raise ValueError(f"{where}: interp_indices {interp_indices.dtype} "
                                 f"{tuple(interp_indices.shape)}, expected int64 [{patches}, 4]")
            if tuple(interp_weights.shape) != (patches, 4) or interp_weights.dtype != torch.float32:
                raise ValueError(f"{where}: interp_weights {interp_weights.dtype} "
                                 f"{tuple(interp_weights.shape)}, expected float32 [{patches}, 4]")
            if tuple(vision_position_ids.shape) != (patches, 2) \
                    or vision_position_ids.dtype != torch.int64:
                raise ValueError(f"{where}: vision_position_ids {vision_position_ids.dtype} "
                                 f"{tuple(vision_position_ids.shape)}, expected int64 [{patches}, 2]")
            if tuple(cu_seqlens.shape) != (2,) or cu_seqlens.dtype != torch.int32 \
                    or cu_seqlens.tolist() != [0, patches]:
                raise ValueError(f"{where}: cu_seqlens {cu_seqlens.dtype} {cu_seqlens.tolist()}, "
                                 f"expected int32 [0, {patches}]")

            pixel_bf16 = pixel_values.to(torch.bfloat16)
            tensors = []
            _append(stream, tensors, "input_ids", "int64", (1, length),
                    input_ids.numpy().tobytes(order="C"))
            _append(stream, tensors, "pixel_values", "bfloat16",
                    (patches, pixel_bf16.shape[1]), _bf16_bytes(pixel_bf16))
            _append(stream, tensors, "image_grid_thw", "int64", (1, 3),
                    grid.numpy().astype(np.int64).tobytes(order="C"))
            _append(stream, tensors, "position_ids", "int64", (3, 1, length),
                    position_ids.numpy().tobytes(order="C"))
            _append(stream, tensors, "interp_indices", "int64", (patches, 4),
                    interp_indices.numpy().astype(np.int64).tobytes(order="C"))
            _append(stream, tensors, "interp_weights", "float32", (patches, 4),
                    interp_weights.numpy().astype(np.float32).tobytes(order="C"))
            _append(stream, tensors, "vision_position_ids", "int64", (patches, 2),
                    vision_position_ids.numpy().astype(np.int64).tobytes(order="C"))
            _append(stream, tensors, "cu_seqlens", "int32", (2,),
                    cu_seqlens.numpy().astype(np.int32).tobytes(order="C"))
            _append(stream, tensors, "feature_index", "int64", (1, length),
                    feature_index.tobytes(order="C"))
            if tuple(t["name"] for t in tensors) != TENSOR_ORDER:
                raise AssertionError("the emitted tensor order drifted from the 9-tensor contract")

            # samples = 1: one System One request row == one branch == one case.
            records.append({"id": case_id, "samples": 1, "tensors": tensors})
            readout_cases.append({"id": case_id, "labels": list(rendered.labels),
                                  "letters_asked": len(rendered.labels),
                                  "temperature": calibration.temperature(rendered.kind,
                                                                         len(rendered.labels))})
            print(f"generate_inputs: {case_id} S={length} P={patches} grid={grid.tolist()}")
            if index == 0:
                print(f"generate_inputs: first rendered prompt (500 chars):\n{prompt[:500]}")

    (args.output_dir / "records.json").write_text(json.dumps({
        "schema": "ge-records/3",
        "file": str((args.output_dir / "inputs.bin").resolve()),
        "output_name": OUTPUT_NAME,
        "slot_token_ids": slot_token_ids,
        "cases": records,
    }, indent=1) + "\n")
    (args.output_dir / "readout.json").write_text(json.dumps({
        "schema": "startlux-text-readout/1",
        "temperature_source": "calibration.json" if calibration_path.is_file()
                              else "default 1.0 (no calibration.json in the model folder)",
        "cases": readout_cases,
    }, indent=1) + "\n")
    print(f"generate_inputs: {len(records)} cases; inputs: {args.output_dir}")


if __name__ == "__main__":
    main()
