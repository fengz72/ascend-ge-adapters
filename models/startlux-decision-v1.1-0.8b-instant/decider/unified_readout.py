"""Real unified-AIR read-out seam for StartLux-Decision v1.1 0.8B Instant (P6 production).

This module wires the official business pipeline (decision.py's StartluxDecision
on the vendored startlux_decide Decider) to ONE native GESession execution of
the exported UNIFIED multimodal AIR.  The single graph call carries exactly the
5 unified inputs -- input_ids / position_ids (the official get_rope_index
planes) / actual_seq_lengths / pixel_values (the static 2-slot pool) /
image_feature_index -- and NOTHING else: no vision features are precomputed
outside the graph, the vision tower runs inside it.

* ``UnifiedGraphSpec.from_io`` validates the AIR's embedded IOSpec against the
  unified contract (exact names, dtypes, dynamic shapes: input_ids [-1,-1],
  position_ids [3,-1,-1], actual_seq_lengths [-1], pixel_values [512,1536]
  STATIC float32, image_feature_index [-1,-1] -> scores [-1,26] float32) and
  takes the caller's explicit max_seq_len bound (the io declares none; it is
  never inferred).
* ``UnifiedReadoutSession.read_scores26(prompt, media)`` encodes ONE request
  row through the OFFICIAL preprocessing (unified_preprocess.OfficialPrepro-
  cessor.encode_media_row: the official TransformersBackend._inputs recipe on
  the official media object the Decider passes -- real PIL images, official
  processor, official smart_resize bounds), builds the validated 5 graph inputs
  (multimodal.build_multimodal_batch with the session's caps), then runs ONE
  ``ge_runtime`` subprocess per read-out (plan mode infer, backend ge_session,
  graph_run_mode=0, must_keep_origin_dtype).  The subprocess is cold-compiled
  and reloaded on EVERY call and the measured compile/load costs are recorded
  per read-out -- no warm-business-QPS claims.  Stdout/stderr go to a regular
  log file (never an inherited pipe), every call has a timeout and the
  returncode / exactly-one-GE_RESULT-line contract is enforced.  Each session
  writes into its own fresh ``session-NNNN`` directory and every read-out keeps
  its raw ``inputs.bin`` / ``outputs.bin`` / ``plan.json`` / log as evidence.
* ``UnifiedReadoutBackend`` is the official Backend with ``reads_images``:
  ``readout(prompt, letter_ids, media)`` runs the graph and returns the RAW
  logits of the requested letters, selected by column from the 26-column
  scores (the folder's letter_token_ids map onto the columns; temperature and
  softmax stay in the official pipeline).  Media arrives as the official
  load_media object and is fed to the official recipe directly -- never
  treated as paths.  Instant models never generate.
* ``startlux_unified_decision`` wires decision.py's StartluxDecision (the
  official Decider, official none/wide-list/image-state semantics untouched)
  to this backend; the caller loads no model weights and creates no NPU vision
  context -- the only NPU work happens inside the ge_runtime subprocess.

Real GE execution is the formally accepted route (P7: the six-stage CLI and the
13-request/17-readout business acceptance ran this session against the formal
unified AIR -- see work/startlux-cannbot/review-p7-cli-independent.json and
work/startlux-validation/p7-final/).  The CPU tests here use a fake runtime
implementing the same JSON plan protocol and the REAL official media/
letter-column path with the real tokenizer/processor.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root
ROOT = MODEL_DIR.parents[1]

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


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


ge_readout = _load_module("startlux_ge_readout_for_unified", "decider/ge_readout.py")
unified_preprocess = _load_module("startlux_unified_preprocess_for_readout",
                                  "model/unified_preprocess.py")
multimodal = _load_module("startlux_multimodal_for_readout", "model/multimodal.py")

if str(MODEL_DIR / "decider" / "vendor") not in sys.path:      # the pinned official package
    sys.path.insert(0, str(MODEL_DIR / "decider" / "vendor"))
from startlux_decide.backends.base import Backend, BackendRequestError  # noqa: E402

ReadoutError = ge_readout.ReadoutError
run_plan = ge_readout.run_plan
read_air_io = ge_readout.read_air_io
parse_timings = ge_readout.parse_timings
source_vocab_size = ge_readout.source_vocab_size

RUNTIME = ge_readout.RUNTIME
INPUT_NAMES = ("input_ids", "position_ids", "actual_seq_lengths", "pixel_values",
               "image_feature_index")
INPUT_DTYPES = {"input_ids": "int64", "position_ids": "int64",
                "actual_seq_lengths": "int32", "pixel_values": "float32",
                "image_feature_index": "int64"}
OUTPUT_NAME = "scores"
LETTER_COLUMNS = 26
SLOT_PIXEL_ROWS = multimodal.PATCH_ROWS            # 256 patch rows per image slot
SLOTS = multimodal.SLOTS                           # 2 fixed global image slots
DEFAULT_OPTIONS = {"graph_run_mode": 0, "precision_mode": "must_keep_origin_dtype"}
MAX_IMAGES_PER_REQUEST = 2


class UnifiedGraphSpec:
    """The unified AIR geometry read from the graph's embedded IOSpec.

    The io declares no length bound (dynamic S), so the caller's explicit
    max_seq_len is REQUIRED -- never inferred.  Batch is fixed at 1 per
    read-out (the business Decider asks one prompt at a time); the graph
    itself stays the dynamic B1..4 single AIR.
    """

    SHAPES = {"input_ids": [-1, -1], "position_ids": [3, -1, -1],
              "actual_seq_lengths": [-1], "pixel_values": [SLOTS * SLOT_PIXEL_ROWS, 1536],
              "image_feature_index": [-1, -1], OUTPUT_NAME: [-1, LETTER_COLUMNS]}
    DYNAMIC_DIMS = {"input_ids": [0, 1], "position_ids": [1, 2],
                    "actual_seq_lengths": [0], "pixel_values": [],
                    "image_feature_index": [0, 1], OUTPUT_NAME: [0]}

    def __init__(self, io, vocab_size=None, max_seq_len=None):
        self.io = io
        self.batch = 1
        self.vocab_size = vocab_size
        if type(max_seq_len) is not int or max_seq_len < 1:
            raise ReadoutError("a dynamic unified AIR needs the caller's explicit "
                               "max_seq_len bound (a positive true int; the io declares "
                               f"none and none is inferred) -- got {max_seq_len!r}")
        self.max_seq_len = max_seq_len
        self.inputs = {node["name"]: node for node in io["inputs"]}
        self.outputs = {node["name"]: node for node in io["outputs"]}
        self.output_shape = (self.batch, LETTER_COLUMNS)

    @classmethod
    def from_io(cls, io, vocab_size=None, max_seq_len=None):
        try:
            ge_readout.validate_io(io)
        except ValueError as error:
            raise ReadoutError(f"the AIR's IOSpec is not valid: {error}") from None
        inputs = {node["name"]: node for node in io["inputs"]}
        outputs = {node["name"]: node for node in io["outputs"]}
        if sorted(inputs) != sorted(INPUT_NAMES):
            raise ReadoutError(f"the AIR inputs {sorted(inputs)} are not the unified "
                               f"contract {sorted(INPUT_NAMES)}")
        if sorted(outputs) != [OUTPUT_NAME]:
            raise ReadoutError(f"the AIR outputs {sorted(outputs)} are not "
                               f"['{OUTPUT_NAME}']")
        for name, dtype in INPUT_DTYPES.items():
            if inputs[name]["dtype"] != dtype:
                raise ReadoutError(f"input {name} dtype {inputs[name]['dtype']} is not "
                                   f"{dtype}")
        for name, shape in cls.SHAPES.items():
            node = inputs.get(name) or outputs[name]
            if list(node["shape"]) != shape:
                raise ReadoutError(f"the IOSpec of {name} (shape {node['shape']}) does not "
                                   f"match the unified contract {shape}")
            dims = list(node.get("dynamic_dims") or [])
            if dims != cls.DYNAMIC_DIMS[name]:
                raise ReadoutError(f"the dynamic_dims of {name} ({dims}) do not match the "
                                   f"unified contract {cls.DYNAMIC_DIMS[name]}")
        if outputs[OUTPUT_NAME]["dtype"] != "float32":
            raise ReadoutError(f"scores dtype {outputs[OUTPUT_NAME]['dtype']} is not "
                               "float32")
        if vocab_size is not None and (type(vocab_size) is not int or vocab_size < 1):
            raise ReadoutError(f"vocab_size must be a positive true int, got {vocab_size!r}")
        return cls(io, vocab_size, max_seq_len)

    @property
    def length_bound(self):
        return self.max_seq_len

    @property
    def scores_bytes(self):
        return math.prod(self.output_shape) * 4


class UnifiedReadoutSession:
    """One read-out = one ge_runtime subprocess over the unified AIR (cold each time).

    Every session creates its own fresh ``session-NNNN`` directory under
    run_dir (never reusing a non-empty one); every read-out gets a subdirectory
    holding plan.json, the raw inputs.bin / outputs.bin (kept as evidence, never
    deleted) and the runtime log.  ``records`` keeps the measured wall /
    compile / load costs so the cold-start overhead of this correctness-first
    flow is reported honestly.
    """

    def __init__(self, device, run_dir, *, runtime_bin=RUNTIME, air_path=None,
                 spec=None, timeout=600.0, seed=42, options=None, preprocessor=None,
                 model_dir=None, vocab_size=None, max_seq_len=None,
                 min_width=unified_preprocess.READOUT_MIN_PHYSICAL_WIDTH):
        self.min_width = int(min_width)
        if self.min_width < 1:
            raise ReadoutError(f"min_width must be >= 1, got {min_width}")
        self.device = int(device)
        if self.device < 0:
            raise ReadoutError(f"device index must be >= 0, got {device}")
        self.runtime_bin = Path(runtime_bin)
        self.timeout = float(timeout)
        self.seed = int(seed)
        self.options = dict(options) if options is not None else dict(DEFAULT_OPTIONS)
        if self.options.get("graph_run_mode", 0) != 0:
            raise ReadoutError("this read-out is graph_run_mode=0 (inference mode) only")
        if "precision_mode" not in self.options:
            self.options["precision_mode"] = DEFAULT_OPTIONS["precision_mode"]
        if preprocessor is None:
            if model_dir is None:
                raise ReadoutError("give either a preprocessor or the model folder for "
                                   "the official no-weight preprocessing")
            preprocessor = unified_preprocess.OfficialPreprocessor(model_dir)
            if vocab_size is None:
                vocab_size = source_vocab_size(model_dir)
        self.preprocessor = preprocessor
        if air_path is None:
            raise ReadoutError("the unified AIR path is required")
        self.air_path = Path(air_path)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        index = 1
        while True:
            candidate = self.run_dir / f"session-{index:04d}"
            try:
                candidate.mkdir()
            except FileExistsError:
                index += 1
                continue
            self.session_dir = candidate
            break
        if spec is None:
            io = read_air_io(self.runtime_bin, self.air_path,
                             self.session_dir / "read-io.log", self.timeout)
            spec = UnifiedGraphSpec.from_io(io, vocab_size, max_seq_len=max_seq_len)
        self.spec = spec
        if max_seq_len is not None and self.spec.max_seq_len != max_seq_len:
            raise ReadoutError(f"max_seq_len disagrees: spec has {self.spec.max_seq_len}, "
                               f"caller passed {max_seq_len}")
        if vocab_size is not None:
            if self.spec.vocab_size is None:
                self.spec.vocab_size = vocab_size
            elif self.spec.vocab_size != vocab_size:
                raise ReadoutError(f"vocab_size disagrees: spec has {self.spec.vocab_size}, "
                                   f"caller passed {vocab_size}")
        self.records: list[dict] = []

    def _next_dir(self):
        index = len(self.records)
        directory = self.session_dir / f"readout-{index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        return index, directory

    def read_scores26(self, prompt, media=None):
        """All 26 letter scores of one request (finite float list), one graph call."""
        index, directory = self._next_dir()
        case_id = f"readout-{index:04d}"
        try:
            arrays = unified_preprocess.encode_readout_row(
                self.preprocessor, prompt, media, self.spec, min_width=self.min_width)
        except ValueError as error:
            raise ReadoutError(f"the read-out does not fit the unified AIR contract "
                               f"(rejected, never truncated): {error}") from None
        inputs_path = directory / "inputs.bin"
        tensors, offset = [], 0
        with inputs_path.open("wb") as stream:
            for node in self.spec.io["inputs"]:
                array = arrays[node["name"]]
                data = array.tobytes(order="C")
                tensors.append({"name": node["name"], "dtype": INPUT_DTYPES[node["name"]],
                                "shape": list(array.shape), "offset": offset,
                                "bytes": len(data)})
                offset += stream.write(data)
        plan = {
            "backend": "ge_session",
            "graph": {"format": "air", "path": str(self.air_path), "om_path": None},
            "io": self.spec.io,
            "inputs": {"file": str(inputs_path), "cases": [
                {"id": case_id, "samples": 1, "tensors": tensors,
                 "output_capacities": {OUTPUT_NAME: self.spec.scores_bytes}}]},
            "device": self.device, "instances": 1, "warmup": 0, "iterations": 1,
            "seed": self.seed, "options": self.options,
            "observability": {"profiling": False, "dump": False},
            "output_dir": str(directory), "mode": "infer",
        }
        (directory / "plan.json").write_text(json.dumps(plan, indent=1))
        log_path = directory / "infer-runtime.log"
        started = time.perf_counter()
        result, text = run_plan(self.runtime_bin, plan, log_path, self.timeout)
        wall = time.perf_counter() - started
        if result.get("status") != "success":
            raise ReadoutError(f"ge_runtime infer failed (status {result.get('status')!r}); "
                               f"log tail:\n{ge_readout._log_tail(text)}")
        scores = self._read_scores(result, directory / "outputs.bin", case_id)
        n_images = 0 if media is None else len(getattr(media, "images", []) or [])
        record = {"index": index, "case_id": case_id,
                  "seq_len": int(arrays["actual_seq_lengths"][0]),
                  "physical_width": int(arrays["input_ids"].shape[1]),
                  "min_width": self.min_width,
                  "images": n_images,
                  "image_tokens": int((arrays["image_feature_index"] >= 0).sum()),
                  "wall_seconds": round(wall, 3),
                  "timings_ms": parse_timings(text), "log": str(log_path),
                  "inputs": str(inputs_path), "outputs": str(directory / "outputs.bin"),
                  "plan": str(directory / "plan.json")}
        self.records.append(record)
        return scores

    def _read_scores(self, result, outputs_path, case_id):
        rows = (result.get("outputs") or {}).get(case_id)
        if not rows or len(rows) != 1 or rows[0].get("instance") != 0:
            found = len(rows) if rows else 0
            raise ReadoutError(f"ge_runtime returned {found} output rows for {case_id}; "
                               f"expected exactly one")
        tensor = next((t for t in rows[0].get("tensors", [])
                       if t.get("name") == OUTPUT_NAME), None)
        if tensor is None:
            raise ReadoutError(f"the captured outputs carry no '{OUTPUT_NAME}' tensor")
        if tensor["dtype"] != "float32":
            raise ReadoutError(f"scores dtype {tensor['dtype']} is not float32")
        shape = tuple(tensor["shape"])
        if shape != self.spec.output_shape:
            raise ReadoutError(f"scores shape {shape} does not match the unified read-out "
                               f"contract {self.spec.output_shape}")
        count = math.prod(shape)
        if tensor["bytes"] != count * 4:
            raise ReadoutError(f"scores bytes {tensor['bytes']} do not match shape "
                               f"{shape} x 4")
        outputs_path = Path(outputs_path)
        if not outputs_path.is_file():
            raise ReadoutError(f"outputs.bin missing after a successful run: {outputs_path}")
        array = np.fromfile(outputs_path, dtype=np.float32, count=count,
                            offset=tensor["offset"])
        if array.size != count:
            raise ReadoutError(f"outputs.bin holds {array.size} of {count} score values")
        values = array.reshape(shape)
        if not np.isfinite(values).all():
            raise ReadoutError(f"scores contain non-finite values: {values.tolist()}")
        return values[0].tolist()


class UnifiedReadoutBackend(Backend):
    """The official Backend whose read-outs come from the unified AIR.

    ``reads_images`` is set: ``readout(prompt, letter_ids, media)`` receives the
    official media object of the request (startlux_decide.images.Media -- real
    PIL images, sizes, token counts) and feeds it to the official preprocessing
    directly.  The 26 score columns are the folder's letter_token_ids in order;
    the requested letters are selected by column and returned as RAW logits
    (temperatures/softmax stay in the official pipeline).  Instant models never
    generate.  Request-level violations raise the official BackendRequestError;
    engine failures propagate as ReadoutError.
    """

    name = "unified-ge-readout"
    reads_images = True

    def __init__(self, session, tokenizer, letters26, context_length=None):
        self.session = session
        self.tokenizer = tokenizer
        vocab = session.spec.vocab_size
        self.letters26 = ge_readout._require_token_ids(letters26, "letters26", vocab)
        if len(self.letters26) != LETTER_COLUMNS or len(set(self.letters26)) != LETTER_COLUMNS:
            raise ReadoutError(f"letters26 must be {LETTER_COLUMNS} unique token ids, got "
                               f"{self.letters26}")
        if session.spec.outputs[OUTPUT_NAME]["shape"][-1] != LETTER_COLUMNS:
            raise ReadoutError("the graph does not expose 26 score columns; letter "
                               "mapping is impossible")
        self.context_length = context_length
        self.column_of = {token: column for column, token in enumerate(self.letters26)}
        self.records: list[dict] = []

    @property
    def spec(self):
        return self.session.spec

    def active_letters(self, letter_ids):
        return [chr(ord("A") + self.column_of[t]) for t in letter_ids]

    async def start(self):
        return None

    async def close(self):
        return None

    async def readout(self, prompt, letter_ids, media=None):
        if not isinstance(prompt, str) or not prompt:
            raise BackendRequestError("the read-out prompt must be a non-empty string")
        asked = ge_readout._require_token_ids(letter_ids, "letter_ids", self.spec.vocab_size)
        if not asked:
            raise BackendRequestError("the read-out asked for no letters")
        unknown = [t for t in asked if t not in self.column_of]
        if unknown:
            raise BackendRequestError(f"the read-out asked for letter token ids {unknown} "
                                      f"which are not among the folder's 26 letter ids "
                                      f"{self.letters26}")
        if len(set(asked)) != len(asked):
            raise BackendRequestError(f"the read-out asked for duplicated letter ids {asked}")
        if media is not None and not hasattr(media, "images"):
            raise BackendRequestError("media must be the official load_media result of the "
                                      "request (or None), not "
                                      f"{type(media).__name__}")
        started = time.perf_counter()
        scores26 = self.session.read_scores26(prompt, media)
        record = {
            "prompt": prompt, "seq_len": self.session.records[-1]["seq_len"],
            "images": self.session.records[-1]["images"],
            "letters_asked": len(asked), "letter_ids": list(asked),
            "active_letters": self.active_letters(asked),
            "scores26": scores26,
            "selected_logits": [scores26[self.column_of[t]] for t in asked],
            "readout_seconds": round(time.perf_counter() - started, 3),
            "session_record": dict(self.session.records[-1]),
        }
        self.records.append(record)
        return list(record["selected_logits"])

    async def generate(self, prompt, max_tokens, sampling, stop, seed):
        raise NotImplementedError("Instant models answer in one read-out and never "
                                  "generate")

    def describe(self):
        return {"name": self.name, "reads_images": True,
                "letters": len(self.letters26),
                "graph": str(self.session.air_path),
                "cold_subprocess_per_readout": True}


def startlux_unified_decision(model_dir, session, tokenizer=None, *, none_check=True,
                              allow_none=False):
    """decision.py's StartluxDecision wired to a UnifiedReadoutBackend.

    Letter ids and temperatures come from the model folder's decision_config
    (read by the official load_model_folder) and the vocab bound from its
    config.json; none are hard-coded.  Returns (StartluxDecision, backend); the
    backend is exposed so callers can harvest its records as evidence.  The
    caller loads no model weights: the tokenizer and the official processor
    settings are the only folder reads.
    """
    bridge = _load_module("startlux_decision_for_unified_readout", "decider/decision.py")
    folder = bridge.load_model_folder(str(model_dir))
    if not folder.vision:
        raise ReadoutError(f"{folder.path}: the unified read-out needs the vision folder "
                           f"(folder kind is {folder.kind!r})")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    vocab = source_vocab_size(model_dir)
    if session.spec.vocab_size is None:
        session.spec.vocab_size = vocab
    elif session.spec.vocab_size != vocab:
        raise ReadoutError(f"vocab_size disagrees: session spec has "
                           f"{session.spec.vocab_size}, the model folder config says {vocab}")
    backend = UnifiedReadoutBackend(session, tokenizer, folder.letter_token_ids)
    decision = bridge.StartluxDecision(folder, tokenizer, backend=backend,
                                       none_check=none_check, allow_none=allow_none)
    return decision, backend
