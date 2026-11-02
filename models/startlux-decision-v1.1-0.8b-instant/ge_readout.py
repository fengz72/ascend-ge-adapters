"""Real GESession read-out callback for StartLux-Decision v1.1 0.8B Instant (P4).

This module wires the official business pipeline (decision.py's StartluxDecision on
the vendored startlux_decide Decider) to a real GE Session execution of the exported
StartLux text AIR.  It is the runtime-only read-out seam the business validation drives:

* ``read_air_io`` reads the IOSpec embedded in the AIR through the canonical
  ``runtime/build/ge_runtime`` binary (plan mode ``read_io``); the AIR is the only
  source of the graph contract.
* ``GraphSpec.from_io`` serves two geometries.  The FIXED B1/S graph -- batch 1,
  one compiled sequence width (S=256 for
  work/startlux-validation/native-full/model-work/build/model.air), scores [1, 26]
  float32 with exactly 26 columns (the callback maps the folder's 26 letter token
  ids onto them) -- is the geometry the real NPU business validation ran on.  The
  DYNAMIC contract of work/startlux-cannbot/p4-io-contract.json (input_ids and
  position_ids [-1, -1], actual_seq_lengths [-1], scores [-1, 26]) is accepted as
  well, B1 only: one read-out's width is the true prompt length with no padding,
  the caller must pass the explicit ``max_seq_len`` bound (the io itself declares
  none and none is inferred), and the expected output is the actual [1, 26] with a
  104-byte capacity.  The dynamic path is implemented and CPU behavior-tested but
  NOT validated on NPU: no complete dynamic AIR model evidence exists yet, so
  nothing dynamic is claimed passed.  Batch read-outs (B>1) stay another framework
  adapter's job.  A prompt longer than the bound (the compiled width, or the
  caller's max_seq_len on the dynamic contract) is REJECTED, never truncated.
* ``GEReadoutSession.read_scores26`` runs ONE ``ge_runtime`` subprocess per read-out
  (plan mode ``infer``, backend ge_session, graph_run_mode=0, must_keep_origin_dtype):
  the graph is cold-compiled and reloaded on every call and the measured compile/load
  costs are recorded per read-out instead of being hidden behind a resident service.
  Stdout/stderr go to a regular log file (never an inherited pipe, which a CANN
  compiler worker can hold open) and every call has a timeout.  Each session writes
  into its own fresh ``session-NNNN`` directory under ``run_dir`` so re-runs never
  overwrite earlier evidence.
* ``GEReadoutCallback`` is the read-out seam decision.py's ReadoutCallbackBackend calls:
  ``__call__(prompt, letter_ids)`` tokenizes the FULL prompt exactly like the official
  transformers backend (``tokenizer.encode(prompt, add_special_tokens=False)``; no cache,
  no shared prefix -- every call reads the whole prompt), runs the AIR, selects the
  columns of the requested letter ids from the 26-column scores and returns those RAW
  logits; the temperature/softmax formulas stay in the official pipeline.

Input values are never silently coerced: token ids and letter ids must be true ints
in the source config's vocab range (bool/float/str are rejected), so an arbitrary
caller cannot smuggle changed values past the interface.

Torch scope: this module itself never imports torch.  The CALLING process normally
loads the official tokenizer with ``transformers.AutoTokenizer.from_pretrained(path,
local_files_only=True)`` exactly like the official backend; that CPU torch import is
not weight loading and creates no NPU context -- all NPU work happens inside the
ge_runtime subprocess.  Run everything from the repository root: plan paths are
resolved from the current working directory.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from core.backends.runtime import execute as _execute  # noqa: E402
from core.contracts import validate_io  # noqa: E402

RUNTIME = Path("runtime/build/ge_runtime")
AIR_DEFAULT = Path("work/startlux-validation/native-full/model-work/build/model.air")
IO_CONTRACT = "work/startlux-cannbot/p4-io-contract.json"
DYNAMIC_READY_GATE = "work/startlux-cannbot/p4-dynamic-ready.json"
DYNAMIC_PENDING_NOTE = (f"the dynamic contract ({IO_CONTRACT}) has no complete AIR model evidence "
                        f"yet; dynamic spec support is implemented but not validated on NPU until "
                        f"{DYNAMIC_READY_GATE} gates the real run")
INPUT_NAMES = ("input_ids", "position_ids", "actual_seq_lengths")
INPUT_DTYPES = {"input_ids": "int64", "position_ids": "int64", "actual_seq_lengths": "int32"}
OUTPUT_NAME = "scores"
LETTER_COLUMNS = 26
PAD_ID = 0
DEFAULT_OPTIONS = {"graph_run_mode": 0, "precision_mode": "must_keep_origin_dtype"}


class ReadoutError(RuntimeError):
    """A read-out could not be served; the message states exactly why."""


def _log_tail(text: str, limit: int = 4000) -> str:
    return text[-limit:] if len(text) > limit else text


def _require_true_int(value, what: str) -> int:
    if type(value) is not int:
        raise ReadoutError(f"{what} must be a true int; got {type(value).__name__} {value!r}")
    return value


def _require_token_ids(values, what: str, vocab_size: int | None) -> list:
    ids = [_require_true_int(t, f"{what}[{i}]") for i, t in enumerate(values)]
    if vocab_size is not None:
        out_of_range = [t for t in ids if not 0 <= t < vocab_size]
        if out_of_range:
            raise ReadoutError(f"{what} fall outside the source config vocab range "
                               f"[0, {vocab_size}): {out_of_range[:8]}")
    return ids


def source_vocab_size(model_dir) -> int:
    """The embedding vocab bound from the model folder's config.json (text_config.vocab_size)."""
    config = json.loads((Path(model_dir) / "config.json").read_text())
    vocab = config.get("vocab_size") or (config.get("text_config") or {}).get("vocab_size")
    if type(vocab) is not int or vocab < 1:
        raise ReadoutError(f"{model_dir}/config.json has no usable vocab_size")
    return vocab


def run_plan(runtime_bin, plan: dict, log_path, timeout: float = 600.0):
    """Execute one ge_runtime plan and return (result, log_text).

    Stdout+stderr of the runtime go to a regular file at ``log_path``; the call is
    bounded by ``timeout`` seconds; exactly one ``GE_RESULT`` line must come back.
    Every deviation raises ReadoutError with the log tail.  The subprocess protocol
    itself (own process group, timeout/abort kill, GE_RESULT parsing) is the shared
    core implementation -- core.backends.runtime.execute.
    """
    result, text, _wall = _execute(runtime_bin, plan, log_path=log_path,
                                    timeout=timeout, error=ReadoutError)
    return result, text


def read_air_io(runtime_bin, air_path, log_path, timeout: float = 600.0) -> dict:
    """The IOSpec embedded in the AIR, read through the runtime binary (read_io)."""
    if not Path(air_path).is_file():
        raise ReadoutError(f"AIR missing: {air_path}; run the export stage first")
    plan = {"mode": "read_io", "graph": {"format": "air", "path": str(air_path), "om_path": None}}
    result, _ = run_plan(runtime_bin, plan, log_path, timeout)
    if "io" not in result:
        raise ReadoutError(f"read_io returned no io: {json.dumps(result)[:500]}")
    return result["io"]


class GraphSpec:
    """The graph geometry inferred from the AIR's embedded IOSpec.

    Two geometries: the FIXED B1/S graph (the only NPU-validated one -- the real
    business validation ran on it) and the DYNAMIC contract of
    work/startlux-cannbot/p4-io-contract.json, B1 served here, whose AIR is still
    pending (see DYNAMIC_PENDING_NOTE: implemented, CPU behavior-tested, not
    claimed passed on NPU).
    """

    DYNAMIC_SHAPES = {"input_ids": [-1, -1], "position_ids": [-1, -1],
                      "actual_seq_lengths": [-1], OUTPUT_NAME: [-1, LETTER_COLUMNS]}
    DYNAMIC_DIMS = {"input_ids": [0, 1], "position_ids": [0, 1],
                    "actual_seq_lengths": [0], OUTPUT_NAME: [0]}

    def __init__(self, io: dict, batch: int, width: int | None, score_columns: int,
                 vocab_size: int | None = None, *, dynamic: bool = False,
                 max_seq_len: int | None = None):
        self.io = io
        self.batch, self.width, self.score_columns = batch, width, score_columns
        self.vocab_size = vocab_size
        self.dynamic = bool(dynamic)
        self.max_seq_len = max_seq_len if self.dynamic else None
        self.inputs = {node["name"]: node for node in io["inputs"]}
        self.outputs = {node["name"]: node for node in io["outputs"]}
        # the output shape every single read-out must return: the fixed graph's
        # compiled [1, 26], or the actual [1, 26] of one B1 dynamic read-out
        if self.dynamic:
            self.output_shape = (batch, score_columns)
        else:
            self.output_shape = tuple(self.outputs[OUTPUT_NAME]["shape"])

    @classmethod
    def from_io(cls, io: dict, vocab_size: int | None = None,
                max_seq_len: int | None = None) -> "GraphSpec":
        try:
            validate_io(io)
        except ValueError as error:
            raise ReadoutError(f"the AIR's IOSpec is not valid: {error}") from error
        inputs = {node["name"]: node for node in io["inputs"]}
        outputs = {node["name"]: node for node in io["outputs"]}
        if sorted(inputs) != sorted(INPUT_NAMES):
            raise ReadoutError(f"the AIR inputs {sorted(inputs)} are not the StartLux text "
                               f"contract {sorted(INPUT_NAMES)}")
        if sorted(outputs) != [OUTPUT_NAME]:
            raise ReadoutError(f"the AIR outputs {sorted(outputs)} are not the "
                               f"['{OUTPUT_NAME}'] contract")
        for name, dtype in INPUT_DTYPES.items():
            if inputs[name]["dtype"] != dtype:
                raise ReadoutError(f"input {name} dtype {inputs[name]['dtype']} is not {dtype}")
        out = outputs[OUTPUT_NAME]
        if out["dtype"] != "float32":
            raise ReadoutError(f"scores dtype {out['dtype']} is not float32")
        if vocab_size is not None and (type(vocab_size) is not int or vocab_size < 1):
            raise ReadoutError(f"vocab_size must be a positive true int, got {vocab_size!r}")
        dynamic = [node["name"] for node in list(io["inputs"]) + list(io["outputs"])
                   if -1 in node["shape"] or node.get("dynamic_dims")]
        if dynamic:
            return cls._from_dynamic_io(io, inputs, outputs, dynamic, vocab_size, max_seq_len)
        if max_seq_len is not None:
            raise ReadoutError("max_seq_len applies to the dynamic contract only; a fixed "
                               "AIR's length bound is its compiled width")
        ids_shape = tuple(inputs["input_ids"]["shape"])
        pos_shape = tuple(inputs["position_ids"]["shape"])
        len_shape = tuple(inputs["actual_seq_lengths"]["shape"])
        if len(ids_shape) != 2 or len(pos_shape) != 2 or len(len_shape) != 1:
            raise ReadoutError(f"input ranks are not [2, 2, 1]: input_ids {ids_shape}, "
                               f"position_ids {pos_shape}, actual_seq_lengths {len_shape}")
        if ids_shape != pos_shape:
            raise ReadoutError(f"input_ids {ids_shape} and position_ids {pos_shape} differ")
        batch, width = ids_shape
        if batch != 1:
            raise ReadoutError(f"this read-out serves the fixed B1 AIR; the graph batch is "
                               f"{batch} (batch read-outs are another framework adapter's job)")
        if width < 1:
            raise ReadoutError(f"compiled width {width} is not positive")
        if len_shape != (batch,):
            raise ReadoutError(f"actual_seq_lengths shape {len_shape} is not ({batch},)")
        out_shape = tuple(out["shape"])
        if len(out_shape) != 2 or out_shape[0] != batch:
            raise ReadoutError(f"scores shape {out_shape} is not [batch, columns]")
        if out_shape[1] != LETTER_COLUMNS:
            raise ReadoutError(f"scores carry {out_shape[1]} columns, not the {LETTER_COLUMNS} "
                               f"letter columns the callback maps the folder's letter_token_ids onto")
        return cls(io, batch, width, out_shape[1], vocab_size)

    @classmethod
    def _from_dynamic_io(cls, io, inputs, outputs, dynamic, vocab_size, max_seq_len):
        """The dynamic contract geometry, B1 read-outs only.

        The io itself declares no length bound, so the caller's explicit max_seq_len
        is REQUIRED (never inferred); batch read-outs (B>1) are another framework
        adapter's job.  The dynamic path is NOT validated on NPU yet
        (DYNAMIC_PENDING_NOTE).
        """
        for name, shape in cls.DYNAMIC_SHAPES.items():
            node = inputs.get(name) or outputs[name]
            dims = list(node.get("dynamic_dims") or [])
            if list(node["shape"]) != shape or dims != cls.DYNAMIC_DIMS[name]:
                raise ReadoutError(f"the dynamic IOSpec of {name} (shape {node['shape']}, "
                                   f"dynamic_dims {node.get('dynamic_dims')}) does not match the "
                                   f"contract shape {shape} / dynamic_dims "
                                   f"{cls.DYNAMIC_DIMS[name]} of {IO_CONTRACT}")
        if type(max_seq_len) is not int or max_seq_len < 1:
            raise ReadoutError(f"a dynamic AIR needs the caller's explicit max_seq_len bound "
                               f"(a positive true int; the io declares none and none is "
                               f"inferred) -- got {max_seq_len!r}; {DYNAMIC_PENDING_NOTE}")
        return cls(io, 1, None, LETTER_COLUMNS, vocab_size, dynamic=True,
                   max_seq_len=max_seq_len)

    @property
    def length_bound(self) -> int:
        """The prompt-length bound: the compiled width (fixed) or the caller-declared
        max_seq_len (dynamic)."""
        return self.max_seq_len if self.dynamic else self.width

    @property
    def scores_bytes(self) -> int:
        return math.prod(self.output_shape) * 4


def encode_padded(input_ids, spec: GraphSpec) -> dict:
    """Right-pad one read-out to the compiled width (the validated B1/S256 encoding).

    Beyond the true length L the graph consumes pad id 0 at position 0 while
    actual_seq_lengths gates every layer at the true L; the native chunk route was
    validated with exactly this encoding (the former
    startlux_npu_adapter._encode_padded, removed with the standard route
    2026-11-02). Token ids are validated as true ints in the source vocab
    range -- never coerced.
    """
    if spec.dynamic:
        raise ReadoutError("encode_padded serves a fixed GraphSpec only; a dynamic spec uses "
                           "encode_b1_dynamic")
    ids = _require_token_ids(input_ids, "input_ids", spec.vocab_size)
    length = len(ids)
    if length < 1:
        raise ReadoutError("the read-out tokenized to 0 tokens; refusing an empty input")
    if length > spec.width:
        raise ReadoutError(f"the read-out needs {length} tokens > compiled width {spec.width}; "
                           f"the AIR is fixed at B{spec.batch}/S{spec.width} and overlong prompts "
                           f"are rejected, never truncated")
    pad = spec.width - length
    return {
        "input_ids": np.asarray([ids + [PAD_ID] * pad], dtype=np.int64),
        "position_ids": np.asarray([list(range(length)) + [PAD_ID] * pad], dtype=np.int64),
        "actual_seq_lengths": np.asarray([length], dtype=np.int32),
    }


def encode_b1_dynamic(input_ids, spec: GraphSpec) -> dict:
    """One B1 read-out on the dynamic contract: width = the TRUE prompt length.

    No padding: input_ids/position_ids are [1, L] with arange(L) positions and
    actual_seq_lengths carries the real length [L].  The length bound is the
    caller-declared max_seq_len (the io declares none); overlong prompts are
    rejected, never truncated.  Token ids are validated as true ints in the source
    vocab range -- never coerced.  NOT validated on NPU yet (DYNAMIC_PENDING_NOTE).
    """
    if not spec.dynamic:
        raise ReadoutError("encode_b1_dynamic serves a dynamic GraphSpec only")
    ids = _require_token_ids(input_ids, "input_ids", spec.vocab_size)
    length = len(ids)
    if length < 1:
        raise ReadoutError("the read-out tokenized to 0 tokens; refusing an empty input")
    if length > spec.max_seq_len:
        raise ReadoutError(f"the read-out needs {length} tokens > the caller-declared "
                           f"max_seq_len {spec.max_seq_len}; overlong prompts are rejected, "
                           f"never truncated ({DYNAMIC_PENDING_NOTE})")
    return {
        "input_ids": np.asarray([ids], dtype=np.int64),
        "position_ids": np.asarray([list(range(length))], dtype=np.int64),
        "actual_seq_lengths": np.asarray([length], dtype=np.int32),
    }


_COMPILE_LINE = re.compile(r"CompileGraph ok .* cost ([0-9.]+) ms")
_LOAD_LINE = re.compile(r"LoadGraph ok .* cost ([0-9.]+) ms")


def parse_timings(log_text: str) -> dict:
    """The cold-start costs the runtime prints per invocation (ms strings -> floats)."""
    timings = {}
    for key, pattern in (("compile_ms", _COMPILE_LINE), ("load_ms", _LOAD_LINE)):
        found = pattern.findall(log_text)
        if found:
            timings[key] = float(found[-1])
    return timings


def _read_scores(result: dict, outputs_path, spec: GraphSpec, case_id: str) -> list:
    rows = (result.get("outputs") or {}).get(case_id)
    if not rows or len(rows) != 1 or rows[0].get("instance") != 0:
        found = len(rows) if rows else 0
        raise ReadoutError(f"ge_runtime returned {found} output rows for {case_id}; "
                           f"expected exactly one")
    tensor = next((t for t in rows[0].get("tensors", []) if t.get("name") == OUTPUT_NAME), None)
    if tensor is None:
        raise ReadoutError(f"the captured outputs carry no '{OUTPUT_NAME}' tensor")
    if tensor["dtype"] != "float32":
        raise ReadoutError(f"scores dtype {tensor['dtype']} is not float32")
    shape = tuple(tensor["shape"])
    if shape != spec.output_shape:
        raise ReadoutError(f"scores shape {shape} does not match the AIR contract "
                           f"{spec.output_shape}")
    count = math.prod(shape)
    if tensor["bytes"] != count * 4:
        raise ReadoutError(f"scores bytes {tensor['bytes']} do not match shape {shape} x 4")
    outputs_path = Path(outputs_path)
    if not outputs_path.is_file():
        raise ReadoutError(f"outputs.bin missing after a successful run: {outputs_path}")
    array = np.fromfile(outputs_path, dtype=np.float32, count=count, offset=tensor["offset"])
    if array.size != count:
        raise ReadoutError(f"outputs.bin holds {array.size} of {count} score values")
    values = array.reshape(shape)
    if not np.isfinite(values).all():
        raise ReadoutError(f"scores contain non-finite values: {values.tolist()}")
    return values[0].tolist()


class GEReadoutSession:
    """One read-out = one ge_runtime subprocess: cold CompileGraph + LoadGraph per call.

    Every session creates its own fresh ``session-NNNN`` directory under ``run_dir``
    (never reusing a non-empty one), so re-running validation cannot overwrite earlier
    evidence.  The session directory collects read-io.log and one subdirectory per
    read-out (plan.json, inputs.bin, outputs.bin, infer-runtime.log); ``records`` keeps
    the measured wall/compile/load costs so the cold-start overhead of this
    correctness-first flow is reported honestly.
    """

    def __init__(self, device, run_dir, *, runtime_bin=RUNTIME, air_path=AIR_DEFAULT,
                 spec=None, timeout=600.0, seed=42, options=None, vocab_size=None,
                 max_seq_len=None):
        self.device = int(device)
        if self.device < 0:
            raise ReadoutError(f"device index must be >= 0, got {device}")
        self.runtime_bin = Path(runtime_bin)
        self.air_path = Path(air_path)
        self.timeout = float(timeout)
        self.seed = int(seed)
        self.options = dict(options) if options is not None else dict(DEFAULT_OPTIONS)
        if self.options.get("graph_run_mode", 0) != 0:
            raise ReadoutError("this validation is graph_run_mode=0 (inference mode) only")
        if "precision_mode" not in self.options:
            self.options["precision_mode"] = DEFAULT_OPTIONS["precision_mode"]
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
            spec = GraphSpec.from_io(io, vocab_size, max_seq_len=max_seq_len)
        self.spec = spec
        if max_seq_len is not None:
            if not self.spec.dynamic:
                raise ReadoutError("max_seq_len applies to the dynamic contract only; the "
                                   f"fixed AIR's length bound is its compiled width "
                                   f"{self.spec.width}")
            if self.spec.max_seq_len != max_seq_len:
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

    def read_scores26(self, input_ids) -> list:
        """All 26 letter scores of one full-prompt read-out (finite float list)."""
        index, directory = self._next_dir()
        case_id = f"readout-{index:04d}"
        values = encode_b1_dynamic(input_ids, self.spec) if self.spec.dynamic \
            else encode_padded(input_ids, self.spec)
        inputs_path = directory / "inputs.bin"
        tensors, offset = [], 0
        with inputs_path.open("wb") as stream:
            for name in INPUT_NAMES:
                array = values[name]
                data = array.tobytes(order="C")
                tensors.append({"name": name, "dtype": INPUT_DTYPES[name],
                                "shape": list(array.shape), "offset": offset, "bytes": len(data)})
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
                               f"log tail:\n{_log_tail(text)}")
        scores = _read_scores(result, directory / "outputs.bin", self.spec, case_id)
        record = {"index": index, "case_id": case_id, "seq_len": len(input_ids),
                  "dynamic": self.spec.dynamic,
                  "width": len(input_ids) if self.spec.dynamic else self.spec.width,
                  "max_seq_len": self.spec.max_seq_len,
                  "wall_seconds": round(wall, 3),
                  "timings_ms": parse_timings(text), "log": str(log_path),
                  "inputs": str(inputs_path), "outputs": str(directory / "outputs.bin"),
                  "plan": str(directory / "plan.json")}
        self.records.append(record)
        return scores


class GEReadoutCallback:
    """readout(prompt, letter_ids) -> the RAW logits of the requested letters.

    The seam decision.py's ReadoutCallbackBackend calls: tokenize the full prompt
    (official ``encode(prompt, add_special_tokens=False)`` -- the caller supplies the
    same official AutoTokenizer the reference backends use; no cache/prefix reuse),
    run the real GE graph, select the requested letter columns from the 26-column
    scores and return them untouched -- the temperature and softmax formulas are the
    official pipeline's business.  ``records`` keeps every prompt, ids, all 26
    scores, the asked letter ids and their letters for evidence.
    """

    def __init__(self, session, tokenizer, letters26):
        self.session = session
        self.tokenizer = tokenizer
        vocab = session.spec.vocab_size
        self.letters26 = _require_token_ids(letters26, "letters26", vocab)
        if len(self.letters26) != LETTER_COLUMNS or len(set(self.letters26)) != LETTER_COLUMNS:
            raise ReadoutError(f"letters26 must be {LETTER_COLUMNS} unique token ids, got "
                               f"{self.letters26}")
        if session.spec.score_columns != LETTER_COLUMNS:
            raise ReadoutError(f"the graph exposes {session.spec.score_columns} score columns, "
                               f"not {LETTER_COLUMNS}; letter mapping is impossible")
        self.column_of = {token: column for column, token in enumerate(self.letters26)}
        self.records: list[dict] = []

    @property
    def spec(self) -> GraphSpec:
        return self.session.spec

    def active_letters(self, letter_ids) -> list:
        return [chr(ord("A") + self.column_of[t]) for t in letter_ids]

    def __call__(self, prompt: str, letter_ids) -> list:
        if not isinstance(prompt, str) or not prompt:
            raise ReadoutError("the read-out prompt must be a non-empty string")
        asked = _require_token_ids(letter_ids, "letter_ids", self.spec.vocab_size)
        if not asked:
            raise ReadoutError("the read-out asked for no letters")
        unknown = [t for t in asked if t not in self.column_of]
        if unknown:
            raise ReadoutError(f"the read-out asked for letter token ids {unknown} which are "
                               f"not among the folder's 26 letter ids {self.letters26}")
        if len(set(asked)) != len(asked):
            raise ReadoutError(f"the read-out asked for duplicated letter ids {asked}")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not isinstance(ids, list):
            raise ReadoutError("the tokenizer did not return a list of token ids (expected "
                               "tokenizer.encode(prompt, add_special_tokens=False))")
        ids = _require_token_ids(ids, "prompt token ids", self.spec.vocab_size)
        if not ids:
            raise ReadoutError("the prompt tokenized to 0 tokens")
        length = len(ids)
        bound = self.spec.length_bound
        if length > bound:
            raise ReadoutError(f"the prompt tokenizes to {length} tokens > the length bound "
                               f"{bound}; overlong prompts are rejected, never truncated "
                               + (f"({DYNAMIC_PENDING_NOTE})" if self.spec.dynamic else
                                  f"(the AIR is fixed at B{self.spec.batch}/S{self.spec.width})"))
        started = time.perf_counter()
        scores26 = self.session.read_scores26(ids)
        record = {
            "prompt": prompt, "input_ids": ids, "seq_len": length,
            "letters_asked": len(asked), "letter_ids": asked,
            "active_letters": self.active_letters(asked),
            "scores26": scores26,
            "selected_logits": [scores26[self.column_of[t]] for t in asked],
            "readout_seconds": round(time.perf_counter() - started, 3),
            "session_record": dict(self.session.records[-1]) if self.session.records else None,
        }
        self.records.append(record)
        return list(record["selected_logits"])


def startlux_decision(model_dir, tokenizer, session, *, none_check=True, allow_none=False):
    """decision.py's StartluxDecision wired to a GEReadoutCallback over ``session``.

    Letter ids and temperatures come from the model folder's decision_config.json
    (read by the official load_model_folder) and the vocab bound from its config.json
    (text_config.vocab_size); none are hard-coded.  Returns (StartluxDecision,
    callback); the callback is exposed so callers can harvest its records as evidence.
    """
    import importlib.util

    module_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("startlux_ge_readout_bridge",
                                                  module_dir / "decision.py")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    folder = bridge.load_model_folder(str(model_dir))
    vocab = source_vocab_size(model_dir)
    if session.spec.vocab_size is None:
        session.spec.vocab_size = vocab
    elif session.spec.vocab_size != vocab:
        raise ReadoutError(f"vocab_size disagrees: session spec has {session.spec.vocab_size}, "
                           f"the model folder config says {vocab}")
    callback = GEReadoutCallback(session, tokenizer, folder.letter_token_ids)
    decision = bridge.StartluxDecision(folder, tokenizer, readout=callback,
                                       none_check=none_check, allow_none=allow_none)
    return decision, callback
