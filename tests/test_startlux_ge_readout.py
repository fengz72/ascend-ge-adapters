"""Behavior tests for the P4 real-GE read-out seam (models/.../ge_readout.py).

The ge_runtime subprocess is replaced by a fake runtime script implementing the same
JSON plan protocol (stdin plan, one GE_RESULT line, outputs.bin with offsets), so the
REAL code paths run: run_plan's log/timeout/result parsing, the session's per-readout
plan building and inputs.bin/outputs.bin handling, the record layout.  The business
wiring tests drive the official Decider (decision.py on the vendored package) with the
FakeTokenizer and marker-driven scores, exactly like tests/test_startlux_decision.py,
proving the callback plugs into the official bridge: letter selection, wide knock-out
over sequential rounds and re-entry of the same callback instance.  No NPU, no weights.
"""
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODEL_DIR = Path("models/startlux-decision-v1.1-0.8b-instant").resolve()
_spec = importlib.util.spec_from_file_location("ge_readout", MODEL_DIR / "decider" / "ge_readout.py")
ge_readout = importlib.util.module_from_spec(_spec)
sys.modules["ge_readout"] = ge_readout
_spec.loader.exec_module(ge_readout)
from ge_readout import (GraphSpec, ReadoutError, encode_b1_dynamic,           # noqa: E402
                        encode_padded, parse_timings, run_plan,
                        source_vocab_size, startlux_decision)

VENDOR = MODEL_DIR / "decider" / "vendor"
if str(VENDOR) not in sys.path:               # the pinned official package (decision.py's path)
    sys.path.insert(0, str(VENDOR))
from startlux_decide.backends.fake import FakeTokenizer    # noqa: E402

TEMPS = {"choice": 0.7, "noul": 1.3, "score": 0.9}
WIDE = {"group": 25, "keep": 3, "residual": 0.001}
LETTERS = list(range(65, 91))                            # FakeTokenizer: ord('A')..ord('Z')
HIGH, LOW, MARK = 2.0, 0.0, "GOLDMARK"
PICK = {"instructions": "Pick one."}
VOCAB = 3_000_000          # above the FakeTokenizer special-token base (~2e6)

FAKE_RUNTIME = r'''#!/usr/bin/env python3
import json, os, struct, sys, time

mode = os.environ.get("FAKE_RUNTIME_MODE", "ok")
io_kind = os.environ.get("FAKE_RUNTIME_IO", "fixed")
plan = json.load(sys.stdin)
if plan.get("mode") == "read_io":
    if io_kind == "dynamic":
        io = {"schema": "ge-io/1",
              "inputs": [
                  {"name": "input_ids", "dtype": "int64", "shape": [-1, -1],
                   "dynamic_dims": [0, 1], "node": "arg0_1"},
                  {"name": "position_ids", "dtype": "int64", "shape": [-1, -1],
                   "dynamic_dims": [0, 1], "node": "arg1_1"},
                  {"name": "actual_seq_lengths", "dtype": "int32", "shape": [-1],
                   "dynamic_dims": [0], "node": "arg2_1"}],
              "outputs": [{"name": "scores", "dtype": "float32", "shape": [-1, 26],
                           "dynamic_dims": [0]}]}
    else:
        io = {"schema": "ge-io/1",
              "inputs": [
                  {"name": "input_ids", "dtype": "int64", "shape": [1, 64], "node": "arg0_1"},
                  {"name": "position_ids", "dtype": "int64", "shape": [1, 64], "node": "arg1_1"},
                  {"name": "actual_seq_lengths", "dtype": "int32", "shape": [1], "node": "arg2_1"}],
              "outputs": [{"name": "scores", "dtype": "float32", "shape": [1, 26]}]}
    print("GE_RESULT " + json.dumps({"io": io}))
    sys.exit(0)
if mode == "hang":
    time.sleep(60)
out_dir = plan["output_dir"]
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "seen-plan.json"), "w") as fh:
    json.dump(plan, fh)
outputs, offset = {}, 0
with open(os.path.join(out_dir, "outputs.bin"), "wb") as fh:
    for node in plan["io"]["outputs"]:
        shape, count = list(node["shape"]), 1
        if any(d < 0 for d in shape):
            # a dynamic output reports the ACTUAL shape: this case's row count
            case = plan["inputs"]["cases"][0]
            ids = next(t for t in case["tensors"] if t["name"] == "input_ids")
            batch = 2 if mode == "dynamic_batch2" else ids["shape"][0]
            shape = [batch if d == -1 else d for d in shape]
        for d in shape:
            count *= d
        values = [float("nan")] + [float(i) for i in range(1, count)] if mode == "nan" \
            else [float(i) for i in range(count)]
        if mode == "short_shape":
            shape, count, values = [shape[0], shape[1] - 1], count - 1, values[:-1]
        fh.write(struct.pack("<%df" % count, *values))
        outputs[node["name"]] = {"name": node["name"], "dtype": node["dtype"], "shape": shape,
                                 "offset": offset, "bytes": count * 4}
        offset += count * 4
print("[INFO] CompileGraph ok (gid=1/1), cost 12.5 ms")
print("[INFO] LoadGraph ok (gid=1, tid=0), cost 1.5 ms")
if mode == "no_result":
    sys.stderr.write("backend exploded\n")
    sys.exit(1)
if mode == "double_result":
    print("GE_RESULT " + json.dumps({"status": "success"}))
    print("GE_RESULT " + json.dumps({"status": "success"}))
    sys.exit(0)
if mode == "failed":
    print("GE_RESULT " + json.dumps({"status": "failed", "error": "backend execution failed"}))
    sys.exit(1)
rows = {case["id"]: [{"instance": 0, "tensors": list(outputs.values())}]
        for case in plan["inputs"]["cases"]}
print("GE_RESULT " + json.dumps({"status": "success", "mode": "infer", "outputs": rows}))
'''


def build_io(width=64, columns=26, batch=1, dynamic=False):
    def shape(dims):
        table = {"b": batch, "s": width, "c": columns, "n": 1}
        return [-1 if (dynamic and d != "n") else table[d] for d in dims]
    io = {"schema": "ge-io/1",
          "inputs": [
              {"name": "input_ids", "dtype": "int64", "shape": shape(["b", "s"]), "node": "arg0_1"},
              {"name": "position_ids", "dtype": "int64", "shape": shape(["b", "s"]), "node": "arg1_1"},
              {"name": "actual_seq_lengths", "dtype": "int32", "shape": shape(["b"]), "node": "arg2_1"}],
          "outputs": [{"name": "scores", "dtype": "float32", "shape": shape(["b", "c"]) if not dynamic
                       else [-1, columns]}]}
    if dynamic:
        io["inputs"][0]["dynamic_dims"] = [0, 1]
        io["inputs"][1]["dynamic_dims"] = [0, 1]
        io["inputs"][2]["dynamic_dims"] = [0]
        io["outputs"][0]["dynamic_dims"] = [0]
    return io


def make_spec(width=64, columns=26, vocab=VOCAB, io=None, max_seq_len=None):
    return GraphSpec.from_io(io if io is not None else build_io(width, columns), vocab,
                             max_seq_len=max_seq_len)


def make_folder(root, temperatures=TEMPS, letters=LETTERS):
    d = Path(root) / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(
        {"jev_prompt_attention": "bidirectional", "max_position_embeddings": 262144,
         "text_config": {"vocab_size": VOCAB}}))
    (d / "decision_config.json").write_text(json.dumps(
        {"prompt_attention": "bidirectional", "temperature_by_type": temperatures,
         "letter_token_ids": letters, "max_options_per_pass": 26, "wide_choice": wide_cfg(),
         "format": "StartLux-Decision-v1.1", "release_date": "2026-10-30"}))
    return d


def wide_cfg():
    return WIDE


class FakeSession:
    """Session stand-in: the same spec/records surface, scores from a callable."""

    def __init__(self, scores, width=64, vocab=VOCAB, spec=None):
        self.scores = scores
        self.spec = spec if spec is not None else make_spec(width=width, vocab=vocab)
        self.records, self.calls = [], []

    def read_scores26(self, input_ids):
        self.calls.append(list(input_ids))
        self.records.append({"index": len(self.records), "wall_seconds": 0.0, "timings_ms": {}})
        return list(self.scores(list(input_ids)))


class MarkerSession(FakeSession):
    """Scores derived from the official prompt's Options block: a letter scores HIGH
    when its option body carries the marker -- the controlled-backend recipe of
    tests/test_startlux_decision.py, fed through the callback's token ids."""

    def __init__(self, tokenizer, width=4096):
        super().__init__(self._from_ids, width=width)
        self.tokenizer = tokenizer

    def _from_ids(self, input_ids):
        prompt = self.tokenizer.decode(input_ids)
        scores = [LOW] * 26
        block = prompt.rsplit("\nOptions:\n", 1)[1]
        for line in block.split("\n"):
            if re.match(r"^[A-Z]\) ", line) and MARK in line:
                scores[ord(line[0]) - 65] = HIGH
        return scores


class TestGraphSpec(unittest.TestCase):
    def test_accepts_the_fixed_b1_contract(self):
        spec = make_spec(width=256, vocab=None)
        self.assertEqual((spec.batch, spec.width, spec.score_columns), (1, 256, 26))
        self.assertEqual(spec.output_shape, (1, 26))
        self.assertEqual(spec.scores_bytes, 104)
        self.assertIsNone(spec.vocab_size)

    def test_scores_must_carry_exactly_26_columns(self):
        with self.assertRaisesRegex(ReadoutError, "25 columns"):
            make_spec(columns=25)

    def test_dynamic_dimensions_without_a_caller_bound_are_refused(self):
        with self.assertRaisesRegex(ReadoutError, "max_seq_len"):
            make_spec(io=build_io(dynamic=True))

    def test_batch_other_than_one_is_refused(self):
        io = build_io()
        io["inputs"][0]["shape"] = [2, 64]
        io["inputs"][1]["shape"] = [2, 64]
        io["inputs"][2]["shape"] = [2]
        io["outputs"][0]["shape"] = [2, 26]
        with self.assertRaisesRegex(ReadoutError, "batch is 2"):
            make_spec(io=io)

    def test_wrong_names_dtypes_and_schema_are_refused(self):
        io = build_io()
        io["inputs"][0]["name"] = "tokens"
        with self.assertRaisesRegex(ReadoutError, "not the StartLux text contract"):
            make_spec(io=io)
        io = build_io()
        io["inputs"][0]["dtype"] = "int32"
        with self.assertRaisesRegex(ReadoutError, "dtype int32 is not int64"):
            make_spec(io=io)
        io = build_io()
        io["schema"] = "other/1"
        with self.assertRaises(ReadoutError):
            make_spec(io=io)

    def test_duplicate_io_names_are_refused(self):
        io = build_io()
        io["outputs"].append(dict(io["outputs"][0]))
        with self.assertRaisesRegex(ReadoutError, "not valid"):
            make_spec(io=io)

    def test_vocab_size_is_validated_and_bounded(self):
        spec = make_spec(vocab=VOCAB)
        self.assertEqual(spec.vocab_size, VOCAB)
        with self.assertRaisesRegex(ReadoutError, "vocab_size"):
            make_spec(vocab=0)

    def test_source_vocab_size_reads_the_folder_text_config(self):
        with tempfile.TemporaryDirectory() as root:
            folder = make_folder(root)
            self.assertEqual(source_vocab_size(folder), VOCAB)


class TestDynamicGraphSpec(unittest.TestCase):
    """The p4-io-contract dynamic geometry: B1 served, explicit caller max_seq_len."""

    def test_accepts_the_dynamic_contract_with_explicit_max_seq_len(self):
        spec = make_spec(io=build_io(dynamic=True), vocab=None, max_seq_len=2048)
        self.assertTrue(spec.dynamic)
        self.assertEqual((spec.batch, spec.width, spec.score_columns), (1, None, 26))
        self.assertEqual(spec.output_shape, (1, 26))
        self.assertEqual(spec.scores_bytes, 104)
        self.assertEqual(spec.max_seq_len, 2048)
        self.assertEqual(spec.length_bound, 2048)
        self.assertIsNone(spec.vocab_size)
        spec = make_spec(io=build_io(dynamic=True), max_seq_len=2048)   # vocab propagates too
        self.assertEqual(spec.vocab_size, VOCAB)

    def test_dynamic_without_the_caller_bound_is_refused(self):
        with self.assertRaisesRegex(ReadoutError, "max_seq_len"):
            make_spec(io=build_io(dynamic=True))

    def test_bad_max_seq_len_values_are_refused(self):
        for bad in (True, 2048.0, "2048", 0, -5, 0.5):
            with self.assertRaisesRegex(ReadoutError, "max_seq_len"):
                make_spec(io=build_io(dynamic=True), max_seq_len=bad)

    def test_dynamic_geometry_deviations_are_refused(self):
        io = build_io(dynamic=True)
        io["outputs"][0]["shape"] = [-1, 25]                    # not the 26 letter columns
        with self.assertRaisesRegex(ReadoutError, "does not match the contract"):
            make_spec(io=io, max_seq_len=2048)
        io = build_io(dynamic=True)
        io["inputs"][2]["shape"] = [-1, -1]                     # lengths are rank 1
        io["inputs"][2]["dynamic_dims"] = [0, 1]
        with self.assertRaisesRegex(ReadoutError, "does not match the contract"):
            make_spec(io=io, max_seq_len=2048)
        io = build_io(dynamic=True)
        io["inputs"][0]["dynamic_dims"] = [0]                   # dims disagree with -1 axes
        with self.assertRaises(ReadoutError):
            make_spec(io=io, max_seq_len=2048)
        io = build_io(dynamic=True)
        io["inputs"][0]["dtype"] = "int32"
        with self.assertRaisesRegex(ReadoutError, "is not int64"):
            make_spec(io=io, max_seq_len=2048)

    def test_a_fixed_graph_with_max_seq_len_is_refused(self):
        with self.assertRaisesRegex(ReadoutError, "dynamic contract only"):
            make_spec(io=build_io(), max_seq_len=2048)

    def test_the_fixed_geometry_is_unchanged(self):
        spec = make_spec(width=256)
        self.assertFalse(spec.dynamic)
        self.assertIsNone(spec.max_seq_len)
        self.assertEqual(spec.length_bound, 256)


class TestTokenValidation(unittest.TestCase):
    def test_encode_padded_builds_the_validated_layout(self):
        spec = make_spec(width=8)
        values = encode_padded([11, 12, 13], spec)
        self.assertEqual(values["input_ids"].tolist(), [[11, 12, 13, 0, 0, 0, 0, 0]])
        self.assertEqual(values["position_ids"].tolist(), [[0, 1, 2, 0, 0, 0, 0, 0]])
        self.assertEqual(values["actual_seq_lengths"].tolist(), [3])
        self.assertEqual(values["actual_seq_lengths"].shape, (1,))
        self.assertEqual(values["input_ids"].dtype.name, "int64")
        self.assertEqual(values["actual_seq_lengths"].dtype.name, "int32")

    def test_overlong_inputs_are_rejected_never_truncated(self):
        spec = make_spec(width=4)
        with self.assertRaisesRegex(ReadoutError, "5 tokens > compiled width 4"):
            encode_padded([1, 2, 3, 4, 5], spec)

    def test_empty_inputs_are_refused(self):
        with self.assertRaisesRegex(ReadoutError, "0 tokens"):
            encode_padded([], make_spec())

    def test_bools_floats_and_strings_are_never_silently_coerced(self):
        spec = make_spec()
        for bad in ([True, 2, 3], [1.0, 2, 3], ["1", 2, 3], [None, 2, 3]):
            with self.assertRaisesRegex(ReadoutError, "true int"):
                encode_padded(bad, spec)

    def test_out_of_vocab_tokens_are_refused(self):
        spec = make_spec(vocab=VOCAB)
        with self.assertRaisesRegex(ReadoutError, "vocab range"):
            encode_padded([1, VOCAB + 5], spec)
        encode_padded([1, VOCAB - 1], spec)                    # boundary id is legal


class TestDynamicEncoding(unittest.TestCase):
    """encode_b1_dynamic: the true unpadded length, the caller-declared bound."""

    def test_encodes_the_true_length_without_any_padding(self):
        spec = make_spec(io=build_io(dynamic=True), max_seq_len=2048)
        values = encode_b1_dynamic([11, 12, 13], spec)
        self.assertEqual(values["input_ids"].tolist(), [[11, 12, 13]])
        self.assertEqual(values["position_ids"].tolist(), [[0, 1, 2]])
        self.assertEqual(values["actual_seq_lengths"].tolist(), [3])
        self.assertEqual(values["input_ids"].dtype.name, "int64")
        self.assertEqual(values["actual_seq_lengths"].dtype.name, "int32")

    def test_overlong_inputs_are_rejected_never_truncated(self):
        spec = make_spec(io=build_io(dynamic=True), max_seq_len=4)
        with self.assertRaisesRegex(ReadoutError, "5 tokens > the caller-declared max_seq_len 4"):
            encode_b1_dynamic([1, 2, 3, 4, 5], spec)
        encode_b1_dynamic([1, 2, 3, 4], spec)                  # exactly at the bound is legal

    def test_empty_and_non_int_inputs_are_refused(self):
        spec = make_spec(io=build_io(dynamic=True), max_seq_len=2048)
        with self.assertRaisesRegex(ReadoutError, "0 tokens"):
            encode_b1_dynamic([], spec)
        for bad in ([True, 2], [1.0, 2], ["1", 2], [None, 2]):
            with self.assertRaisesRegex(ReadoutError, "true int"):
                encode_b1_dynamic(bad, spec)

    def test_out_of_vocab_tokens_are_refused(self):
        spec = make_spec(io=build_io(dynamic=True), vocab=VOCAB, max_seq_len=2048)
        with self.assertRaisesRegex(ReadoutError, "vocab range"):
            encode_b1_dynamic([1, VOCAB + 5], spec)
        encode_b1_dynamic([1, VOCAB - 1], spec)                # boundary id is legal

    def test_a_fixed_spec_is_refused(self):
        with self.assertRaisesRegex(ReadoutError, "dynamic GraphSpec only"):
            encode_b1_dynamic([1, 2], make_spec())

    def test_encode_padded_refuses_a_dynamic_spec(self):
        spec = make_spec(io=build_io(dynamic=True), max_seq_len=2048)
        with self.assertRaisesRegex(ReadoutError, "fixed GraphSpec only"):
            encode_padded([1, 2], spec)


class TestRuntimeExecution(unittest.TestCase):
    """The real run_plan/session paths against a fake runtime binary."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.TemporaryDirectory()
        cls.bin = Path(cls.root.name) / "fake-ge-runtime"
        cls.bin.write_text(FAKE_RUNTIME)
        cls.bin.chmod(0o755)
        cls.run_dir = Path(cls.root.name) / "runs"

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def setUp(self):
        os.environ.pop("FAKE_RUNTIME_MODE", None)
        os.environ.pop("FAKE_RUNTIME_IO", None)

    def tearDown(self):
        os.environ.pop("FAKE_RUNTIME_MODE", None)
        os.environ.pop("FAKE_RUNTIME_IO", None)

    def session(self, **kwargs):
        return ge_readout.GEReadoutSession(device=5, run_dir=self.run_dir,
                                           runtime_bin=self.bin, timeout=30.0, **kwargs)

    def test_one_readout_through_the_whole_plan_protocol(self):
        session = self.session(vocab_size=VOCAB)
        self.assertEqual(session.spec.width, 64)
        self.assertEqual(session.spec.score_columns, 26)
        scores = session.read_scores26([3, 1, 2])
        self.assertEqual(scores, [float(i) for i in range(26)])
        record = session.records[0]
        self.assertEqual(record["seq_len"], 3)
        self.assertEqual(record["timings_ms"], {"compile_ms": 12.5, "load_ms": 1.5})
        plan = json.loads(Path(record["plan"]).read_text())
        self.assertEqual(plan["backend"], "ge_session")
        self.assertEqual(plan["mode"], "infer")
        self.assertEqual(plan["device"], 5)
        self.assertEqual(plan["options"], {"graph_run_mode": 0,
                                           "precision_mode": "must_keep_origin_dtype"})
        self.assertEqual(plan["inputs"]["cases"][0]["output_capacities"], {"scores": 104})
        tensors = {t["name"]: t for t in plan["inputs"]["cases"][0]["tensors"]}
        self.assertEqual([tensors[n]["offset"] for n in
                          ("input_ids", "position_ids", "actual_seq_lengths")], [0, 512, 1024])
        self.assertEqual([tensors[n]["bytes"] for n in
                          ("input_ids", "position_ids", "actual_seq_lengths")], [512, 512, 4])
        self.assertEqual(plan["inputs"]["cases"][0]["samples"], 1)
        expected = encode_padded([3, 1, 2], session.spec)
        written = Path(record["inputs"]).read_bytes()
        self.assertEqual(written, expected["input_ids"].tobytes()
                         + expected["position_ids"].tobytes()
                         + expected["actual_seq_lengths"].tobytes())
        self.assertTrue(Path(record["outputs"]).is_file())

    def test_each_session_gets_its_own_directory_and_reruns_never_overwrite(self):
        first = self.session()
        second = self.session()
        self.assertNotEqual(first.session_dir, second.session_dir)
        self.assertGreater(second.session_dir.name, first.session_dir.name)   # fresh NNNN
        first.read_scores26([1])
        marker = first.session_dir / "readout-0000" / "plan.json"
        third = self.session()
        self.assertGreater(third.session_dir.name, second.session_dir.name)
        self.assertTrue(marker.is_file())

    def test_failed_status_is_a_clear_error(self):
        os.environ["FAKE_RUNTIME_MODE"] = "failed"
        session = self.session()
        with self.assertRaisesRegex(ReadoutError, "status 'failed'"):
            session.read_scores26([1])

    def test_missing_or_duplicated_result_lines_are_errors(self):
        for mode in ("no_result", "double_result"):
            os.environ["FAKE_RUNTIME_MODE"] = mode
            with self.assertRaisesRegex(ReadoutError, "GE_RESULT lines"):
                self.session().read_scores26([1])
            os.environ.pop("FAKE_RUNTIME_MODE", None)

    def test_timeout_kills_the_run_with_a_clear_error(self):
        os.environ["FAKE_RUNTIME_MODE"] = "hang"
        session = self.session()
        session.timeout = 1.0
        with self.assertRaisesRegex(ReadoutError, "did not finish within 1.0s"):
            session.read_scores26([1])

    def test_non_finite_scores_are_refused(self):
        os.environ["FAKE_RUNTIME_MODE"] = "nan"
        session = self.session()
        with self.assertRaisesRegex(ReadoutError, "non-finite"):
            session.read_scores26([1])

    def test_wrong_output_shape_is_refused(self):
        os.environ["FAKE_RUNTIME_MODE"] = "short_shape"
        session = self.session()
        with self.assertRaisesRegex(ReadoutError, "does not match the AIR contract"):
            session.read_scores26([1])

    def test_dynamic_readout_through_the_whole_plan_protocol(self):
        os.environ["FAKE_RUNTIME_IO"] = "dynamic"
        session = self.session(vocab_size=VOCAB, max_seq_len=2048)
        self.assertTrue(session.spec.dynamic)
        self.assertEqual(session.spec.max_seq_len, 2048)
        scores = session.read_scores26([3, 1, 2, 9])
        self.assertEqual(scores, [float(i) for i in range(26)])
        record = session.records[0]
        self.assertEqual(record["seq_len"], 4)
        self.assertTrue(record["dynamic"])
        self.assertEqual(record["width"], 4)                    # the true length, no padding
        self.assertEqual(record["max_seq_len"], 2048)
        plan = json.loads(Path(record["plan"]).read_text())
        self.assertEqual(plan["io"]["outputs"][0]["shape"], [-1, 26])
        self.assertEqual(plan["io"]["outputs"][0]["dynamic_dims"], [0])
        self.assertEqual(plan["inputs"]["cases"][0]["output_capacities"], {"scores": 104})
        tensors = {t["name"]: t for t in plan["inputs"]["cases"][0]["tensors"]}
        self.assertEqual([tensors[n]["shape"] for n in
                          ("input_ids", "position_ids", "actual_seq_lengths")],
                         [[1, 4], [1, 4], [1]])
        self.assertEqual([tensors[n]["bytes"] for n in
                          ("input_ids", "position_ids", "actual_seq_lengths")], [32, 32, 4])
        self.assertEqual([tensors[n]["offset"] for n in
                          ("input_ids", "position_ids", "actual_seq_lengths")], [0, 32, 64])
        expected = encode_b1_dynamic([3, 1, 2, 9], session.spec)
        written = Path(record["inputs"]).read_bytes()
        self.assertEqual(written, expected["input_ids"].tobytes()
                         + expected["position_ids"].tobytes()
                         + expected["actual_seq_lengths"].tobytes())
        self.assertTrue(Path(record["outputs"]).is_file())

    def test_dynamic_readout_must_return_the_actual_1x26(self):
        os.environ["FAKE_RUNTIME_IO"] = "dynamic"
        os.environ["FAKE_RUNTIME_MODE"] = "dynamic_batch2"
        session = self.session(vocab_size=VOCAB, max_seq_len=2048)
        with self.assertRaisesRegex(ReadoutError, "does not match the AIR contract"):
            session.read_scores26([1, 2])

    def test_overlong_dynamic_readout_is_rejected_before_any_execution(self):
        os.environ["FAKE_RUNTIME_IO"] = "dynamic"
        session = self.session(vocab_size=VOCAB, max_seq_len=8)
        with self.assertRaisesRegex(ReadoutError, "max_seq_len 8"):
            session.read_scores26(list(range(9)))
        self.assertEqual(session.records, [])                   # nothing ran, nothing recorded

    def test_dynamic_session_with_a_fixed_graph_and_bound_is_refused(self):
        with self.assertRaisesRegex(ReadoutError, "dynamic contract only"):
            self.session(vocab_size=VOCAB, max_seq_len=2048)    # the default (fixed) fake io

    def test_parse_timings_reads_the_runtime_lines(self):
        text = "[INFO] CompileGraph ok (gid=1/1), cost 36999.4 ms\n" \
               "[INFO] LoadGraph ok (gid=1, tid=0), cost 288.717 ms\n"
        self.assertEqual(parse_timings(text), {"compile_ms": 36999.4, "load_ms": 288.717})
        self.assertEqual(parse_timings("nothing here"), {})


class TestCallback(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()
        self.ramp = [float(i) for i in range(26)]

    def test_selection_returns_raw_logits_in_asked_order(self):
        session = FakeSession(lambda ids: self.ramp)
        callback = ge_readout.GEReadoutCallback(session, self.tok, list(range(32, 58)))
        values = callback("abc", [32, 35, 33])
        self.assertEqual(values, [0.0, 3.0, 1.0])
        record = callback.records[0]
        self.assertEqual(record["prompt"], "abc")
        self.assertEqual(record["input_ids"], [97, 98, 99])
        self.assertEqual(record["active_letters"], ["A", "D", "B"])
        self.assertEqual(record["scores26"], self.ramp)
        self.assertEqual(record["selected_logits"], values)

    def test_letter_ids_are_validated_not_coerced(self):
        session = FakeSession(lambda ids: self.ramp)
        callback = ge_readout.GEReadoutCallback(session, self.tok, list(range(32, 58)))
        with self.assertRaisesRegex(ReadoutError, "true int"):
            callback("abc", [True, 33])
        with self.assertRaisesRegex(ReadoutError, "not among the folder's 26"):
            callback("abc", [32, 99])
        with self.assertRaisesRegex(ReadoutError, "duplicated"):
            callback("abc", [32, 32])
        with self.assertRaisesRegex(ReadoutError, "no letters"):
            callback("abc", [])
        with self.assertRaisesRegex(ReadoutError, "true int"):
            ge_readout.GEReadoutCallback(session, self.tok, [32.0] + list(range(33, 58)))

    def test_overlong_prompt_is_rejected_never_truncated(self):
        session = FakeSession(lambda ids: self.ramp, width=8)
        callback = ge_readout.GEReadoutCallback(session, self.tok, list(range(32, 58)))
        with self.assertRaisesRegex(ReadoutError, "never"):
            callback("abcdefghij", [32])

    def test_empty_prompt_is_refused(self):
        session = FakeSession(lambda ids: self.ramp)
        callback = ge_readout.GEReadoutCallback(session, self.tok, list(range(32, 58)))
        with self.assertRaisesRegex(ReadoutError, "non-empty string"):
            callback("", [32])

    def test_dynamic_callback_enforces_the_caller_bound(self):
        spec = make_spec(io=build_io(dynamic=True), max_seq_len=8)
        session = FakeSession(lambda ids: self.ramp, spec=spec)
        callback = ge_readout.GEReadoutCallback(session, self.tok, list(range(32, 58)))
        values = callback("abc", [32, 35, 33])
        self.assertEqual(values, [0.0, 3.0, 1.0])
        self.assertEqual(callback.spec.length_bound, 8)
        with self.assertRaisesRegex(ReadoutError, "never"):
            callback("abcdefghijk", [32])                       # 11 tokens > max_seq_len 8


class TestOfficialWiring(unittest.TestCase):
    """The official Decider on the real bridge, with marker-driven graph scores."""

    def setUp(self):
        self.tok = FakeTokenizer()
        self.session = MarkerSession(self.tok)
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = make_folder(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def decide(self, questions, state="s", body=None):
        decision, callback = startlux_decision(self.folder, self.tok, self.session)
        return decision.decide(state, questions, body=body), callback

    def test_choice_reads_the_marked_option_through_the_real_graph(self):
        crit = {"a": None, "b": MARK, "c": None}
        decision, callback = self.decide({"q": dict(PICK, criteria=crit)})
        answer = decision.answers["q"]
        self.assertEqual(answer["choice"], "b")
        self.assertEqual(len(callback.records), 1)
        record = callback.records[0]
        self.assertEqual(record["letters_asked"], 4)           # 3 options + the none letter
        self.assertIn("B) b: " + MARK, record["prompt"])
        self.assertEqual(record["scores26"][1], HIGH)          # column B carries the marker
        self.assertEqual(record["selected_logits"][1], HIGH)   # raw logits, no temperature here
        self.assertEqual(decision.usage["input_tokens"], record["seq_len"])

    def test_noul_reports_the_marked_true_option(self):
        crit = {"true": MARK, "false": None}
        decision, callback = self.decide({"q": {"type": "noul", "criteria": crit}})
        self.assertGreater(decision.answers["q"]["noul"], 0.5)
        self.assertEqual(callback.records[0]["letters_asked"], 2)

    def test_wide_27_knocks_out_group_by_group_over_sequential_readouts(self):
        crit = {f"o{i:02d}": (MARK if i == 27 else None) for i in range(1, 28)}
        decision, callback = self.decide({"q": dict(PICK, criteria=crit)})
        self.assertEqual(decision.answers["q"]["choice"], "o27")
        self.assertEqual([r["letters_asked"] for r in callback.records], [14, 13, 7])
        for record in callback.records:                        # selection stays column-exact
            self.assertEqual(record["selected_logits"],
                             [record["scores26"][t - 65] for t in record["letter_ids"]])
        self.assertIn("D) o27: " + MARK, callback.records[2]["prompt"])

    def test_reentry_of_the_same_callback_serves_new_requests(self):
        first, callback = self.decide({"q": dict(PICK, criteria={"a": MARK, "b": None})})
        self.assertEqual(first.answers["q"]["choice"], "a")
        before = [dict(r) for r in callback.records]
        second = self.session_decide_again(callback, {"q": dict(PICK, criteria={"x": None, "y": MARK,
                                                                                "z": None})},
                                           "another state")
        self.assertEqual(second.answers["q"]["choice"], "y")
        self.assertEqual(len(callback.records), 2)
        self.assertEqual(callback.records[0], before[0])        # earlier evidence untouched
        self.assertNotEqual(callback.records[0]["input_ids"], callback.records[1]["input_ids"])

    def session_decide_again(self, callback, questions, state):
        decision, _ = startlux_decision(self.folder, self.tok, self.session)
        decision.backend.readout_fn = callback
        return decision.decide(state, questions, body=None)


class TestModulePurity(unittest.TestCase):
    def test_ge_readout_imports_without_torch(self):
        code = ("import importlib.util, sys\n"
                f"spec = importlib.util.spec_from_file_location('ge_readout', {str(MODEL_DIR / 'decider' / 'ge_readout.py')!r})\n"
                "m = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(m)\n"
                "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)\n"
                "assert 'torch_npu' not in sys.modules\n"
                "print('clean')\n")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             timeout=120, cwd=str(Path.cwd()))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("clean", out.stdout)


if __name__ == "__main__":
    unittest.main()
