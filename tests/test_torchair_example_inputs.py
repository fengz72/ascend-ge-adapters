"""export_spec['example_inputs'] behavior of the torchair exporter.

CPU-only: the validation helper runs directly, and the full export() path is
driven with the torchair/torch_npu modules replaced by fakes (the same
dynamo_export call shape the real path uses), so the tests prove WHICH values
dynamo_export would trace without touching NPU or the real compiler:

  * a normal B1 first case (what core.worker.build passes from the default
    fixture pool) is overridden by the explicit B2/S65 example -- the plain
    CLI build traces the dynamic batch, not the fixture;
  * without example_inputs the legacy behavior is unchanged (the passed
    inputs are traced and still validated against the IOSpec names);
  * wrong keys, dtypes, ranks, static dims and size-1 dynamic-axis examples
    are rejected loudly BEFORE dynamo_export is ever called, so no AIR can
    claim dynamic axes it did not actually trace.
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.exporters.torchair import _example_values, export  # noqa: E402
from core.contracts import validate_io  # noqa: E402


def dynamic_io():
    """The StartLux native dynamic contract ([-1,-1]/[-1,-1]/[-1] -> [-1,26])."""
    return validate_io({"schema": "ge-io/1",
                        "inputs": [
                            {"name": "input_ids", "dtype": "int64", "shape": [-1, -1],
                             "dynamic_dims": [0, 1]},
                            {"name": "position_ids", "dtype": "int64", "shape": [-1, -1],
                             "dynamic_dims": [0, 1]},
                            {"name": "actual_seq_lengths", "dtype": "int32", "shape": [-1],
                             "dynamic_dims": [0]}],
                        "outputs": [{"name": "scores", "dtype": "float32", "shape": [-1, 26],
                                     "dynamic_dims": [0]}]})


def b2_s65_examples():
    """The adapter's deterministic legal trace example (zeros/arange/width)."""
    batch, width = 2, 65
    return {"input_ids": torch.zeros((batch, width), dtype=torch.int64),
            "position_ids": torch.arange(width, dtype=torch.int64).repeat(batch, 1),
            "actual_seq_lengths": torch.full((batch,), width, dtype=torch.int32)}


def b1_fixture_inputs(length=86):
    """What core.worker.build passes today: the encoded first fixture case (B1)."""
    return {"input_ids": torch.zeros((1, length), dtype=torch.int64),
            "position_ids": torch.arange(length, dtype=torch.int64).unsqueeze(0),
            "actual_seq_lengths": torch.tensor([length], dtype=torch.int32)}


class FakeTorchair:
    """Replaces torch_npu/torchair during export(); records the traced values."""

    def __init__(self, names):
        self.names = list(names)
        self.recorded = []
        self.compiler_config = None

    def modules(self):
        outer = self

        class CompilerConfig:
            def __init__(self):
                self.experimental_config = types.SimpleNamespace(frozen_parameter=False)

        def dynamo_export(*values, model, export_path, export_name, dynamic, config):
            outer.recorded.append({"shapes": [tuple(v.shape) for v in values],
                                   "dtypes": [str(v.dtype) for v in values],
                                   "dynamic": dynamic})
            directory = Path(export_path)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{export_name}.air").write_bytes(b"air")
            lines = []
            for index, name in enumerate(outer.names):
                lines += [
                    "node {",
                    f'  name: "arg{2 * index + 2}_1"',
                    '  op: "Data"',
                    "  attr {",
                    '    key: "index"',
                    "    value {",
                    f"      i: {index}",
                    "    }",
                    "  }",
                    "  attr {",
                    '    key: "_ge_adapter_name"',
                    "    value {",
                    f'      s: "{name}"',
                    "    }",
                    "  }",
                    "}",
                ]
            (directory / "dynamo.pbtxt").write_text("\n".join(lines) + "\n")

        def register_fx_node_ge_converter(op):
            def decorator(fn):
                return fn
            return decorator

        npu = types.ModuleType("torch_npu")
        dynamo = types.ModuleType("torch_npu.dynamo")
        tair = types.ModuleType("torch_npu.dynamo.torchair")
        tair.dynamo_export = dynamo_export
        tair.CompilerConfig = CompilerConfig
        dynamo.torchair = tair
        npu.dynamo = dynamo
        top = types.ModuleType("torchair")
        top.register_fx_node_ge_converter = register_fx_node_ge_converter
        return {"torch_npu": npu, "torch_npu.dynamo": dynamo,
                "torch_npu.dynamo.torchair": tair, "torchair": top}


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *args):
        return args[0]


class ExportCall(unittest.TestCase):
    """Drive the real export() with the compiler modules faked out."""

    def run_export(self, spec, inputs, io=None):
        fake = FakeTorchair([node["name"] for node in (io or spec["io"])["inputs"]])
        saved_npu = getattr(torch, "npu", None)
        torch.npu = types.SimpleNamespace(synchronize=lambda: None)
        try:
            with mock.patch.dict(sys.modules, fake.modules()), \
                    tempfile.TemporaryDirectory(dir=Path("work/tmp")) as directory:
                result = export(spec, inputs, Path(directory))
        finally:
            if saved_npu is None:
                delattr(torch, "npu")
            else:
                torch.npu = saved_npu
        return fake, result

    def test_explicit_b2_example_traces_over_the_b1_first_case(self):
        io = dynamic_io()
        spec = {"format": "torch", "model": _Model(), "io": io,
                "example_inputs": b2_s65_examples()}
        fake, result = self.run_export(spec, b1_fixture_inputs(), io)
        self.assertEqual(len(fake.recorded), 1)
        self.assertEqual(fake.recorded[0]["shapes"],
                         [(2, 65), (2, 65), (2,)])
        self.assertEqual(fake.recorded[0]["dtypes"],
                         ["torch.int64", "torch.int64", "torch.int32"])
        self.assertTrue(fake.recorded[0]["dynamic"])
        self.assertEqual(result[1]["inputs"][0]["name"], "input_ids")
        self.assertEqual([node["name"] for node in result[1]["inputs"]],
                         ["input_ids", "position_ids", "actual_seq_lengths"])

    def test_legacy_inputs_are_traced_without_examples(self):
        io = dynamic_io()
        spec = {"format": "torch", "model": _Model(), "io": io}
        inputs = b1_fixture_inputs()
        fake, _ = self.run_export(spec, inputs, io)
        self.assertEqual(fake.recorded[0]["shapes"],
                         [(1, 86), (1, 86), (1,)])
        self.assertTrue(fake.recorded[0]["dynamic"])

    def test_legacy_name_mismatch_still_rejected(self):
        io = dynamic_io()
        spec = {"format": "torch", "model": _Model(), "io": io}
        inputs = b1_fixture_inputs()
        inputs.pop("actual_seq_lengths")
        fake = FakeTorchair([node["name"] for node in io["inputs"]])
        saved_npu = getattr(torch, "npu", None)
        torch.npu = types.SimpleNamespace(synchronize=lambda: None)
        try:
            with mock.patch.dict(sys.modules, fake.modules()):
                with self.assertRaises(ValueError):
                    export(spec, inputs, Path("work/tmp/never"))
        finally:
            if saved_npu is None:
                delattr(torch, "npu")
            else:
                torch.npu = saved_npu
        self.assertEqual(fake.recorded, [])

    def test_bad_examples_rejected_before_any_tracing(self):
        io = dynamic_io()
        bad = dict(b2_s65_examples())
        bad["input_ids"] = bad["input_ids"].to(torch.int32)
        spec = {"format": "torch", "model": _Model(), "io": io, "example_inputs": bad}
        fake = FakeTorchair([node["name"] for node in io["inputs"]])
        saved_npu = getattr(torch, "npu", None)
        torch.npu = types.SimpleNamespace(synchronize=lambda: None)
        try:
            with mock.patch.dict(sys.modules, fake.modules()):
                with self.assertRaises(ValueError):
                    export(spec, b1_fixture_inputs(), Path("work/tmp/never"))
        finally:
            if saved_npu is None:
                delattr(torch, "npu")
            else:
                torch.npu = saved_npu
        self.assertEqual(fake.recorded, [])

    def test_static_axes_are_pinned_with_mark_static(self):
        """dynamic=True symbolizes UNMARKED axes too, so every IOSpec-static
        axis must be pinned (unified mixed contract: position_ids leading 3,
        pixel_values [512,1536]) while declared axes stay mark_dynamic."""
        io = validate_io({"schema": "ge-io/1",
                          "inputs": [
                              {"name": "input_ids", "dtype": "int64", "shape": [-1, -1],
                               "dynamic_dims": [0, 1]},
                              {"name": "position_ids", "dtype": "int64", "shape": [3, -1, -1],
                               "dynamic_dims": [1, 2]},
                              {"name": "actual_seq_lengths", "dtype": "int32", "shape": [-1],
                               "dynamic_dims": [0]},
                              {"name": "pixel_values", "dtype": "float32", "shape": [512, 1536]},
                              {"name": "image_feature_index", "dtype": "int64", "shape": [-1, -1],
                               "dynamic_dims": [0, 1]}],
                          "outputs": [{"name": "scores", "dtype": "float32", "shape": [-1, 4],
                                       "dynamic_dims": [0]}]})
        examples = {"input_ids": torch.zeros((2, 65), dtype=torch.int64),
                    "position_ids": torch.zeros((3, 2, 65), dtype=torch.int64),
                    "actual_seq_lengths": torch.full((2,), 65, dtype=torch.int32),
                    "pixel_values": torch.zeros((512, 1536), dtype=torch.float32),
                    "image_feature_index": torch.zeros((2, 65), dtype=torch.int64)}
        spec = {"format": "torch", "model": _Model(), "io": io, "example_inputs": examples}
        fake = FakeTorchair([node["name"] for node in io["inputs"]])
        marks = {"dynamic": [], "static": []}
        by_id = {id(value): name for name, value in examples.items()}

        def mark_dynamic(tensor, index):
            marks["dynamic"].append((by_id[id(tensor)], index))

        def mark_static(tensor, index):
            marks["static"].append((by_id[id(tensor)], index))

        saved_npu = getattr(torch, "npu", None)
        torch.npu = types.SimpleNamespace(synchronize=lambda: None)
        try:
            with mock.patch.dict(sys.modules, fake.modules()), \
                    mock.patch.object(torch._dynamo, "mark_dynamic", mark_dynamic), \
                    mock.patch.object(torch._dynamo, "mark_static", mark_static), \
                    tempfile.TemporaryDirectory(dir=Path("work/tmp")) as directory:
                export(spec, {}, Path(directory))
        finally:
            if saved_npu is None:
                delattr(torch, "npu")
            else:
                torch.npu = saved_npu
        self.assertEqual(marks["dynamic"],
                         [("input_ids", 0), ("input_ids", 1),
                          ("position_ids", 1), ("position_ids", 2),
                          ("actual_seq_lengths", 0),
                          ("image_feature_index", 0), ("image_feature_index", 1)])
        self.assertEqual(marks["static"],
                         [("position_ids", 0),
                          ("pixel_values", 0), ("pixel_values", 1)])
        self.assertTrue(fake.recorded[0]["dynamic"])


class ExampleValidation(unittest.TestCase):
    """The strict example validator, independent of the compiler modules."""

    def test_valid_b2_s65_ordered_by_iospec(self):
        values = _example_values(torch, dynamic_io(), b2_s65_examples())
        self.assertEqual([tuple(v.shape) for v in values], [(2, 65), (2, 65), (2,)])
        self.assertEqual([v.dtype for v in values],
                         [torch.int64, torch.int64, torch.int32])

    def test_missing_unknown_and_misnamed_keys_rejected(self):
        io = dynamic_io()
        missing = dict(b2_s65_examples())
        missing.pop("position_ids")
        with self.assertRaises(ValueError):
            _example_values(torch, io, missing)
        extra = dict(b2_s65_examples(), scores=torch.zeros(2, 26))
        with self.assertRaises(ValueError):
            _example_values(torch, io, extra)
        renamed = dict(b2_s65_examples())
        renamed["input_idz"] = renamed.pop("input_ids")
        with self.assertRaises(ValueError):
            _example_values(torch, io, renamed)
        with self.assertRaises(ValueError):
            _example_values(torch, io, [b2_s65_examples()["input_ids"]])

    def test_wrong_dtype_rejected(self):
        io = dynamic_io()
        bad = dict(b2_s65_examples())
        bad["actual_seq_lengths"] = bad["actual_seq_lengths"].to(torch.int64)
        with self.assertRaises(ValueError):
            _example_values(torch, io, bad)

    def test_wrong_rank_and_static_dims_rejected(self):
        io = dynamic_io()
        bad = dict(b2_s65_examples())
        bad["input_ids"] = bad["input_ids"].reshape(2, 1, 65)
        with self.assertRaises(ValueError):
            _example_values(torch, io, bad)
        fixed = validate_io({"schema": "ge-io/1",
                             "inputs": [{"name": "input_ids", "dtype": "int64",
                                         "shape": [1, 256]},
                                        {"name": "actual_seq_lengths", "dtype": "int32",
                                         "shape": [1]}],
                             "outputs": [{"name": "scores", "dtype": "float32",
                                          "shape": [1, 26]}]})
        bad = {"input_ids": torch.zeros((2, 256), dtype=torch.int64),
               "actual_seq_lengths": torch.tensor([256, 256], dtype=torch.int32)}
        with self.assertRaises(ValueError):
            _example_values(torch, fixed, bad)

    def test_size_one_dynamic_axis_rejected(self):
        io = dynamic_io()
        b1 = dict(b2_s65_examples())
        b1["input_ids"] = torch.zeros((1, 65), dtype=torch.int64)
        b1["position_ids"] = torch.arange(65, dtype=torch.int64).unsqueeze(0)
        b1["actual_seq_lengths"] = torch.tensor([65], dtype=torch.int32)
        with self.assertRaises(ValueError):
            _example_values(torch, io, b1)
        s1 = dict(b2_s65_examples())
        s1["input_ids"] = torch.zeros((2, 1), dtype=torch.int64)
        s1["position_ids"] = torch.zeros((2, 1), dtype=torch.int64)
        s1["actual_seq_lengths"] = torch.full((2,), 1, dtype=torch.int32)
        with self.assertRaises(ValueError):
            _example_values(torch, io, s1)

    def test_static_axis_may_trace_one(self):
        fixed = validate_io({"schema": "ge-io/1",
                             "inputs": [{"name": "input_ids", "dtype": "int64",
                                         "shape": [1, -1], "dynamic_dims": [1]},
                                        {"name": "actual_seq_lengths", "dtype": "int32",
                                         "shape": [1]}],
                             "outputs": [{"name": "scores", "dtype": "float32",
                                          "shape": [1, 26]}]})
        values = _example_values(torch, fixed, {
            "input_ids": torch.zeros((1, 65), dtype=torch.int64),
            "actual_seq_lengths": torch.tensor([65], dtype=torch.int32)})
        self.assertEqual([tuple(v.shape) for v in values], [(1, 65), (1,)])

    def test_non_tensor_rejected(self):
        import numpy as np
        io = dynamic_io()
        bad = dict(b2_s65_examples())
        bad["actual_seq_lengths"] = np.full(2, 65, dtype=np.int32)
        with self.assertRaises(ValueError):
            _example_values(torch, io, bad)


if __name__ == "__main__":
    unittest.main()
