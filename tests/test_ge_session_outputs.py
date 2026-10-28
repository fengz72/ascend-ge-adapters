"""Opt-in NPU regressions: GE_TEST_DEVICE=0 GE_TEST_TARGET=Ascend910_9382."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.config import ROOT, load_authoring


@unittest.skipUnless(os.environ.get("GE_TEST_DEVICE"), "set GE_TEST_DEVICE for GE hardware tests")
class GeSessionOutputs(unittest.TestCase):
    def setUp(self):
        self.device = int(os.environ["GE_TEST_DEVICE"])
        (ROOT / "work").mkdir(exist_ok=True)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory(dir=ROOT / "work")))
        self.config = load_authoring(ROOT / "models/bge/config.yaml",
                                     target=os.environ.get("GE_TEST_TARGET", "Ascend910_9382"),
                                     source_paths=False)
        self.config["work_dir"] = str(self.directory)
        self.config["model"]["params"] = {}
        # These are generic runtime probes, independent of BGE business references.
        self.config["verification"].pop("script", None)
        self.config["verification"].pop("params", None)
        self.config["source"]["ref"] = str(self.directory / "source.onnx")
        self.assertNotIn("graph_run_mode", self.config["build"]["options"])
        self.config["verification"]["case_ids"] = "all"
        self.config["benchmark"].update(instances=1, iterations=12, warmup=1)

    def model(self, shape, multiple=False):
        import onnx
        from onnx import helper, TensorProto
        axis = 0 if len(shape) == 1 else 1
        begin = 3 if axis == 0 else 2
        output_shape = list(shape)
        output_shape[axis] = 1
        nodes = [helper.make_node("Slice", ["x", "begin", "end", "axes", "steps"], ["y"])]
        outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)]
        if multiple:
            nodes.append(helper.make_node("Cast", ["y"], ["integer"], to=TensorProto.INT64))
            outputs.append(helper.make_tensor_value_info("integer", TensorProto.INT64, output_shape))
        graph = helper.make_graph(nodes, "output-contracts",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)], outputs,
            initializer=[helper.make_tensor(name, TensorProto.INT64, [1], [value])
                         for name, value in (("begin", begin), ("end", begin+1), ("axes", axis), ("steps", 1))])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        model.ir_version = 7
        onnx.checker.check_model(model)
        onnx.save(model, self.config["source"]["ref"])
        self.config["verification"]["outputs"] = {x.name: {"policy": "exact"} for x in outputs}

    def cases(self, shapes):
        return [{"id": f"case-{i:04d}", "samples": shape[0] if len(shape) == 2 else 1,
                 "payload": {"x": (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + i*10).tolist()}}
                for i, shape in enumerate(shapes)]

    def execute(self, cases):
        from core.worker import build, baseline, infer_or_bench, compare_outputs
        build(self.config, cases, self.device)
        baseline(self.config, cases, self.device)
        result = infer_or_bench("infer", self.config, cases, self.directory / "infer", self.device)
        self.assertEqual(result["status"], "success")
        self.assertEqual(compare_outputs(self.config)["status"], "success")
        log = (self.directory / "infer/infer-runtime.log").read_text()
        expected_mode = self.config["build"]["options"].get("graph_run_mode", 0)
        self.assertIn(f"GEInitialize ge.graphRunMode={expected_mode}", log)
        return log

    def test_static_original_slice_reuses_output_buffers_per_instance(self):
        from core.worker import infer_or_bench
        self.model([4])
        cases = self.cases([(4,), (4,)])
        log = self.execute(cases)
        self.assertIn("caller (static compiled)", log)
        self.assertEqual(log.count("Allocated output[0]"), 1)
        with np.load(self.directory / "outputs.npz") as values:
            np.testing.assert_array_equal(values["case-0000/outputs/y"], [3.0])
            np.testing.assert_array_equal(values["case-0001/outputs/y"], [13.0])
        self.config["benchmark"]["instances"] = 2
        result = infer_or_bench("bench", self.config, cases, self.directory / "bench", self.device)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["performance"]["requests"], 12)
        self.assertEqual(result["performance"]["errors"], 0)
        log = (self.directory / "bench/bench-runtime.log").read_text()
        self.assertEqual(log.count("Allocated output[0]"), 2)

    def test_static_multiple_output_dtypes(self):
        self.model([2, 4], multiple=True)
        log = self.execute(self.cases([(2, 4), (2, 4)]))
        self.assertIn("dtype=int64 bytes=16", log)
        self.assertEqual(log.count("Allocated output["), 2)

    def test_dynamic_batch_and_multiple_outputs(self):
        self.model(["batch", 4], multiple=True)
        log = self.execute(self.cases([(1, 4), (2, 4), (1, 4)]))
        self.assertIn("GE (dynamic compiled)", log)
        self.assertNotIn("Allocated output[", log)

    def test_mode1_static_graph_uses_same_output_contract(self):
        self.config["build"]["options"]["graph_run_mode"] = 1
        self.model([2, 4], multiple=True)
        self.execute(self.cases([(2, 4), (2, 4)]))

    def test_static_capacity_too_small_fails_before_execute(self):
        from core.worker import build, infer_or_bench
        self.model([4])
        self.config["model"]["params"]["output_capacities"] = {"y": 1}
        cases = self.cases([(4,)])
        build(self.config, cases, self.device)
        with self.assertRaisesRegex(RuntimeError, "declared capacity"):
            infer_or_bench("infer", self.config, cases, self.directory / "infer", self.device)
