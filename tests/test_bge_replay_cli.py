"""The BGE preparation CLI must stay CPU-only and preserve request contracts."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core import native_onnx
from models.bge import replay


class BgeReplayPreparation(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.graph = self.directory / "model.onnx"
        self.graph.write_bytes(b"small mocked external-weight ONNX")
        self.config = {"source": {"kind": "onnx", "ref": "${UNSET_SOURCE_MODEL}"},
                       "build": {"backend": "ge_session", "dtype": "float32",
                                 "options": {"aicore_num": "4|8", "precision_mode": "must_keep_origin_dtype"}},
                       "model": {"adapter_file": "core/native_onnx.py", "params": {}},
                       "work_dir": str(self.directory)}
        shapes = {"input_ids": [-1, -1], "attention_mask": [-1, -1],
                  "long_cid3_raw": [-1, -1], "long_action": [-1, -1], "long_time_bucket": [-1, -1],
                  "stat_dense_raw": [-1, 20], "stat_ratio": [-1, 20], "stat_count": [-1]}
        self.io = {"schema": "ge-io/1", "inputs": [
            {"name": name, "dtype": dtype, "shape": shapes[name]}
            for name, dtype in replay.INPUT_DTYPES.items()], "outputs": [
                {"name": "scores", "dtype": "float32", "shape": [-1, 10112]}]}
        self.cases = []
        for index, length in enumerate((6, 9, 8, 13, 8)):
            payload = {"input_ids": [[index + 1] * length], "attention_mask": [[1] * length],
                       "long_cid3_raw": [[index] * 128], "long_action": [[1] * 128],
                       "long_time_bucket": [[1] * 128], "stat_dense_raw": [[index] * 20],
                       "stat_ratio": [[.125 + index / 100] * 20], "stat_count": [20]}
            self.cases.append({"id": f"case-{index:04d}", "samples": 1, "payload": payload})
        self.authoring = self.enterContext(patch.object(replay, "load_authoring", return_value=self.config))
        self.data = self.enterContext(patch.object(replay, "load_cases", return_value=self.cases))
        self.read_io = self.enterContext(patch.object(replay, "read_io", return_value=self.io))
        self.adapter = self.enterContext(patch.object(replay, "load_adapter", return_value=native_onnx))
        self.enterContext(patch.object(replay, "graph_files", return_value={
            "path": str(self.graph), "format": "onnx", "om_path": None}))
        self.native_call = self.enterContext(patch("core.backends.runtime.call", side_effect=AssertionError("must not run NPU")))
        self.source_load = self.enterContext(patch("core.sources.load_source", side_effect=AssertionError("must not load weights")))
        self.golden = self.directory / "golden.npz"
        self.golden.write_bytes(b"fixed original golden")

    def args(self, *extra):
        return replay.parser().parse_args(["--device", "10", "--output-dir", str(self.directory / "prepared"), *extra])

    def read_plan(self, args):
        return json.loads(replay.prepare(args).read_text())

    def test_prepare_packs_all_five_cases_and_exact_original_input_dtypes(self):
        plan = self.read_plan(self.args("--prepare-only", "--instances", "4", "--seed", "2027", "--capture"))
        self.assertEqual(plan["instances"], 4)
        self.assertEqual(plan["device"], 10)
        self.assertEqual(plan["seed"], 2027)
        self.assertTrue(plan["replay"]["capture_outputs"])
        self.assertEqual(plan["options"], {"precision_mode": "must_keep_origin_dtype", "graph_run_mode": 0, "aicore_num": "4|8"})
        self.assertEqual(len(plan["inputs"]["cases"]), 5)
        self.assertEqual(plan["workload"], {"kind": "random_cases",
                         "case_ids": [case["id"] for case in self.cases],
                         "sampling": "uniform_with_replacement"})
        binary = Path(plan["inputs"]["file"])
        self.assertEqual(binary.stat().st_size, 17304)
        for case, packed_case in zip(self.cases, plan["inputs"]["cases"]):
            self.assertEqual(packed_case["id"], case["id"])
            self.assertEqual(packed_case["output_capacities"], {"scores": 40448})
            for tensor in packed_case["tensors"]:
                expected = np.asarray(case["payload"][tensor["name"]], dtype=replay.INPUT_DTYPES[tensor["name"]])
                actual = np.fromfile(binary, dtype=tensor["dtype"], count=expected.size, offset=tensor["offset"]).reshape(tensor["shape"])
                np.testing.assert_array_equal(actual, expected)
        self.assertFalse(Path(plan["output_dir"]).exists())
        self.assertEqual(self.golden.read_bytes(), b"fixed original golden")
        self.authoring.assert_called_once_with(self.args().config, source_paths=False)
        self.native_call.assert_not_called()
        self.source_load.assert_not_called()

    def test_yaml_defaults_select_instances_precision_and_cores(self):
        self.config["benchmark"] = {"instances": 10}
        self.config["build"]["options"] = {"aicore_num": "12|24", "precision_mode": "force_fp16"}
        plan = self.read_plan(self.args())
        self.assertEqual(plan["instances"], 10)
        self.assertEqual(plan["options"], {"precision_mode": "force_fp16", "graph_run_mode": 0, "aicore_num": "12|24"})
        override = self.read_plan(self.args("--output-dir", str(self.directory / "override"),
                                            "--instances", "8", "--precision-mode", "must_keep_origin_dtype",
                                            "--aicore-num", "4|8"))
        self.assertEqual(override["instances"], 8)
        self.assertEqual(override["options"], {"precision_mode": "must_keep_origin_dtype", "graph_run_mode": 0, "aicore_num": "4|8"})

    def test_fixed_case_packs_only_the_selected_original_input(self):
        plan = self.read_plan(self.args("--case-id", "case-0002", "--instances", "8"))
        self.assertEqual(plan["instances"], 8)
        self.assertEqual(plan["workload"], {"kind": "fixed_case", "case_ids": ["case-0002"], "sampling": "fixed"})
        self.assertEqual(len(plan["inputs"]["cases"]), 1)
        case = plan["inputs"]["cases"][0]
        self.assertEqual(case["id"], "case-0002")
        self.assertEqual(case["output_capacities"], {"scores": 40448})
        binary = Path(plan["inputs"]["file"])
        self.assertEqual(binary.stat().st_size, 3448)
        for tensor in case["tensors"]:
            expected = np.asarray(self.cases[2]["payload"][tensor["name"]], dtype=replay.INPUT_DTYPES[tensor["name"]])
            self.assertEqual(tensor["dtype"], str(expected.dtype))
            self.assertEqual(tensor["shape"], list(expected.shape))
            actual = np.fromfile(binary, dtype=tensor["dtype"], count=expected.size, offset=tensor["offset"]).reshape(tensor["shape"])
            np.testing.assert_array_equal(actual, expected)
        self.assertFalse(Path(plan["output_dir"]).exists())
        self.assertEqual(self.golden.read_bytes(), b"fixed original golden")
        self.native_call.assert_not_called()
        self.source_load.assert_not_called()

    def test_unknown_fixed_case_is_rejected_before_output_creation(self):
        with self.assertRaisesRegex(ValueError, "unknown BGE case-id"):
            replay.prepare(self.args("--case-id", "case-9999"))
        self.assertFalse((self.directory / "prepared").exists())
        self.adapter.assert_not_called()

    def test_fixed_selection_still_requires_all_five_source_cases(self):
        self.data.return_value = self.cases[:4]
        with self.assertRaisesRegex(ValueError, "all five"):
            replay.prepare(self.args("--case-id", "case-0002"))
        self.assertFalse((self.directory / "prepared").exists())

    def test_fixed_seed_does_not_change_payload_and_repeated_prepare_is_rejected(self):
        first = self.read_plan(self.args("--case-id", "case-0002", "--seed", "2027"))
        binary = Path(first["inputs"]["file"]).read_bytes()
        second = self.read_plan(self.args("--case-id", "case-0002", "--seed", "2028",
                                         "--output-dir", str(self.directory / "second")))
        self.assertEqual(Path(second["inputs"]["file"]).read_bytes(), binary)
        self.assertEqual(first["inputs"]["cases"], second["inputs"]["cases"])
        with self.assertRaises(FileExistsError):
            replay.prepare(self.args("--case-id", "case-0002"))
        self.assertEqual(Path(first["inputs"]["file"]).read_bytes(), binary)

    def test_native_only_preload_appends_original_allocator_without_mutating_python_env(self):
        library = self.directory / "fusion.so"
        library.write_bytes(b"mock shared library; never loaded")
        with patch.dict(os.environ, {"LD_PRELOAD": "/original/allocator.so"}):
            plan = self.read_plan(self.args("--preload", str(library), "--precision-mode", "allow_mix_precision", "--aicore-num", "12|24"))
            self.assertEqual(plan["runtime_env"]["LD_PRELOAD"], f"/original/allocator.so:{library}")
            self.assertEqual(os.environ["LD_PRELOAD"], "/original/allocator.so")
        self.assertEqual(plan["options"]["precision_mode"], "allow_mix_precision")
        self.assertEqual(plan["options"]["aicore_num"], "12|24")

    def test_invalid_parameters_fail_before_config_or_output_creation(self):
        for extra in (("--device", "-1"), ("--instances", "0"), ("--seed", "-1"),
                      ("--min-seconds", "nan"), ("--min-seconds", "400"),
                      ("--min-requests", "0"), ("--max-requests", "10"),
                      ("--aicore-num", "0|24"), ("--aicore-num", "12"), ("--precision-mode", ""), ("--case-id", "")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                replay.prepare(self.args(*extra))
        self.authoring.assert_not_called()
        self.assertFalse((self.directory / "prepared").exists())

    def test_existing_preparation_directory_is_rejected_without_overwriting(self):
        directory = self.directory / "prepared"
        directory.mkdir()
        sentinel = directory / "plan.json"
        sentinel.write_text("keep me")
        with self.assertRaises(FileExistsError):
            replay.prepare(self.args())
        self.assertEqual(sentinel.read_text(), "keep me")
        self.authoring.assert_not_called()

    def test_wrong_graph_float_dtype_or_output_shape_is_rejected(self):
        for which in ("input", "output"):
            io = copy.deepcopy(self.io)
            if which == "input":
                next(node for node in io["inputs"] if node["name"] == "stat_ratio")["dtype"] = "float16"
            else:
                io["outputs"][0]["shape"] = [-1, 10111]
            self.read_io.return_value = io
            with self.subTest(which=which), self.assertRaises(ValueError):
                replay.prepare(self.args())
        self.assertFalse((self.directory / "prepared").exists())

    def test_missing_fifth_case_and_non_ge_route_are_rejected(self):
        self.data.return_value = self.cases[:4]
        with self.assertRaises(ValueError):
            replay.prepare(self.args())
        self.data.return_value = self.cases
        self.config["build"]["backend"] = "atc_om"
        with self.assertRaises(ValueError):
            replay.prepare(self.args())
        self.assertFalse((self.directory / "prepared").exists())

    def test_inline_weight_graph_is_rejected_before_reading_onnx(self):
        with self.graph.open("r+b") as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError):
            replay.prepare(self.args())
        self.read_io.assert_not_called()
        self.assertFalse((self.directory / "prepared").exists())

    def test_graph_override_and_request_windows_are_preserved(self):
        plan = self.read_plan(self.args("--graph", str(self.graph), "--min-seconds", "30", "--max-seconds", "180", "--min-requests", "2000"))
        self.assertEqual(plan["graph"]["path"], str(self.graph))
        self.assertEqual(plan["replay"]["min_seconds"], 30)
        self.assertEqual(plan["replay"]["max_seconds"], 180)
        self.assertEqual(plan["replay"]["min_requests"], 2000)
        self.assertNotIn("runtime_env", plan)
        self.assertEqual(sorted(path.name for path in (self.directory / "prepared").iterdir()), ["inputs.bin", "plan.json"])


if __name__ == "__main__":
    unittest.main()
