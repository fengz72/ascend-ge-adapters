"""Concrete dtype contracts across configuration and native ONNX entry points."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper, TensorProto
import yaml

from core.config import ROOT, load_authoring
from core import native_onnx


class DtypeConfiguration(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def config(self, kind="onnx"):
        source = {"onnx": {"kind": "onnx", "ref": "/not-loaded/source.onnx"},
                  "torch": {"kind": "torch", "factory": "models.tiny.adapter:create_model"},
                  "transformers": {"kind": "transformers", "ref": "/not-loaded/weights", "loader": "AutoModel"}}[kind]
        model = {"name": "dtype-contract"}
        if kind != "onnx":
            model["adapter_file"] = "models/tiny/adapter.py"
        return {"model": model, "source": source, "target": "Ascend910_9382"}

    def load(self, authored):
        path = self.directory / "model.yaml"
        path.write_text(yaml.safe_dump(authored))
        return load_authoring(path)

    def test_all_source_kinds_default_to_concrete_float32(self):
        for kind in ("onnx", "torch", "transformers"):
            with self.subTest(kind=kind):
                config = self.load(self.config(kind))
                self.assertEqual(config["build"]["dtype"], "float32")
                self.assertEqual(config["reference"]["dtype"], "float32")

    def test_both_sections_reject_legacy_automatic_and_invalid_types(self):
        for kind in ("onnx", "torch", "transformers"):
            for section in ("build", "reference"):
                for value in ("preserve", "auto", "float", "double", "complex64", "object",
                              "datetime64", "float128", "", None, 32, []):
                    with self.subTest(kind=kind, section=section, dtype=value):
                        authored = self.config(kind)
                        authored[section] = {"dtype": value}
                        with self.assertRaisesRegex(ValueError, rf"{section}\.dtype"):
                            self.load(authored)

    def test_both_sections_accept_all_supported_concrete_types(self):
        for kind in ("onnx", "torch", "transformers"):
            for dtype in ("float16", "float32", "bfloat16", "float64"):
                with self.subTest(kind=kind, dtype=dtype):
                    authored = self.config(kind)
                    authored.update(build={"dtype": dtype}, reference={"dtype": dtype})
                    config = self.load(authored)
                    self.assertEqual(config["build"]["dtype"], dtype)
                    self.assertEqual(config["reference"]["dtype"], dtype)

    def test_integer_and_boolean_dtypes_follow_source_capabilities(self):
        for dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"):
            for section in ("build", "reference"):
                with self.subTest(dtype=dtype, section=section):
                    authored = self.config("onnx")
                    authored[section] = {"dtype": dtype}
                    self.assertEqual(self.load(authored)[section]["dtype"], dtype)
                    for kind in ("torch", "transformers"):
                        authored = self.config(kind)
                        authored[section] = {"dtype": dtype}
                        with self.assertRaisesRegex(ValueError, "cannot cast Torch model parameters"):
                            self.load(authored)

    def test_inactive_platform_dtypes_are_also_validated(self):
        for section in ("build", "reference"):
            with self.subTest(section=section):
                authored = self.config()
                authored["platforms"] = {"a3": {}, "a5": {section: {"dtype": "preserve"}}}
                with self.assertRaisesRegex(ValueError, rf"{section}\.dtype"):
                    self.load(authored)

    def test_removed_chip_override_is_rejected(self):
        authored = self.config()
        authored["chips"] = {"Ascend910_9382": {"build": {"dtype": "float16"}}}
        with self.assertRaisesRegex(ValueError, "unknown=.*chips"):
            self.load(authored)

    def test_explicit_dtype_overrides_are_retained(self):
        authored = self.config()
        authored.update(build={"dtype": "float16"}, reference={"dtype": "float32"},
                        platforms={"a3": {"build": {"dtype": "float32"},
                                          "reference": {"dtype": "bfloat16"}}})
        config = self.load(authored)
        self.assertEqual(config["build"]["dtype"], "float32")
        self.assertEqual(config["reference"]["dtype"], "bfloat16")
        qwen = load_authoring(ROOT / "models/qwen2.5-0.5b/config/model.yaml",
                              target="Ascend910_9382", source_paths=False)
        self.assertEqual(qwen["build"]["dtype"], "float16")
        self.assertEqual(qwen["reference"]["dtype"], "float32")


class NativeOnnxDtypes(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def model(self, floating=TensorProto.FLOAT):
        names = [("features", floating), ("ids", TensorProto.INT64), ("mask", TensorProto.BOOL)]
        graph = helper.make_graph([helper.make_node("Identity", [n], [n+"_out"]) for n, _ in names],
            "typed-inputs", [helper.make_tensor_value_info(n, dtype, [1, 2]) for n, dtype in names],
            [helper.make_tensor_value_info(n+"_out", dtype, [1, 2]) for n, dtype in names])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        onnx.checker.check_model(model)
        path = self.directory / "source.onnx"
        onnx.save(model, path)
        return str(path)

    def test_concrete_types_leave_other_input_types_unchanged(self):
        for onnx_type, dtype in ((TensorProto.FLOAT, "float32"), (TensorProto.FLOAT16, "float16"),
                                 (TensorProto.DOUBLE, "float64"), (TensorProto.INT8, "int8"),
                                 (TensorProto.BOOL, "bool")):
            for prepare in (native_onnx.prepare_reference, native_onnx.prepare_candidate):
                with self.subTest(dtype=dtype, prepare=prepare.__name__):
                    config = {"build": {"dtype": dtype}, "reference": {"dtype": dtype}}
                    handle = prepare(self.model(onnx_type), config)
                    arrays = handle.encode({"payload": {"features": [[1, 2]],
                                                        "ids": [[7, 9]], "mask": [[True, False]]}})
                    self.assertEqual(arrays["features"].dtype, np.dtype(dtype))
                    self.assertEqual(arrays["ids"].dtype, np.dtype("int64"))
                    self.assertEqual(arrays["mask"].dtype, np.dtype("bool"))
                    actual = handle.run(arrays)
                    for name, array in arrays.items():
                        np.testing.assert_array_equal(actual[name+"_out"], array)
                        self.assertEqual(actual[name+"_out"].dtype, array.dtype)

    def test_integer_declaration_must_match_an_actual_input(self):
        path = self.model()
        for section, prepare in (("reference", native_onnx.prepare_reference),
                                 ("build", native_onnx.prepare_candidate)):
            config = {"build": {"dtype": "float32"}, "reference": {"dtype": "float32"}}
            config[section]["dtype"] = "int8"
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, rf"no input matching {section}\.dtype=int8"):
                prepare(path, config)

    def test_integer_declaration_keeps_floating_auxiliary_inputs(self):
        config = {"build": {"dtype": "int64"}, "reference": {"dtype": "int64"}}
        for prepare in (native_onnx.prepare_reference, native_onnx.prepare_candidate):
            with self.subTest(prepare=prepare.__name__):
                handle = prepare(self.model(), config)
                arrays = handle.encode({"payload": {"features": [[1.5, 2.5]],
                                                    "ids": [[7, 9]], "mask": [[True, False]]}})
                actual = handle.run(arrays)
                self.assertEqual(actual["features_out"].dtype, np.dtype("float32"))
                self.assertEqual(actual["ids_out"].dtype, np.dtype("int64"))
                self.assertEqual(actual["mask_out"].dtype, np.dtype("bool"))
                for name, array in arrays.items():
                    np.testing.assert_array_equal(actual[name+"_out"], array)

    def test_float_mismatch_reports_the_corresponding_section(self):
        path = self.model()
        for section, prepare in (("reference", native_onnx.prepare_reference),
                                 ("build", native_onnx.prepare_candidate)):
            config = {"build": {"dtype": "float32"}, "reference": {"dtype": "float32"}}
            config[section]["dtype"] = "float16"
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, rf"{section}\.dtype=float16"):
                prepare(path, config)

    def test_legacy_dtype_is_rejected_before_reading_an_onnx_model(self):
        for section, prepare in (("reference", native_onnx.prepare_reference),
                                 ("build", native_onnx.prepare_candidate)):
            config = {"build": {"dtype": "float32"}, "reference": {"dtype": "float32"}}
            config[section]["dtype"] = "preserve"
            with self.subTest(section=section), patch("onnx.load") as load:
                with self.assertRaisesRegex(ValueError, rf"{section}\.dtype"):
                    prepare("/not-loaded/source.onnx", config)
                load.assert_not_called()
