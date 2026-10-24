"""Shared model-location contracts for ONNX files and Transformers directories."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import onnx
from onnx import helper, TensorProto
import yaml

from core.config import ROOT, load_authoring
from core.sources import load_source


class SourceReferences(unittest.TestCase):
    def setUp(self):
        (ROOT / "work").mkdir(exist_ok=True)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory(dir=ROOT / "work")))
        self.config_path = self.directory / "configs/model.yaml"
        self.config_path.parent.mkdir()

    def authored(self, kind="onnx", ref="work/source.onnx"):
        model = {"name": "source-ref-contract"}
        source = {"kind": kind, "ref": ref}
        if kind == "transformers":
            model["adapter_file"] = "models/tiny/adapter.py"
            source["loader"] = "AutoModelForCausalLM"
        return {"model": model, "source": source, "target": "Ascend910_9382"}

    def load(self, authored, **kwargs):
        self.config_path.write_text(yaml.safe_dump(authored))
        return load_authoring(self.config_path, **kwargs)

    def onnx_file(self):
        path = self.directory / "source.onnx"
        graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "source-ref",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        onnx.save(model, path)
        return path

    def test_onnx_ref_is_relative_to_project_root_and_loads_original_file(self):
        path = self.onnx_file()
        config = self.load(self.authored(ref=str(path.relative_to(ROOT))))
        self.assertEqual(config["source"]["ref"], str(path))
        self.assertNotIn("path", config["source"])
        self.assertEqual(load_source(config), str(path))
        self.assertEqual(load_source(config, reference=True), str(path))

    def test_environment_reference_resolution_is_shared(self):
        for kind in ("onnx", "transformers"):
            with self.subTest(kind=kind), patch.dict(os.environ, {
                "SOURCE_REF_TEST": str(self.directory.relative_to(ROOT))
            }):
                config = self.load(self.authored(kind, "${SOURCE_REF_TEST}/model"))
                self.assertEqual(config["source"]["ref"], str(self.directory / "model"))

    def test_ref_must_be_a_nonempty_string(self):
        for kind in ("onnx", "transformers"):
            authored = self.authored(kind)
            del authored["source"]["ref"]
            with self.subTest(kind=kind, ref="missing"), self.assertRaisesRegex(ValueError, "source.ref"):
                self.load(authored)
            for ref in (None, "", " ", 7, [], {}):
                with self.subTest(kind=kind, ref=ref), self.assertRaisesRegex(ValueError, "source.ref"):
                    self.load(self.authored(kind, ref))

    def test_old_source_path_is_rejected(self):
        for kind in ("onnx", "transformers"):
            authored = self.authored(kind)
            authored["source"]["path"] = "old-source-path"
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, r"source: unknown=.*path"):
                self.load(authored)

    def test_runtime_only_configuration_does_not_expand_source_environment(self):
        with patch.dict(os.environ):
            os.environ.pop("SOURCE_REF_TEST", None)
            for kind in ("onnx", "transformers"):
                with self.subTest(kind=kind):
                    authored = self.authored(kind, "${SOURCE_REF_TEST}/model")
                    authored["source"]["files"] = ["${SOURCE_REF_TEST}/side-file"]
                    config = self.load(authored, source_paths=False)
                    self.assertEqual(config["source"]["ref"], "${SOURCE_REF_TEST}/model")
                    self.assertEqual(config["source"]["files"], authored["source"]["files"])
                    with self.assertRaisesRegex(ValueError, "source.ref: environment variable SOURCE_REF_TEST"):
                        self.load(authored)

    def test_transformers_loader_receives_the_resolved_ref(self):
        import torch
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        factory = Mock(return_value=model)
        module = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=factory))
        config = self.load(self.authored("transformers", str(self.directory.relative_to(ROOT))))
        with patch.dict(sys.modules, {"transformers": module}):
            self.assertIs(load_source(config, reference=True), model)
        factory.assert_called_once_with(str(self.directory), dtype=torch.float32, local_files_only=True)
        model.to.assert_called_once_with(device="cpu", dtype=torch.float32)

    def test_missing_onnx_ref_file_is_reported_before_execution(self):
        path = self.directory / "missing.onnx"
        config = self.load(self.authored(ref=str(path)))
        with self.assertRaises(FileNotFoundError) as failure:
            load_source(config)
        self.assertEqual(str(failure.exception), str(path))
