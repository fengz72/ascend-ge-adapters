"""Offline candidate scope and FP32 semantic checks; no GE/NPU execution."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMPARISON = HERE.parent / "profile-analysis/comparison/add-layernorm-comparison.json"
NUMERIC_EVIDENCE = []


class AttentionV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temporary.name) / "attention-scope"
        compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
        assert compiler, "local C++17 compiler is required"
        result = subprocess.run([compiler, "-std=c++17", "-O2", str(HERE / "test_attention_scope.cpp"), "-o", str(cls.binary)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def classify(self, names):
        result = subprocess.run([str(self.binary)], input="\n".join(names) + "\n", text=True, capture_output=True, check=True)
        return [int(x) for x in result.stdout.splitlines()]

    def test_actual_48_profile_pairs_select_exactly_24_attention_edges(self):
        report = json.loads(COMPARISON.read_text())
        names = [row["layernorm"] for row in report["pairs"]]
        selected = self.classify(names)
        self.assertEqual(sum(selected), 24)
        for row, accepted in zip(report["pairs"], selected):
            self.assertEqual(accepted, int("/attention/output/" in row["layernorm"]))
            edge = row["source_add_to_layernorm_reduce_edge"]
            self.assertEqual(edge[0], edge[1])

    def test_scope_rejects_unrelated_and_out_of_range_names(self):
        suffix = "/attention/output/LayerNorm/Add_1/LayerNorm"
        positive = [f"/encoder/backbone/encoder/layer.{i}" + suffix for i in (0, 9, 10, 23)]
        negative = [f"/encoder/backbone/encoder/layer.{i}" + suffix for i in (24, -1, "01", "", "x")]
        negative += ["prefix" + positive[0], positive[0].replace("attention/output", "output"),
                     "/encoder/backbone/embeddings/LayerNorm/fused", positive[0].replace("LayerNorm/Add_1/LayerNorm", "LayerNorm/")]
        self.assertEqual(self.classify(positive + negative), [1] * len(positive) + [0] * len(negative))

    def test_fusion_math_graph_safety_and_preload_lifecycle_are_unchanged(self):
        original = (ROOT / "work/bge-opt-20261026/add_layernorm_pass.cpp").read_text()
        wanted = '#include "attention_scope.h"\n' + original
        wanted = wanted.replace('        name.find("/encoder/backbone/encoder/layer.") == std::string::npos ||', '        !IsAttentionOutputLayerNorm(name) ||')
        wanted = wanted.replace('REGISTER_CUSTOM_PASS("BgeResidualAddLayerNormFP32")', 'REGISTER_CUSTOM_PASS("BgeAttentionOutputAddLayerNormFP32V3")')
        wanted = wanted.replace('[BGE_FUSION] AddLayerNorm replacements=', '[BGE_FUSION_V3] attention/output AddLayerNorm replacements=').replace('[BGE_FUSION] failed:', '[BGE_FUSION_V3] failed:')
        self.assertEqual((HERE / "add_layernorm_attention_pass.cpp").read_text(), wanted)
        self.assertEqual((HERE / "preload_scope_filter.h").read_bytes(), (ROOT / "work/bge-opt-20261026/preload_scope_filter.h").read_bytes())
        old = (ROOT / "work/bge-opt-20261026/add_layernorm_pass_v2.cpp").read_text().split('namespace {', 1)[1]
        new = (HERE / "add_layernorm_attention_v3.cpp").read_text().split('namespace {', 1)[1]
        self.assertEqual(new, old)

    def test_float32_semantics_against_double_reference(self):
        # Validate residual-add FP32 rounding and LN epsilon/axes semantics.
        # This is not execution of CANN AddLayerNorm or the full model.
        rng = np.random.default_rng(20261026)
        for sequence, scale in ((6, 1.0), (8, 1e-7), (13, 10.0)):
            x = (rng.normal(size=(1, sequence, 1024)) * scale).astype(np.float32)
            y = (rng.normal(size=x.shape) * scale).astype(np.float32)
            gamma = rng.uniform(.5, 1.5, size=(1024,)).astype(np.float32)
            beta = rng.normal(size=(1024,)).astype(np.float32)
            residual = np.add(x, y, dtype=np.float32)
            centered = residual - residual.mean(axis=-1, keepdims=True, dtype=np.float32)
            variance = np.square(centered).mean(axis=-1, keepdims=True, dtype=np.float32)
            actual = centered / np.sqrt(variance + np.float32(1e-12)) * gamma + beta
            ref = residual.astype(np.float64)
            ref = (ref - ref.mean(axis=-1, keepdims=True)) / np.sqrt(ref.var(axis=-1, keepdims=True) + float(np.float32(1e-12)))
            ref = ref * gamma.astype(np.float64) + beta.astype(np.float64)
            error = float(np.max(np.abs(actual.astype(np.float64) - ref)))
            NUMERIC_EVIDENCE.append({"sequence": sequence, "scale": scale, "max_abs_vs_fp64": error})
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.isfinite(actual).all())
            self.assertLess(error, 1e-5)

    def test_small_variance_distinguishes_wrong_epsilon(self):
        x = np.linspace(-1e-7, 1e-7, 1024, dtype=np.float32)
        variance = x.var(dtype=np.float32)
        correct = x / np.sqrt(variance + np.float32(1e-12))
        changed = x / np.sqrt(variance + np.float32(1e-5))
        self.assertGreater(float(np.max(np.abs(correct - changed))), .09)


if __name__ == "__main__": unittest.main()
