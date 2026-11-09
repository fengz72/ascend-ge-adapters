import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "startlux-decision-v1.1-0.8b-instant"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision_folder():
    """Hermetic decision_config.json + config.json source folder for describe."""
    import json
    import os
    import tempfile
    directory = tempfile.TemporaryDirectory(
        prefix="startlux-decision-", dir=os.environ.get("TMPDIR") or None)
    folder = Path(directory.name)
    (folder / "decision_config.json").write_text(json.dumps({
        "letter_token_ids": list(range(32, 58)),
        "temperature_by_type": {"noul": 1.0411, "choice": 1.4364, "score": 1.4021}}))
    (folder / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 300}}))
    return directory, folder


class TestP3Adapter(unittest.TestCase):
    def setUp(self):
        self.adapter = load("startlux_npu_adapter", "model/startlux_npu_adapter.py")
        self._decision, self.decision_folder = _decision_folder()
        self.addCleanup(self._decision.cleanup)

    def config(self, **override):
        params = {"max_seq_len": 2048, "compiled_seq_len": 256, "chunk_size": 64,
                  "max_batch": 4}
        params.update(override)
        return {"model": {"params": params},
                "source": {"ref": str(self.decision_folder)}}

    def test_softplus_matches_functional(self):
        x = torch.tensor([-100.0, -88.0, -20.0, -1.0, 0.0, 1.0, 19.999, 20.0, 20.001, 50.0, 100.0])
        actual = self.adapter._softplus(x)
        reference = torch.nn.functional.softplus(x)
        self.assertLessEqual(float((actual - reference).abs().max()), 1e-30)
        x = torch.linspace(-30.0, 30.0, 401)
        self.assertLessEqual(float((self.adapter._softplus(x)
                                    - torch.nn.functional.softplus(x)).abs().max()), 1e-7)

    def test_rope_matches_hf_rotary(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

        for head_dim, partial, theta, section in ((256, 0.25, 1e7, [11, 11, 10]), (32, 0.5, 10000.0, [3, 3, 2])):
            cfg = Qwen3_5TextConfig(hidden_size=64, num_attention_heads=2,
                                    num_key_value_heads=1, head_dim=head_dim,
                                    rope_parameters={"rope_type": "default", "rope_theta": theta,
                                                     "partial_rotary_factor": partial,
                                                     "mrope_section": section})
            rotary = Qwen3_5TextRotaryEmbedding(cfg)
            positions = torch.arange(300).view(1, -1)
            reference_cos, reference_sin = rotary(torch.zeros(1, 300, 64), positions)
            inv_freq = 1.0 / (theta ** (torch.arange(0, int(head_dim * partial), 2,
                                                      dtype=torch.float32) / int(head_dim * partial)))
            cos, sin = self.adapter.rope_cos_sin(positions, inv_freq, torch.float32)
            self.assertTrue(torch.equal(reference_cos[0], cos[0]), f"cos mismatch head_dim={head_dim}")
            self.assertTrue(torch.equal(reference_sin[0], sin[0]), f"sin mismatch head_dim={head_dim}")

    def test_describe_report_contract(self):
        report = self.adapter.describe(self.config())
        self.assertIn("reference_params", report)
        self.assertEqual(report["reference_params"]["letter_token_ids"], list(range(32, 58)))
        self.assertEqual(report["reference_params"]["vocab_size"], 300)
        self.assertEqual(set(report["reference_params"]["temperature_by_type"]),
                         {"choice", "score", "noul"})
        self.assertTrue(str(self.decision_folder / "decision_config.json")
                        in report["referenced_files"])
        self.assertFalse(report["global_patch"])
        contract = report["readout_contract"]
        self.assertEqual(contract["bounds"]["max_batch"], 4)
        self.assertEqual(contract["bounds"]["max_seq_len"], 2048)
        self.assertEqual(contract["bounds"]["vocab_size"], 300)
        self.assertTrue(any("native ChunkGatedDeltaRule GDR" in change for change in report["changes"]))
        self.assertFalse(any("pure-Torch chunk GDR" in change for change in report["changes"]))
        self.assertEqual(report["required_dependencies"], ["chunk_gated_delta_rule_proto"])
        self.assertIn("native_gdr", report)
        self.assertIn("dynamic", report)

    def test_describe_reads_source_ref_env_var(self):
        import os
        os.environ["STARTLUX_TEST_REF"] = str(self.decision_folder)
        try:
            report = self.adapter.describe({"model": {"params": {}},
                                            "source": {"ref": "${STARTLUX_TEST_REF}"}})
            self.assertEqual(report["reference_params"]["letter_token_ids"], list(range(32, 58)))
        finally:
            del os.environ["STARTLUX_TEST_REF"]
        with self.assertRaises(ValueError):
            self.adapter.describe({"model": {"params": {}}, "source": {"ref": "${STARTLUX_MISSING_VAR_X}"}})
        with self.assertRaises(ValueError):
            self.adapter.describe({"model": {"params": {}}})

    def test_encode_dynamic_contract(self):
        params = self.adapter._params(self.config())
        good = {"id": "x-b2", "samples": 2, "payload": {
            "input_ids": [[5, 6, 7], [8, 9, 10, 11]],
            "position_ids": [[0, 1, 2], [0, 1, 2, 3]],
            "actual_seq_lengths": [3, 4]}}
        encoded = self.adapter._encode_dynamic(good, params, 300)
        self.assertEqual(encoded["input_ids"].shape, (2, 4))
        self.assertEqual(encoded["input_ids"].dtype, np.int64)
        self.assertEqual(list(encoded["input_ids"][0]), [5, 6, 7, 0])
        self.assertEqual(list(encoded["position_ids"][1]), [0, 1, 2, 3])
        self.assertEqual(encoded["actual_seq_lengths"].tolist(), [3, 4])
        b4 = {"id": "x-b4", "samples": 4, "payload": {
            "input_ids": [[1], [2, 3], [4, 5, 6], [7, 8, 9, 10]],
            "position_ids": [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]],
            "actual_seq_lengths": [1, 2, 3, 4]}}
        self.assertEqual(self.adapter._encode_dynamic(b4, params, 300)["input_ids"].shape, (4, 4))
        # 1D payload means one row
        one_d = {"id": "x-b1", "samples": 1, "payload": {
            "input_ids": [5, 6, 7], "position_ids": [0, 1, 2], "actual_seq_lengths": [3]}}
        self.assertEqual(self.adapter._encode_dynamic(one_d, params, 300)["input_ids"].shape, (1, 3))
        with self.assertRaises(ValueError):
            self.adapter._encode_dynamic({**good, "samples": 1}, params, 300)
        with self.assertRaises(ValueError):
            self.adapter._encode_dynamic({**good, "samples": 5, "payload": {
                **good["payload"], "input_ids": good["payload"]["input_ids"] + [[1]]}}, params, 300)
        with self.assertRaises(ValueError):
            self.adapter._encode_dynamic({"id": "x-r", "samples": 2, "payload": {
                "input_ids": [[5, 6, 7], [8, 9, 10, 11]],
                "position_ids": [[0, 1, 2], [0, 1, 2, 3]],
                "actual_seq_lengths": [3, 5]}}, params, 300)
        with self.assertRaises(ValueError):
            self.adapter._encode_dynamic({"id": "x-r", "samples": 2, "payload": {
                "input_ids": [[5, 6, 7], [8, 9, 10, 11]],
                "position_ids": [[0, 1, 2], [0, 2, 1, 3]],
                "actual_seq_lengths": [3, 4]}}, params, 300)
        with self.assertRaises(ValueError):
            self.adapter._encode_dynamic({"id": "x-r", "samples": 2, "payload": {
                "input_ids": [[5, 6, 7], [8, 9, 10, 999]],
                "position_ids": [[0, 1, 2], [0, 1, 2, 3]],
                "actual_seq_lengths": [3, 4]}}, params, 300)
        with self.assertRaises(ValueError):
            self.adapter._encode_dynamic({"id": "x-r", "samples": 1, "payload": {
                "input_ids": [[5, 6, 7]][0], "position_ids": [0.0, 1.0, 2.0],
                "actual_seq_lengths": [3]}}, params, 300)
        with self.assertRaises(ValueError):
            self.adapter._encode_dynamic({"id": "x-r", "samples": 1, "payload": {
                "input_ids": [[5, 6, 7]][0], "position_ids": [False, True, False],
                "actual_seq_lengths": [3]}}, params, 300)
        with self.assertRaises(ValueError):
            self.adapter._encode_dynamic({"id": "x-r", "samples": 2, "payload": {
                "input_ids": [[5, 6, 7], [8, 9, 10, True]],
                "position_ids": [[0, 1, 2], [0, 1, 2, 3]],
                "actual_seq_lengths": [3, 4]}}, params, 300)
        with self.assertRaises(ValueError):
            self.adapter._encode_dynamic({"id": "x-r", "samples": 2, "payload": {
                "input_ids": [[5, 6, 7], [8, 9, 10, 11.5]],
                "position_ids": [[0, 1, 2], [0, 1, 2, 3]],
                "actual_seq_lengths": [3, 4]}}, params, 300)
        with self.assertRaises(ValueError):
            self.adapter._encode_dynamic({"id": "x-r", "samples": 2, "payload": {
                "input_ids": [[5, 6, 7], [8, 9, 10, 11]],
                "position_ids": [[0, 1, 2], [0, 1, 2, 3]],
                "actual_seq_lengths": [4, 4]}}, params, 300)
        long = list(range(2049))
        with self.assertRaises(ValueError):
            self.adapter._encode_dynamic({"id": "x-r", "samples": 1, "payload": {
                "input_ids": [long], "position_ids": [list(range(2049))],
                "actual_seq_lengths": [2049]}}, params, 300)

    def test_causal_conv4_matches_official_conv1d(self):
        """The fp32 Conv2D node form keeps the official compute order
        (fp32 conv -> cast to input dtype -> SiLU) on transposed views."""
        import torch.nn.functional as F
        channels, seq = 64, 37
        generator = torch.Generator().manual_seed(2026)
        x = torch.randn(2, seq, channels, generator=generator).transpose(1, 2)
        weight = torch.randn(channels, 1, 4, generator=generator) * 0.2
        reference32 = F.conv1d(x.float(), weight.float(), groups=channels,
                               padding=3)[:, :, :seq]
        for dtype, bound in ((torch.float32, 1e-6), (torch.bfloat16, 1e-2)):
            out = self.adapter.causal_conv4(x.to(dtype), weight.to(dtype))
            reference = F.silu(reference32.to(dtype))
            rel = float((out.float() - reference.float()).norm()
                        / reference.float().norm())
            self.assertLess(rel, bound, f"{dtype}: rel {rel}")

    def test_causal_conv4_is_causal(self):
        """Positions only ever see tokens up to t (left zero padding)."""
        generator = torch.Generator().manual_seed(2027)
        channels, seq = 48, 40
        x = torch.randn(2, seq, channels, generator=generator).transpose(1, 2)
        weight = torch.randn(channels, 1, 4, generator=generator) * 0.2
        perturbed = x.clone()
        perturbed[:, :, 20:] += 5.0
        out = self.adapter.causal_conv4(x, weight)
        out_perturbed = self.adapter.causal_conv4(perturbed, weight)
        self.assertTrue(torch.equal(out[:, :, :17], out_perturbed[:, :, :17]))
        self.assertFalse(torch.equal(out[:, :, 17:], out_perturbed[:, :, 17:]))

    def test_rms_norm_fused_formula_matches_official(self):
        """The graph-leg formula of _rms_norm is the official compute domain."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        torch.manual_seed(2028)
        norm = Qwen3_5RMSNorm(64).eval()
        x = torch.randn(3, 7, 64).to(torch.bfloat16)
        with torch.inference_mode():
            official = norm(x)
            x32 = x.float()
            # NpuRmsNorm with a direct bf16 input: fp32 reduction over the
            # bf16 values, fp32 gamma, output in the input dtype
            mine = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True)
                                      + float(norm.eps))
                    * (1.0 + norm.weight.float())).to(x.dtype)
        rel = float((mine.float() - official.float()).norm() / official.float().norm())
        self.assertLess(rel, 1e-6, f"rms fused formula drift {rel}")

    def test_rms_norm_gated_fused_formula_matches_official(self):
        """The graph-leg formula of _rms_norm_gated keeps the official order."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

        torch.manual_seed(2029)
        norm = Qwen3_5RMSNormGated(64).eval()
        x = torch.randn(5, 64).to(torch.bfloat16)
        gate = torch.randn(5, 64).to(torch.bfloat16)
        with torch.inference_mode():
            official = norm(x, gate)
            import torch.nn.functional as F
            x32 = x.float()
            n = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True)
                                  + float(norm.variance_epsilon))
            mine = (norm.weight * n.to(x.dtype) * F.silu(gate.float())).to(x.dtype)
        rel = float((mine.float() - official.float()).norm() / official.float().norm())
        self.assertLess(rel, 1e-6, f"rms gated fused formula drift {rel}")

    def test_gdr_l2norm_rms_equivalent_with_folded_scale(self):
        """The GDR q/k L2-norm as fused RmsNorm(gamma=1/sqrt(D), eps/D)
        reproduces the L2-normalized q/k exactly (the sqrt(D) of the
        mean-domain cancels in gamma); zero rows stay zero."""
        torch.manual_seed(41)
        dim = 128
        q = (torch.randn(2, 7, 4, dim) * 0.5).to(torch.bfloat16)
        k = (torch.randn(2, 7, 4, dim) * 0.5).to(torch.bfloat16)

        def l2_chain(x):
            x32 = x.float()
            return (x32 * torch.rsqrt((x32 * x32).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)

        def rms_equiv(x):
            x32 = x.float()
            y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6 / dim)
            return y.to(torch.bfloat16)

        q_l2, k_l2 = l2_chain(q).float(), l2_chain(k).float()
        gamma = (dim ** -0.5)
        q_r, k_r = rms_equiv(q).float() * gamma, rms_equiv(k).float() * gamma
        scores_old = torch.einsum("bhnd,bhmd->bhnm", q_l2, k_l2) * (dim ** -0.5)
        scores_new = torch.einsum("bhnd,bhmd->bhnm", q_r, k_r) * (dim ** -0.5)
        rel = float((scores_new - scores_old).norm() / scores_old.norm())
        self.assertLess(rel, 0.01, f"gamma-folded scores drift {rel}")
        zero = torch.zeros(1, 1, 1, dim, dtype=torch.bfloat16)
        self.assertEqual(float(rms_equiv(zero).abs().max()), 0.0)

    def test_runtime_decode_is_provenance_free(self):
        report = self.adapter.describe(self.config())
        handle = self.adapter.RuntimeHandle(self.config(), {
            "schema": "ge-io/1",
            "outputs": [{"name": "scores", "dtype": "float32", "shape": [1, 26]}]}, report)
        case = {"id": "q-r0", "samples": 1, "payload": {"question_type": "choice",
                                                        "golden_file": "x.json"}}
        row = handle.decode({"scores": np.zeros((1, 26), dtype=np.float32)}, case)
        self.assertEqual(row["alignment"], {"rows": ["q-r0"], "columns": list(range(32, 58))})
        self.assertNotIn("question_type", row["alignment"])
        self.assertNotIn("golden_file", row["alignment"])
        case2 = {"id": "q-b2", "samples": 2, "payload": {}}
        row2 = handle.decode({"scores": np.zeros((2, 26), dtype=np.float32)}, case2)
        self.assertEqual(row2["alignment"]["rows"], ["q-b2", "q-b2"])
        with self.assertRaises(ValueError):
            handle.decode({"scores": np.zeros((3, 26), dtype=np.float32)}, case2)


class TestNativeGdrContract(unittest.TestCase):
    """Contract of the single native chunk-GDR route (merged from the former
    test_startlux_native_adapter.py when the pure-Torch standard route was
    removed, 2026-11-02)."""

    def setUp(self):
        self.adapter = load("startlux_npu_adapter", "model/startlux_npu_adapter.py")
        self._decision, self.decision_folder = _decision_folder()
        self.addCleanup(self._decision.cleanup)

    def config(self, **override):
        params = {"max_seq_len": 2048, "compiled_seq_len": 256, "chunk_size": 64,
                  "max_batch": 4}
        params.update(override)
        return {"model": {"params": params},
                "source": {"ref": str(self.decision_folder)}}

    def test_describe_dynamic_contract(self):
        report = self.adapter.describe(self.config())
        self.assertTrue(any("native ChunkGatedDeltaRule GDR" in change for change in report["changes"]))
        self.assertFalse(any("pure-Torch chunk GDR" in change for change in report["changes"]))
        self.assertEqual(report["required_dependencies"], ["chunk_gated_delta_rule_proto"])
        dynamic = report["dynamic"]
        self.assertEqual(dynamic["axes"]["input_ids"], [-1, -1])
        self.assertEqual(dynamic["axes"]["position_ids"], [-1, -1])
        self.assertEqual(dynamic["axes"]["actual_seq_lengths"], [-1])
        self.assertEqual(dynamic["axes"]["scores"], [-1, 26])
        self.assertEqual(dynamic["max_batch"], 4)
        self.assertEqual(dynamic["max_seq_len"], 2048)
        self.assertIn("physical padded width", dynamic["length_semantics"])
        self.assertEqual(report["reference_params"]["letter_token_ids"], list(range(32, 58)))

    def test_prepare_runtime_uses_dynamic_encode(self):
        handle = self.adapter.prepare_runtime(self.config(), {"schema": "ge-io/1", "inputs": [
            {"name": "input_ids", "dtype": "int64", "shape": [-1, -1], "dynamic_dims": [0, 1]},
            {"name": "position_ids", "dtype": "int64", "shape": [-1, -1], "dynamic_dims": [0, 1]},
            {"name": "actual_seq_lengths", "dtype": "int32", "shape": [-1], "dynamic_dims": [0]}],
            "outputs": [{"name": "scores", "dtype": "float32", "shape": [-1, 26],
                         "dynamic_dims": [0]}]})
        self.assertIsInstance(handle, self.adapter.RuntimeHandle)
        case = {"id": "n-b2", "samples": 2, "payload": {
            "input_ids": [[5, 6, 7], [8, 9, 10, 11]],
            "position_ids": [[0, 1, 2], [0, 1, 2, 3]],
            "actual_seq_lengths": [3, 4]}}
        encoded = handle.encode(case)
        self.assertEqual(encoded["input_ids"].shape, (2, 4))
        self.assertEqual(encoded["input_ids"].dtype, np.int64)
        self.assertEqual(encoded["actual_seq_lengths"].tolist(), [3, 4])
        five = {"id": "n-b5", "samples": 5, "payload": {
            "input_ids": [[1]] * 5, "position_ids": [[0]] * 5,
            "actual_seq_lengths": [1] * 5}}
        with self.assertRaises(ValueError):
            handle.encode(five)

    def _mock_wrapper(self):
        import torch.nn as nn

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, position_ids, actual_seq_lengths):
                return torch.zeros((input_ids.shape[0], 26), dtype=torch.float32)

        return Wrapper().eval()

    def test_candidate_export_spec_is_dynamic(self):
        report = self.adapter.describe(self.config())
        handle = self.adapter.CandidateHandle(self._mock_wrapper(), self.config(), report)
        spec = handle.export_spec()
        self.assertEqual(spec["format"], "torch")
        by_name = {node["name"]: node for node in spec["io"]["inputs"]}
        self.assertEqual(by_name["input_ids"]["shape"], [-1, -1])
        self.assertEqual(by_name["input_ids"]["dynamic_dims"], [0, 1])
        self.assertEqual(by_name["position_ids"]["shape"], [-1, -1])
        self.assertEqual(by_name["position_ids"]["dynamic_dims"], [0, 1])
        self.assertEqual(by_name["actual_seq_lengths"]["shape"], [-1])
        self.assertEqual(by_name["actual_seq_lengths"]["dynamic_dims"], [0])
        self.assertEqual(spec["io"]["outputs"][0]["shape"], [-1, 26])
        self.assertEqual(spec["io"]["outputs"][0]["dynamic_dims"], [0])
        case = {"id": "n-b2", "samples": 2, "payload": {
            "input_ids": [[5, 6, 7], [8, 9, 10, 11]],
            "position_ids": [[0, 1, 2], [0, 1, 2, 3]],
            "actual_seq_lengths": [3, 4]}}
        inputs = handle.encode(case)
        self.assertEqual(inputs["input_ids"].shape, (2, 4))
        with torch.inference_mode():
            outputs = handle.run(inputs)
        self.assertEqual(outputs["scores"].shape, (2, 26))
        self.assertEqual(handle.output_capacities({"samples": 4}), {"scores": 4 * 26 * 4})

    def _candidate(self, **params):
        return self.adapter.CandidateHandle(self._mock_wrapper(), self.config(**params),
                                             self.adapter.describe(self.config(**params)))

    def test_export_spec_carries_deterministic_b2_s65_example(self):
        from core.contracts import validate_io
        from core.exporters.torchair import _example_values

        handle = self._candidate()
        spec = handle.export_spec()
        example = spec["example_inputs"]
        self.assertEqual(set(example), {"input_ids", "position_ids", "actual_seq_lengths"})
        self.assertEqual(tuple(example["input_ids"].shape), (2, 65))
        self.assertEqual(tuple(example["position_ids"].shape), (2, 65))
        self.assertEqual(tuple(example["actual_seq_lengths"].shape), (2,))
        self.assertEqual(example["input_ids"].dtype, torch.int64)
        self.assertEqual(example["position_ids"].dtype, torch.int64)
        self.assertEqual(example["actual_seq_lengths"].dtype, torch.int32)
        self.assertTrue(torch.equal(example["input_ids"],
                                    torch.zeros((2, 65), dtype=torch.int64)))
        self.assertTrue(torch.equal(
            example["position_ids"], torch.arange(65, dtype=torch.int64).repeat(2, 1)))
        self.assertTrue(torch.equal(example["actual_seq_lengths"],
                                     torch.full((2,), 65, dtype=torch.int32)))
        again = handle.export_spec()["example_inputs"]
        for name in example:
            self.assertTrue(torch.equal(example[name], again[name]),
                            f"example {name} is not deterministic")
        values = _example_values(torch, validate_io(spec["io"]), example)
        self.assertEqual([tuple(v.shape) for v in values], [(2, 65), (2, 65), (2,)])
        io_names = [node["name"] for node in spec["io"]["inputs"]]
        self.assertEqual(io_names, ["input_ids", "position_ids", "actual_seq_lengths"])
        self.assertEqual(spec["io"]["outputs"][0]["shape"], [-1, 26])
        self.assertEqual(spec["io"]["outputs"][0]["dynamic_dims"], [0])

    def test_export_spec_static_axes_when_caps_are_one(self):
        from core.contracts import validate_io
        from core.exporters.torchair import _example_values

        handle = self._candidate(max_batch=1)
        spec = handle.export_spec()
        by_name = {node["name"]: node for node in spec["io"]["inputs"]}
        self.assertEqual(by_name["input_ids"]["shape"], [1, -1])
        self.assertEqual(by_name["input_ids"]["dynamic_dims"], [1])
        self.assertEqual(by_name["actual_seq_lengths"]["shape"], [1])
        self.assertEqual(spec["io"]["outputs"][0]["shape"], [1, 26])
        self.assertEqual(spec["io"]["outputs"][0]["dynamic_dims"], [])
        example = spec["example_inputs"]
        self.assertEqual(tuple(example["input_ids"].shape), (1, 65))
        _example_values(torch, validate_io(spec["io"]), example)
        report = self.adapter.describe(self.config(max_batch=1))
        self.assertEqual(report["dynamic"]["axes"]["input_ids"], [1, -1])
        self.assertEqual(report["dynamic"]["axes"]["scores"], [1, 26])

        handle = self._candidate(max_seq_len=1, compiled_seq_len=1)
        spec = handle.export_spec()
        by_name = {node["name"]: node for node in spec["io"]["inputs"]}
        self.assertEqual(by_name["input_ids"]["shape"], [-1, 1])
        self.assertEqual(by_name["input_ids"]["dynamic_dims"], [0])
        example = spec["example_inputs"]
        self.assertEqual(tuple(example["input_ids"].shape), (2, 1))
        _example_values(torch, validate_io(spec["io"]), example)
        report = self.adapter.describe(self.config(max_seq_len=1, compiled_seq_len=1))
        self.assertEqual(report["dynamic"]["axes"]["input_ids"], [-1, 1])
        self.assertEqual(report["dynamic"]["axes"]["position_ids"], [-1, 1])


if __name__ == "__main__":
    unittest.main()
