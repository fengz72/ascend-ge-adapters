import importlib.util
import unittest
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "startlux-decision-v1.1-0.8b-instant"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision_folder():
    """Hermetic decision_config.json + config.json source folder."""
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


class TestDecoderPrimitives(unittest.TestCase):
    def setUp(self):
        self.adapter = load("startlux_npu_adapter", "model/startlux_npu_adapter.py")
        self._decision, self.decision_folder = _decision_folder()
        self.addCleanup(self._decision.cleanup)

    def test_decision_config_reads_the_folder(self):
        decision = self.adapter._decision_config(self.decision_folder)
        self.assertEqual(decision["letter_token_ids"], list(range(32, 58)))
        self.assertEqual(decision["vocab_size"], 300)

    def test_decision_config_rejects_a_short_letter_table(self):
        import json
        (self.decision_folder / "decision_config.json").write_text(
            json.dumps({"letter_token_ids": list(range(32, 57))}))
        with self.assertRaises(ValueError):
            self.adapter._decision_config(self.decision_folder)

    def test_softplus_matches_functional(self):
        x = torch.tensor([-100.0, -88.0, -20.0, -1.0, 0.0, 1.0, 19.999, 20.0, 20.001, 50.0, 100.0])
        actual = self.adapter._softplus(x)
        reference = torch.nn.functional.softplus(x)
        self.assertLessEqual(float((actual - reference).abs().max()), 1e-30)
        x = torch.linspace(-30.0, 30.0, 401)
        self.assertLessEqual(float((self.adapter._softplus(x)
                                    - torch.nn.functional.softplus(x)).abs().max()), 1e-7)

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



if __name__ == "__main__":
    unittest.main()
