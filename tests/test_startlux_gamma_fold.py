"""Hermetic checks for the StartLuxRMSNorm gamma fold (models/startlux-
decision-v1.1-0.8b-instant/model/modeling_text.py).

from_folder bakes `1.0 + weight` into every text-side StartLuxRMSNorm
parameter after checkpoint load so the exported graph stops recomputing the
add on frozen weights (61 per-request Add nodes otherwise).  These checks pin
the fold contract without NPU: bit-identity with the runtime add, idempotence,
the module-unusable-before-folding hazard, and that the gated norm (a
different class whose weight is a post-norm multiplier, never a gamma offset)
cannot be swept up by the walker's isinstance filter.  The fold count (61) and
the end-to-end bit-identity of the AIR are validated by the integration gates
in the model's validation.md.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "startlux-decision-v1.1-0.8b-instant"


def _import_modeling_text():
    """Import model/modeling_text.py as a standalone module (it is plain
    torch code; npu/torchair are only touched at call time)."""
    if str(MODEL) not in sys.path:
        sys.path.insert(0, str(MODEL))
    spec = importlib.util.spec_from_file_location(
        "startlux_modeling_text_under_test", MODEL / "model" / "modeling_text.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GammaFold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mt = _import_modeling_text()

    def _norm(self, dim, dtype):
        torch.manual_seed(7)
        norm = self.mt.StartLuxRMSNorm(dim)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(dim).to(dtype))
        return norm

    def test_fold_is_bit_identical_to_runtime_add(self):
        for dtype in (torch.bfloat16, torch.float32):
            for dim in (1, 128, 256, 1024):
                norm = self._norm(dim, dtype)
                expected = 1.0 + norm.weight.detach().clone()
                self.assertTrue(norm.fold_gamma())
                self.assertTrue(torch.equal(norm.weight.detach(), expected),
                                f"{dtype}/{dim}: folded weight differs")

    def test_fold_is_idempotent(self):
        norm = self._norm(64, torch.bfloat16)
        self.assertTrue(norm.fold_gamma())
        once = norm.weight.detach().clone()
        self.assertFalse(norm.fold_gamma())
        self.assertTrue(torch.equal(norm.weight.detach(), once))

    def test_unfolded_module_is_identity_gamma_hazard(self):
        # A fresh construction holds zeros: forward would scale by 0 until
        # fold_gamma runs.  The load path folds every instance (loud count
        # assertion in from_folder); this pins that the hazard exists and is
        # the reason the fold must never be skipped.
        norm = self.mt.StartLuxRMSNorm(8)
        self.assertTrue(bool((norm.weight.detach() == 0).all()))
        norm.fold_gamma()
        self.assertTrue(bool((norm.weight.detach() == 1).all()))

    def test_gated_norm_is_not_swept_by_the_walker(self):
        # StartLuxRMSNormGated's weight multiplies the normalized output, it
        # is NOT a gamma offset; the from_folder walker's isinstance filter
        # must never match it (and its forward must keep its own math).
        gated = self.mt.StartLuxRMSNormGated(8)
        self.assertNotIsInstance(gated, self.mt.StartLuxRMSNorm)
        self.assertFalse(hasattr(gated, "fold_gamma"))

    def test_gated_gamma_fold_equivalence(self):
        # The gated-forward fold rests on rms_norm(x, w) == w * rms_norm(x, 1)
        # exactly in real arithmetic (gamma is a pure post-normalization
        # scale).  Pin the identity against a fp32 reference -- the NPU
        # kernel acceptance is the integration gates' job.
        def rms_ref(x, gamma, eps):
            n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
            return gamma * n

        torch.manual_seed(11)
        x = torch.randn(64, 128)
        w = torch.randn(128)
        ones = torch.ones(128)
        eps = 1e-6
        self.assertTrue(torch.allclose(rms_ref(x, w, eps), w * rms_ref(x, ones, eps),
                                       atol=0.0, rtol=0.0),
                        "rms_norm(x, w) != w * rms_norm(x, 1) in fp32 reference")
        # and the source passes the weight as the op gamma (no ones_like)
        source = (MODEL / "model" / "modeling_text.py").read_text()
        gated_block = source.split("class StartLuxRMSNormGated")[1].split("class ")[0]
        self.assertIn("npu_rms_norm(x, self.weight", gated_block)
        self.assertNotIn("ones_like", gated_block)


if __name__ == "__main__":
    unittest.main()
