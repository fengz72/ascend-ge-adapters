"""Hermetic checks for the StartLuxRMSNorm gamma fold (models/startlux-
decision-v1.1-0.8b-instant/model/modeling_text.py).

StartLuxRMSNorm bakes `1.0 + weight` into its parameter through a
load_state_dict post hook: the hook fires only after the raw checkpoint
value has landed, so an unconditional +1 is exactly the invariant "loaded
== gamma in place" (re-load included; no state marker -- a marker that
skipped a re-load would leave raw values in).  These checks pin the fold
contract without NPU: bit-identity with the runtime add, correct re-load
semantics, the module-unusable-before-load hazard, and that the gated norm
(a different class whose weight is a post-norm multiplier, never a gamma
offset) carries no such hook.  The fold count (61) and the end-to-end
bit-identity of the AIR are validated by the integration gates in the
model's validation.md.
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

    @staticmethod
    def _load_raw(norm, raw):
        raw = raw.detach().clone()
        norm.load_state_dict({"weight": raw})
        return raw

    def test_fold_is_bit_identical_to_runtime_add(self):
        for dtype in (torch.bfloat16, torch.float32):
            for dim in (1, 128, 256, 1024):
                norm = self._norm(dim, dtype)
                raw = self._load_raw(norm, norm.weight.detach())
                self.assertTrue(torch.equal(norm.weight.detach(), 1.0 + raw),
                                f"{dtype}/{dim}: folded weight differs")

    def test_reload_folds_again_not_twice(self):
        # the hook is unconditional by design: a re-load overwrites the
        # weight with raw checkpoint values and folds exactly once more, so
        # the parameter never accumulates (+2) and never sits at raw values
        norm = self._norm(64, torch.bfloat16)
        raw = torch.full((64,), 0.5, dtype=torch.bfloat16)
        self._load_raw(norm, raw)
        once = norm.weight.detach().clone()
        self._load_raw(norm, raw)
        self.assertTrue(torch.equal(norm.weight.detach(), once))
        self.assertTrue(torch.equal(norm.weight.detach(), 1.0 + raw))

    def test_unloaded_module_is_identity_gamma_hazard(self):
        # A fresh construction holds zeros: forward would scale by 0 until a
        # load lands.  The hook fires only on load, so this hazard is the
        # price of the checkpoint-shaped init; pinned so the contract is
        # never "fixed" silently.
        norm = self.mt.StartLuxRMSNorm(8)
        self.assertTrue(bool((norm.weight.detach() == 0).all()))
        self._load_raw(norm, torch.zeros(8))
        self.assertTrue(bool((norm.weight.detach() == 1).all()))

    def test_gated_norm_has_no_fold_hook(self):
        # StartLuxRMSNormGated's weight multiplies the normalized output, it
        # is NOT a gamma offset: loading state into it must leave the value
        # untouched (no +1 anywhere).
        gated = self.mt.StartLuxRMSNormGated(8)
        self.assertNotIsInstance(gated, self.mt.StartLuxRMSNorm)
        self.assertFalse(hasattr(gated, "_fold_gamma"))
        raw = torch.full((8,), 0.5)
        gated.load_state_dict({"weight": raw})
        self.assertTrue(torch.equal(gated.weight.detach(), raw))

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
