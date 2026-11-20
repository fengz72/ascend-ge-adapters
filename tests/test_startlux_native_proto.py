"""Contract checks for the project-built ChunkGatedDeltaRule GE prototype
dependency (models/startlux-decision-v1.1-0.8b-instant/native_proto).

The prototype is loaded by a single mechanism -- LD_PRELOAD of the built .so --
so the former python activator (ctypes dlopen + LD_PRELOAD arming) was retired.
These checks are hermetic (no NPU, no CANN context, no GE initialization, no
torch): the activator dir stays retired and the C++ registration source is
present.

export_air.py's startup guard (_require_proto_preloaded) is not covered here.
Importing that module executes its module level, which pulls in torch, the model
package and the causal_conv1d activator installed by prepare.sh, so such a case
is not hermetic and errors out on a checkout without those dependencies.  The
guard runs on every real export instead.  Registration/type/shape/AIR/GE
acceptance evidence lives in the integration gates recorded in
models/startlux-decision-v1.1-0.8b-instant/validation.md.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "startlux-decision-v1.1-0.8b-instant"
NATIVE_PROTO = MODEL / "native_proto"


class ActivatorRetired(unittest.TestCase):
    def test_no_python_activator(self):
        self.assertFalse((NATIVE_PROTO / "python").exists(),
                         "the python activator dir must stay retired (LD_PRELOAD only)")

    def test_cpp_registration_source_present(self):
        self.assertTrue((NATIVE_PROTO / "chunk_gated_delta_rule_proto.cpp").is_file())


if __name__ == "__main__":
    unittest.main()
