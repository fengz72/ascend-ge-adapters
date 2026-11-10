"""Contract checks for the project-built ChunkGatedDeltaRule GE prototype
dependency (models/startlux-decision-v1.1-0.8b-instant/native_proto).

The prototype is loaded by a single mechanism -- LD_PRELOAD of the built .so --
so the former python activator (ctypes dlopen + LD_PRELOAD arming) was retired.
These hermetic checks (no NPU, no CANN context, no GE initialization) cover
export_air.py's startup guard (_require_proto_preloaded) and the absence of the
old activator.  Real registration/type/shape/AIR/GE acceptance evidence lives in
the integration gates recorded in
models/startlux-decision-v1.1-0.8b-instant/validation.md.
"""
import importlib.util
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "startlux-decision-v1.1-0.8b-instant"
EXPORT_AIR = MODEL / "model" / "export_air.py"
NATIVE_PROTO = MODEL / "native_proto"


def _load_export_air():
    """Import export_air.py without running main() (its module level is stdlib-only)."""
    spec = importlib.util.spec_from_file_location("export_air_under_test", EXPORT_AIR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _LdPreloadCase(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("LD_PRELOAD")

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("LD_PRELOAD", None)
        else:
            os.environ["LD_PRELOAD"] = self._saved

    def _set_preload(self, *entries):
        if entries:
            os.environ["LD_PRELOAD"] = os.pathsep.join(entries)
        else:
            os.environ.pop("LD_PRELOAD", None)


class ProtoPreloadGuard(_LdPreloadCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_export_air()

    def test_passes_when_proto_preloaded(self):
        self._set_preload(str(self.mod.PROTO_LIB))
        self.assertIsNone(self.mod._require_proto_preloaded())

    def test_passes_when_proto_among_other_preloads(self):
        self._set_preload("/usr/lib64/libtcmalloc.so.4", str(self.mod.PROTO_LIB))
        self.assertIsNone(self.mod._require_proto_preloaded())

    def test_fails_when_proto_absent(self):
        self._set_preload("/usr/lib64/libtcmalloc.so.4")
        with self.assertRaises(SystemExit) as ctx:
            self.mod._require_proto_preloaded()
        self.assertIn("LD_PRELOAD", str(ctx.exception))

    def test_fails_when_no_preload(self):
        self._set_preload()
        with self.assertRaises(SystemExit):
            self.mod._require_proto_preloaded()


class ActivatorRetired(unittest.TestCase):
    def test_no_python_activator(self):
        self.assertFalse((NATIVE_PROTO / "python").exists(),
                         "the python activator dir must stay retired (LD_PRELOAD only)")

    def test_cpp_registration_source_present(self):
        self.assertTrue((NATIVE_PROTO / "chunk_gated_delta_rule_proto.cpp").is_file())


if __name__ == "__main__":
    unittest.main()
