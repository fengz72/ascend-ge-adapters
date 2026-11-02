"""Contract and install checks for the project-built ChunkGatedDeltaRule GE
prototype dependency (models/startlux-decision-v1.1-0.8b-instant/native_proto).

Hermetic behavior tests: no NPU, no CANN context, no GE initialization.
They exercise the real loading surfaces (env script refusal/dedup, activator
dlopen of a stub library). Real registration/type/shape/AIR/GE acceptance
evidence lives in the integration gates recorded in
models/startlux-decision-v1.1-0.8b-instant/native-validation.md."""
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "startlux-decision-v1.1-0.8b-instant"
ENV_SCRIPT = MODEL / "scripts" / "env_chunk_proto.sh"
ACTIVATOR_SRC = MODEL / "native_proto" / "python" / "chunk_gated_delta_rule_proto.py"
SCRATCH = ROOT / "work" / "startlux-validation" / "native-package" / "tmp"
LIB_NAME = "libchunk_gated_delta_rule_compat_proto.so"


def _ensure_scratch():
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


class EnvScriptBehavior(unittest.TestCase):
    def setUp(self):
        if not _ensure_scratch():
            self.skipTest("scratch dir unavailable")

    def run_bash(self, script):
        return subprocess.run(["bash", "-c", script], capture_output=True,
                              text=True, cwd=ROOT)

    def test_missing_dependency_refused(self):
        r = self.run_bash(f"source {ENV_SCRIPT} work/does-not-exist-at-all; echo rc=$?")
        self.assertEqual(r.returncode, 0)
        self.assertIn("rc=1", r.stdout)
        self.assertIn("does not exist", r.stderr)
        self.assertIn("core.dependencies.prepare", r.stderr)

    def test_exports_deduped_task_level_ld_preload(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            lib = Path(tmp) / "dependencies" / "chunk_gated_delta_rule_proto" / "install" / "lib"
            lib.mkdir(parents=True)
            (lib / LIB_NAME).write_bytes(b"stub")
            tmp_arg = tmp
            r = self.run_bash(
                f"export LD_PRELOAD=/keep/other.so; "
                f"source {ENV_SCRIPT} {tmp_arg}; "
                f"source {ENV_SCRIPT} {tmp_arg} >/dev/null; "
                f"echo \"LD=$LD_PRELOAD\"")
            expected = f"{lib / LIB_NAME}:/keep/other.so"
            ld_lines = [l for l in r.stdout.splitlines() if l.startswith("LD=")]
            self.assertEqual(len(ld_lines), 1)
            self.assertEqual(ld_lines[0], f"LD={expected}")


class ActivatorBehavior(unittest.TestCase):
    def setUp(self):
        if not _ensure_scratch():
            self.skipTest("scratch dir unavailable")

    def _install_layout(self, tmp, with_lib):
        install = Path(tmp) / "dependencies" / "chunk_gated_delta_rule_proto" / "install"
        (install / "python").mkdir(parents=True, exist_ok=True)
        (install / "lib").mkdir(parents=True, exist_ok=True)
        (install / "python" / "chunk_gated_delta_rule_proto.py").write_text(
            ACTIVATOR_SRC.read_text())
        if with_lib:
            stub = Path(tmp) / "stub.cpp"
            stub.write_text("int startlux_proto_stub(void) { return 0; }\n")
            subprocess.run(["g++", "-std=c++17", "-fPIC", "-shared", str(stub),
                           "-o", str(install / "lib" / LIB_NAME)], check=True)
        return install

    def _import_activator(self, install):
        spec = importlib.util.spec_from_file_location(
            "chunk_gated_delta_rule_proto_test", install / "python" / "chunk_gated_delta_rule_proto.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_missing_library_message(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            install = self._install_layout(tmp, with_lib=False)
            with self.assertRaises(FileNotFoundError) as ctx:
                self._import_activator(install)
            self.assertIn("core.dependencies.prepare", str(ctx.exception))

    def test_dlopen_and_ld_preload_armed(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
            install = self._install_layout(tmp, with_lib=True)
            old = os.environ.get("LD_PRELOAD")
            try:
                module = self._import_activator(install)
                self.assertTrue(module._FALLBACK_DONE)
                expected = str(install / "lib" / LIB_NAME)
                self.assertIn(expected, os.environ["LD_PRELOAD"])
                parts = os.environ["LD_PRELOAD"].split(":")
                self.assertEqual(parts.count(expected), 1)
                if old:
                    self.assertIn(old, os.environ["LD_PRELOAD"])
            finally:
                if old is None:
                    os.environ.pop("LD_PRELOAD", None)
                else:
                    os.environ["LD_PRELOAD"] = old


if __name__ == "__main__":
    unittest.main()
