"""Explicit local activator for the project-built ChunkGatedDeltaRule GE
prototype (startlux-chunk-gdr-compat-proto/1.0.0, NON-OFFICIAL compatibility
prototype).

Loading contract (validated for CANN 9.1.0, aarch64, Ascend910_93):
  * Auto vendor loading via ASCEND_CUSTOM_OPP_PATH was investigated and
    REJECTED: with the formal core.dependencies.environment the GE
    initialization process reproducibly dies with SIGSEGV/SIGILL (see
    work/startlux-cannbot/review-native-env-failure.md). The dependency
    therefore installs no opp/ tree and this module performs explicit
    project-local loading only.
  * On import this module:
      1. dlopens the installed prototype .so with RTLD_GLOBAL so that a
         dynamo_export running in this process registers the real REG_OP
         before torchair's unchanged get_registered_ir_def check runs; and
      2. prepends the .so to this worker's LD_PRELOAD so native ge_runtime
         subprocesses spawned by this worker inherit it (task-level env only,
         never a global change).
  * The shell-level equivalent is models/.../scripts/env_chunk_proto.sh
    <work_dir>, sourced after core.dependencies.prepare.

This module never initializes the GE engine or an NPU context, never probes
whether the CANN build already ships the official prototype (such a probe
would require initializing GE), and never disables or fakes any registration
check. When a CANN build ships the official ChunkGatedDeltaRule prototype,
remove this whole dependency entry from the model YAML and unset the
LD_PRELOAD path introduced by the env script instead of loading it.
"""
import ctypes
import os
from pathlib import Path

LIB_NAME = "libchunk_gated_delta_rule_compat_proto.so"
INSTALL_RELPATH = Path("dependencies") / "chunk_gated_delta_rule_proto" / "install"

_FALLBACK_DONE = False


def library_path(work_dir=None):
    """Absolute path of the installed prototype library.

    The module normally lives at <install>/python/, so the library is its
    sibling <install>/lib/<LIB_NAME>; a work_dir is only needed when the
    module is imported from the dependency source tree instead.
    """
    here = Path(__file__).resolve().parent
    candidate = here.parent / "lib" / LIB_NAME
    if candidate.is_file():
        return candidate
    if work_dir is not None:
        candidate = Path(work_dir) / INSTALL_RELPATH / "lib" / LIB_NAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{LIB_NAME} not found next to {here} (looked at {candidate}); "
        "run core.dependencies.prepare for the chunk_gated_delta_rule_proto "
        "dependency first")


def _prepend_ld_preload(path):
    current = os.environ.get("LD_PRELOAD", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if str(path) not in parts:
        parts.insert(0, str(path))
    os.environ["LD_PRELOAD"] = os.pathsep.join(parts)


def ensure_loaded():
    """dlopen the prototype (RTLD_GLOBAL) and arm LD_PRELOAD for subprocesses."""
    global _FALLBACK_DONE
    if _FALLBACK_DONE:
        return
    path = library_path()
    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _prepend_ld_preload(path)
    _FALLBACK_DONE = True


ensure_loaded()
