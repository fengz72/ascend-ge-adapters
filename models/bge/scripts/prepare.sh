#!/usr/bin/env bash
# Build and install the BGE GE fusion pass into the ACTIVE CANN vendors tree
# (standalone; no repository framework dependency).
#
#   bash models/bge/scripts/prepare.sh
#
# Works from any current directory.  Requires a sourced CANN environment:
# ASCEND_HOME_PATH and ASCEND_OPP_PATH must be set (run.sh sources
# /usr/local/Ascend/ascend-toolkit/latest/set_env.sh first).
#
# Installed pass (registers at kAfterBuiltinFusionPass):
#   BgeFusedInferAttentionPass   -> libbge_fused_infer_attention_pass.so
#
# BgeCandidatePrecomputePass is deliberately NOT installed.  It was archived to
# the submodule branch `archive/bge-candidate-precompute-pass` (fork remote) and
# dropped from this model: its measured +10% throughput came with per-round p99
# spread widening from 2.86%/3.99% to 23.45%/16.35%, it was never re-measured on
# the ge_execute stack, and its matcher is far more model-specific than FIAS's.
# See models/bge/README.md.  Because GE loads EVERY .so in the vendors directory,
# this script actively deletes a previously installed copy -- otherwise the pass
# would keep running and keep printing [BGE_PRECOMPUTE] lines.
set -euo pipefail

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "prepare.sh: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$MODEL_DIR/../.." && pwd)"

: "${ASCEND_HOME_PATH:?source CANN set_env.sh first}"
: "${ASCEND_OPP_PATH:?source CANN set_env.sh first}"

ATTN_PASS_DIR="$ROOT/third_party/custom_development_code/fusion_pass/BgeFusedInferAttentionPass"
VENDOR_DIR="$ASCEND_OPP_PATH/vendors/bge_custom_pass/custom_fusion_passes"
# Passes this model no longer uses.  GE auto-loads everything in the vendors
# directory, so "not building it" is not enough -- an installed copy still runs.
#
# libbge_fias_structure_pass.so is this same pass under its former name
# (BgeFiasFusionPass, registered as BgeFusedInferAttentionStructure).  Left in
# place it would be loaded ALONGSIDE the renamed library, and both passes match
# the identical structure, so whichever ran first would decide the fused node
# names and the packaged graph would stop being reproducible.
RETIRED_LIBRARIES=(libbge_candidate_precompute.so libbge_fias_structure_pass.so)

# The pass installs through its own CMakeLists' absolute
# `LIBRARY DESTINATION "$ENV{ASCEND_OPP_PATH}/vendors/bge_custom_pass/custom_fusion_passes"`,
# so no install prefix is passed here and none may be overridden.
#
# (a) Activation: GE auto-loads every .so in `custom_fusion_passes/` and runs
#     each pass it registers -- empirically confirmed on CANN 9.0.0.  That is
#     the ONLY activation path: runtime/main.cpp has no LD_PRELOAD handling
#     (unlike the older bge-opt-100 runner, which had an `unsetenv("LD_PRELOAD")`
#     guard so CANN's Python compiler children would not inherit it), so this
#     host-wide install is what makes the pass take effect.
# (b) The install is host-wide and needs write access to $ASCEND_OPP_PATH.
# (c) The pass no-ops on graphs without its target structure (0 matches is a
#     success and leaves the graph unchanged), so it coexists with other models.
# (d) A stale .so left in that directory is loaded IN ADDITION to any other
#     copy, not instead of it -- GE runs both and the log shows two summaries.
#     Re-run this script after changing pass sources.

build_pass() {
    local name="$1" directory="$2"
    [[ -d "$directory" ]] || die "pass source is missing: $directory"
    step "构建并安装 $name"
    cmake -S "$directory" -B "$directory/build"
    cmake --build "$directory/build" --parallel
    cmake --install "$directory/build"
}

step "清除本模型已停用的 Pass 产物"
for library in "${RETIRED_LIBRARIES[@]}"; do
    if [[ -f "$VENDOR_DIR/$library" ]]; then
        rm -f "$VENDOR_DIR/$library"
        echo "prepare: removed retired $library (GE would otherwise still load it)"
    else
        echo "prepare: no retired $library present"
    fi
done

build_pass bge_fused_attn "$ATTN_PASS_DIR"

step "校验 vendors 安装产物"
for library in libbge_fused_infer_attention_pass.so; do
    [[ -f "$VENDOR_DIR/$library" ]] || die "$library missing from $VENDOR_DIR; install did not land"
    sha256sum "$VENDOR_DIR/$library"
done
for library in "${RETIRED_LIBRARIES[@]}"; do
    [[ ! -f "$VENDOR_DIR/$library" ]] || die "retired $library is still in $VENDOR_DIR"
done
echo "prepare: installed bge_fused_attn -> $VENDOR_DIR"
