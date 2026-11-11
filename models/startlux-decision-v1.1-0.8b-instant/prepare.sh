#!/usr/bin/env bash
# Build and install the StartLux ChunkGatedDeltaRule GE compat prototype into
# the model-local dependency layout work/deps/chunk_gated_delta_rule_proto/
# (the standalone replacement of the framework prepare step; the build script
# itself is the validated one, unchanged).
#
# Everything lands inside GE_INSTALL_DIR (lib + dependency.info); no system path
# is written and no opp/ tree is exposed for GE auto-discovery.  The loading
# contract is a single LD_PRELOAD of the built .so: run.sh injects it into the
# export / infer / bench subprocesses and export_air.py asserts it at startup.
#
# Also builds and installs the WeightNzAndMatMulV3Pass custom fusion pass
# (third_party/custom_development_code/fusion_pass/WeightNzAndMatMulV3Pass)
# into the CANN custom_fusion_passes directory: it rebuilds the constant
# weights of MatMul/MatMulV2 into FRACTAL_NZ as MatMulV3 at compile time
# (no runtime layout casts; 2026-11-09 ticket-team single-case gate:
# MatMulV2 3115us -> MatMulV3 626us per execution, -14.4% total AI time).
# First graph compile after install pays a one-shot TBE cost per matmul
# shape (cached under kernel_meta).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR"
WORK="$MODEL_DIR/work"
ROOT="$(cd "$MODEL_DIR/../.." && pwd)"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

export GE_SRC_DIR="$MODEL_DIR/native_proto"
export GE_BUILD_DIR="$WORK/deps/chunk_gated_delta_rule_proto/build"
export GE_INSTALL_DIR="$WORK/deps/chunk_gated_delta_rule_proto/install"

bash "$MODEL_DIR/scripts/build_chunk_proto.sh"

echo "prepare: installed chunk_gated_delta_rule_proto -> $GE_INSTALL_DIR"

# WeightNzAndMatMulV3Pass: build + install into the ACTIVE CANN's custom
# fusion-pass directory (a system-level install by design; requires write
# access to $ASCEND_OPP_PATH/vendors/).
NZ_PASS_DIR="$ROOT/third_party/custom_development_code/fusion_pass/WeightNzAndMatMulV3Pass"
cmake -S "$NZ_PASS_DIR" -B "$NZ_PASS_DIR/build" > /dev/null
cmake --build "$NZ_PASS_DIR/build" --parallel > /dev/null
cmake --install "$NZ_PASS_DIR/build" > /dev/null
echo "prepare: installed WeightNzAndMatMulV3Pass -> ${ASCEND_OPP_PATH}/vendors/custom_nz_pass/custom_fusion_passes/"

# causal-conv1d (vendored fla AscendC kernel + CausalConv1dGE wrapper): builds
# the vendor package into $ASCEND_OPP_PATH/vendors/causal_conv1d, writes it
# first in the vendors load_priority, and pip-registers the torch binding
# used by modeling_text.causal_conv4 (both export legs and eager calls).
bash "$ROOT/third_party/ascend-ops/causal-conv1d/install.sh"
echo "prepare: installed causal-conv1d -> ${ASCEND_OPP_PATH}/vendors/causal_conv1d/"
