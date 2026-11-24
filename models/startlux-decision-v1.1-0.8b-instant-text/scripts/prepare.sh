#!/usr/bin/env bash
# Build the StartLux ChunkGatedDeltaRule GE compat prototype from native_proto/
# and install the WeightNzAndMatMulV3Pass fusion pass into the active CANN.
#
# The .so stays local (work/deps/chunk_gated_delta_rule_proto/install: lib +
# dependency.info; no opp/ tree, no python activator), so GE never auto-loads
# it: consumers preload it explicitly (run.sh injects LD_PRELOAD into the
# export / infer / bench subprocesses).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$MODEL_DIR/../.." && pwd)"
DEPS="$MODEL_DIR/work/deps/chunk_gated_delta_rule_proto"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

# CANN graph libraries use the pre-c++11 libstdc++ string ABI; the prototype
# must match it or its symbols cannot resolve.
SOC="${STARTLUX_CHUNK_PROTO_SOC:-ascend910_93}"
VERSION="startlux-chunk-gdr-compat-proto/1.0.0"
mkdir -p "$DEPS/install/lib"
g++ -std=c++17 -fPIC -shared -O2 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I"$ASCEND_HOME_PATH/include" \
    "$MODEL_DIR/native_proto/chunk_gated_delta_rule_proto.cpp" \
    -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" \
    -lgraph -lgraph_base \
    -o "$DEPS/install/lib/libchunk_gated_delta_rule_compat_proto.so"
printf 'id=chunk_gated_delta_rule_proto\nversion=%s\nsoc=%s\narch=%s\ncann=%s\nofficial=false\n' \
    "$VERSION" "$SOC" "$(uname -m)" "${ASCEND_HOME_PATH##*/}" > "$DEPS/install/dependency.info"

# WeightNzAndMatMulV3Pass: system-level install into the ACTIVE CANN's
# custom_fusion_passes directory (optional performance pass; the graph still
# runs without it).
NZ_PASS_DIR="$ROOT/third_party/custom_development_code/fusion_pass/WeightNzAndMatMulV3Pass"
cmake -S "$NZ_PASS_DIR" -B "$NZ_PASS_DIR/build" > /dev/null
cmake --build "$NZ_PASS_DIR/build" --parallel > /dev/null
cmake --install "$NZ_PASS_DIR/build" > /dev/null

# causal-conv1d (vendored fla AscendC CausalConv1d + CausalConv1dGE wrapper):
# the fused GDN conv+silu op; installs into opp/vendors and pip-registers the
# torch binding that model/export_air.py imports before dynamo_export.
bash "$ROOT/third_party/ascend-ops/causal-conv1d/install.sh"

# 官方 custom_op_v2 交付布局：GE 扫描 ASCEND_CUSTOM_OPP_PATH 条目下的
# op_graph/lib/<os>/<arch>/*.so 并自行 dlopen；执行期两个 .so 软链进 work/opp。
OPP_GRAPH="$MODEL_DIR/work/opp/op_graph/lib/$(uname -s | tr 'A-Z' 'a-z')/$(uname -m)"
mkdir -p "$OPP_GRAPH"
ln -sf "$DEPS/install/lib/libchunk_gated_delta_rule_compat_proto.so" "$OPP_GRAPH/"
ln -sf "${ASCEND_HOME_PATH}/opp/vendors/causal_conv1d/op_api/lib/libcust_opapi.so" "$OPP_GRAPH/"

echo "prepare: chunk proto -> $DEPS/install; WeightNzAndMatMulV3Pass + causal-conv1d installed; opp root -> $MODEL_DIR/work/opp"
