#!/usr/bin/env bash
# Build the OneJev ChunkGatedDeltaRule GE compat prototype from native_proto/,
# install the WeightNzAndMatMulV3Pass fusion pass into the active CANN, and
# provision the task-local Python deps (transformers 5.17.0).
#
# The .so stays local (work/deps/chunk_gated_delta_rule_proto/install: lib +
# dependency.info; no opp/ tree, no python activator), so GE never auto-loads
# it: consumers preload it explicitly (run.sh injects LD_PRELOAD into the
# export / infer / bench subprocesses).
#
# Python deps: the OneJev checkpoint and the export prototype target
# transformers 5.17.0 (transformers.vision_utils precompute helpers, vision
# forward returning pooler_output with precomputed kwargs).  The container's
# global transformers (5.8.1) is the validated StartLux delivery combination
# and is NOT touched; work/deps/py is prepended to PYTHONPATH by run.sh and
# self-inserted by the model/ scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$MODEL_DIR/../.." && pwd)"
DEPS="$MODEL_DIR/work/deps/chunk_gated_delta_rule_proto"
PY_DEPS="$MODEL_DIR/work/deps/py"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

# ---------------- task-local transformers 5.17.0 (idempotent) ----------------
if PYTHONPATH="$PY_DEPS" python3 -c \
    'import sys, transformers; sys.exit(0 if transformers.__version__ == "5.17.0" else 1)' \
    2>/dev/null; then
    echo "prepare: task-local transformers 5.17.0 already in $PY_DEPS"
else
    mkdir -p "$PY_DEPS"
    python3 -m pip install --no-deps --target "$PY_DEPS" \
        'transformers==5.17.0' 'tokenizers>=0.23.1,<0.24'
    PYTHONPATH="$PY_DEPS" python3 -c \
        'import transformers; assert transformers.__version__ == "5.17.0"'
fi

# ---------------- ChunkGatedDeltaRule compat prototype ----------------
# CANN graph libraries use the pre-c++11 libstdc++ string ABI; the prototype
# must match it or its symbols cannot resolve.
SOC="${ONEJEV_CHUNK_PROTO_SOC:-ascend910_93}"
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

echo "prepare: python deps -> $PY_DEPS; chunk proto -> $DEPS/install; WeightNzAndMatMulV3Pass installed"
