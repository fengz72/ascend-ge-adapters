#!/usr/bin/env bash
# Build the platform Prefix Attention operator and the WeightNzAndMatMulV3Pass
# fusion pass from third_party/custom_development_code, then write
# work/deps/env.sh for run.sh to source.
#
#   bash scripts/prepare.sh [a3|a5]
#
# Requires the CANN environment (ASCEND_HOME_PATH / ASCEND_OPP_PATH); cmake and
# the operator builds fail loudly without it.  The prefix operator stays local
# (work/deps/prefix_attention/install/{python,opp}), so consumers activate it
# explicitly by sourcing work/deps/env.sh; the fusion pass installs into the
# active CANN's vendors/custom_nz_pass/custom_fusion_passes/, where GE picks it
# up on its own (optional performance pass; the graph still runs without it).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
THIRD_PARTY="$(cd "$MODEL_DIR/../.." && pwd)/third_party/custom_development_code"
DEPS="$MODEL_DIR/work/deps"
OP_OUT="$DEPS/prefix_attention"
PLATFORM="${1:-a3}"

# A3: 算子工程自带 build.sh，产出 custom_opp_*.run 装进活动 CANN 的 opp；
# torch_binding 用 pip --target 装到本地 install/python，vendor 目录以软链暴露成
# install/opp（env.sh 再 readlink -f 解析成 CANN 认得的真实路径）。
build_prefix_a3() {
    local src="$THIRD_PARTY/custom_op/prefix-attention/ascendc"
    mkdir -p "$OP_OUT/install"
    (
        cd "$src"
        bash build.sh
        bash "$(ls -t build_out/custom_opp_*.run | head -1)" --quiet \
            --install-path="$ASCEND_HOME_PATH/opp"
        python3 -m pip install --no-deps --no-build-isolation --upgrade \
            --target "$OP_OUT/install/python" "$src/torch_binding"
        ln -sfn "$ASCEND_HOME_PATH/opp/vendors/custom_prefix_attn" "$OP_OUT/install/opp"
    ) > "$OP_OUT/build.log" 2>&1
}

# A5: GE 交付件走 cmake + DESTDIR 暂存，把 CANN opp 子树拷成本地 install/opp，
# python 绑定同样 --target 安装；末尾在装好的环境里 import 一次作为构建自检
# （本机无 A5 实机，这条路径只能靠自检）。
build_prefix_a5() {
    local src="$THIRD_PARTY/custom_op/prefix-attention/cannbot-dsl/ge"
    local build="$OP_OUT/build" stage="$OP_OUT/build/install-root"
    local cann_root="$(readlink -f "$ASCEND_HOME_PATH")"
    mkdir -p "$OP_OUT/install/opp" "$build/tmp"
    (
        export TMPDIR="$build/tmp" MAX_JOBS=2
        export PYTHONPATH="$OP_OUT/install/python:${PYTHONPATH:-}"
        export ASCEND_CUSTOM_OPP_PATH="$OP_OUT/install/opp"
        python3 -c 'import ninja'
        cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
            -DASCEND_HOME_PATH="$cann_root"
        cmake --build "$build" --parallel 2
        DESTDIR="$stage" cmake --install "$build"
        cp -a "$stage$cann_root/opp/." "$OP_OUT/install/opp/"
        python3 -m pip install --no-deps --no-build-isolation --upgrade \
            --target "$OP_OUT/install/python" "$src"
        python3 -c 'import prefix_attn_torch'
    ) > "$OP_OUT/build.log" 2>&1
}

# NZ pass: cmake --install 直接装进活动 CANN 的 custom_fusion_passes 目录，
# 没有本地产物，也不需要 LD_LIBRARY_PATH。
build_nz_pass() {
    local src="$THIRD_PARTY/fusion_pass/WeightNzAndMatMulV3Pass"
    local build="$DEPS/weight_nz/build"
    mkdir -p "$DEPS/weight_nz"
    (
        cmake -S "$src" -B "$build"
        cmake --build "$build" --parallel 2
        cmake --install "$build"
    ) > "$DEPS/weight_nz/build.log" 2>&1
}

mkdir -p "$DEPS"
case "$PLATFORM" in
    a3) build_prefix_a3 ;;
    a5) build_prefix_a5 ;;
    *) echo "prepare.sh: unknown platform $PLATFORM (a3|a5)" >&2; exit 1 ;;
esac
build_nz_pass

{
    printf 'export PYTHONPATH="%s/install/python:${PYTHONPATH:-}"\n' "$OP_OUT"
    printf 'export ASCEND_CUSTOM_OPP_PATH="%s:${ASCEND_CUSTOM_OPP_PATH:-}"\n' \
        "$(readlink -f "$OP_OUT/install/opp")"
    printf 'export TORCH_EXTENSIONS_DIR="%s/work/torch_extensions"\n' "$MODEL_DIR"
} > "$DEPS/env.sh"

echo "prepare: prefix_attention($PLATFORM) -> $OP_OUT/install (log: $OP_OUT/build.log)"
echo "prepare: WeightNzAndMatMulV3Pass -> \$ASCEND_OPP_PATH/vendors/custom_nz_pass (log: $DEPS/weight_nz/build.log)"
echo "prepare: wrote $DEPS/env.sh"
