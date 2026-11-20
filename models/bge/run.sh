#!/usr/bin/env bash
set -euo pipefail

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "run.sh: $*" >&2; exit 1; }
need() { [[ $# -ge 2 ]] || die "$1 needs a value"; }

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$MODEL_DIR/../.." && pwd)"
WORK="$MODEL_DIR/work"
LOGS="$WORK/logs"

# ---------------- 固定参数（唯一可调处） ----------------
PRECISION_MODE=force_fp16     # GE 精度模式：内部 FP16，外部浮点 IO 保持 FP32
AICORE_NUM="12|24"            # 每算子核上限 AIC|AIV（空 = 不限核）
INPUT_SEED=42                 # 输入池生成种子；同种子同 numpy 版本 = 逐字节相同
COSINE_MIN=0.9999             # 精度门限：逐行 cosine 下限
RELATIVE_L2_MAX=0.001         # 精度门限：逐行 relative L2 上限
INSTANCES=10                  # 性能：闭环并发实例数 = 编译 10 份图，串行约 8s/份
WARMUP=1                      # 性能：每实例预热次数；池内 distinct shape 多于它时自动补齐
ITERATIONS=""                 # 性能：空 = 不做 bench；N = 所有实例合计的测量次数
SEED=42                       # 性能：有放回抽样种子
DEVICE="0"                    # 运行设备
# --------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)         need "$@"; DEVICE="$2"; shift 2 ;;
        --iterations)     need "$@"; ITERATIONS="$2"; shift 2 ;;
        --warmup)         need "$@"; WARMUP="$2"; shift 2 ;;
        --instances)      need "$@"; INSTANCES="$2"; shift 2 ;;
        --seed)           need "$@"; SEED="$2"; shift 2 ;;
        --aicore-num)     need "$@"; AICORE_NUM="$2"; shift 2 ;;
        --precision-mode) need "$@"; PRECISION_MODE="$2"; shift 2 ;;
        *)                die "unknown argument: $1" ;;
    esac
done
[[ -n "$DEVICE" ]] || die "--device is required (npu-smi Chip Logic ID)"

# ---------------------------------------------------------------- 1. 环境
step "检查环境（device=$DEVICE）"
BGE_MODEL_PATH="$(cd "$(dirname "$BGE_MODEL_PATH")" && pwd)/$(basename "$BGE_MODEL_PATH")"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE"
export PYTHONPATH="$(python3 -c 'import sysconfig;print(sysconfig.get_path("purelib"))')${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOGS"
rm -f "$LOGS"/*.log   # 清掉上一轮的，免得已删步骤的日志看起来还在跑

# ---------------------------------------------------------------- 2. 生成输入
step "生成输入（随机，seed=${INPUT_SEED}）+ cases.json"
rm -rf "$WORK/inputs"
python3 "$MODEL_DIR/model/generate_cases.py" \
    --output-dir "$WORK/inputs" \
    --seed "$INPUT_SEED" 2>&1 | tee "$LOGS/inputs.log"

# ---------------------------------------------------------------- 3. CPU golden
step "CPU golden（onnxruntime FP32，纯 CPU）"
rm -rf "$WORK/golden"
# 产物是 runtime infer 同构布局（result.json + <case_id>/output_0.npy），第 7 步的
# tools/compare.py 两侧因此共用一个 loader。五例全程约 2s，峰值约 1.5GB RSS。
python3 "$MODEL_DIR/model/golden.py" \
    --model "$BGE_MODEL_PATH" \
    --inputs-dir "$WORK/inputs" \
    --output-dir "$WORK/golden" 2>&1 | tee "$LOGS/golden.log"

# ---------------------------------------------------------------- 4. 融合 Pass
step "构建／安装 GE 融合 Pass"
# prepare.sh 自己校验 vendors 里的安装产物与被淘汰库确实不在，这里不重复。
bash "$MODEL_DIR/scripts/prepare.sh" 2>&1 | tee "$LOGS/prepare.log"

# ---------------------------------------------------------------- 5. runtime
step "构建 C++ runtime"
bash "$ROOT/runtime/build.sh" 2>&1 | tee "$LOGS/runtime-build.log"

# ---------------------------------------------------------------- 6. 精度推理
step "精度推理（ge_execute, $PRECISION_MODE）"
rm -rf "$WORK/infer"
# 省略 --iterations 即 infer：每 case 跑一次、单实例、无预热。图不做中间加工，
# ge_execute 经 aclgrphParseONNX 直接吃内联权重的原始导出。--output-dir 不存在会自建。
"$ROOT/runtime/build/ge_execute" \
    --graph "$BGE_MODEL_PATH" \
    --cases "$WORK/inputs/cases.json" \
    --output-dir "$WORK/infer" \
    --device "$DEVICE" \
    --precision-mode "$PRECISION_MODE" \
    --aicore-num "$AICORE_NUM" 2>&1 | tee "$LOGS/infer.log"

# ---------------------------------------------------------------- 7. 数值比较
step "数值比较（CPU golden vs 设备输出）"
# 两侧都是 npy 布局、输出名都是位置名 output_0，所以按 name 对齐。报告写进 --output
# 目录，即 $WORK/infer/comparison.json。
if ! python3 "$ROOT/tools/compare.py" \
    --golden "$WORK/golden" \
    --output "$WORK/infer" \
    --cosine-min "$COSINE_MIN" \
    --relative-l2-max "$RELATIVE_L2_MAX" 2>&1 | tee "$LOGS/compare.log"; then
    die "accuracy comparison failed; see $LOGS/compare.log and $WORK/infer/comparison.json"
fi

# ---------------------------------------------------------------- 8. 可选性能
if [[ -n "$ITERATIONS" ]]; then
    step "性能测试（instances=$INSTANCES warmup=$WARMUP iterations=$ITERATIONS seed=$SEED aicore=$AICORE_NUM）"
    rm -rf "$WORK/bench"
    "$ROOT/runtime/build/ge_execute" \
        --graph "$BGE_MODEL_PATH" \
        --cases "$WORK/inputs/cases.json" \
        --output-dir "$WORK/bench" \
        --device "$DEVICE" \
        --precision-mode "$PRECISION_MODE" \
        --aicore-num "$AICORE_NUM" \
        --instances "$INSTANCES" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --seed "$SEED" 2>&1 | tee "$LOGS/bench.log"
fi

# ---------------------------------------------------------------- 9. 完成
step "完成"
echo "精度: $WORK/infer/comparison.json"
echo "输出: $WORK/infer/<case_id>/output_0.npy + result.json"
echo "golden: $WORK/golden/<case_id>/output_0.npy + result.json"
if [[ -n "$ITERATIONS" ]]; then
    echo "性能: $WORK/bench/result.json（逐请求记录在其 records 数组里）"
fi
echo "日志: $LOGS/"
