#!/usr/bin/env bash
# The full standalone flow of the Qwen2.5-0.5B packed delivery.
# Run from this adapter directory:
#
#   export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B
#   bash run.sh [--platform a3|a5] [--device <chip>]
#               [--iterations N [--warmup M] [--seed S]
#                           [--instances K] [--aicore-num A|B]]
#
# Steps: dependency build -> inputs -> CPU golden -> ge_execute build ->
# packed AIR export -> infer -> compare; --iterations adds the seeded bench.
set -euo pipefail

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$*"; }

# ---------------- 固定参数（唯一可调处） ----------------
PLATFORM=a3                 # a3: AscendC prefix + int64 bounds; a5: CANNBotDSL prefix + int32
DEVICE_CHIP=2               # ASCEND_RT_VISIBLE_DEVICES: npu-smi 的 Chip Logic ID
DEVICE_INDEX=0              # ge_execute --device: 进程内设备号；ASCEND_RT_VISIBLE_DEVICES
                            # 映射后可见设备从 0 开始，故恒为 0（与上面的 Chip Logic ID 不同）
COSINE_MIN=0.998            # 精度门限：逐行 cosine 下限
RELATIVE_L2_MAX=0.05        # 精度门限：逐行 relative L2 上限
ITERATIONS=""               # 空 = 不做 bench；--iterations N 开启
WARMUP=10                   # 性能：预热请求数
SEED=42                     # 性能：请求抽样种子
INSTANCES=""                # 性能：bench 并发实例数（空 = 平台默认）
AICORE_NUM=""               # 性能：每算子核上限 AIC|AIV（空 = 平台默认）
# --------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform) PLATFORM="$2"; shift 2 ;;
        --device) DEVICE_CHIP="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --instances) INSTANCES="$2"; shift 2 ;;
        --aicore-num) AICORE_NUM="$2"; shift 2 ;;
        -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "run.sh: unknown argument: $1 (see --help)" >&2; exit 1 ;;
    esac
done

# 平台默认（CLI 未给时生效）：已验证的核数预算与实例数。
case "$PLATFORM" in
    a3) INSTANCES="${INSTANCES:-5}"; AICORE_NUM="${AICORE_NUM:-8|16}" ;;
    a5) INSTANCES="${INSTANCES:-3}"; AICORE_NUM="${AICORE_NUM:-16|32}" ;;
    *) echo "run.sh: unknown platform: $PLATFORM (a3|a5)" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------- 1. 环境
step "检查环境及权重路径（platform=$PLATFORM chip=$DEVICE_CHIP）"
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
# CANN 的内嵌 Python（AOE/TE）在 GEInitialize 时 import numpy，而 set_env.sh 只
# 把 CANN 自带的 site-packages 放进 PYTHONPATH；补上调用解释器的那一份。
export PYTHONPATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE_CHIP"
mkdir -p work/logs

# ---------------------------------------------------------------- 2. 依赖构建
step "构建／安装 $PLATFORM prefix 算子与 WeightNzAndMatMulV3Pass"
bash scripts/prepare.sh "$PLATFORM" 2>&1 | tee work/logs/prepare.log
# shellcheck disable=SC1091
source work/deps/env.sh

# ---------------------------------------------------------------- 3. 生成输入
step "生成输入池（100 case × batch16，历史 lognormal 分布）"
python3 model/generate_inputs.py --platform "$PLATFORM" 2>&1 | tee work/logs/inputs.log

# ---------------------------------------------------------------- 4. CPU golden
step "CPU golden（原始 HF FP32 参考）"
python3 model/golden.py 2>&1 | tee work/logs/golden.log

# ---------------------------------------------------------------- 5. runtime
step "构建 ge_execute"
bash ../../runtime/build.sh 2>&1 | tee work/logs/runtime-build.log

# ---------------------------------------------------------------- 6. 导出
step "导出 packed AIR"
python3 model/export_air.py --platform "$PLATFORM" 2>&1 | tee work/logs/export.log

# ---------------------------------------------------------------- 7. 精度推理
step "ge_execute 精度推理"
# ge_execute 默认 force_fp16；显式 must_keep_origin_dtype 才能复现历史精度基线。
../../runtime/build/ge_execute \
    --graph work/build/packed.air \
    --cases work/inputs/cases.json \
    --output-dir work/output \
    --device "$DEVICE_INDEX" \
    --precision-mode must_keep_origin_dtype 2>&1 | tee work/logs/infer.log

# ---------------------------------------------------------------- 8. 数值比较
step "公共工具比较精度"
python3 ../../tools/compare.py \
    --golden work/golden \
    --output work/output \
    --cosine-min "$COSINE_MIN" \
    --relative-l2-max "$RELATIVE_L2_MAX" 2>&1 | tee work/logs/compare.log

# ---------------------------------------------------------------- 9. 性能（可选）
if [[ -n "$ITERATIONS" ]]; then
    step "ge_execute 性能测试（iterations=$ITERATIONS instances=$INSTANCES warmup=$WARMUP seed=$SEED aicore=$AICORE_NUM）"
    # 同 infer：显式 must_keep_origin_dtype，与历史精度基线对齐。
    ../../runtime/build/ge_execute \
        --graph work/build/packed.air \
        --cases work/inputs/cases.json \
        --output-dir work/bench \
        --device "$DEVICE_INDEX" \
        --precision-mode must_keep_origin_dtype \
        --iterations "$ITERATIONS" --warmup "$WARMUP" --seed "$SEED" \
        --instances "$INSTANCES" --aicore-num "$AICORE_NUM" \
        2>&1 | tee work/logs/bench.log
fi

step "完成"
echo "精度: work/output/comparison.json"
if [[ -n "$ITERATIONS" ]]; then
    echo "性能: work/bench/result.json（逐请求记录见 result.json.records）"
fi
echo "日志: work/logs/"
