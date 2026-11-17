#!/usr/bin/env bash
# The full standalone flow of the OneJev1.6-0.8B multimodal delivery.
# Run from this adapter directory:
#
#   export ONEJEV_MODEL_DIR=/path/to/hf-model
#   bash run.sh [--device <chip>]
#               [--iterations N [--warmup M] [--seed S]
#                           [--instances K] [--aicore-num A|B] [--readout <path>]]
#
# Steps: inputs -> CPU golden -> dependency build -> C++ executor build ->
# multimodal AIR export -> infer -> compare; --iterations adds the seeded bench.
set -euo pipefail

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "run.sh: $*" >&2; exit 1; }
usage() { sed -n '2,10p' "${BASH_SOURCE[0]}"; }

PROTO_LIB=work/deps/chunk_gated_delta_rule_proto/install/lib/libchunk_gated_delta_rule_compat_proto.so

# ---------------- 固定参数（唯一可调处） ----------------
DEVICE_CHIP=4                 # ASCEND_RT_VISIBLE_DEVICES: npu-smi 的 Chip Logic ID
DEVICE=0                      # 进程内设备号（可见映射后的编号）
COSINE_MIN=0.999              # 精度门限：逐行 cosine 下限
RELATIVE_L2_MAX=0.02          # 精度门限：逐行 relative L2 上限
ITERATIONS=""                 # 空 = 不做 bench；--iterations N 开启
WARMUP=32                     # 性能：预热请求数（覆盖 (S,P) 形状组合）
SEED=42                       # 性能：请求抽样种子
INSTANCES=1                   # 性能：bench 并发实例数（iterations 为全实例合计）
AICORE_NUM=""                 # 性能：每算子核上限 AIC|AIV（空 = 不限核）
READOUT=""                    # 性能：readout.json 路径（空 = 不计时 CPU decider）
# --------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) [[ $# -ge 2 ]] || die "--device needs a Chip Logic ID value"
                  DEVICE_CHIP="$2"; shift 2 ;;
        --iterations) [[ $# -ge 2 ]] || die "--iterations needs a value"
                      ITERATIONS="$2"; shift 2 ;;
        --warmup) [[ $# -ge 2 ]] || die "--warmup needs a value"
                  WARMUP="$2"; shift 2 ;;
        --seed)   [[ $# -ge 2 ]] || die "--seed needs a value"
                  SEED="$2"; shift 2 ;;
        --instances) [[ $# -ge 2 ]] || die "--instances needs a value"
                     INSTANCES="$2"; shift 2 ;;
        --aicore-num) [[ $# -ge 2 ]] || die "--aicore-num needs an AIC|AIV value"
                      AICORE_NUM="$2"; shift 2 ;;
        --readout) [[ $# -ge 2 ]] || die "--readout needs a path"
                   READOUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

# ---------------------------------------------------------------- 1. 环境
step "检查环境及权重路径"
[[ -d model ]] || die "run from the adapter directory (models/onejev1.6-0.8b)"
[[ -n "${ONEJEV_MODEL_DIR:-}" ]] || die "ONEJEV_MODEL_DIR is not set"
[[ -d "$ONEJEV_MODEL_DIR" ]] || die "ONEJEV_MODEL_DIR $ONEJEV_MODEL_DIR is not a directory"
[[ -f fixtures/cases.jsonl ]] || die "fixtures/cases.jsonl does not exist"
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE_CHIP"
# task-local transformers 5.17.0（容器全局 5.8.1 是 StartLux 验证组合，不动）
export PYTHONPATH="$PWD/work/deps/py:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p work/logs work/tmp
export TMPDIR="$PWD/work/tmp"

# ---------------------------------------------------------------- 2. 生成输入
step "生成输入（fixtures/cases.jsonl）"
python3 model/generate_inputs.py --cases fixtures/cases.jsonl \
    --output-dir work/inputs 2>&1 | tee work/logs/inputs.log

# ---------------------------------------------------------------- 3. CPU golden
step "CPU golden（官方 FP32 参考）"
python3 model/golden.py --inputs-dir work/inputs \
    --output-dir work/golden 2>&1 | tee work/logs/golden.log

# ---------------------------------------------------------------- 4. 依赖构建
step "构建／安装 chunk proto 与 WeightNzAndMatMulV3Pass"
bash scripts/prepare.sh 2>&1 | tee work/logs/prepare.log
[[ -f "$PROTO_LIB" ]] || die "chunk proto missing after prepare: $PROTO_LIB"

# ---------------------------------------------------------------- 5. runtime
step "构建 C++ executor"
bash ../../runtime/build.sh 2>&1 | tee work/logs/runtime-build.log

# ---------------------------------------------------------------- 6. 导出
step "导出 onejev AIR"
mkdir -p work/build
LD_PRELOAD="$PROTO_LIB" python3 model/export_air.py \
    --device "$DEVICE" --export-dir work/build --export-name onejev \
    2>&1 | tee work/logs/export.log
[[ -f work/build/onejev.air ]] || die "export produced no onejev.air"

# ---------------------------------------------------------------- 7. 精度推理
step "C++ executor 精度推理"
LD_PRELOAD="$PROTO_LIB" ../../runtime/build/executor \
    --air work/build/onejev.air \
    --records work/inputs/records.json \
    --output-dir work/output 2>&1 | tee work/logs/infer.log
[[ -s work/output/result.json ]] || die "infer produced no result.json; see work/logs/infer.log"

# ---------------------------------------------------------------- 8. 数值比较
step "公共工具比较精度"
python3 ../../tools/compare.py \
    --golden work/golden \
    --output work/output \
    --cosine-min "$COSINE_MIN" \
    --relative-l2-max "$RELATIVE_L2_MAX" 2>&1 | tee work/logs/compare.log

# ---------------------------------------------------------------- 9. 性能（可选）
if [[ -n "$ITERATIONS" ]]; then
    step "C++ executor 性能测试（iterations=$ITERATIONS instances=$INSTANCES warmup=$WARMUP seed=$SEED aicore=$AICORE_NUM readout=$READOUT）"
    BENCH_ARGS=(--iterations "$ITERATIONS" --warmup "$WARMUP" --seed "$SEED"
                --instances "$INSTANCES")
    [[ -n "$AICORE_NUM" ]] && BENCH_ARGS+=(--aicore-num "$AICORE_NUM")
    [[ -n "$READOUT" ]] && BENCH_ARGS+=(--readout "$READOUT")
    LD_PRELOAD="$PROTO_LIB" ../../runtime/build/executor \
        --air work/build/onejev.air \
        --records work/inputs/records.json \
        --output-dir work/bench \
        "${BENCH_ARGS[@]}" \
        2>&1 | tee work/logs/bench.log
    [[ -s work/bench/result.json ]] || die "bench produced no result.json; see work/logs/bench.log"
fi

step "完成"
echo "精度: work/output/comparison.json"
if [[ -n "$ITERATIONS" ]]; then
    echo "性能: work/bench/result.json + work/bench/requests.csv"
fi
echo "日志: work/logs/"
