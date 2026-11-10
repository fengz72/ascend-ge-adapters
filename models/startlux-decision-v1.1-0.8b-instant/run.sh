#!/usr/bin/env bash
# The full standalone flow of the StartLux unified ONE-AIR route
# (docs/standalone-runtime-plan.md): shell-orchestrated, no `core`.
#
#   export STARTLUX_MODEL_DIR=/path/to/hf-model
#   bash models/startlux-decision-v1.1-0.8b-instant/run.sh [--device <chip>]
#
# Works from any current directory.  Every fixed parameter lives at the top of
# this file -- there is no YAML, no override precedence and no discovery; the
# NPU card is selected with --device (npu-smi Chip Logic ID, default 4).  The
# official preprocessing deps (task-local Pillow/torchvision) are environment
# prerequisites documented in README; missing packages fail naturally at the
# stage that needs them.
set -euo pipefail

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "run.sh: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR"
ROOT="$(cd "$MODEL_DIR/../.." && pwd)"
WORK="$MODEL_DIR/work"
LOGS="$WORK/logs"
RUNTIME_BIN="$ROOT/runtime/build/ge_runtime"
PROTO_LIB="$WORK/deps/chunk_gated_delta_rule_proto/install/lib/libchunk_gated_delta_rule_compat_proto.so"

# ---------------- 固定参数（唯一可调处） ----------------
DEVICE_CHIP=4                 # ASCEND_RT_VISIBLE_DEVICES: npu-smi 的 Chip Logic ID
DEVICE=0                      # 进程内设备号（可见映射后的编号）
COSINE_MIN=0.999              # 精度门限：逐行 cosine 下限
RELATIVE_L2_MAX=0.02          # 精度门限：逐行 relative L2 上限
INSTANCES=1                   # 性能：实例数
WARMUP=32                     # 性能：每实例预热请求数（不消耗测量序列）
ITERATIONS=2048               # 性能：总测量请求数
SEED=42                       # 性能：请求抽样种子
# --------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) [[ $# -ge 2 ]] || die "--device needs a Chip Logic ID value"
                  DEVICE_CHIP="$2"; shift 2 ;;
        *) die "unknown argument: $1 (only --device <chip> is accepted)" ;;
    esac
done

# Task-local Pillow/torchvision of the validated install (environment
# prerequisite; see README).  Prepend for the python stages only.
TASK_LOCAL_DEPS="$ROOT/work/startlux-validation/p6-official/deps"

# ---------------------------------------------------------------- 1. 环境
step "检查环境及权重路径"
[[ -n "${STARTLUX_MODEL_DIR:-}" ]] || die "STARTLUX_MODEL_DIR is not set"
[[ -d "$STARTLUX_MODEL_DIR" ]] || die "STARTLUX_MODEL_DIR $STARTLUX_MODEL_DIR is not a directory"
# shellcheck disable=SC1091
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
# shellcheck disable=SC1091
source "$MODEL_DIR/scripts/env_python.sh"
export PYTHONPATH="$TASK_LOCAL_DEPS/pillow:$TASK_LOCAL_DEPS/torchvision:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE_CHIP"
export TORCH_EXTENSIONS_DIR="$WORK/torch_extensions"
mkdir -p "$WORK" "$LOGS"

# ---------------------------------------------------------------- 2. 生成输入
step "生成输入（官方预处理，一次完成）"
rm -rf "$WORK/inputs" "$WORK/fixtures"
python3 "$MODEL_DIR/generate_inputs.py" 2>&1 | tee "$LOGS/inputs.log"

# ---------------------------------------------------------------- 3. CPU golden
step "CPU golden（官方 FP32 参考）"
rm -rf "$WORK/golden"
python3 "$MODEL_DIR/decider/golden.py" 2>&1 | tee "$LOGS/golden.log"

# ---------------------------------------------------------------- 4. 依赖构建
step "构建／安装 pass 与自定义算子"
bash "$MODEL_DIR/prepare.sh" 2>&1 | tee "$LOGS/prepare.log"
[[ -f "$PROTO_LIB" ]] || die "prototype library missing after prepare: $PROTO_LIB"

# ---------------------------------------------------------------- 5. runtime
step "构建 C++ runtime"
bash "$ROOT/runtime/build.sh" 2>&1 | tee "$LOGS/runtime-build.log"
[[ -x "$RUNTIME_BIN" ]] || die "runtime binary missing: $RUNTIME_BIN"

# ---------------------------------------------------------------- 6. 导出
step "导出单 AIR，生成 IO 与运行请求"
rm -f "$WORK/build/model.air" "$WORK/infer/request.json" "$WORK/bench/request.json"
LD_PRELOAD="$PROTO_LIB" python3 "$MODEL_DIR/model/export_air.py" \
    --device "$DEVICE" --instances "$INSTANCES" --warmup "$WARMUP" \
    --iterations "$ITERATIONS" --seed "$SEED" 2>&1 | tee "$LOGS/export.log"
[[ -f "$WORK/build/model.air" ]] || die "export produced no AIR"
[[ -f "$WORK/infer/request.json" && -f "$WORK/bench/request.json" ]] \
    || die "export produced no requests"

# ---------------------------------------------------------------- 7. 精度推理
step "C++ GESession 精度推理"
rm -f "$WORK/infer/outputs.bin" "$WORK/infer/result.json" \
      "$WORK/infer/comparison.json" "$WORK/infer/business.json"
LD_PRELOAD="$PROTO_LIB" "$RUNTIME_BIN" < "$WORK/infer/request.json" \
    > "$LOGS/infer-runtime.log" 2>&1
sed -n 's/^GE_RESULT //p' "$LOGS/infer-runtime.log" > "$WORK/infer/result.json"
[[ -s "$WORK/infer/result.json" ]] || die "infer produced no GE_RESULT; see $LOGS/infer-runtime.log"

# ---------------------------------------------------------------- 8. 数值比较
step "公共工具比较精度"
python3 "$ROOT/tools/compare.py" \
    --golden "$WORK/golden" \
    --output "$WORK/infer" \
    --cosine-min "$COSINE_MIN" \
    --relative-l2-max "$RELATIVE_L2_MAX" 2>&1 | tee "$LOGS/compare.log"

# ---------------------------------------------------------------- 9. 业务门禁
step "业务概率校验"
python3 "$MODEL_DIR/decider/verify_business.py" 2>&1 | tee "$LOGS/business.log"

# ---------------------------------------------------------------- 10. 空闲确认
step "人工确认卡空闲（性能测试前）"
echo "精度阶段已全部通过。性能测试占用 NPU；确认目标卡（Chip Logic ID $DEVICE_CHIP）空闲。"
read -r -p "已确认卡空闲，开始性能测试? [y/N] " answer
case "$answer" in
    y|Y) ;;
    *) die "已按'卡忙时不测试'中止；稍后可单独重跑 bench（见 README）" ;;
esac

# ---------------------------------------------------------------- 11. 性能测试
step "C++ GESession 性能测试"
rm -f "$WORK/bench/result.json" "$WORK/bench/requests.csv"
LD_PRELOAD="$PROTO_LIB" "$RUNTIME_BIN" < "$WORK/bench/request.json" \
    > "$LOGS/bench-runtime.log" 2>&1
sed -n 's/^GE_RESULT //p' "$LOGS/bench-runtime.log" > "$WORK/bench/result.json"
[[ -s "$WORK/bench/result.json" ]] || die "bench produced no GE_RESULT; see $LOGS/bench-runtime.log"

step "完成"
echo "精度: $WORK/infer/comparison.json + $WORK/infer/business.json"
echo "性能: $WORK/bench/result.json + $WORK/bench/requests.csv"
echo "日志: $LOGS/"
