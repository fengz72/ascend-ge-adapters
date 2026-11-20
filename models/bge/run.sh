#!/usr/bin/env bash
# BGE standalone flow: shell-orchestrated (see models/bge/README.md).
#
#   export BGE_MODEL_PATH=/path/to/model.onnx
#   bash models/bge/run.sh [--device-chip N] [--device N] [--iterations N]
#                          [--warmup N] [--instances N] [--seed N]
#                          [--aicore-num A|B] [--precision-mode M]
#
# Works from any current directory. BGE_MODEL_PATH is the raw ONNX export; it is
# NOT vendored (1.35 GB). The five business samples and both references live in
# fixtures/ (provenance: fixtures/SOURCE.md).
#
# Steps: 1 环境检查 -> 2 生成输入+cases.json -> 3 融合 Pass 构建/安装
#        -> 4 C++ runtime -> 5 精度推理 -> 6 数值与业务比较;
#        --iterations adds the seeded bench after an interactive card-idle check.
#
# 没有打包图这一步：ge_execute 经 aclgrphParseONNX 直接吃业务原始导出（内联权重），
# 实测输出与重打包成外部权重的产物逐比特相同。cases.json 的输入顺序取自
# generate_cases.py 的 DTYPES，即图自身的顺序（ONNX 为冻结交付，见 fixtures/SOURCE.md）。
set -euo pipefail

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "run.sh: $*" >&2; exit 1; }
# 从头部注释块自动生成帮助：跳过 shebang，打印连续的 # 行（去掉 "# " 前缀），
# 遇到第一行非注释即停。不按行号截取，所以改头部注释不会悄悄弄坏 --help。
usage() { awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR"
ROOT="$(cd "$MODEL_DIR/../.." && pwd)"
WORK="$MODEL_DIR/work"
LOGS="$WORK/logs"
RUNTIME_BIN="$ROOT/runtime/build/ge_execute"
FIXTURES="$MODEL_DIR/fixtures"

# ---------------- 固定参数（唯一可调处） ----------------
DEVICE_CHIP=0                 # ASCEND_RT_VISIBLE_DEVICES: npu-smi 的 Chip Logic ID
DEVICE=0                      # 进程内设备号（可见映射后的编号）
PRECISION_MODE=force_fp16     # GE 精度模式：内部 FP16，外部浮点 IO 保持 FP32
AICORE_NUM="12|24"            # 每算子核上限 AIC|AIV（空 = 不限核）
COSINE_MIN=0.9999             # 精度门限：逐行 cosine 下限
RELATIVE_L2_MAX=0.001         # 精度门限：逐行 relative L2 上限
MAX_ABS_ERROR=0.001           # 精度门限：逐行最大绝对误差上限
TOP10_ATOL=0.001              # 业务门禁：Top-10 分数绝对误差上限
INSTANCES=10                  # 性能：闭环并发实例数 = 编译 10 份图，串行约 8s/份
WARMUP=1                      # 性能：每实例预热次数；池内 distinct shape 多于它时自动补齐
ITERATIONS=""                 # 性能：空 = 不做 bench；N = 所有实例**合计**的测量次数
                              # （不是每实例。旧注释的「历史默认 6」配 10 实例只有 6 个请求，
                              #   是退化值；要可比至少 10000，见 Pass README 的测量口径）
SEED=42                       # 性能：有放回抽样种子；同 seed 同 N 下全局序列可复现
# --------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device-chip) [[ $# -ge 2 ]] || die "--device-chip needs a Chip Logic ID value"
                  DEVICE_CHIP="$2"; shift 2 ;;
        --device) [[ $# -ge 2 ]] || die "--device needs an in-process device index"
                  DEVICE="$2"; shift 2 ;;
        --iterations) [[ $# -ge 2 ]] || die "--iterations needs a count"
                  ITERATIONS="$2"; shift 2 ;;
        --warmup) [[ $# -ge 2 ]] || die "--warmup needs a value"
                  WARMUP="$2"; shift 2 ;;
        --instances) [[ $# -ge 2 ]] || die "--instances needs a value"
                  INSTANCES="$2"; shift 2 ;;
        --seed)   [[ $# -ge 2 ]] || die "--seed needs a value"
                  SEED="$2"; shift 2 ;;
        --aicore-num) [[ $# -ge 2 ]] || die "--aicore-num needs an AIC|AIV value"
                  AICORE_NUM="$2"; shift 2 ;;
        --precision-mode) [[ $# -ge 2 ]] || die "--precision-mode needs a value"
                  PRECISION_MODE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

# ---------------------------------------------------------------- 1. 环境
step "检查环境及模型路径"
[[ -n "${BGE_MODEL_PATH:-}" ]] || die "BGE_MODEL_PATH is not set"
[[ -f "$BGE_MODEL_PATH" ]] || die "BGE_MODEL_PATH $BGE_MODEL_PATH is not a file"
BGE_MODEL_PATH="$(cd "$(dirname "$BGE_MODEL_PATH")" && pwd)/$(basename "$BGE_MODEL_PATH")"
# shellcheck disable=SC1091
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi
[[ -n "${ASCEND_OPP_PATH:-}" ]] || die "ASCEND_OPP_PATH is unset after CANN set_env.sh"
VENDOR_DIR="$ASCEND_OPP_PATH/vendors/bge_custom_pass/custom_fusion_passes"
# The CANN native process embeds Python. Match it to the invoking interpreter,
# rather than the container's separate system libpython and site-packages.
read -r python_prefix python_libdir < <(python3 -c \
    'import sys, sysconfig; print(sys.base_prefix, sysconfig.get_config_var("LIBDIR"))')
export PYTHONHOME="$python_prefix"
# 只在 LD_LIBRARY_PATH 非空时拼冒号：尾随的空条目会被 glibc 解释为**当前工作目录**
# （man ld.so: "a zero-length directory name is interpreted as the current working
# directory"），而本脚本可从任意 cwd 运行。正常流程里上面的 set_env.sh 已填充该变量，
# 但 CANN 安装若没设它，就会把 CWD 加进共享库搜索路径。
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="$python_libdir:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$python_libdir"
fi
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TE_PARALLEL_COMPILER="${TE_PARALLEL_COMPILER:-2}"
# Visible-device mapping must be in place before any step touches the device.
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE_CHIP"
# 只预建 $LOGS：它是下面 rm -f 与各步 tee 的目标。inputs/infer/compare/bench 都由
# 各自的步骤先 rm -rf 再自建（ge_execute 也会创建 --output-dir），在这里预建是无效工作。
mkdir -p "$LOGS"
# 清掉上一轮的日志：步骤增删之后残留的 .log 会让人以为某个已删除的步骤还在跑
# （例如阶段 2 去掉垫片后遗留的 pack-outputs.log）。
rm -f "$LOGS"/*.log
# CANN 内嵌的 Python 会起一个 multiprocessing.Manager，它在
# $TMPDIR/pymp-XXXXXXXX/listener-XXXXXXXX 上 bind 一个 AF_UNIX socket。
# sun_path 上限 107 字节，所以 TMPDIR 必须短：$WORK/tmp 在这里是 77 字符，
# 加上后缀共 110 字节，GEInitialize 会以 EC0009 "Python built-in object
# multiprocessing.Manager instantiation failed" 失败。用 /tmp 下的固定短路径，
# 与仓库检出深度无关。
BGE_TMPDIR="/tmp/bge-$(id -u)"
mkdir -p "$BGE_TMPDIR"
export TMPDIR="$BGE_TMPDIR"
for fixture in sample_inputs.json expected_outputs.json cpu-golden.npz; do
    [[ -f "$FIXTURES/$fixture" ]] || die "fixture is missing: $FIXTURES/$fixture"
done

# ---------------------------------------------------------------- 2. 生成输入
step "生成输入（五份业务样例）+ cases.json"
rm -rf "$WORK/inputs"
python3 "$MODEL_DIR/model/generate_cases.py" \
    --samples "$FIXTURES/sample_inputs.json" \
    --output-dir "$WORK/inputs" 2>&1 | tee "$LOGS/inputs.log"
[[ -f "$WORK/inputs/case-0000/input_ids.npy" ]] \
    || die "generate_cases produced no input cases; see $LOGS/inputs.log"
[[ -f "$WORK/inputs/cases.json" ]] \
    || die "generate_cases produced no cases.json; see $LOGS/inputs.log"

# ---------------------------------------------------------------- 3. 融合 Pass
step "构建／安装 GE 融合 Pass"
bash "$MODEL_DIR/scripts/prepare.sh" 2>&1 | tee "$LOGS/prepare.log"
[[ -f "$VENDOR_DIR/libbge_fused_infer_attention_pass.so" ]] \
    || die "fusion pass not installed: $VENDOR_DIR/libbge_fused_infer_attention_pass.so"

# ---------------------------------------------------------------- 4. runtime
step "构建 C++ runtime"
bash "$ROOT/runtime/build.sh" 2>&1 | tee "$LOGS/runtime-build.log"
[[ -x "$RUNTIME_BIN" ]] || die "runtime binary is missing after build: $RUNTIME_BIN"

# ---------------------------------------------------------------- 5. 精度推理
step "精度推理（ge_execute, $PRECISION_MODE）"
rm -rf "$WORK/infer"
mkdir -p "$WORK/infer"
# 省略 --iterations 即 infer：每 case 跑一次、单实例、无预热。
# 输出落在 $WORK/infer/<case_id>/output_<i>.npy，可直接 np.load。
# 图直接喂业务原始导出：ge_execute 经 aclgrphParseONNX 吃内联权重，实测与
# 重打包成外部权重的产物输出逐比特相同，所以不再需要打包这一步。
"$RUNTIME_BIN" \
    --graph "$BGE_MODEL_PATH" \
    --cases "$WORK/inputs/cases.json" \
    --output-dir "$WORK/infer" \
    --device "$DEVICE" \
    --precision-mode "$PRECISION_MODE" \
    --aicore-num "$AICORE_NUM" 2>&1 | tee "$LOGS/infer.log"
[[ -s "$WORK/infer/result.json" ]] \
    || die "infer produced no result.json; see $LOGS/infer.log"

# ---------------------------------------------------------------- 6. 数值与业务比较
step "数值与业务比较（CPU golden + Top-10）"
rm -rf "$WORK/compare"
# compare.py 直接读第 5 步的 <case_id>/output_<i>.npy + result.json；runtime 只按
# 位置报输出名（output_0），compare.py 的 OUTPUT_NAMES 把它映射回 scores。
if ! python3 "$MODEL_DIR/model/compare.py" \
    --golden "$FIXTURES/cpu-golden.npz" \
    --expected "$FIXTURES/expected_outputs.json" \
    --infer-dir "$WORK/infer" \
    --output-dir "$WORK/compare" \
    --cosine-min "$COSINE_MIN" \
    --relative-l2-max "$RELATIVE_L2_MAX" \
    --max-abs-error "$MAX_ABS_ERROR" \
    --atol "$TOP10_ATOL" 2>&1 | tee "$LOGS/compare.log"; then
    die "accuracy comparison failed; see $LOGS/compare.log and $WORK/compare/comparison.json"
fi
[[ -s "$WORK/compare/comparison.json" ]] \
    || die "compare produced no comparison.json; see $LOGS/compare.log"

# ---------------------------------------------------------------- 7. 可选性能
if [[ -n "$ITERATIONS" ]]; then
    step "人工确认卡空闲（性能测试前）"
    echo "精度阶段已全部通过。性能测试占用 NPU；确认目标卡（Chip Logic ID $DEVICE_CHIP）空闲。"
    read -r -p "已确认卡空闲，开始性能测试? [y/N] " answer || answer=""
    case "$answer" in
        y|Y) ;;
        *) die "已按'卡忙时不测试'中止；稍后可单独重跑（见 README）" ;;
    esac

    step "性能测试（instances=$INSTANCES warmup=$WARMUP iterations=$ITERATIONS seed=$SEED aicore=$AICORE_NUM）"
    rm -rf "$WORK/bench"
    mkdir -p "$WORK/bench"
    "$RUNTIME_BIN" \
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
    [[ -s "$WORK/bench/result.json" ]] \
        || die "bench produced no result.json; see $LOGS/bench.log"
fi

# ---------------------------------------------------------------- 8. 完成
step "完成"
echo "精度: $WORK/compare/comparison.json"
echo "输出: $WORK/infer/<case_id>/output_0.npy + $WORK/infer/result.json"
if [[ -n "$ITERATIONS" ]]; then
    echo "性能: $WORK/bench/result.json（逐请求记录在其 records 数组里）"
fi
echo "日志: $LOGS/"
