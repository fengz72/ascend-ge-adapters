#!/usr/bin/env bash
# The dev.txt bypass flow of the StartLux unified ONE-AIR route: an
# INDEPENDENT parallel flow (README "怎么用" -> dev.txt 旁路) that runs the
# business-provided dev.txt query-intent cases through the same two-stage
# split AIR + C++ pipeline as run.sh, without touching the official 5-case
# pool.  All artifacts live under work/devcase/ (never committed).
#
#   export STARTLUX_MODEL_DIR=/path/to/hf-model
#   bash models/startlux-decision-v1.1-0.8b-instant/run_dev.sh [--device <chip>]
#
# Flow: env -> download dev.txt if missing -> generate dev inputs (official
# preprocessing) -> CPU golden -> reuse-or-build deps/AIR/runtime -> infer ->
# numeric compare -> business gate + teacher report (report-only) -> manual
# idle-card confirm -> per-case bench (18 x case_filter) -> pooled-sample
# bench -> summary.  Existing work/build AIRs and work/deps artifacts are
# REUSED when present (the AIR is model-only, case-independent); only
# missing pieces are rebuilt.
set -euo pipefail

step() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "run_dev.sh: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR"
ROOT="$(cd "$MODEL_DIR/../.." && pwd)"
WORK="$MODEL_DIR/work"
DEV_WORK="$WORK/devcase"
LOGS="$DEV_WORK/logs"
RUNTIME_BIN="$ROOT/runtime/build/pipeline_runtime"
PROTO_LIB="$WORK/deps/chunk_gated_delta_rule_proto/install/lib/libchunk_gated_delta_rule_compat_proto.so"
BUILD_DIR="$WORK/build"

# ---------------- 固定参数（唯一可调处） ----------------
DEVICE_CHIP=4                 # ASCEND_RT_VISIBLE_DEVICES: npu-smi 的 Chip Logic ID
DEVICE=0                      # 进程内设备号（可见映射后的编号）
COSINE_MIN=0.999              # 精度门限：逐行 cosine 下限
RELATIVE_L2_MAX=0.02          # 精度门限：逐行 relative L2 上限
WARMUP_CASE=16                # 逐 case bench：预热请求数
ITERATIONS_CASE=400           # 逐 case bench：测量请求数
WARMUP_POOL=32                # 全池 bench：预热请求数（不消耗测量序列）
ITERATIONS_POOL=2048          # 全池 bench：总测量请求数
SEED=42                       # bench 请求抽样种子
DEV_URL="https://storage.360buyimg.com/aiflow-serving/LLM/dev.txt"
DEV_CASES_FILE="$DEV_WORK/source/dev.txt"
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
mkdir -p "$DEV_WORK" "$LOGS"

# ---------------------------------------------------------------- 2. 数据
step "准备 dev.txt（缺失则下载）"
if [[ ! -s "$DEV_CASES_FILE" ]]; then
    mkdir -p "$(dirname "$DEV_CASES_FILE")"
    curl -fsSL "$DEV_URL" -o "$DEV_CASES_FILE" \
        || die "download failed; place the dev.txt file at $DEV_CASES_FILE manually"
fi

step "生成 dev 输入（官方预处理，一次完成）"
rm -rf "$DEV_WORK/inputs"
python3 "$MODEL_DIR/generate_dev_inputs.py" --cases "$DEV_CASES_FILE" 2>&1 | tee "$LOGS/inputs.log"

# ---------------------------------------------------------------- 3. CPU golden
step "CPU golden（官方 FP32 参考，dev 池）"
rm -rf "$DEV_WORK/golden"
python3 "$MODEL_DIR/decider/golden_dev.py" 2>&1 | tee "$LOGS/golden.log"

# ---------------------------------------------------------------- 4. 依赖（复用优先）
step "依赖检查（自定义算子 / runtime / AIR，齐则复用）"
if [[ ! -f "$PROTO_LIB" ]]; then
    step "构建／安装 pass 与自定义算子"
    bash "$MODEL_DIR/prepare.sh" 2>&1 | tee "$LOGS/prepare.log"
fi
if [[ ! -x "$RUNTIME_BIN" ]]; then
    step "构建 C++ runtime"
    bash "$ROOT/runtime/build.sh" 2>&1 | tee "$LOGS/runtime-build.log"
fi
if [[ ! -f "$BUILD_DIR/vision.air" || ! -f "$BUILD_DIR/text.air" || ! -f "$BUILD_DIR/split.json" ]]; then
    step "导出两阶段 split AIR（vision + text）"
    rm -f "$BUILD_DIR/vision.air" "$BUILD_DIR/text.air" "$BUILD_DIR/split.json"
    LD_PRELOAD="$PROTO_LIB" python3 "$MODEL_DIR/model/export_air.py" \
        --device "$DEVICE" 2>&1 | tee "$LOGS/export.log"
fi

# ---------------------------------------------------------------- 5. 精度推理
step "C++ pipeline 精度推理（dev 池，纯文本跳视觉）"
rm -rf "$DEV_WORK/infer"
mkdir -p "$DEV_WORK/infer"
python3 - "$DEV_WORK" "$BUILD_DIR" "$DEVICE" <<'PYREQ' > "$DEV_WORK/infer/request.json"
import json, sys
work, build, device = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(json.dumps({"mode": "infer", "device": device,
                  "vision_air": build + "/vision.air",
                  "text_air": build + "/text.air",
                  "records": work + "/inputs/records.json",
                  "output_dir": work + "/infer"}))
PYREQ
LD_PRELOAD="$PROTO_LIB" "$RUNTIME_BIN" < "$DEV_WORK/infer/request.json" \
    > "$LOGS/infer-runtime.log" 2>&1
sed -n 's/^GE_RESULT //p' "$LOGS/infer-runtime.log" > "$DEV_WORK/infer/result.json"
[[ -s "$DEV_WORK/infer/result.json" ]] || die "infer produced no GE_RESULT; see $LOGS/infer-runtime.log"

# ---------------------------------------------------------------- 6. 数值比较
step "公共工具比较精度"
python3 "$ROOT/tools/compare.py" \
    --golden "$DEV_WORK/golden" \
    --output "$DEV_WORK/infer" \
    --cosine-min "$COSINE_MIN" \
    --relative-l2-max "$RELATIVE_L2_MAX" 2>&1 | tee "$LOGS/compare.log"

# ---------------------------------------------------------------- 7. 业务门禁 + teacher 报告
step "业务概率校验 + teacher 分布对比（报告）"
python3 "$MODEL_DIR/decider/verify_dev.py" 2>&1 | tee "$LOGS/business.log"

# ---------------------------------------------------------------- 8. 空闲确认
step "人工确认卡空闲（性能测试前）"
echo "精度阶段已全部通过。性能测试占用 NPU；确认目标卡（Chip Logic ID $DEVICE_CHIP）空闲。"
read -r -p "已确认卡空闲，开始性能测试? [y/N] " answer
case "$answer" in
    y|Y) ;;
    *) die "已按'卡忙时不测试'中止；稍后可单独重跑 bench（见 README）" ;;
esac

# ---------------------------------------------------------------- 9. 逐 case bench
step "C++ pipeline 逐 case 性能测试（case_filter）"
rm -rf "$DEV_WORK/bench"
mkdir -p "$DEV_WORK/bench/case"
mapfile -t DEV_CASE_IDS < <(python3 - "$DEV_WORK" <<'PYIDS'
import json, sys
records = json.load(open(sys.argv[1] + "/inputs/records.json"))
for case in records["cases"]:
    print(case["id"])
PYIDS
)
for case_id in "${DEV_CASE_IDS[@]}"; do
    out="$DEV_WORK/bench/case/$case_id"
    mkdir -p "$out"
    python3 - "$DEV_WORK" "$BUILD_DIR" "$DEVICE" "$WARMUP_CASE" "$ITERATIONS_CASE" \
        "$SEED" "$case_id" "$out" <<'PYREQ' > "$out/request.json"
import json, sys
work, build, device, warmup, iterations, seed, case_id, out = sys.argv[1:10]
print(json.dumps({"mode": "bench", "device": int(device),
                  "vision_air": build + "/vision.air",
                  "text_air": build + "/text.air",
                  "records": work + "/inputs/records.json",
                  "case_filter": [case_id],
                  "warmup": int(warmup), "iterations": int(iterations),
                  "seed": int(seed), "output_dir": out}))
PYREQ
    LD_PRELOAD="$PROTO_LIB" "$RUNTIME_BIN" < "$out/request.json" \
        > "$LOGS/bench-case-$case_id.log" 2>&1
    sed -n 's/^GE_RESULT //p' "$LOGS/bench-case-$case_id.log" > "$out/result.json"
    [[ -s "$out/result.json" ]] || die "bench produced no GE_RESULT for $case_id; see $LOGS/bench-case-$case_id.log"
    echo "  $case_id done"
done

# ---------------------------------------------------------------- 10. 全池 bench
step "C++ pipeline 全池采样性能测试（dev 池均匀采样）"
out="$DEV_WORK/bench/pool"
mkdir -p "$out"
python3 - "$DEV_WORK" "$BUILD_DIR" "$DEVICE" "$WARMUP_POOL" "$ITERATIONS_POOL" \
    "$SEED" "$out" <<'PYREQ' > "$out/request.json"
import json, sys
work, build, device, warmup, iterations, seed, out = sys.argv[1:8]
print(json.dumps({"mode": "bench", "device": int(device),
                  "vision_air": build + "/vision.air",
                  "text_air": build + "/text.air",
                  "records": work + "/inputs/records.json",
                  "warmup": int(warmup), "iterations": int(iterations),
                  "seed": int(seed), "output_dir": out}))
PYREQ
LD_PRELOAD="$PROTO_LIB" "$RUNTIME_BIN" < "$out/request.json" \
    > "$LOGS/bench-pool.log" 2>&1
sed -n 's/^GE_RESULT //p' "$LOGS/bench-pool.log" > "$out/result.json"
[[ -s "$out/result.json" ]] || die "bench produced no GE_RESULT for the pool; see $LOGS/bench-pool.log"

# ---------------------------------------------------------------- 11. 汇总
step "聚合 bench 摘要"
python3 - "$DEV_WORK" <<'PYSUM' | tee "$LOGS/bench-summary.log"
import json, sys
from pathlib import Path

work = Path(sys.argv[1])
keys = ("qps", "requests", "rows", "e2e_ms", "samples_per_second")
per_case = {}
for directory in sorted((work / "bench" / "case").iterdir()):
    result = json.loads((directory / "result.json").read_text())
    perf = result.get("performance", {})
    per_case[directory.name] = {key: perf.get(key) for key in keys}
pool = None
pool_path = work / "bench" / "pool" / "result.json"
if pool_path.is_file():
    perf = json.loads(pool_path.read_text()).get("performance", {})
    pool = {key: perf.get(key) for key in keys}
summary = {"schema": "startlux-dev-bench-summary/1",
           "per_case": per_case, "pool": pool}
target = work / "bench" / "summary.json"
target.write_text(json.dumps(summary, indent=1) + "\n")
worst = max((entry["e2e_ms"]["p99"] for entry in per_case.values()
             if entry.get("e2e_ms")), default=0.0)
print(f"bench summary: {len(per_case)} per-case runs, worst per-case e2e p99 "
      f"{worst:.3f} ms; pool qps "
      f"{pool['qps'] if pool else 'n/a'}; {target}")
PYSUM

step "完成"
echo "精度: $DEV_WORK/infer/comparison.json + business.json + teacher.json"
echo "性能: $DEV_WORK/bench/summary.json（明细: bench/case/ 与 bench/pool/）"
echo "日志: $LOGS/"
