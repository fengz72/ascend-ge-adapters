#!/bin/bash
# =============================================================================
# run.sh — qwen2.5-0.5b GE 适配管线入口 (薄封装 core/pipeline, 配置驱动)
#
# 用法: ./run.sh --device <N> [pipeline 选项]     (--device 必填: 用哪张卡是运行期事实, 不进 yaml)
#   ./run.sh --device 6                         # 全流程: source→adapt→export→pass→compile→manifest→run→compare
#   ./run.sh --device 6 --skip compile,run,compare   # 只到 golden + bundle (不编译 OM)
#   ./run.sh --device 6 --batch-size 2 --seq-len 16  # 小输入冒烟
#   ./run.sh --device 6 --skip export           # 复用已有 AIR
#   ./run.sh --device 6 --skip export,passes,compile --bench 10   # 复用 AIR/OM, 只跑 runtime + compare
#   ./run.sh --help                             # 完整选项
#
# 配置: config/model.yaml (源/适配/输入/pass/后端/验证)
# 环境: 自动 source env.sh (CANN + PIA 算子 + numpy)
# 注: run/compare 走 C++ runtime (runtime/build/ge_runtime, 先 bash runtime/build.sh);
#     profiling 解析用 tools/parse_profiling.py
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${SCRIPT_DIR}/env.sh"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
exec python3 -m core.pipeline --config "${SCRIPT_DIR}/config/model.yaml" "$@"
