#!/bin/bash
# =============================================================================
# run.sh — qwen2.5-0.5b GE 适配管线入口 (薄封装 core/pipeline, 配置驱动)
#
# 用法: ./run.sh --device <N> [pipeline 选项]     (--device 必填: 用哪张卡是运行期事实, 不进 yaml)
#   ./run.sh --device 6                         # 全流程: source→参考门→adapt→export→pass→compile→manifest→run→compare
#   ./run.sh --device 6 --skip compile,run,compare   # 只到 golden + bundle (不编译 OM)
#   ./run.sh --device 6 --batch-size 2 --seq-len 16  # 小输入冒烟
#   切形态 (prefix/prune): 改 config/model.yaml 的 adapt.params — 产物名自动带后缀, 不互相覆盖
#   ./run.sh --device 6 --skip export           # 复用已有 AIR
#   ./run.sh --device 6 --skip export,passes,compile --bench 10   # 复用 AIR/OM, 只跑 runtime + compare
#   ./run.sh --device 6 --skip export,passes,compile --profiling-parse  # 出 PROF_* 到 io/profiling 并自动解析
#                                                        (只采集不解析: --profiling; 性能测试侧: core.bench --profiling)
#   ./run.sh --device 6 --platform a5           # 覆盖平台 profile (缺省按 device 的 soc 自动探测)
#   ./run.sh --help                             # 完整选项
#
# 平台: config/model.yaml 的 platforms: 段声明各平台事实 (soc/aicore_num/custom_ops/passes);
#       框架用 get_device_properties(--device).name 匹配 soc 自动选定, 产物名带平台后缀不互相覆盖
#
# 两道精度门 (FAIL 即非 0 退出, docs/architecture.md §10):
#   reference — 原版未 patch 的 HF 逐请求前向 vs 适配后 eager  → 验**适配** (--skip reference 跳过)
#   compare   — C++ 运行时输出 vs golden                      → 验**编译/执行**
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
