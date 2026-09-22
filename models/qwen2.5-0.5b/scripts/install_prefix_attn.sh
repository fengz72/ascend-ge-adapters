#!/bin/bash
# =============================================================================
# install_prefix_attn.sh — 构建并安装 PIA 自定义算子 (PrefixInferAttentionScore)
#
# 由 model.yaml 的 `custom_ops:` 声明, 管线在**加载 adapter 之前**执行
# (models/qwen2.5-0.5b/model.py 会 import 它的 torch 绑定)。
# 只有 prefix 链路需要 (adapt.params.prefix: true); 基线链路用不到, 但脚本幂等且
# 已装即跳过, 所以留在配置里无害。
#
# 源码: third_party/ascend-ops/prefix-attention (vendored, **不修改**; build.sh 只在其
#       目录内产 build_out/, 已被该源自己的 .gitignore 忽略)
# 交付: ① CANN 算子包 .run → $ASCEND_HOME_PATH/opp/vendors/custom_prefix_attn/
#       ② torch 绑定包 npu_prefix_infer_attention_score (pip 装, 失败则回退 PYTHONPATH)
# 步骤与版本基线见 third_party/ascend-ops/prefix-attention/README.md
#
# 幂等: vendor 目录 + 绑定包都在 → 只回传 env 后退出 (构建约 4min)。
# 回传 env: 写 KEY=VALUE 到 $GE_ENV_FILE (core/setup_scripts.py 会读进 os.environ,
#           从而传给后续 ATC / ge_runtime 子进程; 进程内 export 是拿不到的)。
# =============================================================================
set -euo pipefail

VENDOR="custom_prefix_attn"
BINDING="npu_prefix_infer_attention_score"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OP_DIR="${REPO_ROOT}/third_party/ascend-ops/prefix-attention"
VENDOR_DIR="${ASCEND_HOME_PATH:?请先 source CANN 的 set_env.sh (需要 ASCEND_HOME_PATH)}/opp/vendors/${VENDOR}"

log()  { echo "[prefix_attn] $*"; }
warn() { echo "[prefix_attn][WARN] $*" >&2; }

# 把算子运行期需要的 env 回传给管线 (等价于 source $VENDOR_DIR/bin/set_env.bash)
emit_env() {
    [ -n "${GE_ENV_FILE:-}" ] || return 0
    {
        echo "ASCEND_CUSTOM_OPP_PATH=${VENDOR_DIR}:${ASCEND_CUSTOM_OPP_PATH:-}"
        echo "LD_LIBRARY_PATH=${VENDOR_DIR}/op_api/lib/:${LD_LIBRARY_PATH:-}"
    } >> "${GE_ENV_FILE}"
}

# ---- 0. 源码在不在 ----
if [ ! -d "${OP_DIR}" ]; then
    warn "算子源码不存在: ${OP_DIR} (见 third_party/README.md 的引入方式)"
    exit 1
fi

# ---- 1. 已装则跳过构建 ----
if [ -d "${VENDOR_DIR}" ] && python3 -c "import ${BINDING}" > /dev/null 2>&1; then
    log "已安装 (${VENDOR_DIR} + ${BINDING}) → 跳过构建 (~4min)"
    emit_env
    exit 0
fi

# ---- 2. 构建 CANN 算子包 (~4min) ----
log "构建 ← ${OP_DIR}"
cd "${OP_DIR}"
bash build.sh

RUN_FILE=$(ls -1 build_out/custom_opp_*.run 2>/dev/null | head -1 || true)
if [ -z "${RUN_FILE}" ]; then
    echo "[prefix_attn][ERROR] 没找到 build_out/custom_opp_*.run (构建失败?)" >&2
    exit 1
fi

# ---- 3. 安装到 CANN opp ----
log "安装 ${RUN_FILE} → ${ASCEND_HOME_PATH}/opp"
bash "${RUN_FILE}" --quiet --install-path="${ASCEND_HOME_PATH}/opp"
[ -d "${VENDOR_DIR}" ] || { echo "[prefix_attn][ERROR] 安装后没出现 ${VENDOR_DIR}" >&2; exit 1; }

# ---- 4. torch 绑定 (README §4: 装 wheel, 或 PYTHONPATH 指向 torch_binding/) ----
if python3 -m pip install --no-deps --quiet "${OP_DIR}/torch_binding"; then
    log "torch 绑定已 pip 安装: ${BINDING}"
else
    warn "pip 安装绑定失败 → 回退 PYTHONPATH (只对本次管线进程及其子进程生效)"
    if [ -n "${GE_ENV_FILE:-}" ]; then
        echo "PYTHONPATH=${OP_DIR}/torch_binding:${PYTHONPATH:-}" >> "${GE_ENV_FILE}"
    fi
    export PYTHONPATH="${OP_DIR}/torch_binding:${PYTHONPATH:-}"
fi

python3 -c "import ${BINDING}" > /dev/null 2>&1 \
    && log "绑定可导入 ✓" \
    || { echo "[prefix_attn][ERROR] ${BINDING} 仍不可导入" >&2; exit 1; }

emit_env
log "完成。验收可跑: python3 ${OP_DIR}/tests/test_torch_binding.py"
