#!/bin/bash
# =============================================================================
# install_prefix_attn_a5.sh — 构建并安装 A5 (Ascend950PR) prefix attention 算子
# (cannbot-dsl GE 交付件, custom_op_v2 PrefixAttention; 910_9382 平台用
# install_prefix_attn.sh 的 AscendC 版 PIA, 两者语义同构但接口/构建体系不同)
#
# 由 model.yaml 的 `custom_ops:` 声明, 管线在**加载 adapter 之前**执行
# (models/qwen2.5-0.5b/model.py 会 import 它的 torch 绑定 prefix_attn_torch)。
# 只有 prefix 链路需要 (adapt.params.prefix: true); 基线链路用不到, 但脚本幂等且
# 已装即跳过, 所以留在配置里无害。
#
# 源码: third_party/ascend-ops/prefix-attention/cannbot-dsl/ge (git submodule,
#       **不修改**; ge/scripts/install.sh 只在其目录内产 ge/build/, 已被该源自己
#       的 .gitignore 忽略)
# 交付: ① GE 交付件 (bisheng 编签入的 .asc → kernel/aicpu .so + libcust_opapi.so)
#          直装 $ASCEND_HOME_PATH/opp/op_graph/ (非 vendors .run 包; CANN 升级会
#          清空 opp, 需重跑本脚本)
#       ② torch 绑定包 prefix_attn_torch (pip -e 装, 失败则回退 PYTHONPATH;
#          注册 torch.ops.ascendc_ops.prefix_attention + torchair converter)
# 构建无需 cannbotdsl (只重生成 kernel 源码时才要); 需要 cmake + bisheng。
# 步骤与版本基线见 third_party/ascend-ops/prefix-attention/cannbot-dsl/ge/README.md
#
# 幂等: 交付件 libcust_opapi.so + 绑定包都在 → 只回传 env 后退出。
# 回传 env: 写 KEY=VALUE 到 $GE_ENV_FILE (core/setup_scripts.py 会读进 os.environ,
#           从而传给后续 ATC / ge_runtime 子进程; 进程内 export 是拿不到的)。
# =============================================================================
set -euo pipefail

BINDING="prefix_attn_torch"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# 源目录优先用 yaml 声明的 path (框架经 $GE_SRC_DIR 传入), 单独手跑时回退到仓库内约定位置
GE_DIR="${GE_SRC_DIR:-${REPO_ROOT}/third_party/ascend-ops/prefix-attention/cannbot-dsl/ge}"
OPP_LIB="${ASCEND_HOME_PATH:?请先 source CANN 的 set_env.sh (需要 ASCEND_HOME_PATH)}/opp/op_graph/lib/linux/aarch64/libcust_opapi.so"

log()  { echo "[prefix_attn_a5] $*"; }
warn() { echo "[prefix_attn_a5][WARN] $*" >&2; }

# 把算子运行期需要的 env 回传给管线 (GE Session/ATC 侧经 ASCEND_CUSTOM_OPP_PATH
# 找到 op_graph 里的交付件; import prefix_attn_torch 也会在进程内自动挂载)
emit_env() {
    [ -n "${GE_ENV_FILE:-}" ] || return 0
    {
        echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_HOME_PATH}/opp:${ASCEND_CUSTOM_OPP_PATH:-}"
    } >> "${GE_ENV_FILE}"
}

# ---- 1. 交付件未装 → 构建 (要构建就得有源码: submodule 未克隆 → 硬失败,
#         model.py 随后 import 绑定会崩, 静默继续只会把错误推到更难查的下游) ----
if [ ! -f "${OPP_LIB}" ]; then
    if [ ! -f "${GE_DIR}/scripts/install.sh" ]; then
        warn "算子源码不存在: ${GE_DIR}"
        warn "  → git submodule update --init --recursive   (需能访问 github.com:443;"
        warn "     该端口在部分网络环境不可达, 离线兜底见 third_party/README.md 的 tarball 步骤)"
        exit 1
    fi

    log "构建 ← ${GE_DIR} (bisheng 编 kernel + cmake 装入 opp/op_graph)"
    bash "${GE_DIR}/scripts/install.sh"

    [ -f "${OPP_LIB}" ] || { echo "[prefix_attn_a5][ERROR] 安装后没出现 ${OPP_LIB}" >&2; exit 1; }
else
    log "GE 交付件已安装 (${OPP_LIB})"
fi

# ---- 2. torch 绑定 (README: pip install -e ge --no-deps) ----
# 首次 import 会 JIT 编译 torch 扩展, 需要 ninja (torch.utils.cpp_extension 硬依赖,
# 厂商镜像常不带) — 缺则补装, 否则下面的导入校验必挂且报错离因果很远
if ! python3 -c "import ninja" > /dev/null 2>&1; then
    log "缺 ninja (JIT 编译 torch 扩展需要) → pip install ninja"
    python3 -m pip install --quiet ninja
fi
if ! python3 -c "import ${BINDING}" > /dev/null 2>&1; then
    if python3 -m pip install --no-deps --quiet -e "${GE_DIR}"; then
        log "torch 绑定已 pip 安装 (editable): ${BINDING}"
    else
        warn "pip 安装绑定失败 → 回退 PYTHONPATH (只对本次管线进程及其子进程生效)"
        if [ -n "${GE_ENV_FILE:-}" ]; then
            echo "PYTHONPATH=${GE_DIR}:${PYTHONPATH:-}" >> "${GE_ENV_FILE}"
        fi
        export PYTHONPATH="${GE_DIR}:${PYTHONPATH:-}"
    fi
else
    log "torch 绑定已可导入: ${BINDING}"
fi

python3 -c "import ${BINDING}" > /dev/null 2>&1 \
    && log "绑定可导入 ✓" \
    || { echo "[prefix_attn_a5][ERROR] ${BINDING} 仍不可导入" >&2; exit 1; }

emit_env
log "完成。验收可跑: python3 -c 'import torch,torch_npu,prefix_attn_torch; print(torch.ops.ascendc_ops.prefix_attention)'"
