#!/bin/bash
# =============================================================================
# install_nz_pass.sh — 构建并安装 WeightNzAndMatMulV3Pass (GE fusion pass)
#
# 由 model.yaml 的 `passes:` 声明, 管线在 ATC 编译前执行 (core/setup_scripts.py)。
# 作用: 常量权重 MatMul ND→FRACTAL_NZ + 换 MatMulV3, 减少 L1 搬运 (ATC 日志里
#       `matmul_match num=169` 即生效)。
#
# 源码: third_party/custom_development_code/fusion_pass/WeightNzAndMatMulV3Pass (submodule)
# 安装: $ASCEND_HOME_PATH/opp/vendors/custom_nz_pass/custom_fusion_passes/lib*.so
#       —— 装到 **pass 自己的 vendor 目录**, 不是 per-model 目录: CANN 对 fusion pass
#       没有隔离机制 (自动扫描 opp/vendors/*/custom_fusion_passes 全部加载), per-model
#       目录只会制造"看起来隔离"的错觉, 还会和别处的同名 pass 撞成重复注册 → ATC/TBE 崩。
# 构建: out-of-source 到 <repo>/.pass_build/, 不污染 submodule 工作树
#
# 幂等: 目标 vendor 已有 .so 就跳过; 别的 vendor 已注册同名 pass 也跳过 (复用那份)。
# =============================================================================
set -euo pipefail

PASS_DIR="WeightNzAndMatMulV3Pass"       # 三方源里的目录名
REG_NAME="MatMulWeightNZPass"            # 注册进 GE 的 pass 名 (冲突判定用)
VENDOR="custom_nz_pass"                  # 安装目标 vendor (pass 自己的名字, 全局唯一一份)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SRC_DIR="${REPO_ROOT}/third_party/custom_development_code/fusion_pass/${PASS_DIR}"
BUILD_DIR="${REPO_ROOT}/.pass_build/${PASS_DIR}"
VENDORS_DIR="${ASCEND_HOME_PATH:?请先 source CANN 的 set_env.sh (需要 ASCEND_HOME_PATH)}/opp/vendors"
DST_DIR="${VENDORS_DIR}/${VENDOR}/custom_fusion_passes"

log()  { echo "[nz_pass] $*"; }
warn() { echo "[nz_pass][WARN] $*" >&2; }

# ---- 0. 源码在不在 (submodule 未克隆则跳过, 不阻断管线) ----
if [ ! -d "${SRC_DIR}" ]; then
    warn "pass 源码不存在: ${SRC_DIR}"
    warn "  → git submodule update --init --recursive"
    exit 0
fi

# ---- 1. 目标 vendor 已装过就跳过 ----
if compgen -G "${DST_DIR}/lib*.so" > /dev/null 2>&1; then
    log "已安装, 跳过构建: ${DST_DIR}"
    exit 0
fi

# ---- 2. 别的 vendor 已注册同名 pass → 复用那份, 不装第二份 ----
# 注: 不能用 `strings | grep -q` — grep -q 命中即退出会让 strings 收到 SIGPIPE,
#     在 set -o pipefail 下整条管道返回 141 (非 0) → 检测恒为假 (实测踩过)。
for so in "${VENDORS_DIR}"/*/custom_fusion_passes/*.so; do
    [ -e "${so}" ] || continue
    hits=$(strings "${so}" 2>/dev/null | grep -c "${REG_NAME}" || true)
    if [ "${hits:-0}" -gt 0 ]; then
        warn "已有 vendor 注册同名 pass ${REG_NAME}: ${so}"
        warn "  同名重复注册会让 ATC/TBE 崩 → 跳过安装, 直接复用已有的那份"
        warn "  (若要改用本仓库这份: 先移除上面那个 vendor 目录, 再重跑)"
        exit 0
    fi
done

# ---- 3. 构建 (out-of-source) ----
log "构建 ${PASS_DIR} ← ${SRC_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake "${SRC_DIR}"
make -j"$(nproc)"

# ---- 4. 安装 ----
SO_COUNT=$(ls -1 "${BUILD_DIR}"/lib*.so 2>/dev/null | wc -l)
if [ "${SO_COUNT}" -eq 0 ]; then
    echo "[nz_pass][ERROR] 构建未产出 lib*.so: ${BUILD_DIR}" >&2
    exit 1
fi
mkdir -p "${DST_DIR}"
cp "${BUILD_DIR}"/lib*.so "${DST_DIR}/"
log "安装完成: ${DST_DIR} (${SO_COUNT} 个 .so)"
log "验证: ATC 日志出现 'matmul_match num=169' 即 pass 生效"
