#!/usr/bin/env bash
# Build the vendored MegaGdnPrefillOp package WITH the MegaGdnPrefill GE
# custom-op-v2 wrapper and the eager flat entry compiled into the same
# libcust_opapi.so (hooked into the vendored build by cmake/custom_build.cmake,
# see the mega-gdn-prefill README). install.sh installs the operator into the
# CANN global opp (vendor package, tbe-custom store requirement) AND registers
# the torch binding via pip (editable -> the vendor package's op_api/python
# targeting the runner's python env via GE_PYTHON) -- the operator imports
# from anywhere, the activator locates the .so and mounts the vendor package
# itself, no shell exports needed. The install dir keeps only dependency.info
# as the core.dependencies identity marker (environment()/activate() degrade
# gracefully without install/{opp,lib,python}).
# Invoked by core.dependencies.prepare with:
#   GE_SRC_DIR    = third_party/ascend-ops/mega-gdn-prefill (the vendoring root)
#   GE_BUILD_DIR  = <work_dir>/dependencies/mega_gdn_prefill/build
#   GE_INSTALL_DIR= <work_dir>/dependencies/mega_gdn_prefill/install
# The vendored build itself reuses the vendoring root's own build cache
# (mega-gdn-prefill/build).
set -euo pipefail
: "${GE_SRC_DIR:?configured dependency source is required}"
: "${GE_INSTALL_DIR:?}"
: "${ASCEND_HOME_PATH:?source CANN set_env.sh first}"

TOP="$(cd "${GE_SRC_DIR}" && pwd)"
ARCH="$(uname -m)"
SOC="${STARTLUX_MEGA_GDN_SOC:-ascend910_93}"
VERSION="startlux-mega-gdn-prefill-v2/2.0.0-merged"

bash "${TOP}/install.sh" -j"$(nproc)" >/dev/null

mkdir -p "${GE_INSTALL_DIR}"
printf 'id=mega_gdn_prefill\nversion=%s\nsoc=%s\narch=%s\ncann=%s\nofficial=false\n' \
  "$VERSION" "$SOC" "$ARCH" "${ASCEND_HOME_PATH##*/}" > "${GE_INSTALL_DIR}/dependency.info"

# Runner discovery (ge OpLibRegister) scans ASCEND_CUSTOM_OPP_PATH roots for a
# root-level libcust_opapi.so and dlopens it so the MegaGdnPrefill REG_OP is
# visible to processes without the python activator (ge_runtime). The symlink
# keeps a single deployed .so (the global opp vendor package installed by
# install.sh); core.dependencies.environment() turns install/opp into
# ASCEND_CUSTOM_OPP_PATH for the build/infer/bench subprocesses.
OPP_ROOT="${ASCEND_OPP_PATH:?install.sh must export ASCEND_OPP_PATH}"
mkdir -p "${GE_INSTALL_DIR}/opp"
ln -sfn "${OPP_ROOT}/vendors/mega_gdn_prefill/op_api/lib/libcust_opapi.so" \
  "${GE_INSTALL_DIR}/opp/libcust_opapi.so"

echo "installed $VERSION -> ${GE_INSTALL_DIR} (soc=$SOC arch=$ARCH)"
