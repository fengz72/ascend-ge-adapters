#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
: "${GE_SRC_DIR:?configured dependency source is required}"
: "${GE_BUILD_DIR:?}"
: "${GE_INSTALL_DIR:?}"
: "${ASCEND_HOME_PATH:?source CANN set_env.sh first}"
PYTHON="${GE_PYTHON:-python3}"
cd "$GE_SRC_DIR"
bash build.sh
mapfile -t PACKAGES < <(find build_out -maxdepth 1 -name 'custom_opp_*.run' -type f | sort)
[ "${#PACKAGES[@]}" -eq 1 ] || { echo 'expected exactly one newly built OPP package' >&2; exit 1; }
bash "${PACKAGES[0]}" --quiet --install-path="$ASCEND_HOME_PATH/opp"
"$PYTHON" -m pip install --no-deps --no-build-isolation --upgrade --target "$GE_INSTALL_DIR/python" "$GE_SRC_DIR/torch_binding"
VENDOR="$ASCEND_HOME_PATH/opp/vendors/custom_prefix_attn"
mkdir -p "$GE_INSTALL_DIR"
ln -sfn "$VENDOR" "$GE_INSTALL_DIR/opp"
