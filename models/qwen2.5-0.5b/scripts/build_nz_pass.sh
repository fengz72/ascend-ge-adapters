#!/usr/bin/env bash
set -euo pipefail
: "${GE_SRC_DIR:?}"
: "${GE_BUILD_DIR:?}"
: "${GE_INSTALL_DIR:?}"
: "${ASCEND_HOME_PATH:?source CANN set_env.sh first}"
DEST="$ASCEND_HOME_PATH/opp/vendors/custom_nz_pass/custom_fusion_passes"
cmake -S "$GE_SRC_DIR" -B "$GE_BUILD_DIR"
cmake --build "$GE_BUILD_DIR" --parallel 2
mkdir -p "$DEST" "$GE_INSTALL_DIR/lib"
cp "$GE_BUILD_DIR"/lib*.so "$DEST/"
cp "$GE_BUILD_DIR"/lib*.so "$GE_INSTALL_DIR/lib/"
