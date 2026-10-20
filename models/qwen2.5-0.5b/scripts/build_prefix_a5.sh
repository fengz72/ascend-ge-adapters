#!/usr/bin/env bash
set -euo pipefail
: "${GE_SRC_DIR:?}"
: "${GE_BUILD_DIR:?}"
: "${GE_INSTALL_DIR:?}"
: "${ASCEND_HOME_PATH:?source CANN set_env.sh first}"
PYTHON="${GE_PYTHON:-python3}"
export MAX_JOBS=2
CANN_ROOT="$(readlink -f "$ASCEND_HOME_PATH")"
STAGE="$GE_BUILD_DIR/install-root"
mkdir -p "$GE_INSTALL_DIR" "$GE_BUILD_DIR/tmp"
export TMPDIR="$GE_BUILD_DIR/tmp"
"$PYTHON" -c 'import ninja'
cmake -S "$GE_SRC_DIR" -B "$GE_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DASCEND_HOME_PATH="$CANN_ROOT"
cmake --build "$GE_BUILD_DIR" --parallel 2
DESTDIR="$STAGE" cmake --install "$GE_BUILD_DIR"
mkdir -p "$GE_INSTALL_DIR/opp"
cp -a "$STAGE$CANN_ROOT/opp/." "$GE_INSTALL_DIR/opp/"
"$PYTHON" -m pip install --no-deps --no-build-isolation --upgrade --target "$GE_INSTALL_DIR/python" "$GE_SRC_DIR"
export PYTHONPATH="$GE_INSTALL_DIR/python:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$GE_INSTALL_DIR/opp"
"$PYTHON" -c 'import prefix_attn_torch'
