#!/usr/bin/env bash
# Build the StartLux ChunkGatedDeltaRule GE prototype (project-built compat,
# non-official) and install the explicit local-loading dependency layout.
# Invoked by core.dependencies.prepare with:
#   GE_SRC_DIR    = models/startlux-decision-v1.1-0.8b-instant/native_proto
#   GE_BUILD_DIR  = <work_dir>/dependencies/<id>/build
#   GE_INSTALL_DIR= <work_dir>/dependencies/<id>/install
# No system paths are written; everything lands inside GE_INSTALL_DIR.
#
# Loading contract (explicit project-local loading; auto vendor loading via
# ASCEND_CUSTOM_OPP_PATH was investigated and REJECTED for this build: the
# formal core.dependencies.environment reproducibly crashes GE init with
# SIGSEGV/SIGILL, see work/startlux-cannbot/review-native-env-failure.md).
# Therefore this script installs only:
#   install/lib/libchunk_gated_delta_rule_compat_proto.so  (the unique library)
#   install/python/chunk_gated_delta_rule_proto.py         (activator module)
#   install/dependency.info                                (identity record)
# No opp/ tree is installed or exported, so GEInitialize never auto-loads it.
# Consumers load it explicitly:
#   - python activator (python_imports): ctypes dlopen RTLD_GLOBAL in-process
#     for dynamo_export, plus task-level LD_PRELOAD for native subprocesses;
#   - scripts/env_chunk_proto.sh <work_dir>: shell-level LD_PRELOAD export.
set -euo pipefail
: "${GE_SRC_DIR:?configured dependency source is required}"
: "${GE_BUILD_DIR:?}"
: "${GE_INSTALL_DIR:?}"
: "${ASCEND_HOME_PATH:?source CANN set_env.sh first}"

ARCH="$(uname -m)"
SOC="${STARTLUX_CHUNK_PROTO_SOC:-ascend910_93}"
VERSION="startlux-chunk-gdr-compat-proto/1.0.0"

mkdir -p "$GE_BUILD_DIR"

# CANN graph libraries are built with the pre-c++11 libstdc++ string ABI;
# the prototype must match it or its symbols cannot resolve.
g++ -std=c++17 -fPIC -shared -O2 -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I"$ASCEND_HOME_PATH/include" \
  "$GE_SRC_DIR/chunk_gated_delta_rule_proto.cpp" \
  -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" \
  -lgraph -lgraph_base \
  -o "$GE_BUILD_DIR/libchunk_gated_delta_rule_compat_proto.so"

# Explicit local-loading layout: lib + python + dependency.info only.
# Drop any earlier opp/ install (including ones from previous experiments);
# it must NOT be exposed for GE auto-discovery.
rm -rf "$GE_INSTALL_DIR/opp"
mkdir -p "$GE_INSTALL_DIR/lib" "$GE_INSTALL_DIR/python"
cp "$GE_BUILD_DIR/libchunk_gated_delta_rule_compat_proto.so" "$GE_INSTALL_DIR/lib/"
cp "$GE_SRC_DIR/python/chunk_gated_delta_rule_proto.py" "$GE_INSTALL_DIR/python/"

printf 'id=chunk_gated_delta_rule_proto\nversion=%s\nsoc=%s\narch=%s\ncann=%s\nofficial=false\n' \
  "$VERSION" "$SOC" "$ARCH" "${ASCEND_HOME_PATH##*/}" > "$GE_INSTALL_DIR/dependency.info"

echo "installed $VERSION -> $GE_INSTALL_DIR (soc=$SOC arch=$ARCH)"
