#!/usr/bin/env bash
# Build the StartLux ChunkGatedDeltaRule GE prototype (project-built compat,
# non-official) and install the explicit local-loading dependency layout.
# Invoked by prepare.sh with:
#   GE_SRC_DIR    = models/startlux-decision-v1.1-0.8b-instant/native_proto
#   GE_BUILD_DIR  = <model>/work/deps/<id>/build
#   GE_INSTALL_DIR= <model>/work/deps/<id>/install
# No system paths are written; everything lands inside GE_INSTALL_DIR.
#
# Loading contract (single mechanism: LD_PRELOAD; auto vendor loading via
# ASCEND_CUSTOM_OPP_PATH was investigated and REJECTED for this build: it
# reproducibly crashes GE init with SIGSEGV/SIGILL).  Therefore this script
# installs only:
#   install/lib/libchunk_gated_delta_rule_compat_proto.so  (the unique library)
#   install/dependency.info                                (identity record)
# No opp/ tree and no python activator are installed, so GEInitialize never
# auto-loads it.  Consumers preload the .so explicitly: run.sh injects
# LD_PRELOAD into the export / infer / bench subprocesses, and export_air.py
# asserts it at startup (_require_proto_preloaded).
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

# Explicit local-loading layout: lib + dependency.info only.
# Drop any earlier opp/ or python/ install (including ones from previous
# experiments): opp/ must NOT be exposed for GE auto-discovery, and the python
# activator is retired (LD_PRELOAD is the single loading mechanism).
rm -rf "$GE_INSTALL_DIR/opp" "$GE_INSTALL_DIR/python"
mkdir -p "$GE_INSTALL_DIR/lib"
cp "$GE_BUILD_DIR/libchunk_gated_delta_rule_compat_proto.so" "$GE_INSTALL_DIR/lib/"

printf 'id=chunk_gated_delta_rule_proto\nversion=%s\nsoc=%s\narch=%s\ncann=%s\nofficial=false\n' \
  "$VERSION" "$SOC" "$ARCH" "${ASCEND_HOME_PATH##*/}" > "$GE_INSTALL_DIR/dependency.info"

echo "installed $VERSION -> $GE_INSTALL_DIR (soc=$SOC arch=$ARCH)"
