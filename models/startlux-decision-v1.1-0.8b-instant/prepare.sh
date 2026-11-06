#!/usr/bin/env bash
# Build and install the StartLux ChunkGatedDeltaRule GE compat prototype into
# the model-local dependency layout work/deps/chunk_gated_delta_rule_proto/
# (the standalone replacement of the framework prepare step; the build script
# itself is the validated one, unchanged).
#
# Everything lands inside GE_INSTALL_DIR (lib + python + dependency.info); no
# system path is written and no opp/ tree is exposed for GE auto-discovery.
# The loading contract stays the validated explicit local loading: export_air
# imports the activator module (dlopen RTLD_GLOBAL) and run.sh sets LD_PRELOAD
# for the export / infer / bench subprocesses only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR"
WORK="$MODEL_DIR/work"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

export GE_SRC_DIR="$MODEL_DIR/native_proto"
export GE_BUILD_DIR="$WORK/deps/chunk_gated_delta_rule_proto/build"
export GE_INSTALL_DIR="$WORK/deps/chunk_gated_delta_rule_proto/install"

bash "$MODEL_DIR/scripts/build_chunk_proto.sh"

echo "prepare: installed chunk_gated_delta_rule_proto -> $GE_INSTALL_DIR"
