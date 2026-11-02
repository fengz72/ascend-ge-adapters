#!/usr/bin/env bash
# Source this script (do NOT execute) after core.dependencies.prepare has
# built the chunk_gated_delta_rule_proto dependency:
#
#   source models/startlux-decision-v1.1-0.8b-instant/scripts/env_chunk_proto.sh \
#          <work_dir>
#
# It exports a task-level LD_PRELOAD pointing at the one installed prototype
# library <work_dir>/dependencies/chunk_gated_delta_rule_proto/install/lib/
# libchunk_gated_delta_rule_compat_proto.so so that the current shell and every
# child process it spawns (python dynamo_export, native ge_runtime, bench
# runners) explicitly load the project-built ChunkGatedDeltaRule GE prototype.
#
# Contract:
#   * deduplicates LD_PRELOAD, keeps any other entries the caller had;
#   * idempotent (sourcing twice leaves a single entry);
#   * refuses with a clear message when the dependency is not built yet;
#   * touches no global/system configuration — the export lives only in this
#     shell (and inherited subprocesses).
#
# Why LD_PRELOAD and not ASCEND_CUSTOM_OPP_PATH auto-discovery: the auto
# vendor layout reproducibly crashed GE initialization (SIGSEGV/SIGILL) under
# the formal core.dependencies.environment; explicit local loading is the
# validated delivery path for this build. See native_proto/README.md and
# work/startlux-cannbot/review-native-env-failure.md.
#
# When the installed CANN build registers the official ChunkGatedDeltaRule
# prototype itself, remove this dependency entry and stop sourcing this
# script instead of double-registering.
env_chunk_proto() {
    local work_dir="${1:?usage: source env_chunk_proto.sh <work_dir>}"
    local lib="$work_dir/dependencies/chunk_gated_delta_rule_proto/install/lib/libchunk_gated_delta_rule_compat_proto.so"
    if [ ! -f "$lib" ]; then
        echo "env_chunk_proto: $lib does not exist;" \
             "run core.dependencies.prepare for chunk_gated_delta_rule_proto first" >&2
        return 1
    fi
    # LD_PRELOAD entries must be independent of the child's working directory.
    lib="$(cd "$(dirname "$lib")" && pwd)/$(basename "$lib")"
    local others="" IFS_BAK="$IFS" part
    IFS=:
    for part in ${LD_PRELOAD:-}; do
        [ -n "$part" ] || continue
        [ "$part" = "$lib" ] && continue
        others="${others:+$others:}$part"
    done
    IFS="$IFS_BAK"
    if [ -n "$others" ]; then
        LD_PRELOAD="$lib:$others"
    else
        LD_PRELOAD="$lib"
    fi
    export LD_PRELOAD
    echo "env_chunk_proto: LD_PRELOAD=$LD_PRELOAD"
}
env_chunk_proto "$@"
