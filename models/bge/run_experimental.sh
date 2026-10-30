#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
variant="${1:?usage: run_experimental.sh fias|precompute stage [pipeline options]}"
shift
fias="$PWD/tools/bge_fias_all_pass/libbge_fias_structure_pass.so"
candidate="$PWD/tools/bge_candidate_precompute/libbge_candidate_precompute.so"
[[ -f "$fias" ]] || { echo "Build passes with bash tools/build_bge_experimental_passes.sh" >&2; exit 2; }
case "$variant" in
  fias) export BGE_FIAS_PASS_SO="$fias" ;;
  precompute)
    [[ -f "$candidate" ]] || { echo "Build passes with bash tools/build_bge_experimental_passes.sh" >&2; exit 2; }
    export BGE_FIAS_PASS_SO="$fias:$candidate" ;;
  *) echo "expected fias or precompute" >&2; exit 2 ;;
esac
exec bash models/bge/run.sh "$@"
