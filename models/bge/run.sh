#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

# The CANN native process embeds Python. Match it to the invoking interpreter,
# rather than the container's separate system libpython and site-packages.
python_prefix="$(python3 -c 'import sys; print(sys.base_prefix)')"
python_libdir="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export PYTHONHOME="$python_prefix"
export LD_LIBRARY_PATH="$python_libdir:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TE_PARALLEL_COMPILER="${TE_PARALLEL_COMPILER:-2}"
mkdir -p work/tmp
export TMPDIR="$PWD/work/tmp"

stage="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi
exec python3 -m core.pipeline "$stage" --config models/bge/config.yaml "$@"
