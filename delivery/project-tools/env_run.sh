#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/cann-9.0.0/set_env.sh
python_prefix="$(python3 -c 'import sys; print(sys.base_prefix)')"
python_libdir="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export PYTHONHOME="$python_prefix"
export LD_LIBRARY_PATH="$python_libdir:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TE_PARALLEL_COMPILER=2
project_root="$(cd "$(dirname "$0")/.." && pwd)"
export TMPDIR="$project_root/work/tmp"
mkdir -p "$TMPDIR"
exec "$@"
