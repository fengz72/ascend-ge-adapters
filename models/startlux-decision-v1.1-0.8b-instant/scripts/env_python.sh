#!/usr/bin/env bash
# Source after the CANN environment, using the container Python that owns NumPy.
# Inlined from the validated A3 container-Python bootstrap (previously sourced
# from models/qwen2.5-0.5b/scripts/env_a3.sh) so this model directory carries
# its own copy and does not reference another model's scripts.
#
#   source models/startlux-decision-v1.1-0.8b-instant/scripts/env_python.sh
#
# Sets PYTHONHOME / LD_LIBRARY_PATH / site-packages PYTHONPATH of the container
# Python, pins 8 BLAS/OpenMP threads and the project TMPDIR.  It installs
# nothing and touches no system configuration.
GE_STARTLUX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GE_STARTLUX_PYTHON_PREFIX="$(python3 -c 'import sys; print(sys.base_prefix)')"
export PYTHONHOME="$GE_STARTLUX_PYTHON_PREFIX"
export LD_LIBRARY_PATH="$PYTHONHOME/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PYTHONHOME/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# TMPDIR stays at the PROJECT-ROOT work/tmp exactly like the validated A3
# bootstrap: GE initialization binds multiprocessing.Manager AF_UNIX sockets
# under TMPDIR, and the 107-byte sun_path limit rejects the much longer
# model-local work/tmp path.
mkdir -p "$GE_STARTLUX_ROOT/work/tmp"
export TMPDIR="$GE_STARTLUX_ROOT/work/tmp"
unset GE_STARTLUX_ROOT GE_STARTLUX_PYTHON_PREFIX
