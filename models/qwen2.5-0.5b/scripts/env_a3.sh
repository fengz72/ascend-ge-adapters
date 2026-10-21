#!/usr/bin/env bash
# Source after the CANN environment, using the container Python that owns NumPy.
GE_QWEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GE_QWEN_PYTHON_PREFIX="$(python3 -c 'import sys; print(sys.base_prefix)')"
export PYTHONHOME="$GE_QWEN_PYTHON_PREFIX"
export LD_LIBRARY_PATH="$PYTHONHOME/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PYTHONHOME/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$GE_QWEN_ROOT/work/tmp"
export TMPDIR="$GE_QWEN_ROOT/work/tmp"
unset GE_QWEN_ROOT GE_QWEN_PYTHON_PREFIX
