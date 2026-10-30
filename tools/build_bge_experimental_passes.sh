#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cann_home="${CANN_HOME:-/usr/local/Ascend/cann-9.0.0}"
source "$cann_home/set_env.sh"
flags=(-std=c++17 -fPIC -D_GLIBCXX_USE_CXX11_ABI=0
  "-I$cann_home/include" "-I$cann_home/opp/built-in/op_graph/inc")
libs=("-L$cann_home/lib64" -lge_compiler -lge_common -lge_common_base
  -lgraph -lgraph_base -lregister -ldl)
g++ -O0 "${flags[@]}" -shared tools/bge_fias_all_pass/bge_fias_all_pass.cpp "${libs[@]}" -o tools/bge_fias_all_pass/libbge_fias_structure_pass.so
g++ -O2 -fopenmp "${flags[@]}" -shared tools/bge_candidate_precompute/bge_candidate_precompute_pass.cpp "${libs[@]}" -o tools/bge_candidate_precompute/libbge_candidate_precompute.so
mkdir -p work/bge-pass-build
for test in test_precompute test_graph_rewrite; do
  g++ -w -O3 -fopenmp "${flags[@]}" "tools/bge_candidate_precompute/$test.cpp" "${libs[@]}" -o "work/bge-pass-build/$test"
  "work/bge-pass-build/$test"
done
printf 'Built BGE experimental passes; host tests passed.\n'
