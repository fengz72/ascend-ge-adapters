# GE build-time candidate-table precomputation (OPT-021)

This experimental CANN 9.0.0 custom pass leaves the original ONNX untouched.
At `kAfterBuiltinFusionPass` it matches the fixed-weight candidate projection,
L2 norm/epsilon clamp, normalization, second projection, two dynamic lookups,
and final score consumer. It computes the normalized and projected `[10112,512]`
tables on the host for **every graph build**, inserts two FP32 GE `Const` nodes,
rewires all consumers, and removes the now-dead computation and three original
weight constants. It has no in-process or disk cache. An unexpected number of
matches fails compilation rather than silently running an unoptimized graph.

Build inside `hejun-cann9-bge`:

```sh
source /usr/local/Ascend/cann-9.0.0/set_env.sh
g++ -std=c++17 -O2 -fopenmp -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I/usr/local/Ascend/cann-9.0.0/include \
  -I/usr/local/Ascend/cann-9.0.0/opp/built-in/op_graph/inc \
  bge_candidate_precompute_pass.cpp \
  -L/usr/local/Ascend/cann-9.0.0/lib64 \
  -lge_compiler -lge_common -lge_common_base -lgraph -lgraph_base -lregister -ldl \
  -o libbge_candidate_precompute.so
```

The host numerical smoke test uses the same CANN headers and link libraries,
but does not initialize GE or use a device:

```sh
g++ -w -std=c++17 -O3 -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I/usr/local/Ascend/cann-9.0.0/include \
  -I/usr/local/Ascend/cann-9.0.0/opp/built-in/op_graph/inc \
  test_precompute.cpp -L/usr/local/Ascend/cann-9.0.0/lib64 \
  -lge_compiler -lge_common -lge_common_base -lgraph -lgraph_base -lregister -ldl \
  -o test_precompute && ./test_precompute
```

`test_graph_rewrite.cpp` uses the same build flags/libraries, substitutes its
filename and output name, and exercises a small GE graph's structural match,
all three consumer rewires, and old-node removal without a GE Session or NPU.

`compile_only.cpp` optionally exercises the real exported ONNX and both passes
through `GEInitialize`/`CompileGraph`, but does not call `aclrtSetDevice`,
`LoadGraph`, or `ExecuteGraph`. It is a build check, not an NPU correctness or
performance result. Link it with the runner's GE/parser/ACL libraries.
Avoid full `DUMP_GE_GRAPH` on a nearly full workspace; the full weight-bearing
dump can consume gigabytes. The pass logs `matched` and `replaced` counts.

For the optional BGE path in this runner, set `BGE_FIAS_PASS_SO` to a colon-separated
`libbge_fias_structure_pass.so:libbge_candidate_precompute.so` list. Its native
runtime loads the passes, then clears `LD_PRELOAD` for CANN compiler workers.
The candidate pass requires `matched=1` and `replaced=1` per graph build.

FP32 host precomputation need not reproduce CANN's `force_fp16` intermediate
rounding exactly. Five business cases and CPU/expected Top10 comparison are
required before considering performance. A final graph/profile check must show
the old candidate kernels absent. Build-time work is intentionally repeated for
each graphId; this experiment does not measure or optimize graph build latency.

The maintained build and inference entrypoints are tools/build_bge_experimental_passes.sh and models/bge/run_experimental.sh. The former builds both pass libraries and runs the two host tests. For the latter, choose precompute and provide the logical device ID after checking device occupancy; then run models/bge/run.sh compare --case-ids all. Historical compile-only and full-transfer replay scripts remain in the isolated checkout; the source and summarized results are archived here. The default model YAML does not load this pass.
