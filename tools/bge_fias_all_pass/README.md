# Structural FusedInferAttentionScore pass

This isolated GE custom pass matches a specific bidirectional padding-mask
attention structure by operator types, edges, constants, shapes and transpose
permutations. It does not require BGE node names or a fixed layer count. Each
matching subgraph is replaced by one `FusedInferAttentionScore`; unmatched
subgraphs are left unchanged. It is not a universal attention matcher.

The supported pattern has BSND Q/K/V reshapes with the same head count and
width, Q/K scaling through the same `1/sqrt(D)` path, and a
`SelectV2(-FLT_MAX, 1 - binary_keep_mask)` additive mask whose BOOL condition
is true for blocked positions. GE currently lowers the last-dimension
selection to `StridedSliceD(begin=[-1], end=[INT64_MAX], strides=[1])`.
The pass checks these details before replacing the score MatMul, mask Add,
Softmax, value MatMul and output Transpose. It also requires untransposed,
unbiased MatMuls and last-axis Softmax. It retains the Q/K/V projections,
BSND reshapes and mask generation. It never constructs a causal mask.

Build in the CANN 9.0.0 container:

```sh
source /usr/local/Ascend/cann-9.0.0/set_env.sh
g++ -std=c++17 -O0 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I/usr/local/Ascend/cann-9.0.0/include \
  -I/usr/local/Ascend/cann-9.0.0/opp/built-in/op_graph/inc \
  bge_fias_all_pass.cpp -L/usr/local/Ascend/cann-9.0.0/lib64 \
  -lge_compiler -lge_common -lge_common_base -lgraph -lgraph_base -lregister -ldl \
  -o libbge_fias_structure_pass.so
```

Set `BGE_FIAS_PASS_SO` to the new library when running the optional pass in this runner.
The runtime preloads it only in the native GE child and clears the preload
before CANN Python workers start. Check `matched_subgraphs`,
`replaced_subgraphs`, and `post_opt_subgraphs` in the log. GE's subsequent
`force_fp16` stage converts Q/K/V and output to FP16; the post-optimization
check requires those descriptors and the BOOL mask.

On the original BGE graph the pass matched 24/24 subgraphs and passed the five
business cases against CPU and expected Top10. Renaming every ONNX node still
matched 24. Swapping the two inputs of one commutative Q-side Mul gave an
equivalent graph with 23 matches and 23 fused nodes, confirming that a
non-matching structure is skipped without a fixed-count requirement. See
`docs/bge/experiments/OPT-020.md` for evidence. The previously tested
fixed-count binary `libbge_fias_all_pass.so` remains separate for OPT-019.

From the runner root, bash tools/build_bge_experimental_passes.sh builds both optional passes and runs the precompute host tests. For five-case inference, use bash models/bge/run_experimental.sh fias infer --device <id> --case-ids all, then run the normal compare stage. The default model YAML does not load this pass.
