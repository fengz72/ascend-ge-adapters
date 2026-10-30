# BGE FusedInferAttentionScore one-layer experiment

This isolated GE custom pass replaces only encoder layer 0's attention with
`FusedInferAttentionScore`. It is not a performance recommendation or a pass for
other models. The original ONNX is unchanged.

## Mask contract

The original ONNX has `Where_2(condition, -FLT_MAX, 1 - expanded_mask)`.
`condition` is the `DT_BOOL` output of `Cast_2(1 - expanded_mask)`, and the
same condition is reused as the FIAS `atten_mask`: `true` means blocked. The
mask is bidirectional `[B,1,S,S]`; `sparse_mode=0` does not add a causal mask.
The original `Expand_1` shape construction (which includes `Where_1`) is
preserved. No mask is reconstructed from the raw input. The all-zero mask
boundary is documented below; the current FP16 route requires a visible key.

Equivalent generation from the public input, if a different graph has no
existing bool edge, is:

```python
# attention_mask: int64[B,S], 1 = visible, 0 = padding
blocked = (attention_mask == 0)                 # bool[B,S]
fias_mask = broadcast_to(blocked[:,None,None,:], (B,1,S,S))
# fias_mask[b,0,q,k] == True  means key k is blocked for query q.
```

Use GE shape operators to derive B and S at runtime; never freeze them to the
five sample lengths. Do not pass the public INT64 keep-mask directly or add a
causal triangle. For this specific graph, the `Where_2` condition is already
the required broadcasted bool tensor, so adding a second mask-building branch
would only increase graph work and create another place for shape mistakes.

The pass takes Q/K/V from their BSND reshapes, before the three transposes.
It replaces the final BNSD-to-BSND transpose too. The two original pre-QK
scales are represented by `scale=0.125` (`D=64`), so numeric order changes and
must be validated against the original model and CPU reference.
GE's built-in transpose optimization may rename the final transpose to
`PartitionedCall_...Transpose_3_Transpose_*`; the matcher accepts this form.
Compiled GE descriptors can retain `-1` for every dynamic dimension, so the
pass also checks the exact layer-0 node identities rather than requiring the
head count and width to be materialized in those descriptors.

The pass runs at `kAfterBuiltinFusionPass`, where original Q/K/V descriptors
are FP32. GE's `force_fp16` stage subsequently converts the FIAS Q/K/V and
output to FP16; a read-only post-optimization check confirmed this. Inserting
the node at `kAfterOriginGraphOptimize` compiled but failed during `LoadGraph`
lowering, so that later stage is not used.

## Build (CANN 9.0.0 container)

```sh
source /usr/local/Ascend/cann-9.0.0/set_env.sh
g++ -std=c++17 -O0 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I/usr/local/Ascend/cann-9.0.0/include \
  -I/usr/local/Ascend/cann-9.0.0/opp/built-in/op_graph/inc \
  bge_fias_pass.cpp -L/usr/local/Ascend/cann-9.0.0/lib64 \
  -lge_compiler -lge_common -lge_common_base -lgraph -lgraph_base -lregister -ldl \
  -o libbge_fias_pass.so
```

Load the pass only in the isolated GE runtime process. Check the log for
`[BGE_FIAS] layer0 replacement=success` and
`[BGE_FIAS] post_opt type=FusedInferAttentionScore input0_dtype=1 ...
output_dtype=1`. `1` is GE `DT_FLOAT16`; the mask remains `DT_BOOL` (`12`).
Create dynamic K/V inputs at IR indexes 1 and 2 with
`create_dynamic_input_byindex_*`; appending them at 29/30 failed during
lowering. Avoid full `DUMP_GE_GRAPH` on this model because it repeatedly
serializes large weights. The pass's optional `BGE_FIAS_DUMP_STRUCTURE=1`
dump omits weights. Do not deploy into global `opp/vendors` yet.

The runner's `run.sh` sets `TMPDIR` under the worktree; for long checkout paths
that can exceed the Unix socket path limit used by CANN's Python manager. For
this isolated experiment, launch the pipeline with a short temporary directory:

```sh
mkdir -p /tmp/bge-fias-tmp
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')"
export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
export TMPDIR=/tmp/bge-fias-tmp
export BGE_FIAS_PASS_SO="$PWD/tools/bge_fias_pass/libbge_fias_pass.so"
python3 -m core.pipeline infer --config models/bge/config.yaml \
  --device <idle-logical-device> --case-ids case-0000
```

The isolated runner injects `LD_PRELOAD` only into its native GE subprocess,
then clears it before CANN's Python compiler workers start. Preloading into
the parent Python pipeline makes its forkserver fail.

Only run after checking that both chips of the physical NPU are free and
holding the normal device-job lock. The five supplied business cases passed
CPU full-score thresholds and expected Top10 IDs, order and scores (dense IDs
are one-based). A synthetic mask `[1,1,1,1,0,0]` also passed against CPU and
preserved Top10 order. With an all-zero mask, both fused and unfused GE
`force_fp16` paths produced nonfinite scores, while CPU FP32 remained finite.
This FP16 route requires at least one visible key token per row. The pass is
still a one-layer experiment; no performance gain has been established.
