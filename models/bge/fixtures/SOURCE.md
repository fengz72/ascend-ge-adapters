# Vendored fixtures: BGE business samples and references

These three files are byte-identical copies taken from the `materials/` directory
that sits next to the 100-machine optimization workspace
(`/export/home/ext.hejun23/workspace/bge-opt-100/materials/`).  They are vendored
so `models/bge/run.sh` no longer depends on a sibling `materials/` directory.

| File | Bytes | SHA-256 |
|---|---:|---|
| `sample_inputs.json` | 27,675 | `66571de2fa8c6a97e52a196a957801ab09b429751ba9cfb94712bb32c660b194` |
| `expected_outputs.json` | 2,526 | `4765cc2b9dac9d2f523f9a684a81082d3b943caaff1fd2c4db90cfdf1b375384` |
| `cpu-golden.npz` | 205,362 | `a59f406eb5782926181963e94cb1f87e1728621dedc3d613f88a7317d8213f35` |

Do not edit these files in place: they are the frozen acceptance workload, and
the thresholds in `run.sh` are calibrated against exactly this input set.

## `sample_inputs.json`

The five business input samples (`case_id` 0..4 plus the eight input features;
`query_token_ids` is bound to the ONNX input `input_ids`).  Produced by the
business side and handed over unchanged; `model/generate_cases.py` imports it
into the `case-NNNN/*.npy` pool.  Its SHA-256 is also recorded in
`cpu-golden-manifest.json` (see below).

## `expected_outputs.json`

The business Top-10 reference (`top10_dense_id`, `top10_score`, `scores_len`
= 10112) for the same five samples, also produced by the business side.  Note
`dense_id = output index + 1`.  `model/verify.py` gates against it.

## `cpu-golden.npz` — a derived artifact

The CPU full-score reference used by the numeric gate: five FP32 `[1, 10112]`
`scores` arrays under the layout `model/compare.py` reads
(`case-NNNN/outputs/scores` plus `case-NNNN/metadata/alignment`).

It was produced by independent CPU inference, not by this repository: the raw
ONNX was run with **onnxruntime 1.26.0** (`CPUExecutionProvider` only,
`intra_op_num_threads=2`, `inter_op_num_threads=1`, sequential execution,
`graph_optimization_level=ORT_ENABLE_ALL`) over the five samples above, with
numpy 1.26.4, and the imported NPY inputs were cross-checked tensor-by-tensor
against the runner's pool.  This matches the baseline stage of the since-removed
``core`` framework (its ``worker.py:baseline`` -> ``native_onnx.py`` path, also
onnxruntime CPU FP32 inference over ``case-*``).

Regenerating it requires the raw ONNX (`BGE_MODEL_PATH`) and onnxruntime.  Both
the generating script and its per-file / per-tensor hashes **are** vendored, as
frozen acceptance evidence rather than as a runnable tool:

| Path | Tracked | Note |
|---|---|---|
| `docs/bge/evidence/20261026/acceptance/build_cpu_golden.py` | yes | 5,730 bytes |
| `docs/bge/evidence/20261026/acceptance/cpu-golden-manifest.json` | yes | 18,645 bytes |

The script is **not runnable as-is**: `build_cpu_golden.py:9-10` hardcodes
`ROOT = Path("/export/home/ext.hejun23/workspace/bge")` and
`OUT = ROOT/"work/bge-opt-20261026/acceptance"`, which are 100-machine paths that
do not exist in this checkout.  Regenerating means copying it and repointing
those two constants at this repository's `BGE_MODEL_PATH` and `fixtures/`.

The manifest's hashes still line up with what is vendored here: it records
`a59f406eb5782926181963e94cb1f87e…` for `cpu-golden.npz` (matches the file), the
model SHA-256 `2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0`,
and the two JSON hashes above.  **Regenerate `cpu-golden.npz` if the ONNX model or
the five samples ever change**; otherwise the numeric gate compares against a
stale reference.  When you do, also diff the graph's `graph.input` order against
`model/generate_cases.py:DTYPES` (see "Not vendored: the ONNX graph" below).

## Not vendored: the ONNX graph

The raw ONNX export is 1.35 GB and is deliberately **not** in this repository.
`run.sh` takes it from the `BGE_MODEL_PATH` environment variable and hands it
straight to `ge_execute`, which parses it through GE's `aclgrphParseONNX`
(inline weights included). Its SHA-256 is
`2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0`.

There used to be a repacking step (`model/build_graph.py`) that rewrote it into
`work/build/model.onnx` + `weights.data` with a `ge_adapters_io` interface
embedded, producing the derived artifact `5e641362d99c9a1afb6dfce70f47b4ac07f1a7c88272fdfc791ea04ae84f4822`.
It was removed after measuring that the raw export yields **bit-identical**
outputs (5/5 cases, `max|diff| = 0.000e+00`) with identical fusion-pass
behaviour. Records citing `5e641362…` refer to that deleted derived artifact.

**Replacing the model is a manual interface check.** `ge_execute` binds inputs
positionally and validates nothing, and the eight input names plus their order
are hardcoded in `model/generate_cases.py:DTYPES` (the output name `scores` and
shape `(1, 10112)` in `model/verify.py`). A new export with a different input
list or order will fail the cosine gate rather than report the mismatch, so when
regenerating `cpu-golden.npz` for a new model, also diff the graph's
`graph.input` order against `DTYPES`.
