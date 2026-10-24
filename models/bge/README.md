# BGE business ONNX on GE Session

This adapter runs the ONNX file selected by `BGE_MODEL_PATH`. The
source file is left unchanged. Its eight inputs include text, behavior history,
and category statistics; the business output is `scores: [batch_size, 10112]`.
The source location is declared as `source.ref`, the same field used by
Transformers model directories.

## Reproduce

Validated location: machine `99`, container `hejun-cann9-bge`, repository
`/export/home/ext.hejun23/workspace/bge/ascend-ge-adapters`, branch
`codex/seven-stage-framework` based on commit `59050e3`.

```bash
cd /export/home/ext.hejun23/workspace/bge/ascend-ge-adapters
export BGE_MODEL_PATH=/export/home/ext.hejun23/workspace/bge/model.onnx
cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build -j4
bash models/bge/run.sh run --device 0
```

Individual stages use the same entry point:

```bash
bash models/bge/run.sh data
bash models/bge/run.sh baseline
bash models/bge/run.sh prepare
bash models/bge/run.sh build --device 0
bash models/bge/run.sh infer --device 0
bash models/bge/run.sh compare
bash models/bge/run.sh bench --device 0
```

Set `BGE_MODEL_PATH` in the invoking shell before `data`, `baseline`, `build`,
or `run`; an unset or empty variable produces a configuration error.
The `prepare`, `infer`, `compare`, and `bench` stages do not
require this variable.

`run.sh` matches CANN's embedded Python to the container's invoking interpreter
using `PYTHONHOME` and its library directory. Without this, the system libpython
can be selected and GE initialization fails with `No module named numpy` even
though the invoking Python has NumPy installed.

## Configuration and compatibility repair

`config.yaml` selects `ge_session`, `Ascend910_9382`,
original FP32 graph/IO, `must_keep_origin_dtype`, one instance and a `4|8`
AIC/AIV budget. The budget
does not provide exclusive access to the shared device.
Both `reference.dtype` and `build.dtype` explicitly specify `float32` for
floating-point data. Integer feature inputs retain their original INT64 types.
GE Session defaults to prediction mode (`graph_run_mode=0`), so the model YAML
does not need to specify it.

In mode 1, the unmodified ONNX graph triggers SIGSEGV in CANN 9.0's
`fe::ConstToAttrStridedSliceV2Pass::MakeConstNode` during `CompileGraph`.
Debug logs confirm `ConstToAttrStridedSliceV2Fusion` is forbidden from being
disabled. Mode 0 compiles and executes the original graph successfully.

The exporter preserves all 30 original Slice nodes, with zero Range nodes.
Validation compares all 3137 nodes, 400
initializers and input/output declarations against the original ONNX. Only
framework IO metadata and external weight storage are added by export. Dynamic
batch, text length and history length are preserved.

The GE runtime now reads `GetCompiledGraphSummary().IsStatic()` after compilation.
Static compiled graphs require caller-provided output tensors: each instance
allocates buffers using the compiled shapes and dtypes, validates declared
capacities, reuses the buffers for every execution, and frees them once during
cleanup (including partial setup failures). Dynamic compiled graphs continue to
receive outputs allocated by GE. This decision follows the compiled graph,
rather than assuming every mode-0 graph is static. BGE is dynamically compiled;
the original rank-one Slice regression is statically compiled and exercises the
new caller-allocation path.

The runtime bridge now writes native stdout/stderr directly to its log file.
This reports native crashes immediately instead of waiting for compiler children
that inherited an output pipe. Errors include the native process exit code.

## Validation scope

The default input pool is imported from `../sample_inputs.json`; no random
features are generated. `query_token_ids` is bound to ONNX `input_ids`, every
sample receives a batch dimension, integer features are INT64 and `stat_ratio`
is FP32. The five supplied cases are:

| Case | Batch | Text length | History length |
|---|---:|---:|---:|
| case-0000 | 1 | 6 | 128 |
| case-0001 | 1 | 9 | 128 |
| case-0002 | 1 | 8 | 128 |
| case-0003 | 1 | 13 | 128 |
| case-0004 | 1 | 8 | 128 |

`data.params.samples_file` selects the input JSON. Paths in BGE script parameters
are relative to the repository root unless absolute. The `data` stage validates
all samples before replacing the persisted pool and removes obsolete case
directories. Other stages read the persisted NPY files without regenerating them.

`baseline` still runs CPU ONNX Runtime on the original model and saves full
10112-dimensional golden score vectors. `compare` checks both:

- Full CPU/GE scores using the configured cosine, relative L2 and absolute error
  thresholds.
- The current `../expected_outputs.json`, via `verification.script`. For every
  selected case, the ten highest GE scores are sorted descending (stable ties),
  their array indices are incremented by 1, and all IDs and their order must
  match `top10_dense_id` exactly. Each score must satisfy absolute error <=
  `verification.params.atol` (default `0.001`), with zero relative tolerance.

Either check failing makes `compare` and `run` fail. Per-rank expected/actual IDs,
scores and absolute errors are recorded in
`accuracy.cases.<case_id>.business.ranks` in the normal result JSON.
With `case_ids: all`, expected and archived case IDs must match exactly; explicit
subsets compare only the selected cases. `scores_sha256_16` is not used as a
floating-point tolerance check.

Changing inputs requires rerunning `data`, `baseline` and `infer`. Changing only
the expected JSON or score tolerance requires rerunning `compare`; it reads the
current file directly and does not need the original model, input JSON or NPU.

These checks establish execution and numerical agreement on the supplied cases.
They do not establish business ranking quality, production performance, all input
shapes, FP16 behavior, or ATC/OM deployment. Benchmark defaults are deliberately
only a six-call smoke check on a shared NPU.

Imported inputs live under `models/bge/io/inputs`; the exported ONNX and its
external `weights.data`, CPU golden, GE outputs, per-stage logs and result JSON
live under the framework's default local directory `work/bge`; no `work_dir`
entry is needed in the model YAML. Keep the exported ONNX and `weights.data`
together. Historical mode-0 validation results remain under `work/bge-mode0`:
`graph_integrity.json` records graph/weight preservation and
`ranking_validation.json` records CPU/GE rankings.
Reimporting inputs removes case directories dropped from the sample JSON,
so reducing the supplied pool cannot retain old cases in later stages.

Regression checks:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
ctest --test-dir runtime/build --output-on-failure
```

Five opt-in device tests cover an original static rank-one Slice, repeated
execution and two instances, mixed float32/int64 outputs, dynamic batch, mode-1
compatibility, and insufficient output capacity. All five passed on device 0:

```bash
export PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')"
export LD_LIBRARY_PATH="$PYTHONHOME/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 TE_PARALLEL_COMPILER=2
GE_TEST_DEVICE=0 GE_TEST_TARGET=Ascend910_9382 \
  python3 -m unittest discover -s tests -p test_ge_session_outputs.py -v
```

The ordinary suite skips these five hardware tests unless
`GE_TEST_DEVICE` is set. The C++ contract test also passes. CPU regressions check
that export preserves original Slice nodes, opsets 11/12/13/14 and runtime
parameters, and that importing the input pool removes stale cases. Business
regressions cover the ID offset, ordered IDs, score tolerance, missing references,
nonfinite values, reading updated expected files, and pipeline failure propagation.

## Appendix: Provided business sample validation (2026-10-22)

The five supplied `../sample_inputs.json` cases were executed individually on
GE Session device 0 in prediction mode (`graph_run_mode=0`), using FP32 and
`must_keep_origin_dtype`. The original graph retains all 30 Slice nodes and
zero Range nodes. Node, initializer value, interface and opset signatures match
between the source and export; the original source file SHA-256 is unchanged.

Expected results come directly from `../expected_outputs.json`. Candidate IDs
are `argsort(-scores[0])[:10] + 1`. All 50 IDs match in order. Scores pass with
absolute tolerance `1e-3` and zero relative tolerance; all observed errors are
also below `1e-4`. The expected scores are supplied to six decimal places.

| Case | Top-10 IDs and order | Maximum absolute score error | Mean absolute score error |
|---|---|---:|---:|
| 0 | Exact match | 6.869653702e-05 | 2.606389084e-05 |
| 1 | Exact match | 9.422024536e-05 | 3.696981087e-05 |
| 2 | Exact match | 7.459005737e-05 | 3.298873844e-05 |
| 3 | Exact match | 8.469063377e-05 | 4.139411488e-05 |
| 4 | Exact match | 9.572097778e-05 | 5.397958374e-05 |

Artifacts: `work/bge-provided-samples/run-20261022-gemode0/` contains
`top10_comparison.json` (all 50 rank comparisons), `outputs.npz` (all five
full score vectors), `graph_integrity.json`, resolved configuration, input
mapping, and `infer-stage/infer-runtime.log`. The reusable validation script is
`work/bge-provided-samples/validate_samples.py`. No synthetic samples were used.

## Appendix: Integrated sample pipeline (2026-10-22)

The default `bash models/bge/run.sh run --device 0` completed all seven stages
with the supplied five samples. Both CPU full-score comparison and the external
Top-10 business checks passed. The original graph and weights are preserved
(30 Slice, zero Range); runtime logs confirm prediction mode 0.

| Case | CPU/GE cosine | CPU/GE relative L2 | CPU/GE max absolute error | Expected Top-10 max absolute error | IDs and order |
|---|---:|---:|---:|---:|---|
| case-0000 | 0.999999978438 | 0.000209465401 | 9.848922491e-05 | 6.869653702e-05 | Exact match |
| case-0001 | 0.999999958158 | 0.00028989152 | 0.0001145526767 | 9.422024536e-05 | Exact match |
| case-0002 | 0.999999952284 | 0.0003110775458 | 0.0001199319959 | 7.459005737e-05 | Exact match |
| case-0003 | 0.999999922195 | 0.0003950748066 | 0.000137783587 | 8.469063377e-05 | Exact match |
| case-0004 | 0.999999905871 | 0.0004473723288 | 0.000215953216 | 9.572097778e-05 | Exact match |

Complete stage results and all 50 rank comparisons are in
`work/bge/results/20261022T071838-a6374be8/result.json`; the same directory
contains `graph_integrity.json` and `infer-runtime.log`. Benchmark input pool
size is 5 (six smoke calls, zero execution errors). The ordinary Python suite
passed 88 tests and skipped five opt-in hardware probes; the C++ contract test
also passed.
