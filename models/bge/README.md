# BGE business ONNX on GE Session

This adapter runs the ONNX file declared by `source.ref`; the current configuration
uses `../materials/model.onnx` relative to the `runner/` root. Keep `runner/` and
`materials/` as sibling directories when moving the project.
Earlier 99-machine configurations used `BGE_MODEL_PATH` in that field. The source
file is left unchanged. Its eight inputs include text, behavior history, and
category statistics; the business output is `scores: [batch_size, 10112]`.

## Current 100-machine configuration and performance

The YAML selects GE Session prediction mode (`graph_run_mode=0`), internal
`force_fp16`, 10 instances, and the `12|24` AIC/AIV per-operator limit.
The external floating input/output contract remains FP32. The original ONNX
keeps all 30 Slice nodes; no Slice-to-Gather rewrite or custom fusion plugin is
used. The user-selected 10-instance configuration passed 2009 full-output
five-case requests and a 600-second random-case stability run. These checks and
the initial formal rounds are in the [10-instance acceptance report](../../docs/bge/analysis/100-fp16-ten-selection-20261027.md).

The later 8/10/12-instance comparison used logical device 2 on the same
100-machine container. Each cell below is the median of three formal 60-second
rounds, including H2D, GE completion synchronization, and full D2H. All 18
rounds had zero request errors and ran without profiling or output capture.
These 2026-10-27 numbers belong to binary SHA256 `8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d`.

| Instances | Fixed QPS | Fixed p99 (ms) | Five-case random QPS | Random p99 (ms) |
|---:|---:|---:|---:|---:|
| 8 | 807.326 | 19.718 | 861.592 | 13.505 |
| **10 (current)** | **884.177** | **14.262** | **903.582** | **19.285** |
| 12 | 851.169 | 33.053 | 862.495 | 35.958 |

The 2026-10-28 dtype-alias fix rebuilt `ge_runtime` (SHA256
`735a3532ecb131de203200a328835af0d58bcddf27f9572eb997b5d6808e43f4`).
This build passed 22,274 five-case full-output requests with zero failures.
Its one-round fixed/random checks yielded 849.606/861.960 QPS with p99
24.404/22.948 ms. These single rounds do not replace the old three-round
medians or establish the new build's stable throughput. See the
[review-fix report](../../docs/bge/analysis/100-runner-review-fix-20261028.md).

For the fixed case, 10 versus 8 instances gained 9.519%, above the 6.670%
within-run variability threshold. In the random scenario, 10 instances had
one valid low round at 730.640 QPS; its observed 4.874% median gain is below
the 19.405% variability threshold. Twelve instances showed no stable gain and
have not completed independent full-output and stability acceptance. Keep the
user-selected 10 instances, without claiming that random-case or production
traffic throughput is optimal. There is no business p99 limit. See the
[per-round comparison](../../docs/bge/analysis/100-fp16-scale-recompare-20261027.md)
for QPS, p99, resource samples, and the anomalous round.

For a new managed fixed-case formal run, use a fresh job name in the **project
root**; omit `--case-id case-0000` for five-case random replay:

```bash
cd /export/home/ext.hejun23/workspace/bge-opt-100
python3 tools/run_one.py --formal --config forcefp16-half10-dtypefix --timeout 500 \
  my-fp16-fixed10 -- --instances 10 --cores '12|24' \
  --precision-mode force_fp16 --seconds 60 --max-seconds 300 \
  --requests 10000 --seed 2053 --case-id case-0000
```

`tools/run_one.py` prepares the plan, checks model and configuration identity,
acquires the physical-device lock, checks occupancy before and after the run,
and verifies the formal request statistics. The performance path does not
capture every output; run full-output pressure verification separately before
accepting a new instance count or model variant. The seven-stage `bench` scope
is `h2d_desc_execute_sync` and omits per-request D2H, so it must not be
compared with these full-transfer QPS values.

## Historical seven-stage reproduction on machine 99

The following commands describe the earlier 2026-10-22 validation on machine
`99`, container `hejun-cann9-bge`, repository
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

`config.yaml` selects `ge_session`, `Ascend910_9382`, original FP32 graph/IO,
`force_fp16` internally, 10 instances, and a `12|24` AIC/AIV budget. The replay
CLI takes instances, precision and cores from this YAML unless explicitly
overridden. Its warmup and time-window parameters remain separate. A core budget
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

The default input pool is imported from `../materials/sample_inputs.json`; no random
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
- The current `../materials/expected_outputs.json`, via `verification.script`. For every
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

The 2026-10-22 checks establish execution and numerical agreement on the
supplied cases for the historical FP32 mode-0 run. They alone do not establish
business ranking quality, production performance, all input shapes, FP16
behavior, or ATC/OM deployment. The later 100-machine `force_fp16` pressure,
stability, and performance results are linked above. The seven-stage `bench`
default remains a six-call smoke check, not the formal full-transfer QPS test.

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
compatibility, and insufficient output capacity. All five passed on device 0
in the earlier 99-machine validation:

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

## Full-transfer replay plan preparation (100 machine)

`replay.py` reads the five persisted BGE cases and an already exported ONNX
with external `weights.data`. It creates a new `inputs.bin` and `plan.json`
without using an NPU. The original inline-weight model is not loaded for this
runtime-only step. Keep the small exported ONNX and `weights.data` together.
The current YAML supplies 10 instances, `force_fp16`, and `12|24` unless the CLI
explicitly overrides them. `--case-id case-0000` creates a real one-case pool;
omitting it selects uniform five-case random replay with replacement. Each
request still performs H2D, GE completion synchronization, and full D2H.

From the 100-machine **project root**, an inspection-only preparation example
is:

```bash
cd /export/home/ext.hejun23/workspace/bge-opt-100
python3 runner/models/bge/replay.py --prepare-only \
  --config runner/models/bge/config.yaml --device 2 \
  --graph materials/exported/model.onnx \
  --output-dir work/prepared/my-bge-fixed10-plan \
  --instances 10 --aicore-num '12|24' --precision-mode force_fp16 \
  --case-id case-0000 --min-seconds 60 --max-seconds 300 \
  --min-requests 10000 --seed 2053
```

The preparation directory must be new. This command writes a plan for review;
`tools/run_one.py` in the project root is the managed execution entry used for
formal measurements. It creates its own plan, verifies original input and
binary identities, locks the physical device, checks both chips before and
after the run, and validates request events. Use a fresh job name for every
run. Device 2 was empty during the 2026-10-27 comparison, but availability
must be checked again before a new run. The per-job timeout remains finite;
there is no current cumulative device-time limit.

`--capture` records each request's full output and is for independent
correctness pressure checks. Formal QPS runs disable capture and profiling.
Any new instance count, precision strategy, graph, or fusion plugin needs its
own full-output correctness and stability acceptance. The separate fixed and
random workloads require separate baselines and reports; a case filtered from
random-run logs is not a fixed-case benchmark. The accepted 10-instance
pressure and stability checks and all 18 recent performance rounds are linked
from the [BGE delivery index](../../docs/bge/README.md).
