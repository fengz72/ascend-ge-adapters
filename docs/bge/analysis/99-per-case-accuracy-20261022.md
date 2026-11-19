# BGE 99 per-case accuracy and seven-stage pipeline (2026-10-22)

99 阶段两次独立验证的原始记录，自 `docs/bge/acceptance.md` 附录原样迁入：业务五例逐 case 误差与七步流水线逐 case 误差。数字、口径与英文原文均不改写。

## 业务样例独立验证（2026-10-22）

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
|---|---:|---:|---:|
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

## 七步样例流水线（2026-10-22）

The default `bash models/bge/run.sh run --device 0` completed all seven stages
with the supplied five samples. Both CPU full-score comparison and the external
Top-10 business checks passed. The original graph and weights are preserved
(30 Slice, zero Range); runtime logs confirm prediction mode 0.

| Case | CPU/GE cosine | CPU/GE relative L2 | CPU/GE max absolute error | Expected Top-10 max absolute error | IDs and order |
|---|---:|---:|---:|---:|---|
| case-0000 | 0.999999978438 | 0.000209465401 | 9.848922491e-05 | 6.869653702e-05 | Exact match |
| case-0001 | 0.999999958158 | 0.00028989152 | 0.0001145526767 | 9.422024536e-05 | Exact match |
| case-0002 | 0.999999952284 | 0.0003110775458 | 0.000119931995 | 7.459005737e-05 | Exact match |
| case-0003 | 0.999999922195 | 0.0003950748066 | 0.000137783587 | 8.469063377e-05 | Exact match |
| case-0004 | 0.999999905871 | 0.0004473723288 | 0.000215953216 | 9.572097778e-05 | Exact match |

Complete stage results and all 50 rank comparisons are in
`work/bge/results/20261022T071838-a6374be8/result.json`; the same directory
contains `graph_integrity.json` and `infer-runtime.log`. Benchmark input pool
size is 5 (six smoke calls, zero execution errors). The ordinary Python suite
passed 88 tests and skipped five opt-in hardware probes; the C++ contract test
also passed.
