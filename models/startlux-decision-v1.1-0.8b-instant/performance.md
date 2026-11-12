# StartLux 决策模型性能记录（P8 唯一正式实测）

正式统一单 AIR（5 输入 1 输出）已于 2026-11-01 在空闲卡上完成唯一正式基准
（job `p8-20261101T025001-1672406`，02:50–02:53 UTC，终态 bench-done、无污染；
主 agent 独立复核 passed：
`work/startlux-cannbot/review-p8-real-independent.json`/`.md`）。本文仅保留该
最终结果、口径与原始证据；历史 P5 纯文本 3IO 诊断准备记录与监控实现细节见
归档快照（`work/startlux-archive/cleanup-20261102/docs/performance.md`）。

## 运行登记

被测 = 正式统一单 AIR `work/startlux-decision/build/model.air`
（1,907,534,423 bytes，sha256
`c1f8816ee25e03119b2522e55adb49ebf42a88692791b179920f5f60221edba1`）；runtime
`runtime/build/ge_runtime`（639,952 bytes，sha256
`692996b5df8a7ed8fdebdc365cdaa9383b6b538868ecf3e019303930d178eecc`）；逻辑设备
1（`ASCEND_RT_VISIBLE_DEVICES=1`，进程内 device 0），单实例，
`graph_run_mode=0`、`precision_mode=must_keep_origin_dtype`，LD_PRELOAD 本地
chunk proto 兼容库。

| job id | 日期 | 逻辑设备 | 用例池 | instances | warmup | iterations | seed | errors | first_hit |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| p8-20261101T025001-1672406 | 2026-11-01 02:50–02:53 UTC | 1 | 11 正式 5IO 用例 | 1 | 32 | 2048 | 42 | 0 | 0 |

2048 请求全部成功，2234 readout 行（B=2 的 mixed 用例抽中 186 次，
2048+186）；true tokens 361889 / padded tokens 571904。

## 最终结果

- wall 74903.106 ms；**27.3419903308 graph executions/s**；**29.8251984370
  readout rows/s**（B 加权）；true tokens/s 4831.428485755985、padded
  tokens/s 7635.250799880048、token 效率 0.6327792776410027（池内分布加权）。
- e2e 延迟（nearest-rank，n=2048）：**P50/P95/P99 = 35.903/43.234/43.293 ms**
  （avg 36.57177587890625）；exec P50/P95/P99 = 35.715/43.045/43.095 ms（avg
  36.37971142578125）；组件均值 h2d 0.19019970703125 ms、desc
  0.000220703125 ms。
- 逐 case e2e（n 与百分位取自主复核独立 JSON）：

| case | n | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|
| text-choice_short-r0 | 201 | 35.899 | 35.953 | 35.991 |
| text-score_short-r0 | 174 | 35.899 | 35.955 | 36.002 |
| text-noul_short-r0 | 207 | 35.900 | 35.963 | 36.138 |
| text-allow_none-r0 | 171 | 35.900 | 35.963 | 36.269 |
| text-wide_27-r0 | 184 | 35.898 | 35.977 | 36.221 |
| text-wide_27-r1 | 176 | 35.900 | 35.960 | 36.068 |
| text-wide_27-r2 | 202 | 35.899 | 35.967 | 36.038 |
| image-red-r0 | 176 | 35.903 | 35.975 | 36.152 |
| image-blue-r0 | 181 | 35.899 | 35.960 | 36.134 |
| image-two-a-b-r0 | 190 | 35.897 | 35.970 | 36.080 |
| mixed-text-image-b2-r0 | 186 | 43.238 | 43.309 | 43.489 |

- 稳态窗口外：compile 27018.0 ms / load 7235.65 ms / warmup 1404.467 ms
  （32 次，含特化 310.746 ms）；HBM 152.40625→3882.40625 MB 为 setup 前后两
  时点**阶段采样，非连续峰值**。

## 口径（如实，不得改述）

- 计时段为 warm graph **h2d+desc+execute+sync**；**不含 D2H**（bench 模式不
  回拷输出）与官方 CPU 预处理（在计时窗口外）；**graph executions/s 不是完整
  业务 QPS**；readout rows/s 不是原始问题/s；token 速率为池内分布加权。
- runtime performance JSON 的 `samples` 是每请求行数统计块（avg 1.0908203125、
  min 1、max 2、P50 1、P99 2）；readout 行总数 2234 由 requests.csv 逐请求
  samples 加权复算得出，不得误读为行总数。
- 抽样：std::mt19937_64(42) %% 11 闭环有放回；plan 精确逐 case 抽中计数与
  runtime/CSV 三方一致（rows_match、draw_counts_match，first_hit=0，
  integrity_violations=[]）。
- 留痕限制：benchmark plan 实际保存在 result.json.plan 字段内，
  result.json.artifacts.plan_json 所指独立文件在磁盘上不存在，不影响 raw
  CSV / native 结果核验；docker-exec 退出码 0 由固定监控协议（非零退出拒绝
  分支 + bench-done 终态）联合证明，未单独持久化数值退出字段。

## 证据路径

- 原始 run：
  `work/startlux-validation/p8-idle-performance/runs/20261101T025002-45aba096/`
  （gate-20261101T025013-a9dafbe2/ 与 bench-20261101T025109-558a6522/，
  含 requests.csv、result.json、summary.json、report.md、inputs.bin、
  outputs.bin、bench-runtime.log）。
- 主 agent 独立复核：`work/startlux-cannbot/review-p8-real-independent.json` /
  `.md`（raw CSV sha256
  `729b9d53957d5b4b7bd4bcdaa9b4db2b11b9b399390d0c05801fc0944c848c4f`、B4 raw
  sha256
  `2b69e5863a086b54238d212dd74417c03dc2b82df1dc710aed1d8706f8b81bea`）。
- host 侧（100 主机 UID1130）：`~/.local/state/startlux-air-monitor/`
  （state.json、monitor.log、jobs/p8-20261101T025001-1672406/）。监控单次任务
  已完成，不再自动重启。

## Appendix: standalone-flow validation measurement (2026-11-05)

The standalone shell flow (docs/standalone-runtime-plan.md, implemented
2026-11-05) was validated end-to-end by `run.sh` launched from a different
working directory on an idle card (Chip Logic ID 4, AICore 0%), same CANN /
container Python / runtime binary.  This appendix records that measurement;
the P8 job above remains the formal benchmark of record.

| item | value |
|---|---|
| entry | `run.sh` full flow (env check -> inputs -> golden -> prepare -> runtime -> export -> infer -> compare -> business -> idle confirm -> bench) |
| device | ASCEND_RT_VISIBLE_DEVICES=4, in-process device 0, idle card |
| AIR | re-exported by `export_air.py`, 1,907,534,423 bytes (identical to the formal AIR size), sha256 `c73c7b622d19e5f81f3683ce29cb8f0441759a23ef9673682bdf2d67712faf05` (exports are not byte-reproducible; the embedded IO contract was verified byte-identical to the formal AIR via runtime `read_io`) |
| precision | 12 cases / 16 rows all pass, worst cosine 0.999995, worst relative L2 0.004004; b4-long 4/4 rows; business gate 11 cases, worst active-probability max abs 0.005507 |
| bench plan | instances 1, warmup 32 (per-instance preheat, not consumed from the measurement sequence), 2048 total measured requests, seed 42, closed loop |
| wall | 76,349.4 ms; warmup 32 runs / 1,419.6 ms; load 7,179.2 ms and specialize 307.4 ms outside the timed window |
| graph executions/s | 26.824065 (a same-parameter repeat measured 27.133183; P8 formal 27.341990 -- within normal run-to-run variance) |
| readout rows/s | 29.260235 (2,234 weighted rows / 2048 requests from requests.csv `samples`; repeat 29.597427; P8 formal 29.825198) |
| e2e latency (n=2048) | P50/P95/P99 = 36.589/43.808/43.929 ms (avg 37.277441) |
| components (avg) | exec 37.072920 ms; h2d 0.202222 ms; desc 0.001000 ms |
| integrity | 2,048 requests, 0 errors; every `first_hit` = 0 (both distinct shapes [B1,S256] and [B2,S256] covered inside warmup; no compile/first-execution pollution in the measure phase) |
| timing scope | warm-graph h2d + descriptor update + execute + sync, no D2H, official CPU preprocessing outside the window -- same scope as P8; graph executions/s is NOT a full business QPS |

Evidence (machine-local, gitignored `work/`):

- requests + result: `models/startlux-decision-v1.1-0.8b-instant/work/bench/{request.json,result.json,requests.csv}`
- precision: `models/startlux-decision-v1.1-0.8b-instant/work/infer/{comparison.json,business.json,result.json}` and `work/golden/`
- full stage logs: `models/startlux-decision-v1.1-0.8b-instant/work/logs/`

## Appendix: mega fused-GDR route measurement (2026-11-05, route rolled back)

The GDR route was temporarily switched to the MegaGdnPrefillV2 fused operator
(`gdn_route=mega`; decision record in validation.md 2026-11-05) and rolled
back the same day: the fused operator is internally 128-aligned and this
model's shapes are too small for the fusion to pay off (see the numbers
below).  This appendix keeps the measurement as investigation evidence only;
the route is NOT delivered.  Same standalone flow, same CANN / container
Python / runtime binary, idle card (Chip Logic ID 3, AICore 0%); the operator
dependency was `work/deps/mega_gdn_prefill` (loading via activator import +
ASCEND_CUSTOM_OPP_PATH, no LD_PRELOAD).  The P8 job above remains the formal
benchmark (native route).

| item | value | native-route reference |
|---|---|---|
| AIR | re-exported by `export_air.py`, 1,905,954,657 bytes, sha256 `201432d308e78e5f638a99fc5cc52fa22eeb4ecf62aca77b63dbba3eb6d51177` (exports are not byte-reproducible; embedded IO verified by runtime `read_io`) | 1,907,534,423 bytes |
| precision | 12 cases / 16 rows all pass, worst cosine 0.999981, worst relative L2 0.010130 | 0.999995 / 0.004004 |
| business gate | 8/11: the three image cases fail `prob_max_abs` 0.012299/0.013162/0.014652 > 0.01 (top1 consistent, margins 0.256/0.392/0.574) -- the inherent mega-route precision already recorded at commit `eb8de21` | 11/11, worst 0.005507 |
| bench plan | instances 1, warmup 32 (2,057.95 ms), 2048 measured requests, seed 42, closed loop | same |
| wall | 99,682.353 ms | 76,349.4 ms (P8 74,903.1) |
| graph executions/s | 20.545261 (2,048 requests, 0 errors) | 26.824065 (P8 27.341990) |
| readout rows/s | 22.411188 (2,234 weighted rows) | 29.260235 (P8 29.825198) |
| e2e latency (n=2048) | P50/P95/P99 = 46.843/48.5/65.705 ms (avg 48.670) | 36.589/43.8/43.929 (avg 37.277) |
| components (avg) | exec 48.482 ms; h2d 0.186093 ms; desc 0.000813 ms | exec 37.072920; h2d 0.202222 |
| per-case e2e P50 (ms) | text B1 46.79–46.82; image B1 46.79–46.83; mixed B2 65.224 | text/image 35.9–36.6; mixed 43.238 |

- Throughput is **~23% below the native route** (20.545 vs 26.824–27.342
  executions/s); the B2 mixed case is disproportionately slower (P50 65.2 vs
  43.2 ms, +51%) while B1 cases are +28% (the fused op processes the full
  padded width through the 128-chunk pipeline per row).
- Integrity: 2,048 requests, 0 errors; every `first_hit` = 0 (both distinct
  shapes covered inside warmup); HBM stage samples 142.4 -> 3,870.4 MB.
- Evidence paths: same `work/bench/`, `work/infer/`, `work/logs/` locations as
  the standalone appendix above (bench run of 2026-11-05 06:36–06:38,
  `run_id` inside `result.json`).

## Appendix: fp32-Conv2D causal_conv4 replacement measurement (2026-11-05)

The causal 4-tap short conv inside the native route was replaced by ONE fp32
depthwise Conv2D node (decision record in validation.md 2026-11-05; the
bf16-input Conv1D/Conv2D path miscompiles this depthwise configuration while
the fp32-input node is correct -- GE graph-mode small-graph gates at
C=6144/S256/S2048, relL2 1.41 vs 2.41e-3 against an fp64 golden, and 27-45%
faster per conv).  Same standalone flow, same CANN / container Python /
runtime binary, idle card (Chip Logic ID 3, AICore 0%), LD_PRELOAD chunk
proto loading unchanged.  The P8 job above was measured with the previous
shift-chain conv.

| item | value | shift-chain conv (2026-11-05 standalone) | P8 formal (native, chip 1) |
|---|---|---|---|
| AIR | re-exported by `export_air.py`, 1,907,534,423 bytes (same size; exports are not byte-reproducible) | 1,907,534,423 | 1,907,534,423 |
| precision | 12 cases / 16 rows all pass, worst cosine 0.999995, worst relative L2 0.004004 -- identical to the pre-replacement values | identical | 0.9999985 / 0.00303 |
| business gate | 11/11, worst active-probability max abs 0.005507 -- identical to the pre-replacement value | 11/11, 0.005507 | 0.0055074 |
| bench plan | instances 1, warmup 32 (1,323.7 ms), 2048 measured requests, seed 42, closed loop | same | same |
| wall | 69,982.0 ms | 76,349.4 ms | 74,903.1 ms |
| graph executions/s | **29.264686** (2,048 requests, 0 errors) | 26.824065 | 27.341990 |
| readout rows/s | 31.922514 (2,234 weighted rows) | 29.260235 | 29.825198 |
| e2e latency (n=2048) | **P50/P99 = 33.561/40.101 ms** (avg 34.168) | 36.589/43.929 (avg 37.277) | 35.903/43.293 |
| components (avg) | exec 33.958 ms | exec 37.072920 | exec 36.379711 |
| per-case e2e P50 (ms) | text/image B1 33.549–33.563; mixed B2 40.004 | text/image 35.9–36.6; mixed 43.238 | 35.9 / 43.2 |

- Net effect: **+9.1% executions/s and −3.0 ms e2e P50 vs the same-day
  shift-chain measurement on the same card** (+7.0% vs the P8 formal
  benchmark), zero precision or business-gate movement, zero dependency
  changes.  The conv micro-gain (0.05–0.25 ms x 18 layers) compounds with
  reduced GE elementwise-node scheduling in the full graph.
- Cross-card note: this run is Chip 3, the shift-chain standalone run Chip 4,
  the P8 job Chip 1 (same Ascend910_9382 SKU); the +9.1% is a same-day
  same-card-class comparison, well above the observed run-to-run variance
  (26.82 vs 27.13 / 27.34 on the old conv).
- Integrity: 2,048 requests, 0 errors; every `first_hit` = 0 (both distinct
  shapes covered inside warmup).
- Evidence paths: same `work/bench/`, `work/infer/`, `work/logs/` locations
  as the standalone appendix above (bench run of 2026-11-05 09:09–09:10).

## Appendix: FIA full-attention replacement measurement (2026-11-05, net regression)

The 6 bidirectional full-attention layers were switched from the eager-math
chain to the CANN `fused_infer_attention_score` operator (BNSD, native GQA
Q8/KV2/D256, per-row `actual_seq_lengths` with an in-graph int64 cast,
sparse_mode=0, no additive mask; projections / Q-K RMSNorm / partial RoPE /
sigmoid gate stay outside).  GE small-graph gates passed first (static +
dynamic B/S, D=256, mixed lengths: valid-prefix relL2 1.9e-3 -- 2.2e-3), then
the full standalone flow on an idle window (Chip 3; the host watchdog
intermittently kills NPU-idle processes, the run was taken in a recovery
window).

| item | value | conv2d-conv baseline (same day, chip 3) |
|---|---|---|
| precision | 12 cases / 16 rows all pass, worst cosine 0.999995, worst relative L2 0.006763 | 0.999995 / 0.004004 |
| business gate | 11/11, worst active-probability max abs 0.006332 | 11/11, 0.005507 |
| graph executions/s | **28.248243** (2,048 requests, 0 errors) | 29.264686 |
| readout rows/s | 30.813757 | 31.922514 |
| e2e latency (n=2048) | P50/P99 = 34.521/41.630 ms (avg 35.398) | 33.561/40.101 |
| per-case e2e P50 (ms) | text/image B1 34.47-34.55; mixed B2 41.010 | 33.55; 40.004 |

- Net effect: **-3.5% executions/s and +0.96 ms e2e P50 vs the conv2d-conv
  baseline** -- the PA kernel's short-sequence scheduling overhead plus the
  q/k/v contiguous copies and the output transpose outweigh the removal of
  the ~15-node eager-math chain per layer.  Precision also moves slightly
  worse (relL2 0.0068 vs 0.0040, business worst 0.0063 vs 0.0055, both still
  inside the gates).
- Recommendation: keep the eager-math full attention (roll the FIA switch
  back) -- it is both faster and numerically closer on this workload's
  short-sequence shapes; revisit FIA only if the workload moves to long
  sequences where PA's asymptotic advantage can materialize.
- Evidence paths: same `work/bench/`, `work/infer/`, `work/logs/` locations
  (bench run of 2026-11-05 13:09-13:11).

## Appendix: MatMul patch-embed replacement measurement (2026-11-06)

The vision patch-embed Conv3d was replaced by the mathematically identical
per-patch MatMul (`MatMulPatchEmbed`, model/vision.py -- stride == kernel
makes the conv a full connection per patch; weight/bias held by reference,
no copies).  Driver: the Conv3DV2 node was MTE1-saturated (mac 0.22, mte1
0.97) at a fixed 6854 us per request including pure-text requests (20% of
all AI time, work/prof-fia).  Same standalone flow, idle card Chip 3, with
the FIA full-attention switch rolled back (eager math) so this gate measures
the clean delivered configuration: fp32-Conv2D conv + eager-math attention +
MatMul patch-embed.

| item | value | conv2d-conv baseline (2026-11-05) |
|---|---|---|
| precision | 12 cases / 16 rows all pass, worst cosine 0.999995, worst relative L2 0.005749 | 0.999995 / 0.004004 |
| business gate | 11/11, worst active-probability max abs 0.007576 | 11/11, 0.005507 |
| graph executions/s | **35.989044** (2,048 requests, 0 errors) | 29.264686 |
| readout rows/s | 39.257580 | 31.922514 |
| e2e latency (n=2048) | **P50/P99 = 27.130/33.897 ms** (avg 27.784) | 33.561/40.101 |
| per-case e2e P50 (ms) | text/image B1 27.10-27.13; mixed B2 33.818 | 33.55; 40.004 |

- Net effect: **+23.0% executions/s and -6.43 ms e2e P50 vs the baseline**
  (P50 27.13 ms), matching the removed 6.9 ms Conv3DV2 task.  Profiling
  re-collection confirms: Conv3DV2 disappears from the operator table; the
  patch-embed now rides inside MatMulV2 (+~25 us/request; MatMulV2 total
  169.9 ms/40 requests vs 170.9 ms before) -- a ~270x reduction of that
  stage.  ChunkGatedDeltaRule becomes the largest AIC operator (11.0%).
- Precision moves slightly (relL2 0.0057 vs 0.0040, business worst 0.0076 vs
  0.0055 -- matmul vs conv accumulation order, ulp-level per patch), all
  inside the gates (11/11 business, 12/12 numeric).
- Evidence paths: `work/bench/`, `work/infer/`, `work/logs/` (bench of
  2026-11-06 03:29-03:31); profiling sessions `work/prof-fia/` (before,
  Conv3DV2 present) and `work/prof-matmul/` (after, Conv3DV2 gone).

## Appendix: official-examples case pool measurement (2026-11-06)

The bench pool was extended from the P8 formal 11 to 16 business cases by the
five official published example cases (StartLuxLabs/StartLux-Decision
examples @ main: the client.py / request.sh support ticket as a B3
one-request-three-questions case and as three B1 single-question cases, and
the image_request.py photo noul question on the red fixture; see
generate_inputs.py `_official_example_cases`).  Same delivered AIR (MatMul
patch-embed + fp32-Conv2D conv + eager-math attention), idle card Chip 3.

| item | value | 11-case pool (same AIR, 2026-11-06) |
|---|---|---|
| requests / rows | 2,048 / **2,432** (0 errors) | 2,048 / 2,234 |
| graph executions/s | 34.576967 | 35.989044 |
| readout rows/s | **41.060149** | 39.257580 |
| e2e P50/P99 (n=2048) | 27.661/38.439 ms (avg 28.918) | 27.130/33.897 |
| per-case e2e P50 (ms) | B1 cases 27.64-27.66; mixed B2 34.396; official-ticket-b3 38.386 | B1 27.10-27.13; mixed 33.82 |

- Precision / business gates on the 17-case precision pool: 17/17 numeric
  (worst cosine 0.999995, relL2 0.005749) and 16/16 business (worst
  active-probability max abs 0.007576) -- the official example cases
  themselves land at relL2 0.0012-0.0048 and prob max abs 0.0001-0.0063 with
  large margins and consistent top-1 (the B3 rows and the three split B1
  cases agree deterministically).
- Throughput moves -3.9% in executions/s because the B3 official case (3
  rows, e2e P50 38.4 ms) enters the equal-weight closed loop, while rows/s
  rises +4.6% (2,432 vs 2,234 weighted rows).  The per-request latency of
  every B1 case is unchanged.
- Evidence paths: `work/bench/`, `work/infer/`, `work/logs/` (bench of
  2026-11-06 06:4x).

## Appendix: official-only case pool measurement (2026-11-06)

The case pool was reduced to ONLY the five official published example cases
(user decision; the P8 formal 11 and the b4-long boundary case were removed,
together with their fixtures).  Same delivered AIR.  Idle card Chip 3.

| item | value |
|---|---|
| requests / rows | 2,048 / 2,916 (0 errors) |
| graph executions/s | 33.823975 |
| readout rows/s | **48.159527** |
| e2e P50/P99 (n=2048) | 27.138/38.335 ms (avg 29.562) |
| per-case e2e P50 (ms) | B1 cases 27.117-27.121; official-ticket-b3 38.214 |
| precision | 5 cases / 7 rows all pass, worst cosine 0.999998, worst relL2 0.004753 |
| business gate | 5/5, worst active-probability max abs 0.006282 |

- rows/s is the pool's own record (+17% vs the 16-case pool): the official
  distribution carries 1.42 rows per request on average (B3 ticket at 21.2%
  of the closed-loop draws); executions/s settles at 33.82 for the same
  reason.  Per-request latencies are unchanged from previous pools.
- Evidence paths: `work/bench/`, `work/infer/`, `work/logs/` (bench of
  2026-11-06 07:0x).

## Appendix: true-length execution (min_width=1) measurement (2026-11-06)

The P4-era minimum physical width 256 was retired: every case now runs at its
TRUE length (width = max row length; the official pool executes at S=85/99,
B3 at 99, the image case at 130).  Driver: the official case pool is entirely
short prompts (33-51% width utilization at the 256 floor).  The short-width
gate on the delivered AIR first re-tested the P4-era "unpadded short-width
execution had measurably unstable boundaries" observation: true-length
execution (S=85/99/130, 3 repeats per case) is BIT-STABLE and within bf16
quantization of both the CPU golden (relL2 <= 0.0037) and the accepted
pad-256 outputs (relL2 <= 0.0046) -- the observation does not reproduce on
this configuration (fp32-Conv2D conv + eager-math attention + MatMul
patch-embed).  Then the full standalone flow was re-gated at min_width=1.

| item | value | pad-256 (same AIR, same day) |
|---|---|---|
| precision | 5/5, worst cosine 0.999998, worst relL2 0.004320 | 0.999998 / 0.004753 |
| business gate | 5/5, worst active-probability max abs 0.005887 | 5/5, 0.006282 |
| requests / rows | 2,048 / 2,916 (0 errors) | same |
| graph executions/s | **36.125397** | 33.823975 |
| readout rows/s | **51.436356** | 48.159527 |
| e2e P50/P99 (n=2048) | 25.532/36.349 ms (avg 27.679) | 27.138/38.335 |
| per-case e2e P50 (ms) | ticket B1 25.109-25.516; image 26.477; B3 33.559 | 27.117-27.121; 27.118; 38.214 |

- Net effect: **+6.8% executions/s and -1.6 ms e2e P50** (B3 case -4.6 ms:
  three rows share the narrowed width).  Both precision and business metrics
  move slightly BETTER than the pad-256 run (the padded region's numerical
  noise is gone).
- Width explains only the S-proportional part of each request; the fixed
  per-request cost (the vision tower always runs in the single AIR, host
  dispatch) bounds the gain at ~2x width reduction -> ~7% latency.
- Evidence paths: `work/bench/`, `work/infer/`, `work/logs/` (bench of
  2026-11-06 07:3x); short-width gate script data under /tmp/opencode (bit
  stability + golden/pad-256 comparisons recorded in validation.md).

## Appendix: RMSNorm fusion + short-width profiling re-collection (2026-11-06)

The eager RMSNorm chains of the text decoder (layer input/post-attention
norms x48, body norm, full-attention q/k norms x12, and the GDR gated norm
x18 per request) were fused into the GE RmsNorm node (`torch.ops.npu.
npu_rms_norm`, model/startlux_npu_adapter.py `_rms_norm` / `_rms_norm_gated`)
with the official compute domain kept exactly (fp32 mean and weight, output
cast back; the gated variant keeps the official weight/silu order; gamma
constant-folded from the frozen parameter; eager legs unchanged).  Same
flow, idle card Chip 3.

| item | value | pre-fusion (min_width=1, same day) |
|---|---|---|
| precision | 5/5, worst cosine 0.999999, worst relL2 **0.002233** | 0.999998 / 0.004320 |
| business gate | 5/5, worst active-probability max abs 0.006454 | 5/5, 0.005887 |
| graph executions/s | **39.801171** (2,048 requests, 0 errors) | 36.125397 |
| readout rows/s | **56.670027** | 51.436356 |
| e2e P50/P99 (n=2048) | 23.192/31.026 ms (avg 25.122) | 25.532/36.349 |
| per-case e2e P50 (ms) | ticket B1 22.893-23.171; image 24.209; B3 30.789 | 25.109-25.516; 26.477; 33.559 |

- Net effect: **+10.2% executions/s and -2.3 ms e2e P50**; precision moves
  clearly BETTER (relL2 0.0043 -> 0.0022 -- the fused fp32 reduction is
  more accurate than the eager chain), business stays inside the gate.
- Profiling re-collection (work/prof-rms, 32 measured requests): the fused
  RmsNorm node appears at 3160x / 4.96% replacing the former Square 115x +
  ReduceMean 79x per request; Mul drops 15.8% -> 11.5%, Cast 12.4% -> 10.9%.
  New top operators: MatMulV2 17.3% (vision fp32 included), GDR
  ChunkGatedDeltaRule 13.0%, Mul 11.5%, Cast 10.9%.
- Evidence paths: `work/bench/`, `work/infer/`, `work/prof-rms/`,
  `work/logs/` (2026-11-06 07:4x-07:5x).

## Appendix: fp16 vision linear compute measurement (2026-11-06)

The vision tower's linear inner products (patch-embed matmul, attention
qkv/proj, MLP fc1/fc2, merger fc1/fc2 -- all replayed op-for-op in
VisionTowerWrapper) run in fp16 while the fp32 activation flow, constant
buffers (pos/rotary tables), norms and softmax stay fp32 (`linear_dtype`
switch, default fp16; bf16 was tested first and FAILS the vision gate at the
real 12-block depth: relL2 0.055 vs fp16 0.0045 -- bf16's 7 mantissa bits do
not survive depth accumulation; official production runs whole-model bf16
under a business-level acceptance, our gate is a feature-level relL2 <= 0.02).

| item | value | fp32-vision baseline (same day) |
|---|---|---|
| precision | 5/5, worst cosine 0.999999, worst relL2 0.006571 | 0.999999 / 0.002233 |
| business gate | 5/5, worst active-probability max abs 0.006454 | 5/5, 0.006454 (identical) |
| graph executions/s | **41.726731** (2,048 requests, 0 errors) | 39.801171 |
| readout rows/s | **59.411693** | 56.670027 |
| e2e P50/P99 (n=2048) | 22.218/29.953 ms (avg 23.963) | 23.192/31.026 |
| per-case e2e P50 (ms) | ticket B1 21.921-22.224; image 23.175; B3 29.792 | 22.893-23.171; 24.209; 30.789 |

- Net effect: **+4.8% executions/s and -1.0 ms e2e P50**; the numeric relL2
  moves 0.0022 -> 0.0066 (fp16 vision propagation, still 3x inside the gate)
  while the business metric is bit-identical to the fp32 baseline.
- Profiling (work/prof-fp16): MatMulV2 17.25% -> 14.32% (171.3 -> 145.7 ms
  total, the fp16 halving of the vision share); Cast rises 10.93% -> 14.35%
  (the per-forward fp16 weight casts are not fully constant-folded -- a
  further ~0.9 ms/request is recoverable by pre-materializing fp16 weight
  copies, left on the backlog).
- Text trunk stays bf16: the ChunkGatedDeltaRule prototype accepts bfloat16
  only (ops-info contract), fp16 equals bf16 throughput, and the text
  numerics have no pain point.
- Evidence paths: `work/bench/`, `work/infer/`, `work/prof-fp16/`,
  `work/logs/` (2026-11-06 08:2x-08:3x).

## Appendix: fp16 full-tower follow-up experiments (2026-11-06, both rolled back)

Two follow-ups to the fp16 vision delivery were built, gated and benched;
both are NET LOSSES and were rolled back (the delivered structure stays:
fp32 residual stream + fp16 linear inner products, 41.73 exec/s).

1. **Pre-materialized fp16 weight copies** (constant-fold the per-forward
   weight Casts): BLOCKED at export -- torchair's positional weight bake
   (runtime input i replaces export_graph.op[i]) mis-aligns when the ~51 new
   fp16 weight Const nodes enter the op list ("incomplete exported Data
   binding"); tiny-model export succeeds, the full dynamic export does not.
2. **fp16 residual stream** (validated numerics: real-tower relL2 0.0051,
   bit-stable, full-model gates pass 5/5 + 5/5 with metrics identical to the
   mixed baseline): benched **40.908 exec/s vs 41.727 (-2.0%)** -- the
   per-layer fp16<->fp32 round-trip Casts cost more than the halved
   activation traffic saves (the weight-copy fold that would have removed
   the Casts is the blocked item above).

| item | fp16 stream (rolled back) | delivered mixed (kept) |
|---|---|---|
| graph executions/s | 40.908414 | **41.726731** |
| e2e P50/P99 (n=2048) | 22.684/30.486 ms | 22.218/29.953 |
| precision / business | 5/5 (relL2 0.006571) / 5/5 (0.006454) | identical |

Evidence: `work/bench/` of the rolled-back run (2026-11-06 11:0x);
validation.md decision record.  Full-fp16 vision is closed at the current
export mechanism: the remaining ~0.9 ms/request of weight Casts is only
recoverable by changing the torchair bake contract or pre-quantizing the
checkpoint itself.

## Appendix: RMSNorm direct-bf16 input measurement (2026-11-07)

The fused RmsNorm nodes now take the bf16 activation DIRECTLY (mixed dtype:
bf16 self + fp32 gamma) instead of the explicit `x.float()` -> norm ->
`.to(bf16)` round-trip -- dropping ~120 Cast nodes per request.  Driver:
the Cast census (op_summary grouping) showed the norm-domain upcast/downcast
as the second-largest Cast group; the NPU eager probe first proved the
kernel's reduction runs in fp32 internally (bf16 input + fp32 ones gamma ==
the explicit fp32-domain path at the bf16 output-quantization level, and the
fp32 gamma removes the only other error source), and the torchair converter
accepts the mixed dtype.

| item | value | explicit fp32-domain (2026-11-06) |
|---|---|---|
| precision | 5/5, worst cosine 0.999997, worst relL2 **0.004039** | 0.999999 / 0.006571 |
| business gate | 5/5, worst active-probability max abs **0.002769** | 5/5, 0.006454 |
| graph executions/s | **43.156994** (2,048 requests, 0 errors) | 41.726731 |
| readout rows/s | **61.448142** | 59.411693 |
| e2e P50/P99 (n=2048) | 21.422/29.198 ms (avg 23.168) | 22.218/29.953 |
| per-case e2e P50 (ms) | ticket B1 21.149-21.425; image 22.376; B3 29.052 | 21.921-23.175; 23.175; 29.792 |

- Net effect: **+3.4% executions/s and -0.8 ms e2e P50**, with BOTH precision
  and business metrics moving clearly better (relL2 0.0066 -> 0.0040, business
  worst 0.0065 -> 0.0028) -- fewer Cast hops also remove quantization
  boundaries on the propagation path.
- Evidence paths: `work/bench/`, `work/infer/`, `work/logs/`
  (2026-11-07 02:1x).

## Appendix: GDR pad-mask removal measurement (2026-11-09)

The ~90 redundant pad-masking Muls per request were removed (the GDR input
zeroing ``mixed * pad`` and the in-layer ``q/k/v * pad`` + ``beta * valid``):
the chunk recurrence is strictly causal, so padded tokens only ever
influence outputs at their own and later positions and the final state --
the readout gathers the true L-1 and the state is discarded (the same
argument the mega route has always relied on; the kernel was re-proven by
the full-model gates below).

| item | value | pre-removal (2026-11-09, same day) |
|---|---|---|
| precision | 5/5, worst cosine 0.999998, worst relL2 0.002798 | 0.999998 / 0.002798 (identical) |
| business gate | 5/5, worst active-probability max abs 0.007095 | 0.007095 (identical) |
| graph executions/s | **45.265265** (2,048 requests, 0 errors) | 41.840-42.536 |
| readout rows/s | **64.449958** | 59.572 |
| e2e P50/P99 (n=2048) | 20.232/28.033 ms (avg 22.089) | 22.117-21.673 / 29.6-29.98 |
| per-case e2e P50 (ms) | ticket B1 20.202-20.208; image 21.217; B3 27.931 | 21.15-21.43; 22.38; 29.05 |

- The precision/business metrics are IDENTICAL to six significant digits to
  the pre-removal run -- direct empirical proof that the kernel's causality
  holds exactly (padded garbage never touches the valid prefix), matching
  the removed masks' no-op status.
- Net effect: **+6.4-8.2% executions/s and -1.4 ms e2e P50**, in line with
  the ~1.5-1.9 ms/estimate of the removed Mul work.
- Evidence paths: `work/bench/`, `work/infer/`, `work/logs/`
  (2026-11-09 03:3x-04:0x).

## Appendix: GDR q/k L2-norm fused as RmsNorm measurement (2026-11-09)

The GDR q/k L2-normalization chains (18 layers x 2 sides of
``x.float() -> mul/rsqrt(sum(x^2)+1e-6) -> .to(bf16)``) were fused into ONE
RmsNorm node per side: ``rms_norm(x, gamma=1/sqrt(D), eps=1e-6/D)`` ==
``x * rsqrt(sum(x^2) + 1e-6)`` exactly (the sqrt(D) of the mean-domain
folding cancels in gamma; kernel-internal fp32 reduction, bf16 output; the
bf16 gamma carries ~1e-4 relative rounding).  The kernel scale stays
``Dk**-0.5`` UNCHANGED: a scale-fold variant (ones gamma + scale D**-1.5)
was NPU-probed WRONG at single-layer relL2 0.85 -- the delta update couples
q/k magnitudes nonlinearly, so the magnitude fold must live in gamma.

| item | value | pre-fusion (P0-1, same day) |
|---|---|---|
| precision | 5/5, worst cosine 0.999998, worst relL2 0.003857 | 0.999998 / 0.002798 |
| business gate | 5/5, worst active-probability max abs 0.003283 | 0.007095 |
| graph executions/s | **49.327130** (2,048 requests, 0 errors) | 45.265265 |
| readout rows/s | **70.233355** | 64.449958 |
| e2e P50/P99 (n=2048) | 18.571/26.016 ms (avg 20.270) | 20.232/28.033 |
| per-case e2e P50 (ms) | ticket B1 18.454-18.575; image 19.513; B3 25.930 | 20.202-20.208; 21.217; 27.931 |

- Net effect: **+9.0% executions/s and -1.7 ms e2e P50**; precision moves
  0.0028 -> 0.0039 (the bf16 gamma's ~1e-4 per-side rounding propagating
  through 18 layers, still 5x inside the gate) while the business metric
  improves.
- Removed per request: ~72 Cast nodes ([B,S,16,128] bf16<->fp32) plus the
  Square/ReduceSum/Rsqrt/Mul chains of the explicit norm.
- Evidence paths: `work/bench/`, `work/infer/`, `work/logs/`
  (2026-11-09 04:3x).

## Appendix: fused RoPE via ApplyRotaryPosEmb measurement (2026-11-09)

The 6 full-attention layers' eager rotate-half chains (~10 small ops per side
per layer) were fused into `npu_apply_rotary_pos_emb` (native BSND layout=1:
q/k [B,S,N,rot], cos/sin [B,S,1,rot], zero reshapes; rotary_mode half == the
exact rotate-half math; the RoPE runs in the natural [B,S,N,D] domain before
the single transpose to BNSD).  Three earlier attempts failed and were
probed: BNSD layout is rejected by the tiler (only 1/4 legal), a TND variant
and a wrongly-shaped BSND call broke the gates, and -- decisively -- the
probe's "reference" was itself wrong (the aclnn EAGER leg's layout=1 reads
[B,S,N,D] as BSH, relL2 0.88 vs pure math); against a PURE-MATH reference
the GE-graph BSND call is exact (relL2 0.0002) for every B/S combination of
the case pool.

| item | value | pre-fusion (P2-5 chain, 2026-11-09) |
|---|---|---|
| precision | 5/5, worst cosine 0.999998, worst relL2 0.003930 | 0.999998 / 0.003857 |
| business gate | 5/5, worst active-probability max abs 0.004551 | 0.003283 |
| graph executions/s | **49.231163** (2,048 requests, 0 errors) | 49.003 (chip-3 window) |
| readout rows/s | **70.096715** | 69.77 |
| e2e P50/P99 (n=2048) | 18.586/26.201 ms | 18.594/26.518 |

- Net effect: throughput at parity-to-slightly-better (+0.5%, within
  run-to-run variance on a different idle chip); precision/business within
  the usual export-nondeterminism band.  The win is structural: ~120 fewer
  graph nodes per request (Mul/Cat/Neg chains) and a smaller AIR
  (1,906 -> 1,708 MB, -10%).
- Note: this run executed on idle chip 8 (chip 3 was occupied by another
  tenant, which had been silently killing the earlier exports).
- Evidence paths: `work/bench/`, `work/infer/`, `work/logs/`
  (2026-11-09 08:5x-09:0x).

## Appendix: fused full-attention via npu_fusion_attention measurement (2026-11-09)

The 6 bidirectional full-attention layers' eager chain (GQA k/v expand,
scores matmul + additive mask, fp32 softmax, probs matmul -- ~15 nodes per
layer) was fused into ONE `npu_fusion_attention` (FA) call per layer: bf16
BNSD, bool [B,1,S,S] pad mask (True = blocked, the exact additive
min-score-mask equivalent; the mask is derived in-graph from the additive
mask), FA consumes the GQA layout natively so the k/v
unsqueeze-expand-reshape drops out.  Eager branch unchanged.

Probes (pure-math reference, all B/S of the case pool): relL2 0.0023 per
layer; eager-chain x2.05 at B1/S99.  A vision-tower FA variant was probed
FIRST and rejected: numerically exact but x0.99 (fp32 short chunks are
device-bound, not host-bound -- no fusion win there).

| item | value | pre-fusion (P2-7, same day) |
|---|---|---|
| precision | 5/5, worst cosine 0.999999, worst relL2 0.007893 | 0.999998 / 0.003930 |
| business gate | 5/5, worst active-probability max abs 0.004143 | 0.004551 |
| graph executions/s | **49.809 / 50.942** (2 runs, 2,048 req, 0 errors) | 49.231 |
| readout rows/s | 70.919 | 70.097 |
| e2e P50/P99 (n=2048) | 18.300/25.948 (repeat 17.885) | 18.586/26.201 |
| per-case e2e P50 (ms) | ticket B1 18.209-18.287; image 19.268; B3 25.674 | 18.454-18.575; 19.513; 25.930 |

- Net effect: **+1.2-3.5% executions/s and -0.3-0.7 ms e2e P50** across
  two runs; precision moves 0.0039 -> 0.0079 (FA's internal probs@v path,
  still 2.5x inside the gate) while business improves slightly.
- The probe-level x2.05 compresses to ~1-3% whole-model because the 6
  attention layers are only ~0.5 ms/request of the e2e.
- Evidence paths: `work/bench/`, `work/infer/`, `work/logs/`
  (2026-11-09 09:2x-09:4x).

## Appendix: WeightNzAndMatMulV3Pass measurement (2026-11-09)

The WeightNzAndMatMulV3Pass custom GE fusion pass
(third_party/custom_development_code/fusion_pass/WeightNzAndMatMulV3Pass)
was built and installed into the CANN custom_fusion_passes directory: it
rebuilds MatMul/MatMulV2 constant weights into FRACTAL_NZ as MatMulV3 at
graph-compile time (host-side, no runtime layout casts; activations and
outputs stay ND).  Installation and build wired into prepare.sh.

| item | value | baseline (af9e05a + mega cleanup) |
|---|---|---|
| precision | 5/5, worst cosine 0.999999, worst relL2 0.004496 | 0.999997 / 0.003933 |
| business gate | 5/5 PASS | 5/5 |
| graph executions/s (mixed closed loop) | 50.606 (2,048 requests, 0 errors) | 50.481 |
| ticket-team single-case AI time | **15,092 us/exec** | 17,642 us/exec (**-14.4%**) |
| MatMulV2 (text projections/MLP) | 0 (replaced) | 3,115 us/exec |
| MatMulV3 (NZ weights) | 626 us/exec | -- |
| ChunkGatedDeltaRule / Cast / Transpose / rest | ~level | ~level |

- The MatMulV2 -> MatMulV3+NZ replacement alone cuts ~2.5 ms of AI time per
  ticket-team execution; the mixed-pool executions/s shows +0.3% (variance
  band) because the closed loop is host-bound at these latencies while the
  single-case AI-time drop is the kernel-level truth.
- First graph compile after install pays a one-shot TBE cost per matmul
  shape (the first infer run may need a generous timeout; kernel_meta
  caches it).
- System-level install by design (CANN vendors dir); prepare.sh rebuilds
  and reinstalls it alongside the chunk prototype.
- Evidence paths: `work/prof-nz/` (ticket-team single-case profile),
  `work/bench/`, `work/infer/` (2026-11-09 12:0x).

## Appendix: FA -> FIA per-row-lengths attention switch (2026-11-10)

The 6 bidirectional full-attention layers moved from npu_fusion_attention
(BNSD + [B,1,S,S] bool pad mask, B*S^2 materialized memory) to
npu_fused_infer_attention_score with per-row true lengths on both query and kv
sides (BSND natural layout, no transposes, NO S^2 mask): graph leg via
torchair.ops (`air` namespace op), eager leg via the torch_npu host op behind
torch.compiler.is_compiling().  S is now bounded by the model's native
max_position_embeddings (262144, executed and numerically correct) instead of
the B*S^2 mask HBM ceiling (FA failed at S=262144: ~68GB mask).

| item | FIA per-row lengths | FA pad mask (pre-switch) |
|---|---|---|
| precision (graph vs CPU FP32 golden) | 5/5, worst cosine 0.999998, worst relL2 0.006006 | 0.999998 / 0.004496 |
| eager smoke vs golden | relL2 0.00798 | 0.00408 |
| business gate | 0.004028 | 0.003431 |
| S=262144 at B=1 | executable, 63 s, sampled relL2 0.0023 | E19999 execution failure |
| graph executions/s (mixed closed loop) | 49.680 | 49.686 |
| e2e P50/P99 (n=2048, ms) | 18.548 / 25.161 | 18.651 / 25.120 |
| ticket-team exec p50 (us) | 18,141 | 18,262 |
| AIR bytes | 1,708,244,818 (-25 KB, dropped transposes) | 1,708,269,731 |

- Throughput parity: the historical 4x single-op slowdown (2026-11-09, B1/S99,
  BNSD + transpose form) is fully offset by the BSND no-transpose form and the
  dropped S^2 mask materialization; the historical -3.5% whole-model revert was
  measured on the BNSD+transpose form.
- Evidence paths: `work/bench/`, `work/infer/`, `work/logs/`
  (2026-11-10 13:0x-14:0x UTC), probes as described in validation.md 2026-11-10.

## Appendix: post-cleanup verification run (2026-11-10)

Regression check after the 2026-11-10 code-review cleanup (dead-code removal,
redundant-validation removal, comment/doc corrections; see validation.md).  The
cleanup claims no numeric change, so this run re-measures the SAME delivered
configuration on the SAME chip and seed as the fused-attention appendix.

Full `run.sh --device 4` for stages 1-9 (precision chain), then stage 11 (bench)
re-run standalone on idle chip 8 -- the physical chip is selected by
`ASCEND_RT_VISIBLE_DEVICES` while `request.json` carries the in-process index 0,
so the bench needs no re-export.  Chip 8 was verified idle (AICore 0%, HBM at
the ~3.2 GB baseline) by two samples 15 s apart immediately before launch, and
released back to 3,255 MB after the run.

| item | value | WeightNzAndMatMulV3Pass appendix (2026-11-09) | fused-attention appendix, chip 8 (2026-11-09) |
|---|---|---|---|
| precision | 5/5, worst cosine 0.9999986085, worst relL2 **0.00449550810581621** | 0.999999 / **0.004496** | 0.999998 / 0.003930 |
| business gate | 5/5 PASS, worst active-probability max abs 0.00343094 | 5/5 PASS | 5/5, worst 0.004551 |
| graph executions/s | 49.489124 (2,048 requests, 0 errors) | 50.606 | 49.231163 |
| readout rows/s | 70.464006 | -- | 70.096715 |
| e2e P50/P99 (n=2048) | 18.785/25.251 ms | -- | 18.586/26.201 ms |
| e2e avg/min/max | 20.204/18.41/28.92 ms | -- | -- |
| exec avg/P99 | 19.998/24.958 ms | -- | -- |
| AIR bytes | 1,708,279,731 (two independent exports, identical size) | -- | 1,708 MB |

Bench detail: `arrival_model` closed_loop, pool_size 5, seed 42,
replacement true, instances 1, warmup 32 runs / 890.374 ms, wall 41,382.83 ms,
distinct_shapes 4, load 3,792.49 ms, specialize 318.972 ms, h2d avg 0.203 ms,
desc avg 0.00095 ms, d2h 0.0, HBM base 504.570 MB / after_setup 4,061.832 MB,
samples per request avg 1.423828125 (min 1, p50 1, p99 3, max 3), errors 0,
`in_flight_at_end` 0.

- **Precision is the strongest evidence of no regression**: worst relL2
  0.00449550810581621 reproduces the recorded 0.004496 digit-for-digit at the
  recorded precision, and worst cosine rounds to the recorded 0.999999.  The
  cleanup touched no operator or numeric path.
- Throughput 49.489 exec/s is -2.2% vs the NZ-pass appendix (50.606) and +0.5%
  vs the chip-8 fused-attention appendix (49.231163); it sits inside the
  historical spread of the delivered configuration (49.231-50.606, ~2.8%).
  The closed loop is host-bound at these latencies, and at measurement time six
  of the sixteen chips (4, 5, 12, 13, 14, 15) were held at ~65 GB HBM by other
  tenants, so host-side contention is the likely cause of landing at the low
  end.  Not attributed to the cleanup.
- Latency shape is unchanged-to-better: P50 +0.199 ms (+1.1%, variance),
  P99 -0.950 ms (-3.6%).
- AIR size 1,708,279,731 bytes from two independent exports confirms the graph
  structure is unchanged.  This is the value the README should have carried:
  its "字节数恒定 1,907,534,423" claim predated the fused RoPE / fused
  full-attention work (1,906 -> 1,708 MB, -10%) and was corrected by this run.
- Evidence paths: `work/bench/` (result.json, requests.csv 2,048 rows),
  `work/infer/` (comparison.json, business.json), `work/logs/`
  (2026-11-10 02:3x-02:5x UTC).


## Appendix: self-owned nn.Module rewrite verification (2026-11-10)

The `model/` tree was rewritten in place as self-owned `nn.Module` classes
(checkpoint-mirroring parameter names, zero-mapping `load_state_dict`,
per-site dtype declarations mirroring vLLM; see validation.md 2026-11-10).
RoPE reverted to the explicit rotate-half chain per the user's stop rule on the
recorded eager/graph layout divergence of `npu_apply_rotary_pos_emb`
(fused-op throughput was measured neutral on 2026-11-09, so the reversion
costs graph size only).  All gates re-run on chip 4 (ASCEND_RT_VISIBLE_DEVICES=4)
after a pre-rewrite HEAD baseline re-export confirmed bit-identical infer
outputs (`e7157ae1...`).

| item | rewritten (single leg) | pre-rewrite baseline |
|---|---|---|
| precision (graph vs CPU FP32 golden) | 5/5, worst cosine 0.999998, worst relL2 0.005362 | 0.999998 / 0.004496 |
| precision (EAGER single-leg NPU vs golden) | 5/5, worst relL2 0.00840 | n/a (no eager path) |
| business gate | 5/5, worst active-probability max abs 0.003790 | 0.003431 |
| golden sentinel (rewrite must not touch oracle) | bit-identical `bfc06e59...` | identical |
| graph executions/s (mixed closed loop, 2,048 req) | 49.686 (quiet chip 14) / 50.186 (quiet chip 4) | 49.231-50.942 band; post-cleanup 49.489 |
| readout rows/s | 70.745 / 71.456 | 70.464 / 70.919 |
| e2e P50/P99 (n=2048, ms) | 18.651/25.120 / 18.433/24.913 | 18.785/25.251 (post-cleanup) |
| errors | 0 | 0 |
| AIR size (bytes) | 1,708,386,627 | 1,708,279,731 |
| AIR op fingerprint | MatMulV2=100, ChunkGatedDeltaRule=36 | identical |

- The +106,896-byte AIR delta is the explicit rotate-half RoPE chain replacing
  the fused op (~120 small nodes back in the graph), the exact structural
  inverse of the 2026-11-09 fused-RoPE appendix; throughput stays in the
  run-to-run band as that appendix already measured ("持平略正（方差内）").
- RoPE re-probe (same day, user question "现在还是不对吗"): the recorded
  eager/graph layout divergence of `npu_apply_rotary_pos_emb` NO LONGER
  reproduces on this build.  Probe: delivery call form (layout="BSND",
  rotary_mode="half", bf16, real GQA geometry q[B,S,8,64]/k[B,S,2,64], theta
  1e7), pure-math reference vs direct eager call vs torchair dynamo_export
  (the delivered export machinery, dynamic dual-axis trace B2/S65) executed by
  ge_runtime at six B/S combos including B=1: both legs relL2 0.0013 (bf16
  quantization) and bit-identical to each other.  Root cause of the historical
  flip: the torchair GE converter maps BOTH "BSH" and "BSND" strings to enum 1
  (GE: 1 == BSND) while the then-current aclnn host read 1 as BSH.
- FUSED RoPE re-landed after the probe (user decision, same day): the rewrite
  appendix's explicit-rotate-half rows above are superseded by the fused-op
  form.  Re-run gates on chip 14: graph worst relL2 0.004496 reproduces the
  pre-rewrite baseline digit-for-digit, business worst 0.003431 matches the
  baseline record with every per-case prob_max_abs better than baseline,
  eager single-leg (incl. non-contiguous slice inputs) relL2 0.00408 vs
  golden, bench 49.622 exec/s / P50 18.684 / P99 25.175 (band), AIR
  1,708,269,731 bytes (baseline level, -10 KB).  See validation.md 2026-11-10.
- Bench window sensitivity (same AIR, same seed, 5 measurements): the closed
  loop is host-bound, and the host was multi-tenant during 2026-11-10
  09:0x-09:1x UTC -- exec/s tracked host load: 50.186 (chip 4, quiet, 08:45),
  37.385 (chip 4, all 8 sampled chips at 200-270 W, inside run.sh), 45.991
  (chip 4, partial), 47.750 (chip 14, host loadavg 54+), **49.686 (chip 14,
  loadavg receding -- the recorded in-band measurement)**.  Precision is
  load-independent: worst relL2 0.005362 reproduced identically across every
  run.  Same methodology caveat as the post-cleanup appendix (idle-chip bench
  under residual host load).
- WeightNzAndMatMulV3Pass: system install untouched; the new AIR carries the
  same 100 constant-weight MatMulV2 nodes, so the compile-time pass's match
  surface is unchanged.  Single-case AI-time profiling (msprof) was not
  repeated; see validation.md 2026-11-10 for the reasoning and the method to
  re-close it via `work/prof-nz/` if needed.
- Evidence paths: `work/bench/` (result.json, requests.csv), `work/infer/`
  (comparison.json, business.json), `work/baseline-pre-rewrite/` (pre-rewrite
  artifacts + sha256), `work/logs/` (2026-11-10 08:0x-08:5x UTC).

## Appendix: text-only vision skip (2026-11-10)

Text-only requests no longer execute the vision AIR at all (records carry
has_images=false; the pipeline runtime feeds a pre-allocated 1x1024 zero
features tensor to the text graph's slot 3 -- the same shape the 4-row dummy
vision output had, and the injection's where(is_image=false) discards it
elementwise: the scores are byte-identical, outputs.bin sha unchanged).

| item | with the 4-row dummy | vision skipped |
|---|---|---|
| infer outputs.bin sha | cff3bbdb... | cff3bbdb... (bit-identical) |
| graph executions/s (mixed pool, 4/5 text-only) | 53.15 | **61.01 (+14.8%)** |
| e2e P50 (n=2048, ms) | 17.27 | **13.90** |
| text-only case e2e P50 (ms) | 17.3 (vision 3.7) | **13.86 (vision 0)** |
| image case e2e P50 (ms) | 19.5 | 19.86 (vision p50 4.8, path unchanged) |

- Cumulative text-only latency vs the unified AIR baseline: 18.55 -> 13.86 ms
  (-25%); vision cost 1 ms (2 static slots) -> 3.7 ms (4-row dynamic dummy)
  -> 0 (skipped, the official media=None behavior).
- Evidence: `work/bench/` (requests.csv: vision_ms all 0 for text-only cases),
  `work/infer/` (2026-11-10 17:0x UTC).

## Appendix: two-stage split AIRs via pipeline_runtime (2026-11-10)

The delivery moved from ONE unified AIR to TWO dynamic AIRs (vision + text)
run by the new minimal C++ pipeline_runtime (positional binding, no io_spec
embedding): the vision stage takes any image count / any official grid (the
official per-call position tables enter as INPUTS -- the position_ids
pattern), attention is FIA TND varlen per-image (fp16 inner math, the one
precision concession), and image_features stay device-resident between the
stages (the vision output's gert::Tensor descriptor is rebuilt over the same
storage and fed directly as the text graph's input slot 3 -- no host round
trip).

| item | split pipeline (53.15 exec/s) | unified AIR (pre-split) |
|---|---|---|
| precision (text vs CPU FP32 golden) | 5/5, worst cosine 0.999998, worst relL2 0.006053 | 0.999998 / 0.006006 |
| eager two-stage vs golden (1img/multires-2/3img/B3/textonly) | 0.00093 / 0.00440 / 0.00500 / 0.00283 / 0.00142 | n/a |
| business gate | 0.004028 | 0.004028 |
| graph executions/s (mixed closed loop) | **53.15** | 49.680 |
| e2e P50/P99 (n=2048, ms) | **17.27 / 24.19** | 18.548 / 25.161 |
| AIR bytes | vision 198,629,675 + text 1,507,553,177 | 1,708,244,818 |
| image capability | **any count <= 16/request, any official grid** | 2 global slots, grid [1,16,16] |
| text-only vision cost | 4 dummy rows (microseconds) | 2 full slots (~1 ms) |

- **Per-graph streams are load-bearing**: with both graphs on ONE stream the
  alternating executes force GE to re-bind the execution plan every call --
  9.08 exec/s, P50 81 ms, P99 818 ms (vision p50 13.4 ms against ~1 ms of
  actual compute located it).  One stream per graph removes it entirely and
  lands ABOVE the unified baseline (+7% exec/s, -1.3 ms P50): the text AIR
  compiles/tils smaller without the vision tower.
- The +7% is a real structural win of the split (smaller per-graph tiling)
  on top of the unlimited-image-count/resolution capability.
- Evidence paths: `work/bench/` (result.json, requests.csv),
  `work/infer/`, `work/logs/` (2026-11-10 15:0x-16:0x UTC), probes as
  described in validation.md 2026-11-10.

## Appendix: GDN causal-conv path switch to the vendored CausalConv1d op (2026-11-11)

`modeling_text.causal_conv4` was replaced by the vendored fla AscendC
kernel behind `torch.ops.ascend_ops.causal_conv1d`
(`third_party/ascend-ops/causal-conv1d`, graph leg = `CausalConv1dGE`
custom-op-v2 wrapper): fused depthwise causal conv + SiLU, dim-last
(B,S,D) bf16, fed directly by the `in_proj_qkv` matmul output.  The old
7-op chain (Transpose-in + TransData->fractal + Conv2D + TransData<-
fractal + tail Slice + Swish + Transpose-out, 128.9 us/layer on the team
case) collapses to the single op, and both layout round-trips disappear.

| item | pre (Conv2D chain) | post (CausalConv1d) |
|---|---|---|
| bench plan | instances 1, warmup 32, 2048 measured requests, seed 42, closed loop, chip 14 | same |
| e2e P50 / P99 / avg (n=2048, ms) | 13.755 / 21.165 / 16.157 | **11.949 / 19.238 / 14.053** (-13.1% / -9.1% / -13.0%) |
| qps | 61.9 (imputed from avg e2e) | **71.16** |
| per-case e2e P50 (ms) | team 13.546, severity 13.549, urgent 13.725, image-damaged 19.448, b3 20.479 | **11.645, 11.644, 11.886, 17.757, 17.040** |
| in-graph op mix per request | Conv2D x18 + TransData x36 + Transpose x37 + Slice x108 + Swish x60 | CausalConv1d x18, TransData x0, Transpose x2.7 |
| AI task time per request (msprof op_statistic, us/req) | 10,445 | **9,156** (-12.3%; both sides exclude MatMulV3 -- the msprof export drops CubeCore rows, same-scope comparison; with the pre-side MatMulV3 restored at 2,687 us/req (187/req, `*_restored.csv`) the full pre AI time is 13,132 us/req, and the conv switch leaves the matmul set untouched, so the relative gain carries) |
| causal-conv path per request (in-graph) | 2,322 (kernel 439.6 + transdata 642.4 + transpose 731.4 + slice/swish/cast tail) | **411.4** (22.9 us/layer) |
| text AIR bytes | 1,708,279,731 | 1,507,412,645 (-11.7%) |
| precision (5 cases vs CPU FP32 golden) | worst cosine 0.999999, worst relL2 0.004004 | worst cosine 0.999999, worst relL2 0.005704 (gate 0.999 / 0.02) |
| business gate | worst active-prob max abs 0.005507 | worst 0.004267 (gate 0.01) |

Integration notes (measured on the same build as this appendix):

- `pipeline_runtime` dlopens the vendor `libcust_opapi.so` (RTLD_GLOBAL)
  before GEInitialize; GE does not discover it on its own in the C++
  runner and the graph compile aborts with `has_ir is 0` otherwise.
- With the op loaded, the process-exit static destruction (after
  GEFinalize) trips a glibc fastbins corruption abort; the runner
  therefore tears down explicitly and leaves via `_exit(0)` on the
  success path (profiling runs keep the normal exit: msprof's atexit
  export completes before the abort, only the exit code is spoiled).
- Evidence: `work/bench/` (post) and `work/bench-baseline-conv2d/`
  (pre, requests.csv), `work/prof-conv1d-msprof/` (msprof, 200-iter
  op_statistic vs `work/prof-team-split/bench-profiling/`, pre side
  additionally `op_statistic_*_restored.csv` via `tools/restore_matmulv3.py`),
  `work/logs/bench-new-runtime.log`.

Long-text dev case (single-case bench, `case_filter dev-intent-q00-r0`,
warmup 16 / 400 measured / seed 42, same chip, same AIRs):

| item | pre (Conv2D chain) | post (CausalConv1d) |
|---|---|---|
| e2e P50 / P99 / avg (n=400, ms) | 26.599 / 26.815 / 26.612 | **22.441 / 22.624 / 22.449** (-15.8%) |
| qps | 37.58 | **44.54** (+18.5%) |
| msprof (416 profiled execs) | - | CausalConv1d 18/req, avg 64.3 us/layer, 7.3% of the (MatMulV3-excluded) AI time; Conv2D / TransData / Transpose all at zero |

Number-source note: the post column cites the 09:15 same-day re-run of the
identical request (`work/devcase/bench-new/case/dev-intent-q00-r0/`,
overwritten in place by the 18-case sweep below).  The first run of the same
config (08:51, evidence since overwritten) read 22.279 / 22.425 / 22.281 ms
and 44.88 qps; the 0.7% spread is within the historical run-to-run band, and
the pre column is unaffected.

Reading the msprof export of this run requires the MatMulV3 caveat: the
raw `op_statistic` drops all 77,792 MatMulV3 rows (the SQE-type UNKNOWN
defect, `tools/restore_matmulv3.py`) and therefore shows
ChunkGatedDeltaRule at "38%" of AI time.  The restored export
(`op_statistic_20261111085318_restored.csv`, same directory, 393,952 rows =
316,160 + 77,792) re-normalizes the picture: **MatMulV3 27.7% and
ChunkGatedDeltaRule 27.5% are tied for the largest share** (6.07 and
6.01 ms/request of the 21.9 ms steady-state device time; 947 device ops per
request at 95.4% device utilization, 22.95 ms/request under msprof vs
22.45 ms without).  The dominant-cost statement in the raw export is an
artifact, not a finding.

Long-text dev pool (18 cases, all text-only, seqlen 1001-1005, dev.txt
bypass flow; pre = the same Conv2D-chain AIRs, post = the CausalConv1d
AIRs):

| item | pre (Conv2D chain) | post (CausalConv1d) |
|---|---|---|
| pooled bench (closed loop, seed 42) | warmup 32 / 2,048 measured requests | per-case sweep: warmup 16 / 400 x 18 cases |
| e2e P50 (ms) | 26.557 (pool avg 26.566, P99 26.805) | **22.237-22.634 per case, mean 22.445** |
| qps | 37.64 (pool) | **44.10-44.95 per case, mean 44.53** (+18.3% on the means) |
| CPU preprocessing (official chain, outside the timing window) | - | avg 1.60 ms/case, worst P99 1.94 ms (`work/devcase/preproc-timing.json`) |

Evidence: `work/devcase/bench/pool/` + `work/devcase/logs/bench-pool.log`
(pre pool), `work/devcase/bench-new/case/*` (post 18-case sweep, 09:15),
`work/devcase/preproc-timing.json`.

## Appendix: RMSNorm gamma pre-fold + gated-norm gamma fold (2026-11-11)

Two independent commits against the long-text devcase profile (the restored
msprof showed 61 per-request `Add` nodes recomputing `1.0 + weight` on frozen
parameters at the iteration head with ~5 us of device idle before each, plus
the gated norm's separate `weight * n` multiply):

- **A1 (`70a4328`, bit-identical)**: `from_folder` bakes `1.0 + weight` into
  every text-side StartLuxRMSNorm parameter (bf16 domain, same operand order)
  with a config-derived count assertion (2/layer + final + 2 per
  full-attention layer = 61).  Same-day equivalent refactor: the fold now
  lives in the class itself as a `register_load_state_dict_post_hook`
  (unconditional +1, no state marker -- a marker that skipped a re-load
  would leave raw values in); the `from_folder` walk and count assertion are
  gone.  AIR bytes unchanged, infer outputs bit-identical, gates unchanged
  (validation.md 2026-11-12).
- **A2 (`0a1e1f6`, numerically equivalent)**: the gated norm passes its
  trained weight as the RmsNorm gamma directly (`rms_norm(x, w) ==
  w * rms_norm(x, 1)`), removing 18 OnesLike + 18 Mul per request.

All numbers below: `dev-intent-q00-r0`, chip 14, warmup 16 / 400 measured /
seed 42, same-session baseline for every comparison.

| item | baseline | A1 | A2 (A1+A2) |
|---|---|---|---|
| AIR outputs vs baseline | - | **bit-identical** (outputs.bin md5 16a42648f00c39eb8098405e0605f7aa) | 468/468 floats differ, max logit |dp| 0.091, cosine 0.999993 |
| numeric gate (vs CPU FP32 golden) | worst cosine 0.999995 / relL2 0.004779 | unchanged (bit-identical) | **0.999993 / 0.004937** (gate 0.999 / 0.02) |
| business gate (dev threshold 0.02) | worst |dp| 0.013914 | unchanged | **0.010254** (improved) |
| text AIR bytes | 1,507,412,645 | 1,507,343,564 (-69 KB) | **1,507,309,063** (-34 KB more) |
| bench e2e avg (ms) | 22.177 | 22.156 (variance band) | **21.607** (two runs 21.681 / 21.533; -0.57 ms / -2.57%) |
| qps | 45.09 | 45.13 | **46.28** (+2.64%) |

msprof re-collection (`work/devcase/prof-gammafold-devq00/`, restored via
`tools/restore_matmulv3.py`) confirms the structural removal:

| metric (steady state, 416 profiled execs) | pre | post |
|---|---|---|
| device ops per request | 947 | **850** (-61 Add, -18 OnesLike, -18 Mul) |
| device busy / idle per request | 21.88 / 1.06 ms | **21.51 / 0.64 ms** |
| device utilization | 95.4% | **97.1%** |
| wall per request under msprof | 22.94 ms | **22.14 ms** (-3.5%) |

Honest reading: the ~0.4 ms prologue-idle saving is profiling-specific
(msprof slows host launches; at bench level A1 measures within the variance
band) -- A1 is kept as a strictly structural cleanup with a bit-identity
proof.  A2's -0.57 ms (the removed Mul node time plus its dispatch) is real
at bench level and is the whole measured gain.

Evidence: `work/devcase/infer-base-a1/` + `infer-a1/` + `infer-a2/`
(outputs.bin pairs), `bench-a1-base/`, `bench-a1/`, `bench-a2-r1/`,
`bench-a2-r2/`, `prof-gammafold-devq00/`; commits 70a4328 / 0a1e1f6.

## Appendix: MegaGdnPrefillV2 long-text re-probe (2026-11-12, stage-1 no-go)

The 2026-11-05 whole-model rejection rested on short-sequence premises (128
alignment waste ~34% at S85, B2 -51%, image-case business fails); the devcase
pool is text-only B=1 at S=1001-1005, so the re-probe was warranted.  Stage 1
was a standalone single-op probe (no model change, no AIR re-export) at the
exact model geometry (NV=16 / DK=DV=128 / C=6144, tokens=1001, batch=1),
operator and install unchanged from the 2026-11-05 fixed build.

| item | result |
|---|---|
| break-even (native chain per GDN layer, post-gamma-fold msprof) | 611 us (CausalConv1d 66 + 3x Slice 39 + 2x L2 RmsNorm 49 + Fill 8 + CGDR 331 + gated RmsNorm 28 + silu/cast/mul/sigmoid tail ~90) |
| mega graph leg (torchair default mode, pipelined) | **685 us/call** (827 on a cold kernel_meta; scratch allocs ~15 us included) |
| mega eager leg | 1,310 us/call (bridge overhead; not the model route) |
| verdict | **LOSS: even the best graph-leg number is +12% over the whole native chain**; x18 layers = +1.3 ms/request (-5.8% throughput) -> NO-GO |
| accuracy side-finding | eager vs CPU FP32 golden relL2 **0.005017** (gate 0.02, better than the S~99 whole-model 0.0101); graph vs eager 0.0; the 1001%128=9 **remainder-chunk path is correct** (never exercised before); the fp16 pipeline does not degrade with length |

Conclusion: the fused-op net-negative finding extends to the long-text
workload at the kernel level -- the 128-alignment argument was necessary but
not sufficient.  Integration-specific risks (three 128x128 mask Consts vs
the positional bake; the per-layer cu_seqlens D2H read-back sync) stay
unexercised by design.  An environment hazard was found and documented while
probing: pip's `fla_npu` ships `fla_npu_opp_env.pth` which hijacks
`ASCEND_CUSTOM_OPP_PATH` at every interpreter start, and mega's activator
(old search order, CUSTOM_OPP_PATH first) then dlopens fla_npu_transformer's
same-named `libcust_opapi.so` (`undefined symbol: mega_gdn_prefill_run_v2`);
causal_conv1d's activator already carries the hardened search order.  See
validation.md 2026-11-12 for the workaround and the follow-up if the op is
ever re-integrated.

Evidence: `work/devcase/mega-probe/` (probe_s1001.py, golden_t1001.pt,
probe.log..probe5.log), chip 14 idle throughout.

## Appendix: GDN zero-state hoisted to one shared Fill (2026-11-12)

Each of the 18 GDN layers zeroed its own `[B416128x128]` initial_state before
the ChunkGatedDeltaRule call (18 Fill nodes, ~8.0 us each, 0.14 ms/request of
device time).  Eliminating the tensor entirely is blocked twice over: the GE
prototype REG_OP makes `initial_state` a REQUIRED input (only `g` is optional;
the eager host likewise rejects None -- "initial_state cannot be empty"), and
the dynamic batch axis rules out a baked Constant (no compile-time B bound,
see the model README).  The route that IS open: the kernel never writes
initial_state (probe: still all-zero after the call, final_state is a
separate output, reuse-vs-fresh bit-identical), so the 18 layers share ONE
zero tensor created once in StartLuxTextModel.forward.  Fill nodes per
request 18 -> 1.

Validation: AIR outputs bit-identical to the pre-hoist build (outputs.bin
md5 14898c64acf4de3b689fdee99220f23c); gates unchanged (worst cosine
0.999993, worst relL2 0.004937, business worst |dp| 0.010254); text.air
1,507,309,063 -> 1,507,247,677 bytes (-61,386, the 17 Fill nodes and their
descriptors).

Bench (idle window, chip 14, 07:0x): same-window A/B with the pre-hoist AIR
rebuilt exactly from commit 555dafb, both legs back-to-back, 2 runs each:

| leg | runs avg (ms) | mean | qps |
|---|---|---|---|
| pre-hoist | 21.640 / 21.559 | 21.599 | 46.30 |
| hoist | 21.456 / 21.257 | **21.356** | **46.82** |
| delta | | **-0.243 ms (-1.12%)** | **+1.14%** |

The four run ranges do not overlap (21.257, 21.456 < 21.559, 21.640); the
gain exceeds the ~-0.14 ms fill-time estimate (the removed dispatch helps
too).  After the A/B the delivered source was restored and re-exported:
AIR byte count reproduced (1,507,247,677) and infer outputs bit-identical
to the benched AIR (md5 14898c64acf4de3b689fdee99220f23c).

msprof re-collection (`work/devcase/prof-zerostate-devq00/`, restored):
Fill 416 total = **1 per request** (was 7,488 = 18), **850 -> 833 device
ops per request**, busy 21.51 -> **21.15 ms**, idle 0.64 -> 0.54 ms,
utilization 97.1 -> **97.5%**, wall under msprof 22.14 -> **21.69 ms**.

Measurement-pollution incident (kept for the record): between ~06:25 and
the idle window all 16 chips were tenant-occupied (chip 14 at 87% AICore /
52 GB); the contended attempts read 37.7-38.1 ms but the pre-hoist control
AIR read the same 37.9 ms/case under the same conditions (21.5/0.55 ~ 39
ms matches the stolen-share arithmetic) -- contention, not regression.
All clean measurements of the day (A2 bench, the mega probe) predate the
occupancy window.

Evidence: `work/devcase/infer-zerostate{,-final}/` (vs `infer-a1hook/`),
`bench-prehoist-r{1,2}` + `bench-zerostate-r{3,4}` (the clean A/B),
`bench-zerostate-r{1,2}` (marked polluted), `prof-zerostate-devq00/`, the
mutation probe logged in validation.md 2026-11-12.
