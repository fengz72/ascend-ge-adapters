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
