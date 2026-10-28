# 100 原始 BGE：首轮独立验收

结论：100 逻辑设备 2 上，原始 FP32 模型的 2000 请求完整输出关联压力通过；fixed-case-0002 与 random-five 各自的第一正式性能轮通过事件与统计契约。**这是首轮证据，不是三轮聚合、最终多实例配置或 10 分钟稳定性通过。** 未设置 p99 硬上限，不作延迟 SLA 通过声明。

测试角色没有启动 NPU 任务，也未重新计算 CPU reference。先仅下载 34 份小文件、约 168KB；收到主控 stage2 完成通知后，在设备任务停止的 CPU 窗口读取原始压力输出和事件，独立执行检查器。所有新远端报告只写入 `/export/home/ext.hejun23/workspace/bge-opt-100/work/tester-initial/`，未覆盖开发结果或 progress。

## 冻结身份与原始压力

实际项目为 `/export/home/ext.hejun23/workspace/bge-opt-100`，容器 `hejun-cann9-bge`，100 主机；逻辑 2 = NPU1/Chip0，目标 BDF `0000:99:00.0`，同时协调同卡 `0000:9B:00.0` 的物理锁和空闲门禁。

| 产物 | 本轮独立校验 SHA256 |
|---|---|
| runner | `8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d` |
| 原始 GE 打包 ONNX | `5e641362d99c9a1afb6dfce70f47b4ac07f1a7c88272fdfc791ea04ae84f4822` |
| 外部 weights.data | `5fabfb4fb33a0db0eaa0537e011df2189150f04b590eca12170dec2cf598dd0d` |

三次作业均为 GE mode=0、FP32、单实例全核；本轮原图未加载融合插件。独立运行身份检查重读实际二进制、图、权重和实际 plan/input，绑定 job request/status、结果目录、runtime seed/pool、事件 run ID。三次 native/包装器退出均为 0，任务进程组停止；前后两芯片 gate 均为 clear。见 [独立 run-identity.json](../../../delivery/supporting-evidence/work/bge-opt-100/tester/initial-independent/run-identity.json) 和 [下载原始小文件清单](../../../delivery/supporting-evidence/work/bge-opt-100/tester/initial-small/collection.json)。

`base-pressure-d2` 的 80,896,000 字节原始 scores 已逐请求独立比较，所有 2000 请求通过。manifest、measure events、runtime cohort 均恰为 2000，request ID 完整为 0～1999；事件与输出的 instance/case/run ID 一一对应。

| 指标 | 独立结果 | 门槛 |
|---|---:|---:|
| 最小 cosine | 0.999999905873287 | ≥0.9999 |
| 最大 relative L2 | 0.0004473780329645365 | ≤0.001 |
| 全向量最大绝对误差 | 0.00021588802337646484 | ≤0.001 |
| Top10 最大分数绝对误差 | 0.00009595939636231066 | ≤0.001 |
| Top10 ID / 顺序 | 全部一致，下标 +1 | 精确一致 |
| 失败请求 | 0 | 0 |
| 五例请求数 | 430 / 421 / 394 / 422 / 333 | 每实例每例 ≥20 |

参考仍为冻结的原始 CPU golden 和用户 expected，没有从候选结果生成参考。外部输出有限 FP32 `[1,10112]`。完整独立报告：[pressure-accuracy.json](../../../delivery/supporting-evidence/work/bge-opt-100/tester/initial-independent/pressure-accuracy.json)。

## 两种场景的第一正式轮

均包含 H2D、描述符处理、GE 执行与同步、完整 scores D2H；闭环单在途，capture/profiling/dump 关闭。测试从原始事件独立重算窗口完成数、QPS、最近秩 p50/p99、预热、排空与原始输入字节身份；没有将压力 capture 轮的耗时用作性能结果。

| 场景 / run | 实际窗口 | 窗口内成功请求 | QPS | p50 | p99 |
|---|---:|---:|---:|---:|---:|
| fixed-case-0002 / base-fixed-r1 | 60.004877128秒 | 10262 | 171.019432 | 5.797430ms | 7.079841ms |
| random-five / base-random-r1 | 62.329194857秒 | 10000 | 160.438459 | 6.041532ms | 8.768116ms |

两轮均达到单轮正式门槛 ≥60 秒且 ≥10000 完成请求；错误、超时、未归档请求和末尾在途为零。固定场景实际池只有 case-0002，随机场景实际池为原五例。两者分别保留基线，不能相互计算优化收益，也不能用 99 的数字替换本机基线。

证据：[fixed 独立报告](../../../delivery/supporting-evidence/work/bge-opt-100/tester/initial-independent/base-fixed-r1-performance.json)、[random 独立报告](../../../delivery/supporting-evidence/work/bge-opt-100/tester/initial-independent/base-random-r1-performance.json)。当前仅各一轮，不计算三轮中位吞吐或宣称性能稳定。

## 执行命令与范围

以下命令均在 100 容器内，仅验证已完成任务的只读证据；`ROOT` 为上文项目根，`OUT=$ROOT/work/tester-initial`。精确 argv、退出码与用时见 [execution.json](../../../delivery/supporting-evidence/work/bge-opt-100/tester/initial-independent/execution.json)。七个检查的退出码全部为 0。

```sh
bash "$ROOT/tools/env_run.sh" python3 "$ROOT/tools/checks/verify_plan_inputs.py" --plan "$ROOT/work/plans/base-pressure-d2.json" --report "$OUT/base-pressure-d2-inputs.json"
bash "$ROOT/tools/env_run.sh" python3 "$ROOT/tools/checks/verify_plan_inputs.py" --plan "$ROOT/work/plans/base-fixed-r1.json" --report "$OUT/base-fixed-r1-inputs.json"
bash "$ROOT/tools/env_run.sh" python3 "$ROOT/tools/checks/verify_plan_inputs.py" --plan "$ROOT/work/plans/base-random-r1.json" --report "$OUT/base-random-r1-inputs.json"
bash "$ROOT/tools/env_run.sh" python3 "$ROOT/tools/checks/validate_run_identity.py" --runs base-pressure-d2 base-fixed-r1 base-random-r1 --identity-manifest "$ROOT/tools/identity.json" --report "$OUT/run-identity.json"
bash "$ROOT/tools/env_run.sh" python3 "$ROOT/tools/checks/verify_outputs.py" pressure --manifest "$ROOT/work/base-pressure-d2/manifest.jsonl" --binary "$ROOT/work/base-pressure-d2/outputs.bin" --events "$ROOT/work/base-pressure-d2/events.jsonl" --instances 1 --min-requests 2000 --min-per-case 20 --report "$OUT/pressure-accuracy.json"
bash "$ROOT/tools/env_run.sh" python3 "$ROOT/tools/checks/validate_performance.py" --performance "$ROOT/work/base-fixed-r1/result.json" --events "$ROOT/work/base-fixed-r1/events.jsonl" --plan "$ROOT/work/plans/base-fixed-r1.json" --instances 1 --min-seconds 60 --min-requests 10000 --report "$OUT/base-fixed-r1-performance.json"
bash "$ROOT/tools/env_run.sh" python3 "$ROOT/tools/checks/validate_performance.py" --performance "$ROOT/work/base-random-r1/result.json" --events "$ROOT/work/base-random-r1/events.jsonl" --plan "$ROOT/work/plans/base-random-r1.json" --instances 1 --min-seconds 60 --min-requests 10000 --report "$OUT/base-random-r1-performance.json"
```

部署在 100 的 `verify_outputs.py`、`workload_identity.py`、`validate_run_identity.py` 与本地受测文件逐项 SHA256 相同。此前发现的 runtime scope/arrival/capture 三项拒绝缺口已修复，独立补充四项 CPU 回归全部通过，见 [migrated-acceptance-guardrails.log](../../../delivery/supporting-evidence/work/bge-opt-100/tester/migrated-acceptance-guardrails.log) 及 [deployed-checker-identity.json](../../../delivery/supporting-evidence/work/bge-opt-100/tester/initial-small/deployed-checker-identity.json)。合成事件只证明检查器逻辑，不用于模型性能数字。

## 证据边界与后续

- 本报告不覆盖三轮正式聚合、多实例或限核配置、融合收益和 10 分钟随机稳定性；这些尚需绑定各自冻结配置独立验收。
- 三次作业的 `plan.monitor_resources` 未设置；现有轮内 `resources.jsonl` 只约 1KB，需另有同期设备/Host PID 见证才能确认轮内没有外部占用。已核对的是前后两芯片空闲及本任务正常退出，不能据此宣称整个窗口均独占。
- 单轮 `formal_round` 是统计与事件契约通过，环境证据边界另列；不冒充最终性能接纳。最终稳定性必须有随机 ≥600 秒的实际 RSS/HBM、约 10 秒采样与吞吐序列，检查资源增量及持续增长趋势。
- 固定输入和五例随机仅代表这五条给定输入，不代表业务到达分布、任意动态 shape 或生产容量。
