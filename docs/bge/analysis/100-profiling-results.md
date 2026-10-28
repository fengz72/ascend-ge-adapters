# 100 BGE Add＋LayerNorm：逐算子采集结果

2026-10-26。**此次局部 kernel 证据显示：全 48 处融合整体降低了目标 kernel 累计时间，但其中 24 个 FFN output 位置融合后更慢。** 这不是模型正式 QPS/p99 验收结论，也不能解释或重现 99 上单轮 −2.02% 的全部原因。

## 采集归属与可用性

分析只对已经结束的两份会话执行 CPU parse/query/export；没有新启动模型/NPU。六条命令均退出 0，原始与导出数据保留在 100：

- 原图：`/export/home/ext.hejun23/workspace/bge-opt-100/work/base-fixed-profile/bench-profiling/PROF_000001_20261026210417674_00011908ICPQNBKL`，native PID 11908。
- 融合 V2：`/export/home/ext.hejun23/workspace/bge-opt-100/work/fusion-fixed-profile/bench-profiling/PROF_000001_20261026210449993_00012632MHLRKARQ`，native PID 12632。
- 两者均为逻辑设备 **2**、Model ID **1**，物理锁为 `100/0000:99:00.0` 与 `100/0000:9B:00.0`。这是实际采集身份，不沿用最初方案的逻辑 14。
- `case-0002`、B1/S8/L128、FP32、GE mode 0、全核、单实例，输入 SHA 均 `27ce13c73cb35762ee7d0396d4425829d7e25a3b8ae2a12b37ed7f96b8690aa5`。native binary、图、输入与插件身份由受管验收结果绑定；分析再次核对该结果与本地拷贝的 plan/result SHA。
- 图 SHA `5e641362d99c9a1afb6dfce70f47b4ac07f1a7c88272fdfc791ea04ae84f4822`，3137 节点、400 initializers；V2 插件 SHA `76ea9fb0f1800369b023e22d71298e508dcfaa6f69f34bcf952af4a1b5fa48ed`。融合日志实际报告 `replacements=48`。

100 的 CPU parse/export 日志与机器报告在 `work/profile-analysis/`。31 份 CSV、step、计划、事件、日志与小 ONNX 已归档到本地 [profile-analysis](../../../delivery/supporting-evidence/work/bge-opt-100/profile-analysis/)。远端 CPU 窗口已释放，此后分析全部在本地进行。

## 如何排除预热并对齐

实际 op_summary 没有 Infer ID，不能假设其包含 20 次测量。两份 query 和 step_trace 都报告 **30 次执行**，事件中均为 10 次预热、20 次测量。

以 step_trace 的 `[Iteration End − Iteration Time, Iteration End]` 为区间，每条 kernel 的完整时间均唯一落入一个区间：原图每步 601 行、融合每步 553 行，全部 30 步一致，零遗漏/跨区间歧义。每个目标 Add、LN、融合节点在每步恰好执行一次。

单实例事件顺序、30 个 step 的顺序、目标节点出现次数完全一致；profiler step-end 与对应 event-complete 的时钟偏移极差仅原图 24.25 μs、融合 25.5 μs。由这些共同证据，将 step 1–10 关联预热、11–30 关联测量。绝对时钟未被假设相等。以下统计只选已关联的 step 11–30。

48 个匹配由实际融合节点后缀 `/BgeAddLayerNorm` 回溯原 LN，并在同一未修改 ONNX 中核对残差 Add 的输出连接到该 LN 的 ReduceMean 输入；不是用所有 Add 的总时间替代。

## 对齐的 kernel 比较

所有目标均为 `AI_VECTOR_CORE`，`Block Num=8`，输入 FLOAT，hidden=1024。每种目标各 960 行（48 位置 × 20 测量执行）。

|粒度|原 Add|原 LN|原 Add＋LN|融合 AddLayerNorm|变化|
|---|---:|---:|---:|---:|---:|
|全部 48 位置，每个目标片段均值|1.901 μs|4.832 μs|6.733 μs|4.607 μs|−31.58%|
|24 个 attention/output，片段均值|2.214 μs|6.661 μs|8.875 μs|4.318 μs|−51.34%|
|24 个 FFN output，片段均值|1.587 μs|3.003 μs|4.590 μs|4.895 μs|+6.65%|
|每个已关联测量执行的 48 片段 kernel 累计|91.227 μs|231.935 μs|323.162 μs|221.113 μs|−102.049 μs|

attention/output 的 24 位置全部更快，范围 −59.99% 至 −46.98%；FFN output 的 24 位置全部更慢，范围 +3.09% 至 +11.33%。每次执行目标任务数由 96 降至 48，全图 CSV 任务行由 601 降至 553。

例如 layer.0 attention/output：原 Add 2.355 μs＋LN 8.224 μs→融合 5.104 μs；同层 FFN output：原 Add 1.598 μs＋LN 3.017 μs→融合 5.075 μs。说明相同 shape/dtype/Block Num 也不能保证所有位置融合都获益；本次不能仅靠这些列确定导致 FFN 变慢的具体 kernel/缓存/调度机制。

## 能得出的结论与限制

- 融合实际执行，且不是所有位置都同向变化。可以把“只融合 24 个 attention/output，保留 FFN 原 Add＋LN”作为具体候选，单独验证精度与无采集性能。
- 局部 kernel 累计减少约 0.102 ms/测量执行，**不等于**端到端延迟减少 0.102 ms。任务可能重叠，Host 提交和依赖间隙也会变化；这里没有把 Task Wait Time 加到计算量中。
- 这是一次短采集 A/B，硬件计数使用 PipeUtilization；采集开销明显。runtime 的 profile QPS/p99 不用于收益表、SLA 或接纳决策。
- 不能据此认定所有 shape、多实例/限核组合都会加速，不能认定此时核数已最优。
- 未捕获该两次运行的最终 GE dump；原始残差边由相同 SHA 的 ONNX、实际 profile 名称、已加载 pass 身份与 48 次替换日志共同约束。不能把旧 99 编译图当作此次完整执行图。
- 不能依据融合节点的四个 output 描述符推断四份数据均写回 HBM，也未测出该候选实际访存量。

机器报告：[add-layernorm-comparison.json](../../../delivery/supporting-evidence/work/bge-opt-100/profile-analysis/comparison/add-layernorm-comparison.json)。960 组对应原始行号、迭代/请求映射、耗时与 Block Num：[matched-kernel-rows.csv](../../../delivery/supporting-evidence/work/bge-opt-100/profile-analysis/comparison/matched-kernel-rows.csv)。分析脚本：[compare_completed_profiles.py](../../../delivery/supporting-evidence/work/bge-opt-100/profiling-dev/compare_completed_profiles.py)。CSV/step 文件 SHA、PID、30 步覆盖、每一目标节点统计均可回溯。
