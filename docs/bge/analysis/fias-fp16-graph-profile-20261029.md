# BGE FP16＋FIAS：最终 GE 图与逐 kernel 诊断

2026-10-29；100 机器、`hejun-cann9-bge` 容器，隔离项目 `merge-review-20261029`。本次只采图和 profiling，不修改原始 ONNX、正式 runner 或 OPT-019 的无采集性能结果。

## 对象与采集

- 构建输入为 `work/bge/build/model.onnx`（SHA-256 `5e641362d99c9a1afb6dfce70f47b4ac07f1a7c88272fdfc791ea04ae84f4822`）及 `weights.data`（`5fabfb4fb33a0db0eaa0537e011df2189150f04b590eca12170dec2cf598dd0d`）。结构 pass SO 为 `libbge_fias_structure_pass.so`（`70fa93511b3805e982990f14cc1be9f17cda6277153053311d323cdec4b2e3fb`）。
- GE Session mode=0、`precision_mode=force_fp16`、`aicore_num=12|24`，物理 NPU 7／逻辑设备 14。固定 `case-0000`：B=1、S=6、行为长 128、统计宽 20。外部输出仍为 FP32 `scores[1,10112]`。
- 图作业只用 1 实例，`DUMP_GE_GRAPH=2`、`DUMP_GRAPH_LEVEL=3`，得到不含权重的 Build 图；profiling 作业用当前 10 实例，10 次/实例预热、总计 100 次测量请求。两者使用同一 ONNX、权重、pass、GE 选项和输入。图和 profiling 分开运行，两个受管任务均 exit code 0；启动前物理 NPU 7 空闲，采集后已释放。
- 完整远端证据目录：`/export/home/ext.hejun23/workspace/bge-opt-100/work/fias-all-20261029/diagnostics-fias-20261029/`。其中 `graph-plan.json`、`profile-plan.json`、`manifest.json`、`job-{graph,profile}/status.json`、`*-stdout.log`、`ge-graph/`、`profile-runtime/bench-profiling/PROF_000001_20261029175515681_00490000MCNNMNJC/`、`msprof-{parse,query,export}.log` 和 `profile-analysis.json` 可复核全过程。

## 最终 GE Build 图

31 MB 无权重图 SHA-256：`87608ea8c8b020f6115cc615f46a0bb9aeab3f1bff0a2f75a5a3ebb3bc5af97b`。图中共有 3,561 个 op 记录，包含常量和形状子图，**不是每次执行的 kernel 数**。pass 日志报告命中/替换 24 处，优化后 24 个 FIAS 的 Q/K/V/输出为 FP16、mask 为 BOOL。

| 结构 | Build 图记录数 | 实际含义 |
|---|---:|---|
| `FusedInferAttentionScore` | 24 | 24 层 attention 已替换，原 QK/Softmax/V 不再各自执行 |
| `MatMulV2` | 144 | encoder 每层六个带 bias 投影；输入、bias、输出描述符均 FP16 |
| `LayerNorm` / `Gelu` | 49 / 24 | 仍是已融合的单算子，不是源图分解链 |
| `LpNormV2` | 5 | 五条 L2 链仅归约部分合并；其后的 `Maximum→Expand→RealDiv` 仍在 |
| `SoftmaxV2` | 1 | 四路门控的 Softmax，不属于 Transformer attention |
| `GatherV2` | 209 | 多数是形状/常量子图记录；实际每请求只执行 10 个 Gather kernel |

图中还留有 `Add` 55、`BatchMatMulV2` 7、`Cast` 13、`Maximum/Expand/RealDiv` 各 7。图记录数与执行次数不一一对应；例如 profiling 每次执行 147 个 `MatMulV2` 和 4 个 `BatchMatMulV2`。

## Profiling 归属与耗时

CANN 9.0.0 `msprof --parse/--query/--export` 全部退出 0，导出非空 `op_summary` 和 `step_trace`。10 个 Model ID 共 200 个 step，事件日志是每实例 10 次预热加总计 100 次测量；按 Model ID 和 `[Iteration End − Iteration Time, Iteration End]` 将 78,400 条 kernel 行唯一归属到 step，遗漏和多重归属均为 0。每 step 恰好 392 条，100 个测量 step 的 op 类型顺序完全一致；剔除前 10 step/实例后才计算下表。`op_summary` 没有 Infer ID，因此不用它直接按请求编号分组。完整分组脚本为 `tools/bge_fias_all_pass/analyze_diagnostic_profile.py`，摘要见 [机器可读结果](../evidence/20261029-fias-fp16-profile-analysis.json)。

下列“μs/请求”是 **100 次测量中该组 kernel 的 Task Duration 总和 ÷100**，10 个实例并发，分组时间可与其他流重叠；分母为每请求平均 4,614.43 μs 的 kernel 时间总和，既不是墙钟延迟，也不能直接换算可提升的 QPS。子项与其父项重叠，不能跨行求和。

| 结构/执行区域 | kernel 数/请求 | 累计 μs/请求 | kernel 时间占比 | 说明 |
|---|---:|---:|---:|---|
| encoder 全部 | 296 | 3,911.45 | 84.77% | 下列 QKV、FIAS、FFN、LN 均在此范围 |
| 其中 Q/K/V 投影 | 72 | 1,021.35 | 22.13% | 通过每个 FIAS 前的三次 MatMul 和稳定任务序列定位 |
| 其中 FFN 上投影＋下投影 | 48 | 1,525.07 | 33.05% | 分别 709.67 / 815.40 μs |
| 其中 FIAS | 24 | 487.02 | 10.55% | 平均每层约 20.29 μs |
| 全部 LayerNorm | 49 | 316.60 | 6.86% | 含 1 个 embedding LN、48 个 encoder LN |
| 候选向量常量链 | 6 | 334.96 | 7.26% | 两次 BatchMatMul 共约 268.35 μs，另有 L2 四核 66.61 μs |
| 长期行为分支 | 33 | 129.71 | 2.81% | Gather、Add、Mul、ReduceSum、除法及 L2 等 |
| 统计分支 | 24 | 70.21 | 1.52% | Gather、权重汇聚及 L2 等 |
| 门控加权片段 | 10 | 49.21 | 1.07% | MatMul、Softmax、三次 Slice/Mul 和两次 Add |
| 五条 L2 归一化链合计 | 20 | 104.37 | 2.26% | 其中候选常量链 66.61 μs；其余四条合计 37.76 μs（0.82%） |

候选常量链值得特别注意：源 ONNX 的 `/cid_projection/MatMul` 两个输入都是 initializer，后续归一化和 `/MatMul` 也只依赖这些固定权重，但本次 profiling 证明这六个 kernel **每次请求都执行**。它可考虑在模型构建时预计算为固定候选表；这属于常量折叠/预计算候选，尚未做正确性或性能试验。QKV 投影是另一大块实际成本，但合成一次宽 MatMul 会增加拆分/布局成本，不能把当前 1.02 ms 直接视为可消除时间。剩余四条动态 L2 链合计占比不到 1%，从本次证据看不宜先于上述两项。

本轮 profiling 的请求级平均延迟 15.80 ms、p99 21.59 ms、100 次测量无错误；这些数字包含采集开销和 10 实例竞争，**不作为正式性能基线**。本报告只定位执行热点，不声明候选优化收益。正式收益需要保持五例完整精度门槛，在相同固定/随机负载和 10 实例配置下关闭 profiling 做独立 A/B。
