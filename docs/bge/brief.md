# BGE 模型优化需求

需求基线，2026-10-26 定稿；执行细节以 [models/bge/README.md](../../models/bge/README.md) 为准。

## 模型与路线

原始模型是业务侧提供的 BGE ONNX（内联权重约 1.35 GB），由环境变量 `BGE_MODEL_PATH` 提供。唯一路线是原始 ONNX → GE Session 推理（`graph_run_mode=0`）：保持原始图与权重，不重新启用 Slice 改写，不走 ATC/OM/ACL 第二路线。

## 接口

已有适配器接口记录为八个输入：`input_ids`、`attention_mask`、`long_cid3_raw`、`long_action`、`long_time_bucket`、`stat_dense_raw`、`stat_ratio`、`stat_count`；其中 `stat_ratio` 为 FP32，其余为 INT64。输出为 FP32 `scores[B,10112]`，支持 batch、文本与行为长度的动态接口。本轮用实际 ONNX 与适配器核对 dtype、范围和维度关系，不仅依赖历史说明。

## 正确性材料与要求

- 输入：仓库内 `models/bge/fixtures/sample_inputs.json`，5 条。
- 预期输出：仓库内 `models/bge/fixtures/expected_outputs.json`，5 条。
- 复用原始模型 CPU ONNX Runtime 的完整分数参考，并逐条比较业务 Top10。`dense_id = 输出索引 + 1`，ID 与顺序精确匹配。
- 当前业务分数阈值：绝对误差 ≤ `0.001`，相对误差 0；当前全量分数阈值：cosine ≥ `0.9999`、relative L2 ≤ `0.001`、max absolute error ≤ `0.001`。保持这些既有阈值，不为通过而放宽或修改 expected。
- 并发、异步、缓冲复用、batch/实例配置变化后，另验请求与输出对应、混合 shape、生命周期和约定稳定性；CPU 比较不混入正式性能计时。

## 可复用入口

运行入口、参数与分步语义见 [models/bge/README.md](../../models/bge/README.md)；`ge_execute` 的命令与产物见 [ge-execute-usage.md](ge-execute-usage.md)。

## 已确认的性能契约与预算

目标：在正确性与资源约束满足的前提下，尽可能提高约定工作负载的成功吞吐，分别报告固定 case 与五条样例随机回放。用户明确暂不设置 p99 上限；持续报告观测 p99 和吞吐—延迟关系，不声称通过延迟 SLA。

1. **p99 上限：无硬性约束。** 不填写虚构的上限，也不将“不受约束”报告为“满足某 SLA”。
2. **计时范围：包含输入输出搬运。** 从每请求首次 H2D 前，到完整 FP32 输出 D2H 完成、Host 可读取后；包含描述符和必要同步。吞吐单位 requests/s，batch=1 时也是业务样本/s，不包含文件 IO、分词或服务端排队。
3. **性能负载：固定单 case＋五例随机两场景。** 固定场景默认选原顺序首个 S8 样例 `case-0002`（B1、L128、stat20），重复完全相同的真实输入；随机场景使用五条真实输入等概率有放回闭环回放，S=6/9/8/13/8、L128。两场景各建基线和最终复测、分别计算收益；所有既有 99 数字仅属于五例随机场景；只解释约定范围，不外推生产容量。
4. **资源与可调范围：全核起步，核数参与调优。** 在一颗可独占的授权 NPU 上比较全核/限核与多实例组合；本轮设计保持 batch=1。
5. **设备占用约束：先设计，等待可用设备。** 不启动性能测试，不终止其他进程。恢复前重新检查占用和物理映射，不能依据历史快照认定空闲。

## 修改与交付

沿既有 ONNX→GE Session 路线，优先分析 profiling 后确定图、搬运、缓冲、调度等候选，不预设某种优化一定有效。原模型与验收数据保持原样；允许在精度门槛和外部 IO 不变时尝试 FP16/混合精度与等价融合，候选另存并保留原图。全局 CANN/驱动或支持范围变化不自动执行。

每个优化点记录到 `docs/bge/experiments/OPT-xxx.md`，原始运行证据保存在实际执行位置并记录回传路径。所有提升、下降、失败、无收益与撤回项都进入 [delivery-100.md](delivery-100.md) 的逐条汇总。
