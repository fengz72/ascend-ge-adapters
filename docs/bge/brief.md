# BGE 模型优化需求

日期：2026-10-26。状态：用户授权使用逻辑设备2恢复执行，先确认空闲，验证原图GE mode=0正确性，建立包含输入输出搬运的基线，再尝试FP16与算子融合，逐项记录。仍执行设备占用门禁。由本地 Codex 的 npu-model-workflow 编排，通过 SSH 在实际容器执行。实施方案见 [design.md](design.md)，恢复信息见 [progress.md](progress.md)。

## 模型与执行位置

- SSH 主机：`99`；容器：`hejun-cann9-bge`。
- 模型项目根：`/export/home/ext.hejun23/workspace/bge`。
- 原始模型：`/export/home/ext.hejun23/workspace/bge/model.onnx`，本次核对存在，大小 1,345,962,755 字节；开始正式实验前重新记录校验值与接口身份。
- 现成 runner：`/export/home/ext.hejun23/workspace/bge/ascend-ge-adapters`；分支 `codex/seven-stage-framework`，本次 HEAD 为 `441b245fc18ee2a78156becf80488ee16ad3474d`，已合入远端更新并提交先前本地修复，工作区干净。
- 实验快照从上述完整版本建立；恢复时重新核查工作区，若新增有效未提交修改及必要未跟踪文件，也须纳入快照。当前模型、样例、ONNX 接口和设备占用身份见 [preflight.json](evidence/preflight.json)。
- 本地 workflow 与模型文档根：`/Users/ext.hejun23/Documents/Codex/2026-10-19/ssh`；本地副本不是远端实际执行目录。运行所需的小脚本按需同步，设备锁须在实际执行主机生效。
- 远端设计文档归档：`/export/home/ext.hejun23/workspace/bge/ascend-ge-adapters/docs/bge`，包含本需求、设计、进度、角色审查与必要证据；同步校验见 [transfer-validation.json](evidence/transfer-validation.json)。

## 唯一路线与现有配置

原始 ONNX → GE Session 推理（mode=0）。先保持原始图与权重，不重新启用 Slice 改写；具体运行选项及编译结果在本轮预检中复核。

已读取远端 `models/bge/config.yaml`：source.ref 为 `${BGE_MODEL_PATH}`，CPU 参考和 GE 构建的浮点 dtype 均为 `float32`，`precision_mode=must_keep_origin_dtype`，target 配置为 `Ascend910_9382`，当前核配置 `aicore_num=4|8`，实例数 1。配置值不等于本轮设备独占或扩容授权，实际设备身份与可用资源另行核对。

已有适配器接口记录为八个输入：`input_ids`、`attention_mask`、`long_cid3_raw`、`long_action`、`long_time_bucket`、`stat_dense_raw`、`stat_ratio`、`stat_count`；其中 `stat_ratio` 为 FP32，其余为 INT64。输出为 FP32 `scores[B,10112]`，支持 batch、文本与行为长度的动态接口。本轮先用实际 ONNX 与远端适配器核对 dtype、范围和维度关系，不仅依赖历史说明。

## 正确性材料与要求

- 输入：`/export/home/ext.hejun23/workspace/bge/sample_inputs.json`，本次读取为 5 条。
- 预期输出：`/export/home/ext.hejun23/workspace/bge/expected_outputs.json`，本次读取为 5 条。
- 复用原始模型 CPU ONNX Runtime 的完整分数参考，并逐条比较业务 Top10。`dense_id = 输出索引 + 1`，ID 与顺序精确匹配。
- 当前业务分数阈值：绝对误差 ≤ `0.001`，相对误差 0；当前全量分数阈值：cosine ≥ `0.9999`、relative L2 ≤ `0.001`、max absolute error ≤ `0.001`。保持这些既有阈值，不为通过而放宽或修改 expected。
- 历史五条样例已通过精度比较，但不是本轮新证据。先核对模型/代码/数据/环境身份，并按实际改动重新验证。
- 并发、异步、缓冲复用、batch/实例配置变化后，另验请求与输出对应、混合 shape、生命周期和约定稳定性；CPU 比较不混入正式性能计时。

## 可复用入口

命令在远端容器的 runner 根执行，使用其原有环境设置与运行脚本：

```sh
export BGE_MODEL_PATH=/export/home/ext.hejun23/workspace/bge/model.onnx
bash models/bge/run.sh data
bash models/bge/run.sh baseline
bash models/bge/run.sh prepare
bash models/bge/run.sh build --device 0
bash models/bge/run.sh infer --device 0
bash models/bge/run.sh compare
bash models/bge/run.sh bench --device 0
```

这些是待实际核验的既有入口，不表示已经获准占用 device 0。运行前检查当前工作副本、依赖、磁盘/内存、设备任务与物理设备映射；保留已有产物，不覆盖旧 run。完整串行入口也可用 `bash models/bge/run.sh run --device 0`，但会重跑各阶段，应按版本与证据选择所需步骤。

## 已确认的性能契约与预算

2026-10-27更新：下列第5项12组/2小时是先前实验约束，用户已明确取消该总量上限，改为使用当时空闲且可独占的设备；仍要求逐作业超时、占用与释放检查、累计设备时间记录。当前100机器实例数搜索及实测停止条件以[设计v6](design.md)和[结果报告](analysis/100-instance-scaling-results-20261027.md)为准。

目标：在正确性与资源约束满足的前提下，尽可能提高约定工作负载的成功吞吐，分别报告固定case与五条样例随机回放。用户明确暂不设置 p99 上限；持续报告观测 p99 和吞吐—延迟关系，不声称通过延迟 SLA。QPS 暂未知，先建立可信基线。

1. **p99 上限：无硬性约束。** 不填写虚构的上限，也不将“不受约束”报告为“满足某 SLA”。
2. **计时范围：包含输入输出搬运。** 从每请求首次 H2D 前，到完整 FP32 输出 D2H 完成、Host 可读取后；包含描述符和必要同步。吞吐单位 requests/s，batch=1 时也是业务样本/s，不包含文件 IO、分词或服务端排队。
3. **性能负载：固定单case＋五例随机两场景。** 固定场景默认选原顺序首个S8样例`case-0002`（B1、L128、stat20），重复完全相同的真实输入；随机场景使用五条真实输入等概率有放回闭环回放，S=6/9/8/13/8、L128。两场景各建基线和最终复测、分别计算收益，所有既有数字仅属于五例随机场景；只解释约定范围，不外推生产容量。
4. **资源与可调范围：全核起步，核数参与调优。** 在一颗可独占的授权 NPU 上比较全核/限核与多实例组合；本轮设计保持 batch=1。原配置 `4|8` 只是历史配置，不是新的资源上限。
5. **有限预算：最多 12 组配置，累计设备占用不超过 2 小时。** 含失败、构建加载、复测和最终稳定性；主控采用含基线最多 12 组的保守计数，具体测量与停止规则见设计。
6. **设备占用约束：先设计，等待可用设备。** 2026-10-24 的只读预检显示 16 个逻辑设备均有外部进程占用；不启动性能测试，不终止其他进程。恢复前重新检查占用和物理映射，不能依据本次或历史快照认定空闲。

当前 benchmark 配置仅预热 1 次、测量 6 次，且现有 bench 不做 D2H，不能用于本轮完整搬运口径的性能基线。需先补测量边界和多实例输出关联验收能力，再开展性能实验。发生器实际施压、缓存未命中、异步完成、队列/错误及搜索后独立验收均按工作流规则检查。

## 修改与交付

沿既有 ONNX→GE Session 路线，优先分析本次 profiling 后确定图、搬运、缓冲、调度等候选，不预设某种优化一定有效。原模型与验收数据保持原样；用户已明确允许在精度门槛和外部IO不变时尝试FP16/混合精度与等价融合，候选另存并保留原图。全局CANN/驱动或支持范围变化不自动执行。

每个优化点记录到本地 `docs/bge/experiments/OPT-xxx.md`，原始运行证据保存在实际执行位置并记录回传路径。所有提升、下降、失败、无收益与撤回项都进入最终 `docs/bge/acceptance.md` 的逐条汇总。当前未创建任何真实 OPT 或声称调优结果。

下一步：完成设计及独立审查，保留待实施的测量修正和测试任务。获得可独占设备并重新核查后，依次完成正确性、可信性能基线、有限搜索和独立验收；当前不启动设备任务。

## 2026-10-26 设备调整

用户确认16个逻辑设备性能一致、卡间差异在误差范围内，并授权改用逻辑设备10。此前设备2新正确性证据保留原设备身份；后续正式基线、候选及复测统一设备10（NPU5/Chip0，PCI0000:89:00.0），启动前重新核查。设备2因外部xllm占用不再发任务，不终止外部进程。总预算不重置。

## 固定与随机双场景补充

用户要求调优时可固定一个case便于比较，最终交付分别给固定case和多case随机的性能。固定case选择及逐输入哈希已记录在[workload-contract-v2.json](evidence/20261026/workload-contract-v2.json)。此为测量协议扩展，保持设备暂停、预算、精度阈值及全五例正确性/压力范围；固定case性能尚未运行。
