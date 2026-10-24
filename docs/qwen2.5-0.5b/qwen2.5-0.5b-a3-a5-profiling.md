# Qwen2.5-0.5B Prefix：A3与A5 profiling对比

> 本文按采集日期保留各版本的配置、逐算子对照及后续实验；报告中的推荐值属于对应时点。当前配置和最近调优入口见[文档索引](README.md)。

原始采集日期：2026-10-21（Asia/Shanghai）。实测设备：A3 `Ascend910_9382` 与A5 `Ascend950PR_958b`，均为逻辑卡1。

历史辅流版本复测（附录F）：A5业务QPS中位数2659.628、最差轮P99 27.131ms，相比历史基线未显示明显提升；逐算子与时间线证据见附录F。正文及附录A–E保留原始采集结果。

2026-10-22补充16|32、batch16、单实例固定输入profiling，结果与原始文件位置见附录G。

2026-10-22完成int32 bounds与提前planner验证，正确性、静态图保护和单实例profiling对比见附录H。

2026-10-22完成算子内部完整bounds缓存，正确性与六轮关闭profiling的同卡对照见附录P：QPS中位数2686.033→2685.188，P99中位数26.515→26.476ms，本配置未见明显性能提升。

原始采集的推荐配置下，A5/A3业务QPS比为 **1.229倍**。固定同一中位输入、单实例、使用各自物理全核预算时，无profiling的QPS比为 **1.482倍**。两种比较回答不同问题：部署吞吐同时受每实例核预算、并发和算子实现影响。

以下三点为2026-10-21原始串行版本的归因。缓存前int32提前planner版本的四实例补采和更新结论见附录O：planner约97.2%已与主流重叠，当前A5 attention kernel相对A3推荐并发快约1.76倍。

- 全核、单实例下，全部169个MatMul累计耗时约 **6.601ms → 4.740ms**，加速 **1.393倍**；MLP gate/up与down占A5设备图时间约45.1%，其加速分别只有1.304与1.489倍。
- A5 Attention计算kernel本身快 **2.540倍**；加上每层AICPU planner和metadata清零后，Attention整条设备路径只有 **1.302倍**。planner与清零占该路径48.7%。
- A3推荐5实例、每实例8|16，A5推荐4实例、每实例16|32。增加实例会拉长单个图的设备任务时间；不能把单实例或某个kernel的加速比直接当作最终QPS比。

## 环境、输入与边界

| 项 | A3 | A5 |
| --- | --- | --- |
| 机器 / 容器 | 99 / hejun-cann9-qwen25 | 850-a5 / hejun-cann910-qwen25 |
| 项目根目录 | /export/home/ext.hejun23/workspace/ascend-ge-adapters | /home/h00909914/ascend-ge-adapters |
| 物理AIC/AIV | 24/48 | 32/64 |
| 推荐限核 / batch / 实例 | 8\|16 / 16 / 5 | 16\|32 / 16 / 4 |
| CANN | 9.0.0 / B250 | 9.1.0 / B243 |
| Python | 3.11.15 | 3.12.13 |
| Torch / TorchNPU | 2.9.0+cpu / 2.9.0.post2 | 2.10.0+cpu / 2.10.0.post2 |
| Transformers | 5.10.1 | 5.10.1 |
| 主仓HEAD | 4288aa6874842ec15e3f44bdc0b22b09a5753125 | 59050e390a8bcd4d01f7beac50b741c008285f2a |
| Attention源码子模块 | b4b30cd73862d19e16242f2d1e82075ace83d770 | 同左 |
| Attention实现 | prefix_910 / FIA-GQA / prefix-merged | prefix_a5 / cannbot DSL生成的AscendC macro |
| weight-NZ pass | 启用 | 未启用 |
| MatMul实际路径 | MatMulV3 ND×NZ，task_time可见 | MatMulV3 ND×ND，op_summary可见 |

两台机器是当前已有部署栈，CANN、Torch与图优化路径不同。本报告给出当前部署中的可观测耗时归因；没有把全部差值归因于芯片微架构，也没有验证“裸硬件理论性能应为1.6倍”的规格前提。设备为共享环境，保留每次运行前后npu-smi快照；A3有较多驻留HBM，计算利用率快照为空闲，无法排除所有瞬时外部干扰。

公共输入为100组batch16，摘要`21defc288cc9848d6f1f71f8a8c6d900327963431e57e77125274b5f5fc8c85b`。两边模型适配源码SHA256相同，同seed的全局请求序列摘要也逐一相同。三个代表case按packed T排序选择：

| 输入 | case_id | packed T | 共享prefix | batch | A5逻辑Q tile填充率 |
| --- | --- | --- | --- | --- | --- |
| short | case-0084 | 1839 | 24 | 16 | 75.6% |
| median | case-0020 | 2041 | 24 | 16 | 66.4% |
| long | case-0088 | 2330 | 23 | 16 | 62.8% |

填充率定义为有效Q行数 / (128 × 所有unit的Q tile数)，仅描述固定128行分块的逻辑尾块填充，不能当作PMU算力利用率。A5将prefix作为独立unit；A3将prefix与第一条请求合并，分块与GQA布局不同。

所有实验使用隔离的YAML副本和输入副本；没有修改两边默认YAML或默认输入池。A5没有拉取或重置分支。完整环境、AIR及依赖二进制摘要保留在两端`work/a3-a5-profiling-20261021/environment.json`。

## 测量口径与预热过滤

普通性能复测关闭profiling，推荐配置各跑seed42/43/44、每轮3000次调用。QPS为业务样本/秒，P99为整batch H2D、描述符、执行和同步耗时，不含服务排队、组batch等待、编译、加载和预热。

profiler采集PipeUtilization、设备task、runtime API与时间线；每个固定输入的单实例采集先预热10次，再测30次；16核补采为100次。推荐并发采集每实例预热10次、共测100次。分析按Model ID分组，用step_trace剔除各图实例最前面的预热步骤，不混入冷启动、编译、首次shape执行或AICPU首调初始化。

设备图时间取每步FP Start至Iteration End；逐算子表为保留测量步骤内各设备task的Task Duration均值。它与无profiling的整batch延迟口径不同。各流水线PMU比率可重叠，不相加；A3 MIX kernel的AIV列为0也不能推断没有向量计算。并发下多个图可重叠，跨图task耗时之和不能当作测量墙钟时间。

特别修正采集缺项：A3的op_summary每图缺少169个NZ MatMul，但task_time含`MatMulV3_ND_NZ_ND_ND_FP16_FP16_FP16_FP16_high_performance_65536`的完整时间。分析从task_time恢复，并通过时间线connection_id关联原MatMul节点；不能仅对op_summary求和。A5另有每图24条MEMCPY_ASYNC，结合源码和每层调用位置归为metadata清零，不能漏掉。

## 无profiling的性能对照

| 场景 | A3 | A5 | A5/A3 |
| --- | --- | --- | --- |
| 推荐配置业务QPS中位数（3×3000次） | 2168.494 | 2664.147 | 1.229× |
| 推荐配置QPS范围 | 2166.909–2180.259 | 2655.138–2664.462 | — |
| 推荐配置最差轮P99 ms | 41.177 | 26.994 | 均<60ms |
| 中位输入、单实例、全核业务QPS | 1362.996 | 2019.967 | 1.482× |
| 同为16核、中位输入、单实例P50 ms | 13.080 | 11.677 | 1.120×（延迟比） |

同16核时A3普通运行有一次269.557ms长尾，P99仍为13.175ms；其QPS为1190.707，A5为1357.694。表中同时展示P50以区分稳定延迟与少量异常，完整均值、最大值和逐调用记录保留，不删除该长尾。

## 逐算子耗时：中位输入、单实例、全核预算

A3使用24|48，A5使用32|64；A5自定义Attention仍固定16个AIC。以下为30次稳态图的逐task算术均值，累计值是每个图中该类别全部调用的平均总时间。K/V与gate/up分别合并同形投影，避免把不同调度顺序误当成同一个节点。

| 类别 | A3次/图 | A5次/图 | A3单次均值 us | A5单次均值 us | A3累计 ms/图 | A5累计 ms/图 | A3耗时/A5耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Q投影 | 24 | 24 | 20.870 | 13.045 | 0.501 | 0.313 | 1.600× |
| K/V投影合计 | 48 | 48 | 9.994 | 8.006 | 0.480 | 0.384 | 1.248× |
| O投影 | 24 | 24 | 22.766 | 13.816 | 0.546 | 0.332 | 1.648× |
| MLP gate/up合计 | 48 | 48 | 66.434 | 50.965 | 3.189 | 2.446 | 1.304× |
| MLP down | 24 | 24 | 78.292 | 52.574 | 1.879 | 1.262 | 1.489× |
| 裁剪后的输出头 | 1 | 1 | 6.302 | 2.685 | 0.006 | 0.003 | 2.347× |
| Attention计算kernel | 24 | 24 | 88.640 | 34.893 | 2.127 | 0.837 | 2.540× |
| A5 Attention planner | 0 | 24 | — | 30.691 | 0.000 | 0.737 | — |
| A5 metadata清零 | 0 | 24 | — | 2.470 | 0.000 | 0.059 | — |
| AddRmsNorm | 47 | 47 | 23.394 | 8.724 | 1.100 | 0.410 | 2.682× |
| RmsNorm | 2 | 2 | 15.737 | 5.838 | 0.031 | 0.012 | 2.696× |
| RoPE | 24 | 24 | 25.516 | 8.534 | 0.612 | 0.205 | 2.990× |
| Swish | 24 | 24 | 28.705 | 19.431 | 0.689 | 0.466 | 1.477× |
| Mul | 24 | 24 | 16.809 | 15.063 | 0.403 | 0.362 | 1.116× |
| GatherV2 | 4 | 4 | 9.089 | 4.866 | 0.036 | 0.019 | 1.868× |
| Add | 1 | 1 | 9.771 | 7.132 | 0.010 | 0.007 | 1.370× |
| Sub | 1 | 1 | 2.815 | 2.552 | 0.003 | 0.003 | 1.103× |
| bounds int64→int32 | 0 | 1 | — | 2.174 | 0.000 | 0.002 | — |
| Slice | 1 | 1 | 11.413 | 3.469 | 0.011 | 0.003 | 3.291× |

| 汇总 | A3 ms | A5 ms | A3/A5 |
| --- | --- | --- | --- |
| 全部MatMul（169次） | 6.601 | 4.740 | 1.393× |
| Attention设备路径（kernel+planner+清零） | 2.127 | 1.633 | 1.302× |
| 全部已捕获设备task累计 | 11.625 | 7.862 | 1.479× |
| 设备图时间均值 | 11.741 | 8.227 | 1.427× |
| 设备图时间中位数 | 11.736 | 7.923 | 1.481× |
| 图内未被设备task覆盖的区间 | 0.117 | 0.365 | — |

A5图时间均值高于中位数，保留了采集中的长步骤；所有算子均值仍使用全部30个稳态步骤。图内未覆盖区间包含调度、提交间隔等未归属设备task的时间，不直接等同CPU计算，也不与Host API耗时重复相加。

## Prefix Attention实现差异及实测影响

| 方面 | A3 | A5 |
| --- | --- | --- |
| 计算实现 | AscendC FIA模板链，FiaBlockCubeNonQuantGqa；prefix与req0合并 | cannbot DSL生成AscendC，128×128 Q/KV macro流水线，2个AIV半块 |
| GQA / 输入 | 14 Q heads、2 KV heads、D64；TND；bounds int64 | 同一GQA数学问题；bounds转int32，多一条Cast/图 |
| prefix处理 | prefix-merged，第一请求与prefix合并 | prefix是独立unit，其他请求按prefix+private逻辑取K/V |
| 规划方式 | FIA tiling与分核；实际图内kernel有24个AIC block预算 | 每层metadata清零→AICPU planner→macro kernel |
| Attention核数 | 随GE预算；本次全核24 | GE交付件kAicCores=16，全核32的实验仍只launch16 |
| 中位输入每层计算kernel us | 88.640 | 34.893 |
| 每层planner us | 无对应设备task | 30.691 |
| 每层metadata清零 us | 无对应设备task | 2.470 |

A5 planner和metadata清零合计每图 **0.796ms**，约占A5设备图均值的9.7%。planner的aicpu明细另记录平均Task_time=15.525us、Dispatch_time=4.836us；这些是同一planner的另一统计视图，不再叠加到Task Duration。

输入长度变化下，Attention路径的影响如下：

| 输入 | A3 kernel ms | A5 kernel ms | A5 planner+清零 ms | kernel加速比 | 完整Attention路径加速比 |
| --- | --- | --- | --- | --- | --- |
| short | 1.886 | 0.605 | 0.728 | 3.118× | 1.414× |
| median | 2.127 | 0.837 | 0.796 | 2.540× | 1.302× |
| long | 2.266 | 0.934 | 0.864 | 2.427× | 1.260× |

源码定位：`third_party/ascend-ops/prefix-attention/cannbot-dsl/ge/src/kernel_launcher.h:23`固定kAicCores；`ge/src/prefix_attention_op.cc`在每次Execute中清零metadata并调用planner和kernel；`prefix_attn/metadata.py`描述分核盒；`prefix_attn/attention.py`描述q_head/q_tile循环、两遍softmax和流水线。A3入口为`ascendc/op_kernel/prefix_infer_attention_score.cpp`，语义与tiling改动详见`ascendc/docs/FIA_DELTA.md`。这些路径均位于相同的b4b30cd子模块版本中。

A3源码还包含`ascendc/op_host/op_api/prefix_infer_attention_score_custom_op.cpp`的同步D2H包装支路。但本次稳态轨迹中未观察到每层两次拷贝对应的48次/图开销；可见Memcpy约为输入H2D数量级，设备task覆盖图时间约99%。因此不把这段源码的D2H逻辑直接认定为本次瓶颈，也不把等待stream完成的Host API时长再次相加。具体运行分支或缓存机制若要优化，仍需单独确认。

A3的NZ MatMul与A5的ND MatMul也不是相同布局路径。未做关闭NZ pass的A/B，本报告不估算NZ pass的独立收益；仅确认它属于比较环境差异，169个MatMul均已计入耗时。

## 并发为何进一步压缩吞吐差距

同一中位输入、同样开启profiling，保持各自推荐限核，只改变实例数：

| 平台 | 限核 | 实例1图时间均值 ms | 推荐实例 | 并发图时间均值 ms | 单图时间膨胀 | 采集内业务QPS |
| --- | --- | --- | --- | --- | --- | --- |
| A3 | 8\|16 | 21.561 | 5 | 34.258 | 1.589× | 2320.485 |
| A5 | 16\|32 | 11.959 | 4 | 23.797 | 1.990× | 2630.320 |

A3从1到5实例时，图时间约膨胀1.59倍；A5从1到4实例时约膨胀1.99倍。并发让设备资源竞争变强，更多实例并不等于成比例增加可用算力。每实例核预算也不是独占绑核，不能把8×5或16×4解释成物理核心数。采集内QPS受profiling开销、固定输入和短测量段影响，不用于取代前面的三轮普通吞吐结果。

## 可以优先验证的优化方向

1. **A5将planner和metadata清零提到每个请求/图执行的一次公共准备阶段**：24层使用相同bounds、head配置和16核策略，当前重复24次规划。若只按已测串行task费用估计，省去23次planner约0.706ms/图，再省23次清零约0.057ms/图。该估计没有实施验证，并发、调度和缓存会影响真实收益；metadata必须按请求/图实例隔离，不能无条件跨请求缓存。
2. **A5让Attention核数成为显式可验证的参数**：当前GE预算32不会改变固定16核的Attention。应以不同核数变体做正确性和端到端A/B，不能直接假设32核更快；更宽kernel可能降低并发吞吐。
3. **继续优化占比最大的MatMul/MLP**：当前169个MatMul约快1.39倍，MLP约占A5设备图45%。可围绕ND/NZ、tiling、gate/up布局、Swish与Mul融合做独立A/B，并配合Memory/MemoryAccess组进一步验证带宽或计算限制；本轮PipeUtilization与时长证据不足以断言全部MLP都是带宽瓶颈。

以上为后续实验建议，本轮没有修改算子实现、默认限核、batch或实例数，也没有承诺优化后必达1.6倍。

## 图表

![A3与A5 profiling对比](../../images/qwen2.5-0.5b-a3-a5-profiling.png)

图中operator累计值均来自中位输入、单实例、全核采集；普通吞吐来自各自推荐配置的三轮复测。1.6虚线只表示本次问题中的参照值。

## 附录A：全部运行记录

状态均来自原始结果，prof=on的QPS与P99带采集开销。输入full为100组随机池，short/median/long为前述单case。

| 平台 | 实验 | AIC/AIV | 实例 | 输入 | 调用数 | seed | prof | 业务QPS | P50 ms | P99 ms | max ms | 状态 | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A3 | plain-opt-s42 | 8/16 | 5 | full | 3000 | 42 | off | 2166.909 | 36.620 | 41.036 | 42.707 | success | 20261021T075753-4d7237f8 |
| A3 | profile-median-opt-i1 | 8/16 | 1 | median | 30 | 42 | on | 736.258 | 21.711 | 21.911 | 21.911 | success | 20261021T075943-78bb8710 |
| A3 | plain-opt-s43 | 8/16 | 5 | full | 3000 | 43 | off | 2168.494 | 36.557 | 41.177 | 42.495 | success | 20261021T080119-b4a785b8 |
| A3 | plain-opt-s44 | 8/16 | 5 | full | 3000 | 44 | off | 2180.259 | 36.404 | 40.797 | 41.763 | success | 20261021T080316-ecbe54b0 |
| A3 | plain-median-full-i1 | 24/48 | 1 | median | 1000 | 42 | off | 1362.996 | 11.733 | 11.836 | 13.142 | success | 20261021T080502-4977038c |
| A3 | profile-short-full-i1 | 24/48 | 1 | short | 30 | 42 | on | 1430.730 | 11.141 | 11.372 | 11.372 | success | 20261021T080558-47a23a44 |
| A3 | profile-median-full-i1 | 24/48 | 1 | median | 30 | 42 | on | 1345.540 | 11.875 | 11.933 | 11.933 | success | 20261021T080643-6ebba159 |
| A3 | profile-long-full-i1 | 24/48 | 1 | long | 30 | 42 | on | 1235.731 | 12.920 | 13.327 | 13.327 | success | 20261021T080727-129a2e46 |
| A3 | profile-median-opt-concurrent | 8/16 | 5 | median | 100 | 42 | on | 2320.485 | 34.401 | 35.538 | 35.538 | success | 20261021T080811-d0f8ffdf |
| A3 | plain-median-c16-i1 | 16/32 | 1 | median | 1000 | 42 | off | 1190.707 | 13.080 | 13.175 | 269.557 | success | 20261021T081151-85aab800 |
| A3 | profile-median-c16-i1 | 16/32 | 1 | median | 30 | 42 | on | 949.375 | 13.318 | 106.645 | 106.645 | success | 20261021T081248-0c3a7cd9 |
| A3 | profile-median-c16-i1-repeat | 16/32 | 1 | median | 100 | 43 | on | 1201.294 | 13.305 | 13.916 | 13.916 | success | 20261021T081848-0ab0df92 |
| A5 | plain-opt-s42 | 16/32 | 4 | full | 3000 | 42 | off | 2664.462 | 23.820 | 26.994 | 52.730 | success | 20261021T075804-769caead |
| A5 | profile-median-opt-i1 | 16/32 | 1 | median | 30 | 42 | on | 1314.104 | 11.795 | 22.445 | 22.445 | success | 20261021T075930-6adcbbf6 |
| A5 | plain-opt-s43 | 16/32 | 4 | full | 3000 | 43 | off | 2655.138 | 23.901 | 26.977 | 73.552 | success | 20261021T080052-99d8fc7e |
| A5 | plain-opt-s44 | 16/32 | 4 | full | 3000 | 44 | off | 2664.147 | 23.863 | 26.818 | 56.006 | success | 20261021T080225-a6db2f71 |
| A5 | plain-median-full-i1 | 32/64 | 1 | median | 1000 | 42 | off | 2019.967 | 7.897 | 8.015 | 19.266 | success | 20261021T080348-e851513e |
| A5 | profile-short-full-i1 | 32/64 | 1 | short | 30 | 42 | on | 2056.256 | 7.411 | 17.935 | 17.935 | success | 20261021T080438-4c7c5b4f |
| A5 | profile-median-full-i1 | 32/64 | 1 | median | 30 | 42 | on | 1893.028 | 8.147 | 17.287 | 17.287 | success | 20261021T080520-150f3348 |
| A5 | profile-long-full-i1 | 32/64 | 1 | long | 30 | 42 | on | 1650.046 | 9.386 | 17.936 | 17.936 | success | 20261021T080603-cab1485d |
| A5 | profile-median-opt-concurrent | 16/32 | 4 | median | 100 | 42 | on | 2630.320 | 23.463 | 53.273 | 53.273 | success | 20261021T080648-6703323f |
| A5 | plain-median-c16-i1 | 16/32 | 1 | median | 1000 | 42 | off | 1357.694 | 11.677 | 12.237 | 22.320 | success | 20261021T081147-795fe349 |

## 附录B：profiling窗口与异常说明

| 平台 | 实验 | 保留图数 | 剔除预热 | 图时间均值 ms | 中位数 ms | 最大值 ms | 设备task累计 ms/图 | 未覆盖区间 ms/图 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A3 | profile-median-opt-i1 | 30 | 10 | 21.561 | 21.553 | 21.698 | 21.435 | 0.126 |
| A3 | profile-short-full-i1 | 30 | 10 | 11.022 | 11.010 | 11.157 | 10.894 | 0.128 |
| A3 | profile-median-full-i1 | 30 | 10 | 11.741 | 11.736 | 11.806 | 11.625 | 0.117 |
| A3 | profile-long-full-i1 | 30 | 10 | 12.786 | 12.774 | 12.934 | 12.666 | 0.120 |
| A3 | profile-median-opt-concurrent | 100 | 50 | 34.258 | 34.241 | 35.407 | 34.134 | 0.124 |
| A3 | profile-median-c16-i1 | 30 | 10 | 16.263 | 13.157 | 106.441 | 12.998 | 3.264 |
| A3 | profile-median-c16-i1-repeat | 100 | 10 | 13.148 | 13.135 | 13.308 | 12.988 | 0.160 |
| A5 | profile-median-opt-i1 | 30 | 10 | 11.959 | 11.597 | 22.223 | 11.553 | 0.405 |
| A5 | profile-short-full-i1 | 30 | 10 | 7.556 | 7.201 | 17.659 | 7.138 | 0.418 |
| A5 | profile-median-full-i1 | 30 | 10 | 8.227 | 7.923 | 17.062 | 7.862 | 0.365 |
| A5 | profile-long-full-i1 | 30 | 10 | 9.480 | 9.185 | 17.692 | 9.126 | 0.354 |
| A5 | profile-median-opt-concurrent | 100 | 40 | 23.797 | 23.216 | 52.785 | 22.888 | 0.909 |

A3首次16核profiling有一次约106ms的整batch长步骤，设备图未覆盖区间均值被拉高；原记录保留，补采100次用于核对稳定状态。补采设备图均值13.148ms、中位数13.135ms。A5全核采集中也保留长步骤，因而均值与中位数存在差别。没有把异常计为0，也没有从均值中删除测量段样本。

## 附录C：每层主要算子累计耗时

单位us，为30个稳态图的均值。K/V及gate/up为每层两次调用合计，其他列为单次；所有24层均展示。

### A3：中位输入、单实例、全核

| 层（从0开始） | Q | K/V | O | gate/up | down | Attention kernel | planner | metadata清零 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 21.093 | 20.007 | 22.858 | 132.559 | 78.203 | 87.460 | 0.000 | 0.000 |
| 1 | 21.061 | 20.057 | 22.769 | 132.132 | 78.014 | 89.428 | 0.000 | 0.000 |
| 2 | 20.867 | 19.905 | 23.086 | 134.821 | 78.385 | 88.595 | 0.000 | 0.000 |
| 3 | 21.026 | 19.808 | 22.627 | 138.347 | 78.430 | 88.867 | 0.000 | 0.000 |
| 4 | 20.553 | 19.955 | 23.001 | 138.761 | 78.156 | 89.500 | 0.000 | 0.000 |
| 5 | 20.877 | 20.095 | 22.735 | 134.780 | 78.291 | 88.703 | 0.000 | 0.000 |
| 6 | 20.870 | 19.993 | 22.782 | 131.954 | 78.132 | 89.130 | 0.000 | 0.000 |
| 7 | 20.909 | 19.917 | 22.640 | 131.763 | 78.164 | 88.125 | 0.000 | 0.000 |
| 8 | 20.893 | 20.044 | 22.867 | 131.428 | 78.293 | 87.867 | 0.000 | 0.000 |
| 9 | 20.807 | 19.928 | 22.763 | 131.929 | 78.305 | 88.283 | 0.000 | 0.000 |
| 10 | 20.954 | 19.956 | 23.258 | 131.893 | 78.264 | 89.092 | 0.000 | 0.000 |
| 11 | 20.914 | 19.771 | 22.697 | 131.599 | 78.316 | 88.745 | 0.000 | 0.000 |
| 12 | 20.809 | 19.962 | 22.605 | 132.006 | 78.130 | 89.102 | 0.000 | 0.000 |
| 13 | 20.617 | 20.241 | 22.764 | 131.736 | 78.268 | 88.387 | 0.000 | 0.000 |
| 14 | 20.827 | 20.189 | 22.235 | 131.669 | 78.420 | 88.873 | 0.000 | 0.000 |
| 15 | 20.955 | 19.898 | 23.358 | 131.987 | 78.259 | 88.537 | 0.000 | 0.000 |
| 16 | 20.801 | 19.839 | 22.841 | 132.872 | 78.288 | 88.378 | 0.000 | 0.000 |
| 17 | 20.813 | 19.985 | 22.625 | 132.245 | 78.366 | 88.407 | 0.000 | 0.000 |
| 18 | 20.881 | 19.995 | 22.351 | 131.599 | 78.342 | 89.220 | 0.000 | 0.000 |
| 19 | 20.833 | 20.011 | 22.778 | 132.578 | 78.554 | 88.185 | 0.000 | 0.000 |
| 20 | 20.912 | 19.941 | 23.025 | 132.329 | 78.366 | 89.408 | 0.000 | 0.000 |
| 21 | 20.973 | 20.063 | 22.745 | 133.697 | 78.285 | 88.171 | 0.000 | 0.000 |
| 22 | 20.745 | 20.009 | 22.421 | 132.505 | 78.274 | 88.406 | 0.000 | 0.000 |
| 23 | 20.895 | 20.162 | 22.543 | 131.661 | 78.495 | 88.481 | 0.000 | 0.000 |

### A5：中位输入、单实例、全核

| 层（从0开始） | Q | K/V | O | gate/up | down | Attention kernel | planner | metadata清零 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 13.027 | 15.986 | 13.765 | 101.892 | 52.602 | 35.064 | 35.299 | 2.518 |
| 1 | 12.997 | 15.823 | 13.878 | 102.199 | 52.536 | 34.681 | 31.073 | 2.595 |
| 2 | 12.985 | 16.106 | 13.795 | 101.550 | 52.567 | 34.931 | 30.527 | 2.511 |
| 3 | 12.971 | 15.929 | 13.802 | 101.618 | 52.557 | 35.179 | 30.598 | 2.468 |
| 4 | 13.031 | 15.824 | 13.785 | 101.982 | 52.580 | 34.621 | 30.508 | 2.500 |
| 5 | 13.091 | 16.074 | 13.818 | 102.050 | 52.672 | 35.131 | 30.800 | 2.482 |
| 6 | 13.033 | 15.997 | 13.816 | 102.291 | 52.555 | 34.252 | 30.281 | 2.508 |
| 7 | 13.065 | 16.014 | 13.814 | 101.720 | 52.617 | 35.595 | 30.498 | 2.470 |
| 8 | 13.099 | 16.029 | 13.828 | 101.897 | 52.543 | 34.576 | 30.687 | 2.462 |
| 9 | 13.050 | 15.959 | 13.769 | 101.740 | 52.544 | 34.805 | 30.536 | 2.621 |
| 10 | 13.071 | 15.977 | 13.849 | 102.159 | 52.605 | 34.987 | 30.526 | 2.495 |
| 11 | 13.055 | 16.106 | 13.854 | 102.068 | 52.574 | 34.683 | 30.536 | 2.498 |
| 12 | 13.080 | 16.017 | 13.828 | 101.744 | 52.537 | 35.124 | 30.291 | 2.468 |
| 13 | 13.067 | 16.089 | 13.805 | 102.020 | 52.574 | 34.752 | 30.370 | 2.504 |
| 14 | 13.067 | 16.038 | 13.963 | 101.920 | 52.618 | 34.927 | 30.428 | 2.458 |
| 15 | 13.025 | 16.040 | 13.777 | 102.246 | 52.548 | 34.699 | 30.506 | 2.433 |
| 16 | 13.015 | 15.902 | 13.784 | 101.797 | 52.566 | 35.146 | 30.478 | 2.409 |
| 17 | 13.090 | 16.181 | 13.797 | 102.001 | 52.514 | 35.215 | 30.572 | 2.419 |
| 18 | 13.014 | 15.926 | 13.801 | 101.709 | 52.558 | 34.589 | 30.267 | 2.401 |
| 19 | 13.042 | 16.078 | 13.783 | 101.927 | 52.570 | 35.172 | 30.310 | 2.391 |
| 20 | 13.066 | 16.007 | 13.786 | 102.203 | 52.530 | 34.410 | 30.414 | 2.389 |
| 21 | 13.059 | 16.102 | 13.832 | 101.700 | 52.692 | 35.329 | 30.295 | 2.411 |
| 22 | 13.039 | 16.040 | 13.849 | 102.050 | 52.570 | 34.459 | 30.229 | 2.404 |
| 23 | 13.050 | 16.032 | 13.805 | 101.854 | 52.546 | 35.115 | 30.566 | 2.473 |

## 附录D：完整逐节点时长

以下覆盖全部设备task：A3每图321条，A5每图370条（含24条metadata清零）。每行是在同一采集内按类别/顺序识别的节点，均有30次保留测量。两个平台分别列出，避免对同形K/V、gate/up或融合前后的节点强行逐一配对。跨平台比较采用主表和附录C的语义分组。

### A3逐节点

| 类别 | 组内序号 | 层 | 均值 us | 中位数 us | 次数 | 原始节点或kernel |
| --- | --- | --- | --- | --- | --- | --- |
| GatherV2 | 1 | — | 10.689 | 10.680 | 30 | GatherV2 |
| RmsNorm | 1 | — | 23.210 | 22.120 | 30 | RmsNorm |
| K/V投影合计 | 1 | 0 | 10.075 | 9.980 | 30 | MatMulV2_1_v3 |
| K/V投影合计 | 2 | 0 | 9.932 | 9.900 | 30 | MatMulV2_2_v3 |
| Q投影 | 1 | 0 | 21.093 | 21.020 | 30 | MatMulV2_v3 |
| GatherV2 | 2 | — | 14.293 | 14.300 | 30 | GatherV2_1 |
| GatherV2 | 3 | — | 6.443 | 6.400 | 30 | GatherV2_2 |
| RoPE | 1 | — | 24.858 | 24.840 | 30 | ApplyRotaryPosEmb |
| Attention计算kernel | 1 | 0 | 87.460 | 87.791 | 30 | PrefixInferAttentionScore |
| O投影 | 1 | 0 | 22.858 | 23.140 | 30 | MatMul/v2_v3 |
| AddRmsNorm | 1 | — | 25.858 | 25.820 | 30 | RmsNorm_1/AddRmsNorm |
| MLP gate/up合计 | 1 | 0 | 66.612 | 66.741 | 30 | MatMul_1/v2_v3 |
| MLP gate/up合计 | 2 | 0 | 65.947 | 65.981 | 30 | MatMul_2/v2_v3 |
| Swish | 1 | — | 28.819 | 28.750 | 30 | Swish |
| Mul | 1 | — | 16.728 | 16.730 | 30 | Mul |
| MLP down | 1 | 0 | 78.203 | 78.141 | 30 | MatMul_3/v2_v3 |
| AddRmsNorm | 2 | — | 20.487 | 20.340 | 30 | RmsNorm_2/AddRmsNorm |
| Q投影 | 2 | 1 | 21.061 | 21.020 | 30 | MatMulV2_3_v3 |
| K/V投影合计 | 3 | 1 | 10.056 | 10.040 | 30 | MatMulV2_4_v3 |
| K/V投影合计 | 4 | 1 | 10.001 | 9.960 | 30 | MatMulV2_5_v3 |
| RoPE | 2 | — | 26.175 | 25.730 | 30 | ApplyRotaryPosEmb_1 |
| Attention计算kernel | 2 | 1 | 89.428 | 88.781 | 30 | PrefixInferAttentionScore_1 |
| O投影 | 2 | 1 | 22.769 | 22.770 | 30 | MatMul_4/v2_v3 |
| AddRmsNorm | 3 | — | 26.951 | 26.860 | 30 | RmsNorm_3/AddRmsNorm |
| MLP gate/up合计 | 3 | 1 | 66.422 | 66.401 | 30 | MatMul_5/v2_v3 |
| MLP gate/up合计 | 4 | 1 | 65.710 | 65.801 | 30 | MatMul_6/v2_v3 |
| Swish | 2 | — | 28.688 | 28.630 | 30 | Swish_1 |
| Mul | 2 | — | 16.823 | 16.760 | 30 | Mul_1 |
| MLP down | 2 | 1 | 78.014 | 77.921 | 30 | MatMul_7/v2_v3 |
| AddRmsNorm | 4 | — | 20.593 | 20.050 | 30 | RmsNorm_4/AddRmsNorm |
| Q投影 | 3 | 2 | 20.867 | 20.920 | 30 | MatMulV2_6_v3 |
| K/V投影合计 | 5 | 2 | 9.939 | 9.880 | 30 | MatMulV2_7_v3 |
| K/V投影合计 | 6 | 2 | 9.966 | 9.920 | 30 | MatMulV2_8_v3 |
| RoPE | 3 | — | 25.909 | 25.790 | 30 | ApplyRotaryPosEmb_2 |
| Attention计算kernel | 3 | 2 | 88.595 | 87.651 | 30 | PrefixInferAttentionScore_2 |
| O投影 | 3 | 2 | 23.086 | 22.660 | 30 | MatMul_8/v2_v3 |
| AddRmsNorm | 5 | — | 26.226 | 26.220 | 30 | RmsNorm_5/AddRmsNorm |
| MLP gate/up合计 | 5 | 2 | 67.225 | 67.191 | 30 | MatMul_10/v2_v3 |
| MLP gate/up合计 | 6 | 2 | 67.596 | 67.521 | 30 | MatMul_9/v2_v3 |
| Swish | 3 | — | 28.771 | 28.710 | 30 | Swish_2 |
| Mul | 3 | — | 16.798 | 16.700 | 30 | Mul_2 |
| MLP down | 3 | 2 | 78.385 | 78.381 | 30 | MatMul_11/v2_v3 |
| AddRmsNorm | 6 | — | 20.255 | 19.910 | 30 | RmsNorm_6/AddRmsNorm |
| K/V投影合计 | 7 | 3 | 9.940 | 9.950 | 30 | MatMulV2_10_v3 |
| K/V投影合计 | 8 | 3 | 9.868 | 9.830 | 30 | MatMulV2_11_v3 |
| Q投影 | 4 | 3 | 21.026 | 21.100 | 30 | MatMulV2_9_v3 |
| RoPE | 4 | — | 24.034 | 23.800 | 30 | ApplyRotaryPosEmb_3 |
| Attention计算kernel | 4 | 3 | 88.867 | 89.022 | 30 | PrefixInferAttentionScore_3 |
| O投影 | 4 | 3 | 22.627 | 22.610 | 30 | MatMul_12/v2_v3 |
| AddRmsNorm | 7 | — | 26.088 | 25.940 | 30 | RmsNorm_7/AddRmsNorm |
| MLP gate/up合计 | 7 | 3 | 70.644 | 70.431 | 30 | MatMul_13/v2_v3 |
| MLP gate/up合计 | 8 | 3 | 67.703 | 67.691 | 30 | MatMul_14/v2_v3 |
| Swish | 4 | — | 28.657 | 28.630 | 30 | Swish_3 |
| Mul | 4 | — | 16.775 | 16.650 | 30 | Mul_3 |
| MLP down | 4 | 3 | 78.430 | 78.501 | 30 | MatMul_15/v2_v3 |
| AddRmsNorm | 8 | — | 20.590 | 20.080 | 30 | RmsNorm_8/AddRmsNorm |
| Q投影 | 5 | 4 | 20.553 | 20.520 | 30 | MatMulV2_12_v3 |
| K/V投影合计 | 9 | 4 | 10.000 | 9.960 | 30 | MatMulV2_13_v3 |
| K/V投影合计 | 10 | 4 | 9.955 | 9.830 | 30 | MatMulV2_14_v3 |
| RoPE | 5 | — | 25.781 | 25.750 | 30 | ApplyRotaryPosEmb_4 |
| Attention计算kernel | 5 | 4 | 89.500 | 89.351 | 30 | PrefixInferAttentionScore_4 |
| O投影 | 5 | 4 | 23.001 | 23.080 | 30 | MatMul_16/v2_v3 |
| AddRmsNorm | 9 | — | 26.215 | 26.170 | 30 | RmsNorm_9/AddRmsNorm |
| MLP gate/up合计 | 9 | 4 | 70.330 | 70.331 | 30 | MatMul_17/v2_v3 |
| MLP gate/up合计 | 10 | 4 | 68.432 | 68.441 | 30 | MatMul_18/v2_v3 |
| Swish | 5 | — | 28.751 | 28.710 | 30 | Swish_4 |
| Mul | 5 | — | 16.823 | 16.700 | 30 | Mul_4 |
| MLP down | 5 | 4 | 78.156 | 78.141 | 30 | MatMul_19/v2_v3 |
| AddRmsNorm | 10 | — | 20.225 | 19.680 | 30 | RmsNorm_10/AddRmsNorm |
| Q投影 | 6 | 5 | 20.877 | 20.950 | 30 | MatMulV2_15_v3 |
| K/V投影合计 | 11 | 5 | 10.060 | 10.080 | 30 | MatMulV2_16_v3 |
| K/V投影合计 | 12 | 5 | 10.035 | 10.030 | 30 | MatMulV2_17_v3 |
| RoPE | 6 | — | 25.983 | 25.860 | 30 | ApplyRotaryPosEmb_5 |
| Attention计算kernel | 6 | 5 | 88.703 | 89.091 | 30 | PrefixInferAttentionScore_5 |
| O投影 | 6 | 5 | 22.735 | 22.570 | 30 | MatMul_20/v2_v3 |
| AddRmsNorm | 11 | — | 26.291 | 26.260 | 30 | RmsNorm_11/AddRmsNorm |
| MLP gate/up合计 | 11 | 5 | 68.306 | 68.451 | 30 | MatMul_21/v2_v3 |
| MLP gate/up合计 | 12 | 5 | 66.474 | 66.421 | 30 | MatMul_22/v2_v3 |
| Swish | 6 | — | 28.627 | 28.610 | 30 | Swish_5 |
| Mul | 6 | — | 16.768 | 16.700 | 30 | Mul_5 |
| MLP down | 6 | 5 | 78.291 | 78.271 | 30 | MatMul_23/v2_v3 |
| AddRmsNorm | 12 | — | 20.178 | 20.060 | 30 | RmsNorm_12/AddRmsNorm |
| Q投影 | 7 | 6 | 20.870 | 20.910 | 30 | MatMulV2_18_v3 |
| K/V投影合计 | 13 | 6 | 9.973 | 9.920 | 30 | MatMulV2_19_v3 |
| K/V投影合计 | 14 | 6 | 10.021 | 9.880 | 30 | MatMulV2_20_v3 |
| RoPE | 7 | — | 25.879 | 25.750 | 30 | ApplyRotaryPosEmb_6 |
| Attention计算kernel | 7 | 6 | 89.130 | 89.121 | 30 | PrefixInferAttentionScore_6 |
| O投影 | 7 | 6 | 22.782 | 22.590 | 30 | MatMul_24/v2_v3 |
| AddRmsNorm | 13 | — | 26.701 | 26.690 | 30 | RmsNorm_13/AddRmsNorm |
| MLP gate/up合计 | 13 | 6 | 65.834 | 65.861 | 30 | MatMul_25/v2_v3 |
| MLP gate/up合计 | 14 | 6 | 66.120 | 66.301 | 30 | MatMul_26/v2_v3 |
| Swish | 7 | — | 28.794 | 28.710 | 30 | Swish_6 |
| Mul | 7 | — | 16.875 | 16.670 | 30 | Mul_6 |
| MLP down | 7 | 6 | 78.132 | 78.161 | 30 | MatMul_27/v2_v3 |
| AddRmsNorm | 14 | — | 20.003 | 19.870 | 30 | RmsNorm_14/AddRmsNorm |
| Q投影 | 8 | 7 | 20.909 | 20.940 | 30 | MatMulV2_21_v3 |
| K/V投影合计 | 15 | 7 | 9.980 | 9.950 | 30 | MatMulV2_22_v3 |
| K/V投影合计 | 16 | 7 | 9.937 | 9.930 | 30 | MatMulV2_23_v3 |
| RoPE | 8 | — | 25.772 | 25.790 | 30 | ApplyRotaryPosEmb_7 |
| Attention计算kernel | 8 | 7 | 88.125 | 87.421 | 30 | PrefixInferAttentionScore_7 |
| O投影 | 8 | 7 | 22.640 | 22.610 | 30 | MatMul_28/v2_v3 |
| AddRmsNorm | 15 | — | 26.403 | 26.380 | 30 | RmsNorm_15/AddRmsNorm |
| MLP gate/up合计 | 15 | 7 | 66.002 | 66.021 | 30 | MatMul_29/v2_v3 |
| MLP gate/up合计 | 16 | 7 | 65.762 | 65.771 | 30 | MatMul_30/v2_v3 |
| Swish | 8 | — | 28.654 | 28.630 | 30 | Swish_7 |
| Mul | 8 | — | 16.813 | 16.750 | 30 | Mul_7 |
| MLP down | 8 | 7 | 78.164 | 78.151 | 30 | MatMul_31/v2_v3 |
| AddRmsNorm | 16 | — | 20.099 | 19.860 | 30 | RmsNorm_16/AddRmsNorm |
| Q投影 | 9 | 8 | 20.893 | 20.940 | 30 | MatMulV2_24_v3 |
| K/V投影合计 | 17 | 8 | 10.085 | 10.040 | 30 | MatMulV2_25_v3 |
| K/V投影合计 | 18 | 8 | 9.959 | 10.020 | 30 | MatMulV2_26_v3 |
| RoPE | 9 | — | 25.961 | 25.870 | 30 | ApplyRotaryPosEmb_8 |
| Attention计算kernel | 9 | 8 | 87.867 | 87.551 | 30 | PrefixInferAttentionScore_8 |
| O投影 | 9 | 8 | 22.867 | 22.330 | 30 | MatMul_32/v2_v3 |
| AddRmsNorm | 17 | — | 26.307 | 26.280 | 30 | RmsNorm_17/AddRmsNorm |
| MLP gate/up合计 | 17 | 8 | 65.892 | 65.981 | 30 | MatMul_33/v2_v3 |
| MLP gate/up合计 | 18 | 8 | 65.536 | 65.471 | 30 | MatMul_34/v2_v3 |
| Swish | 9 | — | 28.659 | 28.630 | 30 | Swish_8 |
| Mul | 9 | — | 16.696 | 16.700 | 30 | Mul_8 |
| MLP down | 9 | 8 | 78.293 | 78.381 | 30 | MatMul_35/v2_v3 |
| AddRmsNorm | 18 | — | 19.964 | 20.040 | 30 | RmsNorm_18/AddRmsNorm |
| Q投影 | 10 | 9 | 20.807 | 20.910 | 30 | MatMulV2_27_v3 |
| K/V投影合计 | 19 | 9 | 9.923 | 9.870 | 30 | MatMulV2_28_v3 |
| K/V投影合计 | 20 | 9 | 10.005 | 9.920 | 30 | MatMulV2_29_v3 |
| RoPE | 10 | — | 24.334 | 24.090 | 30 | ApplyRotaryPosEmb_9 |
| Attention计算kernel | 10 | 9 | 88.283 | 87.921 | 30 | PrefixInferAttentionScore_9 |
| O投影 | 10 | 9 | 22.763 | 22.850 | 30 | MatMul_36/v2_v3 |
| AddRmsNorm | 19 | — | 26.007 | 25.890 | 30 | RmsNorm_19/AddRmsNorm |
| MLP gate/up合计 | 19 | 9 | 66.408 | 66.491 | 30 | MatMul_37/v2_v3 |
| MLP gate/up合计 | 20 | 9 | 65.522 | 65.681 | 30 | MatMul_38/v2_v3 |
| Swish | 10 | — | 28.658 | 28.640 | 30 | Swish_9 |
| Mul | 10 | — | 16.751 | 16.700 | 30 | Mul_9 |
| MLP down | 10 | 9 | 78.305 | 78.331 | 30 | MatMul_39/v2_v3 |
| AddRmsNorm | 20 | — | 20.449 | 20.100 | 30 | RmsNorm_20/AddRmsNorm |
| Q投影 | 11 | 10 | 20.954 | 20.930 | 30 | MatMulV2_30_v3 |
| K/V投影合计 | 21 | 10 | 10.041 | 10.040 | 30 | MatMulV2_31_v3 |
| K/V投影合计 | 22 | 10 | 9.915 | 9.960 | 30 | MatMulV2_32_v3 |
| RoPE | 11 | — | 26.175 | 25.930 | 30 | ApplyRotaryPosEmb_10 |
| Attention计算kernel | 11 | 10 | 89.092 | 88.892 | 30 | PrefixInferAttentionScore_10 |
| O投影 | 11 | 10 | 23.258 | 22.940 | 30 | MatMul_40/v2_v3 |
| AddRmsNorm | 21 | — | 26.325 | 26.260 | 30 | RmsNorm_21/AddRmsNorm |
| MLP gate/up合计 | 21 | 10 | 65.830 | 65.861 | 30 | MatMul_41/v2_v3 |
| MLP gate/up合计 | 22 | 10 | 66.062 | 66.021 | 30 | MatMul_42/v2_v3 |
| Swish | 11 | — | 28.745 | 28.730 | 30 | Swish_10 |
| Mul | 11 | — | 16.693 | 16.720 | 30 | Mul_10 |
| MLP down | 11 | 10 | 78.264 | 78.161 | 30 | MatMul_43/v2_v3 |
| AddRmsNorm | 22 | — | 20.305 | 19.960 | 30 | RmsNorm_22/AddRmsNorm |
| Q投影 | 12 | 11 | 20.914 | 20.970 | 30 | MatMulV2_33_v3 |
| K/V投影合计 | 23 | 11 | 9.925 | 9.800 | 30 | MatMulV2_34_v3 |
| K/V投影合计 | 24 | 11 | 9.846 | 9.800 | 30 | MatMulV2_35_v3 |
| RoPE | 12 | — | 25.928 | 25.740 | 30 | ApplyRotaryPosEmb_11 |
| Attention计算kernel | 12 | 11 | 88.745 | 88.671 | 30 | PrefixInferAttentionScore_11 |
| O投影 | 12 | 11 | 22.697 | 22.540 | 30 | MatMul_44/v2_v3 |
| AddRmsNorm | 23 | — | 26.275 | 26.240 | 30 | RmsNorm_23/AddRmsNorm |
| MLP gate/up合计 | 23 | 11 | 65.696 | 65.661 | 30 | MatMul_45/v2_v3 |
| MLP gate/up合计 | 24 | 11 | 65.904 | 65.891 | 30 | MatMul_46/v2_v3 |
| Swish | 12 | — | 28.737 | 28.660 | 30 | Swish_11 |
| Mul | 12 | — | 16.741 | 16.720 | 30 | Mul_11 |
| MLP down | 12 | 11 | 78.316 | 78.311 | 30 | MatMul_47/v2_v3 |
| AddRmsNorm | 24 | — | 20.552 | 20.150 | 30 | RmsNorm_24/AddRmsNorm |
| Q投影 | 13 | 12 | 20.809 | 20.840 | 30 | MatMulV2_36_v3 |
| K/V投影合计 | 25 | 12 | 10.041 | 10.080 | 30 | MatMulV2_37_v3 |
| K/V投影合计 | 26 | 12 | 9.921 | 9.910 | 30 | MatMulV2_38_v3 |
| RoPE | 13 | — | 25.913 | 25.900 | 30 | ApplyRotaryPosEmb_12 |
| Attention计算kernel | 13 | 12 | 89.102 | 88.891 | 30 | PrefixInferAttentionScore_12 |
| O投影 | 13 | 12 | 22.605 | 22.640 | 30 | MatMul_48/v2_v3 |
| AddRmsNorm | 25 | — | 26.823 | 26.770 | 30 | RmsNorm_25/AddRmsNorm |
| MLP gate/up合计 | 25 | 12 | 66.270 | 66.421 | 30 | MatMul_49/v2_v3 |
| MLP gate/up合计 | 26 | 12 | 65.736 | 65.791 | 30 | MatMul_50/v2_v3 |
| Swish | 13 | — | 28.737 | 28.660 | 30 | Swish_12 |
| Mul | 13 | — | 16.904 | 16.690 | 30 | Mul_12 |
| MLP down | 13 | 12 | 78.130 | 78.081 | 30 | MatMul_51/v2_v3 |
| AddRmsNorm | 26 | — | 20.511 | 19.900 | 30 | RmsNorm_26/AddRmsNorm |
| Q投影 | 14 | 13 | 20.617 | 20.600 | 30 | MatMulV2_39_v3 |
| K/V投影合计 | 27 | 13 | 10.133 | 10.200 | 30 | MatMulV2_40_v3 |
| K/V投影合计 | 28 | 13 | 10.109 | 10.160 | 30 | MatMulV2_41_v3 |
| RoPE | 14 | — | 25.721 | 25.700 | 30 | ApplyRotaryPosEmb_13 |
| Attention计算kernel | 14 | 13 | 88.387 | 88.931 | 30 | PrefixInferAttentionScore_13 |
| O投影 | 14 | 13 | 22.764 | 22.900 | 30 | MatMul_52/v2_v3 |
| AddRmsNorm | 27 | — | 26.211 | 26.190 | 30 | RmsNorm_27/AddRmsNorm |
| MLP gate/up合计 | 27 | 13 | 65.707 | 65.701 | 30 | MatMul_53/v2_v3 |
| MLP gate/up合计 | 28 | 13 | 66.029 | 66.031 | 30 | MatMul_54/v2_v3 |
| Swish | 14 | — | 28.635 | 28.600 | 30 | Swish_13 |
| Mul | 14 | — | 16.801 | 16.700 | 30 | Mul_13 |
| MLP down | 14 | 13 | 78.268 | 78.341 | 30 | MatMul_55/v2_v3 |
| AddRmsNorm | 28 | — | 19.828 | 19.820 | 30 | RmsNorm_28/AddRmsNorm |
| Q投影 | 15 | 14 | 20.827 | 20.910 | 30 | MatMulV2_42_v3 |
| K/V投影合计 | 29 | 14 | 10.027 | 10.160 | 30 | MatMulV2_43_v3 |
| K/V投影合计 | 30 | 14 | 10.163 | 10.190 | 30 | MatMulV2_44_v3 |
| RoPE | 15 | — | 23.983 | 24.080 | 30 | ApplyRotaryPosEmb_14 |
| Attention计算kernel | 15 | 14 | 88.873 | 89.301 | 30 | PrefixInferAttentionScore_14 |
| O投影 | 15 | 14 | 22.235 | 22.220 | 30 | MatMul_56/v2_v3 |
| AddRmsNorm | 29 | — | 26.354 | 26.200 | 30 | RmsNorm_29/AddRmsNorm |
| MLP gate/up合计 | 29 | 14 | 66.032 | 65.991 | 30 | MatMul_57/v2_v3 |
| MLP gate/up合计 | 30 | 14 | 65.637 | 65.621 | 30 | MatMul_58/v2_v3 |
| Swish | 15 | — | 28.715 | 28.650 | 30 | Swish_14 |
| Mul | 15 | — | 16.818 | 16.730 | 30 | Mul_14 |
| MLP down | 15 | 14 | 78.420 | 78.381 | 30 | MatMul_59/v2_v3 |
| AddRmsNorm | 30 | — | 20.349 | 20.070 | 30 | RmsNorm_30/AddRmsNorm |
| Q投影 | 16 | 15 | 20.955 | 20.970 | 30 | MatMulV2_45_v3 |
| K/V投影合计 | 31 | 15 | 9.959 | 9.960 | 30 | MatMulV2_46_v3 |
| K/V投影合计 | 32 | 15 | 9.939 | 9.980 | 30 | MatMulV2_47_v3 |
| RoPE | 16 | — | 25.971 | 25.770 | 30 | ApplyRotaryPosEmb_15 |
| Attention计算kernel | 16 | 15 | 88.537 | 88.001 | 30 | PrefixInferAttentionScore_15 |
| O投影 | 16 | 15 | 23.358 | 22.890 | 30 | MatMul_60/v2_v3 |
| AddRmsNorm | 31 | — | 26.239 | 26.180 | 30 | RmsNorm_31/AddRmsNorm |
| MLP gate/up合计 | 31 | 15 | 65.800 | 65.861 | 30 | MatMul_61/v2_v3 |
| MLP gate/up合计 | 32 | 15 | 66.187 | 66.251 | 30 | MatMul_62/v2_v3 |
| Swish | 16 | — | 28.792 | 28.720 | 30 | Swish_15 |
| Mul | 16 | — | 16.775 | 16.710 | 30 | Mul_15 |
| MLP down | 16 | 15 | 78.259 | 78.291 | 30 | MatMul_63/v2_v3 |
| AddRmsNorm | 32 | — | 19.853 | 19.800 | 30 | RmsNorm_32/AddRmsNorm |
| Q投影 | 17 | 16 | 20.801 | 20.830 | 30 | MatMulV2_48_v3 |
| K/V投影合计 | 33 | 16 | 9.907 | 9.870 | 30 | MatMulV2_49_v3 |
| K/V投影合计 | 34 | 16 | 9.932 | 9.890 | 30 | MatMulV2_50_v3 |
| RoPE | 17 | — | 25.967 | 25.920 | 30 | ApplyRotaryPosEmb_16 |
| Attention计算kernel | 17 | 16 | 88.378 | 87.882 | 30 | PrefixInferAttentionScore_16 |
| O投影 | 17 | 16 | 22.841 | 22.800 | 30 | MatMul_64/v2_v3 |
| AddRmsNorm | 33 | — | 26.221 | 26.240 | 30 | RmsNorm_33/AddRmsNorm |
| MLP gate/up合计 | 33 | 16 | 66.212 | 66.181 | 30 | MatMul_65/v2_v3 |
| MLP gate/up合计 | 34 | 16 | 66.660 | 66.691 | 30 | MatMul_66/v2_v3 |
| Swish | 17 | — | 28.642 | 28.630 | 30 | Swish_16 |
| Mul | 17 | — | 16.822 | 16.750 | 30 | Mul_16 |
| MLP down | 17 | 16 | 78.288 | 78.321 | 30 | MatMul_67/v2_v3 |
| AddRmsNorm | 34 | — | 20.505 | 20.300 | 30 | RmsNorm_34/AddRmsNorm |
| Q投影 | 18 | 17 | 20.813 | 20.770 | 30 | MatMulV2_51_v3 |
| K/V投影合计 | 35 | 17 | 10.053 | 10.160 | 30 | MatMulV2_52_v3 |
| K/V投影合计 | 36 | 17 | 9.931 | 9.940 | 30 | MatMulV2_53_v3 |
| RoPE | 18 | — | 26.068 | 25.830 | 30 | ApplyRotaryPosEmb_17 |
| Attention计算kernel | 18 | 17 | 88.407 | 88.041 | 30 | PrefixInferAttentionScore_17 |
| O投影 | 18 | 17 | 22.625 | 22.500 | 30 | MatMul_68/v2_v3 |
| AddRmsNorm | 35 | — | 26.251 | 26.190 | 30 | RmsNorm_35/AddRmsNorm |
| MLP gate/up合计 | 35 | 17 | 66.027 | 66.051 | 30 | MatMul_69/v2_v3 |
| MLP gate/up合计 | 36 | 17 | 66.218 | 66.111 | 30 | MatMul_70/v2_v3 |
| Swish | 18 | — | 28.744 | 28.660 | 30 | Swish_17 |
| Mul | 18 | — | 16.714 | 16.700 | 30 | Mul_17 |
| MLP down | 18 | 17 | 78.366 | 78.461 | 30 | MatMul_71/v2_v3 |
| AddRmsNorm | 36 | — | 20.377 | 19.950 | 30 | RmsNorm_36/AddRmsNorm |
| Q投影 | 19 | 18 | 20.881 | 20.940 | 30 | MatMulV2_54_v3 |
| K/V投影合计 | 37 | 18 | 9.931 | 9.900 | 30 | MatMulV2_55_v3 |
| K/V投影合计 | 38 | 18 | 10.064 | 10.010 | 30 | MatMulV2_56_v3 |
| RoPE | 19 | — | 24.226 | 24.070 | 30 | ApplyRotaryPosEmb_18 |
| Attention计算kernel | 19 | 18 | 89.220 | 89.551 | 30 | PrefixInferAttentionScore_18 |
| O投影 | 19 | 18 | 22.351 | 22.260 | 30 | MatMul_72/v2_v3 |
| AddRmsNorm | 37 | — | 26.491 | 26.520 | 30 | RmsNorm_37/AddRmsNorm |
| MLP gate/up合计 | 37 | 18 | 65.877 | 65.961 | 30 | MatMul_73/v2_v3 |
| MLP gate/up合计 | 38 | 18 | 65.722 | 65.801 | 30 | MatMul_74/v2_v3 |
| Swish | 19 | — | 28.656 | 28.630 | 30 | Swish_18 |
| Mul | 19 | — | 16.964 | 16.710 | 30 | Mul_18 |
| MLP down | 19 | 18 | 78.342 | 78.401 | 30 | MatMul_75/v2_v3 |
| AddRmsNorm | 38 | — | 20.255 | 20.000 | 30 | RmsNorm_38/AddRmsNorm |
| Q投影 | 20 | 19 | 20.833 | 20.880 | 30 | MatMulV2_57_v3 |
| K/V投影合计 | 39 | 19 | 10.002 | 10.020 | 30 | MatMulV2_58_v3 |
| K/V投影合计 | 40 | 19 | 10.009 | 9.940 | 30 | MatMulV2_59_v3 |
| RoPE | 20 | — | 25.975 | 25.880 | 30 | ApplyRotaryPosEmb_19 |
| Attention计算kernel | 20 | 19 | 88.185 | 88.382 | 30 | PrefixInferAttentionScore_19 |
| O投影 | 20 | 19 | 22.778 | 22.700 | 30 | MatMul_76/v2_v3 |
| AddRmsNorm | 39 | — | 26.274 | 26.250 | 30 | RmsNorm_39/AddRmsNorm |
| MLP gate/up合计 | 39 | 19 | 66.289 | 66.421 | 30 | MatMul_77/v2_v3 |
| MLP gate/up合计 | 40 | 19 | 66.289 | 66.351 | 30 | MatMul_78/v2_v3 |
| Swish | 20 | — | 28.630 | 28.620 | 30 | Swish_19 |
| Mul | 20 | — | 16.909 | 16.690 | 30 | Mul_19 |
| MLP down | 20 | 19 | 78.554 | 78.641 | 30 | MatMul_79/v2_v3 |
| AddRmsNorm | 40 | — | 20.398 | 19.850 | 30 | RmsNorm_40/AddRmsNorm |
| Q投影 | 21 | 20 | 20.912 | 20.950 | 30 | MatMulV2_60_v3 |
| K/V投影合计 | 41 | 20 | 9.915 | 9.770 | 30 | MatMulV2_61_v3 |
| K/V投影合计 | 42 | 20 | 10.026 | 9.990 | 30 | MatMulV2_62_v3 |
| RoPE | 21 | — | 25.766 | 25.750 | 30 | ApplyRotaryPosEmb_20 |
| Attention计算kernel | 21 | 20 | 89.408 | 89.142 | 30 | PrefixInferAttentionScore_20 |
| O投影 | 21 | 20 | 23.025 | 23.220 | 30 | MatMul_80/v2_v3 |
| AddRmsNorm | 41 | — | 26.339 | 26.270 | 30 | RmsNorm_41/AddRmsNorm |
| MLP gate/up合计 | 41 | 20 | 66.125 | 66.131 | 30 | MatMul_81/v2_v3 |
| MLP gate/up合计 | 42 | 20 | 66.204 | 66.391 | 30 | MatMul_82/v2_v3 |
| Swish | 21 | — | 28.639 | 28.620 | 30 | Swish_20 |
| Mul | 21 | — | 16.757 | 16.750 | 30 | Mul_20 |
| MLP down | 21 | 20 | 78.366 | 78.391 | 30 | MatMul_83/v2_v3 |
| AddRmsNorm | 42 | — | 20.310 | 19.960 | 30 | RmsNorm_42/AddRmsNorm |
| Q投影 | 22 | 21 | 20.973 | 20.930 | 30 | MatMulV2_63_v3 |
| K/V投影合计 | 43 | 21 | 10.102 | 10.170 | 30 | MatMulV2_64_v3 |
| K/V投影合计 | 44 | 21 | 9.961 | 9.860 | 30 | MatMulV2_65_v3 |
| RoPE | 22 | — | 26.062 | 25.890 | 30 | ApplyRotaryPosEmb_21 |
| Attention计算kernel | 22 | 21 | 88.171 | 87.681 | 30 | PrefixInferAttentionScore_21 |
| O投影 | 22 | 21 | 22.745 | 22.880 | 30 | MatMul_84/v2_v3 |
| AddRmsNorm | 43 | — | 26.449 | 26.350 | 30 | RmsNorm_43/AddRmsNorm |
| MLP gate/up合计 | 43 | 21 | 67.108 | 67.141 | 30 | MatMul_85/v2_v3 |
| MLP gate/up合计 | 44 | 21 | 66.589 | 66.481 | 30 | MatMul_86/v2_v3 |
| Swish | 22 | — | 28.738 | 28.730 | 30 | Swish_21 |
| Mul | 22 | — | 16.832 | 16.660 | 30 | Mul_21 |
| MLP down | 22 | 21 | 78.285 | 78.311 | 30 | MatMul_87/v2_v3 |
| AddRmsNorm | 44 | — | 20.481 | 20.130 | 30 | RmsNorm_44/AddRmsNorm |
| Q投影 | 23 | 22 | 20.745 | 20.760 | 30 | MatMulV2_66_v3 |
| K/V投影合计 | 45 | 22 | 9.973 | 9.940 | 30 | MatMulV2_67_v3 |
| K/V投影合计 | 46 | 22 | 10.036 | 10.040 | 30 | MatMulV2_68_v3 |
| RoPE | 23 | — | 25.921 | 25.850 | 30 | ApplyRotaryPosEmb_22 |
| Attention计算kernel | 23 | 22 | 88.406 | 88.501 | 30 | PrefixInferAttentionScore_22 |
| O投影 | 23 | 22 | 22.421 | 22.240 | 30 | MatMul_88/v2_v3 |
| AddRmsNorm | 45 | — | 26.315 | 26.260 | 30 | RmsNorm_45/AddRmsNorm |
| MLP gate/up合计 | 45 | 22 | 66.383 | 66.361 | 30 | MatMul_89/v2_v3 |
| MLP gate/up合计 | 46 | 22 | 66.122 | 66.151 | 30 | MatMul_90/v2_v3 |
| Swish | 23 | — | 28.669 | 28.630 | 30 | Swish_22 |
| Mul | 23 | — | 16.935 | 16.740 | 30 | Mul_22 |
| MLP down | 23 | 22 | 78.274 | 78.191 | 30 | MatMul_91/v2_v3 |
| AddRmsNorm | 46 | — | 20.548 | 20.080 | 30 | RmsNorm_46/AddRmsNorm |
| Q投影 | 24 | 23 | 20.895 | 20.910 | 30 | MatMulV2_69_v3 |
| K/V投影合计 | 47 | 23 | 10.016 | 10.030 | 30 | MatMulV2_70_v3 |
| K/V投影合计 | 48 | 23 | 10.146 | 10.240 | 30 | MatMulV2_71_v3 |
| RoPE | 24 | — | 24.014 | 23.900 | 30 | ApplyRotaryPosEmb_23 |
| Attention计算kernel | 24 | 23 | 88.481 | 88.562 | 30 | PrefixInferAttentionScore_23 |
| O投影 | 24 | 23 | 22.543 | 22.610 | 30 | MatMul_92/v2_v3 |
| AddRmsNorm | 47 | — | 26.797 | 26.760 | 30 | RmsNorm_47/AddRmsNorm |
| MLP gate/up合计 | 47 | 23 | 65.905 | 65.901 | 30 | MatMul_93/v2_v3 |
| MLP gate/up合计 | 48 | 23 | 65.756 | 65.801 | 30 | MatMul_94/v2_v3 |
| Swish | 24 | — | 28.771 | 28.690 | 30 | Swish_23 |
| Mul | 24 | — | 16.910 | 16.720 | 30 | Mul_23 |
| MLP down | 24 | 23 | 78.495 | 78.401 | 30 | MatMul_95/v2_v3 |
| Add | 1 | — | 9.771 | 9.600 | 30 | Add_47 |
| StridedSliceV3 | 1 | — | 11.413 | 11.060 | 30 | StridedSliceV2 |
| Sub | 1 | — | 2.815 | 2.800 | 30 | Sub |
| GatherV2 | 4 | — | 4.929 | 4.920 | 30 | GatherV2_3 |
| RmsNorm | 2 | — | 8.265 | 8.250 | 30 | RmsNorm_48 |
| 裁剪后的输出头 | 1 | 24 | 6.302 | 6.340 | 30 | MatMul_96/v2_v3 |

### A5逐节点

| 类别 | 组内序号 | 层 | 均值 us | 中位数 us | 次数 | 原始节点或kernel |
| --- | --- | --- | --- | --- | --- | --- |
| GatherV2 | 1 | — | 8.698 | 8.713 | 30 | GatherV2 |
| RmsNorm | 1 | — | 6.615 | 6.620 | 30 | RmsNorm |
| K/V投影合计 | 1 | 0 | 7.976 | 7.978 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 2 | 0 | 8.010 | 8.047 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| Q投影 | 1 | 0 | 13.027 | 12.995 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| GatherV2 | 2 | — | 3.671 | 3.746 | 30 | GatherV2_1 |
| GatherV2 | 3 | — | 3.580 | 3.637 | 30 | GatherV2_2 |
| RoPE | 1 | — | 8.455 | 8.415 | 30 | ApplyRotaryPosEmb |
| bounds int64→int32 | 1 | — | 2.174 | 2.170 | 30 | Cast |
| A5 metadata清零 | 1 | 0 | 2.518 | 2.510 | 30 | N/A |
| A5 Attention planner | 1 | 0 | 35.299 | 30.447 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 1 | 0 | 35.064 | 35.032 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 1 | 0 | 13.765 | 13.777 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 1 | — | 8.224 | 8.308 | 30 | RmsNorm_1/AddRmsNorm |
| MLP gate/up合计 | 1 | 0 | 51.333 | 51.326 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 2 | 0 | 50.559 | 50.506 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 1 | — | 18.625 | 18.569 | 30 | Swish |
| Mul | 1 | — | 14.693 | 14.680 | 30 | Mul |
| MLP down | 1 | 0 | 52.602 | 52.609 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 2 | — | 9.268 | 9.238 | 30 | RmsNorm_2/AddRmsNorm |
| Q投影 | 2 | 1 | 12.997 | 12.963 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 3 | 1 | 7.937 | 7.917 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 4 | 1 | 7.886 | 7.965 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 2 | — | 8.515 | 8.502 | 30 | ApplyRotaryPosEmb_1 |
| A5 metadata清零 | 2 | 1 | 2.595 | 2.514 | 30 | N/A |
| A5 Attention planner | 2 | 1 | 31.073 | 30.517 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 2 | 1 | 34.681 | 34.579 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 2 | 1 | 13.878 | 13.909 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 3 | — | 8.134 | 8.186 | 30 | RmsNorm_3/AddRmsNorm |
| MLP gate/up合计 | 3 | 1 | 51.563 | 51.525 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 4 | 1 | 50.637 | 50.630 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 2 | — | 19.180 | 19.157 | 30 | Swish_1 |
| Mul | 2 | — | 15.014 | 14.891 | 30 | Mul_1 |
| MLP down | 2 | 1 | 52.536 | 52.520 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 4 | — | 8.758 | 8.828 | 30 | RmsNorm_4/AddRmsNorm |
| Q投影 | 3 | 2 | 12.985 | 12.969 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 5 | 2 | 8.069 | 8.099 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 6 | 2 | 8.036 | 8.059 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 3 | — | 8.649 | 8.633 | 30 | ApplyRotaryPosEmb_2 |
| A5 metadata清零 | 3 | 2 | 2.511 | 2.483 | 30 | N/A |
| A5 Attention planner | 3 | 2 | 30.527 | 30.221 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 3 | 2 | 34.931 | 35.102 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 3 | 2 | 13.795 | 13.801 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 5 | — | 8.250 | 8.314 | 30 | RmsNorm_5/AddRmsNorm |
| MLP gate/up合计 | 5 | 2 | 51.210 | 51.200 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 6 | 2 | 50.340 | 50.354 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 3 | — | 16.132 | 16.113 | 30 | Swish_2 |
| Mul | 3 | — | 16.668 | 16.587 | 30 | Mul_2 |
| MLP down | 3 | 2 | 52.567 | 52.570 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 6 | — | 8.872 | 8.904 | 30 | RmsNorm_6/AddRmsNorm |
| K/V投影合计 | 7 | 3 | 8.002 | 7.956 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 8 | 3 | 7.927 | 7.918 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| Q投影 | 4 | 3 | 12.971 | 12.938 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 4 | — | 8.513 | 8.535 | 30 | ApplyRotaryPosEmb_3 |
| A5 metadata清零 | 4 | 3 | 2.468 | 2.459 | 30 | N/A |
| A5 Attention planner | 4 | 3 | 30.598 | 30.306 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 4 | 3 | 35.179 | 35.215 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 4 | 3 | 13.802 | 13.811 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 7 | — | 8.386 | 8.462 | 30 | RmsNorm_7/AddRmsNorm |
| MLP gate/up合计 | 7 | 3 | 51.081 | 50.989 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 8 | 3 | 50.537 | 50.543 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 4 | — | 18.430 | 18.318 | 30 | Swish_3 |
| Mul | 4 | — | 14.811 | 14.691 | 30 | Mul_3 |
| MLP down | 4 | 3 | 52.557 | 52.522 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 8 | — | 9.365 | 9.360 | 30 | RmsNorm_8/AddRmsNorm |
| Q投影 | 5 | 4 | 13.031 | 13.046 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 9 | 4 | 7.885 | 7.952 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 10 | 4 | 7.940 | 7.958 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 5 | — | 8.565 | 8.505 | 30 | ApplyRotaryPosEmb_4 |
| A5 metadata清零 | 5 | 4 | 2.500 | 2.467 | 30 | N/A |
| A5 Attention planner | 5 | 4 | 30.508 | 30.004 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 5 | 4 | 34.621 | 34.486 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 5 | 4 | 13.785 | 13.765 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 9 | — | 8.262 | 8.285 | 30 | RmsNorm_9/AddRmsNorm |
| MLP gate/up合计 | 9 | 4 | 51.501 | 51.447 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 10 | 4 | 50.482 | 50.491 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 5 | — | 19.559 | 19.557 | 30 | Swish_4 |
| Mul | 5 | — | 15.247 | 15.166 | 30 | Mul_4 |
| MLP down | 5 | 4 | 52.580 | 52.580 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 10 | — | 9.022 | 9.010 | 30 | RmsNorm_10/AddRmsNorm |
| Q投影 | 6 | 5 | 13.091 | 13.079 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 11 | 5 | 8.041 | 8.017 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 12 | 5 | 8.033 | 8.028 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 6 | — | 8.479 | 8.481 | 30 | ApplyRotaryPosEmb_5 |
| A5 metadata清零 | 6 | 5 | 2.482 | 2.438 | 30 | N/A |
| A5 Attention planner | 6 | 5 | 30.800 | 30.453 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 6 | 5 | 35.131 | 34.983 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 6 | 5 | 13.818 | 13.809 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 11 | — | 8.325 | 8.346 | 30 | RmsNorm_11/AddRmsNorm |
| MLP gate/up合计 | 11 | 5 | 51.588 | 51.520 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 12 | 5 | 50.462 | 50.451 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 6 | — | 19.939 | 19.742 | 30 | Swish_5 |
| Mul | 6 | — | 14.941 | 14.914 | 30 | Mul_5 |
| MLP down | 6 | 5 | 52.672 | 52.651 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 12 | — | 9.419 | 9.415 | 30 | RmsNorm_12/AddRmsNorm |
| Q投影 | 7 | 6 | 13.033 | 13.053 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 13 | 6 | 7.984 | 7.982 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 14 | 6 | 8.012 | 8.023 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 7 | — | 8.454 | 8.441 | 30 | ApplyRotaryPosEmb_6 |
| A5 metadata清零 | 7 | 6 | 2.508 | 2.507 | 30 | N/A |
| A5 Attention planner | 7 | 6 | 30.281 | 30.261 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 7 | 6 | 34.252 | 34.294 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 7 | 6 | 13.816 | 13.791 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 13 | — | 8.275 | 8.307 | 30 | RmsNorm_13/AddRmsNorm |
| MLP gate/up合计 | 13 | 6 | 51.660 | 51.582 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 14 | 6 | 50.631 | 50.647 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 7 | — | 19.720 | 19.703 | 30 | Swish_6 |
| Mul | 7 | — | 15.146 | 15.076 | 30 | Mul_6 |
| MLP down | 7 | 6 | 52.555 | 52.563 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 14 | — | 9.200 | 9.122 | 30 | RmsNorm_14/AddRmsNorm |
| Q投影 | 8 | 7 | 13.065 | 13.079 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 15 | 7 | 8.020 | 8.062 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 16 | 7 | 7.994 | 7.986 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 8 | — | 8.603 | 8.553 | 30 | ApplyRotaryPosEmb_7 |
| A5 metadata清零 | 8 | 7 | 2.470 | 2.458 | 30 | N/A |
| A5 Attention planner | 8 | 7 | 30.498 | 30.289 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 8 | 7 | 35.595 | 35.620 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 8 | 7 | 13.814 | 13.813 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 15 | — | 8.329 | 8.323 | 30 | RmsNorm_15/AddRmsNorm |
| MLP gate/up合计 | 15 | 7 | 51.314 | 51.248 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 16 | 7 | 50.406 | 50.422 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 8 | — | 20.049 | 20.005 | 30 | Swish_7 |
| Mul | 8 | — | 14.761 | 14.627 | 30 | Mul_7 |
| MLP down | 8 | 7 | 52.617 | 52.556 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 16 | — | 9.372 | 9.276 | 30 | RmsNorm_16/AddRmsNorm |
| Q投影 | 9 | 8 | 13.099 | 13.108 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 17 | 8 | 8.053 | 8.057 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 18 | 8 | 7.976 | 7.994 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 9 | — | 8.493 | 8.492 | 30 | ApplyRotaryPosEmb_8 |
| A5 metadata清零 | 9 | 8 | 2.462 | 2.442 | 30 | N/A |
| A5 Attention planner | 9 | 8 | 30.687 | 30.258 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 9 | 8 | 34.576 | 34.472 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 9 | 8 | 13.828 | 13.825 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 17 | — | 8.419 | 8.389 | 30 | RmsNorm_17/AddRmsNorm |
| MLP gate/up合计 | 17 | 8 | 51.313 | 51.307 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 18 | 8 | 50.584 | 50.603 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 9 | — | 19.435 | 19.453 | 30 | Swish_8 |
| Mul | 9 | — | 15.024 | 15.055 | 30 | Mul_8 |
| MLP down | 9 | 8 | 52.543 | 52.531 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 18 | — | 9.300 | 9.315 | 30 | RmsNorm_18/AddRmsNorm |
| Q投影 | 10 | 9 | 13.050 | 13.079 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 19 | 9 | 7.991 | 8.002 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 20 | 9 | 7.968 | 7.944 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 10 | — | 8.571 | 8.575 | 30 | ApplyRotaryPosEmb_9 |
| A5 metadata清零 | 10 | 9 | 2.621 | 2.453 | 30 | N/A |
| A5 Attention planner | 10 | 9 | 30.536 | 30.040 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 10 | 9 | 34.805 | 34.709 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 10 | 9 | 13.769 | 13.777 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 19 | — | 8.334 | 8.358 | 30 | RmsNorm_19/AddRmsNorm |
| MLP gate/up合计 | 19 | 9 | 51.428 | 51.329 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 20 | 9 | 50.312 | 50.319 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 10 | — | 19.829 | 19.855 | 30 | Swish_9 |
| Mul | 10 | — | 14.877 | 14.848 | 30 | Mul_9 |
| MLP down | 10 | 9 | 52.544 | 52.547 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 20 | — | 9.210 | 9.157 | 30 | RmsNorm_20/AddRmsNorm |
| Q投影 | 11 | 10 | 13.071 | 13.073 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 21 | 10 | 8.047 | 8.014 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 22 | 10 | 7.930 | 7.963 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 11 | — | 8.593 | 8.575 | 30 | ApplyRotaryPosEmb_10 |
| A5 metadata清零 | 11 | 10 | 2.495 | 2.482 | 30 | N/A |
| A5 Attention planner | 11 | 10 | 30.526 | 30.213 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 11 | 10 | 34.987 | 34.868 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 11 | 10 | 13.849 | 13.823 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 21 | — | 8.317 | 8.352 | 30 | RmsNorm_21/AddRmsNorm |
| MLP gate/up合计 | 21 | 10 | 51.656 | 51.599 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 22 | 10 | 50.503 | 50.503 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 11 | — | 19.739 | 19.687 | 30 | Swish_10 |
| Mul | 11 | — | 15.080 | 14.963 | 30 | Mul_10 |
| MLP down | 11 | 10 | 52.605 | 52.559 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 22 | — | 9.265 | 9.302 | 30 | RmsNorm_22/AddRmsNorm |
| Q投影 | 12 | 11 | 13.055 | 13.051 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 23 | 11 | 8.109 | 8.084 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 24 | 11 | 7.998 | 7.990 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 12 | — | 8.649 | 8.630 | 30 | ApplyRotaryPosEmb_11 |
| A5 metadata清零 | 12 | 11 | 2.498 | 2.465 | 30 | N/A |
| A5 Attention planner | 12 | 11 | 30.536 | 30.134 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 12 | 11 | 34.683 | 34.661 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 12 | 11 | 13.854 | 13.869 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 23 | — | 8.035 | 8.055 | 30 | RmsNorm_23/AddRmsNorm |
| MLP gate/up合计 | 23 | 11 | 51.480 | 51.449 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 24 | 11 | 50.588 | 50.585 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 12 | — | 19.121 | 19.051 | 30 | Swish_11 |
| Mul | 12 | — | 14.965 | 15.003 | 30 | Mul_11 |
| MLP down | 12 | 11 | 52.574 | 52.551 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 24 | — | 8.820 | 8.717 | 30 | RmsNorm_24/AddRmsNorm |
| Q投影 | 13 | 12 | 13.080 | 13.055 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 25 | 12 | 8.020 | 8.022 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 26 | 12 | 7.997 | 7.986 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 13 | — | 8.508 | 8.412 | 30 | ApplyRotaryPosEmb_12 |
| A5 metadata清零 | 13 | 12 | 2.468 | 2.448 | 30 | N/A |
| A5 Attention planner | 13 | 12 | 30.291 | 30.104 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 13 | 12 | 35.124 | 35.182 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 13 | 12 | 13.828 | 13.790 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 25 | — | 8.358 | 8.428 | 30 | RmsNorm_25/AddRmsNorm |
| MLP gate/up合计 | 25 | 12 | 51.334 | 51.372 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 26 | 12 | 50.409 | 50.368 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 13 | — | 19.217 | 19.166 | 30 | Swish_12 |
| Mul | 13 | — | 14.876 | 14.886 | 30 | Mul_12 |
| MLP down | 13 | 12 | 52.537 | 52.533 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 26 | — | 9.351 | 9.314 | 30 | RmsNorm_26/AddRmsNorm |
| Q投影 | 14 | 13 | 13.067 | 13.073 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 27 | 13 | 8.058 | 8.092 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 28 | 13 | 8.031 | 8.029 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 14 | — | 8.440 | 8.417 | 30 | ApplyRotaryPosEmb_13 |
| A5 metadata清零 | 14 | 13 | 2.504 | 2.481 | 30 | N/A |
| A5 Attention planner | 14 | 13 | 30.370 | 30.246 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 14 | 13 | 34.752 | 34.677 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 14 | 13 | 13.805 | 13.808 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 27 | — | 8.031 | 8.043 | 30 | RmsNorm_27/AddRmsNorm |
| MLP gate/up合计 | 27 | 13 | 51.468 | 51.443 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 28 | 13 | 50.552 | 50.555 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 14 | — | 19.825 | 19.851 | 30 | Swish_13 |
| Mul | 14 | — | 15.155 | 15.194 | 30 | Mul_13 |
| MLP down | 14 | 13 | 52.574 | 52.518 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 28 | — | 9.171 | 9.075 | 30 | RmsNorm_28/AddRmsNorm |
| Q投影 | 15 | 14 | 13.067 | 13.064 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 29 | 14 | 8.036 | 8.017 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 30 | 14 | 8.001 | 7.978 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 15 | — | 8.608 | 8.622 | 30 | ApplyRotaryPosEmb_14 |
| A5 metadata清零 | 15 | 14 | 2.458 | 2.449 | 30 | N/A |
| A5 Attention planner | 15 | 14 | 30.428 | 30.158 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 15 | 14 | 34.927 | 34.913 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 15 | 14 | 13.963 | 13.934 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 29 | — | 8.339 | 8.331 | 30 | RmsNorm_29/AddRmsNorm |
| MLP gate/up合计 | 29 | 14 | 51.470 | 51.422 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 30 | 14 | 50.451 | 50.363 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 15 | — | 19.495 | 19.400 | 30 | Swish_14 |
| Mul | 15 | — | 14.963 | 14.912 | 30 | Mul_14 |
| MLP down | 15 | 14 | 52.618 | 52.555 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 30 | — | 9.283 | 9.293 | 30 | RmsNorm_30/AddRmsNorm |
| Q投影 | 16 | 15 | 13.025 | 13.007 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 31 | 15 | 8.014 | 8.035 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 32 | 15 | 8.026 | 8.040 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 16 | — | 8.532 | 8.494 | 30 | ApplyRotaryPosEmb_15 |
| A5 metadata清零 | 16 | 15 | 2.433 | 2.442 | 30 | N/A |
| A5 Attention planner | 16 | 15 | 30.506 | 30.419 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 16 | 15 | 34.699 | 34.632 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 16 | 15 | 13.777 | 13.771 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 31 | — | 8.070 | 8.081 | 30 | RmsNorm_31/AddRmsNorm |
| MLP gate/up合计 | 31 | 15 | 51.676 | 51.675 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 32 | 15 | 50.569 | 50.576 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 16 | — | 19.826 | 19.691 | 30 | Swish_15 |
| Mul | 16 | — | 15.233 | 15.207 | 30 | Mul_15 |
| MLP down | 16 | 15 | 52.548 | 52.580 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 32 | — | 9.214 | 9.171 | 30 | RmsNorm_32/AddRmsNorm |
| Q投影 | 17 | 16 | 13.015 | 13.020 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 33 | 16 | 7.938 | 7.899 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 34 | 16 | 7.964 | 7.986 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 17 | — | 8.565 | 8.519 | 30 | ApplyRotaryPosEmb_16 |
| A5 metadata清零 | 17 | 16 | 2.409 | 2.407 | 30 | N/A |
| A5 Attention planner | 17 | 16 | 30.478 | 30.273 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 17 | 16 | 35.146 | 35.142 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 17 | 16 | 13.784 | 13.794 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 33 | — | 8.411 | 8.454 | 30 | RmsNorm_33/AddRmsNorm |
| MLP gate/up合计 | 33 | 16 | 51.453 | 51.447 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 34 | 16 | 50.344 | 50.313 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 17 | — | 19.844 | 19.733 | 30 | Swish_16 |
| Mul | 17 | — | 14.872 | 14.808 | 30 | Mul_16 |
| MLP down | 17 | 16 | 52.566 | 52.564 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 34 | — | 9.277 | 9.216 | 30 | RmsNorm_34/AddRmsNorm |
| Q投影 | 18 | 17 | 13.090 | 13.076 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 35 | 17 | 8.102 | 8.120 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 36 | 17 | 8.079 | 8.090 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 18 | — | 8.620 | 8.608 | 30 | ApplyRotaryPosEmb_17 |
| A5 metadata清零 | 18 | 17 | 2.419 | 2.399 | 30 | N/A |
| A5 Attention planner | 18 | 17 | 30.572 | 30.263 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 18 | 17 | 35.215 | 35.078 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 18 | 17 | 13.797 | 13.781 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 35 | — | 8.370 | 8.343 | 30 | RmsNorm_35/AddRmsNorm |
| MLP gate/up合计 | 35 | 17 | 51.399 | 51.401 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 36 | 17 | 50.601 | 50.574 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 18 | — | 19.995 | 19.934 | 30 | Swish_17 |
| Mul | 18 | — | 14.935 | 14.942 | 30 | Mul_17 |
| MLP down | 18 | 17 | 52.514 | 52.490 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 36 | — | 9.402 | 9.412 | 30 | RmsNorm_36/AddRmsNorm |
| Q投影 | 19 | 18 | 13.014 | 12.981 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 37 | 18 | 7.888 | 7.886 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 38 | 18 | 8.037 | 8.089 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 19 | — | 8.504 | 8.505 | 30 | ApplyRotaryPosEmb_18 |
| A5 metadata清零 | 19 | 18 | 2.401 | 2.380 | 30 | N/A |
| A5 Attention planner | 19 | 18 | 30.267 | 30.144 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 19 | 18 | 34.589 | 34.462 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 19 | 18 | 13.801 | 13.780 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 37 | — | 8.033 | 8.049 | 30 | RmsNorm_37/AddRmsNorm |
| MLP gate/up合计 | 37 | 18 | 51.399 | 51.384 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 38 | 18 | 50.310 | 50.277 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 19 | — | 19.804 | 19.634 | 30 | Swish_18 |
| Mul | 19 | — | 14.775 | 14.704 | 30 | Mul_18 |
| MLP down | 19 | 18 | 52.558 | 52.559 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 38 | — | 9.215 | 9.168 | 30 | RmsNorm_38/AddRmsNorm |
| Q投影 | 20 | 19 | 13.042 | 13.071 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 39 | 19 | 8.062 | 8.024 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 40 | 19 | 8.017 | 8.048 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 20 | — | 8.503 | 8.381 | 30 | ApplyRotaryPosEmb_19 |
| A5 metadata清零 | 20 | 19 | 2.391 | 2.380 | 30 | N/A |
| A5 Attention planner | 20 | 19 | 30.310 | 30.159 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 20 | 19 | 35.172 | 35.094 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 20 | 19 | 13.783 | 13.754 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 39 | — | 8.300 | 8.270 | 30 | RmsNorm_39/AddRmsNorm |
| MLP gate/up合计 | 39 | 19 | 51.524 | 51.511 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 40 | 19 | 50.403 | 50.410 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 20 | — | 19.846 | 19.556 | 30 | Swish_19 |
| Mul | 20 | — | 15.195 | 15.116 | 30 | Mul_19 |
| MLP down | 20 | 19 | 52.570 | 52.580 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 40 | — | 9.355 | 9.395 | 30 | RmsNorm_40/AddRmsNorm |
| Q投影 | 21 | 20 | 13.066 | 13.062 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 41 | 20 | 7.995 | 8.003 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 42 | 20 | 8.012 | 7.973 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 21 | — | 8.530 | 8.506 | 30 | ApplyRotaryPosEmb_20 |
| A5 metadata清零 | 21 | 20 | 2.389 | 2.363 | 30 | N/A |
| A5 Attention planner | 21 | 20 | 30.414 | 30.086 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 21 | 20 | 34.410 | 34.361 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 21 | 20 | 13.786 | 13.784 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 41 | — | 8.047 | 8.024 | 30 | RmsNorm_41/AddRmsNorm |
| MLP gate/up合计 | 41 | 20 | 51.619 | 51.627 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 42 | 20 | 50.584 | 50.534 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 21 | — | 19.623 | 19.564 | 30 | Swish_20 |
| Mul | 21 | — | 15.049 | 15.007 | 30 | Mul_20 |
| MLP down | 21 | 20 | 52.530 | 52.541 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 42 | — | 8.949 | 8.981 | 30 | RmsNorm_42/AddRmsNorm |
| Q投影 | 22 | 21 | 13.059 | 13.046 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 43 | 21 | 8.155 | 8.143 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 44 | 21 | 7.946 | 7.959 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 22 | — | 8.414 | 8.416 | 30 | ApplyRotaryPosEmb_21 |
| A5 metadata清零 | 22 | 21 | 2.411 | 2.388 | 30 | N/A |
| A5 Attention planner | 22 | 21 | 30.295 | 30.058 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 22 | 21 | 35.329 | 35.338 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 22 | 21 | 13.832 | 13.840 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 43 | — | 8.321 | 8.395 | 30 | RmsNorm_43/AddRmsNorm |
| MLP gate/up合计 | 43 | 21 | 51.334 | 51.303 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 44 | 21 | 50.365 | 50.285 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 22 | — | 19.712 | 19.634 | 30 | Swish_21 |
| Mul | 22 | — | 15.029 | 15.087 | 30 | Mul_21 |
| MLP down | 22 | 21 | 52.692 | 52.552 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 44 | — | 9.457 | 9.395 | 30 | RmsNorm_44/AddRmsNorm |
| Q投影 | 23 | 22 | 13.039 | 13.067 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 45 | 22 | 7.978 | 7.954 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 46 | 22 | 8.062 | 8.064 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 23 | — | 8.520 | 8.479 | 30 | ApplyRotaryPosEmb_22 |
| A5 metadata清零 | 23 | 22 | 2.404 | 2.380 | 30 | N/A |
| A5 Attention planner | 23 | 22 | 30.229 | 30.052 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 23 | 22 | 34.459 | 34.362 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 23 | 22 | 13.849 | 13.798 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 45 | — | 8.150 | 8.088 | 30 | RmsNorm_45/AddRmsNorm |
| MLP gate/up合计 | 45 | 22 | 51.578 | 51.599 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 46 | 22 | 50.472 | 50.479 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 23 | — | 19.451 | 19.568 | 30 | Swish_22 |
| Mul | 23 | — | 15.249 | 15.191 | 30 | Mul_22 |
| MLP down | 23 | 22 | 52.570 | 52.587 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 46 | — | 9.266 | 9.233 | 30 | RmsNorm_46/AddRmsNorm |
| Q投影 | 24 | 23 | 13.050 | 13.040 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 47 | 23 | 8.004 | 7.985 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| K/V投影合计 | 48 | 23 | 8.029 | 8.035 | 30 | aclnnAddmm_MatMulV3Common_MatMulV3 |
| RoPE | 24 | — | 8.543 | 8.486 | 30 | ApplyRotaryPosEmb_23 |
| A5 metadata清零 | 24 | 23 | 2.473 | 2.449 | 30 | N/A |
| A5 Attention planner | 24 | 23 | 30.566 | 30.104 | 30 | _Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface |
| Attention计算kernel | 24 | 23 | 35.115 | 35.075 | 30 | _Z36kernel__PrefixMacroKernel___call___023KernelArgTensor_half_3dS_S_S_f26KernelArgTensor_int32_t_1dS0_ |
| O投影 | 24 | 23 | 13.805 | 13.784 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| AddRmsNorm | 47 | — | 8.487 | 8.516 | 30 | RmsNorm_47/AddRmsNorm |
| MLP gate/up合计 | 47 | 23 | 51.428 | 51.431 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| MLP gate/up合计 | 48 | 23 | 50.426 | 50.389 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Swish | 24 | — | 19.939 | 19.947 | 30 | Swish_23 |
| Mul | 24 | — | 14.959 | 14.953 | 30 | Mul_23 |
| MLP down | 24 | 23 | 52.546 | 52.554 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |
| Add | 1 | — | 7.132 | 7.149 | 30 | Add_47 |
| StridedSliceV2 | 1 | — | 3.469 | 3.365 | 30 | StridedSliceV2 |
| Sub | 1 | — | 2.552 | 2.528 | 30 | Sub |
| GatherV2 | 4 | — | 3.516 | 3.545 | 30 | GatherV2_3 |
| RmsNorm | 2 | — | 5.061 | 5.058 | 30 | RmsNorm_48 |
| 裁剪后的输出头 | 1 | 24 | 2.685 | 2.705 | 30 | aclnnMm_MatMulV3Common_MatMulV3 |

## 附录E：复现与原始证据

两端实验目录均为各自项目根下的`work/a3-a5-profiling-20261021`。其中保留隔离configs、输入池、attempts.jsonl、environment.json、原始PROF会话、msprof导出日志、profile-analysis.json及audit.json。所有run_id均可在各自`work/qwen2.5-0.5b/results/<run_id>`找到完整配置和日志。

```bash
# 在对应机器、容器和项目根目录中；A3还需source env_a3.sh
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
python3 -m core.pipeline bench --config work/a3-a5-profiling-20261021/configs/profile-median-full-i1.yaml --device 1 --profiling
msprof --export=on --output=<本次结果目录>/bench-profiling/PROF_<会话>
python3 work/a3-a5-profiling-20261021/analyze_profile.py
python3 work/a3-a5-profiling-20261021/audit_profile.py
```

本次22次运行共22570次模型调用；12组profiling保留570个测量图。逐调用QPS/P99复算、零错误、输入摘要、相同seed序列、预热剔除、每图169 MatMul/24 Attention以及A5每图24 planner/24清零检查均通过。各自默认配置与默认输入池前后摘要相同。

交付分析记录包使用JSON/JSONL保存原始表字段，不包含CSV文件。大体积原始PROF二进制与时间线继续保存在两台服务器上述目录。文档文件名为`qwen2.5-0.5b-a3-a5-profiling.md`，记录包为`qwen-a3-a5-profiling-20261021.zip`。

## 附录F：A5 AICPU planner 辅流版本复测（2026-10-21）

### 结论与范围

默认 `16|32`、batch16、4实例下，辅流版本相对本文历史 A5 基线**未显示明显端到端提升**。
三轮业务 QPS 中位数为 **2659.628**（历史2664.147，**-0.170%**），
最差轮整 batch P99为 **27.131ms**（历史26.994ms，**+0.137ms / +0.508%**），仍低于60ms。
本轮按历史栈与当前栈直接比较，没有另外构建同CANN主流对照；算子变化以profiling实测列示。
按未复测的历史A3中位数2168.494计算，当前A5/A3约为1.226倍，仍接近原来的1.229倍。

环境为850-a5 / `hejun-cann910-stream`、逻辑卡1、CANN `9.3.0~weekly.20261014.01`，
GE Compiler / Executor由 `upstream/9.3.0-beta.1` @ `cec5518fe3eb39dd7eec7b1fd124280b2ff6eaf2`
配套构建，`ascend-ops` 为 `0d59d480eda0d2dfae3d3e7026ac0bbb0ab1dc29`。
Torch / TorchNPU为2.10.0+cpu / 2.10.0.post2，Transformers为5.10.1。
同一AIR和算子库用于本轮全部运行；普通性能关闭profiling，沿用本文业务QPS和整batch延迟口径。
100组batch16输入池摘要仍为 `21defc288cc9848d6f1f71f8a8c6d900327963431e57e77125274b5f5fc8c85b`，
每个seed的全局请求序列均与对应历史记录一致。普通测量共9000次调用，错误数均为0，
每实例均先预热全部100个输入shape，未把编译、加载或预热计入测量。

### 普通性能逐轮对比

| seed | 调用数 | 历史业务QPS | 当前业务QPS | QPS变化 | 历史P99 ms | 当前P99 ms | 当前max ms | 当前run_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 3000 | 2664.462 | 2659.628 | -0.181% | 26.994 | 26.919 | 53.945 | 20261021T132813-f0cbaa10 |
| 43 | 3000 | 2655.138 | 2653.560 | -0.059% | 26.977 | 27.131 | 55.952 | 20261021T132939-1365564c |
| 44 | 3000 | 2664.147 | 2662.196 | -0.073% | 26.818 | 26.823 | 53.850 | 20261021T133131-836947b4 |

当前QPS范围2653.560–2662.196，历史范围2655.138–2664.462；两者高度重叠。
逐次记录重算的P99与业务QPS均与result.json一致，所有测量调用的 `first_hit=0`。

### 逐算子对比：中位输入、单实例、32|64

复用本文 `case-0020`（T=2041、prefix=24、batch16），预热10次后保留30个稳态图。
Attention内核仍固定16个AIC。下表单位us，累计值为每图该类全部调用的平均总时间。
负变化表示耗时减少。原表将StridedSliceV2简称为Slice，此处按实际算子类型列示。

| 算子 | 次/图 | 历史单次 us | 当前单次 us | 历史累计 us/图 | 当前累计 us/图 | 单次变化 |
| --- | --- | --- | --- | --- | --- | --- |
| Q投影 | 24 | 13.045 | 13.053 | 313.087 | 313.275 | +0.06% |
| K/V投影合计 | 48 | 8.006 | 7.967 | 384.276 | 382.397 | -0.49% |
| O投影 | 24 | 13.816 | 13.798 | 331.581 | 331.141 | -0.13% |
| MLP gate/up合计 | 48 | 50.965 | 50.871 | 2446.336 | 2441.795 | -0.19% |
| MLP down | 24 | 52.574 | 52.530 | 1261.775 | 1260.722 | -0.08% |
| 输出头 | 1 | 2.685 | 2.588 | 2.685 | 2.588 | -3.64% |
| Attention kernel | 24 | 34.893 | 34.962 | 837.441 | 839.096 | +0.20% |
| AICPU planner | 24 | 30.691 | 30.727 | 736.593 | 737.445 | +0.12% |
| metadata清零 | 24 | 2.470 | 0.386 | 59.286 | 9.267 | -84.37% |
| AddRmsNorm | 47 | 8.724 | 8.118 | 410.018 | 381.528 | -6.95% |
| RmsNorm | 2 | 5.838 | 5.897 | 11.676 | 11.794 | +1.01% |
| RoPE | 24 | 8.534 | 8.144 | 204.826 | 195.445 | -4.58% |
| Swish | 24 | 19.431 | 19.272 | 466.335 | 462.526 | -0.82% |
| Mul | 24 | 15.063 | 16.823 | 361.517 | 403.742 | +11.68% |
| GatherV2 | 4 | 4.866 | 4.659 | 19.465 | 18.637 | -4.26% |
| Add | 1 | 7.132 | 7.242 | 7.132 | 7.242 | +1.54% |
| Sub | 1 | 2.552 | 2.455 | 2.552 | 2.455 | -3.79% |
| bounds Cast | 1 | 2.174 | 1.513 | 2.174 | 1.513 | -30.41% |
| Slice / StridedSliceV2 | 1 | 3.469 | 3.200 | 3.469 | 3.200 | -7.73% |

| 指标 | 历史 ms | 当前 ms |
| --- | --- | --- |
| 全部MatMul | 4.740 | 4.732 |
| Attention kernel+planner+清零 | 1.633 | 1.586 |
| 全部计算/清零task累计 | 7.862 | 7.806 |
| 设备图时间均值 | 8.227 | 8.359 |
| 设备图时间中位数 | 7.923 | 7.945 |
| 图内未被计算/清零task覆盖区间 | 0.365 | 0.553 |

planner的完整Task Duration仍约30.7us，独立AICPU子明细的Task_time为15.503us
（历史15.525us）；这是同一任务的不同视图，不重复相加。
清零从历史MEMCPY_ASYNC 2.470us变为当前MEMSET_ASYNC 0.386us，每图约省50us；
Attention计算kernel基本不变，169个MatMul累计时间也基本不变。
设备图均值仍包含测量段内长步骤；中位数7.945ms与历史7.923ms接近，不移除长步骤。

### 默认并发下的算子对比：中位输入、16|32、4实例

与历史相同，先每实例预热10次，再共测100次；按Model ID各剔除最前面的10个预热步骤。

| 算子 | 历史单次 us | 当前单次 us | 变化 |
| --- | --- | --- | --- |
| Q投影 | 64.903 | 65.434 | +0.82% |
| K/V投影合计 | 37.976 | 38.742 | +2.02% |
| O投影 | 66.118 | 66.443 | +0.49% |
| MLP gate/up合计 | 146.872 | 144.438 | -1.66% |
| MLP down | 158.238 | 153.796 | -2.81% |
| Attention kernel | 94.833 | 95.797 | +1.02% |
| AICPU planner | 38.226 | 37.671 | -1.45% |
| metadata清零 | 2.581 | 0.415 | -83.92% |

并发设备图均值 23.797ms → 23.598ms；这是固定输入、带采集开销的短窗口，不能替代三轮普通吞吐结果。

### 辅流是否隐藏了planner开销

采集确认单实例主/辅流为51/52；四实例分别为50/54、51/55、53/56、58/57。
每个稳态图均有24个Attention kernel、24个planner和24次清零；主/辅流不同。
每图另外存在48次EVENT RECORD、48次EVENT WAIT和48次EVENT_RESET。

在保留的全部130个稳态图中，**planner与同一模型主流计算的时间交集均为0**。
当前依赖仍为：

```text
主流：Q/K/V与RoPE完成 → record(input_ready) → wait(metadata_ready) / reset → Attention
辅流：                     wait(input_ready) / reset → 清零 → planner → record(metadata_ready)
```

因此本次实现改变了任务所在stream，但没有把本模型的planner提前到QKV/MLP计算期间。
四实例之间可以交错执行，这与同一模型内隐藏planner耗时是两回事。
24层仍分别规划相同bounds，没有减少planner调用数。

同步event独占、未与计算/清零task重叠的区间，单实例每图约42.325us，四实例每图约52.790us。
EVENT WAIT会与辅流planner重叠，不能将所有event任务时长与planner相加。
图内未覆盖区间的跨版本差值还包含调度间隔和长步骤，不能全部归因于event自身执行时间。
结合普通性能、算子耗时和时间线，当前结论是：**辅流已生效，但尚未得到明确性能收益**。
后续若追求吞吐改善，应验证提前规划或跨层复用metadata；单纯换流不会消除当前依赖等待。

### 采集与原始记录

| 采集 | AIC/AIV | 实例 | 保留稳态图 | 剔除预热 | 业务QPS（带采集） | P99 ms（带采集） | run_id |
| --- | --- | --- | --- | --- | --- | --- | --- |
| profile-median-full-i1 | 32/64 | 1 | 30 | 10 | 1812.935 | 20.706 | 20261021T133258-89638623 |
| profile-median-opt-concurrent | 16/32 | 4 | 100 | 40 | 2637.609 | 52.173 | 20261021T133346-011fb664 |

逐算子主表使用完整的task_time/op_summary窗口：单实例720个planner、并发2400个planner均已计入。
并发aicpu子明细在部分Task ID超过65535后，其Stream/Task ID关联存在不一致，
仅2082/2400条能按原始键关联，因此未用该子明细的局部均值代替全量Task Duration。
并发稳态窗口内GatherV2统计数为350次（历史400次），未将该类纳入并发逐算子加速比表；
单实例逐算子表各类别调用数一致。
单实例aicpu子明细720/720完整关联。

原始result.json与逐调用requests.jsonl位于表中对应的 `work/qwen2.5-0.5b/results/<run_id>/`。
采集原始PROF数据、导出JSONL、环境摘要、请求序列核对和分析脚本位于
`work/attached-stream-benchmark-20261021/` 及相应结果目录；结果表保存在本附录，不另保留CSV文件。
默认模型YAML、输入池、AIR和已安装算子在本轮测量期间均未修改。

## 附录G：A5 16|32、batch16、单实例重新采集（2026-10-22）

本次沿用附录F的 `profile-median-opt-concurrent.yaml` profiling 配置，仅将 `benchmark.instances` 从4改为1；限核、输入、预热、正式调用数、seed和采集选项均保持一致。未实现提前planner或跨层复用，本次仍为现有辅流版本。

| 项目 | 本次配置 / 结果 |
| --- | --- |
| 环境 | 850-a5 / hejun-cann910-stream，CANN 9.3.0，逻辑卡1 |
| AIC/AIV / batch / 实例 | 16/32 / 16 / 1 |
| 固定输入 | case-0020，T=2041，prefix=24，16个业务样本 |
| 输入摘要 | `83c64fad54daa4c6968b38f49a8df917fbb167f85bf39c398cd59aa51cd95269`（仅该固定case） |
| 预热 / 保留正式图 | 10 / 100 |
| seed / AIC metrics | 42 / PipeUtilization |
| 业务QPS（开启profiling） | 1349.886 |
| 整batch平均 / P50 / P99延迟 ms（开启profiling） | 11.847 / 11.765 / 20.469 |
| 设备图平均 / 中位数 ms | 11.651 / 11.561 |
| 失败调用 | 0 |
| run_id | `20261022T082226-364fc732` |

上述QPS和P99来自固定输入、开启profiling的100次采集，包含采集开销，不替代无profiling的100-case输入池性能结论。P99沿用仓库整batch H2D、执行和同步口径。

### 算子耗时

下表均为剔除预热后的单次算子平均耗时；前两列比较相同限核下的并发影响，最后一列保留此前满核单实例作为背景。满核结果与本次核预算不同，不直接用于判断版本收益。

| 算子 / 阶段 | 每图调用数 | 本次16/32单实例 μs | 附录F 16/32四实例 μs | 附录F 32/64单实例 μs |
| --- | --- | --- | --- | --- |
| Q projection | 24 | 21.533 | 65.434 | 13.053 |
| KV projections | 48 | 6.750 | 38.742 | 7.967 |
| O projection | 24 | 22.507 | 66.443 | 13.798 |
| MLP gate/up | 48 | 89.106 | 144.438 | 50.871 |
| MLP down | 24 | 99.343 | 153.796 | 52.530 |
| Output head | 1 | 2.497 | 25.869 | 2.588 |
| Attention kernel | 24 | 34.831 | 95.797 | 34.962 |
| Attention planner (AICPU) | 24 | 30.964 | 37.671 | 30.727 |
| Attention metadata memset | 24 | 0.386 | 0.415 | 0.386 |
| AddRmsNorm | 47 | 9.753 | 17.476 | 8.118 |
| RmsNorm | 2 | 6.129 | 10.714 | 5.897 |
| ApplyRotaryPosEmb | 24 | 12.110 | 19.794 | 8.144 |
| Swish | 24 | 20.331 | 43.539 | 19.272 |
| Mul | 24 | 22.189 | 54.943 | 16.823 |
| GatherV2 | 4 | 4.928 | —（统计不完整） | 4.659 |
| Add | 1 | 9.319 | 16.272 | 7.242 |
| Sub | 1 | 2.354 | 3.170 | 2.455 |
| Cast | 1 | 1.625 | 4.090 | 1.513 |
| StridedSliceV2 | 1 | 3.379 | 5.390 | 3.200 |

每图169次MatMul累计平均 8.045ms；Attention计算、planner与清零累计平均 1.588ms，其中planner为 0.743ms。每图Attention/planner/清零各24次，100个正式图对应2400次planner，AICPU子明细2400/2400完整关联。

### 主辅流与一致性核对

本次主流为60，辅流为59。100个正式图的planner与同模型主流计算交集均为0。每图仍有48次EVENT RECORD、48次EVENT WAIT和48次EVENT_RESET；event独占且未与计算/清零重叠的区间平均为46.633μs。

这与附录F结论一致：planner已经运行在辅流，但当前QKV完成后发射、主流立即等待的依赖关系仍然存在。限核16/32的单实例基线已补齐，可用于后续提前规划方案的同配置对比。

已核对临时YAML相对上次profiling配置仅修改实例数；正式模型YAML、AIR、已安装算子库及上次profiling配置的采集前后SHA256均一致，算子库也与附录F使用的文件摘要一致。采集前逻辑卡1无运行进程。

### 原始文件位置

- profiling根目录：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T082226-364fc732/bench-profiling`。
- 原始PROF目录：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T082226-364fc732/bench-profiling/PROF_000001_20261022082233254_00000704CJAIAINQ`。
- 结果与逐调用记录：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T082226-364fc732` 中的 `result.json`、`requests.jsonl`。
- 配置、环境快照、逐算子与stream分析：`/home/h00909914/ascend-ge-adapters/work/attached-stream-single-instance-20261022`。

保留原始PROF、timeline JSON与6张导出JSONL表；转换索引记录列名、行数和SHA256，不保留CSV文件。

## 附录H：int32 bounds 输入与提前 planner（2026-10-22）

本次在850-a5的hejun-cann910-stream中，将A5 prefix_a5的bounds从模型输入端改为int32，删除图内int64→int32 Cast。input_ids、position_ids及A3/FIA输入仍为int64；未新增token、前端算子调用或GE Pass，原prefix_attention调用签名保持不变。

### 输入就绪与内存依赖

运行器先完成同步H2D，再启动执行。最终100-case地址跟踪确认：2400次planner发射全部直接读取同一个外部bounds输入缓冲区；100次H2D的68字节内容逐项匹配case数据，均早于本轮planner发射，没有复制到另一个planner输入地址，编译图中Cast数为0。

| 核对项 | 最终结果 |
| --- | --- |
| bounds地址（本次进程） | `0x120000022000` |
| bounds dtype / shape | int32 / [17] |
| H2D次数 / 每次字节数 | 100 / 68 |
| planner次数 / metadata缓冲区数 | 2400 / 24 |
| bounds额外搬运 / Cast | 0 / 0 |
| 编译图bounds_preloaded=true节点数 | 24 |

静态GE图存在不同的输入绑定行为：固定shape校验中，外部H2D地址为0x120000117000，而录制的planner使用模型内部地址0x12004d200000。无条件删除入口等待时，第三次更换bounds内容出现错误。因此最终实现的bounds_preloaded属性默认false，保留静态图及图内生产bounds所需的主→辅流同步；TorchAir仅对带原生适配导出器绑定标记、直接Data输入且bounds/query有符号动态维度的路径启用提前规划。不能仅凭int32 dtype删除所有路径的等待。

当前Qwen实际使用RT2动态执行，Execute在运行时发射任务；不能用V1示例的HEAD录制行为推断24个planner全部在模型起点启动。优化消除的是QKV→planner的设备等待，实际提前量由下面的时间线数据确认。

RT2的原workspace由主流分配器管理，提前在辅流写入可能与其复用时序冲突。本次将metadata改成按ACL context、模型辅流及节点名独立持有的16 KiB缓冲区，每轮仍在辅流清零和规划。24层每实例有效metadata为384 KiB，不含分配器对齐开销。所有路径保留planner→kernel的record/wait/reset；提前路径每节点仅一个完成event。沿用显式清理接口，同时释放event与metadata；同一优化实例须完成上一轮后再执行下一轮。

### 正确性验证

| 检查 | 结果 |
| --- | --- |
| 编码与导出契约 | A5/A3/FIA各100个case，Torch编码与运行时数组逐项一致；A5 bounds=int32，其他输入保持原dtype；int32越界被拒绝 |
| 100-case端到端对比 | 12800个输出值与int32、保留入口等待的基线逐位一致，最大差值0 |
| case-0000独立FP32基线 | 16个业务样本全部通过；最小cosine=0.999997816254，最大relative L2=0.003342528654 |
| 静态图固定shape基础校验 | [RESULT] calls=3 checked=5505024 max_abs=0.000186502934 rmse=8.7945268e-06 status=PASS; [CLEANUP] status=PASS |
| 四动态图实例、同名节点、提前规划开启 | [RESULT] models=4 calls=12 checked=22020096 max_abs=0.000186502934 rmse=8.7945268e-06 status=PASS; [CLEANUP] status=PASS |

固定shape为T=2048、batch16、N1=14、N2=2、D=64、FP16，轮换Q/K/V数值及bounds内容；只同步执行流，不使用全设备同步掩盖缺失依赖。四实例测试采用动态图声明、固定实际shape，覆盖独立资源隔离。临时校验程序放在work目录，未增加仓库内多流测试文件。

### 单实例profiling对比

沿用附录G：逻辑卡1、16|32、batch16、单实例、固定case-0020（T=2041、prefix=24）、预热10次、正式100次、seed42、PipeUtilization。采集关闭地址跟踪和图dump，配置与附录G一致。输入池业务内容摘要保持21defc288cc9848d6f1f71f8a8c6d900327963431e57e77125274b5f5fc8c85b。

| 指标 | 附录G：原同步 | 本次：int32与提前规划 |
| --- | --- | --- |
| 业务样本QPS（带profiling） | 1349.886 | 1413.009（+4.68%） |
| 整batch平均延迟 ms | 11.847 | 11.310 |
| 整batch P99 ms | 20.469 | 29.382 |
| 设备图平均 / 中位数 ms | 11.651 / 11.561 | 11.071 / 10.883 |
| 各计算/清零task累计均值 ms | 11.451 | 11.449 |
| planner与同模型主流计算重叠 μs/图 | 0.000 | 708.755 |
| event独占覆盖 μs/图 | 46.633 | 1.780 |
| EVENT RECORD / WAIT / RESET，每图 | 48 / 48 / 48 | 24 / 24 / 24 |
| run_id | `20261022T082226-364fc732` | `20261022T101723-0aba6757` |

本次planner与本模型主流计算的平均交集为0.709ms/图，占planner累计任务耗时约95.2%。每图仍执行24次planner，尚未进行跨层复用。重叠区间不能再与主流计算时间直接相加；任务耗时累计并非图延迟。

上表为单次、固定输入、开启profiling的100次窗口，P99包含采集扰动；没有重测关闭profiling后的100-case普通吞吐，不将本表推广为生产QPS或P99收益。

### P99与长步骤

本次P99由首个正式调用req=0拉高：整batch 29.382ms，对应设备图28.854ms；其中计算/清零task覆盖10.991ms，未覆盖区间17.863ms，而正式窗口的未覆盖区间中位数为0.156ms。该图task累计11.616ms，接近全窗口均值11.449ms。P99确实高于附录G，不能声称尾延迟改善；现有数据只定位到图内长间隔，未将其直接归因为某个kernel或采集机制。

### 单次算子平均耗时

| 算子 / 阶段 | 原同步 μs | 本次 μs | 本次每图次数 |
| --- | --- | --- | --- |
| Q projection | 21.533 | 21.540 | 24 |
| KV projections | 6.750 | 6.761 | 48 |
| O projection | 22.507 | 22.486 | 24 |
| MLP gate/up | 89.106 | 89.102 | 48 |
| MLP down | 99.343 | 99.345 | 24 |
| Output head | 2.497 | 2.689 | 1 |
| Attention kernel | 34.831 | 34.971 | 24 |
| Attention planner (AICPU) | 30.964 | 31.024 | 24 |
| Attention metadata memset | 0.386 | 0.540 | 24 |
| AddRmsNorm | 9.753 | 9.892 | 47 |
| RmsNorm | 6.129 | 6.235 | 2 |
| ApplyRotaryPosEmb | 12.110 | 12.127 | 24 |
| Swish | 20.331 | 20.338 | 24 |
| Mul | 22.189 | 21.450 | 24 |
| GatherV2 | 4.928 | 5.157 | 4 |
| Add | 9.319 | 9.176 | 1 |
| Sub | 2.354 | 2.794 | 1 |
| Cast | 1.625 | — | 0 |
| StridedSliceV2 | 3.379 | 3.426 | 1 |

### 采集尝试与共享环境

| 尝试 | 逻辑卡 | 结果 | run_id |
| --- | --- | --- | --- |
| 1 | 1 | 启动失败：507899，驱动SQ/CQ远端分配失败，无有效样本 | `20261022T100615-b98318a4` |
| 2 | 1 | 成功：100次正式调用，0错误 | `20261022T101723-0aba6757` |

第二次以独立进程在原指定1号卡成功采集，没有换卡。重试前后该卡已无xllm进程，SMI利用率为0；首次驱动错误保留为失败记录，不计入性能。原始环境快照保存在before.txt/after.txt。


### 原始数据与复现配置

- profiling目录：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T101723-0aba6757/bench-profiling`。
- 原始PROF：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T101723-0aba6757/bench-profiling/PROF_000001_20261022101735054_00032704NKCCJIFN`。
- 结果和逐调用记录：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T101723-0aba6757` 中的result.json、requests.jsonl。
- 配置、输入地址审计、正确性、逐算子和主辅流分析：`/home/h00909914/ascend-ge-adapters/work/bounds-int32-early-planner-20261022`。

保留原始PROF、timeline JSON及导出JSONL表，不保留CSV。新profiling采集期间正式模型YAML、AIR、算子库和来源profiling配置的摘要均保持不变；算子与AIR相对附录G为本次修改后的版本。代码尚未提交，工作树diff和文件SHA256记录在同目录的change-audit.json。


## 附录 I：空闲卡复测P99（2026-10-22）

在附录H代码和产物不变的情况下，再次于逻辑卡1执行相同profiling：16|32、batch16、单实例、固定case-0020（T=2041、prefix=24）、预热10次、正式100次、seed42、PipeUtilization。采集前后npu-smi显示1号卡利用率为0且无运行进程；采集前所有卡也均无运行进程。正式模型YAML、AIR、算子库及来源配置的SHA256与附录H成功采集结束时一致，采集期间也未变化。没有修改预热次数、剔除首个正式调用或调整仓库P99算法。

| 指标 | 附录H | 空闲卡复测 |
| --- | --- | --- |
| 业务样本QPS（带profiling） | 1413.009 | 1426.133 |
| 整batch平均延迟 ms | 11.310 | 11.213 |
| 整batch P50 ms | 11.132 | 11.121 |
| 整batch P99 / 最大延迟 ms | 29.382 | 19.786 |
| 最慢正式调用 | req=0（iteration=11） | req=0（iteration=11） |
| 最慢调用的设备图 ms | 28.854 | 19.452 |
| 最慢图计算/清零task覆盖 ms | 10.991 | 10.940 |
| 最慢图未被上述task覆盖的区间 ms | 17.863 | 8.513 |
| 设备图平均 / 中位数 ms | 11.071 / 10.883 | 10.977 / 10.883 |
| 各计算/清零task累计均值 ms | 11.449 | 11.453 |
| planner与同模型主流计算重叠 μs/图 | 708.755 | 705.420 |
| event独占覆盖 μs/图 | 1.780 | 1.860 |
| EVENT RECORD / WAIT / RESET，每图 | 24 / 24 / 24 | 24 / 24 / 24 |
| run_id | `20261022T101723-0aba6757` | `20261022T105624-5644191b` |

100次正式调用全部成功。首个正式调用仍是明显离群点：H2D=0.089ms、execute+sync=19.695ms、整batch=19.786ms；其余99次延迟范围11.027–11.295ms，均值11.126ms（仅用于解释分布，未从正式统计中删除）。本窗口仓库P99等于最大值。最慢图task累计11.616ms，接近全窗口均值11.453ms；正式窗口未覆盖区间中位数仅0.147ms。

结论：空闲卡复测后P99从29.382ms降至19.786ms，但首个正式调用的长间隔仍然复现，不能将原异常完全归因于其他任务占卡。算子平均耗时基本不变，planner仍与主流重叠约0.705ms/图；本次没有发现planner计算累计耗时足以解释该长间隔。当前证据定位到首个正式调用的图内未覆盖区间，尚不足以确定具体运行时或profiling根因。此结果仍属于开启profiling的固定输入窗口，不替代普通性能统计。

### 单次算子耗时复核

| 算子 / 阶段 | 附录H μs | 空闲卡复测 μs |
| --- | --- | --- |
| Add | 9.176 | 9.194 |
| AddRmsNorm | 9.892 | 9.938 |
| ApplyRotaryPosEmb | 12.127 | 12.134 |
| Attention kernel | 34.971 | 34.991 |
| Attention metadata memset | 0.540 | 0.537 |
| Attention planner (AICPU) | 31.024 | 30.871 |
| GatherV2 | 5.157 | 5.168 |
| KV projections | 6.761 | 6.761 |
| MLP down | 99.345 | 99.350 |
| MLP gate/up | 89.102 | 89.134 |
| Mul | 21.450 | 21.508 |
| O projection | 22.486 | 22.513 |
| Output head | 2.689 | 2.421 |
| Q projection | 21.540 | 21.546 |
| RmsNorm | 6.235 | 6.206 |
| StridedSliceV2 | 3.426 | 3.192 |
| Sub | 2.794 | 2.725 |
| Swish | 20.338 | 20.404 |

### 原始数据

- 配置、环境快照、摘要及分析：`/home/h00909914/ascend-ge-adapters/work/bounds-int32-p99-recheck-20261022`。
- 结果与逐调用记录：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T105624-5644191b` 中的result.json、requests.jsonl。
- profiling：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T105624-5644191b/bench-profiling/PROF_000001_20261022105633387_00045733DQHECHQA`。
- 导出6张表均转存JSONL并保留列名与SHA256索引，不保留CSV。模型、算子与配置均未修改，本次仅新增复测记录。


## 附录 J：关闭profiling的单实例性能复测（2026-10-22）

按性能测试口径关闭profiling，继续使用逻辑卡1、16|32、batch16、单实例、固定case-0020（T=2041、prefix=24）、预热10次、正式100次、seed42。与附录I的唯一配置差异为observability.profiling=false；执行时再次显式传入profiling=False。result.json中配置profiling=false且performance.profiling_enabled=false，无profiling输出目录。采集前后1号卡无运行进程；AIR、算子库、正式模型YAML及来源配置SHA256与附录I一致，测试期间未变。

| 指标 | 附录I：开启profiling | 本次：关闭profiling |
| --- | --- | --- |
| 业务样本QPS | 1426.133 | 1444.377 |
| 整batch平均延迟 ms | 11.213 | 11.027 |
| 整batch P50 ms | 11.121 | 10.844 |
| 整batch P99 / 最大延迟 ms | 19.786 | 30.278 |
| 整batch最小延迟 ms | 11.027 | 10.750 |
| 正式调用数 / 错误数 | 100 / 0 | 100 / 0 |
| 最慢正式调用 | req=0 | req=0 |
| run_id | `20261022T105624-5644191b` | `20261022T110204-03e09a53` |

关闭profiling后，最慢调用仍为req=0：H2D=0.081ms、execute+sync=30.196ms、整batch=30.278ms。其余99次延迟为10.750–10.964ms，均值10.833ms；该分布说明仅用于定位离群点，所有100次均保留在正式QPS和P99统计中。本窗口仓库P99等于最大值。

结论：不启用profiling时仍有首个正式调用的延迟尖峰，不能将该现象完全归因于profiling。由于本次没有设备时间线，不将附录I测得的图内间隔直接当作本轮尖峰原因。该结果是固定输入的单实例100次性能窗口；没有重跑4实例或100-case长窗口，不将其推广到那些配置。

### 原始数据

- 配置、环境快照及摘要：`/home/h00909914/ascend-ge-adapters/work/bounds-int32-performance-no-profile-20261022`。
- 结果、bench.log与逐调用记录：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T110204-03e09a53` 中的result.json、bench.log、requests.jsonl。
- 未开启profiling，不产生PROF目录；逐调用CSV已转换为JSONL后删除，没有保留CSV。


## 附录 K：空闲卡3单实例profiling（2026-10-22）

按用户要求使用空闲卡重新采集：逻辑卡3、16|32、batch16、单实例、固定case-0020（T=2041、prefix=24）、预热10次、正式100次、seed42、PipeUtilization。启动检查、采集前及采集后SMI快照均显示3号卡无运行进程；启动检查时计算利用率为0。result.json核实执行device=3。正式模型YAML、AIR、算子库和来源配置SHA256与附录I一致且本轮未变化。所有100次正式调用成功。

本轮用于设备时间线与算子分析，开启profiling；普通性能结果仍参见关闭profiling的附录J。本次未重跑普通性能。

| 指标 | 卡3本次profiling |
| --- | --- |
| 设备图平均 / 中位数 ms | 11.087 / 10.876 |
| 各计算/清零task累计均值 ms | 11.445 |
| planner与本模型主流重叠 μs/图 | 707.805 |
| event独占覆盖 μs/图 | 1.863 |
| EVENT RECORD / WAIT / RESET，每图 | 24 / 24 / 24 |
| 带profiling的业务样本QPS | 1413.554 |
| 带profiling的整batch平均 / P50 ms | 11.306 / 11.089 |
| 带profiling的整batch P99 / 最大延迟 ms | 29.552 |
| run_id | `20261022T111827-95504de4` |

首个正式调用req=0（iteration=11）仍是最慢调用：整batch29.552ms、execute+sync=29.479ms，设备图28.940ms。该图计算/清零task累计11.607ms，任务覆盖10.953ms，未覆盖区间17.986ms；正式窗口未覆盖区间中位数0.155ms。其余99次整batch延迟10.974–12.177ms。全部调用均计入正式统计，没有剔除离群点。

在另一张空闲卡上仍观察到首个正式调用的长间隔；planner与主流继续重叠。当前时间线只能定位到任务未覆盖区间，不直接断言具体运行时根因。

### 单次算子平均耗时

| 算子 / 阶段 | 附录I卡1 μs | 本次卡3 μs | 本次每图次数 |
| --- | --- | --- | --- |
| Add | 9.194 | 9.267 | 1 |
| AddRmsNorm | 9.938 | 9.875 | 47 |
| ApplyRotaryPosEmb | 12.134 | 12.122 | 24 |
| Attention kernel | 34.991 | 34.934 | 24 |
| Attention metadata memset | 0.537 | 0.537 | 24 |
| Attention planner (AICPU) | 30.871 | 31.046 | 24 |
| GatherV2 | 5.168 | 5.177 | 4 |
| KV projections | 6.761 | 6.755 | 48 |
| MLP down | 99.350 | 99.320 | 24 |
| MLP gate/up | 89.134 | 89.106 | 48 |
| Mul | 21.508 | 21.461 | 24 |
| O projection | 22.513 | 22.505 | 24 |
| Output head | 2.421 | 2.445 | 1 |
| Q projection | 21.546 | 21.546 | 24 |
| RmsNorm | 6.206 | 6.213 | 2 |
| StridedSliceV2 | 3.192 | 3.230 | 1 |
| Sub | 2.725 | 2.728 | 1 |
| Swish | 20.404 | 20.224 | 24 |

### 路径与采集状态

- profiling目录：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T111827-95504de4/bench-profiling`。
- 原始PROF：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T111827-95504de4/bench-profiling/PROF_000001_20261022111835639_00070132KCPMPJKD`。
- 结果与逐调用记录：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T111827-95504de4` 中的result.json、requests.jsonl。
- 配置、SMI快照、逐算子及主辅流分析：`/home/h00909914/ascend-ge-adapters/work/bounds-int32-idle-device3-profile-20261022-111751`。
- 之前1号卡采集启动时有xllm驻留，用户要求切换空闲卡后，该任务已自然完成，记录为superseded（run_id：20261022T111653-943f3694），不用于本轮对比，未停止其他用户进程。记录保存在`/home/h00909914/ascend-ge-adapters/work/bounds-int32-single-instance-profile-20261022-111620`。
- 6张导出表转换为JSONL并记录列名与SHA256，不保留CSV。代码、模型和正式配置未修改。


### 流58注册任务与流59首次planner长耗时

进一步关联msprof时间线的Host API、connection_id及aicpu明细后，可将原先的“任务未覆盖区间”细化为明显的Host首次调用开销。以下为同一份卡3采集，无新增运行或代码修改。

流58不是本算子的planner辅流；实际有两个KERNEL_AICPU任务（Task ID=16、4457），分别为183883.578μs和396.366μs，且op_summary没有对应的模型算子。它们分别落在两个Host线程首次执行aclrtRegisterCpuFunc的区间内，并通过connection_id关联到当次PrefixAttention的Node@launch。因此证据指向注册/加载链路触发的运行时内部AICPU任务，不能将其计为额外一层planner；profiling没有暴露该内部任务的具体函数名，不能仅凭KERNEL_AICPU标签断言内部实现细节。官方接口说明确认aclrtRegisterCpuFunc用于注册AI CPU算子信息并获取函数句柄：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/appdevgapi/aclcppdevg_03_1897.html 。

| 阶段 | Host线程 | aclrtRegisterCpuFunc耗时 | 流58内部任务 | 流59第一个planner |
| --- | --- | --- | --- | --- |
| 第1次预热 | 70132 | 193.005ms | 183.884ms | 7.914ms（task=17） |
| 第1次正式调用（req=0） | 76663 | 9.400ms | 0.396ms | 0.189ms（task=4458） |
| 同线程后续调用 | 76663 | 中位数0.590μs | 无新增KERNEL_AICPU | 稳态planner中位数30.574μs |

流59的_Z28_prefix_attn_metadata_kernel23_PrefixAttnMetadataArgs_interface就是DSL生成的metadata planner。kernel_launcher.cc通过PREFIX_ATTN_AICPU_LAUNCH调用_prefix_attn_metadata_kernel_do；生成的codegen/aicpu/_prefix_attn_metadata_kernel_shim.asc在该函数中使用_prefix_attn_metadata_kernel<<<1, 0, stream>>>(s)发射，注册/launch细节由编译器生成。

最显眼的7.914ms发生在第1次预热，不是正式窗口P99的那个planner。该任务的aicpu明细为Dispatch_time=5946μs、Task_time=1812.242μs、Total_time=7844.781μs；后续稳态对应字段的中位数为5μs、15.354μs、22.364μs。时间线任务跨度与aicpu明细采用不同边界，不直接混加；Compute_time和Memcpy_time均未填有效分解，不能将Task_time等同于纯算法计算时间。该数据表明第一次的额外耗时主要落在调度/首次执行路径，固定输入并没有导致首层规划工作量扩大。第二层已降至96.883μs，第三层38.059μs，随后约30μs。

正式首调用的planner仅188.862μs，aicpu Task_time=17.922μs，因此29.552ms的端到端延迟并不是metadata算法计算了十几毫秒。在该次时间线中，Host的aclnnAddmmGetWorkspaceSize首次调用为5.200ms，aclrtRegisterCpuFunc首次调用为9.400ms；二者各自位于约7.284ms和9.891ms的设备计算空闲区间。API耗时和任务/空闲区间有包含、并行关系，不相加为总耗时。

runtime/bench.cpp:133–164在调用BenchPool的线程中完成预热，而180–185行之后才创建正式std::thread并调用threadEnter。时间线也确实显示预热使用70132、正式测量使用76663；两线程分别有240次和2400次注册API调用。由此判断，当前预热覆盖了shape及预热线程，却没有覆盖正式工作线程首次使用的ACL/runtime路径，是首个正式调用尖峰的重要原因。不能从这份trace进一步证明注册缓存一定采用线程局部存储，或把全部17.987ms归给单一注册API。

建议将每实例预热放到正式执行的同一个持久工作线程中，完成threadEnter、输入与各shape预热后统一进入计时阶段。保持正式样本全量统计，不以删除首样本代替预热修复。是否彻底消除尖峰仍需修改后关闭profiling重测确认；本轮仅分析，未改benchmark代码。

提取证据：上述实验目录中的aicpu-cold-start-analysis.json，含两个线程首次注册、planner样例及长空闲区间。


## 附录 L：同工作线程预热修复与关闭profiling性能验证（2026-10-22）

### 修改内容与计时口径

runtime/bench.cpp的BenchPool原先在调用线程预热，之后才创建正式工作线程。现改为每个实例先进入自己的工作线程并调用threadEnter，在该线程完成预热后继续执行正式请求；所有工作线程都预热成功后，由协调线程统一开始计时并放行。工作线程在正式请求循环中不再访问这个启动锁。线程创建失败、threadEnter抛异常或任一实例预热失败时，都唤醒等待线程并回收资源，不开始正式测量。

预先分配每实例的预热序列，保持原有每个distinct shape至少执行一次的规则、随机补齐请求的分配顺序、seed及正式请求序列；warmup=0时也仍保留原有的shape覆盖规则。本轮固定输入只有一种shape，仍为每轮预热10次、正式100次，没有额外丢弃首个正式样本。多实例的预热现在分别在各工作线程执行，而非主线程串行执行。

正式QPS的墙钟窗口现在从全部工作线程预热完成后开始，排除了线程创建、threadEnter及预热；原版窗口还包含工作线程创建和首次进入的开销。整batch延迟仍为H2D、描述更新、execute及同步，P99计算公式不变。QPS差异不能全部归为算子加速。本轮未修改模型、attention实现、AIR、算子库及正式模型YAML。

### 验证与环境

- 已重新编译ge_runtime与runtime_contracts，runtime_contracts通过。
- 扩展已有tests/runtime_contracts.cpp，检查预热与正式执行始终使用实例的同一工作线程、慢实例预热完成前不开始测量、预热失败及threadEnter异常能退出并释放资源；原有采样复现、请求分配、部分测量失败及统计检查继续通过。
- 卡3当时存在其他进程，因此使用确认空闲的逻辑卡1。三轮各自启动检查、运行前和结束后SMI快照均显示1号卡无进程；没有停止其他用户任务。
- A5、16|32、batch16、单实例、固定case-0020（T=2041、prefix=24）、seed42；每轮独立启动进程，预热10次、正式100次，连续3轮。
- 三轮result.json均核实profiling=false、performance.profiling_enabled=false，无PROF或bench-profiling目录。AIR、算子库、正式模型YAML、来源配置及固定输入池摘要与附录J一致。新运行时二进制及源码在三轮期间保持不变。

### 全量性能记录

下表QPS均为业务样本/秒；所有正式请求都纳入统计。在每轮100次的窗口中，仓库P99等于最大延迟。附录J与本轮均使用逻辑卡1，关闭profiling。

| 轮次 | QPS | 平均延迟 ms | P50 ms | P99 ms | 首个正式请求 ms | 最慢req（从0计数） | 错误数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 附录J：修改前 | 1444.377 | 11.027 | 10.844 | 30.278 | 30.278 | 0 | 0 |
| 修改后第1轮 | 1403.194 | 11.395 | 11.189 | 17.319 | 10.972 | 86 | 0 |
| 修改后第2轮 | 1441.205 | 11.092 | 11.089 | 13.274 | 11.027 | 50 | 0 |
| 修改后第3轮 | 1478.785 | 10.813 | 10.811 | 11.369 | 10.813 | 75 | 0 |

三轮共300次正式请求均成功，first_hit均为0，没有丢弃任何样本。首个正式请求为10.813–11.027ms，均接近各轮稳态延迟，原先约30ms且固定出现在req=0的尖峰在三次独立启动中均未复现，支持“正式工作线程未得到预热”这一判断。

但不能宣称所有P99抖动都已消失：第1轮在req=86出现17.319ms，第2轮在req=50出现13.274ms，第3轮P99为11.369ms（req=75）。前两轮最慢请求的execute+sync分别为17.210ms和13.175ms，H2D分别为0.106ms和0.097ms；这些尖峰已不集中在首个请求。关闭profiling的本轮没有设备时间线，尚不能确定剩余波动的具体原因。三轮QPS中位数1441.205，保留全部轮次，不以第3轮替代其他结果。

本次结论限定于固定输入、单实例配置；没有重新测试4实例或100-case混合输入池。

### 原始数据与复现

- 第1轮：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T114043-6cec779f`，含result.json、requests.jsonl、bench.log；run_id=`20261022T114043-6cec779f`。
- 第2轮：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T114229-b66fc9be`，含result.json、requests.jsonl、bench.log；run_id=`20261022T114229-b66fc9be`。
- 第3轮：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T114357-443f8dc0`，含result.json、requests.jsonl、bench.log；run_id=`20261022T114357-443f8dc0`。

配置、运行脚本、SMI快照、测试日志、修改前后源码、diff、SHA256及全部轮次摘要保存在`/home/h00909914/ascend-ge-adapters/work/same-worker-warmup-20261022-113718`。三轮各自的performance-median-16-i1.yaml位于repeat-1、repeat-2、repeat-3子目录。逐调用CSV转换为JSONL后删除；测试生成的临时CSV也已清理，未保留CSV。

本次修改了runtime/bench.cpp、runtime/bench.h及已有测试tests/runtime_contracts.cpp，并追加本文档；此前attention相关未提交修改予以保留，未提交或推送。


## 附录 M：同工作线程预热后的单实例profiling（2026-10-22）

使用附录L修改后的运行时重新采集：启动时确认空闲的逻辑卡1、16|32、batch16、单实例、固定case-0020（T=2041、prefix=24）、预热10次、正式100次、seed42、PipeUtilization。三份SMI快照（启动检查、运行前、运行后）均显示卡1无运行进程；运行前利用率为0。AIR、算子库、正式模型YAML、来源配置、运行时二进制及bench.cpp/bench.h的SHA256与附录L第3轮结束时一致，本轮期间也未改变。所有100次正式调用成功。

本轮开启profiling，用于分析时间线；普通性能结果仍以附录L关闭profiling的三轮为准。

### 同线程预热的时间线证据

| 检查 | 本次结果 |
| --- | --- |
| aclrtRegisterCpuFunc调用线程 | 预热和正式全部在Host线程92084，共2640次 |
| 流58 KERNEL_AICPU | 全部仅1次，121.932ms，发生于第1次预热 |
| 正式窗口流58 KERNEL_AICPU | 0次 |
| 第1次预热的注册接口 | 129.568ms |
| 首个正式调用的第1次注册接口 | 1.020μs |
| 正式窗口注册接口中位数 / 最大值 | 0.760 / 34.590μs |
| 流59第1次预热的metadata planner | 6.290ms |
| 流59正式首调用第1层planner | 31.184μs |
| 正式首调用整batch / 设备图 | 11.684 / 11.374ms |

修改前附录K中，预热与正式测量分属70132和76663两个Host线程，正式首调用出现9.400ms注册接口及额外的流58内部任务。本轮没有该现象：冷启动的注册和首个metadata长任务均留在预热阶段，正式首调用的planner已回到约31μs。此证据支持附录L观察到的首请求尖峰修复效果。

### 正式窗口摘要与剩余波动

| 指标 | 本次带profiling窗口 |
| --- | --- |
| 业务样本QPS | 1161.060 |
| 整batch平均 / P50 ms | 13.771 / 12.775 |
| 整batch P99 / 最大值 ms | 32.566 |
| 最慢正式调用 | req=47，iteration=58 |
| 设备图平均 / 中位数 ms | 13.527 / 12.521 |
| 各计算/清零task累计均值 ms | 11.438 |
| 计算/清零task未覆盖区间均值 ms | 2.619 |
| planner与同模型主流重叠 μs/图 | 518.905 |
| event独占覆盖 μs/图 | 6.335 |
| EVENT RECORD / WAIT / RESET，每图 | 24 / 24 / 24 |

本次没有删除任何正式样本，100次窗口中仓库P99仍等于最大值。最慢的是req=47：H2D=0.083ms、execute+sync=32.480ms、整batch32.566ms；设备图32.375ms，其中计算/清零task累计11.459ms、任务覆盖11.300ms、未覆盖区间21.076ms。该调用的24次planner累计0.756ms，第1层planner为31.423μs，没有正式阶段重新注册/加载的证据。

时间线显示该图存在多段约0.5–1.5ms的计算空闲区间，同时有若干Host launch区间拉长；不能将总延迟解释为某个metadata kernel计算了二十多毫秒。全窗口task累计均值与附录K接近（11.445ms→11.438ms），但图未覆盖区间增加，planner与主流重叠也从707.805μs/图降至518.905μs/图。该变化说明本轮执行/发射节奏有较大波动，尚不足以从这份trace确定是主机调度、运行时还是采集开销造成；不直接归因于其他任务，也不将带profiling的QPS/P99替代普通性能结果。

附录K使用卡3，本次使用卡1，因此上述历史数值用于定位现象，不视为严格同卡性能A/B。本次仅重新采集和分析，未改动附录L的代码。

### 单次算子平均耗时

| 算子 / 阶段 | 附录K μs | 本次 μs | 本次每图次数 |
| --- | --- | --- | --- |
| Add | 9.267 | 9.144 | 1 |
| AddRmsNorm | 9.875 | 9.903 | 47 |
| ApplyRotaryPosEmb | 12.122 | 12.105 | 24 |
| Attention kernel | 34.934 | 34.888 | 24 |
| Attention metadata memset | 0.537 | 0.524 | 24 |
| Attention planner (AICPU) | 31.046 | 30.924 | 24 |
| GatherV2 | 5.177 | 5.107 | 4 |
| KV projections | 6.755 | 6.752 | 48 |
| MLP down | 99.320 | 99.331 | 24 |
| MLP gate/up | 89.106 | 89.090 | 48 |
| Mul | 21.461 | 21.398 | 24 |
| O projection | 22.505 | 22.503 | 24 |
| Output head | 2.445 | 2.569 | 1 |
| Q projection | 21.546 | 21.536 | 24 |
| RmsNorm | 6.213 | 6.213 | 2 |
| StridedSliceV2 | 3.230 | 3.199 | 1 |
| Sub | 2.728 | 2.755 | 1 |
| Swish | 20.224 | 20.174 | 24 |

### 原始数据

- run_id：`20261022T114940-d157480a`。
- profiling目录：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T114940-d157480a/bench-profiling`。
- 原始PROF：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T114940-d157480a/bench-profiling/PROF_000001_20261022114950711_00085454JDLNPRBR`。
- 结果和逐调用数据：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T114940-d157480a` 中的result.json、requests.jsonl。
- 配置、环境快照、算子/主辅流分析及聚焦时间线证据：`/home/h00909914/ascend-ge-adapters/work/same-worker-profiling-20261022-114920`，其中focused-trace-analysis.json记录注册线程、流58任务、流59样例、最慢步骤及Host长任务。
- 保留原始PROF、timeline JSON、6张导出JSONL表及其列名/SHA256索引，不保留CSV。


## 附录 N：A5默认参数性能验证（2026-10-22）

按当前预设运行，关闭profiling：逻辑卡1、16|32、batch16、4实例，直接使用models/qwen2.5-0.5b/config/model.yaml；没有覆盖实例数、迭代次数或输入池。默认正式请求为1000次、seed42。使用附录L的同工作线程预热版本、int32 bounds及提前planner实现，当前模型与算子版本保持不变。运行前后卡1无运行进程，运行前利用率为0，没有停止其他用户任务。

默认输入目录为models/qwen2.5-0.5b/io/inputs，共100个变长case，逐项核对均为batch16；业务输入池摘要仍为21defc288cc9848d6f1f71f8a8c6d900327963431e57e77125274b5f5fc8c85b。结果包含89种shape，正式请求抽到了100个case。配置warmup=10是每实例最低次数；沿用仓库shape覆盖规则，每实例实际预热89次，4实例共356次，预热墙钟2.746s。全部实例预热成功后再开始正式计时。

| 指标 | 本次默认参数结果 |
| --- | --- |
| 业务样本QPS（样本/秒） | 2681.175 |
| 整batch吞吐（batch/秒） | 167.573 |
| 整batch平均延迟 ms | 23.800 |
| 整batch P50 ms | 23.679 |
| 整batch P99 ms | 26.749 |
| 整batch最大延迟 ms | 31.481 |
| 正式请求数 / 业务样本数 | 1000 / 16000 |
| 错误数 | 0 |
| 正式测量墙钟 ms | 5967.534 |
| profiling_enabled | false |
| run_id | `20261022T115820-b644b307` |

| 实例 | 正式请求数 | 平均延迟 ms | P99 ms | 最大延迟 ms |
| --- | --- | --- | --- | --- |
| instance_0 | 250 | 23.842 | 26.749 | 31.481 |
| instance_1 | 250 | 23.767 | 26.851 | 30.424 |
| instance_2 | 250 | 23.794 | 26.813 | 27.102 |
| instance_3 | 250 | 23.797 | 26.566 | 26.909 |

本次默认配置业务样本QPS为2681.175，聚合P99为26.749ms，4个实例的P99均低于60ms。没有删去首个请求或其他离群点，全部1000条记录的first_hit=0且错误数为0。该结果是当前默认1000次变长输入窗口的单次测量，不将其与固定case单实例profiling或其他迭代次数的历史结果作严格性能A/B。

result.json核实profiling=false、performance.profiling_enabled=false，无profiling目录。正式模型YAML、AIR、算子库、运行时二进制、bench.cpp/bench.h及适配器源码在本轮前后SHA256均一致。本次没有修改预设或代码。

### 原始数据

- 结果与逐调用记录：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T115820-b644b307` 中的result.json、requests.jsonl及bench.log。
- 默认YAML快照、运行脚本、SMI快照、摘要与前后SHA256：`/home/h00909914/ascend-ge-adapters/work/default-a5-performance-20261022-115756`。
- requests.csv已转换为JSONL后删除，不保留CSV。


## 附录 O：当前A5与A3吞吐差距及Attention重点分析（2026-10-22）

### 比较口径和新增采集

当前A5使用附录N的普通性能：16|32、batch16、4实例、默认100-case输入池、1000次请求、关闭profiling，QPS=2681.175。A3复用2026-10-21的推荐配置8|16、batch16、5实例，3×3000次普通测量中最高QPS=2180.259（seed44，run_id=20261021T080316-ecbe54b0）；其中位数为2168.494。本轮未登录A3重跑，当前A5也未重新穷举参数，因此比较的是“当前A5预设”与“A3历史推荐配置最佳测量值”，不能宣称已经找到新版本A5的全局峰值。

两端普通性能使用相同100-case输入池，摘要21defc288cc9848d6f1f71f8a8c6d900327963431e57e77125274b5f5fc8c85b；但测量日期、seed、窗口长度和运行时版本有差别。本次比值为1.2298倍；若以1.6倍为目标，则A5需达到3488.414业务样本/秒，比当前还需约30.1%吞吐提升。这里将1.6作为目标，不将其视为已经验证的所有算子、所有输入的硬件必达比例。

为避免沿用旧同步版本的Attention归因，本轮在空闲逻辑卡1补采当前4实例profiling：16|32、batch16、固定case-0020（T=2041、prefix=24）、每实例预热10次、正式总共100次、seed42、PipeUtilization。全部成功；运行前后及启动快照均显示卡1无进程，运行前利用率为0。模型、算子库、运行时及配置在采集前后摘要一致。新增run_id=20261022T120752-b0b1c926，使用附录L的同线程预热与int32 bounds提前planner版本。此固定输入profiling仅用于逐算子分析，不替代附录N的变长输入普通QPS。

A3使用历史同一case的5实例profiling（profile-median-opt-concurrent，run_id=20261021T080811-d0f8ffdf）。以下比较是各自推荐核预算和并发下的可见task耗时，包含资源竞争影响，不是等核、裸芯片速度比较。

### 当前Attention kernel已经超过1.6倍，MatMul整体尚未达到

| 类别 | A3每图累计 ms | 当前A5每图累计 ms | A3耗时/A5耗时 |
| --- | --- | --- | --- |
| Attention计算kernel，24次 | 4.144 | 2.352 | 1.762倍 |
| 全部MatMul，169次 | 23.523 | 16.568 | 1.420倍 |
| 其中MLP gate/up+down | 16.540 | 11.144 | 1.484倍 |
| 全部计算/清零task累计 | 34.134 | 23.585 | — |
| 设备图平均时间 | 34.258 | 22.961 | — |

每层Attention kernel为172.674μs→97.986μs，已快1.762倍。当前A5中，Attention计算占设备task累计约10.0%，全部MatMul约70.2%，其中MLP约47.3%（MLP已包含在MatMul内，不能再相加）。MLP gate/up单次227.775μs→149.829μs约1.52倍，down 233.605μs→164.674μs约1.42倍。整网收益因此受到占比更大的MatMul/MLP限制，不能仅由Attention kernel的加速推导。

累计task时间与设备图墙钟不同，planner和主流计算有重叠，跨实例也存在重叠。上表占比用于判断优化重心，不将它们解释为可直接线性相减的严格并发吞吐上限。若固定其余执行开销，Attention kernel减半只对应约5%量级的task工作量下降，不能据此期待整网自动增加约30%的吞吐；资源争用变化的真实收益仍需A/B。

### Planner现已基本重叠，旧版串行归因需要更新

| 当前4实例，每图均值 | 时间 |
| --- | --- |
| 24次AICPU planner累计 | 0.810ms |
| planner与本模型主流计算交集 | 0.788ms |
| planner交集占累计比例 | 97.22% |
| 未与本模型主流计算重叠的planner任务时长 | 0.023ms |
| metadata清零累计 | 0.018ms |
| event独占覆盖 | 1.720μs |
| EVENT RECORD / WAIT / RESET | 24 / 24 / 24 |

正文历史版本将kernel、planner、清零串行相加得到约1.30倍，适用于当时的实现。当前planner约97.2%已与主流覆盖；继续把0.810ms完整叠加到2.352ms kernel上，会重复计时。未重叠时长也不自动等于端到端关键路径，因为可能落在其他图间隔内。

仍有24层重复规划和清零，及对应Host launch、事件管理工作。其输出仅依赖本次请求的bounds、head配置与核数，各层Q/K/V数值变化不影响规划，所以“每请求、每图实例预处理一次，24层复用”仍值得实验。但收益重点将是减少重复工作与发射次数，不能再按省去23次planner累计时长来承诺同等图延迟收益。输入bounds每轮会变化，不能仅按设备地址或总T缓存；资源及完成依赖需按请求/实例隔离，全部消费者结束后再复用。

### Attention并发伸缩：单实例优势被明显压缩

| Attention kernel单次可见耗时 | A3，8|16 | A5，16|32 |
| --- | --- | --- |
| 各自单实例 | 126.815μs | 34.888μs |
| 推荐并发 | 5实例：172.674μs | 4实例：97.986μs |
| 并发/单实例膨胀 | 1.362倍 | 2.809倍 |

A5 kernel本身没有随实例数改变，同一输入从约35μs增至98μs，说明推荐并发下的资源竞争/执行交错已明显影响单kernel可见耗时；不能直接用单实例kernel优势推算峰值吞吐。具体是Cube、Vector、搬运还是调度约束，现有PipeUtilization不能完全区分，不直接认定HBM带宽饱和。A3与A5每实例限核并非独占绑核，不能把8×5、16×4当作物理核心总数。

普通性能也可以用闭环近似解释：A3历史最佳平均batch延迟36.649ms，A5当前为23.800ms，单batch延迟比约1.540；实例数因子为4/5，二者相乘约1.232，与实际QPS比1.230接近。这是观测关系，不意味着直接加到5实例就会按比例提升；增加实例也会继续拉长kernel和图的耗时。

### 源码中Attention可进一步验证的限制

1. 固定核数及尾块保护：ge/src/kernel_launcher.h:23固定kAicCores=16，LaunchAicpuPlanner传入macro_cost=1；metadata.py:157–163会执行owner_limit=aic_core_num-1，因此最多15个owner承担计算，尽管launch的block数是16。扩大GE核预算不会自动得到32个Attention计算owner。这是单次kernel分配上限，不能推断4实例全卡只有15核工作。该减一逻辑有宏流水线尾块保护原因，不能直接删除后假定正确或更快。
2. 固定128行Q tile及独立prefix：attention.py:615–650按unit、q_head、q_tile遍历；本case有17个unit，24个Q tile/head、共336个head-tile，Q行填充率2041/(24×128)=66.44%，prefix单独tile只有24/128=18.75%。这说明尾块与小prefix存在填充损失，不是PMU算力利用率。源码层面将prefix与req0合并可使本case的tile/head从24变23，减少一个tile约4.2%，其整网收益不能据此直接推断。
3. GQA复用与softmax成本：A3是FiaBlockCubeNonQuantGqa专用模板并采用prefix-merged；A5虽通过q_head//7选择同一KV head，但按Q head独立构建Q/K/V tile，源码未显式将7个Q head打包到一个共同QK/PV工作项。可验证GQA打包或KV的L1复用，不能把重复load指令简单当作7倍HBM读量，缓存可能复用。A5的softmax在UB中先求max、再重读scores求exp/sum；这两遍向量处理没有重新计算第二次QK MatMul。对短序列、小D=64，填充、向量和流水线固定成本可能限制峰值算力兑现。
4. 更小tile需整体实验：按同一case计算，Q tile=64的逻辑行填充率可到79.73%，但tile/head从24增加到40，且现有双AIV各64行的布局、metadata代价模型和channel路由都需要适配。不能只看到填充率上升就认定更快。

已核对历史b4b30cd→当前0d59d480：ascendc目录及DSL attention.py/metadata.py均未发生提交间差异；本轮没有修改这些实现。当前GE同步和metadata缓冲管理有此前未提交改动，因此旧版本性能只用作明确标注的历史基线。

### 优化顺序建议

先以当前版本重新验证3–5实例和配套限核的吞吐，确认A5新峰值；Attention侧优先验证每请求一次planner及显式可调核数/有效owner数，再做prefix合并、GQA复用和tile变体。要补齐整网约30%吞吐差额，同时需要处理当前占比约70%、仅快1.42倍的MatMul路径，特别是MLP与A3 NZ/A5 ND布局差异；没有关闭NZ pass的对照，不估算NZ单独带来的收益。

### 原始证据与新增采集记录

- 当前普通性能：附录N，run_id=20261022T115820-b644b307。
- A3历史普通最佳：run_id=20261021T080316-ecbe54b0，基线原始路径见上文及附录A。
- 本次4实例profiling：`/home/h00909914/ascend-ge-adapters/work/qwen2.5-0.5b/results/20261022T120752-b0b1c926/bench-profiling/PROF_000001_20261022120800479_00105597CGEHJOFG`。
- 本次带profiling固定输入窗口：业务QPS=2731.042、P99=25.563ms、100次正式请求、0错误；此数值不与变长输入普通QPS混用。
- 配置、环境快照、完整逐算子/主辅流分析、shape计算、源码快照及历史差异审计：`/home/h00909914/ascend-ge-adapters/work/attention-a3-a5-analysis-20261022-120732`。
- comparison-summary.json与a3-baseline-evidence.json保存比值计算与历史基线，6张导出表已转为JSONL并保存列名/SHA256，不保留CSV。本轮仅采集和分析，没有修改运行时代码或算子实现。

## 附录 P：完整 bounds 的算子内部缓存与四实例性能（2026-10-22）

### P.1 检查点与实现边界

修改前先提交可恢复检查点，未推送：主仓 `ccb8283deb6b8b4d80e7241c390512b34df70838`
（`perf(qwen): preload bounds and warm up benchmark workers`），ascend-ops
`fbf0400872079046c1b494188c5d8bd2749d868a`
（`perf(prefix-attention): overlap metadata planning with model execution`）。本节缓存改动为此检查点之上的工作区修改。

不修改原生 GE 执行器、项目 runtime、前端适配器、默认配置或输入池。沿用 CANN 9.3.0 weekly、
已安装的 output-size 修复及同工作线程预热；AIR 摘要保持
`fb5a0e6af78ca25a622fae0020570b1fb71c20baab9766b6631b0c841979baf8`。

缓存仅在 GE 算子内部使用：每个 `(ACL context, 主流, 辅流)` 有一个固定条目，各层共享；
每个 `(ACL context, 主流, 辅流, 节点名)` 保留独立 16 KiB metadata 输出。
完整键包括 bounds 长度、所有 int32 值、Q/KV head 数、AIC 核数、macro_cost 和 planner/metadata ABI。
不以地址、shape、总 token 数或哈希作为完整匹配条件。比较、计算及保存键使用同一次 bounds 快照。

命中时完整复制 16 KiB 到节点私有输出；未命中时先清零完整输出、执行原 planner、保存键和结果，最后发布 valid。
初始化和所有缓存读写均在同一辅流上排序，不需要设备锁。Host 锁仍只保护资源表和分配。
保留输入就绪规则及 planner → kernel 同步；Attention 从不直接读取共享缓存。
条目最多保存 256 个 bounds 元素，超过时完整规划并绕过缓存，不截断、不扩容。
当前私有缓存结构为 17,444 字节/执行通道，另计分配器开销；24 层节点私有输出仍为 384 KiB/实例。
同一实例须串行完成执行，多实例可并发。已有显式清理接口一并释放缓存，卸载约束不变。

原 DSL planner 及生成的计算 body 未改动，保留原入口；新 AICPU 包装复用原 body。
`dump_kernel_src.py` 在生成后自动附加缓存入口，`add_metadata_cache.py` 可在无 DSL 环境恢复包装，幂等检查通过。
ABI/布局或影响结果的 planner 常量、算法变化时须递增 `kPlannerCacheAbi` 并重建配套交付件。
仍有 24 次 AICPU launch；同一请求通常由 24 次完整规划变为 1 次规划和 23 次比较/复制。
这不是消除 launch 或 event 开销的方案。

### P.2 正确性与隔离验证

临时验证驱动均放在 `work/bounds-cache-20261022/`，未增加仓库多流测试文件。

| 检查 | 覆盖 | 结果 |
|---|---|---|
| 原 planner 对照 | 4 条并发流、每流 20 次，完整 16 KiB 逐字节比较，共 1,310,720 字节 | 全部一致 |
| 同 shape 不同 bounds | 同一设备地址、长度、总 T、prefix 不变，仅改变中间边界；另测 prefix 改变及重复命中 | 正确失效或命中，无误用 |
| 其他键与容量 | Q/KV heads、核数、macro_cost、bounds 长度变化；300 元素超容量 | 全部与原 planner 一致；超容量回退 |
| ABI 与写越界 | 每流额外 1 次 ABI 失效；缓存与输出尾部 64 字节保护区 | 每流 hits=10、misses=9、bypass=2；保护区完整 |
| GE 四实例 | 4 个模型使用相同节点名，每模型 2 个节点，固定 T=2048、batch16，轮换 bounds；32 次图执行/64 次算子执行 | 117,440,512 个元素通过独立 FP32 参考，max abs=0.000186502934，RMSE=9.00648737e-06 |
| 模型卸载 | 四模型执行完成后 RemoveGraph，再调用已有清理接口 | PASS |
| Qwen 全输入池 | 100 case、12,800 个评分，与缓存前输出比较；各层 QKV 不同但 bounds 相同 | 逐位一致 |
| Qwen FP32 参考 | case0000，沿用仓库 compare 口径 | PASS，min cosine=0.999997816254，max relative L2=0.003342528654 |
| 实际缓存共享 | 单实例 100 case 的 2,400 次 launch 地址跟踪 | 1 个共享缓存、24 个不同的节点私有 metadata 输出 |

AICPU 与 GE 四实例检查在逻辑卡 1 尚可用时完成；之后卡 1 出现 Critical/设备启动失败，
全模型验证及下述性能均切至空闲逻辑卡 3。未对其他任务或设备执行复位操作。
地址跟踪仅用于正确性，不带入性能测试。临时驱动曾遇到链接/不可复制 Tensor 的编译问题，修正后通过；
第一次 `RTLD_NEXT` 跟踪在首个 launch 后中断，改为显式 artifact handle 后采集完整；
卡 1 的设备检测失败重试也未计入性能。过程保存在 validation.log、trace-attempt1.json、
correctness-attempt2-device1.json 中。

### P.3 关闭 profiling 的默认配置测试

统一参数：`16|32`、batch16、4 实例、100 case 输入池、1000 次正式调用/16,000 业务样本、seed42。
每实例沿用 warmup=10 的配置及完整 shape 覆盖，实际共 356 次预热，并在正式执行的同一工作线程完成。
输入池摘要保持 `21defc288cc9848d6f1f71f8a8c6d900327963431e57e77125274b5f5fc8c85b`。
profiling=false、dump=false，无 LD_PRELOAD/地址跟踪。QPS 为业务样本/秒；延迟含 H2D、描述符处理、执行与同步，
不含服务排队或组 batch 等待。各轮配置、AIR、runtime 和运行中的算子库摘要前后一致。

先运行 3 轮缓存版本，再在同一空闲卡临时安装检查点 `fbf0400` 构建的无缓存交付件运行 3 轮对照，
两组均无 profiling。对照后已恢复缓存交付件，并逐文件校验摘要。下表全部保留，不挑选最好一轮。

| 版本/轮次 | 逻辑卡 | QPS（业务样本/s） | 平均 ms | P50 ms | P99 ms | 最大 ms | errors | run id |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 附录 N 历史默认无缓存 | 1 | 2681.175 | 23.800 | 23.679 | 26.749 | 31.481 | 0 | `20261022T115820-b644b307` |
| 缓存 1 | 3 | 2685.188 | 23.814 | 23.705 | 26.476 | 26.861 | 0 | `20261022T125551-042ad759` |
| 缓存 2 | 3 | 2685.020 | 23.801 | 23.693 | 26.474 | 27.191 | 0 | `20261022T125734-775609f2` |
| 缓存 3 | 3 | 2689.179 | 23.779 | 23.632 | 26.517 | 27.689 | 0 | `20261022T125915-178b25e4` |
| 同卡无缓存对照 1 | 3 | 2686.033 | 23.806 | 23.681 | 26.602 | 27.418 | 0 | `20261022T130110-d200740c` |
| 同卡无缓存对照 2 | 3 | 2686.544 | 23.792 | 23.680 | 26.515 | 27.484 | 0 | `20261022T130250-6c92b176` |
| 同卡无缓存对照 3 | 3 | 2683.629 | 23.804 | 23.713 | 26.349 | 27.253 | 0 | `20261022T130432-5211e1ac` |

| 三轮独立窗口中位数 | QPS（业务样本/s） | P99 ms |
|---|---:|---:|
| 同卡无缓存对照 | 2686.033 | 26.515 |
| 缓存 | 2685.188 | 26.476 |

缓存相对同卡无缓存对照：QPS **-0.031%**，P99 **-0.147%**。
相对附录 N 历史默认结果：QPS **+0.150%**，P99 **-1.021%**。
中位数分别统计各指标，并非合成的一次运行；正式调用全部成功。
三轮窗口不能单独证明微小差异有统计显著性，换卡的历史比较也不能单独归因于缓存。本配置未观察到明显吞吐提升，P99基本持平且均低于60ms。

该实现已验证完整键复用与多实例隔离，吞吐收益以本表为准。结合附录 O 中旧 planner 已约 97.2% 与主流重叠的观测，
减少重复规划未必直接减少整图关键路径；24 次 launch、同步和每层 16 KiB 复制仍存在，MatMul 与 Attention 主计算未改动。
本节未重新开启 profiling，不能把端到端变化直接等同于某个算子的耗时变化。

### P.4 记录与复现

算子源码：`third_party/ascend-ops/prefix-attention/cannbot-dsl/ge/`。
过程与完整数据：`work/bounds-cache-20261022/`，包括 correctness-result.json、launch-trace.jsonl、
source-checks.json、comparison-summary.json、final-artifact-sha256.json、operator-tracked.patch、
operator-new-files.json、operator-complete.patch（含新增文件）、各 window/control 子目录的原生 result、请求 JSONL、配置及测试前后 npu-smi。
本次生成的请求 CSV 已转为 JSONL 并删除；摘要及全部性能结果记录在本附录表格。

本次直接构建当前 ascend-ops 工作区，默认 model.yaml 中的 dependency revision 仍为原来的 `0d59d480`；
它不能代表本次未提交缓存源码。恢复与复现应使用上述检查点、工作区改动及实际交付件摘要。
提交新算子时应配套更新依赖版本记录。

缓存交付件 SHA256：

| 文件 | SHA256 |
|---|---|
| libcust_opapi.so | `fa27965ad3bfe28e4866538eb753754979e880fa5ba5a5ff3955dfa07c49fc25` |
| aicpu_metadata.so | `a3bafdbb20d3db36d363e11219c2970583f142d4f67e518e550e7886d4aaf118` |
| kernel_fp16.so | `5760970b68f7bd162d88c15d199b7b742e9bdbb48ab5bd699bd6cb40d83315ef` |

## 附录Q：2026-10-22 A3复测与A5无缓存重新调优

A3使用99/hejun-cann9-qwen25，默认8|16、batch16、5实例，最终性能改用当时空闲逻辑卡复验（逐run记录设备）。
初测中的设备占用风险已标记；以下仅使用空闲卡复验。
三轮3000调用的业务QPS为2094.122/2126.997/2170.244，P99为42.101/41.440/40.913ms；
QPS中位数与原文档2168.494相差-1.914%。

A5使用850-a5/hejun-cann910-stream、CANN9.3.0、空闲逻辑卡3，先恢复无缓存算子检查点fbf0400。
本次64次扫描与确认运行后，16|32、batch16、3实例的三轮长窗口QPS中位数为2712.967，最差轮P99为20.021ms。
标准默认入口再次验证QPS=2714.666、P99=19.880ms；100case/12800评分与原基线逐位一致，FP32参考通过。
A5精确SoC的默认配置和依赖版本记录已同步；此前附录中的4实例与缓存版本是历史试验状态。

两端分别重新采集推荐核预算单实例、不限核单实例及推荐并发profiling。
本次不限核A5 attention实际启动32个block，旧“全核”记录实际上只有16个，不能直接混为同一kernel配置。
关闭profiling的不限核单实例QPS比为1.676倍，推荐配置峰值比为1.275倍；
不限核单实例attention每层耗时85.485→22.136µs（3.862倍），MLP累计仅加速1.360倍。
优化应优先检查MLP/MatMul和激活读写，再评估attention的GQA分块、流水尾部与核预算。

详细口径、全部性能表（含失败）、24层attention及169个MatMul逐算子表、原始profiling目录、
正确性和交付件摘要见 [A3/A5复测与无缓存调优](qwen2.5-0.5b-a3-a5-retuning.md)。
两端过程目录均为项目根下`work/a3-a5-retuning-20261022/`，导出CSV已转为JSONL并删除。

## 附录R：A5 MatMulV3权重NZ对照（2026-10-23）

已在现有WeightNzAndMatMulV3Pass中适配已有MatMulV3：编译期重排常量权重，并重建节点清除ND实现选择缓存。
同卡三轮业务QPS中位数2711.228→2767.918（+2.091%），最差轮P99 20.096→19.799ms；
全量100case/12800评分逐位一致。并发MLP的MTE2开销降低，单实例基本不变。
本节实验采用进程局部启用，A5默认配置保持ND；最终Pass已提交到个人fork的main。全部记录与逐算子指标见
[A5权重NZ实验](qwen2.5-0.5b-a5-weight-nz.md)。
