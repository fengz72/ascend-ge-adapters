# BGE ONNX 静态融合分析

日期：2026-10-24（北京时间）。范围：原始 ONNX 数据流、99 容器内实际 CANN 9.0.0 算子包和已有历史日志。主控分析，SE 核查已有实现，测试独立复核模式与数值契约。未改写模型、未初始化 NPU、未编译/执行新 GE 图、未测量性能。

## 1. 结论

原图有六类值得核查的结构。其中 **LayerNorm、GeluV2、MatMul 的 bias 融合** 优先核查；**AddLayerNorm、FlashAttentionScore、L2Normalize** 带有额外的模式或数值条件。目标 CANN 已有相应实现，不需要先假定要开发新的计算 kernel；ONNX/GE 接入、自动匹配和精度仍需分别证明。

| 源图结构 | 数量 | 已安装 CANN 目标 | 当前静态判断 |
|---|---:|---|---|
| LayerNorm 的九个基础计算算子 | 49 | LayerNorm / LayerNormV3，FP32 kernel | 完整数据流匹配；有内置融合 Pass；保留最后轴与 epsilon≈1e-12 |
| Erf GELU 的五个基础计算算子 | 24 | GeluV2(approximate="none")，FP32 kernel | 数学语义对应；内置 GELU Pass 存在，但实际目标、未知 shape 门槛须核查 |
| MatMul + 一维静态 bias Add | 145 | 带 bias 的 MatMul/MatMulV2/MatMulV3；已证实 V3 全 FP32 kernel | 有 MatMulBiasAddFusionPass；rank3 激活的合法 lowering、广播和布局待看最终图 |
| residual Add + LayerNorm | 48 | AddLayerNorm，全 FP32 kernel | 有融合 Pass，但存在内部运行模式/变体限制线索；与上述 LayerNorm 重叠 |
| QK、padding mask、Softmax、V MatMul | 24 | FlashAttentionScore，Q/K/V FP32 kernel | 可保留为条件候选；需要验证短序列、mask、scale、布局和 GE 接入 |
| ReduceL2 + Clip + Expand + Div | 5 | L2Normalize，FP32 ND 实现 | epsilon 要转换到平方和域；其中一组为纯常量链，先查常量折叠 |

数量是**源图结构计数**，不是已确认未融合的执行 kernel 数，也不能直接换算为收益。不同组存在重叠：48 个 AddLayerNorm 包含在 49 个 LayerNorm 中；FFN 的 MatMul+bias 与后续 GELU可组成更大区域，不能将各项百分比或源节点消除数相加。

原始模型 SHA256=`2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0`，opset14、3137 节点、400 个 initializer、无中间 value_info。结构显示24层 Transformer、hidden=1024、16 heads、head_dim=64、FFN=4096；本轮五例 B=1、S=6/9/8/13/8。中间维度由连接、权重与 reshape 常量推导，未冒充完整 shape inference。

## 2. 可以对应已有算子的具体结构

### LayerNorm：49 组

```text
x ──ReduceMean(-1)──mean
└──Sub(mean)──d──Pow(2)──ReduceMean(-1)──Add(epsilon)──Sqrt──std
             └────────────────Div(std)──Mul(gamma)──Add(beta)──y
```

一组在 embedding，另外每层 attention output 和 FFN output 各一组，共 `1+24×2=49`。gamma/beta 都为 `[1024]` FP32，归约保留维度，人口方差，epsilon 的实际 FP32 值为 `9.999999960041972e-13`。

例如 `/encoder/backbone/encoder/layer.0/attention/output/LayerNorm/ReduceMean` 到同前缀 `Add_1`，九个非 Constant 节点可以表达为一个 LayerNorm。49 组内部中间张量均无组外消费者；边界 y 的残差、Q/K/V 和 Shape 消费者必须保留。

CANN 已安装 LayerNorm/LayerNormV3 schema、目标 `ascend910_93` FP32 实现与预编译 `.o`；融合库中有实际定义的 LayerNormONNXFusionPass、LayerNormInferenceFusionPass。LayerNorm 与 V3 的统计输出契约不同，原图只需 y 时也必须正确处理额外输出。不得使用常见默认 epsilon=1e-5 替代原值。

### Erf GELU：24 组

```text
x ──Div(1.4142135381698608)──Erf──Add(1)──Mul(x)──Mul(0.5)──y
```

每层 FFN intermediate 一组。例如 layer.0 的 `/intermediate/intermediate_act_fn/Div` 至 `Mul_1`。源激活宽度4096，五个计算节点构成完整闭包。

对应目标优先使用 **GeluV2(approximate="none")**。安装源码中的 Gelu 使用 tanh 近似，FastGelu 使用另一种近似，不能因名称相似就当等价。GeluV2 的 none 分支也采用具体的数值实现，乘除和舍入次序可能不同，仍需原阈值精度回归。

已找到 FP32、approximate=none 的 GeluV2 kernel；GeluONNXFusionPass 存在，但不能由 Pass 名推断输出一定是 GeluV2。其诊断文本还含 unknown shape 排除提示，须以当前最终 GE 图和日志确认。

### MatMul + Bias：145 组

```text
x[..., K] ──MatMul(W[K,N])──Add(bias[N])──y[...,N]
```

| 位置 | 权重 shape | bias shape | 数量 |
|---|---|---|---:|
| Q/K/V 与 attention output | `[1024,1024]` | `[1024]` | 96 |
| FFN intermediate | `[1024,4096]` | `[4096]` | 24 |
| FFN output | `[4096,1024]` | `[1024]` | 24 |
| 最后 gate 投影 | `[512,4]` | `[4]` | 1 |

145 个 MatMul 输出都只供对应 bias Add 消费。另有24个 attention MatMul 后的 Add，另一端为动态 padding mask，不是静态 bias，未混入计数。

CANN 的 MatMul 家族带 bias 输入，且有 MatMulBiasAddFusionPass。大多数 encoder 输入为 rank3 `[B,S,K]`；ONNX Gemm 是 rank2，不能直接改名替换。需要 GE 已有 lowering、BatchMatMul 处理或保持动态 leading dims 的 reshape。先查最终 MatMul 是否已经带 bias，避免重复开发融合。

### residual Add + LayerNorm：48 组

```text
branch + residual ──LayerNorm(gamma,beta,epsilon)──y
```

每层 attention 和 FFN 各一组。残差 Add 输出只进入同一 LayerNorm 内部，具备融合闭包条件。embedding 末次 Add 也接 LayerNorm，但有广播/embedding语义条件，单独列作条件区域，不计入48个标准残差组。

目标 AddLayerNorm 已有全 FP32 kernel。内置库包含 AddLayerNormFusionPass，但诊断字符串存在 graphRunMode 与 AddLayerNormV3 的分支限制，尚未读取该分支源码；不能据此声称 mode=0 一定命中，也不能扩大成“整个算子只支持训练”。继续保留 mode=0，后续查最终图或显式目标节点。进一步把 dense bias 一起融合时还要保留原始加法顺序与广播语义。

### Attention：24 组

```text
Q ──Mul(sqrt(1/sqrt(64)))──┐
Kᵀ──Mul(sqrt(1/sqrt(64)))──MatMul──Add(padding_mask)──Softmax(-1)──MatMul(V)
```

CANN 9.0 目标包中 **FlashAttentionScore 确实存在 Q/K/V 全 FP32 的 kernel**；不能笼统地说 FP32 无法使用 Attention 融合。PromptFlashAttention 和 FusedInferAttentionScore 在该目标描述中没有 FP32 Q，不能用这两个变体替代当前 FP32 模型。

源 Q/V 为 `[B,16,S,64]`，K 已转置为 `[B,16,64,S]`；目标通常接收未转置 K 的布局，须正确还原。源缩放分摊在 Q/K，数学 scale 为0.125；合并后累积顺序可能改变。源 mask 是双向 padding mask，0或-FLT_MAX，目标 bool/uint8 mask 的1通常代表屏蔽；不能把原 INT64 keep-mask直接传入或开启causal。

必须单独验证 S=6/8/9/13 的短序列、混合 padding、全 mask 行、keep_prob=1、scale只应用一次及动态形状。短S的融合开销也可能抵消收益，当前不预测百分比。尚无本模式自动转为 FlashAttentionScore 的命中证据，显式GE节点或ONNX自定义域映射仍是后续接入工作。

### L2Normalize：5 组，1组纯常量

```text
原图：x / max(sqrt(sum(x*x)), epsilon)
CANN：x / sqrt(max(sum(x*x), eps))
对应：eps = epsilon² ≈ 1e-24
```

五组分别为 candidate、query、long、stat 和最终融合向量。直接传 eps=1e-12 会改变近零向量结果：`x=[1e-8,0]` 时原结果首项为1，错误参数可得到约0.01。应使用原始 epsilon 的平方，继续验证零/极小向量及实现参数范围。

第一组 candidate 的输入投影两端都是 initializer，连同归一化和后续常量投影可能已被 GE 常量折叠；这部分不能按每请求运行收益计算。其余四组依赖实际输入。没有找到足以证明当前模式自动融合的专用 Pass。

## 3. 已有实现与当前已融合状态的区别

本轮 SE 分别核对 GE schema、目标 ops-info、实现源码、预编译 kernel `.o`、Pass 导出函数与解析登记，详见 [CANN 安装核查](cann-fusion-inventory.md)。核对不是仅看 aclnn 头文件或库中一个算子名称。

在 `runner/work/bge` 和 `bge/work` 搜索未找到可供核验的最终 GE pbtxt。历史五例运行 `20261022T071838-a6374be8` 的日志为 `graphRunMode=0`、`output allocation=GE (dynamic compiled)`；记录见 [历史证据检查](../evidence/fusion-history-check.json)。因此目前不能断言49个LN或24个GELU仍被拆分，也不能断言它们已全部融合。unknown shape 和 mode 限制是后续需要核实的实际问题。

原图 opset14，标准 [LayerNormalization 从opset17开始](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)，[Gelu 从opset20开始](https://onnx.ai/onnx/operators/onnx__Gelu.html)。不能在opset14中直接加入这些标准节点，也不能只改全模型opset数字。安装包虽有部分ONNX自定义域解析登记，端口、属性及GE Session加载仍未验证。优先保留原图并观察内置融合，未命中再做隔离接入实验。

没有把 MatmulBiasGelu、MulsAddSoftmax 或 EmbeddingLayerNorm 列作“本机已有可直接用的融合目标”：本次未找到其公开目标证据。ScaledMaskedSoftmax有experimental标记和mask常量语义差异；Swin专用的MaskedSoftmaxWithRelPosBias需要原图没有的相对位置bias，均不强行套用。

## 4. 下一步与本轮完成状态

建议执行顺序：

1. 设备可独占后，用原图 mode=0 的同次编译保存最终 GE 图/融合日志，确认已融合、未融合及常量折叠区域。
2. 先看 LayerNorm、GeluV2 和 MatMul+bias；只为尚未覆盖的模式建立候选，不重复实现已有结果。
3. 再核查 AddLayerNorm 的模式限制；FlashAttentionScore 和 L2Normalize各作为明确条件的独立候选，避免一次改变opset、mask、精度及多个子图。
4. 每次实施前创建固定OPT记录；执行后完整CPU分数和Top10均沿用原阈值，再以H2D+GE+D2H独立测量。没有实施前不分配虚构的优化结果。

完整结构、位置和闭包检查见 [匹配结果](../evidence/onnx-fusion-patterns.json)，可用 [静态分析脚本](../tools/analyze_fusion_patterns.py) 从 [图结构元数据](../evidence/onnx-fusion-graph.json) 复现。独立复核见 [测试报告](fusion-independent-audit.md) 及其证据。此脚本只分析元数据，不重写或执行ONNX。

本轮结果：静态分析与已安装实现核查完成；模型/权重、expected、运行代码、dtype与mode未改变；设备占用预算仍为0，优化尝试数仍为0。融合命中、可运行性、模型精度和吞吐收益均未新增验收结论。
