# BGE ONNX 融合候选：目标 CANN 安装包核查

记录时间：2026-10-24（北京时间）。SE 只读调查；未初始化 NPU、编译/加载/执行 GE 图或采集性能。执行环境为 SSH `99` → 容器 `hejun-cann9-bge`，安装根 `/usr/local/Ascend/cann-9.0.0`，`ascend-toolkit/latest` 实际指向此目录，`opp/version.info` 声明 `Version=9.0.0`。目标为 `Ascend910_9382`，芯片族算子描述为 `ascend910_93`；目标 platform INI 中 `SoC_version=Ascend910_9382`、`Short_SoC_version=Ascend910_93`、`AIC_version=AIC-C-220` 可对应此目录。

本报告区分四层证据：① GE schema；②目标 SoC 的 ops-info 支持列表；③实现源码/已安装预编译 kernel；④ ONNX parser 或 GE fusion pass 入口。四层均存在仍不等于本 BGE 的 GE mode=0 最终图已融合，也不证明性能收益。主控对原始图的逐边匹配和候选计数见 [onnx-fusion-patterns.json](../evidence/onnx-fusion-patterns.json)。

## 结论与建议顺序

优先核验 **49 处 LayerNorm、145 处 MatMul+Bias、24 处精确 GELU** 的最终 GE 图。它们在目标安装包中有对应 FP32 实现，LayerNorm/MatMul+Bias/GELU 还找到已实现的内置 fusion pass。**48 处 residual Add+LayerNorm** 具有 AddLayerNorm 全 FP32 kernel，但必须核实当前 inference mode 与内部 pass 限制。**24 处 attention** 可以考察 FlashAttentionScore，安装包真实存在 FP32 Q/K/V kernel；它和只支持半精度/量化 Q 的 PromptFlashAttention、FusedInferAttentionScore 不能混为一谈。**5 处 L2 归一化** 有语义匹配的 L2Normalize 实现，但需把 epsilon 转换成平方域，且其中 1 处输入全常量，应先看是否已被常量折叠。

| 原图模式（主控静态计数） | 本 CANN 已有目标 | 目标 FP32 证据 | 接入/限制 | 建议 |
|---|---|---|---|---|
| ReduceMean→Sub→Pow(2)→ReduceMean→Add(eps)→Sqrt→Div→Mul(gamma)→Add(beta)，49 处 | LayerNorm / LayerNormV3 | GE schema + ascend910_93 动态 TBE + FP32 预编译 `.o` | LayerNormONNXFusionPass、LayerNormONNXMULFusionPass、LayerNormInferenceFusionPass；原始 eps≈1e-12 必须保留 | 最先看已有最终图，优先复用内置融合 |
| MatMul→Add(一维 bias)，145 处，含 144 个 encoder 和 1 个 gate | MatMul/MatMulV2 的 optional bias；MatMulV3 | MatMulV3 全 FP32 ND+ND+bias+output 预编译 `.o`；MatMul 动态实现支持 bias | MatMulBiasAddFusionPass 的 DefinePatterns/Fusion 导出符号，配置也列名；需核实 rank/broadcast/布局处理 | 高优先，避免重做编译器已经完成的 bias 融合 |
| x→Div(sqrt2)→Erf→Add(1)→Mul(x)→Mul(0.5)，24 处 | **GeluV2(approximate="none")** | schema、910_93 ops-info、动态实现、FP32+none 的 high_precision `.o` | GeluONNXFusionPass 存在，但日志提示 unknown shape 可能排除；不能据此假定输出就是 GeluV2 | 高优先；先确认内置 pass 的实际目标及精度 |
| residual Add→LayerNorm，48 处 | AddLayerNorm | schema、全 FP32 ND ops-info tuple、全 FP32 kernel `.o` | AddLayerNormFusionPass 已实现；库中还存在 mode/LayerNormV3 限制提示；eps 默认1e-5，必须显式保留1e-12 | 第二步，和 LayerNorm 替换有依赖/重叠，不能把节点数或收益相加 |
| QKᵀ→scale/mask→Softmax→V，24 处 | **FlashAttentionScore** | Q/K/V 全 FP32 ops-info tuple、FP32 `.o`，输出 attention_out 可 FP32 | 原图 BNSD、16头、D64、S6/8/9/13；K 要还原非转置表示；mask/scale/排布和短S适配需逐项验证；未找到据以确认本模型自动融合的入口 | 保留候选，不能因 dtype 排除，也不能直接宣称能融合/有收益 |
| ReduceL2→Clip(norm,min=1e-12)→Expand→Div，5 处 | L2Normalize(axis=[-1], eps≈1e-24) | schema、910_93 FP32 ND ops-info、动态 TBE 源码 | GE 算子 eps 位于 sqrt 内；没有找到能证明当前模式自动融合的专用 pass；1处纯常量 | 较低优先，先看常量折叠和已生成融合内核 |

## 各目标的语义与支持边界

### LayerNorm / AddLayerNorm

已安装头文件 `opp/built-in/op_graph/inc/ops_proto_legacy.h:39392` 的 `LayerNorm` 定义为：

```text
mean = mean(x, reduce_axis, keepdims=True)
variance = mean((x-mean)^2, reduce_axis, keepdims=True)
y = gamma * ((x-mean)/sqrt(variance+epsilon)) + beta
```

输入/输出允许 DT_FLOAT/DT_FLOAT16/DT_BF16。`begin_norm_axis` 和 `begin_params_axis` 必须正确映射原图最后一维，gamma/beta 为 1024 维；epsilon 直接使用原图常量值。schema 的默认 epsilon、ops-info 默认、Python 实现默认并不完全一致，说明 **显式保留 epsilon** 是必要条件，不能从默认值推导等价。

`LayerNormV3` 的第二统计输出为 rstd，`LayerNorm` 为 variance。当前原图只需要归一化结果，但重写仍要正确处理额外输出/消费者；不能在出现统计量外部消费者时混用两个契约。已安装 `layer_norm.json` 的全 FP32 预编译选择项具有 x/gamma/beta/y/mean/variance 全 FP32、ND 格式，已核对相应 `.o` 存在。

`AddLayerNorm` 在 `ops_proto_nn.h:4077` 注册，输入 x1/x2/gamma/beta，可选 bias，输出 y/mean/rstd/x。910_93 ops-info 中的第 9 组为全 FP32 ND；`AddLayerNorm_normal_input_all_fp32_high_performance.o` 存在。原图 residual 路径要保证加法顺序、广播、额外消费者与 eps 保留。**只有 kernel 存在，不能证明 mode=0 内置 AddLayerNormFusionPass 会匹配**。库内日志包含：

```text
Only Fusion AddLayerNormV3 when graphRunMode is true.
AddLayerNormFusionPass is train mode
AddLayerNormFusionPass is not train mode
```

这些是内部分支限制线索，尚未读取对应实现源码，不能扩大为“整个 AddLayerNorm 仅训练可用”。无需因此改回 mode=1；应在当前 mode=0 路线做最终图核验或以后单独构造显式目标节点验证。

### 精确 GELU 必须与近似版本区分

安装的 `dynamic/gelu_v2.py` 注册 `GeluV2`，`approximate="none"` 分支对应标准 Gaussian CDF/Erf GELU，允许 FP32；实际实现使用数值逼近计算 Erf/CDF，因此仍需按模型原有阈值比对，不能许诺位级相同。`gelu_v2.json` 中 FP32+none 的 high_precision 与 high_performance 预编译条目和 `.o` 均存在。

安装的 **`Gelu` Python 实现明确使用 tanh 近似**（0.044715 多项式），`FastGelu` 是 1.702 系数的 sigmoid 类近似。schema 的“GELU”名称不能证明它和原图 Erf 版本相同。未经精度授权和复测，不把 Gelu/FastGelu 直接当等价替换。主张精确融合时优先明确 **GeluV2(approximate=none)**。

内置融合库中 `GeluONNXFusionPass::DefinePatterns/Fusion`、`GeluFusionPass::DefinePatterns/Fusion` 真实导出，且包含 `GeluONNXFusionPass cannot be applied for unknown shape` 日志文本。此处 unknown shape 是 pass 运行时看到的 GE shape；原 ONNX 是动态模型，不能直接推出 GE 实例化后仍然 unknown shape。

### MatMul + Bias

GE `MatMul`/`MatMulV2` schema 和目标 ops-info 带 optional bias；`MatMulV3` 提供 ND/ND、全 FP32输入、FP32 bias 和 FP32输出组合，预编译 `.o` 存在。原图多数三维 [B,S,K] 乘二维权重，再做末维 bias 广播；具体使用 MatMul/BatchMatMul/MatMulV3、reshape 和布局折叠由解析与编译决定，不能只凭数学公式断言无需 reshape 就可直接塞入任一算子。

内置 `MatMulBiasAddFusionPass::DefinePatterns/Fusion` 已实现，`fusion_config.json` 给出 pass 名和序号。应先查看原始模式在最终图中是否仍然分离；若已经作为 MatMul 的 bias 输入或 UB fusion 实现，不再把 ONNX 中的 Add 计为实际待消除 kernel。

### Attention：FlashAttentionScore 是 FP32 候选，另外两个不是本轮直接替换

910_93 `FlashAttentionScore` 的 input0/1/2 含两组 FP32 tuple；kernel selection 中有 Q/K/V=FP32、attention_out=FP32 的条目，且实际文件存在：

```text
opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_transformer/
flash_attention_score/FlashAttentionScore_9d6a04df15c0ec6754a9d9860ceb8208.o
```

对应 metadata `coreType=MIX`，kernel 内含 AIC/AIV 混合任务。这是 schema 之外的真实实现证据，但其具体 tiling 对当前 B=1、N=16、D=64、S∈{6,8,9,13} 是否接受，尚未调用 tiling/编译来证明。

`fusion_ops.h:84` 的契约支持 BNSD/BSND/BSH/SBH/TND。BGE 当前 Q/V=[B,16,S,64]，原 K 已转为 [B,16,64,S]，目标接受原始 K 布局后自己执行 Kᵀ；替换必须识别并移除/还原该转置。原图 Q/K 各乘 sqrt(1/sqrt(64))，数学上合并为 scale=0.125。浮点运算顺序改变仍须做精度验证。也可以先保留已缩放 Q/K、令目标 scale=1，以减少第一轮语义变更，是否保留额外 Mul 需实测决定。

原图是双向 attention 的 additive padding mask（0 / -FLT_MAX）。目标 atten_mask 使用 bool/uint8，必须转换为 **true/1 表示屏蔽**，并满足目标所允许的 [B,1,S,S]/[S,S] 等 shape；不能直接把原 INT64 keep-mask 当 mask，也不能开启 causal 模式。全 masked 行、含 padding 的边界输入、输出最后 reshape/transpose、keep_prob=1 以及 pre/next token 范围需明确。不要使用保留的 padding_mask 参数来猜测 additive-mask 接入。

目标 `PromptFlashAttention`、`FusedInferAttentionScore` 的 query dtype 列表无 FP32，其现有 kernel selection 也没有 FP32 Q 项；因此在本轮保留原始 FP32 的范围内不推荐直接替换。`FlashAttentionScore` 的 API 文档虽称训练算子，但并不等于其无 dropout 正向不能用于推理；当前路线要通过 GE operator/parser 接入证明，不能用 aclnn 单算子可调用替代 GE 可用性。

### L2Normalize epsilon 的位置

安装的动态 TBE 实现 `impl/ops_legacy/dynamic/l2_normalize.py:63-70`：

```text
square_sum = reduce_sum(x*x, axis, keepdims=True)
y = x / sqrt(max(square_sum, eps))
```

原图为 `x/max(sqrt(square_sum), 1e-12)`，因此数学语义对齐使用 **eps=(1e-12)^2≈1e-24**。直接填 eps=1e-12 会把最小分母变为1e-6，改变近零向量的输出。需要 FP32 的零向量/极小值/正常值回归。未在此安装包的目标 ops-info 与 schema 查到 `L2NormalizeV2`，不可凭其他版本印象采用这个名字。

### Softmax 与更大融合：有算子，但不宜凭名字套用

`ScaledMaskedSoftmax` 有 FP32 ND kernel，mask 是 bool、支持广播，且支持 fixed_triu_mask=false；不过安装头文件明确标为 experimental / please do not use，旧 Python 实现使用 -10000 掩码值，不能直接视为原图 -FLT_MAX 的逐点等价。原图 Q/K 的 scale 已在 MatMul 前，若只融合 mask+softmax，应避免重复缩放。`ScaledMaskedSoftmaxV2` 的910_93 ops-info 和kernel selection也有 FP32 ND+bool，但本次已查头文件未找到其公共 REG_OP、未验证 parser/GE 接入，因此仅保留内部实现存在的事实。

`MaskedSoftmaxWithRelPosBias` 支持 FP32，但 schema 注释明确为 Swin Transformer 专用，必需 relative_pos_bias；当前 BGE 没有对应相对位置偏置，不推荐用补零 bias 强行套用。

本次在目标 ops-info/公开 schema 未发现名为 `MatmulBiasGelu`、`MulsAddSoftmax`、`BertEmbeddings` 或 Embedding+LayerNorm 的可直接采用目标。没有发现并不证明所有命名/内部 UB fusion 都不存在；只是不能把这些名称列作本环境已核实的融合算子。

## 原生 ONNX 与 GE 入口

原始 BGE 为 **opset14**。ONNX 标准 [LayerNormalization 从17开始](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)，[Gelu 从20开始](https://onnx.ai/onnx/operators/onnx__Gelu.html)，不能在 opset14 图里直接创建同名标准节点并声称合法等价。ONNX operator 与 GE operator 不共享名字/属性契约，GE LayerNorm/GeluV2 本身不是 opset14 标准 ONNX 节点。

已安装 `libops_all_onnx_plugin.so` 中有 `ai.onnx::17::LayerNormalization`、`ai.onnx::18::LayerNormalization`、`ParseParamsLayerNormalization` 和 `ParseOpToGraphLayerNormalization` 的登记/函数文本，但这不能证明解析后一定保留成一个 GE LayerNorm 节点，解析器也可能扩图。没有查到标准 `ai.onnx::20::Gelu` 的登记证据。

安装插件同时包含 `npu::1::NPUGeluV2` 和真实导出的 `domi::ParseParamsNpuGeluV2(ge::Operator const&, ge::Operator&)`，也包含 `npu::1::NPUFlashAttention` 等自定义域登记。这表明存在现成自定义算子解析入口的候选；使用时仍要核实端口顺序、attribute schema、输出个数、目标映射以及当前 GE Session import 路径是否加载对应插件。不能仅从字符串就宣称“把ONNX op_type改成GeluV2/FlashAttentionScore即能运行”。

目前对原图更稳妥的第一步是 **不改 ONNX，核验内置融合后的最终 GE 图**。若内置模式未命中，再做单个隔离 GE pass 或已支持自定义 ONNX映射的最小实验；不要先同时升级全模型 opset、改 mask、替换全部 LayerNorm/GELU 与 attention。

## 证据索引与未运行项

- [安装身份与重点库 SHA256](../evidence/cann-fusion-inventory-identity.json)。
- [目标 ops-info 原始条目](../evidence/cann-fusion-inventory-ops.json)。
- [GE schema 与原文件行号](../evidence/cann-fusion-inventory-schema.json)。
- [实现源码只读快照](../evidence/cann-fusion-inventory-impl.json)。
- [预编译 kernel selection](../evidence/cann-fusion-inventory-kernels.json)。
- [FP32 kernel metadata 与对应 .o 存在性](../evidence/cann-fusion-inventory-binary-availability.json)。
- [已定义导出函数与诊断字符串](../evidence/cann-fusion-inventory-pass-symbols.json)。
- [融合配置、平台 INI 与 ONNX 登记文本](../evidence/cann-fusion-inventory-registration.json)。
- [附加算子、schema 与排除依据](../evidence/cann-fusion-inventory-extra-ops.json)。
- [ScaledMaskedSoftmaxV2 ops-info/实现/kernel selection](../evidence/cann-fusion-inventory-softmaxv2.json)。

主控读取的历史 `20261022T071838-a6374be8` infer/bench 日志明确 graphRunMode=0 且 output allocation=GE（dynamic compiled）；当前没有可用 GE pbtxt。因此 unknown-shape 与 mode 限制是实际待核实事项，不能写成原图已自动融合。

没有编译/运行重写图、没有证明具体子图已命中或未命中、没有测量收益，没有改动 CANN 全局安装目录。官方在线CANN文档可用性与版本内容混杂，因此本次芯片/dtype实现结论以 **99 已安装9.0.0包** 为主，ONNX版本边界以 ONNX 官方 schema 为准。短S、mask边界、kernel具体tiling限制和实际mode=0命中情况需后续设备独占条件满足时验证。
