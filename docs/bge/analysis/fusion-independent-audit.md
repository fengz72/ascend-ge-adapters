# BGE 融合候选独立静态契约审查

审查时间：2026-10-24（北京时间）。角色：测试。执行范围：只读原始 ONNX 的公共静态结构证据和 99 / `hejun-cann9-bge` 内的 CANN 9.0 安装源码、头文件；未改写模型，未编译、加载、执行 NPU 图，未进行性能测试。

结论：原图存在 49 处 LayerNorm 分解、24 处 Erf GELU、145 处静态 bias 的 MatMul+Add，可以进入后续 GE 融合核查；48 处残差 Add+LayerNorm 是更大范围的重叠候选。5 处 L2Normalize 和 24 处 Attention 也有模式，但必须先处理 epsilon、FP32、mask、scale 与动态 shape 契约，不能直接按算子名称替换。这里的“可替换”指数学与数据流候选，未证明 GE 自动命中或已有最终 kernel，也未证明性能收益。

## 模型身份与证据边界

- 原始路径：`/export/home/ext.hejun23/workspace/bge/model.onnx`。
- SHA256：`2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0`。
- ONNX opset 14，3137 个节点、400 个 initializer；图没有中间 `value_info`。
- `input_ids`、`attention_mask`、长期行为输入为动态 batch/sequence/long_len 的 INT64；`stat_ratio` 为 FP32；输出 `scores` 为 `[batch_size,10112]` FP32。浮点权重为 FP32。中间 shape 由边、reshape 常量和权重维度推导，不冒充已完成全图 shape inference。
- 源图说明可匹配结构，不证明运行时已经融合。安装了 aclNN 接口、op 定义或 TBE 实现，也不等于现有 ONNX→GE 路线能够导入该节点或自动命中此模式。
- 模型结构来自 [onnx-fusion-graph.json](../evidence/onnx-fusion-graph.json)。独立遍历结果为 [pattern-contracts](../evidence/fusion-independent-pattern-contracts.json)，安装源码摘录为 [cann-contracts](../evidence/fusion-independent-cann-contracts.json)，数值反例为 [numeric-witnesses](../evidence/fusion-independent-numeric-witnesses.json)。

主控逐边匹配脚本 `docs/bge/tools/analyze_fusion_patterns.py` 的输出也已独立复核：模型身份、LN/GELU边界输出集合、四类计数、中间消费者闭包及1处常量L2均与独立结果一致。48处标准残差和24处attention与原图逐层结构一致。交叉检查结果见 [crosscheck](../evidence/fusion-independent-crosscheck.json)。

## 候选及契约判断

| 候选 | 精确位置/数量 | 源图已核实条件 | 判断及必需条件 |
|---|---|---|---|
| LayerNorm 分解→LayerNorm | embedding 1 处；24 层各 attention/output 和 output 2 处，共49处 | ReduceMean→Sub→Pow(2)→ReduceMean→Add(eps)→Sqrt→Div→Mul(gamma)→Add(beta)，归约轴均为[-1]，hidden=1024；gamma/beta `[1024]` FP32；内部无外部消费者 | 数学可替换候选；必须显式 epsilon≈1e-12、最后轴、人口方差、FP32累计；需核实 GE 接入方式和实际融合状态 |
| 残差Add+LayerNorm→AddLayerNorm | 24层×2=48处；embedding末次Add另1处条件候选 | 残差加法输出仅流向同一LN的ReduceMean和Sub，保留LN边界输出即可 | 条件可替换；48个标准残差与上述49个LN重叠，不能重复累计收益；融合 bias 时需检查加法顺序 |
| Erf GELU→GeluV2(approximate=none) | 24处 `/intermediate/intermediate_act_fn` | Div(√2常量)→Erf→Add(1)→Mul(x)→Mul(0.5)；内部无外部消费者；FP32 | 数学可替换候选；CANN `GeluV2` 的 none 与本图语义一致，但实现及乘除次序不同，需误差验证；不能无条件替为 `Gelu`/`FastGelu` |
| MatMul+静态bias Add | 每层Q/K/V、attention输出dense、FFN两层共6处×24=144，再加gate1，共145处 | bias `[1024]`/`[4096]`/`[4]` FP32，沿最后一维广播；MatMul输出仅该Add消费；权重静态FP32 | 可进入已有MatMul bias融合核查；encoder激活为rank3时不能原样改成ONNX rank2 Gemm，须由GE合法lowering或保持shape的变换承接 |
| L2分解→L2Normalize | candidate、query、long、stat、final共5处 | `x / max(sqrt(sum(x²,axis=-1)),1e-12)`，keepdims=1；Clip→Expand→Div | 条件候选；CANN源码是`sqrt(max(sum(x²),eps))`，必须传原epsilon的平方≈1e-24，再核实FP32、零/极小向量和动态shape；candidate分支为纯常量链，应先排查常量折叠 |
| QK+mask+Softmax+V→Attention融合 | 24处 attention/self；16 heads，head_dim=64 | Q/K前各乘`√(1/√64)`，Softmax轴=-1，接V MatMul；无dropout；mask由输入padding生成 | 条件候选；必须证实当前910_93/FP32/GE路线支持、scale只应用一次、非因果mask及全mask行语义一致、layout/dynamic shape合法 |

## 不可忽略的数值与图边界

### LayerNorm 的 epsilon 与方差

所有 49 组 epsilon 的实际 FP32 值均为 `9.999999960041972e-13`，接近 `1e-12`，不是常见默认 `1e-5`。形式为 `mean((x-mean(x))²)`，不是 RMSNorm，也不是样本方差。CANN LayerNorm/AddLayerNorm 若使用不同归约顺序、Welford 或乘倒数实现，数学相同也可能出现 FP32 末位差异。

LN 组内 Sub 有两个消费者，分别用于平方和标准化，这是合法内部扇出；独立遍历确认49组内部输出无组外消费者。LN最终输出的Q/K/V、残差和Shape等多消费者是组边界，融合必须保留同名/等价输出。

AddLayerNorm 若进一步把 dense bias 融入 optional bias，需核实原始 `(matmul+bias)+residual` 与内核执行顺序。不可因“加法结合律”承诺 FP32 逐位等价。embedding最后一次Add也满足加法接LN形状，但不应把整段三个Gather和两个Add自动认作同一个已有EmbeddingLayerNorm匹配。

### GELU 的名称不代表同一近似

本图24组统一为 `0.5*x*(1+erf(x/1.4142135381698608))`。CANN安装源码 `.../dynamic/gelu.py` 使用 tanh 近似；`.../dynamic/gelu_v2.py` 支持 `approximate="none"` / `"tanh"`，支持FP32，none表示Erf语义。实际 `gelu_compute_erf` 使用多项式/exp实现，并非直接调用源图同一条Erf链，不能保证逐位一致。CPU数学反例中x=3时，Erf与tanh公式相差约0.0004123；24层累积后的业务输出必须实测，不能拿单点误差代替最终验收。

GELU源x进入Div和Mul的两路分支是该GELU内部数据依赖，不能只保留相邻Erf一个节点就删除其余数据边。MatMul+bias+GELU大融合必须覆盖bias Add的两路内部消费者。

### L2Normalize 必须平方 epsilon

原始约束在范数上，CANN已安装TBE实现 `.../dynamic/l2_normalize.py:63–70` 在平方和上取max。以 `x=[1e-8,0]` 为例，原始归一化首项为1；直接把 `eps=1e-12` 传入CANN形式得到约0.01。数学上需要传 `eps=(9.999999960041972e-13)²≈1e-24`，并检查实现允许的epsilon范围与精度。

五组中第一个candidate L2的MatMul两端都是initializer；其归一化及后续仅依赖常量的子图可能已被GE折叠。源图节点数量不能当作每请求必然执行次数或收益。

### Attention 的 mask 与 scale

输入 `attention_mask` 为0/1 padding可见性；图经1-mask、bool、Where构造加性mask。形状推导为 `[B,1,S,S]`，有效位置0、屏蔽位置 `-3.4028234663852886e38`（FP32最小有限值）。这是双向padding mask，不是因果三角mask。若融合内核改为bool/uint8掩码，须确认1/0含义、广播维和padding语义；不得默认启用causal。

源图Q/K各乘 `sqrt(1/sqrt(64))`，二者乘积数学上等于0.125。换成QK后乘0.125或交由attention scale属性后，FP32舍入/累积顺序不同；不能重复乘scale。原图使用最小有限值而非负无穷：全mask行在稳定softmax计算下可得到均匀分布，而一些融合attention会输出零或采用负无穷处理。必须覆盖该输入，或由正式输入契约明确排除；不能暗改支持范围。

Q/K/V的reshape与transpose由动态batch/sequence shape链构造。融合不能把实测S硬编码成模型支持范围；Q的Shape消费者也必须保留或用合法替代shape路径承接。仅25个Softmax里24个属于attention，剩余gate Softmax不应归为FlashAttention。

### Opset 与路线

原模型opset14不包含标准 `LayerNormalization`（自opset17）或 `Gelu`（自opset20）。不能在保留opset14时塞入这些标准节点；也不能仅改opset数字而跳过其他算子schema兼容检查。优先检查GE已有图融合pass、其命中条件及最终编译图；确需显式替换时，再选择目标版本已验证的CANN/ONNX接入方式。aclnn接口存在不能代替GE接入证据。

当前精度配置是FP32且must_keep_origin_dtype。仅支持FP16/BF16的fusion变体不属于本轮无精度变更的直接候选；若要降精度，应另列实验并先保持业务门槛，不能隐式cast来让融合“命中”。

## 后续验收要求

设备恢复后，先保留原版baseline并导出同一次构建的最终GE图/融合日志，确认上述候选哪些已融合、哪些已常量折叠。每次只验证一个候选或明确的组合，不用源节点消除数代替收益。

1. 子图精度覆盖：LN常量/近常量/大数输入与epsilon；GELU正负/尾部；L2零向量/1e-8/epsilon附近向量；MatMul不同B/S与bias广播；Attention混合padding、全有效、全mask及边界长度。
2. 五条给定样例与原始CPU完整scores对比，沿用既定cosine≥0.9999、relative L2≤0.001、max abs≤0.001；再与expected Top10按dense_id=index+1逐ID、顺序、分数绝对误差≤0.001校验。
3. 动态输入、输出shape/dtype和非有限值检查；多实例/并发配置另做逐请求输入输出关联验收。
4. 核查最终执行kernel的FP32/累计模式、shape/layout及实际执行次数；再按H2D+GE+D2H边界独立测吞吐和p99。

本轮静态结构审查通过；融合可运行性、最终图命中、精度和性能均未验收，当前不接纳任何性能收益声明。

## 主报告定稿前独立复核

复核对象：[fusion-static-analysis.md](fusion-static-analysis.md)，SHA256 `3ea30fa17ffab47747245ad789d6922d57e99e7da2b3752639213d8cf77ce378`。本次仅检查本地静态文档和已保存证据，未调用设备。

结论：静态分析报告复核通过，无必须修正项；本结论仅说明报告与证据一致，不新增融合运行、精度或性能验收。

- 六类计数为 LayerNorm 49、Erf GELU 24、MatMul+bias 145、标准 residual AddLayerNorm 48、Attention 24、L2Normalize 5，与独立遍历一致；AddLayerNorm重叠和一处常量L2已明确。
- 已抽核SE保存的目标kernel selection与binary存在性：LayerNorm/GeluV2/AddLayerNorm的实际FP32数据tuple、MatMulV3的FP32 x1/x2/bias/y、FlashAttentionScore的FP32 Q/K/V/attention_out均有对应binary_exists=true。这里不将可选索引/mask/offset参数误称为必须全FP32。
- Pass证据确有定义导出函数；主报告正确限制为“存在”，未据名称断言当前BGE已融合，也未把AddLayerNorm内部mode分支扩大成整个算子不可推理。历史无最终GE图，实际命中仍未确认。
- opset14、LayerNorm epsilon约1e-12、GeluV2 none、L2 epsilon平方约1e-24、FP32累计/舍入、非因果padding mask、-FLT_MAX与全mask行、Q/K分摊scale、动态shape和rank3 Gemm限制均已保留。
- 当前源码匹配、安装实现与最终运行状态三个层次清楚；未把source节点计数换算为收益、未新增OPT结果或设备测试结论。
