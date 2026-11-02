# StartLux-Decision v1.1 0.8B Instant 适配方案

> 本文是适配方案与历史设计记录：分阶段设计、门槛与过程结论保持原样。当前
> 最终验收状态以 [../validation.md](../validation.md) 与
> [../performance.md](../performance.md) 为准（P0–P8 全部完成/通过，正式
> config 已切换统一图，P5 性能已由 P8 正式基准验收）。文中引用的
> native-validation.md、unified-validation.md、decision-validation.md 等已
> 从模型根移除的历史文件，其原字节快照在
> `work/startlux-archive/cleanup-20261102/docs/` 可找到。

## 1. 目标与范围

在 ascend-ge-adapters 中新增独立适配器，复用 data、baseline、prepare、build、infer、compare、bench 七步管线。模型经 TorchAir 导出 AIR，由原生 GESession mode=0 编译、加载、执行。主 agent 负责任务拆分、代码/文档审查、独立核验；实现与实验由容器内 cannbot 执行。每阶段完成后报告证据，审查通过再进入下一阶段。

先完成文本/JSON 的 choice、noul、score 及 None/宽选项语义，再验收变长批处理和性能；图像支持作为完整能力的后续阶段。当前验收平台仅声明 Ascend910_9382（A3），不推断 A5 兼容性。Instant 不生成文本，不实现 AutoThink；一个问题组或分组淘汰请求可能对应多次 read-out。

## 2. 固定环境与已知证据

- 主机 SSH alias: 100；容器: hejun-cann910-jev。
- 工程: /export/home/ext.hejun23/workspace/ascend-ge-adapters。
- 镜像: swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-a3-openeuler24.03-py3.12。
- Python 3.12.13；CANN 9.1.0；torch/torch_npu 2.12.0；transformers 5.8.1；TorchAir 使用 torch_npu.dynamo.torchair。
- 上游推理代码固定 commit: a99977d9332decc9fb27a2488b4a83b533b6cd5f；主agent已逐文件核对33份源码/许可材料的Git blob哈希，全部匹配。
- 模型固定 revision: 6496ae641cc71b149ce9dad69bd97f2a44e7bf11；完整权重 1746942632 字节，主agent独立计算 SHA256=f788c8e01fd453dfbf2effb23a06148d38238b9e8a872ab8faacbb49a03cecd1，与 HF LFS oid 一致。材料准备已完成，见 work/startlux-materials/materials-report.md。
- 16 个逻辑设备可见；NPU FP16/BF16 matmul 已通过。
- 工程 tiny 模型 AIR 导出、原生 runtime 编译、GESession mode=0 动态 shape 推理已通过；3 case 最大误差约 1.19e-7。证据位于 work/environment-check-cann910。
- npu_chunk_gated_delta_rule：BF16 单算子执行通过（B=1,L=65,Nk=Nv=16,Dk=Dv=128）。早期导出报缺少 ChunkGatedDeltaRule IR 注册（历史记录；官方文档含 GE 静态图示例，不等于官方不支持图模式）。该问题已由项目本地 REG_OP 兼容原型解决：依赖经 core.dependencies 正式整理，通过真实 prepare/environment/activate、子进程 IR 探针、正式 env 下 AIR 重导出与原生 GESession 数值/隔离/重入门禁（native-validation.md），原生路线 B1/S256 整模 7/7 通过（work/startlux-validation/native-full/）。标准算子小图 B1/B2 异长、卷积、双向注意力随机输入门禁为 P1 历史；其暴露的原 Neumann 幂级数求逆在高相关 key 下的严重数值消减已由块三角加倍求逆修复（gated_delta.py 当前实现，独立压力原型 relL2 约 1e-7~6e-7，真实层输入对照与整模 CPU/GE 通过），标准路线 B1/S256 整模 7/7 通过。随机小图通过仍不可单独推断真实权重可用。
- chunk DeltaNet、recurrent DeltaNet、causal conv 具有 TorchAir converter；接口存在不等于实际图执行已通过。
- 融合因果卷积文档限定 kernel=3；模型需要 kernel=4，首版采用等价标准运算。
- 实施记录模型 HF revision、官方代码 commit、工程 commit、包版本、镜像 ID、输入 seed 与命令，不用 main/latest 作为可复现身份。

## 3. 模型语义

模型: https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant
官方推理代码: https://github.com/StartLuxLabs/StartLux-Decision

加载类为 Qwen3_5ForConditionalGeneration，提取 model.language_model 与输出 head。文本骨干 24 层、hidden=1024：18 层因果 Gated DeltaNet，6 层双向 full attention。full attention 为 Q=8/KV=2/head_dim=256，partial RoPE 旋转64维。DeltaNet 为16个 key/value heads、维度128、因果卷积宽度4。

- 全注意力对每条完整 prompt 双向读取；DeltaNet 与短卷积保持因果。
- 每个问题构造完整 state/question/options，禁止复用共享 prefix 表示或沿用 Qwen 的 causal mask。
- 官方 prompt、chat template、thinking-off 后缀与选项映射必须保持一致；校验 A-Z 是读出位置的独立单 token。
- 温度和 letter_token_ids 读取 decision_config.json，不自行硬编码。
- 每条序列的卷积与递归状态独立，调用之间从零开始，丢弃末状态；不使用 KV cache。

## 4. 计算实现

保留原 HF embedding、线性投影、MLP、残差与权重。使用模型专用 nn.Module wrapper/实例适配；避免污染 Qwen 的进程全局 patch。golden 与候选在独立进程运行。

| 模块 | 首版策略 | 验证要点 |
|---|---|---|
| full attention | 先标准 matmul/softmax 对齐，再接支持该配置的融合注意力 | 双向；padding隔离；Q/K norm；sigmoid输出gate；Q=8/KV=2/D=256 |
| DeltaNet | 优先 npu_chunk_gated_delta_rule | Q/K L2 norm；scale仅一次；g为FP32；状态布局；长度定义；序列reset |
| causal conv | kernel=4 depthwise Conv1d，或4路位移乘加+SiLU | 权重顺序；左零填充；不跨行；边界长度 |
| RMSNorm | FP32规约，保留1+weight语义 | 与普通RMSNorm区分 |
| gated norm | 原公式先norm后SiLU gate | 权重dtype、计算顺序 |
| RoPE | 保留partial RoPE/原始文本位置规则 | 仅旋转64维，不套Qwen2全维RoPE |
| head | 单独复制A-Z对应26行，FP32投影 | 不破坏tied输入embedding；取最后有效位置 |

BF16主干；g、敏感归约及最终投影保留FP32。GE使用 must_keep_origin_dtype，不沿用 force_fp16。原生DeltaNet算子的状态dtype遵循接口，用实际误差决定替代策略。core.sources 当前统一 .to(dtype)，因此需在StartLux加载/适配边界保证 A_log 等原始FP32值没有提前被舍入，dt_bias按checkpoint原始dtype保真后在公式中转FP32；必要时从checkpoint恢复，不能简单BF16再转FP32。

若原生算子/图路径不满足需求，先给出最小复现与具体错误。可用固定长度纯Torch分块实现作为正确性兜底；不得用CPU推理冒充GESession推理，也不得依靠逐token Python循环作为最终性能方案。只有证明确有必要后再开发AscendC算子或修改第三方代码。

## 5. AIR与批处理契约

| 名称 | shape | dtype |
|---|---|---|
| input_ids | [B,S] | int64 |
| position_ids | [B,S] | int64 |
| actual_seq_lengths | [B] | int32 |
| scores | [B,26] | float32 |

第一阶段 B=1、固定S整模验证；再开放B/S动态维、右侧padding与混合长度。full attention只屏蔽padding key；读出位置 L-1。DeltaNet若将B*S展平，必须按物理S划分行，不能把各行有效长度误用于展平分段。其内部长度向量与全注意力累计边界分别转换，不共用错误定义。无效尾部不得跨行传播。

首批覆盖长度64/128/256/512/1024/2048及63/65/127/129边界。2048只是首版建议上限，不是模型原生上限；超限明确拒绝，不静默截断。长度分桶是组batch策略，动态不支持时明确交付固定档位。packed在正确性闭环后依据profiling决定，且每个问题保留完整prompt，不引入prefix共享。

## 6. CPU业务层

采用官方 prompt.py、modelcfg.py、readout.py、pipeline.py 的固定版本语义，保留许可及来源。每个问题在有效选项上 softmax(logits / temperature_by_type)。实现choice、noul的P(yes)、score期望值/confidence、None追加及概率拆分、allow_none、宽列表分组/淘汰与单选项特例。Instant拒绝auto/always thinking。保留 request_id、question_id、read-out轮次、选项顺序、batch行号对齐。

原生图只输出26列logits；解码与温度放在CPU。框架一个sample定义为一次read-out，业务原始问题数/请求数另计。infer/bench使用持久化输入不加载原始模型权重；业务在线入口可以依赖tokenizer和decision轻量材料，但不在runtime执行HF模型forward。

## 7. 工程文件与职责

模型目录 models/startlux-decision-v1.1-0.8b-instant：
- adaptation-plan.md：本方案；持续记录设计调整及理由。
- README.md：简介、依赖、使用命令和限制。
- startlux_npu_adapter.py：prepare_reference / prepare_candidate / prepare_runtime，encode/run/decode/export_spec及输出容量。
- decision.py：官方业务语义桥接、模板、tokenizer与解码。
- generate_cases.py：持久化输入生成/读取，真实决策fixtures和可重复压力输入。
- verify.py：独立业务校验，不只检查张量相似度。
- config/model.yaml：唯一模型配置；source.ref=${STARTLUX_MODEL_DIR}，source.loader=Qwen3_5ForConditionalGeneration。
- validation.md：阶段进展、命令、精度与性能证据及未完成项；文件名用英文，调优统计附录不新增CSV汇总。
- 必要的小型算子实验与回归测试；大权重、上游只读副本、AIR、原始日志放work，不提交。

测试放 tests/test_startlux_adapter.py。通用core/runtime原则上复用，确有框架缺陷时最小修复并回归。未经明确需要不重构Qwen、不改第三方子模块、不提交或推送。

## 8. 分阶段执行及门禁

| 阶段 | 任务 | 进入下一阶段的条件 |
|---|---|---|
| P0 | 固定版本、准备官方代码/模型材料、记录环境 | 可复现身份和输入明确，无凭据泄露 |
| P1 | kernel4卷积、双向注意力、DeltaNet的数值与AIR实验 | CPU/NPU对齐；关键算子AIR可由原生GESession执行 |
| P2 | 官方CPU golden与业务编码解码 | 真实choice/noul/score及None/宽列表fixtures通过 |
| P3 | batch1真实权重整模导出、GESession推理 | 逐层/最终logits对齐，业务结果通过 |
| P4 | 变长/批处理、重复调用/状态隔离 | padding及跨请求无污染、边界长度通过 |
| P5 | 完整输入池性能及必要优化 | 明确口径、无profiling正式复测、正确性不退化 |
| P6 | 图像与MRoPE | 单图/多图/图文混合，视觉特征和最终决策对齐 |
| P7 | 主agent审查与交付 | 源码、文档、命令、模型身份、精度、性能证据一致 |

P1具体矩阵：DeltaNet覆盖B=1/2、异长、63/64/65/127/128/129，真实16heads和D=128；full attention用Q=8/KV=2/D=256，验证末token影响前部、padding key不影响有效token；conv4验证权重方向、短序列和跨行隔离。先单算子再单层，失败保留最小复现，不直接启动大批量调优。

golden用官方双向FP32参考；增加BF16参考以区分量化与逻辑误差。初始建议门槛：26列logits cosine>=0.999、relative L2<=0.02；概率max abs error<=0.01；参考top1-top2>0.02的样本选择一致。初始门槛待实测确认并写入YAML，不能为掩盖错误直接放宽。结构、字段、对齐、None/选项顺序必须严格一致，无NaN/Inf。

## 9. 性能与部署

复用data→baseline→prepare→build→infer→compare→bench。准备依赖步骤可在全内置算子时为空；每个步骤只执行本步，不静默补跑。AIR内保存IOSpec，不自行引入通用清单/缓存体系。runtime运行不依赖原始权重；图编译、加载和预热独立计时，不纳入稳态吞吐。

先单实例，再扫描batch和实例数；统计read-out/s、原始问题/s、请求/s、有效token/s、P50/P95/P99、HBM、编译/加载时间。必须注明H2D/D2H与业务预后处理边界。多问题/宽列表请求不能把read-out数当请求数。正式结果无profiling，profiling单独采集并记录开销；不继承Qwen最佳核数/实例，不承诺未实测QPS。

## 10. 多模态阶段

沿官方AutoProcessor处理图像、占位符展开、视觉特征填入和get_rope_index，保留MRoPE语义。优先完整多模态AIR（视觉塔+特征填入+decoder+head）；若动态视觉图无法导出，拆vision AIR与decision AIR，通过模型专用编排层调用GESession。设置明确图像数量/分辨率/展开token上限，分别验收单图、多图、图文混合。必要D2H/H2D单独计量，不假设现有runtime已有设备端多图串联。

## 11. 交付与审查

交付模型适配源码、唯一YAML、AIR与依赖、轻量tokenizer/decision材料、精度/业务证据及性能报告。主agent逐阶段审查diff、代码、文档、原始日志和输出，独立复核关键计算与声明。所有未通过或未执行项如实记录，不把小图通过表述为整模通过。


## 官方 GE 图模式复核（历史记录，2026-10-30）

用户指出的官方文档明确包含 GE 图模式示例，不得将当前缺失 IR 报错概括为该算子不支持图模式。
API: https://www.hiascend.com/document/detail/zh/Pytorch/2610/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_chunk_gated_delta_rule.md
版本说明: https://www.hiascend.com/document/detail/zh/Pytorch/2610/releasenote/docs/zh/release_notes/release_notes.md
主 agent 用浏览器读取正文，确认：
- ENABLE_ACLNN=false，CompilerConfig.mode=max-autotune，torch.compile(... dynamic=False) 为 GE 示例。
- 示例 bs2,seqlen100，actual_seq_lengths 先 pad/cumsum 再去首元素，传入 [100,200] 累计结束位置。
- release table v2.12.0-26.1.0: PyTorch2.12.0，TorchNPU26.1.0，whl2.12.0，CANN9.1.0，Python3.10~3.13。当前基础版本组合符合官方表，不能未经证据归因为版本不配套。
- 算子新增接口表标注 CANN8.5.0及以后。
- 分别报告 eager / torch.compile GE / dynamo_export AIR / 原生GESession，不得混淆。
- 标准算子实现保留为数值对照及兼容路径，优先核清原生融合路径。
- 复核后进展：原生融合路径经兼容原型 + 显式局部加载验证通过（native-validation.md），原生整模 B1/S256 7/7（work/startlux-validation/native-full/）；标准路线整模 7/7（validation.md P3 节）。


## 实测驱动的验收补充

- 原生依赖：当前安装仅缺 chunk 的 GE 原型注册；项目自建兼容原型（不是官方文件）经未修改的真实 IR 查询、AIR、原生 GESession 和独立 NumPy FP64 逐 token 递推复核通过，B1/B2 输出相对误差约 0.00374。最小直接 vendor 路径曾完成自动注册，但正式 core.dependencies.environment 环境中 GE 初始化出现 SIGSEGV，尚不能将自动加载视为交付方式。当前采用已通过的显式局部加载：Python ctypes dlopen + native 子进程任务级 LD_PRELOAD，由模型环境脚本在 prepare 后设置；不改系统配置。正式整理与隔离测试已完成并通过（native-validation.md §4：真实 prepare/environment/activate、子进程 IR 探针、正式 env 下 B2 packed AIR 重导出、原生 GESession 数值/隔离/重入全过；eager/GE 见 B1/S256 整模 7/7，work/startlux-validation/native-full/）。
- 原生长度：当前安装的 kernel 源码和实测要求逐行长度，真正 packed 的 63/65 输入传 [63,65]；网页 GE 示例累计结束位置与本安装不一致。对于 flatten 的右 padding B*S 布局，原生 GDR 应传物理宽度 [S,S,...] 并在各行独立 mask/gather true L，不能把 true L 或累计结束位置直接套到含 padding 的扁平输入。正式 adapter 必须记录此版本边界。
- 数值稳定性：数学上等价的 Neumann 幂级数在高度相关 key 下不具备 FP32 数值稳定性。新增相关 key/高 beta/弱衰减门禁以及真实层输入对照；随机输入通过仅为小图证据。块三角逆方案的独立数值原型在同一压力集达到约 1e-7~6e-7 相对误差，已正式落地为 gated_delta.py 当前实现：真实 checkpoint 层输入对照通过，标准路线整模 CPU 复测与 GE B1/S256 7/7 通过；Neumann 幂级数求逆保留为历史记录（validation.md P1b 历史注记）。
- 量化与业务分支：固定同 prompt 的 26 logits 继续对照 FP32 oracle，原数值和概率门槛不变。真实多轮请求同时保留 FP32 与相同 dtype 的原始 HF 参考；BF16 本身可令宽列表 keep=3 边界的入围项改变。已有 27 选项示例终选相同，但按原 option ID 对齐的全分布 FP32/BF16 最大差异约 0.164864，不能声称通过 FP32 的 0.01 全分布概率门槛。部署多轮应对照同 dtype 官方语义，且单独如实报告 FP32 量化分支差异；不能只看最终 choice 相同就宣称全分布一致。

## 当前状态索引（2026-10-31；最终状态见 ../validation.md）

阶段状态以 [../validation.md](../validation.md) 为唯一权威记录。截至 2026-11-02
的最终状态：P0–P8 全部完成/通过——P0–P2 完成，P3 standard+native 两路线
B1/S256 整模 7/7 通过（主 agent 独立复算 7+7，review-p3-final-independent.json；
最大 active-K 概率误差 standard 0.0036457 / native 0.0051981），P4（动态
shape/B2B4、业务 GESession 回调、正式整合）与 P6（图像）已完成，P7 交付与
P5 性能均已验收（P5 由 P8 空闲窗口正式基准完成，B4 long 补验 4/4 通过）。
正式唯一 config 已切换统一图（`config/model.yaml`，unified_adapter）。本节
以下"进行中/未验收"表述为 2026-10-31 时的阶段历史：P4（动态 shape/B2B4、
业务 GESession 回调、正式整合）与 P6（图像）进行中，P5 性能与 P7 交付未验收。
上文早期"执行中/待"表述已按完成证据更新，历史记录保留原样；当时的"正式唯一
config 仍为标准路线（config/model.yaml），原生路线整合为 P4 任务"为阶段
历史。

## P6 单一多模态 AIR 原型验收补记（2026-10-31）

P6 统一图目标达成：单一 AIR（FP32 视觉塔 + 官方 masked_scatter 注入 + 官方
3 平面 MRoPE + 文本 decoder + 26 列读出头，1.91GB）经原生 GESession 一池
一次编译加载全部 12 cases / 38 行 gates 通过，门限未放宽（cosine 最差
0.9999964、relL2 最差 0.00627 vs 独立 HF FP32 oracle；matched-precision
oracle relL2 最差 0.01146；active-K 概率误差最差 0.00449；19/19 top1 一致），
单 AIR 换图隔离（图像行变 / 纯文本行 bitwise 不变 / 同池重复 bitwise 一致）
与官方 Decider 业务原型重放 7/7 choice 一致。专项记录见
`unified-validation.md`，证据与复跑命令见
`work/startlux-cannbot/p6-single-air-ready.json`。此为原型验收：正式生产
YAML/public callback/正式 CLI 的最后集成仍由主 agent 协调（config/model.yaml
未动），P5 性能待卡空闲，真实业务 public factory 在下一阶段验收。（阶段历史
注记，2026-11-02：该阶段后续均已完成——正式 config 已切换统一图并通过 P7 六
阶段 CLI/业务验收，P5 性能已由 P8 空闲窗口正式基准验收，见
[../validation.md](../validation.md) 与 [../performance.md](../performance.md)。）
