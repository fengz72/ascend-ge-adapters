# StartLux-Decision v1.1 0.8B Instant 优化点汇总

日期：2026-11-10。状态：**汇总索引**（只读整理，不引入新实测）。

本文把散落在 [../performance.md](../performance.md)（16 个附录实测表）与
[../validation.md](../validation.md)（27 条变更记录）中的调优点收敛为一张
总表：做了什么、收益多少、优化前后平均时延、当前是否采纳。**原始实测数据与
证据路径以上述两文件为准**；本文只做汇总与交叉核对，若与二者冲突以其为准。
采纳状态已用交付代码与 git 提交逐项复核（见文末「采纳状态的核对依据」）。

## 0. 统一口径

- 计时段为 warm graph **h2d + desc + execute + sync**，**不含 D2H**（bench 模式
  不回拷输出）与官方 CPU 预处理（在计时窗口外）。
- 时延列格式为 e2e **avg（P50）**，单位 ms，n=2048 请求；未记录 avg 的条目只给
  P50 并标注。
- 吞吐为 **graph executions/s**，闭环有放回抽样、seed 42、instances 1、warmup 32
  （预热不消耗测量序列）、2048 测量请求、0 错误。
- `graph executions/s` **不是完整业务 QPS**，`readout rows/s` 不是原始问题/s。
- 各附录分别在 chip 1/3/4/8 测得（同 Ascend910_9382 SKU）；交付态历史散布
  49.231–50.606 exec/s（~2.8%），**±0.5% 级差异一律记为方差带内，不当作收益**。

## 1. 已采纳的优化（按实测时间顺序）

| # | 优化点 | 优化前 avg(P50) ms | 优化后 avg(P50) ms | 吞吐变化 | 收益与依据 |
|---|---|---|---|---|---|
| 1 | `causal_conv4`：4 抽头移位乘加链 → 单个 **fp32 depthwise Conv2D** 节点（`padding=(0,3)`、groups=C） | 37.277 (36.589) | 34.168 (33.561) | 26.824 → **29.265**（+9.1%；vs P8 正式 +7.0%） | conv 微收益（0.05–0.25 ms × 18 层）叠加 GE elementwise 调度减少。**bf16 输入的 Conv1D/Conv2D 会错编译**（relL2 1.41 vs fp64 golden），fp32 输入正确（2.41e-3）；CANN 内置 FusedCausalConv1d/MaskedCausalConv1d 因 K 固定 3 不可用 |
| 2 | 视觉 patch-embed **Conv3d → 等价 MatMul**（`MatMulPatchEmbed`，stride==kernel 使 conv 退化为逐 patch 全连接，权重按引用持有） | 34.168 (33.561) | 27.784 (27.130) | 29.265 → **35.989**（+23.0%） | 驱动证据：Conv3DV2 节点 MTE1 饱和（mac 0.22 / mte1 0.97），固定 6854 µs/请求且纯文本请求也付，占全部 AI 时间 20%。替换后 Conv3DV2 从算子表消失，patch-embed 并入 MatMulV2（+~25 µs/请求）——该阶段 **~270× 降低**；ChunkGatedDeltaRule 升为最大 AIC 算子（11.0%） |
| 3 | 退役 256 最小物理宽度 → **case 按真长执行**（`min_width=1`，宽度 = max 行真长） | 29.562 (27.138) | 27.679 (25.532) | 33.824 → **36.125**（+6.8%） | 官方用例池全是短 prompt（256 下限下宽度利用率仅 33–51%）；B3 用例 −4.6 ms（三行共享收窄宽度）。前置门禁证明真长执行 **bit-stable**，P4 期「未补齐短宽执行边界不稳」的观察在本配置下不复现。精度/业务反而略好（补齐区数值噪声消失）。收益被每请求固定成本（视觉塔在单 AIR 中恒运行、host 派发）限制在 ~7% |
| 4 | 文本 decoder 的 eager RMSNorm 链（层输入/后注意力 ×48、body norm、full-attention q/k norm ×12、GDR gated norm ×18）→ **GE RmsNorm 节点融合**（`npu_rms_norm`） | 27.679 (25.532) | 25.122 (23.192) | 36.125 → **39.801**（+10.2%） | 计算域严格保持（fp32 mean 与 weight、输出 cast 回；gated 保持官方 weight/silu 顺序；gamma 从冻结参数常量折叠）。profiling 复采（`work/prof-rms`，32 个测量请求）：融合 RmsNorm 节点出现 3160 次 / 占 4.96%，替代原先**每请求** Square 115 次 + ReduceMean 79 次；Mul 15.8→11.5%、Cast 12.4→10.9%。**精度明显变好**（relL2 0.0043→0.0022，融合 fp32 归约比 eager 链更准） |
| 5 | 视觉塔线性内积 **fp32 → fp16**（patch-embed matmul、attention qkv/proj、MLP fc1/fc2、merger fc1/fc2；fp32 激活流、常量 buffer、norm、softmax 保持 fp32） | 25.122 (23.192) | 23.963 (22.218) | 39.801 → **41.727**（+4.8%） | profiling：MatMulV2 17.25→14.32%（171.3→145.7 ms，视觉份额减半）；Cast 10.93→14.35%（每 forward 的 fp16 权重 cast 未完全常量折叠，残余 ~0.9 ms/请求 → 由 #10 解决）。**bf16 先测先弃**：真实 12-block 深度下 relL2 0.055 vs fp16 0.0045，7 位尾数扛不住深度累积。文本主干保持 bf16（ChunkGatedDeltaRule 原型按 ops-info 契约只接受 bf16） |
| 6 | 融合 RmsNorm **直喂 bf16 激活**（混 dtype：bf16 self + fp32 gamma），消除 `x.float()`/`.to(bf16)` 显式往返 | 23.963 (22.218) | 23.168 (21.422) | 41.727 → **43.157**（+3.4%） | 省 ~120 Cast/请求（普通 norm 61 + GDR gated 18 组），由 Cast 实证分组（op_summary）驱动。前置 NPU eager 验证 kernel 内部归约为 **fp32**（直喂 bf16 1.66e-3 == 显式 fp32 域路径的输出量化水平），torchair converter 接受混 dtype。**精度与业务双双变好**（relL2 0.0066→0.0040、业务 0.0065→0.0028）——减少 Cast 跳数同时移除了传播路径上的量化边界 |
| 7 | RmsNorm **gamma 直接以 bf16 计算**（`1.0 + norm.weight` 在 bf16 域；gated 的 ones 同改） | 23.168 (21.422) | 未记录 avg（P50 22.117 / 21.673，同 AIR 重跑） | 43.157 → 41.84 / 42.54（**方差带重叠，无显著差异**） | 差分下钻（pbtxt 保留 + 两版对照导出）定案：前端图 `[1024]` Cast 为 **0**，该 cast 是 **GE 图编译期 tiling 选路插入的**，即 fp32 gamma 版的交付 kernel **一直在消费 bf16 gamma**，「fp32 gamma 精度」从未到达 kernel。故本改动数值等价、前端更干净、编译器少插 48 个 Cast；48 个 `[1024]` Cast 理论值仅 ~64 µs/请求 = 0.3%，被方差淹没 |
| 8 | 删除 GDR 路径**冗余 pad 掩码乘法**（`mixed * pad` 与层内 `q/k/v * pad`、`beta * valid`，合计 ~90 Mul/请求，P0-1） | 未记录 avg（P50 22.117–21.673） | 22.089 (20.232) | 41.840–42.536 → **45.265**（+6.4~8.2%） | 依据：chunk 递推严格因果，pad token 只影响自身及之后位置与最终 state，而 readout gather 真长 L-1、state 丢弃 → per-row padding 永不泄漏。**实证**：删除后数值/业务指标与删除前**六位有效数字完全一致**（relL2 0.002798 / 业务 0.007095），直接证明掩码确为 no-op。收益与移除的 ~1.5–1.9 ms/请求 Mul 工作量相符 |
| 9 | GDR q/k **L2 归一链 → 每侧一个 RmsNorm 节点**（18 层 × 2 侧，P0-2） | 22.089 (20.232) | 20.270 (18.571) | 45.265 → **49.327**（+9.0%） | `rms_norm(x, gamma=1/sqrt(D), eps=1e-6/D)` 与 `x·rsqrt(sum(x²)+1e-6)` **精确等价**（mean 域折叠的 sqrt(D) 在 gamma 中抵消）。消除 ~72 次 `[B,S,16,128]` bf16↔fp32 Cast + 显式 Square/ReduceSum/Rsqrt/Mul 链。**scale 折叠方案被实测否决**：ones gamma + kernel scale D^-1.5 在真实单层探针 relL2 0.85（delta 更新对 q/k 模长非线性耦合，模长折叠必须放 gamma）；kernel scale 保持 `Dk**-0.5` 不变。精度 0.0028→0.0039（bf16 gamma 的 ~1e-4/侧舍入经 18 层传播，仍 5× 门限余量），业务变好 |
| 10 | **预量化视觉权重为 fp16 常量**（wrapper 构造时原位 `param.data.to(fp16)`，P2-5） | — | —（bench 49.00，方差带内） | 49.327 → 49.00（方差带内） | 关键手法：**原位替换 dtype 使导出图的权重 Const 数量不变**，torchair positional bake 保持对齐（此前 51 个**额外** buffer Const 才是 bake 冲突根源，见 §2）。数值与 P2-5 前**逐位一致**（relL2 0.003857 / 业务 0.003283）。profiling：总 Cast **501 → 362/请求**（vision `[512,768]` 权重 cast 36 次及关联激活 cast 已消），残余为 norm 残差域与 GDR l2norm 域的 upcast 无损最小集 |
| 11 | **融合 RoPE** `npu_apply_rotary_pos_emb`（6 层 full-attention 的 eager rotate-half 链，每侧每层 ~10 小算子，P2-7） | 未记录 avg（P50 18.594） | 未记录 avg（P50 18.586） | 49.003 → **49.231**（+0.5%，持平略正、方差内） | 交付形态：RoPE 在**天然 `[B,S,N,D]` 域**（transpose 前）以 **BSND** 调用（q/k `[B,S,N,rot]` 切片、cos/sin `[B,S,1,rot]`、**零 reshape**）；`rotary_mode="half"` 即精确 rotate-half 数学。GE 图 + BSND 对**纯数学参考** relL2 0.0002，覆盖用例池全部 B/S 组合。eager 腿保留显式数学（aclnn eager 的 layout=1 按 BSH 解读、非 BSND，自身即错 0.88）。**收益是结构性的**：~120 节点/请求、AIR **1906 → 1708 MB（−10%）** |
| 12 | **融合 full-attention** `npu_fusion_attention`（6 层双向 attention 的 eager 链：GQA k/v expand、scores matmul + 加性 mask、fp32 softmax、probs matmul，每层 ~15 节点，P2-8） | 未记录 avg（P50 18.586） | 未记录 avg（P50 18.300 / 复测 17.885） | 49.231 → **49.809 / 50.942**（+1.2~3.5%） | 每层一次 FA 调用：bf16 **BNSD**、bool `[B,1,S,S]` pad mask（True=屏蔽，与加性 min-score mask 等价、图内由加性 mask 派生）、FA 原生消费 GQA 故 k/v unsqueeze-expand-reshape 消失。probe（纯数学参考）：relL2 0.0023/层、eager 链 **×2.05**（B1/S99）。probe 级 ×2.05 压缩到整模 ~1–3%，因 6 层 attention 仅占 e2e ~0.5 ms。精度 0.0039→0.0079（FA 内部 probs@v 路径，仍 2.5× 门限余量），业务略好 |
| 13 | **WeightNzAndMatMulV3Pass** 自定义 GE 融合 Pass（`third_party/custom_development_code/fusion_pass/`，`prepare.sh` 固化构建+安装） | 单 case AI 17,642 µs/执行 | 单 case AI **15,092 µs/执行** | 混合闭环 50.481 → 50.606（+0.3%，方差带） | 在图编译期（`kAfterOriginGraphOptimize`）把 MatMul/MatMulV2 的常量权重重排为 **FRACTAL_NZ** 并重建为 **MatMulV3**——host 期转换、**无运行时布局 cast**、激活与输出保持 ND。MatMulV2 3,115 µs 整体替换为 MatMulV3 **626 µs（−80%）**，文本投影链最大受益，单 case AI 时间 **−14.4%（−2.5 ms/执行）**。混合闭环只 +0.3% 是因为该延迟下闭环 host-bound，**单 case AI 时间才是 kernel 级真相**。注意：安装后首次图编译按 matmul 形状逐一 TBE 编译（首跑需较长超时，`kernel_meta` 缓存后续免）；系统级安装至 CANN vendors 目录 |

## 2. 已否决 / 回退的实验（保留为调查证据，**均未采纳**）

| 实验 | 优化前 → 后 | 结果与根因 |
|---|---|---|
| **MegaGdnPrefillV2** 整块融合 GDR 路线（`gdn_route=mega`，vendored custom-op-v2） | 37.277 (36.589) → **48.670 (46.843)** ms；26.824 → **20.545** exec/s | **−23% 吞吐**、B1 +28% / B2 mixed **+51%**（P50 65.2 vs 43.2 ms）；业务门禁 **8/11 FAIL**（三个图像用例 `prob_max_abs` 0.0123/0.0132/0.0147 > 0.01）。根因：融合算子内部按 **128 对齐**设计，本模型 shape 偏小（hidden 1024、Nk=Nv=16/Dk=Dv=128、有效长度多在 S256 量级），融合收益为负。**同日回退**，代码后于 2026-11-09 移除 |
| **FIA** `fused_infer_attention_score` 替换 6 层 full-attention（BNSD、原生 GQA Q8/KV2/D256、per-row `actual_seq_lengths`、sparse_mode=0） | 34.168 (33.561) → **35.398 (34.521)** ms；29.265 → **28.248** exec/s | **−3.5%**、+0.96 ms P50；精度也略差（relL2 0.0040→0.0068、业务 0.0055→0.0063，均在门限内）。根因：PA kernel 的短序列调度开销 + q/k/v 连续拷贝 + 输出转置，超过移除每层 ~15 节点 eager 链的收益。**回退**，建议仅在负载转向长序列时复测 |
| **FIA 复测**（在 P2-8 已上线 FA 的布局链上，三变体：per-row lengths / bool mask / 无 mask） | 单算子 B1/S99：FIA **0.182 ms** vs FA **0.045 ms** vs eager 0.084 ms | FIA 比 FA **慢 4×**、比 eager 慢 2.2×。PromptFlashAttention kernel 在短序列（S≤130）调度开销主导，与早期 FIA 整模调查（−3.5%）根因一致。数值侧 lengths/boolmask 变体 relL2 0.0023–0.0024（与 FA 同级）✓、无 mask 在有 pad 时 0.36–0.50（预期内）。**不可行，维持 FA** |
| **FA 布局变体 BSND / BSH**（目标省每层 4 个 transpose） | BSND 单算子 ×1.00；BSH 单算子 **×1.12** | **BSND**：eager 数值与 BNSD 逐位一致，但 **torchair converter 只支持 BSH/BNSD**（`input_layout != BSH & BNSD` 直接 NotImplementedError），图模式不可用。**BSH**：`[B,S,N*D]` 纯 reshape 零转置且略快，但 **GQA 推断失败**（relL2 0.99997）——H 拼接维无法区分 2 heads×256 与 8 heads×64，FA 按 head_num=8 错切 k/v；expand 到 8 heads 再传则开销回到原点。**维持 BNSD + transpose** |
| **视觉塔 FA 变体** | 单算子 **×0.99** | 数值精确但无收益：fp32 短 chunk 是 **device-bound 而非 host-bound**。**先测先弃**（在 P2-8 之前即排除） |
| **fp16 残差流**（全塔 fp16） | (22.218) → **(22.684)** P50；41.727 → **40.908** exec/s | **−2.0%**。逐层 fp16↔fp32 往返 Cast 的代价超过激活流量减半的节省。数值侧本身是过的（真实塔 relL2 0.0051、bit-stable、整模 5/5+5/5 与混合基线同指标）——**纯性能净负**。**回退**，交付结构保持 fp32 残差流 + fp16 线性内积 |
| **预物化 fp16 权重副本**（常量折叠每 forward 的权重 Cast，目标收回 ~0.9 ms/请求） | — | **导出即受阻**：torchair 的 positional weight bake（runtime input i 替换 `export_graph.op[i]`）在 ~51 个**新增** fp16 权重 Const 进入 op 列表时错位（"incomplete exported Data binding"）；小模型导出成功、完整动态导出失败。**受阻**；后由 #10 的**原位 dtype 替换**（Const 数量不变）达成同一目标 |
| **GDR/attention 投影合并**（GDR 四投影 qkv/z/b/a 与 FA 三投影 q/k/v 构造时预拼接为单权重，单 matmul + 切片，目标消除 ~54 次/请求小 matmul 启动，P2-6） | 49.00 → **45.78** exec/s | **−6.6%，净负回退**。profiling 定位：合并后的 `[B,S,8224]` 切片成为 **StridedSlice 新大头**（~53 次/请求大切片 + `[B,S,6144]` 子切片链），且非连续视图向 conv4/transpose 下传使 **TransData 翻倍（307→787 µs/请求）**、MatMulV2 时间上升——切片与非连续代价远超 matmul 启动节省。结论：分开的 Linear 输出各自连续，是该图结构下的更优形态 |
| **GDR conv Conv1D 化**（`F.pad(x32,(3,0)) + Conv1D(groups=C)`，目标消除输出切片 StridedSliceV3 358 µs/e 与 conv 的 NCHW TransData 377 µs/e） | 单 case AI 15,092 → **15,572 µs/e**；bench −3%（49.0） | **+480 µs 净负，回退**。门禁 5/5+5/5 是过的，但 profiling 定位：torchair 把 `aten.conv1d` 转换为**同一 Conv2D 节点**，unsqueeze→TransData 依旧（pad 后宽度 102 使 TransData 反而更大，376 µs）；`F.pad` 引入新的 fp32 全宽拷贝（`[1,6144,102]`×18 层，Cast 组 +209 µs）；输出切片确实消失（StridedSliceV3 1383→965 µs）但被前两项抵消反超。结论：GE 的 conv1d/conv2d 共享实现 + F.pad 拷贝代价使该方向不成立 |
| **融合 RoPE 的 TND / reshape 变体**（P2-7 定位过程中的两轮 probe） | 动态图 relL2 **~0.88–0.89**（语义级全错） | 本 CANN 构建的 GE 原生 ApplyRotaryPosEmb tiling 仅接受 BSH(1)/TND(4)、拒 BNSD(3)；**BSND 源 + `reshape(-1)`/`reshape(B*S)`（qwen 同构）在动态图下 B1/B3/S85/99/130 全错**，BNSD 源 + transpose→reshape 则 B3/S99 逐位正确而 **B1 全错**；aclnn eager 两种布局全对、静态图（trace=执行形状）全对。qwen2.5-0.5b 的先例是**真 varlen 单动态轴**（q 全程 TND、无 B 维概念），与本项目的 **B×S 双动态轴**不同。**变体否决**；最终交付形态见 §1 #11（天然 BSND 域、零 reshape，对纯数学 relL2 0.0002 全 B/S 正确） |

## 3. 非性能类变更（不影响数值/时延，列出以免与优化混淆）

| 变更 | 日期 | 性质 |
|---|---|---|
| 用例池 11 正式 → 16（加官方 examples）→ **仅官方 5 个** | 2026-11-06 | **工作负载定义变化，不是优化**。16 用例池 −3.9% exec/s 但 +4.6% rows/s（B3 三行用例进入等权闭环）；仅官方 5 用例池 rows/s **48.160**（+17%，官方分布平均 1.42 行/请求、B3 占闭环抽样 21.2%），exec/s 33.824。**各 B1 用例的每请求时延在池间不变** |
| 收敛为 native chunk GDR **单路线**（删纯 Torch 标准路线） | 2026-11-02 | 结构决策。接受 CANN 升级/换 SoC 时无标准路线兜底的风险 |
| `ge_runtime` 子进程执行流程去重（三处 → 一处） | 2026-11-02 | 代码结构 |
| 目录重组（`model/` 与 `decider/`） | 2026-11-05 | 代码结构，运行产物路径不变 |
| 独立脚本流程迁移（`run.sh` 串行调度，去 `core`/YAML/adapter/handle） | 2026-11-05 | 交付形态。迁移验收实测 26.824 exec/s、P50 36.589 ms（与 P8 正式 27.342/35.903 同在方差带） |
| 代码审查清理（删死代码 / 修错误 / 削冗余校验 / 注释去过程化） | 2026-11-10 | 声称无数值变化，**已 NPU 复测证实**：worst relL2 0.00449550810581621 与记录的 0.004496 逐位复现，AIR 两次独立导出均 1,708,279,731 bytes |
| 兼容原型加载收敛为**单一 `LD_PRELOAD` 契约**（删 python activator） | 2026-11-10 | 加载机制重构，**无数值/算子路径变化**（`.so` 编译命令与图内容均未动）。最小 case 复验：`.so` 重建 ok、guard 无 `LD_PRELOAD` 时正确报错、`ge_runtime` CompileGraph ok、cosine 0.99999976 / relL2 0.001033 |

## 4. 整体轨迹

| 阶段 | 吞吐 exec/s | e2e avg ms | e2e P50 ms | rows/s |
|---|---|---|---|---|
| **P8 正式基准**（2026-11-01，唯一正式实测；11 用例池、pad-256、全 eager、Conv3d patch-embed、移位链 conv） | 27.342 | 36.572 | 35.903 | 29.825 |
| **当前交付态**（2026-11-10 清理后复测；5 官方用例、真长执行、全部已采纳优化在位） | **49.489** | **20.204** | **18.785** | **70.464** |
| 累计 | **+81.0%** | **−44.8%** | **−47.7%** | **+136%** |

**同口径纯优化链**（固定为「仅官方 5 用例池」，剔除池变化影响）：

| 起点 | 终点 | 吞吐 | e2e avg |
|---|---|---|---|
| 官方 5 用例池 + pad-256（#3 之前） | 当前交付态 | 33.824 → 49.489（**+46.3%**） | 29.562 → 20.204 ms（**−31.7%**） |

逐项累乘校验（exec/s，取各步实测比值 #3 ×1.068、#4 ×1.102、#5 ×1.048、#6
×1.034、#7+#8 合并 ×1.049、#9 ×1.090、#10 ×0.993、#11 ×1.005、#12 ×1.023（取
1.012~1.035 中值）、#13 ×1.003）：33.824 → **50.51**，落在实测终点区间 49.489
（最新复测）～50.606（NZ Pass 附录峰值）之内，与 ~2.8% 的历史散布一致。

## 5. 必须随表引用的口径限制

1. **跨池对比不可直接读作纯优化收益**：P8 正式基准是 11 用例 + pad-256，当前是
   5 官方用例 + 真长。§4 的「累计 +81.0%」混入了工作负载定义变化；**#3（真长
   执行）是优化，池收敛不是**。要只看优化，用 §4 的同口径链（+46.3%）。
2. **跨卡与方差**：各附录在 chip 1/3/4/8 测得（同 SKU）。#1 的 +9.1% 是同日同
   卡类对比，明显高于旧 conv 上观察到的 run-to-run 方差（26.82 / 27.13 / 27.34）。
   交付态历史散布 49.231–50.606（~2.8%）。
3. **2026-11-10 复测的 49.489 落在散布低端**（−2.2% vs NZ Pass 附录、+0.5% vs
   chip-8 融合 attention 附录）：测量时 16 卡中 6 卡（4/5/12/13/14/15）被其他
   租户占至 ~65 GB HBM，host 侧争用是可能原因，**未归因于清理**（精度逐位复现
   是无回归的最强证据）。
4. **闭环 host-bound**：#13 的混合闭环只 +0.3% 而单 case AI 时间 −14.4%，说明在
   当前延迟量级下闭环吞吐受 host 侧限制，**kernel 级收益要用单 case AI 时间看**。
5. **导出非字节可复现**：AIR 字节数在不同导出间会变（融合前 1,907,534,423 /
   1,906,330,769，融合后稳定 1,708,279,731）；嵌入的 IO 契约经 runtime `read_io`
   核对。README 曾长期携带过期的「字节数恒定 1,907,534,423」，已由 2026-11-10
   复测修正。
6. **精度门限**：逐行 cosine ≥ 0.999、relative L2 ≤ 0.02；业务门禁为 GE vs
   **相同 dtype 官方 BF16 动态参考**的 active 概率 |Δ| ≤ 0.01。已采纳优化的最差
   记录为 relL2 0.007893（#12，2.5× 余量）；wide_27 末轮 FP32 第 6 名 cause_18
   vs BF16/GE cause_23 属跨 dtype 已知翻转区，按披露处理不计失败。

## 6. 采纳状态的核对依据

采纳状态**不以文档叙述为准，而以交付代码 + git 提交为准**，逐项复核结果：

| 优化点 | 代码/提交证据（2026-11-10 模块化重写后的新位置；历史位置见 git 历史） |
|---|---|
| #1 fp32 Conv2D conv | `model/modeling_text.py` `def causal_conv4`（fp32 depthwise Conv2D + `padding=(0,3)` + 输出切片），调用于 `StartLuxGatedDeltaNet.forward` |
| #2 MatMul patch-embed | `model/modeling_vision.py` `class MatMulPatchEmbed`（参数名 `proj` 镜像 checkpoint） |
| #4/#6/#7/#9 RmsNorm 系列 | `model/modeling_text.py` `StartLuxRMSNorm.forward`（gamma=(1+w)）、`StartLuxRMSNormGated.forward`（gamma=w，ones）、`StartLuxGatedDeltaNet.forward`（GDR q/k L2norm，`l2_gamma` buffer） |
| #5 fp16 视觉线性 | `model/modeling_vision.py` 各 Linear/Conv 声明处 `dtype=torch.float16` + `_fp16_linear`（fp32 残差流内积降精度） |
| #11 RoPE | `model/modeling_text.py` `StartLuxAttention.forward`：**融合算子回切（2026-11-10）**——`npu_apply_rotary_pos_emb(..., layout="BSND", rotary_mode="half")`；当日 probe 证明本构建两腿（aclnn eager / torchair 图）逐位一致（relL2 0.0013 vs 纯数学，六种 B/S 含 B1），历史 layout=1 语义错位不复现；详见 validation.md 2026-11-10 条目 |
| #12 融合 FA | `model/modeling_text.py` `StartLuxAttention.forward`：`npu_fusion_attention(..., "BNSD", atten_mask=pad_mask, sparse_mode=0)`，单腿无分支 |
| #13 NZ Pass | `prepare.sh:43-47` 构建 + `cmake --install` 至 `${ASCEND_OPP_PATH}/vendors/custom_nz_pass/custom_fusion_passes/` |
| #11 + #12 的提交 | `af9e05a perf(startlux): fused RoPE and fused full-attention (P2-7, P2-8)`，正文明载 "a botched git-checkout during the BSND experiment had silently reverted both patches; **this commit re-lands them** (gates re-passed)"，并给出 "Official 5-case bench: 49.2 -> 50.5 exec/s, e2e P50 18.6 -> 18.0 ms" |
| mega 路线已移除 | `76f29cf refactor(startlux): remove the mega GDR route code` |

### 待修正的文档失实（**尚未修改**）

`../validation.md` 的「2026-11-09 变更：融合 RoPE 实验回退（P2-7，GE 原生算子
动态 batch bug）」条目末尾仍写着「**结论与回退（维持）**：… P2-7 回退，eager
RoPE 链保留」，与交付态矛盾：

- 同条目内的「**复活与定案**」段已给出交付形态（天然 BSND 域、零 reshape、对纯
  数学 relL2 0.0002）与 bench 49.231 exec/s；
- 其后的「补充证据（二轮 probe）」测的是 **BSND 源 + `reshape(-1)`** 的 qwen 同构
  变体（动态图全错），**与交付形态不是同一变体**；
- 提交 `af9e05a` 已 re-land，代码 `:292-305` 在用；
- `../validation.md` 的 2026-11-10 条目与 `../performance.md` 的 post-cleanup 附录
  都把 AIR 1906→1708 MB（−10%）归因于「融合 RoPE / 融合 full-attention」。

即该「回退（维持）」结论是**复活定案之前的旧结论，未随 re-land 更新**。本文
§1 #11 按代码实况记为**已采纳**，§2 末行只把 *TND / reshape 变体* 记为否决。

## 7. 证据索引

| 优化点 | performance.md 附录 | validation.md 条目 | profiling |
|---|---|---|---|
| P8 正式基准 | 正文「最终结果」 | 「正式精度验收」 | — |
| 独立流程迁移验收 | standalone-flow validation measurement (2026-11-05) | — | — |
| mega 融合路线 | mega fused-GDR route measurement (2026-11-05, route rolled back) | 2026-11-05 MegaGdnPrefillV2 切换尝试与同日回退 | — |
| #1 fp32 Conv2D conv | fp32-Conv2D causal_conv4 replacement measurement (2026-11-05) | 2026-11-05 causal_conv4 替换为 fp32 depthwise Conv2D 节点 | — |
| FIA 整模 | FIA full-attention replacement measurement (2026-11-05, net regression) | 2026-11-09 FIA 复测（P2-8 补充） | — |
| #2 MatMul patch-embed | MatMul patch-embed replacement measurement (2026-11-06) | 2026-11-06 视觉 patch-embed Conv3D 替换为等价 MatMul（含 FIA 调查回退） | `work/prof-fia`（前）、`work/prof-matmul`（后） |
| 用例池 11→16 | official-examples case pool measurement (2026-11-06) | 2026-11-06 输入池扩充官方 examples case | — |
| 用例池 →仅官方 5 | official-only case pool measurement (2026-11-06) | 2026-11-06 case 池收敛为仅官方 examples | — |
| #3 真长执行 | true-length execution (min_width=1) measurement (2026-11-06) | 2026-11-06 退役 256 最小物理宽度 | — |
| #4 RmsNorm 融合 | RMSNorm fusion + short-width profiling re-collection (2026-11-06) | 2026-11-06 文本 RMSNorm 融合为 GE RmsNorm 节点 | `work/prof-rms` |
| #5 fp16 视觉线性 | fp16 vision linear compute measurement (2026-11-06) | 2026-11-06 视觉塔线性计算域 fp16 | `work/prof-fp16` |
| fp16 残差流 / 权重物化 | fp16 full-tower follow-up experiments (2026-11-06, both rolled back) | 2026-11-06 fp16 全塔后续实验（均回退） | — |
| #6 RmsNorm 直喂 bf16 | RMSNorm direct-bf16 input measurement (2026-11-07) | 2026-11-07 RmsNorm 直喂 bf16（混 dtype，省 ~120 Cast/请求） | — |
| #7 gamma 直喂 bf16 | —（仅 validation.md） | 2026-11-09 RmsNorm gamma 直喂 bf16（差分下钻定案） | — |
| #8 pad 掩码删除 | GDR pad-mask removal measurement (2026-11-09) | 2026-11-09 删除 GDR 路径冗余 pad 掩码乘法（P0-1） | — |
| #9 GDR q/k L2norm | GDR q/k L2-norm fused as RmsNorm measurement (2026-11-09) | 2026-11-09 GDR q/k L2norm 融合为 RmsNorm（P0-2） | — |
| #10 预量化视觉权重 | —（仅 validation.md） | 2026-11-09 预量化视觉权重为 fp16 常量（P2-5） | — |
| 投影合并（否决） | —（仅 validation.md） | 2026-11-09 GDR/attention 投影合并实验（P2-6，净负回退） | — |
| conv Conv1D 化（否决） | —（仅 validation.md） | 2026-11-09 GDR conv Conv1D 化实验（净负，回退） | — |
| #11 融合 RoPE | fused RoPE via ApplyRotaryPosEmb measurement (2026-11-09) | 2026-11-09 融合 RoPE 实验回退（P2-7）**← 结论段失实，见 §6** | — |
| FA 布局变体（否决） | —（仅 validation.md） | 2026-11-09 FA 布局变体测试（BSND/BSH 均不可行，维持 BNSD） | — |
| #12 融合 FA | fused full-attention via npu_fusion_attention measurement (2026-11-09) | 2026-11-09 full-attention 融合为 npu_fusion_attention（P2-8） | — |
| #13 NZ Pass | WeightNzAndMatMulV3Pass measurement (2026-11-09) | 2026-11-09 安装 WeightNzAndMatMulV3Pass 自定义融合 Pass | `work/prof-nz` |
| 清理后复测 | post-cleanup verification run (2026-11-10) | 2026-11-10 代码审查清理 | — |
| LD_PRELOAD 单一契约 | —（仅 validation.md） | 2026-11-10 兼容原型加载收敛为单一 LD_PRELOAD 契约 | — |

原始 run 产物在模型内 `work/{bench,infer,logs,prof-*}/`（gitignored）；P8 正式
基准的原始证据与主 agent 独立复核路径见 `../performance.md` 的「证据路径」。
