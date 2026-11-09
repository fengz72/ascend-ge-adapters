# StartLux-Decision v1.1 0.8B Instant 验证记录（当前状态）

环境身份：容器 hejun-cann910-jev，CANN 9.1.0，Python 3.12.13，torch 2.12.0+cpu，
torch_npu 2.12.0，transformers 5.8.1，设备 Ascend910_9382，runtime 二进制
`runtime/build/ge_runtime`。GE Session：
`graph_run_mode=0`、`precision_mode=must_keep_origin_dtype`。

最终交付（已验收）：视觉塔 + 图像特征注入 + 三轴 MRoPE + 文本 decoder + 26 列
letter 读出头整合于**单一 AIR、单次 GESession**。正式 config 唯一：
`config/model.yaml`（unified_adapter）。AIR/runtime 身份：

- AIR `work/startlux-decision/build/model.air`：1,907,534,423 bytes，sha256
  `c1f8816ee25e03119b2522e55adb49ebf42a88692791b179920f5f60221edba1`
- runtime `runtime/build/ge_runtime`：639,952 bytes，sha256
  `692996b5df8a7ed8fdebdc365cdaa9383b6b538868ecf3e019303930d178eecc`

## 正式精度验收（全部通过）

- **六阶段 CLI**（data/baseline/prepare/build/infer/compare，正式
  `config/model.yaml`）：11 cases / 12 rows，min cosine 0.9999985326 / max
  relL2 0.0030342605 / 业务 active 概率 max 0.0055074189。主 agent 独立复核：
  `work/startlux-cannbot/review-p7-cli-independent.json`；AIR 结构独立核对
  （单图 + critical weights）：`work/startlux-cannbot/review-p7-air-structure-independent.json`。
- **官方业务验收**：13 请求 17 真实 GE 读出，证据
  `work/startlux-validation/p7-final/`。两种判据口径不同：
  - GE vs 相同 dtype 官方 BF16 动态参考（部署级判据）：**17/17 全过**，worst
    cosine 0.9999968 / relL2 0.00947 / 概率 |Δ| 0.00985；重入与 finalize 亦过。
  - GE vs 官方 FP32 同 prompt：**16/16 过**（17 读出中 16 有同 prompt FP32
    匹配）；wide_27 末轮 FP32 第 6 名 cause_18 vs BF16/GE cause_23，属跨
    dtype 已知翻转区，按披露处理不计失败，不称全 FP32 业务分布门限通过
    （`work/startlux-validation/p7-final/compare-report.json`）。
- **边界矩阵**：18 cases / 27 rows 数字门限全过（min cosine 0.9999509 / max
  relL2 0.0153549；B1 S<=2048、B2 long [129,2048]、B4 short 及同会话未改动行
  三组 bitwise 全过）：
  `work/startlux-cannbot/review-p7-boundary-independent.json`。
- **B4 long [1,65,129,2048] 补验（已通过）**：P7 时期共享卡资源不足真实 OOM；
  2026-11-01 P8 空闲窗口以 fresh GESession 补验 4/4 行全过（vs 独立原官方
  CPU 金标：min cosine 0.9999947830951117 / max relL2 0.0040037066209951165，
  gate wall 56.006 s）。**OOM 已解决**；P7 OOM 历史证据保留于
  `review-p7-boundary-independent.json`，不删除。

## CPU 单测

主 agent 2026-10-31 独立重跑 13 个相关模块（六基础模块 + 统一图五模块 +
multimodal + torchair exporter 门禁）共 **317 项全部通过**：
`work/startlux-cannbot/review-p7-unittest-final.log`（27.8s）与
`review-p7-unittest-final-state.log`（Ran 317 tests in 27.635s，OK）。清理
（2026-11-02）后主 agent 再次复跑 317 项全部通过：
`work/startlux-cannbot/cleanup-20261102/unittest.log`（Ran 317 tests in
 28.531s，OK）。全仓套件另有 4 个预存 error（上游 onnx 缺包
 ModuleNotFoundError，非本模型引入），不据此宣称全 repo 通过。

## 2026-11-05 变更：目录重组（model/ 与 decider/）

**决策**（用户拍板）：模型根下的散置 Python 文件按职责归组——

- **`model/`**（模型构建与导出）：`model.py`、`startlux_npu_adapter.py`、
  `multimodal.py`、`vision.py`、`unified_preprocess.py`、`export_air.py`；
- **`decider/`**（decider 业务语义与官方管线）：`decision.py`、
  `reference.py`、`golden.py`、`ge_readout.py`、`unified_readout.py`、
  `verify_business.py`、`fixtures/`、`vendor/`（官方 startlux_decide
  逐字节子集）；
- 根目录保留流程入口与基础设施：`run.sh`、`prepare.sh`、
  `generate_inputs.py`、README/validation/performance、`docs/`、`scripts/`、
  `native_proto/`、`work/`。

实施：git mv 保留历史；所有文件的 `MODEL_DIR` 语义改为模型根
（`parents[1]`），`_load_module` 调用改带 `model/`/`decider/` 前缀，
`vendor`/`fixtures` 引用改 `decider/` 子路径（含 adapter `_params` 的
`fixtures_file` 默认值）；`run.sh` 三处阶段路径、tests 7 个文件路径同步。
验证：全部 startlux CPU 单测 242 项通过；数据阶段冒烟通过
（`generate_inputs.py` 12 cases / `decider/golden.py` 16 rows /
`model/verify_business.py` 11/11 PASS）；`model.py`/`export_air.py` import
链冒烟通过。运行产物路径（`work/`、AIR、bench 请求）不变。

## 2026-11-09 变更：GDR q/k L2norm 融合为 RmsNorm（P0-2）

**决策**（用户拍板）：GDR 的 q/k L2 归一链（18 层 × 2 侧的
`x.float()→mul/rsqrt(sum+1e-6)→.to(bf16)`）各融合为一个 RmsNorm 节点：
`rms_norm(x, gamma=1/sqrt(D), eps=1e-6/D)` 与 `x·rsqrt(sum(x²)+1e-6)`
**精确等价**（mean 域折叠的 sqrt(D) 在 gamma 中抵消；kernel 内部 fp32
归约、bf16 输出；bf16 gamma 带 ~1e-4 相对舍入）。

- **scale 折叠方案被实测否决**：ones gamma + kernel scale D**-1.5 的变体
  在真实单层 NPU 探针 relL2 0.85（delta 更新对 q/k 模长非线性耦合，
  模长折叠必须放在 gamma 而非 scale）；scale 网格 {D^-1.5, D^-1, D^-0.5,
  1} 全不匹配；**gamma=1/sqrt(D) + scale 保持 D^-0.5 → relL2 0.0029**
  （量化级）定案。首次整模门禁曾因 scale 折叠失败（全 case relL2
  0.22–0.64），修正后通过。
- 整模门禁（`run.sh --device 3`）：数值 5/5（worst cosine 0.999998 /
  relL2 0.003857——bf16 gamma 的 1e-4/侧舍入经 18 层传播，仍 5× 门限
  余量）；业务 5/5（worst 0.003283）。
- bench（2,048 请求 0 错误）：**49.327 exec/s、70.233 rows/s（+9.0%）**；
  e2e P50 **18.571 ms**（−1.7 ms）；B1 ticket 18.5 / image 19.5 / B3
  25.9 ms。消除 ~72 次 [B,S,16,128] cast + 显式归约链。测量表见
  performance.md 附录。
- CPU 单测：新增 gamma 折叠等价 1 项（公式含 gamma=1/sqrt(D)），全套
  通过。
- **register_buffer 化**（后续微调）：`gdr_l2_gamma` 改为 decoder 的
  非持久 buffer（`persistent=False`，随 `.to(device)` 迁移、图内一次
  常量折叠，替代每层 `torch.full`）——导出无 bake 冲突（单个常量
  buffer，与上次 51 个权重 buffer 的失败量级不同），整模门禁数值与
  torch.full 版**逐位一致**（0.003857 / 0.003283）；bench 48.26
  exec/s（方差带内，常量化收益 <0.5% 属预期）。

## 2026-11-09 变更：FA 布局变体测试（BSND/BSH 均不可行，维持 BNSD）

**测试**（用户提议）：P2-8 的 FA 若改用 BSND（q/k/v 保持天然 [B,S,N,D]
域、省每层 4 个 transpose）或 BSH（纯 reshape）是否更快。

- **BSND**：eager 数值与 BNSD 逐位一致、单算子 x1.00（同速）——但
  **torchair converter 只支持 BSH/BNSD**（`input_layout != BSH & BNSD`
  直接 NotImplementedError），图模式不可用。
- **BSH**：[B,S,N*D] 纯 reshape 零转置、单算子 x1.12 略快——但 **GQA
  推断失败**（relL2 0.99997）：BSH 的 H 拼接维无法区分 2 heads×256 与
  8 heads×64，FA 按 head_num=8 错切 k/v。expand 到 8 heads 再传则引入
  的开销回到原点。
- **结论**：BNSD + transpose 维持现役。

## 2026-11-09 变更：FIA 复测（P2-8 补充，不可行维持 FA）

**复测**（用户提议）：在 P2-8 已上线 FA 的布局链上复测
`npu_fused_infer_attention_score`（FIA/PromptFlashAttention）三变体
（per-row lengths / bool mask / 无 mask，bf16 BNSD GQA 8/2，真实 B/S
组合，纯数学参考）：

- 数值：lengths 与 boolmask 变体 relL2 0.0023–0.0024（与 FA 同级）✓；
  无 mask 在有 pad 时 0.36–0.50（无屏蔽，预期内）。
- **性能（B1/S99）：FIA 0.182 ms vs FA 0.045 ms vs eager 0.084 ms——
  FIA 比 FA 慢 4 倍、比 eager 慢 2.2 倍**。PromptFlashAttention kernel
  在短序列（S≤130）上调度开销主导，与早期 FIA 整模调查（-3.5% 回退）
  根因一致。**结论：FIA 不可行，FA（npu_fusion_attention）保持现役。**

## 2026-11-09 变更：full-attention 融合为 npu_fusion_attention（P2-8）

**决策**（用户提议"FA 那里是 host bound"）：6 层双向 full-attention 的
eager 链（GQA expand、scores matmul + 加性 mask、fp32 softmax、probs
matmul，每层 ~15 节点）融合为每层一次 `npu_fusion_attention`（bf16
BNSD、bool [B,1,S,S] pad mask True=屏蔽=加性 mask 等价、FA 原生 GQA——
k/v expand 消失）。

- **前置 probe**（纯数学参考）：文本侧全 B/S 组合 relL2 0.0023、eager 链
  x2.05（B1/S99）；**vision 侧先测先弃**——数值精确但 x0.99（fp32 短
  chunk 是 device-bound，融合无收益）。
- 整模门禁：数值 5/5（worst relL2 **0.007893**——FA 内部 probs@v 精度路
  径，仍 2.5× 门限余量）、业务 5/5（0.004143）。
- bench 两次：**49.809 / 50.942 exec/s**（+1.2–3.5%），e2e P50
  18.300 / 17.885 ms；probe 级 x2.05 压缩到整模 ~1–3% 是因为 6 层
  attention 仅占 e2e ~0.5ms。测量表见 performance.md 附录。

## 2026-11-09 变更：融合 RoPE 实验回退（P2-7，GE 原生算子动态 batch bug）

**实验**（用户提议）：6 层 full-attention 的 eager rotate-half RoPE 链
（每侧每层 ~10 个小算子）替换为 `npu_apply_rotary_pos_emb`。

定位过程（逐层 probe）：
- **layout 约束**：本 CANN 构建的 GE 原生 ApplyRotaryPosEmb tiling 仅接受
  BSH(1)/TND(4)，BNSD(3) 被拒（`Parameter layout ... should be 1 or 4`）
  ——改为 TND（[B,N,S,rot] 切片 transpose reshape [T,N,64]，cos/sin
  [T,1,64]）。
- **eager aclnn 路径正确**：TND+half 对 eager 数学 relL2 0.003（量化级）。
- **GE 图模式静态正确**：B2/S99 trace+执行 0.0000（逐位）。
- **GE 图模式动态 B=1 全错**：trace B2/S65 动态导出后，执行 B1 的
  任意 S 全部 relL2 ~0.89（语义级错误），B3/S99 却 0.0000——**GE 原生
  算子对 B=1 的动态执行存在实现 bug**（B 维特化路径）。
- 整模门禁 4/5 FAIL（relL2 0.019–0.074，S=99 最差、全体 B1 case 偏移），
  与 probe 的 B1 bug 及 6 层累积传播相符。

**复活与定案（2026-11-09 三轮，用户提供官方算子文档后）**：官方文档
确认 layout=1 即 **BSND**（[b,s,n,d]），且用户指出 qwen2.5-0.5b 生产已用
该算子。三轮 probe 用**纯数学参考**（而非 aclnn eager——后者的 layout=1
按 BSH 解读，自身即错 0.88，是前两轮全部"图错"结论的根源）后定案：
**GE 图 + BSND 对纯数学 relL2 0.0002，全部 B/S 组合正确**。

**交付形态**：RoPE 在天然 [B,S,N,D] 域（transpose 前）以 BSND 调用融合
算子（q/k [B,S,N,rot] 切片、cos/sin [B,S,1,rot]、零 reshape；eager 腿
保留显式数学——aclnn eager 的 layout=1 非 BSND）。整模门禁 5/5 + 5/5
（relL2 0.003930 / 业务 0.004551）；bench（chip 8 空闲卡，chip 3 被他人
占用曾致导出反复被杀——非 watchdog）**49.231 exec/s、70.097 rows/s**，
e2e P50 18.586 ms——吞吐持平略正（方差内），结构收益 ~120 节点/请求与
AIR -10%。测量表见 performance.md 附录。

**补充证据（2026-11-09 二轮 probe，含 qwen 先例的边界）**：用户指出
qwen2.5-0.5b 生产上已用同一算子（TND+half）——但该先例为**真 varlen 单
动态轴**（q 全程 TND，`reshape(-1)` 自投影输出，无 B 维概念），与本项目
的 **B×S 双动态轴**不同。二轮 probe 补齐边界：

- BSND 域（切片→`reshape(-1)`/`reshape(B*S)`，无转置链，qwen 同构）：
  动态图下 **B1/B3/S85/99/130 全部 relL2 ~0.88**（全错，比 BNSD 源的
  仅 B1 错更广）；
- BNSD 源 + transpose→reshape：动态图 B3/S99 = 0.0000、B1 全错；
- aclnn eager 两种布局全对；静态图（trace=执行形状）全对。

**结论与回退（维持）**：GE 原生 ApplyRotaryPosEmb 在本构建的**双动态轴
（B×S）图上不可靠**——行为随输入布局与 B 值不可预测地变化（BSND 源全错、
BNSD 源 B1 错、eager/静态全对），qwen 先例的真 varlen 单轴场景未覆盖该组
合。P2-7 回退，eager RoPE 链保留。若后续 CANN 版本修复动态双轴 bug，可
凭 probe 的多形状执行模式复测。

## 2026-11-09 变更：GDR/attention 投影合并实验（P2-6，净负回退）

**实验**（用户提议）：GDR 四投影（qkv/z/b/a，同输入）与 full-attention
三投影（q/k/v）构造时预拼接为单权重，单 matmul + 切片，消除 ~54 次/请求
小 matmul 启动。实施含两处返工（GDR in_proj 无 bias 的健壮合并、
buffer 注册目标错到子 Linear），全链通过后整模门禁 5/5 + 5/5（relL2
0.003933）。

**bench 45.78 exec/s（vs P2-5 的 49.00，−6.6%）——净负，回退**。profiling
定位：合并后的 `[B,S,8224]` 切片成为 StridedSlice 新大头（~53 次/请求
大切片 + `[B,S,6144]` 子切片链），且非连续视图向 conv4/transpose 下流传
播使 TransData 翻倍（307→787µs/请求）、MatMulV2 时间上升——切片与非
连续代价远超 matmul 启动节省。**结论：分开的 Linear 输出各自连续，是
该图结构下的更优形态；投影合并在 torchair/GE 的切片实现下不成立**。

## 2026-11-09 变更：预量化视觉权重为 fp16 常量（P2-5）

**决策**（用户拍板）：vision 塔 Linear/Conv 权重在 wrapper 构造时**原位
替换**为 fp16（`param.data.to(fp16)`，构造顺序 `visual.float()` 移到构造
前避免被 upcast 撤销）——导出图的权重 Const **数量不变、dtype 变 fp16**，
torchair positional bake 保持对齐（此前 51 个**额外** buffer Const 才是
bake 冲突根源）；bf16 存储值 → fp16 无损（8 位尾数 ⊂ 10 位，量级远低于
fp16 上限），数值应逐位等价。

- 实施附带：model.py `_expected_param_dtype` 的 vision 规则拆分
  （Linear/Conv → fp16、norm 等其余 → fp32）；fp16 wrapper 构造变异共享
  visual 的顺序语义在单测中固化。
- 整模门禁（`run.sh --device 3`）：数值 5/5（relL2 0.003857）、业务 5/5
  （0.003283）——**与 P2-5 前逐位一致**（预量化与运行时 cast 同值）。
- bench：49.00 exec/s（方差带内）；profiling 复采：总 Cast **501 → 362
  /请求**（vision 的 `[512,768]` 权重 cast 36 次及关联激活 cast 已消），
  残余 362 次主要为 `[1,99,1024]` 级的 norm 残差域与 GDR l2norm 域
  （upcast 无损域内最小集）。
- 结转：vision 激活边界 cast（`[512,3,12,64]`/`[512,3072]` 约 36
  次/请求，~350µs）为混合精度结构固有，此前 fp16 残差流实验已证净负。

## 2026-11-09 变更：安装 WeightNzAndMatMulV3Pass 自定义融合 Pass

**决策**（用户提议）：构建并安装
`third_party/custom_development_code/fusion_pass/WeightNzAndMatMulV3Pass`
到 CANN 自定义融合 Pass 目录（`${ASCEND_OPP_PATH}/vendors/custom_nz_pass/
custom_fusion_passes/`），并在 `prepare.sh` 中固化构建+安装步骤。该 Pass
在图编译期（kAfterOriginGraphOptimize）将 MatMul/MatMulV2 的常量权重
重排为 FRACTAL_NZ 并重建为 MatMulV3——host 期转换、无运行时布局 cast、
激活与输出保持 ND。

- **门禁**：数值 5/5（worst cosine 0.999999 / relL2 0.004496）、业务
  5/5，与基线同级（导出抖动内）。
- **性能**：混合闭环 50.606 exec/s（方差带）；**ticket-team 单 case AI
  时间 17,642 → 15,092 µs/执行（−14.4%）**——MatMulV2 3,115µs 整体替换
  为 MatMulV3 626µs（−80%），文本投影链是最大受益者。测量表见
  performance.md 附录。
- **注意**：安装后首次图编译按 matmul 形状逐一 TBE 编译 MatMulV3
  kernel（首跑需较长超时，kernel_meta 缓存后续免）；系统级安装（CANN
  vendors 目录），prepare.sh 每次重建重装。

## 2026-11-09 变更：删除 GDR 路径冗余 pad 掩码乘法（P0-1）

**决策**（用户拍板）：删除 GDR 路径的全部 pad 掩码乘法——
`forward_embeddings` 的 `mixed * pad`（GDR 层输入清零）与
`_gated_delta_native` 的 `q/k/v * pad`、`beta * valid`（合计 ~90 次/请求
Mul，~1.5–1.9 ms/请求 AI 时间）。依据：chunk 递推严格因果，pad token 只
影响自身及之后位置与最终 state——readout gather 真长 L-1、state 丢弃，
per-row padding 永不泄漏（与 mega 路线 docstring 同一论证；native 传真
理宽度使 pad 区参与递推，但污染同样只落在无害区）。

- **kernel 因果性实证**：删除后整模门禁数值/业务指标与删除前**六位有
  效数字完全一致**（relL2 0.002798 / 业务 0.007095）——pad garbage 对有
  效区输出零影响，掩码确为 no-op。
- 门禁（`run.sh --device 3`，AIR 重导出 1,906,330,769 bytes）：数值 5/5
  （worst cosine 0.999998 / relL2 0.002798）、业务 5/5（worst 0.007095）。
- bench（2,048 请求 0 错误）：**45.265 exec/s、64.450 rows/s
  （+6.4–8.2%）**；e2e P50 **20.232 ms**（−1.4 ms）；B1 ticket 20.2 /
  image 21.2 / B3 27.9 ms。测量表见 performance.md 附录。
- 保留：full-attention 的加性 attn_mask（softmax 双向、key 列必须屏蔽）。
  docstring/describe 的 native 语义段同步为 no-pad-masking 表述。全套
  CPU 单测 247 项通过。

## 2026-11-09 变更：RmsNorm gamma 直喂 bf16（差分下钻定案）

**背景**：RmsNorm 直喂 bf16 后，profiling 仍见 `[1024]` FLOAT→BF16 ×
48 次/请求的 Cast，怀疑 gamma 被下转。差分下钻（pbtxt 保留 + 两版对照
导出）定案：

- **前端图无此 cast**：fp32 gamma 与 bf16 gamma 两版导出的 pbtxt 中
  RmsNorm 均为 79 个、gamma 输入均直连 `Add`/`OnesLike` 节点、`[1024]`
  Cast 节点均为 **0**——cast 是 **GE 图编译期（tiling 选路）插入的**，
  不是 converter（converter 对 npu_rms_norm 直接透传，ops-info 也原生
  声明 (bf16 x, fp32 gamma) 组合）。
- 即：fp32 gamma 版的**交付 kernel 实际一直在用 bf16 gamma**（编译器把
  fp32 gamma 编译期 cast 掉）——"fp32 gamma 精度"从未真正到达 kernel。
- **修法**（用户提议）：gamma 直接以 bf16 计算（`1.0 + norm.weight`，
  bf16 域加法与 fp32 计算→bf16 舍入数值相同）；gated 的 ones 同改 bf16
  （全 1 无量化）。数值与现状**等价**（kernel 消费相同），前端更干净、
  编译器少插 48 个 Cast。

门禁与 bench（2026-11-09，`run.sh --device 3`）：

- 数值 5/5（worst cosine 0.999998 / relL2 0.002798）、业务 5/5（worst
  0.007095）——与 fp32 gamma 版（0.004039 / 0.002769）的差异在 AIR 导出
  非确定性的抖动量级内，均远低于门限。
- bench（2,048 请求 0 错误）：41.84 / 42.54（同 AIR 重跑）exec/s vs
  fp32 版 43.16——**run-to-run 方差带重叠，无显著差异**（48 个
  [1024] Cast 理论值仅 ~64µs/请求 = 0.3%，被方差淹没）。
- 取舍：保持 bf16 gamma（前端图更干净、编译期少 48 个插入 Cast，数值
  与性能不劣）。

## 2026-11-07 变更：RmsNorm 直喂 bf16（混 dtype，省 ~120 Cast/请求）

**决策**（用户拍板，Cast 实证分组驱动）：融合 RmsNorm 节点改为**直喂 bf16
激活 + fp32 gamma**（混 dtype），消除原 `x.float()`/`.to(bf16)` 的显式
往返 Cast（~120 次/请求，普通 norm 61 + GDR gated 18 组）。

前置 NPU eager 验证（2026-11-07）：
- kernel 内部归约为 **fp32**（gated 全 1 gamma：直喂 bf16 1.66e-3 ==
  显式 fp32 域路径的输出量化水平；fp32 输入对照 4.8e-8）；
- fp32 gamma 消除唯一残余误差源（bf16 gamma 量化 +0.0008）；
- torchair converter 接受混 dtype。

整模门禁（`run.sh --device 3`）：数值 5/5（worst cosine 0.999997 /
relL2 **0.004039**）、业务 5/5（worst **0.002769**）——**均显著优于基线**
（0.006571 / 0.006454，Cast 边界减少同时移除了传播路径上的量化边界）。
bench（2,048 请求 0 错误）：**43.157 exec/s、61.448 rows/s（+3.4%）**；
e2e P50 21.422 ms（−0.8 ms）。测量表见 performance.md 附录。

## 2026-11-06 变更：fp16 全塔后续实验（权重物化受阻 + fp16 残差流负收益，均回退）

**背景**：fp16 视觉塔交付（fp32 流 + fp16 线性内积，41.73 exec/s）后两项
后续实验：

- **fp16 权重常量折叠**（预物化 fp16 权重副本以消除每 forward 的权重
  Cast，~0.9 ms/请求目标）：**受阻于导出机制**——torchair 的 positional
  weight bake（运行时输入 i 按位置替换 export_graph.op[i]）在新增 ~51 个
  fp16 权重 Const 节点后错位（全模动态导出报 "incomplete exported Data
  binding"；tiny 静态导出可过）。回退物化。
- **fp16 残差流**（层间 hidden 以 fp16 存储，层边界 fp32 计算域）：数值
  全过（真实塔 relL2 0.0051、bit-stable；整模 5/5 + 5/5，指标与基线完全
  一致），但 bench **40.91 exec/s（−2.0%）**——层间 fp16↔fp32 往返 Cast
  开销超过激活带宽减半的收益（且能消这些 Cast 的权重折叠恰是受阻项）。
  回退至 fp32 流。

**结论**：fp16 视觉塔的最终交付形态即混合版（fp32 流 + fp16 线性内积 +
运行时权重 Cast）；剩余 ~0.9 ms/请求的权重 Cast 仅在改造 torchair bake
契约或预量化 checkpoint 时才可回收。全套单测 247 项通过。

## 2026-11-06 变更：视觉塔线性计算域 fp16（已验证混合精度）

**决策**（用户拍板）：视觉塔线性内积（patch-embed matmul、attention
qkv/proj、MLP fc1/fc2、merger fc1/fc2，wrapper 内 op-for-op 重画）改为
fp16；fp32 激活流、常量表（pos/rotary）、norm/softmax 全部保持 fp32
（`linear_dtype` 开关，默认 fp16；`vision_linear_dtype` 模型参数可回
fp32）。

- 前置调研（真实权重 + 真实 12 层深度 + 真实 fixture，CPU 精度模拟）：
  bf16 线性 relL2 **0.055 FAIL**（尾数 7 位经 12 层累积放大；浅塔模拟
  0.0044 具误导性）；**fp16 线性 relL2 0.0045 PASS**（尾数 10 位，4.5×
  门限余量）。P6 的"BF16 fails"（0.11）根因为整塔量化的 dtype 混乱链
  （复现报错佐证）+ bf16 深度累积。
- 官方先例：官方 vLLM 生产即全模型 bf16（业务口径验收，FP8/Q8_0 亦背书
  "largely preserve"）——我们的门限为特征级 relL2 ≤0.02（更严格），
  fp16 在该严格门限内通过，无需改口径。文本 trunk 保持 bf16
  （ChunkGatedDeltaRule 原型仅接受 bfloat16；fp16=bf16 吞吐）。
- 整模门禁（`run.sh --device 3`）：数值 5/5（worst cosine 0.999999 /
  relL2 0.006571，fp16 传播后仍 3× 门限余量）；业务 5/5（worst
  0.006454，与 fp32 基线**完全一致**）。
- bench（2,048 请求 0 错误）：**41.727 exec/s、59.412 rows/s（+4.8%）**；
  e2e P50 22.218 ms（−1.0 ms）。profiling：MatMulV2 17.25%→14.32%
  （vision 份额减半生效）；Cast 升至 14.35%（fp16 权重 cast 未完全常量
  折叠，~0.9 ms/请求可再回收——列入 backlog）。
- 新增单测 2 项（fp32 模式回归 + fp16 门限），全套 **247 项通过**。

## 2026-11-06 变更：文本 RMSNorm 融合为 GE RmsNorm 节点

**决策**（用户拍板，profiling 驱动）：文本 decoder 的 eager RMSNorm 链
（每请求 48+1+12 次普通 norm + 18 次 GDR gated norm）融合为
`torch.ops.npu.npu_rms_norm`（GE RmsNorm 节点，`_rms_norm`/`
`_rms_norm_gated`，官方计算域精确保持：fp32 mean/weight、输出 cast 回、
gated 变体保持官方 weight/silu 顺序、gamma 由冻结参数常量折叠；eager 腿
不变）。另补回 FIA 回退时 git restore 意外丢失的 causal_conv4 等价单测
2 项，新增 RMS 融合公式等价单测 2 项（245 项全过）。

- 整模门禁（`run.sh --device 3`）：数值 5/5（worst cosine 0.999999 /
  relL2 **0.002233**，融合后显著变好）；业务 5/5（worst 0.006454）。
- bench（2,048 请求 0 错误）：**39.801 exec/s、56.670 rows/s（+10.2%）**；
  e2e P50 23.192 ms（−2.3 ms）。
- profiling 复采（work/prof-rms）：RmsNorm 融合节点 3160 次/4.96% 取代原
  Square 115 + ReduceMean 79 次/请求；Mul 15.8%→11.5%、Cast 12.4%→10.9%；
  新热点排序：MatMulV2 17.3%、ChunkGatedDeltaRule 13.0%、Mul 11.5%。

## 2026-11-06 变更：退役 256 最小物理宽度（case 按真长执行，min_width=1）

**决策**（用户拍板）：P4 时代的 pad256 策略（min_width=256，保守过渡——
保持动态图输入与固定宽度时代逐位一致的验收方式）正式退役；case 按真长
执行（宽度=max 行真长；官方池 S=85/99/130）。

- **先导实验**（先证伪 P4 的"短宽边界不稳定"观察）：在交付 AIR 上以
  min_width=1 重打包官方 case 执行 3 次——5 case 全部**逐位稳定**；对
  CPU golden relL2 ≤0.0037、对已验收 pad-256 输出 relL2 ≤0.0046（bf16
  量化级）。P4 观察在当前配置（fp32-Conv2D conv + eager-math attention +
  MatMul patch-embed）不复现。
- **整模门禁**（`run.sh --device 3`）：数值 5/5（worst cosine 0.999998 /
  relL2 0.004320）、业务 5/5（worst 0.005887）——均优于 pad-256 基线
  （0.004753 / 0.006282）。
- **bench**（2,048 请求 0 错误）：36.125 exec/s、**51.436 rows/s（+6.8%）**；
  e2e P50 25.532 ms（−1.6 ms）；B1 25.1–25.5 / image 26.5 / B3 33.6 ms。
  宽度收益受每请求固定开销（vision 塔恒跑 + host 调度）限制：宽度 ~2×
  缩减 → 延迟 ~7% 下降。
- 契约更新：generate_inputs caps min_width=1；unified_preprocess/
  README 的 pad256 叙述改为 min_width=1 真长执行。P4 pad256 验收记录
  保留于本文件历史章节。

## 2026-11-06 变更：case 池收敛为仅官方 examples（5 个）

**决策**（用户拍板）：输入 case 池**只保留官方仓库 examples 的 5 个
case**（official-ticket-b3-r0 / official-ticket-{team,urgent,severity}-r0 /
official-image-damaged-r0），删除 P8 正式 11 与 b4-long 边界 case 及其
fixtures（`decider/fixtures/p2b-readouts.json`、`b4-long-rows.json`），并
清理死代码链（`_text_cases`/`_image_cases`/`_b4_long_case`/
`_render_official_prompts`、蓝 fixture、adapter 的 `_fixture`/
`fixtures_file` 参数及其单测）。bench 池 = 精度池 = 5 case。

- 整模门禁（`run.sh --device 3`）：数值 **5/5**（worst cosine 0.999998 /
  relL2 0.004753）、业务 **5/5**（worst 0.006282）。
- bench（2,048 请求 0 错误，2,916 行）：33.824 exec/s、**48.160 rows/s**
  （官方分布平均 1.42 行/请求，B3 占抽样 21.2%）；B1 case P50
  27.117–27.121 ms、B3 38.214 ms。测量表见 performance.md 附录。
- 形状覆盖变化披露：官方 5 case 均为 S256/B1–B3；**B4 与 S2048 的形状/
  边界覆盖不再由 case 池承载**（图能力不变，边界常量与校验仍在，
  b4-long 验收记录保留于本文件历史章节）。

## 2026-11-06 变更：输入池扩充官方 examples case（5 个，17-case 门禁）

**决策**（用户拍板）：按官方仓库 StartLuxLabs/StartLux-Decision 的
examples（client.py / request.sh / image_request.py @ main）构造正式输入
case——支持工单 state 的三类型问题（team choice / urgent noul / severity
score，官方单请求多问题的形态以 B3 单 case 承载，每问题一行）+ 同三问题
的 B1 拆分 case + 配送照片 noul 问题（红 fixture 图承载，官方示例图片不随
repo 分发）。渲染与编码全部走既有官方管线（from_systemone /
place_images / plain_prompt + 官方预处理），rows_meta 按选项数取前 N 字母
（noul 恒 2）。

- 池规模：12 → **17 cases**（16 business + b4-long）；bench 池 11 → 16。
- 整模门禁（`run.sh --device 3`）：数值 **17/17**（worst cosine
  0.999995 / relL2 0.005749）；业务 **16/16**（worst 0.007576）。官方
  case 自身 relL2 0.0012–0.0048、prob_max_abs 0.0001–0.0063、top1 全一致、
  margin 大（choice 0.885 / noul 0.426–0.757 / score 0.158）；B3 行与 B1
  拆分 case 结果确定性一致。
- bench（2,048 请求 0 错误，2,432 行）：34.577 exec/s（−3.9% vs 11-case
  池：B3 case 进入等权闭环拉低请求级吞吐）、**41.060 rows/s（+4.6%）**；
  B1 case 的 P50 不变（27.64–27.66 ms）、B3 38.386 ms。测量表见
  performance.md 附录。

## 2026-11-06 变更：视觉 patch-embed Conv3D 替换为等价 MatMul（含 FIA 调查回退）

**决策**（用户拍板，profiling 驱动）：GE profiling（work/prof-fia）显示
Conv3DV2（视觉 patch-embed，fp32）为最大热点——每请求固定 6854 µs、占全部
AI 时间 20%，且**纯文本请求也在跑**（单图架构代价）；指标 mac 0.22 /
mte1 0.97 = 访存受限、退化 tiling。stride==kernel 使该卷积在数学上等价于
每 patch 全连接，改为单个 MatMulV2（`model/vision.py` 的 `MatMulPatchEmbed`：
持有官方 conv 引用、无权重拷贝、输入 cast 语义与官方一致）。

- 前置调查（同日）：`fused_infer_attention_score` 替换 6 层双向 full
  attention 已通过全部门禁（数值 12/12 relL2 0.0068、业务 11/11 worst
  0.0063、bench 28.25 exec/s）但**净负收益 −3.5%**（配套 TransData/Cast 与
  host 调度大于 kernel 节省，FIA 仅占 0.65% AI 时间）——**已回退**为
  eager-math attention；测量表见 performance.md 附录。
- 整模门禁（MatMul patch-embed + conv2d conv + eager-math attention，
  `run.sh --device 3`）：数值 12/12（worst cosine 0.999995 / relL2 0.005749，
  略移但远低于门限）、业务 11/11（worst 0.007576）。
- bench（2,048 请求 0 错误）：**35.989 exec/s（+23.0%）**、e2e P50
  **27.130 ms（−6.43 ms）**；profiling 复采确认 Conv3DV2 消失、该阶段开销
  ~25 µs/请求（约 270×），ChunkGatedDeltaRule 成为最大 AIC 算子（11.0%）。
- CPU 单测：新增 `MatMulPatchEmbed` 等价性 2 项（fp32 对官方 Conv3d ≤1e-6、
  wrapper 接线），全套 244 项通过。

## 2026-11-05 变更：MegaGdnPrefillV2 融合算子切换尝试与同日回退

**决策**（用户拍板）：尝试将 GDR 路线由 native ChunkGatedDeltaRule 切换为
**MegaGdnPrefillV2 整块融合算子**（vendored
`third_party/ascend-ops/mega-gdn-prefill`，custom-op-v2），完成全量门禁后
**同日回退 native 单路线**。原因：融合算子内部按 **128 对齐**设计，本模型
shape 偏小（hidden 1024、Nk=Nv=16/Dk=Dv=128、有效长度多在 S256 量级），
融合收益为负。mega 分支保留为 `gdn_route="mega"` 休眠选项（代码、
`scripts/build_mega_gdn.sh`、vendored 算子包均在）。

切换期间的全量门禁结果（Chip 3 空闲卡，独立流程完整重跑）：

- 流程接线（已回退，记录备查）：`prepare.sh` → `build_mega_gdn.sh`（装入
  CANN 全局 opp + pip 注册 torch 绑定）；`export_air.py` import activator；
  export/infer/bench GE 子进程改用 `ASCEND_CUSTOM_OPP_PATH=<install>/opp`
  替代 chunk 原型 LD_PRELOAD。接线本身全程无故障。
- **数值门禁**：12 cases / 16 rows 全过，worst cosine 0.999981 / relL2
  0.010130（native 0.999995 / 0.004004）。逐用例看，图像用例 logits 误差
  与文本同量级（relL2 0.0028–0.0038 vs 文本 0.0015–0.0044）。
- **业务门禁 8/11**：3 个图像用例 `prob_max_abs` 0.0123/0.0132/0.0147 >
  0.01（top1 一致、margin 0.256/0.392/0.574）——logits 误差经温度归一化放
  大所致，与 `eb8de21` 记录的路线固有精度一致（native 为 11/11，worst
  0.0055）。
- **bench**：2,048 请求 0 错误，20.545 graph exec/s、22.411 rows/s、e2e P50
  46.843 ms（B2 用例 P50 65.2 ms）——对比 native 26.824–27.342 exec/s、
  P50 36.6 ms，**吞吐约 −23%、B2 用例 −51%**（详见 performance.md 附录）。
- **回退执行**：默认 `gdn_route` 还原为 native；`prepare.sh`/`export_air.py`/
  `run.sh` 恢复 chunk 原型显式局部加载契约；单测还原并通过（240 项）。
  回退后 `run.sh --device 3` 全量重跑精度链恢复通过（数值 12/12：worst
  cosine 0.999995 / relL2 0.004004；业务 11/11：worst 0.005507，与 11-05
  凌晨 native 验收一致）；`work/` 精度产物已覆盖为 native，mega 调查的
  bench 残留已清理（native 正式基准仍为 P8 与 standalone 附录记录）。
  `run.sh` 新增 `--device <Chip Logic ID>` 卡选择参数（默认 4）为本日唯一
  保留变更。

## 2026-11-05 变更：causal_conv4 替换为 fp32 depthwise Conv2D 节点

**决策**（用户拍板，"零依赖方案 B"）：`causal_conv4` 内部实现由显式 4 抽头
移位乘加改为**单个 fp32 depthwise Conv2D 节点**（[B,C,1,S]×[C,1,1,4]、
groups=C、padding=(0,3) 切前 S），函数签名/调用点/数值顺序（fp32 conv →
cast 输入 dtype → SiLU）不变；CANN 内置 FusedCausalConv1d/MaskedCausalConv1d
因 **K 固定 3** 不可用，vendored xllm CausalConv1d（K=4）需引入 vendor 包
依赖，未采用。

前置证据（GE 图模式小图门禁，C=6144/groups=C/kernel=4，S256 与 S2048）：
- bf16 输入的 Conv1D 与 Conv2D **均**错编译（relL2 1.41 vs fp64 golden；
  P3 复现，torchair 将 conv1d 转同一 Conv2D 节点）；
- fp32 输入的 Conv2D 正确（2.41e-03 = BF16 输出量化预期，与移位形式及
  eager 完全同值），图模式单 conv 快 27%（S256）–45%（S2048）；
- CANN 内置 `npu_fused_causal_conv1d` / `npu_masked_causal_conv1d` 文档
  明确 K/W 固定 3，直接排除。

整模门禁（2026-11-05，`run.sh --device 3` 全量重跑）：

- 数值 12 cases / 16 rows 全过：worst cosine 0.999995 / relL2 0.004004，
  **与替换前逐项一致**（conv 替换无数值可见影响）。
- 业务 11/11 全过：worst 0.005507，与替换前一致。
- bench（2,048 请求 0 错误）：**29.265 exec/s**（+9.1% vs 同日同卡类移位
  形式 26.824；+7.0% vs P8 正式 27.342）；e2e P50 **33.561 ms**（−3.0 ms）；
  P99 40.101 ms。测量表见 performance.md 附录。
- 新增 CPU 单测 2 项（等价性：fp32 ≤1e-6 / bf16 ≤1e-2 对官方 conv1d 顺
  序、非连续转置视图；因果性：t 仅依赖 ≤t 的 token）；startlux 全套
  CPU 单测通过。

## 2026-11-02 变更：收敛为 native chunk GDR 单路线

**决策**（用户拍板）：纯 Torch 标准路线与原生融合路线并存的结构收敛为
**仅 native 路线**——接受 CANN 升级/换 SoC 时无标准路线兜底的风险（proto
为项目自建注册库，失效时必须先重建重验才能导出）。

- `startlux_npu_adapter.py`：native `_gated_delta`（q/k 真实长度 FP32 L2
  归一 + BF16 cast、零初始态、逐行物理宽度语义）合入基座；`describe`
  契约含 `native_gdr`/`dynamic`/`required_dependencies`；`RuntimeHandle`
  /`CandidateHandle` 全部转为动态 io（B2/S65 确定性 trace 示例）；删除
  `_encode_padded` 固定 B=1 路径。图内容不变（native trace 从不经过被删
  代码），正式 AIR 无需重导出。
- 删除：`startlux_native_adapter.py`（并入基座）、`gated_delta.py`（纯
  Torch GDR）、`operator_checks.py`（P1 门禁历史入口）。
- 保留更正：`ge_readout.py` **不是过程文件**——`unified_readout.py` 与
  `scripts/run_unified_performance.py` 加载其共享实现（`run_plan`/
  `read_air_io`/`parse_timings` 等），与其测试一并保留。
- 测试：`test_startlux_adapter.py` 重写（删 GDR 单测 6 项、conv 单测 2 项、
  wrapper 级 CPU forward 1 项、`_encode_padded` 契约 1 项；并入原
  native 契约用例 5 项，`test_startlux_native_adapter.py` 删除）；
  `test_startlux_multimodal.py` 删 4 项 CPU 上执行 decoder/unified
  forward 的用例（native op 无 CPU 实现）。**302 项全部通过**（Ran 302
  tests in 26.611s，OK；317 → 302 = −10 −6 +5 −4）。
- 覆盖转移：CPU 上的整图 wrapper 级数学验证随标准路线一并消失（native op
  无 CPU 实现），NPU 侧证据（P3 native 7/7、P6/P7 验收、compare 复核）
  为权威覆盖。

## 2026-11-02 变更：ge_runtime 子进程执行流程去重（三处 → 一处）

**决策**（用户拍板，可维护性优先）：ge_runtime 子进程执行协议此前有三套
重复实现——`core/backends/runtime.call`（subprocess.run 简单版，CLI 六阶段
用）、`ge_readout.run_plan`（timeout + 证据保留，业务读出用）、
`run_unified_performance.ge_call_polling`（Popen + abort 轮询 + 进程组
kill，P8 用）——收敛为 `core/backends/runtime.execute` 一套权威实现。

- `execute(runtime_bin, plan, *, log_path, timeout, should_abort,
  poll_interval, error, aborted_error)`：超集语义（Popen + start_new_session
  进程组隔离、timeout/abort 触发 killpg、恰好一条 GE_RESULT、failed-status
  非零退出返回带 error 详情的 result、wall 计时、log_path 缺省走临时文
  件）；异常类型按调用方参数化，全部公共接口/异常类型/返回结构保持不变。
- `call()` 变薄委托（签名不变）：core/worker、test_framework、
  test_bge_replay_cli 的 patch 点零改动；全部路径附带获得进程组隔离
  kill（此前仅 P8 有）。
- `ge_readout.run_plan` / `run_unified_performance.ge_call_polling` 委托
  execute，仅保留各自异常映射（ReadoutError / WorkerError+Aborted 含
  abort reason）；`_kill_own_group` 删除。P8 脚本变更无 NPU 复验（NPU
  侧已停测），行为等价由下述测试保证。
- **边界放宽记录**：ge_readout 原已 import `core.contracts.validate_io`
  （repo-root 自举），本次追加 import `core.backends.runtime`；适配器文件
  （load_adapter 加载的 unified_adapter.py）仍不 import core，隔离规则
  适用范围不变。
- 行为保持证据：新增 execute 协议单测 9 项（missing-binary 错误类型、
  单/多 GE_RESULT、failed-status 返回、非零退出、非 JSON、timeout kill、
  abort 仅杀本组、临时日志）；test_review_contracts 两处 patch
  subprocess.run 的用例改真实假二进制（fork-child 场景断言拆分以适配新
  消息格式）；**302 项 startlux 单测全过**（含 ge_readout 47 项与
  unified_readout timeout 项）；全仓 406 项 4 errors 全为预存 onnx 缺包，
  与既有记录口径一致，零新增。

## 精度验收的历史支撑（细节在归档）

P0 材料固定（上游 commit/模型 revision/权重 SHA 逐文件核对，
`work/startlux-materials/materials-report.md`；vendor 13 模块逐字节一致
`work/startlux-cannbot/review-vendor.json`）、P1 算子级门禁、P2 官方 CPU
FP32/BF16 业务语义、P3 B1/S256 两路线整模 7/7（`review-p3-final-independent.json`）、
P4 动态整模 27 cases（`review-p4-dynamic-raw-independent.json`）、P4 业务回调
（`review-p4-business-independent.json`）、P6 视觉塔 FP32 AIR（relL2
0.000286066/0.000292276，`review-p6-vision-raw-independent.json`）等过程记录
与原始文档见 `work/startlux-archive/cleanup-20261102/`（decision-validation.md、
native-validation.md、unified-validation.md 等快照）及 work/ 下各证据目录。

## P8 独立复核（2026-11-02，passed）

`work/startlux-cannbot/review-p8-real-independent.json` / `.md`：从原始 run 目录
独立复算 B4 门禁与 bench 全量统计（raw CSV sha256
`729b9d53957d5b4b7bd4bcdaa9b4db2b11b9b399390d0c05801fc0944c848c4f`、B4 raw
sha256 `2b69e5863a086b54238d212dd74417c03dc2b82df1dc710aed1d8706f8b81bea`），
AIR/runtime SHA 与字节数一致，随机序列逐请求一致，结论 passed。性能数值见
`performance.md`。

## 原始证据路径

- P8/B4 补验与 bench：
  `work/startlux-validation/p8-idle-performance/runs/20261101T025002-45aba096/`
  （`gate-20261101T025013-a9dafbe2/`：gate-report.json、plan.json、inputs.bin、
  outputs.bin、infer-runtime.log；`bench-20261101T025109-558a6522/`：result.json、
  summary.json、report.md、requests.csv、bench-runtime.log、inputs.bin）
- P7 正式验收：`work/startlux-cannbot/review-p7-cli-independent.json`、
  `review-p7-air-structure-independent.json`、
  `review-p7-boundary-independent.json`、
  `work/startlux-validation/p7-final/`、
  `work/startlux-cannbot/p7-main-review-findings.md`
- 主 agent 复核：`work/startlux-cannbot/p8-main-final-review.json`（accepted）、
  `review-p8-real-independent.json`/`.md`
