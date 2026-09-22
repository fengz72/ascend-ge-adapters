# PrefixInferAttentionScore 相对 FIA 源仓的改动说明（维护指南）

本文档记录本算子相对于 **ops-transformer `origin/9.0.0` @ e88633357** 中 FIA（`attention/common` 模板链 + `attention/fused_infer_attention_score`）的全部改动，供后续维护与 FIA 版本升级时重放。

## 0. 总览：三层改动

| 层 | 内容 | 载体 | FIA 行为是否改变 |
|---|---|---|---|
| **L1 prefix-in-Q 使能** | 编译期/运行期标志 `prefixInQ`，prefix KV 内嵌主 KV 头部，batch 间共享 | vendored common 文件（源仓 `attention/common`，7 文件 +163/-24） | 否（全部 gated on `prefixInQFlag`/`FIAT::prefixInQ`，FIA 侧恒 false） |
| **L2 独立仓化** | 从源仓解耦为独立可编单算子仓：目录重组、include 路径、自包含前导、本算子自有文件 | 本仓全部非 common 文件 + include 改动 | 不适用（本仓只编 PIA，不产 FIA 二进制） |
| **L3 prefix-merged + 性能** | batch 0'=[prefix;req0] 合并；GQA 高性能 cube 块；分核容差修复/两段择优 | vendored common 文件（本仓工作区，15 文件）+ 自有文件 | 否（gated on `prefixMergedFlag`/`FIAT::prefixMerged`，FIA 侧恒 false） |

**门控纪律（重要）**：所有对 vendored common 文件的行为改动均以 `prefixInQ`/`prefixMerged` 标志门控，标志为 false 时代码路径与 FIA 逐位一致。升级 FIA 时只需以新版 FIA 为底重放这些 gated 补丁，无需理解 FIA 自身逻辑的变化。

## 1. 目录映射（vendored ↔ 源仓）

| 本仓 | 源仓（e88633357） | 说明 |
|---|---|---|
| `op_host/common_impl/`（顶层 .h/.cpp） | `attention/common/op_host/` | split_core、fia_tiling_info/shape/templates_registry/base |
| `op_host/common_impl/arch32/` | `attention/common/op_host/arch32/` | FiaTilingNonQuant tiling 模板 |
| `op_host/common_impl/include/` | `attention/common/op_host/include/` | tiling_base 框架（未改） |
| `op_host/fia_tiling_impl/` | `attention/fused_infer_attention_score/op_host/arch32/` 同名文件 | FIA 的 tiling 结构/parser（仅 include 路径改动） |
| `op_kernel/common/op_kernel/` | `attention/common/op_kernel/` | kernel 模板链（含 arch32/、memcopy/） |

## 2. vendored common 文件改动明细

逐文件列出全部 delta（相对源仓 e88633357，含 L1+L3），维护时**改动这些文件必须保持门控与 FIA 等价性**。

### 2.1 `op_kernel/common/op_kernel/fia_public_define.h`
- FIAType 新增第 12 个模板参数 `PREFIX_IN_Q`（默认 false）→ `static constexpr bool prefixInQ`（L1）
- FIAType 新增第 13 个模板参数 `PREFIX_MERGED`（默认 false）→ `static constexpr bool prefixMerged`（L3）
- `ConstInfo` 新增 `bool prefixInQFlag`（L1）
- FIA 侧两参数恒为默认 false，模板展开结果不变

### 2.2 `op_kernel/common/op_kernel/arch32/fia_kernel_nonquant.h`（调度器）
- **L1**：`InitTilingData` 中 `constInfo.prefixInQFlag = FIAT::prefixInQ`；`GetTaskDealMode` 中 batch 0 的 KV 长度豁免（`!(prefixInQFlag && bIdx==0)` 才 `+= systemPrefixLen`）；`s1PosOffset = (prefixInQFlag && bIdx != 0) ? P : 0`（batch 1+ 的 Q 绝对位置用于 causal 对角线计算，`CalcCurS2StartEndWithSparse` 消费）
- **L3**：act 数组"视角后移一个元素"共三处（见 §4.1 链条）：`InitActualSeqLenQ`/`InitActualSeqLenKV` 的 `SetGlobalBuffer` 指针 +1；`Init` 中传给 `matmulService`/`vectorService` 的原始 act 指针 +8B。均 `if constexpr (FIAT::prefixMerged)` 门控
- `fia_block_vec_flashdecode.h`：FD 路径同款 batch 0 KV 豁免（L1，5 行）

### 2.3 `op_kernel/common/op_kernel/arch32/fia_block_vec_nonquant.h`（向量块）
- **L1**：`ElewiseCompute` 中压缩 mask（sparse 2/3/4）场景 batch 1+ 的 `maskInfo.gs1StartIdx += systemPrefixLen * gSize`（mask 行偏移参与对角线计算；sparse 0/1 的显式 mask 行号即批内行号，不加偏移）

### 2.4 `op_kernel/common/op_kernel/arch32/fia_block_cube_nonquant.h`（泛化 cube 块）
- **L1**：`Init` 中 shared prefix 输入未连接时，prefix GM 别名到主 KV batch0 数据指针（`ListTensorDesc` 解析；FIA 的独立 prefix 张量路径保留：`keySharedPrefix != nullptr ? : 别名`）
- **L3**：`DealMm1SingleMKN`/`DealMm2SingleMKN` 的 KV 拷贝分发加 batch 0' 豁免：`systemPrefixFlag && !(prefixInQFlag && bIdx==0)` 时走 prefix 分段拷贝，否则主 KV 整段拷贝（batch 0' 的 KV 区=[prefix;req0] 连续段）。**当前 kernel 入口未使用此块**（已换 GQA 块），保留改动仅为回滚便利

### 2.5 `op_kernel/common/op_kernel/arch32/fia_block_cube_nonquant_gqa.h`（GQA 高性能 cube 块）
- **L1**（源仓层 +151 行）：同 2.4 的 prefix 别名；`CopyKToL1`/`CopyVToL1` 的 prefix/主 KV 分段拷贝（`prefixDealSize`/`normalDealSize` 切分，主 KV 部分 `s2Idx = s2Start - prefixLen`、基底 `GetTBase(bIdx)`）
- **L3**：两处拷贝分发加 batch 0' 整段拷贝豁免（与 2.4 同款条件）。**此块是当前 kernel 入口的在用 cube 块**

### 2.6 `op_kernel/common/op_kernel/memcopy/offset_calculator_v2.h`
- **L1**：`GM_KV_TND` 类新增 `InitForPrefix`（TND 前缀为 [T,N,D] 独立张量、无 act，按 T 总长初始化布局）

### 2.7 `op_host/common_impl/fia_tiling_info.h`
- **L1**：`FiaTilingInfo` 新增 `bool prefixInQFlag`
- **L3**：新增 `bool prefixMergedFlag`（注释含生效口径说明）

### 2.8 `op_host/common_impl/split_core.cpp`（分核，L3 主战场）
- **`GetBatchAssignTolerance`**（新增函数）：batch 级分配容差原为"当前 batch 末块代价/2"（面向 FIA 均匀 batch）；prefixInQ 下以待放入单元自身代价兜底（`max(lastBlockCost, bN2Cost) / 2`），否则 merged batch 0' 的小尾 M 块（代价 42）使容差过紧，486 的大单元放不进单核被行切分为 FD 规约
- **`SplitCore` 两段择优**（prefixInQ 门控）：
  - pass1 仅接受无 FD 方案（整 batch 整行不跨核；prefill 的 FD 向量规约开销未计入 cost 模型，实测净亏，aiv +25%）
  - pass2 FD 回退：仅当 FD 方案 maxCost < FD-free 最优的 2/3 时启用（保护单超大 batch 无 batch 级并行度的场景）
  - 并列决胜：maxCost 并列时优先"使用核数更多但 ≤ bN2 单元总数(bSize×n2Size)"的方案（单元不串行；超出单元数的核必然空闲）
  - **完整性校验**：`tmpResult.usedCoreNum != 0` 才可接受——被 costLimit 剪枝中途放弃的残缺方案（bN2End 部分为 0）曾被误接受，导致 kernel 读垃圾任务区间（数值错/aicore 挂死）
  - 诊断开关：`PIA_SPLIT_CORES=N`（强制核数）、`PIA_SPLIT_FDFREE=1`（禁用 FD 回退）
- `split_core.h`：`BaseInfo::prefixInQ` 注释更新

### 2.9 `op_host/common_impl/arch32/fia_tiling_nonquant.cpp`（tiling 模板）
- **`CreateSplitInput`**（prefixMergedFlag 门控）：生效 bSize=B-1、act 数组按索引 +1 物化（batch 0' 的 s1/s2=act[1]，batch j'=差分），与 `GetS1SeqSize`/`GetS2SeqSize` 的 cumsum 差分公式及 batch 0 豁免逐位吻合
- **`FillTilingBaseParams`**（同门控）：下发 bSize=B-1、`actualSeqS1Dims/S2Dims`=B-1（kernel 侧指针 +1 后 `GetTSize()=act[B-1]=T` 保持正确）
- **`CalcMBaseSize`**：保持 512，注释记录 mBase 256 试验无收益（见 §5）

### 2.10 `op_host/fia_tiling_impl/fused_infer_attention_score_tiling_info_parser.h`
- 仅 include 路径改动（`../common_impl/...` 替代源仓相对路径），无行为变化

## 3. 本算子自有文件（非 vendored，全部为新增）

| 文件 | 职责 |
|---|---|
| `op_host/prefix_infer_attention_score_tiling.cpp/.h` | tiling 入口：复用 FiaInfoParser 解析 → 硬编码 prefix-in-Q+merged 语义（置 prefixInQFlag/prefixMergedFlag、prefix 长度 P=act_kv[0]、s1Size 改写为合并后最大 Q 长度）→ 自有 checker → 注册表分派 FiaTilingNonQuant。含 `PiaGenSimplifiedKey`（GE 图模式二进制匹配，未注册会回退 JIT 导致输出错乱） |
| `op_host/prefix_infer_attention_score_tiling_check.cpp/.h` | 自有约束校验：仅 TND/NoQuant/无 PSE/sparse 0或2；act_q≡act_kv 为**严格递增** cumsum([P,L0,...]) 且 **actLen≥2**；sparse_mode=0 仅支持共享 mask（2D 或 batch 维=1，merged 下逐 batch mask 无法映射）；P+max(L)≤2048（压缩 mask） |
| `op_host/prefix_infer_attention_score_def.cpp` | op def（31 输入 + 16 attr，与 FIA 同序；shared prefix 输入槽位保留但永不连接） |
| `op_host/prefix_infer_attention_score_infershape.cpp` | 输出 shape 推导（=query.shape），挂接 OpDef 供 GE InferShapePass |
| `op_host/op_api/aclnn_prefix_infer_attention_score{.cpp,.h,_full.cpp}` | aclnn 三段式接口 |
| `op_host/op_api/prefix_infer_attention_score_custom_op.cpp` | GE V2 EagerExecuteOp 回调（图模式 GE 路径，31 输入 + 16 attr 镜像 def） |
| `op_host/fia_tiling_impl/prefix_infer_attention_score_base_tiling.h` | 基础 tiling 注册（独立命名的等价布局 struct，避免与 per-key 注册同名冲突） |
| `op_kernel/prefix_infer_attention_score.cpp` | kernel 入口：4 个 tiling key（fp16/bf16 × 非split/splitKv），`FiaKernelNonQuant + FiaBlockCubeNonQuantGqa + FiaBlockVecNonQuant + FiaBlockVecFlashDecode`，FIAType 第 12/13 参数均 true；自包含 scalar 前导（源仓经由 IFA/PFA 完整前导获得，本仓闭包自携带） |
| `op_host/common_impl/arch32/fia_host_prelude.h` | host 侧自包含前导（同上原因） |
| `op_graph/prefix_infer_attention_score_proto.h` | 图融合识别用头（输入/attr 镜像 def，文档注释须与现行契约一致） |
| `torch_binding/` | torch 接入独立包（schema+eager+torchair converter，aclnn 符号经 dlopen 解析） |
| `tests/` | test_torch_binding.py（三层验收：eager vs CPU golden / `.tensor` vs `.default` bit 级 / GE 整图 vs eager bit 级；全 PASS 为准出） |

## 4. 关键机制（改动时必须保持的不变量）

### 4.1 prefix-merged 的 act 索引+1 链条（四处一致，缺一即静默错乱）

外部契约不变：act=cumsum([P, L0, ..., L_{B-1}])，B 个元素。kernel 侧合并解释为 B-1 个 batch：

```
batch 0' = [prefix;req0]: Q/KV 基底=0,        长度=act[1]=P+L0  (方阵causal, 不叠加P)
batch j' = 请求j:         基底=act[j'],       长度=act[j'+1]-act[j'], KV长度再+P(前缀区可见)
```

实现 = "act 视图后移一个元素 + dims=B-1"，四处必须同时成立：
1. **host 物化**（`CreateSplitInput`）：分核输入按 act[i+1] 填充
2. **host 下发**（`FillTilingBaseParams`）：bSize/dims 均为 B-1
3. **kernel 调度器**（`InitActualSeqLenQ/KV`）：GM 视图指针 +1 元素
4. **kernel 块入参**（`Init` 传给 matmulService/vectorService 的原始指针）：+8 字节

配套：batch 0 的 KV 长度豁免（`GetTaskDealMode`）+ batch 0 的 KV 整段拷贝豁免（GQA 块 CopyK/VToL1）+ batch 1+ 的 mask 行偏移（vec 块，L1 已有）。

### 4.2 分核两段择优的边界
- FD-free 偏好是经验结论（prefill 场景 FD 规约 aiv 开销 > 分核均衡收益），若未来 shape 演化（如单 batch 超大 KV），靠 2/3 回退门槛兜底；门槛是经验值
- 残缺方案校验（`usedCoreNum != 0`）是**正确性修复**，任何重构 SplitCore 时不可丢
- 借助 cost 模型不感知的开销（FD 规约、任务串行化），故存在并列决胜这类启发式；修改需重做性能 A/B（msprof `Task Duration` 口径，对照 FIA 基线，README §5）

### 4.3 s1Size 改写（tiling 入口）
merged 后最大 Q 长度=max(act[1], max L_j)，用于 mm1/mm2 workspace 的 M 轴尺寸（`CalcMmResSize`），不改写会 workspace 不足。

## 5. 试验记录（避免重复踩坑）

| 试验 | 结论 |
|---|---|
| mBaseSize 512→256（加深流水/缩 drain） | 两轮全场景 ±5% 噪声内无收益，seq=218 略差——aiv 长杆根因是绝对工作量（scores GM 往返/mask/softmax），非流水深度。已回退，注释留在 `CalcMBaseSize` |
| 任务预取深度 3→4（`FIA_PRELOAD_TASK_CACHE_SIZE`） | no-op：`ExecuteTask` 槽位算术硬编码滞后量（vec1=t-1, vec2=t-2），扩环不改变滞后。真加深需 PRELOAD_NUM workspace 环 + 同步 event 环改造，收益不成立 |
| cube 块泛化版（FiaBlockCubeNonQuant） | 已弃用换 GQA 版（对齐 FIA 生产包 HIGH_PERFORMANCE_GQA 同款 Q2V2KP3 tiling），泛化块改动保留仅为回滚 |
| aiv=0.0 现象 | FIA 生产包 op_summary 的 aiv 列为采集口径假象，FIA 实为同款 MIX 流水；勿据此推断"FIA 纯 AIC" |

## 6. FIA 版本升级重放指引

1. 以源仓新版本 `attention/common` 覆盖本仓 `op_host/common_impl/` + `op_kernel/common/` 对应文件（保持本仓目录结构，修 include 路径）
2. 重放 §2 各文件的 gated 补丁（改动均为局部、带 `prefix-in-Q`/`prefix-merged` 注释标记，可逐块 grep 定位）
3. `fia_tiling_impl/` 用源仓 FIA 同名文件覆盖后重放 include 路径改动
4. 重建 + 全量回归：`test_torch_binding.py`（golden/bit级/GE图，全 PASS 为准出）
5. 性能 A/B：msprof `Task Duration` 口径（300 次取 min，含 FIA V1 基线同进程对照），对照 README §5 的 FIA/PIA 基线区间
