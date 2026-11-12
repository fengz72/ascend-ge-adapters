# StartLux-Decision v1.1 0.8B Instant 验证记录（当前状态）

环境身份：容器 hejun-cann910-jev，CANN 9.1.0，Python 3.12.13，torch 2.12.0+cpu，
torch_npu 2.12.0，transformers 5.8.1，设备 Ascend910_9382，runtime 二进制
`runtime/build/ge_runtime`。GE Session：
`graph_run_mode=0`、`precision_mode=must_keep_origin_dtype`。

最终交付（已验收）：视觉塔 + 图像特征注入 + 三轴 MRoPE + 文本 decoder + 26 列
letter 读出头整合于**单一 AIR、单次 GESession**。交付形态为独立脚本流程
（`run.sh` 串行调度，无 YAML / adapter / handle，见
[docs/standalone-runtime-plan.md](docs/standalone-runtime-plan.md)）；下文中
`config/model.yaml` / `unified_adapter` 等框架路由条目为历史记录，对应文件已
删除。AIR/runtime 身份：

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

2026-11-10 模块化重写后：`tests/test_startlux_adapter.py` 与
`tests/test_startlux_multimodal.py` 按用户决策删除（单腿 NPU 算子使新模块
无法在 CPU forward，相关断言由 NPU eager/graph 双路验证接管，见当日条目）；
`tests/test_startlux_unified_preprocess.py`（23 项）与
`tests/test_startlux_native_proto.py` 改路径后全部通过。

## 2026-11-12 调查（负结果，未集成）：MegaGdnPrefillV2 长文本复测（S=1001 stage-1 单算子探针，否决）

**动机**：2026-11-05 整模否决（−23% 吞吐）的三个理由全部是短序列前提——
128 对齐浪费（S85 ~34%）、B2 mixed −51%、8/11 业务 FAIL 集中在图像用例。
devcase 池是纯文本 S=1001–1005、B=1、业务阈值 0.02，前提已全部改变，且
msprof 显示 mega 可吸收的原生链达 **~611 µs/GDN 层**（CausalConv1d 66 +
3×StridedSlice 39 + 2×L2 RmsNorm 49 + Fill 8 + CGDR 331 + gated RmsNorm 28 +
silu/cast/mul/sigmoid 尾 ~90，gamma 折叠后口径）。故先做 stage-1 单算子
探针（`work/devcase/mega-probe/probe_s1001.py`，几何与模型一致
NV=16/DK=DV=128/C=6144，tokens=1001/batch=1），不改模型、不重导出 AIR。

**结果（chip 14，算子与安装均为 2026-11-05 修复版原样在位）**：

- **计时（决定性）**：graph 腿（torchair 默认模式，模型 AIR 路径同构）
  **685 µs/层**（另一次 827，kernel_meta 冷热差；均含每次调用的 scratch
  分配 ~15 µs）、eager 腿 1,310 µs/层（桥开销，模型路径不适用）——
  **连最佳值都高于整条原生链 611 µs/层（+12%）**，18 层放大为
  +1.3 ms/请求（−5.8% 吞吐）。**NO-GO**，与 2026-11-05 结论一致：融合
  收益为负**扩展到长文本负载**，非短序列假象。
- **精度（附带数据点）**：eager vs CPU fp32 golden（逐 token 递推参考，
  matmul 形式重写并经 T=64 与原 einsum 版交叉验证 max diff 3e-5 = 1 ulp）
  **relL2 0.005017**（门限 0.02，4× 余量；比 S≈99 整模的 0.0101 更好）；
  graph vs eager rel_l2 = 0.0。**1001%128=9 余块路径正确**（此前从未测过
  非整块长度），fp16 内部流水不随长度劣化。
- 同时避免了集成才会暴露的两项已知风险：3 个 128×128 mask Const 对
  positional bake 的冲击、每层一次的 cu_seqlens D2H 回读同步（18 次/请求）。

**环境事故与修复（记录为坑）**：pip 装的 `fla_npu` 包携带
`site-packages/fla_npu_opp_env.pth`，**每个 python 进程启动时**把
`ASCEND_CUSTOM_OPP_PATH` 指向自身 opp；mega activator 的 `find_op_lib` 是
旧搜索顺序（先 `ASCEND_CUSTOM_OPP_PATH`），被劫持后 dlopen 了
`fla_npu_transformer` 的同名 `libcust_opapi.so`，JIT 桥报
`undefined symbol: mega_gdn_prefill_run_v2`。**causal_conv1d 的 activator
已修过此顺序（先 `$ASCEND_OPP_PATH`，0ffc5b6 的 wrapper 加固）**，mega
未跟进。探针用启动包装器重设 `ASCEND_CUSTOM_OPP_PATH=$ASCEND_OPP_PATH`
绕过（`/tmp/opencode/run_probe.py` 模式）；若未来重新集成 mega，需同步修
`third_party/ascend-ops/mega-gdn-prefill/python/mega_gdn_prefill.py` 的搜索
顺序并重装。另：torch extension 的 file_baton 锁在 kill -9 后残留会使后续
所有运行死等，删 `torch_extensions/mega_gdn_prefill_bridge/` 即恢复。

证据：`work/devcase/mega-probe/`（probe_s1001.py、golden_t1001.pt、
probe.log…probe5.log、探针期间 chip 14 idle）。

## 2026-11-11 变更：RMSNorm gamma 预折叠 + gated norm gamma 折叠（两笔独立提交）

msprof（dev-intent-q00-r0 长文本，restored 口径）定位：迭代头部 97 个小算子
中 **61 个 Add（`1.0 + norm.weight`，49×[1024] + 12×[256]）与 18 个
OnesLike（gated ones）是冻结权重的常量算术**，每个前有 ~5 µs 设备空转
（profiling 下 host 发射受限）；另有 gated norm 的 `weight * n` 独立
Mul 节点 18 次/请求（0.30 ms 设备时间）。

- **A1（位级等价，`70a4328`）**：`from_folder` 在 load_state_dict 后把
  `1.0 + weight` 原位烘进每个文本侧 `StartLuxRMSNorm` 参数（bf16 域、同
  操作数序 → **位级一致**），forward 直用 `self.weight`；计数断言由 config
  推导（2/层 + 终层 + 2/full-attention 层 = 61）。验证：**AIR 输出逐字节
  相同**（infer outputs.bin md5 `16a42648f00c39eb8098405e0605f7aa` 与改前
  一致）、门禁原样通过、text.air −69 KB（61 节点消失）。bench −0.02 ms
  （方差带内）——msprof 下的前导空转收益是 profiling 特有的 host 开销，
  如实记为结构性清理。
- **A2（数值等价，`0a1e1f6`）**：`rms_norm(x, w) == w * rms_norm(x, 1)`
  严格成立（gamma 是纯后置缩放），gated norm 直传 `self.weight` 为
  gamma，消除 18×OnesLike + 18×Mul（kernel 在 fp32 域施加 gamma，少一次
  bf16 舍入）。与被否决的 2026-11-09 scale 折叠（sqrt(D) 进 kernel scale
  喂非线性递推，relL2 0.85）性质不同。验证：worst cosine 0.999993 /
  relL2 0.004937（位级基线 0.004779，余量 4×）、业务 worst |dp|
  **0.013914→0.010254（变好）**；bench ×2：avg 21.681/21.533 ms（均值
  **21.607，−0.57 ms/−2.57%**，同会话基线 22.177），qps 45.09→46.28。
- **msprof 复采**（`work/devcase/prof-gammafold-devq00/`）：**947→850
  ops/请求**（−97 = 61 Add + 18 OnesLike + 18 Mul），设备 busy 21.88→21.51
  ms、idle 1.06→**0.64 ms**、利用率 95.4→**97.1%**、msprof 下 wall
  22.94→**22.14 ms（−3.5%）**。
- 新增 hermetic 单测 `tests/test_startlux_gamma_fold.py`（5 项：折叠位级、
  幂等、未折叠即失效的坑、walker 类过滤、gated 等价性 fp32 参考 + 源契约）。

证据：`work/devcase/{infer-base-a1,infer-a1,infer-a2,bench-a1-base,bench-a1,
bench-a2-r1,bench-a2-r2,prof-gammafold-devq00}/`；测量表见 performance.md
当日附录。

## 2026-11-11 变更（负结果，已回退）：手动 npu_add_rms_norm 替换 InplaceAddRmsNorm

**问题**（用户提议）：team profiling 显示 48 个 InplaceAddRmsNorm 融合点各带一对
bf16→f32 / f32→bf16 Cast（[1,99,1024] 双向 ×48，~534us/请求）——当时归因于
AddRmsNormFusionGraphPass 的 fp32 域。能否手动调用
`torch_npu.npu_add_rms_norm` 绕过融合 pass、消除 cast？

**实验**：tuple-flow 重构 DecoderLayer/TextModel（mlp_out + pending 残差流，
48 个 Add+RmsNorm 位点全部改为显式 `npu_add_rms_norm` 调用）。

**结果**：
- eager：`npu_add_rms_norm` 确实直收直出 bf16（y/sum 均 bf16，sum 与
  bf16(fp32 加法) 位精确，数值 relL2 0.0028 同级）——eager 路径无 cast；
- **图路径 cast 原样保留**：`[1,99,1024]` bf16→f32 ×48 + `[99,1024]`
  f32→bf16 ×48 与替换前逐形状相同；
- AI 时间 13,238 → 13,616 us/exec（+2.9%，方差带内持平略差）；内核
  AddRmsNorm（手动）540us vs InplaceAddRmsNorm（pass 融合）495us——同族
  内核，手动路由无优势；
- 门禁：eager 冒烟 5 类 case PASS（0.0011–0.0058），图 infer 门禁 PASS
  （worst relL2 0.006053 与基线同值）。

**结论（2026-11-11 深挖修正，用户指正后重新定案）**：上面"内核族要求 fp32"的
归因是**错的**。追加三轮证据：
1. **TBE 内核二进制元数据**（`opp/built-in/.../add_rms_norm/AddRmsNorm_3_high_performance.json`
   与 `inplace_add_rms_norm/InplaceAddRmsNorm_3_high_performance.json` 的
   supportInfo）：**x1/x2/gamma 均为 bfloat16，输出 x1/x2 也 bfloat16**（唯一
   f32 输出是 rstd [..,1]，尺寸可忽略）——内核原生 bf16（与 ops-nn 9.1.0
   README 一致）；
2. **原始导出图（保留 dynamo.pbtxt 的 4 层迷你图）**：norm 输出 bf16、
   BatchMatMul x1 bf16，norm↔matmul 边界**零 cast**——模型代码与 dynamo
   导出全程干净；
3. **自定义 WeightNzAndMatMulV3Pass 源码**：`ReplaceWithMatMulV3` 将 x1
   desc 原样复制（不改 dtype、不插 cast）。

**归属（计数证据定案）**：cast 的形状/数量与 **MatMulV3 的输入/输出位**
完全对齐——[1,99,1024] bf16→f32 ×48 = 每层 norm1+norm2 输出喂投影（24+24）；
[99,1024] f32→bf16 ×48 = down_proj（24）+ out_proj（18+6）输出回 bf16 流；
[1,99,3584]/[99,3584] ×24 = MLP 中间界。norm 内核 bf16 进出（supportInfo），
不可能产出 f32 [99,1024]——**f32 往返 attach 在 MatMulV3 激活值边界**，由
编译期（我们 NZ pass 之后的 CANN 内部阶段）插入；旁证：本 SoC 的
MatMulV3/MatMulV2 runtime KB 仅有 fp16/fp32 条目、无 bf16。手动
`npu_add_rms_norm` 失败的完整解释：cast 本就不在 norm 边界，换 norm 的
调用方式不触及它。InplaceAddRmsNorm 无 torch/torchair 暴露的调用入口
（torch.ops.npu 与 torchair ge_apis 均查无此算子），“直接手动调用”
不可达，且按归属也不解决问题。

**新的可行杠杆（未测）**：fusion switch 配置禁用这两个内置 pass，让残差走
朴素路径——67 个非融合 RmsNorm 位点（`RmsNorm_normal_all_bf16` 内核）实测
**零 cast**，证明朴素路径无此开销。预估：省 ~534us cast，付 ~48 次额外 Add
内核（~3us/次），净赚 ~300-400us/请求。GE Session 侧的禁用方式需查
（ATC 有 `--fusion_switch_file`，Session 选项名待确认）。
**代码已回退**（outputs 位相同恢复 `cff3bbdb…`）；pipeline_runtime 的
profiling + case_filter 能力保留（未提交）。

实验证据：`work/prof-team-split/`（替换前）、`work/prof-team-fused/`
（替换后，两者 op/cast 形状对照）。

## 2026-11-10 变更：纯文本请求完全跳过视觉阶段（对齐官方 media=None 行为）

**决策**（用户提议）：拆分后纯文本调用仍把 4 行哑元跑一遍视觉 AIR（bench
requests.csv 实测 **3.7ms/请求**，占 e2e 17.3ms 的 ~21%）——官方行为是
`media is None → 不跑视觉塔`，两阶段形态现在可以完整对齐。

**设计**：
- `generate_inputs.py`：每 case 记录 `has_images`（从编码行是否携带
  image_grid_thw 判定）；records 格式不变，视觉哑元张量仍打包（格式统一、
  eager 调试仍可跑视觉），runtime 只是**不执行**；
- `pipeline_runtime.cpp`：`has_images=false` 的 case **完全跳过视觉图的
  H2D 与 ExecuteGraphWithStreamAsync**，text 图输入槽 3 直喂**预分配的
  1×1024 零特征张量**（aclrtMemset 清零一次）——形状与 4 行哑元视觉输出
  **完全一致**（[1,16,16] 哑元的 T_m = 4/4 = 1），零新增形状特化；
- **位相同性证明**：注入是 `where(is_image, gathered, embeds)`，is_image
  全 false 时逐元素选择与 gathered 值无关 → 零特征与哑元视觉输出产出
  **逐字节相同的 scores**（实测 outputs.bin sha 前后一致 `cff3bbdb…`）。

**验证**（chip 14）：

| 门禁 | 跳过前 | 跳过后 |
|---|---|---|
| infer outputs.bin | `cff3bbdb…` | **`cff3bbdb…`（位相同）** |
| compare / business | 0.006053 / 0.004028 PASS | 同值 PASS |
| bench（2048 req，0 error） | 53.15 exec/s、P50 17.27ms | **61.01 exec/s（+14.8%）、P50 13.90ms** |
| 文本 case e2e p50 | 17.3ms（vision 3.7ms） | **13.86ms（vision 0）** |
| 图像 case e2e p50 | — | 19.86ms（vision p50 4.8ms，路径不变） |

纯文本请求累计收益自统一 AIR 基线（18.55ms，2 槽哑元照算 ~1ms 视觉）：
视觉成本 1ms→3.7ms→**0**；e2e 18.55→13.86ms（**-25%**）。

## 2026-11-10 变更：两阶段拆分（动态视觉 AIR + 文本 AIR）对齐官方不限图数/分辨率

**决策**（用户拍板）：对齐官方"不限数量、不限分辨率"。视觉路径的三个动态维度
（数量、每图分辨率、逐图形状派生）此前被静态槽一次性固定（SLOTS=2、grid
[1,16,16]）；FIA 切换后 S² 掩码已除，唯一剩余障碍是视觉形状派生量。方案按
"视觉/文本拆分 + 关键降险设计"实施：

- **官方位置表作图输入**（对应文本侧 position_ids 先例）：
  `fast_pos_embed_interpolate`/`rot_pos_emb` 是逐图 Python 循环，图内张量化
  易踩顺序坑——改为 CPU 每调用用**官方方法**算好 `vision_pos_embeds
  [T_v,768]`/`rotary_cos,sin [T_v,64]`/`cu_seqlens [n]` 四表作为图输入，
  图内零插值逻辑、表值=官方精确值（pos_embed 表从 checkpoint 读 BF16
  →F32 精确上移，`vision_tables_provider` 绑定官方未绑定方法，同
  `official_rope_index_provider` 模式）；
- **视觉注意力 → FIA TND varlen**：官方逐图非因果全注意力 =
  `npu_fused_infer_attention_score(input_layout="TND", actual_seq_qlen/kvlen
  =累计 cu)`——与文本侧同族机制（bsnd 按行长 0.0022 的先例），段内双向；
  **唯一精度让步：FIA 无 fp32，视觉注意力由 fp32 数学降为 fp16**（旋转仍在
  fp32 官方函数、残差流/表 fp32），门禁把关；
- **两阶段拆分**：vision AIR（5 动态输入 → image_features [T_m,1024]）+
  text AIR（image_features 作为输入替代 pixel_values，注入 index_select+where
  不变）；编码期策略对齐官方：每请求 ≤16 图（vendored MAX_IMAGES）、每图
  256..4096 patch 行（MIN/MAX_PIXELS/256）、h/w 偶数、纯文本 4 行哑元；
  SLOTS/build_static_buffers/固定网格拒绝全部删除。

**runtime**（用户拍板 C++ 最小实现，不带 io_spec）：
`runtime/pipeline_runtime.cpp`（~530 行）：位置绑定（无 write_io/read_io）、
**features 设备常驻直连**（vision 输出 gert::Tensor 重建描述符直喂 text 输入
槽 3，不经主机 D2H/H2D）、infer/bench 双模式（输出格式兼容 compare.py/
verify_business.py）。run.sh 切换到 export_air（两阶段导出，即原 export_split 更名合并——单 AIR 导出
流程连同 write_io 嵌入一并删除）+ pipeline_runtime。

**踩坑与修复**（eager 阶段全部即时暴露）：bridge `ids_rows` 已是 list 再调
`.tolist()`（UnboundLocalError/AttributeError 定位）；`validated` 文本行分支
未初始化；`from_systemone` spec 是单个问题 dict 不是 {q:...} 包装；单流交替
两图 → **GE 每次执行重绑执行计划**（吞吐 9.08 exec/s、P50 81ms、P99 818ms，
requests.csv 分解 vision p50 13.4ms/纯计算 ~1ms 定位）——**每图独立 stream 修
复后 53.15 exec/s、P50 17.27ms**（比统一 AIR 49.7/18.5 快 ~7%，text 图更小
tiling 更优）。

**验证**（chip 14）：

| 门禁 | 结果 |
|---|---|
| 25 项 CPU 测试（含 3 图/3 全局/混合分辨率接受断言新） | 通过 |
| golden 哨兵 | 位相同 `bfc06e59` |
| eager 两阶段冒烟 vs 官方 CPU golden（5 类 case） | 1图 0.00093 / 混合分辨率 2图 0.00440 / 3图 0.00500 / B3 混合 0.00283 / 纯文本哑元 0.00142，全 PASS |
| 两 AIR 导出 | vision 198,629,675 + text 1,507,553,177 bytes（parse_bindings 双射） |
| pipeline infer vs golden | cos 0.999998 / relL2 0.006053 PASS |
| 业务门禁 | 0.004028 PASS |
| bench（2048 req，0 error） | **exec/s 53.15、P50 17.27ms、P99 24.19ms**（统一 AIR：49.68/18.55/25.16） |

**能力对照**（此前 vs 现在）：每请求图数 2 全局 → **16（官方上限）**；每图分辨率
固定 256² → **256²..1024² 任意官方网格**；纯文本视觉成本 1ms（2 槽照算）→
4 行哑元（微秒级）。

## 2026-11-10 变更：编码期上限改为模型自身边界（S 从 config 派生、B 取消上限）

**决策**（用户提问"这些限制是不是可以去掉了"）：FIA 切换后 S 与 B 均无算子级
约束，手选常数失去依据，改为：

- `max_seq_len` 默认 None → **从 folder config 派生**
  `config.text_config.max_position_embeddings`（当前 checkpoint 262144，模型
  真实位置上限；S=262145 编码期拒绝、262144 接受，实测确认）；
- `max_batch` 默认 None → **无编码期上限**（仅保留 batch>=1 检查；显式传参仍
  可收紧，测试沿用）；B 的真实约束是运行期 HBM 的 B·S 交互，README 披露；
- `MAX_BATCH`/`MAX_SEQ_LEN` 模块常数及 unified_preprocess 的 re-export 删除，
  generate_inputs 不再显式传参。

**验证**：24 项 CPU 测试通过（新增 test_row_beyond_the_model_native_cap_rejected
覆盖派生路径；fake config fixture 补 max_position_embeddings=4096）；行为实测
（真实 config）：S=262144 收/262145 拒、B=5000 收、显式 max_batch=2 拒 3 行、
空批拒；generate_inputs 哨兵位相同（`62122c8c`）。**编码期纯 CPU 逻辑，图与
AIR 零改动，无需重导出**（上一条目的全部图腿门禁继续有效）。

## 2026-11-10 变更：full-attention 切换 FIA 按行长注意力（S 上限升至模型原生 262144）

**决策**（用户拍板）：为解除 S² 掩码带来的长度限制，6 层双向 full-attention 由
`npu_fusion_attention`（FA，BNSD + [B,1,S,S] 布尔掩码，B·S² 物化内存是硬约束：
B1/S262144≈68GB>64GB HBM 执行失败）切换为 `npu_fused_infer_attention_score`
（FIA/PromptFlashAttention）**按行长变体**：`actual_seq_lengths` +
`actual_seq_lengths_kv` 双侧传每行真长（int64 张量、运行期进图），**无任何 S²
掩码**；图腿调用 torchair 的 `air` 命名空间算子（`torchair.ops.`，图专用），
eager 腿按用户决策以 `torch.compiler.is_compiling()` 分派到
`torch_npu.npu_fused_infer_attention_score`（torch.ops.npu host 实现，同 kernel）。

**语义实证**（probe，B3/S130 lengths [130,85,61] vs 纯数学双向参考）：
lengths 变体 per-row relL2 0.0021-0.0024（eager 与图腿同值，BNSD/BSND 同
kernel）；对照（无 lengths）padding 行 0.61/0.78——证明 per-row lengths 确实
屏蔽 padding keys。用户贴出的 sparse_mode 文档描述的是 atten_mask **掩码矩阵**
模式（0-4 全要 S1×S2 矩阵）；per-row lengths 是正交的 varlen 机制，实测支持
"固定矩形 [B,S] 布局上的行内双向注意力"。

**probe 踩坑记录**（全部已实证定位）：①FIA 拒绝非连续输入（文档"不支持非连
续的Tensor"），`transpose` 后直传报 561002=ACLNN_ERR_INNER_TILING_ERROR——
解法为 **BSND 布局免转置**（q/k/v 天然 [B,S,N,D] 连续，顺带省掉 FA 形态每层
4 个 transpose）；②probe 参考实现两处 GQA 展开写错（`repeat` 应为
`repeat_interleave`、`head % KV` 应为 `head // (HEADS//KV)`）曾致 relL2≈1 的
假阴性——对照单头手算定位后修正；③torchair.ops 版 eager 抛
NotImplementedError（图专用，Meta impl 齐全可被 dynamo 追踪）。

**验证**（chip 14，全门禁）：

| 门禁 | FIA | FA（切换前） |
|---|---|---|
| eager 冒烟 vs golden | relL2 0.00798 | 0.00408 |
| 图 vs golden | cos 0.999998 / relL2 **0.006006** | 0.004496 |
| 业务门禁 | 0.004028 | 0.003431 |
| **S=262144（模型原生上限）** | **可执行，63s，数值正确** | **执行失败（68GB 掩码）** |
| B8/S99、B1/S8192 vs golden | 0.00256 / 0.00281 | — |
| bench（2048 req，0 error） | **exec/s 49.68、P50 18.548、ticket-team 18.1ms** | 49.686 / 18.651 / 18.3ms |
| AIR | 1,708,244,818（−25KB，免转置） | 1,708,269,731 |

**性能结论**：历史担心的 FIA 4× 单算子慢（2026-11-09，B1/S99，BNSD+转置形
态）在整模上**完全被 BSND 免转置 + 免掩码物化抵消**——吞吐持平（49.68 vs
49.686 exec/s），ticket-team 单 case 略快；历史 −3.5% 回退是 BNSD+转置形态
所致。**S 上限**：MAX_SEQ_LEN 131072→**262144**（=模型原生
max_position_embeddings；B·S² 约束解除，残余约束为 B·S 激活 HBM 与 B 阶梯
1024 上限，README 披露不变）。golden/inputs 哨兵位相同（`bfc06e59`/
`62122c8c`），23 项 CPU 测试通过，位置式权重 bake 重验（图输出正确即证）。

## 2026-11-10 变更：放开 MAX_BATCH / MAX_SEQ_LEN（4→1024、2048→131072，实测包线定值）

**决策**（用户拍板"先把这两个限制放开"）：旧值来自 P0 方案的首版交付范围
（非模型原生上限，见 adaptation-plan.md:74；模型原生 max_position_embeddings
=262144）。放开采用**实测包线定值**：图本就是 B/S 双动态轴（无需重导出），
用现有 AIR 经 ge_runtime 逐形状探测，上限 = 已验证的最大形状。

**探测**（chip 14，合成文本行经 build_multimodal_batch 打包，每形状独立
请求隔离失败；脚本未入库：/tmp/opencode/boundary_probe*.py，方法如上可复现）：

- S 阶梯（B=1）：4096/8192/16384/32768/65536/**131072 全过**；**262144 编译
  通过但执行失败**（E19999 Inner Error，量级与 [B,1,S,S] 布尔掩码物化
  ~68GB 超 64GB HBM 吻合）；
- B 阶梯（S=99）：8/16/32/64/128/256/512/**1024 全过未触顶**（B 轴为软边界，
  受 HBM 约束）；
- 组合角点全过：B2/S131072、B4/S32768、B8/S16384、B16/S8192、B32/S4096、
  B64/S2048、B128/S1024（掩码若物化 B2·S131072² ≈ 34GB 亦可通过）。

**数值锚定**（超越旧包线的形状，图 vs 官方 CPU FP32 golden，同一合成输入）：

| 形状 | worst cosine | worst relL2 | 判定 |
|---|---|---|---|
| B=8, S=99（新 B 包线内） | 1.000000 | 0.00297 | PASS |
| B=1, S=8192（超旧 S 上限 4×） | 0.999996 | 0.00581 | PASS |

**定值**：`MAX_BATCH = 1024`、`MAX_SEQ_LEN = 131072`（写入
modeling_multimodal.py 顶部，注释含探测依据）。**披露**：两轴为独立上限，
极端角点（如 B1024×S131072）超出 HBM（激活 ~B·S、pad 掩码 ~B·S²），在运行
期以 GE 内部错误失败而非编码期拒绝——README 使用限制已如实记载。

**回归**：23 项 CPU 测试通过（其中 test_row_beyond_the_cap_rejected 原以硬
编码 2049 行测旧上限，已改为从单一来源取 `MAX_SEQ_LEN + 1`——测机制而非数
字）；golden/inputs 哨兵位相同（`bfc06e59` / `62122c8c`，官方 case 池输入不
变）；编码期超限拒绝随新默认值生效（S=131073、B=1025 拒绝，S=131072 接受，
实测确认）。

## 2026-11-10 变更："未用语言/框架自带能力"审计（stdlib/去重复）

**审计**（用户提问驱动）：逐文件排查"手写了语言或框架已提供的能力"的地方。

**修整（4 处）**：

- `golden.py` / `generate_inputs.py` 各自手写的 `STARTLUX_MODEL_DIR` 读取+
  校验函数删除，改用包内单一来源 `modeling_multimodal.source_folder()`（连同
  两个文件里仅为它存在的 `import os` 一起删除；`verify_business.py` 按设计
  不 import 模型包、独立读 folder 侧——保持不动，见 2026-11-10 清理条目）；
- `modeling_multimodal.build_multimodal_batch` 的 4 行 `_TokenIds` 类改用
  `types.SimpleNamespace`（同文件既有惯用法）。**注意**：该函数内局部变量
  `types`（mm_token_type_ids 张量）会遮蔽模块级 `import types`，故必须
  `from types import SimpleNamespace` 按名导入——首轮改动因此触发
  UnboundLocalError（UnboundLocalError 被 CPU 测试与 generate_inputs 立即
  捕获，哨兵门禁证明了其有效性），修正后 23 项测试通过；
- `unified_preprocess.image_data_uri` 的手写 mime 字典改用 stdlib
  `mimetypes.guess_type(name) or "image/png"`（官方 examples/image_request.py
  同款用法）。mime 在本流程中只流入 `media.data_uris` 属性（golden/
  generate_inputs 均不读取），对编码产物零影响——inputs 哨兵位相同证明。

**判定为"框架真缺能力、现行方案已是耦合最低"而不动的**：

- `export_air.parse_bindings`（36 行 pbtxt 文本解析）：torchair
  `dynamo_export` 无返回值，导出图是 backend 内部局部对象（导出即出
  `_graph_stack` 栈），文本解析回填需未安装的 tensorflow proto——替代方案只有
  monkey-patch torchair 私有函数 `make_export_graph` 捕获图对象，对私有签名
  的耦合高于对 pbtxt 文本格式的耦合，维持现状；
- `modeling_text` 的 `_softplus`（aten.softplus 无 GE converter）、
  `causal_conv4`（bf16 conv 误编译）、MRoPE gather 索引（官方 in-place 切片
  不可导出且 gather 形态已证位精确）、`modeling_vision.build_static_buffers`
  的 SimpleNamespace 方法绑定（官方方法是实例方法，构造完整模型才能直接调，
  而那会加载全部权重）——均为已记录的数值/导出性约束；
- `verify_business.py` 的 N_COLUMNS=26 硬编码：它承担"folder 侧 26 ==
  scores 侧 26"双侧契约的独立半边，若改为 `len(letters)` 派生则校验恒真，
  保持硬编码；
- 各入口脚本的 `Path(__file__)` 根锚点：sys.path 引导必须先于包 import
  （鸡生蛋），无法从包内获取。

**验证**：23 项 CPU 测试通过；golden 哨兵位相同 `bfc06e59`；inputs 哨兵位
相同 `62122c8c`；被追踪进 AIR 的代码（modeling_text / modeling_vision）零
改动，无需重导出。

## 2026-11-10 变更：model/ 转为 Python 包（删 _load_module 按路径加载机制）

**决策**（用户提议）：`model/` 加空 `__init__.py` 成为包，按路径 +
合成名注册的 `_load_module` 机制整体删除（该机制存在的唯一原因就是目录
不是包；其 "register before exec: dynamo resolves traced functions by
__module__" 注释防护的问题，包导入天然满足——`model.modeling_text` 等真实
模块名本就在 `sys.modules`）。

- 删除 5 处 `_load_module`（modeling_multimodal / unified_preprocess /
  export_air / generate_inputs / golden）+ 测试的 `load` helper 及全部合成
  模块名；`MODEL_DIR` 从 modeling 文件中彻底消失，入口脚本保留的仅用于
  定位 `work/` 数据目录；
- 入口脚本各自把模型根目录 `sys.path.insert` 一次（export_air 与
  unified_preprocess 位于包内、golden 位于 decider/、generate_inputs 位于
  根、测试位于 tests/），`run.sh` **零改动**；
- `modeling_multimodal` 对 `modeling_text` 保持函数级惰性导入（torch_npu
  免导入属性不变，golden/generate_inputs 的 CPU 路径不变）。

**验证**（chip 14）：golden 哨兵位相同 `bfc06e59`；generate_inputs 哨兵
位相同 `62122c8c`；23 项 CPU 测试通过；eager 冒烟 relL2 0.00408（与包化前
逐值一致）；导出 + 推理门禁 relL2 0.004496 / 业务 0.003431（与融合算子
回切版完全一致）；AIR 字节数不变 1,708,269,731（sha 因图节点名内嵌模块名
而异，语义等价由门禁证明）；bench 49.917 exec/s、P50 18.523 ms、0 error
（带内）。

## 2026-11-10 变更：model/ 模块化重写为自持权重 nn.Module 树（HF 风格）

**决策**（用户逐项拍板）：(1) dtype 在模块声明处指定、照 vLLM；(2) 原地替换
`model/`；(3) 参照 cann-recipes-infer `modeling_qwen.py`（纯 `nn.Module` +
自写加载，不 subclass `PreTrainedModel`）；(4) 删除全部
`torch.compiler.is_compiling()` 双腿分支，单腿直接调 NPU 算子；(5) vision 的
`fast_pos_embed_interpolate` / `rot_pos_emb` 继续 import 官方方法；(6) 双向
注意力硬编码，不读 `jev_prompt_attention`。

**RoPE 定案（停止协议触发）**：`npu_apply_rotary_pos_emb` 的 `layout=1` 在
GE 图腿解读为 BSND、在 aclnn eager 腿解读为 BSH（本文件 2026-11-09 条目与
performance.md 附录各记录一次，relL2 0.88 级语义分歧）。按用户判据"算，停下
来"，融合算子弃用，RoPE 恢复**显式 rotate-half 数学**（即原 eager 腿实现，
其图内形态与 P2-5 已验收链相同）——单腿在 eager 与 graph 两腿均正确。代价与
2026-11-09 实测一致：吞吐持平（方差内），AIR +107KB。

**同日复测修正（probe，用户问"现在还是不对吗"后执行）**：以交付调用形态
（`layout="BSND", rotary_mode="half"`、bf16、真实 GQA 形状 q[B,S,8,64]/
k[B,S,2,64]、官方 theta 1e7）做三方对账——纯数学参考 vs eager 直调 vs
torchair `dynamo_export`（复用 export_air 全套机制，trace B2/S65 动态双轴）
经 `ge_runtime` 执行六种 B/S（含历史 bug 场景 B=1）：**两腿全部 relL2
0.0013（bf16 量化级）且逐位一致，历史分歧在当前构建不复现**。根因定位：
torchair GE converter（`ge_converter/custom/apply_rotary_pos_emb.py:30-37`）
把字符串 "BSH" 与 "BSND" 混叠映射到同一枚举值 1（GE 侧 1=BSND），而当时的
aclnn host 把 1 解读为 BSH——同一枚举两腿语义相反；当前构建该错位已消除。
**融合算子回切（同日，用户在 probe 结果后拍板"换回融合算子"）**：
`StartLuxAttention.forward` 恢复 `npu_apply_rotary_pos_emb`（BSND + half，
旧交付验证过的图腿形态：q/k [B,S,N,rot] 切片、cos/sin [B,S,1,rot]；eager 侧
切片传入后不再被读取，in-place 语义无害）。eager 侧**非连续切片输入**是
probe（连续张量）之外唯一未测组合，由下方 eager 冒烟覆盖。接口风险点记录
在案（converter 的 in-place 警告与 "BSH" 字符串混叠），当前构建两腿一致
已实证。probe 脚本未入库（临时 `/tmp/opencode/rope_probe.py`），方法与
判据如上可复现。

**回切后全链复验**（chip 14，eager 冒烟 + 导出/推理/双门禁 + bench）：

| 门禁 | 融合算子（回切后） | 显式数学（回切前） | 重写前基线 |
|---|---|---|---|
| eager 单腿 vs golden | cos 0.999999 / relL2 **0.00408** | 0.999998 / 0.00840 | — |
| 图 vs golden | cos 0.999999 / **relL2 0.004496**（与基线全精度逐位一致） | 0.005362 | 0.004496 |
| 业务门禁 | worst 0.003431（与基线记录值一致；逐 case prob_max_abs 全部更贴 golden） | 0.003790 | 0.003431 |
| bench（2048 req） | exec/s 49.622、P50 18.684、P99 25.175、0 error | 49.686 / 18.651 | 49.489 / 18.785（post-cleanup） |
| AIR | **1,708,269,731 bytes**（较基线 −10KB） | 1,708,386,627 | 1,708,279,731 |

结论：融合算子回切后，新模块树的图输出在 worst relL2 上**全精度复现**重写
前基线（0.00449550810581621），业务逐 case 概率误差全部优于基线；eager 单
腿（含非连续切片输入）0.00408 亦达量化级。AIR 大小回到基线水平。

**新结构**（`model/`，6 → 5 个文件）：

- `modeling_text.py`：`StartLuxRMSNorm`(gamma=(1+w)，zeros-init) /
  `StartLuxRMSNormGated`(gamma=w，ones-init——两者语义不同，类 docstring 各自
  声明) / `StartLuxMLP` / `StartLuxGatedDeltaNet` / `StartLuxAttention` /
  `StartLuxDecoderLayer` / `StartLuxTextModel`（含 MRoPE 表）；
- `modeling_vision.py`：`MatMulPatchEmbed`(参数名 `proj` 保持 checkpoint 零映射) /
  `StartLuxVisionAttention`(带 `num_key_value_groups=1`，官方 eager 接口所需) /
  `StartLuxVisionMLP` / `StartLuxVisionBlock` / `StartLuxVisionPatchMerger`
  (精确 `nn.GELU()`，与 MLP 的 gelu_pytorch_tanh 不同) / `StartLuxVisionTower`
  (`build_static_buffers` 加载后经官方方法预算常量)；
- `modeling_multimodal.py`：`StartLuxModel`/`StartLuxForDecision`（参数树镜像
  checkpoint：`model.visual.*` + `model.language_model.*`，`load_state_dict` 零
  映射）+ `from_folder`/`load`/`export_spec` + 原组批桥/`io_spec`/
  `official_rope_index_provider` 原样迁移；`modeling_text` 惰性加载（golden/
  generate_inputs 的 CPU-only 导入属性保持）。

**加载契约**（`from_folder`）：safetensors → `load_state_dict(strict=False)`，
断言 `missing == []` 且 unexpected 恰为 `mtp.*`（15 张量，MTP 头 Instant 不
用；HF `Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`
官方声明可忽略）→ `derive_decision_head`（tied embedding 26 行 F32 克隆，
`tie_word_embeddings=True` 有断言）→ A_log F32 计数/dtype 断言 → 官方方法预算
vision 常量 buffer → device-only `.to`。**随之删除**：`_expected_param_dtype`
逐名 dtype 契约表、`_verify_candidate_placement` 的 dtype 部分（保留 5 行
device 断言）、`_restore_and_verify` 的 A_log 特例复原、vision `inv_freq` 重生成
workaround（新路径逐张量 `copy_`，从不对整模型 cast dtype）。dtype 与 vLLM
`qwen_gdn_linear_attn.py:804-831` 逐项对齐：A_log 显式 F32、`dt_bias`/
`linear_attn.norm.weight` 跟随模型 dtype（bf16）、vision Linear/Conv weight+bias
fp16、LayerNorm 与 `pos_embed` fp32。

**验证**（chip 4，基线先行固化：HEAD 重跑 export+infer 与既有产物位相同
`e7157ae1…`，证明 08:19 前基线有效）：

| 门禁 | 结果 | 基线 |
|---|---|---|
| golden 哨兵（重写不改 oracle） | **位相同** `bfc06e59…` | `bfc06e59…` |
| eager NPU 单腿 vs golden（5 case/7 行） | cos 0.999998 / relL2 0.00840 | —（旧实现无 eager 路径对照） |
| 图（新 AIR）vs golden | cos 0.999998 / **relL2 0.005362** PASS | relL2 0.004496 |
| 业务门禁 | max abs 0.003790 PASS | 0.003431 |
| bench（2048 req，0 error） | exec/s **49.686**（14 号卡干净窗口）/**50.186**（4 号卡干净窗口）、P50 18.651/18.433 ms | 带内：49.231–50.942、post-cleanup 49.489/P50 18.785 |
| 剩余 CPU 单测 | 23 项通过 | — |
| AIR 算子指纹 | 与基线一致（MatMulV2=100、ChunkGatedDeltaRule=36） | 同 |
| AIR 大小 | 1,708,386,627 bytes（+107KB，显式 RoPE 节点） | 1,708,279,731 |

关键风险关闭：**frozen_parameter 位置式权重 bake 与新参数枚举对齐**——新
AIR 经 ge_runtime 推理输出正确（上表图门禁），导出内 `parse_bindings` 双射校验
通过。**WeightNzAndMatMulV3Pass**：系统安装未动、新图匹配面与基线一致（AIR
内 100 个常量权重 MatMulV2，pass 在 ge_runtime 编译期生效），mixed exec/s 带
内；单 case AI-time profiling 未复测（基线口径为 msprof AI 时间，与 bench
exec wall 口径不可直接对账），如需闭环可按 2026-11-09 `work/prof-nz/` 方法
复测。

bench 窗口敏感性：闭环 host-bound，2026-11-10 09:0x-09:1x UTC 主机多租户
（loadavg 峰值 54/100/181，npu-smi 全机 200–270W），同 AIR 同种子 5 次测量
exec/s 随负载波动：50.186 → 37.385（run.sh 内，全机忙）→ 45.991 → 47.750
（14 号卡，load 54+）→ **49.686**（14 号卡，load 回落，记录值）。精度与负载
无关：worst relL2 0.005362 每次逐位复现。完整 run.sh（chip 4）exit=0 全门禁
通过。详见 performance.md 当日附录。

eager 证据另含：单腿 eager 前向 vs 旧图输出 cos 0.9999998 / relL2 0.00733
——与基线图的差异是算子内部精度级（FA 图/eager 实现路径不同），非语义差。
eager smoke 脚本：`/tmp/opencode/eager_smoke.py`（临时，未入库；复跑方法见
本条目描述：`mm.load(0)` + `build_multimodal_batch` + 逐 case 前向对比）。

同日修正本文件一处历史遗留矛盾：下文 2026-11-09"补充证据（二轮 probe）"段
的"结论与回退（维持）：P2-7 回退，eager RoPE 链保留"系**被三轮 probe 推翻
的过时结论**（二轮以 aclnn eager 为参考，而 aclnn eager 的 layout=1 自身即
错 0.88）；实际交付一直是"图腿融合 RoPE + eager 腿显式数学"，直至本次重写
统一为单腿显式数学。

## 2026-11-10 变更：兼容原型加载收敛为单一 LD_PRELOAD 契约（删 python activator）

**决策**（用户拍板）：将 ChunkGatedDeltaRule 兼容原型的加载由"python activator
（import 时 ctypes dlopen RTLD_GLOBAL + arm 任务级 LD_PRELOAD）叠加 run.sh
LD_PRELOAD"两套重叠机制，收敛为**单一 `LD_PRELOAD` 契约**。仅改加载机制，
**不触碰** `.so` 的 `REG_OP`/infer、图内容、GESession 选项，数值行为不变。

动机（冗余实证）：`run.sh` 已对 export/infer/bench 三阶段注入
`LD_PRELOAD=$PROTO_LIB`；export 进程启动时动态链接器已全局加载 `.so` 并跑完
`REG_OP` 静态注册，故 activator 的 `ctypes.CDLL(RTLD_GLOBAL)` 为重复加载、
`_prepend_ld_preload` 因去重判断空转；`export_air.py` 的 `write_io` 子进程经
`subprocess.run`（未传 `env=`）继承 `os.environ` 里 run.sh 设的 `LD_PRELOAD`，
无需 activator 再 arm。即 activator 在 run.sh 与 README（均已 `LD_PRELOAD`）
两条路径下都无独有能力。且 `LD_PRELOAD` 在解释器启动前完成注册，时序上早于
并稳于"import 时 dlopen"（后者依赖 `_activate_proto()` 必须早于 torchair
`get_registered_ir_def` 检查的手工时序）。native `ge_runtime`（infer/bench）
本就是纯 `LD_PRELOAD`、无 python，已证明该机制自足。

改动：删除 `native_proto/python/chunk_gated_delta_rule_proto.py`（activator）；
`scripts/build_chunk_proto.sh` 不再安装 `install/python`（并 `rm -rf` 旧
`opp/` 与 `python/`，安装布局收敛为 lib + dependency.info）；`export_air.py`
以 `_require_proto_preloaded()`（启动即校验 `LD_PRELOAD` 含原型 `.so`，缺失则
明确报错指向 `prepare.sh` / `LD_PRELOAD=<proto.so>`，不再静默失败在 torchair
检查里）替代 `_activate_proto()`；`prepare.sh` / `startlux_npu_adapter.py` /
模型 README / `native_proto/README.md` 注释同步。关闭条件不变（CANN 自带同名
注册时移除 prepare.sh 依赖构建与 run.sh 的 LD_PRELOAD 注入；禁止双注册）。

验证（CPU，hermetic）：`tests/test_startlux_native_proto.py` 重写为契约测试
（guard 通过/失败 4 项 + activator 已退役 2 项）**6/6 通过**；`export_air.py`
`py_compile` 通过；`build_chunk_proto.sh`/`prepare.sh`/`run.sh` `bash -n` 通过。
**NPU 全链门禁（`run.sh` 精度 + 性能）待在空闲卡重跑**：因加载路径变更，需确认
export 在 `LD_PRELOAD` 下 `REG_OP` 注册成功、`write_io` 子进程继承、infer/bench
数值与性能不变；通过前不宣称交付态等价。

## 2026-11-10 变更：代码审查清理（删死代码 / 修错误 / 削冗余校验 / 注释去过程化）

**决策**（用户拍板）：按代码审查结论清理模型目录，只保留独立脚本流程实际使用
的功能；不改动任何数值行为，因此**不需要重跑 NPU 精度或性能门禁**。

### 修正的错误

- `run.sh`：`die()` 定义在参数解析循环之后，`bash run.sh --bogus` 实际输出
  `die: command not found` 并以 127 退出，真正的错误信息丢失。已将
  `step()`/`die()` 上移至 `set -euo pipefail` 之后；实测两种误用均打印正确
  信息并以 1 退出。
- `model/vision.py`：`forward()` 中 `stream` 变量赋值后从未使用，而其上方注释
  宣称"fp16 residual stream … Validated on the real 12-block tower: relL2
  0.0079"。实际循环全程 FP32（`patch_embed` 与 `_linear` 均 `.to(x.dtype)`
  回到 fp32）。按用户决策**删变量 + 将注释改写为当前真实结构**（FP32 残差流 +
  FP16 Linear/Conv 内积），不实现 fp16 残差流，故数值行为不变。
- 视觉精度口径自相矛盾：`model.py` 的 placement 报告写 `"vision": "float32"`、
  `multimodal.py` 的 `describe()["precision"]` 与 `weight_storage`、README
  混精度条目均称视觉塔 FP32，但 `model.py:_expected_param_dtype` 强制 7 类
  vision Linear/Conv 参数为 **float16**（P2-5 预量化）。已统一为
  "FP32 残差流 + FP16 Linear/Conv 内积"，placement 报告拆为
  `vision_linear` / `vision_residual_stream` 两字段。
- README 称 mega 分支"保留为 `gdn_route="mega"` 休眠选项"，但该代码已由
  提交 76f29cf 全部删除（grep 无 `gdn_route`）。已改为"代码已移除，测量证据
  保留"。
- README / 本文 / `docs/standalone-runtime-plan.md` 称 `decider/fixtures/`
  （`p2b-readouts.json`、`b4-long-rows.json`）存在，实际目录不存在；
  `generate_inputs.py` 的 `FIXTURES` 常量亦从未使用。已删常量并更正文档。
- README 与本文写 `model/verify_business.py`，实际路径为
  `decider/verify_business.py`。已更正。
- `scripts/env_chunk_proto.sh` 从未被 source，且其拼接路径
  `<work_dir>/dependencies/...` 与 `prepare.sh` 实际安装位置 `work/deps/...`
  不一致（即使 source 也必然报 "does not exist"）。已删除，并清理
  `native_proto/README.md`、`chunk_gated_delta_rule_proto.cpp`、
  `scripts/build_chunk_proto.sh` 中的引用。
- `model/export_air.py`：`build_request()` 的列表推导内部每条 record 重建一次
  `set(case_ids)`。已提出为局部变量，并改用集合差直接算 missing。

### 删除的死代码（均经 AST + 全仓 grep 双重确认 0 生产引用）

按 `_load_module` 精确 import 图验证，以下文件不在 `run.sh` 的调用图内，仅被
测试引用，已连同其测试删除：

| 文件 | 行数 |
|---|---|
| `decider/ge_readout.py` | 701 |
| `decider/unified_readout.py` | 461 |
| `decider/reference.py` | 339 |
| `decider/decision.py` | 164 |
| `scripts/monitor_idle_benchmark.py` | 1243 |
| `scripts/env_chunk_proto.sh` | 56 |
| `tests/test_startlux_ge_readout.py` | 663 |
| `tests/test_startlux_unified_readout.py` | 759 |
| `tests/test_startlux_decision.py` | 927 |
| `docs/production-integration.md` | 178 |

框架路由残留（README 已声明交付形态"无 YAML / adapter / handle"）：
`startlux_npu_adapter.py` 的 `RuntimeHandle` / `CandidateHandle` /
`ReferenceHandle` / `prepare_reference`（全仓 0 调用）/ `prepare_candidate` /
`prepare_runtime` / `_rows_of` / `_payload_rows` / `_pad_batch` /
`_encode_dynamic` / `_alignment` / `_dynamic_axes` / `_dynamic_io` /
`_example_geometry` / `adapt()`；`multimodal.py` 的 `MultimodalRuntimeHandle` /
`MultimodalCandidateHandle` / `prepare_candidate` / `prepare_runtime`。

单点死符号：`multimodal.model_rope_index_provider`、
`example_inputs(device=)` 的 device 分支、`MultimodalDecoder.slots`、
`IMAGE_TOKENS`（与 `vision.MERGED_ROWS` 重复）、三处未使用的 `ROOT`；
`StartLuxTextDecoder.forward`（仅经已删的 `CandidateHandle.run` 可达）、
`_rope`、模块级 `rope_cos_sin`、`mrope_cos_sin` 的 2D 分支、`_gated_delta`
纯转发空壳、`self.chunk_size` 与整条 `chunk_size` 参数链、`self.embedding_shape`、
`adapt()` 的 `wrapper.width` 与整条 `compiled_seq_len` 链、`_gated_delta` 中
未用的 `value_dim` 解包；`unified_preprocess` 的 5 个死别名
（`GRID_THW`/`PATCH_ROWS`/`MERGED_ROWS`/`SLOTS`/`IMAGE_TOKENS_PER_IMAGE`）、
`encode_case_rows`、`encode_readout_row`、`pad_graph_inputs` 与整条
`min_width` 管道（独立流程恒传 1 → `target == width` 早退，整个函数是
no-op）、随之失效的 `numpy` 导入；`generate_inputs` 的 `FIXTURES`、
`_validate_against_io`、`_save_row_encodings` 的死返回值；`golden` 的
`self.source`、`self.vocab_size`、`ROOT`、`N_COLUMNS`；`vision` 的
`rotary_inv_freq_restored`；activator 的 `INSTALL_RELPATH`、
`library_path(work_dir)` 的 fallback 分支、`_FALLBACK_DONE` 去重（模块级只
import 一次）。

`multimodal.validate_batch_arrays`（105 行）一并删除：它原是已删的
`MultimodalRuntimeHandle.encode`（外部 JSON payload 入口）的数组级门禁，移除
`build_multimodal_batch` 末尾的二次调用后已无生产调用方；其检查项对
`build_multimodal_batch` 自身构造出的数组恒真（feature index 的 0,1,2,… 顺序
由 slot 计数器保证、padding 区保持 zero/-1 初始化）。对应的
`TestBatchValidation` / `TestPaddingIndexStrictness` 两个测试类删除；活路径的
拒绝行为由 `TestBridgeInputHygiene`（走 `build_multimodal_batch`）覆盖。

### 收敛的间接层

- `describe()` 的 ~60 键报告字典收敛为 `multimodal.io_spec()`（无参，纯静态
  字面量）：独立流程只消费 `io.inputs` / `io.outputs`，其余
  （`bounds.slot_scope`、`grid_policy`、`feature_order`、`cpu_side.*`、
  `precision.*`、`changes`、`readout_contract`、`dynamic`、`native_gdr`、
  `required_dependencies`、`referenced_files`）为与模块 docstring 重复的散文，
  已删除。`startlux_npu_adapter.describe()` 整体删除。
- `_params(config)` 删除：三个调用方全部传 `"params": {}`，全套校验
  （allowed 集合、未知键拒绝、正整数、`compiled_seq_len <= max_seq_len`）
  校验的是硬编码默认值。边界常量单一来源收敛为 `multimodal.MAX_BATCH` /
  `MAX_SEQ_LEN` / `SLOTS` 与 `vision.GRID_THW`；`vision_linear_dtype` 配置项
  删除，`build_unified_model` 直接传 `torch.float16`（`VisionTowerWrapper`
  的 `linear_dtype` 参数保留，测试仍以 fp32 作为位精确对照）。
- `_source_ref(config)` 的 `${VAR}` 正则替换与相对路径解析删除：两个调用方都
  传已 resolve 的绝对路径。`_decision_config(config)` 改为
  `_decision_config(folder)`，返回值收敛为 `{letter_token_ids, vocab_size}`
  （`temperature_by_type` / `token_ids` / `folder` 仅被已删的 `describe()`
  消费；温度校验在唯一使用方 `verify_business._row_report` 内做）。
- `model.py` 的 `_config()` / `io_contract()`、`generate_inputs.py` 的
  `_config()`、`golden.py` 的 config 字典删除；`build_unified_model(model)`
  自行从 `model.name_or_path` 取 folder，`load()` 返回
  `(unified, placement)`。
- `build_graph_inputs` 的 `hasattr(row["input_ids"], "dim")` 双模式嗅探删除，
  只接受已编码行（唯一生产调用方 `generate_inputs.py` 本就两步调用）。

### 削除的冗余校验

同进程内重复校验与对自造数据的校验：`build_multimodal_batch` 末尾对
`validate_batch_arrays` 的二次调用；`_check_row_structure` 中两处恒假检查
（`len(groups) != n_images` 与 pad 连续性——前者由 pads 总数 + 每组等长蕴含，
后者由 pads 是升序全量位置、groups 是其等长切块蕴含）；`_row_groups` 的重复
计算（改为接住 `_check_row_structure` 的返回值）；`pixel_blocks.append(None
if pixels is None else pixels)` 的等价三元；`_require_cpu_tensor(types, ...,
types.dtype)` 把自身 dtype 当期望值传入导致 dtype 检查恒真（改为直接查
dtype，并把 `types_row` 比对移入"调用方提供了 types"的分支——原写法在
`types is None` 时比对恒真）；`model.py` 的 `source.dtype != bfloat16` 自检
（上一行刚 `.to(dtype=torch.bfloat16)`）；`vision.py` 预量化循环的
`param is not None` 冗余与 fp16 上溢检查（对 bf16→fp16 视觉权重不可能触发，
且权重在 NPU 上，**每个参数强制一次 D2H 同步**）、`slots < 1`（唯一调用方传
常量 2）、手工搬 buffer 至 target device（唯一生产调用方 `model.py` 就是
`unified.to(target)`，且 `_verify_candidate_placement` 会逐名检查 device）；
`vision_linear_dtype` 的双重校验；letters 一致性与 tied head 的重复校验
（`_restore_and_verify` 保留 tied 检查——head26 克隆自 `embed_tokens`，未
tied 会静默投影错行，属数值契约）；`generate_inputs` 对自身 20 行前硬编码
构造的数据的 5 处自检（26 列、case id 去重、`samples == len(rows)`、占位符
计数、`_validate_against_io`）；`export_air` 的二次 `_validate_io`、被首个
条件蕴含的 `len(bindings) != len(names)`、包 argparse `type=int` 结果的
`int()`、与 `run.sh` 重复的删 AIR；`run.sh` 的 `[[ -f "$PROTO_LIB" ]]` /
`[[ -x "$RUNTIME_BIN" ]]` / 两处导出产物存在性检查（`prepare.sh`、
`runtime/build.sh`、`export_air.py` 均在 `set -e` 下自行失败）；图片数 ≤2 的
三重复查（收敛到 `build_multimodal_batch` 单一权威门禁，并把硬编码 2 改为
`slots`）；`min_width` 正整数的两处重复校验；`golden.py` 的 `samples` 正整数
/ shape / finite 三项自检（数据来自同流程上游，`torch.stack` 形状由构造保证，
非有限值会使 `tools/compare.py` 的 cosine 门限失败）。

**保留**的校验：跨进程/跨阶段文件边界（AIR 嵌入 IO 的 `_validate_io(embedded)`、
`outputs.bin` 字节范围、`records.json` / `cases.json` 的 schema 标记、
`STARTLUX_MODEL_DIR`、`ge_runtime` 的 returncode 与"恰好一行 GE_RESULT"）与
数值契约（A_log 位精确、head26 位精确、tied head、grid `[1,16,16]`、B/S/图数
上限、feature index 值域、`_check_row_structure` 的 marker 结构）。
`verify_business.py` 的 26 列检查保留——它直接读 `decision_config.json`
（不经 `_decision_config`），是"folder 侧列数 == scores 侧列数"这一双侧契约的
folder 半边。

### 注释去过程化

源码注释中的变更日志、验收数字与不可访问的 `work/*.json` 路径（`work/` 不提交
Git）已移除或改写为"为什么"：`startlux_npu_adapter.py`（模块 docstring 的
路线删除史、`_rms_norm` 的 "NPU-verified 2026-11-06 / pbtxt diff probe"、
`causal_conv4` 的 "Route evidence (2026-11-05 …)" 段、L2-norm 的
"NPU-probed relL2 0.0029 / probed WRONG at relL2 0.85"、P2-7 / P2-8 的探针
数字）；`vision.py`（模块 docstring 的 P6 报告路径、`MatMulPatchEmbed` 的
GE profiling 数字、`_linear` 的 "bf16 FAILS … relL2 0.055"、预量化循环的
"the buffer-copy approach failed exactly there"、rotary 的
`review-vision-rotary-buffer-cast.json` 引用）；`multimodal.py`（精度段的
"validated in p6-official … rel_l2 ~0.11"、`build_unified_model` docstring 的
`p4-dynamic, 27/27 GE` 与 `core.sources.load_source` 叙事）；
`unified_preprocess.py`（17 行 P4 pad256 退役史、processor 组合的失败史与
p6 runner 路径、`DEPS_HINT` 中不存在的 `p6-unified/env.sh`）；
`generate_inputs.py`（8 行 min_width 退役叙事）；`export_air.py`（4 处
"migrated from the validated framework exporter"、"the formal-11 bench pool"
——实际池为 5 case、以及与 run.sh 顶部重复的固定参数数值）；`golden.py` /
`verify_business.py`（"the migration of the validated …" 与 "the accepted
policy"）。数值证据本身保留在本文与 `performance.md`。

### 未改动项（明确记录）

- `decider/vendor/` 的 13 个官方模块**全部保留**（用户决策：先保留，后续再
  精简）。删除 `decision.py` / `reference.py` 后，活路径只需
  `images` / `prompt` / `backends.bidirectional` 及其传递依赖
  `backends.base`（经 `backends/__init__.py`）与 `__init__.py`；
  `modelcfg` / `controls` / `pipeline` / `readout` / `loops` /
  `backends.fake` / `backends.transformers` 变为不可达。SOURCE.md 的
  集成点说明已更新，子集仍与固定上游 commit 逐字节一致。
- `_load_module` 的 6 份逐字拷贝**未合并**：其模块命名（如
  `startlux_multimodal_for_unified`）对 dynamo 的 `__module__` 解析与测试的
  模块同一性是承重的，抽公共模块需改 `sys.path` 注入方式并重跑 NPU 导出验证，
  超出本次"不改数值行为、不依赖 NPU"的清理范围。已确认 6 份中 3 份丢失了
  "# register before exec" 说明注释（拷贝漂移的实证），留待后续统一。
- `docs/adaptation-plan.md` 未改（文件头已自声明为历史设计记录，其中的
  `config/model.yaml` 等引用属史实）。
- `verify_business.py` 的 `business: false` 跳过分支保留：当前 5 个 case 全为
  `business: True`，该分支在交付流程中不可达，但 `business` 标志同时是
  `generate_inputs.py` 的 bench 池选择器，属有文档的既定行为而非防御性代码。
- `fusion_result.json` 与 `kernel_meta/` 落在模型源码目录（均已 gitignore），
  属 GE dump 落点配置问题，未在本次范围内调整。

### 验证

- startlux CPU 单测 **51 项全部通过**（`tests.test_startlux_native_proto` /
  `test_startlux_adapter` / `test_startlux_multimodal` /
  `test_startlux_unified_preprocess`），清理前基线为 236 项（含已删除的
  3 个测试文件）。全部剩余用例覆盖活路径。
- 全仓 `unittest discover`：204 项，4 个 error 全为预存 onnx 缺包
  （`test_dtype_contracts` / `test_source_contracts` 导入期、
  `test_review_contracts` 2 项用例内），与本次改动无关。
- 9 个 Python 入口全部 import 冒烟通过，签名一致性核对：
  `_decision_config(folder)`、`build_unified_model(model)`、
  `StartLuxTextDecoder(body, letters26)`、`load(device)`、
  `export_example(wrapper, device)`、`build_graph_inputs(rows, *, vocab_size,
  max_batch, max_seq_len)`、`OfficialPreprocessor(model_dir)`、
  `build_pool(folder, letters)`、`OfficialReference(source, folder)`。
- `bash -n run.sh` 通过；`run.sh --bogus` 与 `run.sh --device`（缺值）实测
  打印正确错误信息并以 1 退出（修复前为 `die: command not found` / 127）。
- 净变化：**32 个文件，+775 / −8,041 行**。模型目录 Python 源码（不含
  vendor）由 **6,715 行降至 2,616 行**；startlux 测试由 4,259 行降至
  1,101 行。删除的 6,715 − 2,616 中，整文件删除占 3,068 行
  （`monitor_idle_benchmark.py` 1,243、`ge_readout.py` 701、
  `unified_readout.py` 461、`reference.py` 339、`decision.py` 164、activator
  净减 156）。

### NPU 真实流程验证（2026-11-10，`run.sh --device 4`，chip 4 空闲）

清理后**已重跑真实 `run.sh`**，精度链路（阶段 1–9）全部通过；bench 阶段按
本次验证范围（导出与推理链路）在第 10 步人工确认门喂 `n` 主动中止，未执行。

| 阶段 | 结果 |
|---|---|
| 2 生成输入 | 5 cases / 7 rows，precision 池 5 / bench 池 5（重构后的 `build_pool(folder, letters)`、`io_spec()`、`_decision_config(folder)` 在真实材料上正常；已删的 `_validate_against_io` / `min_width` 无回归） |
| 3 CPU golden | 官方 FP32 参考 7 行写出 |
| 4 依赖构建 | proto 库 + WeightNzAndMatMulV3Pass 均安装成功 |
| 5 runtime | 构建成功 |
| **6 导出** | `model.air` **1,708,279,731 bytes**；`write_io` 经 ge_runtime 回读的嵌入 IO 与 `io_spec()` **逐项一致**（`input_ids [-1,-1]` dyn[0,1] / `position_ids [3,-1,-1]` dyn[1,2] / `actual_seq_lengths [-1]` dyn[0] / `pixel_values [512,1536]` 静态无 dynamic_dims / `image_feature_index [-1,-1]` dyn[0,1] → `scores [-1,26]` dyn[0]）；`parse_bindings` 取到真实节点名 `arg2_1/arg5_1/arg7_1/arg8_1/arg10_1`，双射绑定校验通过 |
| **7 推理** | 5 cases `status: success`；`outputs.bin` 728 bytes 与 `result.json` 索引**完全自洽**（B3 case `[3,26]`@0、四个 B1 case `[1,26]`@312/416/520/624，max(offset+bytes) == 文件字节数，无空洞无越界） |
| 8 数值比较 | **PASS**：5 cases，worst cosine **0.9999986085101928**（门限 ≥0.999）、worst relL2 **0.00449550810581621**（门限 ≤0.02） |
| 9 业务门禁 | **PASS**：5/5，worst active-probability max abs **0.00343094**（门限 ≤0.01），全部 top1 一致，参考 margin 0.158–0.885 |
| 10 空闲确认 | 喂 `n` → 修复后的 `die` 打印"已按'卡忙时不测试'中止"并以 1 退出（**A1 修复在真实流程中验证**） |

与既有基线对比（performance.md 融合后附录，2026-11-09：worst cosine 0.999998
/ relL2 0.003930、业务 worst 0.004551）：本次 cosine 同档、relL2 0.004496 与
业务 0.00343 均落在文档自述的"export-nondeterminism band"内（历史 5-case 记录
relL2 跨 0.002233–0.004753），且本次在不同卡（chip 4 vs chip 3/8）执行。

**图结构未变的旁证**：本次两次独立导出（02:23 与 02:35）字节数均为
1,708,279,731，与 performance.md 记录的融合后 1,708 MB 一致；清理未触碰任何
算子/数值路径。据此发现并修正 README 的一处失实：其"字节数恒定
1,907,534,423"是融合 RoPE / 融合 full-attention（提交 af9e05a，1,906 → 1,708
MB，−10%）之前的旧值。


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

**结论（2026-11-10 修正）**：本段原判定"P2-7 回退，eager RoPE 链保留"**已被
同日三轮 probe 推翻并废弃**——二轮 probe 以 aclnn eager 为参考，而 aclnn
eager 的 layout=1 按 BSH 解读、自身即错 0.88，其全部"图错"结论均系参考错误
（三轮以纯数学参考重测：GE 图 + BSND 对全 B/S 组合 relL2 0.0002）。实际交付
自始为"图腿融合 RoPE（BSND）+ eager 腿显式数学"。2026-11-10 模块化重写后按
用户停止协议（layout=1 的 eager/图语义分歧算错误）进一步统一为**单腿显式
rotate-half 数学**，融合算子整体弃用，见当日条目。

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

## 2026-11-09 变更：GDR conv Conv1D 化实验（净负，回退）

**实验**（"深入" #1 布局优化）：causal_conv4 由 Conv2D+padding=(0,3)+输出
切片改为 `F.pad(x32,(3,0)) + Conv1D(groups=C)`——目标消除输出切片
（StridedSliceV3 [1,6144,1,102] 358µs/e）与 conv 的 NCHW TransData
（377µs/e）。

**结果**：门禁 5/5+5/5 过，但 ticket-team 单 case AI 时间 **15,092 →
15,572 µs/e（+480µs，净负）**，bench 49.0 exec/s（−3%）。profiling 定位：
- torchair 将 aten.conv1d 转换为同一 **Conv2D** 节点（P2-7 调查时已知），
  unsqueeze→TransData 依旧（[1,6144,1,102] 376µs——pad 后宽度 102 使
  TransData 反而更大）；
- F.pad 引入新的 fp32 全宽拷贝（[1,6144,102]×18 层），Cast 组上升
  209µs；
- 输出切片确实消失（StridedSliceV3 1383→965µs），但被上述两项抵消
  反超。
**结论**：GE 的 conv1d/conv2d 共享实现 + F.pad 的拷贝代价使该方向不成立；
Conv2D+padding=(0,3)+切片维持现役（在 NZ Pass 基线上已是最优形态）。

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
