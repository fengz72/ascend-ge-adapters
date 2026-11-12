# StartLux-Decision v1.1 0.8B Instant 待办优化点（backlog）

日期：2026-11-12。状态：**候选方案集**。

本文收录**尚未落地**的优化点，每节一个，含修改方案、预期收益、执行顺序与
门禁。与 [optimization-summary.md](optimization-summary.md)（已采纳/已否决
的只读汇总索引）互补。

记录约定：

- 每节自带状态标签：**计划**（待 probe 裁决）/ **候选**（未立项）/ **已落地**
  / **已否决**。落地或否决后**保留原文供追溯**，只改状态标签，并按惯例回填
  [../performance.md](../performance.md) 英文文件名附录表、
  [../validation.md](../validation.md) 变更记录、optimization-summary.md
  §1（或 §2）与 §7 证据索引；不新增 CSV 汇总文件。
- 新增优化点追加为新节，编号顺延；每节修改方案须锚定当前交付代码的精确
  行号（行号漂移时先重锚定再动手）。

## 1. text 图 StridedSlice → SplitV（slice 切片换带宽型算子）

状态：**计划（待 probe 裁决）**。

### 1.1 是什么

text 图每执行 **90 个 StridedSlice 节点**（18 个 GDN 层 ×3 +
6 个 full-attention 层 ×6），全部由 `aten.slice` 切片表达式产生、lowering 为
StridedSliceV2/V3。该家族的成本是**固定的、与字节数无关**（实测 12–16
µs/节点），在官方池短序列形状下整图约 **1.0 ms/执行**的纯切片开销（按
per-op 现状加总 1,024 µs；旁证：conv Conv1D 化实验记录的 StridedSliceV3
总量 965–1,383 µs/执行，同量级）。换成 `torch.split_with_sizes`（lowering
为 ge.SplitV，带宽型、截距≈0）可把 90 节点压到 **36 个**。

动机链：P2-6 投影合并实验（optimization-summary §2）试图消除小 matmul
启动，却把合并后的大切片变成新大头而净负回退——分开的 Linear + 显式切片
是该图结构下的既成形态，本点只优化切片本身。

### 1.2 机理与证据

- `aten.split_with_sizes.default` **有** GE converter（torchair 随 torch_npu
  发行，`_ge_concrete_graph/ge_converter/aten/split_with_sizes.py:8-21`）：
  `ge.SplitV(self, split_sizes, dim, num_split=len(meta_outputs))`，
  **无等长限制**；而现注释（`model/modeling_text.py:200`）所指的
  `aten.split.sizes`（`torch.split` 传 list）与 `aten.chunk` 均无 converter
  ——**两个不同 ATen 算子**，必须写 `torch.split_with_sizes`。
- converter 把 dim 直接 dtype_promote 成 INT32（同文件 :17-20）；
  declare_supported 的用例全为正轴，负轴是否被 GE 归一化未验证 →
  **dim 一律写正数**。
- declare_supported 只登记测试用例、不做准入拦截
  （`fx2ge_converter.py:382-390`）；bf16 未登记（仅 F32/U8）不阻塞，
  由 probe 判据实测确认。
- declare 用例 `Support(F32(4,1,1664), [1408,128,128], 2)` 几乎就是 GDR
  三分站点的形状族（1664 = key_dim×2 + value_dim），先验很强。
- 成本证据（devcase B1/S1001，全 bf16）——StridedSlice 固定开销 vs
  带宽型算子：

| 输出大小 | StridedSliceV3 | 同字节 ConcatD |
|---|---|---|
| 0.26 MB | 13.55 µs | — |
| 1.03 MB | 14.38 µs | 5.04 µs |
| 4.10 MB | 11.66 / 15.80 µs | 9.05 µs |

- Split 家族按带宽走、截距≈0（两个独立点，同 fp32）：SplitVD 3.15 MB /
  4.06 µs（775 GB/s，12 block）；Unpack 4.72 MB / 6.16 µs（766 GB/s）。
  同图受控对照：StridedSliceD 786 KB（48 block）9.28 µs vs SplitVD
  1.57 MB（12 block）4.06 µs——**搬 2 倍字节花 44% 时间**。

### 1.3 修改点（4 站点，model/modeling_text.py）

**① GDR q/k/v 三分（:143-147）**，现状：

```python
mixed = causal_conv4(mixed_qkv, self.conv1d.weight)
q = mixed[..., :self.key_dim].reshape(batch, width, -1, self.head_k_dim)
k = mixed[..., self.key_dim:2 * self.key_dim].reshape(
    batch, width, -1, self.head_k_dim)
v = mixed[..., 2 * self.key_dim:].reshape(
    batch, width, -1, self.head_v_dim)
```

改为：

```python
mixed = causal_conv4(mixed_qkv, self.conv1d.weight)
q, k, v = torch.split_with_sizes(
    mixed, [self.key_dim, self.key_dim, self.value_dim], dim=2)
q = q.reshape(batch, width, -1, self.head_k_dim)
k = k.reshape(batch, width, -1, self.head_k_dim)
v = v.reshape(batch, width, -1, self.head_v_dim)
```

**② attn q/gate 拆分（:199-201）**，现状：

```python
qkv = self.q_proj(hidden).view(batch, width, -1, self.head_dim * 2)
# explicit slices: aten.chunk / aten.split.sizes have no GE converter
q, gate = qkv[..., :self.head_dim], qkv[..., self.head_dim:]
```

改为：

```python
qkv = self.q_proj(hidden).view(batch, width, -1, self.head_dim * 2)
# split_with_sizes -> ONE ge.SplitV; aten.chunk / aten.split.sizes have no
# GE converter, and aten.slice lowers to StridedSliceV2 whose cost is fixed
# (~12-16 us/node) rather than bandwidth-bound
q, gate = torch.split_with_sizes(qkv, [self.head_dim, self.head_dim], dim=3)
```

**③④ partial-RoPE 两腿（:209-214）**，现状：

```python
q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(
    q2d[..., :self.rotary_dim], k2d[..., :self.rotary_dim],
    cos.unsqueeze(2), sin.unsqueeze(2),
    layout="BSND", rotary_mode="half")
q2d = torch.cat((q_rot, q2d[..., self.rotary_dim:]), dim=-1)
k2d = torch.cat((k_rot, k2d[..., self.rotary_dim:]), dim=-1)
```

改为（`q_rot_src/k_rot_src` 命名避免与 rope 输出遮蔽）：

```python
q_rot_src, q_pass = torch.split_with_sizes(
    q2d, [self.rotary_dim, self.head_dim - self.rotary_dim], dim=3)
k_rot_src, k_pass = torch.split_with_sizes(
    k2d, [self.rotary_dim, self.head_dim - self.rotary_dim], dim=3)
q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(
    q_rot_src, k_rot_src, cos.unsqueeze(2), sin.unsqueeze(2),
    layout="BSND", rotary_mode="half")
q2d = torch.cat((q_rot, q_pass), dim=-1)
k2d = torch.cat((k_rot, k_pass), dim=-1)
```

要点：

- dim 一律写正数（dim=2/3），不写 -1。
- `q/k/v` 的后续 reshape 保留：SplitV 输出是新分配的连续张量，reshape
  为免费 view，保留以缩小 diff。
- cat 结构（:213-214）不动，其 ConcatD 成本两路径相同、不属本点收益。
- FIA 的连续性约束不受影响（其输入是 cat 输出，本方案不触碰）。
- eager 腿 `split_with_sizes` 返回 view（末维切分非连续），下游 reshape
  会物化连续副本；且现状 :210 本就向 rope 传非连续 slice 视图——同类
  输入有先例。

### 1.4 预期收益

成本模型：traffic/770 GB/s + ~3.5 µs 家族下限。**probe 实测前均为估计。**

官方池形状（B1/S99，固定开销区，收益最大）：

| 站点 | 现状/层 | SplitV/层 | 省/层 | ×层数 |
|---|---|---|---|---|
| GDR 三分 | 3×10.58 = 31.7 µs | ~3.5 µs | 28.2 | ×18 = 508 µs |
| attn q/gate | 2×14.0 = 28.0 µs | ~3.5 µs | 24.5 | ×6 = 147 µs |
| attn q rope 腿 | 11.55+13.73 = 25.3 µs | ~3.5 µs | 21.8 | ×6 = 131 µs |
| attn k rope 腿 | 10.80+11.52 = 22.3 µs | ~3.5 µs | 18.8 | ×6 = 113 µs |
| **合计** | | | | **≈ −899 µs/exec** |

对当前同口径基线 9,156 µs/执行（causal-conv1d 附录，performance.md:849，
不含 MatMulV3）≈ **−9.8%**；对 conv1d 前口径 10,445 µs 为 −8.6%；全口径
（含 MatMulV3 恢复，~11.8 ms）约 −7.6%。占比：GDR 56% / attention 44%。

长文本形状（B1/S1001，带宽区，收益缩水；单 case AI 21,900 µs 实测）：

| 站点 | 现状/层 | SplitV/层 | ×层数 |
|---|---|---|---|
| GDR 三分（24.6 MB traffic） | 35.0 µs | 31.9 µs | ×18 = −56 µs |
| attn q/gate（16.4 MB） | 31.6 µs | 21.3 µs | ×6 = −62 µs |
| attn q rope（8.2 MB） | 30.0 µs | 10.6 µs | ×6 = −116 µs |
| attn k rope（2.05 MB） | 29.1 µs | 2.7 µs | ×6 = −158 µs |
| **合计** | | | **≈ −392 µs/exec（−1.8%）** |

占比：**attention 86%** / GDR 14%——GDR 不达标时 attention-only 变体在
长文本下保留 86% 收益，独立成立。

- 节点数：90 → 36 SplitV（**−54/exec**），StridedSliceV3 在 text 图归零
  （devcase 实测 90 个/exec 全部来自这 4 个站点，无第 7 个 signature）。
- 不确定带：S=99 **−5% ~ −9%**；S=1001 **−0.9% ~ −2.6%**——取决于
  SplitV 在 bf16 大 shape 下的实际带宽与 block 分配（观测到的 SplitVD
  只用 12 block，StridedSlice 用 48）。
- **e2e 不承诺**。两个相反先例：#13（NZ Pass）单 case AI −14.4% 只换来
  闭环 +0.3%（host-bound）；causal-conv1d（AI −12.3% 且 −241 节点/请求）
  换来 e2e P50 −13.1%。本次 −54 节点 + AI −8.6%（S99 口径），按 conv
  那次的作用机制（节点数下降同时减轻 GE 任务调度）粗估 **e2e P50
  0 ~ −4%**，必须实测，落在 2.8% 方差带内（optimization-summary §5.2）
  就按方差带内记录。

### 1.5 执行顺序与门禁

**第 1 步 probe（go/no-go 硬门禁，不改交付代码）**：单层迷你模块复刻真实
producer 链（causal_conv4 输出直连 GDR 三分站点、RMSNorm 后接 RoPE
站点），真实 shape/dtype（bf16），两个形状各跑一遍：S=99（官方池）与
S=1001（dev case）——只测 S=99 会全是固定开销、只测 S=1001 会低估官方
池收益。torchair 导出 + 编译 + GE profiling（`tools/parse_profiling.py`
读 op_summary，记录 per-op µs 与 Block Num）。环境引导：`scripts/env_python.sh`
+ LD_PRELOAD ChunkGatedDeltaRule 原型（同 export 类流程）。

判据（全部满足才继续）：

1. 图中出现 SplitV/SplitVD 节点，且 StridedSliceV2/V3 归零（若 dynamo 把
   split_with_sizes 分解回 slice，直接否决，零成本退出）；
2. 输出与现路径**逐位一致**（纯数据搬移）；contiguity 断言对象 =
   **reshape 后喂 kernel 的张量**（q/k/v、q2d/k2d、rope 输入）——eager
   腿 raw split view 末维切分必然非连续，不作断言对象；图中不得出现新增
   TransData；追加**整模 eager A/B 逐位断言**（两腿共享 modeling 代码，
   一次覆盖）；
3. attn 三站点合计：S=99 < 40 µs/层（基线 75.6）、S=1001 < 45 µs/层
   （基线 90.7）；
4. GDR 站点：S=99 < 25 µs/层（基线 31.7）或 S=1001 < 30 µs/层（基线
   35.0）；不达标则 **GDR 不改、只改 attention**（占 S=99 收益 43.5% /
   S=1001 86%，独立成立）。

**第 2 步 整模导出核对**：SplitV 计数 36/exec、StridedSliceV3 在 text 图
归零、Data/Const 绑定仍双射（本方案不动任何参数，不触发 §2 记录过的
positional bake 冲突）、记录 AIR 字节数（当前 1,507,412,645）。

**第 3 步 精度门禁**：`decider/verify_*.py` 5/5 + 业务门禁；要求复现当前
交付的 worst relL2 0.005704 与业务 worst 0.004267（performance.md:852-853；
逐位一致则必然复现）。

**第 4 步 性能门禁**：两个形状各一次 msprof 单 case 采集（case_filter +
profiling 请求选项已落地），与 21,900 µs/exec（S=1001）及官方池基线同口径
对比 + op_summary 差分；**用 `tools/restore_matmulv3.py` 修复 CubeCore 行**，
否则 AI 总量会漏掉全部 MatMulV3；再跑官方 5 用例 bench（instances 1、
warmup 32、n=2048、seed 42）取 exec/s 与 e2e avg/P50/P99。

**第 5 步 文档回填**：performance.md 英文文件名附录表（两形状各一行，含
probe 的 per-op 与 Block Num）、validation.md 变更记录、
optimization-summary.md §1 新增一行 + §7 证据索引；不新增 CSV 汇总文件。

### 1.6 风险与处置

| 风险 | 处置 |
|---|---|
| dynamo 把 split_with_sizes 分解回 slice | probe 判据 1 直接否决，零成本退出 |
| GE SplitV 不接受负轴 | 已规避：全部写正数 dim=2/3 |
| bf16 未在 declare_supported 登记（只有 F32/U8） | 装饰器不做拦截（fx2ge_converter.py:382-390 已核实）；probe 判据 2/3 实测确认 |
| SplitV 大 shape 只给 12 block、带宽上不去 | probe 在 S=1001 实测 Block Num 与 per-op；GDR 段按判据 4 单独裁决 |
| SplitV 后被插 TransData | probe 判据 2 否决——P2-6 先例：非连续下传使 TransData 翻倍是净负主因 |
| eager 腿视图非连续的下游接受性 | 现状 :210 本就传非连续 slice 视图给 rope（同类先例）+ 整模 A/B 逐位断言覆盖 |
| e2e 落在方差带内 | 按 optimization-summary §5.4 口径以单 case AI 时间为 kernel 级真相记录，e2e 如实标注「方差带内」 |

### 1.7 关联

#11 融合 RoPE（③④ 站点所在）、#12 FIA、P2-6 投影合并否决（切片存在的
历史根因）、#16 零态共享（probe-first + 位级门禁方法先例）。

## 2. 视觉塔 rotate_half 的 StridedSlice（候选，未立项）

`model/modeling_vision.py:39` 从 transformers 导入
`apply_rotary_pos_emb_vision`，:96 调用；其内部 rotate_half 走 chunk →
每 block 4 个 StridedSlice（256,12,64→256,12,32，混合池共 1,440 次 /
20.4 ms）。与 §1 同型（固定开销区、op 数为 §1 的 7 倍），但需本地
vendored 一份该函数（不能改上游），且只在图像请求上生效。**建议 §1
probe 通过后再单独评估**，方法与门禁直接复用 §1.5。
