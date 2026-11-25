# StartLux-Decision v1.1 0.8B Instant Text 优化执行计划

- 日期：2026-11-24
- 状态：**已执行**（P0 的 E1/E2 与 P1-1/P1-2/P1-3 已跑，结果回填 §8；P1-1/P1-2 采纳，E1/E2/P1-3 否决/撤回；P1-4 MLP-only 已测，P1-5 全模块已测待采纳）
- 目标：把 `-text` 交付态从 **21.27 ms（P50 21.36）** 压到 **~19.5–20.0 ms**，追平参考图 A 的 19.65 ms（**已达成并超过**：P1-5 实测 **19.178 ms**，低于目标区间且 **0.472 ms 优于 A**；待采纳）
- 范围：`models/startlux-decision-v1.1-0.8b-instant-text`
- 依据：2026-11-24 的两份 profiling 对比（A 在主机 99，B 在本机 100）

---

## 0. 怎么用这份文档

1. 先读 §1–§4 理解差距来源，不要跳过 §3 的"口径陷阱"。
2. 严格按 §5 的优先级执行，每项**单独改、单独 A/B**，不要合并提交。
3. 每项都用 §6 的流程采集/验证，用 §7 的门禁判定。
4. 结果回填 §8 的执行记录表，并按 §9 更新 README 与英文附录。

---

## 1. 环境与路径

### 1.1 主机 / 容器

| 角色 | 主机 | 容器 | 适配目录 |
| --- | --- | --- | --- |
| B（被测，本机） | `ssh 100` | `hejun-cann910-jev` | `/export/home/ext.hejun23/workspace/ascend-ge-adapters/models/startlux-decision-v1.1-0.8b-instant-text` |
| A（参考基线） | `ssh 99` | `hejun-cann910-jev` | `/export/home/meisanyou1/fzy/jev/prof/slx08b/r1_base_graph` |

进入容器：
```bash
ssh 100
docker exec -it hejun-cann910-jev bash
```

### 1.2 关键路径

- 被测模型代码：`model/modeling_text.py`（331 行；下文行号以此版本为准）
- 导出：`model/export_air.py`
- 依赖/融合 Pass：`scripts/prepare.sh`
- 采集/分析工具：`../../tools/parse_profiling.py`、`../../tools/restore_matmulv3.py`
- 当前 AIR：`work/build/text.air`（约 1.5 GB，2026-11-24 07:47）
- 本次 B profiling：
  `work/prof-text/profiling/PROF_000001_20261124113155878_00407364CMQPNPIF/mindstudio_profiler_output/`
  - `op_summary_20261124113342_restored.csv`（分析用这个）
  - `op_statistic_20261124113342.csv`
  - `step_trace_20261124113342.csv`
- A profiling（主机 99，容器内）：
  `/export/home/meisanyou1/fzy/jev/prof/slx08b/r1_base_graph/A03-R40-I191-99-4100006.JD.LOCAL_2464330_20261117194459042_ascend_pt/PROF_000001_20261117194459074_02464330BROLECDJ/mindstudio_profiler_output/`
  - `op_summary_20261117194501.csv`、`op_statistic_20261117194501.csv`、`step_trace_20261117194501.csv`

> A 在另一台主机/另一个 CANN 绘图流程（`r1_base_graph`）上采集，**跨机绝对时间只能作方向参考**；所有"是否采纳"的判定必须在同一张卡、同一窗口内做 A/B。

---

## 2. 当前基线（交付态 B）

来源：`README.md` §3.2 / §3.4（2026-11-24，逻辑芯片 4，CANN 9.1.0，Ascend910_9382）。

| 指标 | 值 |
| --- | --- |
| 稳态 bench（warmup 32 + 1000 req，seed 42） | e2e avg **21.27 ms** / p50 21.26 / p99 21.39，QPS 47.0，0 错误 |
| 逐 case 稳态（17 个） | min 21.245 / P50 21.358 / avg 21.436 / max 21.975 ms |
| 图冷启动 | CompileGraph 12481.2 ms，LoadGraph 2496.8 ms |
| 精度 | cosine worst 0.999993（≥0.999）；relative L2 worst 0.004937（≤0.02） |
| 说明 | 首个 case 500.819 ms 含冷开销，不计入稳态 |

同期 conv 路线 A/B（提交 `246f1f3` 正文）：Conv2D 基线 25.05 ms、`causal_conv1d_qkv` 融合 22.51 ms、**`causal_conv1d` 交付 21.27 ms**。

---

## 3. 对比口径（先看这里，避免误判）

1. **step 时间**：B 的 `step_trace` 里 24 ms 的步是 **msprof 采集抖动**（README bench max 21.975 可证），不代表模型。真实对比用"干净步"：A 19.65 ms vs B 21.36 ms（P50）。B 全 18 步均值 21.97 ms 不要拿来当基线。
2. **MatMulV3 行是重建的**：msprof 这批构建会丢 CubeCore/MIX 行，原始 `op_summary_20261124113342.csv` 里**没有 MatMulV3**（11620 行、640 op/iter）；`_restored.csv` 里才有 186 个 MatMulV3（14868 行、826 op/iter）。分析前必须先跑 `tools/restore_matmulv3.py`；MatMulV3 的 duration 是近似的，**结论要看 shape/数量/趋势，不要死抠单算子微秒**。
3. **op 数量口径一致**：A、B 都是每 iter **18 个 `ChunkGatedDeltaRule`**、67 个 `RmsNorm`、48 个 `InplaceAddRmsNorm`、6 个 attention，hidden 均 `1001×1024` → 同一模型、同一负载（S=1001）。A 的 `op_statistic` 计数被放大了 4/3（按 4 iter 聚合），**不要用 op_statistic 的 Count 做逐 iter 对比，用 op_summary 的时间窗**。
4. **A 无 MatMulV3、无大批 Cast**：A 的线性层是 `AI_CORE` 的 ND `MatMul`，B 是 `MIX_AIC` 的 NZ `MatMulV3`。这是两条不同的图实现，不是同一份代码的两个开关——但差距高度集中在这一点上（见 §4）。

---

## 4. 对比结果与根因

### 4.1 总量分解（每 step）

| | A | B（干净步） | 差 |
| --- | ---: | ---: | ---: |
| step 总时长 | 19.65 ms | 21.36 ms | **+1.71 ms** |
| 算子 duration 合计 | 19.57 ms | 21.00 ms | **+1.42 ms** |
| 调度空泡（busy 利用率） | 0.10 ms（99.5%） | 0.42 ms（98.0%） | +0.32 ms |

逐 step 时间：A = 19.665/19.642/19.679（3 步，极稳）；B 干净步 21.26–21.40，另有若干 23.8–24.2 的抖动步（采集噪声）。

### 4.2 逐算子差（每 iter，us；窗口见 §6.3）

| `OP Type` | A n / us | B n / us | Δ(B−A) |
| --- | ---: | ---: | ---: |
| `MatMulV3` | 0 / 0 | 186 / 6021 | **+6021** |
| `MatMul` | 163 / 4850 | 1 / 3 | −4847 |
| `Cast` | 97 / 459 | 232 / 1519 | **+1060** |
| `StridedSliceV3` | 0 / 0 | 90 / 1183 | +1183 |
| `StridedSliceD` | 49 / 741 | 0 / 0 | −741 |
| `Mul` | 37 / 348 | 49 / 726 | +379 |
| `SlxQkvConv` | 18 / 1696 | 0 / 0 | −1696 |
| `CausalConv1d` | 0 / 0 | 18 / 1227 | +1227 |
| `SwiGlu` | 24 / 474 | 0 / 0 | −474 |
| `Swish` | 0 / 0 | 42 / 568 | +568 |
| `FlashAttentionScore` | 6 / 506 | 0 / 0 | −506 |
| `FusedInferAttentionScore` | 0 / 0 | 6 / 585 | +585 |
| `InplaceAddRmsNorm` | 48 / 1279 | 48 / 1044 | −234 |
| `AutomaticBufferFusionOp` | 58 / 593 | 18 / 209 | −384 |
| `RmsNorm` | 67 / 1547 | 67 / 1466 | −81 |
| `ChunkGatedDeltaRule` | 18 / 5986 | 18 / 6002 | +16（一致） |
| A 独有（SplitVD/Transpose/Add/Neg/ConcatV2D…） | ~970 | ~110 | −860 |
| **合计** | **19574** | **20997** | **+1422** |

结论：`ChunkGatedDeltaRule` 完全一致（同 kernel 同 shape），差距不在 GDN 主体；核心是 **MatMul 形态（+1.17 ms）** 和 **Cast 爆炸（+1.06 ms）**，`StridedSlice`/`Mul` 再添 ~0.8 ms；B 的 conv 反而更快（−0.47 ms）。

### 4.3 Cast 结构（最关键证据）

B 的 232 个 `Cast` 里，有 **48 对是整块 hidden 的无效往返**：

```
RmsNorm/InplaceAddRmsNorm (bf16 输出)
    -> Cast  BF16→FLOAT  shape "1,1001,1024"   ×48
    -> Cast  FLOAT→BF16  shape "1001,1024"     ×48
    -> MatMulV3 (bf16) ...
```

每个 RMSNorm 之后都紧跟这对"bf16→fp32→bf16"，然后才进 `MatMulV3`。A 在同样位置**没有**这对 cast（A 的 `Cast` 只有 gated norm 的 `16016,128` 18 对、g 的 `1,1001,16` 18 个等，和 B 相同）。48 = 24 层 × 2 个 norm（input_layernorm + post_attention_layernorm），所以是每个 norm 输出后的固定产物。

此外 B 还有 `1001,3584` 的 24 对、`1001,2048` 的 6 对等，都落在线性层附近。

### 4.4 MatMul 形态

- A：`MatMul`，`AI_CORE`，ND 权重（如 `in=1001,1024;7168,1024 out=1001,7168`），163 个/iter，4.85 ms。
- B：`MatMulV3`，`MIX_AIC`，NZ 权重（如 `in=1001,1024;224,64,16,16 out=1001,3584`），186 个/iter，6.02 ms（重建值）。
- B 比 A 多 23 个 matmul：MLP 的 gate/up 在 A 是**单个 7168 投影**（一个 `MatMul` + 一个 `SwiGlu`），在 B 是**两个 3584**（48 个 matmul + `Swish` + `Mul`）。
- B 的小 N=16 matmul（`in_proj_a/b`，36 个）在 `MatMulV3` 上 10.4 us/个，是 A 同形的 2.3 倍（4.5 us/个）。

### 4.5 根因假设（按可信度）

1. **`WeightNzAndMatMulV3Pass` 是 +1.17 ms 与 96 个无效 cast 的共同来源**（高可信）。
   该 Pass 在 `kAfterOriginGraphOptimize` 把 ND `MatMul` 常量化成 NZ `MatMulV3`，自定义实现，恰好也只在这条图上有；关掉它应回到 A 的 `MatMul` 形态。**必须先做 §5.1 的 E1 验证。**
2. **MLP gate/up 未融合**（高可信）：`StartLuxMLP.forward` 是 eager 两投影 + SiLU + Mul；A 用单投影 + `SwiGlu` 融合算子。
3. **切片未走带宽型算子**（高可信，且 backlog 已有方案）：B 的 90 个 `StridedSliceV3` 全部来自 4 个站点（q/k/v 三分、q/gate、rope 两腿），固定 12–16 us/节点。
4. attention `FIA` 在 S=1001 略慢于 A 的 `FA`（+80 us，低优先）。

---

## 5. 优化方案（按 ROI / 风险排序）

> 约定：每项都写清楚"改哪里 / 怎么做 / 预期 / 风险 / 门禁"。除 P0 外都需要改代码。

### P0. 不改模型的两个 A/B 实验（先做，决定后续方向）

#### E1（最高优先）关闭 `WeightNzAndMatMulV3Pass` 复测

- 该 Pass 安装位置：`$ASCEND_HOME_PATH/opp/vendors/custom_nz_pass/custom_fusion_passes/libnz_weight_pass.so`
  （本机实际：`/usr/local/Ascend/cann-9.1.0/opp/vendors/custom_nz_pass/custom_fusion_passes/`）
- 关闭：
  ```bash
  mv "$ASCEND_HOME_PATH/opp/vendors/custom_nz_pass/custom_fusion_passes/libnz_weight_pass.so" \
     "$ASCEND_HOME_PATH/opp/vendors/custom_nz_pass/custom_fusion_passes/libnz_weight_pass.so.off"
  ```
- 重新 export + CompileGraph + infer/compare + bench + profiling（见 §6）。
- 恢复：把 `.off` 改回，或直接 `bash scripts/prepare.sh` 重新安装（`prepare.sh:38-41`）。
- **假设**：关闭后 `MatMulV3` 消失、`Cast` 从 232 降到 ~100、`MatMul`(AI_CORE,ND) 出现，延迟 −0.8 ~ −1.5 ms。
- **实测（2026-11-25，chip 11 同窗 A/B）**：方向相反——关 pass 反而 **+0.611 ms（+3.02%）**，Pass 有益，详见 §8。
- **判据**：
  - 图中 `MatMulV3` → 0，`Cast` 明显下降（重点看那 48 对 `1,1001,1024`/`1001,1024` 是否消失）；
  - 精度门禁过（§7）；
  - 若反而退化为 `MatMulV2` 且更慢 → **立即回退**，只记录"NZ Pass 对 -text 有益"的结论，转 P1。
- 风险：Pass 是系统级安装（`$ASCEND_HOME_PATH`），注意改完恢复，别影响同机其他任务。

#### E2 attention `FIA` ↔ `FA`

- 位置：`model/modeling_text.py:211-227`（当前 `npu_fused_infer_attention_score`）。
- 做法：改为 `npu_fusion_attention`（与 A 的 `FlashAttentionScore` 对齐），按 BSND/BNSD 传参；S=1001 是 FA 的优势区。
- 预期 −80 us（B 0.585 ms → ~0.505 ms）。
- 风险：布局/掩码语义；必须过精度门禁。历史 backlog 里 S=99 的结论是 FIA 慢 4×，长文本未复测，这次一并补上。

### P1. 代码级优化（改 `model/modeling_text.py`）

#### P1-1 MLP gate/up 合并 + `npu_swiglu`（`StartLuxMLP`，`:62-77`）

- 现状 `:76-77`：
  ```python
  return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
  ```
  两条 `1024→3584` matmul + fp32 `Swish` + `Mul` + 配套 cast。
- 做法：
  1. 加载权重后把 `gate_proj.weight`、`up_proj.weight` 在输出维 concat 成 `gate_up_proj`（用 `register_load_state_dict_post_hook` 或 `install_*` 方法，保持在 checkpoint 名不被破坏）。
  2. forward 改 `self.down_proj(torch.ops.npu.npu_swiglu(self.gate_up_proj(x)))`。
  - `torch.ops.npu.npu_swiglu` 已在环境中确认存在（`dir(torch.ops.npu)`）。
- 预期：matmul 少 24 个（48→24）+ 省 fp32 Swish/Mul/cast，**−0.4 ~ −0.5 ms**。
- 风险：`npu_swiglu` 的 gate/up 顺序与 dtype 语义；**先做单算子 probe** 再入图。
- 与已否决的 P2-6 的区别：P2-6 是"合并后还要 StridedSlice 再切"导致净负；这里 `npu_swiglu` 内部直接消费拼接、不产生额外切片，所以可以试。

#### P1-2 `StridedSlice` → `SplitV`（backlog `docs/optimization-backlog.md` §1 已写）

- 站点：
  - `:133-137` GDR q/k/v 三分
  - `:196` attention q/gate 拆分
  - `:202-203` partial-RoPE q/k 两腿
- 做法：用 `torch.split_with_sizes(..., dim=正数)`（lowering 为 `ge.SplitV`；**不要**用 `torch.split`/`chunk`，它们没有 converter）。
- 预期：90 → 36 节点，S=1001 下 **−0.3 ~ −0.4 ms**（backlog §1.4 长文本估算 −392 us）。
- 风险与门禁：见 backlog §1.5/§1.6（先单层 probe，判据：SplitV 出现且 StridedSlice 归零、逐位一致、无新增 TransData）。

#### P1-3 清理残留 norm cast（仅当 E1 后仍有）

- `StartLuxRMSNorm.forward:38-40` 的 `return y.to(x.dtype)`；
- `StartLuxRMSNormGated.forward:57-59` 的 `gate.float()`。
- 注意：gated norm 的 18 对 `16016,128` cast **A 也有**，不是回归项，优先级最低；只有 E1 没消掉那 48 对 `1,1001,1024` 时才需要定位（保留 `dynamo.pbtxt` 再删，见 `export_air.py:192` 目前会删掉它）。
- 预期 0.1 ~ 0.6 ms。

#### P1-4 MLP 显式 2D（`squeeze(0)`/`unsqueeze(0)`，在 P1-1+P1-2 之上）

- 根因已定位：torchair 对 3D 输入的 `nn.Linear` 发 `BatchMatMul`，GE 改写为 2D MatMul，并用 `trans_Cast` bf16→fp32→bf16 完成 3D→2D flatten（按节点名前缀拆分：combo 图 trans_Cast 174 节点 / 1386.0 µs = AI 时间 6.86%，model-code Cast 58 / 358.5 µs = 1.78%）。
- 做法：`StartLuxMLP.forward` 改 `h = torch.ops.npu.npu_swiglu(self.gate_up_proj(x.squeeze(0))); return self.down_proj(h).unsqueeze(0)`。B 固定 STATIC 1（`io_spec()` 的 `mark_static`），`squeeze/unsqueeze` 无动态 shape 运算。
- 实测（chip 13，2026-11-25，两轮 A1(combo)→B(squeeze)→A2(combo)）：combo mean 20.168 → squeeze mean 19.609 ms（pooled −0.5587 ms / −2.77%），分布不重叠；trans_Cast 174→78 节点（−96 / −608.9 µs，−43.9%），总 AI 20193.1→19536.2 µs（−656.9 µs）；精度 bit-neutral 0.999995 / 0.004146。
- **状态：MLP-only 增量阶段（非终态）**；全模块扩展见 P1-5。
- 冷编译首次 234,074.5 ms（234 s），warm 11,097–11,265 ms（combo 11,356–12,764 ms）；`kernel_meta/` 不随仓库交付，新部署付一次。
- 剩余 78 trans_Cast / 777.1 µs（3.98%）在 attention/GDN 投影，由 P1-5 一并消除。
- 更正：首试「CompileGraph 挂起 >8 min」为误诊——冷编译需 234 s，8 min 被杀过早；`io_spec()` 固定 B=1 STATIC、`b*s` 折叠为 `s`，无动态乘积问题。

#### P1-5 显式 2D 扩展到 GDN + attention（在 P1-1+P1-2+P1-4 之上）

- 机制：GE 每个 norm 输出只插一对 `trans_Cast`，同一张量喂的其余投影复用它 flatten 后的结果；因此**同一模块内所有投影必须一起转 2D**，只转一个只是搬移开销——MLP-only 174→78，全模块 174→**0**。
- 做法（`model/modeling_text.py`，相对 pristine 69 insertions / 26 deletions，patch `work/p16-full2d.patch`）：
  - MLP 同 P1-4。
  - GDN：`h2 = hidden.squeeze(0)` 喂 `in_proj_qkv/z/b/a`；`in_proj_qkv(h2).unsqueeze(0)` 立即恢复 3D，保持 vendored `causal_conv1d` 与 `dim=2` 的 `split_with_sizes` 不变；`z` reshape 到 `(width,-1,head_v_dim)`；`b/a` 变 2D `[S,16]` 后删掉 `beta,g = beta.flatten(0,1), g.flatten(0,1)`；输出 `norm(out,z).reshape(width,-1)` 再 `out_proj(out).unsqueeze(0)`。
  - attention：`h2 = hidden.squeeze(0)` 喂 `q/k/v_proj`；`q_gate` view `(width,-1,head_dim*2)`，q/gate 的 `split_with_sizes` 从 dim=3 移到 dim=2；`gate` 保持 2D `[S,2048]`；`q_heads/k_heads/v` 各自 `.unsqueeze(0)` 恢复 4D（RoPE 与 FIA 需 4D BSND）；输出 `out.reshape(width,-1)` 再 `o_proj(out * torch.sigmoid(gate)).unsqueeze(0)`（gate 必须 2D）。
- 实测（chip 4，2026-11-25 07:53–07:57，A1(combo)→B(full2d)→A2(combo)）：combo mean 20.139 → full2d **19.178 ms**（−0.961 ms / −4.77%）；qps 49.627→52.116（+5.02%）；累计 vs 原基线 21.3055 **−2.127 ms / −9.99%**，0.472 ms 优于 A 19.65。结构：ops 724→568、AI 20193.1→19132.1 µs（−1061.0 µs / −5.25%）、trans_Cast 174→**0**、all Cast 232→76（1744.5→575.4 µs）、MatMulV3 162 不变；精度 bit-neutral 0.999995 / 0.004146。
- model-code Cast 58→76（+18）非回归：多出的 18 个 `FLOAT→BF16 "16016,128"`（166.4 µs）是 gated norm `.to(x.dtype)`，combo 里被折进 `trans_Cast` 的 downcast；适配器消失后独立成节点。
- 次要收益：HBM rise 2174 MB（combo ~4050），warm compile 9349.3 ms（combo 11221.6/12038.9）。冷编译首次 504,141.3 ms（504 s；`kernel_meta/` 144 MB/29 → 254 MB/34，证明在真编译），`kernel_meta/` 不交付，新部署付一次。
- **状态：已测待采纳**（源码已含改动，但 `work/build/text.air` 仍为 combo 图，sha256 first 200 MB `aa48270bda9c`）。

### P2. 可选 / 暂不做

- 小 matmul 合并：`in_proj_a`（`1024→16`）和 `in_proj_b`（`1024→16`）合成一个 `1024→32`，再切两半。可省 ~0.1–0.2 ms，但引入切片，先看 E1 结果。
- **不做**（已有否决证据，别重复踩）：
  - GDR/attention 投影合并（P2-6，切片/TransData 净负）；
  - Mega GDN 融合（P0/P1 历史否决）；
  - `g` 的 `a.float()` 改 bf16（A 同样 fp32，且影响数值）。

---

## 6. 执行流程

### 6.1 环境

```bash
cd /export/home/ext.hejun23/workspace/ascend-ge-adapters/models/startlux-decision-v1.1-0.8b-instant-text
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export STARTLUX_MODEL_DIR=/path/to/hf-model     # 现有会话沿用其取值
export ASCEND_RT_VISIBLE_DEVICES=4              # 逻辑芯片 4（与基线一致）
```

### 6.2 标准一轮（export → infer → compare → bench）

一键（推荐，避免漏 `--precision-mode` 和 `--device`）：
```bash
bash run.sh --device 4 --iterations 1000 --warmup 32 --seed 42
```

或分步：
```bash
bash scripts/prepare.sh
bash ../../runtime/build.sh
LD_PRELOAD=work/deps/chunk_gated_delta_rule_proto/install/lib/libchunk_gated_delta_rule_compat_proto.so \
  python3 model/export_air.py --device 0 --export-dir work/build --export-name text
ASCEND_CUSTOM_OPP_PATH="$PWD/work/opp" \
  ../../runtime/build/ge_execute --graph work/build/text.air --cases work/inputs/cases.json \
  --output-dir work/output --device 0 --precision-mode must_keep_origin_dtype
python3 ../../tools/compare.py --golden work/golden --output work/output \
  --cosine-min 0.999 --relative-l2-max 0.02
ASCEND_CUSTOM_OPP_PATH="$PWD/work/opp" \
  ../../runtime/build/ge_execute --graph work/build/text.air --cases work/inputs/cases.json \
  --output-dir work/bench --device 0 --precision-mode must_keep_origin_dtype \
  --iterations 1000 --warmup 32 --seed 42
```

### 6.3 profiling 采集

原 B 采集口径（见 `work/prof-text/profiling/.../host/sample.json`）：`acl/runtime on`、`ai_core task-based`、`task-time on`、`export on`。可用模板（每轮换 `--tag` 和 `--output`）：

```bash
TAG=prof-text-e1        # 本次实验名
rm -rf work/$TAG
ASCEND_RT_VISIBLE_DEVICES=4 \
ASCEND_CUSTOM_OPP_PATH="$PWD/work/opp" \
msprof --output=work/$TAG/profiling \
  --application="env ASCEND_RT_VISIBLE_DEVICES=4 ASCEND_CUSTOM_OPP_PATH=$PWD/work/opp \
    ../../runtime/build/ge_execute --graph work/build/text.air --cases work/inputs/cases.json \
    --output-dir work/$TAG --device 0 --precision-mode must_keep_origin_dtype" \
  --ai-core=on --aicpu=on --runtime=on --ascendcl=on --task-time=on --export=on
PROF=$(ls -d work/$TAG/profiling/PROF_* | tail -1)
python3 ../../tools/restore_matmulv3.py --profiling_dir "$PROF"   # 必须，修复 MatMulV3 行
```

> `msprof` 参数以本机 `msprof --help` 为准；关键是 `restore_matmulv3.py` 一定要在分析前跑。

> **工具修正（2026-11-25）**：上面的 `msprof --application=...` 模板不可用——application 模式已废弃，且会把 `env VAR=val ...` 前缀里的裸 `env` 当路径解析（rc=255），`--export` 在 application 模式下也被忽略。实测路径：`ge_execute --profiling`（GEInitializeV2 选项，写入 `<output-dir>/profiling`）→ `msprof --export=on --output="<PROF_dir>"` → `tools/restore_matmulv3.py`。`msprof` 要求 `--output` 目录已存在（否则 ~1 s 内报 "Analysis output path ... does not exist" 且不启动）；`ge_execute` 按文件后缀判图格式，AIR 备份必须保留 `.air` 后缀。

### 6.4 复现 §4 的逐算子对比（可直接跑）

```python
# python3 op_delta.py <A_op_summary.csv> <B_op_summary_restored.csv>
import csv, sys
from collections import defaultdict
def load(p): return list(csv.DictReader(open(p)))
def f(x):
    try: return float(x)
    except: return 0.0
def agg(rows, t, end):
    d = defaultdict(lambda: [0, 0.0])
    for r in rows:
        s = f(r["Task Start Time(us)"])
        if end - t <= s < end:
            d[r["OP Type"]][0] += 1
            d[r["OP Type"]][1] += f(r["Task Duration(us)"])
    return d
# A iter0：t=19665.552, end=1794915899182009.512
# B iter0：t=21405.047, end=1795519944443441.438  （均取自 step_trace）
A = agg(load(sys.argv[1]), 19665.552, 1794915899182009.512)
B = agg(load(sys.argv[2]), 21405.047, 1795519944443441.438)
for k in sorted(set(A) | set(B), key=lambda k: -(max(A.get(k,[0,0])[1], B.get(k,[0,0])[1]))):
    an, av = A.get(k, [0, 0.0]); bn, bv = B.get(k, [0, 0.0])
    print(f"{k:26s} A {an:4d}/{av:8.1f}  B {bn:4d}/{bv:8.1f}  d {bv-av:+8.1f}")
```

MatMul shape 归因（定位 gate/up、小 N=16）：
```python
import csv
from collections import defaultdict
def f(x):
    try: return float(x)
    except: return 0.0
def mm(path, t, end, names):
    d = defaultdict(lambda: [0, 0.0])
    for r in csv.DictReader(open(path)):
        s = f(r["Task Start Time(us)"])
        if end - t <= s < end and r["OP Type"] in names:
            d[(r["Input Shapes"], r["Output Shapes"])][0] += 1
            d[(r["Input Shapes"], r["Output Shapes"])][1] += f(r["Task Duration(us)"])
    for k, v in sorted(d.items(), key=lambda x: -x[1][1])[:10]:
        print(f"n={v[0]:3d} sum={v[1]:8.1f}  {k[0][:44]} -> {k[1][:24]}")
```

---

## 7. 验收门禁（每项都要过）

| 项 | 门限 | 当前值 |
| --- | --- | --- |
| 精度 cosine（逐行最差） | ≥ 0.999 | 0.999993 |
| relative L2（逐行最差） | ≤ 0.02 | 0.004937 |
| 业务门禁（`decider/verify_business.py`，如适用） | 与基线一致 | — |
| 性能 | 同卡、同 bench 参数（1000 req / warmup 32 / seed 42），记录 avg/P50/P99/QPS | avg 21.27 / p50 21.26 / p99 21.39 / 47.0 QPS |
| 图结构 | `restore_matmulv3.py` 后 op_summary 与基线差分；关注 `MatMulV3`/`Cast`/`StridedSlice` 计数 | — |

判定原则（沿用 `docs/optimization-summary.md` §5）：e2e 落在历史 2.8% 方差带内就记"方差带内"，以单 case AI 时间（kernel 级）为准；不要用跨机 A 的绝对值当门禁。

---

## 8. 执行记录（后续会话填写）

| 实验 | 改动 | 精度 cos / relL2 | e2e avg / p50 / p99 (ms) | QPS | op 关键变化 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 基线（交付态） | — | 0.999993 / 0.004937 | 21.27 / 21.26 / 21.39 | 47.0 | MatMulV3 186、Cast 232、StridedSliceV3 90 | — |
| E1 关 NZ Pass | 关闭 `WeightNzAndMatMulV3Pass`（同一 AIR 改 `.so`/`.so.off` 编译） | 0.999995 / 0.005790 | 20.886 / 20.883 / 21.021（pass-off，chip 11 同窗；NZ-on mean 20.274，Δ **+0.611 ms / +3.02%**） | 47.856（pass-off；NZ-on 49.243 / 49.355） | MatMulV3 186（MIX_AIC）→ MatMulV2 186（AI_CORE）；同 iter 其余计数全同（826 ops、Cast 232、StridedSliceV3 90、RmsNorm 67、InplaceAddRmsNorm 48、Swish 42、Sigmoid 24、CausalConv1d 18、ChunkGatedDeltaRule 18） | 否决：pass-off 反而慢 **+0.611 ms（+3.02%）**，Pass 有益；方向与 §5 假设（−0.8~−1.5 ms）相反，刚过 ±2.8% 带、属温和收益。旧「争用归一化 ~30%」估计作废——逐 op 膨胀比 1.03（Sin/Cos）–2.79（MatMul）不统一，因子不可辨识，估计跨 +29.9%~−6.4%（跨零）；37.712 ms / 37374.4 µs 仅作污染测量。结构性结论不变：唯一差 MatMulV3→MatMulV2，推翻 §4.5 假设 #1（48 对 cast 原样保留） |
| E2 FIA→FA | `:211-227` | 无（converter 拒 BSND，未导出） | 无（不可导出） | 无（不可导出） | torchair fx2ge 仅支持 `input_layout ∈ {BSH, BNSD}`，BSND 抛 `NotImplementedError`（npu_fusion_attention.py:34-35,68-71）；现码为 BSND | 不可行，撤回：复现 §2 既有否决；改 BNSD 需 S=1001 下 4 个 transpose，对声称 −80 µs 为负 ROI |
| P1-1 MLP+swiglu | `:62-77` | 0.999995 / 0.004146 | 20.868 / 20.861 / 20.972（chip 4；vs baseline-1 21.352 −0.484 ms / −2.27%） | 47.896（chip 4；baseline-1 46.813） | `SwiGlu` 出现；`Swish` 44→20、`Mul` 68→44、`Cast` 165→141，各 −24（对应 24 层 MLP）；AIR −88,124 B | 采纳：e2e −2.27%（vs baseline mean −2.05%）在 ±2.8% 带内；按 kernel 级证据采纳（op 计数各 −24、单算子 probe PASS：`silu(first_half)*second_half`，relL2 0.001660 vs 反序 0.613299）；chip-13 run1/run2 同向佐证 |
| P1-2 SplitV | `:133-137,196,202-203` | 0.999993 / 0.004937（与基线逐位一致） | 20.556 / 20.548 / 20.710（chip 4；vs baseline-1 −0.796 ms / −3.73%） | 48.625（chip 4；baseline-1 46.813） | `StridedSliceV2` 92→0；`SplitV` 出现（40） | 采纳：−3.73%（vs baseline mean −3.52%）超 ±2.8% 带；过 backlog §1.5 双门禁（dynamo 未分解回 slice、输出逐位一致）；chip-13 佐证 |
| P1-1+P1-2 组合 | `:62-77` + `:133-137,196,202-203` | 0.999995 / 0.004146 | 20.109 / 20.104 / 20.266（chip 4；vs baseline-1 −1.242 ms / −5.82%，vs baseline mean −5.62%） | 49.698（chip 4） | `SwiGlu` 出现、`StridedSliceV2` 92→0、`SplitV` 40 | 采纳：chip-4 规范值 −5.62% / −1.197 ms（大于 chip-13 的 −4.53%）；绝对时延 20.109 vs chip-13 20.095/20.138（spread 0.21%，跨芯片一致）；combo 自身距 ~19.5–20.0 目标尚差 0.109 ms，全链 P1-5 已达标（见下行） |
| P1-3 norm cast | `:38-40,57-59` | 无新测量（no-op） | 无（no-op） | 无（no-op） | pbtxt（fx2ge 后、GE pass 前）`RmsNorm` 115 = 编译后 `RmsNorm` 67 + `InplaceAddRmsNorm` 48；编译后 `Cast` 232，其中 136 非 hidden-pair 对应 139 个显式 fp32 站点；hidden-pair 为 `trans_Cast_*` 3D→2D flatten 往返；按节点名前缀精确拆分（combo 图）trans_Cast 174 节点 / 1386.0 µs = AI 时间 6.86%，model-code Cast 58 / 358.5 µs = 1.78%，合计 232 / 1744.5 µs，非 norm 残留 | 按原写法否决（no-op：`npu_rms_norm(bf16 x, bf16 w)` 返回 bf16，`return y.to(x.dtype)` 不消任何 cast；`gate.float()` 属 `"16016,128"` 家族，参考图 A 同样有）；新 backlog 候选：GE 以 bf16→fp32→bf16 实现 matmul 前 3D→2D flatten。**更正**：显式 2D flatten 首试并非挂起——冷编译需 234,074.5 ms（234 s），8 min 被杀系过早；`kernel_meta/` 于 06:54 新增 `kernel_meta_temp_9864803020239436846`，后续同图编译 11,097–11,265 ms（combo 11,356–12,764 ms）；`io_spec()` 固定 B=1 STATIC、`b*s` 折叠为 `s`，无动态乘积问题。冷编译成本是已量化权衡，由 P1-4 处理 |
| P1-4 MLP 显式 2D | `:62-77`（`x.squeeze(0)` / `h.unsqueeze(0)`） | 0.999995 / 0.004146（bit-neutral） | 19.536 / 19.682（2 样本；combo mean 20.168 → squeeze mean 19.609，pooled −0.5587 ms / −2.77%；p50 combo 20.140 → squeeze 19.604，−0.5365 ms / −2.66%） | 50.969（combo 49.562；+2.84%） | trans_Cast 174→78 节点（1386.0→777.1 µs，−43.9%）；ALL Cast 232→136（1744.5→1140.8 µs）；MatMulV3 162 不变；total ops 724→628（−96） | **MLP-only 增量阶段（非终态，见 P1-5）**：e2e −2.77% 在 ±2.8% 边缘，按 kernel 证据（AI −656.9 µs、trans_Cast −608.9 µs）判定可采纳；分布不重叠（gap 0.257 ms）、精度 bit-neutral；MLP-only 值 19.609 ms；首次冷编译 234,074.5 ms 为量化权衡 |
| P1-5 全模块显式 2D | `:62-77` + `StartLuxGatedDeltaNet.forward` / `StartLuxAttention.forward` | 0.999995 / 0.004146（bit-neutral） | 19.178 / 19.174 / 19.274（chip 4；combo mean 20.139 → 19.178，−0.961 ms / −4.77%） | 52.116（combo 49.627；+5.02%） | ops 724→568、AI 20193.1→19132.1 µs（−1061.0 / −5.25%）；trans_Cast 174→0；all Cast 232→76（1744.5→575.4 µs）；MatMulV3 162 不变；model-code Cast 58→76 为 gated-norm 析出 | 已测待采纳：−4.77% 超 ±2.8% 带，90.6% kernel saving 落到 e2e；累计 vs 原基线 21.3055 −2.127 ms / −9.99%，19.178 ms 低于 ~19.5–20.0 目标并 0.472 ms 优于 A 19.65；warm compile 9349.3 ms、HBM rise 2174 MB；首次冷编译 504,141.3 ms |

---

## 9. 文档回填约定

- 采纳/否决结论回填：本文 §8；`docs/optimization-summary.md` §1/§2 与 §7 证据索引（该文档在 `models/startlux-decision-v1.1-0.8b-instant/`，是同级非 text 版，按项目惯例共用）。
- 实测数字写英文文件名附录表（`performance.md` 范式），保留测量日期/芯片/计时边界等 provenance。
- 模型 `README.md` 只写当前状态（§3.2 性能、§3.4 精度直接改写），不写"曾经如何"。
- 提交信息用英文 Conventional Commits，例如 `perf(startlux-text): fuse MLP gate/up via npu_swiglu`。

---

## 10. 风险与回退

| 风险 | 处置 |
| --- | --- |
| E1 关闭 NZ Pass 后变 `MatMulV2` 更慢 | 立即恢复 `.so`，记录结论，转 P1 |
| `npu_swiglu` 语义/顺序不对 | 单算子 probe 对纯数学参考；不通过则放弃 P1-1 |
| SplitV 动态图被 dynamo 分解回 slice | probe 判据 1 直接否决，零成本退出 |
| 系统级 Pass 改动影响同机其他任务 | 实验窗口内独占；结束立即恢复 |
| profiling 抖动误判 | 以 bench avg/P50 与干净步 op_summary 为准，不看单次 24 ms 步 |
| MatMulV3 重建值失真 | 只用于方向判断；采纳看同窗口 bench + 精度 |
