# OneJev1.6-0.8B 多模态极简流程

> **状态：hold，暂缓适配。** `run.sh` 不可执行，阻塞有三处：它调用的 `runtime/build/executor` 已不在仓库；`model/generate_inputs.py` 产出打包格式 `inputs.bin` + `records.json`（schema `ge-records/3`），而 `ge_execute` 要的是 `cases.json`（schema `ge-cases/1`）加每 case 独立 npy；bench 的 `--readout` 把 CPU decider 计入 e2e，`ge_execute` 没有对应参数。迁移因此不是换参数，需要重写输入生成并先定 readout 计时方案。不依赖执行端的部件各自独立：`model/generate_inputs.py`、`model/golden.py`、`scripts/prepare.sh`、`model/export_air.py`、`scripts/verify/`；`tools/compare.py` 需要推理产物，同样受阻。本目录当前无 `tests/` 覆盖。

本目录是 [OneJev1.6-0.8B](https://huggingface.co/OmniJev/OneJev1.6-0.8B)
多模态 System One 决策模型的**单图+文本**极简交付：每个部件一个自包含脚本、
部件间零 import 依赖，只通过 `work/` 下的文件契约交互，共用一个 7 输入动态
多模态 AIR（`input_ids` [1,S] + `pixel_values` [P,1536] + 5 个位置/预计算
张量 → `scores` [1,26]）。B=1 由图钉死；S（token 数）与 P（图像 patch 数）
是仅有的动态轴。适配流程、迁移手册与全部已核实事实见
[docs/adaptation-plan.md](docs/adaptation-plan.md)。

## 1. 背景

### 1.1 官方地址

| 项目 | 地址 |
| ---- | ---- |
| 模型地址 | https://huggingface.co/OmniJev/OneJev1.6-0.8B |
| 官方推理/决策包 | https://github.com/OmniJev/Flide （PyPI `flide`，格式 `onejev-slot-v2`） |
| 模型介绍 | https://github.com/OmniJev/OneJev |

架构为 `Qwen3_5ForConditionalGeneration`（Qwen3.5-0.8B 微调平均）：文本主干
24 层 = 18 GatedDeltaNet 线性注意力 + 6 全注意力（**causal**，本 checkpoint
无 modeling_overrides）；vision 塔 12 层，patch16/merge2。读出 = tied lm_head
的 A–Z 槽位行 logits 前 K 列 → softmax(/T)；无 calibration.json → **温度
1.0**。prompt 渲染为官方 `onejev-labels-v2`（vendor/ 节选自 flide 0.1.16，
prompt/calibrate 与官方 wheel 逐字节一致，渲染结果已与官方包 14/14
byte-identical 交叉验证）。

### 1.2 业务约束（v1）

- **batch = 1**、**每 case 恰一张图 + 一个问题分支**（debias=1）。
- 纯文本 case、多图、视频、choice>256 分块、debias>1、MTP 均不在 v1 范围。
- 验证数据：`fixtures/cases.jsonl`（官方 System One 请求风格构造，14 条：
  choice K=2..8 / noul / score 3–5 级；5 张确定性生成的截图风 PNG，
  S=388..1083、P=1024..3520）。
- transformers 版本硬约束：checkpoint 与导出补丁面向 **5.17.0**（task-local
  安装于 `work/deps/py`，脚本自校验版本）；容器全局 5.8.1 是 StartLux 交付
  验证组合，**不得改动**。

目录中的五个文件契约部件：

| 部件 | 输入 | 输出（默认路径） |
| ---- | ---- | ---- |
| `model/generate_inputs.py` | cases.jsonl + 权重目录 | `work/inputs/{inputs.bin, records.json, readout.json}` |
| `model/golden.py` | `work/inputs` + 权重目录 | `work/golden/{outputs.bin, result.json}` |
| `model/export_air.py` | 权重目录 | `work/build/onejev.air` |
| 图执行 runner（**当前无实现，见顶部 hold 说明**） | `onejev.air` + `work/inputs` | `work/output/{outputs.bin, result.json}`，bench 另加逐调用记录 |
| `tools/compare.py`（复用） | golden 与 infer 输出 | `work/output/comparison.json` |

另含 `run.sh`（一键串联）、`scripts/prepare.sh`（依赖构建）、
`scripts/verify/`（输入审计与 rope/prompt/契约三个交叉验证脚本）、
`model/modeling_onejev.py`（Decision 包装 + 全部 NPU/导出补丁）、
`native_proto/`（ChunkGatedDeltaRule GE 原型，LD_PRELOAD 契约同 -text）、
`vendor/`（flide 节选）与 `model/model.py`（历史原型脚本，仅存档，
被上述部件取代）。

## 2. 运行

### 2.1 环境信息

| 组件 | 版本 |
| ---- | ---- |
| CANN（Ascend-cann-toolkit） | 9.1.0（V100R001C11SPC001B243，aarch64-linux） |
| NPU 驱动（npu-smi） | 25.5.t6.b001 |
| 硬件 | Ascend910_9382 |
| Python | 3.12.13（容器 `/usr/local/python3.12.13`） |
| torch / torch_npu | 2.12.0+cpu / 2.12.0 |
| transformers | **5.17.0（task-local `work/deps/py`）**；容器全局 5.8.1 不动 |
| numpy / Pillow | 2.5.1 / 13.0.0 |

### 2.2 运行脚本

```bash
export ONEJEV_MODEL_DIR=/path/to/hf-model        # 权重材料，不自动下载
# 当前在「图执行 runner 构建」一步失败，见顶部 hold 说明
bash run.sh --device 0                            # 精度全流程
bash run.sh --device 0 --iterations 2000 --instances 1   # 追加 bench
```

`run.sh` 依次执行：生成输入 → CPU golden → 依赖构建 → 图执行 runner 构建
（**当前无实现，见顶部 hold 说明**）→ 多模态 AIR 导出 → 精度推理 →
`tools/compare.py` 门禁（cosine≥0.999、relative L2≤0.02）；`--iterations`
追加多实例 bench（`--instances`、`--aicore-num`、`--readout
work/inputs/readout.json` 把 CPU decider 计入 e2e）。

### 2.3 怎么导出 AIR

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ONEJEV_MODEL_DIR=/path/to/hf-model
bash scripts/prepare.sh
LD_PRELOAD=work/deps/chunk_gated_delta_rule_proto/install/lib/libchunk_gated_delta_rule_compat_proto.so \
    python3 model/export_air.py --device 0 --export-dir work/build --export-name onejev
```

产物 `work/build/onejev.air`（约 1.7 GB）。导出后自动做 Data 节点绑定双射
校验（7 输入、逻辑名与 io_spec 一致）并删除 dynamo.pbtxt。

## 3. 运行时

### 3.1 输入输出

| 张量 | 方向 | dtype | shape | 语义与约束 |
| ---- | ---- | ----- | ----- | ----- |
| `input_ids` | 输入 | int64 | [1, S] | 官方渲染 prompt，image pad 已展开为 P/4 个 |
| `pixel_values` | 输入 | bfloat16 | [P, 1536] | 官方 image processor 产物主机侧 cast bf16 |
| `position_ids` | 输入 | int64 | [3,1,S] | 官方 `get_rope_index` MRoPE 三轴 |
| `interp_indices` | 输入 | int64 | [P, 4] | 位置表 bilinear 采样 tap（48×48 表） |
| `interp_weights` | 输入 | float32 | [P, 4] | 对应权重 |
| `vision_position_ids` | 输入 | int64 | [P, 2] | vision 轴向 (h,w) id |
| `feature_index` | 输入 | int64 | [1, S] | image token 位 → merged 特征行 0..P/4-1，其余 -1 |
| `scores` | 输出 | float32 | [1, 26] | A–Z 26 列 letter logits |

`image_grid_thw` 与 `cu_seqlens` **不是图输入**：v1 补丁下其全部消费者都由
上表预计算张量供数（单图注意力忽略 cu 分段），二者对图是死输入（torchair
会将其 DCE）；每 case 的真实 grid 以 `records.json` 的 `grid` 元数据字段
携带，供 golden 的官方整模前向使用。records schema 为 `ge-records/3`
（tensors 按位绑定图输入）。

### 3.2 精度

2026-11-17 旧环境（CANN 9.1.0，Ascend910_9382，逻辑芯片 6）产品化 AIR +
ge-records/3 executor 全 14 case 闭环：

- `tools/compare.py` 门禁 **PASS**：worst cosine **0.999979**（门限 0.999），
  worst relative L2 **0.008327**（门限 0.02）；golden 为 CPU FP32 官方整模
  前向（与 9→7 输入契约的图路径交叉验证 cosine=1.000000000）。
- 图冷启动（不计入请求时延）：CompileGraph 27666.5 ms、LoadGraph
  8278.4 ms（本机 TBE kernel 缓存已热；全新目录首跑会显著更慢，-text 参考
  首跑 CompileGraph ~17.5 min）。
- 逐 case 模型段时延（infer 单次顺序执行；S/P 见 fixtures 表）：

| case | S | P | ms |
| ---- | --: | --: | --: |
| onejev-00 | 417 | 1024 | 334.224（首 case 含冷开销） |
| onejev-01 | 399 | 1024 | 45.556 |
| onejev-02 | 417 | 1024 | 40.754 |
| onejev-03 | 388 | 1024 | 45.219 |
| onejev-04 | 617 | 1792 | 55.196 |
| onejev-05 | 588 | 1792 | 56.676 |
| onejev-06 | 603 | 1792 | 54.074 |
| onejev-07 | 953 | 3072 | 96.547 |
| onejev-08 | 903 | 3072 | 92.992 |
| onejev-09 | 1049 | 3520 | 112.850 |
| onejev-10 | 1083 | 3520 | 113.028 |
| onejev-11 | 430 | 1200 | 49.185 |
| onejev-12 | 1011 | 3520 | 112.336 |
| onejev-13 | 1039 | 3520 | 112.770 |

稳态（除首 case）40.8–113.0 ms，随 S/P 近线性。

### 3.3 性能数据

2026-11-17 旧环境（逻辑芯片 6，共享机器，不限核）单实例闭环 bench：
固定 seed 42 从 14 case 池有放回抽样，预热 32 请求后测 **2000 请求**，
readout decider 计入 e2e；**0 错误**。e2e = H2D+执行+同步+D2H+readout。

| 指标 | 值 |
| ---- | --: |
| QPS（单实例） | **13.95** |
| e2e avg / p50 / p99 (ms) | 71.70 / 53.96 / 113.87 |
| exec avg / p50 / p99 (ms) | 71.30 / 53.50 / 113.22 |
| H2D avg (ms) | 0.37 |
| D2H / decider avg (ms) | 0.03 / ~0.001 |
| HBM 占用（单实例） | ~3.2 GB |

时延分布由混合 case 池决定（S=388..1083、P=1024..3520，见 §3.2 逐 case 表；
p50≈S≈400-600 档、p99≈S≈1000+ 档）。多实例扩展与 `12|24` 限核对照的正式
3-seed 数据待新环境按 docs/adaptation-plan.md §6.7 补测（旧环境一次 4 实例
尝试在限核 kernel 重编译期间被外部 SIGTERM 中断，无有效数据）。
