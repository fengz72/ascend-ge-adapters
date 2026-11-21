## 1. 背景

业务侧提供的 BGE ONNX 导出，内联权重约 1.35 GB。

骨干是 24 层双向 padding-mask 注意力（BSND 布局，Q/K 共享 `1/sqrt(D)` 缩放链）；打分头对 10112 个固定候选做「投影 → L2 归一化（带 epsilon 下限）→ 二次投影」，输出 `scores`。

候选打分：一次请求给一条文本 query、一段行为历史与一组类目统计，输出 10112 个候选的分数，业务消费其中的 **Top-10**（`dense_id = 输出下标 + 1`）。八个输入特征里七个是 INT64、`stat_ratio` 是 FP32；batch 与文本/行为长度动态，业务请求为 batch=1。


## 2. 环境

### 2.1 Ascend910_9382

| 项目 | 值 |
|---|---|
| SoC | `Ascend910_9382`|
| CANN | 9.0.0 |
| Python | 3.11.15（`/usr/local/python3.11.15`） |
| numpy | 1.26.4 |
| onnxruntime | 1.26.0 |

ONNX 直喂 GE，不经 torch，所以没有 Torch / TorchNPU / Transformers 依赖。

## 3. 运行指南

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export BGE_MODEL_PATH=/path/to/model.onnx        # 业务原始导出，内联权重，约 1.35 GB
bash models/bge/run.sh --device 0
```

### 3.1 输入与输出

| # | 名称 | shape | dtype | 生成值域 |
|---:|---|---|---|---|
| 0 | `input_ids` | `[-1, -1]` | int64 | token id 101…12419；文本长 6…13 |
| 1 | `attention_mask` | `[-1, -1]` | int64 | 全 1（batch=1，宽度即真实长度，无 padding） |
| 2 | `long_cid3_raw` | `[-1, 128]` | int64 | 前 L 位 1…42689，其余 0 |
| 3 | `long_action` | `[-1, 128]` | int64 | 前 L 位 0…2，其余 0 |
| 4 | `long_time_bucket` | `[-1, 128]` | int64 | 前 L 位是序号 1…L，其余 0 |
| 5 | `stat_dense_raw` | `[-1, 20]` | int64 | 前 `stat_count` 位 1…43887，其余 0 |
| 6 | `stat_ratio` | `[-1, 20]` | float32 | 与 `stat_dense_raw` 同位置非零，取值 (0, 0.973] |
| 7 | `stat_count` | `[-1]` | int64 | 1…4，等于非零统计槽数 |


| 输出 | shape | dtype | 业务语义 |
|---|---|---|---|
| `scores` | `[-1, 10112]` | float32 | 10112 个候选的分数，`dense_id = 下标 + 1`；列顺序由图自身决定，仓库里没有描述它的文件 |


### 3.2 Patch

`BgeFusedInferAttentionPass`: 源码在 `third_party/custom_development_code/fusion_pass/BgeFusedInferAttentionPass`。

| | 内容 |
|---|---|
| 改写 | 每层的 score `BatchMatMulV2` → mask `Add` → `SoftmaxV2` → value `BatchMatMulV2` → 输出 `Transpose`，以及 Q/K 两侧的 `1/sqrt(D)` 缩放 `Mul`，共 7 个节点，替换为一个内置 `FusedInferAttentionScore`（`input_layout="BSND"`、`sparse_mode=0`、`softmax_lse_flag=false`、`num_heads=num_key_value_heads`） |
| 注册阶段 | `kAfterBuiltinFusionPass` |

### 3.3 产物路径

```text
work/
├── inputs/    # 第 2 步：case-000N/{八个特征}.npy + cases.json（ge-cases/1）
├── golden/    # 第 3 步：result.json + case-000N/output_0.npy（onnxruntime CPU FP32 参考）
├── infer/     # 第 6 步：result.json + case-000N/output_0.npy（float32 [1,10112]）；第 7 步把 comparison.json 也写进这里
├── bench/     # 第 8 步（可选）：result.json（config 回显、qps、e2e/exec/d2h 分位、逐实例、逐请求 records）
└── logs/      # 每步一份 *.log；第 1 步先清掉上一轮的
```

## 4. 精度和性能

| 平台 | case | cosine 最小 | relative L2 最大 | 业务 QPS | Avg | P99 |
|---|---|---|---|---|---|---|
| `Ascend910_9382` | 全量 5 组 | 0.999996 | 0.002666 | 989.907 | 10.084 | 13.773 |

精度标准见 [^1]，性能口径见 [^2]。表中数字来自 2026-11-21 在 Chip Logic ID 7 上的同一次 `run.sh --device 7 --iterations 30000`。

[^1]: 逐行判定（每 case 一行 10112 列），门限 cosine ≥ 0.9999、relative L2 ≤ 0.003，`tools/compare.py` 要求全部行同时满足两条才算通过。golden 由 `model/golden.py` 用 onnxruntime 1.26.0 CPU FP32 在 `model/generate_cases.py` 的 seed 42 随机池上生成；设备侧是 `ge_execute --precision-mode force_fp16 --aicore-num 12|24`。relative L2 上限按 FP16 实测极值标定：`FusedInferAttentionScore` 只接受 FP16/BF16/INT8，`must_keep_origin_dtype` 编译期即报 EZ3002，因此没有 FP32 口径可选。五例的 Top-10（`dense_id = 输出下标 + 1`）ID 与顺序与 golden 完全一致。
[^2]: 性能按整池测量，不是逐 case，故只给一行。`ge_execute` bench 模式：10 实例闭环并发、seed 42 有放回抽样，30000 请求落在 30.306 s 窗口内、0 错误，每实例 3000 次，五例各 5925–6096 次。业务 QPS = 窗口内样本数 / 窗口时长，batch=1 时等于请求吞吐。计时边界是逐请求 e2e，含 H2D + execute + sync + 全量 D2H（`timed_phases`），不含编译（10 份图共 79.948 s）、加载（每实例 1.051–1.196 s）与预热（40 次共 0.411 s；池内 4 个 distinct shape 多于 `--warmup 1`，由 runtime 自动补齐）。Avg / P99 为 e2e 均值与 99 分位；同轮 e2e min 4.234 / max 18.013 ms，exec avg 9.932 / p99 13.629 ms，h2d avg 0.119 ms，d2h avg 0.033 ms。单轮结果，未做多轮中位与实例数扫描。

