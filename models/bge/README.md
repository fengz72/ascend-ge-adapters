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
| `Ascend910_9382` | 全量 5 组 | | | | | |

