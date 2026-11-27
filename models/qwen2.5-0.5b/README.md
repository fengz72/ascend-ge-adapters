## 1. 背景

[Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)，权重目录由 `QWEN_MODEL_DIR` 给出，流程不自动下载。骨干是 24 层 Qwen2 decoder：hidden 896、GQA 14 个注意力头 / 2 个 KV 头（head_dim 64）、intermediate 4864、RMSNorm + SiLU MLP + half RoPE、vocab 151936，`lm_head` 与 `embed_tokens` 绑定。读出头不是全词表投影：`install_score_head()` 从 tied embedding 取出 8 行（token id 15–22）作 `[8, 896]` 评分头，对每条请求末 token 的 hidden 做 FP32 矩阵乘，得到 `[N, 8]` 分数。

业务是共享 prefix 的批量打分：每个 case 一段 prefix 只参与一次计算，其后跟随 16 条业务请求（batch 固定 16），每条请求取末 token 的 8 个固定 token 分数，列序由 [config/target_tokens.json](config/target_tokens.json) 定义。输入池 100 个 case，prefix 20–25 token，请求总长按 lognormal 抽样（`mu=4.997`、`sigma=0.167`，截断到 `[prefix+1, 218]`）；一次前向完成 prefix + 全部请求，无 KV-cache。


## 2. 环境

### 2.1 Ascend910_9382（`--platform a3`）

| 项目 | 值 |
|---|---|
| SoC | `Ascend910_9382` |
| CANN | ≥ 9.1.0（实测 9.1.0，V100R001C11SPC001B243，aarch64-linux）；Prefix AscendC 算子的四参数 `MallocOutputTensor` 与 `OP_MODULE_ID` 日志符号在 9.0.0 不存在 |
| NPU 驱动 | npu-smi 25.5.t6.b001 |
| Python | 3.12.13（`/usr/local/python3.12.13`） |
| Torch | 2.12.0+cpu |
| TorchNPU | 2.12.0（torchair 内联于 `torch_npu.dynamo.torchair`，提供 `dynamo_export`） |
| Transformers | 5.8.1（只用于 CPU golden） |
| numpy | 2.5.1 |
| safetensors | 0.8.0（导出时读 `model.safetensors`） |
| ninja | 1.13.2（`scripts/prepare.sh` 的构建自检 `import ninja`） |
| cmake / GCC | 3.27.9 / 12.3.1（构建 Prefix 算子与 NZ pass） |


### 2.2 Ascend950PR_958b（`--platform a5`）

| 项目 | 值 |
|---|---|
| SoC | `Ascend950PR_958b` |
| CANN | ≥ 9.3.0（a5 的 GE 交付件依赖 attached 辅流接口，9.2.x 没有该接口） |
| NPU 驱动 | npu-smi 25.6.rc3 |
| Python | 3.12.16（`/usr/local/python3.12.16`） |
| Torch | 2.12.0+cpu |
| TorchNPU | 2.12.0（提供 torchair `dynamo_export`） |
| Transformers | 5.8.1（只用于 CPU golden） |
| numpy | 2.5.4 |
| safetensors | 0.8.0（导出时读 `model.safetensors`） |
| ninja | 1.13.2（`scripts/prepare.sh` 的 a5 构建自检 `import ninja`） |
| cmake / GCC | 3.31.12 / 12.3.1（构建 Prefix 算子与 NZ pass） |


## 3. 运行指南

```bash
cd models/qwen2.5-0.5b                          # run.sh 用相对路径，必须在本目录执行
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B     # 权重材料，不自动下载
bash run.sh --platform a3 --device 2            # --device 是 npu-smi 的 Chip Logic ID
```


### 3.1 输入与输出

| 名称 | shape | dtype | 说明 |
|---|---|---|---|
| `input_ids` | `[-1]` | int64 | packed token：先 prefix，再各请求私有 token；`0 ≤ id < 151936` |
| `position_ids` | `[-1]` | int64 | prefix 段自 0 起，每条请求自 `len(prefix)` 重新计数；取值 `< 2048` |
| `actual_seq_lengths` | `[(batch+1)]` | int64（a3）／int32（a5） | 各段累计结束位置，`batch` = 请求数（池内恒 16） |

| 输出 | shape | dtype | 业务语义 |
|---|---|---|---|
| `scores`（产物文件名是位置名 `output_0`） | `[-1, 8]` | float32 | 每条业务请求末 token 的 8 个评分列，列顺序来源 [config/target_tokens.json](config/target_tokens.json) |


### 3.2 Patch

| 替换目标 | 替换为 | 来源与安装位置 |
|---|---|---|
| 每层的 packed 因果注意力，且共享 prefix 的 K/V 只存一份、只算一份 | a3：AscendC `PrefixInferAttentionScore`（`torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score.tensor`，`sparse_mode=2` 读固定 `[2048, 2048]` 压缩因果掩码、`pre_tokens=INT_MAX` 与 `next_tokens=0` 即因果，bounds 作为图输入保持动态）；a5：CANNBotDSL `PrefixAttention`（`torch.ops.ascendc_ops.prefix_attention`，FP16 GE 交付件） | `third_party/custom_development_code/custom_op/prefix-attention/{ascendc,cannbot-dsl/ge}`；`scripts/prepare.sh <platform>` 构建到 `work/deps/prefix_attention/install/{python,opp}`，由 `work/deps/env.sh` 的 `PYTHONPATH` + `ASCEND_CUSTOM_OPP_PATH` 激活 |
| `MatMul`/`MatMulV2`/`MatMulV3` 常量权重的 ND 排布 | 自定义融合 pass `MatMulWeightNZPass`（工程名 `WeightNzAndMatMulV3Pass`）：权重 ND → FRACTAL_NZ，节点统一重建为 `MatMulV3`；注册阶段 `kAfterOriginGraphOptimize` | `third_party/custom_development_code/fusion_pass/WeightNzAndMatMulV3Pass`；`cmake --install` 装进活动 CANN 的 `vendors/custom_nz_pass/custom_fusion_passes/`，GE 自行加载。可选性能 pass，缺省图仍可运行，生效证据是编译日志的 `matmul_match num is 169 / matmul_effect num is 169` |
| 读出头：checkpoint 无独立 `lm_head`（与 `embed_tokens` 绑定） | `install_score_head(token_ids)` 从 tied embedding 取 8 行派生 `[8, 896]` FP32 评分头，在 `load_state_dict(strict=True)` 与 `requires_grad_(False)` 之后调用 | [model/modeling_qwen.py](model/modeling_qwen.py)、[model/export_air.py](model/export_air.py) |
| 导出侧的动态轴与位置绑定 | torchair `dynamo_export(dynamic=True)` + `experimental_config.frozen_parameter`；reshape 的动态轴必须写 `-1`（写显式符号尺寸会让 dynamo 提升 SymInt 链、多出 Data 节点并生成 GE 无法推断的 `Pack`），导出后删掉重复冻结权重的 `dynamo.pbtxt` | [model/export_air.py](model/export_air.py) |

### 3.3 产物路径

```text
work/
├── deps/              # 第 2 步 scripts/prepare.sh：prefix_attention/install/{python,opp}、weight_nz/build、env.sh
├── inputs/            # 第 3 步 model/generate_inputs.py：cases.json（ge-cases/1）+ case-XXXX/{input_ids,position_ids,actual_seq_lengths}.npy
├── golden/            # 第 4 步 model/golden.py：result.json + case-XXXX/output_0.npy（HF FP32 CPU 参考，[16, 8]）
├── build/             # 第 6 步 model/export_air.py：packed.air（993754309 字节，FP16 冻结权重）
├── output/            # 第 7 步 ge_execute：result.json + case-XXXX/output_0.npy；第 8 步 tools/compare.py 把 comparison.json 写进这里
├── bench/             # 第 9 步（--iterations）ge_execute：result.json（config 回显、pool、qps、samples_per_second、e2e 分位、逐实例、逐请求 records）；多 seed 轮次写在 bench-<seed>/
├── logs/              # 每步一份 *.log
└── torch_extensions/  # TORCH_EXTENSIONS_DIR：算子 torch binding 的 JIT 缓存
```


## 4. 精度和性能

| 平台 | case | cosine 最小 | relative L2 最大 | 业务 QPS | Avg | P99 |
|---|---|---|---|---|---|---|
| `Ascend910_9382`（a3） | seq: avg 150 p99 218 / prefix=[20, 25] | 0.9994395 | 0.033784 | 2185.282 | 36.556 | 40.862 |
| `Ascend950PR_958b`（a5） | seq: avg 150 p99 218 / prefix=[20, 25] | 0.9996369 | 0.028409 | 2706.961 | 17.686 | 20.288 |

a3 精度两列为 §2.1 环境下 100 case / 1600 样本全量实测。a3 性能三列口径为 CANN 9.0.0、
Python 3.11.15、torch 2.9.0，bench `8|16`、5 实例、3000 请求 × 3 轮（seed 42/43/44）的
业务 QPS 中位数与 e2e avg / p99 中位数。
