# StartLux-Decision v1.1 0.8B Instant 纯文本极简流程（-text）

本目录是 [StartLux-Decision-v1.1-0.8B-Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant)
决策模型的**纯文本**极简交付：每个部件一个自包含脚本、部件间零 import
依赖，只通过 `work/` 下的文件契约交互，共用一个 2 输入动态 text AIR
（`input_ids` [1,S] / `position_ids` [3,1,S] → `scores` [1,26]）。
B=1 由图钉死；

## 1. 背景

### 1.1 官方地址

| 项目             | 地址                                                                                          |
| ---------------- | --------------------------------------------------------------------------------------------- |
| 模型地址         | https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant                    |
| 官方 git 推理脚本 | https://github.com/StartLuxLabs/StartLux-Decision/tree/main/examples                          |


### 1.2 业务约束

- **batch = 1**
- **仅纯文本**：不接收图片；case 文件中任何携带图片的行都会报错
- **验证数据**：`fixtures/dev.txt`（业务 dev 池，18 条 case，`dev-intent-q00`
  ~ `dev-intent-q08` 各 `r0`/`r1`），实测 token 长度 S = 1001–1005。
- 每条 case 为一行 JSONL，字段固定为 `{state, questions, targets, meta}`；
  恰一个 `choice` 问题、2–26 个 criteria，targets 为 criteria 上的概率分布
  （和 ≈ 1）。经官方 prompt 渲染（`from_systemone → messages →
  plain_prompt`，chat template 关闭 thinking）后
  `tokenizer.encode(prompt, add_special_tokens=False)` 得到 `input_ids`。

目录中的五个文件契约部件：

| 部件                                              | 输入                       | 输出（默认路径）                                                    |
| ------------------------------------------------- | -------------------------- | ------------------------------------------------------------------- |
| `model/generate_inputs.py`                        | dev.txt JSONL case 文件    | `work/inputs/{inputs.bin, records.json}`                            |
| `model/golden.py`                                 | `work/inputs` + 权重目录   | `work/golden/{outputs.bin, result.json}`                            |
| `model/export_air.py`                             | 权重目录                   | `work/build/text.air`                                               |
| `runtime/build/executor`（`runtime/executor.cpp`） | `text.air` + `work/inputs` | `work/output/{outputs.bin, result.json}`，bench 另加 `requests.csv` |
| `tools/compare.py`（复用）                        | golden 与 infer 输出       | `work/output/comparison.json`                                       |

另含 `run.sh`（一键串联）与 `scripts/prepare.sh`（依赖构建），不属于文件
契约部件。

边界与限制：


- 二进制依赖由本目录 `scripts/prepare.sh` 构建（源码在 `native_proto/`）：
  chunk proto 需 `LD_PRELOAD`（导出与执行都需要），
  WeightNzAndMatMulV3Pass 为可选性能 pass（缺省图仍可运行）；
  causal-conv1d vendor 算子不安装。
- `work/build/text.air` 首次在新目录编译时按形状逐一 TBE 编译，首跑偏慢
  属预期。

## 2. 运行

### 2.1 环境信息

本机已验证组合（版本核对日期 2026-11-12）：

| 组件                        | 版本                                          |
| --------------------------- | --------------------------------------------- |
| CANN（Ascend-cann-toolkit） | 9.1.0（V100R001C11SPC001B243，aarch64-linux） |
| NPU 驱动（npu-smi）         | 25.5.t6.b001                                  |
| 硬件                        | Ascend910_9382                                |
| Python                      | 3.12.13（容器 `/usr/local/python3.12.13`）     |
| torch                       | 2.12.0+cpu                                    |
| torch_npu                   | 2.12.0                                        |
| transformers                | 5.8.1                                         |
| numpy                       | 2.5.1                                         |


### 2.2 运行脚本


```bash
export STARTLUX_MODEL_DIR=/path/to/hf-model        # 权重材料，不自动下载
bash run.sh --device 0                                        # 精度全流程
```

`run.sh` 依次执行：生成输入 → CPU golden → 依赖构建 → C++ executor 构建
→ text AIR 导出 → 精度推理 → `tools/compare.py` 门禁


### 2.3 怎么导出 AIR

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export STARTLUX_MODEL_DIR=/path/to/hf-model
bash scripts/prepare.sh
LD_PRELOAD=work/deps/chunk_gated_delta_rule_proto/install/lib/libchunk_gated_delta_rule_compat_proto.so \
    python3 model/export_air.py --device 0 --export-dir work/build --export-name text
```

产物为 `work/build/text.air`（当前约 1.5 GB）。

## 3. 运行时

### 3.1 输入输出

| 张量           | 方向 | dtype   | shape   | 语义与约束                        |
| -------------- | ---- | ------- | ------- | --------------------------------- |
| `input_ids`    | 输入 | int64   | [1, S]  | token 序列；`0 ≤ id < vocab_size` |
| `position_ids` | 输入 | int64   | [3,1,S] | MRoPE 三轴位置；纯文本即三轴 `arange(S)` |
| `scores`       | 输出 | float32 | [1, 26] | A–Z 26 列 letter logits           |



### 3.2 性能数据


