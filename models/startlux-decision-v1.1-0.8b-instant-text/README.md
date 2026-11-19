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

| 部件 | 输入 | 输出（默认路径） |
| ---- | ---- | ---------------- |
| `model/generate_inputs.py` | dev.txt JSONL case 文件 | `work/inputs/cases.json`（`ge-cases/1`）+ 每 case 一个 `work/inputs/<case_id>/{input_ids,position_ids}.npy` + `work/inputs/readout.json` |
| `model/golden.py` | `work/inputs` + 权重目录 | `work/golden/<case_id>/output_0.npy` + `work/golden/result.json`（`cases` 索引） |
| `model/export_air.py` | 权重目录 | `work/build/text.air` |
| `runtime/build/ge_execute`（`runtime/`） | `text.air` + `work/inputs/cases.json` | `work/output/<case_id>/output_0.npy` + `work/output/result.json`；bench 只写 `work/bench/result.json`，逐请求在 `.records` |
| `tools/compare.py`（复用） | golden 与 infer 输出 | `work/output/comparison.json` |

另含 `run.sh`（一键串联）与 `scripts/prepare.sh`（依赖构建），不属于文件
契约部件。

边界与限制：


- 二进制依赖由本目录 `scripts/prepare.sh` 构建（源码在 `native_proto/`）：
  chunk proto 需 `LD_PRELOAD`（导出与执行都需要），
  WeightNzAndMatMulV3Pass 为可选性能 pass（缺省图仍可运行）；
  causal-conv1d vendor 算子不安装。
- `work/build/text.air` 首次在新目录编译时按形状逐一 TBE 编译，首跑偏慢
  属预期。
- `work/inputs/readout.json`（`startlux-text-readout/1`）记录每 case 的
  labels（criteria keys 顺序）、letters_asked 与官方 choice 温度，供业务侧
  read-out 使用；本目录的流程不消费它。

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

`run.sh` 依次执行：生成输入 → CPU golden → 依赖构建 → `ge_execute` 构建
→ text AIR 导出 → 精度推理 → `tools/compare.py` 门禁；`--iterations N` 追加 bench


### 2.3 怎么导出 AIR

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export STARTLUX_MODEL_DIR=/path/to/hf-model
bash scripts/prepare.sh
LD_PRELOAD=work/deps/chunk_gated_delta_rule_proto/install/lib/libchunk_gated_delta_rule_compat_proto.so \
    python3 model/export_air.py --device 0 --export-dir work/build --export-name text
```

产物为 `work/build/text.air`（当前约 1.5 GB）。

### 2.4 怎么单独跑一次推理

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=4                  # npu-smi 的 Chip Logic ID
LD_PRELOAD=work/deps/chunk_gated_delta_rule_proto/install/lib/libchunk_gated_delta_rule_compat_proto.so \
    ../../runtime/build/ge_execute \
    --graph work/build/text.air \
    --cases work/inputs/cases.json \
    --output-dir work/output \
    --device 0 \
    --precision-mode must_keep_origin_dtype
```

两个参数容易漏，后果不同：

- `--device` 是**必填**项，指进程内设备号。`ASCEND_RT_VISIBLE_DEVICES` 映射后
  可见设备从 0 开始编号，所以恒为 0，与上面的 Chip Logic ID 不是一回事
  （`run.sh` 里分别是 `$DEVICE` 与 `$DEVICE_CHIP`）。漏传直接 exit 2。
- `--precision-mode` 必须显式传 `must_keep_origin_dtype`：`ge_execute` 的默认
  值是 `force_fp16`，漏传**不会报错**，但 `run.sh` 的精度门禁（cosine
  ≥ 0.999、relative L2 ≤ 0.02）过不去。

`LD_PRELOAD` 在推理阶段同样必需：图里含 `ChunkGatedDeltaRule`，`ge_execute` 的
`CompileGraph` 也要用到 `native_proto/` 注册的算子原型。

## 3. 运行时

### 3.1 输入输出

| 张量 | 方向 | dtype | shape | 语义与约束 |
| ---- | ---- | ----- | ----- | ---------- |
| `input_ids` | 输入 | int64 | [1, S] | token 序列；`0 ≤ id < vocab_size` |
| `position_ids` | 输入 | int64 | [3,1,S] | MRoPE 三轴位置；纯文本即三轴 `arange(S)` |
| `output_0`（语义名 `scores`） | 输出 | float32 | [1, 26] | A–Z 26 列 letter logits |

输入经 `work/inputs/cases.json` 索引：每个 case 只有 `id` / `samples` /
`inputs[].{name,path}`，`path` 相对 `cases.json` 所在目录。**JSON 里不写
dtype 与 shape**，`ge_execute` 一律从 npy 头读；`inputs` 数组顺序就是图输入
顺序（位置绑定，`name` 只进日志）。

输出名必须是位置名 `output_0`：`ge_execute` 按 `"output_" + 下标` 命名
（`runtime/main.cpp:245`），GE 的语义张量名 `scores` 不透传。`golden.py` 因此
也写 `output_0`，`tools/compare.py` 按 name 对齐两侧。infer 产物布局
`work/output/<case_id>/output_0.npy` + `result.json` 的 `cases` 索引，与
golden 同构。



### 3.2 性能数据

口径：单实例 infer，`fixtures/dev.txt` 全池 18 case，不限核（2026-11-13）。

图冷启动（不计入请求时延）：

| 阶段         |        耗时 |
| ------------ | ----------: |
| CompileGraph |  17460.2 ms |
| LoadGraph    | 3676.36 ms |

首个 case 的 614.693 ms 含首次执行冷开销，单独列出；其余 17 个 case 为稳态：

| 指标 |                ms |
| ---- | ----------------: |
| min  |           24.9399 |
| P50  |           25.0753 |
| avg  |            25.193 |
| max  |           25.8392 |

逐 case 模型段耗时（S 为 token 长度）：

| case              |    S |        text ms |
| ----------------- | ---: | -------------: |
| dev-intent-q00-r0 | 1001 |        614.693 |
| dev-intent-q00-r1 | 1001 |        24.9774 |
| dev-intent-q01-r0 | 1003 |        25.8058 |
| dev-intent-q01-r1 | 1003 |        25.1426 |
| dev-intent-q02-r0 | 1005 |        25.5748 |
| dev-intent-q02-r1 | 1005 |        25.0582 |
| dev-intent-q03-r0 | 1002 |        25.5652 |
| dev-intent-q03-r1 | 1002 |        24.9828 |
| dev-intent-q04-r0 | 1002 |        25.0139 |
| dev-intent-q04-r1 | 1002 |        24.9828 |
| dev-intent-q05-r0 | 1004 |        25.8392 |
| dev-intent-q05-r1 | 1004 |        25.0753 |
| dev-intent-q06-r0 | 1002 |        24.9399 |
| dev-intent-q06-r1 | 1002 |        24.9402 |
| dev-intent-q07-r0 | 1004 |        25.1266 |
| dev-intent-q07-r1 | 1004 |        25.0217 |
| dev-intent-q08-r0 | 1004 |        25.1545 |
| dev-intent-q08-r1 | 1004 |        25.0836 |

### 3.3 并发扩展（case0 固定回放，`aicore_num=12|24`）

2026-11-16（CANN 9.1.0，Ascend910_9382，逻辑芯片 6）多实例闭环 bench：固定
`dev-intent-q00-r0`（S=1001）、每算子核上限 `12|24`。每档 3 轮（seed
42/43/44），每实例预热 32 请求后测 2000 请求；`--iterations` 为全实例合计
（=2000×N）。30 轮全部 0 错误，逐轮 QPS 极差 ≤3.0。e2e = H2D+执行+同步+D2H
（`result.json.timed_phases` 明示）；exec 仅含执行+同步，h2d/d2h 各
~0.026 ms。逐轮原始数据在 `work/bench-scaling/inst<N>-r<r>/`；bench 的逐请求
记录写在 `work/bench/result.json` 的 `.records` 数组（`instance` / `request` /
`case_id` / `shape_key` / `samples` / `h2d_ms` / `exec_ms` / `d2h_ms` /
`e2e_ms` / `first_hit` / `error`）。

跨轮中位数（QPS 与延迟单位 ms）：

| N | QPS | e2e avg | e2e p99 | exec avg | exec p99 | HBM 占用 |
|--:|----:|--------:|--------:|---------:|---------:|---------:|
| 1 | 31.5 |  31.70 |  31.93 |  31.64 |  31.84 |  3.3 GB |
| 2 | 52.0 |  38.48 |  38.85 |  38.42 |  38.79 |  6.6 GB |
| 3 | 66.5 |  45.12 |  47.68 |  45.07 |  47.63 |  9.8 GB |
| 4 | 71.3 |  56.07 |  58.43 |  56.01 |  58.37 | 13.1 GB |
| 5 | 72.3 |  69.15 |  72.05 |  69.09 |  71.99 | 16.4 GB |
| 6 | 68.5 |  87.59 |  91.62 |  87.53 |  91.54 | 19.7 GB |
| 7 | 67.0 | 104.50 | 108.80 | 104.44 | 108.74 | 23.0 GB |
| 8 | 63.2 | 126.61 | 131.51 | 126.55 | 131.45 | 26.3 GB |
| 9 | 67.5 | 133.19 | 176.00 | 133.13 | 175.94 | 29.5 GB |
| 10 | 67.2 | 148.60 | 205.73 | 148.53 | 205.67 | 32.8 GB |

结论：吞吐在 N=4–5 达峰（~72 QPS，单实例 31.5 的 2.3 倍），N≥6 饱和并
小幅回落（63–68）；e2e 从 N≥5 起随实例数近似线性上升，N≥9 出现明显 p99
长尾（排队抖动）；HBM 每实例 ~3.3 GB 线性增长。对照参考：单实例默认
（不限核）bench 为 ~25.2 ms，`12|24` 限核使单实例慢 ~25%。同为共享机器
（他人在跑多卡任务），绝对值受整机负载影响。
