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

两个参数是旧 `executor` 没有的，漏传的后果不同：

- `--device` 是**必填**项，指进程内设备号。`ASCEND_RT_VISIBLE_DEVICES` 映射后可见
  设备从 0 开始编号，所以恒为 0，与上面的 Chip Logic ID 不是一回事（`run.sh`
  里分别是 `$DEVICE` 与 `$DEVICE_CHIP`）。漏传直接 exit 2。
- `--precision-mode` 默认是 `force_fp16`，而旧 `executor` 把
  `must_keep_origin_dtype` 硬编码在 session options 里。漏传**不会报错**，但
  下面 §3.3 的精度基线不可复现。

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

下表是 2026-11-13 旧 `executor` 的记录，保留作对照基线；迁移到 `ge_execute`
后的复测见 §3.4，冷启动两项与首 case 时延均在同一量级。

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

> **口径已变（2026-11-19 迁移到 `ge_execute` 后）**：`ge_execute` 不提供
> `--readout`，CPU decider 计时能力已移除，`run.sh` 也不再接受该参数。e2e
> 现在是 `h2d + execute + sync + d2h`（`result.json.timed_phases` 明示），
> 比下表少一个 readout 项。该项实测 ~0.001 ms，占单实例 e2e（31.70 ms）的
> 0.004%，低于共享机器的噪声底（本节逐轮 QPS 极差就有 ≤3.0），因此下表的
> e2e 列在噪声内仍可比。另外 `requests.csv` 不再产出，逐请求数据改在
> `work/bench/result.json` 的 `.records` 数组（字段 `instance` / `request` /
> `case_id` / `shape_key` / `samples` / `h2d_ms` / `exec_ms` / `d2h_ms` /
> `e2e_ms` / `first_hit` / `error`）。
>
> `readout.json` 仍由 `generate_inputs.py` 产出（labels / letters_asked /
> temperature），但当前没有任何消费者。

2026-11-16 容器（CANN 9.1.0，Ascend910_9382）逻辑芯片 6，executor 多实例闭环
bench：固定 `dev-intent-q00-r0`（S=1001）、每算子核上限 `12|24`、CPU readout
decider 计入 e2e。每档 3 轮（seed 42/43/44），每实例预热 32 请求后测
2000 请求；iterations 为全实例合计（=2000×N）。30 轮全部 0 错误，逐轮
QPS 极差 ≤3.0。e2e = H2D+执行+同步+D2H+readout；exec 仅含执行+同步
（h2d/d2h 各 ~0.026 ms、decider ~0.001 ms，均可忽略）。逐轮原始数据与
requests.csv 在 `work/bench-scaling/inst<N>-r<r>/`。

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

### 3.4 附录：迁移到 ge_execute 的验证记录（2026-11-19）

`28389ba` 把三个旧运行时收敛成单个 `ge_execute`，`f210ed2` 迁移了
`models/qwen2.5-0.5b`，本次按同一模式迁移本目录。三处契约变化：
`ge-records/3`（`inputs.bin` + 带 offset/bytes 的 `records.json`）→
`ge-cases/1`（`cases.json` + 每 case 独立 npy）；infer 产物
`outputs.bin` + `result.json.outputs` → `<case_id>/output_0.npy` +
`result.json.cases`；`requests.csv` 不再产出。`--readout` 无对应实现，已按
§3.3 的口径说明移除。

改动范围（算法与数值路径未动）：

| 文件 | 改动 | 未动的部分 |
| ---- | ---- | ---------- |
| `model/generate_inputs.py` | 落盘改 `cases.json` + `<case_id>/{input_ids,position_ids}.npy`；删 `inputs.bin`/`records.json` 与 `OUTPUT_NAME`；新增 `INPUT_NAMES` | `_load_dev_rows()` 的 dev.txt 业务校验、case_id 编号、prompt 渲染与 tokenize、三处 `ValueError`、`readout.json`、源文件 sha256 摘要 |
| `model/golden.py` | `_load_cases()` 改读 `cases.json` + npy；产物改 `<case_id>/output_0.npy` + `cases` 索引；`OUTPUT_NAME` → `output_0`；删掉 `ge-records/3` 的 schema 断言 | 打分数学（`bidirectional.text_forward` → `hidden[0,-1]` → `@ head_rows.T` → float32 `[1,26]`）、`letter_ids()`、线程 pin |
| `run.sh` | infer/bench 换 `ge_execute` + `--graph`/`--cases`，补 `--device "$DEVICE"` 与 `--precision-mode must_keep_origin_dtype`；删 `--readout`/`READOUT` | 环境检查、`prepare.sh`、`export_air.py`、`compare.py` 调用、`LD_PRELOAD`、精度门限 |
| `model/export_air.py`、`scripts/prepare.sh`、`native_proto/`、`tools/compare.py`、`runtime/` | 无改动 | — |

实测（本机 Chip Logic ID 1，CANN 9.1.0，Ascend910_9382，Python 3.12.13 /
numpy 2.5.1 / torch 2.12.0+cpu / torch_npu 2.12.0 / transformers 5.8.1，与
§2.1 同一组合；共享机器，16 张卡 HBM 均已被他人占用 44–55 GB，AICore 0%）：

| 项目 | 命令要点 | 结果 |
| ---- | -------- | ---- |
| runtime 重建 | `rm -rf runtime/build && bash runtime/build.sh` | 只产出 `ge_execute`（404320 B）与 `runtime_contracts`（464000 B），旧 `executor`/`ge_runtime`/`pipeline_runtime` 残留已随目录删除不再生 |
| CPU 契约测试 | `ctest --output-on-failure` | 1/1 Passed（npy 读写与拒绝路径、cases 解析、抽样/预热计划、统计与百分位、JSON 落盘） |
| CLI 冒烟 | `ge_execute --help` | exit 0，usage 与新参数表一致 |
| AIR 真机 infer | 沿用 2026-11-13 的 `work/build/text.air`（1507491002 B，未重跑 `export_air.py`）；1 case、S=1001、`input_ids` 取 `[4,999]` 的随机 int64；`--device 0 --precision-mode must_keep_origin_dtype`，`LD_PRELOAD` 注入 chunk proto | exit 0；`allocation=GE (dynamic compiled)`；`CompileGraph ok (gid=1/1) cost 17449.5 ms`、`LoadGraph ok (gid=1) cost 3841.2 ms, 2 input buffer(s)`（对照 §3.2 的 17460.2 / 3676.36 ms）；融合 pass 生效 `matmul_match num is 187 / matmul_effect num is 186`；产出 `smoke-0000/output_0.npy` **float32 [1,26] 104 bytes**，C-contiguous、全有限、argmax=5；e2e 587.983 ms（首 case 冷启动，对照 §3.2 的 614.693 ms）、d2h avg 0.039 ms |
| `result.json` 结构 | 同上 | `mode=infer`、`status=success`、`timed_phases=[h2d,execute,sync,d2h]`、`pool={cases:1,distinct_shapes:1}`、`config.precision_mode=must_keep_origin_dtype`；`cases[0].outputs[0]` = `{name:output_0, dtype:float32, shape:[1,26], bytes:104, path:...}`，`path` 为 CWD 相对（`--output-dir` 传的是相对路径） |
| compare 管路 | 复刻 `golden.py` 迁移后的写盘逻辑造参考目录（相对 `result.json` 的 path + `cases` 索引），`tools/compare.py --cosine-min 0.999 --relative-l2-max 0.02` | exit 0，`compare: 1 cases, worst cosine 1.000000, worst relative L2 0.000000`，PASS；`comparison.json` 为 `ge-compare/1`、`rows=1`、输出键 `output_0`。**注意这是管路验证不是精度验证**——参考值取自 infer 输出自身，cosine 恒为 1.0 只证明 `compare.py` 能按 `cases` 键分派到 npy loader、能解析两种 path 拼法、能按 `output_0` 对齐、`rows` 取自 `[1,26]` 的 `shape[0]` |
| Python 回归 | `python3 -m unittest discover -s tests -p 'test_*.py'` | Ran 154 tests，OK（skipped=5），无 failure/error |

**未验证范围**（阻塞：本机 `STARTLUX_MODEL_DIR` 未设置，`generate_inputs.py`
需要 tokenizer、`golden.py` 需要 FP32 权重）：

- **精度门限（cosine ≥ 0.999 / rel-L2 ≤ 0.02）尚未在 `ge_execute` 上实测**。
  历史基线是旧 executor 的 `work/output/comparison.json`：18 case / 18 row
  全 PASS，worst cosine `0.9999877883685806`、worst rel-L2
  `0.005193270205303367`。补跑 `bash run.sh` 后应与之逐位吻合；吻合即证明
  迁移数值中性（qwen 迁移时同样逐位一致）。
- 18 case / 5 种 distinct shape（S ∈ 1001..1005）的真实池未跑过，冒烟只有
  1 case / 1 shape。图 S 轴动态（`export_air.py` 的 `dynamic_dims`），不触发
  `ge_execute` 的「静态图 + 多 shape」护栏；bench `--warmup 16` ≥ 5，预热会
  自动覆盖全部 distinct shape。
- bench 路径（`--iterations`）未跑，§3.3 的并发扩展表未在 `ge_execute` 上复测。
- AIR 未重新导出，沿用 2026-11-13 产物；`prepare.sh` 未重跑，沿用
  `work/deps/` 里 2026-11-13 构建的 chunk proto `.so`。

补跑后请把结果追加到本节表格，不要改写已有行。冒烟产物在
`work/smoke-ge/`（Git 忽略）。

