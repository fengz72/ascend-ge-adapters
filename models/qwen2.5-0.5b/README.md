# Qwen2.5-0.5B 共享 prefix 极简流程

本目录是 [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) 的自包含交付：
每个部件一个自包含脚本、部件间零 import 依赖，只通过 `work/` 下的文件契约
交互，共用一个 3 输入动态 packed AIR（`input_ids` [T] / `position_ids` [T] /
`actual_seq_lengths` [N+1] → `scores` [N, 8]）。整个流程自包含于本目录。

## 1. 背景

### 1.1 官方地址

| 项目     | 地址                                          |
| -------- | --------------------------------------------- |
| 模型地址 | https://huggingface.co/Qwen/Qwen2.5-0.5B      |

### 1.2 业务约束

- **共享 prefix**：每个 case 只有一段 prefix 参与计算一次，其后跟随 16 条业务
  请求；prefix Q/KV 不重复计算。
- **固定 batch = 16**（`config/target_tokens.json` 之外的池参数固化在
  `model/generate_inputs.py`）。
- **输入池**：100 个 case，每 case 共享 prefix=20–25 token + batch16 请求；请求
  总长沿用历史 lognormal 分布（`mu=4.997`、`sigma=0.167`、上限 218）。
- **评分列**：每个业务请求取末 token 的 8 个固定 token 分数，列序由
  [config/target_tokens.json](config/target_tokens.json) 定义。
- **无 KV-cache**：一次前向完成 prefix + 全部请求。

目录中的五个文件契约部件：

| 部件                                                 | 输入                        | 输出（默认路径）                                                     |
| ---------------------------------------------------- | --------------------------- | -------------------------------------------------------------------- |
| `model/generate_inputs.py`                           | `QWEN_MODEL_DIR` 的 config.json | `work/inputs/cases.json` + `work/inputs/case-XXXX/*.npy`（schema `ge-cases/1`） |
| `model/golden.py`                                    | `work/inputs` + 权重目录    | `work/golden/<case>/output_0.npy` + `work/golden/result.json`        |
| `model/export_air.py`                                | 权重目录                    | `work/build/packed.air`                                              |
| `runtime/build/ge_execute`（源码 `runtime/`）        | `packed.air` + `work/inputs` | `work/output/<case>/output_*.npy` + `result.json`，bench 逐请求数据在同目录 `result.json.records` |
| `tools/compare.py`（复用）                           | golden 与 infer 输出        | `work/output/comparison.json`                                        |

另含 `run.sh`（一键串联）与 `scripts/prepare.sh`（依赖构建：平台 Prefix 算子 +
WeightNZ pass，并生成 `work/deps/env.sh`）；二者不属于文件契约部件。

边界与限制：

- 二进制依赖由 `scripts/prepare.sh <platform>` 从 `third_party/custom_development_code`
  子模块构建（本机从零约 53 s）：平台 Prefix Attention 算子装入
  `work/deps/prefix_attention/install/{python,opp}`，WeightNzAndMatMulV3Pass 装进
  活动 CANN 的 `vendors/custom_nz_pass/custom_fusion_passes/`，为可选性能 pass
  （缺省图仍可运行；生效证据是编译日志的 `matmul_match num is 169`）。
- 首次使用需初始化子模块：

  ```bash
  git submodule update --init third_party/custom_development_code
  ```

- `work/build/packed.air` 首次在新目录编译时按形状逐一 TBE 编译，首跑偏慢属预期。
- 全量跑一次的实测量级（本机 12 核、共享负载）：依赖构建 ~50 s（缓存命中）、CPU
  golden ~11 min（8 线程）、导出 ~2 min、ge_execute 编译+加载 ~7 s（TBE 缓存命中）、
  100 case 精度推理 ~2 s、比较 ~1 s。
- `model/generate_inputs.py` 需要 `QWEN_MODEL_DIR` 下的 `config.json`
  （仅读 `vocab_size`），不加载权重；`model/golden.py`、`model/export_air.py`
  需要完整权重。

## 2. 运行

### 2.1 环境信息

本机已验证组合（版本核对日期 2026-11-16）：

| 组件                        | 版本                                          |
| --------------------------- | --------------------------------------------- |
| CANN（Ascend-cann-toolkit） | 9.0.0（V100R001C10SPC001B250，aarch64-linux） |
| NPU 驱动（npu-smi）         | 25.5.t6.b001                                  |
| 硬件                        | Ascend910_9382                                |
| Python                      | 3.11.15                                       |
| torch                       | 2.9.0+cpu                                     |
| torch_npu                   | 2.9.0.post2                                   |
| transformers                | 5.10.1                                        |
| numpy                       | 1.26.4                                        |

### 2.2 运行脚本

```bash
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B    # 权重材料，不自动下载
bash run.sh --platform a3 --device 2           # --device 为 npu-smi 的 Chip Logic ID
```

`run.sh` 依次执行：环境检查与 `set_env.sh` → 依赖构建（`scripts/prepare.sh`）
→ 生成输入 → CPU golden → ge_execute 构建（`runtime/build.sh`）→ packed AIR
导出 → 精度推理 → `tools/compare.py` 门禁。

- `--platform a3|a5`：选择 Prefix 算子与 bounds dtype（默认 `a3`）。
- `--device <chip>`：`ASCEND_RT_VISIBLE_DEVICES` 使用 `npu-smi` 的 Chip Logic ID。
- 性能测试追加 `--iterations N`，可选 `--warmup M`、`--seed S`、
  `--instances K`、`--aicore-num A|B`：

  ```bash
  bash run.sh --platform a3 --device 2 --iterations 1000
  ```

  未给 `--instances` / `--aicore-num` 时按平台默认（a3：5 实例、`8|16`；
  a5：3 实例、`16|32`）。bench 读取完整输入池，不走精度文件。

### 2.3 单独执行某个部件

各部件的路径都是固定的（`work/` 下的文件契约），没有路径参数；单独执行只需
CANN 环境 + `QWEN_MODEL_DIR`，涉及 NPU 的步骤还要 `source work/deps/env.sh`：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B
export ASCEND_RT_VISIBLE_DEVICES=2                       # npu-smi 的 Chip Logic ID
# CANN 内嵌 Python 需要看到容器 numpy（run.sh 同样会做）
export PYTHONPATH="$(python3 -c 'import site; print(site.getsitepackages()[0])'):${PYTHONPATH:-}"

bash scripts/prepare.sh --platform a3 && source work/deps/env.sh
python3 model/generate_inputs.py --platform a3           # -> work/inputs
python3 model/golden.py                                  # -> work/golden（纯 CPU）
python3 model/export_air.py --platform a3                # -> work/build/packed.air
../../runtime/build/ge_execute --graph work/build/packed.air \
    --cases work/inputs/cases.json --output-dir work/output \
    --device 0 --precision-mode must_keep_origin_dtype
python3 ../../tools/compare.py --golden work/golden --output work/output \
    --cosine-min 0.998 --relative-l2-max 0.05
```

`--device 0` 是 ge_execute 的**进程内**设备号：导出 `ASCEND_RT_VISIBLE_DEVICES`
后可见设备从 0 开始，与上面 `--device <chip>` 的 Chip Logic ID 是两个概念。
`--precision-mode must_keep_origin_dtype` 必须显式给出：ge_execute 默认
`force_fp16`，不传则无法复现历史精度基线。

产物为 `work/build/packed.air`。导出使用 torchair `dynamo_export`，每个图输入
仅第 0 轴动态，因此一个 AIR 覆盖整个变长输入池。

## 3. 运行时

### 3.1 输入输出

| 张量                 | 方向 | dtype                          | shape   | 语义与约束                                                        |
| -------------------- | ---- | ------------------------------ | ------- | ----------------------------------------------------------------- |
| `input_ids`          | 输入 | int64                          | [T]     | packed token：先 prefix，再各请求私有 token；`0 ≤ id < vocab_size` |
| `position_ids`       | 输入 | int64                          | [T]     | prefix 自 0 起，每条请求自 `len(prefix)` 重新计数                  |
| `actual_seq_lengths` | 输入 | int64（a3）/ int32（a5）       | [N+1]   | 各段累计结束位置；`N` = 每 case 请求数（batch16 时为 16）         |
| `scores`             | 输出 | float32                        | [N, 8]  | 每个业务请求末 token 的 8 个评分列，列序见 `config/target_tokens.json` |

packed 布局为 `segments = [prefix] + requests`：
`input_ids = concat(segments)`；`position_ids` 的 prefix 段从 0 开始、每条请求从
`len(prefix)` 开始；`actual_seq_lengths = cumsum(段长)`，共 N+1 个值。业务行取
每条请求的末 token，即 `bounds[1:] - 1`；共享 prefix 只计算一次。A3 用
AscendC `PrefixInferAttentionScore`（int64 bounds），A5 用 CANNBotDSL
`PrefixAttention`（int32 bounds），两者输入张量数值一致。

输入经 `work/inputs/cases.json`（schema `ge-cases/1`）索引：每个 case 的三个
输入各自存一个 npy（`case-XXXX/{input_ids,position_ids,actual_seq_lengths}.npy`），
`inputs` 数组顺序即图输入顺序，dtype/shape 由 runtime 从 npy 头读取。infer 与
golden 产物同构：每个 case 一个 `<case_id>/output_0.npy`（`[N, 8]` float32），
外加 `result.json` 的 `cases` 索引。

### 3.2 性能数据

2026-11-16 本机实测：CANN 9.0.0、Ascend910_9382、Chip Logic ID 2、batch16、
100 组变长输入池。共享机器（他人在跑多卡任务），绝对值受整机负载影响。

图冷启动（不计入请求时延）：

| 阶段                                        |        耗时 |
| ------------------------------------------- | ----------: |
| CompileGraph（首份，TBE 冷缓存，不限核）    | 480103 ms |
| CompileGraph（首份，TBE 冷缓存，`8\|16`）   | 232853 ms |
| CompileGraph（每份，TBE 缓存命中）          |  4.7–7.0 s |
| LoadGraph（每实例）                         |  1.28–1.97 s |

`--aicore-num` 是编译期上限，改核数会触发一次重新编译；同进程内后续图实例命中
缓存。5 实例 bench 的 HBM 增量 4741 MB（948 MB/实例）。

精度推理（单实例、不限核、100 case 各一次）：首个 case 29.458 ms 含首次执行冷
开销，单独列出；其余 99 个 case 为稳态。

| 指标 |        ms |
| ---- | --------: |
| min  |   11.4034 |
| P50  |   12.0479 |
| avg  |   12.0863 |
| max  |   13.4046 |

bench（`--iterations 3000 --instances 5 --aicore-num 8|16`，每轮预热覆盖池内全部
distinct shape，seed 42/43/44 三轮，3×3000 请求全 0 错误，逐实例 QPS
均匀 27.327）：

| 指标                       |         值 |
| -------------------------- | ---------: |
| 业务 QPS（三轮中位）       |  2186.262 |
| 模型调用 QPS（三轮中位）   |   136.641 |
| e2e avg（中位）            |  36.5565 ms |
| e2e P50（中位）            |  36.2325 ms |
| e2e P99（中位／三轮最差）  |  40.7956 / 40.9928 ms |
| H2D / D2H avg              |  ~0.041 / ~0.026 ms |

e2e = H2D + 执行 + 同步 + D2H；H2D/D2H 合计 ~0.067 ms，可忽略。对照历史记录
（[验证记录](../../docs/qwen2.5-0.5b/validation.md) A3 节：同为 `8|16`、batch16、
5 实例，三轮各 10000 次调用）：业务 QPS 中位 2172.093、最差 P99 41.266 ms，
本次自包含流程与之一致。逐轮原始数据与逐请求记录（`result.json.records`）在
`work/bench*/`。

**迁移到 `ge_execute` 后的复测（2026-11-19，同一张卡、整机空闲）**：同配置三轮
业务 QPS 中位 2185.282、模型调用 QPS 中位 136.580（逐实例 27.316）、e2e avg 中位
36.5565 ms、P50 中位 36.2920 ms、P99 中位 40.8620 / 三轮最差 40.8900 ms，3×3000
请求全 0 错误；精度推理稳态 99 case min 11.5780 / P50 12.1460 / avg 12.2540 /
max 13.5660 ms（首 case 31.2240）。与上表逐项同分布，无回退。另修正一处口径：
池内 distinct shape 实为 **89** 个（100 个 case 里有重复的 T），不是 100；
`ge_execute` 的 `result.json.pool.distinct_shapes` 直接给出该值，每轮预热
445 次 = 5 实例 × 89。详见[验证记录](../../docs/qwen2.5-0.5b/validation.md)
的「ge_execute 迁移复测（2026-11-19）」节。

### 3.3 精度实测

门限由 `run.sh` 固定为逐行 `cosine ≥ 0.998`、逐行 `relative L2 ≤ 0.05`；
`tools/compare.py` 结果写入 `work/output/comparison.json`。

| 平台 | case / 样本 | 最小 cosine | 最大 relative L2 | 结果 |
| ---- | ----------: | ----------: | ---------------: | ---- |
| A3（Ascend910_9382） | 100 / 1600（全量） | 0.999439 | 0.033784 | PASS |
| A3，迁移到 `ge_execute` 后复测（2026-11-19） | 100 / 1600（全量） | 0.9994394615744572 | 0.033784056043048426 | PASS（与上一行逐位一致） |
| A5（Ascend950PR_958b） | 未实测（本机 CANN 9.0.0 无 A5 辅流接口） | — | — | — |

复测能逐位一致的前提是显式传 `--precision-mode must_keep_origin_dtype`：
`ge_execute` 默认 `force_fp16`，而旧 `executor` 把它硬编码在 session options 里。

对照历史记录：迁移到自包含流程前的旧框架路径 A3 全量 100 组/1600 样本的最小 cosine 为
0.998909550002、最大 relative L2 为 0.046848644966（评分头输出 FP16）；本流程
的评分头在 FP32 中计算，同一门限下余量明显变大（0.999439 / 0.033784）。

## 使用限制

- 面向标准全注意力、SiLU、默认 RoPE 的推理，不支持训练、HF `generate()`、
  KV-cache 或滑动窗口注意力。
- **A5** 需要 CANN ≥ 9.3.0 的辅流接口；本机（CANN 9.0.0）无法验证 A5 路径。
- 交付精度为 FP16；CPU golden 为 FP32 独立参考。
- 评分列固定为 `config/target_tokens.json` 的 8 个 token，不随输入改变。
- Prefix Attention 与 WeightNZAndMatMulV3Pass 来自
  `third_party/custom_development_code` 子模块的固定提交。

## 相关文档

- [适配说明](../../docs/qwen2.5-0.5b/adaptation.md)
- [维护记录](../../docs/qwen2.5-0.5b/maintenance.md)
- [验证记录](../../docs/qwen2.5-0.5b/validation.md)与[文档索引](../../docs/qwen2.5-0.5b/README.md)
