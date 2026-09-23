# Qwen2.5-0.5B Ascend NPU 部署指南

> 当前架构（`core/` + `runtime/`）的部署说明。历史版本（v1.0 / v2.0，基于已退役的 `atb/` 工具链）
> 见 git 历史；实测报告归档在 `docs/reports/`、`docs/aicore/`、`docs/prefix-attention/`。

## 1. 概述

Qwen2.5-0.5B 以 varlen packed（变长拼接）形态部署到 Ascend NPU：

```
PyTorch (NPU 融合算子) → AIR (torchair dynamo_export) → OM (ATC) → ACL 执行
                        └──────────────────────────────────────→ GeSession 在线执行 (免 ATC)
```

**核心特性**
- 2D TND：hidden 全程 `[T, D]` 无 batch 维（消除 Pack 算子）
- NPU 融合算子：`npu_rms_norm` + `npu_apply_rotary_pos_emb`(TND/half) + `npu_infer_attention_score`
- RoPE cos/sin 图外预计算成图常量（`[1, max_seq_len, 64]`），图内只留 2 个 Gather
- 权重经 `frozen_parameter=1` 冻结进图（AIR ~1GB，OM ~1.4GB）
- 动态 shape：T（总 token 数）与 N（请求数）均为动态维；**同一 OM 可跨 shape 复用**（实测 T=32 导出的 OM 直接跑 T=2080）
- 可选：NZ 权重 pass（`WeightNzAndMatMulV3Pass`）、lm_head 词表剪裁、prefix-attention（PIA）变体、限核

## 2. 环境

| 项目 | 版本 |
|---|---|
| SoC | Ascend910_9382 |
| CANN Toolkit | 9.0.0 |
| Python | 3.11.15 |
| PyTorch / torch_npu | 2.9.0 / 2.9.0.post2 |
| transformers | 5.10.1 |
| 镜像 | `quay.io/jd_xllm/xllm-ai:xllm-dev-a3-arm-cann9-20260605` |

环境初始化由 `env.sh` 负责（`run.sh` 会自动 source）：CANN `set_env.sh` + PIA 算子 vendor +
把本机 site-packages 注入 `PYTHONPATH`（GE 的 tbe pywrapper 用内嵌 python3，缺 numpy 会让
`GEInitialize` 返回 -1）。

## 3. 快速开始

```bash
cd <repo>
bash runtime/build.sh                       # 一次性: 构建 C++ 运行时 → runtime/build/ge_runtime

# 全链路 (source→adapt→golden→AIR→io_spec→bundle→ATC→OM→manifest→run→compare)
./models/qwen2.5-0.5b/run.sh --device 6 --batch-size 2 --seq-len 16

# 默认配置 (batch=10, seq=208)
./models/qwen2.5-0.5b/run.sh --device 6
```

> **pass / 自定义算子怎么装的**：`model.yaml` 的 `passes:` 与 `custom_ops:` 填的是**用户脚本路径**
> （`scripts/install_nz_pass.sh`、`scripts/install_prefix_attn.sh`），管线分别在 ATC 编译前、
> 加载 adapter 前执行（`core/setup_scripts.py`）。脚本幂等：已装即跳过；检测到别的 vendor 已注册
> 同名 pass 也跳过并复用那份（同名重复注册会让 ATC/TBE 崩，docs §7.1）。所以**不需要** `--skip passes`；
> 确实想跳过时用 `--skip ops,passes`。

产物（默认落在 `models/qwen2.5-0.5b/`，`--work-dir` 可改；后续 `--skip export` 必须带同一 `--work-dir`）：

```
air/qwen2.5-0.5b.air  air/qwen2.5-0.5b.io_spec.json  air/dynamo.pbtxt
om/qwen2.5-0.5b_linux_aarch64.om
verification/{bundle.json, inputs/*.bin, golden_logits.bin, outputs/{output_0.bin, outputs.json}}
deploy/manifest.json
```

## 4. 模型 I/O 规格

**图输入序 ≠ forward 入参序**（torchair 的 Data index 序由导出决定，`_source_name` 属性给出真名，
`core/graph.py` 自动配对，io_spec 按图序落盘）：

| 图序 | 图节点 | 语义 | Shape | Dtype | 动态维 |
|---|---|---|---|---|---|
| 0 | `arg1_1` | actual_seq_lengths（累积长度） | `[N]` | int64 | dim0 |
| 1 | `arg4_1` | input_ids（所有序列 token 拼接） | `[T]` | int64 | dim0 |
| 2 | `arg7_1` | position_ids | `[T]` | int64 | dim0 |

输出：`logits [N, 151936] float16`（每条序列末 token）；启用词表剪裁后为 `[N, len(target_tokens)]`。

`bundle.json` 按 **forward 入参序**（input_ids, position_ids, actual_seq_lengths）记录具体 shape
与 .bin；C++ 运行时按 **logical 名**与 io_spec 配对，不按位置。

## 5. 两种执行后端

| 后端 | 配置 | 特点 |
|---|---|---|
| OM/ACL（离线） | `backend.type: om_acl` | 需 ATC（~8min），产物可移植，运行期开销低 |
| GeSession（在线） | `backend.type: ge_session` | 免 ATC，直接加载 AIR；`CompileGraph` ~10s/图实例，首次执行含 shape 特化 ~250ms |

切换后端只改 `config/model.yaml` 的 `backend.type`，复用已有 AIR/bundle：

```bash
./models/qwen2.5-0.5b/run.sh --device 6 --skip export,passes,compile
```

绕开 Python 直接调 C++（部署态可用 `--input logical:shape:file` 替代 bundle）：

```bash
./runtime/build/ge_runtime models/qwen2.5-0.5b/deploy/manifest.json --device 6 --bench 20
```

## 6. 性能与精度

延迟 / 吞吐（`--warmup`、`--bench` 为延迟口径；`--threads`、`--requests`、`--sweep` 为吞吐口径）：

```bash
# 延迟
./models/qwen2.5-0.5b/run.sh --device 6 --skip export,passes,compile --warmup 10 --bench 100
# 吞吐 (线程档扫描; 1 worker ↔ 1 实例)
./runtime/build/ge_runtime <manifest> --device 6 --sweep 1,2,4,8 --requests 800 --warmup 10
# 变长负载 (多实例 + 请求池回放, 出归档报告)
python3 -m core.bench --scenario models/qwen2.5-0.5b/bench/varlen.yaml --device 6
#   → models/qwen2.5-0.5b/results/<run_id>/{run,perf,accuracy}.json + perf.md/accuracy.md
#   请求池由 scenario 的 generate 脚本按 lognormal 分布生成 (幂等, 见 scripts/gen_requests.py)
# 限核 (在线后端)
./models/qwen2.5-0.5b/run.sh --device 6 --skip export,passes,compile --runtime-opt --aicore_num=12
```

实测（batch=2 / seq=16 → T=32, N=2，device 6）：

| 指标 | OM/ACL | GeSession |
|---|---|---|
| 延迟 p50 | 3.25 ms | 3.01 ms |
| QPS（1/2/4 线程，各 60 请求） | 308 / 364 / 362 | 318 / 371 / 391 |
| 精度 vs NPU-eager golden | cosine 0.99995719，rel_l2 9.2e-3 | 同左（输出逐字节一致） |

判定门限：cosine > 0.9999 且 relative_l2 < 0.01（`tools/compare.py`）；shape 不一致直接判失败，
不做 flatten/截断兜底。大 batch / prefix / 限核的历史数据见 `docs/reports/`、`docs/aicore/`、
`docs/prefix-attention/`；`core.bench` 的归档报告见 `results/`（含两后端同负载对比样例）。

变长负载实测（2 实例、lognormal 池 200 套 / 138 种 shape、batch=10、T≈1273–1694，device 8）：

| | om_acl | ge_session |
|---|---|---|
| QPS | 107.7 | 107.2 |
| e2e avg / p99 | 18.54 / 20.20 ms | 18.59 / 20.41 ms |
| HBM 每实例 | ~1212 MB | **~604 MB** |
| 启动每实例 | 2.9 s | **22.2 s**（含每份图 ~10 s 的 CompileGraph） |
| warmup 特化（138 shape） | 1.83 s | 1.73 s |

## 7. Profiling 与 dump

```bash
# OM/ACL: 生成 acl.json 交给 aclInit → PROF_* 会话
./runtime/build/ge_runtime <manifest> --device 6 --bench 20 \
    --profiling --profiling_output ./prof_om --profiling_aic_metrics PipeUtilization
# GeSession: 经 GEInitialize 全局选项开启 (同一组参数)
./runtime/build/ge_runtime <manifest> --device 6 --bench 20 --profiling --profiling_output ./prof_ge
# 解析
python3 tools/parse_profiling.py parse-and-export --profiling_dir ./prof_om
python3 tools/parse_profiling.py summary --profiling_dir ./prof_om
# 逐算子数据 dump (仅 OM/ACL)
./runtime/build/ge_runtime <manifest> --device 6 --dump --dump_path ./dump_om --dump_mode output
python3 tools/parse_dump.py ...   # 见 tools/README.md
```

## 8. 可选变体（都在 `config/model.yaml`）

| 变体 | 配置 | 说明 |
|---|---|---|
| 词表剪裁 | `adapt.params.prune_token_file: config/target_tokens.json` | lm_head 输出维 151936 → N，减少 D2H |
| prefix-attention | 用 `config/model.prefix.yaml`（`adapt.params.prefix: true` + `inputs.prefix_len: 20`） | PIA 算子（`npu_prefix_infer_attention_score`，KV 内嵌），由 `custom_ops` 脚本装到 `opp/vendors/custom_prefix_attn/`；导出名自动变 `qwen2.5-0.5b-prefix`，产物与基线互不覆盖 |
| 限核（离线） | `backend.aicore_num: 12`（整数按 1:2 拆成 `12|24`） | OM 文件名带 `_c12_24` 后缀 |
| 限核（在线） | `--runtime-opt --aicore_num=12` | 经 `GEInitialize` 注入 |
| 图常量长度 | `graph.dynamic.max_seq_len: 2048` | RoPE 表与因果 mask 长度同源；决定 position_ids 上限 |

## 9. 已知限制

- **单输出契约**：bundle 只记一个 golden（`golden_logits.bin`），compare 只比一个输出 —— 当前
  契约假设 CausalLM 单 logits 输出（docs §5.4）。
- **随机 varlen 负载生成未移植**：旧 `atb/bench_latency.cpp` 的 RequestGenerator（对数正态序列
  长度分布 + 闭环随机请求）随 `atb/` 退役，需要时从 git 历史取；通用替代是用 `tools/varlen.py`
  生成多组 bundle，逐组跑 `ge_runtime`。
- **pass 全局唯一**：fusion pass 无 per-model 隔离，同名 pass 只能有一份；安装脚本已做检测（§3）。
- **NPU 显存**：export 阶段约需 1.2GB（0.5B fp16 + 上下文）；卡被占满时进程会被 SIGKILL（exit 137）。
