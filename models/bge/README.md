# BGE

## 是什么

将业务提供的 BGE ONNX 模型适配为 GE Session 推理。输入为八个业务特征：文本（样例 JSON 中的 `query_token_ids` 绑定到 ONNX `input_ids`）与 `attention_mask`、行为历史 `long_cid3_raw`/`long_action`/`long_time_bucket`、类目统计 `stat_dense_raw`/`stat_ratio`/`stat_count`（七个 INT64 加一个 FP32），输出 FP32 `scores[B,10112]`，batch 与文本/行为长度动态。保留原始图结构（30 个 Slice 全部不改写、零 Range），不使用自定义融合插件；外部浮点 IO 保持 FP32，内部 `force_fp16`。

输入池由业务样例 JSON 导入，不做随机生成。验收同时比对 CPU 全量分数（cosine、相对 L2、绝对误差）与业务 Top-10（ID、顺序及分数绝对误差）。需求、设计、交付报告与全部调优/分析档案见 [BGE 文档索引](../../docs/bge/README.md)。

## 怎么用

模型、样例与期望输出依赖仓库根的兄弟目录 `materials/`（`model.onnx`、`sample_inputs.json`、`expected_outputs.json`，以及 GE 加载用的 `exported/`）。本仓库不含这些文件：`data`/`baseline`/`build` 需要它们，`prepare`/`infer`/`compare`/`bench` 不需要。

历史七步复现（99 机器首次验证形态，基于提交 `59050e3`，当时 `source.ref` 为 `${BGE_MODEL_PATH}`）。在仓库根目录执行，路径与卡号按实际环境替换：

```bash
cd /path/to/ascend-ge-adapters
export BGE_MODEL_PATH=/path/to/model.onnx
cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build -j4
bash models/bge/run.sh run --device 0
```

各步骤可独立执行：`bash models/bge/run.sh data|baseline|prepare|build --device 0|infer --device 0|compare|bench --device 0`。当前配置 `source.ref` 已改为 `../materials/model.onnx`（100 机器布局：`runner/` 与 `materials/` 同级）。`run.sh` 将 CANN 内嵌 Python 与调用解释器对齐（PYTHONHOME 及其库目录），否则 GE 初始化会报 `No module named numpy`。

100 机器的正式性能测量（固定/随机双场景、含全量 D2H 口径）使用项目根受管入口 `tools/run_one.py`：它自建计划、核对模型与配置身份、锁定物理设备并在运行前后检查占用。计划准备用 `models/bge/replay.py`（`--prepare-only` 为唯一模式，不占设备；`--case-id` 选固定场景，省略为五例等概率随机）。完整命令、统计口径与身份门禁见[交付报告第 4 节](../../docs/bge/delivery-100.md#4-交付件与复现)。

回归测试：

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
ctest --test-dir runtime/build --output-on-failure
```

五个 opt-in 设备测试（设置 `GE_TEST_DEVICE` 后运行 `tests/test_ge_session_outputs.py`）覆盖静态 rank-1 Slice、多实例复跑、混合 float32/int64 输出、动态 batch、mode-1 兼容及输出容量不足。

## 性能表现

当前默认配置（原始 ONNX、GE Session 预测模式 `graph_run_mode=0`、内部 `force_fp16`、10 实例、每算子核上限 `12|24`）在 100 机器逻辑设备 2 的实测。每格为三轮 60 秒正式窗口中位数；计时含 H2D、描述符、GE 完成同步与全量 D2H，18 轮均零请求错误：

| 实例数 | 固定 case QPS | 固定 p99 (ms) | 五例随机 QPS | 随机 p99 (ms) |
|---:|---:|---:|---:|---:|
| 8 | 807.326 | 19.718 | 861.592 | 13.505 |
| **10（当前）** | **884.177** | **14.262** | **903.582** | 19.285 |
| 12 | 851.169 | 33.053 | 862.495 | 35.958 |

固定场景 10 实例相对 8 实例 +9.519%（超过本批 6.670% 波动门槛）；随机场景 +4.874%，低于 19.405% 门槛，收益未证实。上表绑定二进制 SHA256 `8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d`。2026-10-28 dtype 别名修复后的新二进制（`735a3532ecb131de203200a328835af0d58bcddf27f9572eb997b5d6808e43f4`）已通过五例 22,274 请求全量输出校验，稳定吞吐尚待独立三轮复测。七步 `bench` 口径（`h2d_desc_execute_sync`）不含逐请求 D2H，与本表不可比。10 实例配置的五例全量输出（2009 请求）与 600 秒稳定性验收见[十实例报告](../../docs/bge/analysis/100-fp16-ten-selection-20261027.md)，逐轮数据见[复测报告](../../docs/bge/analysis/100-fp16-scale-recompare-20261027.md)与[修复报告](../../docs/bge/analysis/100-runner-review-fix-20261028.md)。

## 使用限制

- 仅覆盖给定五条 batch=1 样例（文本长 6/9/8/13/8、行为长 128、统计宽 20）；`dense_id = 输出下标 + 1`。未设置业务 p99 上限，不声称满足生产 SLA。
- CANN 9.0 下 `graph_run_mode=1`（训练模式）会使原始图在 `fe::ConstToAttrStridedSliceV2Pass` 内 SIGSEGV，必须使用 mode=0。ATC/OM 部署未验证。
- 10 实例为用户指定配置，不声称实例数或随机场景吞吐最优；12 实例未完成独立全量输出与稳定性验收。
- 更换实例数、精度策略、图或融合插件需重新执行全量输出正确性与稳定性验收；`--capture` 压力检查与正式 QPS 轮分开，性能路径不采集逐请求输出。
- 更换输入需重跑 `data`/`baseline`/`infer`；仅更换期望 JSON 或分数阈值只需重跑 `compare`，它直接读取当前文件，不依赖原始模型、输入 JSON 或 NPU。

## Appendix: Experimental FIAS and precompute (2026-10-29 / 30)

The default YAML still runs the original ONNX without a custom pass. These
optional GE passes and their host tests are now in this runner. Build them
inside the CANN 9.0.0 container:

    bash tools/build_bge_experimental_passes.sh

The normal data, baseline and build stages must first create the input pool,
CPU golden and exported ONNX in work/bge. For a fresh checkout, run those
stages through models/bge/run.sh with the same chosen device and target.
After confirming an idle, exclusive physical NPU, run five-case accuracy on
the chosen logical device ID (set DEVICE_ID to that integer):

    bash models/bge/run_experimental.sh fias infer --device "$DEVICE_ID" --case-ids all
    bash models/bge/run.sh compare --case-ids all
    bash models/bge/run_experimental.sh precompute infer --device "$DEVICE_ID" --case-ids all
    bash models/bge/run.sh compare --case-ids all

The pass environment applies to the native GE child only. The seven-stage
bench omits per-request D2H and is not comparable with the full-transfer
figures below. These experiments ran in the earlier isolated checkout, using
GE mode 0, force_fp16, 10 instances, 12|24 cores, 60-second no-profiling
rounds, and H2D plus full D2H. The new runner build has not been independently
rebenchmarked.

| Optimization / workload | Before QPS / avg / p99 ms | After QPS / avg / p99 ms | QPS change | Decision |
|---|---:|---:|---:|---|
| OPT-018 first-layer FIAS, fixed case0, 30s exploration | 875.043 / n.a. / 16.102 | 873.593 or 913.729 / n.a. / 26.885 or 15.482 | -0.17% or +4.42% | No stable benefit |
| OPT-019 all 24 FIAS, fixed case0 round 1 | 865.5 / n.a. / 29.10 | 1060.4 / n.a. / 12.78 | +22.5% | Accepted in isolated experiment |
| OPT-019 all 24 FIAS, fixed case0 round 2 | 878.7 / n.a. / 25.06 | 1074.6 / n.a. / 12.45 | +22.3% | Accepted in isolated experiment |
| OPT-019 all 24 FIAS, five-case random round 1 | 782.1 / n.a. / 27.64 | 1049.6 / n.a. / 12.92 | +34.2% | Accepted in isolated experiment |
| OPT-019 all 24 FIAS, five-case random round 2 | 741.1 / n.a. / 36.90 | 1054.7 / n.a. / 12.81 | +42.3% | Accepted in isolated experiment |
| OPT-020 structural matching | Same FIAS behavior | Five-case outputs identical | Not independently measured | Accepted as matcher refactor |
| OPT-021 precompute, fixed case0 pair 1 | 1107.576 / 9.027 / 12.076 | 1190.210 / 8.400 / 11.554 | +7.461% | Accepted in isolated experiment |
| OPT-021 precompute, fixed case0 pair 2 | 1070.228 / 9.342 / 12.665 | 1200.948 / 8.325 / 13.791 | +12.214% | Accepted in isolated experiment |
| OPT-021 two-pair pooled | 1088.902 / 9.182 / 12.441 | 1195.579 / 8.362 / 12.471 | +9.797% | Fixed case0 only; pooled p99 +0.241% |

OPT-021 compares FIAS-only with FIAS plus precomputation. It is not an
additive gain over OPT-019's original-model comparison. Both pairs used
opposite A/B order and had zero errors. Five-case full-output and expected
Top10 checks passed. No random-case performance test or new-runner performance
acceptance has been completed for OPT-021. Full records:
[OPT-018](../../docs/bge/experiments/OPT-018.md),
[OPT-019](../../docs/bge/experiments/OPT-019.md),
[OPT-020](../../docs/bge/experiments/OPT-020.md),
[OPT-021](../../docs/bge/experiments/OPT-021.md), and
[case0 summary](../../docs/bge/evidence/20261030-precompute-case0-summary.json).
The earlier [GE graph and profiling analysis](../../docs/bge/analysis/fias-fp16-graph-profile-20261029.md)
explains the candidate constant-chain cost. No post-precompute final graph or
profiling capture has yet confirmed which kernels disappeared.
