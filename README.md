# Ascend GE model adapters

每个模型一个自包含交付目录，入口是目录内的 `run.sh`。共用的执行能力由 `runtime/` 的单个二进制 `ge_execute` 提供，离线分析由 `tools/` 提供。没有跨模型的编排框架、没有模型 YAML、没有统一命令入口；数据生成、CPU golden、图导出、精度门禁和业务门禁都在各模型目录内，部件间零 import，只通过各自 `work/` 下的文件契约交互。

## BGE 当前适配交付（100 机器）

当前用户指定配置是原始 ONNX、GE Session mode=0、内部 `force_fp16`、10 实例、每算子核上限 `12|24`；外部浮点输入输出保持 FP32。请从 [BGE 文件索引](docs/bge/README.md) 和 [交付报告](docs/bge/delivery-100.md) 开始。100 项目根的 `materials/` 与 `work/` 分别保存大模型和原始作业证据。

2026-10-27 在 100 机器逻辑设备 2 上，无 profiling、包含 H2D/GE 同步/全量 D2H 的正式三轮复测：固定 `case-0000` 的 8/10/12 实例 QPS 中位数为 **807.326 / 884.177 / 851.169**；五例随机回放为 **861.592 / 903.582 / 862.495**。10 实例固定场景比 8 实例高 9.519%，超过本批波动门槛；随机场景 10 实例有一轮仅 730.640 QPS，不能确认吞吐收益。12 实例没有稳定优势，维持用户指定的 10 实例。逐轮 QPS、p99、资源与门禁见 [8/10/12 实例复测](docs/bge/analysis/100-fp16-scale-recompare-20261027.md)；模型入口说明见 [BGE README](models/bge/README.md)。

2026-10-28 修复输出 dtype 别名与材料/交付路径后重建了 runtime。上述三轮中位数属于旧二进制；新构建已通过五例全量输出及固定/随机各一轮性能检查，稳定吞吐尚待三轮复核。证据和两个二进制的 SHA256 见 [runner 修复报告](docs/bge/analysis/100-runner-review-fix-20261028.md)。

## 模型交付

| 目录 | 图执行后端 | 说明 |
|---|---|---|
| [`models/bge`](models/bge/README.md) | `ge_execute`，原始 ONNX 直喂 | 业务原始导出经 `aclgrphParseONNX` 直接加载，无打包图中间步骤 |
| [`models/qwen2.5-0.5b`](models/qwen2.5-0.5b/README.md) | `ge_execute`，3 输入动态 packed AIR | 适配、验证、调优与 profiling 见[文档索引](docs/qwen2.5-0.5b/README.md) |
| [`models/startlux-decision-v1.1-0.8b-instant-text`](models/startlux-decision-v1.1-0.8b-instant-text/README.md) | `ge_execute`，单图 AIR | 文本路 |
| [`models/onejev1.6-0.8b`](models/onejev1.6-0.8b/README.md) | **无可执行后端** | `run.sh` 调 `runtime/build/executor`，bench 还需要 `--readout`；`ge_execute` 两者都不提供 |
| [`models/startlux-decision-v1.1-0.8b-instant`](models/startlux-decision-v1.1-0.8b-instant/README.md) | **无可执行后端** | 两阶段 vision → text，`run.sh` 调 `runtime/build/pipeline_runtime`；`ge_execute` 只支持单图，需要先定方案 |

已验证平台、精度与性能数字、使用限制写在各自的 README 里，本页不汇总，也不代替各模型的验证范围声明。

## 环境

在 CANN 容器的项目根目录执行，使用容器 Python：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
python3 -m pip install -r requirements.txt
bash runtime/build.sh
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p work/tmp
export TMPDIR="$PWD/work/tmp"
```

各模型的 `run.sh` 自行加载 CANN 与算子环境、自行构建 `ge_execute`、自行安装本模型需要的算子和 pass；上面的 `runtime/build.sh` 只在需要单独重建运行时时用。

`TMPDIR` 必须短：GEInitialize 起的 `multiprocessing.Manager` 在 `$TMPDIR/pymp-*/listener-*` 上 bind AF_UNIX socket，`sun_path` 上限 107 字节。CANN 原生进程内嵌 Python，必须与调用方解释器对齐，`run.sh` 会导出 `PYTHONHOME` 与 `LIBDIR`。

## 独立运行时 ge_execute

`runtime/` 只有一个二进制 `ge_execute`：加载一张 AIR 或 ONNX 图，用 GE Session 按 `cases.json`（schema `ge-cases/1`）描述的 npy 输入跑 infer 或 bench；实例数、warmup、运行次数、限核、profiling 全部由命令行指定。模型侧只提供图、`cases.json` 和 npy，不需要 io 契约文件。用法、`cases.json` 规范、计时口径、限制与验证记录见 [runtime 说明](runtime/README.md)。

```bash
bash runtime/build.sh
runtime/build/ge_execute --graph work/build/model.onnx --cases work/inputs/cases.json \
    --output-dir work/bench --device 0 --instances 10 --warmup 1 --iterations 1000 --aicore-num 12|24
```

`--device` 是必填的进程内设备号，`ASCEND_RT_VISIBLE_DEVICES` 映射后恒为 0。`--iterations N` 在所有实例间合计，`N>0` 即 bench、`0` 即 infer；`--instances K` 编译 K 份图并串行 AddGraph+CompileGraph，启动时间随 K 线性增长。`--aicore-num A|B` 是编译期每算子核上限，不预留、不隔离、不等于绑核。`--graph-run-mode` 默认 0（推理模式），mode=1 在 CANN 9.0 上 SIGSEGV。

`--precision-mode` 默认 `force_fp16`。已实测的精度基线是在 `must_keep_origin_dtype` 下取得的，复现这些数字必须显式传参；`models/qwen2.5-0.5b` 与 `models/startlux-decision-v1.1-0.8b-instant-text` 的 `run.sh` 已经显式传了。

`ge_execute` 不提供：ATC/OM 后端、两阶段 pipeline、stdin plan 协议、readout decider 阶段、io 契约校验。上表两个「无可执行后端」的模型就是被这些缺口挡住的。

## 依赖源码

两个子模块都在实际使用中，不是历史遗留：

- `third_party/custom_development_code` 跟踪[上游仓库](https://gitcode.com/CharmanderFang/custom_development_code)的 `main` 分支。提供 Prefix Attention 算子（A3/A5 源码在 `custom_op/prefix-attention/`，使用说明见[算子文档](third_party/custom_development_code/custom_op/prefix-attention/README.md)）、`fusion_pass/WeightNzAndMatMulV3Pass`（`models/qwen2.5-0.5b` A5 与 `models/startlux-decision-v1.1-0.8b-instant-text` 使用）与 `fusion_pass/BgeFusedInferAttentionPass`（`models/bge` 使用）。
- `third_party/ascend-ops` 提供 `causal-conv1d` 与 `mega-gdn-prefill`，由 `models/startlux-decision-v1.1-0.8b-instant/prepare.sh` 安装。

依赖由各模型自己的构建脚本安装，安装位置和方式见各模型 README。以 `models/qwen2.5-0.5b/scripts/prepare.sh` 为例：Prefix 算子装进模型目录的 `work/deps/`，NZ Pass 装进 `${ASCEND_OPP_PATH}/vendors/custom_nz_pass/custom_fusion_passes/`（由 GE 自动加载），并生成 `work/deps/env.sh` 供 `run.sh` source。

## 目录

```text
models/       每个模型一个自包含交付：数据脚本、导出、门禁、run.sh、原始输入
runtime/      ge_execute: GE Session 单图执行 (AIR/ONNX)、并发实例、限核、测量、采集
tools/        profiling/dump 离线分析、结果比较
tests/        模型侧行为测试与 runtime CPU 契约测试
docs/         模型文档、验证与调优记录
third_party/  自定义算子、GE pass、插件源码
images/       文档图片统一平铺存放
work/         本机生成的图、依赖、golden、output 和结果，Git 忽略
```

## 测试

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
ctest --test-dir runtime/build --output-on-failure
```

`tests/` 的 Python 用例按路径加载各模型目录的模块，全部设计为 CPU 用例、不需要 NPU，覆盖 `tools/compare.py` 的双布局加载与门禁、Qwen 输入池写入的 `ge-cases/1` 契约、StartLux 统一预处理接缝、StartLuxRMSNorm gamma 折叠，以及 StartLux ChunkGatedDeltaRule 原型依赖的启动守卫。最后一项需要 `causal_conv1d` 可导入，该包由 `models/startlux-decision-v1.1-0.8b-instant/prepare.sh` 安装；未安装时它的 `setUpClass` 直接报错，其余用例不受影响。

`tests/runtime_contracts.cpp` 是 `ge_execute` 的 CPU 契约测试，不链 GE、不需要 NPU，由 `runtime/CMakeLists.txt` 构建、经 ctest 运行，覆盖抽样一致、余数分配、预热线程与屏障、失败记录与释放。

`tools/compare.py` 按 `result.json` 的 `cases`/`outputs` 键自动分派新旧两种结果布局，因此不同时期产出的结果目录可以混比。它只依赖 NumPy，不加载模型模块、Torch 或 TorchNPU。

## 相关文档

- [BGE 文件索引](docs/bge/README.md)、[交付报告](docs/bge/delivery-100.md)、[ge_execute 用法](docs/bge/ge-execute-usage.md)、[验收](docs/bge/acceptance.md)
- [Qwen 使用说明](models/qwen2.5-0.5b/README.md)与[适配、验证、调优及 profiling 索引](docs/qwen2.5-0.5b/README.md)
- [runtime 说明](runtime/README.md)、[离线工具](tools/README.md)
- [模型 README 模板](docs/model-readme-template.md)
- [CANN 辅流接口核验](docs/cann-eager-attached-stream-support.md)：已在 `V100R001C12B140` / 850-a5 逻辑卡 1 实测端到端通过；`V100R001C12B086` 与 `V100R001C11SPC001B243` 实测无 `RequestAttachedStream`。其他版本与设备待验证。
