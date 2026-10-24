# Ascend GE model adapters

一个模型一份 YAML，模型改动集中在独立适配文件。框架提供运行能力，优化策略交给 Skills。

## 环境

在 CANN 容器的项目根目录执行，使用容器 Python：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
python3 -m pip install -r requirements.txt
bash runtime/build.sh
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p work/tmp
export TMPDIR="$PWD/work/tmp"
```

A3 原生GE执行前，在CANN环境之后加载容器Python环境：

```bash
source models/qwen2.5-0.5b/scripts/env_a3.sh
```

## 七个独立步骤

每条命令只执行本步，缺少输入、图或依赖时直接报错，不补跑前置步骤。没有 check、resolve、verify 或 profile 命令。

| 步骤 | 命令 | 读取 | 产出 |
|---|---|---|---|
| 1 | data | 模型 YAML、生成脚本 | 持久化输入 NPY |
| 2 | baseline | 已有输入、原始模型 | golden.npz |
| 3 | prepare | 选定依赖源码、构建脚本 | 算子/pass/plugin 安装目录 |
| 4 | build | 已有输入、原始模型、适配代码、已安装依赖 | AIR / ONNX，以及 ATC 路径的 OM |
| 5 | infer | 已有图、输入、运行适配代码 | outputs.npz，可采集 profiling |
| 6 | compare | 已有 golden、output、YAML 阈值 | 精度结果 |
| 7 | bench | 已有图、完整输入池、性能参数 | 吞吐、延迟、逐次调用记录，可采集 profiling |

Qwen 依次执行：

```bash
python3 -m core.pipeline data --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b
python3 -m core.pipeline baseline --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b
python3 -m core.pipeline prepare --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b
python3 -m core.pipeline build --config models/qwen2.5-0.5b/config/model.yaml --device 6
python3 -m core.pipeline infer --config models/qwen2.5-0.5b/config/model.yaml --device 6
python3 -m core.pipeline compare --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b
python3 -m core.pipeline bench --config models/qwen2.5-0.5b/config/model.yaml --device 6 --instances 1
```

run 只按上述顺序调用同一套步骤函数。比较不通过时汇总选定 case 的精度错误并停止，独立 bench 仍可以运行。

```bash
python3 -m core.pipeline run --config models/qwen2.5-0.5b/config/model.yaml --device 6
```

## Python 调用

命令行与 Python 共用 core/pipeline.py。execute 读取当前 YAML、选择平台并应用本次参数覆盖；main 只解析命令行和显示结果。execute 的 config 参数只接收一个 YAML 路径，不接收已解析字典或配置列表。

```python
from core.pipeline import execute

result = execute(
    "bench",
    "models/qwen2.5-0.5b/config/model.yaml",
    device=6,
    instances=4,
)
print(result["performance"])
```

execute(stage, config, *, device=None, target=None, targets=None, case_ids=None, instances=None, profiling=None) 支持与命令行相同的运行覆盖。case_ids 可以是 ["case-0000", "case-0001"] 或 "all"；targets 是可选芯片目录 YAML 路径。覆盖只保存在内存和实际结果中，不改模型 YAML。Python 调用错误直接抛出异常，命令行显示错误并以非零状态退出。

独立 cli.py 已删除，不保留旧模块入口。report 是读取已有结果的工具命令，不属于七步管线。

## 精度与性能范围

Qwen 默认 verification.case_ids 为 case-0000。baseline、infer、compare 都只处理所选 case；日常只计算一组 golden，最终验收可以各步传 --case-ids all，或 run --case-ids all。精度默认单实例，与 benchmark.instances 分开。主流程没有候选 eager 精度门禁；候选融合算子替换仍在 build 导出前应用。

bench 始终读取完整输入池，不需要 golden、output 或精度证据，也不读取原始权重。infer/bench/compare/prepare 不要求 QWEN_MODEL_DIR；该变量供 data/baseline/build/run 读取原始模型使用。

```bash
python3 -m core.pipeline bench --config models/qwen2.5-0.5b/config/model.yaml --device 6 --instances 1
python3 -m core.pipeline bench --config models/qwen2.5-0.5b/config/model.yaml --device 6 --instances 8
```

--instances 只在本次调用覆盖 YAML，不改 YAML，不生成临时 YAML。iterations 是所有实例合计的模型调用次数。Qwen 当前默认 batch=16，1000 次调用对应 16000 个业务样本。按 seed 生成全局有放回抽样序列，实例数只改变分工；每实例至少预热所有输入 shape。导出、GE 编译、加载、预热不计入性能时间。

## 图、数据和运行结果

Qwen 原始输入位于 models/qwen2.5-0.5b/io/inputs，每个 case 一个目录，包含 input_ids.npy、position_ids.npy、actual_seq_lengths.npy。当前默认 100 组、batch16；保留历史 batch10 的生成方法并确定性扩展请求。data 负责生成/更新；其余步骤只读取已有文件，不调用生成脚本的 generate。

图位于 work/<模型名>/build。具名 IO 信息直接写入 AIR 图属性或 ONNX metadata，GE 运行从图读取，不生成 cases.json、build.json、lock.json、io.json 或交付清单。ATC 路径保留导出图和 model.om；运行 OM 时从旁边的导出图读取相同 IO。原始模型、模型参数或输入编码改变后显式重跑 data/build；框架不自动判断旧图是否可复用。

baseline 和 infer 分别在 work/<模型名> 保存 golden.npz、outputs.npz，只包含本次选择的 case；每个 case 保留业务输出与对齐信息。compare 要求为所有实际业务输出配置策略，缺失、多余或未配置的输出直接报错；不允许空策略绕过精度比较。compare 只读取这两个数据文件，无需输入目录、NPU、原始模型或依赖安装目录。精度和性能结果、日志、requests.csv 位于 work/<模型名>/results/<run>。测量过程中执行失败时仍保存已完成调用、失败调用和部分统计，并将本次结果标为 failed；初始化失败不生成虚假的性能值。内部传输使用的 inputs.bin/outputs.bin 在执行结束删除。

每次命令读取当前 YAML。修改实例数直接重跑 bench；GE Session 限核在加载 AIR 时设置，ATC 限核在编译 OM 时设置，因此 ATC 编译参数改变需要显式重新 build。修改依赖源码后显式 prepare。没有自动缓存命中或兼容旧接口的分支。

## Profiling 与平台

infer、bench 或 run 可传 --profiling，也可使用 YAML 的 observability.profiling。采集复用当前图，普通 bench 的随机负载也支持采集；性能结果会标明 profiling_enabled，带采集开销的结果单独看待。GE Session dump 尚不支持。

--device 指定本次设备；按精确 SoC 选择模型 YAML 的 platforms 和可选 chips。离线步骤用 --target 指定精确 SoC。精确芯片默认配置由 YAML 的 chips 定义：Ascend910_9382 使用8|16、batch16、5实例；Ascend950PR_958b 的NZ推荐最优配置为16|32、batch16、3实例。核预算不表示独占绑核。A3 和 A5 默认均启用权重 NZ Pass。依赖脚本接收 GE_SRC_DIR、GE_BUILD_DIR、GE_INSTALL_DIR、GE_PYTHON；依赖交付件保存在 work 下，NZ Pass 同时安装到 ${ASCEND_OPP_PATH}/vendors/custom_nz_pass/custom_fusion_passes/，由 GE 自动加载。

```text
models/       模型适配、数据脚本、唯一模型 YAML、构建脚本和原始输入
third_party/  自定义算子、GE pass、插件源码
targets/      精确芯片目录
core/         七步编排、源模型加载、导出、比较和原生运行边界
runtime/      GE Session / ATC OM 执行、并发、限核、测量、采集
tools/        profiling/dump 离线分析
tests/        行为与运行契约测试
docs/         项目说明、验证与调优文档
images/       文档图片统一平铺存放
work/         本机生成的图、依赖、golden、output 和结果，Git 忽略
```

```bash
python3 -m core.pipeline report work/qwen2.5-0.5b/results/<run>/result.json
python3 -m unittest discover -s tests -p 'test_*.py'
ctest --test-dir runtime/build --output-on-failure
```

每次性能运行保存带版本号的原始结果和逐调用记录，可通过 core.results.export_benchmarks 导出统一 CSV，字段与计量口径见 docs/benchmark-results.md。

## 相关文档

- [A3调优记录](docs/qwen2.5-0.5b/qwen2.5-0.5b-a3-tuning.md)
- [A3/A5复测与无缓存调优](docs/qwen2.5-0.5b/qwen2.5-0.5b-a3-a5-retuning.md)
- [A3/A5逐算子profiling对比](docs/qwen2.5-0.5b/qwen2.5-0.5b-a3-a5-profiling.md)
- [A5权重NZ实验与回归记录](docs/qwen2.5-0.5b/qwen2.5-0.5b-a5-weight-nz.md)
- [A5默认NZ与限核、实例数调优](docs/qwen2.5-0.5b/qwen2.5-0.5b-a5-nz-tuning.md)
- CANN辅流接口核验：[9.1.0](docs/cann-9.1.0-eager-attached-stream-support.md)、[9.2.0](docs/cann-9.2.0-eager-attached-stream-support.md)、[9.3.0](docs/cann-9.3.0-eager-attached-stream-support.md)
