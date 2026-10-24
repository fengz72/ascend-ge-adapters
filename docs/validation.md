# 框架验证记录

模型适配、正确性和性能实验分别见 [Qwen 验证记录](qwen2.5-0.5b/validation.md)和[模型文档索引](qwen2.5-0.5b/README.md)。本页记录通用执行契约及 Tiny、ONNX 回归，不维护模型默认配置。

## 适配器重构期间的框架回归（2026-10-24）

A5 主工作区及 A3 隔离副本分别通过 59 项 Python 测试，覆盖框架契约、模型适配、结果保存及错误传播。真实 Qwen 导出与精度范围单独记在模型验证文档。本次文档整理没有重跑设备测试。

## 提交前框架回归（2026-10-20）

以下保留当时环境和结果，只代表该次验证。

环境为 850-a5 的 hejun-cann910-qwen25 容器、CANN 9.1.0、Python 3.12.13、Ascend950PR_958b（32 AIC / 64 AIV）。使用当时空闲的设备1，运行前后未观察到其他同卡驻留进程；其他设备上的服务没有被操作。以下是框架回归，不计入调优报告的164次测量，也不替代设备6的三轮长测。

53项 Python 测试通过。新增回归覆盖：空/漏配/多配精度策略、真实输出名不一致、包含 metadata/alignment 的业务输出名、ONNX 冒号/斜杠/百分号/中文名称、NPY文件名可逆转义，以及失败性能结果贯穿 native→worker→pipeline 后的保留。未执行真实设备故障注入。

Release 构建成功，CTest 的 runtime_contracts（1/1）通过。C++ 测试使用始终执行的 CHECK，覆盖固定/变长请求池、逐输出容量、采样顺序不随实例数变化，以及测量中途失败后保留3次成功调用和1次失败调用。旧 Bundle/Manifest、固定输入后端分支和未使用的 describe_adapter 已删除；静态输入仍可通过统一请求池运行。

### 实际模型回归

| 模型 | 执行方式与精度范围 | 性能范围 | 结果记录（项目内） |
|---|---|---|---|
| 普通 Torch Tiny | 完整七步 run，3个变长 case、7个业务样本，两个输出均通过 | 双实例，总计21次调用，0错误 | work/tiny-ranker/results/20261020T110032-52fa04d0/result.json |
| 原生 ONNX | 完整七步 run，100个 case、200个业务样本，两个输出均通过 | 双实例，总计21次调用，0错误 | work/onnx-ranker/results/20261020T110709-d09893b1/result.json |

ONNX 示例直接使用 features:0、bias/input、encoder/scores:0、encoder/indices 等原始张量名，从生成NPY、CPU golden、导出图、GE Session到比较完整跑通；没有别名清单。Tiny和ONNX的21次短测只验证执行契约，不用于性能排名。

首次Tiny调用误覆盖了CANN的PYTHONPATH，在build阶段报No module named tbe，失败记录20261020T105925-da613a2d保留。保留set_env.sh配置的PYTHONPATH后重新完整run通过；没有修改框架去绕过环境错误。

## pipeline 统一入口回归（2026-10-20）

入口已统一为 core.pipeline 的 execute/main，core/cli.py 和旧入口已删除，命令行、Python 调用均由 execute 读取一份当前模型 YAML 并应用运行覆盖。实际执行旧模块返回 No module named core.cli。

- 44 项 Python 测试通过。覆盖统一入口、每次重新读取 YAML、参数覆盖及校验、错误透传、七步顺序、比较失败停止、独立 bench 不补跑前置阶段，以及现有数据/模型/运行契约。
- Tiny 在 A5 卡 6 分别通过七个独立命令，又通过完整 run；三组精度通过，独立三实例 bench 与完整 run 的双实例 bench 各 21 次调用均无错误。
- ONNX 使用 Python execute("run", "models/onnx-example/config.yaml", device=6) 完成七步，100 组精度通过，双实例 21 次调用无错误。

本次完整 run 结果：

- Tiny：work/tiny-ranker/results/20261020T020936-6f8e6af0/result.json。
- ONNX：work/onnx-ranker/results/20261020T021043-167edfa9/result.json。

新入口的 report 对完整 run 和独立 bench 均正常显示核数、实例数、精度状态和性能数据。Qwen 的独立 bench、profiling 及当时的精度边界已移至模型验证记录。

## 七步与独立调用（2026-10-19）

环境：Ascend950PR_958b，32 AIC / 64 AIV，CANN 9.1.0，容器 Python 3.12.13，逻辑卡 6。此次验证 GE Session，未测试 ATC。

- 40 项 Python 行为测试通过，覆盖七步顺序、单步不补跑、单组/多组/全量选择、无权重运行配方、NPY 历史分布、移除 check/verify/profile/resolve、比较汇总、输出名与业务元信息不冲突、实例数 CLI 覆盖等。
- 普通 Torch Tiny 完整 run 通过；又分别调用 data、baseline、prepare、build、infer、compare、bench 七条命令通过。独立精度只生成/比较一组，独立 bench 使用完整三组池、三实例。
- ONNX 完整七步 run 通过，100 组精度通过、双实例短测 21 次调用无错误。最终又独立执行 baseline、build、infer、compare，100 组精度通过。
- 缺少 AIR、inputs、依赖安装目录时均直接报错，没有生成输入、安装依赖或重导出。负向测试目录已清理。compare 在无输入目录、无权重、无模型适配文件和无已安装依赖的条件下仍能比较已有 golden/output。

Tiny 完整 run：work/tiny-ranker/results/20261019T130206-9f6311c6/result.json；独立七步位于其 results 下 20261019T132318 至 132532 的各次记录。
ONNX 完整 run：work/onnx-ranker/results/20261019T130404-1db4bd15/result.json；最终独立 compare：work/onnx-ranker/results/20261019T132719-73c7a685/result.json。

## Profiling（2026-10-19）

- Tiny 独立 bench --profiling：双实例、随机 21 次调用，无错误，performance.profiling_enabled=true。采集 52 个文件、313615 字节，证明采集复用真实 bench 路径。

Tiny 性能采集：work/tiny-ranker/results/20261019T132532-968f4911/bench-profiling。

infer 和 bench 的采集目录分别带步骤名称；带 profiling 开销的性能结果不与普通测量混淆。没有独立 profile 流程。

## 产物与验证边界

新流程不生成 cases.json、build.json、lock.json、io.json、job/plan 文件、实例 YAML 或交付清单。IO 在 AIR/ONNX 图内。内部打包输入和原始输出在结束后删除，保留实际 golden.npz、outputs.npz、运行结果和采集数据。

上述 2026-10-19 至 2026-10-20 的设备验证覆盖当时 A5 环境的 GE Session；不能据此推断其他设备或 ATC 路径已经验收。ATC 专项验收仍需按对应模型记录范围，GE Session dump 尚不支持。
