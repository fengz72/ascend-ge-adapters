# 模型验证状态（2026-10-20）

## 新增 A3 Ascend910_9382 验证

99机器、hejun-cann9-qwen25容器、CANN9.0.0、设备1（24 AIC/48 AIV）已跑通Qwen2.5-0.5B Prefix。芯片配置为8|16、batch16、5实例；三轮各10000次调用，业务QPS中位数2172.093、最差P99 41.266ms。100组/1600样本精度通过，最小cosine=0.998909550002、最大relative L2=0.046848644966；54项Python测试和CTest 1/1通过。

全部57次压测尝试、失败与SIGTERM中断、补测、环境修复和原始数据见 [A3调优记录](qwen2.5-0.5b/qwen2.5-0.5b-a3-tuning.md)。下文保留A5历史验证记录，其中“A3无实机”描述对应当时的测试范围。

2026-10-21补充：默认batch16输入池已恢复并核对摘要；新增输入batch一致性拦截，55项Python测试通过；默认5实例1000次调用复验业务QPS=2165.776、P99=41.100ms，100组/1600样本精度复验通过。详见上述调优文档。

## A5 历史验证记录


## 本次框架修复的提交前验证（batch16，2026-10-20）

环境为 850-a5 的 hejun-cann910-qwen25 容器、CANN 9.1.0、Python 3.12.13、Ascend950PR_958b（32 AIC / 64 AIV）。使用当时空闲的设备1，运行前后未观察到其他同卡驻留进程；其他设备上的服务没有被操作。以下是框架回归，不计入调优报告的164次测量，也不替代设备6的三轮长测。

53项 Python 测试通过。新增回归覆盖：空/漏配/多配精度策略、真实输出名不一致、包含 metadata/alignment 的业务输出名、ONNX 冒号/斜杠/百分号/中文名称、NPY文件名可逆转义，以及失败性能结果贯穿 native→worker→pipeline 后的保留。未执行真实设备故障注入。

Release 构建成功，CTest 的 runtime_contracts（1/1）通过。C++ 测试使用始终执行的 CHECK，覆盖固定/变长请求池、逐输出容量、采样顺序不随实例数变化，以及测量中途失败后保留3次成功调用和1次失败调用。旧 Bundle/Manifest、固定输入后端分支和未使用的 describe_adapter 已删除；静态输入仍可通过统一请求池运行。

### 实际模型回归

| 模型 | 执行方式与精度范围 | 性能范围 | 结果记录（项目内） |
|---|---|---|---|
| 普通 Torch Tiny | 完整七步 run，3个变长 case、7个业务样本，两个输出均通过 | 双实例，总计21次调用，0错误 | work/tiny-ranker/results/20261020T110032-52fa04d0/result.json |
| 原生 ONNX | 完整七步 run，100个 case、200个业务样本，两个输出均通过 | 双实例，总计21次调用，0错误 | work/onnx-ranker/results/20261020T110709-d09893b1/result.json |
| Qwen Prefix FP16 | 七步分别独立调用，默认 case-0000、batch16，共16个样本通过 | 完整100组池，四实例、1000次调用、16000个样本，0错误 | 分步骤记录见下表 |

ONNX 示例直接使用 features:0、bias/input、encoder/scores:0、encoder/indices 等原始张量名，从生成NPY、CPU golden、导出图、GE Session到比较完整跑通；没有别名清单。Tiny和ONNX的21次短测只验证执行契约，不用于性能排名。

Qwen 固定为16|32、batch16、4实例、seed42，Prefix源码 kAicCores=16；prepare 重新构建的 libcust_opapi.so SHA256 为 51d16a5bed62e49e911cd7f15121c096ea0a805a6d45f35ef5d7fea7b4101adc，与调优时的16核库一致。

| Qwen独立步骤 | run_id | 实际结果 |
|---|---|---|
| data | 20261020T105946-a33c8483 | 100个case、1600个样本、300个NPY，固定模型输入目录 |
| baseline | 20261020T105946-29f2a667 | 未patch的CPU FP32原始HF模型，case-0000的16条参考输出 |
| prepare | 20261020T105957-8bf6cb2d | 显式重建安装A5 Prefix依赖 |
| build | 20261020T110129-fb39b85d | 显式重新导出FP16 AIR，IO嵌入图内 |
| infer | 20261020T110443-e51e0a5e | 读取已有AIR和输入，单实例输出一组 |
| compare | 20261020T110514-e8639972 | 16/16通过；最小cosine=0.9999978163，最大relative L2=0.0033425287 |
| bench | 20261020T110525-0801aebb | 1000次调用，业务QPS=2657.744，模型调用QPS=166.109，p50=23.753ms，p99=27.007ms，max=54.762ms，0错误 |

上述记录均在 work/qwen2.5-0.5b/results/<run_id>/result.json。bench 保存1000条requests.csv记录，每条samples=16、各实例250次；日志确认 ge.aicoreNum=16|32。infer、compare和bench在未设置QWEN_MODEL_DIR时完成；没有补跑数据、golden、依赖安装或导出步骤。每次步骤执行结束后均没有遗留inputs.bin/outputs.bin，也未生成cases/build/lock/io等清单文件。

profiling 复用已有AIR，由独立 infer --profiling 完成，run_id=20261020T110630-50a481e5。msprof --export=on 成功，导出后89个文件、6710798字节，op_statistic包含Prefix metadata和PrefixMacroKernel各24次。采集数据位于同次结果目录的infer-profiling；随后独立compare（20261020T110658-a1da19b5）仍通过。性能短测关闭profiling。

首次Tiny调用误覆盖了CANN的PYTHONPATH，在build阶段报No module named tbe，失败记录20261020T105925-da613a2d保留。保留set_env.sh配置的PYTHONPATH后重新完整run通过；没有修改框架去绕过环境错误。

### 输入迁移与未验证范围

Qwen默认data.directory已固定为 models/qwen2.5-0.5b/io/inputs，实际内容为batch16，池SHA256为21defc288cc9848d6f1f71f8a8c6d900327963431e57e77125274b5f5fc8c85b，与历史batch16调优池一致。历史batch10的完整输入、golden.npz和outputs.npz在执行新baseline/infer前保留于 work/qwen2.5-0.5b/validation/history-batch10：

| 历史batch10数据 | SHA256 |
|---|---|
| 输入池（规范case摘要） | 69892db56429877350ce7f55480f1f3f831e6342e9a1f14ff7be3235986a55a4 |
| golden.npz | 1e1231cbfdbddffc2b6d3ebf2471214fb6269f4479d6c7ea469c70854774caf5 |
| outputs.npz | 17dd6ff881beee0ba29c3ef3ab7ab18b7af336a6fee80199420dde7859376227 |

全部164次历史调优结果和文末附录数据保持不变。本次没有进行batch16全100组精度验收；下节的100组全量精度结论仅针对历史batch10。A3没有实机，12|24限核、prefix_910和weight_nz pass均未实测；其他A5芯片也未验证。ATC本次未执行，ACL后端随原生程序编译通过不代表ATC路径验收；GE Session dump仍不支持。


## 历史 Qwen Prefix FP16 全量验收（batch10，2026-10-20）

按要求保持原 FP16 路径，build.dtype 为 float16，precision_mode 为 force_fp16。临时尝试的 FP32 残差/RMSNorm 已撤回，适配文件与本次定位前逐字节一致；third_party 算子源码未修改。验收阈值只调整此模型 YAML 的 verification.outputs.scores：

```yaml
verification:
  outputs:
    scores:
      policy: numeric
      scope: per_sample
      cosine_min: 0.998
      relative_l2_max: 0.05
```

输入仍为同一历史 100 组，每组 batch=10，共 1000 个业务样本。golden 重新使用未 patch 的 HF 模型在 CPU FP32 下生成，之后没有修改；没有将 golden 改成 FP16。

对同一 CPU FP32 golden 的全量对照：

| 候选执行路径 | 旧 0.999/0.02 阈值失败组数 | 当前 0.998/0.05 阈值失败组数 | 最小 cosine | 最大 relative L2 |
|---|---:|---:|---:|---:|
| 原始 HF FP16（NPU） | 4 | 0 | 0.998912059 | 0.046643686 |
| FIA eager FP16 | 2 | 0 | 0.999310607 | 0.039486116 |
| Prefix eager FP16 | 3 | 0 | 0.999068564 | 0.049276981 |
| Prefix GE Session FP16 | 2 | 0 | 0.999475372 | 0.034476226 |

原始 HF FP16 的旧阈值失败为 case-0002/r8、case-0030/r4、case-0091/r8、case-0093/r3。Prefix GE 的旧阈值失败为 case-0002/r8、case-0091/r8，最大逐元素绝对误差为 0.046730756759643555。此前 FIA 只验证三个控制 case 通过，不能据此推断其全量精度通过；本次补齐全 100 组后，旧阈值下也有两组失败。

按当前 YAML 在 A5 卡 6 重新执行 infer --case-ids all 和 compare --case-ids all：100 组、1000 个样本全部通过。GE 输出与调整阈值前逐值一致，归档的业务 scores 和 AIR 输出均为 float16。本次改变的是验收阈值，数值误差范围没有改变。框架默认比较策略和其他模型未修改。

实际运行结果：

- 原始 CPU FP32 golden：work/qwen2.5-0.5b/results/20261020T022208-0eba6b60/result.json。
- 旧阈值全量失败记录：work/qwen2.5-0.5b/results/20261020T022925-2f21120e/result.json。
- 当前 FP16 全量 infer：work/qwen2.5-0.5b/results/20261020T023508-c4256c0c/result.json。
- 当前阈值全量 compare：work/qwen2.5-0.5b/results/20261020T023557-ab2f535d/result.json。

CPU FP32 golden SHA256：1e1231cbfdbddffc2b6d3ebf2471214fb6269f4479d6c7ea469c70854774caf5。复用了修改阈值前的 FP16 AIR，没有重新导出或修改算子；未新增配置快照和清单。44 项 Python 测试通过。A3 无实机，ATC 按要求本次未测试。

以下为此前使用旧阈值时的验证记录；其中的“未通过”和“尚未修复”描述当时的 0.999/0.02 阈值状态。

## pipeline 统一入口回归（2026-10-20）

入口已统一为 core.pipeline 的 execute/main，core/cli.py 和旧入口已删除，命令行、Python 调用均由 execute 读取一份当前模型 YAML 并应用运行覆盖。实际执行旧模块返回 No module named core.cli。

- 44 项 Python 测试通过。覆盖统一入口、每次重新读取 YAML、参数覆盖及校验、错误透传、七步顺序、比较失败停止、独立 bench 不补跑前置阶段，以及现有数据/模型/运行契约。
- Tiny 在 A5 卡 6 分别通过七个独立命令，又通过完整 run；三组精度通过，独立三实例 bench 与完整 run 的双实例 bench 各 21 次调用均无错误。
- ONNX 使用 Python execute("run", "models/onnx-example/config.yaml", device=6) 完成七步，100 组精度通过，双实例 21 次调用无错误。
- Qwen Prefix 使用 python3 -m core.pipeline run 完成七步：生成/读取历史 100 组 NPY、单组原始 golden、依赖准备、AIR 导出、单实例 GE 推理、精度比较、完整池随机性能测试。默认 case-0000 精度通过，最大 relative L2=0.0033425286544669563；单实例 1000 次调用零错误，业务吞吐 1216.918/s。阈值保持 cosine >= 0.999、relative L2 <= 0.02。

本次完整 run 结果：

- Tiny：work/tiny-ranker/results/20261020T020936-6f8e6af0/result.json。
- ONNX：work/onnx-ranker/results/20261020T021043-167edfa9/result.json。
- Qwen：work/qwen2.5-0.5b/results/20261020T021142-c321eb17/result.json。

独立 bench 回归时移除 QWEN_MODEL_DIR，复用上述 run 的 AIR、100 组输入和已安装依赖，保持 16|32 预算、seed=42，每组总计 1000 次调用：

| 调用方式 | 实例数 | 业务样本/s | p50 ms | p99 ms | 错误 |
|---|---:|---:|---:|---:|---:|
| 命令行 main → execute | 1 | 1215.42 | 8.285 | 9.171 | 0 |
| Python execute | 2 | 2207.98 | 9.013 | 10.200 | 0 |

- 单实例：work/qwen2.5-0.5b/results/20261020T021529-3c338266/result.json。
- 双实例：work/qwen2.5-0.5b/results/20261020T021616-619c1f75/result.json。

两次均只记录 bench，无前置步骤；317 个既有文件的大小和纳秒修改时间保持不变，涵盖模型 YAML、AIR、全部输入、golden、output 和依赖安装文件。两次 1000 条全局抽样序列 SHA256 均为 0774f62ec8798adfe7a46ed8244f2fe4820e38548878f1c5f5821c5df1ca4d1e。没有新增清单、配置快照或实例 YAML，内部打包输入/输出在执行后删除。

Python execute("infer", ..., device=6, profiling=True) 在未设置权重路径的条件下成功采集，原始数据为 54 个文件、1031925 字节，msprof --export=on 成功。算子统计包含 Prefix metadata/planner 和 PrefixMacroKernel 各 24 次任务。采集后通过新命令入口独立 compare，默认单组精度仍通过。

- 采集：work/qwen2.5-0.5b/results/20261020T021707-737f8062/infer-profiling。
- 比较：work/qwen2.5-0.5b/results/20261020T021737-b461f679/result.json。

新入口的 report 对完整 run 和两条独立 bench 均正常显示核数、实例数、精度状态和性能数据。本次 Qwen 没有重跑全 100 组精度，历史全量失败仍未修复；A3 没有实机，ATC 按要求未测试。

以下保留 2026-10-19 七步独立执行验证记录。A3 无实机，ATC 本次仍未测试；Qwen 历史全量精度问题见文末。

环境：Ascend950PR_958b，32 AIC / 64 AIV，CANN 9.1.0，容器 Python 3.12.13，逻辑卡 6。此次验证 GE Session，未测试 ATC。

## 七步与独立调用

- 40 项 Python 行为测试通过，覆盖七步顺序、单步不补跑、单组/多组/全量选择、无权重运行配方、NPY 历史分布、移除 check/verify/profile/resolve、比较汇总、输出名与业务元信息不冲突、实例数 CLI 覆盖等。
- 普通 Torch Tiny 完整 run 通过；又分别调用 data、baseline、prepare、build、infer、compare、bench 七条命令通过。独立精度只生成/比较一组，独立 bench 使用完整三组池、三实例。
- ONNX 完整七步 run 通过，100 组精度通过、双实例短测 21 次调用无错误。最终又独立执行 baseline、build、infer、compare，100 组精度通过。
- Qwen Prefix 完整 run 按顺序完成七步，默认只生成一组 golden、单实例 GE 输出与比较；最大 relative L2=0.0033425286544669563，未放宽 cosine >= 0.999、relative L2 <= 0.02。随后完整 100 组池随机测量 1000 次调用，0 错误，业务吞吐 1215.587/s。
- 缺少 AIR、inputs、依赖安装目录时均直接报错，没有生成输入、安装依赖或重导出。负向测试目录已清理。compare 在无输入目录、无权重、无模型适配文件和无已安装依赖的条件下仍能比较已有 golden/output。

Qwen 完整 run：work/qwen2.5-0.5b/results/20261019T130550-47e42bf2/result.json。
Tiny 完整 run：work/tiny-ranker/results/20261019T130206-9f6311c6/result.json；独立七步位于其 results 下 20261019T132318 至 132532 的各次记录。
ONNX 完整 run：work/onnx-ranker/results/20261019T130404-1db4bd15/result.json；最终独立 compare：work/onnx-ranker/results/20261019T132719-73c7a685/result.json。

## 独立 bench 的 1–8 实例实测

全程移除 QWEN_MODEL_DIR。每条命令只改变 --instances，复用同一 AIR、同一 100 组 NPY 输入、16|32 核数预算、batch=10、seed=42，每组共 1000 次调用。8 组共 8000 次调用，0 错误。

| 实例数 | 业务样本/s | p50 ms | p99 ms | 错误 |
|---:|---:|---:|---:|---:|
| 1 | 1173.77 | 8.547 | 10.194 | 0 |
| 2 | 2224.46 | 8.985 | 10.212 | 0 |
| 3 | 2571.00 | 11.573 | 13.021 | 0 |
| 4 | 2601.98 | 15.262 | 17.147 | 0 |
| 5 | 2549.41 | 19.477 | 21.431 | 0 |
| 6 | 2457.37 | 23.790 | 32.302 | 0 |
| 7 | 2470.85 | 27.975 | 33.928 | 0 |
| 8 | 2408.57 | 32.122 | 35.163 | 0 |

基于大小和纳秒修改时间核对 317 个文件，模型 YAML、AIR、全部输入、golden、output 和依赖安装文件均未变化。bench 没有读取原始权重或补跑任何前置步骤。GE Session 仍需加载/在线编译图、建立各实例并预热，这些属于实际 GE 执行准备，不是重新导出 AIR，也不计入测量。

汇总：work/qwen2.5-0.5b/results/instances-1to8-sevensteps-20261019/summary.csv，各次实际结果路径见 CSV。延迟口径为 H2D、描述符、execute/synchronize，不含编译、加载、预热、业务预处理或服务排队。限核预算不表示独占绑核。

输入池 SHA256 仍为 69892db56429877350ce7f55480f1f3f831e6342e9a1f14ff7be3235986a55a4；1000 次全局抽样序列 SHA256 仍为 0774f62ec8798adfe7a46ed8244f2fe4820e38548878f1c5f5821c5df1ca4d1e，8 组均一致。

## Profiling

- Tiny 独立 bench --profiling：双实例、随机 21 次调用，无错误，performance.profiling_enabled=true。采集 52 个文件、313615 字节，证明采集复用真实 bench 路径。
- Qwen 独立 infer --profiling：未设置权重路径，读取已有 AIR 和所选输入，保存一组 output。原始采集 54 个文件、1077929 字节；msprof --export=on 成功，导出后共 89 个文件、6868929 字节。统计包含 Prefix metadata/planner 与 PrefixMacroKernel 各 24 次任务。
- 采集后独立 compare 仍通过，结果位于 work/qwen2.5-0.5b/results/20261019T132815-d5606194/result.json。

Qwen 采集：work/qwen2.5-0.5b/results/20261019T132741-fe130474/infer-profiling。
Tiny 性能采集：work/tiny-ranker/results/20261019T132532-968f4911/bench-profiling。

infer 和 bench 的采集目录分别带步骤名称；带 profiling 开销的性能结果不与普通测量混淆。没有独立 profile 流程。

## 文件与仍未验证的边界

新流程不生成 cases.json、build.json、lock.json、io.json、job/plan 文件、实例 YAML 或交付清单。IO 在 AIR/ONNX 图内。内部打包输入和原始输出在结束后删除，保留实际 golden.npz、outputs.npz、运行结果和采集数据。Qwen 原始输入目录仍为 100 个 case、300 个 NPY，无 JSON 数据文件。

Qwen 本轮只验证默认单组精度，不能宣称全 100 组已通过。历史全量 GE 精度曾在 case-0002/r8、case-0091/r8 失败，最大 relative L2=0.0344762，尚未修复；当前主流程不再由候选 eager 检查提前阻断 GE 验证。

A3 没有实机，12|24 预算、算子和 GE pass 未验证。当前 A5 958b 以外的 SKU 也未验证。ATC 按要求本轮未测；其独立执行仍保留导出图作为 IO 来源，需要后续专项验证。GE Session dump 尚不支持。
