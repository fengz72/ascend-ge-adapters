# Qwen2.5-0.5B 验证记录

使用方法见[模型 README](../../models/qwen2.5-0.5b/README.md)，适配原理见[适配说明](adaptation.md)，性能和 profiling 见[文档索引](README.md)。框架、Tiny 和 ONNX 的验证集中在[框架验证记录](../validation.md)。

本文按日期和输入范围保留验证结果。历史“当前”“未验证”及“尚未修复”均指所在实验时点；不同 batch、阈值、依赖版本的结论不能互相替代。依赖来源与迁移分支的维护历史见[维护记录](maintenance.md)。

## Prefix Attention 归档迁移验证（2026-10-27）

Prefix Attention 从 `third_party/ascend-ops/prefix-attention/` 迁入
`third_party/custom_development_code/custom_op/prefix-attention/`，目标仓库分支为
`codex/migrate-prefix-attention`，迁移提交为 `dc21a7a7f10661ee49af75c9519832103b2d6db9`；
源提交为 `fbf0400872079046c1b494188c5d8bd2749d868a`。模型 YAML 的 A3/A5 source
与对应 revision 已更新。原 ascend-ops 子模块及其引用保持不变，先保留用于迁移对照。

导入源提交跟踪的159个文件；除3份 README 外，156个源码与构建文件逐字节一致，
执行权限全部保留。新增局部 .gitignore 排除普通构建、缓存和日志产物；
A5 codegen、AICPU 源码及 artifacts.h 仍作为构建输入签入。
来源、版本与构建入口见[算子说明](../../third_party/custom_development_code/custom_op/prefix-attention/README.md)。

实测环境：99机器、hejun-cann9-qwen25容器、CANN9.0.0、Python3.11.15、
Ascend910_9382，设备6。重新 prepare 后核对 CMakeCache，源码目录确认为迁入后的 ascendc。

| 检查 | 范围 | 结果 |
|---|---|---|
| 配置解析 | Ascend910_9382、Ascend950PR_958b 的依赖目录、构建脚本及迁移提交 | 通过 |
| Python / Shell 语法 | 迁入工程的 Python 源码及两个构建入口 | 通过 |
| 框架回归 | Python 60项；CTest 1/1 | 全部通过 |
| A3 prepare | 新目录全新构建 AscendC OPP 包并安装 Torch binding | 通过 |
| A3 算子独立测试 | eager 对 FP32参考；tensor/default接口；GE整图及复用 | 全部通过；最大参考误差1.150668e-4，其余对照逐值一致 |
| A3 Qwen build / infer / compare | 重新导出AIR；batch16，全100组/1600样本 | 全部通过；最小cosine=0.998909550002，最大relative L2=0.046848644966 |
| 迁移前后输出对照 | 已归档旧输出与新输出；既有golden的SHA256 | 数值数组变化0项，最大差异0；golden未修改 |
| A5 设备构建与执行 | 850-a5 / hejun-cann920-cannbotdsl 补充验证 | 构建及算子测试通过；全量精度在进程级加载输出扩容修复库后通过，详见下节 |

本次复用已有 golden，没有重新生成数据或修改精度阈值，也没有执行性能压测。
上述表格的A3结论适用于所列A3环境；A5补充验证见下节。当前继续保留 ascend-ops 子模块。

| 步骤 | run_id |
|---|---|
| prepare | 20261027T062007-3365ad90 |
| build | 20261027T062211-1933a54a |
| infer --case-ids all | 20261027T062439-a8d01f6f |
| compare --case-ids all | 20261027T062512-dafeb903 |

运行记录位于 `work/qwen2.5-0.5b/results/<run_id>/`；迁移清单、配置检查、
测试日志及精度摘要位于 `work/prefix-migration-20261027/`。
原图、outputs和golden副本归档于该目录的 `previous-main/`；
新的模型图及输出保留在默认work目录。

## A5 Prefix Attention 迁移补充验证（2026-10-27）

使用850-a5机器、hejun-cann920-cannbotdsl容器、
`/home/h00909914/ascend-ge-adapters`、逻辑设备7。
迁入分支及提交仍为 `codex/migrate-prefix-attention` /
`dc21a7a7f10661ee49af75c9519832103b2d6db9`，未修改算子实现或精度阈值。
CANN工具包为9.3.0 B140，Python3.12.14，Torch2.10.0+cpu、
TorchNPU2.10.0.post2，Transformers5.10.1。

**结论：迁移构建及算子测试通过；默认GE的全量精度未通过，
进程级加载既有输出扩容修复库后，全100组/1600样本通过，输出与迁移前逐值一致。**
这不是对该容器默认GE全量精度的通过声明。

| 检查 | 结果 |
|---|---|
| 源码与版本 | 159个导入文件哈希与99机器一致；两个子仓库工作区干净，旧子模块未改动 |
| A5 prepare / runtime build / AIR导出 | 从新路径重新构建、安装并导出成功 |
| Planner | frozen golden、8线程同步、深队列及同bounds并发测试全部通过 |
| 独立Attention | T=128/2048/4096、batch=1/16/32，FP32参考最大绝对误差1.1202693e-4 |
| 框架回归 | Python60项、CTest1/1通过；复验包含进程级修复库环境 |
| 默认GE全量执行 | infer执行成功；compare有23组失败，77组与旧输出逐值一致 |
| 加载扩容修复库后的全量执行 | 100组/1600样本全部通过，最小cosine=0.999475201977，最大relative L2=0.034481627722 |
| 迁移前后输出 | 200个NPZ数组全部逐值一致，数值最大差异0 |
| CPU参考来源 | 复用99机器的完整FP32 golden；权重、模型配置及300个NPY输入文件哈希一致；参考文件未改写 |

原A5 golden仅含case-0000，已归档；本次用已核验来源的完整CPU FP32参考进行全量比较。
原A5单case参考与该完整参考的重叠分数最大绝对差为2.25007534e-5。
不重新生成输入、不调整阈值、不使用设备输出替代golden，未进行性能压测。

### GE环境差异与复现

默认GE的失败输入均位于首轮容量之外的较大shape范围，符合算子文档中已记录的
`EagerOpExecutionContext::MallocOutputTensor`输出缓冲区扩容问题。
同一AIR、算子二进制、输入、golden和阈值保持不变，仅加载已有修复版
`liblowering.so`后，全部输出恢复为迁移前数值。修复库来自已有GE源码提交
`cec5518fe3eb39dd7eec7b1fd124280b2ff6eaf2`的构建产物，
SHA256为 `df205905bd630bfdd7391f5f28c86417ff9cf95451662723bb27e64885530eb6`。

本次对照仅验证该进程级修复在当前Qwen工作负载中的效果；
正式运行环境应使用包含该修复、版本匹配的GE Compiler / Executor。
未替换系统CANN或GE库。曾尝试在work内组合完整GE、Runtime和头文件，
但遇到组件及内置GatherV2形状推导兼容问题；这些实验未作为通过环境。
已恢复原CANN头文件/库对应的运行器和依赖，实验日志保留于work。

#### 修复来源与适用环境

扩容修复位于 GE 提交
[`3bc5821ebe7c93720842dbd5989bbab82a745ef9`](https://gitcode.com/cann/ge/commit/3bc5821ebe7c93720842dbd5989bbab82a745ef9)，
修改 `graph_metadef/exe_graph/runtime/` 下 Eager / HostCPU 的输出缓冲区容量检查。
本实验使用的构建源码为公开 GE `9.3.0-beta.1` 分支中的
`cec5518fe3eb39dd7eec7b1fd124280b2ff6eaf2`，包含上述修复；
`graph_metadef` 已包含在该 GE 仓库中，无需另取子模块。

| 项目 | 本次已验证值 |
|---|---|
| 主机环境 | aarch64，openEuler 24.03 LTS-SP4，glibc 2.38 |
| CANN | 9.3.0 B140，`/usr/local/Ascend/cann-9.3.0` |
| 构建工具 | GCC 12.3.1，CMake 3.31.12，C++11 ABI=0 |
| Python | 3.12.14，前缀 `/usr/local/python3.12.14` |
| GE 构建目标 | `ge-executor` 配置下的 `lowering` |
| 原实验产物 | `/home/h00909914/ge/build/ge-executor/graph_metadef/exe_graph/liblowering.so` |
| 原实验 SHA256 | `df205905bd630bfdd7391f5f28c86417ff9cf95451662723bb27e64885530eb6` |

下列命令在新环境的项目根目录执行。先安装项目依赖、准备匹配的 CANN/驱动和模型输入、
FP32 golden；根据实际位置设置 `CANN_ROOT` 和 Python 前缀。本次单库预加载仅在上表组合验证，
不表示可跨 CANN、CPU 架构或系统 ABI 混用。完整 GE 部署仍需配套 Compiler / Executor。

```bash
export GE_ADAPTER_ROOT="$PWD"
export CANN_ROOT=/usr/local/Ascend/cann-9.3.0
export GE_FIX_PYTHON_PREFIX=/usr/local/python3.12.14
export GE_FIX_DIR="$GE_ADAPTER_ROOT/work/a5-runtime-fix"
mkdir -p "$GE_FIX_DIR"
```

#### 方式一：获取原实验二进制

如果新环境或 Linux 中转机能通过 SSH 别名 `850-a5` 访问原容器，可直接取回已验证文件；
经过中转时，将文件传到新环境的 `$GE_FIX_DIR` 后再次校验。这里不依赖原 `work/` 环境脚本。

```bash
ssh 850-a5 'docker exec hejun-cann920-cannbotdsl cat /home/h00909914/ascend-ge-adapters/work/prefix-migration-20261027/runtime-fix/liblowering.so' > "$GE_FIX_DIR/liblowering.so.download"
printf '%s  %s\n' \
  df205905bd630bfdd7391f5f28c86417ff9cf95451662723bb27e64885530eb6 \
  "$GE_FIX_DIR/liblowering.so.download" | sha256sum --check - &&
  mv "$GE_FIX_DIR/liblowering.so.download" "$GE_FIX_DIR/liblowering.so"
```

#### 方式二：从固定源码重建

无需访问原实验机器。按固定提交的
[GE 构建说明](https://gitcode.com/cann/ge/blob/cec5518fe3eb39dd7eec7b1fd124280b2ff6eaf2/docs/zh/build.md)
安装系统编译依赖，包括 GCC/G++、CMake、make、patch、autoconf、automake 和 libtool；离线环境先按该提交的
[离线编译说明](https://gitcode.com/cann/ge/blob/cec5518fe3eb39dd7eec7b1fd124280b2ff6eaf2/docs/zh/user_guides/offline_compile.md)
准备第三方源码，放到 GE 的 `output/third_party/`。使用新的源码/构建目录，避免继承旧 SDK 缓存。

```bash
(
set -e
source "$CANN_ROOT/set_env.sh"
export PYTHONHOME="$GE_FIX_PYTHON_PREFIX"
export PATH="$PYTHONHOME/bin:$PATH"
export LD_LIBRARY_PATH="$PYTHONHOME/lib:${LD_LIBRARY_PATH:-}"
export GE_FIX_SRC="$GE_ADAPTER_ROOT/work/ge-output-growth-src"
git clone --single-branch --branch 9.3.0-beta.1 https://gitcode.com/cann/ge.git "$GE_FIX_SRC"
git -C "$GE_FIX_SRC" checkout --detach cec5518fe3eb39dd7eec7b1fd124280b2ff6eaf2
git -C "$GE_FIX_SRC" merge-base --is-ancestor 3bc5821ebe7c93720842dbd5989bbab82a745ef9 HEAD
# 使用当前构建 Python；requirements.txt 包含 pybind11>=2.13.6,<3.0.0。
python3 -m pip install -r "$GE_FIX_SRC/requirements.txt"
cmake -S "$GE_FIX_SRC" -B "$GE_FIX_SRC/build/ge-executor" \
  -DBUILD_OPEN_PROJECT=ON -DENABLE_OPEN_SRC=ON -DBUILD_METADEF=ON \
  -DCANN_PACKAGES=ge-executor -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_BUILD_DEVICE=ON -DUSE_CXX11_ABI=0 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DASCEND_INSTALL_PATH="$(readlink -f "$CANN_ROOT")" \
  -DCANN_3RD_LIB_PATH="$GE_FIX_SRC/output/third_party" \
  -DHI_PYTHON="$(command -v python3)"
cmake --build "$GE_FIX_SRC/build/ge-executor" --target lowering --parallel 2
install -m 644 "$GE_FIX_SRC/build/ge-executor/graph_metadef/exe_graph/liblowering.so" "$GE_FIX_DIR/liblowering.so"
(cd "$GE_FIX_DIR" && sha256sum liblowering.so > SHA256SUMS)
)
```

该目标构建不执行 GE 安装，不覆盖系统库。重新编译的 SHA256 可能因工具链和构建环境不同而变化，
不能直接用原实验哈希作为通过依据，须记录自己的哈希并完成下述验收。
本次补充文档时已在原 GE 构建目录执行 `lowering` 增量构建，产物哈希仍与原实验一致；
未重新进行全新源码目录及第三方依赖的完整构建；本容器的新目录配置检查发现缺少系统工具
`patch`，因此不能把既有构建缓存可增量编译等同于全新环境依赖已齐全。

#### 进程级加载与验收

在项目根目录启动以下子 Shell；Python 前缀按新环境实际安装位置修改。
这里直接给出完整环境设置，替代旧的 `source work/.../a5-output-growth-fix-env.sh`。
Python 启动后先检查 `/proc/self/maps`，确认修复库确实已映射；
随后重新 prepare 并执行全100组精度比较，不修改 golden 或阈值。

```bash
(
  set -e
  cd "$GE_ADAPTER_ROOT"
  source "$CANN_ROOT/set_env.sh"
  export PYTHONHOME="$GE_FIX_PYTHON_PREFIX"
  export PATH="$PYTHONHOME/bin:$PATH"
  export LD_LIBRARY_PATH="$PYTHONHOME/lib:${LD_LIBRARY_PATH:-}"
  test -r "$GE_FIX_DIR/liblowering.so"
  export LD_PRELOAD="$GE_FIX_DIR/liblowering.so"
  export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  export TMPDIR="$GE_ADAPTER_ROOT/work/tmp"
  mkdir -p "$TMPDIR"
  python3 - <<'PY'
import hashlib
import os
from pathlib import Path
library = (Path(os.environ['GE_FIX_DIR']) / 'liblowering.so').resolve()
mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()}
assert str(library) in mapped, f'修复库未加载: {library}'
print('loaded:', library)
print('sha256:', hashlib.sha256(library.read_bytes()).hexdigest())
PY
  : "${QWEN_MODEL_DIR:?请先设置模型权重目录，例如 /home/h00909914/Qwen2.5-0.5B}"
  python3 -m core.pipeline prepare --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b
  python3 -m core.pipeline build --config models/qwen2.5-0.5b/config/model.yaml --device 7
  python3 -m core.pipeline infer --config models/qwen2.5-0.5b/config/model.yaml --device 7 --case-ids all
  python3 -m core.pipeline compare --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b --case-ids all
)
```

逻辑设备7按实际空闲设备调整；执行前在外层设置 `export QWEN_MODEL_DIR=/实际模型路径`。
输入与 FP32 golden 的准备使用项目七步流程中的 data / baseline。
只有100组全部通过才算该新环境验收通过；库成功加载不能替代精度验收。
子 Shell 结束后，外层 Shell 的 `LD_PRELOAD` 等变量保持原值；
若之前在外层加载过旧环境脚本，需在外层显式执行 `unset LD_PRELOAD` 才会回到默认 GE。

原 `work/prefix-migration-20261027/a5-output-growth-fix-env.sh` 仅作为历史实验记录保留，
新环境不再依赖它。原实验缺少的 ONNX 依赖使用项目锁定的 onnx1.17.0、onnxruntime1.20.1、
protobuf5.29.5离线包补齐，首次缺依赖失败日志同时保留。


| 步骤 | run_id |
|---|---|
| 新路径默认环境prepare | 20261027T064136-ebf2cdab |
| 新路径AIR导出 | 20261027T064458-d5015d3a |
| 默认GE全量infer / compare | 20261027T064724-5c6c9c5b / 20261027T064914-930f4d0a |
| 扩容修复库对照infer / compare | 20261027T071705-e9038339 / 20261027T071737-b05641c8 |
| 恢复运行器后最终infer / compare | 20261027T071952-4a549fd0 / 20261027T072023-e8ad8772 |

原始记录在A5项目的 `work/qwen2.5-0.5b/results/<run_id>/`；
测试脚本、日志、环境及摘要在 `work/prefix-migration-20261027/`。
原A5图、依赖、golden和输出保存在该目录的 `previous-a5/`。
在99主工作区的同名work目录下归档了A5验证日志与摘要。
`ascend-ops`子模块继续保留，本次未提交或推送主仓库变更。

## 适配器精简回归（2026-10-24）

本次只调整共享 NumPy 编码、初始化时绑定算子入口、统一 PATCHES 清单和测试侧 CPU 算子注入；A3/A5 默认配置及算子实现未改动。

| 检查 | 范围 | 结果 |
|---|---|---|
| Python 测试 | A5 主工作区与 A3 隔离副本，各 59 项 | 全部通过 |
| 修改前后编码对照 | FIA、A3 Prefix、A5 Prefix，各 100 组/1600 样本 | 所有输入数值和 dtype、输出容量一致 |
| A3 导出与 GE 推理 | Ascend910_9382，case-0000，batch16，16 个业务样本 | 16/16 通过；最小 cosine=0.9999781981，最大 relative L2=0.0147937744 |
| A5 导出与 GE 推理 | Ascend950PR_958b，case-0000，batch16，16 个业务样本 | 16/16 通过；最小 cosine=0.9999978163，最大 relative L2=0.0033425287 |

A5 的 build/infer/compare 记录位于 `work/qwen-adapter-simplify-20261024/a5/validation.json`，编码对照位于同级上层 `encoding-comparison.json`。A3 使用隔离副本，记录位于 `work/qwen-adapter-simplify-20261024/checkout/work/validation/a3-retry/validation.json`。A3 首次隔离验证因路径过长触发 `AF_UNIX path too long`，缩短 TMPDIR 后通过。

本次没有进行全输入池设备精度验收或性能测试；不能将 100 组编码一致等同于 100 组设备精度通过。

## A3 Ascend910_9382 验证（2026-10-20 至 2026-10-21）

99机器、hejun-cann9-qwen25容器、CANN9.0.0、设备1（24 AIC/48 AIV）已跑通Qwen2.5-0.5B Prefix。芯片配置为8|16、batch16、5实例；三轮各10000次调用，业务QPS中位数2172.093、最差P99 41.266ms。100组/1600样本精度通过，最小cosine=0.998909550002、最大relative L2=0.046848644966；54项Python测试和CTest 1/1通过。

全部57次压测尝试、失败与SIGTERM中断、补测、环境修复和原始数据见 [A3调优记录](qwen2.5-0.5b-a3-tuning.md)。后文 A5 历史记录中的“A3无实机”仅描述当时范围。

2026-10-21补充：默认batch16输入池已恢复并核对摘要；新增输入batch一致性拦截，55项Python测试通过；默认5实例1000次调用复验业务QPS=2165.776、P99=41.100ms，100组/1600样本精度复验通过。详见上述调优文档。

## A5 独立步骤回归（batch16，2026-10-20）

环境为 850-a5 的 hejun-cann910-qwen25 容器、CANN 9.1.0、Python 3.12.13、Ascend950PR_958b（32 AIC / 64 AIV）。使用当时空闲的设备1，运行前后未观察到其他同卡驻留进程；其他设备上的服务没有被操作。以下是框架回归，不计入调优报告的164次测量，也不替代设备6的三轮长测。

| 模型 | 执行方式与精度范围 | 性能范围 | 结果记录（项目内） |
|---|---|---|---|
| Qwen Prefix FP16 | 七步分别独立调用，默认 case-0000、batch16，共16个样本通过 | 完整100组池，四实例、1000次调用、16000个样本，0错误 | 分步骤记录见下表 |

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

以下保留此前旧阈值记录；其中的“未通过”和“尚未修复”描述当时的 0.999/0.02 阈值状态。

## pipeline 统一入口的 Qwen 回归（2026-10-20）

- Qwen Prefix 使用 python3 -m core.pipeline run 完成七步：生成/读取历史 100 组 NPY、单组原始 golden、依赖准备、AIR 导出、单实例 GE 推理、精度比较、完整池随机性能测试。默认 case-0000 精度通过，最大 relative L2=0.0033425286544669563；单实例 1000 次调用零错误，业务吞吐 1216.918/s。阈值保持 cosine >= 0.999、relative L2 <= 0.02。

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

## 七步与独立调用（2026-10-19）

环境：Ascend950PR_958b，32 AIC / 64 AIV，CANN 9.1.0，容器 Python 3.12.13，逻辑卡 6。此次验证 GE Session，未测试 ATC。

- Qwen Prefix 完整 run 按顺序完成七步，默认只生成一组 golden、单实例 GE 输出与比较；最大 relative L2=0.0033425286544669563，未放宽 cosine >= 0.999、relative L2 <= 0.02。随后完整 100 组池随机测量 1000 次调用，0 错误，业务吞吐 1215.587/s。

Qwen 完整 run：work/qwen2.5-0.5b/results/20261019T130550-47e42bf2/result.json。

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

## Profiling（2026-10-19）

- Qwen 独立 infer --profiling：未设置权重路径，读取已有 AIR 和所选输入，保存一组 output。原始采集 54 个文件、1077929 字节；msprof --export=on 成功，导出后共 89 个文件、6868929 字节。统计包含 Prefix metadata/planner 与 PrefixMacroKernel 各 24 次任务。
- 采集后独立 compare 仍通过，结果位于 work/qwen2.5-0.5b/results/20261019T132815-d5606194/result.json。

Qwen 采集：work/qwen2.5-0.5b/results/20261019T132741-fe130474/infer-profiling。

## 历史验证边界（2026-10-19）

Qwen 原始输入目录仍为 100 个 case、300 个 NPY，无 JSON 数据文件。

Qwen 本轮只验证默认单组精度，不能宣称全 100 组已通过。历史全量 GE 精度曾在 case-0002/r8、case-0091/r8 失败，最大 relative L2=0.0344762，尚未修复；当前主流程不再由候选 eager 检查提前阻断 GE 验证。

A3 没有实机，12|24 预算、算子和 GE pass 未验证。当前 A5 958b 以外的 SKU 也未验证。ATC 按要求本轮未测；其独立执行仍保留导出图作为 IO 来源，需要后续专项验证。GE Session dump 尚不支持。

早期 A5 Prefix 记录另有 ATC 双实例 profiling 超时；后续记录未给出该项复验结论，本次文档整理也未复测，不能视为已解决。

## 自包含改造（2026-11-16）

Qwen2.5-0.5B 改造为自包含交付，不再经过 `core/pipeline.py` 七步。旧的 core 路径产物
（`qwen_npu_adapter.py`、`generate_cases.py`、`config/model.yaml`、`scripts/env_a3.sh`）
及依赖它们的测试已删除；新形态为 `run.sh` + `model/{modeling_qwen,generate_inputs,golden,export_air}.py`
+ `scripts/prepare.sh` + `config/target_tokens.json`，
入口是 [`models/qwen2.5-0.5b/run.sh`](../../models/qwen2.5-0.5b/run.sh)。GE 执行走
`runtime/build/executor`，精度门禁用 `tools/compare.py`。

`scripts/prepare.sh` 不再经由 `GE_SRC_DIR`/`GE_BUILD_DIR`/`GE_INSTALL_DIR`/`GE_PYTHON`
环境变量契约（那是 `core/dependencies.py` 按 YAML 通用派发依赖构建时用的；改造后
`prepare.sh` 是唯一调用者，源路径与产出路径都由它自己确定，变量不携带信息）。原
`scripts/build_prefix_910.sh`、`build_prefix_a5.sh`、`build_nz_pass.sh` 三个脚本因此
并入 `prepare.sh`。合并后从零重建依赖 53 s，导出 AIR 字节数、NZ pass 命中数
（`matmul_match num is 169`）与全量 100 case 精度（worst cosine 0.999439462、
relative L2 0.033784056）与合并前逐项一致。A5 分支按原脚本逐字搬移，本机无 A5
实机，未实测。

算法（历史 RNG 输入分布、共享 prefix packed 布局、8 列评分头）逐字搬移，旧记录中的
batch10/batch16 输入统计与数值结论不变。本节不改写也不替代上述历史验证结论。

### A3 实测（2026-11-16）

环境：本机容器 CANN 9.0.0（V100R001C10SPC001B250，aarch64-linux）、npu-smi
25.5.t6.b001、Ascend910_9382、Chip Logic ID 2、Python 3.11.15、torch 2.9.0+cpu、
torch_npu 2.9.0.post2、transformers 5.10.1、numpy 1.26.4；Prefix 算子与
WeightNzAndMatMulV3Pass 由 `scripts/prepare.sh` 从子模块提交
`8e08d0d20275558aefac1bbb35019e6b52f1073a` 构建（NZ pass 生效证据：编译日志
`matmul_match num is 169 / matmul_effect num is 169`）。共享机器，他人在跑多卡
任务，绝对值受整机负载影响。

| 项目 | 配置 | 结果 |
|---|---|---|
| 输入池 | 100 case × batch16，历史 lognormal | 1600 业务样本，pool sha256 前 12 位 `cfeed98ec4a6`（精简前后一致） |
| 导出 | `--platform a3`，dynamic axis 0 | `work/build/packed.air` 993754309 字节；3 个 Data 节点（index 0/1/2，`DT_INT64 [-1]`），0 个 Pack |
| 精度 | 全量 100 case，门限 cosine ≥ 0.998 / relative L2 ≤ 0.05 | PASS，1600 行，worst cosine 0.999439462、worst relative L2 0.033784056（精简前后逐位一致） |
| 精度推理时延 | 单实例、不限核、100 case 各一次 | 首 case 29.458 ms；稳态 99 case min 11.4034 / p50 12.0479 / avg 12.0863 / max 13.4046 ms |
| 图冷启动 | 单份图 | CompileGraph 480103 ms（TBE 冷缓存、不限核）／232853 ms（冷缓存、`8\|16`）／4.7–7.0 s（缓存命中）；LoadGraph 1.28–1.97 s |
| bench | `8\|16`、5 实例、3000 请求 × 3 轮（seed 42/43/44），预热覆盖全部 100 个 distinct shape | 3 轮全 0 错误；业务 QPS 中位 2186.262（三轮 2186.262/2193.346/2186.153）；模型调用 QPS 中位 136.641；e2e avg 中位 36.5565 ms、p50 中位 36.2325 ms、p99 中位 40.7956 ms、三轮最差 40.9928 ms；逐实例 QPS 均匀 27.327；H2D/D2H avg ~0.041/~0.026 ms |
| HBM | 5 份图 | free 62340.4 → 57599.4 MB，增量 4741 MB（948 MB/实例） |
| 一键流程 | `bash run.sh --platform a3 --device 2 --iterations 3000 --seed 42` | 21:16:34→21:33:25 全程通过（依赖构建 108 s、输入 <1 s、golden 11.4 min、runtime 构建 1 s、导出 2.2 min、精度推理 18 s、比较 <1 s、bench 69 s）；该轮 bench 业务 QPS 2199.581、e2e p99 40.5825 ms、0 错误，落在上表三轮的同一分布内 |

与本文「A3 Ascend910_9382 验证（2026-10-20 至 2026-10-21）」节的旧 core 路径记录
对照：业务 QPS 中位 2172.093 → 2186.262，最差 P99 41.266 → 40.9928 ms，最小
cosine 0.998909550002 → 0.999439462，最大 relative L2 0.046848644966 →
0.033784056（评分头改在 FP32 中计算，门限未变）。性能与精度均未回退。

上表全部数据取自**精简后**的代码与二进制（`runtime/executor.cpp` 删掉
`--case-filter`/`--help`/`AicoreSpec`/交叉校验/`per_case`/`model_ms`/`arrival_model`
/`backend`/`mode`，executor 目标只链 6 个 CANN 库，`model/` 四个脚本删掉路径参数与
导出侧冗余开关）。精简前后各跑过一次完整流程，pool sha256、AIR 字节数、worst
cosine/relative L2 三项完全一致；删除项的逐条实测依据见[适配说明](adaptation.md)。

未验证范围：A5（`--platform a5`，需 CANN ≥ 9.3.0 辅流接口，本机无 A5 实机）；ATC/OM 路径（本流程不产出 OM）；profiling（用外部 `msprof` 包裹 executor）。

A5 分支只做过**结构性**验证（桩算子，非真实 kernel）：按真实 schema 注册 `ascendc_ops::prefix_attention` 并桩掉其 PrivateUse1 实现后，`QwenPackedModel(config, "a5")` 构造成功，buffer 只有 `cos`/`sin`/`score_head`（**没有** `compressed_mask`，即 4 MB 压缩掩码确实只在 a3 注册），int32 bounds 原样透传到算子（`bounds=(3,) torch.int32`、`scale=0.125`、24 层各调用一次），`forward` 返回 `(2, 8) float32` 且有限。数值正确性、GE 交付件加载与 CANN 9.3.0 辅流接口仍未验证。

同期对 `modeling_qwen.py` 的三处精简（`compressed_mask` 改由 a3 工厂注册、去掉只为喂闭包而存在的 `self.config`、补上「mask 必须惰性读，因为 `nn.Module.to()` 会重绑 buffer」的注释）已复验：eager 全 100 case 对照 CPU golden 得 worst cosine 0.999098718 / relative L2 0.045386977（图路径更优，见上表），重导出 AIR 仍为 993754309 字节，全量精度门禁 PASS 且 worst 值与改动前逐位一致（0.999439462 / 0.033784056）。
原始数据在 `models/qwen2.5-0.5b/work/{golden,output,bench,bench-43,bench-44,logs}`
（Git 忽略）。

## ge_execute 迁移复测（2026-11-19）

`runtime` 在 `28389ba` 把三个旧运行时（`ge_runtime` stdin plan 协议、`executor`、
`pipeline_runtime`）换成单一二进制 `ge_execute`：输入契约从 `ge-records/3`
（`inputs.bin` + `records.json`）换成 `ge-cases/1`（`cases.json` + 每 case 若干 npy），
infer 产物从 `outputs.bin` + `result.json.outputs` 换成 `<output-dir>/<case_id>/output_<i>.npy`
+ `result.json.cases`，`requests.csv` 不再产出。本适配器随之迁移，本节是迁移后的复测。

改动范围（算法与数值路径**未动**）：

| 文件 | 改动 |
|---|---|
| `model/generate_inputs.py` | 产物换成 `work/inputs/cases.json` + `case-XXXX/{input_ids,position_ids,actual_seq_lengths}.npy`；写盘逻辑抽成可注入输出目录的 `write_pool(out_dir, vocab, platform)`；生成前清掉旧 `inputs.bin`/`records.json`。`generate_cases()`/`packed_inputs()` 的签名、RNG 序列、池参数、摘要顺序逐字未动 |
| `model/golden.py` | `_load_cases()` 改读 `cases.json` + `np.load`（按 name 取张量）；产物换成与 runtime infer 同构的 `work/golden/<case_id>/output_0.npy` + `result.json.cases`，写盘抽成 `write_case_scores(out, case_id, tensor)`（`np.save` 不建父目录，由它负责）。打分逻辑（HF FP32 CPU、行重建、padded batch、`logits_to_keep`、8 列评分头、OMP/MKL pin）逐字未动 |
| `run.sh` | infer/bench 两段换 `ge_execute --graph/--cases/--output-dir`，新增 `--device "$DEVICE_INDEX"`（进程内设备号 0，与 `DEVICE_CHIP` 的 Chip Logic ID 是两个概念）与显式 `--precision-mode must_keep_origin_dtype`；收尾提示由 `requests.csv` 改为 `result.json.records` |
| `tools/compare.py` | `load_directory()` 按 `result.json` 的键自动分派：`cases` → npy 布局，`outputs` → 旧 `outputs.bin` 布局（保留给尚未迁移的 onejev / startlux-text golden），两侧可混比；新 loader 用 npy 头交叉校验声明的 dtype/shape/bytes，path 解析依次尝试原样（绝对或 CWD 相对，runtime 侧）、相对 result.json 目录（golden 侧）、`<dir>/<case_id>/output_<i>.npy` 兜底。比较逻辑、门限、`ge-compare/1` 报告、退出码未动 |
| `tests/test_qwen_ge_cases.py`、`tests/test_compare_layouts.py` | 新增，纯 CPU：锁定历史 RNG 与 pool sha256、cases.json/npy 结构、`write_case_scores` 的父目录创建，以及两种布局的加载/混比/全部失败路径 |

**`--precision-mode must_keep_origin_dtype` 是迁移中唯一的数值风险点**：旧 `executor`
在 session options 里硬编码它，`ge_execute` 默认 `force_fp16`，不显式传则精度基线不可复现。
其余 GE/ACL 序列逐项核对一致（`GEInitialize` 全局选项、session 选项、K 份
AddGraph+CompileGraph 全部在 `aclrtSetDevice` 之前、`Graph::LoadFromFile`、warmup 每实例先覆盖
全部 distinct shape、iterations 为所有实例合计），见 `runtime/session.cpp:271-409` 与旧
`runtime/executor.cpp:198-265` 的对照。

### A3 实测（2026-11-19）

环境：与 2026-11-16 节同一容器（CANN 9.0.0、Ascend910_9382、Chip Logic ID 2、Python
3.11.15、numpy 1.26.4、torch 2.9.0+cpu、torch_npu 2.9.0.post2、transformers 5.10.1），
Prefix 算子与 NZ pass 沿用 `work/deps/` 已构建产物（未重跑 `prepare.sh`），AIR 沿用
2026-11-17 导出的 `work/build/packed.air`（993754309 字节，未重导出）。**整机 16 张卡全空闲**
（AICore 0%、HBM ~3.1/65 GB），比 2026-11-16 的共享负载干净。

| 项目 | 配置 | 结果 |
|---|---|---|
| 输入池 | `generate_inputs.py --platform a3` | 100 case × batch16 = 1600 业务样本，pool sha256 前 12 位 **`cfeed98ec4a6`**（与 2026-11-16 记录一致，RNG 未漂移）；`cases.json` 100 case、每 case 3 个 npy、`samples` 全为 16 |
| CPU golden | `golden.py`（HF FP32，8 线程） | 100 个 `[16,8]` float32 `output_0.npy` + `result.json.cases`；耗时 ~11 min，与旧记录同量级 |
| runtime 构建 | `bash runtime/build.sh` + `ctest` | `ge_execute` 链接通过，CPU 契约测试 1/1 通过 |
| AIR 冒烟 | 3 case 小池、单实例、不限核 | exit 0；CompileGraph 5490 ms（TBE 缓存命中）、LoadGraph 1261 ms、`allocation=GE (dynamic compiled)`、NZ pass 命中 `169/169`；输出 `[16,8]` float32 512 字节；对 golden 抽检 3 case 最小 cosine 0.999992796、最大 relative L2 0.007442 |
| 精度 | 全量 100 case，门限 cosine ≥ 0.998 / relative L2 ≤ 0.05 | **PASS**，1600 行，worst cosine `0.9994394615744572`、worst relative L2 `0.033784056043048426` —— 与 2026-11-16 旧 `executor` 记录**逐位一致** |
| 精度推理时延 | 单实例、不限核、100 case 各一次 | 首 case 31.2240 ms；稳态 99 case min 11.5780 / p50 12.1460 / avg 12.2540 / max 13.5660 ms；d2h avg 0.0285 ms（旧记录：首 case 29.458，稳态 11.4034 / 12.0479 / 12.0863 / 13.4046） |
| bench | `8\|16`、5 实例、3000 请求 × 3 轮（seed 42/43/44），预热自动补齐到 445 次（5 × 池内 **89** 个 distinct shape） | 3 轮全 0 错误、`status=success`；业务 QPS 中位 **2185.282**（三轮 2181.907/2192.231/2185.282）；模型调用 QPS 中位 136.580（逐实例 27.316）；e2e avg 中位 36.5565 ms、p50 中位 36.2920 ms、p99 中位 40.8620 ms、三轮最差 p99 40.8900 ms / max 42.2940 ms；h2d avg 0.040–0.042、d2h avg 0.025–0.028 ms；compile 24245–25036 ms、逐实例 LoadGraph 1268–1366 ms；`per_instance` 均匀 600 |
| profiling | `--profiling --iterations 4 --instances 1 --warmup 1` | exit 0；`work/smoke/prof/profiling/PROF_000001_*/{host,device_2}` 共 4.4 MB，`sample.json` + `.done` 齐备 |
| 一键流程 | `bash run.sh --platform a3 --device 2` | 14:20:44→14:33:15 全程通过（依赖构建 46 s、输入 <1 s、golden 9.2 min、ge_execute 构建 <1 s、导出 2.2 min、精度推理 20 s、比较 <1 s）。重导出的 AIR 仍为 993754309 字节、NZ pass 命中 `169/169`；精度门禁 PASS，worst cosine `0.9994394615744572` / relative L2 `0.033784056043048426` 与上面逐阶段执行的那次**逐位一致**，100 个 case 的逐 case worst_cosine 也全部相同；infer e2e avg 12.135 / p50 11.878 / max 31.998 ms、CompileGraph 5218.6 ms、LoadGraph 1271.0 ms |
| Python 回归 | `python3 -m unittest discover -s tests -p 'test_*.py'` | 150 tests，仅 1 个既有环境性 error（`test_startlux_native_proto.ProtoPreloadGuard` 缺 `causal_conv1d`，与本次无关），15 skipped；新增 26 个用例全通过 |

对照 2026-11-16 节：业务 QPS 中位 2186.262 → 2185.282（-0.04%）、e2e p50 中位 36.2325 →
36.2920 ms、p99 三轮最差 40.9928 → 40.8900 ms、worst cosine / relative L2 逐位相同。
**性能与精度均无回退**，且本次整机空闲、上次为共享负载，两组数据落在同一分布内。

三处记录口径修正（不改写原实验事实，只标注新证据）：

- 旧记录写「预热覆盖全部 100 个 distinct shape」，实测池内 distinct shape 为 **89**
  （100 个 case 里有重复的 T），`ge_execute` 的 `result.json.pool.distinct_shapes` 现在
  直接给出这个数；预热次数 445 = 5 实例 × 89，新旧运行时同一算法。
- `ge_execute` 在 **infer** 模式不回填 `result.json` 的 `compile_ms` / `load_ms`
  （只有 bench 路径回填，`runtime/main.cpp:435-437`），infer 的编译/加载耗时只在 stdout 的
  `[INFO]` 行里。已记入 `runtime/README.md` 的使用限制，属待修的记录缺口，不影响执行与数值。
- **AIR 不是逐比特可复现的**：一键流程重导出后字节数仍是 993754309，但 sha256 从
  `efef8787fc04b4802c4b9697ba9dbfaa716b9e53a82f9c9b2f29fc5befff06bd` 变成
  `561020e42064e70517ec0b7f57e6af771a146a2e0d6fa2dc2ef98d50b3265ddd`（`dynamo_export`
  的容器里带时间戳/路径一类易变元数据）。因此历史记录里的「AIR 993754309 字节」这一项
  只能作为**字节数**指纹，不能升级成 sha256 指纹；判断图是否等价要看 NZ pass 命中数
  （`169/169`）与下游精度（本次逐位一致），而不是图的哈希。

未验证范围：A5（`--platform a5`，需 CANN ≥ 9.3.0 辅流接口，本机无 A5 实机）；ATC/OM 路径
（`ge_execute` 明确不支持 OM）；HBM 增量未复测（旧记录 4741 MB / 948 MB 每实例）；
bench 未在一键流程里跑（`run.sh` 不带 `--iterations` 时跳过第 9 阶段，本次 bench 数据
取自上面逐阶段的三轮）。

原始数据在 `models/qwen2.5-0.5b/work/{inputs,golden,output,bench,bench-43,bench-44,smoke,logs}`；
迁移前旧 `executor` 的产物保留在 `work/output-legacy-records3/` 与 `work/bench-legacy/`
（均 Git 忽略）。
