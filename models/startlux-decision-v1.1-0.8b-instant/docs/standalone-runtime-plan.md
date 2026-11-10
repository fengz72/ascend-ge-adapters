# StartLux 独立脚本与 C++ runtime 执行方案

日期：2026-11-04。状态：**已实施**（2026-11-05 完成迁移并通过全流程验收）。
本文保留原始设计与逐项对齐的决策记录；下方目录树为当时的目标结构，实施时
Python 文件按职责归入 `model/` 与 `decider/` 子目录（见 validation.md 的
2026-11-05 目录重组条目），且 2026-11-10 的代码审查清理删除了不在 `run.sh`
调用图内的模块（`decision.py`、`reference.py`、`ge_readout.py`、
`unified_readout.py`、`fixtures/`、`scripts/monitor_idle_benchmark.py`、
`scripts/env_chunk_proto.sh`）。当前实际结构以 README 的目录清单为准。

## 1. 目标与范围

将模型整理成独立示例：Shell 串行调度，Python 直接完成数据、参考计算和 AIR 导出，现有 C++ runtime 完成推理与性能，公共工具完成精度比较。不依赖 `core`，不保留 YAML、adapter、handle 等流程框架接口。

原则是约定大于配置：采用固定目录、固定执行顺序和固定测试参数，不增加插件发现、配置覆盖、自动重试或断点调度。迁移已验证实现，不借此重写模型算法。

先沿用已验收的 native ChunkGatedDeltaRule 路线，不在本次迁移中同时切换 Mega 算子，也不设计多路线运行时配置。Mega 路线另行验证后再决定替换。

## 2. 目标目录

以下结构是迁移后的目标，不表示这些入口当前均已存在。

```text
ascend-ge-adapters/
├── runtime/                         # 复用现有 C++ runtime
├── tools/
│   └── compare.py                   # 公共精度比较工具
└── models/startlux-decision-v1.1-0.8b-instant/
    ├── README.md
    ├── run.sh                       # 完整流程入口
    ├── prepare.sh                   # 构建安装 pass 与自定义算子
    ├── model.py                     # 单图模型组合入口
    ├── export_air.py                # 导出 AIR 与运行请求
    ├── generate_inputs.py           # 生成输入及用例索引
    ├── golden.py                    # 官方 CPU 参考
    ├── verify_business.py           # 业务概率门禁（迁移自 unified_verify.py）
    ├── multimodal.py                # 以下为 model.py 复用的已验证模块，保持原位
    ├── startlux_npu_adapter.py
    ├── vision.py
    ├── decision.py
    ├── reference.py
    ├── unified_preprocess.py
    ├── scripts/                     # env_python.sh、算子构建与显式加载、P8 监控存档
    ├── fixtures/                    # 固定测试材料
    ├── vendor/                      # 必需的官方参考代码
    ├── native_proto/                # 当前路线需要的兼容原型
    └── work/                        # 全部生成产物，不提交 Git
        ├── inputs/
        ├── golden/
        │   ├── outputs.bin
        │   └── result.json
        ├── build/
        │   └── model.air
        ├── infer/
        │   ├── request.json
        │   ├── outputs.bin
        │   ├── result.json
        │   └── comparison.json
        ├── bench/
        │   ├── request.json
        │   ├── result.json
        │   └── requests.csv
        ├── deps/
        └── logs/
```

模型侧保留五个 Python 入口（model、export_air、generate_inputs、golden、verify_business）、两个 Shell 入口。计算实现通过 import 复用上述既有已验证模块，不物理合并；`vendor/` 和必要的原型源码不为减少文件数而展开复制进 `model.py`。

新的模型内 `work/` 与现有工程根目录的 `work/startlux-decision/` 分开；迁移期间保留旧 AIR 和全部验收证据。实施时将模型内 `work/` 加入 Git 忽略规则。

## 3. 环境和参数约定

权重路径沿用环境变量，不复制权重到模型目录，不自动下载或搜索候选路径：

```bash
export STARTLUX_MODEL_DIR=/path/to/hf-model
```

`model.py` 和 `golden.py` 读取同一个变量。`run.sh` 开始时检查变量已设置、目录存在；不做依赖 import 预检，缺包由对应阶段自然报错。

| 项目 | 约定 |
|---|---|
| 工作目录 | 本模型目录下的 `work/` |
| 权重 | `STARTLUX_MODEL_DIR` |
| 官方预处理依赖 | 任务局部 Pillow/torchvision，环境前置，安装方式见 README |
| 后端 | `ge_session` |
| AIR | `work/build/model.air` |
| 设备 | 在 `run.sh` 顶部固定设备号，区分可见设备映射与进程内编号 |
| 精度门限 | cosine ≥ 0.999、relative L2 ≤ 0.02，逐样本验收 |
| 性能参数 | 单实例、warmup32、总测量请求2048、seed42 |

其他参数固定在 `run.sh` 顶部，不提供 YAML、参数覆盖优先级或自动发现。`run.sh` 根据自身位置定位模型目录和工程根目录，从任何当前目录启动都一致。

环境沿用既有已验证组合；现有版本与限制见模型 README。移除流程框架不意味着更换 CANN、Torch、TorchNPU、Transformers 或 SoC。官方预处理依赖的任务局部 Pillow/torchvision 属环境前置条件，README 记录已验证安装路径；不做 import 预检，缺包时保留现有 DEPS_HINT 报错提示。

## 4. 各入口职责

### 4.1 model.py

- 组合视觉塔、图像特征注入、三轴 MRoPE、文本 decoder 和26列字母读出头，提供直接的权重加载函数和一个完整 `nn.Module`，最终导出一张 AIR。
- 保留已验收的视觉 FP32、文本 BF16、A_log 从 checkpoint 恢复 FP32、FP32 读出头及现有算子实现。
- 不包含 pipeline adapter、handle、配置解析和业务调度。
- 计算实现通过 import 复用 `multimodal.py`、`startlux_npu_adapter.py`、`vision.py` 等已验证模块，不物理合并、不复制源码；这些模块及其 CPU 单测保持不动。官方参考代码保留在 `vendor/`。

图边界维持 B≤4、S≤2048、最小物理宽度256、每次调用全局最多两张图片；使用已验收的固定图像处理网格及双图槽位。不截断超限请求。

### 4.2 generate_inputs.py

- 读取固定 fixtures，调用官方预处理。
- 保存预处理后的 Tensor、必要的原始用例信息和输入索引。
- 固定定义精度用例集及性能输入池，二者可以复用输入文件。
- 精度用例集包含 B4 混合有效长度用例（行长 1/65/129/2048），与普通用例同样生成、同样参与 golden 与逐样本比较，不设单独补验步骤。
- 组批与编码时强制图边界（B≤4、S≤2048、最小物理宽度256、全局最多两张图片），超限拒绝；边界值写入 README 使用限制。
- 不让 golden 和导出阶段分别重复预处理。

模型边界、输入含义及 IO 定义保留明确的单一来源，不在多个脚本复制同一套常量：边界常量沿用 `multimodal.py` / `vision.py` 顶层的既有定义，新入口直接 import。用例索引由数据阶段生成，实际图节点信息由导出阶段生成；`golden.py` 不导入这些模块（26 列字母 id 与温度读官方 `decision_config.json`）。

### 4.3 golden.py

- 从相同权重加载独立的官方 CPU 参考实现。
- 使用已保存的预处理结果，生成与 AIR 一致的26列 logits。
- 统一读出位置、字母顺序、有效长度、padding 和图片语义。
- 不调用 `model.py` 的适配计算逻辑，不导入 NPU 注册模块。
- 输出 `outputs.bin` 和 `result.json`，结构与 runtime infer 结果完全同构（顶层 `status`/`mode`/`backend`，`outputs` 按 case→rows→tensors），比较工具统一处理两侧、无需特判。

参考计算独立不等于重复实现预处理。CPU FP32 数值比较与既有 BF16 业务分支差异分别记录，不扩大原有验收结论，不通过放宽门限消除差异；已有精度事实仍以 `validation.md` 为准。

### 4.4 export_air.py

- 加载 `model.py`，直接调用 TorchAir 导出，不导入 `core` exporter。
- 显式导入所需 pass、converter 和算子注册。
- 保留动态 B、S，不能将示例输入尺寸固化。
- 从实际导出结果取得节点名称、顺序和 dtype，不手写可能漂移的节点映射。
- 沿用现有 IO 嵌入能力及已验证的导出逻辑，必要时调用 runtime 的 `write_io` 模式。
- 自动生成 `infer/request.json`、`bench/request.json`，用户不编辑请求 JSON。

两个请求使用同一张 AIR。显式设置既有 GESession 选项，包括 `graph_run_mode` 和 `must_keep_origin_dtype`，不依赖 runtime 默认精度策略。

### 4.5 prepare.sh 与环境设置

`prepare.sh` 只负责构建、安装必要 pass 和自定义算子，复用现有构建逻辑，模型专用安装产物放在固定的 `work/deps/`。构建缓存可以沿用原有构建系统的位置，不额外实现缓存管理。

`run.sh` 显式设置后续子进程需要的库路径、Python 路径及算子加载变量。不能在子 Shell 内 `export` 后假定父进程环境已改变。A3 容器 Python 引导从 `models/qwen2.5-0.5b/scripts/env_a3.sh` 复制收敛为本模型 `scripts/env_python.sh`，不引用其他模型目录。

继续使用当前已验证的显式局部加载方式，避免兼容原型重复注册。确实需要系统安装的操作在 README 和执行输出中明确说明，不隐藏在普通导入过程中。

### 4.6 verify_business.py

- 迁移 `unified_verify.py` 的业务概率校验逻辑到新路径，纯 CPU NumPy，不加载权重、不碰 NPU。
- 读取 `STARTLUX_MODEL_DIR` 的官方 `decision_config.json`（26 个字母 token id 与各题型温度）和用例 rows_meta，对 `infer/` 与 `golden/` 的 26 列 logits 做校验。
- 公共比较工具保持模型无关，业务语义不并入 `tools/compare.py`。

## 5. 完整执行顺序

```text
检查环境及权重路径
    ↓
生成输入
    ↓
CPU golden
    ↓
构建／安装 pass、自定义算子，设置后续环境
    ↓
构建 C++ runtime
    ↓
导出单 AIR，生成 IO 与运行请求
    ↓
C++ GESession 精度推理
    ↓
公共工具比较精度
    ↓
业务概率校验
    ↓
人工确认卡空闲（打印提醒并等待确认）
    ↓
C++ GESession 性能测试
```

各 Python 阶段使用独立进程；CPU golden 在本流程 NPU 依赖注册前完成。复用 `runtime/build.sh` 构建 `runtime/build/ge_runtime`，不另建模型专用 C++ runtime。

C++ runtime 保留现有 stdin JSON 接口：

```bash
"$RUNTIME" < "$WORK/infer/request.json"
"$RUNTIME" < "$WORK/bench/request.json"
```

上述变量由 `run.sh` 定义。脚本将 stdout 中的 `GE_RESULT` 保存为对应目录的 `result.json`，完整 stdout/stderr 留在 `work/logs/`。不修改 C++ 命令行接口以适配新的调度方式。

## 6. 结果格式与公共精度比较

golden 与推理结果采用兼容的 Tensor 索引：case、输出名称、dtype、shape、offset、bytes，复用 runtime 当前的输出结构（行内 `instance` 字段固定为0，保持同构）；golden 不需要模拟 bench 专属的计时与性能状态字段。

```text
golden/：outputs.bin + result.json
infer/ ：outputs.bin + result.json
```

比较命令固定在 `run.sh`：

```bash
python3 "$ROOT/tools/compare.py" \
  --golden "$WORK/golden" \
  --output "$WORK/infer" \
  --cosine-min 0.999 \
  --relative-l2-max 0.02
```

`tools/compare.py` 仅依赖 NumPy，不导入模型模块、Torch、TorchNPU 或 `core`。现有实现是依赖 `core.verify` 的 .npy 位置参数工具，仓库内无其他使用方，原地重写为新 CLI 并删除旧接口。第一版只支持当前需要的 F32 输出，不预先设计多格式转换、指标插件和输出策略配置。

比较行为：

1. 按 case 和输出名称匹配，检查用例、输出、shape、索引与二进制数据完整性。
2. 拒绝 NaN、Inf。
3. 转 FP64，沿最后一维逐行计算 cosine 和 relative L2；当前 `[B,26]` 即逐样本验收，避免整体指标掩盖单行失败。
4. 明确处理零向量，不通过随意添加 epsilon 将不满足条件的结果变为通过。
5. 所有行同时满足两个门限才通过。
6. 控制台打印最差指标和失败项，保存 `infer/comparison.json`；通过返回0，失败返回非零并阻止性能阶段继续。

### 6.1 业务概率门禁

数值比较通过后运行 `verify_business.py`（与 P7 验收同口径）：按 rows_meta 与官方 `decision_config.json`，对 K 个有效选项列在官方温度下做 softmax，检查候选与 golden 的 max |Δp| ≤ 0.01、参考 top1-top2 概率间隔 > 0.02 时候选 top1 一致、全部 26 列有限。失败返回非零并阻止性能阶段。

## 7. 性能测试

性能输入池与精度用例集分别固定。正式计时前应覆盖输入池实际形状，确认现有 runtime 预热机制有效，测量记录无首次编译等污染。warmup32 为每实例预热请求数、不消耗测量序列（已按 runtime/bench.h 现有实现核对）；2048指总测量请求数。

原始 `requests.csv` 保留在 `work/bench/`，不新增 CSV 汇总文件。性能结果及口径写入英文文件名文档的附录表格，并链接原始证据。

报告明确区分请求数、graph executions/s、batch/readout 行数和 rows/s；readout 总数从原始请求的 samples 列加权计算，不把每请求 samples 的统计对象当作总行数。保留各用例延迟和整体吞吐。

计时范围继续是 warm graph 的 H2D＋描述符更新＋execute＋sync，不包含官方 CPU 预处理和 D2H，不称完整业务 QPS。HBM 阶段采样不称峰值。

精度通过不代表卡已空闲。进入性能测试前 run.sh 打印提醒并等待人工确认卡空闲；不内置利用率轮询、不接入后台监控、不绕过确认启动第二份 benchmark。（原 `scripts/monitor_idle_benchmark.py` 曾保留为 P8 历史证据与手动工具，2026-11-10 清理时删除：它不在本流程调用图内，P8 监控数值已存档于 performance.md。）

## 8. 运行、重跑和失败行为

目标完整执行命令：

```bash
export STARTLUX_MODEL_DIR=/path/to/hf-model
bash models/startlux-decision-v1.1-0.8b-instant/run.sh
```

`run.sh` 使用 `set -euo pipefail`，任一步失败立即停止。每一步开始时只移除自己将生成的最终结果，失败后不能误用旧 AIR、旧 golden 或旧推理结果；同时保留失败日志，不自动清空整个 `work/`。

依赖构建复用已有缓存，不反复清理。需要单独重跑时直接调用对应 Python 文件或 C++ 请求，README 给出命令。不新增阶段调度器、自动重试、断点恢复或运行记录数据库。

## 9. 迁移与验收

迁移顺序：

1. 保留现有源码、正式 AIR 和全部验收证据，先实现独立入口及模型内新 `work/`。
2. 迁移现有已验证计算与导出逻辑，不同时修改第三方算子实现。
3. 完成 CPU、单 AIR 精度和空闲条件下的性能验证。
4. 全部通过后，再移除旧 pipeline 专用配置和包装代码：删除 `config/model.yaml`、`config/model-mega.yaml`、`unified_adapter.py`、`unified_verify.py`、`verify.py`、`generate_unified_cases.py`、`scripts/run_unified_performance.py`；保留被复用的 `multimodal.py`、`startlux_npu_adapter.py`、`vision.py`、`decision.py`、`reference.py`、`unified_preprocess.py`、`scripts/monitor_idle_benchmark.py` 及其既有 CPU 单测；业务门禁逻辑完成向 `verify_business.py` 的迁移后删除原文件。实施修正（2026-11-05）：`ge_readout.py` 与 `unified_readout.py` 一并保留——保留的 `unified_readout.py`（GESession 读出会话）在模块级依赖 `ge_readout.py`，二者及其单测不在删除范围。调整受影响的测试引用（删除被移除文件对应的 test_startlux_unified_adapter / test_startlux_unified_verify / test_startlux_verify / test_startlux_unified_cases），此前不提前删除当前正式入口。实施修正（2026-11-06）：为满足"独立入口不直接或间接依赖 `core`"的验收项，`core.backends.runtime.execute`（ge_runtime 子进程协议，未用的 abort 钩子已去除）与 `core.contracts.validate_io`（IOSpec 校验）自 `ebb6fbd` 内联进 `ge_readout.py` 并标注来源；`unified_readout.py` 改用 `ge_readout.validate_io`，二者不再 import `core`。实施修正（2026-11-10）：代码审查确认 `decision.py`、`reference.py`、`ge_readout.py`、`unified_readout.py`、`scripts/monitor_idle_benchmark.py` 均不在 `run.sh` 的调用图内（按 `_load_module` 精确 import 图验证），仅被各自单测引用，已连同 `tests/test_startlux_{decision,ge_readout,unified_readout}.py` 一并删除；`scripts/env_chunk_proto.sh` 从未被 source 且路径与 `prepare.sh` 的实际安装位置不一致，同样删除。2026-11-05 与 11-06 两次"保留"修正随之失效。`decider/vendor/` 的 13 个官方模块按用户决策全部保留。详见 validation.md 的 2026-11-10 条目。

验收项：

- 独立入口不直接或间接依赖 `core`。
- 官方 CPU golden 与 NPU 使用相同预处理输入及读出语义，参考计算独立。
- AIR 为包含完整数值前向的单图；动态尺寸和既有边界用例通过，包括精度用例集内的 B4 混合有效长度用例。
- 精度逐样本通过既有门限，业务概率门禁通过，跨 dtype 业务差异保持原有披露。
- 性能符合空闲及预热要求，无首次命中污染，原始记录完整，计时范围准确。
- 从不同当前目录启动、单独重跑推理及比较均正常，不误用上一次失败前留下的产物。
- README 更新为真实已实施入口、环境和证据路径；新方案验收前仍保留现有流程说明。

本方案不要求实现额外通用框架；验收聚焦迁移后计算、IO 和执行语义的等价性。
