# BGE 固定 / 随机双场景独立审查

2026-10-26。结论：**工作流测量约定、本地准备 CLI 和新版独立验收器的 CPU 准备通过；远端集成与固定场景设备性能均未验收。** 本次没有使用 SSH、NPU 或 PID observer；没有修改历史证据、模型、样例、expected、阈值或冻结 C++。

## 契约与材料核查

已读取工作流入口、测量/需求约定、需求/设计/调优记录/交付模板，以及 BGE brief、design v4、acceptance、progress 和 workload-contract-v2.json。七份工作流文件与 `/Users/ext.hejun23/Documents/workspace/skills/npu-model-workflow` 对应文件逐字节相同。

固定场景选 `case-0002`，理由是原始长度 `[6,9,8,13,8]` 的中位长度为 8，并取原顺序第一个匹配样例，选择发生于固定场景计时前。独立从归档原始 17,304 字节输入包按原计划提取八张量，逐一核对 dtype、shape、字节数和 SHA，固定池为 3,448 字节，SHA 为 `27ce13c73cb35762ee7d0396d4425829d7e25a3b8ae2a12b37ed7f96b8690aa5`。同为 S8 的 case-0004 内容不同，不能互换。

约定明确两场景独立建基线、分别计算收益与 p99；默认最终采用同一冻结配置，场景专用配置需分别验收。固定输入换 seed 不产生新输入；固定池必须实际只有一条，不能由 random-five 分组时延推算固定吞吐。全部历史性能仍属于 random-five，175.321044 QPS 基线与 461.333304 QPS 八实例搜索值没有改名或混入固定结果。

完整五例数值/Top10、每实例×case 至少 20 次的请求输出压力，以及至少 600 秒随机混合 shape 稳定性均保留。双场景增加的运行计入既有 7,200 秒预算，不新增时长；同配置换场景不新增唯一配置名额。设计要求重估固定/随机×基线/最终×三轮共十二个正式 run，加全五例压力、稳定性和 20% 余量，且至少预留 50 分钟。当前余额 5,583.320090 秒；**恢复前仍须根据实际配置更新数值预算估算**，不能把旧 50 分钟自动当作新增场景已获运行准入。

## 准备 CLI 独立 CPU 验证

审查对象为 `work/bge-opt-20261026/measurement-src/models/bge/replay.py`、对应 CLI 测试和 README。`--case-id` 真正过滤传给 encode 的池；省略时保留五例。未知 ID 在创建输出目录及 adapter 准备之前被拒绝；固定选择仍要求完整五例源池，旧目录拒绝覆盖。计划记录 workload kind、实际 case_ids 和采样规则；输出接口与计时参数保持。

独立执行原十二项回归，**12/12 通过**。覆盖固定原始 payload、五例默认池、未知/缺失 case、两种 seed 下固定字节不变、拒绝覆盖、错误类型/shape、原 allocator 的 native-only preload 等。测试禁止 native call/权重加载，并确认 runtime 子目录没有创建。静态核对既有 Sampler 的 poolSize=1 分支返回索引 0；八个冻结 C++ 文件 SHA 与 measurement-identity.json 全部相同。

本地环境为 Python 3.13.3、NumPy 2.5.1、PyYAML 6.0.3。PyYAML 使用任务隔离的绝对 PYTHONPATH。实际命令在 measurement-src 中执行：

```bash
PYTHONPATH=/Users/ext.hejun23/Documents/Codex/2026-10-19/ssh/work/bge-opt-20261026/python-deps \
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python3 -m unittest tests/test_bge_replay_cli.py -v
```

完整日志为 `work/bge-opt-20261026/acceptance-v2/cli-independent-cpu.log`。

## 发现的验收缺口与本地副本适配

旧独立脚本写死五例：性能检查要求全部五例预热，计划检查要求五条输入，已完成运行身份要求 pool_size=5 和整个五例 input SHA。合法固定池会被拒绝。按主控授权，仅在 `work/bge-opt-20261026/acceptance-v2` 建立三个脚本副本及小型共享模块，历史 `evidence/20261026/acceptance` 完全保留。

| 文件 | 本次变化 |
|---|---|
| workload_identity.py | 用冻结 CPU manifest SHA 和原始输入包 SHA 验证来源；按所选原 case 推导完整输入字节；逐张量核对内容、dtype、shape、偏移与容量；约束实际池及显式 workload 一致 |
| verify_plan_inputs.py | 调用共享身份检查；合法 fixed 为原五例之一的一条，legacy 无 metadata 仍必须为完整 random-five |
| verify_outputs.py | 性能按实际计划的 case 集合检查预热、事件和 runtime sampling；固定场景需提供实际 plan；拒绝 result 与外部传入 plan 不同。完整分数与五例压力函数保持原样；pressure CLI 禁止把 2000 请求 / 每实例每 case 20 次门槛调低 |
| validate_run_identity.py | 保留 binary/graph/job/run/plugin 身份检查，用实际 workload 池大小和独立输入身份替换固定的 pool5/整包 SHA 断言；增加本地归档根选择 |

没有把 3,448 字节写成所有固定 case 的通用限制。五个合法 ID 分别从原始输入包与独立 manifest 推导其字节；case-0002 只是当前交付默认。更换固定 ID 需重建同场景基线，不能事后选最快样例。

性能检查使用 `--plan <实际plan.json>`；result.json 本身含实际 plan 路径时可自动读取。没有 plan 的历史性能检查继续按严格 random-five 处理。参考默认使用实验根的原 acceptance 目录；本地 CPU 准备可只读使用已归档参考。真实完整分数/压力仍需原 expected 文件，不能用测试 fixture 替代。

## 新验收器回归与边界

定向 CPU 回归 **13/13 通过**，日志 `acceptance-v2/cpu-regression.log`。覆盖：五个合法固定 ID、默认及 legacy 随机池、固定和随机性能事件、假固定标签但实际五条池、未知/缺失 ID、元数据与实际池不一致、篡改输入字节、相同 shape 串 case、无固定 plan、runtime pool 不一致、计划 CLI、已完成固定运行身份，以及五例压力保持全覆盖和 CLI 最小门槛。

已完成运行身份测试中的 binary/graph digest 使用显式 CPU fixture；真实输入字节检查保留。压力正例使用归档 CPU 分数与归档 expected 排名信息，属于验收器测试。所有合成事件均仅验证校验逻辑，**不提供任何 NPU 正确性或性能证明**。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python3 -m unittest discover -s work/bge-opt-20261026/acceptance-v2 \
  -p test_workload_identity.py -v
```

另比较 AST，确认 `compare_one`、`pressure_check`、`npz_check` 与历史版本相同；历史三个脚本 SHA 与 delivery-manifest 原记录一致。`acceptance-v2/validation.json` 记录本次文件/日志 SHA、12+13 测试通过和未验证边界。

已就绪：双场景规则、本地 CLI CPU 准入、新验收工具 CPU 准备、真实原输入身份及历史数据保护。待完成：主控审阅/归档新工具、恢复 SSH 后远端集成与文件身份复核、恢复资源后重新门禁和预算估算、固定场景基线/候选实测、最终双场景三轮及原压力/稳定性验收。旧聚合入口若复用，需显式采用新版性能检查并传入实际场景计划；历史聚合与历史报告本次未改。不以本地 CPU 通过宣称远端集成、固定性能或最终配置接纳。
