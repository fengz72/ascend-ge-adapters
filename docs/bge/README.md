# BGE 适配文件索引

当前执行项目位于 100 机器的 `hejun-cann9-bge` 容器：
`/export/home/ext.hejun23/workspace/bge-opt-100`。100 项目以 `runner/docs/bge/` 为文档正本，项目根 `docs/bge/` 是兼容链接；本地工作区保留副本。99 机器的历史结果单独见 [acceptance.md](acceptance.md)。

## 当前状态

- **2026-10-28 runner 修复**：输出 dtype 别名、材料相对路径和交付路径问题已修复；新二进制通过五例 22,274 请求全量输出校验及固定/随机各一轮 60 秒性能检查。旧三轮性能中位数与新构建单轮结果分开记录，见[修复与验证报告](analysis/100-runner-review-fix-20261028.md)。
- **当前选用配置**：原始 ONNX、GE Session `graph_run_mode=0`、内部 `force_fp16`、10 实例、每算子核上限 `12|24`，无自定义融合插件。外部浮点输入/输出仍为 FP32。五例完整输出和 600 秒稳定性见 [FP16 十实例报告](analysis/100-fp16-ten-selection-20261027.md)；8/10/12 实例同设备双场景三轮对照见[复测报告](analysis/100-fp16-scale-recompare-20261027.md)。
- **原 FP32 阶段交付**：8 实例、`12|24` 的完整正式对照与稳定性仍保留在 [100 交付报告](delivery-100.md)。最新 FP16 对照中，10 实例固定 case 相对 8 实例中位吞吐 +9.519%，随机五例因一轮低值未证实收益；当前配置由用户指定，不声称实例数整体最优。
- **其他未接纳候选**：Add+LayerNorm 融合 V2/V3、FP32 10/12/16 实例上探、FP16 12 实例上探。优化记录 `OPT-001`～`OPT-017` 全部保留。
- **范围**：只有给定五条样例；Top10 的 `dense_id` 为输出下标加 1。没有业务 p99 上限或生产流量分布。

上游同步与切换前 FP32 五例回归见[同步报告](analysis/100-upstream-sync-validation-20261027.md)。

## 文件位置

|用途|位置|说明|
|---|---|---|
|需求与设计|[brief.md](brief.md)、[design.md](design.md)|路线、输入输出、精度与测量契约|
|交付结论|[delivery-100.md](delivery-100.md)|最终配置、正确性、固定/随机双场景性能及全部优化记录汇总|
|当前进度|[progress.md](progress.md)|按时间追加的执行与决策记录；顶部是最新状态|
|优化点|[experiments/](experiments/)|`OPT-001`～`OPT-017`，包括失败和未接纳候选|
|分析与独立审查|[analysis/](analysis/)|性能、融合、精度、测试审查等专题报告|
|机器可读证据|[evidence/](evidence/) 与 [交接工作文件](../../delivery/supporting-evidence/README.md)|99/100 的汇总、身份、验收与逐轮复核文件；原始运行事件仍在项目根 `work/`|
|原始输入与模型|远端 `materials/`|`model.onnx`、`sample_inputs.json`、`expected_outputs.json`、CPU golden；`exported/` 是供 GE 加载的打包图与权重|
|运行框架|远端 `runner/`|`codex/seven-stage-framework` 分支；BGE 配置位于 `models/bge/config.yaml`，原生执行程序位于 `runtime/`|
|设备作业与验收工具|远端 `tools/`|`run_one.py`、`control.py`、身份/配置账本及校验器|
|原始作业|远端 `work/`|每个作业独立目录，包含计划、请求事件、资源样本、门禁和状态；不覆盖旧作业名|
|候选融合插件|远端 `candidates/`、`candidates-v3/`|V2/V3 实验产物，均未纳入当前交付配置|
|源码快照与文件清单|[delivery/](../../delivery/README.md)|当前未提交框架改动的补丁、新增源文件和核心路径清单；不复制大模型或原始作业|

## 复现入口

先检查实际设备与同芯片占用，再在远端项目根使用新作业名运行 `tools/run_one.py`；完整命令与统计口径见 [交付报告第 4 节](delivery-100.md#4-交付件与复现)。`tools/identity.json` 当前指向逻辑设备 2；临时换卡后的测试结果不能仅凭旧账本重放。远端 `runner/` 有未提交改动，交付快照位于 `delivery/`，当前 Git base 为 `21293e3`；未提交 BGE 源码改动仍需应用 `runner/delivery/source-snapshot/`。
