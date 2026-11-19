# BGE 适配文件索引

本目录是 2026 年 10 月在 99 / 100 机器上完成的 BGE 优化记录与归档。
当前模型入口是 [models/bge/README.md](../../models/bge/README.md)。
当前运行时 `ge_execute` 的用法是 [ge-execute-usage.md](ge-execute-usage.md)。

## 归档分区

|用途|文档|说明|
|---|---|---|
|需求与设计|[brief.md](brief.md)、[design.md](design.md)|需求基线、接口与数值门槛、测量与判定口径、设备与预算约定|
|99 阶段验收|[acceptance.md](acceptance.md)|99 机器本轮的正确性、基线、搜索与预算归档；逐 case 误差见[附录](analysis/99-per-case-accuracy-20261022.md)|
|100 交付结论|[delivery-100.md](delivery-100.md)|当前配置、正式性能与全部优化点的汇总；配置与数字以本篇为准|
|时间线|[progress.md](progress.md)|按日期索引的阶段结论、预算里程碑与决策顺序|
|当前运行时|[ge-execute-usage.md](ge-execute-usage.md)|`ge_execute` 的 `cases.json` 规范、命令、产物、计时口径与限制|
|分析与独立审查|[analysis/](analysis/)|性能、融合、精度、实例数、实测与审查等专题报告|
|优化点|[experiments/](experiments/)|`OPT-001`～`OPT-017`，包括失败和未接纳候选|
|冻结证据|[evidence/](evidence/)|机器可读的汇总、身份、验收与逐轮复核文件；只读，不随文档改动|

2026-10-28 的 runner 修复及其“新二进制只有单轮”的口径见[修复报告](analysis/100-runner-review-fix-20261028.md)。

本仓库不含 `materials/`、`work/` 与 100 项目的 `tools/`、`runner/`、`candidates*/`；这些位于 100 项目根 `/export/home/ext.hejun23/workspace/bge-opt-100`，原始运行事件仍在其中。
