# BGE 当前适配文件交接

当前用户指定并完成五例精度/600秒稳定性验证的配置：原始 ONNX、GE Session mode=0、内部 `force_fp16`、10 实例、每算子核上限 `12|24`、无自定义融合插件；外部浮点 IO 保持 FP32。完整结论见 [`../docs/bge/delivery-100.md`](../docs/bge/delivery-100.md)，文件导览见[`../docs/bge/README.md`](../docs/bge/README.md)。

最新同设备三轮对照见[8/10/12实例复测](../docs/bge/analysis/100-fp16-scale-recompare-20261027.md)：固定`case-0000`中位QPS分别为807.326/884.177/851.169，五例随机分别为861.592/903.582/862.495；随机10实例有一轮730.640QPS，因此随机场景的吞吐优势未证实。维持用户指定的10实例配置，完整逐轮证据保留在项目根`work/`。

2026-10-28 的 [runner 修复报告](../docs/bge/analysis/100-runner-review-fix-20261028.md)记录新旧二进制身份与新构建的五例全量输出、固定/随机单轮性能检查。上段三轮性能数字绑定旧二进制；新构建尚未完成三轮稳定吞吐复核。旧二进制与当时身份清单归档于 `previous-binary/`。

- `manifest.json`：核心模型、输入、参考输出、二进制、配置和文档的实际 SHA256 与字节数。
- `source-snapshot/tracked.patch`：相对 `runner/` Git base commit 的已跟踪源码改动。
- `source-snapshot/new/`：`runner/` 中未跟踪的新增源文件，按原相对路径保存。
- `../../materials/`：原始模型、样例、CPU golden 和 GE 打包图/权重；不在本目录复制大权重。
- `../`：当前实际构建与运行的 `runner/` 源码；`../../tools/`：项目根的设备门禁、配置和验收入口。
- `../../work/`：逐作业原始证据，保留原作业名与路径；`../../candidates*/`：未接纳的融合候选。

从 `manifest.json` 的 `runner.base_commit` 获取基础仓库版本，对 `runner/` 应用补丁，再将 `source-snapshot/new/` 中的文件按相对路径复制回 `runner/`，即可重建这次未提交的源码状态。模型配置通过 `../materials/` 引用同级材料目录；历史作业证据仍使用记录中的原路径。

仓库内的 `fixtures/`、`exported-graph/`、`project-tools/` 和 `rejected-candidates/` 是轻量交接副本；大权重与逐请求原始证据仍在项目根。运行受管作业时继续使用项目根 `tools/run_one.py`；新环境还需按其设备、构建产物和材料路径重新生成 `tools/identity.json`，不能沿用本机的冻结身份清单。

`review-scripts/` 保留本轮性能与精度复核脚本；`supporting-evidence/` 收纳分析报告直接引用的 38 个本地工作文件。这两部分是复核材料，不替代项目根 `work/<job>/` 原始设备证据。
