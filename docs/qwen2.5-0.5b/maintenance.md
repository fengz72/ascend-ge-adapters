# Qwen2.5-0.5B 维护记录

本文记录依赖来源与迁移分支的维护历史；正确性与设备验证见[验证记录](validation.md)，使用方法见[模型 README](../../models/qwen2.5-0.5b/README.md)。

## 上游合入后的依赖统一（2026-10-27）

`custom_development_code` 子模块来源改为
`https://gitcode.com/CharmanderFang/custom_development_code.git`，跟踪 `main`，
固定提交为 `8e08d0d20275558aefac1bbb35019e6b52f1073a`。
该上游已合入 Prefix Attention（!35）及 WeightNZ/MatMulV3（!34）。
A3/A5 的 Prefix 算子与 WeightNZ pass 四项依赖的 source 均位于该子模块，
revision 统一为该提交；原个人 fork 迁移分支保留，`ascend-ops` 继续用于历史对照。

已通过 Git 目录差异检查：上游 Prefix Attention 与 `c6b18c8` 完全一致，
上游 WeightNZ/MatMulV3 pass 与 `fba9f846` 完全一致。因此恢复的是此前设备验证过的
算子和 pass 源码组合；[验证记录](validation.md)中的历史提交号继续保留，不改写原实验事实。
这次仅切换仓库来源和固定版本，没有修改算子/pass 实现或 GE 系统库，
A5 默认 GE 输出缓冲区扩容问题的限制仍适用。

99 / hejun-cann9-qwen25 与 850-a5 / hejun-cann920-cannbotdsl 两处主仓工作区
均已同步来源、配置、子模块提交与本文记录。99 使用 CANN 9.0.0，A5 使用 CANN 9.3.0；
A5 验证沿用[验证记录](validation.md)中的进程级输出扩容修复库环境。

| 检查 | A3（99） | A5（850-a5） |
|---|---|---|
| A3/A5 配置解析及实际源码检查 | 四项依赖均来自上游，revision 与 HEAD 一致 | 同左 |
| prepare | Prefix Attention 与 WeightNZ pass 重新构建、安装通过 | 通过 |
| Python 框架回归 | 60/60 通过 | 60/60 通过 |
| Runtime CTest | 1/1 通过 | 1/1 通过 |
| 上游 WeightNZ host 回归 | 开启 `NZ_WEIGHT_PASS_BUILD_TESTS` 构建，CTest 1/1 通过 | 同左 |

两处测试日志均位于各自工作区的 `work/custom-upstream-20261027/`。
自包含流程不再经过 core 七步，运行产物按阶段落在模型目录的 `work/`：
`work/deps/`（Prefix 算子与 NZ pass）、`work/inputs/`、`work/golden/`、
`work/build/`、`work/output/`、`work/bench/`、`work/logs/`。
本次未重新执行完整模型精度或性能测试；此前设备验证范围及 GE 环境限制仍按[验证记录](validation.md)记录。

## 迁移分支整理（2026-10-27）

按要求将 `codex/migrate-prefix-attention` 重建到上游基线
`ddca283e1f3d0488c4b1c4ce7672fb9bb128a83c`，当前迁移提交为
`c6b18c8d6bbb1f3bd93b624258ca0ad654307b8e`。分支历史不包含
`fba9f846cc31792fb155cb8c5653b06a74ea4137`，也不包含其WeightNZ/MatMulV3改动；
相对上游仅保留Prefix Attention迁入及相关README改动。

已核对Prefix Attention目录及迁移README与原提交 `dc21a7a7f10661ee49af75c9519832103b2d6db9`
逐字节一致，模型配置中两项Prefix依赖revision已同步。[验证记录](validation.md)中的A3/A5设备验证仍按实际
运行时的原提交记录；该时点包含独立WeightNZ提交，不能据此宣称移除该提交后的
完整模型组合已经重新验证。WeightNZ依赖的独立版本声明本次未调整。

## 运行时契约迁移到 ge_execute（2026-11-19）

`runtime` 在 `28389ba` 把三个旧运行时换成单一二进制 `ge_execute`，本适配器的
`work/` 文件契约随之改变。**依赖与图未动**：`work/deps/`（Prefix 算子 + NZ pass）
沿用已构建产物、未重跑 `scripts/prepare.sh`，`work/build/packed.air` 沿用
2026-11-17 的导出（993754309 字节）、未重跑 `export_air.py`；子模块固定提交仍是
`8e08d0d20275558aefac1bbb35019e6b52f1073a`。

| 阶段 | 旧产物（`ge-records/3` / `ge-compare/1`） | 新产物（`ge-cases/1` / npy 同构） |
|---|---|---|
| `generate_inputs.py` | `work/inputs/{inputs.bin, records.json}` | `work/inputs/cases.json` + `work/inputs/case-XXXX/{input_ids,position_ids,actual_seq_lengths}.npy` |
| `golden.py` | `work/golden/{outputs.bin, result.json(outputs)}` | `work/golden/<case_id>/output_0.npy` + `work/golden/result.json(cases)` |
| 运行时 infer | `work/output/{outputs.bin, result.json(outputs)}` | `work/output/<case_id>/output_0.npy` + `work/output/result.json(cases)` |
| 运行时 bench | `work/bench*/{result.json(performance), requests.csv}` | `work/bench*/result.json`（顶层扁平统计 + `records` 逐请求） |

两个脚本都会在写新产物前清掉自己那一层的旧文件（`inputs.bin`/`records.json`、
`outputs.bin`），避免两种布局混在同一目录里让 `tools/compare.py` 猜。迁移前的旧
运行时产物已整体挪到 `work/output-legacy-records3/` 与 `work/bench-legacy/`
（Git 忽略）留档，没有删除。

`tools/compare.py` 改为按 `result.json` 的键自动分派布局（`cases` → npy，
`outputs` → 旧 `outputs.bin`），**旧布局的读取能力保留**，因为 onejev1.6-0.8b 与
startlux-decision-v1.1-0.8b-instant-text 的 golden 仍在写它。本适配器是四个待迁移
模型里第一个完成迁移的；其余三个的现状：

| 模型 | 阻塞点 |
|---|---|
| `onejev1.6-0.8b`、`startlux-decision-v1.1-0.8b-instant-text` | `run.sh` 仍调已删除的 `executor`，且 bench 传 `--readout`；`ge_execute` 尚未实现 readout decider 阶段 |
| `startlux-decision-v1.1-0.8b-instant` | 两阶段（vision → text）pipeline，`ge_execute` 明确只支持单图，不在范围内 |
| `bge`、`tiny`、`onnx-example` | 走 `core.pipeline` → `core/backends/runtime.py`，后者仍指向已删除的 `ge_runtime` 与 stdin plan 协议，需要单独迁移 core 侧 |

复测数据与逐项对照见[验证记录](validation.md)的「ge_execute 迁移复测（2026-11-19）」节；
`ge_execute` 侧的 AIR 分支验证记录见 [runtime/README.md](../../runtime/README.md) 附录。
