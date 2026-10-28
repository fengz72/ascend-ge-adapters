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

两处测试日志均位于各自工作区的 `work/custom-upstream-20261027/`；
prepare 的正式运行记录位于 `work/qwen2.5-0.5b/results/`。
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
