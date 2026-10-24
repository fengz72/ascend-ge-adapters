# 平台和芯片

[targets.yaml](../targets/targets.yaml) 按 A3/A5 列出精确 SoC。设备号来自 `--device`，识别后将模型 YAML 的公共段和 `platforms.<family>` 顺序合并。每次命令只读取一个模型 YAML；`--targets` 或 Python execute 的 `targets` 参数可指定芯片目录。

```bash
python3 -m core.pipeline bench --config models/qwen2.5-0.5b/config/model.yaml --device 6
python3 -m core.pipeline prepare --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b
```

模型优化、依赖、核数、batch 和实例数统一由模型 YAML 的公共段及 platforms 维护；同一平台使用一套参数，不再按 SoC 叠加覆盖。芯片目录只管理硬件身份，可声明 `expected_capacity` 与设备查询结果核对；不从 A3/A5 名称推断实际容量。

修改实例数后直接重跑 bench；修改其他参数按[七步流程](../README.md#七个独立步骤)执行相应阶段。GE 核数预算不表示独占绑核。

平台选择成功不代表模型或算子已验收。Qwen 的已验证 SoC、配置与实测范围见[模型文档索引](qwen2.5-0.5b/README.md)，本页不维护模型默认值或历史性能。
