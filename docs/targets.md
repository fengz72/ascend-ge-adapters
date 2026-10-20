# 平台和芯片

targets/targets.yaml 按 A3/A5 列出精确 SoC。设备号来自 --device，识别后将模型 YAML 的公共段、platforms.<family>、chips.<soc> 顺序合并。每次 execute/命令行只接收一个 model.yaml；--targets 或 execute 的 targets 参数可指定芯片目录。

```bash
python3 -m core.pipeline bench --config models/qwen2.5-0.5b/config/model.yaml --device 6
python3 -m core.pipeline prepare --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b
```

修改 benchmark.instances 后重复同一 bench 命令，不生成实例 YAML。芯片容量由设备查询，不从 A3/A5 名称推断。可在目录为客户 SKU 声明 expected_capacity，以实际查询进行核对。

Qwen 当前 platforms.a3.build.options.aicore_num 为 12|24，a5 为 16|32；没有芯片级预算覆盖。A3 选择 prefix_910 和 weight_nz，A5 选择 prefix_a5。模型优化、依赖和核数选项均在模型 YAML，目录只管理硬件身份。

A5 Ascend950PR_958b 已有实测；A3 和其他 A5 SoC 没有环境，尚未验证。平台选择成功只证明配置选择成功，不代表模型或算子已验收。GE 核数预算不表示独占绑核或任意限核能力。
