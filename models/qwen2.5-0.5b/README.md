# Qwen2.5-0.5B

## 是什么

将 Hugging Face Qwen2.5-0.5B 适配为无 KV-cache 的 packed 推理，支持 A3/A5 Prefix Attention、融合 RMSNorm/RoPE 和所选 token 的业务评分，复用原生 HF DecoderLayer、MLP 和线性层。

[适配原理](../../docs/qwen2.5-0.5b/adaptation.md)、[验证记录](../../docs/qwen2.5-0.5b/validation.md)及调优、profiling 报告统一收录在[模型文档索引](../../docs/qwen2.5-0.5b/README.md)。

## 怎么用

依赖配套的 CANN、TorchNPU/TorchAir、Transformers 和模型权重。先按[项目安装说明](../../README.md#环境)初始化容器环境、安装 Python 依赖并构建原生运行器；当前适配验证使用 Transformers 5.10.1。

在项目根目录执行，`--device` 选择空闲逻辑卡，下面的卡号仅为示例：

```bash
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B
python3 -m core.pipeline run --config models/qwen2.5-0.5b/config/model.yaml --device 6
```

A3 执行前还需加载容器 Python 环境：

```bash
source models/qwen2.5-0.5b/scripts/env_a3.sh
```

配置统一维护在 [config/model.yaml](config/model.yaml)：公共段配置共用参数，`platforms.a3/a5` 配置平台算子、依赖、核数预算和实例数，不再使用 `chips` 覆盖层。`--device` 自动识别芯片，离线步骤使用 `--target`。输出评分列由 [config/target_tokens.json](config/target_tokens.json) 定义。

Prefix Attention 与 WeightNZ/MatMulV3 pass 的来源仓库、固定版本与构建入口统一维护在 [config/model.yaml](config/model.yaml)；Prefix Attention 源码见 [custom_op/prefix-attention](../../third_party/custom_development_code/custom_op/prefix-attention/README.md)（A3 用 `ascendc/`，A5 用 `cannbot-dsl/ge/`）。

七个步骤可独立执行，命令见[项目流程](../../README.md#七个独立步骤)。已有图、输入及依赖后，可以直接覆盖实例数运行性能：

```bash
python3 -m core.pipeline bench --config models/qwen2.5-0.5b/config/model.yaml --device 6 --instances 3
```

原始权重用于 data、baseline、build、run；infer、compare、bench、prepare 无需设置 `QWEN_MODEL_DIR`。默认精度 case、全量验收选择及性能测量口径见项目流程和模型 YAML。

## 性能表现

默认配置实测，batch16、同一 100 组变长输入池、seed42；P99 含 H2D、描述符、执行与同步，不含编译、预热和服务排队。

| 平台 | 核数/实例 | 业务 QPS | P99 |
|---|---|---:|---:|
| A3（Ascend910_9382） | 8\|16、5 实例 | 2172.093（三轮各 10000 次调用的中位） | 41.266 ms（三轮最差） |
| A5（Ascend950PR_958b） | 16\|32、3 实例 | 2768.499 | 19.475 ms |

A5 为单轮 1000 次调用扫描值，未做三轮长窗口确认。完整测量口径、验证边界及受干扰记录见[文档索引](../../docs/qwen2.5-0.5b/README.md)。

## 使用限制

- 面向标准全注意力、SiLU、默认 RoPE 的推理，不支持训练、HF `generate()`、KV-cache 或滑动窗口注意力。
- A5 Prefix 需要 CANN 9.3.0 及以上的辅流接口；动态 shape 还要求 GE 包含输出缓冲区扩容修复并使用版本匹配的 Compiler / Executor；当前交付支持 FP16、14/2 个 Q/KV heads、head dim 64。A3/A5 的实际版本、硬件及验收范围见[验证记录](../../docs/qwen2.5-0.5b/validation.md)，其他 SoC 需要单独验证。
- Patch 影响当前进程中的 HF Qwen2 类。原始 golden 与候选模型必须在独立进程中执行，且不能重复适配已裁剪模型。部署使用导出的 AIR 及平台算子运行库。
- 已有输入池的实际 batch 必须与 YAML 一致；修改输入参数后显式执行 data。修改模型或依赖后按项目流程重新 build 或 prepare。
- 默认使用 FP16。BF16 输出的运行期容量计算仍存在已知问题；GE Session dump 尚不支持，ATC 的历史验证范围见验证记录。
