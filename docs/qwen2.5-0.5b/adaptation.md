# Qwen2.5-0.5B 适配说明

安装和快速上手见[模型 README](../../models/qwen2.5-0.5b/README.md)，实验与验证入口见[文档索引](README.md)。本文说明当前源码的计算路径和输入输出契约。

## 源码分工

| 文件 | 职责 |
|---|---|
| [qwen_npu_adapter.py](../../models/qwen2.5-0.5b/qwen_npu_adapter.py) | HF 方法替换、模型初始化、输入编码、输出评分及 AIR 运行配方 |
| [generate_cases.py](../../models/qwen2.5-0.5b/generate_cases.py) | 保留历史分布的输入生成和读取 |
| [model.yaml](../../models/qwen2.5-0.5b/config/model.yaml) | 公共参数及 A3/A5 平台的算子、依赖和运行参数 |
| [target_tokens.json](../../models/qwen2.5-0.5b/config/target_tokens.json) | 有序业务评分 token |
| [构建脚本目录](../../models/qwen2.5-0.5b/scripts) | 平台算子与权重 NZ Pass 构建 |

`adapt()` 原地准备传入模型，保留 HF 模块和原始权重，仅在裁剪 lm_head 时复制指定行。`target_tokens.json` 按配置路径读取；使用默认相对路径交付时需与适配文件保持目录关系。

## Patch 与计算路径

`PATCHES` 统一列出五个替换目标，`apply_patches()` 和适配报告共用这份清单：

| HF 目标 | 本文件替换函数 | 改动 |
|---|---|---|
| `Qwen2RMSNorm.forward` | `rms_norm_forward` | 使用 `npu_rms_norm`。 |
| `apply_rotary_pos_emb` | `rotary_pos_emb` | packed TND 布局及 NPU RoPE 融合算子。 |
| `Qwen2Attention.forward` | `attention_forward` | 保留原 Q/K/V/O 投影，选择 FIA、A5 Prefix 或 A3 Prefix。 |
| `Qwen2Model.forward` | `packed_model_forward` | packed 输入和 RoPE 表，调用原 HF DecoderLayer/MLP，选择各请求末 token。 |
| `Qwen2ForCausalLM.forward` | `score_forward` | 调用原 lm_head，输出业务 scores。 |

Patch 影响当前进程中的 HF Qwen2 类，使用独立的开发/导出容器或进程。每次适配加载原始模型，不对已经裁剪的模型重复 adapt。原始 golden 必须在 patch 前由独立进程生成；已应用 patch 的进程会拒绝生成参考结果。当前针对容器中的 Transformers 5.10.1 验证。

这些替换面向无 KV-cache 的 packed 推理接口，不用于训练或 HF `generate()`。部署直接加载生成的 AIR，无需导入或执行 Python patch；自定义算子的原生运行库仍按平台配置提供。

模型 forward 使用 packed token，预计算 RoPE 表并按 position IDs Gather；DecoderLayer 和 MLP 继续调用原 HF 实现。Attention 初始化时绑定 FIA、A3 Prefix 或 A5 Prefix 入口，forward 保留 Q/K/V/O 投影和 TND reshape。bounds 通过参数逐层传递，不写入每层的可变请求状态。

## 输入编码与结果对齐

模型 Handle 和 RuntimeHandle 共用 NumPy 编码；模型执行时再转换为 Tensor。RuntimeHandle 只处理数组和输出容量，不加载权重或应用 HF patch。CPU RMSNorm、RoPE 和 Attention 参考算子由[测试](../../tests/test_qwen_adapter.py)注入，不进入生产 NPU 路径。

原始 HF 参考输入按请求补齐为二维张量，FIA 为每条请求重复 prefix；Prefix 路径只保存一次共享 prefix，再拼接各请求私有 tokens。position IDs 在每条请求中从 prefix 长度继续，bounds 为各段累计结束位置。A5 Prefix 的 bounds 编码为 int32，FIA/A3 为 int64；token 和 position IDs 始终为 int64。

候选在最终 norm 前选择每个业务请求的末 token，Prefix 模式跳过独立 prefix 段。原始 HF 参考使用 `logits_to_keep` 提取所需位置的完整词表输出。按 target token 顺序选择评分列，解码结果同时携带请求 ID 和评分列的对齐信息。验收阈值只由模型 YAML 的 `verification.outputs.scores` 决定。

## 输入池

generate_cases.py 保留 Git f416fff 的历史算法：NumPy RandomState(0)，先抽共享 prefix=20–25，再抽 batch=10 的总长度 round(exp(normal(4.997,0.167)))，截断上限 218；token 使用独立 seed*100003+case_index。历史 batch10 的 100 组实际总长均值 149.345、P99 212.02、范围 89–218。当前默认 batch16 保留前10条历史请求并确定性扩展，实际总长均值149.769、P99 218、范围84–218，不修改样本凑统计量。此脚本实现 generate(config)，在 data.directory 指定的 models/qwen2.5-0.5b/io/inputs 中保存 case-0000/ 至 case-0099/，每个 case 目录直接包含三个一维 int64 输入张量：input_ids.npy、position_ids.npy、actual_seq_lengths.npy。NPY 自带 dtype 和 shape，读取时使用 np.load(path, allow_pickle=False)，不需要 bundle.json 或数据清单。每次 pipeline 的首步检查当前生成结果，只补齐、修复或更新变化的 case；减少 count 时清理多出的 case 目录。返回的数据直接从这些文件读取，文件不变时不重写。该目录由 Git 忽略，可删除后重新生成。单独运行时 --config 接收同一模型 YAML，保存数据并展示统计。

```yaml
data:
  script: models/qwen2.5-0.5b/generate_cases.py
  directory: models/qwen2.5-0.5b/io/inputs
  params:
    count: 100
    batch_size: 16
    # 其余分布参数见 config/model.yaml
```

目录如下（共 100 个 case、300 个 NPY 文件）：

```text
io/inputs/
  case-0000/
    input_ids.npy
    position_ids.npy
    actual_seq_lengths.npy
  case-0001/
    input_ids.npy
    position_ids.npy
    actual_seq_lengths.npy
  ...
```

文件采用共享 prefix 的 packed 布局：input_ids 先放一次 prefix，再放各请求的私有 tokens；position_ids 的 prefix 从 0 开始，每条私有序列从 prefix 长度开始；actual_seq_lengths 保存各段累计结束位置，当前 batch=16 时共有 17 个值。读取时按 bounds 恢复同一批业务请求，case ID 取目录名，request ID 按顺序为 r0、r1 等。参考模型和 FIA 由适配文件编码为各自的输入，A3/A5 Prefix 的候选输入与这些 NPY 张量数值一致，A5 bounds 的 dtype 转换见上文。

先独立准备数据，随后 bench 直接复用已有输入；bench 不自动执行 data：

```bash
python3 -m core.pipeline data --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b
```

以上 NPY 是持久化输入：三个数组均为 int64。A5 候选编码时将 bounds 转为 int32，数值保持一致。修改生成参数后显式运行 data，其他步骤只读取已有输入，不生成数据或输入清单。历史 batch10 与 batch16 的精度结论分别保留在[验证记录](validation.md)。

## Prefix 算子接入

A3 使用 `prefix_910`，A5 使用 `prefix_a5`，具体依赖版本以模型 YAML 为准。A5 支持变长 T、prefix、bounds 和实际 batch，当前交付的 dtype、头数与 CANN 接口要求见模型 README。

早期 A5 eager 桥接修复 `b4b30cd` 将 `getCurrentNPUStream().stream(false)` 改为 `stream()`，先提交 TorchNPU 主机队列中的 Q/K/V 工作，再提交 ACL 自定义算子；该修复没有增加逐算子设备同步。之后的辅流 planner、int32 bounds 和同步调整以[逐算子 profiling 报告](qwen2.5-0.5b-a3-a5-profiling.md)中的对应实验为准，不能将早期桥接实现视为所有后续版本的同步方式。

CANN 辅流接口核验分别见 [9.1.0](../cann-9.1.0-eager-attached-stream-support.md)、[9.2.0](../cann-9.2.0-eager-attached-stream-support.md)、[9.3.0](../cann-9.3.0-eager-attached-stream-support.md)。
