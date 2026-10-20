# Qwen2.5-0.5B 适配与改动说明

交付内容是修改后的源码、配置和构建脚本。源码中的修复已经生效，客户按本文和项目 README 使用即可。

## 改了哪些地方

| 文件 | 改动和作用 |
|---|---|
| `qwen_npu_adapter.py` | 通过 `apply_patches()` 集中 patch HF 方法；`adapt(raw_model, config)` 原地准备模型并应用 patch。融合算子、PrefixAttention、packed 输入、所选输出 head 和业务编码都在这一文件，复用 HF 的 DecoderLayer、MLP 和线性层。 |
| `generate_cases.py` | 由 YAML 声明的 pipeline 首步脚本，使用 NumPy 保留历史长度分布和 token RNG；不依赖 core 或模型适配文件。 |
| `config/model.yaml`、`config/target_tokens.json` | 原始模型位置、数据生成脚本与参数、精度标准、输出 token、公共模型参数、A3/A5 平台候选和测量参数。客户设置自己的模型路径和业务输出。 |
| `config/model.yaml` 中的 `platforms` / 可选 `chips` | 自动应用 A3/A5 候选及精确芯片覆盖，明确选择 Prefix 算子、GE pass、源码位置和构建脚本。 |
| 项目 `targets/targets.yaml` | 按 A3/A5 集中列出精确 SoC，可声明客户 SKU 和实测容量；不承载模型参数。 |
| `scripts/build_prefix_a5.sh`、`build_prefix_910.sh`、`build_nz_pass.sh` | 在对应 CANN 容器中构建算子或 GE pass，使用容器 Python。 |
| 项目 `third_party/ascend-ops/prefix-attention/cannbot-dsl/ge/prefix_attn_torch/prefix_attn_bridge.cpp` | A5 Prefix eager 桥接将 `getCurrentNPUStream().stream(false)` 改为 `stream()`，先提交 TorchNPU 主机队列里的 Q/K/V 任务，再提交 ACL 自定义算子；没有加入逐算子设备同步。修复已提交为 `b4b30cd`。 |

`config/target_tokens.json` 必须与适配文件保持相对位置。原始权重由客户提供；`adapt()` 修改传入的模型对象，保留 HF 权重，仅在裁剪输出时复制指定的 lm_head 行。模型被设为推理模式。

## Patch 列表

`apply_patches()` 中直接列出五个替换目标：

| HF 目标 | 本文件替换函数 | 改动 |
|---|---|---|
| `Qwen2RMSNorm.forward` | `rms_norm_forward` | 使用 `npu_rms_norm`。 |
| `apply_rotary_pos_emb` | `rotary_pos_emb` | packed TND 布局及 NPU RoPE 融合算子。 |
| `Qwen2Attention.forward` | `attention_forward` | 保留原 Q/K/V/O 投影，选择 FIA、A5 Prefix 或 A3 Prefix。 |
| `Qwen2Model.forward` | `packed_model_forward` | packed 输入和 RoPE 表，调用原 HF DecoderLayer/MLP，选择各请求末 token。 |
| `Qwen2ForCausalLM.forward` | `score_forward` | 调用原 lm_head，输出业务 scores。 |

Patch 影响当前进程中的 HF Qwen2 类，使用独立的开发/导出容器或进程。每次适配加载原始模型，不对已经裁剪的模型重复 adapt。原始 golden 必须在 patch 前由独立进程生成；已应用 patch 的进程会拒绝生成参考结果。当前针对容器中的 Transformers 5.10.1 验证。

这些替换面向无 KV-cache 的 packed 推理接口，不用于训练或 HF `generate()`。部署直接加载生成的 AIR，无需导入或执行 Python patch；自定义算子的原生运行库仍按平台配置提供。


## 使用

唯一模型配置是 config/model.yaml。权重路径由 QWEN_MODEL_DIR 提供给 data、baseline、build、run；infer、compare、bench、prepare 不需要原始权重环境变量。完整七步入口如下，默认精度只用 case-0000，性能用完整 100 组输入：

```bash
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B
python3 -m core.pipeline run --config models/qwen2.5-0.5b/config/model.yaml --device 6
```

各步均可独立调用，主项目 README 列出了七条命令。单步命令不补跑任何前置步骤。data 生成输入，baseline 生成 golden，prepare 构建依赖，build 应用 HF patch 并导出，infer 读取已有图运行 GE，compare 比较已有输出，bench 只跑性能。没有强制候选 eager 精度门禁。

已执行 build 后，测试实例数无需重导出、生成 golden 或重复精度：

```bash
python3 -m core.pipeline bench --config models/qwen2.5-0.5b/config/model.yaml --device 6 --instances 1
python3 -m core.pipeline bench --config models/qwen2.5-0.5b/config/model.yaml --device 6 --instances 8
```

--instances 临时覆盖 YAML，不改文件。iterations 为所有实例合计的调用次数；当前默认每次 batch=16，1000 次对应 16000 个业务样本。性能读取完整输入池按 seed 有放回抽样，实例数不改变全局抽样序列。GE 编译、加载、每实例全部输入 shape 的预热不计入测量。

RuntimeHandle 与 HF patch 同文件交付，只处理数组的 encode/decode 和输出容量，不加载权重或应用 patch。AIR 自带具名 IO，bench 不依赖 build.json、lock.json、golden 或精度证据。infer/bench 的 --profiling 复用已有图，性能采集结果会标记 profiling_enabled。

## 数据与精度

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

文件采用共享 prefix 的 packed 布局：input_ids 先放一次 prefix，再放各请求的私有 tokens；position_ids 的 prefix 从 0 开始，每条私有序列从 prefix 长度开始；actual_seq_lengths 保存各段累计结束位置，当前 batch=16 时共有 17 个值。读取时按 bounds 恢复同一批业务请求，case ID 取目录名，request ID 按顺序为 r0、r1 等。参考模型和 FIA 由适配文件编码为各自的输入，A3/A5 Prefix 的候选输入与这些 NPY 张量逐元素一致。

先独立准备数据，随后 bench 直接复用已有输入；bench 不自动执行 data：

```bash
python3 -m core.pipeline data --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b
```

原始模型在独立进程中先运行，再对候选应用 HF patch。精度默认 case-0000，可在 verification.case_ids 配多条或 all，或给 baseline/infer/compare/run 传 --case-ids。baseline 只为所选 case 生成原始 CPU FP32 golden，infer 单实例运行同一选择并保存 output，compare 按 YAML 阈值完成所选 case 的比较；run 在通过后继续性能。

```bash
python3 -m core.pipeline baseline --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b --case-ids all
python3 -m core.pipeline infer --config models/qwen2.5-0.5b/config/model.yaml --device 6 --case-ids all
python3 -m core.pipeline compare --config models/qwen2.5-0.5b/config/model.yaml --target Ascend950PR_958b --case-ids all
```

参考输出使用有序 target_tokens.json，对每个业务请求比较 cosine >= 0.998、relative L2 <= 0.05。候选继续使用 build.dtype: float16 和 precision_mode: force_fp16；适配代码保持原 FP16 路径，不增加 FP32 残差或输出层。该阈值只配置在此模型 YAML 中，不改变框架或其他模型的默认阈值。参考使用 HF logits_to_keep，只取所需位置的完整词表 logits，不 patch 原始模型。不同模型的业务输入输出仍由各自适配文件定义，框架不解释 Qwen 参数。

## 平台与边界

platforms.a3 选择 prefix_910、weight_nz，GE 核数预算 12|24；platforms.a5 选择 prefix_a5，当前默认核数预算 16|32。--device 识别芯片自动选择配置，不保存卡号。Prefix 的 T、prefix、bounds、实际 batch 可变；不重复配置 Prefix batch。

历史 A5 验证（16|32核、batch=10）已按当前 FP16 阈值通过完整 100 组、1000 个业务样本。最小 cosine=0.9994753718802383，最大 relative L2=0.034476225797271505；同一输出按旧 0.999/0.02 阈值仍有两组失败。当时只调整验收阈值，保留相同 FP16 模型和 CPU FP32 golden。A3 和其他 A5 芯片未实测，ATC 本轮暂不验证。历史性能和 profiling 见 docs/validation.md。只有 model.yaml、适配文件、数据生成脚本、实际依赖源码和构建脚本随项目交付，work 不交付。

## A5 性能调优（2026-10-20）

当前 A5 默认：限核 **16|32**、**batch_size=16**、**instances=4**；Prefix 源码 kAicCores=16，对应算子已重建安装。data.directory 固定为 models/qwen2.5-0.5b/io/inputs，含100组 batch16 输入；与历史调优 batch16 输入池摘要一致。该配置此前三轮各10000次调用，业务QPS中位数 2662.044，最差轮整 batch p99 为26.865ms。原 batch10 输入、golden 和 output 保留在 work/qwen2.5-0.5b/validation/history-batch10；本次框架修改的 batch16 验证范围及结果见 docs/validation.md，不能将历史 batch10 全量精度结论直接推广到 batch16 全量。测试范围、选型理由、趋势图、复现命令及全部164次测试结果见 [调优记录](../../docs/qwen2.5-0.5b/调优记录.md)。
