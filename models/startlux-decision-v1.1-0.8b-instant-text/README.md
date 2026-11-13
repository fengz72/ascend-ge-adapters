# StartLux-Decision v1.1 0.8B Instant 纯文本极简流程（-text）

## 是什么

[StartLux-Decision-v1.1-0.8B-Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant)
决策模型的**纯文本**极简交付：每个部件一个自包含脚本、部件间零 import 依赖，
只通过 `work/` 下的文件契约交互，共用一个 2 输入动态 text AIR
（`input_ids` [1,S] / `position_ids` [3,1,S] → `scores` [1,26]；
**B=1 由图钉死**——io spec 轴 0 声明为静态，B≠1 请求在图绑定处即报错）。
图片（多模态）能力不在本目录——完整双阶段流程见
[../startlux-decision-v1.1-0.8b-instant/](../startlux-decision-v1.1-0.8b-instant/README.md)。

五个部件与文件契约：

| 部件                                              | 输入                       | 输出（默认路径）                                                    |
| ------------------------------------------------- | -------------------------- | ------------------------------------------------------------------- |
| `model/generate_inputs.py`                        | dev.txt JSONL case 文件    | `work/inputs/{inputs.bin, records.json}`                            |
| `model/golden.py`                                 | `work/inputs` + 权重目录   | `work/golden/{outputs.bin, result.json}`                            |
| `model/export_air.py`                             | 权重目录                   | `work/build/text.air`                                               |
| `runtime/build/executor`（`runtime/executor.cpp`） | `text.air` + `work/inputs` | `work/output/{outputs.bin, result.json}`，bench 另加 `requests.csv` |
| `tools/compare.py`（复用）                        | golden 与 infer 输出       | `work/output/comparison.json`                                       |

另含 `run.sh`（一键串联）与 `scripts/prepare.sh`（依赖构建），不属于文件契约部件。

要点：每条 case 恒 B=1、真实长度（无 padding）；纯文本 MRoPE
`position_ids` 即三轴 arange（已对官方 `get_rope_index` 输出逐位验证）；
`model/modeling_text.py` 是**完整的 NPU 可运行模型文件**（`StartLuxTextModel`：
embed → 混合解码器 trunk → 末 token 26 列读出，
`forward(input_ids, position_ids)` 直接返回 scores，
读出头由 `install_decision_head(letter_token_ids)` 从加载后的词嵌入行派生），
与父目录同名文件的功能性差异恰好三处：`_causal_depthwise_conv` 改标准算子路线（见
"使用限制"）、父目录外层读出 wrapper 并入 `StartLuxTextModel`、
B=1 钉死并删去 `actual_seq_lengths` 输入（长度全部形状派生）；
`vendor/`
是官方 `startlux_decide` 固定 commit 子集的 6 模块最小闭包（prompt 渲染 +
bidirectional 参考注意力，provenance 见 `vendor/SOURCE.md`）；golden 是
独立的官方 CPU FP32 参考（官方 `Qwen3_5ForConditionalGeneration` +
官方双向注意力补丁），不复用 `modeling_text`（后者全是 NPU host op，
CPU 不可执行）。

### 模型输入与输出

text AIR 的动态 IO 契约（executor 按此固定顺序位置绑定；B=1 由图静态钉死，
唯一动态轴是 S）：

| 张量             | 方向 | dtype   | shape   | 语义与约束                                       |
| ---------------- | ---- | ------- | ------- | ------------------------------------------------ |
| `input_ids`      | 输入 | int64   | [1, S]  | token 序列；`0 ≤ id < vocab_size`                |
| `position_ids`   | 输入 | int64   | [3,1,S] | MRoPE 三轴位置；纯文本即三轴 `arange(S)`         |
| `scores`         | 输出 | float32 | [1, 26] | A–Z 26 列 letter logits（FP32 读出）             |

输入数据的来源与约束：

- **case 文件**（dev.txt JSONL）：每行 `{state, questions, targets, meta}`，
  恰一个 `choice` 问题、2–26 个 criteria、targets 为 criteria 上的概率分布
  （和 ≈1）；经官方 prompt 渲染（`from_systemone → messages → plain_prompt`，chat template 关闭 thinking）后 `tokenizer.encode(prompt, add_special_tokens=False)` 得 `input_ids`。
- **每 case B=1、真实长度**（无 padding），dev 池实测 S ≈ 1001–1005；
  上界 `S ≤ max_position_embeddings = 262144`。**没有
  `actual_seq_lengths` 输入**——B=1 契约下铺出来的宽度就是真实长度，
  records 格式无法表达 padding（错误数据不可表达），图内所有长度消费
  （attention 长度、GDN 行宽、读出索引）全部形状派生。
- **records.json**（`startlux-text-records/2`）：`{file: inputs.bin, cases: [{id, text_tensors[2]}]}`，text_tensors 携带 name/dtype/shape/offset/bytes，
  指向 `inputs.bin` 中的连续字节段。

模型关键参数（`config.json` text_config）：24 层（18 层 Gated DeltaNet 线性
注意力 + 6 层双向 full attention）、hidden 1024、vocab 248320、MRoPE
section [11,11,10]（interleaved，theta 1e7，partial rotary 0.25）、GDN 因果
卷积核宽 4、`tie_word_embeddings=True`（26 列读出头取 `embed_tokens` 的
letter token 行，`decision_config.json`）。

## 怎么用

环境信息（本机已验证组合，版本核对日期 2026-11-12）：

| 组件                        | 版本                                          |
| --------------------------- | --------------------------------------------- |
| CANN（Ascend-cann-toolkit） | 9.1.0（V100R001C11SPC001B243，aarch64-linux） |
| NPU 驱动（npu-smi）         | 25.5.t6.b001                                  |
| 硬件                        | Ascend910_9382                                |
| Python                      | 3.12.13（容器 `/usr/local/python3.12.13`）     |
| torch                       | 2.12.0+cpu                                    |
| torch_npu                   | 2.12.0                                        |
| transformers                | 5.8.1                                         |
| numpy                       | 2.5.1                                         |

golden（CPU 参考）只需 Python/torch/transformers；导出与执行另需 CANN
环境与 LD_PRELOAD 的 chunk proto（见下）。

0. 一键串联（在适配器目录内运行，等价于下面 ①–⑤，可选追加 bench）：

```bash
export STARTLUX_MODEL_DIR=/path/to/hf-model
bash run.sh                                         # 精度全流程
bash run.sh --iterations 400 --warmup 16 --seed 42  # 追加性能测试
```

`run.sh` 依次执行：生成输入 → CPU golden → 依赖构建（`scripts/prepare.sh`）
→ C++ executor 构建 → text AIR 导出 → 精度推理 → `tools/compare.py` 门禁
→（可选）bench；可用 `--device <chip>` / `--cases <file>` 覆盖默认值。

1. 前置（一次性；`run.sh` 会幂等重跑依赖构建）：

```bash
export STARTLUX_MODEL_DIR=/path/to/hf-model        # 权重材料，不自动下载
# 本机已验证材料：work/startlux-materials/model-6496ae641cc71b149ce9dad69bd97f2a44e7bf11
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
bash scripts/prepare.sh        # 本目录 native_proto/ 构建 chunk proto + 装 WeightNzAndMatMulV3Pass
PROTO_LIB=work/deps/chunk_gated_delta_rule_proto/install/lib/libchunk_gated_delta_rule_compat_proto.so
```

case 数据默认读随仓库固定的 `fixtures/dev.txt`（业务 dev 池），换数据用
`python3 model/generate_inputs.py --cases <file>`。

2. 分步执行（在适配器目录内运行，所有路径即默认值）：

```bash
python3 model/generate_inputs.py               # ① case → work/inputs
python3 model/golden.py                        # ② 官方 CPU 参考 → work/golden
LD_PRELOAD="$PROTO_LIB" python3 model/export_air.py --device 0   # ③ → work/build/text.air
export ASCEND_RT_VISIBLE_DEVICES=4             # npu-smi 的 Chip Logic ID
LD_PRELOAD="$PROTO_LIB" ../../runtime/build/executor       # ④ infer → work/output
python3 ../../../tools/compare.py --golden work/golden --output work/output \
    --cosine-min 0.999 --relative-l2-max 0.02  # ⑤ 精度门禁
```

3. bench（`--iterations` 出现即 bench 模式；`--warmup`/`--seed` 依附于它）：

```bash
LD_PRELOAD="$PROTO_LIB" ../../runtime/build/executor \
    --case-filter dev-intent-q00-r0 --warmup 16 --iterations 400 --seed 42
# 结果在 work/output/result.json 的 performance 块 + requests.csv
```

4. profile：msprof 外层包裹 executor 即可（自动注入 PROF 环境并导出
   op_summary/task_time），executor 无需任何 profiling 参数。

## 使用限制

- 仅纯文本 case；输入文件中任何携带图片的行都会报错（多模态走父目录）。
- **B=1 由导出的图强制执行**（io spec 轴 0 静态 + executor 复验）：
  B≠1 请求在图绑定处即报错，不会静默算错；records 格式无法表达
  padding，铺出来的宽度就是真实长度。将来若需混长组批（padding
  batching），须重导一张 3 输入（含 `actual_seq_lengths`）的图。
- GDN 因果卷积走**内置算子路线**（`_causal_depthwise_conv`：Cast → Conv2D(对称
  padding attr) → Slice → Swish，父目录融合前的已验证实现），**不依赖
  causal-conv1d vendor 算子**——2026-11-12 共享 OPP 里的 `custom_xllm_math`
  适配（`REG_OP(CausalConv1d)` 同名注册且 load_priority 更高）使所有含
  `CausalConv1dGE` 的图执行失败，本路线对此免疫；代价为性能回退（父流程
  历史实测该路线比融合算子慢 ~13-15%），vendor 适配稳定后可切回。
- 二进制依赖由本目录 `scripts/prepare.sh` 构建（源码在 `native_proto/`，父目录保有
  同一份副本）：chunk proto `LD_PRELOAD`（导出与执行都需要）、
  WeightNzAndMatMulV3Pass（可选性能 pass，缺省图仍可运行）；causal-conv1d
  vendor 算子不安装。
- golden 在 CPU 加载 FP32 官方模型，峰值约 4GB 内存；无 NPU、无
  LD_PRELOAD 依赖。
- `work/build/text.air` 首次在新目录编译时按形状逐一 TBE 编译，首跑偏慢
  属预期。
- 权重目录需含 `decision_config.json`（26 个 letter token id）且
  `tie_word_embeddings=True`；不满足时导出 fail-loudly。
