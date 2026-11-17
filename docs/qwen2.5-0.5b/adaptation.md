# Qwen2.5-0.5B 适配说明

安装和快速上手见[模型 README](../../models/qwen2.5-0.5b/README.md)，实验与验证入口见[文档索引](README.md)。本文说明自包含流程的计算路径和部件间文件契约；流程只通过 `work/` 下的文件交互，不导入 `core/`。

## 源码分工

| 文件 | 职责 |
|---|---|
| [model/modeling_qwen.py](../../models/qwen2.5-0.5b/model/modeling_qwen.py) | NPU 可执行的 packed trunk、平台 Prefix 算子绑定和 8 列评分头 |
| [model/generate_inputs.py](../../models/qwen2.5-0.5b/model/generate_inputs.py) | 历史 RNG 输入池与 `ge-records/3` 张量索引 |
| [model/golden.py](../../models/qwen2.5-0.5b/model/golden.py) | 未 patch 的 CPU FP32 独立参考与 ge-compare/1 输出布局 |
| [model/export_air.py](../../models/qwen2.5-0.5b/model/export_air.py) | torchair `dynamo_export` 导出动态 packed AIR |
| [runtime/executor.cpp](../../runtime/executor.cpp) | 单 AIR 的位置绑定执行（精度/性能） |
| [config/target_tokens.json](../../models/qwen2.5-0.5b/config/target_tokens.json) | 有序业务评分 token（8 列） |
| [scripts/prepare.sh](../../models/qwen2.5-0.5b/scripts/prepare.sh) | 平台 Prefix 算子与权重 NZ Pass 构建，生成 `work/deps/env.sh` |

`run.sh` 串联各步骤；`scripts/prepare.sh` 构建依赖，二者不属于契约部件。

## 计算路径

`QwenPackedModel.forward(input_ids, position_ids, actual_seq_lengths)` 返回 float32
`[N, 8]` 评分。输入是共享 prefix 的 packed 布局：`segments = [prefix] + requests`，
`actual_seq_lengths = cumsum(段长)`（N+1 个值）。prefix 的 Q/KV 只保存并计算一次，
业务行取每条请求的末 token，即 `bounds[1:] - 1`。

| 模块 | 实现 |
|---|---|
| RMSNorm | `torch.ops.npu.npu_rms_norm`，输出回到输入 dtype |
| RoPE | 构造时预计算 `[2048, head_dim]` 的 cos/sin 表（FP16），按 `position_ids` Gather；`torch_npu.npu_apply_rotary_pos_emb(layout="TND", rotary_mode="half")` |
| Attention | 构造时按平台绑定 Prefix 算子，forward 保留 Q/K/V/O 投影与 TND reshape |
| MLP | `down_proj(silu(gate_proj(x)) * up_proj(x))`，24 层 |
| 评分头 | 末次 norm 后按 `bounds[1:]-1` 选末 token，`float @ score_head^T` |

平台 Prefix 算子：

| 平台 | 入口 | bounds | 说明 |
|---|---|---|---|
| a3 | `torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score.tensor` | int64 | AscendC `PrefixInferAttentionScore`；`sparse_mode=2` 读取固定 `[2048,2048]` 压缩因果掩码，`pre_tokens=INT_MAX`、`next_tokens=0` 即因果；bounds 作为图输入保持动态 |
| a5 | `torch.ops.ascendc_ops.prefix_attention` | int32 | CANNBotDSL `PrefixAttention`（FP16 交付） |

两个工厂签名统一为 `(config, model)`，在 `QwenPackedModel.__init__` 里绑定一次、
24 层共享同一个闭包，因此 `forward` 里没有平台分支。算子包是 side-effect import
（`torch.ops` 只是注册表的惰性视图，包本身不导出可调用对象），写在工厂内部而不是
模块顶部：只有选中的平台才付 JIT 编译与 `ASCEND_CUSTOM_OPP_PATH` 的代价。
`[2048,2048]` 压缩掩码由 **a3 工厂自己** `register_buffer`（a5 的 buffer 只有
`cos`/`sin`/`score_head`，实测确认），闭包里按 `model.compressed_mask` **惰性**读取
——`nn.Module.to()` 会把 buffer 重新绑定成新张量，构造期捕获引用会拿到迁移前的
CPU 副本。

Qwen2.5-0.5B 将 `lm_head` 与 `embed_tokens` 绑定，checkpoint 不含独立 head 张量。
`install_score_head(token_ids)` 在 `load_state_dict` 之后、`forward` 之前从 tied
embedding 取对应行作为评分头。trunk 权重从 `model.safetensors` 的 `model.*`
键严格加载。

### 逐件删除实测（eager NPU 对照 CPU golden，case-0000/0001）

| 部件 | 删掉后 | 结论 |
|---|---|---|
| `QwenRMSNorm` 末尾的 `.to(x.dtype)` | min cosine 0.999983251 / max relL2 0.016223159，与保留时逐位相同 | **已删**：`npu_rms_norm` 对 FP16 输入就返回 FP16（只有 rstd 是 FP32） |
| `install_score_head` 的 `with torch.no_grad()` | 导出与精度不变 | **已删**：改由 `load_model` 先 `requires_grad_(False)` 再装头 |
| `atten_mask=None`（保留 `sparse_mode=2`） | min cosine **-0.991** / max relL2 6.19 | 不可删：压缩因果模板是算子的必需输入 |
| `sparse_mode=0` 且不给 mask | 同上（-0.991 / 6.19） | 不可删 |
| 压缩 mask 缩到 `[256,256]` | min cosine **-0.980** / max relL2 3.86 | 不可缩：模板尺寸固定 `[2048,2048]`，`MAX_SEQ_LEN` 同时限定 RoPE 表深度与算子的 `prefix+最长请求 ≤ 2048` |
| `hidden.float() @ score_head.T` 的 FP32 读出 | 未测（历史 FP16 读出的最大 relL2 为 0.0468，已贴近 0.05 门限） | 保留：FP32 读出把最大 relL2 降到 0.0338，且 `tools/compare.py` 只支持 float32 |
| `actual_seq_lengths[1:].to(torch.int64)` | 未测（a3 下 bounds 本就是 int64） | 保留：a5 的 bounds 是 int32，`index_select` 需要 int64 索引 |

## 输入池

`generate_inputs.py` 逐字保留历史输入生成脚本的 NumPy RNG 序列：
`RandomState(0)`，先抽共享 prefix 长度 20–25，再由前 10 条
`round(exp(normal(4.997, 0.167)))` 得到请求长度并截断到 `[prefix+1, 218]`；token
使用独立 `RandomState(seed*100003+case_index)`。第 10..15 条（batch16 扩展）由
sha256 派生的 per-case RNG 补足，不改动前 10 条历史请求。

历史统计量（算法逐字搬移，供对照）：

| 池 | 总长均值 | P99 | 范围 |
|---|---:|---:|---:|
| 历史 batch10 | 149.345 | 212.02 | 89–218 |
| 当前 batch16 | 149.769 | 218 | 84–218 |

每个 case 生成三个一维张量并写入 `work/inputs/inputs.bin`，索引写入
`work/inputs/records.json`（schema `ge-records/3`）：

| 张量 | dtype | 语义 |
|---|---|---|
| `input_ids` | int64 | packed token：先 prefix，再各请求私有 token |
| `position_ids` | int64 | prefix 自 0 起，每条请求自 `len(prefix)` 重新计数 |
| `actual_seq_lengths` | int64（a3）/ int32（a5） | 各段累计结束位置，N+1 个 |

`records.json` 记录每个 case 的 tensors 偏移、dtype、shape 和 `samples`；
`input_ids`/`position_ids` 始终 int64，仅 bounds dtype 随平台变化。该阶段只需
`QWEN_MODEL_DIR` 下 `config.json` 的 `vocab_size`，不加载权重、不需要 NPU。

## CPU golden

`golden.py` 用 `AutoModelForCausalLM` 以 float32 在 CPU 上加载同一权重目录，
不应用任何方法 patch、不导入模型包。按 `input_ids` 与 bounds 重建
`prefix + request_i`，每个 case 补齐为 `[N, W]` 的 padded batch，用
`attention_mask`/`position_ids` 取唯一真实末位置
`(attention_mask.sum(1) - 1).unique(sorted=True)`，只物化这些位置的 logits，
再按 `config/target_tokens.json` 的 8 个 token 取列得到 `[N, 8]`。

输出为 `tools/compare.py` 读取的 ge-compare/1 布局：`work/golden/outputs.bin`
（拼接的 `[N, 8]` float32）+ `work/golden/result.json`
（`outputs[case_id] = [{instance: 0, tensors: [...]}]`）。该阶段不需要 NPU 或
`LD_PRELOAD`，`OMP_NUM_THREADS`/`MKL_NUM_THREADS` 在导入 torch 前固定为 8。

## AIR 导出

`export_air.py` 用 torchair `dynamo_export` 导出单图，只设一个开关：
`CompilerConfig.experimental_config.frozen_parameter = True`（把 FP16 权重烘进图）。
`dynamic=True` 让每个输入的第 0 轴符号化，因此变长输入池共用一个 AIR，
`scores [N, 8]` 的第 0 轴随之动态。trace 示例 `T=20`（prefix 4 + 两条 8-token
请求，bounds `(4, 12, 20)`）：动态轴必须以 ≥2 的尺寸 trace，否则会被特化成静态图。

torchair 2.9.0（torch 2.9.0+cpu / torch_npu 2.9.0.post2）实测结论，逐条都做过
「删掉后重导出 + 全量 100 case 精度门禁」验证，删掉后导出图的算子直方图与保留时
逐项相同、worst cosine/relative L2 完全一致（0.999439462 / 0.033784056）：

- **reshape 的动态轴必须写 `-1`**（唯一必须遵守的一条）。写成显式符号尺寸
  （`hidden.view(hidden.shape[0], heads, dim)`）会让 dynamo 额外提升一条 SymInt
  链：导出图多出 2 个 Data 节点（共 5 个，破坏位置绑定），并为每个 reshape 生成
  `Pack(Shape 取值, 常量, 常量)` 拼形状，而该 Pack 的输入被连到同名的 cos 常量表
  上，`CompileGraph` 直接失败于 `op[Pack], the input shape dims should be equal`。
  改用 `-1` 后 Pack 归零、Data 恰为 3 个（index 0/1/2 与 records 的 tensors
  顺序一致，dtype/shape 为 `DT_INT64 [-1]`）。
- **不需要** `torch._dynamo.mark_dynamic`：`dynamic=True` 已经把三个输入的第 0 轴
  符号化成 `dim: -1`。
- **不需要** `torch._dynamo.config.patch(specialize_float=True)`：本模型闭包里的
  Python float（norm eps、attention scaling）没有被提升成额外的 f64 图输入，Data
  节点仍是 3 个。
- **不需要** StartLux `-text` 导出里的 `bind_input` 自定义算子、
  `_ge_adapter_name` 标记、Data 节点前置重排与 `dynamo.pbtxt` 双射校验：那套是
  torchair 2.12.0 位置烘焙被 SymInt 链打乱的 workaround，本流程用 `-1` 从源头
  避免了 SymInt 链。图输入顺序由 `dynamo_export(*values)` 的实参顺序决定，正确性
  由 `tools/compare.py` 门禁兜底。
- **不需要**导出后的 `torch.npu.synchronize()`、`.eval()`、以及预先 `unlink` 旧
  AIR：`dynamo_export` 会就地覆盖已存在的只读 `.air`（实测重复导出成功、产物
  可用），模型里没有 dropout/batchnorm 之类依赖 training 标志的模块。

导出后删除 `dynamo.pbtxt`（重复冻结权重的 protobuf 文本，本图约 3.6 GB），保留
`work/build/packed.air`（993754309 字节，FP16 冻结权重）。导出进程需已
`source work/deps/env.sh`，使平台 Prefix 算子可导入。

## 执行

`runtime/executor.cpp` 加载一个 AIR，按 `records.json` 将 `tensors[i]` 位置绑定到
图输入 i（张量名仅作信息），逐 case 写入 `work/output/outputs.bin` 与
`result.json`，布局与 golden 一致，交由 `tools/compare.py` 比较。带
`--iterations N` 时进入 bench：`--instances K` 个实例闭环并发、每实例预热、
seed 抽样、`--aicore-num A|B` 设每算子核上限，输出延迟与 `requests.csv`。
executor 只保留 `tools/compare.py` 与人工阅读需要的字段：`result.json` 顶层为
`status` + `outputs`（infer）或 `status` + `performance`（bench），逐请求明细在
`requests.csv`（`per_case`、`model_ms` 等可由它推导的字段已删除）。
`--aicore-num` 不做格式校验，非法值由 GE 自己拒绝（实测
`Invalid_Argument(E10001): Value garbage for parameter ge.aicoreNum is invalid`）。

`GEInitialize` 会拉起 CANN 内嵌 Python（AOE/TE）并 `import numpy`，而
`set_env.sh` 只把 CANN 自带的 site-packages 放进 `PYTHONPATH`；容器 Python 装在
别处时必须补上它的那一份，否则报
`Environment_Error_Import_Python_Module_Failed: No module named 'numpy'`。
`run.sh` 在 source CANN 环境后做这件事（旧 `scripts/env_a3.sh` 的同类职责）。

## 平台与限制

A5 Prefix 需要 CANN ≥ 9.3.0 的辅流接口；当前 A3（CANN 9.0.0）实测，A5 路径未在
本机验证。交付 dtype 为 FP16，CPU golden 为 FP32。CANN 辅流接口核验分别见
[9.1.0](../cann-9.1.0-eager-attached-stream-support.md)、
[9.2.0](../cann-9.2.0-eager-attached-stream-support.md)、
[9.3.0](../cann-9.3.0-eager-attached-stream-support.md)。
