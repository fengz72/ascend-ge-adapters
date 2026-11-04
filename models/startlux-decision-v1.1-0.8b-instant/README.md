# StartLux-Decision v1.1 0.8B Instant Ascend GE 适配

## 是什么

[StartLux-Decision-v1.1-0.8B-Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant)
决策模型（Qwen3_5ForConditionalGeneration 骨干：文本 24 层、hidden=1024，18 层
Gated DeltaNet + 6 层双向 full attention；读出 A–Z 26 列 letter logits）的独立 GE
适配目录。

最终交付（已验收）：视觉塔 + 图像特征注入 + 三轴 MRoPE + 文本 decoder + 26 列
letter 读出头整合在**单一 AIR、单次 GESession** 中执行；拆分多图仅允许用于调试
精度。正式 config 唯一：`config/model.yaml`（unified_adapter）。正式 AIR
`work/startlux-decision/build/model.air`（1,907,534,423 bytes，sha256
c1f8816ee25e03119b2522e55adb49ebf42a88692791b179920f5f60221edba1），路径不变。

当前精度/性能验收状态见 [validation.md](validation.md) 与
[performance.md](performance.md)；适配方案见
[docs/adaptation-plan.md](docs/adaptation-plan.md)，生产集成说明见
[docs/production-integration.md](docs/production-integration.md)。历史阶段文档
（决策语义、native 依赖、P6 统一图过程记录）与本次清理前的原始文档快照见
[work/startlux-archive/cleanup-20261102/](../../work/startlux-archive/cleanup-20261102/)。

独立脚本与 C++ runtime 执行的待实施方案见
[docs/standalone-runtime-plan.md](docs/standalone-runtime-plan.md)；该文档记录目标结构，
当前正式入口与验收结果保持不变。

目录内容与保留原因：

- `unified_adapter.py` / `unified_preprocess.py` / `unified_readout.py` /
  `unified_verify.py` / `generate_unified_cases.py` / `multimodal.py`：正式
  统一图路线与 CPU 单测直接加载的模块。
- `startlux_npu_adapter.py` / `decision.py` / `reference.py` / `ge_readout.py` /
  `verify.py`：**正式路线与 CPU 测试共享依赖，不是废文件**——`decision.py`/
  `reference.py` 承载官方业务语义与独立 CPU FP32 oracle，`ge_readout.py` 为
  真实 GESession read-out 共享实现（`unified_readout.py` 与
  `scripts/run_unified_performance.py` 加载），`startlux_npu_adapter.py` 为
  文本 decoder 适配器（2026-11-02 起收敛为 native chunk GDR 单路线，原
  `startlux_native_adapter.py`/`gated_delta.py`/`operator_checks.py` 已按
  决策删除），`verify.py` 为验收钩子；tests/ 下 CPU 单测从本目录逐文件加载
  它们。
- `vendor/`：官方推理代码逐字节子集（13 个 Python 模块与固定上游 commit
  一致，`work/startlux-cannbot/review-vendor.json`）。
- `native_proto/` 与 `scripts/`：ChunkGatedDeltaRule GE 兼容原型源码、构建/
  显式局部加载环境脚本、P8 监控与基准脚本（均保留原路径）。
- `config/model.yaml`、`fixtures/`：正式唯一模型配置与固定读出 fixtures。

## 怎么用

环境：CANN 容器（CANN 9.1.0、torch 2.12.0+cpu、torch_npu 2.12.0、
transformers 5.8.1，Ascend910_9382）与 NPU；依赖为 CANN + 容器 Python +
任务局部 pip --target 包（Pillow/torchvision，PYTHONPATH 注入），正式
AIR `work/startlux-decision/build/model.air`、runtime
`runtime/build/ge_runtime`。CPU 单测不需要 NPU，但仍按
`work/startlux-validation/p7-final/env.sh` 的 scoped 环境为准；work/ 中保存
模型材料、任务局部依赖和验收证据，正式运行依赖这些材料，本目录不是自包含
离线包。

CPU 单测（pytest 未安装，用例基于 unittest；2026-11-02 native-only 收敛后
302 项全过，收敛前 317 项全过，见 `validation.md`）：

```bash
python3 -m unittest tests.test_startlux_decision tests.test_startlux_native_proto
python3 -m unittest tests.test_startlux_adapter tests.test_startlux_ge_readout
python3 -m unittest tests.test_startlux_verify tests.test_startlux_multimodal
python3 -m unittest tests.test_startlux_unified_adapter tests.test_startlux_unified_preprocess
python3 -m unittest tests.test_startlux_unified_readout tests.test_startlux_unified_verify
python3 -m unittest tests.test_startlux_unified_cases tests.test_torchair_example_inputs
```

正式六阶段 CLI（唯一正式配置 = 统一图 `config/model.yaml`；2026-10-31 P7 已
真实通过一次）：先 source 正式 scoped 环境（CANN + A3 容器 Python + 8 线程 +
任务局部 Pillow/torchvision + STARTLUX_MODEL_DIR；ASCEND_RT_VISIBLE_DEVICES=1
取 npu-smi info -m 的 Chip Logic ID 1，对进程暴露为 runtime device 0，可按卡况
改配），prepare 之后做 canonical proto 的一次加载（勿再 source 携带其他
compat.so 的旧任务 env，双注册已知会崩）：

```bash
source work/startlux-validation/p7-final/env.sh
CFG=models/startlux-decision-v1.1-0.8b-instant/config/model.yaml
python3 -m core.pipeline data     --config $CFG --target Ascend910_9382
python3 -m core.pipeline baseline --config $CFG --target Ascend910_9382
python3 -m core.pipeline prepare  --config $CFG --target Ascend910_9382
source models/startlux-decision-v1.1-0.8b-instant/scripts/env_chunk_proto.sh \
  work/startlux-decision
python3 -m core.pipeline build    --config $CFG --device 0
python3 -m core.pipeline infer    --config $CFG --device 0
python3 -m core.pipeline compare  --config $CFG --target Ascend910_9382
```

业务读出入口（正式 AIR + 官方 Decider，每 read-out 一个冷 ge_runtime 子进程）：
`work/startlux-validation/p7-final/run_p7_business.py`。算子级门禁（P1 历史
入口）与历史实验入口的命令在归档文档快照中保留，不在本文重复。

## 使用限制

- 图边界：官方 processed grid [1,16,16]（256x256 输入）；每次调用全局最多 2
  张图（pixel_values 静态 [512,1536] 双图位）；B<=4、S<=2048、最小物理宽度
  256（短行右 pad 至该下限，actual_seq_lengths 保真长）；超限请求拒绝、不截
  断。边界证据与 B4 补验见 `validation.md`。
- 混精度：视觉塔 FP32 + 文本 decoder BF16；原 checkpoint 视觉权重 BF16 存储、
  本工程导出前转 FP32；A_log 自 checkpoint 逐层恢复 F32、head26 F32。BF16
  视觉路线 relL2 0.1139/0.1095 未过门限，已归档于
  `work/startlux-validation/p6-official/archive/bf16-failed-route`。
- wide_27 跨 dtype 分支披露（保留）：FP32 末轮第 6 名入围者 cause_18，官方
  BF16 与 GE 均为 cause_23（GE 落在 BF16 分支）；不能宣称全 FP32 业务分布门限
  通过，业务判据为 GE vs 相同 dtype 官方 BF16 动态参考，GE vs FP32 同 prompt
  16/16 过（详见 `validation.md`）。
- processor 链：AutoProcessor 整体 factory 加载失败（Qwen3VLVideoProcessor
  backend setter），官方组件分别构建（Qwen2VLImageProcessorPil +
  Qwen3VLProcessor + Qwen2Tokenizer）成功。
- conv4：生产统一使用共享 `causal_conv4`（4-tap 位移乘加 + SiLU）。实测
  TorchAir/GE 对真实配置（C=6144、kernel=4、groups=C、padding=3）的 Conv1D
  转换存在数值错误，该结论仅针对该实测配置。
- GDR：唯一路线为原生 CANN kernel + 项目兼容原型（非官方文件，仅 IR 注册不含
  计算内核），加载方式为显式局部加载（`env_chunk_proto.sh`/LD_PRELOAD；自动
  vendor 加载已复现崩溃并弃用），CANN 升级/换 SoC 需重跑其门禁且**无标准路线
  兜底**（纯 Torch 标准路线已按决策删除，P3 standard 7/7 验收转为历史声明）。
- 性能：唯一正式实测为 P8 单实例 warm 稳态基准，口径与数值见
  `performance.md`；graph executions/s 不是完整业务 QPS。NPU 侧测试已停止，
  不自动重启监控或基准。
