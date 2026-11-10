# StartLux-Decision v1.1 0.8B Instant Ascend GE 适配（独立脚本流程）

## 是什么

[StartLux-Decision-v1.1-0.8B-Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant)
决策模型（Qwen3_5ForConditionalGeneration 骨干：文本 24 层、hidden=1024，18 层
Gated DeltaNet + 6 层双向 full attention；读出 A–Z 26 列 letter logits）的独立
GE 适配目录。

交付形态为**独立脚本流程**（[docs/standalone-runtime-plan.md](docs/standalone-runtime-plan.md)
定稿方案，2026-11-05 实施并通过全流程验收）：Shell 串行调度
（`run.sh`），Python 直接完成数据（`generate_inputs.py`）、官方 CPU 参考
（`decider/golden.py`）与 AIR 导出（`model/model.py` + `model/export_air.py`），现有 C++ runtime
完成推理与性能，公共工具（`tools/compare.py`）完成精度比较，业务概率门禁由
`decider/verify_business.py` 承担。**不依赖 `core` 流程框架**，无 YAML / adapter /
handle；固定参数集中在 `run.sh` 顶部。

单图交付：视觉塔 + 图像特征注入 + 三轴 MRoPE + 文本 decoder + 26 列读出头整
合在单一动态 AIR、单次 GESession 中执行。验收状态见
[validation.md](validation.md)（历史精度事实）与
[performance.md](performance.md)（P8 正式基准 + 独立流程验证测量附录）。

目录内容：

- **独立流程入口**：`run.sh`（完整流程）、`prepare.sh`（构建安装 pass 与自
  定义算子）、`generate_inputs.py`（输入池生成）、
  `scripts/env_python.sh`（A3 容器 Python 引导）。
- **`model/`**（模型构建与导出）：`model.py`（单图组合入口）、
  `startlux_npu_adapter.py`（native chunk GDR 文本 decoder）、`multimodal.py`
  （统一图组合与组批校验，图边界常量与 IO 契约单一来源）、`vision.py`
  （固定网格视觉塔：FP32 残差流 + FP16 Linear/Conv 内积）、
  `unified_preprocess.py`（官方 CPU 预处理）、
  `export_air.py`（AIR 导出与请求生成）。
- **`decider/`**（官方 CPU 参考与业务门禁）：`golden.py`（官方 CPU golden）、
  `verify_business.py`（业务概率门禁）、`vendor/`（官方推理代码逐字节子集，
  13 个 Python 模块与固定上游 commit 一致；当前流程用到
  `images` / `prompt` / `backends.bidirectional` 及其传递依赖，其余模块按
  SOURCE.md 的固定 commit 子集原样保留）。
- `native_proto/` 与 `scripts/`：ChunkGatedDeltaRule GE 兼容原型源码、
  构建脚本与显式局部加载 activator。
- `prepare.sh` 同时构建安装 **WeightNzAndMatMulV3Pass** 自定义融合 Pass
  （`third_party/custom_development_code/fusion_pass/`）：图编译期将
  MatMul/MatMulV2 常量权重转 FRACTAL_NZ 重建为 MatMulV3（系统级安装至
  CANN vendors 目录；安装后首次图编译按形状逐一 TBE 编译、kernel_meta
  缓存）。2026-11-09 实测：ticket-team 单 case AI 时间 −14.4%。
- `work/`：全部生成产物（inputs / golden / build / infer / bench / deps /
  logs / fixtures），不提交 Git。

## 怎么用

环境（沿用既有已验证组合）：CANN 容器（CANN 9.1.0、torch 2.12.0+cpu、
torch_npu 2.12.0、transformers 5.8.1，Ascend910_9382）与 NPU。前置条件：

1. 权重材料通过环境变量提供（不自动下载）：

```bash
export STARTLUX_MODEL_DIR=/path/to/hf-model
# 本机已验证材料：work/startlux-materials/model-6496ae641cc71b149ce9dad69bd97f2a44e7bf11
```

2. **官方预处理的 Pillow/torchvision 是环境前置依赖**（任务局部安装，勿系统
   安装）：已验证安装为
   `work/startlux-validation/p6-official/deps/{pillow,torchvision}`（wheel
   sha256 文件在同目录）；`run.sh` 会自动把它们前置到 PYTHONPATH。缺包时会在
   生成输入阶段自然报错（DEPS_HINT 指向上述路径）。

完整流程（从任何当前目录启动一致；bench 前会等待人工确认卡空闲）：

```bash
bash models/startlux-decision-v1.1-0.8b-instant/run.sh [--device <chip>]
```

NPU 卡用 `--device` 指定（npu-smi 的 Chip Logic ID，默认 4）；其余固定参数在
`run.sh` 顶部（精度门限 cosine 0.999 / relL2 0.02、单实例 / warmup 32 /
总测量请求 2048 / seed 42），是唯一可调处。
输入 case 池 = **仅官方仓库 examples 的 5 个业务 case**（工单三问题 B3/
三个 B1 拆分 + 配送照片 noul；bench 池 = 精度池）。流程：检查环境及权重 →
生成输入（官方预处理一次完成）→ CPU golden →
构建/安装 pass 与自定义算子（`work/deps/`）→ 构建 C++ runtime（复用
`runtime/build.sh`）→ 导出单 AIR 并自动生成 infer/bench 请求 → 精度推理 →
数值比较（`tools/compare.py`）→ 业务概率校验 → **人工确认卡空闲** → 性能
测试。产物在 `work/infer/`（comparison.json + business.json）与
`work/bench/`（result.json + requests.csv），日志在 `work/logs/`。

单独重跑（每条只执行本步，缺输入直接报错）：

```bash
python3 models/startlux-decision-v1.1-0.8b-instant/generate_inputs.py   # 数据
python3 models/startlux-decision-v1.1-0.8b-instant/decider/golden.py    # 官方 CPU 参考
bash   models/startlux-decision-v1.1-0.8b-instant/prepare.sh            # 依赖构建
# 导出（LD_PRELOAD 指向 work/deps/.../install/lib/libchunk_gated_delta_rule_compat_proto.so）：
LD_PRELOAD=<proto.so> python3 models/startlux-decision-v1.1-0.8b-instant/model/export_air.py \
    --device 0 --instances 1 --warmup 32 --iterations 2048 --seed 42
LD_PRELOAD=<proto.so> runtime/build/ge_runtime < models/.../work/infer/request.json
python3 tools/compare.py --golden models/.../work/golden --output models/.../work/infer \
    --cosine-min 0.999 --relative-l2-max 0.02
python3 models/startlux-decision-v1.1-0.8b-instant/decider/verify_business.py
```

CPU 单测（pytest 未安装，用例基于 unittest）：

```bash
python3 -m unittest tests.test_startlux_native_proto tests.test_startlux_adapter
python3 -m unittest tests.test_startlux_multimodal tests.test_startlux_unified_preprocess
python3 -m unittest tests.test_torchair_example_inputs
```

## 使用限制

- 图边界：官方 processed grid [1,16,16]（256x256 输入）；每次调用全局最多 2
  张图（pixel_values 静态 [512,1536] 双图位）；B≤4、S≤2048；**case 按真长执行**
  （宽度 = max 行真长，无补齐下限）；超限拒绝、不截断。边界常量单一来源为
  `multimodal.py` / `vision.py` 顶层，生成输入时强制执行。
- 混精度：视觉塔 FP32 残差流 + FP16 Linear/Conv 内积，文本 decoder BF16；
  A_log 自 checkpoint 逐层恢复 F32、head26 F32。
- wide_27 跨 dtype 分支披露（保留）：FP32 末轮第 6 名入围者 cause_18，官方
  BF16 与 GE 均为 cause_23；业务判据为 GE vs 相同 dtype 官方 BF16 动态参考，
  详见 `validation.md`。
- processor 链：AutoProcessor 整体 factory 加载失败，官方组件分别构建成功；
  conv4 使用共享 `causal_conv4`。
- GDR：唯一路线为原生 CANN kernel + 项目兼容原型（非官方文件，仅 IR 注册不
  含计算内核），加载方式为显式局部加载（activator dlopen + run.sh 对
  export/infer/bench 子进程注入 LD_PRELOAD；自动 vendor 加载已复现崩溃并弃
  用），CANN 升级/换 SoC 需重跑其门禁且无标准路线兜底。MegaGdnPrefillV2
  融合算子路线实测净负（融合算子内部 128 对齐、本模型 shape 偏小：B1 −28% /
  B2 −51%，吞吐 −23%），代码已移除，测量证据见 validation.md /
  performance.md。
- 性能：口径与数值见 [performance.md](performance.md)；graph executions/s
  不是完整业务 QPS。bench 前需人工确认卡空闲。
- AIR 导出非字节可复现（同源同参数多次导出 sha256 不同、字节数恒定
  1,708,279,731，嵌入 IO 契约经 runtime write_io/read_io 验证一致）；正式证据
  以数值验收为准。字节数曾为 1,907,534,423，融合 RoPE 与融合 full-attention
  后降至当前值（−10%，见 performance.md 附录）。
