# StartLux-Decision v1.1 0.8B Instant Ascend GE 适配（独立脚本流程）

## 是什么

[StartLux-Decision-v1.1-0.8B-Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-0.8B-Instant)
决策模型（Qwen3_5ForConditionalGeneration 骨干：文本 24 层、hidden=1024，18 层
Gated DeltaNet + 6 层双向 full attention；读出 A–Z 26 列 letter logits）的独立
GE 适配目录。

交付形态为**独立脚本流程**（[docs/standalone-runtime-plan.md](docs/standalone-runtime-plan.md)
定稿方案，2026-11-05 实施并通过全流程验收）：Shell 串行调度
（`run.sh`），Python 直接完成数据（`generate_inputs.py`）、官方 CPU 参考
（`decider/golden.py`）与两阶段 split AIR 导出（`model/export_air.py`，产出
vision.air + text.air），C++ pipeline runtime
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
- **`model/`**（模型构建与导出，Python 包 `model`）：`modeling_text.py`（自持权重文本
  decoder：参数名镜像 checkpoint 零映射加载、A_log 显式 F32、融合 RoPE（BSND）、
  FIA 按行长注意力（图腿 torchair `air` 算子 / eager 腿 torch_npu host 实现，
  无 S² 掩码））、`modeling_vision.py`
  （动态视觉塔：任意图数/任意官方网格，官方位置表作为图输入，FIA TND varlen
  逐图注意力，fp16 内积/fp32 流）、
  `modeling_multimodal.py`（两阶段拆分：视觉/文本 IO 契约、组批桥（任意图数/
  网格 + 官方位置表 provider）、`from_folder` 逐张量加载与对账）、
  `unified_preprocess.py`（官方 CPU 预处理）、
  `export_air.py`（两阶段 split AIR 导出）；执行端
  `runtime/pipeline_runtime.cpp`（最小 C++ 两阶段 runner：vision AIR →
  设备常驻 features → text AIR）。
- **`decider/`**（官方 CPU 参考与业务门禁）：`golden.py`（官方 CPU golden）、
  `verify_business.py`（业务概率门禁）、`vendor/`（官方推理代码逐字节子集，
  13 个 Python 模块与固定上游 commit 一致；当前流程用到
  `images` / `prompt` / `backends.bidirectional` 及其传递依赖，其余模块按
  SOURCE.md 的固定 commit 子集原样保留）。
- `native_proto/` 与 `scripts/`：ChunkGatedDeltaRule GE 兼容原型源码与构建脚本
  （加载靠单一 `LD_PRELOAD` 契约，无 python activator；纯文本交付目录
  `../startlux-decision-v1.1-0.8b-instant-text/native_proto/` 保有同一份源码副本）。
- `prepare.sh` 同时构建安装 **WeightNzAndMatMulV3Pass** 自定义融合 Pass
  （`third_party/custom_development_code/fusion_pass/`）：图编译期将
  MatMul/MatMulV2 常量权重转 FRACTAL_NZ 重建为 MatMulV3（系统级安装至
  CANN vendors 目录；安装后首次图编译按形状逐一 TBE 编译、kernel_meta
  缓存）。2026-11-09 实测：ticket-team 单 case AI 时间 −14.4%。
- `prepare.sh` 还构建安装 **causal-conv1d** vendor 算子包
  （`third_party/ascend-ops/causal-conv1d`：vendored fla AscendC kernel +
  `CausalConv1dGE` GE 包装 + torch 绑定）：GDN 层因果卷积由 7 算子链
  （Transpose×2 + TransData×2 + Conv2D + Slice + Swish）融合为单算子
  （含 SiLU、dim-last 直连 in_proj 输出），图腿经 torchair converter 落
  `CausalConv1dGE` 节点，`pipeline_runtime` 在 GEInitialize 前 dlopen
  `libcust_opapi.so`（C++ 进程无 activator；GE 自动发现不到该库）。
  2026-11-11 实测：bench e2e P50 −13.1%、每请求 AI 时间 −12.3%，见
  performance.md 附录。
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
python3 -m unittest tests.test_startlux_native_proto tests.test_startlux_unified_preprocess
python3 -m unittest tests.test_torchair_example_inputs
```

### dev.txt 旁路（业务 query 意图 case，独立流程）

`run_dev.sh` 是与主流程并行的独立旁路，**不改动官方 5 case 池**：把业务
提供的 dev.txt（电商 query 意图分类，18 行 = 9 query × 2 选项序变体，
4B teacher 标注 targets）跑通"数据 → CPU golden → 复用或重建 AIR/依赖 →
精度推理 → 数值比较 → 业务门禁 + teacher 分布对比 → 逐 case bench（18 ×
case_filter）+ 全池采样 bench → 汇总"。入口：

```bash
export STARTLUX_MODEL_DIR=/path/to/hf-model
bash models/startlux-decision-v1.1-0.8b-instant/run_dev.sh [--device <chip>]
```

脚本：`generate_dev_inputs.py`（数据，官方渲染链）、`decider/golden_dev.py`
（复用官方 `OfficialReference`）、`decider/verify_dev.py`（业务门禁 +
teacher 报告）。case 文件默认 `work/devcase/source/dev.txt`（缺失自动从
业务 URL 下载）；产物全部在 `work/devcase/`（不提交）。差异口径：dev 池
业务概率门限 0.02（平坦 7 选项分布放大 |Δp|，官方池的 0.01 不适用，详见
`verify_dev.py` 常量注释）；teacher 对比（top1 一致率 / 逐标签 Δp / KL /
r0–r1 选项序不变性）**仅报告**，不阻断、不影响退出码。

## 使用限制

- dev.txt 旁路：teacher 分布对比仅报告、不设门禁；dev 池业务概率门限为
  0.02（与官方池 0.01 不同口径，原因见 `decider/verify_dev.py` 常量注释）；
  旁路产物全部在 `work/devcase/`，不提交 Git。
- 图边界：**对齐官方**——任意图片数量（每请求 ≤ 官方 serve 上限 16）与任意官方
  smart_resize 网格（每图 256..4096 patch 行，即 256²..1024² 像素，保持长宽比）；
  纯文本调用**完全跳过视觉阶段**（runtime 直喂 1×1024 零特征张量，注入
  where 丢弃、得分位相同——与官方 media=None 不跑视觉塔的行为一致）；
  **S ≤ 模型原生 `max_position_embeddings`（当前 checkpoint 为 262144，编码期从
  folder config 派生；S=262144 已实测可执行且数值正确）；B 无编码期上限**——
  真实约束是运行期 HBM（激活 ~B·S），超内存组合以 GE 内部错误失败而非静默截断；
  **case 按真长执行**（宽度 = max 行真长，无补齐下限）；超限拒绝、不截断。
- 混精度：视觉塔 FP32 残差流/位置表 + FP16 Linear/Conv 内积 + **FP16 FIA
  注意力（逐图 varlen，唯一精度让步，门禁把关）**；文本 decoder BF16；
  A_log 以模块声明 F32（vLLM 同款，加载即位精确）、head26 F32。
- 交付形态：**两阶段 split AIR**（vision.air + text.air）由
  `runtime/pipeline_runtime` 串联执行——features 设备常驻直连（不经主机）、
  每图独立 stream（单流交替两图会强制 GE 重绑执行计划，实测吞吐 -83%）。
- wide_27 跨 dtype 分支披露（保留）：FP32 末轮第 6 名入围者 cause_18，官方
  BF16 与 GE 均为 cause_23；业务判据为 GE vs 相同 dtype 官方 BF16 动态参考，
  详见 `validation.md`。
- processor 链：AutoProcessor 整体 factory 加载失败，官方组件分别构建成功；
  conv4 使用共享 `causal_conv4`。
- GDR：唯一路线为原生 CANN kernel + 项目兼容原型（非官方文件，仅 IR 注册不
  含计算内核），加载方式为单一 `LD_PRELOAD` 契约（run.sh 对 export/infer/bench
  子进程注入，export_air.py 启动即校验；自动 vendor 加载已复现崩溃并弃用），
  CANN 升级/换 SoC 需重跑其门禁且无标准路线兜底。MegaGdnPrefillV2
  融合算子路线实测净负（融合算子内部 128 对齐、本模型 shape 偏小：B1 −28% /
  B2 −51%，吞吐 −23%），代码已移除，测量证据见 validation.md /
  performance.md。
- GDR conv：GDN 因果卷积依赖 causal-conv1d vendor 包（prepare.sh 安装至
  CANN vendors，系统级）：`torch.ops.ascend_ops.causal_conv1d` 图腿落
  `CausalConv1dGE`（EagerExecuteOp），数值为 kernel 固有 bf16 行为
  （与上游 fla-npu wheel 逐位一致；端到端 relL2 0.0057，见 performance.md
  附录）。限制：dense prefill、`(B,S,D)` 布局、D 需 16 倍数、W∈{2,3,4}。
  C++ runtime 在 GEInitialize 前 dlopen 该 `libcust_opapi.so`，进程退出时
  其静态析构在 GEFinalize 后触发 glibc fastbins 损坏——runtime 成功路径
  已显式 teardown 后 `_exit(0)` 规避；带 GE profiling 的运行保留正常退出
  （msprof 的 atexit 导出先完成，仅退出码受损，预期 134/abort）。
- 性能：口径与数值见 [performance.md](performance.md)；graph executions/s
  不是完整业务 QPS。bench 前需人工确认卡空闲。
- AIR 导出非字节可复现（同源同参数多次导出 sha256 不同、字节数恒定
  1,507,412,645（causal-conv1d 替换后；曾为 1,708,279,731，融合 RoPE 与
  融合 full-attention 后为该值），嵌入 IO 契约经 runtime write_io/read_io
  验证一致）；正式证据以数值验收为准。
