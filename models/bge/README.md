# BGE Ascend GE 适配

## 1. 背景

### 1.1 模型来源

业务侧提供的 BGE ONNX 导出，内联权重约 1.35 GB，SHA-256 `2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0`。由环境变量 `BGE_MODEL_PATH` 给出，**不入库、不自动下载**；已验证材料位置见 [`fixtures/SOURCE.md`](fixtures/SOURCE.md)。

骨干是 24 层双向 padding-mask 注意力（BSND 布局，Q/K 共享 `1/sqrt(D)` 缩放链）；打分头对 10112 个固定候选做「投影 → L2 归一化（带 epsilon 下限）→ 二次投影」。基线图保留原始结构：30 个 Slice 全部不改写、零 Range。

随仓库提交的三个 fixture（`sample_inputs.json`、`expected_outputs.json`、`cpu-golden.npz`）都来自业务侧或由其派生，是**冻结的验收工作负载**，来源、SHA-256 与再生成方式见 [`fixtures/SOURCE.md`](fixtures/SOURCE.md)。

### 1.2 业务场景

候选打分：一次请求给一条文本 query、一段行为历史与一组类目统计，输出 10112 个候选的分数，业务消费其中的 **Top-10**（`dense_id = 输出下标 + 1`）。八个输入特征里七个是 INT64、`stat_ratio` 是 FP32；batch 与文本/行为长度动态，给定样例均为 batch=1（文本长 6/9/8/13/8、行为长 128、统计宽 20）。

业务强相关的部分——五份样例、Top-10 参考、`dense_id` 的 1 基偏移——都由业务侧提供并原样入库，本目录不重新推导，也不做随机生成。

需求、设计、交付报告与全部调优/分析档案见 [BGE 文档索引](../../docs/bge/README.md)；`ge_execute` 的接入细节与逐轮验证记录见 [ge-execute-usage.md](../../docs/bge/ge-execute-usage.md)。

## 2. 环境

### 2.1 Ascend910_9382

| 项目 | 值 |
|---|---|
| SoC | `Ascend910_9382`（Ascend910_93 系列） |
| 物理核（AIC/AIV） | 未单独核实；运行时按每算子上限 `12\|24` 限制（编译期上限，**不预留、不隔离、不等于绑核**） |
| CANN | 9.0.0 |
| Python | 3.11.15（`/usr/local/python3.11.15`） |
| numpy | 1.26.4 |
| Torch / TorchNPU / Transformers | **不需要**——ONNX 直喂 GE，不经 torch |
| onnx / onnxruntime | 运行**不需要**；仅再生成 `cpu-golden.npz` 时用 onnxruntime 1.26.0（`CPUExecutionProvider`） |
| 验证范围 | 已实测 `Ascend910_9382` + CANN 9.0.0；其他 SoC 与 CANN 版本待验证 |

本容器装了**两个 Python 3.11**，SONAME 相同但是两套东西：PATH 上的 `/usr/local/python3.11.15`（3.11.15，有 numpy 1.26.4）与系统 `/usr/bin/python3`（3.11.6，**没有 numpy**），而 `ldconfig -p | grep libpython3.11` 默认解析到后者。GE 编译算子时在原生进程里内嵌一个 CPython（TBE/TE 走 Python，还会起 multiprocessing 工作进程），`set_env.sh` 只设 `PYTHONPATH` 与 CANN 的 lib 目录、**不设 `PYTHONHOME`**，所以必须由调用方把内嵌解释器钉到正在驱动的这个解释器上，否则 GE 初始化报 `No module named numpy`。`run.sh` 已内建这两行 export，换机器时用下面两条自检：

```bash
ldconfig -p | grep libpython3.11
python3 -c "import sys, numpy; print(sys.base_prefix, numpy.__file__)"
```

## 3. 运行指南

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export BGE_MODEL_PATH=/path/to/model.onnx        # 业务原始导出，内联权重，约 1.35 GB
bash models/bge/run.sh [--device-chip N] [--device N]
                       [--iterations N] [--warmup N] [--instances N] [--seed N]
                       [--aicore-num A|B] [--precision-mode M]
```

`run.sh` 自己完成 CANN 环境检查、`PYTHONHOME` / `LD_LIBRARY_PATH` 对齐、主机 BLAS 线程钉（`OPENBLAS/OMP/MKL_NUM_THREADS=2`、`TE_PARALLEL_COMPILER=2`）与 `TMPDIR` 设置，不需要手工准备。

- `--device-chip` 是 npu-smi 的 **Chip Logic ID**（导出为 `ASCEND_RT_VISIBLE_DEVICES`），`--device` 是映射后的进程内设备号，通常为 0。默认 `--device-chip 0`。
- 从任何当前目录启动结果一致。
- `--iterations N` 开启第 7 步性能测试，**前面有人工确认卡空闲的交互**；省略则只跑精度。
- 只有第 5、7 步需要 NPU；第 2 步（生成输入）与第 6 步（比较）纯 CPU，可单独重跑。单独重跑任何一步只执行本步，缺输入直接报错、不补跑前置。
- 固定参数唯一可调处是 `run.sh` 顶部的「固定参数（唯一可调处）」块，本页不复制参数表。

步骤（逐步打印 banner，每步日志落在 `work/logs/`，第 1 步先清掉上一轮的 `*.log`）：

| # | 步骤 | 需要 NPU |
|---|---|---|
| 1 | 环境检查：`BGE_MODEL_PATH`、CANN `set_env.sh`、内嵌 Python 对齐、fixtures 齐备 | 否 |
| 2 | 生成输入（五份业务样例）+ `cases.json` | 否 |
| 3 | 构建并安装融合 Pass | 否 |
| 4 | 构建 C++ runtime | 否 |
| 5 | 精度推理（单实例、无预热） | **是** |
| 6 | 数值与业务比较 | 否 |
| 7 | 可选性能（需人工确认卡空闲） | **是** |

### 3.1 输入与输出

输入按 `cases.json` 的 `inputs` 数组**位置**绑定到图输入，`ge_execute` 不做 io 契约校验。八个输入的顺序、dtype 与静态 shape 由 `model/generate_cases.py` 的 `DTYPES` 与 `validate_arrays` 声明并逐例校验：

| # | 名称 | shape | dtype | 说明 |
|---:|---|---|---|---|
| 0 | `input_ids` | `[-1, -1]` | int64 | 文本 token；样例 JSON 里叫 `query_token_ids`，导入时改名 |
| 1 | `attention_mask` | `[-1, -1]` | int64 | 只接受 0 与 1 |
| 2 | `long_cid3_raw` | `[-1, 128]` | int64 | 行为历史 |
| 3 | `long_action` | `[-1, 128]` | int64 | 行为历史 |
| 4 | `long_time_bucket` | `[-1, 128]` | int64 | 行为历史 |
| 5 | `stat_dense_raw` | `[-1, 20]` | int64 | 类目统计 |
| 6 | `stat_ratio` | `[-1, 20]` | float32 | 唯一的 FP32 输入 |
| 7 | `stat_count` | `[-1]` | int64 | 取值必须在 `[0, 20]` |

所有输入还会被校验有限性；`input_ids`/`attention_mask` 的第二维是文本长（五例为 6/9/8/13/8），`long_*` 的第二维是行为长（五例均为 128）。

| 输出 | shape | dtype | 业务语义 |
|---|---|---|---|
| `scores` | `[-1, 10112]` | float32 | 10112 个候选的分数；`dense_id = 下标 + 1`；列顺序由图自身决定，业务 Top-10 参考在 `fixtures/expected_outputs.json` |

精度模式：外部浮点 IO 保持 FP32，内部 `force_fp16`（`run.sh` 的 `PRECISION_MODE`）。

**输入顺序是承重的。** `ge_execute` 按位置绑定且不校验，而八个输入里 `input_ids`/`attention_mask` 同为 int64 `[-1, seq]`、`long_cid3_raw`/`long_action`/`long_time_bucket` 同为 int64 `[-1, 128]`——共 12 种排列的 dtype/shape 签名完全相同，喂错了 GE 无从判断。顺序不在运行时从图里推导，因为读图 header 要花约 3 GB RSS（权重内联在 protobuf 里，`onnx.load(load_external_data=False)` 跳不过它；实测 0.34–0.41 s、峰值 2647–3948 MB），而这八个名字与 dtype 本来就已经硬编码在 `DTYPES` 里。顺序错在 runtime 层静默、在流水线层响亮：输出会偏离，第 6 步 cosine ≥ 0.9999 的门禁立刻失败。**换图时必须人工核对这八个名字与顺序**，那本来就要重新生成 `cpu-golden.npz`。

### 3.2 Patch

图改写由一个 GE 自定义融合 Pass 完成：`BgeFusedInferAttentionPass`（源码在 `third_party/custom_development_code/fusion_pass/BgeFusedInferAttentionPass`，目录名、CMake project 与注册名三者一致）。

| | 内容 |
|---|---|
| 改写 | 每层的 score `BatchMatMulV2` → mask `Add` → `SoftmaxV2` → value `BatchMatMulV2` → 输出 `Transpose`，以及 Q/K 两侧的 `1/sqrt(D)` 缩放 `Mul`，共 7 个节点，替换为一个内置 `FusedInferAttentionScore`（`input_layout="BSND"`、`sparse_mode=0`、`softmax_lse_flag=false`、`num_heads=num_key_value_heads`） |
| 保留 | Q/K/V 投影、BSND `Reshape`、全部层共享的 mask 生成链 |
| 匹配方式 | 纯结构匹配，不依赖节点名与层数；不命中的链原样保留、不报错 |
| 注册阶段 | `kAfterBuiltinFusionPass` |

BGE 原始导出图 24 层全部命中。细节（判据、控制边策略、属性继承、stage 选择的实测依据）见该 Pass 自己的 README。

**加载契约**：`scripts/prepare.sh` 用路径直接构建并 `cmake --install` 到 `${ASCEND_OPP_PATH}/vendors/bge_custom_pass/custom_fusion_passes/`，由 **GE 自动发现加载**，不经任何依赖声明 schema（因此没有 `revision` 字段，也不受子模块指针影响）。`runtime/main.cpp` 没有任何 `LD_PRELOAD` 处理，所以写入 vendors 是**唯一**的生效路径。

- 安装是**主机级**的，需要对 `$ASCEND_OPP_PATH` 有写权限，且对主机上所有模型可见。
- **目录里的每个 `.so` 都会被加载，不是同名覆盖。** 多个库并存时全部运行，同一次编译出现多轮汇总行。改完 Pass 源码必须重跑 `prepare.sh`；排查异常日志先核对 vendors 里的产物身份。
- 因此「停用某个 Pass」不能只是不再构建它。`prepare.sh` 维护一份 `RETIRED_LIBRARIES`，每次运行都主动删除其中已安装的产物并在末尾断言它们确实不在——本 Pass 的历史库名也在该列表里，否则两个注册名匹配同一结构、谁先改写不确定，产物不再可复现。
- Pass 对不含目标结构的图空转（0 匹配即成功返回、图不变），可与其它模型的 Pass 共存。
- Pass 的 host 单测（ctest）属开发者步骤，`prepare.sh` 不执行。

**部署后必须核对两项，缺一不算生效**：

```
[BGE_FUSED_ATTN] replaced_subgraphs=24
```

以及编译配置确实是 `precision_mode=force_fp16`。Pass 不读任何会话选项、对任何精度路线都照常融合，所以**汇总行不能证明降精度委托成立**——它把降精度委托给 GE（在 `force_fp16` 转换之前跑、按 FP32 描述符接好 FIAS，由 GE 下一阶段收窄），委托不成立时图仍然编译、加载、执行成功，日志照样打印 24，只能靠精度比对发现（实测记录见 Pass README 第 3 节）。

BGE 图上汇总行出现 0、出现不等于 24 的值、或出现多轮，都是**部署故障**（Pass 未加载、图结构未命中、或 vendors 里有陈旧产物），应停止上线排查，不能当作良性空转；只有其它模型的图上 `replaced_subgraphs=0` 才是正常空转。日志里出现任何 `[BGE_PRECOMPUTE]` 或 `[BGE_FIAS]` 行同样是部署故障，说明 vendors 里还留着 `RETIRED_LIBRARIES` 该清掉的库。

汇总行落在 `work/logs/infer.log`（第 5 步）与 `work/logs/bench.log`（第 7 步）——`ge_execute` 的人类可读摘要走 stdout、Pass 的汇总走 stderr，`run.sh` 用 `2>&1 | tee` 合到同一个文件。**轮次等于实例数**：infer 单实例所以 1 轮，bench `--instances 10` 编译 10 份图、10 轮（每轮都应是 24）。判断有没有陈旧 `.so` 要看**同一轮内**是否出现两套前缀，而不是数总行数。命中数不足时日志只给数字不给原因：逐候选诊断不在产物里，要定位是哪一层、卡在哪一步需按 Pass README 第 1 节的说明临时重建一个拒绝原因探测；这一行也区分不了「没匹配到」与「匹配到但被控制边策略跳过」，分辨方法是看有没有 `refused=control_edge` 行。

### 3.3 产物路径

```text
work/
├── inputs/                 # 第 2 步：case-000{0..4}/*.npy（八个特征各一个）+ cases.json（ge-cases/1）
├── infer/                  # 第 5 步：result.json + <case_id>/output_0.npy（float32 [1,10112]，np.load 直接可用）
├── compare/comparison.json # 第 6 步：逐 case 的数值门与业务门明细 + 汇总判定
├── bench/result.json       # 第 7 步（可选）：config 回显、qps、e2e/exec/d2h 分位、逐实例、逐请求 records
└── logs/*.log              # 每步一份；第 1 步先清掉上一轮的
```

`work/` 全部被 `.gitignore` 忽略。图不做任何中间加工：`ge_execute` 经 `aclgrphParseONNX` 直接吃 `$BGE_MODEL_PATH` 指向的原始导出，所以图的身份就是业务交付物的身份。

第 6 步的 `model/compare.py` 直接读第 5 步的产物，没有中间格式。`ge_execute` 只按位置报输出名（`output_0`），`compare.py` 的 `OUTPUT_NAMES` 常量把它映射回 `scores`（与 `verify.py` 硬编码的同一个名字），并断言个数与 `result.json` 一致；每张数组的 shape/dtype 都会与 `result.json` 的声明核对，截断或陈旧的 npy 是响亮失败而不是错的比较结果。`--infer-dir` 是**权威**：按 `<output-dir>/<case_id>/output_<i>.npy` 重建路径、只从 `result.json` 取文件名，不跟随其中记录的绝对路径，否则搬迁或复制过的结果树会让门禁静默读到别处的陈旧 npy 并照常出报告。

### 3.4 回归测试

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
ctest --test-dir runtime/build --output-on-failure
```

`model/generate_cases.py` 与 `model/verify.py` 的函数签名（`generate(config)`/`load(config)`、`compare(config, outputs)`，收普通 dict）服务于本目录的调用方：`model/compare.py` 调 `verify.py` 的 `compare()`，`run.sh` 第 2 步调 `generate_cases.py` 的 CLI。两者入口形态不同：`generate_cases.py` 有 CLI，`verify.py` **只有模块入口、没有 CLI**。

## 4. 精度和性能

| 平台 | case | cosine 最小 | relative L2 最大 | 最大绝对误差 | Top-10 分数最大绝对误差 | 业务 QPS | Avg | P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Ascend910_9382` | case-0000 | 0.9999998886957745 | 0.000523171 | 0.000316441 | 0.000305625 | — | — | — |
| `Ascend910_9382` | case-0001 | 0.9999997515415282 | 0.000716033 | 0.000363559 | 0.000340375 | — | — | — |
| `Ascend910_9382` | case-0002 | 0.9999998323033370 | 0.000642238 | 0.000354290 | 0.000353875 | — | — | — |
| `Ascend910_9382` | case-0003 | 0.9999997845740241 | 0.000662108 | 0.000316322 | 0.000195609 | — | — | — |
| `Ascend910_9382` | case-0004 | 0.9999997904757376 | 0.000698134 | 0.000430822 | 0.000430938 | — | — | — |

五例全通过、0 失败例，Top-10 的 ID、顺序、分数五例全对。性能三列为空：**当前栈上没有可引用的性能数字**，见 [^2]。

精度标准见 [^1]，性能口径见 [^2]，性能优化文档见 [^3]。

[^1]: 两道门，**不可互相替代**。数值门对 `fixtures/cpu-golden.npz` 逐行比 cosine ≥ `0.9999`、相对 L2 ≤ `0.001`、最大绝对误差 ≤ `0.001`，管的是全部 10112 维分数整体像不像；业务门（`model/verify.py` 对 `fixtures/expected_outputs.json`）比 Top-10 的 ID 集合、顺序与分数绝对误差 ≤ `0.001`，管的是业务真正消费的那 10 个名次对不对。golden 是 onnxruntime 1.26.0 CPU FP32 的独立参考（进程隔离，不走 Ascend），再生成方式见 `fixtures/SOURCE.md`。**不能用收紧数值门限来省掉业务门**：五例参考分数里最紧的相邻名次分差是 `5.26e-4`（case-0000 的 rank 8→9），翻转这一对需要单侧误差超过 `2.63e-4`，而五例实测最大绝对误差的最小值就是 `3.16e-4`——门限收紧到能保证排序的程度，case-0000/0001/0004 直接判失败；逐例比值是 1.20× / 1.37× / 0.26× / 0.13× / 1.51×，**3 例已超自己的翻转界**，排序没翻是因为最大误差恰好没落在临界分数对上。FP16 的噪声底高于最紧翻转界，没有可调空间。`comparison.json` 的逐位复现依赖 BLAS 线程钉：cosine 与相对 L2 由 `np.dot`/`np.linalg.norm` 在 float64 上归约，OpenBLAS 归约顺序随线程数变化，同一份输入在 `OPENBLAS_NUM_THREADS` 取 2/1/12 时末位分别给 `…128`/`…161`/`…084`；`run.sh` 钉 2，手工调 `compare.py` 不限线程看到的末位差异不是回归，与线程无关的不变量是 `work/infer/` 下的 npy 字节。

[^2]: **当前栈（`ge_execute` + vendors 自动加载）上没有任何已验收的性能数字。** 第 7 步能产出 `qps` / `e2e_*` / `exec_*` / `d2h_avg`，计时边界是 `e2e = h2d + execute + sync + 全量 D2H`，但它是**单次 seeded 测量**，没有轮间中位数、没有身份门禁与设备独占门禁，不足以支撑任何增益结论。`--warmup M` 是每实例 M 次，且池内 distinct shape 多于 M 时自动补齐到覆盖全部 shape（五例池有 4 种 shape，`--instances 2 --warmup 1` 实际预热 8 次）。历史数字有两组，都**不可**与当前栈并列：其一是受管入口（项目根 `tools/run_one.py` + `models/bge/replay.py`，四轮 60 秒窗口取中位数 + 身份门禁 + 独占门禁）测的实例数扫描，10 实例固定 case 884.177 QPS / p99 14.262 ms、五例随机 903.582 QPS / p99 19.285 ms，绑定二进制 SHA-256 `8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d` 且**不含融合 Pass**，那条入口不在本目录流程内、本目录无法复现；其二是融合 Pass 的收益（固定 +25.723%、随机 +17.849%，尾延迟极差从 76.66%/91.50% 收敛到 2.86%/3.99%），运行栈是 `ge_runtime` SHA-256 `f4b68730…`、计时边界 `h2d_desc_execute_sync_d2h`，绑定 Pass 产物而非本目录。

[^3]: 融合 Pass 的实测记录与口径见 [`BgeFusedInferAttentionPass/README.md`](../../third_party/custom_development_code/fusion_pass/BgeFusedInferAttentionPass/README.md) 第 4 节；实例数扫描的逐轮数据见[十实例报告](../../docs/bge/analysis/100-fp16-ten-selection-20261027.md)、[复测报告](../../docs/bge/analysis/100-fp16-scale-recompare-20261027.md)与[修复报告](../../docs/bge/analysis/100-runner-review-fix-20261028.md)。

**验收口径**（上表的 provenance）：2026-11-20 14:53:33–14:53:55，Chip Logic ID 0 / `0000:9D:00.0`，CANN 9.0.0，`precision_mode=force_fp16`，`aicore_num=12|24`，全程 22 秒。运行前后均 8/8 张卡（16 个 chip 视图）报无进程、全机 0 个进程行、chip 0 HBM 3157 → 3157 MB。产物身份：`ge_execute` `dfe7f38faac66ae59fccd48e8b2144aca4b5963bdaee83f35bade881ceda4bf8`、Pass 构建树 `b3ab21b5aa77ad4cbc7342b1cd41d9d4cd73671640543a6f9ac80fbe6a2b2141`（vendors 副本因 `cmake --install` 剥离 RPATH 与之差 240 字节、大小相同，核对身份认准 `build/` 那份）、图 `2080f876…`。`CompileGraph ok` 8550.4 ms、`LoadGraph ok` 1078.8 ms、8 input buffers；`logs/infer.log` 共 11 行、Pass 占 1 行；`refused=control_edge`、`remove_failed`、`cannot resolve`、`replacement_failed`、`InitIoReuseFlag`、`netoutput_node`、`E13028`、`E19999`、`skipped=` 全 0 命中。逐轮记录见 [ge-execute-usage.md](../../docs/bge/ge-execute-usage.md) 第 9 节。

## 5. 使用限制

- 仅覆盖给定五条 batch=1 样例（文本长 6/9/8/13/8、行为长 128、统计宽 20）。未设置业务 p99 上限，**不声称满足生产 SLA**。
- 只支持 GE Session 推理，图执行模式用 GE 默认的推理模式。**ATC/OM 部署未验证。**
- 验证范围只有 `Ascend910_9382` + CANN 9.0.0；换 SoC 或换 CANN 版本需重跑第 5、6 步门禁。
- 10 实例为用户指定配置，**不声称实例数或随机场景吞吐最优**；12 实例未完成独立全量输出与稳定性验收。
- `fixtures/cpu-golden.npz` 是**派生产物**（onnxruntime CPU FP32 跑五份样例得到）。更换 ONNX 图或更换样例都必须重新生成它，否则数值门禁比的是过时参考；再生成脚本在仓库里（`docs/bge/evidence/20261026/acceptance/build_cpu_golden.py`）但硬编码了 100 机器路径，需先改两个常量，详见 `fixtures/SOURCE.md`。
- 触发重新验收的改动：换图、换样例、换融合 Pass 或其源码、换精度策略、换实例数、换 `aicore_num`。更换输入需重跑第 2、5、6 步；仅更换 `fixtures/expected_outputs.json` 或分数门限只需重跑第 6 步（它直接读当前文件，不依赖原始模型、输入 JSON 或 NPU）。
- **精度推理是单实例、无预热**：省略 `--iterations` 即 infer 模式，每 case 跑一次。`--instances` 只在第 7 步用；实例数 > 1 时每实例都编译一份图（1.35 GB 的 BGE 约 8 s/份，串行）。
- **`ge_execute` 没有超时开关。** 挂死的 run 只能靠外部 `timeout(1)` 或人工终止；从外部杀掉 `run.sh` 时注意 `ge_execute` 可能留下存活的 CANN 编译 worker（它们继承 stdout 且比被杀的父进程活得久）。
- **`cases.json` 的输入顺序是位置绑定，运行时不校验。** 顺序错在 runtime 层不报错，但会让输出偏离、在第 6 步的 cosine 门禁上响亮失败。顺序硬编码在 `generate_cases.py` 的 `DTYPES` 键序里；换图时必须人工核对这八个名字与顺序，并重跑第 2 步。
- **融合 Pass 的自定义路线是唯一的，没有兜底。** 它只装在 `$ASCEND_OPP_PATH/vendors/bge_custom_pass/custom_fusion_passes/`，GE 自动发现；该目录是主机级共享的，同目录其它 `.so` 会一并加载。Pass 不命中时是空转成功而不是失败，所以**只能靠日志核对发现**，不会有任何编译期报错。
- **`TMPDIR` 必须短。** CANN 内嵌 Python 的 `multiprocessing.Manager` 在 `$TMPDIR/pymp-XXXXXXXX/listener-XXXXXXXX` 上 bind AF_UNIX socket，`sun_path` 上限 107 字节。`run.sh` 因此用 `/tmp/bge-$(id -u)` 而不是 `work/tmp`——后者在本仓库是 77 字符，加后缀 110 字节，会让 `GEInitialize` 以 `EC0009 ... multiprocessing.Manager instantiation failed` 失败。
- 可复现性：图不做中间加工，产物身份就是业务交付物的 SHA-256；Pass 的 vendors 副本因 `cmake --install` 剥离 RPATH 与构建树副本差 240 字节（大小相同），核对身份要认准 `build/` 那份。不支持 dump。
