# BGE 接入 ge_execute

面向 BGE 适配：新运行时 `ge_execute` 的用法、要提供的输入、产物、精度门禁接法、环境变量与限制。完整规范见 [runtime/README.md](../../runtime/README.md)；本文只讲 BGE 这条路径。

对应提交：`2bd6f97 feat(runtime): replace the three runtimes with a single ge_execute`（2026-11-18）。

## 1. 现状与影响

`runtime/` 只有 `ge_execute` 一个二进制：吃「**图 + cases.json + npy**」，加载 AIR 或 ONNX，用 GE Session 跑 infer 或 bench。

- **`models/bge` 已迁移**：`run.sh` 第 5、7 步直接调 `ge_execute`，`model/driver.py` 已删除。`models/startlux-*` 的两个 run.sh 还没有可执行后端。
- run.sh 共 7 步（另加收尾）：环境检查 → 生成输入+`cases.json` → 融合 Pass → 构建 runtime → 精度推理 → 数值与业务比较 →（可选）性能。`--graph` 直接指业务原始导出。
- BGE 侧不再需要：`io.json`（**整个文件与 `model/build_graph.py`**）、`inputs.bin` 打包、`output_capacities`、driver 的 pack/decode、`work/build/`（曾存 1.3 GB 重复权重）。输入顺序硬编码在 `generate_cases.py:DTYPES` 的键序（已核对与图一致），输出名硬编码在 `compare.py:OUTPUT_NAMES` 与 `verify.py`——`ge_execute` 只给位置名 `output_0`。
- 已用现有 `model.onnx` 与五例输入验证：**输出与旧 `ge_runtime` 逐比特一致**，对 `fixtures/cpu-golden.npz` 的门禁全过（见 §7）。线程数变化会让 cosine/相对 L2 差几个 ULP（见 `models/bge/README.md` 的精度门禁小节）；对同一份产物重跑，`comparison.json` 的全部数值字段不变。

## 2. 环境

### 2.1 必需

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')"
export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0                     # npu-smi 的 Chip Logic ID
export TMPDIR="/tmp/bge-$(id -u)"; mkdir -p "$TMPDIR"  # 必须短，见下
bash runtime/build.sh                                  # -> runtime/build/ge_execute
```

`TMPDIR` 必须短：GEInitialize 起的 `multiprocessing.Manager` 会在 `$TMPDIR/pymp-XXXXXXXX/listener-XXXXXXXX` 上 bind 一个 AF_UNIX socket，`sun_path` 上限 107 字节，超了会以 `EC0009` 失败。

融合 Pass 仍由 `models/bge/scripts/prepare.sh` 装到 `$ASCEND_OPP_PATH/vendors/bge_custom_pass/custom_fusion_passes/`，GE 自动加载；生效时日志里能看到 `[BGE_FUSED_ATTN] replaced_subgraphs=24`，这是本 Pass 唯一的汇总行。出现任何 `[BGE_PRECOMPUTE]` 行都属部署故障——说明 vendors 里还留着已停用的 `BgeCandidatePrecomputePass` 库，`prepare.sh` 应当已删除它。

### 2.2 为什么要 `PYTHONHOME` + `LD_LIBRARY_PATH`

GE 编译算子时**在原生进程里内嵌一个 CPython**（TBE/TE 走 Python，还会起 multiprocessing 工作进程）。本容器装了**两个 Python 3.11**，SONAME 相同但是两套完全不同的东西：

| | 路径 | 版本 | libpython | numpy |
|---|---|---|---|---|
| PATH 上的 `python3` | `/usr/local/python3.11.15` | 3.11.15 | 23 MB | 1.26.4 ✓ |
| 系统 | `/usr/bin/python3` | 3.11.6 | 5.2 MB | **没有** ✗ |

而动态加载器默认解析到的是系统那套：

```
$ ldconfig -p | grep libpython3.11
        libpython3.11.so.1.0 => /lib64/libpython3.11.so.1.0     # 3.11.6，无 numpy
```

`set_env.sh` 只设 `PYTHONPATH`（CANN 自己的 site-packages）和 CANN 的 lib 目录，**不设 `PYTHONHOME`**，也不把 python 的 libdir 加进 `LD_LIBRARY_PATH`。所以不设这两行时，内嵌解释器会加载 `/lib64` 那套 libpython，并按它编译进去的 prefix 去 `/usr/...` 找 stdlib 与 site-packages——那棵树下没有 numpy，而 TBE 编译算子要用 numpy。

两行 export 的作用就是把内嵌解释器钉到你正在驱动的这个解释器上：`PYTHONHOME` 决定 stdlib/site-packages 去哪找，`LD_LIBRARY_PATH` 决定 `libpython3.11.so.1.0` 优先解析到哪个。用 `sys.base_prefix` 而不是 `sys.prefix`，是为了在 venv 里也拿到底层真实安装。

**换机器时自检**（两者 prefix 不一致、或 ldconfig 指向那套没有 numpy，就必须设）：

```bash
ldconfig -p | grep libpython3.11
python3 -c "import sys, numpy; print(sys.base_prefix, numpy.__file__)"
```

### 2.3 主机线程数（与 `--aicore-num` 无关）

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TE_PARALLEL_COMPILER=2
```

本机 `nproc = 12`，numpy 链 `openblas64 (DYNAMIC_ARCH)`；OpenBLAS/OpenMP/MKL 默认线程数都等于核数，而 TBE 还会 fork 多个编译 worker，每个 worker import numpy 又各带一套线程池 → 12 × N 个主机线程挤 12 核。后果不是算错，是编译变慢、**测量抖动**（计时循环的 h2d/e2e 是普通主机线程，主机 CPU 被抢时 p99 会被拉高）、以及抢其他租户的 CPU。`TE_PARALLEL_COMPILER` 控制 TBE 并行编译进程数，直接影响 `compile_ms`。

⚠️ 这三个限的是**主机 CPU 的 BLAS 线程**，`--aicore-num` 限的是 **NPU 上每算子的 AIC|AIV 核数**，两者完全不同。

| 场景 | 线程数 | TE_PARALLEL_COMPILER |
|---|---|---|
| 编译期（prepare / 首次跑） | 4–8 | 4–8 |
| 测量期（bench） | **2** | 2 |

仓库现状不一致：根 `README.md` 写 8，`models/bge/run.sh` 写 2。测量用 2；正式复测时把当时取值记进结果，否则跨轮次不可比。

## 3. cases.json：BGE 侧唯一要新提供的东西

**`inputs` 数组顺序 = 图输入顺序**（位置绑定，不做名字匹配），BGE 固定这 8 个：

```
input_ids, attention_mask, long_cid3_raw, long_action,
long_time_bucket, stat_dense_raw, stat_ratio, stat_count
```

**已并进 `model/generate_cases.py`**，run.sh 第 2 步产出 `work/inputs/cases.json`：

```bash
python3 model/generate_cases.py --samples fixtures/sample_inputs.json --output-dir work/inputs
```

顺序取自 `DTYPES` 的键序，**就是上面那 8 个名字**（已与图的 `graph.input` 顺序核对一致，去掉 `--io` 前后产出的 `cases.json` 完全相同）。不从 `io.json` 推导顺序：ONNX 是冻结交付，八个名字/dtype/静态 shape 本来就硬编码在 `DTYPES` 与 `validate_arrays` 里、输出名本来就硬编码在 `verify.py` 里，而读图 header 要花约 3 GB RSS（权重内联在 protobuf，`load_external_data=False` 跳不过）。用 3 GB 去防一个政策上已冻结的变更不划算。

顺序错了在 runtime 层是静默的（BGE 八个输入里 `input_ids`/`attention_mask` 同为 int64 `(1,seq)`、`long_cid3_raw`/`long_action`/`long_time_bucket` 同为 int64 `(1,128)`，共 12 种排列的 dtype/shape 签名完全相同，GE 无从判断），但在流水线层是响亮的：输出会偏离，第 6 步 cosine ≥ 0.9999 对 `cpu-golden.npz` 的门禁立刻失败。**换图时人工核对这八个名字与顺序**，那本来就要重新生成 golden（见 `fixtures/SOURCE.md`）。

规则：

- `path` 相对 cases.json 所在目录（绝对路径原样用）；`models/*/work/` 已被 gitignore，放那里不会脏工作区。
- **dtype/shape 由 npy 头决定，JSON 里不用写**，也不会和数据漂移。
- `samples` 可选，默认 1，只影响 `samples_per_second` 统计。`generate_cases.py` 填的是 `input_ids` 的 batch 维。
- `id` 必须唯一，用于日志、infer 输出目录名和 records 里的 `case_id`。
- 各 case 的输入个数必须一致（device 缓冲按位置预分配到池内最大字节）。

## 4. 命令

```bash
# 精度推理（五例）
runtime/build/ge_execute \
  --graph "$BGE_MODEL_PATH" \
  --cases models/bge/work/inputs/cases.json \
  --output-dir models/bge/work/infer --device 0 \
  --precision-mode force_fp16 --aicore-num 12|24

# 性能
runtime/build/ge_execute \
  --graph "$BGE_MODEL_PATH" \
  --cases models/bge/work/inputs/cases.json \
  --output-dir models/bge/work/bench --device 0 \
  --precision-mode force_fp16 --aicore-num 12|24 \
  --instances 10 --warmup 1 --iterations 1000 --seed 42
```

| 参数 | BGE 常用值 | 语义 |
|---|---|---|
| `--iterations N` | 1000 | **所有实例合计**的测量次数；`0`（默认）= infer，每 case 跑一次、单实例、无预热 |
| `--instances K` | 10 | 闭环并发实例数 = 编译 K 份图，**串行约 8s/份**（10 实例 ≈ 80s 启动） |
| `--warmup M` | 1 | **每实例** M 次；池内 distinct shape 多于 M 时自动补齐到覆盖全部 shape（GE 首次执行某 shape 才特化，这笔代价必须留在预热段）。不消耗测量序列、不计入统计 |
| `--seed` | 42 | 有放回抽样种子；同 seed 同 N 下全局序列可复现，K 只改分工 |
| `--aicore-num` | `12\|24` | 每算子 AIC\|AIV 上限（GEInitialize 设 `ge::ir_option::AICORE_NUM`）。编译期上限，**不预留、不隔离、不等于绑核** |
| `--precision-mode` | `force_fp16` | 内部 FP16，外部浮点 IO 保持 FP32 |
| `--graph-run-mode` | `0`（默认） | mode=1 在 CANN 9.0 上 SIGSEGV，别用 |
| `--profiling` | 关 | 产出 `<output-dir>/profiling/PROF_*`，msprof 可解析；带采集开销的结果单独看待 |
| `--format` | 按后缀推断 | `.onnx` → onnx，`.air` → air |

## 5. 产物与计时口径

- **infer**：`<output-dir>/<case_id>/output_0.npy`（BGE 是 float32 `[1,10112]`，`np.load` 直接可用，**不需要 decode 脚本**）+ `result.json`（逐输出 dtype/shape/path/bytes）。
- **bench**：`result.json` —— `qps`、`e2e_avg|min|p50|p99|max`、`exec_avg|p99`、`d2h_avg`、`warmup_runs|warmup_ms`、`compile_ms`、逐实例 `load_ms`、`per_instance`、逐请求 `records`（`h2d_ms`/`exec_ms`/`d2h_ms`/`e2e_ms`/`first_hit`）、`config` 全参数回显。
- **计时口径**：`e2e = h2d + execute + sync + 全量 D2H`，infer 与 bench 一致（`result.json` 的 `timed_phases` 明示）。图编译、LoadGraph、预热不计入；infer 的 `wall_ms` 也不含写 npy。
- 退出码：`0` 成功；`1` 运行期失败（已完成的统计仍写进 `result.json` 并标 `status=failed`）；`2` 用法错误。
- stdout 是人类可读摘要，机器读 `result.json`。

**与旧 `ge_runtime` bench 口径的差别**：旧的正式复测也是「包含 H2D/GE 同步/全量 D2H」，所以量级可比；唯一区别是主机落地缓冲——旧实现每请求重新分配（`resize` 一个新 vector），新实现复用每实例 scratch、稳态零分配。对 40KB 输出影响很小，但严格说新口径少算了每请求的主机分配/释放，跨二进制对比时要写明。

## 6. 精度门禁怎么接

**`compare.py` 直接读 `ge_execute` 的产物**，没有中间格式。run.sh 第 6 步：

```bash
python3 model/compare.py --golden fixtures/cpu-golden.npz \
    --expected fixtures/expected_outputs.json \
    --infer-dir work/infer --output-dir work/compare \
    --cosine-min 0.9999 --relative-l2-max 0.001 --max-abs-error 0.001 --atol 0.001
```

要点：

- 输出名：`ge_execute` 只按位置报 `output_0`，而 golden npz 用图自己的名字 `scores`，映射来自 `compare.py` 的模块常量 `OUTPUT_NAMES = ("scores",)`（与 `verify.py` 硬编码的同一个名字），并断言个数与 `result.json` 一致。
- `--infer-dir` 指向 `ge_execute --output-dir`，读其中的 `result.json` 加 `<case_id>/output_<i>.npy`。
- `result.json` 里的 `path` 在 `--output-dir` 为相对路径时是相对**运行时 cwd** 的，所以先按记录值找、找不到再按 `<infer-dir>/<case_id>/<文件名>` 重建，不依赖 cwd。run.sh 传绝对路径，走第一条。
- 校验 `status == success`、`mode == infer`、输出个数与 `OUTPUT_NAMES` 一致、每张数组的 `shape`/`dtype` 与 `result.json` 声明一致、case ID 不重复——截断或陈旧的 npy 是响亮失败而不是错的比较结果。
- 只搬运字节、不加语义。

`compare.py` 直接读 npy 目录，不产生中间 `outputs.npz`：对同一份 `work/infer/` 重跑，`comparison.json` 的全部数值字段与旧的 npz 路径基线相同，唯一差异是 `infer` 字段从 npz 路径变成目录路径；把 `outputs.npz` 移走后结果不变。

两道门禁为什么都必须保留、以及为什么不能靠收紧数值门限省掉业务门，见 [`models/bge/README.md`](../../models/bge/README.md) 的「精度门禁：为什么是两道而不是一道」——结论是五例里最紧的相邻名次分差 `5.26e-4` 与本 FP16 实现的实测误差 `2.36e-4 ~ 3.38e-4` 只差 1.16 倍，**不存在既能放行当前实现又能保证排序的数值门限**。

## 7. 实测参考（2026-11-18，本机 Chip Logic ID 3，CANN 9.0.0）

| 场景 | 结果 |
|---|---|
| 五例 infer | 与旧 `ge_runtime` 产物**逐比特一致**（5/5 case max abs diff `0.00e+00`）；对 `fixtures/cpu-golden.npz` cosine `1.000000`、rel_l2 ≤ `6.32e-04`、max_abs ≤ `3.38e-04`，满足 `0.9999 / 1e-3 / 1e-3` 门限；e2e avg 60.599ms（p50 6.423 / max 278.831，d2h avg 0.044） |
| bench `--instances 2 --warmup 1 --iterations 20 --aicore-num 12\|24` | 20 请求 0 错误，warmup 自动补齐到 8 次（2 实例 × 4 distinct shape），qps 463.1，e2e p50 4.116 / p99 4.951 / avg 4.273ms（exec avg 4.140 + d2h avg 0.032），compile 16.0s，逐实例 load ≈ 1.1s |

⚠️ 这是**卡被其他作业占用前的短轮**（20 请求），只证明功能与口径正确，**不是性能结论**；正式数字仍按三轮复测流程走（每轮 ≥60 秒 / ≥10000 成功请求，前后设备门禁，见 [复测报告](analysis/100-fp16-scale-recompare-20261027.md)）。

## 8. 限制与坑

- **输入顺序错了不会报错**（没有 io 校验），只会得到错的结果 → `cases.json` 的 `inputs` 顺序必须与图一致。这是去掉 io 契约的代价。
- BGE 的 ONNX 是动态 batch → 输出由 GE 分配，变长池可以混不同 shape（五例池里就有 4 种 shape）。
- 若换成**静态图**，池里只能有一种 shape：多 shape 会在 Init 直接报错（`compiles to a static shape but the case pool has N distinct shapes`）。原因是 GE 对静态图喂错 shape **不报错**，会静默按编译期 shape 解释字节；而 GE 没有查询编译期输入 shape 的 API，所以只能拦到这一层。
- npy 必须 **C-order**；dtype 白名单 13 种（float16/bfloat16/float32/float64、int8/16/32/64、uint8/16/32/64、bool）；`fortran_order=True`、大端、object/字符串/复数一律报错，不转置不兜底。bfloat16 要按 uint16 存并在 JSON 里标 `"dtype": "bfloat16"`。
- **启动时间随实例数线性增长**：每实例一份 AddGraph+CompileGraph 且必须串行（同一 graphId 只能 LoadGraph 一次）。逐份耗时打在 `[INFO] CompileGraph ok (gid=i/K), cost ..., allocation=caller|GE`，汇总在 `result.json` 的 `compile_ms`/`load_ms`。
- `graph_run_mode` / `aicore_num` / `--profiling` 都是 GEInitialize 的全局选项，进程内只生效一次，改了要重跑进程。
- 不支持 OM（ATC 产物）、dump、两阶段 pipeline、请求池时序回放（旧 `replay`）。
- **精度门禁的翻转界**：当前形态下 3/5 例实测误差已超自己的翻转界（`1.20×` / `1.37×` / `1.51×`；双 Pass 形态是 `1.16×` / `1.04×` / `1.19×`）。因此「不存在既能放行当前实现又能保证 Top-10 排序的数值门限」这一结论变强而非变弱。
- **性能未重测**：停用预计算 Pass 后相对原始图的总增益从固定 `+38.527%` / 随机 `+29.418%` 回落到仅 FIAS 的 `+25.723%` / `+17.849%`，两组数字同样出自旧 `ge_runtime` + `LD_PRELOAD` 栈；当前 `ge_execute` + vendors 栈上从未测过任何一组。要引用性能数字必须先重测。

## 9. 验证里程碑

|日期|结论|cosine|相对 L2|最大绝对误差|证据指针|
|---|---|---|---|---|---|
|2026-11-18|端到端 7 步退出 0，全程 32 秒；`CompileGraph ok` 7983.9 ms、`LoadGraph ok` 1117.3 ms，五例最差值与历史记录逐位相同|—|—|—|`ge_execute` SHA-256 `ef3a5e95…`；提交 `2bd6f97`|
|2026-11-18|`compare.py` 改读 `<infer-dir>/result.json` + 逐 case npy，10 条负向路径全部 exit 1；数值字段与基线全同|—|—|—|同上|
|2026-11-19|原始 1.35 GB 内联权重 ONNX 直喂：`CompileGraph ok` 9307.7 ms、`LoadGraph ok` 1078.4 ms，五例与打包图逐比特相同；删除 `build_graph.py`、`io.json`、`work/build/`|—|—|—|原始模型 SHA-256 `2080f876…`|
|2026-11-19|设备侧复验：7 步退出 0，`CompileGraph ok` 9283.6 ms / `LoadGraph ok` 1065.5 ms，五例逐位相同|`0.9999998228985413`|`0.0006315230319931038`|`0.00033801794052124023`|原始 ONNX SHA-256 `2080f876…`|
|2026-11-19|停用 `BgeCandidatePrecomputePass`，`BGE_PRECOMPUTE` 0 命中；`CompileGraph ok` 9013.4 ms / `LoadGraph ok` 1213.5 ms|`0.9999997515415282`|`0.0007160327352242834`|`0.00043082237243652344`|Pass 库 `076dd785…`；上游 `d221027` 删掉 qwen 测试后 Python 测试为 111 个|
|2026-11-19|Pass 重构为单个 `BgeFusedInferAttentionPass`，汇总行一行；`CompileGraph ok` 9057.5 ms / `LoadGraph ok` 1185.1 ms，五例与 Pass README 精度表逐位相同|`0.9999997515415282`|`0.0007160327352242834`|`0.00043082237243652344`|Pass 构建树产物 `b3ab21b5…`；子模块 `b633f55`|

三轮设备侧验收的逐轮 provenance：

- 2026-11-19 10:15:39–10:16:01：Chip Logic ID 0，全程 22 秒；运行前后 NPU 0 均报 `No running processes`、全机 0 个进程行，chip 0 HBM 3156 → 3156 MB；九个异常关键字全 0；Top-10 五例全对。
- 2026-11-19 15:07:22–15:07:47：Chip Logic ID 0，全程 25 秒；前后均 8/8 个 NPU 报 `No running processes`、全机 0 个进程行，chip 0 HBM 3155 → 3154 MB；`logs/infer.log` 13 行（双 Pass 形态 16 行），`BGE_PRECOMPUTE` 0 命中、九个异常关键字全 0；Top-10 五例全对。
- 2026-11-19 16:58:33–16:58:56：Chip Logic ID 0，全程 23 秒；前后均 8/8 个 NPU 报 `No running processes`、全机 0 个进程行，chip 0 HBM 3155 → 3155 MB；`logs/infer.log` 11 行、本 Pass 占 1 行；16 个「不得出现」关键字全 0；Top-10 五例全对。

各项的完整产物路径、哈希与逐条检查清单见 `models/bge/README.md` 的使用限制与验收记录。
