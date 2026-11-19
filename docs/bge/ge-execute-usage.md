# BGE 接入 ge_execute

面向 BGE 适配：新运行时 `ge_execute` 的用法、要提供的输入、产物、精度门禁接法、环境变量与限制。完整规范与全部验证记录见 [runtime/README.md](../../runtime/README.md)；本文只讲 BGE 这条路径。

对应提交：`2bd6f97 feat(runtime): replace the three runtimes with a single ge_execute`（2026-11-18）。

## 1. 现状与影响

`runtime/` 现在只有一个二进制 `ge_execute`：吃「**图 + cases.json + npy**」，加载 AIR 或 ONNX，用 GE Session 跑 infer 或 bench。

- 旧的 `ge_runtime`（stdin plan 协议）、`executor`、`pipeline_runtime` 已删除。
- **`models/bge` 已完成迁移**（见第 9 节）：`run.sh` 第 6/8 步直接调 `ge_execute`，`model/driver.py` 已删除。`models/startlux-*` 仍待迁移。
- run.sh **少了「打包 ONNX 图」这一步**，现在是 8 步：环境检查 → 生成输入+`cases.json` → 融合 Pass → 构建 runtime → 精度推理 → 数值与业务比较 →（可选）性能 → 完成。`--graph` 直接指业务原始导出。（阶段 1 曾把打包图提到第 2 步以便 `cases.json` 从 `io.json` 取序；阶段 3 删掉打包图后该重排也随之撤销。）
- BGE 侧不再需要：`io.json`（**整个文件与 `model/build_graph.py` 都已删除**）、`inputs.bin` 打包、`output_capacities`、`driver.py` 的 pack/decode、`work/build/`（曾存 1.3 GB 重复权重）。输入顺序硬编码在 `generate_cases.py:DTYPES` 的键序（已核对与图一致），输出名硬编码在 `compare.py:OUTPUT_NAMES` 与 `verify.py`——`ge_execute` 只给位置名 `output_0`。
- 已用现有 `model.onnx` 与五例输入验证：**输出与旧 `ge_runtime` 逐比特一致**，对 `fixtures/cpu-golden.npz` 的门禁全过（见第 7 节）。迁移后另行离线复核：五例最差值在 `OPENBLAS_NUM_THREADS=2` 下与历史记录逐位相同（线程数变化会让 cosine/相对 L2 差几个 ULP，见 `models/bge/README.md` 的精度门禁小节）；阶段 2 让 `compare.py` 直接读 npy 目录后，对同一份产物重跑，`comparison.json` 的全部数值字段仍与阶段 1 基线完全相同。

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

GE 编译算子时**在原生进程里内嵌一个 CPython**（TBE/TE 走 Python，还会起 multiprocessing 工作进程），运行日志里能直接看到它：

```
File "/usr/local/Ascend/cann-9.0.0/python/site-packages/tbe/common/repository_manager/route.py"
File "/usr/local/python3.11.15/lib/python3.11/multiprocessing/managers.py"
```

本容器装了**两个 Python 3.11**，SONAME 相同但是两套完全不同的东西：

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

顺序取自 `DTYPES` 的键序，**就是上面那 8 个名字**（已与图的 `graph.input` 顺序核对一致，去掉 `--io` 前后产出的 `cases.json` 完全相同）。

阶段 1 曾让顺序从 `io.json` 推导，阶段 3 撤销了：ONNX 是冻结交付，八个名字/dtype/静态 shape 本来就硬编码在 `DTYPES` 与 `validate_arrays` 里、输出名本来就硬编码在 `verify.py` 里，而读图 header 要花约 3 GB RSS（权重内联在 protobuf，`load_external_data=False` 跳不过）。用 3 GB 去防一个政策上已冻结的变更不划算。

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

迁移期曾有一个 `model/pack_outputs.py` 垫片把 npy 打包回 `outputs.npz`（`core/worker.py:save_rows` 布局）以复用旧的 `compare.py`，**已在阶段 2 删除**。删除前后对同一份 `work/infer/` 做过对比：`comparison.json` 的全部数值字段完全相同，唯一差异是 `infer` 字段从 npz 路径变成目录路径；把 `outputs.npz` 移走后重跑结果不变。

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

## 9. 待办

已完成（2026-11-18，阶段 1）：

- [x] `models/bge/run.sh` 第 6/8 步改为直接调 `ge_execute`；`model/driver.py` 已删除
- [x] `model/generate_cases.py` 顺带产出 `cases.json`（第 3 节）。顺序一度取自 `io.json`，**阶段 3 改回 `DTYPES` 键序**
- [x] 第 6 节的打包片段实现为 `model/pack_outputs.py`，run.sh 第 7 步调用（**阶段 2 已删除**，见下）
- [x] 打包图提到第 2 步、生成输入降到第 3 步（`cases.json` 依赖 `io.json`）——**阶段 3 删除打包图后该重排已撤销**，回到生成输入为第 2 步
- [x] 离线验证：`ge_execute` 五例输出与旧 `ge_runtime` 逐比特相同；垫片的 npz 与旧 `driver.py` 逐键逐比特相同；`compare.py` 在 `OPENBLAS_NUM_THREADS=2` 下复现历史五例最差值到最后一位；`tests/` 的 13 个 BGE 用例全过
- [x] **端到端重跑验收**（2026-11-18 19:43:28–19:44:00，Chip Logic ID 0，独占门禁满足：运行前后全机 0 个进程行、8/8 NPU 报 `No running processes`、chip 0 HBM 3157 → 3149 MB）：`run.sh` 第 1–7 步退出 0，全程 32 秒；`CompileGraph ok` 7983.9 ms、`LoadGraph ok` 1117.3 ms、8 input buffers；两个 Pass 汇总行各 1 轮、九个异常关键字全 0；打包图 SHA-256 仍为 `5e641362…`；五例全过、最差值与历史记录**逐位相同**。`ge_execute` SHA-256 `ef3a5e95b6e3d302208f6e1ec38c4446b21bd025b8393b0eb918509bb6a66c81`。详见 `models/bge/README.md` 使用限制第一条。

已完成（2026-11-18，阶段 2）：

- [x] `model/compare.py` 改为直接读 `<infer-dir>/result.json` + `<case_id>/output_<i>.npy`；`model/pack_outputs.py` 与垫片调用一并删除，不再有中间的 `outputs.npz`。（`--io` 一度必填，**阶段 3 改为 `OUTPUT_NAMES` 常量**）
- [x] 离线复验：对阶段 1 那轮留下的同一份 `work/infer/`，`comparison.json` 的**全部数值字段**与阶段 1 基线完全相同，唯一差异是 `infer` 字段从 npz 路径变成目录路径；移走 `outputs.npz` 后重跑结果不变
- [x] **设备侧复验**（2026-11-18 20:39:54–20:40:25，Chip Logic ID 0，独占门禁满足：前后全机 0 个进程行、8/8 NPU 空闲、chip 0 HBM 3143 → 3149 MB）：`run.sh` 第 1–7 步退出 0，五例最差值仍与历史记录**逐位相同**，Pass 汇总行各 1 轮、九个异常关键字全 0，`CompileGraph ok` 8164.9 ms / `LoadGraph ok` 1081.8 ms
- [x] 负向路径逐条实测：缺 `result.json`、`status != success`、`mode == bench`、输出个数与 `io.json` 不符、`shape`/`dtype` 声明不符、case ID 重复、npy 缺失、`io.json` 缺失、`cases` 为空——10 条全部 exit 1。**其中「npy 缺失」最初误报成功**，暴露出 `result.json` 记绝对路径会让 `--infer-dir X` 静默读到 X 以外的产物；已改为 `--infer-dir` 权威（按 `<output-dir>/<case_id>/output_<i>.npy` 重建，只从记录取文件名），并加权威性检验（副本 npy 改全 0 → 读到副本、cosine 判 0.0）
- [x] 删除 `model/verify.py` 的死 CLI（`main`/`load_rows`/`_row`/`_unwrap_scores` 共 69 行，168 → 93 行）。全仓库对该文件只有两处 `script:` 配置，都经 `core/verify.py:11` 的 `spec_from_file_location` 只调 `compare()`；`compare.py` 也是 `import verify as business` 后调 `compare()`。没有任何地方执行过它的 CLI，而它读的 npz 在阶段 2 后也不再产生
- [x] `model/` 从 5 个文件 1109 行降到 4 个文件 771 行（净 −338）：`driver.py` −394（删除）、`verify.py` −75（死 CLI）、`generate_cases.py` +53、`compare.py` +78（`_load_infer` 吸收原垫片的读取与校验）；`pack_outputs.py` 在本轮内建了又删，净 0

待办：

- [ ] 卡空闲后补跑两项：`--profiling` 产出 `PROF_*`、静态图 + 多 shape 池的护栏（两项都在 `Session::Init`，D2H 改动没碰；在含 D2H 之前的构建上已通过）
- [ ] 下次动到第 2–5 步（生成输入、Pass、runtime、推理）时要重新端到端验收一次；只动第 6 步的比较逻辑可以沿用离线复验（对同一份 `work/infer/` 比对 `comparison.json` 的全部数值字段）
- [ ] `models/startlux-*` 迁移到 `ge_execute`（两个 run.sh 目前同样没有可执行后端）。迁移时可一并考虑把数值门收敛到 `tools/compare.py`——但它现在读的是 `result.json` + 单个 `outputs.bin`（offset 索引），要先学会读逐 case npy 目录（`models/bge/model/compare.py:_load_infer` 已是一个可参考的实现），并补 `--max-abs-error`；业务门按 `standalone-runtime-plan.md` 的既定原则**不并入**公共工具。

已完成（2026-11-19，阶段 3：删除打包图）：

- [x] **实测 `ge_execute` 直接吃原始 1.35 GB 内联权重 ONNX**：`CompileGraph ok` 9307.7 ms、`LoadGraph ok` 1078.4 ms、8 input buffers，五例输出与打包图那轮**逐比特相同**（5/5 `max|diff| = 0.000e+00`），Pass 汇总行各 1 轮、8 个异常关键字全 0。原始模型 SHA-256 `2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0`，与 `fixtures/SOURCE.md` 记录一致
- [x] **删除 `model/build_graph.py`（216 行）与 `io.json`、`work/build/`（1.3 GB 重复权重）**。上一轮标为待议的 `_fast_path` + `_external_locations`（37 行从未执行）随之消失
- [x] `--graph` 直接指 `$BGE_MODEL_PATH`；run.sh 去掉打包图步、编号回到 8 步（阶段 1 的重排随之撤销）
- [x] `generate_cases.py` 去掉 `--io` 与 `graph_input_order()`，`cases.json` 顺序用 `DTYPES` 键序——产出与从 `io.json` 取序那版**完全相同**
- [x] `compare.py` 去掉 `--io`，改用模块常量 `OUTPUT_NAMES = ("scores",)` 按位置映射输出名；`comparison.json` 顶层少一个 `io` 字段，全部数值字段与基线相同
- [x] 负向路径重测 9 条（`io.json` 那条已不适用，换成 `OUTPUT_NAMES` 个数不符）全部 exit 1，正向 exit 0；全量单测只剩既有的 startlux `causal_conv1d` 缺失（当时 146 个，上游 `d221027` 删掉 qwen 测试后为 111 个）
- [x] `model/` 从 4 个文件 771 行降到 **3 个文件 537 行**（累计从 5 个文件 1109 行降下来，净 −572）
- [x] 文档同步：`models/bge/README.md`（步骤表、顺序取舍与 3 GB RSS 实测、验收记录）、`fixtures/SOURCE.md`（不再重打包、换图时的人工接口核对）
- [x] **设备侧端到端重验收**（2026-11-19 10:15:39–10:16:01，Chip Logic ID 0，全程 22 秒，独占门禁完全满足：运行前后 NPU 0 均报 `No running processes`、全机 0 个进程行、chip 0 HBM 3156 → 3156 MB 完全一致）：`run.sh` 8 步退出 0；五例最差值 cosine `0.9999998228985413`、相对 L2 `0.0006315230319931038`、最大绝对误差 `0.00033801794052124023`，与历史记录**逐位相同**；Pass 汇总行各 1 轮、九个异常关键字全 0；`CompileGraph ok` 9283.6 ms / `LoadGraph ok` 1065.5 ms；原始 ONNX SHA-256 核对为 `2080f876…`；`work/build` 确认已不存在；`work/logs/` 里已无 `build.log` 与 `pack-outputs.log`（第 1 步新增的陈旧日志清理生效）

已完成（2026-11-19，阶段 4：停用 BgeCandidatePrecomputePass）：

- [x] 源码归档到子模块分支 **`archive/bge-candidate-precompute-pass`**（`fork` 远端，指向 `0430d1ec`，含全部 7 个文件，`git ls-remote` 已核实）。建分支时未切换分支、未碰工作区，`bge-fias-precompute-pass` 上那 5 项未提交的 FIAS 改动原样保留
- [x] `scripts/prepare.sh` 不再构建/安装它，并新增 `RETIRED_LIBRARIES` 机制：**主动删除**已安装的停用产物 + 末尾断言其确实不在。因为 GE 会加载 vendors 目录下每一个 `.so`，「不再构建」不等于「不再执行」
- [x] 已物理删除 `$ASCEND_OPP_PATH/vendors/bge_custom_pass/custom_fusion_passes/libbge_candidate_precompute.so`（5,220,648 字节），目录下只剩 `libbge_fias_structure_pass.so`（`076dd78529a90930…`）
- [x] `run.sh` 产物校验只留 FIAS 库；`models/bge/README.md` 新增「已停用：BgeCandidatePrecomputePass」小节（5 条理由 + 精度对照 + 代价），日志门禁把**任何 `[BGE_PRECOMPUTE]` 行**列为部署故障
- [x] **设备侧端到端重验收**（2026-11-19 15:07:22–15:07:47，Chip Logic ID 0，全程 25 秒，独占门禁完全满足：运行前后均 8/8 个 NPU 报 `No running processes`、全机 0 个进程行、chip 0 HBM 3155 → 3154 MB）：`run.sh` 8 步退出 0；五例全过、0 失败，最差值 cosine `0.9999997515415282`（与 `BgeFusedInferAttentionPass/README.md:150` 的仅-FIAS 记录**逐位相同**）、相对 L2 `0.0007160327352242834`、最大绝对误差 `0.00043082237243652344`；Top-10 五例全对；`logs/infer.log` **13 行**（双 Pass 形态 16 行，少的 3 行正是预计算汇总），FIAS 三条各 1 轮、`BGE_PRECOMPUTE` **0 命中**、九个异常关键字全 0；`CompileGraph ok` 9013.4 ms / `LoadGraph ok` 1213.5 ms
- [x] 精度门禁的翻转界表按当前形态刷新：3/5 例实测误差已超自己的翻转界（1.20× / 1.37× / 1.51×，双 Pass 形态是 1.16× / 1.04× / 1.19×），「不存在既能放行当前实现又能保证 Top-10 排序的数值门限」这一结论**变强而非变弱**

待办（阶段 4 新增）：

- [ ] **性能未重测。** 停用预计算 Pass 后相对原始图的总增益从固定 +38.527% / 随机 +29.418% 回落到仅 FIAS 的 +25.723% / +17.849%，而这两组数字同样出自旧 `ge_runtime` + `LD_PRELOAD` 栈，**当前 `ge_execute` + vendors 栈上从未测过任何一组**。要引用性能数字必须先重测
- [ ] 子模块三件事（都不是本次改动引入的）：主仓库钉的指针 `8e08d0d2` 里**两个 BGE Pass 都不存在**，新克隆 + `submodule update` 会让 `prepare.sh` 直接 die；已验收的 FIAS 产物 `076dd785…` 是从**未提交**源码构建的（那 5 项单库合并改动），仅凭 git 无法复现；`.gitmodules` 写 `branch = main` 但工作在 `bge-fias-precompute-pass`
- [ ] 预计算 Pass 的源码是否要从工作分支 `bge-fias-precompute-pass` 上删除（归档分支已保住它）。破坏性操作，未执行

已完成（2026-11-19，阶段 5：Pass 重构 + 子模块历史压缩）：

- [x] 子模块 `third_party/custom_development_code` 的 15 个提交压缩为**单个** commit `b633f55`（`feat(fusion_pass): add the BGE fused-infer-attention rewrite pass`），`--force-with-lease` 推送到 `fork/bge-fias-precompute-pass`（`0430d1e...b633f55`，lease 基准经 `ls-remote` 核对）。压缩前建了本地保险分支 `backup/pre-squash-2f53c8f`
- [x] `BgeCandidatePrecomputePass` 从工作分支删除（7 个文件 + `fusion_pass/README.md` 索引条目 22 共 6 行）。归档分支 `archive/bge-candidate-precompute-pass`（`0430d1e`，7 个文件）删除前后都核实过完好
- [x] 主仓库子模块指针 `8e08d0d` → `b633f55`。**这同时修掉了一个既有缺口**：旧指针 `8e08d0d2` 里两个 BGE Pass 都不存在，新克隆 + `submodule update` 会让 `prepare.sh` 在 `[[ -d "$ATTN_PASS_DIR" ]]` 直接 die
- [x] Pass 重构后的接口变化已在 `models/bge` 侧适配（目录 `BgeFusedInferAttentionPass`、库 `libbge_fused_infer_attention_pass.so`、注册名 `BgeFusedInferAttentionPass`、日志前缀 `[BGE_FUSED_ATTN]`、汇总行从三行变**一行** `replaced_subgraphs=N`、不再读精度路线）。`prepare.sh` 的 `RETIRED_LIBRARIES` 同时列出 `libbge_candidate_precompute.so` 与旧库名 `libbge_fias_structure_pass.so`——改名后旧库若留在 vendors 会与新库**同时生效**，两个注册名匹配同一结构，谁先改写不确定
- [x] **设备侧端到端重验收**（2026-11-19 16:58:33–16:58:56，Chip Logic ID 0，全程 23 秒，独占门禁完全满足：前后均 8/8 NPU 报 `No running processes`、全机 0 个进程行、chip 0 HBM 3155 → 3155 MB 完全一致）：`run.sh` 8 步退出 0；五例最差值 cosine `0.9999997515415282`、相对 L2 `0.0007160327352242834`、最大绝对误差 `0.00043082237243652344`，与 Pass README 精度表**逐位相同**；Top-10 五例全对；`logs/infer.log` **11 行**、本 Pass 占 1 行；16 个「不得出现」关键字全 0（含 `BGE_FIAS`/`BGE_PRECOMPUTE` 两个旧前缀与 `matched_subgraphs`/`precision_mode=`/`skipped=` 三个重构前的日志键）；`CompileGraph ok` 9057.5 ms / `LoadGraph ok` 1185.1 ms。Pass 构建树产物 `b3ab21b5aa77ad4cbc7342b1cd41d9d4cd73671640543a6f9ac80fbe6a2b2141` 与 Pass README 记录一致
