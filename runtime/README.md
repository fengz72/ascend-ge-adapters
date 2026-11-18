# runtime: ge_execute

## 是什么

一个 C++ 二进制 `ge_execute`：加载一张 **AIR 或 ONNX** 图 → GE Session 编译并加载 → 按 `cases.json` 描述的输入跑 **infer 或 bench**。模型侧只交三样东西：图、`cases.json`、npy 输入文件；运行时负责读入、执行、写输出。

实例数、warmup、运行次数、限核、profiling 全部由命令行指定。没有 io 契约校验，没有 stdin plan 协议，没有 ATC/OM 后端，没有两阶段 pipeline，没有请求池回放。

| 文件 | 职责 |
|---|---|
| `main.cpp` | CLI 解析、infer/bench 编排、result.json |
| `session.cpp` | GE + ACL 全部细节（pimpl，GE 头只出现在这里） |
| `npy.cpp` | npy 读与写（v1.0/v2.0 头，C-order，白名单 dtype） |
| `cases.cpp` | `cases.json` → 内存 case 池 |
| `stats.cpp` | 抽样计划、warmup 计划、计时聚合 |
| `../tests/runtime_contracts.cpp` | CPU 契约测试（不链 GE，不需要 NPU） |

## 怎么用

### 构建

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
bash runtime/build.sh                                  # -> runtime/build/ge_execute
ctest --test-dir runtime/build --output-on-failure     # CPU 契约测试
```

CANN 原生进程内嵌 Python，必须与调用方解释器对齐；`TMPDIR` 必须短（GEInitialize 起的 `multiprocessing.Manager` 在 `$TMPDIR/pymp-*/listener-*` 上 bind AF_UNIX socket，`sun_path` 上限 107 字节）：

```bash
export PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')"
export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
export TMPDIR="/tmp/ge-$(id -u)"; mkdir -p "$TMPDIR"
export ASCEND_RT_VISIBLE_DEVICES=<npu-smi 的 Chip Logic ID>
```

### 命令

```bash
# infer: 每个 case 跑一次, 单实例, 输出写 <output-dir>/<case_id>/output_<i>.npy
runtime/build/ge_execute --graph work/build/model.onnx --cases work/inputs/cases.json \
    --output-dir work/infer --device 0

# bench: K 个实例闭环并发, 合计 N 次测量
runtime/build/ge_execute --graph work/build/model.onnx --cases work/inputs/cases.json \
    --output-dir work/bench --device 0 \
    --instances 10 --warmup 1 --iterations 1000 --seed 42 \
    --aicore-num 12|24 --precision-mode force_fp16
```

| 参数 | 默认 | 语义 |
|---|---|---|
| `--graph` | 必填 | AIR 或 ONNX 图路径 |
| `--cases` | 必填 | `cases.json` 路径 |
| `--output-dir` | 必填 | 产物目录，不存在会创建 |
| `--device` | 必填 | 进程内设备号（`ASCEND_RT_VISIBLE_DEVICES` 映射后通常是 0） |
| `--format` | 按后缀推断 | `air` 或 `onnx` |
| `--iterations N` | 0 | **所有实例合计**的测量次数；`N>0` 即 bench，`0` 即 infer |
| `--instances K` | 1 | 闭环并发实例数 = 编译 K 份图（每份一次 AddGraph+CompileGraph，串行） |
| `--warmup M` | 0 | **每实例** M 次预热；池内 distinct shape 多于 M 时自动补齐到覆盖全部 shape。不消耗测量序列、不计入统计 |
| `--seed S` | 42 | 有放回抽样种子；同 seed 同 N 下全局序列可复现，K 只改分工 |
| `--aicore-num A\|B` | 空 | 每算子 AIC\|AIV 核上限，GEInitialize 设 `ge::ir_option::AICORE_NUM`。编译期上限，**不预留、不隔离、不等于绑核** |
| `--precision-mode` | `force_fp16` | GE session 精度模式 |
| `--graph-run-mode` | 0 | 0 = 推理模式（项目默认；mode=1 在 CANN 9.0 上 SIGSEGV） |
| `--profiling` | 关 | 经 GEInitialize 全局选项开启，产出 msprof 可解析的 `PROF_*` 会话 |
| `--aic-metrics` | `PipeUtilization` | profiling 的 AIC 指标 |

### cases.json（schema `ge-cases/1`）

模型侧自己产。`inputs` 数组**顺序即图输入顺序**（位置绑定），`dtype`/`shape` 一律从 npy 头读，不在 JSON 里重复声明：

```json
{"cases": [
  {"id": "case-0000",
   "samples": 16,
   "inputs": [{"name": "input_ids",    "path": "case-0000/input_ids.npy"},
              {"name": "position_ids", "path": "case-0000/position_ids.npy"}]}]}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `cases[].id` | 是 | 非空且唯一；用于日志、infer 输出目录名、records 里的 `case_id` |
| `cases[].samples` | 否 | 业务样本数，默认 1；只喂 `samples` / `samples_per_second` 统计 |
| `cases[].inputs[].path` | 是 | 相对 `cases.json` 所在目录，绝对路径原样用 |
| `cases[].inputs[].name` | 否 | 只用于日志，不做名字匹配 |
| `cases[].inputs[].dtype` | 否 | 仅用于标注 numpy 表示不了的类型：bfloat16 按 uint16 存时写 `"dtype": "bfloat16"` |

各 case 的输入个数必须一致（device 缓冲按位置预分配到池内最大字节）。

### 产物与退出码

- **infer**：`<output-dir>/<case_id>/output_<i>.npy`（模型侧 `np.load` 直接可用，不需要 decode 脚本）+ `result.json`（`cases` 数组逐输出给 dtype/shape/path/bytes）。
- **bench**：`result.json` —— `config` 全参数回显、`pool`、`status`、`requests`/`errors`/`samples`、`wall_ms`/`qps`/`samples_per_second`、`e2e_avg|min|p50|p99|max`、`exec_avg|p99`、`d2h_avg`、`warmup_runs`/`warmup_ms`、`compile_ms`、逐实例 `load_ms`、`per_instance`、逐请求 `records`（含 `h2d_ms`/`exec_ms`/`d2h_ms`/`e2e_ms`/`first_hit`）。
- **profiling**：`<output-dir>/profiling/PROF_*/{host,device_N}`。
- 退出码：`0` 成功；`1` 运行期失败（已完成的统计仍写进 `result.json` 并标 `status=failed`）；`2` 用法错误。
- stdout 是人类可读摘要，机器读 `result.json`。

计时口径：infer 与 bench 同口径，`e2e = h2d + execute + sync + 全量 D2H`（`result.json` 里 `timed_phases` 明示）。bench 每请求同样把输出读回主机，只是丢弃数据（`CollectOutputs(discard=true)`，主机侧复用每实例 scratch 缓冲，稳态零分配，测到的是纯传输而不是分配器噪声）；预热段走完全相同的路径，所以 scratch 增长与 GE 输出缓冲复用都在预热付掉。infer 的 `wall_ms` 是逐 case e2e 之和，不含写 npy。图编译、LoadGraph、预热都不计入测量。

## 使用限制

- **只支持单图**。两阶段（vision → text、中间张量设备常驻）不在范围内。
- **只支持 GE Session**。ATC 编译出的 OM 不支持；dump 不支持。
- **不做 io 契约校验**：输入个数/顺序/dtype/shape 是否与图匹配交给 GE 判断。唯一护栏是"静态编译图 + 池内多于一种 shape"在 Init 直接报错——实测 GE 对静态图喂错 shape **不报错**，会静默按编译期 shape 解释字节。池内 shape 与图自身静态 shape 不一致这种查不出来（GE 没有查询编译期输入 shape 的 API，那需要 io 契约）。
- **npy 只接受 C-order 与 13 个白名单 dtype**（float16/bfloat16/float32/float64、int8/16/32/64、uint8/16/32/64、bool）；`fortran_order=True`、大端、object/字符串/复数一律报错，不转置不兜底。
- **启动时间随实例数线性增长**：每实例一份 AddGraph+CompileGraph 且必须串行（同一 graphId 只能 LoadGraph 一次）。实测 1.35GB BGE ONNX 约 8s/份，tiny 图约 0.5s/份。逐份耗时打在 `[INFO] CompileGraph ok (gid=i/K), cost ...` 与 `result.json` 的 `compile_ms`/`load_ms`。
- `graph_run_mode`、`aicore_num`、profiling 都是 GEInitialize 的全局选项，进程内只生效一次，改了要重跑进程。
- **AIR 分支尚未在本机实测**：仓库内没有 AIR 图，容器也没有 torchair/aclgraph 可合成。已验证的是格式分派与错误路径（`cannot load graph '...' as air`）；`Graph::LoadFromFile` 与旧运行时完全同一调用。首次接 AIR 模型时补一次下面的冒烟。

## 附录：验证记录（2026-11-18，本机 Chip Logic ID 3，CANN 9.0.0）

下表前五行与"计时口径含 D2H"的当前构建一致（e2e = h2d + execute + sync + 全量 D2H）：

| 场景 | 图 | 结果 |
|---|---|---|
| BGE 五例 infer | `models/bge/work/build/model.onnx`（1.35GB，动态，融合 pass 生效） | 与旧 `ge_runtime` 产物**逐比特一致**（5/5 case max abs diff `0.00e+00`）；对 `fixtures/cpu-golden.npz` cosine `1.000000`、rel_l2 ≤ `6.32e-04`、max_abs ≤ `3.38e-04`，满足 `models/bge/run.sh` 的 `0.9999 / 1e-3 / 1e-3` 门限；e2e avg 60.599ms（p50 6.423 / max 278.831，d2h avg 0.044） |
| BGE 短 bench | 同上 | `--instances 2 --warmup 1 --iterations 20 --aicore-num 12\|24`：20 请求 0 错误，warmup 自动补齐到 8 次（2 实例 × 4 distinct shape），qps 463.1，e2e p50 4.116 / p99 4.951 / avg 4.273ms（exec avg 4.140 + d2h avg 0.032），compile 16.0s |
| tiny 静态 ONNX | MatMul+Add+ArgMax，2 输出（float32 + int64） | 走 caller 分配分支（`allocation=caller (static compiled)`）；scores 与 numpy 差 `1.19e-07`（1 ULP，融合累加顺序），indices 精确一致；d2h avg 0.028ms |
| tiny 动态 ONNX | 同图，N 动态，rows=2/3/1 | 走 GE 分配分支；变长输出 `[2,3]/[3,3]/[1,3]` 与 `[2]/[3]/[1]` 正确，数值同上；d2h avg 0.036ms |
| tiny bench | 静态图 | `--instances 2 --warmup 2 --iterations 21 --aicore-num 8\|16`：21 请求 0 错误，warmup 4 次，e2e p50 0.081ms（exec avg 0.033 + d2h avg 0.026） |
| CLI 契约 | —（不需要 NPU） | `--help` exit 0；缺必填/未知 flag/`--instances 0`/`--warmup -1`/`--graph-run-mode 2`/`--format om`/后缀无法推断/`--aicore-num 12` 全部 exit 2；`cases.json` 缺失 exit 1 |
| CPU 契约测试 | —（不需要 NPU） | `ctest` 1/1 通过：npy 读写往返与全部拒绝路径、cases 解析与错误路径、抽样/预热计划、统计与百分位（含 `d2h_avg` / 逐请求 `d2h_ms`）、JSON 落盘 |

待卡空闲补跑（两项都在含 D2H 之前的构建上已通过，且都不在 D2H 改动的代码路径上——profiling 与静态图护栏都在 `Session::Init`）：

| 场景 | 图 | 已有结果（旧构建） |
|---|---|---|
| profiling | 动态图 | `--profiling --iterations 4`：`profiling/PROF_000001_*/{host,device_3}` 共 256K，含 `sample.json` + `.done` |
| 静态图护栏 | 静态图 + 3 种 shape 的池 | `compiles to a static shape but the case pool has 3 distinct shapes`，exit 1 |

2026-11-18 16:52 起本机 chip 3 被其他作业接管（HBM 从 2.9GB 涨到 56409/65536MB），此后两次运行在启动约 8 秒时被外部 SIGTERM（exit 143，TBE 报 `main process disappeared`），不是本运行时的缺陷；设备验证随即停止，上表两行留待补跑。

复现用的图、cases.json 与产物在 `work/ge-execute-smoke/`（Git 忽略）。
