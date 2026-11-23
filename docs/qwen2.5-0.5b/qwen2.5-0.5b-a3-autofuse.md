# Qwen2.5-0.5B Prefix：A3 自动融合（AUTOFUSE_FLAGS）评估

> 本文保留 2026-11-23 的 A3 autofuse 评估结果。后文配置和结论属于该轮实验；当前配置和最近报告入口见[文档索引](README.md)。

采集日期：2026-11-23。Ascend910_9382（a3）、CANN 9.0.0（V100R001C10SPC001B250，aarch64-linux）、逻辑设备 2、profiling 关闭。评估对象是 `AUTOFUSE_FLAGS="--enable_autofuse=true;--autofuse_enable_pass=reduce,concat"`。

## 实测结论

**A3 不引入 AUTOFUSE_FLAGS：默认交付路径开不起来，绕开之后也没有任何融合发生，收益为零。**

- 生产路径（`ge::Session::CompileGraph(gid)`，不带输入）一开 AF 就编译失败：autofuse 的符号化前置只拿到 GE 的哑元 shape `DUMMY_SHAPE={-3}`，报 `Symbolize graph input failed`（error 1343225857，Parameter invalid）。
- 换用带输入的编译入口 `BuildGraph(gid, inputs)` 后 AF 编译通过（`autofuser.cpp` 的 Lowering 28.2ms / Lifting 0.9ms 正常执行），但**执行图与不开 AF 完全一致**：`ExecuteGraphAfterSplit` 两侧都是 31169 个算子、321 个 `LaunchKernelWithHandle`、58 类算子直方图逐项相同；把节点名里的 PID 归一化后仅剩 1446 行差异，全部是 `bt:` 二进制属性里的运行期地址，结构差异 0 行。
- 精度与性能因此都不变：100 case / 1600 行输出与基线**逐位相同**（max|diff| = 0），worst cosine `0.9994394615744572`、worst relative L2 `0.033784056043048426`，过门禁（0.998 / 0.05）；单流业务 QPS 中位数 43.519（不开 AF）vs 43.085（AF），未受同租户干扰的轮次两侧区间重叠（43.373–43.633 vs 43.422–43.637），差 <0.2%，在噪声内。
- 代价是编译变慢：单图中位 5.3s → 7.5s（+40%），且 AF 侧编译时间在 7.3–14.2s 之间双峰波动（noAF 稳定在 5.1–5.4s）。
- 白名单从 `reduce,concat` 扩到 7 个 pass 全开（`reduce,concat,matmul,slice,split,gather,transpose`）结论不变：输出仍与基线逐位相同，QPS 43.468 落在基线区间内。

没有命中的原因与既有优化重叠：169 个 MatMul 已由自定义 `MatMulWeightNZPass` 统一重建为 `MatMulV3`（权重 FRACTAL_NZ），归一化走融合 `RmsNorm`，注意力走自定义 `PrefixInferAttentionScore`，autofuse 的 reduce/concat 类 lowering 在这张图上找不到可融合模式。

评估期间为绕开符号化而给 `runtime/` 加过的 `--compile-case` 开关（默认关闭，走 `BuildGraph`）已回滚，patch 存档在 `models/qwen2.5-0.5b/work/af/compile-case.patch`；交付路径仍是 `CompileGraph`。

## 环境与测量口径

| 项目 | 实际值 |
|---|---|
| 芯片 / 设备 | Ascend910_9382（a3），`ASCEND_RT_VISIBLE_DEVICES=2`、`--device 0` |
| 图 | `work/build/packed.air`（torchair `dynamo_export(dynamic=True)`，FP16 冻结权重，三个动态输入 origin shape 均为 `[-1]`） |
| 输入池 | `work/inputs/cases.json`：100 case、batch 固定 16、89 种 distinct shape |
| 精度模式 | `must_keep_origin_dtype`（与历史精度基线同口径；`ge_execute` 默认是 `force_fp16`） |
| 精度门禁 | 逐行 cosine ≥ 0.998、relative L2 ≤ 0.05，`tools/compare.py` 对 HF FP32 CPU golden |
| 性能口径 | `--iterations 3000 --warmup 10 --instances 1 --aicore-num 8\|16`，profiling 关闭；e2e 为整 batch 的 H2D + 描述符处理 + 执行 + 同步 + D2H，不除以 batch；编译与加载不计入 |
| 业务 QPS | 业务样本数 / 测量段墙钟秒数 = 模型调用 QPS × 16 |
| A/B 方式 | 同 seed 内交替跑各配置（dyn-noaf → bc-noaf → bc-af），抵消同租户偶发活动造成的漂移 |
| 并发降级原因 | 测量期间同租户在 16 张卡上各占 ~59GB HBM（每卡仅剩 ~3GB 空闲），`--instances 5` 加载即 `rtMalloc fail, purpose: page caching, size:992870400`；本轮全部为单流口径，**不能与模型 README 里 5 实例的业务 QPS 直接比较** |

配置名含义：`dyn-noaf` = 生产路径（`CompileGraph`，不开 AF）；`bc-noaf` = `BuildGraph` 带输入编译、不开 AF；`bc-af` = `BuildGraph` + AF `reduce,concat`；`bc-af-all` = `BuildGraph` + AF 7 个 pass 全开。

## 1. 默认路径为什么开不起来

评估时抓到的 GE plog 摘录（原始日志目录已清理，复现脚本见附录）：

```text
[symbolic_shape_symbolizer.cc:65]  BuildDataSymbolizeInfo: Node[arg2_1] is not supported symbolize, input origin shape is [-3].
[symbolic_shape_symbolizer.cc:327] SymbolizeRootGraph: Assert ((ret) == ge::SUCCESS) failed
[symbolic_shape_symbolizer.cc:399] Symbolize: Assert ((SymbolizeRootGraph(graph, graph_inputs)) == ge::SUCCESS) failed
[autofuse_optimize.cc:205]         Run: Symbolize graph input failed, graph graph_1
[inner_session.cc:756]             CompileGraph: failed, graph_id:1, inputs size:0
[ge_api.cc:1042]                   CompileGraph: Compile graph failed, error code:1343225857
```

`[-3]` 不是 AIR 里的维值，而是 GE 的 `DUMMY_SHAPE`（`include/graph/types.h:27-32`：`UNKNOWN_DIM=-1`、`UNKNOWN_DIM_NUM=-2`、`DUMMY_SHAPE={-3}`）。编译期没有输入张量（`inputs size:0`）时 Data 节点拿到的就是这个哑元，而 autofuse 的符号化前置拒绝哑元、整个 autofuse 中止并让编译失败。不开 AF 时同一条路径不会走符号化，所以生产路径一直是正常的。

绕开方式是给编译期喂真实 shape。v1 下只能换另一个接口 `ge::Session::BuildGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs)`（`ge_api.h:165`），实测有两处与 `CompileGraph` 不同，必须一起处理：

1. `BuildGraph` 之后 `GetCompiledGraphSummary` 返回 null（`ge_api.cc:1058 graph needs to be compiled first`），只能按动态图走，输出由 GE 分配；
2. `BuildGraph` 内部已完成 load，再调 `LoadGraph` 会报 `Graph has been loaded`（`graph_manager.cc:1648`），该路径要跳过 `LoadGraph`。

这两个副作用都是 v1 `BuildGraph` 特有的；runtime 切到 GE API v2 后，带输入编译是同一个 `CompileGraph` 的两参重载，summary 与 `LoadGraph` 都正常，见第 6 节。

带输入编译本身是中性的：`bc-noaf` 的 QPS 中位数 43.424、100 case infer e2e 1216.5ms，与 `dyn-noaf` 的 43.519 / 1227.8ms 同级，输出逐位相同。

## 2. 精度

`tools/compare.py --cosine-min 0.998 --relative-l2-max 0.05`，100 case / 1600 行全过：

| 配置 | 结果目录 | worst cosine | worst relative L2 | 与 dyn-noaf 逐位比较 |
|---|---|---|---|---|
| dyn-noaf | `work/output-b-dyn` | 0.9994394615744572 | 0.033784056043048426 | — |
| bc-noaf | `work/output-b-noaf` | 0.9994394615744572 | 0.033784056043048426 | 100/100 case max\|diff\| = 0 |
| bc-af | `work/output-b-af` | 0.9994394615744572 | 0.033784056043048426 | 100/100 case max\|diff\| = 0 |
| bc-af-all | `work/output-b-af-all` | 0.9994394615744572 | 0.033784056043048426 | 100/100 case max\|diff\| = 0 |

worst 值与本轮之前的历史基线（`work/output/comparison.json`）也完全一致，即 AF 没有引入任何数值变化。

## 3. 执行图对比：融合有没有发生

`DUMP_GE_GRAPH=1 DUMP_GRAPH_LEVEL=2 DUMP_GRAPH_PATH=<dir>` 各做一次单 case（case-0037，T=2053）编译，对比最后两个阶段的 `ge_proto_*.txt`：

| 阶段 | 算子总数 noAF | 算子总数 AF | 算子类型直方图 | `LaunchKernelWithHandle` |
|---|---:|---:|---|---:|
| `ExeGraphBeforeOptimize`（00000014） | 24274 | 24274 | 58 类逐项相同 | — |
| `ExecuteGraphAfterSplit`（00000015） | 31169 | 31169 | 58 类逐项相同 | 321 / 321 |

两侧 md5 不同，差异全部来自运行期标识：节点名里的 PID（`SplitConstTensor_dynamic_const_<pid>_168_94`）与 `bt:` 属性里的地址。归一化 PID 后 stage14 剩 1446 行差异，逐行分类 100% 是 `bt:`，非 `bt:` 差异 0 行。`LaunchKernelWithHandle` 321 就是每次前向的设备 kernel 数，AF 前后一致，说明没有算子被合并。

AF 侧另有一处旁证：dump 的 `RunCustomPassAfterBuiltinFusionPass`（00000006）与 `PreRunAfterOptimizeOriginalGraph`（00000007，autofuse 在该阶段内跑）两个文件字节数完全相同（2946177546）。

原始 dump 每侧 ~72GB，巨型阶段文件已清理，只留下 stage 12/14/15（`work/af/dump-{af,noaf}-keep/`，共 ~900MB）；算子直方图在 `work/af/hist-{af,noaf}-stage1{4,5}.txt`，复现脚本见附录。

## 4. 性能

单流口径（`--instances 1 --iterations 3000 --warmup 10 --aicore-num 8|16`），每轮 3000 次模型调用 / 48000 个业务样本，零执行错误：

| 配置 | 轮数 | 业务 QPS 中位 | QPS 范围 | e2e p50 中位 (ms) | e2e p99 中位 (ms) | 单图编译 (s) |
|---|---:|---:|---|---:|---:|---:|
| dyn-noaf | 6 | 43.519 | 43.373–43.633 | 22.114 | 26.596 | 5.3 |
| bc-noaf | 3 | 43.424 | 39.564–43.638 | 22.191 | 26.641 | 6.6 |
| bc-af | 6 | 43.085 | 41.624–43.637 | 22.157 | 29.100 | 7.5（7.3–14.2） |
| bc-af-all | 1 | 43.468 | — | 22.117 | 26.588 | 7.4 |

逐 seed 明细（业务 QPS / e2e p50 / e2e p99 / 单图编译秒）：

| seed | dyn-noaf | bc-noaf | bc-af | bc-af-all |
|---|---|---|---|---|
| 42 | 43.456 / 22.115 / 26.625 / 5.2 | 43.424 / 22.191 / 26.641 / 8.5 | 41.624 / 22.592 / 61.732 / 14.2 | 43.468 / 22.117 / 26.588 / 7.4 |
| 43 | 43.498 / 22.118 / 26.630 / 5.4 | 43.638 / 22.039 / 26.543 / 6.6 | 43.559 / 22.075 / 26.583 / 7.5 | |
| 44 | 43.564 / 22.090 / 26.516 / 5.3 | 39.564 / 23.515 / 62.482 / 6.6 | 42.300 / 22.692 / 31.604 / 13.2 | |
| 45 | 43.633 / 22.052 / 26.583 / 5.4 | | 43.637 / 22.075 / 26.579 / 7.4 | |
| 46 | 43.540 / 22.114 / 26.522 / 5.3 | | 43.422 / 22.123 / 26.596 / 7.3 | |
| 47 | 43.373 / 22.279 / 26.608 / 5.1 | | 42.748 / 22.192 / 36.592 / 7.4 | |

读法与干扰判定：

- p99 ≈ 26.6ms 的轮次是干净轮次；p99 抬到 31.6–62.5ms 的轮次受同租户偶发活动干扰，其 QPS 不可用于配置间比较。干扰在 AF 与 noAF 轮次都出现（seed44 的 `bc-noaf` 掉到 39.564、p99 62.482），不是 AF 引入。
- 只取干净轮次：dyn-noaf 43.373–43.633（6 轮全部干净）vs bc-af 43.422–43.637（3 轮干净），区间重叠，AF 无可辨收益；这与第 3 节"执行图完全一致"互为印证。
- 100 case infer 的 e2e 也同样重合：dyn-noaf 1227.8ms、bc-noaf 1216.5ms、bc-af 1233.2ms、bc-af-all 1221.9ms。
- 唯一稳定的差异是编译期：AF 的单图编译中位 7.5s vs noAF 5.3s（+2.2s，+40%），且 AF 侧呈双峰（7.3–7.5s 与 13.2–14.2s 两组，noAF 全部落在 5.1–5.4s，`bc-noaf` 6.6–8.5s）。按 5 实例口径推算，启动至少多 ~11s。

## 5. 一处需要避免的误判

打开 GE verbose 日志（`ASCEND_GLOBAL_LOG_LEVEL=0/1` + `ASCEND_PROCESS_LOG_PATH`）本身会把 infer 拖慢约 6 倍：同一 AF 配置，带 INFO 日志时 100 case e2e 7579.3ms（avg 75.8ms/case），不带日志时 1233.2ms（avg 12.3ms/case）。评估编译期行为时要分开跑，不能把日志开销记到 AF 头上。

## 6. GE API v2 下复测（2026-11-23，runtime 提交 `9a2f3d3` 之后）

第 1 节的失败与绕过都是 v1 口径。runtime 切到 GE API v2（`ge/ge_api_v2.h` + `libge_runner_v2.so` + `ge::GeSession`）后，带输入编译不再是另一个接口，而是同一个 `CompileGraph` 的重载：

```cpp
Status CompileGraph(uint32_t graph_id);                                         // 交付路径
Status CompileGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs);   // AF 需要的入口
```

复测时同样给 `ge_execute` 加了个默认关闭的开关（编译时用样例 case 的主机字节构造 `ge::Tensor`：`TensorDesc(Shape, FORMAT_ND, dtype)` + 主机指针 + 字节数，走两参重载，其余流程不变），**测完已回滚，v2 上不需要 patch**。口径与第 2–4 节一致（同 `packed.air`、同 100 case 池、`must_keep_origin_dtype`、`--instances 1`、chip 2）：

- **默认路径在 v2 上仍然开不起来**：不带样例输入的单参 `CompileGraph(gid)` + AF 复测（2026-11-23 18:55）报同一组错——`symbolic_shape_symbolizer.cc:65 Node[arg2_1] is not supported symbolize, input origin shape is [-3]` → `autofuse_optimize.cc:205 Symbolize graph input failed` → `Compile graph failed, error code:1343225857`，只是出口从 v1 的 `ge_api.cc:1042` 变成 `ge_api_v2.cc:800`，`ge_execute` 侧表现为 `CompileGraph(gid=1) failed: E19999: Inner Error!` 后 exit 1。v2 降低的是"绕过"的代价，不是 AF 的前置要求。
- **v1 `BuildGraph` 的两个副作用在 v2 消失**：`GetCompiledGraphSummary` 正常（日志仍是 `allocation=GE (dynamic compiled)`，`IsStatic()` = false），`LoadGraph(gid, {}, stream)` 正常（`LoadGraph ok (gid=1), cost 1234.0 ms`），不再需要"跳过 LoadGraph"和"summary 为 null 时降级"。
- **符号化通过、autofuse 真的跑了**：`symbolic_shape_symbolizer.cc:310 SymbolizeShape: Symbolize data node arg2_1, index 0, value 2053, symbol s0`（arg4_1 = 2053、arg15_1 = 17，即样例 case-0037 的真实 shape），`autofuser.cpp:147 [Lowering] 28092 µs`、`:165 [Lifting] 933 µs`；第 1 节那三条报错与 error 1343225857 均不再出现。
- **精度**：100 case / 1600 行 PASS，worst cosine `0.9994394615744572`、rel L2 `0.033784056043048426`，与不开 AF 的输出 **100/100 逐位相同**；池内 89 种 shape 全部正确执行，说明带输入编译没有把图变成 shape 特化。
- **性能仍无收益**（`--iterations 3000 --warmup 10 --aicore-num 8|16`，同 seed 交替，两轮 p99 均 ≤ 26.6ms，未受同租户干扰）：

| seed | 不开 AF：qps / p50 / p99 / 单图编译 | 开 AF：qps / p50 / p99 / 单图编译 | 差值 |
|---|---|---|---|
| 42 | 43.631 / 22.038 / 26.487 / 5.152s | 43.683 / 22.012 / 26.467 / 6.355s | qps +0.12%，编译 +1.20s |
| 43 | 43.600 / 22.064 / 26.573 / 5.150s | 43.620 / 22.051 / 26.565 / 6.332s | qps +0.05%，编译 +1.18s |

qps/p50/p99 的差异都在噪声内（v2 不开 AF 的既有参照：seed42 43.643、seed43 43.576），唯一稳定成本是每份图编译 +1.2s（+23%，v1 时是 +40%）。

**结论不变：A3 不引入 AUTOFUSE_FLAGS。** v2 降低的只是"开启 AF"的工程代价，收益侧仍为 0——第 3 节的执行图对比说明这张图上 reduce/concat 没有可融合模式，与用哪个 API 版本无关。若以后图形态变化（出现大量未融合的 reduce/concat 链，或改走静态导出），两参 `CompileGraph` 就是现成的评估入口。

## 附录：原始产物与复现

产物都在 `models/qwen2.5-0.5b/work/`（gitignored）：`af/EVIDENCE.md`（证据摘录）、`af/compile-case.patch`（**v1 时代**的 `--compile-case` 开关，只对 `9a2f3d3` 之前的 runtime 有效）、`af/{phase2,phase3,phase4,ab2}.sh`（复现脚本）、`af/hist-*.txt`（算子直方图）、`af/plog-v2-af/`（第 6 节的符号化取证）、`logs/{infer,compare,bench}-*.log`、`bench-*-i3000/result.json`（逐请求记录在 `records`）、`output-b-*/comparison.json`、`bench-v2af-*/result.json` 与 `output-v2af-*/comparison.json`（第 6 节）。

```bash
cd models/qwen2.5-0.5b
# 第 1-5 节（v1 口径）：先加回 v1 时代的开关
git apply work/af/compile-case.patch && bash ../../runtime/build.sh
bash work/af/phase2.sh   # A: AF 编译证据(dump + DEBUG plog)  B: 干净 infer A/B + 精度  C: bench 矩阵 seed42-44
bash work/af/phase3.sh   # noAF 图 dump，与 AF 对比算子构成
bash work/af/phase4.sh   # 7 pass 全开的 AF + seed45-47 配对
bash work/af/ab2.sh <dyn-noaf|bc-noaf|bc-af|bc-af-all> <seed> <iterations>   # 单轮
```

第 6 节（v2 口径）没有存档 patch：改动就是编译时按开关改调两参 `CompileGraph(gid, inputs)`、inputs 用样例 case 的主机字节构造，其余流程（summary、`LoadGraph`、执行）一行不动。

`--instances 5` 的生产口径本轮未测（当时 HBM 被同租户占满，每卡仅剩 ~3GB，5 份图实例加载即 `rtMalloc fail`）；由于执行图与输出都逐位一致，5 实例只会得到同样的零收益结论，未安排补测。
