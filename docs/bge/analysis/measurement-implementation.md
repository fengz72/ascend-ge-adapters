# BGE 完整搬运测量入口实现

日期：2026-10-26。开发自测报告；基础测量能力修正，不是一个模型性能优化点。没有由开发角色提交任何 NPU 作业，本文不宣称原图或候选的新设备精度、QPS、p99 已通过。

## 实现与冻结身份

基于 `441b245fc18ee2a78156becf80488ee16ad3474d` 完整快照，仅修改独立副本 `/export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026/runner` 的八个文件：

- `runtime/bench.h/.cpp`：增加可选回放契约与终态字段；将公共分位数改为最近秩；性能 JSON 明确计时口径、窗口与 cohort。
- `runtime/replay.cpp`：新增固定时间窗、带请求数门槛和上限的闭环回放。每实例独立 worker，在同一线程覆盖每个真实 case 的预热，再统一开始测量；全局 seed 预生成有放回序列，request_id 全局唯一。
- `runtime/backends/gesession_backend.h/.cpp`：使用既有 `CollectOutputs` 完整 D2H，抑制高频正常日志；校验输出 dtype、shape、实际逻辑字节和非空地址。每实例独立 Host 输出容器，压力输出 sink 同步完成后才可进入下一请求。
- `runtime/main.cpp`：在原执行 JSON 增加可选 `replay` 对象；保存每请求 events，压力模式额外保存完整输出和 manifest。保持旧 infer 和固定次数 bench 的入口。
- `runtime/CMakeLists.txt`、`tests/runtime_contracts.cpp`：链接新入口并补 CPU 回归。

本地可审阅补丁：`work/bge-opt-20261026/measurement.patch`；源码副本：`work/bge-opt-20261026/measurement-src`；逐文件 SHA：`work/bge-opt-20261026/measurement-identity.json`。补丁 SHA-256：`b0190bb626f9a808d66ff5d4c19d3df638ed34be5b87257c62e4c013021b0c15`。

远端 `runtime/build/ge_runtime` SHA-256：`8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d`。冻结后交独立测试角色审查；后续设备运行使用该二进制身份。未修改原始模型、权重、输入、expected、阈值、模型 YAML 或全局配置，也没有提交 Git commit。

## 最小调用方式

复用现有 plan 的 `backend/graph/io/inputs/options/observability/output_dir`，保持 `mode=bench`、`instances=1`、`iterations` 为正整数；设备由主控设置（当前授权目标为逻辑设备 10）。正式基线增加：

```json
{
  "replay": {
    "run_id": "baseline-2027",
    "min_seconds": 60,
    "max_seconds": 300,
    "min_requests": 10000,
    "max_requests": 2000000,
    "warmup_per_case": 10,
    "capture_outputs": false
  }
}
```

沿用 plan 的 seed；搜索为 42，独立验收为 2027/2028/2029。`precision_mode=must_keep_origin_dtype` 和不指定 `aicore_num` 为原始全核基线；不同精度/融合候选仍使用相同测量工具。主控设备锁、占用检查、超时、预算和资源监控仍由已有外部任务工具负责，本实现不建立第二套调度器。

压力验证使用相同入口，`capture_outputs=true`，设置独立 run_id 与目录。所有请求均执行完整 D2H，随后同步写入 `outputs.bin` 与 `manifest.jsonl`；文件归档在该请求 latency 终点之后。因此压力模式 QPS 含归档对发放的影响，明确标记 `capture_enabled=true`，不可当作正式性能结果。

## 计时、统计与证据字段

测量请求在 dispatcher 锁内获准发放的时刻记录 `started_ns`，包含获准后至首次 H2D 的短调度间隙、H2D、描述符、GE 执行与同步、完整 D2H 和本次 Host 输出分配；D2H 返回后记录 `completed_ns`。同一 clock 的纳秒时间戳用于所有请求和窗口。这样截止前获准的请求会完整排空，截止后不再发放新请求。

按设计保留原来的每请求 HostTensor/data 分配，且纳入计时；本次没有混入 Host 容量预分配/复用优化。Device 输入输出所有权与释放策略沿用既有 runner，本次不重写。

达到最短时长且已成功完成请求数满足门槛后停止新发；到最大时长或序列容量上限也停止并排空。主 QPS 和 samples/s 只数 `completed_ns <= window_end_ns` 的成功项；`cohort_qps` 和 `cohort_successes` 单列，不能混作窗口完成率。`statistics_sufficient` 仅在窗口内成功完成数、最短时长和零错误均满足时为 true；执行成功而统计不足会明确返回 false，不据此验收性能。

`events.jsonl` 包含 run_id、request_id、instance、case_id、shape_key、phase、status、error_stage、started/scheduled/sent/received/completed_ns、H2D/desc/execute/D2H/e2e 分项和搬运字节。闭环 scheduled/sent/received 与实际发放相同；不声称开放到达或排队性能。measurement request_id 为全局整数，warmup 为 `warmup-{instance}-{index}` 字符串以保持唯一。

`manifest.jsonl` 仅包含 measure 请求，按 run_id/request_id/instance/case_id 关联；每行 tensors 含 name、dtype、shape、offset、bytes。二进制是无 header 的完整原始输出拼接，各行占独立字节区间；BGE 为 FP32 `[1,10112]`、40448 字节。sink 在同一个锁内分配 offset、写二进制与 manifest，且检查 flush 失败。任何 H2D/execute/D2H/sink 失败保留终态阶段并停止新发，不将失败计为 success。

性能 JSON 包括 `latency_scope=h2d_desc_execute_sync_d2h`、`arrival_model=closed_loop`、`slo_status=not_defined`、窗口起止/最后完成时间、window/cohort 成功数、排空耗时和截止时在途数。原 `requests.csv` 为兼容辅助输出，新契约以 events 为准。

## CPU 自测与构建

本地 clang++ 编译并执行 CPU 契约成功。远端在 CANN 9.0.0 环境执行：

```sh
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build -j4
ctest --test-dir runtime/build --output-on-failure
```

完整二进制构建成功，`runtime_contracts` 1/1 通过。日志位于本地 `work/bge-opt-20261026/measurement-build.log`。

新增契约覆盖：最近秩 p99 的 100/1000 样本边界；受控时钟证明 D2H 等待在 latency 内；D2H 失败保留阶段且不调用成功 sink；sink 抛异常不变成成功；固定时间上限和尾部排空；仅一条窗口后完成的 cohort 不能使统计充分或得到窗口 samples/s；两实例各自保留当前 case 与输出关联，且完整全局序号与 seed 序列一致；events 的完整搬运字节与唯一 warmup/measure 标识。既有固定次数、线程预热屏障、部分失败和请求池测试继续通过。

CPU 结果不能证明 CANN 实际内核、输出生命周期、精度或吞吐；冻结后二进制仍需主控先做新 D2H 路径压力精度验证，再建立同口径基线。

## 回退与边界

以原完整快照 `441b245` 重建独立副本即可回退；原 runner checkout 未修改。已在本地保留八个旧文件内容以便补丁核对。没有将缓冲复用、图优化或低精度收益计入基础测量变更。

沿用底层 `Execute` 的同步和原有资源销毁路径；设备同步失败或外部超时后不能由本工具单独证明硬件已安全释放，应由外部任务管理和后置进程/设备核对决定。内存分配失败、进程崩溃或文件持久化失败均不得据部分记录宣称通过。性能、压力/稳定性和模型优化的最终接纳由主控与独立测试依据新的实际设备证据决定。

## 附录：可复现的 Python 准备入口（2026-10-26）

新增 `models/bge/replay.py`、`tests/test_bge_replay_cli.py` 并追加模型 README。该补充只准备 plan，不调度或执行 NPU；没有修改上述八个冻结 C++ 文件、融合 so 或二进制。远端二进制 SHA 仍为 `8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d`。

入口使用 `load_authoring(source_paths=False)`、现有 `data.load`、原生 adapter 的 `prepare_runtime`、`worker.encode` 和 `runtime.read_io`。仅加载已有 NPY 和小型外部权重打包图的 IO metadata，不读取原始 1.35 GB 权重、不重新生成输入或参考；严格保持五例 batch=1、八输入 INT64/FP32 和 FP32 scores 输出。

`--output-dir` 必须全新，只写 `inputs.bin`、`plan.json`；plan 的 output_dir 为未创建的 `runtime/` 子目录，与受管 launcher 的拒绝覆盖策略一致。默认 graph 来自 config 已构建产物，也可 `--graph` 指向已有外部权重打包 ONNX。配置文件仍不写 work_dir。

参数覆盖 device、instances、precision-mode、可选 aicore-num、min/max-seconds、min/max-requests、seed、warmup-per-case、run-id 与 capture。未指定核数表示全核，明确去掉 YAML 的历史 `4|8` 限制。`--prepare-only` 为唯一执行模式，默认行为也是仅准备；不存在裸 `--run`。实际设备占用门禁、锁、超时和预算由主控既有 run_checked/device_job 负责。

可选 `--preload` 只在 plan.runtime_env.LD_PRELOAD 写入当前继承值追加库路径，Python 环境不变。受管 launcher 须仅应用到 native 子进程并保留实际父 job 的 allocator；准备与执行环境不一致时不能用覆盖赋值丢掉后注入的 allocator。

远端 CPU 测试 8/8 通过，覆盖：完整五例 17304 字节打包与逐 tensor dtype/shape/数值核对；实例数不被强制改成 1；原 golden 不修改；不调用 native runtime 或 source loader；参数错误/非 GE 路线/少样例/错误 IO/大 inline 模型在准备前拒绝；已有目录不覆盖；窗口和 graph 覆盖参数；preload 追加并不污染 Python 环境。`--help` 正常。日志：`work/bge-opt-20261026/replay-cli-tests.log`。本地系统 Python 缺 PyYAML，因此最终测试在实际远端容器的 Python 执行；未据本地依赖缺失改全局环境。

独立可审阅补丁为 `work/bge-opt-20261026/replay-cli.patch`，逐文件与补丁身份为 `replay-cli-identity.json`，供主控集成到 canonical runner 后再做对应离线验证。本开发角色仍未自行执行 NPU。

## 附录：canonical 集成后的交付只读审查

主控已通知：冻结测量入口与 Python 准备 CLI 已集成到 canonical `/export/home/ext.hejun23/workspace/bge/ascend-ge-adapters`，canonical 二进制与本报告冻结 SHA 一致，独立 CPU 测试及原始五例/压力/基线已通过。本角色本次仅只读核对源码、模型 README、YAML 与受管 launcher，没有重复构建、测试或提交 NPU 任务；最终推荐实例数仍待主控冻结，不能把中间搜索值当作交付配置。

只读材料保存于本地 `work/bge-opt-20261026/canonical-delivery-review.txt` 和 `canonical-delivery-lines.txt`。

### 容易误用的两条入口

canonical `models/bge/README.md:20` 的 `run.sh run`、`:32` 的 `run.sh bench` 仍经七步执行入口产生不含 replay 对象的 plan，后端在 `runtime/backends/gesession_backend.cpp:613` 选择旧 `BenchPool`。该路径保留兼容语义，计时为 H2D/描述符/GE+同步，不含每请求 D2H。它仍读取 YAML 的 `aicore_num=4|8`、warmup=1、iterations=6，不能复现本轮全核、固定窗口、完整搬运的性能结果。

新 `models/bge/replay.py:111` 至 `:124` 使用独立契约：默认移除 YAML 的历史核数限制，显式传递 CLI 实例数、seed 与时间窗，每例预热 10 次；plan 的 iterations=1 是旧 JSON 必填字段占位，并不把回放限制成一次。后端根据 replay.enabled 走新 `ReplayPool`。因此 YAML 旧值不影响使用新计划产生的本轮结果，但前置快速示例会使读者误入另一计时口径。

最小文档建议是在 README 的 Reproduce 标题之后，立即加入以下说明并链接后面的完整搬运章节；原始兼容示例无需为此修改代码或 YAML：

> The seven-stage `run` and `bench` examples below are historical smoke checks using the YAML's `4|8` core limit and six benchmark calls. Their bench scope is `h2d_desc_execute_sync`, without D2H. For the 2026-10-26 performance results, use [Full-transfer replay plan preparation](#full-transfer-replay-plan-preparation), then submit its plan through the managed launcher. The replay CLI uses all cores by default and its own request/time-window parameters.

配置说明处的 `4|8` 和六次调用同样标为历史冒烟值即可。不得把迭代次数改大后声称旧 bench 已具有 D2H 口径。

### 补齐从准备到受管执行的一步

README 当前的新章节仅给准备命令，随后泛称使用 `run_checked.py/device_job.py`，这两个 wrapper 并非 canonical 项目里的相对路径。完整交付文档至少应给出本轮实际 helper 的绝对路径或其交付包位置和版本，避免读者准备完计划后改用旧 `run.sh bench`。

本主机当前 wrapper 的命令形式如下；仅为经只读源码核对后的文档示例，本角色没有执行。run_id、目录与 job 名必须全新，`--instances` 在最终配置冻结后填写。基线仍明确为 1 实例：

```sh
cd /export/home/ext.hejun23/workspace/bge/ascend-ge-adapters
python3 models/bge/replay.py --prepare-only \
  --device 10 --config models/bge/config.yaml \
  --output-dir work/bge/replay/baseline-2027-new \
  --instances 1 --seed 2027 \
  --precision-mode must_keep_origin_dtype \
  --min-seconds 60 --max-seconds 300 --min-requests 10000
python3 /export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026/tools/run_checked.py \
  --job baseline-2027-new --timeout 650 \
  --binary /export/home/ext.hejun23/workspace/bge/ascend-ge-adapters/runtime/build/ge_runtime \
  /export/home/ext.hejun23/workspace/bge/ascend-ge-adapters/work/bge/replay/baseline-2027-new/plan.json
```

显式 `--binary` 很重要：本轮 wrapper 当前默认仍指向隔离 runner 的二进制；虽然主控确认目前两个二进制 SHA 相同，交付复现命令不应依赖这种偶然一致。该 wrapper 当前还明确固定逻辑设备 10 与物理 BDF，并维护当前任务的累计预算；它不是任意设备通用入口。已有预算或设备门禁拒绝必须保留，不能绕过 wrapper 继续跑。

正式性能结果应明确显示 `status=success`、`statistics_sufficient=true`、`latency_scope=h2d_desc_execute_sync_d2h` 且 `capture_enabled=false`；压力输出校验另用 capture=true 的新计划。最终文档应引用已冻结实例/核数/精度与 binary/plan 身份，不用历史六次冒烟结果作为收益分母。

### README 最小修正已落实

经主控明确授权，已仅修改 canonical `models/bge/README.md`：在旧 Reproduce 示例前说明无 D2H 的历史冒烟口径；将 `4|8`、warmup=1、iterations=6 标明为旧配置字段；完整搬运章节补受管执行命令，并用 `--binary` 显式选择 canonical 二进制。示例保持 1 实例全核基线，最终设置链接到 `../../docs/bge/acceptance.md`，没有提前指定搜索中的实例数。

该 README 是 CLI 初版 README 的后续文档修正，身份从 `e49415f5a17834e4411d502094a5b9040612f72a82230c11283daf74e1dae014` 更新为 `3670b40d05530f2500f07e28b8b9b8502d6f7b05163f60fc644384ba3ed56c86`（具体以 `work/bge-opt-20261026/readme-delivery-identity.json` 为准）。初版 `replay-cli-identity.json` 是当时三个文件的历史身份，不应再当作当前 README 哈希；CLI 源码和测试身份不变。新 README-only 差异为 `work/bge-opt-20261026/readme-delivery.patch`。

canonical 测量二进制再次只读核对仍为 `8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d`。未修改 CLI、C++、so、YAML、expected、全局 OPT/progress 或独立测试快照，也未构建或启动 NPU。

## 固定case与五例随机准备入口补充

用户新增双场景要求后，在可信本地副本`work/bge-opt-20261026/measurement-src`准备`--case-id`。先验证五例来源与原IO，再选择实际编码请求池；固定case-0002只打包原第三例3448字节，dtype/shape/内容全部保留。不传选项仍五例17304字节、均匀有放回回放。plan记录workload.kind/case_ids/sampling；未知ID和空ID落盘前拒绝，固定输入换seed不改变payload，已有目录拒绝覆盖。原C++和8efca二进制未修改。

本地12个CLI CPU回归PASS；原有测试的临时目录改为resolve()以兼容macOS /var→/private/var别名，并未放宽路径/输入断言。PyYAML6.0.3仅安装在任务隔离python-deps用于CPU测试，未修改系统环境。该验证使用mock图IO并真实打包8项输入，不表示真实GE固定场景已运行。

SSH99当前在bastion阶段断开，未能读取或更新canonical文件，远端集成和固定设备测量均待完成。已生成基于已确认canonical三文件SHA的[独立补丁](../evidence/20261026/dual-workload-replay.patch)、[身份](../evidence/20261026/dual-workload-replay-identity.json)与带原SHA校验的[集成脚本](../evidence/20261026/integrate_dual_workload.py)；恢复连接后需核对原身份再集成、复跑Linux CPU回归。不覆盖旧模型/NPY/golden/expected或历史测量证据，不自动启动设备。
