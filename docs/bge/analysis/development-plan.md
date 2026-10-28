# BGE 开发实施设计（尚未实施）

日期：2026-10-24。开发角色交付；仅设计和只读源码核查。本文件不代表已修改 runner、重新执行模型或取得性能收益。

与主控 `../design.md` v1 配套；以下接口名是实现建议，测量规则、配置名额、窗口/请求门槛、seed、压力/稳定性和停止预算以主设计为准，不能以本文件另行改变工作负载。

## 1. 基线、范围与启动条件

实际执行位置固定为 SSH `99` → `hejun-cann9-bge` → `/export/home/ext.hejun23/workspace/bge/ascend-ge-adapters`。代码基线为 `441b245fc18ee2a78156becf80488ee16ad3474d`，预检工作区干净；源码证据为 `../evidence/runner-source-441b245.txt`，身份与设备证据为 `../evidence/preflight.json`。后续实现从这一完整快照建立独立工作副本和独立 work/results/cache，集成后重新记录身份。

唯一模型路线：原始 ONNX → GE Session mode=0。保持原图、FP32、`must_keep_origin_dtype`、八输入各自 dtype、五条输入与 expected、完整分数和 Top10 阈值。业务 ID 为输出索引加 1；不重新启用 Slice 改写，不把降精度、修改权重或增添支持范围混入本轮。

用户最新要求先完成设计；当前 16 个逻辑 NPU 都有外部进程占用，暂停性能测试，也不为本设计启动图编译/加载、NPU 正确性或 profiling。恢复时由主控重新检查授权设备与物理映射、外部 PID/HBM、运行任务和主机锁；发现全部候选设备占用即保持暂停，不杀他人进程、不选“低利用率但已占用”的设备试跑。设备锁不能约束其他用户，性能轮次前后还需核查外部负载。

后续预算上限为 12 组候选配置、累计设备占用 2 小时。构建/加载、失败、正确性、搜索、复测、稳定性及清理都计入占用预算，不通过改 run_id 重置。预算耗尽就交付已有证据与未验收项。p99 目前没有数值上限：可记录回放吞吐—p99 曲线、推荐配置，不能标注“满足 p99 SLO”或生产容量达标。候选编号和预算账本由主控维护，本文件不分配 OPT。

## 2. 已确认的 runner 缺口

| 位置 | 当前行为 | 必须补齐的验收能力 |
|---|---|---|
| `runtime/bench.h/.cpp` 的 `BenchPool` | 固定请求数；H2D/描述符 → Execute；计时止于 Execute 返回 | 含 D2H 完成的边界、公共时间窗口、完整请求事件、失败和尾部排空 |
| `runtime/backends/gesession_backend.cpp` 的 `Execute` | `ExecuteGraphWithStreamAsync` 后 `aclrtSynchronizeStream`，因此已等待 GE 完成 | 必须再把全部业务输出拷回 Host；异步提交成功不能当完成 |
| 同文件 `CollectOutputs` | 仅 capture 路径调用；逐输出分配 `HostTensor.data` 后同步 D2H、打印日志 | 让正式 bench 也执行全部 D2H；高频打印和磁盘归档不混入性能测量 |
| `runtime/main.cpp` | bench 无 output capture；infer 按 case_name 汇总输出，没有唯一调用 ID | 压力轮次按 `run_id + request_id` 关联重复/交错请求，保留 instance/case/shape |
| `core/backends/runtime.py` | `mode != bench` 时实例数固定为 1 | 不宣称现有 infer 已覆盖多实例；新增 BGE 专用压力入口使用同一 bench 执行路径 |
| `core/worker.py` 的 `infer_or_bench` | infer 仅接受每个 case 的一份 instance0 输出 | 保留单实例推理契约，压力结果由 BGE 专用校验器解析，不削弱原断言 |

旧六次冒烟结果和不含 D2H 的 QPS 不能成为新计时范围的收益分母。计时契约修复完成后，重新建立“全核、单实例、同一新工具”的可信基线；修复本身没有优化收益声明。

## 3. 最小改动位置和接口

复用现有请求池、GE runner、编码与验证模块。只新增模型专属回放入口和校验器；不引入新 workflow 引擎、通用服务调度平台或另一套模型执行栈。以下是待开发接口，当前仓库没有这些新增字段/命令。

| 文件 | 建议最小改动 |
|---|---|
| `runtime/bench.h` | 扩展 `BenchOptions`：可选 `measure_ms`、`copy_outputs`、受预算约束的最大请求数；扩展 `RequestRecord`：唯一 ID、phase/status、时间戳、D2H 时间、错误阶段；`PoolResources` 增加明确的输出完成钩子 |
| `runtime/bench.cpp` | 在同一 worker 内依次 setInputs → execute/sync → 输出钩子；新增固定时窗的闭环分发；修正最近秩 p99（当前 `times[N*0.99]` 在 N=100/1000 时错一位）；保留原固定计数入口兼容性，旧输出明确标记计时范围；新增 JSONL 请求事件归档 |
| `runtime/backends/gesession_backend.cpp/.h` | GE 输出完成钩子调用完整 D2H，允许“仅 Host 可读”和“复制后交付独立 capture sink”两种用途；从输出函数拆出高频日志。新增选项显式要求 D2H，未配置完成钩子时拒绝执行该契约 |
| `runtime/main.cpp` | 解析可选执行 JSON 中的测量契约；压力 capture 行增加 request_id/case_id/instance/shape；bench perf 轮次只留事件、不写全量输出；失败保留部分事件且结果为 failed |
| `models/bge/replay.py`（新增） | 读取现有配置，复用 `load_authoring`、BGE 数据生成、`core.worker.encode` 和 `core.backends.runtime.call` 构造执行 JSON；支持正确性压力与性能两种明确用途，实例数/核数/窗口/seed/独立结果目录来自实验清单 |
| `models/bge/replay_verify.py`（新增） | 按 request_id 校验每一次压力输出，调用既有全分数与 BGE Top10 比较能力，生成逐请求及汇总报告；明确要求完整请求集合 |
| `tests/runtime_contracts.cpp`、BGE 专用 Python 测试 | 使用假后端和固定数组验证计时边界、事件与关联、错误及资源生命周期；不要求真实 NPU |

先不扩展全局 YAML、七步 CLI 和 `core.worker` 的推理语义。模型辅助把已校验参数写入调用 JSON，新增运行字段只在 runtime 的输入边界严格校验。这比为了 BGE 回放改动所有模型入口更小。若未来其他模型确有同一需求，再另行抽取。

输出完成钩子建议语义为 `finishOutputs(instance, request, record)`：成功仅代表该请求全部输出在本实例拥有的 Host 缓冲内可读；压力模式还可在完成之后将该缓冲交给 capture sink。它不能只触碰一个输出字节、校验地址或只做 checksum 来替代完整搬运。

## 4. 同步、缓冲所有权与异常

第一阶段继续保持一个 worker 对应一个 graphId/stream，每实例最多一个在途请求，不把流水异步同时纳入测量修复。每实例输入 device 缓冲、静态输出 device 缓冲、Host 输出对象均独立；共享的请求输入只读。不得用跨实例全局输出 vector 或同一个可变 Host 缓冲。

单请求顺序：

```text
分配唯一 request_id、选择 case、记录开始
H2D 全部输入 + 按该请求实际 shape 重建输入 Tensor
ExecuteGraphWithStreamAsync
aclrtSynchronizeStream，确认 GE 完成
检查输出个数/dtype/实际shape/字节容量/非空地址
将所有业务输出完整同步 D2H
记录 completed_ns，状态 success
压力模式：把独立快照按 request_id 归档，或同步写完后再允许本实例下一请求
本实例才可复用输入与输出缓冲
```

`CollectOutputs` 的当前 Host 分配行为先保留在测量边界内，避免将缓冲预分配伪装成计时工具修复；如果之后需要 Host 缓冲预分配/页锁定内存，单独作为候选，并对基线和候选用同一完成定义。性能轮次的 D2H 结果只在 Host 留存到该请求完成，不做 CPU 参考比较、不写完整输出文件。压力模式额外归档开销不用于性能数字。

静态/动态输出依旧依据编译 summary 的 `IsStatic()`，不能由 mode=0 或 ONNX 动态维度直接推断。静态编译图的 device 输出由本实例分配、容量检查、复用并统一释放；动态编译图由 GE 返回输出，D2H 之前保留相应 Tensor 和有效地址，完成之后才允许进入下一请求。不要因新增 D2H 顺便重写已有动态输出释放策略；如需变动，先用本机同版 SDK/样例核实归属、单独诊断和回归。

任一阶段失败均记录相同 request_id 的 error 及失败阶段，禁止产生 success。共享 stop 标志阻止新请求继续发放，已启动 worker 按能确认的同步/完成路径退出，再 join，再释放资源。异常 unwind 不得跳过 join，不得在 stream 尚可能使用时释放 device 内存。若 GE 卡死或同步失败导致无法确认完成，外部设备任务超时负责终止并保存任务状态；该轮无效，不伪造 timeout 成功或“已释放 NPU”。清理需核对进程及设备状态。

压力 capture 必须拥有独立输出快照：同步文件 sink 完成后才能复用当前 Host 缓冲；若未来改为异步 sink，队列中必须移动/复制为独立所有者并设有有界容量。文件偏移与元数据在同一临界区分配，禁止多个 worker 并发覆盖。

## 5. 计时与请求事件契约

本轮回放选择闭环：worker 完成一次请求后才发下一次。每实例一个 worker，客户端并发等于实例数。该结果用于五条样例回放的吞吐比较，不声称开放到达业务容量，也不隐藏不存在的服务排队/网络时间。

计时从开始 H2D 前到完整输出 D2H 返回后；包含实际每请求描述符和 Host 输出分配，所有精度比较、Top10 排序、归档和调度准备单独处理。每实例预热必须覆盖五条样例的四种不同 shape 至少各 10 次，并且使用正式 worker 和同一 D2H 路径。编译、加载、预热独立记录；测量中实际发生的缓存未命中或特化保留。

用 C++ `steady_clock` 的同一 ns 时钟记录全部事件；不要混用 Python epoch 时间计算延迟。建议每个已发放请求恰好一行：

```json
{"run_id":"...","request_id":"...-000001","case_id":"case-0003","instance":1,"shape_key":"...","phase":"measure","status":"success","scheduled_ns":1000000,"sent_ns":1000000,"received_ns":1000000,"h2d_end_ns":1100000,"execute_end_ns":1200000,"completed_ns":1300000,"h2d_bytes":3528,"d2h_bytes":40448,"error_stage":null}
```

闭环没有外部计划到达：`scheduled/sent/received` 显式记为同一实际发放时刻并写入 `arrival_model=closed_loop`；不能借这些相等字段宣称开放到达发送无滞后。必要时分别记录 `h2d_begin/desc_end/execute_begin/d2h_begin`，exec 与 D2H 分项不覆盖端到端。warmup 同 schema、phase 分开，不在事后移动慢请求。

测量采用所有 worker 公共 `window_start_ns/window_end_ns`，起点前预设规则：搜索最少 30 秒且 2000 请求，上限 180 秒；正式最少 60 秒且 10000 请求，上限 300 秒。到达最短时长但请求不足时继续到请求门槛或上限；上限仍不足明确统计不足。达到实际 window_end 后停止发放并排空所有已发放请求，保留最后完成时间和窗口末在途数。主指标分别列出：窗口内成功完成数/窗口秒数、发送 cohort 成功数/窗口秒数、排空耗时；二者名称明确，后者不冒充稳态完成速率。p99 对整个发送 cohort 的终态请求计算，使用 `sorted[ceil(0.99*N)-1]`，包含窗口末仍在途请求最终完成/报错的延迟；pending、漏行、重复 ID 或错误使该轮无效，不能只取快完成子集。

回放序列从五条真实样例等概率、有放回采样；沿用现有 `Sampler`，由主设计的 seed 预先生成有界全局序列，全局计数器分发唯一序号，保存各 case 实际计数。固定模型/数据、精度、窗口规则和种子用于解释 A/B；搜索 seed=42，独立验收 seed=2027/2028/2029，不能改为无放回均匀循环使随机负载规则改变。不要从五条样例本身声称有五个独立延迟样本；p99 来自足够多的重复请求，且只代表这个有限分布。

本机脚本 `request_metrics.py` 可离线核对事件。p99 上限为空时不要传一个虚构的大数让工具给 PASS；自行输出统计与 `slo_status=not_defined`，最终是否接纳由主控和测试按设计决定。选择哪种统计辅助并不替代尾部、错误、稳态和独立验收。

## 6. 输出验证路径

1. 固定 model/sample/expected/代码/SDK 身份；原图结构及权重身份必须可追溯。CPU ONNX Runtime 五条完整分数生成/读取可信 golden，与 expected 的业务信息分别核对。
2. 设备可用后先用既有 `infer → compare` 验证单实例原图 mode=0。本轮新版本成功才能称“本轮通过”，历史 pass 单列引用。
3. 新输出完成路径单实例五条逐输出验精度，然后在拟交付实例/核数配置做独立压力轮次；与正式性能使用同一二进制、图和 runtime 参数，额外输出 sink 是明确的校验模式。
4. 压力输入用同五条样例重复交错，包括相同 shape 的两个不同 case；manifest 建立 `request_id → case_id/input_sha/instance`。case 与内容绑定而非按完成顺序猜测。每条输出检查 FP32 `[1,10112]`、finite、全量 CPU 阈值、Top10 ID+1、顺序、分数绝对误差≤0.001 且 rtol=0。全分数 cosine≥0.9999、relative L2≤0.001、max abs≤0.001。
5. 对每条压力请求调用现有比较函数时，可用只含对应 case 的配置副本，避免 `case_ids=all` 的完整五条集合检查误用；外围校验器必须先验证全局 manifest、每 case 覆盖、请求集合和唯一性，原业务比较函数/expected/阈值不改。不能只取每实例最后一条输出或每 case 任意一条。
6. 发现漏失、重复、instance/case 串数据、坏 dtype/shape、NaN/Inf 或数值/排名失败，一律错误；结果为空也失败。稳定性和资源曲线由测试角色独立记录。本轮五条覆盖 B=1、S=6/8/9/13、L=128，不能据此验收任意动态维度或新增 batch。

## 7. 有意义的 CPU 回归计划

以下全是后续实现的测试建议，当前未执行这些新增测试。

| 目标 | 不依赖 NPU 的验证方法 | 必须检测的故障 |
|---|---|---|
| D2H 完成边界 | `tests/runtime_contracts.cpp` 的假资源在输出钩子等待受控 latch；等待前断言无 success/completed，释放后才完成 | Execute 返回就提前结束计时；D2H 失败仍报成功 |
| 固定窗口和尾部排空 | 以可注入时钟/受控步骤推进 dispatcher；跨 window_end 的请求必须保留终态，之后不再发新请求 | 少算慢请求、停止窗口后继续发放、用排空 cohort 当窗口完成 QPS |
| 最近秩统计 | 固定升序 1..100、1..1000 的整数延迟，断言 p99 分别为 99/990；一条、空数据和错误请求另测 | 当前索引偏一位；空数据或剔除错误后给虚假 p99 |
| 多实例请求归属 | 假后端按 instance/request/case 写入可区分值，并故意反序完成；校验唯一 ID 和完整覆盖 | 共用缓冲、按完成顺序关联、重复/遗漏请求 |
| 缓冲复用边界 | 假后端在 finish 阻塞期间捕捉下一次 setInputs；用分配/释放计数器覆盖 setup 和运行部分失败 | 当前请求输出未读完就覆盖；异常时 double free/漏释放/join 前释放 |
| 失败与异常 | 在 H2D、execute、D2H、sink 各点注入失败/异常，部分线程创建失败沿用现有场景 | 继续调度新请求、少报错误、join 死锁、部分结果变成功 |
| 输出身份与业务规则 | `tests/test_bge_replay_verify.py` 合成可信的 FP32 scores，逐项注入错 ID/排名/0基ID/误差/shape/dtype/NaN | 把 set 相等当顺序一致、阈值被放宽、NaN 被忽略 |
| manifest 完整性 | 缺失/重复 request_id、相同 shape 不同 case、未知 ID、重复 case、截断输出文件/越界 offset | 只验最后一个或抽样验一条、文件错位仍 pass |
| 配置和身份 | mock `runtime.call` 检查辅助入口生成的 plan 与原图路径/FP32/核数/实例/窗口/输出模式；错误参数必须在调用前拒绝 | 强制单实例、错误设备、改 dtype、缺 D2H、复用旧结果目录 |
| 旧行为回归 | 保留现有固定请求数的 `runtime_contracts`、现有 BGE 验证和 dtype/config 测试 | 仅为回放破坏七步入口和其他模型 |

受控时钟/latch 不靠脆弱的“睡眠后大于某毫秒”断言。CPU 测试只能证明 runner 契约和校验器行为；不能证明 CANN 的真实异步生命周期、核限制生效或 GE 精度。硬件验证延后，状态保持未运行。

## 8. 实施交接和回退

建议先做计时/事件契约及其 CPU 回归，再做 BGE 压力校验入口，最后由测试从冻结快照独立审查。当前用户只要求先完成设计，以上实施步骤暂不启动。后续获准且设备门禁通过后，先校验新工具、新建全核单实例基线，再开展限核×多实例候选。默认全核应通过移除 `aicore_num` 限制表达，并核实实际可用核数和 SDK 生效证据；当前历史 `4|8` 不是新全核基线。

核数候选的合法 AIC|AIV 组合从设备与同版接口确认，不预填未知硬件常数；实例数受可用 HBM 和加载成本限制。第一轮先避免叠加异步流水、Host 内存优化或图改写，以便解释限核和实例收益。若限核配置只有在多实例时更快，保留该组合及其相同输出关联压力验收结果。

每组候选均使用主控预分配的固定记录，记录父快照、改动、生效证据、命令、失败与运行时间、结果和回退。计时工具变化导致历史基线不可比时明确重建基线，不计算虚假百分比。正式性能搜索/复测与 profiling 分开。

回退以固定代码快照和独立配置为单位，保留原始模型、原配置与历史证据；不覆盖源模型、不修改 expected。交付给测试的是完整新快照、二进制/图/数据校验值、命令、request events、压力输出及验证报告和未执行清单。最终建议配置尚未产生；本文件不宣称任何 QPS 或 p99 收益。
