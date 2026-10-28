# 100 迁移：独立测试准备审查

结论：CPU 验收逻辑与双场景准备入口已独立复跑通过；这不构成 100 上的模型正确性、性能或稳定性通过。测试角色未发起 SSH 或 NPU 任务，未修改 runner、原模型、expected、精度阈值、旧 99 证据或全局 progress。

## 独立复跑

项目根下执行：

```sh
python3 work/bge-opt-100/tester/run_cpu_readiness.py
```

完整 argv、工作目录、PYTHONPATH、退出码、源码/输入 SHA256 和日志哈希见 [cpu-readiness.json](../../../delivery/supporting-evidence/work/bge-opt-100/tester/cpu-readiness.json)。脚本在测试专属目录组装本地源快照，并且只在内存中调整旧验收器的本地 fixture 路径；验收逻辑和阈值不变。此测试组装不是 100 上的冻结 build。

| 检查 | 本轮结果 | 原始证据 |
|---|---|---|
| fixed/random replay 准备 CLI | 12 个测试通过 | [dual-workload-cli.log](../../../delivery/supporting-evidence/work/bge-opt-100/tester/dual-workload-cli.log) |
| acceptance-v2 场景和输入身份 | 13 个测试通过 | [workload-identity-v2.log](../../../delivery/supporting-evidence/work/bge-opt-100/tester/workload-identity-v2.log) |
| 旧验收器对抗检查 | 15 项通过 | [legacy-adversarial.log](../../../delivery/supporting-evidence/work/bge-opt-100/tester/legacy-adversarial.log) |
| 完整分数、Top10 和最近秩自检 | 8 项通过 | [legacy-selftest.json](../../../delivery/supporting-evidence/work/bge-opt-100/tester/legacy-selftest.json) |
| 100 原图 GE mode=0 数值正确性 | 未执行 | 等待设备侧冻结证据 |
| 100 fixed/random 性能和最终稳定性 | 未执行 | 等待设备侧冻结证据 |

对抗测试确认可拒绝：同 shape 但错误 case 的输出、事件与 manifest 关联错误、重复/缺少响应、输出二进制重叠或尾随字节、FP16 外部输出、NaN、完整分数越界、Top10 顺序交换、D2H 字节不足、请求失败、重叠在途、预热不足以及把排空后完成数除以原窗口。场景测试拒绝“实际五条池却标固定 case”、错误 case、输入内容被改、固定场景 runtime pool_size 不为 1，且固定 case 压力不能替代五例全覆盖。

## 可复用入口与固定契约

1. [acceptance-v2/verify_outputs.py](../../../delivery/supporting-evidence/work/bge-opt-20261026/acceptance-v2/verify_outputs.py) 的 `compare_one`、`npz_check`、`pressure_check`、`perf_check` 是本次复用的主入口。[workload_identity.py](../../../delivery/supporting-evidence/work/bge-opt-20261026/acceptance-v2/workload_identity.py) 核对原始八输入的 dtype、shape、内容哈希和实际打包池；场景不能仅靠标签判定。
2. 完整输出必须为有限 FP32 `[1,10112]`。数值门槛：cosine ≥ 0.9999、relative L2 ≤ 0.001、max absolute error ≤ 0.001。Top10 使用稳定降序，下标加 1 得到 `dense_id`，ID 和顺序精确一致，分数 atol=0.001/rtol=0。
3. 压力验收通过 `(run_id, request_id)` 一一关联 events 与 manifest，再核对 instance/case/phase 和该 case 的完整向量。输出区域必须完整、无间隙/重叠/未引用尾部。CLI 不允许把压力门槛降到 2000 请求或每 instance×case 20 次以下。
4. `perf_check --plan` 同时适用于实际单条池的 fixed-case-0002 和完整五条池的 random-five。它从 events 重算窗口成功数/QPS、排空、最近秩 p50/p99，检查每实例闭环、连续全局 request ID、预热和 H2D/D2H 字节数。正式三轮按每轮 ≥60 秒且 ≥10000 窗口内成功请求，搜索按 ≥30 秒且 ≥2000；两种场景分别统计，不能合并。
5. 旧 [verify_capture.py](../evidence/20261026/acceptance/verify_capture.py) 可验 infer 的五例 `result.json + outputs.bin`；`verify_outputs.py npz` 可验五例 NPZ。二者都不能替代多实例请求关联压力。

100 上完成独立身份适配后，命令形态应保持如下。路径由冻结交接给出，`A` 是 100 的独立验收目录；不应原样使用旧 99 的绝对路径。

```sh
python3 "$A/verify_plan_inputs.py" --plan "$PLAN" --report "$REPORT_INPUT"
python3 "$A/verify_outputs.py" npz --outputs "$OUTPUT_NPZ" --report "$REPORT_NUMERIC"
python3 "$A/verify_outputs.py" pressure --instances 8 --min-requests 2000 --min-per-case 20 --manifest "$MANIFEST" --binary "$OUTPUT_BIN" --events "$EVENTS" --report "$REPORT_PRESSURE"
python3 "$A/verify_outputs.py" perf --instances 8 --min-seconds 30 --min-requests 2000 --warmup-min 10 --plan "$PLAN" --performance "$RESULT" --events "$EVENTS" --report "$REPORT_SEARCH"
python3 "$A/verify_outputs.py" perf --instances 8 --min-seconds 60 --min-requests 10000 --warmup-min 10 --plan "$PLAN" --performance "$RESULT" --events "$EVENTS" --report "$REPORT_FORMAL"
```

实例数必须来自实际冻结配置，上面 8 只是示例。`pressure` 的 events/manifest 必须来自完整五条池，不能将 fixed 性能输出用于该验收。没有 p99 上限，结果维持 `p99_limit=null` / `SLO_acceptance=not_configured`。

## 迁移前必须补齐的身份适配

| 缺口 | 影响与所需证据 |
|---|---|
| `verify_outputs.py EXPECTED` 固定 `/export/home/ext.hejun23/workspace/bge/expected_outputs.json` | 100 新项目需解析其自身只读 materials 路径并核对原 expected SHA256，不能因路径不通而跳过比较。CPU golden/manifest 也应从 100 交付包明确读取。 |
| `validate_run_identity.py` 固定逻辑 10、99 主机/BDF、旧 runner/packed graph/plugin 哈希 | 新 runner 构建哈希、100 主机/BDF、逻辑卡、每次实际加载插件与计划身份需单独冻结；保留旧 99 检查器作为旧证据。新哈希并非可任意放行的白名单。 |
| 旧 `aggregate_rounds.py`、`verify_stability.py` 同样固定 device10/物理5/0/BDF | 需使用 100 实测物理映射，分别聚合 fixed/random。三轮 QPS 取中位数、p99 合并原始延迟，不能平均三个 p99。 |
| 原 CPU golden 和 packed 输入来源要随材料迁移 | 原模型 SHA256 `2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0`；sample `66571de2fa8c6a97e52a196a957801ab09b429751ba9cfb94712bb32c660b194`；expected `4765cc2b9dac9d2f523f9a684a81082d3b943caaff1fd2c4db90cfdf1b375384`。移植不要求重新制造相同 NPZ，但必须验证旧 golden 身份和独立 CPU 来源。 |
| 本地双场景补丁需在 100 实际集成 | 已通过本地准备测试不等于远端集成完成。固定池应实际仅含 case-0002，检查原八输入内容和 pool_size=1；随机池为原五例。 |

`perf_check` 是事件数学验收器，单独运行不证明性能环境有效。外层还必须检查 native/包装器退出码、作业清理、原图 30 Slice/0 Range 及权重语义、GE mode=0、真实精度参数、实际实例数、`capture=false`、profiling/dump 关闭、scope=`h2d_desc_execute_sync_d2h`、arrival_model=`closed_loop`，并核对新二进制/图/插件哈希、设备空闲门禁和实际运行中的外部负载。完整压力还需将 manifest/event 数量与 runtime 宣布的请求总数绑定，防止两份事件同时缺失而仅满足最小数量。

三轮生命周期、随机场景连续 10 分钟资源稳定性以及退出后真实设备释放仍是独立项目。旧稳定性脚本保留了资源趋势人工审查，并不因满足资源增量数字就自动证明没有持续增长；需要检查约 10 秒采样完整性、前后各两分钟中位数和吞吐趋势。逻辑卡 14 与 15 的共享资源和宿主 PID 也应纳入本轮资源门禁。

## 下一次独立验收交接

请交付冻结目录或清单：材料/CPU reference/hash、新源码快照和补丁、新 runner 与 plugin hash、100 卡的逻辑到物理映射、实际命令/环境与 plan、device_job request/status、结果/events/压力 manifest+binary、资源 JSONL、运行前后门禁与本任务 Host PID 见证。测试从这些只读材料重新计算；开发方的 PASS 汇总本身不是验收依据。
