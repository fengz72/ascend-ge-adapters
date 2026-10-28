# BGE 2026-10-26 独立验收记录

角色：独立测试。用户最初授权在逻辑设备 2 恢复原图 GE mode=0 正确性、包含输入输出搬运的基线，以及 FP16 和算子融合实验；随后主控传达用户改用逻辑设备 10，后续正式基线、候选和最终验收统一使用设备 10。设备 2 的新五例正确性保留原设备身份，不冒充设备 10 的性能基线。本文仅据真实证据更新通过项；测试不启动 NPU 作业、不修改模型、expected 或精度阈值。

远端证据根为 `/export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026/acceptance`。原 runner HEAD 为 `441b245fc18ee2a78156becf80488ee16ad3474d`；计时实现由开发在隔离 runner 修改，冻结后另行审查。

## 当前独立结论（2026-10-26 04:00 UTC）

原始模型保持 **30 Slice、0 Range**，GE mode=0 正确性、CPU 完整分数/业务 Top10 对比、设备 10 初始三轮单实例基线均通过。FP16 两种策略和 AddLayerNorm 融合的压力数值均通过，但搜索没有显示吞吐收益。全核多实例搜索中八实例暂时最好，观察 QPS **461.333304**、p99 **18.243470 ms**；相对初始三轮基线中位数观察 +163.14%，**尚未最终验收或接纳**。

| 配置 | 压力正确性 | 有效纯性能搜索 QPS | p99 / ms | 当前验收含义 |
|---|---|---:|---:|---|
| 原始 FP32，全核单实例 | 2000 请求通过 | 175.321044（三轮中位数） | 6.819411（合并） | 初始正式基线通过 |
| Cube FP16 | 2000 请求通过 | 157.291151 | 8.852352 | 单轮无吞吐收益 |
| FP32 融合 V1 | 2000 请求通过 | 无有效值 | — | 外层125/后台子进程失败 |
| 自动混精 FP16 | 2000 请求通过 | 152.631777 | 7.450638 | 单轮无吞吐收益 |
| FP32 融合 V2 | 2000 请求通过 | 171.781605 | 6.137444 | 单轮无明确吞吐收益 |
| FP32 全核双实例 | 2001 请求通过 | 269.899590 | 7.914773 | 单轮搜索观察 |
| FP32 全核四实例 | 2003 请求通过 | 370.212634 | 11.448209 | 单轮搜索观察 |
| FP32 全核八实例 | 2007 请求通过 | 461.333304 | 18.243470 | 暂定搜索优胜，未最终验收 |
| FP32 全核十六实例 | 5015 请求通过 | 无有效值 | — | native SIGKILL/外层125，根因未定 |
| 八实例半核12\|24 | 尚未运行 | — | — | pre-gate 外部占用拒绝启动 |

设备 10 后来出现外部 holder PID 4088730/136MB，主控停止所有新设备实验。最终候选三轮交错复测、10 分钟稳定性及其前后压力复测尚未执行；因此不能将搜索收益写成最终交付收益。只读 PID observer 已停止、退出 0。后续只有在获得可用设备资源并重新通过门禁后才能继续这些剩余验收。

## 参考与材料身份

独立重新计算三份文件 SHA-256，与需求设计完全一致：

| 文件 | SHA-256 |
|---|---|
| model.onnx | `2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0` |
| sample_inputs.json | `66571de2fa8c6a97e52a196a957801ab09b429751ba9cfb94712bb32c660b194` |
| expected_outputs.json | `4765cc2b9dac9d2f523f9a684a81082d3b943caaff1fd2c4db90cfdf1b375384` |

旧 golden 可读，且旧 run 声明 CPU baseline 成功，但路径记录没有完整绑定 ORT 配置及 golden 哈希。因此本轮直接从原始 `model.onnx` 和五份 JSON 输入独立生成新 CPU golden。强制 `CPUExecutionProvider`、`intra_op_num_threads=2`、`inter_op_num_threads=1`、顺序执行；独立构造 `input_ids=query_token_ids` 等八路输入，并逐张量核对 runner 的 NPY 输入内容。

结果为 `cpu-golden.npz`，SHA-256 `a59f406eb5782926181963e94cb1f87e1728621dedc3d613f88a7317d8213f35`，包含五份 FP32 `[1,10112]` 完整 scores。ORT 为 1.26.0、NumPy 为 1.26.4；`cpu-golden-manifest.json` 保留版本、providers、配置、文件哈希与每例输入输出 dtype、shape、有效字节和内容哈希。五例与旧 CPU golden 逐 bit 完全一致；新 CPU golden 的 Top10 对 expected 全部通过，最大误差 `6.056747436922194e-7`。CPU session 加五次推理耗时约 1.83 秒，不作为 NPU 性能数字。

## 原始模型 GE mode=0 正确性

主控提供的新设备输出 `original-correctness/result.json` 与 `outputs.bin` 已独立解析。核对运行退出码为 0、五个 case 各有 instance 0 唯一输出，dtype 为 FP32、shape 为 `[1,10112]`、每份 40,448 字节，二进制偏移无重叠、无空洞和尾部未关联字节。逐请求比较完整分数和 expected Top10，`dense_id=index+1`，采用稳定降序排序。

| case | cosine | relative L2 | 完整分数 max abs | Top10 max abs | ID 与顺序 |
|---|---:|---:|---:|---:|---|
| 0000 | 0.9999999784421971 | 0.00020944930563095084 | 0.0000984780490398407 | 0.00006857732772824754 | 全部一致 |
| 0001 | 0.9999999581557469 | 0.00028990405981721125 | 0.00011455267667770386 | 0.00009416064071654517 | 全部一致 |
| 0002 | 0.999999952276987 | 0.0003110954810913381 | 0.00011995434761047363 | 0.00007459005737309177 | 全部一致 |
| 0003 | 0.9999999222346481 | 0.00039497795896009555 | 0.00013775378465652466 | 0.00008478004074097267 | 全部一致 |
| 0004 | 0.9999999058732865 | 0.0004473780329645363 | 0.00021588802337646484 | 0.00009595939636231066 | 全部一致 |

五例通过全部固定门槛：finite、cosine≥0.9999、relative L2≤0.001、完整分数 max abs≤0.001；50 个 Top10 项的 ID 和顺序精确一致，score atol=0.001、rtol=0。结果见 `original-mode0-correctness.json`，绑定 result、二进制与 CPU golden 的 SHA。此结果只覆盖五条给定样例，尚不代表压力或性能验收。

另从原始与 GE 打包 ONNX 直接读取并逐项比较：3137 个节点的序列、属性与连线 hash 一致，400 个 initializer 的名称、dtype、shape 与完整内容 hash 一致，IO、opset 14、稀疏权重和函数定义一致；两图均保留 30 个 Slice、0 个 Range。打包 ONNX SHA 为 `5e641362d99c9a1afb6dfce70f47b4ac07f1a7c88272fdfc791ea04ae84f4822`。`graph-integrity.json` 留存两图全部签名，`verify_graph_integrity.py` 可重新执行；metadata 和外部权重存储方式差异不会冒充计算改写。

## 独立验证器与后续验收

远端 `verify_outputs.py` 支持五例 NPZ、逐请求输出 manifest 与二进制、性能 events 三种离线验证。固定阈值写入验证器，不接受候选配置覆盖。自测覆盖正常 golden、FP16 输出错误、NaN、完整向量超差、相同 shape 错 case、Top10 错序，以及 100/1000 请求最近秩 p99，八项均通过。

`test_acceptance.py` 另以 CPU 合成 events/输出执行 15 项对抗验证：正常重放、case 关联错误、同 shape 错分数、重复/缺失请求、二进制覆盖/额外字节、窗口和排空统计、D2H 不完整、失败请求、NaN 分项、闭环请求重叠、缺预热、cohort 冒充窗口完成数量、统计不足。全部通过，结果为 `verifier-adversarial-tests.json`。这些合成数据仅验证验收器，绝不作为 NPU 正确性或性能证据。

压力验证按 `(run_id,request_id)` 关联 events 和 manifest，核对 instance、case、phase、真实输出 shape/dtype/字节/偏移及完整分数和 Top10；拒绝缺失、重复、串 case、二进制重叠或未引用内容。最终契约按设计要求至少 2000 请求，且每实例每 case 至少 20 次；较小的五例正确性运行不会被记成最终压力通过。

性能验证要求每请求实际 H2D、完整 40,448 字节 D2H、同一单调时钟的开始与完成、闭环每实例不重叠、每实例每 case 至少 10 次含 D2H 预热。独立从窗口内成功完成数和实际秒数计算 QPS，从原始延迟用 `ceil(0.99*N)-1` 计算 p99，分别报告排空与 cohort 数。无 p99 上限时只报告观察值。正式基线/独立验收每轮至少 60 秒且窗口内至少 10000 次成功；不能因统计量不足放宽冻结门槛。

以上为独立验收器和固定契约。后续各节按真实证据列出新增通过项；没有用旧计时数据、历史 PASS 或候选生成的 expected 代替本轮运行。

## 计时实现冻结与基线准入

冻结的隔离 runner 二进制 SHA 为 `8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d`，补丁 SHA 为 `b0190bb626f9a808d66ff5d4c19d3df638ed34be5b87257c62e4c013021b0c15`。测试逐一独立核对八份源码和 binary 与冻结身份相同，在自己的 `acceptance/measurement-cpu-build` 重新配置和编译 CPU `runtime_contracts`，再执行 ctest，1/1 通过；运行前后源码和 binary 未变化。原始命令、返回码与 stdout 保存在 `measurement-independent-cpu.json` 和 `.log`。

独立审查确认 GE stream 同步后执行完整真实 D2H，再记录 Host 输出就绪；每实例每 case 10 次预热也走该路径。压力归档在请求计时完成后、该实例下一次执行之前同步完成；共享文件通过 mutex 串行化偏移，binary/manifest 写入及 flush 失败会使该请求和轮次失败。请求时钟从 dispatch 锁内获准发放开始，额外包括到首次 H2D 的短 Host 间隙，截止前发放的请求完成后排空；窗口成功计数、样本吞吐与统计充分性都按窗口内完成计算。p99 使用最近秩。

预审发现的两个统计问题已修复并加入 CPU 回归：排空请求不能凑够窗口样本门槛，`samples_per_second` 不能以 cohort 样本除窗口；受控时钟测试明确构造窗口完成 0、cohort 1、inflight 1、drain>0，结果统计不足且窗口样本吞吐为 0。多 worker 回归另核对相同 shape 的不同 case、全局采样顺序和请求 ID 关联。

Host 输出保持每实例独占，当前每请求仍创建/分配 HostTensor data 并将分配包含在延迟。主控明确按开发计划保留这一基线行为；Host 缓冲预分配/容量复用若实施，必须单列优化点，不能声称此次计时修正已经实现该优化。

`plans/base-pressure.json` 与三轮 base 计划已独立验证输入二进制全部 17,304 字节和 CPU manifest 一致，dtype、shape、每 case 输出容量和图路径正确；设备统一为 10，GE mode=0、原始 FP32 图、未限核、1 实例、无 profiling。压力计划为至少 2000 请求；正式三轮分别 seed 2027/2028/2029、至少 60 秒且 10000 请求、最多 300 秒。验证结果分别保存为 `base-*-input-identity.json`。计时和计划获准进入主控设备实验；此准入不等于真实压力/性能通过。

## 设备 10 基线压力正确性

`base-pressure` 使用冻结 binary，在设备 10、原始 mode=0 FP32 图、全核单实例执行。独立从 2000 个 measure event 关联 2000 份完整输出，每请求均为 FP32 `[1,10112]`；输出偏移覆盖整个二进制，无缺失、重复、重叠或未引用内容。case 0000～0004 分别出现 430、421、394、422、333 次，超过每 case 20 次门槛。

全部 2000 请求完整分数与 Top10 通过，0 请求失败。最坏完整 max abs 为 `0.00021588802337646484`，relative L2 为 `0.0004473780329645365`，Top10 max abs 为 `9.595939636231066e-5`。结果保存在 `base-pressure-correctness.json`，绑定 events/manifest/raw 输出和 CPU golden 哈希。

额外逐 event 检查 50 次含 D2H 预热、每请求实际输入字节、完整 40,448 字节 D2H、时序、闭环无重叠及发布统计值，均通过；诊断结果为 `base-pressure-timing-diagnostic.json`。该运行启用完整输出落盘，只用于压力正确性和采集路径验证，其吞吐和延迟不进入正式性能基线或收益比较。

## 第一组三轮正式性能基线

统一设备 10、冻结测量 binary、原始图与未限核单实例，关闭压力输出归档。测试逐 event 重新计算窗口完成率和最近秩，并与 runtime 发布值逐一比较；所有 31,312 个测量请求均成功，50 次含 D2H 预热/轮、输入输出字节、时序、闭环无重叠及样本量均通过。

| 轮次 | seed | 窗口秒数 | 成功请求 | QPS | p99 / ms |
|---|---:|---:|---:|---:|---:|
| base-round1 | 2027 | 62.586816903 | 10000 | 159.77805702275083 | 6.730881 |
| base-round2 | 2028 | 60.004205712 | 10520 | 175.32104416967806 | 7.111864 |
| base-round3 | 2029 | 60.005052061 | 10792 | 179.85152298558222 | 7.153775 |

三轮 QPS 中位数为 **175.32104416967806**；合并全部请求的 p50 为 **5.660719 ms**，p99 为 **6.819411 ms**。p99 未取三轮平均；每 case 数量、分位数、分项和 10 秒窗口均保留在 `baseline-performance-summary.json`。

轮间相对极差 `(max-min)/median` 为 **11.449547347781035%**，存在明显波动。后续正式收益必须用冻结候选与原基线交错独立复测，不能挑最高单轮，也不能在差异尚未超过噪声时声称确定提升。用户未设延迟 SLA，此处只报告实测尾延迟。

设备资源按 NPU 5/chip 0/Phy-ID 10/BDF `0000:89:00.0` 单独审查，不把其他芯片进程视为本设备干扰。round2、round3 的 npu-smi 进程均与 host `docker top` 的本任务父子链对应；round3 还直接读取 host `/proc/<pid>/status` 的 NSpid，将 host 3020236 映射到 container 2002152。round1 采样只显示单个 host PID 2972475，但未及时取得直接 PID 映射；保留主控前后空闲门禁及单进程的边界证据，明确这一历史身份局限，不补造直接映射。见 `host-pid-witnesses.jsonl` 和汇总中的 resource_evidence。第一组基线成立，最终收益仍待交错复验。

## Add + LayerNorm 融合候选离线审查

独立读取 `candidates/add_layernorm_pass.cpp` 与对应测试，再独立执行 `candidates/test_add_layernorm_pass`，退出码 0。有效模式融合，以及错误 epsilon、非末轴、残差外部消费者、统计输出被使用、控制边、混合输入 dtype 六种应拒绝场景，共七项测试通过。只加载 GE 图/schema 库，没有调用 GEInitialize、aclInit 或设备任务；证据为 `fusion-cpu-tests.json` 和 `fusion-cpu-tests.log`。

直接核对当前 CANN 9.0.0 的 `opp/built-in/op_proto/inc/nn_norm_ops.h`：AddLayerNorm 必选输入为 x1、x2、gamma、beta，可选 bias 不连接；四个输出依次为 y、mean、rstd、x。候选将原输出 y 的消费者转接到输出 0，原统计输出无人消费才允许替换；残差 Add 也必须只有该 LayerNorm 一名消费者。epsilon 明确写回原 FP32 1e-12；`additional_output` 默认 false，输出 x 无消费者，符合当前模式。源码另外限制 BGE encoder 节点名、FP32、三维末维 1024、gamma/beta `[1024]`、末轴与无控制边。未发现阻断试验的问题，安装 schema 摘录保存为 `installed-add-layernorm-schema.txt`。

当次插件 SHA-256 为 `8767095f04348ed6097c769d71909421409397cedef534c4c7650d61df6cfff5`；源码 SHA-256 为 `e4d2894adbd1a0684f240581d79a1d88f0d2f5bdcaed29477c31106834e2ceb5`。七项 CPU 图测试只证明匹配/拒绝、接线与属性契约；实际替换数量、目标 CANN 编译和完整分数还必须由主控的设备试验提供，不能据此宣称吞吐收益或设备精度通过。

## FP16 Cube 候选正确性

`fp16-correctness` 的选项为 GE mode=0、`precision_mode=cube_fp16in_fp32out`，保留原始图和原始输入输出 dtype。独立核对计划与 packed 输入字节均与原参考一致，设备为 10；host PID 3096396/container PID 2002863 的运行身份已留见证。主控/SE 另保留最终 GE 图来确认实际内部精度，不能由 FP32 输出推断内部仍为 FP32。

独立验证全部 2000 个 captured 请求，五 case 数量仍为 430、421、394、422、333。0 错误、遗漏、重复或错 case，输出全部 FP32 `[1,10112]` 且有限，完整分数及 Top10 均通过既定门槛，无需放宽阈值。

| case | 最低 cosine | 最大 relative L2 | 完整分数 max abs | Top10 max abs |
|---|---:|---:|---:|---:|
| 0000 | 0.999999944950109 | 0.00033302323619808887 | 0.00013321638107299805 | 0.00005981544494626512 |
| 0001 | 0.999999920212007 | 0.0004000642342119006 | 0.00017148256301879883 | 0.00013367852020262916 |
| 0002 | 0.9999998908884228 | 0.00047207150972162907 | 0.00018070638179779053 | 0.00007517481803892556 |
| 0003 | 0.9999999010123676 | 0.0004452795796205249 | 0.000166378915309906 | 0.00010105210876465431 |
| 0004 | 0.9999998929720024 | 0.0004639440336980705 | 0.00024280697107315063 | 0.00010504179382325729 |

全部请求的 Top10 ID 和顺序精确一致。证据为 `fp16-pressure-correctness.json`，含每 case 最坏指标、首请求 Top10、raw 输出/manifest/events/golden 哈希；`fp16-input-identity.json` 绑定计划和输入。此项仅正确性通过，候选的性能收益和接纳状态仍待纯性能搜索与正式独立复验。

随后 `fp16-search` 关闭 capture/profiling，独立验证搜索口径的 30.001687769 秒、4719 次成功请求，0 错误，QPS **157.29115096238104**、p50 **6.160764 ms**、p99 **8.852352 ms**。device 10 的资源采样仅出现直接映射的本任务 host PID 3182933，无外部进程观测。相对首组三轮基线中位数观察约 -10.28% QPS，未显示收益，且这一差值接近原基线 11.45% 轮间范围；一轮搜索不足以宣布确定退化或正式接纳。证据为 `fp16-search-statistics.json`，含独立复算、每 case 和 10 秒窗口。搜索通过表示测量有效，不表示优化收益成立。

## FP32 AddLayerNorm 候选设备正确性

`fusion-correctness` 使用同一冻结 runner、原始图、mode=0 和 `must_keep_origin_dtype`，仅通过本任务 job 的 `LD_PRELOAD` 加载已审插件。独立核对 job argv 中插件路径和真实 SHA，与七项 CPU 测试时身份相同；所有材料、结果、计划、events 的 run ID 和配置也相符，见 `completed-run-identities.json`。

2000 个请求的完整 FP32 scores 与 Top10 全部通过，0 失败、遗漏、重复或错 case。最坏 max abs 为 **0.0002160295844078064**，relative L2 为 **0.000447579821078645**，最低 cosine 为 **0.9999999057935363**，Top10 max abs 为 **9.601900100708605e-5**。每 case 仍为 430/421/394/422/333 次，原始输出和独立逐 case 指标保存在 `fusion-pressure-correctness.json`。host PID 3258217/container PID 2005676 运行见证已归档。

SE 已另外检查真实 pass 日志和 GE 图，确认 48 处 AddLayerNorm 生效，kernel 为 `AddLayerNorm_normal_input_all_fp32_high_performance`；正确性通过并不自动意味着性能收益。编译阶段 Python signal 的 TypeError 被主控单独保留为诊断，根因尚未确定，不能仅凭日志将其归因于某个子进程或插件继承。

后续 `fusion-search` 虽 runtime 返回 0 并产出请求记录，但外层 `jobs/fusion-search/status.json` 明确为 `reason=background_children`、`exit_code=125`、`task_returncode=0`，保留两个未及时退出的 PID，最终由任务管理清理进程组。测试直接读取该状态，判定整个搜索**失败**，不将其 runtime 的 QPS 作为有效性能结果或接纳依据。设备随后实际释放不抵消生命周期失败。原版本证据保留，受限加载方案仅作为新候选验证修复假设；尚未证明根因。

独立聚合和身份验证工具已强制检查外层任务 `finished/exit0/process_group_stopped`，不能仅凭 runtime 成功通过。之前 base、Cube FP16 和 fusion-correctness 各轮外层均为退出 0，因此已经建立的正确性与初始基线通过项不受此次失败影响。三轮基线的前后真实设备空闲 gate 与任务时间、独立 task ID 已另行审查，保存为 `baseline-lifecycle.json`；没有把 helper 仅表示 OS 进程管理的 `device_release_checked=false` 改写为硬件释放证据。

## 自动混合精度候选正确性

`mixed-correctness` 使用 `allow_mix_precision_fp16`，外层任务和 runtime 均退出 0。独立核验全部 2000 份完整分数与业务 Top10，0 失败、缺失、重复或错 case；最低 cosine 为 **0.9999997969163102**，最大 relative L2 为 **0.0006383212320449013**，完整分数 max abs 为 **0.0004622936248779297**，Top10 max abs 为 **0.0004623124999999506**。所有 Top10 ID 和顺序精确一致，五 case 均超出 20 次覆盖要求，无需局部回退或放宽阈值。完整逐 case 指标见 `mixed-pressure-correctness.json`。

`mixed-search` 独立按 30 秒/2000 请求搜索契约复算通过，外层任务退出 0：窗口 30.000305847 秒、4579 次成功，QPS **152.63177726762726**、p50 **6.502108 ms**、p99 **7.450638 ms**，无错误。资源采样仅映射本任务 host PID 3507537，没有本设备外部进程观测。相对初始基线中位数观察约 -12.94% QPS，单轮搜索未显示收益，主控暂不接纳；没有运行候选三轮正式验收。证据为 `mixed-search-statistics.json`。

## 融合 V2 作用域修正的独立准入

新候选插件 SHA 为 `76ea9fb0f1800369b023e22d71298e508dcfaa6f69f34bcf952af4a1b5fa48ed`。V2 包含原融合源码，另外只在实际 basename 为 `ge_runtime` 的进程启动时，从继承的 `LD_PRELOAD` 精确剔除自身的 canonical path；保留其他库，运行中的插件映射不卸载。普通 Python/其他可执行程序不修改原环境。主控的 launcher 只对 native Popen 增加 `plan.runtime_env`，Python launcher 自身不预载插件；主 C++ binary 未变化。这里是在验证修复假设，并未证明旧残留进程的根因。

测试在自己的 `acceptance/v2-independent-build` 复制冻结源码并重新构建，独立复跑七项图契约、十一项路径/分隔符/别名过滤契约和三项真实 preload/child 集成，全部通过。另用真正冻结的 76ea 插件再执行三项集成，确认 native 保持自身映射、后续 Python child 不映射自身且保留 tcmalloc、普通程序保留原始 preload。结果为 `v2-independent-cpu-validation.json`，没有设备调用。

独立审查还发现旧 V1 与初版 V2 计划会覆盖容器默认 tcmalloc preload。主控已将所有未启动的 V2 计划修正为 `/usr/lib64/libtcmalloc.so.4.5.16:<冻结V2插件绝对路径>`；测试复核五份计划，见 `v2-plan-review.json`，因此新候选保持基线 allocator。V1 的 allocator 差异作为另一个可比性限制保留。V2 获得设备试验准入，其真实融合、2000 请求正确性、进程回收与性能仍待实际证据。

随后 `fusion-v2-correctness` 的外层任务/runtime 均退出 0，进程组已停止；runtime 日志独立记录 **48** 处替换、继承 preload 清除自身标记，runtime/task 日志的 TypeError 均为 **0**。完整 2000 份输出独立通过，最大 abs **0.0002160295844078064**、relative L2 **0.000447579821078645**、最低 cosine **0.9999999057935363**、Top10 max abs **9.601900100708605e-5**，ID/顺序全部一致，无缺失、重复或失败。raw 输出 SHA 与 V1 相同，说明本次五样例数值逐 bit 一致。见 `fusion-v2-pressure-correctness.json` 和 `fusion-v2-runtime-diagnostics.json`。

V2 在这次设备正确性任务中未复现旧版本的 signal TypeError 与后台进程失败，不据此宣称旧问题根因已经确定。此次 host 只读探测到达时任务已结束，因此没有补造直接 PID 映射；前后真实空闲 gate 和外层成功证据保留。后续实验开启本轮限时、每十秒只读的本任务 host PID 见证采集，任务结束时由测试停止并保留退出状态。

`fusion-v2-search` 的外层任务/runtime 均为退出 0，独立统计 30.003212442 秒、5154 成功请求，QPS **171.78160538520112**、p50 **5.881082 ms**、p99 **6.137444 ms**。相对初始基线中位数 QPS 观察约 -2.02%，差异处于基线 11.45% 轮间波动内，未显示明确吞吐收益。尾延迟观察改善仍是一轮搜索结果，不能据此作为最终接纳。资源样本仅为直接映射的本任务 host PID 3604415；证据 `fusion-v2-search-statistics.json`。

## 双实例压力与交付入口

`multi2-correctness` 为原始 FP32 图、GE mode=0、未限核、同逻辑设备 10 的两个独立实例。全体 **2001** 个 measure cohort 请求均完成独立逐请求分数与 Top10 验证，包括窗口截止后的一个尾部请求；没有误要求恰好 2000，也没有把该尾部计入窗口吞吐。

0 失败、遗漏、重复或错 case，十个 instance×case 组合最少出现 159 次，均满足至少 20 次要求。最大完整 abs 为 **0.00021588802337646484**、relative L2 为 **0.0004473780329645365**、Top10 abs 为 **9.595939636231066e-5**，ID 和顺序全一致。证据为 `multi2-pressure-correctness.json`。外层退出 0、进程组停止和前后真实设备空闲门禁在 `completed-candidate-lifecycle.json` 逐任务举证；同文件也包含混精与融合 V2 的已完成任务。

对双/四实例各 correctness/search/三轮 plan 共十份作了独立离线审查，确认原始 graph、packed 输入、scope、mode、dtype、默认核数和每实例五 case 预热不变，仅实例数改变。四实例在此项审查中只是准备，不能写为已运行；见 `multi-instance-plan-review.json`。

新增 `models/bge/replay.py` 的 prepare-only CLI 已独立代码审查，并在 canonical 仓库执行八项 CPU 回归全部通过。另用真实 config、现有小型外部权重 ONNX 的 IO 和真实 NPY 准备一次，产出 17,304 字节输入逐张量与独立 CPU manifest 一致，runtime 子目录未创建、没有设备调用；preload 只追加写到计划，保留默认 allocator，不修改 Python 进程环境；旧目录拒绝覆盖。见 `replay-cli-independent-tests.json/.log`、`replay-cli-real-preparation.json` 和 `cli-real-input-identity.json`。

计时修正已集成 canonical 仓库，测试实际重新计算 canonical 与隔离 runner 的二进制 SHA，均为冻结的 **8efca50a...630d**，逐字节一致，见 `canonical-binary-identity.json`。CLI 与 README 改动没有改变 C++ binary，这个 CPU 交付准入没有被冒充为另一次 NPU 数值或性能测试。

`multi2-search` 独立搜索统计通过：窗口 **30.00004558 秒**，窗口内 **8097** 成功，另有 **1** 个尾部完成，共 cohort **8098**；QPS **269.89958993255635** 使用 8097 除窗口计算。p50 为 **7.409078 ms**、p99 为 **7.914773 ms**，外层任务退出 0、进程组清理正常，资源样本仅映射本任务 host PID 3695741。相对初始基线中位数观察约 **+53.95%** 吞吐，超过先前 11.45% 轮间范围；p99 同时观察增加约 16.1%。这是一轮搜索，正式接纳仍需冻结配置后独立三轮及交错基线复验。见 `multi2-search-statistics.json`。

## 四实例压力正确性

`multi4-correctness` 已执行，独立核验全部 **2003** 个 measure cohort 请求（窗口内 2000，另 3 个尾部），没有把尾部遗漏或混计窗口吞吐。0 失败、重复、缺失或错 case，二十个 instance×case 组合最少覆盖 **69** 次，均超过至少 20 次要求，包含相同 shape 不同内容的 case 0002/0004。

所有完整 FP32 scores 与 Top10 通过，worst max abs 为 **0.00021588802337646484**、relative L2 为 **0.0004473780329645365**、Top10 abs 为 **9.595939636231066e-5**，ID/顺序全部一致。外层任务退出 0、设备实际释放由主控门禁留证；数值与请求关联证据为 `multi4-pressure-correctness.json`。压力吞吐不进入性能比较；随后纯性能搜索的独立结果如下。


`multi4-search` 的独立统计为 30.001677326 秒、11,107 个窗口成功请求，另有 3 个尾部，共 cohort 11,110。QPS 为 **370.21263442409173**，p50 **10.791432 ms**、p99 **11.448209 ms**；外层任务退出 0，资源样本仅映射本任务 host PID 3774043。相对初始基线中位数观察 +111.16% QPS，相对双实例搜索观察 +37.17%，但尾延迟同时增加；一轮搜索不能代替最终三轮交错复验。见 `multi4-search-statistics.json`。

## 八实例压力与搜索

`multi8-correctness` 独立核验全部 **2007** 个 measure cohort 请求，包含窗口内 2000 和 7 个尾部；40 个 instance×case 组合最少覆盖 **36** 次，均超过至少 20 次门槛。完整 FP32 分数、finite、Top10 ID、顺序和分数全部通过，无缺失、重复或失败。最大完整 abs 为 **0.00021588802337646484**、relative L2 **0.0004473780329645365**、Top10 abs **9.595939636231066e-5**。证据为 `multi8-pressure-correctness.json`，没有因多实例放宽阈值。

`multi8-search` 独立统计为 **30.002169569 秒**，窗口内 **13,841** 次成功，另有 **7** 个尾部，共 cohort **13,848**；QPS **461.3333035188673**，p50 **17.328750 ms**、p99 **18.243470 ms**。外层退出 0，资源样本的 host PID 3897293 与任务进程链直接对应，无本设备外部进程观测。相对初始基线中位数观察 +163.14% QPS，相对四实例搜索 +24.61%；该值仍是一轮搜索，最终接纳需三轮交错对照、压力复测与稳定性检查。见 `multi8-search-statistics.json`。

已只读审查 16 实例及 8/16 实例半核六份计划：原始 graph/IO、五 case 的全部 40 个输入张量字节、mode=0、FP32、device 10、计时边界及无 profiling/dump/插件均一致。16 实例使用 800 次含 D2H 预热和 5000 次压力请求；半核计划只新增 `aicore_num=12|24`。计划通过不代表设备执行或实际 instance×case 覆盖通过，后续仍必须逐请求验真。见 `multi16-half-plan-review.json`。


## 十六实例压力正确性与生命周期

`multi16-correctness` 独立验证全部 **5015** 个 measure cohort 请求，窗口内 5000 加 15 个尾部，80 个 instance×case 组合最少 **36** 次；全部完整 FP32 分数、finite、Top10 ID/顺序/分数通过，0 缺失、重复、失败或错 case。最坏完整 abs **0.00021588802337646484**、relative L2 **0.0004473780329645365**、Top10 abs **9.595939636231066e-5**，阈值不变。证据为 `multi16-pressure-correctness.json`。纯性能收益仍需关闭 capture 后实测，不能使用压力吞吐代替。

对 multi2/4/8 的 correctness/search 和 multi16-correctness 共七个已结束进程周期独立审查：原图、输入、frozen binary、计划与事件 run ID 身份一致；外层退出 0、进程组已停止，且每次前后真实 device10 空闲 gate 与任务时间绑定。见 `multi-instance-run-identities.json`、`multi-instance-lifecycle.json`。

四组多实例压力另逐 event 检查含 D2H 的每实例每 case 预热、实际完整 H2D/40,448 字节 D2H、闭环时序、窗口/尾部计数与发布统计，均通过；证据为各 `multiN-correctness-timing-diagnostic.json`。这些只证明压力采集与计时契约，未进入正式性能接纳。


`multi16-search` **失败**，与前述压力正确性通过分开保留：native `returncode=-9`，外层 `exit_code=125 / task_returncode=247 / reason=background_children`，没有 runtime/performance 结果，不能填入有效搜索 QPS。主控已停止提高并发。直接 PID 见证映射主进程 host 4034834 → container 2029055，另有同父子链 ge_runtime 子进程 4037417 → 2029109；同次 11 个资源样本中主进程 RSS 最高 25,338,176 KiB。

测试随后只读采集 cgroup、Docker 限额和内核证据，保存于 `multi16-search-sigkill-container-evidence.json`、`multi16-search-sigkill-host-evidence.json`。当前为 cgroup v1，所以 `memory.events` 不存在；`memory.failcnt=0`，`memory.oom_control` 的 oom_kill=0，memory limit 为 9223372036854710272，Docker Memory/MemorySwap/MemoryReservation 均为 0。宿主 `/proc/vmstat` 的 oom_kill=0；可读 dmesg 中未见匹配本任务 PID 的 OOM 记录，仅保留同任务 child 的 Ascend CQ 诊断。Docker `OOMKilled=false` 是容器状态，不是单进程原因判据。上述为事后快照，不能把 SIGKILL 直接断言为 OOM；目前根因未定。


## 外部占用暂停与观察器退出

`half8-correctness-before.json` 在 **03:58:14 UTC** 返回外部 host PID **4088730**，HBM **136MB**。测试另于 **04:00:54 UTC** 在容器内只读执行同一物理5/chip0的 proc-mem 查询，返回码 0，仍看到相同 PID/136MB；未初始化模型、未启动设备任务。`jobs/half8-correctness` 不存在，说明门禁在运行前生效。

宿主 `/proc` 和 `ps` 的必要身份字段将该进程映射为 NSpid 4088730 → 325016，启动时间为本地 11:57:45，进程为 python，指定 `device-10.json` 配置及 dspark-validation-holder/torch_npu 参数 token 匹配；它不属于本任务 ge_runtime/launcher 父子链。未归档他人的完整命令参数。初次在宿主直接执行 npu-smi 的 DCMI 失败原样保留，实际占用结论使用容器内成功查询及主控门禁，不能把失败查询解释为空闲。见 `device10-external-holder-20261026.json`。

主控要求暂停后，测试写入停止标记，有界只读 PID observer session **5943** 响应退出，实际 exit code **0**，此前轮换的 session 14345 也已退出 0。见 `host-observer-lifecycle.jsonl`。没有遗留观察任务，也没有启动半核、组合优化或最终验收设备作业。


## 阶段性交付文档独立复核

2026-10-26 04:07 UTC 交接审查通过。只读核对 `acceptance.md`、`progress.md` 和 OPT-001～009 全部记录：六组有效搜索的 QPS/p99、四组多实例完整输出请求数及最小组合覆盖、初始三轮基线 31,312 请求/175.32104416967806 中位 QPS/6.819411 ms 合并 p99 均与独立原始证据一致。融合 V1 与十六实例 search 的外层失败未进入有效性能结果，十六实例 SIGKILL 根因未定、半核未启动、八实例未最终接纳等状态表述准确。九条优化记录均包含固定七节，文档内本地 Markdown 链接均能解析。

本地归档 delivery-manifest 的 **475** 项按声明对 gzip 解压后的原字节逐项复核，字节数与 SHA-256 全部匹配；11 份大输出按清单留在远端，不把归档摘要冒充原输出。此处只核查文档和现有材料，没有重启设备任务或只读观察器。

审查指出的进度措辞已经修正并再次读取确认：V2 改为“两次设备运行未复现 V1 清理失败，旧问题根因未证实”，没有把观察不复现写成已确认因果修复；OPT-007/008 中与已完成实验矛盾的旧时态也已移除。阶段性报告审查通过只表示记录与证据一致，**不表示最终性能验收通过**。最终候选三轮交错复测、10 分钟稳定性与前后压力复测仍未执行，等待资源恢复后按冻结契约继续。
