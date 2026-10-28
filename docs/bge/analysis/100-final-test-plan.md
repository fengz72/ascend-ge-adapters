# 100 最终独立验收入口

离线入口已准备：[verify_100_delivery.py](../../../delivery/supporting-evidence/work/bge-opt-100/tester/verify_100_delivery.py)。它只读取已完成作业与冻结材料，不运行 runtime、不调用 NPU API。当前设备任务仍由主控串行执行；本文件不是最终验收结果。

本地 CPU 回归：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s work/bge-opt-100/tester -p test_delivery_audit.py -v
```

12 项通过，见 [delivery-audit-cpu.log](../../../delivery/supporting-evidence/work/bge-opt-100/tester/delivery-audit-cpu.log)。包括三轮吞吐中位数与原始延迟合并 p99、配置漂移拒绝、窗口排空与部分十秒桶、原 RSS/HBM 门槛、缺少首末资源、采样间隙的保留、趋势需审查以及持锁内 gate 和至少三个生命周期。

## 使用方式

将单独 tester 文件放入 100 的测试目录，在设备测量结束后的 CPU 窗口运行。以下名称为最终交接参数示例，不代表这些 run 已存在或已经完成。

```sh
bash "$ROOT/tools/env_run.sh" python3 "$ROOT/work/tester/verify_100_delivery.py" \
  --project-root "$ROOT" \
  --fixed-runs final-fixed-r1 final-fixed-r2 final-fixed-r3 \
  --random-runs final-random-r1 final-random-r2 final-random-r3 \
  --stability-run final-stability \
  --lifecycle-runs final-random-r1 final-random-r2 final-stability \
  --pci 0000:99:00.0 --npu-id 1 --chip-id 0 \
  --report "$ROOT/work/tester-final/delivery-audit.json"
```

`ROOT=/export/home/ext.hejun23/workspace/bge-opt-100`。默认调用已部署 `tools/checks` 的冻结输入、数值和 runtime 身份逻辑；可用 `--checks-dir` 或 `--identity` 指定交接的检查器/身份位置。大权重按未变更的文件 stat 身份缓存一次哈希，避免每轮重复读取；不能在正式性能窗口执行大文件哈希与输出比较。

## 执行与判定范围

- fixed/random 各三个不同正式轮次，每轮 ≥60 秒且 ≥10000 窗口内成功请求、每实例每 case ≥10 次预热；随机正式 seed 为 2027/2028/2029。固定池实际仅 case-0002，随机实际原五条。两场景分别报告各轮 QPS、p50/p99、10 秒窗口、QPS 中位数/相对极差和合并原始延迟 p99，不平均 p99。
- 六轮必须对应同一冻结 binary/graph/plugin/options/实例配置。稳定性及生命周期也绑定同一配置；图/精度变化后不能拿旧配置生命周期或资源结果补齐。
- 生命周期至少三个不同 task ID，均正常 exit、native/包装器为 0、进程组停止，前后两芯片 proc-mem 均 clear；必须覆盖混合五例。100 gate 位于持锁区间，按 `job.started ≤ before_gate ≤ after_gate ≤ job.finished` 检查。
- 随机稳定性 ≥600 秒，事件检查拒绝错误、超时、重复/缺失请求、未归档与不一致排空。完整 D2H 仍在计时中；稳定性未逐分数比较，不冒充压力精度验收。
- 资源首末各 120 秒中位数比较沿用原门槛：RSS 增量 ≤max(128MiB,初始5%)，HBM 增量 ≤max(64MB,初始1%)。资源没有达到旧脚本至少12条、首末各至少3条的要求或命令失败时拒绝；同时保留真实采样间隔、斜率、正负增量、Host PID 及完整序列供审查。

## 必须由测试角色结合证据完成的审查

原设计没有规定持续增长的斜率门槛、10 秒采样允许的最大间隔、吞吐下降的统计阈值。入口不新设这些阈值，也不把满足资源增量自动写成最终 PASS。数值门槛失败会返回非零并标失败；数值契约满足时报告 `passed=null` / `requires_trend_sampling_and_exclusivity_review`，退出 0 只表示原始数学检查完成。

测试角色仍须检查：资源是否形成平台期或持续增长、采样是否足以覆盖首末和中间过程、十秒吞吐是否持续下降、Host PID 是否属于本任务、运行中是否有外部负载。证据不够明确时写未验收，不能因为前后 gate 空闲就认定整个窗口独占。

此外，最终精度压力必须单独绑定该冻结配置的完整输出验收：至少2000请求、每实例每 case ≥20、request ID—输出关联、完整分数及 Top10 下标+1。正式性能与稳定性没有收集每一完整向量的持久化副本，不能替代此精度压力。最终报告还需核对初始同场景基线、全部 OPT、设备时长预算、失败/退化记录及交付配置，不能由本工具独立宣告整轮工作完成。

## 已冻结命名的最终对照入口

主控现已冻结推荐配置为原图 FP32、GE mode=0、八实例、每算子核上限 `12|24`、不加载融合插件。对照入口为 [verify_100_comparison.py](../../../delivery/supporting-evidence/work/bge-opt-100/tester/verify_100_comparison.py)，仅在收到全部已完成证据和 CPU 审查窗口后运行：

```sh
bash "$ROOT/tools/env_run.sh" python3 "$ROOT/work/tester/verify_100_comparison.py" \
  --project-root "$ROOT" --docs-root "$ROOT/docs/bge" \
  --delivery-doc "$ROOT/docs/bge/delivery-100.md" \
  --historical-budget "$ROOT/docs/bge/evidence/20261026/budget.json" \
  --pressure-run half8-pressure --stability-run final-best-stability \
  --best-fixed-r3 final-best-fixed-r3-retry1 \
  --report "$ROOT/work/tester-final/comparison.json"
```

它使用 `final-base-{fixed,random}-r{1..3}` 和 `final-best-{fixed,random}-r{1..3}`，并复用前面的冻结身份、原始事件、候选稳定性与生命周期检查。两边各自按三轮吞吐中位数和合并原始 p99 报告，每个场景单独计算收益。变化需达到3%且超过基线/候选轮间相对极差的较大值，才超过原先约定的实际波动门槛；这是约定的收益判据，不是统计置信区间。p99 变化另列，无硬上限，不自动拒绝吞吐提高同时延迟增加的候选。

本次候选固定第三轮原作业 `final-best-fixed-r3` 以SIGTERM失败，没有有效完整性能结果。`--best-fixed-r3` 显式选择同配置的 `final-best-fixed-r3-retry1` 作为待独立检查的第三轮，并同步替换有效作业门禁列表；省略参数仍沿用原名。失败原作业保留在全部job预算中，报告另列 `failed_jobs_charged`，不进入有效三轮统计，也不退还设备费用或配置名额。补测是否有效仍由原身份、输入和正式统计门槛检查决定，不能只按名称接纳。

最终压力直接重读 `half8-pressure` 的原始 manifest/events/完整输出，核对八实例同一冻结配置、至少2000请求、每 instance×case 至少20次，并绑定 runtime cohort 和连续全局请求 ID。12 个正式作业、压力作业及候选生命周期的双芯片前后 gate 均纳入检查。

配置数使用历史真实账本，不使用可复用 key 数代替：99 `config_count=9`（含未迁为可复用 key 的旧 V1 融合），100 `registry.known=8`、新增 `half8/fusion8-half/attention8-half` 三个真实配置，总计 **12/12**。旧 V1 的无效性能仍占已经发生的配置名额。设备费用从历史1616.67991秒加100每个完成作业UTC起止重新计算，包含失败与门禁拦截；核对当前 ledger、7200秒上限及缺状态/未完成任务。

交付文档核对 `OPT-001` 至 `OPT-011` 共11份固定七节记录，指定的最终报告每个OPT恰好一行七列，保留失败、负收益和不接纳项。本次通过 `--delivery-doc` 明确选择 `docs/bge/delivery-100.md`，保留99历史 `acceptance.md` 原样；省略参数时仍默认读取 `docs-root/acceptance.md`。结构检查通过后仍需人工逐项核对数字原始来源、场景/主机及是否接纳；结构完整不等于数值内容已经验收。

本地比较/预算/文档覆盖20项CPU回归通过，包含指定100交付文档且历史acceptance不变、失败原轮计费与显式补测选择的回归，见 [final-comparison-cpu.log](../../../delivery/supporting-evidence/work/bge-opt-100/tester/final-comparison-cpu.log)。连同原控制器、scope和稳定性逻辑，当前测试目录合计49项独立CPU回归通过，见 [all-independent-cpu.log](../../../delivery/supporting-evidence/work/bge-opt-100/tester/all-independent-cpu.log)。这些均为本地合成/只读材料测试，本次准备没有连接100或启动设备任务。
