# BGE 本轮优化结果与恢复交接

日期：2026-10-26。**已有结果已归档；按用户选择等待设备释放，最终性能未验收。** 原始模型GE mode=0正确性、完整搬运基线及有限候选搜索已完成；最佳有效搜索点是FP32全核八实例，但尚未执行冻结后的三轮独立复测和10分钟稳定性，不能将搜索值作为正式交付收益。

## 执行范围、环境与身份

SSH99，容器hejun-cann9-bge；项目根`/export/home/ext.hejun23/workspace/bge`，canonical仓库`/export/home/ext.hejun23/workspace/bge/ascend-ge-adapters`。本轮路线原ONNX→GE Session，graph_run_mode=0，不改写Slice。最初五例正确性在逻辑设备2，其后出现外部xllm；用户授权逻辑设备10并确认16卡等性能，全部正式基线和搜索统一逻辑10/NPU5/Chip0/PCI0000:89:00.0。

CANN9.0.0、Ascend910_9382、driver25.5.t6.b001、firmware7.8.0.5.216；父代码441b245fc18ee2a78156becf80488ee16ad3474d，测量binary SHA256=8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d。8个测量文件和prepare-only CLI已集成canonical，二进制与隔离runner逐bit相同。开发及独立C++契约1/1、CLI8/8通过；README明确旧run/bench是历史冒烟且缺D2H，不能复现本轮口径。当前代码改动未commit/push。

原始model SHA256=2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0；sample=66571de2fa8c6a97e52a196a957801ab09b429751ba9cfb94712bb32c660b194；expected=4765cc2b9dac9d2f523f9a684a81082d3b943caaff1fd2c4db90cfdf1b375384。重新生成的独立ORT CPU golden=a59f406eb5782926181963e94cb1f87e1728621dedc3d613f88a7317d8213f35。原图与外置权重打包图3137节点、400权重内容和opset/IO逐项一致，均30Slice/0Range。

只覆盖给定五条输入B1、S=6/9/8/13/8、L128、stat宽度20；七项INT64输入与stat_ratio FP32，输出FP32 scores[1,10112]。全核通过不设置aicore_num实现；静态SoC配置24 AIC/48 AIV。源权重FP32；降精度及融合仅作为隔离编译候选，不替换原模型。具体内部精度与敏感算子见各OPT及[SE报告](analysis/precision-fusion-plan.md)。

## 固定case与多case随机分别报告

按用户新增约定，调优可固定单case，最终交付并列两场景；以下历史OPT表和已有性能数值均属于random-five，保持原证据身份。固定case默认选择原顺序首个中位长度S8样例case-0002，实际输入B1/S8/L128/stat20，每请求重复完全相同内容，选择发生在测量前。详见[双场景契约与输入哈希](evidence/20261026/workload-contract-v2.json)。

|场景|同场景原FP32单实例基线QPS / p99|当前候选QPS / p99|最终交付验收|
|---|---|---|---|
|固定case-0002|未运行|未运行|未验收，待设备释放后建基线及独立复测|
|五例随机random-five|175.321044 / 6.819411ms（三轮）|461.333304 / 18.243470ms（八实例单轮搜索）|三轮候选复测与10分钟稳定性仍未执行|

本地prepare-only入口已补`--case-id`并通过12个CPU回归；实际一例池与默认五例池已核对。新版计划/性能/身份验收工具另存版本并通过13项独立CPU回归，全五例压力要求保持，见[审查报告](analysis/dual-workload-review.md)。SSH99当前不可达，远端集成/新版文档同步及固定case设备测量尚未完成，见[补丁身份](evidence/20261026/dual-workload-replay-identity.json)。

两场景分别计算相对本场景基线的收益，不能用固定候选对随机基线算提升，也不能用随机日志的单case子集替代固定case实测。计时仍包含完整H2D+GE同步+全量D2H，完整正确性和最终请求关联压力仍覆盖全部五例。双场景复测沿用原预算，重新估算后预留验收；没有新增设备运行或性能值。

## 正确性与可信性能基线

独立参考ORT1.26.0/NumPy1.26.4/CPU provider、2 intra/1 inter。全部完整输出要求finite、cosine≥0.9999、relative L2≤0.001、max abs≤0.001。Top10按dense_id=下标+1比较，ID和顺序精确相同；分数atol0.001/rtol0。并列分数按较小下标优先。阈值和expected未因候选调整。

原始mode0五例及新测量路径2000请求压力均PASS：最坏完整max abs=0.00021588802337646484，relative L2=0.0004473780329645365，min cosine=0.9999999058732865；Top10最大误差9.595939636231066e-5，ID/顺序全部一致。详见[独立验收](analysis/optimization-acceptance.md)及[压力报告](evidence/20261026/acceptance/base-pressure-correctness.json)。

计时从请求获准发放到Host完整输出就绪，包含H2D、描述符设置、GE异步完成同步、完整40448字节D2H及本次Host输出分配；不含文件读取、tokenization、服务端排队、参考比对和采集落盘。五例等概率有放回、闭环每实例一个在途请求，各实例每case预热10次。正式基线每轮≥60秒且窗口成功≥10000，capture/profiling=false；QPS只计窗口内完成，尾部全部排空并纳入cohort延迟。p99使用nearest-rank，不平均各轮p99。

|指标|首个可信FP32全核单实例基线|当前八实例最佳有效搜索|用途限制|
|---|---|---|---|
|成功requests/s|三轮159.778057 / 175.321044 / 179.851523；中位数175.321044|461.333304|八实例仅30.002秒搜索，观察+163.14%，非正式交付收益|
|p99 ms|三轮合并31312请求6.819411|18.243470|不同容量点观察+167.52%；未设置延迟SLA|
|关联压力|2000请求PASS|2007 cohort PASS；40组合最少36次|完整分数和Top10逐请求核验|
|资源|单实例setup HBM约1.7GB|八实例setup HBM约10.7GB|非共享服务总资源承诺|
|最终三轮/10分钟稳定性|基线三轮已完成|未执行|被新外部占用阻塞，未验收|

基线轮间相对极差11.4495%；变化不足3%或未超过父/候选较大轮间波动时不判显著提升。八实例搜索相对四实例+24.61%，但仍需要冻结后独立序列和交错基线复测。正式性能均未混入完整输出归档；压力的QPS不作收益值。独立PID/实际设备前后门禁对已完成任务核查，无同期外部设备负载观测；锁只约束遵守相同锁的作业，不声称全机资源预约。

## 全部调优记录

单位为requests/s / ms；除明确标记的基线三轮外，候选值均为搜索。每行差值相对于本行指定父版本，吞吐为正表示提高、p99为正表示延迟增加。单轮搜索与三轮基线的差异仅作方向筛选，不作统计显著性或最终接纳。源图49 LayerNorm、24 Gelu及144 MatMul+bias本已被GE自动融合，不重复计入新增收益。

|编号|一句话内容|父版本：吞吐 / p99|候选：吞吐 / p99|观察变化|是否纳入最终配置|原因|
|---|---|---|---|---|---|---|
|[OPT-001](experiments/OPT-001.md)|Cube FP16输入、FP32输出|单实例 FP32 175.321 / 6.819|157.291 / 8.852352（搜索）|QPS -10.28%；p99 +29.81%|否|2000请求精度通过；吞吐观察下降，未证明收益|
|[OPT-002](experiments/OPT-002.md)|48处Add+LayerNorm融合V1|单实例 FP32 175.321 / 6.819|未测得（运行无效）|不适用|否|完整分数通过，但search外层125；性能无效，且allocator被替换|
|[OPT-003](experiments/OPT-003.md)|CANN自动FP16混精|单实例 FP32 175.321 / 6.819|152.632 / 7.450638（搜索）|QPS -12.94%；p99 +9.26%|否|2000精度通过，吞吐观察下降；低范数输入未扩展验收|
|[OPT-004](experiments/OPT-004.md)|融合V2限定native加载并保留allocator|单实例 FP32 175.321 / 6.819|171.782 / 6.137444（搜索）|QPS -2.02%；p99 -10.00%|否|48融合生效、2000精度/清理通过，吞吐差异在波动内|
|[OPT-005](experiments/OPT-005.md)|FP32全核双实例|单实例 FP32 175.321 / 6.819|269.900 / 7.914773（搜索）|QPS +53.95%；p99 +16.06%|否（未最终验收）|2001请求精度通过，搜索有效，继续更高并发|
|[OPT-006](experiments/OPT-006.md)|FP32全核四实例|单实例 FP32 175.321 / 6.819|370.213 / 11.448209（搜索）|QPS +111.16%；p99 +67.88%|否（未最终验收）|2003请求精度通过，搜索有效，八实例更高|
|[OPT-007](experiments/OPT-007.md)|FP32全核八实例|单实例 FP32 175.321 / 6.819|461.333 / 18.243470（搜索）|QPS +163.14%；p99 +167.52%|否（待设备后验收）|2007请求精度通过，当前最佳有效搜索点；三轮/10min未执行|
|[OPT-008](experiments/OPT-008.md)|FP32全核十六实例|单实例 FP32 175.321 / 6.819|未测得（运行无效）|不适用|否|5015请求精度通过；search native SIGKILL，外层125，无性能值|
|[OPT-009](experiments/OPT-009.md)|八实例半核12\|24|八实例全核搜索 461.333 / 18.243|未运行|不适用|否（未运行）|启动前发现外部占位进程，门禁拦截，无新配置或占用计费|

9个已分配OPT均有固定七节记录；包括无收益、失败、资源阻塞。原FP32基线仍为可信回退，当前没有通过全部必需项的最终优化配置；不会将各点百分比累加。

## 失败、资源阻塞与预算

OPT002融合V1数学正确但search outer125/background_children，原库保留；OPT004修正native-only加载并保留tcmalloc，在两次设备运行中未复现清理失败，但无吞吐收益，不纳入默认配置。

OPT008十六实例先通过5015个完整输出压力，随后search在图15/16编译后SIGKILL（native=-9），外层125并完成本任务子进程组清理，事后真实device10空闲。无可用性能结果。cgroup failcnt/oom_kill=0、宿主vmstat oom_kill=0、未找到同期OOM记录，**原因未定，不能写成已确认OOM**；不再提高并发。

11:58半核实验启动前device10出现外部PID4088730/约136MB。宿主ps确认11:57:45启动的torch_npu idle-context-and-allocation / dspark-validation-holder，路径属于wangyifan194的dspark-pipeline-validation资源占位，与本任务无关。门禁拒绝启动OPT009，没有job或模型执行，不终止外部进程。用户已明确选择“先归档已有结果，设备释放后再继续”。本轮停止发起设备任务；后续恢复先重新确认可独占设备，沿用现有证据与预算。新占用不使此前已完成测量自动失效。

已实际执行9/12组唯一配置，保守设备占用1616.679910/7200秒，余额5583.320090秒（含编译/加载、失败、清理和所有重复任务）。半核门禁失败不新增配置或设备费用。原始完整输出、最终GE dump和完整日志保留在实际实验根；必要身份、计划、事件、独立报告、补丁和清理证据归档[evidence/20261026](evidence/20261026/)，大输出的字节数/哈希/远端位置见delivery-manifest.json。

## 复现与恢复

在99容器canonical仓库中，模型data/build产物已存在。下例是保留兼容的五例随机准备入口，不查询或占用设备；使用新目录。固定case调优在集成新版入口后显式添加`--case-id case-0002`，并换用新的固定场景output-dir；不传该参数仍为五例随机：

```bash
cd /export/home/ext.hejun23/workspace/bge/ascend-ge-adapters
python3 models/bge/replay.py --prepare-only \
  --config models/bge/config.yaml --device 10 \
  --output-dir work/bge/replay/baseline-new --instances 1 --seed 2027 \
  --min-seconds 60 --max-seconds 300 --min-requests 10000
```

八实例候选将instances显式设8，其它相同；核数省略表示全核，不受YAML历史4|8影响。先核查实际物理设备、外部进程及本任务旧状态，确认可独占后使用受管入口：

```bash
python3 /export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026/tools/run_checked.py \
  /export/home/ext.hejun23/workspace/bge/ascend-ge-adapters/work/bge/replay/baseline-new/plan.json \
  --job baseline-new --timeout 650 \
  --binary /export/home/ext.hejun23/workspace/bge/ascend-ge-adapters/runtime/build/ge_runtime
```

恢复不重新计预算、不覆盖旧run、不跳过前置设备门禁。恢复后先补固定case原FP32单实例基线，再开展OPT009半核的固定case对照与全五例压力；有收益才考虑四分核，最多再一个最佳N/core上的融合V2组合，每个新组合先分编号并独立验数。最终冻结配置后使用seed2027/2028/2029进行三轮≥60秒且≥10000请求，与同口径单实例基线交错；专门完整输出压力每实例×case≥20，最终10分钟稳定性、≥3次加载/销毁及设备/锁释放全部通过，才更新最终接纳和收益。至少预留50分钟设备预算，不能缩短门槛换取完成状态。

本轮暂缺：半核/最佳并发融合组合、冻结后性能复验、10分钟稳定性。更大shape/生产输入分布、明确p99约束和极端数值边界不在现有五例验收范围。代码未commit/push，源模型和样例只读。
