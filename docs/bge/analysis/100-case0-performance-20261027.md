# BGE 最优配置固定 case-0000 性能复测

日期：2026-10-27。主机100、容器`hejun-cann9-bge`、项目根`/export/home/ext.hejun23/workspace/bge-opt-100`。本次只测当前已接纳配置在`case-0000`上的性能，没有引入新优化配置，也没有重写原始Slice。

## 配置与测量口径

原始FP32 ONNX图经GE Session执行，`graph_run_mode=0`，`precision_mode=must_keep_origin_dtype`，8个独立实例，每算子`aicore_num=12|24`，batch1、每实例单在途，不加载自定义融合插件。目标为逻辑设备2，即100的NPU1/Chip0、PCI`0000:99:00.0`；同卡Chip1、PCI`0000:9B:00.0`纳入锁和前后门禁。打包图SHA256为`5e641362d99c9a1afb6dfce70f47b4ac07f1a7c88272fdfc791ea04ae84f4822`，测量二进制SHA256为`8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d`，与[最终交付配置](../delivery-100.md)一致。

请求池**仅包含`case-0000`**：`input_ids`和`attention_mask`均为`[1,6]`，长期行为长128，统计宽20。八个输入张量每请求H2D合计3416字节，FP32 `scores[1,10112]`每请求D2H 40448字节。三轮输入二进制SHA256均为`6a6fe4cbd2e443001ed12a34bfca48483b3d7dfa7c0c02c11361f30db501d5c7`，执行事件中的每个测量请求均标记`case-0000`。每实例预热10次；正式窗口每轮至少60秒且至少10000成功请求，闭环计时包含H2D、描述符设置、GE完成同步与全量D2H，不包含文件读取、tokenization或服务排队。无profiling、无输出capture；精度引用此前五例全量压力验证，本次不声称新增逐分数校验。

## 三轮结果

|作业|seed|窗口秒数|窗口内成功请求|QPS|p50 ms|p99 ms|错误|
|---|---:|---:|---:|---:|---:|---:|---:|
|`case0-best-r1-20261027`|2027|60.003588|34944|582.365|13.741|14.210|0|
|`case0-best-r2-20261027`|2028|60.001360|35184|586.387|13.643|14.119|0|
|`case0-best-r3-20261027`|2029|60.000060|36008|600.133|13.329|13.806|0|

**QPS三轮中位数586.387**，相对极差3.03%。三轮共106136次窗口内成功完成；发送窗口内的完整请求cohort共106157次，全部成功并完成尾部排空。对三轮cohort原始延迟按nearest-rank合并计算，**p50 13.577 ms、p99 14.148 ms、最大20.429 ms**。p99不是三轮p99的平均值。没有给定p99业务上限，因此不报告延迟SLA通过。

每轮受管作业退出码0，输入、图、二进制、配置身份和性能事件校验通过；双芯片前后门禁均通过。每轮60秒窗口内有5个资源快照，目标芯片各只见该轮同一宿主PID，同卡另一芯片未见进程；测试结束后两芯片`proc-mem`均为空。资源采样是离散观测，不保证发现采样间隙的瞬态外部负载。

本次新增设备时间460.300314秒；累计**6226.241481/7200秒**，余**973.758519秒**，实际唯一配置仍为12/12。它只给出已接纳配置在`case-0000`的性能，不构成`case-0000`上单实例基线与八实例候选的收益比较；`case-0002`和五例随机结果使用不同负载，不与本结果混算百分比。

## 证据与复现

[本地汇总JSON](../../../delivery/supporting-evidence/work/bge-opt-100/case0-best-summary-20261027.json)保存三轮指标、原始事件与门禁文件SHA256及最终空闲检查。100项目中的原始事件、资源快照、受管作业状态和结果保留在`work/case0-best-r{1,2,3}-20261027/`、`work/jobs/case0-best-r{1,2,3}-20261027/`及同名`*-performance.json`、`*-run-identity.json`、`*-before-gate.json`、`*-after-gate.json`。复现须先确认设备可独占、预算充足，并使用全新作业名：

```bash
cd /export/home/ext.hejun23/workspace/bge-opt-100
python3 tools/run_one.py --formal --config half8 --timeout 350 \
  case0-best-repro-01 -- --instances 8 --cores '12|24' \
  --seconds 60 --max-seconds 300 --requests 10000 --seed 2031 \
  --case-id case-0000
```
