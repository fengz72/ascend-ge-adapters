# 100 固定第三轮：SIGTERM 失败离线审查

结论：`final-best-fixed-r3` 的native runtime以SIGTERM退出，未形成有效完整性能轮；**信号的发送者和触发条件仍未知**。保留失败作业并计费，使用明确新名称的同配置补测接受独立验收。不能将这次失败改写为已确认的OOM、GE编译器崩溃或supervisor超时，也不能因补测成功而宣称整次流程零失败。

本报告只读取主控提供的本地文件：[失败作业及gate](../../../delivery/supporting-evidence/work/bge-opt-100/final-best-fixed-r3-failure.txt)、[runtime日志与结果](../../../delivery/supporting-evidence/work/bge-opt-100/final-best-fixed-r3-runtime.txt)、[事后容器状态](../../../delivery/supporting-evidence/work/bge-opt-100/r3-failure-host-state.txt)。未连接100、启动设备任务或修改正在运行的作业。

结构化提取、门禁时序和原始来源SHA见 [r3-failure-offline-review.json](../../../delivery/supporting-evidence/work/bge-opt-100/tester/r3-failure-offline-review.json)。

## 可以确认

- 原图 FP32、mode=0、8实例、`ge.aicoreNum=12|24` 的8次CompileGraph和8次LoadGraph均记录成功。随后没有成功的 `GE_RESULT`，launcher结果为 `returncode=-15`，native用时96.648659秒。
- supervisor记录 `reason=exited`、`exit_code=task_returncode=241`、`process_group_stopped=true`。当前launcher使用 `sys.exit(native_returncode)`；native的−15转换为POSIX 8位退出码241，与这条返回链一致，241不是另一种NPU错误码。
- supervisor从14:25:24.829035到14:27:08.005826 UTC，保守费用 **103.176791秒**。它包含启动、前后门禁和收尾，不用较短native用时替代计费。
- 前后两芯片proc-mem均为空，gate时间位于受管作业内，最终进程组停止。这支持本任务NPU上下文已释放；不能自动扩展为Host IPC等所有资源均无遗留。
- 约22:30:06 UTC+8的事后快照显示 `OOMKilled=false`、`restart_count=0`、`mem_limit=0`，目标两芯片仍无进程。

## 不能据此确定

| 现有日志 | 证据边界 |
|---|---|
| SIGTERM / −15 | 确定了退出信号，未记录发送者。可能来自其他进程、运行环境或进程自身的终止逻辑，现有材料无法区分 |
| supervisor `reason=exited`、约103秒结束 | 没有显示既定350秒timeout/cancel路径；不支持将它称为本工具超时。仍需同次request/cancel状态或信号日志做最终归属 |
| TBE `main process disappeared` | 在主runtime消失后报告编译子进程发现父进程不在，不能反推TBE主动杀死runtime或原始编译失败 |
| resource_tracker的12个semaphore warning | 说明异常退出时发现待清理对象，并不证明持续Host资源泄漏，也不证明所有清理已经完成 |
| OOMKilled=false / 没有容器重启 | 支持没有观测到容器整体OOMKill/重启，不排除子进程、宿主或上层cgroup/调度器行为。Docker mem_limit=0仅表示没有该项显式容器内存上限 |
| 无GE_RESULT | 足以判定没有有效完整性能结果；没有带时间的测量开始/预热完成标记时，不能严格仅凭它证明测量窗口一定尚未开始 |

主控将失败位置描述为测量前；独立证据当前能确认的范围是“8图加载成功后、有效完整性能结果产出前”。若存在阶段标记或已写出的请求事件，可在最终原始交接时补充定位，不能补造缺失的日志。

现阶段无需为解释一次信号额外启动模型或profiling。如已有同次宿主/调度器终止记录、supervisor cancel文件与请求、kernel或cgroup内存事件，可只读纳入；没有则保持“SIGTERM来源未定”。成功补测也不能单独证明原失败根因。

## 性能轮与预算处理

最终比较入口新增 `--best-fixed-r3 final-best-fixed-r3-retry1`，候选固定三轮及其有效门禁列表均采用r1/r2/retry1。失败原run的native/outer非零，不能进入均值、中位数或p99池，也不能填0 QPS后与成功轮平均。

预算审查仍遍历全部已完成job，包含原失败run与补测run，并单列失败计费项。两者属于同一冻结half8配置，重复测量不新增唯一配置，历史累计仍为12/12；所有真实占用都计入7200秒总账。

比较逻辑新增CPU回归验证：显式选择补测时有效12轮列表不含失败原名，但原失败的103.176791秒仍在总账且配置数不变。只有补测实际身份、输入、0退出、≥60秒且≥10000完成等原要求全过，才可成为有效第三轮。最终仍需原始证据独立复核后给出结论。
