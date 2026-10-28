# 100 BGE：GE 逐算子 profiling 交接

开发只读核对 100 容器 CANN 9.0.0 和新 runner；没有采集 NPU 数据、没有启动模型、没有修改远端。2026-10-26。

## 真实采集入口

`/export/home/ext.hejun23/workspace/bge-opt-100/runner/runtime/main.cpp` 读取 `observability.profiling`，将 profiling 交给 GE 后端；并把 `performance.profiling_enabled` 标记为 true。`runtime/backends/gesession_backend.cpp` 在 GEInitialize 设置：

```text
ge.exec.profilingMode = "1"
ge.exec.profilingOptions = {
  "output": "<plan.output_dir>/bench-profiling",
  "task_time": "on",
  "runtime_api": "on",
  "aic_metrics": "PipeUtilization"
}
```

这是真实 GE 采集入口，不是占位日志。SDK `/usr/local/Ascend/cann-9.0.0/include/external/ge_common/ge_api_types.h:57` / `:58` 确认两个 key。runner 的 Destroy 最后调用 GEFinalize，正常退出后再分析，不能在进程仍运行时把残缺文件当采集完成。

主控从原图/融合 V2 已验证的固定 `case-0002` plan 各复制一份，保持 FP32、device、核数、单实例、输入字节、预热数和执行数完全相同，使用新 run ID 和新结果目录；仅 `observability.profiling=true`，`dump=false`。可明确 `aic_metrics="PipeUtilization"`；若仅要任务时间且不采硬件计数，可用空字符串，源码不会设置 aic_metrics 字段。两侧必须相同。

短诊断建议每 case 预热 10 次、测量 20 次：replay `min_seconds=0.01,max_seconds=2,min_requests=20,max_requests=20,warmup_per_case=10,capture_outputs=false`，其余字段不变。此轮可能因达到 max_requests 而被 runtime 标记 statistics_sufficient=false，这是短诊断，不作性能接纳；先确认任务成功和真实执行数。所有设备占用仍计入预算，设备队列和双 BDF 锁由主控管理。

采集覆盖启动与预热，不能直接把 CSV 总数除以 20。先用 Model ID / Infer ID、step/timeline 与执行记录核对实际执行次数；能证明一一映射时才排除预热。无法关联时保留“含预热的同配置累计 kernel 数据”，不声称单次稳态耗时或 p99 根因。

## CANN 9.0.0 实际工具

100 已只读运行 `/usr/local/Ascend/cann-9.0.0/bin/msprof --help`，退出 0。该路径链接到 `tools/profiler/bin/msprof`。确认支持：

```bash
/usr/local/Ascend/cann-9.0.0/bin/msprof --parse=on --output=/absolute/path/PROF_<exact-session>
/usr/local/Ascend/cann-9.0.0/bin/msprof --query=on --output=/absolute/path/PROF_<exact-session>
/usr/local/Ascend/cann-9.0.0/bin/msprof --export=on --output=/absolute/path/PROF_<exact-session>
```

export 支持 `--summary-format=csv`（默认 csv）、`--model-id`、`--iteration-id`。帮助声称默认 iteration 1，安装 Python 导出器又会依芯片/数据格式选择 all-export 或 graph-export；因此必须查看实际 Model/Infer 范围及日志，不能假设默认导出了全部调用。需要指定某个 model/iteration 时先 query，使用实际存在的 ID，不推断 ID 从 0 或 1 开始。不要把内部 Python 的 `--iteration-count`（C++ help 未展示）当已经验证的 CLI 承诺。

正常输出包括 PROF 会话的 host/device 原始数据、解析后的 sqlite 与 `mindstudio_profiler_output` 等导出目录，实际层级以现场文件为准。安装代码明确解析完成标志名 `all_file.complete`；此标志只证明解析阶段，不单独证明采集、导出或归属正确。有效证据要求：native/受管任务退出 0、GE_RESULT 成功、正常 GEFinalize、非空原始数据、parse/export 退出 0、非空 op_summary 和合法表头、目标进程/模型/设备归属与预期执行数一致。导出日志会报告 `... data ... has been exported to ...`。

若 native 入口无有效采集文件，可使用 msprof 外层应用采集作为单独回退（不是和 native profiling 同时开启）：

```bash
/usr/local/Ascend/cann-9.0.0/bin/msprof \
  --output=/absolute/new/profile-output \
  --task-time=on --runtime-api=on --ge-api=l1 --aic-metrics=PipeUtilization \
  python3 /export/home/ext.hejun23/workspace/bge-opt-100/tools/run_runtime.py \
  /export/home/ext.hejun23/workspace/bge-opt-100/runner/runtime/build/ge_runtime \
  /absolute/new/profile-plan.json
```

这是设备受管 job 的内部命令模板，仍需设备门禁、锁、超时、清理与预算；不能裸跑绕过 wrapper。融合库仍由 plan.runtime_env 注入 native child，不能在外层给 Python/msprof 预加载。回退 plan 必须 `observability.profiling=false`，但结果另行标记“msprof 外部采集”，不能因为 runtime flag false 就纳入正式性能。

## 实际 CSV 字段与分析边界

已查看安装文件 `/usr/local/Ascend/cann-9.0.0/tools/profiler/profiler_tool/analysis/msconfig/msprof_export_data_config.py`：

- op_summary：`Model Name,Model ID,Task ID,Stream ID,Infer ID,Op Name,OP Type,OP State,Task Type,Task Start Time(us),Task Duration(us),Task Wait Time(us),Block Num,Mix Block Num,HF32 Eligible`。
- op_statistic：`Model Name,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)`。
- step_trace：`Iteration ID,FP Start(us),BP End(us),Iteration End(us),Iteration Time(us),FP to BP Time(us),Iteration Refresh(us),Data Aug Bound(us)`。
- ai_core_op_report 还能追加 `Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats`，具体是否有数据看实际导出。

动态/单算子导出模式会省略部分列（安装代码明确单算子省略 Model Name/Infer ID），所以缺 Infer ID 时只能做聚合，不能伪造按请求归一化。Task Duration 单位为 us；不得以缺列默认 0 掩盖解析错误。

融合后以 `.../BgeAddLayerNorm` 的名字和 `OP Type=AddLayerNorm` 定位。原图需从最终图对齐 48 对残差 Add 与 LayerNorm，不能拿全部 Add 的时间与融合 kernel 比。分别比较对应 kernel 的次数、累计/平均耗时、等待时间、Block Num、输入 dtype/shape、Host launch/API 间隙。kernel 时间可能重叠，累计和不等于 GE 墙钟；单次 profile 只提供诊断线索，不证明稳定吞吐负收益。

## 本地最小工具

`/Users/ext.hejun23/Documents/Codex/2026-10-19/ssh/work/bge-opt-100/profiling-dev/profile_evidence.py`：只接受一份明确 PROF_* 会话；可 CPU parse/export，传播失败，拒绝多个 CSV 时默认选择“最新”；检查微秒单位、非有限数和重复行，保留原始 CSV 行号、文件 SHA、Model/Infer 范围、按类型/名称累计统计和 Add/LayerNorm/AddLayerNorm 原行。未证明 warmup 关联时不作剔除，不输出 QPS 或性能 PASS。

```bash
python3 profile_evidence.py --session /absolute/path/PROF_<exact> --export --report profile-report.json
# 多个CSV时加 --csv /absolute/path/PROF_<exact>/.../op_summary_....csv
# 已确认模型/执行关联后才加 --model-id ACTUAL_ID --infer-ids ACTUAL_ID_1 ACTUAL_ID_2
```

四项 CPU 回归通过，日志位于 `work/bge-opt-100/profiling-dev/cpu-tests.log`。旧 runner/tools/parse_profiling.py 会吞 parse/export 失败并默认选择最新 session，适合交互查看但不能单独作为完整性验收。新脚本也只验证导出文件；设备执行、产物身份和完整性仍由主控/测试交叉确认。
