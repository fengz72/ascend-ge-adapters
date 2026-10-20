# 性能结果格式

每次执行独立保存 result.json 和运行日志；完成测量后保存 requests.csv。若测量途中失败，仍保存部分 performance 和 requests.csv，其中包括失败调用，status=failed；QPS 与分位数只统计成功调用，不能将失败运行当作达标性能。若在初始化阶段失败，可能没有逐调用 CSV，保留原始错误而不补造调用记录。JSON 使用 schema: ge-run/1；原生性能对象使用 schema: ge-performance/1。失败运行也保留状态、配置和原始错误，缺失的性能值不填零。

result.json 包含 run_id、started_at（UTC）、duration_seconds、stage、device、完整生效 config，以及实际 performance。duration_seconds 包含本步 worker 的启动、GE 编译/加载、预热和测量；performance.wall_ms 仅为测量段。普通性能扫描关闭 profiling。

performance.qps 是模型调用次数/s，performance.samples_per_second 是业务样本/s。batch 改变时使用业务样本/s 排名。e2e_ms 是一次完整 batch 的 H2D、描述符处理、执行和同步耗时，不能除以 batch；不含组 batch 等待、服务排队或业务后处理。

requests.csv 保存每次调用的 req、instance、input_set、shape_key、samples、h2d_ms、desc_ms、exec_ms、e2e_ms、error 和 first_hit，支持重算分位数、分析慢 case 与各实例差异。

实验编排可在同一个 result.json 中添加 experiment 信息：id、phase、repeat、kernel_aic_cores、kernel_library_sha256、batch_size、pool_sha256、输入长度统计等。模型相关的源码读取和调优策略由实验脚本或 Skill 完成，core 不读取自定义算子的头文件，也不实现参数搜索。

统一 CSV 可从原始 JSON 重新生成：

```python
from core.results import export_benchmarks

export_benchmarks(result_paths, "measurements.csv")
```

每行对应一次实际测量。列包括模型/后端/芯片/设备/精度、C/V 核数与实际自定义算子核数、batch、实例数、seed、请求次数、输入池标识、调用 QPS、业务 QPS、p50/p99/max（ms）、错误数、profiling 状态、HBM（MB）、预热和测量耗时、原始结果路径。CSV 不改变原始 JSON，也不增加运行清单或配置快照文件。
