# 100 设备编排与参数化验收：独立 CPU 审查

本轮没有启动 SSH/NPU，也未修改控制器或验收器实现。实际 100 模型、二进制和设备运行结果另行验收。

## 控制器

独立回归命令：

```sh
python3 -m unittest discover -s work/bge-opt-100/tester -p test_control_guardrails.py -v
bash -n work/bge-opt-100/env_run.sh
```

最新 `control.py` 的 13 项测试通过，shell 语法检查通过。证据：[control-guardrails.log](../../../delivery/supporting-evidence/work/bge-opt-100/tester/control-guardrails.log)、[control-review.json](../../../delivery/supporting-evidence/work/bge-opt-100/tester/control-review.json)。测试将 subprocess 全部替换成 CPU stub，不执行 stub 二进制或加载 stub `.so`。

已验证：

- 历史占用从 1616.67991 秒继续计费；新成功和失败任务均追加实际持有时间，没有因迁移清零。
- running 状态，以及已经创建目录/request 但缺 status 的任务，均阻止新任务。缺 status 曾被最初版本忽略，主控修复后独立复测通过。
- 改变 binary/graph 内容、输出目录不匹配、复用已有 job 目录时拒绝启动。
- known 和 new 配置名均与实际配置身份比较，同名不同实例/选项不能绕过名额；主控已修复最初可覆写/跳过比较的问题。
- 剩余预算不足时不发起设备任务；当前启动门槛除 timeout 外还留出 35 秒余量。
- 有效冻结配置能正常构造双物理设备锁 `100/0000:81:00.0`、`100/0000:83:00.0` 并通过 `guarded` 入口运行。
- 持锁任务的前置 gate 忙时不运行 runtime；后置 gate 未确认释放时不报告成功。

控制器部署位置应为 `<project>/tools/control.py`，此时 `ROOT=parents[1]` 正确。本地文件只是待部署源，不应在本地原位置直接执行真实 `run`。`plan.instances` 由准备入口生成整数，output_dir 为 `<project>/work/<job>`；控制器检查该精确路径。设备锁只协调遵守该共享锁的任务，外部占用仍须通过同卡两芯片的实际门禁和运行资源记录核验。

边界：预算是保守累计 UTC 实际作业时间（含锁等待）；清理必须先确认本任务进程停止，不能为满足时长强行释放锁。35 秒不是对异常清理的绝对上界；临近预算时仍需主控保留清理与最终验收余量。CPU 测试没有证明真实环境 `set_env.sh`、PYTHONHOME、驱动、锁根跨容器互斥或 Host PID 映射已经通过。

## 100 参数化验收

新 [acceptance](../../../delivery/supporting-evidence/work/bge-opt-100/acceptance/README.md) 已完成 reference/golden/expected/original-inputs、100 physical identity 和新 artifact hash 的参数化。自带 22 个 CPU 测试已由测试角色重新执行通过，见 [migrated-acceptance-independent.log](../../../delivery/supporting-evidence/work/bge-opt-100/tester/migrated-acceptance-independent.log)。这覆盖路径迁移仍保留哈希、固定池、插件只进入 native、保留 allocator、物理锁集合和不可降低 CLI 门槛等约束。

测试环境变量均指向本地只读归档：

```sh
export PYTHONPATH="$PWD/work/bge-opt-20261026/python-deps:$PWD/work/bge-opt-100/acceptance"
export BGE_REFERENCE_DIR="$PWD/docs/bge/evidence/20261026/acceptance"
export BGE_ORIGINAL_INPUTS="$PWD/docs/bge/evidence/20261026/original-inputs/inputs.bin"
export BGE_EXPECTED_OUTPUTS="$PWD/work/bge-provided-samples/expected_outputs.json"
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s work/bge-opt-100/acceptance -p test_workload_identity.py -v
python3 -m unittest discover -s work/bge-opt-100/tester -p test_migrated_acceptance_guardrails.py -v
```

补充四项独立性能语义回归时，最初为 1 通过、3 失败：runtime performance 显式写 `latency_scope=execute_only`、`arrival_model=open_loop` 或 `capture_enabled=true` 时，`perf_check` 仍返回通过。计划中的 capture/profiling/dump 已能被拒绝，但应同时核对实际 runtime 报告；其中 capture 不一致还会被 run identity 检查器拦截，scope/arrival 不一致则缺少门禁。

初次补充回归的失败情况保留于上文发现记录；[migrated-acceptance-guardrails.log](../../../delivery/supporting-evidence/work/bge-opt-100/tester/migrated-acceptance-guardrails.log) 现为修复后的最新独立回归日志。scope 语义缺口不影响未计时的数值正确性任务，但必须在性能验收前关闭。

后续复核：该缺口已修复，补充四项独立对抗回归现为 4/4 通过，并已核对受测检查器与 100 实际部署文件 SHA 相同。100 逻辑2首轮压力和双场景事件已独立复验，见 [100-initial-acceptance.md](100-initial-acceptance.md)。上文问题描述保留为发现及修复记录，不代表仍然阻塞。
