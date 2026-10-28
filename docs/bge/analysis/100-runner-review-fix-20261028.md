# 100 机器 runner review 问题修复（2026-10-28）

执行位置：`100/hejun-cann9-bge` 的 `/export/home/ext.hejun23/workspace/bge-opt-100`。本次修复不改原始 ONNX、样例、精度门槛、GE mode=0、`force_fp16`、10 实例或 `12|24` 限核。

## 修复内容

1. GE 静态编译输出和动态执行输出统一按规范化 dtype 比较：`float` 等价于 `float32`，`double` 等价于 `float64`；未知类型和真正不一致的类型仍拒绝。回归测试覆盖接受别名与拒绝不一致类型。
2. `runner/models/bge/config.yaml` 的模型、输入样例和预期输出改用相对 `runner/` 根的 `../materials/`，整个项目目录迁移时只需保持 `runner/` 与 `materials/` 同级。当前机器解析后仍是原来的三个材料文件。
3. 修正 `runner/delivery/README.md` 的源码与项目工具路径。新环境的受管工具身份清单必须按新设备和构建产物重新生成；旧身份与二进制已保存在 `runner/delivery/previous-binary/`。

## 身份与验证

|项目|SHA256|
|---|---|
|2026-10-27 三轮性能二进制（归档）|`8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d`|
|本次修复后 `runner/runtime/build/ge_runtime`|`735a3532ecb131de203200a328835af0d58bcddf27f9572eb997b5d6808e43f4`|

`cmake --build runtime/build -j4`、C++ `runtime_contracts` 及 13 项 BGE Python 回放测试通过；当前配置的三个相对材料路径均解析到现有项目根 `materials/`。新二进制使用独立受管配置键 `forcefp16-half10-dtypefix`，历史 `forcefp16-half10` 仍绑定旧二进制。

逻辑设备 2 上，三个作业的前后双芯片空闲门禁、进程退出和独立输入身份校验均通过。性能作业关闭 profiling 和逐输出归档，计时包含 H2D、GE 执行同步和完整 D2H；压力作业单独归档并检查全部输出。

|验证|新二进制结果|判定|
|---|---|---|
|五例随机全量输出压力|22,274 请求，0 失败；最大绝对误差 0.0005681、relative L2 最大 0.0008141；Top10 ID、顺序、分数门槛通过|正确性通过|
|固定 `case-0000`，60 秒单轮|50,977 个窗口内成功请求，849.606 QPS，p99 24.404 ms，0 错误|单轮性能校验通过|
|五例随机，60 秒单轮|51,718 个窗口内成功请求，861.960 QPS，p99 22.948 ms，0 错误|单轮性能校验通过|

这两项单轮 QPS 分别比旧二进制 2026-10-27 三轮中位数 884.177、903.582 低 3.910% 和 4.606%。测量不在同一时间交错进行，不能据此断言修复造成回退，也不能把旧三轮中位数当作新二进制的正式性能验收。若需要确认新构建的稳定吞吐，应按固定与随机场景分别做同设备交错三轮复测，并继续核对 p99、资源和独占门禁。

机器可读摘要见 [证据](../evidence/20261028-runner-review-fix.json)；原始计划、输出、事件、资源及门禁保留在项目根 `work/dtypefix-fp16-ten-{pressure,fixed,random}-20261028/` 和对应的 `work/jobs/`、`work/*-performance.json`、`work/*-accuracy.json`。
