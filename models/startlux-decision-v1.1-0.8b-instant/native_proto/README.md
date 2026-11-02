# native_proto

项目自建的 ChunkGatedDeltaRule GE AscendIR 原型注册库（**非官方兼容原型**，
`startlux-chunk-gdr-compat-proto/1.0.0`）。

- 存在原因：当前 CANN 9.1.0 (aarch64, Ascend910_93) 构建含该算子的 kernel、
  tiling 与 aclnn 宿主实现，但未在 GE OpFactory 注册 AscendIR 原型
  （libopsproto.so 有 RecurrentGatedDeltaRule、无 ChunkGatedDeltaRule）。
- 作用：仅执行真实 `REG_OP` 注册与 shape/dtype infer，让未改动的 torchair
  `get_registered_ir_def` 检查通过；不定义任何计算内核。
- schema 依据：torch_npu 2.12.0 converter、CANN 9.1.0
  `aic-ascend910_93-ops-info-transformer.json`、`ops_proto_legacy.h` 同族
  REG_OP 风格、torch_npu meta 注册。详见源码头注释。
- 加载：**显式项目局部加载**（本构建的验收交付方式，不再宣称默认自动加载）。
  `ASCEND_CUSTOM_OPP_PATH` 自动 vendor 加载在正式 `core.dependencies.environment`
  下复现 GE 初始化 SIGSEGV/SIGILL（见 `work/startlux-cannbot/review-native-env-failure.md`），
  故依赖不安装/不暴露 `opp/` 目录；由
  ① `python/chunk_gated_delta_rule_proto.py`（activate 时 ctypes dlopen RTLD_GLOBAL，
  并为本 worker 的 native 子进程设置任务级 `LD_PRELOAD`），或
  ② `scripts/env_chunk_proto.sh <work_dir>`（prepare 后在父 shell source，
  去重导出 `LD_PRELOAD`，标准 build/infer/bench 子进程全部继承）显式加载。
- 关闭条件：当所用 CANN 构建已自带同名原型注册时，从模型 YAML 中移除本依赖条目，
  并停止 source 上述 env 脚本（unset 其引入的 `LD_PRELOAD` 路径）；
  禁止双注册，禁止无证据地自动覆盖官方注册。

构建入口：`models/startlux-decision-v1.1-0.8b-instant/scripts/build_chunk_proto.sh`
（由 `core.dependencies.prepare` 以 `GE_SRC_DIR/GE_BUILD_DIR/GE_INSTALL_DIR` 调用）。
