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
- 加载：**单一 `LD_PRELOAD` 契约**（本构建的验收交付方式，不再宣称默认自动加载）。
  `ASCEND_CUSTOM_OPP_PATH` 自动 vendor 加载会复现 GE 初始化 SIGSEGV/SIGILL，
  故依赖不安装/不暴露 `opp/` 目录，也不再装 python activator；任何触碰 GE 的
  进程都以 `LD_PRELOAD=<install>/lib/libchunk_gated_delta_rule_compat_proto.so`
  启动——`run.sh` 对 export/infer/bench 子进程注入。动态链接器在解释器启动前
  完成 `REG_OP` 注册（早于任何 torchair/GE import），native 子进程经
  `os.environ` 继承同一 `LD_PRELOAD`。
- 关闭条件：当所用 CANN 构建已自带同名原型注册时，移除 `scripts/prepare.sh` 中的本依赖
  构建与 `run.sh` 中的 `LD_PRELOAD` 注入；禁止双注册，禁止无证据地自动覆盖官方注册。

构建入口：本适配器的 `scripts/prepare.sh`（直接编译本目录的 .cpp 到
`work/deps/chunk_gated_delta_rule_proto/install/`）；父目录保有同一份源码
副本，两处内容一致。
