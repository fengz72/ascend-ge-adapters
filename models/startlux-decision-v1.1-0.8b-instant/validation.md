# StartLux-Decision v1.1 0.8B Instant 验证记录（当前状态）

环境身份：容器 hejun-cann910-jev，CANN 9.1.0，Python 3.12.13，torch 2.12.0+cpu，
torch_npu 2.12.0，transformers 5.8.1，设备 Ascend910_9382，runtime 二进制
`runtime/build/ge_runtime`。GE Session：
`graph_run_mode=0`、`precision_mode=must_keep_origin_dtype`。

最终交付（已验收）：视觉塔 + 图像特征注入 + 三轴 MRoPE + 文本 decoder + 26 列
letter 读出头整合于**单一 AIR、单次 GESession**。正式 config 唯一：
`config/model.yaml`（unified_adapter）。AIR/runtime 身份：

- AIR `work/startlux-decision/build/model.air`：1,907,534,423 bytes，sha256
  `c1f8816ee25e03119b2522e55adb49ebf42a88692791b179920f5f60221edba1`
- runtime `runtime/build/ge_runtime`：639,952 bytes，sha256
  `692996b5df8a7ed8fdebdc365cdaa9383b6b538868ecf3e019303930d178eecc`

## 正式精度验收（全部通过）

- **六阶段 CLI**（data/baseline/prepare/build/infer/compare，正式
  `config/model.yaml`）：11 cases / 12 rows，min cosine 0.9999985326 / max
  relL2 0.0030342605 / 业务 active 概率 max 0.0055074189。主 agent 独立复核：
  `work/startlux-cannbot/review-p7-cli-independent.json`；AIR 结构独立核对
  （单图 + critical weights）：`work/startlux-cannbot/review-p7-air-structure-independent.json`。
- **官方业务验收**：13 请求 17 真实 GE 读出，证据
  `work/startlux-validation/p7-final/`。两种判据口径不同：
  - GE vs 相同 dtype 官方 BF16 动态参考（部署级判据）：**17/17 全过**，worst
    cosine 0.9999968 / relL2 0.00947 / 概率 |Δ| 0.00985；重入与 finalize 亦过。
  - GE vs 官方 FP32 同 prompt：**16/16 过**（17 读出中 16 有同 prompt FP32
    匹配）；wide_27 末轮 FP32 第 6 名 cause_18 vs BF16/GE cause_23，属跨
    dtype 已知翻转区，按披露处理不计失败，不称全 FP32 业务分布门限通过
    （`work/startlux-validation/p7-final/compare-report.json`）。
- **边界矩阵**：18 cases / 27 rows 数字门限全过（min cosine 0.9999509 / max
  relL2 0.0153549；B1 S<=2048、B2 long [129,2048]、B4 short 及同会话未改动行
  三组 bitwise 全过）：
  `work/startlux-cannbot/review-p7-boundary-independent.json`。
- **B4 long [1,65,129,2048] 补验（已通过）**：P7 时期共享卡资源不足真实 OOM；
  2026-11-01 P8 空闲窗口以 fresh GESession 补验 4/4 行全过（vs 独立原官方
  CPU 金标：min cosine 0.9999947830951117 / max relL2 0.0040037066209951165，
  gate wall 56.006 s）。**OOM 已解决**；P7 OOM 历史证据保留于
  `review-p7-boundary-independent.json`，不删除。

## CPU 单测

主 agent 2026-10-31 独立重跑 13 个相关模块（六基础模块 + 统一图五模块 +
multimodal + torchair exporter 门禁）共 **317 项全部通过**：
`work/startlux-cannbot/review-p7-unittest-final.log`（27.8s）与
`review-p7-unittest-final-state.log`（Ran 317 tests in 27.635s，OK）。清理
（2026-11-02）后主 agent 再次复跑 317 项全部通过：
`work/startlux-cannbot/cleanup-20261102/unittest.log`（Ran 317 tests in
28.531s，OK）。全仓套件另有 4 个预存 error（上游 onnx 缺包
ModuleNotFoundError，非本模型引入），不据此宣称全 repo 通过。

## 2026-11-02 变更：收敛为 native chunk GDR 单路线

**决策**（用户拍板）：纯 Torch 标准路线与原生融合路线并存的结构收敛为
**仅 native 路线**——接受 CANN 升级/换 SoC 时无标准路线兜底的风险（proto
为项目自建注册库，失效时必须先重建重验才能导出）。

- `startlux_npu_adapter.py`：native `_gated_delta`（q/k 真实长度 FP32 L2
  归一 + BF16 cast、零初始态、逐行物理宽度语义）合入基座；`describe`
  契约含 `native_gdr`/`dynamic`/`required_dependencies`；`RuntimeHandle`
  /`CandidateHandle` 全部转为动态 io（B2/S65 确定性 trace 示例）；删除
  `_encode_padded` 固定 B=1 路径。图内容不变（native trace 从不经过被删
  代码），正式 AIR 无需重导出。
- 删除：`startlux_native_adapter.py`（并入基座）、`gated_delta.py`（纯
  Torch GDR）、`operator_checks.py`（P1 门禁历史入口）。
- 保留更正：`ge_readout.py` **不是过程文件**——`unified_readout.py` 与
  `scripts/run_unified_performance.py` 加载其共享实现（`run_plan`/
  `read_air_io`/`parse_timings` 等），与其测试一并保留。
- 测试：`test_startlux_adapter.py` 重写（删 GDR 单测 6 项、conv 单测 2 项、
  wrapper 级 CPU forward 1 项、`_encode_padded` 契约 1 项；并入原
  native 契约用例 5 项，`test_startlux_native_adapter.py` 删除）；
  `test_startlux_multimodal.py` 删 4 项 CPU 上执行 decoder/unified
  forward 的用例（native op 无 CPU 实现）。**302 项全部通过**（Ran 302
  tests in 26.611s，OK；317 → 302 = −10 −6 +5 −4）。
- 覆盖转移：CPU 上的整图 wrapper 级数学验证随标准路线一并消失（native op
  无 CPU 实现），NPU 侧证据（P3 native 7/7、P6/P7 验收、compare 复核）
  为权威覆盖。

## 2026-11-02 变更：ge_runtime 子进程执行流程去重（三处 → 一处）

**决策**（用户拍板，可维护性优先）：ge_runtime 子进程执行协议此前有三套
重复实现——`core/backends/runtime.call`（subprocess.run 简单版，CLI 六阶段
用）、`ge_readout.run_plan`（timeout + 证据保留，业务读出用）、
`run_unified_performance.ge_call_polling`（Popen + abort 轮询 + 进程组
kill，P8 用）——收敛为 `core/backends/runtime.execute` 一套权威实现。

- `execute(runtime_bin, plan, *, log_path, timeout, should_abort,
  poll_interval, error, aborted_error)`：超集语义（Popen + start_new_session
  进程组隔离、timeout/abort 触发 killpg、恰好一条 GE_RESULT、failed-status
  非零退出返回带 error 详情的 result、wall 计时、log_path 缺省走临时文
  件）；异常类型按调用方参数化，全部公共接口/异常类型/返回结构保持不变。
- `call()` 变薄委托（签名不变）：core/worker、test_framework、
  test_bge_replay_cli 的 patch 点零改动；全部路径附带获得进程组隔离
  kill（此前仅 P8 有）。
- `ge_readout.run_plan` / `run_unified_performance.ge_call_polling` 委托
  execute，仅保留各自异常映射（ReadoutError / WorkerError+Aborted 含
  abort reason）；`_kill_own_group` 删除。P8 脚本变更无 NPU 复验（NPU
  侧已停测），行为等价由下述测试保证。
- **边界放宽记录**：ge_readout 原已 import `core.contracts.validate_io`
  （repo-root 自举），本次追加 import `core.backends.runtime`；适配器文件
  （load_adapter 加载的 unified_adapter.py）仍不 import core，隔离规则
  适用范围不变。
- 行为保持证据：新增 execute 协议单测 9 项（missing-binary 错误类型、
  单/多 GE_RESULT、failed-status 返回、非零退出、非 JSON、timeout kill、
  abort 仅杀本组、临时日志）；test_review_contracts 两处 patch
  subprocess.run 的用例改真实假二进制（fork-child 场景断言拆分以适配新
  消息格式）；**302 项 startlux 单测全过**（含 ge_readout 47 项与
  unified_readout timeout 项）；全仓 406 项 4 errors 全为预存 onnx 缺包，
  与既有记录口径一致，零新增。

## 精度验收的历史支撑（细节在归档）

P0 材料固定（上游 commit/模型 revision/权重 SHA 逐文件核对，
`work/startlux-materials/materials-report.md`；vendor 13 模块逐字节一致
`work/startlux-cannbot/review-vendor.json`）、P1 算子级门禁、P2 官方 CPU
FP32/BF16 业务语义、P3 B1/S256 两路线整模 7/7（`review-p3-final-independent.json`）、
P4 动态整模 27 cases（`review-p4-dynamic-raw-independent.json`）、P4 业务回调
（`review-p4-business-independent.json`）、P6 视觉塔 FP32 AIR（relL2
0.000286066/0.000292276，`review-p6-vision-raw-independent.json`）等过程记录
与原始文档见 `work/startlux-archive/cleanup-20261102/`（decision-validation.md、
native-validation.md、unified-validation.md 等快照）及 work/ 下各证据目录。

## P8 独立复核（2026-11-02，passed）

`work/startlux-cannbot/review-p8-real-independent.json` / `.md`：从原始 run 目录
独立复算 B4 门禁与 bench 全量统计（raw CSV sha256
`729b9d53957d5b4b7bd4bcdaa9b4db2b11b9b399390d0c05801fc0944c848c4f`、B4 raw
sha256 `2b69e5863a086b54238d212dd74417c03dc2b82df1dc710aed1d8706f8b81bea`），
AIR/runtime SHA 与字节数一致，随机序列逐请求一致，结论 passed。性能数值见
`performance.md`。

## 原始证据路径

- P8/B4 补验与 bench：
  `work/startlux-validation/p8-idle-performance/runs/20261101T025002-45aba096/`
  （`gate-20261101T025013-a9dafbe2/`：gate-report.json、plan.json、inputs.bin、
  outputs.bin、infer-runtime.log；`bench-20261101T025109-558a6522/`：result.json、
  summary.json、report.md、requests.csv、bench-runtime.log、inputs.bin）
- P7 正式验收：`work/startlux-cannbot/review-p7-cli-independent.json`、
  `review-p7-air-structure-independent.json`、
  `review-p7-boundary-independent.json`、
  `work/startlux-validation/p7-final/`、
  `work/startlux-cannbot/p7-main-review-findings.md`
- 主 agent 复核：`work/startlux-cannbot/p8-main-final-review.json`（accepted）、
  `review-p8-real-independent.json`/`.md`
