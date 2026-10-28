# BGE 迁移至 100：开发交接

2026-10-26。开发角色仅做本地和远端只读调查；没有修改 99/100 文件、没有初始化 NPU、没有启动设备任务。本文给主控迁移、构建和排队使用，不能作为 100 正确性或性能通过证据。

## 1. 源快照与运行材料

99 / `hejun-cann9-bge`：

|材料|绝对路径 / 身份|
|---|---|
|canonical 代码（包含有效未提交改动）|`/export/home/ext.hejun23/workspace/bge/ascend-ge-adapters`|
|原隔离 runner|`/export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026/runner`|
|原调优工具、配置、插件和验收|`/export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026/{tools,plans,candidates,acceptance}`|
|原始模型|`/export/home/ext.hejun23/workspace/bge/model.onnx`，1345962755 bytes，SHA256 `2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0`|
|五条输入|`/export/home/ext.hejun23/workspace/bge/sample_inputs.json`，SHA256 `66571de2fa8c6a97e52a196a957801ab09b429751ba9cfb94712bb32c660b194`|
|业务输出|`/export/home/ext.hejun23/workspace/bge/expected_outputs.json`，SHA256 `4765cc2b9dac9d2f523f9a684a81082d3b943caaff1fd2c4db90cfdf1b375384`|
|小 ONNX、外置权重|`/export/home/ext.hejun23/workspace/bge/ascend-ge-adapters/work/bge/build/{model.onnx,weights.data}`，必须一起迁移；小 ONNX SHA256 `5e641362d99c9a1afb6dfce70f47b4ac07f1a7c88272fdfc791ea04ae84f4822`|
|归档后的 NPY 输入|`/export/home/ext.hejun23/workspace/bge/ascend-ge-adapters/models/bge/io/inputs/case-0000` 至 `case-0004`，每 case 8 个 NPY|
|独立 CPU 参考|`/export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026/acceptance/cpu-golden.npz`，SHA256 `a59f406eb5782926181963e94cb1f87e1728621dedc3d613f88a7317d8213f35`|
|独立 CPU manifest|同目录 `cpu-golden-manifest.json`，SHA256 `ba2afe4a2ab5c867174b5f515966f77413e79409d113c5a305fff7130bb3561f`|
|原打包输入|`/export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026/original-inputs/inputs.bin`，SHA256 `14ccd8bc7e4fb3f6437fbb7c1b47fc710c2cd2eba9ce220d1a1715a8936f7a44`|

99 canonical 与隔离 runner 的 `models/bge/replay.py` 当前均为 7997 bytes，SHA256 `3c0af6fdaae170563198e614075a45b6860c12ff69320d7f9d3c947ae0ecb00e`，尚未集成固定 case 补丁。原隔离 native binary SHA256 为 `8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d`；100 应重新构建并记录新 SHA，不能将旧 SHA 守卫直接用于新二进制。

## 2. 100 实际 CPU 环境

只读查询：Python `/usr/local/python3.11.15/bin/python3`，Python 3.11.15，libdir `/usr/local/python3.11.15/lib`；NumPy 1.26.4、PyYAML 6.0.3、ONNX 1.17.0、ONNX Runtime 1.26.0、protobuf 3.20.2；CMake 3.27.9、g++ 12.3.1。

`/usr/local/Ascend/cann-9.0.0/{set_env.sh,include,lib64}`、`/usr/lib64/libtcmalloc.so.4.5.16` 均存在。默认 `LD_PRELOAD=/usr/lib64/libtcmalloc.so.4.5.16`。

仓库 requirements 文本固定 ORT 1.20.1、protobuf 5.29.5，但**已验收的 99 独立 CPU manifest 记录 ORT 1.26.0 / NumPy 1.26.4，与 100 当前实际环境一致**。无需为了文本版本全局降级/升级；保留实际环境身份并重算参考核对。若后续确需依赖调整，使用项目隔离环境，不覆盖容器全局包。

## 3. 构建命令

以下假定新根为 `/export/home/ext.hejun23/workspace/bge-opt-100`，所有命令由主控执行。

```bash
cd /export/home/ext.hejun23/workspace/bge-opt-100/runner
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONHOME="$(python3 -c 'import sys; print(sys.base_prefix)')"
export LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TE_PARALLEL_COMPILER=2
cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release -DASCEND_HOME=/usr/local/Ascend/cann-9.0.0
cmake --build runtime/build -j8
ctest --test-dir runtime/build --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py'
```

不要带入旧 `runtime/build/CMakeCache.txt`。C++17、旧 ABI `-D_GLIBCXX_USE_CXX11_ABI=0` 已在 CMake 中配置；头文件还依赖仓库 `third_party`。链接 `ge_compiler/ge_runner/ge_common/ge_common_base/graph/graph_base/fmk_onnx_parser/parser_common/ascendcl/pthread`。普通 Python 测试中的 5 个 opt-in 设备测试必须保持未设置 `GE_TEST_DEVICE`，CPU 验证不能冒充设备通过。

## 4. 固定 case 补丁集成

本地准备包：`/Users/ext.hejun23/Documents/Codex/2026-10-19/ssh/work/bge-opt-20261026/dual-workload-replay.patch`，SHA256 `dff15d0301c8126348305b46034a8f29211769902bafb352436bdb1423d45859`。

只修改 `models/bge/replay.py`、`tests/test_bge_replay_cli.py`、`models/bge/README.md`，不改 C++。在新 runner 基线先 `git apply --check`，再 apply；不要原样运行本地 `integrate_dual_workload.py`，它硬编码 99 路径及旧 binary 身份。

```bash
git apply --check /export/home/ext.hejun23/workspace/bge-opt-100/tools/dual-workload-replay.patch
git apply /export/home/ext.hejun23/workspace/bge-opt-100/tools/dual-workload-replay.patch
python3 -m unittest discover -s tests -p test_bge_replay_cli.py -v
```

预期 replay.py SHA256 `0b5ac3dde39951bd4976cbc342a54d3a2bb22ffcb2805d00304a26ef0f10b3b0`、测试 SHA256 `599497ff6bf43feaaed3ed2984166cc7997167575529cdc85d76c8ee3d946925`；若另有必要路径更新，以 diff 和新 SHA 建立新身份。CPU preparation 测试原先 12 项通过，100 上仍需重跑。

新布局需要在项目 YAML 中把 `data.params.samples_file`、`verification.params.expected_outputs` 改成 `materials` 下实际绝对路径；`source.ref` 仍可 `${BGE_MODEL_PATH}`。**迁移源 YAML 历史留有 `aicore_num: 4|8`，不得把它当全核基线**。replay CLI 默认会移除该限核；建议新项目 YAML 也清掉历史限核。

## 5. preparation 与真实运行契约

固定 case 示例（只准备文件，不访问设备）：

```bash
cd /export/home/ext.hejun23/workspace/bge-opt-100/runner
python3 models/bge/replay.py --prepare-only \
  --config models/bge/config.yaml --device 14 \
  --graph /export/home/ext.hejun23/workspace/bge-opt-100/materials/exported/model.onnx \
  --output-dir /export/home/ext.hejun23/workspace/bge-opt-100/work/base-fixed-r1 \
  --case-id case-0002 --instances 1 --seed 2027 \
  --min-seconds 60 --max-seconds 300 --min-requests 10000
```

随机五例省略 `--case-id`，使用独立新目录和 run ID。默认 `precision_mode=must_keep_origin_dtype`、`graph_run_mode=0`、全核；固定 case 实际只有一个输入池元素，不能以随机结果中的子集代替。每个请求完整 scores D2H 40448 bytes。prepare 读取归档 NPY 与小 ONNX 元数据，不需要原 1.35 GB 模型；不能将原 inline-weight model 传给 `--graph`（CLI 拒绝超过 64 MB 的图）。

八实例半核：增加 `--instances 8 --aicore-num '12|24'`。它是编译核数上限，不提供实例独占，也不能把 `8×12` 理解为预留核数。C++ 在 `GEInitialize` 用 `ge::ir_option::AICORE_NUM` 设置；日志应核对生效值，图变化之后重新验收。

融合：增加 `--preload /export/home/ext.hejun23/workspace/bge-opt-100/candidates/libbge_add_layernorm_pass_v2.so`。prepare 追加保留 allocator；只由 `run_runtime.py` 对 native child 注入。**不得把融合 LD_PRELOAD 放到 Python launcher 外层**。

正确性压力：随机五例 `--capture --min-requests 2000 --min-seconds 1`（仍设合理 max-seconds，单实例/多实例均需每个 instance×case 至少 20 次独立校验）；纯性能不得 capture、dump 或 profiling。

native 接收 JSON stdin，不接收 plan 命令行参数。受管设备流程为 `device_job.py → env_run.sh → run_runtime.py <binary> <plan>`；`run_runtime.py` 创建全新 `plan.output_dir`，写 `runtime.log/resources.jsonl/result.json`，从 `GE_RESULT` 提取 native 结构化结果。运行时写 events.jsonl、performance 以及 capture 的 manifest.jsonl/outputs.bin。新的受管 wrapper 完成适配前，不能直接绕过它裸跑。

## 6. 融合 V2 CPU 构建

需一同迁移 candidates 中 `add_layernorm_pass.cpp`、`add_layernorm_pass_v2.cpp`、`preload_scope_filter.h`、`test_add_layernorm_pass.cpp`、`test_preload_scope.cpp`、`preload_probe.cpp`、`build_add_layernorm_v2.py`。

```bash
python3 /export/home/ext.hejun23/workspace/bge-opt-100/candidates/build_add_layernorm_v2.py \
  --cann /usr/local/Ascend/cann-9.0.0 \
  --directory /export/home/ext.hejun23/workspace/bge-opt-100/candidates
```

该脚本仅构建与 CPU 验证：7 图契约、11 preload filter、3 native/child/non-native 环境集成以及 `ldd -r`。临时 so 完整构建后原子替换目标。所需固定 Python 和 tcmalloc 路径均已在 100 确认存在。产物身份重新计算，不能沿用 99 so SHA 验收。

## 7. 必改的旧 launcher / 验收硬编码

|文件|必须处理的假设|
|---|---|
|旧 tools/run_checked.py|ROOT、`plan.device==10`、门禁 `npu-smi -i 5 -c 0`、`0000:89:00.0`、99 主机物理锁身份、旧 runner cwd/binary、旧锁根、结束门禁文案|
|旧 tools/env_run.sh|TMPDIR 固定到 99 项目目录；新项目设独立 tmp、缓存和编译目录；保留 Python embedding 环境修正|
|旧 tools/run_series.py|计划目录、旧 job 列表和 launcher 路径；不能直接提交历史整串实验|
|旧 tools/device_job.py|工具本体通用，无 99/device10 硬编码；调用侧传新 cwd、run-dir、共享 lock-root 和已核实的物理设备身份|
|旧 acceptance/validate_run_identity.py 与本地 acceptance-v2 同名文件|device10、99 PCI/host、旧 binary SHA、插件 SHA、结果直接位于 root/name 的布局、allocator 前缀守卫|
|acceptance-v2/verify_outputs.py|EXPECTED 固定 99 根，需配置成新 materials/expected_outputs.json；保留阈值和全部 ID+1/order 校验|
|acceptance-v2/workload_identity.py|默认寻找 sibling acceptance 和 sibling original-inputs；新布局必须放对目录或显式配置，保留冻结 manifest 与输入 SHA|
|本地 remote.py/integrate_*.py/prepare_*.py 等部署脚本|普遍硬编码 ssh99、旧项目根与旧计划，不应批量执行，优先用新参数化入口|

旧 run_checked 仅累计其 ROOT/jobs；迁移后若 jobs 为空会把累计预算错误清零。新主控必须保留已消耗约 1616.68 / 7200 设备秒，余约 5583.32 秒及 3 组新配置预算，运行前预留超时与最终稳定性预算。100 的逻辑 14/15 与 NPU/chip/PCI 映射、同卡负载由主控现场再次查证，不从 99 身份替换字符串推断。

验收工具分两层：数值/输入身份工具仍使用不可变 manifest；设备/产物身份工具使用新冻结运行 manifest，不能把设备身份条件删掉。性能复测须每场景 ≥60 秒且 ≥10000 完成，3 轮、事件独立重算；全量五例压力单独跑，最终至少 10 分钟稳定性。低于门槛的搜索只报告搜索值，不作交付 PASS。

## 8. 验证尚未完成

未构建 100 runner 或融合库、未部署固定 case 补丁、未检查新物理卡独占状态、未执行 100 模型正确性、基线、候选或 profiling。所有命令和路径是主控执行交接，不是已完成实验。
