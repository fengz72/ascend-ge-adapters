# A5 CANN 9.3.0 辅流能力核验

核验日期：2026-10-21。

**结论：`850-a5` 的 `hejun-cann910-stream` 容器当前安装包支持 `EagerOpExecutionContext::RequestAttachedStream()`。已在 A5 逻辑卡 1 编译并运行上游 `eager_attached_stream_add` 在线示例，辅流申请、kernel 下发、模型重放和 10 轮精度校验全部通过。**

## 实际安装版本

| 项目 | 实际值 |
|---|---|
| SSH / 容器 | `850-a5` / `hejun-cann910-stream` |
| 实际安装路径 | `/usr/local/Ascend/cann-9.3.0` |
| Toolkit 元数据 | `9.3.0`，`V100R001C12B140`，aarch64 |
| GE Executor / Compiler / Metadef / Runtime 元数据 | 均为 `9.2.0`，时间戳 `20261009_000323778` |
| 测试设备 | A5 Ascend950PR，逻辑卡 1；运行前无可见计算进程 |
| 测试目录 | `/home/h00909914/work/eager-attached-stream-check-9.3.0-20261021` |
| 上游示例固定提交 | `28aede1f69af4f15b1432ccf6ba9e29db6f9cfb7` |

Toolkit 名称与内部组件版本字段不同。结论针对本次实际安装文件和运行结果，不能仅以某个组件显示 `9.2.0` 判断功能缺失。容器沿用了原名称，其创建时的镜像标签也不代表重新安装后的组件版本。

## 接口与执行链路

安装头文件 `include/exe_graph/runtime/eager_op_execution_context.h:131` 已声明：

```cpp
rtStream RequestAttachedStream(const ge::AscendString &key);
```

递归检查 `lib64` 下共享库，按真实路径去重后共 206 个文件，全部完成动态符号扫描：

| 库 | 关键导出 |
|---|---|
| `liblowering.so` | `gert::EagerOpExecutionContext::RequestAttachedStream(ge::AscendString const&)` |
| `libdavinci_executor.so` | `ge::AttachedStreamCollection::RequestAttachedStream(...)`、`UnbindAndDestroy()` |

示例在模型加载、任务下沉期间进入 `Execute()`，申请 `eager_aux` 辅流，以 event 同步主流和辅流，在辅流上调用 `aclrtLaunchKernelV2()`，之后通过 `ExecuteGraphWithStreamAsync()` 重放模型。实际运行已覆盖这一链路。

示例来源：[固定提交的示例目录](https://gitcode.com/cann/ge/tree/28aede1f69af4f15b1432ccf6ba9e29db6f9cfb7/examples/custom_op/eager_attached_stream_add)。保存一份原始源码，并另建 `sample-device1` 用于测试；源码差异核对确认仅将 `ge.exec.deviceId` 从 `0` 改为 `1`，算子、kernel 和精度判定均未修改。

## 附录：全部验证结果

| 编号 | 验证 | 返回码 / 结果 |
|---|---|---|
| 1 | 安装头文件检查 | 存在 `RequestAttachedStream()` 声明 |
| 2 | 动态符号扫描 | 206/206 文件成功；存在公开接口与 V1 辅流管理导出 |
| 3 | 接口最小探针编译 | 0，成功 |
| 4 | 最小探针仅链接 `-llowering` | 1，缺少 `AscendString` 构造函数所在的 `libgraph` 显式依赖 |
| 5 | 最小探针链接 `-llowering -lgraph` | 0，成功 |
| 6 | `LD_BIND_NOW=1` 加载最小探针 | 0，成功；main 立即返回，实际接口调用由后续 GE 示例覆盖 |
| 7 | 示例 CMake configure | 0，成功 |
| 8 | 示例构建，`-j4` | 0，成功；生成自定义算子 SO 与 Session 可执行文件 |
| 9 | 可执行文件与自定义算子 SO 的 `ldd` | 无缺失依赖，无旧 `cann-9.1.0` 或 `cann-9.2.0-beta.2` 路径 |
| 10 | A5 逻辑卡 1 在线端到端运行 | 0，成功，进程总耗时 36.223 秒，含初始化、RTC 编译、构图、加载和执行 |
| 11 | 辅流 kernel 下发 | 日志打印非空辅流句柄，`blocks=8` |
| 12 | 模型重放精度 | 预热 3 轮，随后 10 轮校验全部通过，每轮 8192 个 float32 元素 |

编号 4 的失败来自最小探针链接命令漏写库依赖；补全显式依赖后成功，原始示例 CMake 已包含所需库。保留首次失败记录与日志。

运行日志关键输出：

```text
[AttachedStream] kernel launched on attached stream 0x58690dd8, blocks=8
[online] precision check passed (8192 elements)
...
========== Eager AttachedStream Online E2E: ALL PASS ==========
```

`precision check passed` 实际出现 10 次，进程返回码为 0。

### 复现命令

在上述测试目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
cmake -S sample-device1 -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
export ASCEND_CUSTOM_OPP_PATH="$PWD/sample-device1/output:${ASCEND_CUSTOM_OPP_PATH:-}"
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd build
timeout --signal=TERM --kill-after=10s 180s ./eager_attached_stream_session_run
```

### 与此前安装包的对照

| 实际安装包 | 接口声明 | 接口导出 | 本次或历史验证结论 |
|---|---|---|---|
| 9.1.0 B243，`hejun-cann910-qwen25` | 无 | 无 | 最小探针与原始示例均编译失败 |
| 9.2.0-beta.2 B086，`hejun-cann910-stream` 重新安装前 | 无 | 无 | 最小探针编译失败 |
| 当前 9.3.0 B140，`hejun-cann910-stream` | 有 | 有 | V1 在线示例端到端、10 轮精度全通过 |

本次确认了示例覆盖的**单模型 V1 在线执行能力**；没有验证 RT2、多模型并发、离线 OM 路径或 Prefix Attention 的性能收益。

### 证据保存

`cann-9.3.0-eager-attached-stream-evidence-20261021.zip` 保存原始与逻辑卡 1 调整后的示例源码、唯一差异补丁、源码摘要、安装头文件、组件元数据、动态符号扫描、构建与探针日志、运行日志、依赖解析和运行前后 NPU 快照。不包含 CSV、共享库或可执行二进制。

安装头文件 SHA-256：`0c2122ab41371c25f8137a5437802e60331721d36bd9e56158e8013dfc311078`。
