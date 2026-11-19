# A5 CANN 辅流（Eager Attached Stream）支持核验

核验日期：2026-10-21。

## 1. 结论与版本要求

上游 `eager_attached_stream_add` 示例依赖的 `gert::EagerOpExecutionContext::RequestAttachedStream()` 在 CANN 9.3.0 `V100R001C12B140` 安装包中存在声明与动态导出，并已在 `850-a5` 的 A5 逻辑卡 1 上端到端跑通：辅流申请、kernel 下发、模型重放和 10 轮精度校验全部通过。CANN 9.1.0 `V100R001C11SPC001B243` 与 9.2.0-beta.2 `V100R001C12B086` 的实际安装包均无此接口，最小探针编译失败。

三版本一行对照：9.1.0 `V100R001C11SPC001B243`（`hejun-cann910-qwen25`）不可用、9.2.0-beta.2 `V100R001C12B086`（`hejun-cann910-stream` 重新安装前）不可用、9.3.0 `V100R001C12B140`（`hejun-cann910-stream`）可用；接口声明/导出与库文件数明细见第 4 节。

版本下限的出处：`third_party/ascend-ops/prefix-attention/cannbot-dsl/ge/CMakeLists.txt:156-166` 用 `check_cxx_source_compiles` 同时探测 `gert::EagerOpExecutionContext::RequestAttachedStream` 与 `gert::ResourceUsageContext::ReportAttachedStream` / `ge::ResourceUsageReporter`，探测失败即报 `FATAL_ERROR "PrefixAttention requires CANN >= 9.3.0 with GE RequestAttachedStream and ResourceUsageReporter"`；实现侧 `third_party/ascend-ops/prefix-attention/cannbot-dsl/ge/src/prefix_attention_op.cc:232` 调用 `ctx->RequestAttachedStream(kPlannerStreamKey)`。`models/qwen2.5-0.5b/README.md` 的使用限制同样写明 A5 需要 CANN ≥ 9.3.0 的辅流接口。

范围限定：以上结论只覆盖本次实际安装的三个组件组合与 `850-a5` 设备。9.3.0 的端到端结果已在 `V100R001C12B140` / A5 逻辑卡 1 实测；其他 9.3.0 构建、其他设备、RT2 路径、多模型并发与离线 OM 路径均待验证。9.1.0 与 9.2.0 的「不可用」不代表所有同名版本或定制补丁包。

## 2. 接口契约

上游[示例目录](https://gitcode.com/cann/ge/tree/master/examples/custom_op/eager_attached_stream_add)通过 Git 获取并阅读；网页抓取工具无法访问该页面。为避免 master 后续变化，固定读取提交 `28aede1f69af4f15b1432ccf6ba9e29db6f9cfb7`，提交时间 `2026-10-20 15:31:55 +0800`。

示例执行链路：

1. 用 RTC 编译 Ascend C Add kernel，构建 GE 图。
2. V1 模型加载、任务下沉时进入自定义算子的 `Execute()`。
3. 调用 `ctx->RequestAttachedStream("eager_aux")`，申请由 GE 管理的辅流。
4. 主流 record 输入就绪 event，辅流 wait；辅流执行 Add，record 完成 event，主流 wait。
5. `ExecuteGraphWithStreamAsync()` 重放模型；预热 3 轮，随后校验 10 轮、每轮 8192 个 float32 元素的加法结果。

这项能力涉及的不只是创建一条 ACL stream。上游实现还包括：

| 层次 | 必要支持 |
|---|---|
| 公开 API | `EagerOpExecutionContext::RequestAttachedStream(const ge::AscendString &key)` |
| 上下文转发 | 通过额外输出槽 `kArgsHandler` 获取 `ArgsHandler`，再获取 `AttachedStreamProvider` |
| V1 执行器 | 模型拥有 `AttachedStreamCollection`，创建物理流并以 `ACL_MODEL_STREAM_FLAG_HEAD` 绑定模型 |
| 生命周期 | 模型管理辅流；同模型相同 key 复用；销毁前同步、解绑及销毁流 |
| 算子侧 | 用 event 显式保证主流和辅流之间的数据依赖 |

源码依据：[算子实现](https://gitcode.com/cann/ge/blob/28aede1f69af4f15b1432ccf6ba9e29db6f9cfb7/examples/custom_op/eager_attached_stream_add/ge/custom_op.cpp)、[上下文接口](https://gitcode.com/cann/ge/blob/28aede1f69af4f15b1432ccf6ba9e29db6f9cfb7/inc/graph_metadef/external/exe_graph/runtime/eager_op_execution_context.h)、[上下文转发](https://gitcode.com/cann/ge/blob/28aede1f69af4f15b1432ccf6ba9e29db6f9cfb7/graph_metadef/exe_graph/runtime/eager_op_execution_context.cc)、[V1 辅流管理](https://gitcode.com/cann/ge/blob/28aede1f69af4f15b1432ccf6ba9e29db6f9cfb7/runtime/v1/graph/load/model_manager/attached_stream_collection.cc)。

该固定提交也包含 `Rt2AttachedStreamCollection` 和 RT2 provider 接线，不能把当前 master 的支持范围概括成“只有 V1”。在线示例自身验证的是 V1 下沉、重放链路，RT2 未在 9.1.0 环境验证。

## 3. 核验方法

每个实际安装包都按同一顺序执行以下四步，版本差异只体现在结果上。

1. **头文件声明检查**：在 `<CANN>/include/exe_graph/runtime/eager_op_execution_context.h` 查找 `RequestAttachedStream()`，并核对额外输出槽是否含上游转发所需的 `kArgsHandler`。
2. **lib64 递归去重符号扫描**：对安装目录 `lib64` 下所有共享库按真实路径去重，逐个执行 `nm -D -C`，统计文件总数并匹配公开接口与相关内部类符号。
3. **最小负向探针**：只使用安装包原始头文件，不修改系统组件；分别编译 `GetStream()` 与 `RequestAttachedStream("eager_aux")` 探针。
4. **原始示例构建**：上游示例原样复制到隔离目录，先 source 当前安装包的 `set_env.sh`，使用安装包自带头文件、库和 `_GLIBCXX_USE_CXX11_ABI=0` 构建；不修改系统安装或现有 Prefix 算子。

最小负向探针源码：

```cpp
#include "exe_graph/runtime/eager_op_execution_context.h"
gert::rtStream probe(gert::EagerOpExecutionContext *ctx) {
  return ctx->RequestAttachedStream("eager_aux");
}
```

最小探针编译命令：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I"$ASCEND_HOME_PATH/include" -c probe-attached-stream.cc
```

## 4. 版本对照表

| 实际安装包 | 接口声明 | 接口导出 | 库文件数 | 验证结论 |
|---|---|---|---:|---|
| 9.1.0 B243，`hejun-cann910-qwen25` | 无 | 无 | 203 | 最小探针与原始示例均编译失败 |
| 9.2.0-beta.2 B086，`hejun-cann910-stream` 重新安装前 | 无 | 无 | 204 | 最小探针编译失败 |
| 9.3.0 B140，`hejun-cann910-stream` | 有 | 有 | 206 | V1 在线示例端到端、10 轮精度全通过 |

9.2.0 的接口槽位比 9.1.0 更齐全（新增 `MallocReadOnlyDevArgs()` 与 `kArgsHandler`），但仍无 `RequestAttachedStream`。这说明「槽位齐全」不等于「支持辅流」，版本下限不能按槽位或组件版本号推断，只能按目标接口的声明、导出与实际运行结果判断。

## 5. 9.3.0 实测（V100R001C12B140）

### 实际安装版本

| 项目 | 实际值 |
|---|---|
| SSH / 容器 | `850-a5` / `hejun-cann910-stream` |
| 实际安装路径 | `/usr/local/Ascend/cann-9.3.0` |
| Toolkit 元数据 | `9.3.0`，`V100R001C12B140`，aarch64 |
| GE Executor / Compiler / Metadef / Runtime 元数据 | 均为 `9.2.0`，时间戳 `20261009_000323778` |
| 测试设备 | A5 Ascend950PR，逻辑卡 1；运行前无可见计算进程 |
| 测试目录 | `/home/h00909914/work/eager-attached-stream-check-9.3.0-20261021` |
| 上游示例固定提交 | `28aede1f69af4f15b1432ccf6ba9e29db6f9cfb7` |

Toolkit 名称与内部组件版本字段不同：Toolkit 显示 `9.3.0`，而 GE Executor / Compiler / Metadef / Runtime 元数据仍是 `9.2.0`。结论针对实际安装文件和运行结果，不能仅以某个组件显示 `9.2.0` 判断功能缺失。容器沿用了原名称，其创建时的镜像标签也不代表重新安装后的组件版本。

### 接口与符号

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

### 验证结果

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

编号 4 的失败来自最小探针链接命令漏写库依赖；补全显式依赖后成功，原始示例 CMake 已包含所需库。首次失败记录与日志保留在证据包内。

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

已确认的是示例覆盖的**单模型 V1 在线执行能力**；RT2、多模型并发、离线 OM 路径与 Prefix Attention 性能收益未验证。

## 6. 9.2.0-beta.2（V100R001C12B086）

| 项目 | 实际值 |
|---|---|
| 容器 | `hejun-cann910-stream`，running |
| Docker 镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-beta.2-950-openeuler24.03-py3.12` |
| CANN 目录 | `/usr/local/Ascend/cann-9.2.0-beta.2` |
| Toolkit | `9.2.0-beta.2`，`V100R001C12B086` |
| GE Executor / Compiler / Metadef / Runtime | 均为 `9.2.0-beta.2`，时间戳 `20260826_163105218` |
| 证据目录 | `/home/h00909914/work/eager-attached-stream-check-9.2.0-20261021` |

| 检查 | 结果 |
|---|---|
| 安装头文件中 `RequestAttachedStream()` | 无声明 |
| 安装头文件中 `GetStream()` | 有声明 |
| 安装头文件中 `MallocReadOnlyDevArgs()`、`kArgsHandler` | 均存在，比 9.1.0 环境多出这些接口或槽位；不能据此认为已支持辅流 |
| 共享库动态符号扫描 | `lib64` 递归去重后 204 个文件，全部扫描成功，无 `RequestAttachedStream` 匹配 |
| 基础接口符号 | `liblowering.so` 导出 `GetStream()`、`MallocReadOnlyDevArgs()` |
| 底层模型绑定符号 | 有 `aclmdlRIBindStream`，不足以替代 GE 托管辅流 API |
| `GetStream()` 最小探针编译 | 返回码 0，成功 |
| `RequestAttachedStream("eager_aux")` 最小探针编译 | 返回码 1，失败：类没有该成员 |
| 原始示例构建或 NPU 执行 | 未执行；目标接口的最小编译探针已失败 |

核心报错：

```text
error: ‘class gert::EagerOpExecutionContext’ has no member named ‘RequestAttachedStream’
```

范围限定：本节只覆盖实际安装的 9.2.0-beta.2 B086，不能外推至其他 9.2.0 构建或定制补丁包。上游接口来源沿用固定提交 `28aede1f69af4f15b1432ccf6ba9e29db6f9cfb7`，未重新解析变化中的 master。

## 7. 9.1.0（V100R001C11SPC001B243）

| 项目 | 实际值 |
|---|---|
| SSH | `850-a5` |
| 容器 | `hejun-cann910-qwen25` |
| 项目根目录 | `/home/h00909914/ascend-ge-adapters` |
| 当前项目提交 | `59050e390a8bcd4d01f7beac50b741c008285f2a` |
| CANN 实际目录 | `/usr/local/Ascend/cann-9.1.0` |
| Toolkit | `9.1.0`，`V100R001C11SPC001B243`，aarch64 |
| GE Executor | `9.1.0`，组件时间戳 `20261015_165011190` |
| GE Compiler | `9.1.0`，组件时间戳 `20260730_231653901` |
| Metadef / Runtime | 均为 `9.1.0`，组件时间戳 `20260730_231653901` |
| 编译器 | GCC 12.3.1，CMake 3.27.9 |
| 隔离核验目录 | `/home/h00909914/ascend-ge-adapters/work/eager-attached-stream-check-20261021` |

GE Executor 的组件时间戳与其他组件不同，因此核验依据为实际文件和构建结果，而非仅比较 Toolkit 的版本号或文件日期。

### 头文件与运行库

安装目录的 `include/exe_graph/runtime/eager_op_execution_context.h` 有 `GetStream()`，没有 `RequestAttachedStream()`；其额外输出槽也只有 `kWorkSpace`，缺少上游转发所需的 `kArgsHandler`。

递归扫描安装目录 `lib64` 下所有共享库，按真实路径去重后共 **203** 个文件，`nm -D -C` 全部成功：

| 符号 | 实际结果 |
|---|---|
| `EagerOpExecutionContext::RequestAttachedStream` | 无定义或引用匹配 |
| `AttachedStreamCollection` | 无动态符号匹配；单凭内部类符号缺失不作为决定性证据 |
| `EagerOpExecutionContext::GetStream() const` | 在 `liblowering.so` 中导出 |
| `aclmdlRIBindStream` | 在 `libacl_rt.so`、`libascendcl.so` 中导出 |

因此，当前存在底层流绑定 API，但没有示例使用的 GE 公开入口及配套接口。仅替换头文件或手动补一行成员声明无法补齐运行库实现和 provider 接线。

### 验证结果

| 编号 | 验证 | 返回码 | 结果 |
|---|---|---:|---|
| 1 | `GetStream()` 最小探针编译 | 0 | 成功，确认基础 include 环境可用 |
| 2 | `RequestAttachedStream("eager_aux")` 最小探针编译 | 1 | 失败：类不存在该成员 |
| 3 | `GetStream()` 探针链接 `liblowering.so` | 0 | 成功，确认基础接口运行库链接可用 |
| 4 | `LD_BIND_NOW=1` 加载并运行上述探针可执行文件 | 0 | 成功；main 立即返回，仅验证加载与符号重定位，未调用上下文或 NPU |
| 5 | 上游原始示例 CMake configure | 0 | 成功 |
| 6 | 上游原始示例 build，`-j4` | 2 | 失败于 `ge/custom_op.cpp:92`，同样缺少 `RequestAttachedStream` |
| 7 | 安装目录共享库符号扫描 | 203/203 文件成功 | 未找到 `RequestAttachedStream` 导出 |
| 8 | NPU 上端到端执行、精度或性能测试 | 未执行 | 原始示例无法生成算子交付件；没有端到端成功结果 |

核心报错：

```text
error: 'class gert::EagerOpExecutionContext' has no member named 'RequestAttachedStream'
```

原始示例构建命令：

```bash
cmake -S upstream-example -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

## 8. 与 Prefix Attention 的关系

当前 A5 Prefix 实现使用 `ctx->GetStream()`，依次提交 metadata 清零、AICPU planner 和 Attention kernel。9.1.0 与 9.2.0-beta.2 实际安装包均已导出 `GetStream()`，该既有路径可用；9.3.0 B140 在此之上提供了框架托管辅流接口。要通过示例 API 改成框架托管辅流，需要完整匹配的 GE/Metadef 组件及依赖，不能仅更新算子代码或头文件。

即便支持辅流，也不能由此直接推导吞吐提升：planner 生成的 metadata 是 Attention 的输入，两者之间仍有依赖，移到辅流不会自动消除这段耗时。需要让独立工作发生重叠，或减少重复规划，再用 profiling 验证收益；Prefix Attention 的性能收益尚未验证。

示例在源码中明确假定单模型，其 event 缓存在按算子类型复用的实例中。若将方案用于当前多实例并发业务，还需按模型或执行实例隔离同步资源，并验证不同输入、多实例重放、销毁与重载。示例的 10 轮校验不能替代这些验证。

## 9. 证据与哈希

| 安装包 | 证据包 | 内容 | 安装头文件 SHA-256 |
|---|---|---|---|
| 9.3.0 B140 | `cann-9.3.0-eager-attached-stream-evidence-20261021.zip` | 原始与逻辑卡 1 调整后的示例源码、唯一差异补丁、源码摘要、安装头文件、组件元数据、动态符号扫描、构建与探针日志、运行日志、依赖解析、运行前后 NPU 快照；不含 CSV、共享库或可执行二进制 | `0c2122ab41371c25f8137a5437802e60331721d36bd9e56158e8013dfc311078` |
| 9.2.0-beta.2 B086 | `cann-9.2.0-eager-attached-stream-evidence-20261021.zip` | 组件信息、安装头文件快照、最小探针、编译日志、动态符号扫描结果及文档；无 CSV | `c929ebacc00e7131c6c0e8429e49e95401ddd6d90f2d3c7a5549cc43e7560559` |
| 9.1.0 B243 | `cann-9.1.0-eager-attached-stream-evidence-20261021.zip` | 固定提交的示例源码、相关上游接口与 provider 源码、安装头文件快照、组件信息、SHA-256、所有探针与构建日志、`validation.json`、`library-symbols.json`；无 CSV | `607e06350703a5eb430e6c403e9359cb7bdb7dd7f7021e1939e006eccc0cf081` |
