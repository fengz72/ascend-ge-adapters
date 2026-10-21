# A5 CANN 9.1.0 Eager 辅流功能核验

核验日期：2026-10-21。

## 结论

**当前 A5 容器安装的 CANN 9.1.0（B243）不能直接使用 `eager_attached_stream_add` 示例所展示的框架托管辅流功能。** 已在目标容器完成头文件检查、动态符号扫描、最小探针编译以及上游原始示例构建。关键接口 `gert::EagerOpExecutionContext::RequestAttachedStream()` 的声明和导出实现均缺失，原始示例因此构建失败。

结论针对本次实际安装的组件组合。没有据此推断所有名为 9.1.0 的定制补丁包，也没有验证某一更高版本安装包已支持该功能。

## 上游示例实现

用户指定的[示例目录](https://gitcode.com/cann/ge/tree/master/examples/custom_op/eager_attached_stream_add)通过 Git 获取并阅读；网页抓取工具无法访问该页面。为避免 master 后续变化，固定读取提交：

`28aede1f69af4f15b1432ccf6ba9e29db6f9cfb7`，提交时间 `2026-10-20 15:31:55 +0800`。

示例执行链路如下：

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

该固定提交也包含 `Rt2AttachedStreamCollection` 和 RT2 provider 接线，不能把当前 master 的支持范围概括成“只有 V1”。本次链接中的在线示例自身验证的是 V1 下沉、重放链路，未在当前 9.1.0 环境验证 RT2。

## A5 实际环境

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

## 附录：实际验证结果

所有构建均先 source 当前安装包的 `set_env.sh`，使用安装包自带头文件、库和 `_GLIBCXX_USE_CXX11_ABI=0`。上游示例原样复制到隔离目录；未修改系统安装或现有 Prefix 算子。

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

最小负向探针：

```cpp
#include "exe_graph/runtime/eager_op_execution_context.h"
gert::rtStream probe(gert::EagerOpExecutionContext *ctx) {
  return ctx->RequestAttachedStream("eager_aux");
}
```

最小探针命令：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I"$ASCEND_HOME_PATH/include" -c probe-attached-stream.cc
```

原始示例构建命令：

```bash
cmake -S upstream-example -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

### 对 Prefix Attention 优化的意义

当前 A5 Prefix 实现使用 `ctx->GetStream()`，依次提交 metadata 清零、AICPU planner 和 Attention kernel。当前安装包支持这一既有路径；要通过示例 API 改成框架托管辅流，需要先提供完整匹配的 GE/Metadef 组件及依赖，或对当前版本作完整后移植。不能仅更新算子代码或头文件。

即便支持辅流，也不能由此直接推导吞吐提升：planner 生成的 metadata 是 Attention 的输入，两者之间仍有依赖，移到辅流不会自动消除这段耗时。需要让独立工作发生重叠，或减少重复规划，再用 profiling 验证收益。

示例在源码中明确假定单模型，其 event 缓存在按算子类型复用的实例中。若将方案用于当前多实例并发业务，还需按模型或执行实例隔离同步资源，并验证不同输入、多实例重放、销毁与重载。示例的 10 轮校验不能替代这些验证。

后续可在独立容器中验证包含完整接口与 provider 支持的匹配组件版本。当前未安装或测试其他版本，因此没有把“升级至 9.3.0 即可”作为已验证结论。

### 证据文件

`cann-9.1.0-eager-attached-stream-evidence-20261021.zip` 保存固定提交的示例源码、相关上游接口与 provider 源码、安装头文件快照、组件信息、SHA-256、所有探针与构建日志、`validation.json` 和 `library-symbols.json`。没有 CSV 文件。

安装头文件 SHA-256：`607e06350703a5eb430e6c403e9359cb7bdb7dd7f7021e1939e006eccc0cf081`。
