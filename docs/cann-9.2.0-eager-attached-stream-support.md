# A5 CANN 9.2.0-beta.2 辅流接口核验

核验日期：2026-10-21。

**结论：`850-a5` 上 `hejun-cann910-stream` 容器的实际安装包没有 `gert::EagerOpExecutionContext::RequestAttachedStream()`，无法直接使用上游 `eager_attached_stream_add` 示例的 GE 托管辅流接口。**

| 项目 | 实际值 |
|---|---|
| 容器 | `hejun-cann910-stream`，running |
| Docker 镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-beta.2-950-openeuler24.03-py3.12` |
| CANN 目录 | `/usr/local/Ascend/cann-9.2.0-beta.2` |
| Toolkit | `9.2.0-beta.2`，`V100R001C12B086` |
| GE Executor / Compiler / Metadef / Runtime | 均为 `9.2.0-beta.2`，时间戳 `20260826_163105218` |
| 证据目录 | `/home/h00909914/work/eager-attached-stream-check-9.2.0-20261021` |

## 附录：检查与编译结果

| 检查 | 结果 |
|---|---|
| 安装头文件中 `RequestAttachedStream()` | 无声明 |
| 安装头文件中 `GetStream()` | 有声明 |
| 安装头文件中 `MallocReadOnlyDevArgs()`、`kArgsHandler` | 均存在，比此前 9.1.0 环境多出这些接口或槽位；不能据此认为已支持辅流 |
| 共享库动态符号扫描 | `lib64` 递归去重后 204 个文件，全部扫描成功，无 `RequestAttachedStream` 匹配 |
| 基础接口符号 | `liblowering.so` 导出 `GetStream()`、`MallocReadOnlyDevArgs()` |
| 底层模型绑定符号 | 有 `aclmdlRIBindStream`，不足以替代 GE 托管辅流 API |
| `GetStream()` 最小探针编译 | 返回码 0，成功 |
| `RequestAttachedStream("eager_aux")` 最小探针编译 | 返回码 1，失败：类没有该成员 |
| 原始示例构建或 NPU 执行 | 未执行；目标接口的最小编译探针已失败 |

探针仅使用安装包原始头文件，未修改系统组件。命令：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I"$ASCEND_HOME_PATH/include" -c probe-attached-stream.cc
```

```cpp
#include "exe_graph/runtime/eager_op_execution_context.h"
gert::rtStream probe(gert::EagerOpExecutionContext *ctx) {
  return ctx->RequestAttachedStream("eager_aux");
}
```

核心报错：

```text
error: ‘class gert::EagerOpExecutionContext’ has no member named ‘RequestAttachedStream’
```

结论只覆盖当前实际安装的 **9.2.0-beta.2 B086**，不能外推至其他 9.2.0 构建或定制补丁包。上游接口来源仍采用前次固定的 GE 提交 `28aede1f69af4f15b1432ccf6ba9e29db6f9cfb7`，本次没有重新解析变化中的 master。

证据包 `cann-9.2.0-eager-attached-stream-evidence-20261021.zip` 含组件信息、安装头文件快照、最小探针、编译日志、动态符号扫描结果及文档，无 CSV。

安装头文件 SHA-256：`c929ebacc00e7131c6c0e8429e49e95401ddd6d90f2d3c7a5549cc43e7560559`。
