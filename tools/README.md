# tools

通用工具脚本（不依赖具体模型结构）：

| 文件 | 用途 |
|---|---|
| `varlen.py` | varlen 输入生成（token 拼接 + position ids，基线/prefix） |
| `atc_utils.py` | ATC 编译（AIR → OM，含 NzWeightPass / 限核支持） |
| `compare.py` | 精度对比（golden vs 实际输出，多种指标） |
| `parse_dump.py` | CANN ACL dump 数据解析（protobuf v2.0 格式 → npy → 分析） |
| `parse_profiling.py` | CANN profiling 数据解析（msprof 封装 + CSV/JSON 分析） |

Python 引用: `from tools.varlen import generate_varlen_inputs`（仓库根在 PYTHONPATH 时）。
