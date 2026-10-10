# tools

通用工具脚本（不依赖具体模型结构）：

| 文件 | 用途 |
|---|---|
| `varlen.py` | varlen 输入生成（token 拼接 + position ids，基线/prefix 两形态）+ **落盘/读回 schema**（`write_inputs`/`load_inputs`，输入脚本与框架共用）+ **结构不变量自检**（`check_varlen`：`asl` 须 cumsum、`position` 每段从起点递增、`asl[-1]==T`） |
| `atc_utils.py` | ATC 编译（AIR → OM）。`build_atc_argv` 是纯函数（argv 可打印复核）、`run_atc` 起子进程；`normalize_aicore` 归一限核规格并给出 OM 文件名后缀。**不含 fusion pass 安装** —— pass/自定义算子归 `core/setup_scripts.py` + `models/<m>/scripts/install_*.sh`（装进 `opp/vendors/` 后由 CANN 自动扫描，ATC 侧无需任何开关） |
| `compare.py` | 精度比对。**比对数学的唯一定义处** —— `core/verify.py` 的两道门都调它，不自己重写指标。判定只看 cosine + relative L2（逐元素统计仅参考显示）。库侧不做 shape 兜底，CLI 侧 flatten+截断但打显式 WARN（两套策略刻意不同，见模块 docstring） |
| `sweep.py` | 并发档位扫描（逐档调 `core.bench`，每档一个 `ge_runtime` 子进程；汇总 scaling 表） |
| `parse_dump.py` | CANN ACL dump 数据解析（protobuf v2.0 → npy → list/summary/show/compare/diff/plot）。**仅 `om_acl` 后端**：dump 经 `acl.json` + `aclInit(configPath)` 生效，`ge_session` 下给了 `--dump*` 会 WARN 并忽略（`runtime/main.cpp`），而 `model.yaml` 的默认后端是 `ge_session` —— 要用先把 `backend.type` 改成 `om_acl` |
| `parse_profiling.py` | CANN profiling 数据解析（msprof 封装 + CSV/JSON 分析）。两后端都支持；三个入口（`ge_runtime` / `core.pipeline` / `core.bench`）都有 `--profiling*` 开关，`core.backend.parse_profiling` 是共用的解析封装 |

Python 引用: `from tools.varlen import generate_varlen_inputs`（仓库根在 PYTHONPATH 时）。

`compare.py` / `parse_dump.py` / `parse_profiling.py` 都可独立当 CLI 跑（`--help`）；
`varlen.py` / `atc_utils.py` / `sweep.py` 是库（`sweep.py` 也有 CLI）。
