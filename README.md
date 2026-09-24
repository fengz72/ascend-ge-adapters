# Ascend GE Adapters

把客户 torch 模型搬到华为 Ascend NPU 的 GE 上**高效运行**（2 种后端），并内建精度/性能验证的 onboarding 管线。

```
Source ──[Adapt]──> Graph(AIR) ──[Passes]──> Backend ──> outputs ──> compare(golden)
  ①torch            + io_spec    ATC 装 pass    OM/ACL | GeSession
              └── adapt **前**的原版 HF 前向 = reference ──> compare(golden)   # 验"适配"
```

- **设计文档**：[`docs/architecture.md`](docs/architecture.md)（契约、动态 shape 策略、pass 子系统、适配协议、实测结论）
- **逐模型文档**：`models/<model>/docs/`（部署指南 + 实测报告）
- **适配新模型**：照 `models/qwen2.5-0.5b/model.py` 复制修改 + 写 `config/model.yaml`，框架零改动

## 目录

| 路径 | 职责 |
|---|---|
| `core/` | 通用框架（Python）：config / source / adapter / exporter / graph / passes / backend / verify / pipeline |
| `runtime/` | 通用执行运行时（C++）：单入口 + manifest 分发 + `acl_backend`(OM) / `gesession_backend`(AIR) + bench |
| `models/<name>/` | 逐模型：`model.py`(Adapter) + `config/model.yaml`(声明) + 产物 + docs |
| `tools/` | varlen 输入生成 / ATC 封装 / 精度比对 / dump·profiling 解析（见 `tools/README.md`） |
| `tests/` | 回归门：`tiny_e2e`（需 NPU 的脚本，极小模型跑通两后端闭环） |
| `third_party/` | 三方源，一律**只读**：`custom_development_code`(submodule, fusion pass) + `ascend-ops`(submodule, PIA 自定义算子) + `nlohmann/json.hpp` |
| `docs/` | 架构设计 |

## 三份契约

| 契约 | 文件 | 内容 |
|---|---|---|
| **io_spec** | `air/<name>.io_spec.json` | 图接口：node/logical/dtype/format + 动态维 `-1`；**按图 Data 序** |
| **bundle** | `io/bundle.json` | 一组具体输入：**具体 shape** + .bin 路径 + golden + provenance；**按 forward 序** |
| **manifest** | `io/manifest.json` | 运行时入口：backend / 图与 OM 路径 / io_spec / device / bundle（路径相对，可移植） |

两侧靠 **logical 名**配对（不按位置）——图 Data 序 ≠ forward 入参序（实测，见 docs §15）。

## 快速开始

```bash
# 0) 环境 (CANN + vendor 算子 + PYTHONPATH)
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 1) 构建 C++ 运行时 (一次性)
bash runtime/build.sh                       # → runtime/build/ge_runtime

# 2) 跑一个模型的全链路 (导出→ATC→执行→比对)
./models/qwen2.5-0.5b/run.sh --device 6 --batch-size 2 --seq-len 16

# 3) 切形态 (改 model.yaml 的 adapt.params: prefix/prune; 产物名自动带后缀, 不互相覆盖)
#    配置只描述"当前形态", 历史形态靠 git — 见 docs/architecture.md §5.2

# 4) 只跑执行 + 比对 (复用已有 AIR/OM/bundle)
./models/qwen2.5-0.5b/run.sh --device 6 --skip export,passes,compile --warmup 10 --bench 100

# 5) 变长负载性能测试 (多实例 + 请求池回放 → 归档报告; 口径在 model.yaml 的 bench 段)
python3 -m core.bench --config models/qwen2.5-0.5b/config/model.yaml --device 6
#   → models/qwen2.5-0.5b/results/<run_id>/{run,perf}.json + perf.md (只出性能; 精度归 run.sh 两道门)
#   覆盖: --instances 8 --requests 4000

# 5b) 并发档位扫描 (逐档起进程, 汇总 scaling 表; 档位循环不进 C++)
python3 tools/sweep.py --config models/qwen2.5-0.5b/config/model.yaml --device 6 --levels 1,2,4,8
#   → models/qwen2.5-0.5b/results/sweep-<ts>.md (QPS/加速比/并行效率/p99/HBM 一张表)

# 6) 吞吐 / 限核 / profiling
./runtime/build/ge_runtime models/qwen2.5-0.5b/io/manifest.json --device 6 \
    --threads 4 --requests 800
./runtime/build/ge_runtime <manifest> --device 6 --bench 20 --profiling --profiling_output ./prof
python3 tools/parse_profiling.py parse-and-export --profiling_dir ./prof
```

`ge_runtime` 的完整选项：`--output_dir --device --warmup --bench --threads --requests
--bench-plan --graph_run_mode --precision_mode --aicore_num --output_reserve --dump*
--profiling*`（`--help`）。**一次只测一档并发**（`--threads`）——档位扫描是"同一件事跑 N 遍"，
归 `tools/sweep.py`（每档独立进程，资源彻底建/销，HBM 归还干净）。
性能/精度报告的格式与归档见 `models/<model>/results/README.md`。

## 测试

```bash
python3 tests/tiny_e2e.py --device 6        # 极小模型全链路, 两后端 (几百 MB 显存, 分钟级)
./models/qwen2.5-0.5b/run.sh --device 6     # 真实模型全链路 (含下面两道精度门)
```

**两道精度门**（都在 `run.sh` 里，各隔离一个变量，FAIL 即非 0 退出）：

| 门 | 比对 | 证明 | 产物 |
|---|---|---|---|
| `reference` | 原版**未 patch** 的 HF 逐请求前向 vs 适配后 eager golden | **适配**正确（融合算子替换、varlen 打包、prefix 语义、末 token 取值、lm_head 剪裁） | `io/reference.json` |
| `compare` | C++ 运行时输出 vs golden | **编译**正确（AIR→OM/GeSession、图序喂入、动态 shape 特化） | `io/outputs/` |

`reference` 必须在 `adapt()` 之前算（patch 是进程级类属性），且需要 adapter 实现
`unpack_requests`（把打包的图输入还原成逐请求）与 `reference_columns`（输出被剪裁时取哪些列）；
未实现则 WARN 跳过。门限比 `compare` 松一档（跨实现，见 `core/verify.py` 的 `REF_*`）。
跳过：`--skip reference`。

## 依赖

见 `requirements.txt`（Python）与 `runtime/CMakeLists.txt`（C++：CANN 的 `ascendcl` / `ge_*` / `graph*`，
C++17，`-D_GLIBCXX_USE_CXX11_ABI=0`）。GE 在线路径额外要求把本机 site-packages 注入 `PYTHONPATH`
（tbe pywrapper 用内嵌 python3，缺 numpy 会让 `GEInitialize` 返回 -1）——`models/*/env.sh` 已处理。
