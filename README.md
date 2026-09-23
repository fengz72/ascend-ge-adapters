# Ascend GE Adapters

把客户模型（3 种形态）搬到华为 Ascend NPU 的 GE 上**高效运行**（2 种后端），并内建精度/性能验证的 onboarding 管线。

```
Source ──[Adapt]──> Graph(AIR|ONNX) ──[Passes]──> Backend ──> outputs ──> compare(golden)
  ①hub名/②torch/③onnx    + io_spec        ATC 装 pass    OM/ACL | GeSession
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
| `tests/` | 回归门：`test_*.py`(纯 CPU, pytest) + `tiny_e2e`/`tiny_onnx_e2e`/`smoke`(需 NPU, 脚本) |
| `third_party/` | 三方源，一律**只读**：`custom_development_code`(submodule, fusion pass) + `ascend-ops`(vendored, PIA 自定义算子) + `nlohmann/json.hpp` |
| `docs/` | 架构设计 |

## 三份契约

| 契约 | 文件 | 内容 |
|---|---|---|
| **io_spec** | `air/<name>.io_spec.json` | 图接口：node/logical/dtype/format + 动态维 `-1`；**按图 Data 序** |
| **bundle** | `verification/bundle.json` | 一组具体输入：**具体 shape** + .bin 路径 + golden + provenance；**按 forward 序** |
| **manifest** | `deploy/manifest.json` | 运行时入口：backend / 图与 OM 路径 / io_spec / device / bundle（路径相对，可移植） |

两侧靠 **logical 名**配对（不按位置）——图 Data 序 ≠ forward 入参序（实测，见 docs §15）。

## 快速开始

```bash
# 0) 环境 (CANN + vendor 算子 + PYTHONPATH)
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh

# 1) 构建 C++ 运行时 (一次性)
bash runtime/build.sh                       # → runtime/build/ge_runtime

# 2) 跑一个模型的全链路 (导出→ATC→执行→比对)
./models/qwen2.5-0.5b/run.sh --device 6 --batch-size 2 --seq-len 16

# 3) 跑变体 (model.yaml 的 variants: 段, 只写与基线的差异)
./models/qwen2.5-0.5b/run.sh --device 6 --variant prefix

# 4) 只跑执行 + 比对 (复用已有 AIR/OM/bundle)
./models/qwen2.5-0.5b/run.sh --device 6 --skip export,passes,compile --warmup 10 --bench 100

# 5) 变长负载性能测试 (多实例 + 请求池回放 → 归档报告)
python3 -m core.bench --scenario models/qwen2.5-0.5b/bench/varlen.yaml --device 6
#   → models/qwen2.5-0.5b/results/<run_id>/{run,perf,accuracy}.json + perf.md/accuracy.md

# 6) 吞吐 / 限核 / profiling
./runtime/build/ge_runtime models/qwen2.5-0.5b/deploy/manifest.json --device 6 \
    --sweep 1,2,4,8 --requests 800
./runtime/build/ge_runtime <manifest> --device 6 --bench 20 --profiling --profiling_output ./prof
python3 tools/parse_profiling.py parse-and-export --profiling_dir ./prof

# 7) 直接调 C++ 运行时 (部署态, 无 bundle)
./runtime/build/ge_runtime <manifest> --device 6 \
    --input actual_seq_lengths:2:/path/asl.bin --input input_ids:32:/path/ids.bin \
    --input position_ids:32:/path/pos.bin
```

`ge_runtime` 的完整选项：`--output_dir --input --device --warmup --bench --threads --requests
--sweep --bench-plan --graph_run_mode --precision_mode --aicore_num --output_reserve --dump*
--profiling*`（`--help`）。性能/精度报告的格式与归档见 `models/<model>/results/README.md`。

## 测试

```bash
pytest                                      # 纯 CPU 单测 (秒级, 不需 NPU/torch_npu)
python3 tests/tiny_e2e.py --device 6        # 极小模型全链路, 两后端 (几百 MB 显存)
python3 tests/tiny_onnx_e2e.py --device 6   # 形态③ ONNX → ATC(fw=5) → OM → 部署态执行
python3 tests/smoke.py --device 6               # qwen2.5-0.5b 真实权重全链路
```

约定：`tests/test_*.py` = pytest 收集（CI 可跑）；其余 `tests/*.py` = 需 NPU 的脚本，手动跑。

## 依赖

见 `requirements.txt`（Python）与 `runtime/CMakeLists.txt`（C++：CANN 的 `ascendcl` / `ge_*` / `graph*`，
C++17，`-D_GLIBCXX_USE_CXX11_ABI=0`）。GE 在线路径额外要求把本机 site-packages 注入 `PYTHONPATH`
（tbe pywrapper 用内嵌 python3，缺 numpy 会让 `GEInitialize` 返回 -1）——`models/*/env.sh` 已处理。
