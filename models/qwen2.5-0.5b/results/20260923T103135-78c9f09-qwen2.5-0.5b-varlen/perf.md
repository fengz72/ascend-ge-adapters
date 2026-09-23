# 性能报告 — qwen2.5-0.5b-varlen

- run_id: `20260923T103135-78c9f09-qwen2.5-0.5b-varlen`
- backend: **om_acl** · device: 8 · soc: Ascend910_9382
- manifest: `/tmp/opencode/e2e/deploy/manifest.json`
- 负载: instances=2 · requests=100 · warmup=50 · seed=0
- 请求池: `/export/home/ext.hejun23/workspace/ascend-ge-adapters/models/qwen2.5-0.5b/bench/requests` — 200 套 / 138 种 shape
- 环境: git `78c9f09044939b4d26bc4b36d236130c2fbe67bb` · CANN `cann-9.0.0` · torch_npu `2.9.0.post2`

## 聚合

| 指标 | 值 |
|---|---|
| wall | 928.689 ms |
| **QPS** | **107.68 req/s** |
| e2e avg / p50 / p99 / max | 18.541 / 18.589 / 20.198 / 20.198 ms |
| e2e min | 17.179 ms |
| HBM (建实例前 → 后) | 50749 → 53173 MB |
| warmup | 276 次 / 3415.2 ms |
| errors | 0 |

## 每实例

| instance | req | QPS | e2e avg | e2e p99 | exec avg | exec p99 | h2d avg | desc avg | load ms | shapes | 特化 ms | err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| instance_0 | 50 | 53.84 | 18.559 | 20.037 | 18.490 | 19.966 | 0.065 | 0.003 | 2897.2 | 138 | 1832.4 | 0 |
| instance_1 | 50 | 53.84 | 18.523 | 20.198 | 18.454 | 20.123 | 0.064 | 0.003 | 2881.1 | 138 | 1564.2 | 0 |

## 口径

- `e2e` = h2d + desc(重设该请求 shape) + execute+sync；`exec` 只含 execute+sync。
- 每实例 = 一份独立加载的模型（ACL 每实例独立 `aclmdlLoadFromFile`；GeSession 单 Session 多图，每实例一份 `CompileGraph`+`LoadGraph`），1 worker ↔ 1 实例，无锁。
- `特化` = 该实例**首次**命中某 shape 的请求 exec 耗时之和（GeSession 显著，ACL≈0）；warmup 已覆盖 每实例 × 每 shape，测量段一般不再付这笔。
- 请求池由模型侧脚本按分布生成（语义自洽由模型侧保证），各实例用 `seed + instance_id` 独立随机抽样，可复现。
- 精度不在本报告内（见 `accuracy.md`）：性能跑不落盘输出，避免 D2H 污染延迟。
