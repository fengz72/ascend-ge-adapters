# 性能报告 — qwen2.5-0.5b-varlen

- run_id: `20260923T103313-78c9f09-qwen2.5-0.5b-varlen`
- backend: **ge_session** · device: 8 · soc: Ascend910_9382
- manifest: `/tmp/opencode/e2e/deploy/manifest_ge.json`
- 负载: instances=2 · requests=60 · warmup=50 · seed=0
- 请求池: `/export/home/ext.hejun23/workspace/ascend-ge-adapters/models/qwen2.5-0.5b/bench/requests` — 200 套 / 138 种 shape
- 环境: git `78c9f09044939b4d26bc4b36d236130c2fbe67bb` · CANN `cann-9.0.0` · torch_npu `2.9.0.post2`

## 聚合

| 指标 | 值 |
|---|---|
| wall | 559.946 ms |
| **QPS** | **107.15 req/s** |
| e2e avg / p50 / p99 / max | 18.585 / 18.555 / 20.412 / 20.412 ms |
| e2e min | 16.861 ms |
| HBM (建实例前 → 后) | 50751 → 51958 MB |
| warmup | 276 次 / 3212.0 ms |
| errors | 0 |

## 每实例

| instance | req | QPS | e2e avg | e2e p99 | exec avg | exec p99 | h2d avg | desc avg | load ms | shapes | 特化 ms | err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| instance_0 | 30 | 53.58 | 18.635 | 20.360 | 18.557 | 20.248 | 0.076 | 0.000 | 22231.9 | 138 | 1730.2 | 0 |
| instance_1 | 30 | 53.58 | 18.534 | 20.412 | 18.467 | 20.351 | 0.066 | 0.000 | 21950.2 | 138 | 1462.5 | 0 |

## 口径

- `e2e` = h2d + desc(重设该请求 shape) + execute+sync；`exec` 只含 execute+sync。
- 每实例 = 一份独立加载的模型（ACL 每实例独立 `aclmdlLoadFromFile`；GeSession 单 Session 多图，每实例一份 `CompileGraph`+`LoadGraph`），1 worker ↔ 1 实例，无锁。
- `特化` = 该实例在 **warmup 段**首次命中各 shape 的 exec 耗时之和（ACL 是 tiling 缓存建立，GeSession 还含图特化，两者都显著）；warmup 覆盖 每实例 × 每 shape，故测量段不再付这笔 —— 这也意味着**请求池的 distinct shape 数直接决定 warmup 成本**。
- 测量段若出现 warmup 未覆盖的 shape，csv 里 `first_hit=1` 且 stderr 会 WARN（说明延迟被特化污染）。
- 请求池由模型侧脚本按分布生成（语义自洽由模型侧保证），各实例用 `seed + instance_id` 独立随机抽样，可复现。
- 精度不在本报告内（见 `accuracy.md`）：性能跑不落盘输出，避免 D2H 污染延迟。
