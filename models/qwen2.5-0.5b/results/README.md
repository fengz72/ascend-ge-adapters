# results — 性能基线 (curated) + 本地产物

**两类东西，别混**：

| | 入库? | 内容 |
|---|---|---|
| **本文件 `README.md`** | ✅ 跟踪 | curate 后的**性能基线**：背书的数字 + provenance + 复现口径 |
| `<run_id>/`、`index.json`、`sweep-*.md` | ❌ gitignore | 每次 `core.bench` / `sweep.py` 的**过程产物**，可重跑再生，只作本地参考 |

理由：run 目录是机器生成、可复现的过程数据（`perf.md` 还带本地绝对路径、`index.json` 无限增长会撞
merge 冲突）——入库只会积累陈旧产物（曾有 run 指向已死的 `/tmp` manifest，就是教训）。**调试完把最终
数字 curate 进本文件即可**，原始 run 目录留本地。这与仓库「配置只描述当前形态、历史靠 git」的哲学一致
（docs/architecture.md §5.2）。

**不含精度**：精度由 `run.sh` 的两道门负责（`io/reference.json` + 门② compare，见 docs §10）。

---

## 当前基线

> 在空闲卡上重跑后，用新数字覆盖本节（旧数字靠 git 历史留存）。

**qwen2.5-0.5b · prefix+prune 形态 · ge_session**

| 项 | 值 |
|---|---|
| **QPS** | **178.12 req/s** |
| e2e avg / p50 / p99 / max | 22.43 / 22.49 / 23.72 / 24.74 ms |
| 负载 | instances=4 · requests=2000 · warmup=50 · 池 200 套 / 150 种 shape |
| HBM | 374 → 1323 MB（建 4 实例前 → 后） |
| errors | 0 |
| 形态 | prefix=true · prune=target_tokens.json · batch=10 · seq=208 · prefix_len=20 · fp16 |
| 环境 | device 8 · soc Ascend910_9382 · git `d5eb115` · CANN `9.0.0` · torch_npu `2.9.0.post2` · transformers `5.10.1` |
| 时间 | 2026-10-07 |

两后端资源模型对比（ACL vs GeSession 的 HBM / 启动代价）见 docs/architecture.md §10「多实例的资源模型」。

---

## 怎么产生一份 run（本地）

```bash
# 1) 先有产物 (AIR/OM/manifest/bundle)
./models/qwen2.5-0.5b/run.sh --device 8

# 2) 跑性能测试 (请求池由 model.yaml 的 bench.pool 脚本按分布生成, 落 io/pool/, 幂等)
python3 -m core.bench --config models/qwen2.5-0.5b/config/model.yaml --device 8
#    覆盖: --instances 8 --requests 4000 --warmup 50
#    → results/<run_id>/{run,perf}.json + perf.md + perf_requests.csv (全本地, gitignored)

# 3) 并发档位扫描 (逐档起进程, 每档一份 run 目录 + 一张汇总表)
python3 tools/sweep.py --config models/qwen2.5-0.5b/config/model.yaml --device 8 --levels 1,2,4,8
#    → results/sweep-<ts>.md (本地): instances / QPS / 加速比 / 并行效率 / e2e avg·p99 / HBM

# 4) 满意后, 把 perf.md 的最终数字 + provenance curate 进上面「当前基线」节
```

run 目录内容（本地参考，均 gitignored）：

```
<时间戳>-<git短sha>-<model>-bench/   # run_id
├── run.json          # 快照: git/CANN/torch_npu 版本、device、soc、model.yaml 全文、性能摘要
├── perf.json         # C++ 出的性能数据 (聚合 + 每实例 + 阶段耗时 + HBM)
├── perf.md           # 人读表 (聚合 / 每实例 / 口径说明) ← curate 的素材
├── perf_requests.csv # 逐请求明细: instance,req,input_set,tokens,shape_key,first_hit,h2d/desc/exec/e2e_ms,error
└── plan.json         # 交给 C++ 的 bench plan
```
