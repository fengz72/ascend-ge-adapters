# results — 性能报告归档

每次 `python3 -m core.bench --config …` 产出一个 run 目录，**json + md 入库**（小、可 diff、
可归档），`perf_requests.csv`、`plan.json` 被 gitignore（体积大或可重新生成）。
**不含精度**：精度由 `run.sh` 的两道门负责（`io/reference.json` + 门② compare，见 docs §10）。

```
results/
├── index.json                       # 历次 run 一行摘要 (趋势/归档索引)
├── sweep-<时间戳>.md                # 并发档位扫描汇总 (tools/sweep.py, 每档仍各有自己的 run 目录)
└── <时间戳>-<git短sha>-<model>-bench/   # run_id
    ├── run.json          # 快照: git/CANN/torch_npu 版本、device、soc、model.yaml 全文、性能摘要
    ├── perf.json         # C++ 出的性能数据 (聚合 + 每实例 + 阶段耗时 + HBM)
    ├── perf.md           # 人读表 (聚合 / 每实例 / 口径说明)
    ├── perf_requests.csv # 逐请求明细 (gitignore): instance,req,input_set,tokens,shape_key,
    │                     #   first_hit,h2d_ms,desc_ms,exec_ms,e2e_ms,error
    └── plan.json         # 交给 C++ 的 bench plan (gitignore)
```

## 怎么产生一份

```bash
# 1) 先有产物 (AIR/OM/manifest/bundle)
./models/qwen2.5-0.5b/run.sh --device 8

# 2) 跑性能测试 (请求池由 model.yaml 的 bench.pool 脚本按分布生成, 落 io/pool/, 幂等)
python3 -m core.bench --config models/qwen2.5-0.5b/config/model.yaml --device 8
#    覆盖: --instances 8 --requests 4000 --warmup 50
#    产物在别处时: --manifest <path>/io/manifest.json

# 3) 并发档位扫描 (逐档起进程, 每档一份 run 目录 + 一张汇总表)
python3 tools/sweep.py --config models/qwen2.5-0.5b/config/model.yaml --device 8 --levels 1,2,4,8
#    → results/sweep-<ts>.md: instances / QPS / 加速比 / 并行效率 / e2e avg·p99 / HBM
```

## 已归档的样例 run

| run_id | backend | 说明 |
|---|---|---|
| `20260923T103135-…-varlen` | om_acl | 开发机样例：device 8、2 实例、100 请求、lognormal 池（200 套 / 138 种 shape，batch=10、T≈1273–1694）。**manifest 指向当时的 /tmp 产物**，仅作格式与量级参考，不是正式基线 |
| `20260923T103313-…-varlen` | ge_session | 同上负载，60 请求。可与上一条直接对比两后端 |

两条样例的结论（同负载、2 实例）：QPS 107.7 vs 107.2、e2e avg 18.54 vs 18.59 ms —— 打平；
但 **HBM 每实例 1212MB(ACL) vs 604MB(GE)**，**启动 2.9s/实例(ACL) vs 22.2s/实例(GE，含
每份图 ~10s 的 CompileGraph)**。正式基线请在空闲卡上用 `run.sh` 产出的产物重跑并覆盖此表。
