# results — 性能/精度报告归档

每次 `python3 -m core.bench --scenario …` 产出一个 run 目录，**json + md 入库**（小、可 diff、
可归档），`raw/`、`perf_requests.csv`、`plan.json` 被 gitignore（体积大或可重新生成）。

```
results/
├── index.json                       # 历次 run 一行摘要 (趋势/归档索引)
└── <时间戳>-<git短sha>-<scenario>/   # run_id
    ├── run.json          # 快照: git/CANN/torch_npu 版本、device、soc、scenario 全文、性能摘要
    ├── perf.json         # C++ 出的性能数据 (聚合 + 每实例 + 阶段耗时 + HBM)
    ├── perf.md           # 人读表 (聚合 / 每实例 / 口径说明)
    ├── accuracy.json     # 精度: cosine / rel_l2 / max_abs / 门限 / 判定
    ├── accuracy.md
    ├── perf_requests.csv # 逐请求明细 (gitignore): instance,req,input_set,tokens,shape_key,
    │                     #   first_hit,h2d_ms,desc_ms,exec_ms,e2e_ms,error
    ├── plan.json         # 交给 C++ 的 bench plan (gitignore)
    └── raw/              # 精度跑的输出 .bin / outputs.json (gitignore)
```

## 怎么产生一份

```bash
# 1) 先有产物 (AIR/OM/manifest/bundle)
./models/qwen2.5-0.5b/run.sh --device 8

# 2) 跑性能测试 (请求池由 scenario 的 generate 脚本按分布生成, 幂等)
python3 -m core.bench --scenario models/qwen2.5-0.5b/bench/varlen.yaml --device 8
#    覆盖: --instances 8 --requests 4000 --warmup 50 ; 只测性能: --skip-accuracy
#    产物在别处时: --manifest <path>/deploy/manifest.json
```

## 已归档的样例 run

| run_id | backend | 说明 |
|---|---|---|
| `20260923T103135-…-varlen` | om_acl | 开发机样例：device 8、2 实例、100 请求、lognormal 池（200 套 / 138 种 shape，batch=10、T≈1273–1694）。**manifest 指向当时的 /tmp 产物**，仅作格式与量级参考，不是正式基线 |
| `20260923T103313-…-varlen` | ge_session | 同上负载，60 请求。可与上一条直接对比两后端 |

两条样例的结论（同负载、2 实例）：QPS 107.7 vs 107.2、e2e avg 18.54 vs 18.59 ms —— 打平；
但 **HBM 每实例 1212MB(ACL) vs 604MB(GE)**，**启动 2.9s/实例(ACL) vs 22.2s/实例(GE，含
每份图 ~10s 的 CompileGraph)**。正式基线请在空闲卡上用 `run.sh` 产出的产物重跑并覆盖此表。
