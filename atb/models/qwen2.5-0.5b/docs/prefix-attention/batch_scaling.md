# PIA batch × thread 扩展性测试 (GESession 12|24)

> 测试日期: 2026-09-16
> 测试环境: Ascend910_9382, device 12 (空闲), host load ~38, CANN 9.0.0
> 模型: qwen2.5-0.5b-prefix (prefix-attention 分支, AIR 在线)
> 算子: npu_prefix_infer_attention_score commit `8e79993` (prefix-merged, KV 内嵌版)
> 精度: 清理 JIT 缓存后逐位一致 (见 benchmark.md)
> 测试工具: `atb/build/bench_ge_latency --batch-size N` (新增参数)

## 1. 测试目标

PIA 的 token 节省 = `(batch-1)×P` (prefix 只算一份), batch 越大节省越多。
测试 batch 从 10 到 32、thread 1-6 时 QPS/延迟的变化, 找到延迟约束下的最优
(batch, thread) 组合。

## 2. 工作负载与条件

| 项 | 值 |
|---|---|
| seq_len | lognormal(4.997, 0.167) clip [1,218] — avg≈150, p99≈218 (随机) |
| prefix P | 每请求均匀随机 [20, 25] |
| batch_size | 10, 12, 16, 20, 24, 28, 32 (`--batch-size N`) |
| 线程数 sweep | 1, 2, 3, 4, 5, 6 (独立线程闭环, 每线程独立图实例) |
| 限核 | 12\|24 (AIC\|AIV 1:2) |
| requests / warmup | 8000 / 50 (全部 batch 档均 8000 请求) |

## 3. 原始数据

### 3.1 QPS (req/s)

| batch\thread | 1T | 2T | 3T | 4T | 5T | 6T |
|---|---|---|---|---|---|---|
| 10 | 76.26 | 140.47 | 163.22 | **167.97** | 167.05 | 165.73 |
| 12 | 65.23 | 123.81 | 138.56 | **148.08** | 135.24 | 142.04 |
| 16 | 50.34 | 90.78 | 106.05 | **109.37** | 109.13 | 108.86 |
| 20 | 42.22 | 75.90 | 87.93 | **90.10** | 89.93 | 89.09 |
| 24 | 38.82 | 68.52 | 75.45 | 76.59 | 75.74 | **78.42** |
| 28 | 31.65 | 56.76 | 64.68 | **67.55** | 66.59 | 63.86 |
| 32 | 29.15 | 54.21 | **61.75** | 60.95 | 59.10 | 57.79 |

(粗体 = 该 batch 的 QPS 峰值; 全部 batch 均为 8000 请求)

### 3.2 E2E 延迟 avg (ms)

| batch\thread | 1T | 2T | 3T | 4T | 5T | 6T |
|---|---|---|---|---|---|---|
| 10 | 13.11 | 14.22 | 18.37 | 23.80 | 29.92 | 36.19 |
| 12 | 15.33 | 16.14 | 21.61 | 27.00 | 36.02 | 41.19 |
| 16 | 19.86 | 22.03 | 28.15 | 36.55 | 45.78 | 54.94 |
| 20 | 23.68 | 26.33 | 34.10 | 44.38 | 55.58 | 67.32 |
| 24 | 25.76 | 29.18 | 39.74 | 52.17 | 65.97 | 76.48 |
| 28 | 31.59 | 35.22 | 46.36 | 59.17 | 75.01 | 93.91 |
| 32 | 34.30 | 36.85 | 48.49 | 65.60 | 84.56 | 103.79 |

### 3.3 E2E 延迟 p99 (ms)

| batch\thread | 1T | 2T | 3T | 4T | 5T | 6T |
|---|---|---|---|---|---|---|
| 10 | 14.25 | 16.92 | 19.75 | 25.28 | 31.51 | 37.91 |
| 12 | 21.18 | 18.33 | 46.38 | 29.47 | 93.33 | 99.31 |
| 16 | 22.30 | 24.82 | 38.38 | 40.00 | 49.25 | 89.94 |
| 20 | 26.55 | 29.47 | 37.77 | 48.39 | 59.64 | 71.52 |
| 24 | 28.22 | 36.30 | 44.39 | 56.96 | 70.91 | 83.01 |
| 28 | 35.01 | 39.27 | 51.20 | 66.08 | 123.64 | 99.61 |
| 32 | 52.00 | 50.75 | 52.91 | 70.46 | 89.84 | 109.35 |

### 3.4 物理 Token 吞吐 (tok/s, 峰值档)

| batch | 峰值 Token/s | 峰值线程 |
|---|---|---|
| 10 | 217,737 | 4T |
| 12 | 229,688 | 4T |
| 16 | 225,443 | 4T |
| 20 | 231,522 | 4T |
| 24 | 241,789 | 6T |
| 28 | 242,361 | 4T |
| 32 | 253,138 | 3T |

## 4. 分析

### 4.1 Token 吞吐饱和

物理 Token 吞吐在 batch 10-28 区间稳定在 **~220-240k tok/s**, 设备算力是瓶颈。
QPS 随 batch 增大而下降, 纯粹因为单请求 token 变多 — 不是效率降低。

### 4.2 最优线程数随 batch 递减

| batch 范围 | 最优线程 | 解释 |
|---|---|---|
| 10-20 | 4T | 标准 12\|24 限核饱和点 |
| 24 | 4-6T | 接近饱和, 6T 略优 |
| 28 | 4T | 5T+ 开始过饱和 |
| 32 | **3T** | 大 batch 单请求占满更多算力, 3T 即饱和; 4T+ QPS 反降 |

### 4.3 E2E 延迟线性增长

E2E avg 在任意线程数下近似随 batch 线性增长 (batch 每增 4, 延迟约增 20-30%),
因为单请求 token 量 = P + batch×own 线性增长。

### 4.4 batch=32 的过饱和现象

batch=32 在 4T+ 出现 QPS 下降 (61.75→60.95→59.10→57.79), 额外并发只增加排队延迟。
这是 12\|24 限核下的单请求粒度过大 — 单请求已占用足够多核, 多线程并发无法并行加速。
batch=32 的 Token 吞吐峰值 253k tok/s (3T) 为全 batch 最高。

## 5. 延迟约束下的最优 (batch, thread) 组合

| 延迟 SLO | 推荐配置 | QPS | 语义 tok/s |
|---|---|---|---|
| ≤15ms | batch=10, 1T | 76.3 | ~114k |
| ≤20ms | batch=10, 2T | 140.5 | ~211k |
| ≤25ms | batch=10, 3T | 163.2 | ~245k |
| ≤35ms | batch=10, 4T | **168.0** | **~252k** |
| ≤50ms | batch=20, 2T | 75.9 | ~227k |
| 无约束 | **batch=10, 4T** | **168.0** | **~252k** |

(语义 tok/s = QPS × batch × avg_seq_len, 即用户视角的 token 处理量)

## 6. 结论

1. **batch=10 + 4T 是 12|24 限核下的 QPS 最优组合** (168 QPS, 语义 ~252k tok/s)
2. **增大 batch 不提升总吞吐** (设备已饱和在 ~230k tok/s), 只增加单请求延迟
3. **大 batch 的价值在于语义吞吐** (prefix 节省比例 (B-1)×P/(B×seq) 随 B 增大),
   但在当前 P=20-25 与 seq~150 的比例下, 语义增益有限 (~1.7% 从 B=10 到 B=32)
4. 如需提高单请求吞吐 (大 batch), 需放宽延迟 SLO 或增加核数

## 7. 工具改动

`bench_ge_latency` 新增 `--batch-size N` 参数 (默认 10, 上限 40):
- `BATCH_FIXED_VAL` 改为可变 (由 CLI 覆盖)
- `MAX_BATCH_SIZE` 11→40, `MAX_TOTAL_TOKENS` 相应扩大

## 8. 复现命令

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source $ASCEND_HOME_PATH/opp/vendors/custom_prefix_attn/bin/set_env.bash
export PYTHONPATH=/usr/local/python3.11.15/lib/python3.11/site-packages:$PYTHONPATH

for BS in 10 12 16 20 24 28 32; do
    ./build/bench_ge_latency \
        --model air/qwen2.5-0.5b-prefix.air \
        --sweep 1,2,3,4,5,6 \
        --requests 8000 --warmup 50 \
        --prefix 20-25 --aicore-num 12 \
        --batch-size $BS \
        --device-id 12
done
```

原始日志: `/tmp/opencode/clean_pia_12c.log` (batch=10),
`/tmp/opencode/bs20.log`, `/tmp/opencode/bs24.log`, `/tmp/opencode/bs28.log`,
`/tmp/opencode/bs32_r2a.log` (batch=32 1-3T), `/tmp/opencode/bs32_r2b.log` (batch=32 4-6T)
