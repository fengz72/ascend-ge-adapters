# PrefixInferAttentionScore 独立自定义算子工程

prefix-in-Q prefill attention 独立算子交付件。同一请求内多个 batch 共享相同 prefix（如 system prompt）时，prefix 的 Q/KV/输出只存一份、只算一份，减少计算与显存。kernel 基于 vendored FIA 模板链（源仓 `attention/common`，零改动引入），prefix-in-Q 语义由算子自身承载。

**维护指引**：相对源仓 FIA（ops-transformer `origin/9.0.0` @ e88633357）的全部改动清单、关键机制不变量与版本升级重放步骤见 [docs/FIA_DELTA.md](docs/FIA_DELTA.md)。

## 1. 算子语义

```
aclnnPrefixInferAttentionScore(
    query, key, value,                   # 均TND; q=[P+sum(L), N_q, D], k/v=[P+sum(L), N_kv, D]
                                         # (prefix Q/KV在各张量头部, 所有batch共享一份)
    attenMask,                           # 可选; 压缩causal [2048,2048] bool (sparse_mode=2)
    actualSeqLengths,                    # cumsum([P, L0, L1, ...])
    actualSeqLengthsKv,                  # cumsum([P, L0, L1, ...]) 与act_q逐元素相等
    numHeads, scale, numKeyValueHeads, sparseMode, preTokens, nextTokens)
```

- 布局：仅 TND；dtype：NoQuant fp16/bf16；sparse_mode：0 或 2
- `P = act_kv[0]`（无需单独传 prefix 长度）；**batch 0' = [prefix; req0] 合并为单 batch**（q_len=kv_len=P+L₀，方阵 causal，prefix 行恰好只见 prefix 列 → 数学精确等价），batch j'+1 为请求 j（q_len=L_j，kv=prefix区+自身区）
- 约束：P>0，actLen≥2（至少一个请求），act 严格递增（所有 batch 非空），逐 batch q_len==kv_len（act_q 与 act_kv 为相同 cumsum 数组）；压缩 mask 下 P+max(L)≤2048；sparse_mode=0 显式 mask 仅支持共享 mask（2D 或 batch 维=1）
- **act 配置必须与喂入数据的 batch 切分一致**：不匹配时（act 累加值与 T 自洽的情况下）无任何报错、结果静默错误
- 数学：causal 下各 batch 的 prefix 段输出逐位相同 → 只算一份，精确等价；输出 `[P+sum(L), N_q, D]`（prefix 段输出在头部）

## 2. 环境要求（版本基线）

| 项 | 版本 |
|---|---|
| SoC | Ascend910_93（芯片实测 Ascend910_9382） |
| CANN | 9.0.0 |
| OS | openEuler 24.03 aarch64 |
| Python | 3.11.15 |
| torch / torch_npu | 2.9.0+cpu / 2.9.0.post2 |
| 源仓基线 | ops-transformer `origin/9.0.0` @ e88633357（分支基点） |

## 3. 构建与安装

```bash
cd prefix_infer_attention_score
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 设置ASCEND_HOME_PATH等
bash build.sh        # ~4min, 产物 build_out/custom_opp_openEuler_aarch64.run
```

安装（安装后目录为 `$ASCEND_HOME_PATH/opp/vendors/custom_prefix_attn/`）：

```bash
bash build_out/custom_opp_openEuler_aarch64.run --quiet --install-path=$ASCEND_HOME_PATH/opp
```

## 4. 调用方式

安装 wheel（或 `PYTHONPATH` 指向 `torch_binding/`）后 import 即可用：

```python
import npu_prefix_infer_attention_score
out = torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score(
    query, key, value,   # q=[P+sum(L),N_q,D], k/v=[P+sum(L),N_kv,D] (prefix在头部)
    atten_mask=mask,
    actual_seq_lengths=[...], actual_seq_lengths_kv=[...],  # 均为cumsum([P,L0,...])
    num_heads=14, scale=scale, num_key_value_heads=2, sparse_mode=2)
```

wheel 构建：`cd torch_binding && pip wheel . --no-deps -w dist`。


## 5. 性能数据

基线：FIA V1（`torch_npu.npu_fused_infer_attention_score`，TND 重复布局，prefix 逐 batch 重复，q/k/v 为 B×seq 行）。kernel 口径（msprof `Task Duration`，300 次取 min，同 chip 多轮复测，b=10）。

| 场景 | FIA V1 | PIA | FIA/PIA |
|---|---|---|---|
| prefix=20, seq=150, fp16 | 0.056ms  | 0.063ms | **0.89x** |
| prefix=20, seq=150, bf16 | 0.056ms  | 0.060ms | **0.93x** |
| prefix=25, seq=218, fp16 | 0.082ms  | 0.083ms | **0.99x** |
| prefix=25, seq=218, bf16 | 0.082ms  | 0.082ms | **1.00x** |
| 随机负载 lognormal, fp16 | 0.064ms | 0.070ms | **0.91x** |
| 随机负载 lognormal, bf16 | 0.059ms | 0.065ms | **0.91x** |

随机负载场景（对齐模型侧 RequestGenerator）：每请求总长 `sl = round(exp(N(μ=4.997, σ=0.167)))` 截断 [1,218]（中位数≈148/均值≈150），prefix 长度 U[20,25]

## 6 交付清单

| 交付物 | 生成 |
|---|---|
| `*.run`（CANN 算子包） | `bash build.sh` → `build_out/custom_opp_openEuler_aarch64.run` |
| `*.whl`（torch 接入包） | `cd torch_binding && pip wheel . --no-deps -w dist` |

安装顺序：先装 .run（§3），再 `pip install` wheel；验收跑 `tests/test_torch_binding.py`（全 PASS 为准出）。


## 7. 注意事项
- **transformers 5.x eager mask 是加性**（`attn_weights + mask`，0/-inf float），不是 bool masked_fill——构造块对角 mask 必须用加性语义
- **interface 返回约定 [B, T, h, d]**（forward 直接 `reshape(B,T,-1)`）；返回 [B,h,T,d] 时元素数恰好相同 → reshape 静默乱序
- **sdpa 对照勿用 `is_causal=True`**：那是全序列因果（跨请求泄漏），packed 需块对角加性 attn_mask
- **bf16×24层 logits 绝对阈值不可用**：任何两个正确 attention 实现的端到端偏差 ~0.3-0.7（bf16 单 ULP 在 logit=16 处即 0.0625）；验收用 argmax 一致 + 不劣于 sdpa 对照
- decode 未支持（prefill-only）：`use_cache=False`；interface 内校验 `q_len == packed T` 防误用
