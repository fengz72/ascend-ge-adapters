# 精度报告

| 项 | 值 |
|---|---|
| 输出 | logits |
| shape | [2, 151936] |
| dtype | float16 |
| golden | `/tmp/opencode/e2e/verification/bundle.json` |
| cosine | 0.99995719 |
| relative_l2 | 9.206946e-03 |
| max_abs | 1.367188e-01 |
| 门限 | cosine > 0.9999 且 rel_l2 < 0.01 |
| **判定** | **PASS** |

> 单请求、与性能跑分开（性能跑不落盘输出）。golden = NPU-eager，比对口径见 `tools/compare.py`；shape 不一致直接判失败，不做 flatten/截断。
