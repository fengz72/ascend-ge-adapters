#!/usr/bin/env python3
"""qwen2.5-0.5b 性能测试请求池生成器 — 按分布产 K 套 varlen 输入 (bundle + .bin)。

框架不生成模型专属负载 (语义自洽只有模型侧能保证: asl 必须是 cumsum、position 每段
从 0 递增、asl[-1] == T), 所以这一步是**用户脚本**, 由 bench scenario 的 `generate:`
声明 (core/setup_scripts.py 执行, 传 $GE_SRC_DIR / $GE_ENV_FILE)。

每套请求写成 <out>/req_NNN/{bundle.json, inputs/*.bin}:
    bundle.inputs 按 **forward 入参序** (input_ids, position_ids, actual_seq_lengths),
    logical 名与 io_spec 一致 — C++ 按 logical 名配对 (docs §5.4)。
    bundle.golden 只写 shape ([N, vocab], 无文件): 供运行时推导输出缓冲上限。

分布 (对齐旧 atb/bench_latency 的 RequestGenerator 口径):
    lognormal  每请求长度 = round(exp(N(mu, sigma))) 截断到 [min_len, max_len]
    fixed      每请求长度 = seq_len
    batch      请求条数 N: 固定 --batch, 或在 [--batch-min, --batch-max] 内均匀取

用法:
    python3 gen_requests.py --out /path/requests --count 200 --dist lognormal \
        --mu 4.997 --sigma 0.167 --max-len 218 --batch 10 --vocab 151936 --seed 0
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from tools.varlen import generate_varlen_from_lens   # noqa: E402

FORWARD_ORDER = ["input_ids", "position_ids", "actual_seq_lengths"]


def sample_lengths(args, rng):
    """返回 (batch, [每请求长度])。"""
    if args.batch_min and args.batch_max:
        batch = int(rng.randint(args.batch_min, args.batch_max + 1))
    else:
        batch = args.batch
    if args.dist == "fixed":
        return batch, [args.seq_len] * batch
    raw = np.exp(rng.normal(args.mu, args.sigma, size=batch))
    lens = np.clip(np.round(raw), args.min_len, args.max_len).astype(int)
    return batch, lens.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="请求池输出目录")
    p.add_argument("--count", type=int, default=100, help="生成多少套请求")
    p.add_argument("--dist", choices=["lognormal", "fixed"], default="lognormal")
    p.add_argument("--mu", type=float, default=4.997)
    p.add_argument("--sigma", type=float, default=0.167)
    p.add_argument("--min-len", type=int, default=1)
    p.add_argument("--max-len", type=int, default=218)
    p.add_argument("--seq-len", type=int, default=208, help="--dist fixed 时的每请求长度")
    p.add_argument("--batch", type=int, default=10, help="每套请求的条数 N")
    p.add_argument("--batch-min", type=int, default=0)
    p.add_argument("--batch-max", type=int, default=0)
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="参数相同也重新生成")
    args = p.parse_args()

    params = {k: v for k, v in vars(args).items() if k not in ("out", "force")}
    meta_path = os.path.join(args.out, "pool_meta.json")
    if not args.force and os.path.exists(meta_path):
        try:
            old = json.load(open(meta_path))
            if old.get("params") == params and old.get("count") == args.count:
                print(f"[gen_requests] 已存在同参数请求池 ({old.get('count')} 套) → 跳过: {args.out}")
                return 0
        except (OSError, ValueError):
            pass

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    totals, shapes = [], set()
    for k in range(args.count):
        batch, lens = sample_lengths(args, rng)
        ids, pos, cum = generate_varlen_from_lens(lens, vocab_size=args.vocab,
                                                 seed=args.seed * 100003 + k)
        req_dir = os.path.join(args.out, f"req_{k:04d}")
        os.makedirs(os.path.join(req_dir, "inputs"), exist_ok=True)

        tensors = {"input_ids": ids.numpy().astype(np.int64),
                   "position_ids": pos.numpy().astype(np.int64),
                   "actual_seq_lengths": np.array(cum, dtype=np.int64)}
        entries = []
        for logical in FORWARD_ORDER:                     # bundle 按 forward 序
            arr = tensors[logical]
            rel = f"inputs/{logical}.bin"
            arr.tofile(os.path.join(req_dir, rel))
            entries.append({"logical": logical, "shape": list(arr.shape), "file": rel})

        json.dump({"inputs": entries,
                   "golden": {"logical": "logits", "shape": [batch, args.vocab],
                              "file": "golden_logits.bin"},   # 只有 shape, 无文件 (性能测试不比对)
                   "provenance": {"seq_lens": lens, "total_tokens": int(tensors["input_ids"].size),
                                  "seed": args.seed * 100003 + k}},
                  open(os.path.join(req_dir, "bundle.json"), "w"), indent=2)
        totals.append(int(tensors["input_ids"].size))
        shapes.add(tuple(int(x) for x in tensors["input_ids"].shape) + (batch,))

    totals = np.array(totals)
    meta = {"params": params, "count": args.count,
            "stats": {"total_tokens_min": int(totals.min()),
                      "total_tokens_median": int(np.median(totals)),
                      "total_tokens_max": int(totals.max()),
                      "distinct_shapes": len(shapes)},
            "forward_order": FORWARD_ORDER}
    json.dump(meta, open(meta_path, "w"), indent=2)

    print(f"[gen_requests] 生成 {args.count} 套请求 → {args.out}")
    print(f"[gen_requests]   T(total_tokens): min={totals.min()} median={int(np.median(totals))} "
          f"max={totals.max()} | distinct shapes={len(shapes)} | dist={args.dist} "
          f"batch={args.batch}{'/' + str(args.batch_min) + '-' + str(args.batch_max) if args.batch_min else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
