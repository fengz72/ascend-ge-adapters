#!/usr/bin/env python3
"""qwen2.5-0.5b 性能测试请求池生成器 — 按分布产 K 套 varlen 输入 (bundle + .bin)。

框架不生成模型专属负载 (语义自洽只有模型侧能保证: asl 必须是 cumsum、position 每段
从 0 递增、asl[-1] == T), 所以这一步是**用户脚本**, 由 bench scenario 的 `generate:`
声明 (core/setup_scripts.py 执行, 传 $GE_SRC_DIR / $GE_ENV_FILE)。

每套请求写成 <out>/req_NNN/{bundle.json, inputs/*.bin}:
    bundle.inputs 按 **forward 入参序** (input_ids, position_ids, actual_seq_lengths),
    logical 名与 io_spec 一致 — C++ 按 logical 名配对 (docs §5.4)。
    bundle.golden 只写 shape ([N, 输出宽], 无文件): 供运行时推导输出缓冲上限
    (输出宽 = lm_head 剪裁后的词表宽, 由 --prune-tokens 指向的 json 决定; 未剪裁则 = --vocab)。

分布 (对齐旧 atb/bench_latency 的 RequestGenerator 口径):
    lognormal  每请求长度 = round(exp(N(mu, sigma))) 截断到 [min_len, max_len]
    fixed      每请求长度 = seq_len
    batch      请求条数 N: 固定 --batch, 或在 [--batch-min, --batch-max] 内均匀取
    prefix     --prefix P 或 Pmin-Pmax: 每套请求共享一份 packed prefix (PIA 形态),
               分布产的是每请求**总长** (P + own), 下界自动抬到 P+1 (own>=1);
               act = cumsum([P, own...]) 共 N+1 个元素 — 与基线布局**不通用**

用法:
    python3 gen_requests.py --out /path/requests --count 200 --dist lognormal \
        --mu 4.997 --sigma 0.167 --max-len 218 --batch 10 --vocab 151936 --seed 0
    # PIA prefix + lm_head 剪裁 (输出宽 = len(target_tokens)):
    python3 gen_requests.py --out /path/requests --prefix 20-25 \
        --prune-tokens models/qwen2.5-0.5b/config/target_tokens.json ...
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from core.config import load_target_tokens                              # noqa: E402
from tools.varlen import generate_prefix_varlen_from_lens, generate_varlen_from_lens  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
FORWARD_ORDER = ["input_ids", "position_ids", "actual_seq_lengths"]


def out_width(args):
    """golden 输出宽 = lm_head 剪裁后的词表宽 (未剪裁则 = --vocab)。

    剪裁宽度的事实源是 target_tokens.json 本身 (与 model.yaml 的 prune_token_file 同一份),
    不在 scenario 里再抄一个数字 — 抄错了运行时按 golden shape 分配的输出缓冲就对不上。
    """
    if not args.prune_tokens:
        return args.vocab
    path = args.prune_tokens
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(_REPO_ROOT, path)
    return len(load_target_tokens(path))


def parse_prefix(spec):
    """--prefix → (P_min, P_max)。"" / 0 → (0,0) 基线布局; "20" → (20,20); "20-25" → (20,25)。

    P 每套请求抽一次 (套内共享一份 prefix — packed 布局的前提), 套间随机。
    """
    if not spec or str(spec) == "0":
        return 0, 0
    parts = str(spec).split("-")
    lo = int(parts[0])
    hi = int(parts[1]) if len(parts) > 1 else lo
    if lo < 1 or hi < lo:
        raise ValueError(f"--prefix 须是 P 或 Pmin-Pmax (P>=1, Pmax>=Pmin): {spec!r}")
    return lo, hi


def sample_lengths(args, rng, prefix_len=0):
    """返回 (batch, [每请求**总长**]) — prefix 形态下总长 = P + own, 故下界抬到 P+1。"""
    if args.batch_min and args.batch_max:
        batch = int(rng.randint(args.batch_min, args.batch_max + 1))
    else:
        batch = args.batch
    if args.dist == "fixed":
        return batch, [args.seq_len] * batch
    min_len = max(args.min_len, prefix_len + 1)
    if min_len > args.max_len:
        raise ValueError(f"--max-len {args.max_len} <= prefix {prefix_len} → 每请求总长无解 "
                         f"(需 > P); 抬高 --max-len 或降低 prefix")
    raw = np.exp(rng.normal(args.mu, args.sigma, size=batch))
    lens = np.clip(np.round(raw), min_len, args.max_len).astype(int)
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
    p.add_argument("--prune-tokens", default="",
                   help="target_tokens.json 路径 (同 model.yaml 的 prune_token_file): "
                        "golden 输出宽 = len(token_ids), 不填则 = --vocab")
    p.add_argument("--prefix", default="",
                   help="PIA prefix 形态: P 或 Pmin-Pmax (每套请求抽一个 P, 套内共享); 空 → 基线布局")
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

    p_lo, p_hi = parse_prefix(args.prefix)
    out_vocab = out_width(args)
    rng = np.random.RandomState(args.seed)
    totals, shapes, prefixes = [], set(), []
    for k in range(args.count):
        prefix_len = int(rng.randint(p_lo, p_hi + 1)) if p_lo else 0
        batch, lens = sample_lengths(args, rng, prefix_len)
        own = [length - prefix_len for length in lens]      # 每请求自有段 (总长 - P)
        if prefix_len:
            ids, pos, cum = generate_prefix_varlen_from_lens(
                own, prefix_len, vocab_size=args.vocab, seed=args.seed * 100003 + k)
        else:
            ids, pos, cum = generate_varlen_from_lens(
                lens, vocab_size=args.vocab, seed=args.seed * 100003 + k)
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

        prov = {"seq_lens": lens, "total_tokens": int(tensors["input_ids"].size),
                "seed": args.seed * 100003 + k}
        if prefix_len:
            prov["prefix_len"] = prefix_len               # 总长 = P + own; act 有 batch+1 个元素
            prov["own_lens"] = own
        json.dump({"inputs": entries,
                   "golden": {"logical": "logits", "shape": [batch, out_vocab],
                              "file": "golden_logits.bin"},   # 只有 shape, 无文件 (性能测试不比对)
                   "provenance": prov},
                  open(os.path.join(req_dir, "bundle.json"), "w"), indent=2)
        totals.append(int(tensors["input_ids"].size))
        prefixes.append(prefix_len)
        shapes.add(tuple(int(x) for x in tensors["input_ids"].shape) + (batch,))

    totals = np.array(totals)
    meta = {"params": params, "count": args.count,
            "stats": {"total_tokens_min": int(totals.min()),
                      "total_tokens_median": int(np.median(totals)),
                      "total_tokens_max": int(totals.max()),
                      "distinct_shapes": len(shapes)},
            "forward_order": FORWARD_ORDER}
    if p_lo:
        meta["stats"]["prefix_len_min"] = int(min(prefixes))
        meta["stats"]["prefix_len_max"] = int(max(prefixes))
    json.dump(meta, open(meta_path, "w"), indent=2)

    print(f"[gen_requests] 生成 {args.count} 套请求 → {args.out}")
    print(f"[gen_requests]   T(total_tokens): min={totals.min()} median={int(np.median(totals))} "
          f"max={totals.max()} | distinct shapes={len(shapes)} | dist={args.dist} "
          f"batch={args.batch}{'/' + str(args.batch_min) + '-' + str(args.batch_max) if args.batch_min else ''}")
    if p_lo:
        print(f"[gen_requests]   prefix packed: P∈[{min(prefixes)}, {max(prefixes)}] "
              f"(声明 {args.prefix}) | 物理 token = P+Σown, act 形状 [batch+1] | 输出宽 {out_vocab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
