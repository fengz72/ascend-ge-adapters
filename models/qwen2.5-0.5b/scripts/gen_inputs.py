#!/usr/bin/env python3
"""qwen2.5-0.5b 图输入生成器 — 产**一套** varlen 输入落盘 (inputs.json + inputs/*.bin)。

框架不生成模型专属负载 (语义自洽只有模型侧能保证: asl 必须是 cumsum、position 每段从起点
递增、asl[-1] == T), 所以这一步是**用户脚本**, 由 model.yaml 的 `inputs.script` 声明
(core/setup_scripts.py 执行)。与 bench.pool 的 gen_requests.py 同一套接口、同一份产物
schema (tools/varlen.write_inputs); 区别是本脚本只产一套、形状固定 (export trace 与
verify golden 共用), gen_requests 按分布产 K 套供压测回放。

落盘而非在进程里造 (原来是 adapter.build_inputs), 于是: 激励成为三层 (L0 reference /
L1 golden / L2 图) 共用的**独立产物**, L0/L1 的 golden 可脱离 export 重跑, 生成阶段不占卡。

--vocab 由框架从**已加载模型**注入 (= model.get_input_embeddings().weight.shape[0]),
**不是** config.vocab_size —— lm_head 剪裁会改写后者 (如改成 8), 用它生成 token 就只覆盖
embedding 的前 8 行。手工复现时从权重目录的 safetensors header 取同一个数。

--prefix-len 的**存在与否**由 adapt.params.prefix 管辖 (core.pipeline._inputs_args 注入或
移除), 故翻 prefix 一个开关即无缝切形态, 不必手改 yaml 的 inputs.args。

用法 (框架调用; 手工复现):
    python3 gen_inputs.py --out models/qwen2.5-0.5b/io --vocab 151936 \
        --batch-size 10 --seq-len 208 --seed 0                 # 基线 (FIA)
    python3 gen_inputs.py ... --prefix-len 20                  # prefix packed 形态 (PIA)
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from tools.varlen import (check_varlen, generate_prefix_varlen_inputs,   # noqa: E402
                          generate_varlen_inputs, write_inputs)

FORWARD_ORDER = ["input_ids", "position_ids", "actual_seq_lengths"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="输出目录 (框架传 <base>/io)")
    p.add_argument("--vocab", type=int, required=True,
                   help="embedding 行数 (框架从已加载模型注入; 不是 config.vocab_size)")
    p.add_argument("--batch-size", type=int, default=10, help="请求条数 N")
    p.add_argument("--seq-len", type=int, default=208, help="每请求长度 (prefix 形态含 P)")
    p.add_argument("--prefix-len", type=int, default=0,
                   help=">0 → prefix packed 形态 (act = cumsum([P, L...]), 段数 N+1); "
                        "不传 → 基线 (act = cumsum([L...]), 段数 N)")
    p.add_argument("--seed", type=int, default=0, help="token 随机种子 (可复现)")
    args = p.parse_args()

    if args.vocab <= 0:
        raise SystemExit(f"[gen_inputs] --vocab 须 >0, got {args.vocab}")
    if args.prefix_len:
        ids, pos, act, lens = generate_prefix_varlen_inputs(
            args.batch_size, args.seq_len, args.prefix_len,
            vocab_size=args.vocab, seed=args.seed)
        own = lens
        asl = act
    else:
        if args.prefix_len < 0:
            raise SystemExit(f"[gen_inputs] --prefix-len 不能为负, got {args.prefix_len}")
        ids, pos, lens, cum = generate_varlen_inputs(
            args.batch_size, args.seq_len, vocab_size=args.vocab, seed=args.seed)
        own = None
        asl = cum

    tensors = {"input_ids": ids.squeeze(0).numpy().astype(np.int64),
               "position_ids": pos.squeeze(0).numpy().astype(np.int64),
               "actual_seq_lengths": np.array(asl, dtype=np.int64)}

    # 落盘前自检结构不变量 — 脚本写错就地失败, 不要等喂进图后表现为 tiling 崩/精度全错
    segments = check_varlen(tensors["input_ids"], tensors["position_ids"],
                            tensors["actual_seq_lengths"], prefix_len=args.prefix_len)
    expect = args.batch_size + 1 if args.prefix_len else args.batch_size
    if segments != expect:
        raise SystemExit(f"[gen_inputs] 段数 {segments} ≠ 期望 {expect} "
                         f"(prefix 形态 act 有 N+1 个元素, 基线 N 个)")

    prov = {k: v for k, v in vars(args).items() if k != "out"}
    prov["total_tokens"] = int(tensors["input_ids"].size)
    prov["segments"] = segments
    if own is not None:
        prov["own_lens"] = [int(x) for x in own]
    os.makedirs(args.out, exist_ok=True)
    path = write_inputs(args.out, tensors, FORWARD_ORDER, prov)

    form = f"prefix packed (P={args.prefix_len}, own={own[0] if own else 0})" if args.prefix_len \
        else "基线 varlen"
    print(f"[gen_inputs] {form}: N={args.batch_size} seq_len={args.seq_len} "
          f"T={prov['total_tokens']} 段数={segments} vocab={args.vocab} seed={args.seed}")
    print(f"[gen_inputs]   → {path}")
    print(f"[gen_inputs]   " + " ".join(
        f"{k}{tuple(v.shape)}:{v.dtype}" for k, v in tensors.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
