"""
变长 (varlen) 输入处理 — token 拼接、位置编码、落盘/读回、结构不变量

通用工具, 不依赖具体模型结构。

落盘 schema (write_inputs / load_inputs) 由**输入生成脚本**与**框架**共用:
脚本产 <out>/{inputs.json, inputs/*.bin}, pipeline 读回成张量喂给三层 (L0 reference /
L1 golden / L2 trace+bundle)。inputs.json 的 inputs 段与 bundle.json 的 inputs 段同形
({logical, shape, file}), 只多一个 dtype —— bundle 不记 dtype 是因为 C++ 从 io_spec 取,
而输入脚本在 io_spec 之前跑, Python 侧读回 .bin 必须自己知道 dtype。
"""

import torch

INPUTS_JSON = "inputs.json"
INPUTS_SUBDIR = "inputs"


def generate_varlen_inputs(batch_size, seq_len, vocab_size=0, seed=None):
    """生成等长 varlen 输入 (不需要 tokenizer)。

    Args:
        batch_size:  序列条数
        seq_len:     每条序列的 token 数
        vocab_size:  >0 时生成 [0, vocab_size) 的随机 token; 0 则全 0
        seed:        随机种子 (可复现)

    Returns:
        concat_ids:  [1, total_len] token ids
        concat_pos:  [1, total_len] 拼接后的 position ids
        seq_lens:    list[int] 每条序列的长度
        cum_seq_lens: list[int] 累积长度 (用于 actual_seq_lengths)
    """
    seq_lens = [seq_len] * batch_size
    concat_ids = _tokens(batch_size * seq_len, vocab_size, seed).unsqueeze(0)
    pos_ids = [torch.arange(seq_len) for _ in range(batch_size)]
    concat_pos = torch.cat(pos_ids).unsqueeze(0)
    cum_seq_lens = []
    acc = 0
    for s in seq_lens:
        acc += s
        cum_seq_lens.append(acc)
    return concat_ids, concat_pos, seq_lens, cum_seq_lens


def _tokens(n, vocab_size, seed):
    """n 个 token id: vocab_size>0 → seeded 随机 (可复现); 否则全 0。"""
    if not vocab_size or vocab_size <= 0:
        return torch.zeros(n, dtype=torch.long)
    import numpy as np

    rng = np.random.RandomState(seed)
    return torch.from_numpy(rng.randint(0, vocab_size, size=n).astype(np.int64))


def generate_varlen_from_lens(seq_lens, vocab_size=0, seed=None):
    """按**给定的每请求长度**生成 varlen 输入 (性能测试的请求池用)。

    与 generate_varlen_inputs 的区别: 每条请求长度可各不相同 (服从任意分布),
    token ids 随机 (vocab_size>0 时) 或全 0。

    Args:
        seq_lens:   list[int] 每条请求的 token 数 (>=1)
        vocab_size: >0 时生成 [0, vocab_size) 的随机 token; 0 则全 0
        seed:       随机种子 (可复现)

    Returns:
        concat_ids:  [total] int64
        concat_pos:  [total] int64 每段 0..L-1
        cum_seq_lens: list[int] 累积长度 (actual_seq_lengths)
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    ids, pos, cum, acc = [], [], [], 0
    for length in seq_lens:
        length = int(length)
        if length < 1:
            raise ValueError(f"seq_lens 每项须 >=1, got {length}")
        if vocab_size > 0:
            ids.append(rng.randint(0, vocab_size, size=length).astype(np.int64))
        else:
            ids.append(np.zeros(length, dtype=np.int64))
        pos.append(np.arange(length, dtype=np.int64))
        acc += length
        cum.append(acc)
    return (torch.from_numpy(np.concatenate(ids)),
            torch.from_numpy(np.concatenate(pos)),
            cum)


def generate_prefix_varlen_inputs(batch_size, seq_len, prefix_len, vocab_size=0, seed=None):
    """生成 packed prefix-in-Q varlen 输入 (不需要 tokenizer)。

    布局: [prefix(P), req0(L), req1(L), ...], 每请求总长 = seq_len = P + L。
    与 generate_varlen_inputs(batch, seq_len) 语义等价 (每条请求 = prefix + own,
    prefix 逐条重复展开), 供 FIA 基线与 prefix 算子精度互证。

    act 契约 (KV 内嵌版算子): cumsum([P, L0, L1, ...]) — prefix 独立成 batch 0
    (P = act[0]), 请求 i 为 batch i+1, act_q ≡ act_kv。

    Args:
        vocab_size: >0 时生成 [0, vocab_size) 的随机 token; 0 则全 0
        seed:       随机种子 (可复现)

    Returns:
        concat_ids:  [1, P + batch*L] token ids
        concat_pos:  [1, P + batch*L] prefix 行 0..P-1, 每请求行 P..seq_len-1
        act:         list[int] cumsum([P, L, L, ...]) (batch+1 个元素)
        own_lens:    list[int] 每请求自有长度 L
    """
    if prefix_len <= 0 or prefix_len >= seq_len:
        raise ValueError(f"需要 0 < prefix_len < seq_len, got prefix={prefix_len}, seq={seq_len}")
    own = seq_len - prefix_len
    total = prefix_len + batch_size * own

    concat_ids = _tokens(total, vocab_size, seed).unsqueeze(0)
    pos_prefix = torch.arange(prefix_len)
    pos_req = torch.arange(prefix_len, seq_len)
    concat_pos = torch.cat([pos_prefix] + [pos_req] * batch_size).unsqueeze(0)

    own_lens = [own] * batch_size
    act, acc = [], 0
    acc += prefix_len
    act.append(acc)
    for L in own_lens:
        acc += L
        act.append(acc)
    return concat_ids, concat_pos, act, own_lens


def generate_prefix_varlen_from_lens(own_lens, prefix_len, vocab_size=0, seed=None):
    """按**给定的每请求自有长度**生成 packed prefix-in-Q varlen 输入 (性能测试请求池用)。

    与 generate_prefix_varlen_inputs 的区别: 每条请求的自有长度 L_i 可各不相同
    (服从任意分布), 而 prefix 长度 P 全套共享一份 (packed 布局的意义所在)。
    与 generate_varlen_from_lens 的区别: 头部多一段 prefix, act 多一个元素 (batch+1)。

    布局/契约同 generate_prefix_varlen_inputs:
        ids/pos = [prefix(P), req0(L0), req1(L1), ...]
        pos     prefix 行 0..P-1, 请求 i 行 P..P+L_i-1
        act     cumsum([P, L0, L1, ...]) — prefix 独立成 batch 0, 请求 i 为 batch i+1

    Args:
        own_lens:   list[int] 每条请求的自有 token 数 (>=1, 不含 prefix)
        prefix_len: 共享 prefix 长度 P (>=1)
        vocab_size: >0 时生成 [0, vocab_size) 的随机 token; 0 则全 0
        seed:       随机种子 (可复现)

    Returns:
        concat_ids:  [P + sum(own_lens)] int64
        concat_pos:  [P + sum(own_lens)] int64
        act:         list[int] 累积长度 (actual_seq_lengths, batch+1 个元素)
    """
    import numpy as np

    if prefix_len < 1:
        raise ValueError(f"prefix_len 须 >=1, got {prefix_len}")
    rng = np.random.RandomState(seed)

    def rand_tokens(n):
        if vocab_size > 0:
            return rng.randint(0, vocab_size, size=n).astype(np.int64)
        return np.zeros(n, dtype=np.int64)

    ids, pos, act = [], [], []
    ids.append(rand_tokens(prefix_len))
    pos.append(np.arange(prefix_len, dtype=np.int64))
    acc = prefix_len
    act.append(acc)
    for own in own_lens:
        own = int(own)
        if own < 1:
            raise ValueError(f"own_lens 每项须 >=1, got {own}")
        ids.append(rand_tokens(own))
        pos.append(np.arange(prefix_len, prefix_len + own, dtype=np.int64))
        acc += own
        act.append(acc)
    return (torch.from_numpy(np.concatenate(ids)),
            torch.from_numpy(np.concatenate(pos)),
            act)


# ==================== 落盘 / 读回 (脚本与框架共用的 schema) ====================

def write_inputs(out_dir, tensors: dict, logical_order, provenance=None) -> str:
    """{logical: ndarray} 按 logical_order 落盘 → <out>/inputs/<logical>.bin + inputs.json。

    logical_order = **forward 入参序** (与 bundle/io_spec 同一口径; C++ 按 logical 名配对,
    但顺序仍要一致 —— pipeline 会拿它与 adapter.io_input_nodes 的 logical 序对账)。
    返回 inputs.json 路径。
    """
    import json
    import os

    extra = set(tensors) - set(logical_order)
    if extra:
        raise ValueError(f"tensors 有 logical_order 未声明的键: {sorted(extra)}")
    missing = [n for n in logical_order if n not in tensors]
    if missing:
        raise ValueError(f"logical_order 声明了但 tensors 里没有: {missing}")

    os.makedirs(os.path.join(out_dir, INPUTS_SUBDIR), exist_ok=True)
    entries = []
    for logical in logical_order:
        arr = tensors[logical]
        rel = f"{INPUTS_SUBDIR}/{logical}.bin"
        arr.tofile(os.path.join(out_dir, rel))
        entries.append({"logical": logical, "shape": [int(x) for x in arr.shape],
                        "dtype": str(arr.dtype), "file": rel})
    path = os.path.join(out_dir, INPUTS_JSON)
    with open(path, "w") as f:
        json.dump({"inputs": entries, "provenance": provenance or {}},
                  f, indent=2, ensure_ascii=False)
    return path


def load_inputs(out_dir, device=None):
    """读回 write_inputs 的产物 → (tuple[Tensor], list[logical], provenance)。

    校验文件字节数 == prod(shape) × dtype 字节数: 产物被截断/与声明不一致时当场失败,
    而不是喂进图后表现为 tiling 崩或精度全错 (与 graph.from_air 拒绝按位置硬配同一哲学)。
    device 非空则搬上去 —— 脚本产 CPU .bin (生成阶段不占卡), 上卡是框架的事。
    """
    import json
    import os

    import numpy as np

    path = os.path.join(out_dir, INPUTS_JSON)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"输入未生成: {path} 不存在 — 先跑 inputs 脚本 (即去掉 --skip export)")
    with open(path) as f:
        meta = json.load(f)
    entries = meta.get("inputs") or []
    if not entries:
        raise ValueError(f"{path} 的 inputs 为空")

    out, logical = [], []
    for e in entries:
        logical.append(e["logical"])
        fp = os.path.join(out_dir, e["file"])
        dt = np.dtype(e["dtype"])
        shape = tuple(int(x) for x in e["shape"])
        expect = int(np.prod(shape)) * dt.itemsize if shape else dt.itemsize
        actual = os.path.getsize(fp)
        if actual != expect:
            raise ValueError(
                f"{fp} 字节数 {actual} ≠ shape{shape} × {dt} = {expect} — 产物被截断, 或与 "
                f"{path} 的声明不一致 (重跑 inputs 脚本)")
        t = torch.from_numpy(np.fromfile(fp, dtype=dt).reshape(shape))
        out.append(t.to(device) if device is not None else t)
    return tuple(out), logical, meta.get("provenance") or {}


# ==================== 结构不变量 (生成侧自检 + 加载侧复验) ====================

def check_varlen(ids, pos, asl, prefix_len=0):
    """varlen 打包的结构不变量; 返回段数 (= batch, prefix 形态为 batch+1)。

    框架**不生成**模型专属负载 (语义自洽只有模型侧能保证), 但**验**语义自洽不需要模型
    知识, 就是下面几条等式。生成脚本落盘前调一次, 就把"脚本写错 → 图静默算错"变成当场失败。

        prefix_len>0: asl = cumsum([P, L0, L1, ...]) → asl[0] == P, 段 0 是共享 prefix
        prefix_len=0: asl = cumsum([L0, L1, ...])
        两形态共同:   asl[-1] == T (ids 元素数), asl 严格递增且首项 >0
                      position 每段递增: 基线各段 0..L_i-1; prefix 的段 0 是 0..P-1,
                      段 i>0 是 P..P+L_i-1 (packed 布局下 prefix 物理上只存一份)
    """
    ids = torch.as_tensor(ids).reshape(-1)
    pos = torch.as_tensor(pos).reshape(-1)
    asl_t = torch.as_tensor(asl).reshape(-1)
    asl = [int(x) for x in asl_t.tolist()]
    total = ids.numel()

    if pos.numel() != total:
        raise ValueError(f"position_ids 元素数 {pos.numel()} ≠ input_ids {total}")
    if not asl:
        raise ValueError("actual_seq_lengths 为空")
    if asl[0] <= 0 or any(b <= a for a, b in zip(asl, asl[1:])):
        raise ValueError(f"actual_seq_lengths 须严格递增且首项 >0: {asl}")
    if asl[-1] != total:
        raise ValueError(f"asl[-1]={asl[-1]} ≠ T={total} — 段长累积须等于 token 总数")
    if prefix_len and asl[0] != prefix_len:
        raise ValueError(f"prefix 形态 asl[0] 须 == prefix_len={prefix_len}, got {asl[0]}")

    bounds = [0] + asl
    for i in range(len(asl)):
        lo, hi = bounds[i], bounds[i + 1]
        start = prefix_len if (prefix_len and i > 0) else 0
        want = torch.arange(start, start + (hi - lo), dtype=pos.dtype)
        if not torch.equal(pos[lo:hi], want):
            got = pos[lo:hi][:4].tolist()
            raise ValueError(
                f"第 {i} 段 position_ids 须是 arange({start}, {start + hi - lo}) "
                f"(每段独立从起点递增), got {got}{'...' if hi - lo > 4 else ''}")
    return len(asl)
