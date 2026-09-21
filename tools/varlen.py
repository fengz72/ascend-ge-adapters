"""
变长 (varlen) 输入处理 — token 拼接与位置编码

通用工具, 不依赖具体模型结构。
"""

import torch


def generate_varlen_inputs(batch_size, seq_len):
    """直接生成全 0 token ids 的 varlen 输入, 不需要 tokenizer。

    Args:
        batch_size:  序列条数
        seq_len:     每条序列的 token 数

    Returns:
        concat_ids:  [1, total_len] 全 0 token ids
        concat_pos:  [1, total_len] 拼接后的 position ids
        seq_lens:    list[int] 每条序列的长度
        cum_seq_lens: list[int] 累积长度 (用于 actual_seq_lengths)
    """
    seq_lens = [seq_len] * batch_size
    concat_ids = torch.zeros(batch_size * seq_len, dtype=torch.long).unsqueeze(0)
    pos_ids = [torch.arange(seq_len) for _ in range(batch_size)]
    concat_pos = torch.cat(pos_ids).unsqueeze(0)
    cum_seq_lens = []
    acc = 0
    for s in seq_lens:
        acc += s
        cum_seq_lens.append(acc)
    return concat_ids, concat_pos, seq_lens, cum_seq_lens


def prepare_varlen_inputs(tokenizer, input_texts):
    """将多条文本拼接为 varlen 格式的输入。

    Returns:
        concat_ids:  [1, total_len] 拼接后的 token ids
        concat_pos:  [1, total_len] 拼接后的 position ids
        seq_lens:    list[int] 每条文本的长度
        cum_seq_lens: list[int] 累积长度 (用于 actual_seq_lengths)
    """
    all_ids, pos_ids = [], []
    for text in input_texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        all_ids.append(ids)
        pos_ids.append(torch.arange(ids.shape[0]))
    seq_lens = [x.shape[0] for x in all_ids]
    concat_ids = torch.cat(all_ids).unsqueeze(0)
    concat_pos = torch.cat(pos_ids).unsqueeze(0)
    cum_seq_lens = []
    acc = 0
    for s in seq_lens:
        acc += s
        cum_seq_lens.append(acc)
    return concat_ids, concat_pos, seq_lens, cum_seq_lens


def generate_prefix_varlen_inputs(batch_size, seq_len, prefix_len):
    """生成 packed prefix-in-Q varlen 输入 (全 0 token, 不需要 tokenizer)。

    布局: [prefix(P), req0(L), req1(L), ...], 每请求总长 = seq_len = P + L。
    与 generate_varlen_inputs(batch, seq_len) 语义等价 (每条请求 = prefix + own,
    prefix 逐条重复展开), 供 FIA 基线与 prefix 算子精度互证。

    act 契约 (KV 内嵌版算子): cumsum([P, L0, L1, ...]) — prefix 独立成 batch 0
    (P = act[0]), 请求 i 为 batch i+1, act_q ≡ act_kv。

    Returns:
        concat_ids:  [1, P + batch*L] 全 0 token ids
        concat_pos:  [1, P + batch*L] prefix 行 0..P-1, 每请求行 P..seq_len-1
        act:         list[int] cumsum([P, L, L, ...]) (batch+1 个元素)
        own_lens:    list[int] 每请求自有长度 L
    """
    if prefix_len <= 0 or prefix_len >= seq_len:
        raise ValueError(f"需要 0 < prefix_len < seq_len, got prefix={prefix_len}, seq={seq_len}")
    own = seq_len - prefix_len
    total = prefix_len + batch_size * own

    concat_ids = torch.zeros(total, dtype=torch.long).unsqueeze(0)
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
