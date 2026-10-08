"""
Qwen2.5-0.5b 模型适配 — 全部替换实现 + 运行期注入 + Qwen25Adapter

【新模型适配指引】复制本文件为 models/<model>/model.py, 然后:
    1. 改 MODELING 绑定和 patch_specs 里的类名 (Qwen2* → 目标架构类名)
    2. 按模型调整 attention forward (head 配置/布局) 与注入 (rotary 表结构)
    3. 结构差异大时不继承本文件, 直接继承 core.adapter.GeModelAdapter 自写
    机制层 (adapt/patch/setup) 见 core/adapter.py — 无需改动。

替换项 (类级 patch, 对所有实例生效):
    Qwen2ForCausalLM.forward       → 2D varlen 模型接口 (图边界, 见 _varlen_forward)
    Qwen2RMSNorm.forward           → npu_rms_norm
    apply_rotary_pos_emb           → npu_apply_rotary_pos_emb (TND)
    Qwen2RotaryEmbedding.forward   → cos/sin 表 Gather
    Qwen2Attention.forward         → 2D varlen (FIA 基线 / PIA prefix, 由 _prefix_mode 区分)

运行期注入 (实例级, 由 adapt 自动调用 adapter.setup, 无需手工调用):
    模式标志 _prefix_mode → 模型与每层 self_attn (forward 的 last 索引、
    attention 的 FIA/PIA 算子选择)
    图常量: atten_mask [max_seq_len, max_seq_len] bool + cos/sin 表 [1, max_seq_len, D] fp16
            (max_seq_len 由 pipeline 从 cfg.graph.dynamic.max_seq_len 透传)
    lm_head vocab 剪裁 (prune_tokens 非空时) — 输出维度 vocab_size → len(tokens)
    actual_seq_lengths 不注入 — patched forward 每次调用以入参覆盖每层
    self_attn 的属性 (成为图 Data 节点, 换值/换 shape 不重编译)

adapt 两阶段: apply_patches (类级行为替换) → setup (实例级适配: 手术 + 常量 + 标志)

用法 (管线走 core/pipeline.py; 下面是手工复现的顺序):
    import torch, torch_npu
    from transformers import AutoModelForCausalLM
    torch_npu.npu.set_device(0)                  # 设备由调用方管理
    from model import Qwen25Adapter
    raw = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).npu()

    adapter = Qwen25Adapter()                    # FIA 基线
    # adapter = Qwen25Adapter(prefix=True)       # PIA prefix
    # adapter = Qwen25Adapter(prune_tokens=ids)  # + lm_head 剪裁

    inputs = adapter.build_inputs(raw, batch_size=10, seq_len=208)  # 激励, 必须在 adapt 之前
    model = adapter.adapt(raw, max_seq_len=2048)                    # L0 → L1, 破坏性不可逆

    适配后接口 (即导出图的边界):
    logits = model(input_ids, position_ids, actual_seq_lengths)     # [N, vocab]
"""

import torch
import torch.nn as nn
import torch_npu
from torchair.ops import npu_fused_infer_attention_score as _torchair_fia
from transformers.models.qwen2 import modeling_qwen2

from core.adapter import GeModelAdapter
from core.graph import IoNode
from tools.varlen import generate_varlen_inputs, generate_prefix_varlen_inputs

try:
    import npu_prefix_infer_attention_score  # noqa: F401  PIA 算子注册 (schema + eager + torchair converter)
    _PREFIX_IA_OP = torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score
except ImportError:
    _PREFIX_IA_OP = None


# ==================== 组件级替换实现 ====================

def _rms_norm_forward(self, hidden_states):
    """RMSNorm: 8 个小算子 → npu_rms_norm 1 个融合算子。"""
    return torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """RoPE rotate: 一次处理 Q/K, 原生 TND 布局。"""
    cos = cos.squeeze(0).unsqueeze(1)   # [1, T, D] → [T, 1, D]
    sin = sin.squeeze(0).unsqueeze(1)
    q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(
        q, k, cos, sin, layout="TND", rotary_mode="half"
    )
    # 图编译时算子可能返回 4D [1,T,N,D], 显式 reshape 回 3D, 否则下游 num_heads 推断错误
    q_embed = q_embed.reshape(-1, q.size(1), q.size(2))
    k_embed = k_embed.reshape(-1, k.size(1), k.size(2))
    return q_embed, k_embed


def _rotary_emb_forward(self, x, position_ids):
    """RoPE cos/sin: 从预注册图常量表按 position_ids Gather (图内仅 2 个 Gather)。"""
    return self._cos_table[:, position_ids, :], self._sin_table[:, position_ids, :]


def _attention_forward(self, hidden_states, position_embeddings, **kwargs):
    """2D varlen attention (替换 Qwen2Attention.forward)。

    基线 (self._prefix_mode=False): npu_fused_infer_attention_score 推理算子,
    图编译走 torchair Tensor 入口 (避免 dynamo specialize), eager 走
    torch_npu 入口。
    prefix 模式 (self._prefix_mode=True, adapter.setup 注入):
    npu_prefix_infer_attention_score (KV 内嵌版) — q/k/v 全长 TND, 行序
    [prefix, req0, ...], prefix KV 位于 k/v 头部由算子内部处理, 无需切分;
    actual_seq_lengths = act = cumsum([P, L0, ...]), prefix 独立成 batch 0。

    共同约定: hidden_states [T, D] 全程 2D 无 batch 维, reshape 只用 int
    常量 + -1 (不提取 SymInt, 消除 GE Pack 算子)。sparse_mode=2 压缩 causal
    (mode=3 语义等价但图编译落慢的 MIX tiling 分支), 算子按
    actual_seq_lengths 生成 block-diagonal mask; GQA 由 num_key_value_heads
    原生支持。
    """
    num_heads = int(self.config.num_attention_heads)
    num_kv_heads = int(self.config.num_key_value_heads)
    head_dim = int(self.head_dim)

    q = self.q_proj(hidden_states).reshape(-1, num_heads, head_dim)
    k = self.k_proj(hidden_states).reshape(-1, num_kv_heads, head_dim)
    v = self.v_proj(hidden_states).reshape(-1, num_kv_heads, head_dim)

    # 经 modeling 模块属性调用 — 与 HF 原版调用机制一致, 实现由 patch 状态决定
    q, k = modeling_qwen2.apply_rotary_pos_emb(q, k, *position_embeddings)

    actual_seq_lengths = self.actual_seq_lengths_tensor
    if self._prefix_mode:
        attn_output = _PREFIX_IA_OP.tensor(
            q, k, v,
            atten_mask=self.atten_mask,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths,
            num_heads=num_heads,
            scale=self.scaling,
            num_key_value_heads=num_kv_heads,
            pre_tokens=2147483647,
            next_tokens=0,
            sparse_mode=2,
        )
    else:
        fia = (_torchair_fia if torch.compiler.is_compiling()
               else torch_npu.npu_fused_infer_attention_score)
        attn_output, _ = fia(
            q, k, v,
            num_heads=num_heads,
            input_layout="TND",
            scale=self.scaling,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths,
            num_key_value_heads=num_kv_heads,
            atten_mask=self.atten_mask,
            sparse_mode=2,
        )

    attn_output = attn_output.reshape(-1, int(self.config.hidden_size)).contiguous()
    return self.o_proj(attn_output), None


# ==================== 模型级 forward (图边界) ====================

def _varlen_forward(self, input_ids, position_ids, actual_seq_lengths):
    """适配后的模型接口 (替换 Qwen2ForCausalLM.forward): 2D varlen, 输出末 token logits。

    输入 (成为图 Data 节点):
        input_ids          [T] int64 — 所有序列 token 拼接
        position_ids       [T] int64 — 图内 cos/sin 按 position_ids Gather
        actual_seq_lengths [N] int64 — 基线: 各序列累积长度;
                           prefix: act = cumsum([P, L0, ...]), prefix 独立成 batch 0
    hidden 全程 2D [T, D] 无 batch 维 (避免 GE Pack); 绕开原版 forward 的
    labels/cache/mask 分支, 只保留一条计算路径。
    输出: logits [N, vocab] — 基线取每序列末 token (actual_seq_lengths-1);
          prefix 取每请求末 token (act[1:]-1, 丢掉 prefix 段结束行)。
    模式由 self._prefix_mode 区分 (adapter.setup 注入, 两个模式共用本 forward)。
    """
    for layer in self.model.layers:
        layer.self_attn.actual_seq_lengths_tensor = actual_seq_lengths

    m = self.model
    hidden = m.embed_tokens(input_ids)
    position_embeddings = m.rotary_emb(hidden, position_ids)
    for layer in m.layers:
        hidden = layer(
            hidden,
            attention_mask=None,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            use_cache=self.config.use_cache,
        )
    last = actual_seq_lengths[1:] - 1 if self._prefix_mode else actual_seq_lengths - 1
    hidden = hidden.index_select(0, last)
    hidden = m.norm(hidden)
    return self.lm_head(hidden)


# ==================== 运行期注入 (patch 前提) ====================

def _build_causal_mask(max_seq_len, device):
    """[max_seq_len, max_seq_len] 上三角因果掩码 (bool), 用于 sparse_mode=2 压缩 causal。

    长度须与 RoPE 表 (max_seq_len) 同源, 否则长序列下 mask 与 position 覆盖范围不一致。
    """
    return torch.triu(
        torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device), diagonal=1
    )


def _precompute_rope_cos_sin(model, max_seq_len, device):
    """预计算 RoPE cos/sin 表注册为 buffer, 配合 frozen_parameter 成为图常量。"""
    rotary_emb = model.model.rotary_emb
    position_ids = torch.arange(max_seq_len, dtype=torch.long, device=device).unsqueeze(0)

    inv_freq = rotary_emb.inv_freq
    scaling = rotary_emb.attention_scaling

    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(device)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * scaling).to(dtype=torch.float16)
    sin = (emb.sin() * scaling).to(dtype=torch.float16)

    rotary_emb.register_buffer('_cos_table', cos, persistent=False)
    rotary_emb.register_buffer('_sin_table', sin, persistent=False)


# ==================== lm_head vocab 剪裁 (结构手术) ====================

def prune_lm_head(model, token_ids):
    """裁剪 lm_head 仅保留 token_ids 对应的行 (通用 CausalLM 手术)。

    输出维度从 vocab_size 降为 len(token_ids)。
    若 tie_word_embeddings=True, 先 clone 解绑再裁剪, 不影响 embed_tokens。
    """
    if getattr(model.config, 'tie_word_embeddings', False):
        model.lm_head.weight = nn.Parameter(
            model.model.embed_tokens.weight.data.clone()
        )
        print("[prune] 解除 lm_head/embed_tokens weight tying")

    device = model.lm_head.weight.device
    target_ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)
    pruned_weight = model.lm_head.weight.data.index_select(0, target_ids_t).clone()

    hidden_size = model.config.hidden_size
    new_lm_head = nn.Linear(hidden_size, len(token_ids), bias=False)
    new_lm_head.weight = nn.Parameter(pruned_weight)
    model.lm_head = new_lm_head.to(device).half()
    model.config.vocab_size = len(token_ids)

    print(f"[prune] lm_head: [{len(token_ids)}, {hidden_size}]")


# ==================== 适配器 ====================

class Qwen25Adapter(GeModelAdapter):
    """Qwen2.5-0.5b (modeling_qwen2)。

    prefix=True 使用 KV 内嵌 PIA 算子; prune_tokens 非空时剪裁 lm_head vocab。
    """

    MODELING = modeling_qwen2

    def __init__(self, prefix=False, prune_tokens=None):
        super().__init__()
        if prefix and _PREFIX_IA_OP is None:
            raise RuntimeError(
                "prefix 模式需要 npu_prefix_infer_attention_score 算子 (安装见 env.sh)")
        self.prefix = prefix
        self.prune_tokens = prune_tokens
        self.pruned_token_ids = None

    def setup(self, model, device='npu', max_seq_len=2048):
        """实例级适配 (由 adapt 自动调用): lm_head 剪裁 + 模式标志 + 图常量注入。

        1. lm_head vocab 剪裁 (prune_tokens 非空时) — 结构手术,
           输出维度 vocab_size → len(tokens)
        2. 模式标志 _prefix_mode → 模型与每层 self_attn (forward 的 last 索引、
           attention 的 FIA/PIA 算子选择); 创建时烙定, 之后不可变:
           运行期翻标志会导致 eager 与已导出图行为分裂
        3. 图常量: atten_mask [max_seq_len, max_seq_len] bool + RoPE cos/sin 表
           [1, max_seq_len, D] — 两者同源于 max_seq_len (pipeline 从
           cfg.graph.dynamic.max_seq_len 透传; 表长决定 position_ids 可用上限)
        actual_seq_lengths 不注入: patched forward 每次调用都会以入参
        覆盖每层 self_attn 的属性 (eager 与图导出同一路径); 若有人绕过
        patched forward 直接调层, 缺属性会立刻 AttributeError (好过静默
        用 dummy 值算错)。
        """
        if self.prune_tokens:
            prune_lm_head(model, self.prune_tokens)
            self.pruned_token_ids = self.prune_tokens

        model._prefix_mode = self.prefix
        atten_mask = _build_causal_mask(max_seq_len, device)
        for layer in model.model.layers:
            layer.self_attn._prefix_mode = self.prefix
            layer.self_attn.register_buffer('atten_mask', atten_mask)
        _precompute_rope_cos_sin(model, max_seq_len, device)

    def patch_specs(self):
        """patch 集合与 prefix 模式无关 (模式是实例状态, 由 setup 注入)。"""
        m = self.MODELING
        return [
            (m.Qwen2ForCausalLM,     "forward",              _varlen_forward),
            (m.Qwen2RMSNorm,         "forward",              _rms_norm_forward),
            (m,                      "apply_rotary_pos_emb", _apply_rotary_pos_emb),
            (m.Qwen2RotaryEmbedding, "forward",              _rotary_emb_forward),
            (m.Qwen2Attention,       "forward",              _attention_forward),
        ]

    # ---- 输入接口 (export trace 与 verify golden 共用) ----

    def build_inputs(self, model, batch_size=10, seq_len=208, prefix_len=0, seed=0, **kwargs):
        """生成 varlen 输入 (input_ids, position_ids, actual_seq_lengths), NPU 张量。

        prefix 模式由 self.prefix 决定 (创建时定); prefix_len 是 **prefix 形态专属**参数,
        非 prefix 形态**忽略**它 (而非报错) —— 这样 prefix_len 可在 yaml 常驻, 翻
        adapt.params.prefix 一个开关即无缝切形态 (bench 侧的 --prefix 去留同理由
        core.bench._form_args 按 flag 管辖)。仅 prefix=true 却 prefix_len<=0 才报错 (真错误)。

        token 是 **seeded 随机** (可复现): 全 0 token 会让每条请求的输入逐字节相同,
        精度比对退化成"同一行比 N 次"。词表宽取自 embedding 权重形状, **不是**
        config.vocab_size — lm_head 剪裁会把后者改成剪裁宽度 (如 8), 用它生成 token
        就只覆盖 embedding 的前 8 行。
        """
        vocab = model.get_input_embeddings().weight.shape[0]
        if self.prefix:
            if prefix_len <= 0:
                raise ValueError("prefix 模式需要 prefix_len > 0")
            concat_ids, concat_pos, act, own_lens = generate_prefix_varlen_inputs(
                batch_size, seq_len, prefix_len, vocab_size=vocab, seed=seed)
            print(f"  batch_size={batch_size}, prefix_len={prefix_len}, "
                  f"own_len={own_lens[0]}, total_tokens={prefix_len + sum(own_lens)} "
                  f"(FIA 基线: {batch_size * seq_len})")
            asl = act
        else:
            concat_ids, concat_pos, seq_lens, cum_seq_lens = generate_varlen_inputs(
                batch_size, seq_len, vocab_size=vocab, seed=seed)
            ignored = f" (忽略 prefix_len={prefix_len}: 非 prefix 形态)" if prefix_len else ""
            print(f"  batch_size={batch_size}, total_tokens={sum(seq_lens)}, "
                  f"cum_seq_lens[-1]={cum_seq_lens[-1]}{ignored}")
            asl = cum_seq_lens

        return (concat_ids.squeeze(0).npu(),
                concat_pos.squeeze(0).npu(),
                torch.tensor(asl, dtype=torch.int64, device='npu'))

    def mark_dynamic(self, inputs, **kwargs):
        input_ids, position_ids, asl = inputs
        torch._dynamo.mark_dynamic(input_ids, 0)      # T 动态
        torch._dynamo.mark_dynamic(position_ids, 0)   # T 动态
        if not self.prefix:
            torch._dynamo.mark_dynamic(asl, 0)        # 基线 N 动态; prefix 的 act 形状固定 batch+1
        return inputs

    def io_input_nodes(self, inputs, **kwargs):
        """io_spec 输入节点 (logical 序 = forward 序; dim0 动态, prefix 的 act 静态)。"""
        logical = ['input_ids', 'position_ids', 'actual_seq_lengths']
        nodes = []
        for i, (t, lg) in enumerate(zip(inputs, logical)):
            if self.prefix and i == 2:
                shape, dyn = list(t.shape), []           # act 形状固定 batch+1
            else:
                shape, dyn = [-1] + list(t.shape)[1:], [0]
            nodes.append(IoNode(logical=lg, dtype=str(t.dtype).replace('torch.', ''),
                                format='ND', shape=shape, dynamic_dims=dyn))
        return nodes

    # ---- 原版参考比对 (core/verify.py: 隔离"适配"这一个变量) ----

    def unpack_requests(self, inputs):
        """打包的图输入 → 逐请求 (input_ids, position_ids), 供原版 HF 一条一条前向。

        prefix 形态: ids = [prefix(P), own_0, own_1, ...], act = cumsum([P, L0, L1, ...])
            → 请求 i = prefix ++ own_i, position 0..P+L_i-1
              (packed 布局与"每请求把 prefix 重复展开"语义等价 — 算子按 act 分段,
               prefix 只在物理上存一份)
        基线形态: act = cumsum([L0, L1, ...]) → 请求 i = 段 i, position 0..L_i-1

        position 一律用 arange 重新生成, **不取**图输入里的 position_ids: 参考侧要的
        是"一条独立请求"的语义真值, 打包时 position 若写错正好由比对暴露。
        """
        ids, _, act = inputs
        bounds = [int(x) for x in act.tolist()]
        requests = []

        def push(tokens):
            requests.append((tokens,
                             torch.arange(tokens.numel(), dtype=torch.long, device=ids.device)))

        if self.prefix:
            prefix = ids[:bounds[0]]
            for i in range(len(bounds) - 1):
                push(torch.cat([prefix, ids[bounds[i]:bounds[i + 1]]]))
        else:
            prev = 0
            for end in bounds:
                push(ids[prev:end])
                prev = end
        return requests

    def reference_columns(self):
        """lm_head 剪裁 → 参考输出只取保留的那些列 (与图输出同宽)。"""
        return self.prune_tokens
