import sys, torch, torch_npu, torchair
from transformers import Qwen3_5ForConditionalGeneration


class Decision(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.model
        self.register_buffer(
            "rows", model.lm_head.weight.detach()[[r[0] for r in letters]].float()
        )

    def forward(self, ids, pixels, grid, pos, ii, iw, vp, cu, feature_index):
        feats = self.backbone.visual(
            pixels,
            grid_thw=grid,
            interp_indices=ii,
            interp_weights=iw,
            precomputed_vpos=vp,
            cu_seqlens=cu,
        ).pooler_output
        text = self.backbone.get_input_embeddings()(ids)
        selected = feats.index_select(
            0, feature_index.clamp_min(0).flatten()
        ).reshape_as(text)
        embeds = torch.where(
            (feature_index >= 0).unsqueeze(-1), selected.to(text.dtype), text
        )
        hidden = self.backbone.language_model(
            inputs_embeds=embeds, position_ids=pos, use_cache=False
        ).last_hidden_state
        scores = hidden[:, -1].float() @ self.rows.T
        return scores


from transformers.models.qwen3_5 import modeling_qwen3_5 as m


def single_image_attention(
    self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs
):
    n = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states)
        .reshape(n, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    q, k = m.apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    q, k, v = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
    out = m.eager_attention_forward(
        self, q, k, v, attention_mask=None, scaling=self.scaling, dropout=0.0
    )[0]
    return self.proj(out.reshape(n, -1).contiguous())


m.Qwen3_5VisionAttention.forward = single_image_attention
from torchair import ge, register_fx_node_ge_converter


@torch.library.custom_op("onejev::chunk", mutates_args=())
def chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    s: torch.Tensor,
    l: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.npu.npu_chunk_gated_delta_rule(
        q, k, v, beta=b, initial_state=s, actual_seq_lengths=l, g=g, scale=128**-0.5
    )[0]


@chunk.register_fake
def fake(q, k, v, b, s, l, g):
    return torch.empty_like(v)


@register_fx_node_ge_converter(torch.ops.onejev.chunk.default)
def conv(q, k, v, b, s, l, g, meta_outputs=None):
    return ge.custom_op(
        "ChunkGatedDeltaRule",
        inputs=dict(
            query=q, key=k, value=v, beta=b, initial_state=s, actual_seq_lengths=l, g=g
        ),
        attrs={"scale_value": ge.attr.Float(128**-0.5)},
        outputs=["out", "final_state"],
    )[0]


def delta(query, key, value, g, beta, **kwargs):
    B, S, H, D = value.shape

    def norm(t):
        f = t.float()
        return (
            (f * torch.rsqrt(f.square().sum(-1, keepdim=True) + 1e-6))
            .to(value.dtype)
            .flatten(0, 1)
        )

    o = chunk(
        norm(query),
        norm(key),
        value.flatten(0, 1),
        beta.to(value.dtype).flatten(0, 1),
        value.new_zeros((B, H, D, D)),
        torch.full((B,), S, device=value.device, dtype=torch.int32),
        g.float().flatten(0, 1),
    )
    return o.reshape(B, S, H, D), None


m.torch_chunk_gated_delta_rule = delta


def supplied_positions(grid, merge, kwargs):
    return kwargs.pop("precomputed_vpos")


m.get_vision_position_ids = supplied_positions
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(".")
letters = [tok.encode(chr(65 + i), add_special_tokens=False) for i in range(26)]
assert (
    all(len(row) == 1 for row in letters) and len(set(row[0] for row in letters)) == 26
)


def export_softplus(t):
    return torch.where(t > 20, t, torch.log1p(torch.exp(torch.clamp(t, max=20))))


m.F.softplus = export_softplus
torch._dynamo.config.specialize_float = True


def ge_causal_conv4(hidden_states, weight, bias=None, activation=None, **kwargs):
    if bias is not None or activation not in ("silu", "swish") or weight.shape[-1] != 4:
        raise ValueError("Expected bias-free four-tap SiLU causal convolution")
    channels = weight.shape[0]
    y = torch.nn.functional.conv2d(
        hidden_states.float().unsqueeze(2),
        weight.float().reshape(channels, 1, 1, 4),
        groups=channels,
        padding=(0, 3),
    )
    y = y[..., : hidden_states.shape[-1]].squeeze(2).to(hidden_states.dtype)
    return torch.nn.functional.silu(y)


m.causal_conv1d_fn = ge_causal_conv4
model = (
    Decision(
        Qwen3_5ForConditionalGeneration.from_pretrained(
            ".", dtype=torch.bfloat16, attn_implementation="eager"
        )
    )
    .eval()
    .to("npu:0")
)
ids = torch.tensor([[248053] + [248056] * 64 + [248054, 32, 33]], device="npu:0")
grid = torch.tensor([[1, 16, 16]], device="npu:0")
pos = (
    torch.arange(ids.shape[1], device="npu:0")
    .view(1, 1, -1)
    .expand(3, 1, -1)
    .contiguous()
)
pos, _ = model.backbone.get_rope_index(
    ids.cpu(), (ids.cpu() == 248056).int(), image_grid_thw=grid.cpu()
)
pos = pos.to(ids.device)
from transformers import vision_utils as vu

v = model.backbone.visual
cgrid = grid.cpu()
ii, iw = vu.get_vision_interpolation_indices_and_weights(
    cgrid,
    v.num_grid_per_side,
    mode=v.interpolation_mode,
    align_corners=v.interpolation_align_corners,
    spatial_merge_size=2,
)
vp = vu.get_vision_position_ids(cgrid, 2)
cu = vu.get_vision_cu_seqlens(cgrid)
import torchair._utils.export_utils as eu

orig_convert = eu._convert_data_to_const


def aligned(inputs, graph, path, names):
    graph.op.sort(
        key=lambda op: (0, op.attr["index"].i) if op.type == "Data" else (1, 0)
    )
    for i, x in enumerate(inputs):
        if id(x) in names:
            assert graph.op[i].type == "Data" and graph.op[i].attr["index"].i == i
    return orig_convert(inputs, graph, path, names)


eu._convert_data_to_const = aligned
# The prior experiment kept the full textual graph; inspect this candidate from its AIR directly.
eu.dump_graph = lambda *args, **kwargs: None
torchair.dynamo_export(
    ids,
    torch.zeros(256, 1536, device="npu:0", dtype=torch.bfloat16),
    grid,
    pos,
    *[a.to(ids.device) for a in (ii, iw, vp, cu)],
    torch.tensor([[-1] + list(range(64)) + [-1] * 3], device=ids.device),
    model=model,
    export_path=sys.argv[1],
    export_name="onejev",
    dynamic=True
)