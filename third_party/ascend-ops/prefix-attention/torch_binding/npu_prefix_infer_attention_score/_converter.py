# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details.
# You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""torchair GE converter注册(图模式: torch.compile / dynamo_export)。

含默认值一致性断言(防线): torch.export将==schema默认值的实参从fx节点kwargs规约掉,
converter以自身Python默认值兜底——两层默认值不一致即静默丢参。
"""

import inspect

import torch


def _assert_defaults_match_schema(converter):
    op = torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score
    schema_defaults = {a.name: a.default_value for a in op.default._schema.arguments
                       if a.default_value is not None}
    sig = inspect.signature(converter)
    for name, sval in schema_defaults.items():
        if name not in sig.parameters:
            continue
        cval = sig.parameters[name].default
        if cval is inspect.Parameter.empty:
            continue
        if cval != sval:
            raise RuntimeError(
                f"converter默认值与算子schema默认值不一致: {name} schema={sval!r} "
                f"converter={cval!r}; fx层会规约掉==schema默认值的实参, converter默认值兜底"
                "将导致图模式静默丢参, 请对齐三层默认值(schema/_converter.py/def.cpp)")


def register_converter():
    try:
        import torchair  # noqa: F401
        from torchair._ge_concrete_graph.ge_converter.converter_utils import (
            Tensor, Optional, Union, List, DataType, TensorSpec, dtype_promote,
        )
        import torchair.ge as ge
    except ImportError as e:
        import warnings
        warnings.warn(f"torchair不可用({e}), 图模式converter未注册, 仅支持eager调用")
        return

    @torchair.register_fx_node_ge_converter(
        torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score.default)
    def convert_npu_prefix_infer_attention_score(
        query: Tensor, key: Tensor, value: Tensor, *, atten_mask: Optional[Tensor] = None,
        actual_seq_lengths: Optional[Union[List[int], Tensor]] = None,
        actual_seq_lengths_kv: Optional[Union[List[int], Tensor]] = None,
        num_heads: int = 1, scale: float = 1.0, pre_tokens: int = 2147483647,
        next_tokens: int = 2147483647, num_key_value_heads: int = 0, sparse_mode: int = 2,
        meta_outputs: TensorSpec = None,
    ):
        # 默认值必须与_schema.py的schema默认值严格一致(上方断言校验)
        if actual_seq_lengths is not None:
            actual_seq_lengths = dtype_promote(actual_seq_lengths, target_dtype=DataType.DT_INT64)
        if actual_seq_lengths_kv is not None:
            actual_seq_lengths_kv = dtype_promote(actual_seq_lengths_kv, target_dtype=DataType.DT_INT64)
        outputs = ge.custom_op(
            "PrefixInferAttentionScore",
            inputs={
                "query": query, "key": [key], "value": [value], "atten_mask": atten_mask,
                "actual_seq_lengths": actual_seq_lengths,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
            },
            outputs=["attention_out", "softmax_lse"],
            attrs={
                "num_heads": ge.attr.Int(num_heads), "scale": ge.attr.Float(scale),
                "pre_tokens": ge.attr.Int(pre_tokens), "next_tokens": ge.attr.Int(next_tokens),
                "input_layout": ge.attr.Str("TND"),
                "num_key_value_heads": ge.attr.Int(num_key_value_heads),
                "sparse_mode": ge.attr.Int(sparse_mode), "inner_precise": ge.attr.Int(1),
                "block_size": ge.attr.Int(0), "antiquant_mode": ge.attr.Int(0),
                "softmax_lse_flag": ge.attr.Bool(False), "key_antiquant_mode": ge.attr.Int(0),
                "value_antiquant_mode": ge.attr.Int(0), "query_quant_mode": ge.attr.Int(0),
                "pse_type": ge.attr.Int(0), "out_dtype": ge.attr.Int(0),
            },
        )
        return outputs[0]

    _assert_defaults_match_schema(convert_npu_prefix_infer_attention_score)

    @torchair.register_fx_node_ge_converter(
        torch.ops.npu_ops_transformer.npu_prefix_infer_attention_score.tensor)
    def convert_npu_prefix_infer_attention_score_tensor(
        query: Tensor, key: Tensor, value: Tensor, *, atten_mask: Optional[Tensor] = None,
        actual_seq_lengths: Optional[Tensor] = None,
        actual_seq_lengths_kv: Optional[Tensor] = None, num_heads: int = 1,
        scale: float = 1.0, pre_tokens: int = 2147483647,
        next_tokens: int = 2147483647, num_key_value_heads: int = 0, sparse_mode: int = 2,
        meta_outputs: TensorSpec = None,
    ):
        """Tensor传参converter: act_seq作为图Data输入(device tensor), 运行时由V2回调
        D2H读当前值 — 换值不触发重编译(动态batch切分场景)。
        注意: act tensor须作为forward入参传入; 挂module属性会被dynamo当常量guard。"""
        if actual_seq_lengths is not None:
            actual_seq_lengths = dtype_promote(actual_seq_lengths, target_dtype=DataType.DT_INT64)
        if actual_seq_lengths_kv is not None:
            actual_seq_lengths_kv = dtype_promote(actual_seq_lengths_kv, target_dtype=DataType.DT_INT64)
        outputs = ge.custom_op(
            "PrefixInferAttentionScore",
            inputs={
                "query": query, "key": [key], "value": [value], "atten_mask": atten_mask,
                "actual_seq_lengths": actual_seq_lengths,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
            },
            outputs=["attention_out", "softmax_lse"],
            attrs={
                "num_heads": ge.attr.Int(num_heads), "scale": ge.attr.Float(scale),
                "pre_tokens": ge.attr.Int(pre_tokens), "next_tokens": ge.attr.Int(next_tokens),
                "input_layout": ge.attr.Str("TND"),
                "num_key_value_heads": ge.attr.Int(num_key_value_heads),
                "sparse_mode": ge.attr.Int(sparse_mode), "inner_precise": ge.attr.Int(1),
                "block_size": ge.attr.Int(0), "antiquant_mode": ge.attr.Int(0),
                "softmax_lse_flag": ge.attr.Bool(False), "key_antiquant_mode": ge.attr.Int(0),
                "value_antiquant_mode": ge.attr.Int(0), "query_quant_mode": ge.attr.Int(0),
                "pse_type": ge.attr.Int(0), "out_dtype": ge.attr.Int(0),
            },
        )
        return outputs[0]

    _assert_defaults_match_schema(convert_npu_prefix_infer_attention_score_tensor)
