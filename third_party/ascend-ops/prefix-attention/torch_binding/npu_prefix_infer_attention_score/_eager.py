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

"""eager dispatch(PrivateUse1): torch算子 → pybind扩展 → aclnn。"""

import torch
from torch.library import impl

from ._builder import load_extension
from ._schema import AS_LIBRARY, OP_NAME

op_module = load_extension()  # JIT编译/加载(带缓存)


@impl(AS_LIBRARY, OP_NAME, "PrivateUse1")
def npu_prefix_infer_attention_score(query, key, value, atten_mask=None,
                                      actual_seq_lengths=None, actual_seq_lengths_kv=None,
                                      num_heads=1, scale=1.0, pre_tokens=2147483647,
                                      next_tokens=2147483647, num_key_value_heads=0, sparse_mode=2):
    return op_module.npu_prefix_infer_attention_score(
        query, key, value, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
        num_heads, scale, pre_tokens, next_tokens, num_key_value_heads, sparse_mode)


def _tensor_to_int_list(t):
    if t is None:
        return None
    if t.device.type != "cpu":
        t = t.cpu()
    return t.to(torch.int64).tolist()


@impl(AS_LIBRARY, OP_NAME + ".tensor", "PrivateUse1")
def npu_prefix_infer_attention_score_tensor(query, key, value, atten_mask=None,
                                            actual_seq_lengths=None, actual_seq_lengths_kv=None,
                                            num_heads=1, scale=1.0, pre_tokens=2147483647,
                                            next_tokens=2147483647, num_key_value_heads=0,
                                            sparse_mode=2):
    """Tensor传参overload: eager下取值转list走aclnn; 图模式(torchair)下act_seq作为图输入。"""
    return op_module.npu_prefix_infer_attention_score(
        query, key, value, atten_mask,
        _tensor_to_int_list(actual_seq_lengths), _tensor_to_int_list(actual_seq_lengths_kv),
        num_heads, scale, pre_tokens, next_tokens, num_key_value_heads, sparse_mode)
