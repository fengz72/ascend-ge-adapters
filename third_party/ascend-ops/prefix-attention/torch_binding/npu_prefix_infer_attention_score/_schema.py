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

"""torch schema正本。

⚠ 默认值三层对齐: 本schema / CANN op def(def.cpp) / torchair converter(_converter.py)
的默认值必须逐项一致。torch.export会将==schema默认值的实参从fx节点kwargs中规约掉,
fx2ge调converter时以其Python签名默认值兜底, 两层不一致即图模式静默丢参。
"""

import torch
from torch.library import Library, impl

# torch命名空间沿用npu_ops_transformer(与历史扩展/测试保持drop-in兼容)
AS_LIBRARY = Library("npu_ops_transformer", "DEF")

OP_NAME = "npu_prefix_infer_attention_score"

SCHEMAS = [
    # .default: int[]传参(eager语义)
    "npu_prefix_infer_attention_score(Tensor query, Tensor key, Tensor value, "
    "*, Tensor? atten_mask=None, int[]? actual_seq_lengths=None, "
    "int[]? actual_seq_lengths_kv=None, int num_heads, float scale=1.0, "
    "int pre_tokens=2147483647, int next_tokens=2147483647, int num_key_value_heads=0, "
    "int sparse_mode=2) -> Tensor",
    # .tensor: act_seq经int64 tensor传参(图模式下作Data输入直达kernel)
    "npu_prefix_infer_attention_score.tensor(Tensor query, Tensor key, Tensor value, "
    "*, Tensor? atten_mask=None, Tensor? actual_seq_lengths=None, "
    "Tensor? actual_seq_lengths_kv=None, int num_heads, float scale=1.0, "
    "int pre_tokens=2147483647, int next_tokens=2147483647, int num_key_value_heads=0, "
    "int sparse_mode=2) -> Tensor",
]


def register():
    for schema in SCHEMAS:
        AS_LIBRARY.define(schema)

    @impl(AS_LIBRARY, OP_NAME, "Meta")
    def npu_prefix_infer_attention_score_meta(query, key, value, atten_mask=None,
                                              actual_seq_lengths=None, actual_seq_lengths_kv=None,
                                              num_heads=1, scale=1.0, pre_tokens=2147483647,
                                              next_tokens=2147483647, num_key_value_heads=0,
                                              sparse_mode=2):
        torch._check(
            query.dim() == 3,
            lambda: "query must be 3D TND tensor [T, N, D], but got " + str(query.dim()) + "D.")
        # TND prefix-in-Q: attention_out = query.shape (prefix输出在头部)
        return query.new_empty(query.size())

    @impl(AS_LIBRARY, OP_NAME + ".tensor", "Meta")
    def npu_prefix_infer_attention_score_tensor_meta(query, key, value, atten_mask=None,
                                                      actual_seq_lengths=None,
                                                      actual_seq_lengths_kv=None, num_heads=1,
                                                      scale=1.0, pre_tokens=2147483647,
                                                      next_tokens=2147483647, num_key_value_heads=0,
                                                      sparse_mode=2):
        return npu_prefix_infer_attention_score_meta(
            query, key, value, atten_mask, actual_seq_lengths, actual_seq_lengths_kv,
            num_heads, scale, pre_tokens, next_tokens, num_key_value_heads, sparse_mode)


register()
