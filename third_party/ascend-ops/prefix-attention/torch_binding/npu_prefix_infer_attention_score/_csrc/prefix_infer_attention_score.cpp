/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software and you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details.
 * You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prefix_infer_attention_score.cpp
 * \brief torch_npu适配: npu_prefix_infer_attention_score → aclnnPrefixInferAttentionScore
 * TND prefix-in-Q prefill attention: query/key/value均=[prefix, b0, ..., bn](prefix KV在key/value头部,
 * P=act_kv[0]), act_q与act_kv均为cumsum([P, L0, L1, ...]), 输出=[prefix_out, b0_out, ..., bn_out]
 */
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
const int DIM_TND = 3;

/**
 * @brief ACLNN Wrapper for aclnnPrefixInferAttentionScore
 */
at::Tensor npu_prefix_infer_attention_score(const at::Tensor &query, const at::Tensor &key,
                                             const at::Tensor &value,
                                             const c10::optional<at::Tensor> &atten_mask,
                                             const c10::optional<std::vector<int64_t>> &actual_seq_lengths,
                                             const c10::optional<std::vector<int64_t>> &actual_seq_lengths_kv,
                                             int64_t num_heads, double scale, int64_t pre_tokens,
                                             int64_t next_tokens, int64_t num_key_value_heads, int64_t sparse_mode)
{
    TORCH_CHECK(query.dim() == DIM_TND && key.dim() == DIM_TND && value.dim() == DIM_TND,
                "query/key/value must be 3D TND tensors [T, N, D], but got query.dim(): ", query.dim(),
                ", key.dim(): ", key.dim(), ", value.dim(): ", value.dim());
    auto dtype = query.scalar_type();
    TORCH_CHECK((dtype == at::kHalf) || (dtype == at::kBFloat16),
                "query dtype must be float16 or bfloat16, but got ", dtype);
    TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                key.scalar_type() == value.scalar_type(),
                "query/key/value must have the same dtype");
    int64_t num_kv_heads = (num_key_value_heads == 0) ? num_heads : num_key_value_heads;
    TORCH_CHECK(num_heads % num_kv_heads == 0,
                "num_heads(", num_heads, ") must be divisible by num_key_value_heads(", num_kv_heads, ")");
    TORCH_CHECK((sparse_mode == 0) || (sparse_mode == 2), "sparse_mode must be 0 or 2, but got ", sparse_mode);

    // aclnn层key/value为tensorlist; 用vector持有数据, 避免initializer_list临时数组悬垂
    std::vector<at::Tensor> key_vec = {key};
    std::vector<at::Tensor> value_vec = {value};
    at::TensorList key_list(key_vec);
    at::TensorList value_list(value_vec);
    // aclIntArray参数: optional<vector> → optional<IntArrayRef>
    c10::optional<at::IntArrayRef> seq_q;
    c10::optional<at::IntArrayRef> seq_kv;
    if (actual_seq_lengths.has_value()) {
        seq_q = at::IntArrayRef(*(actual_seq_lengths));
    }
    if (actual_seq_lengths_kv.has_value()) {
        seq_kv = at::IntArrayRef(*(actual_seq_lengths_kv));
    }

    at::Tensor attention_out = at::empty(query.sizes(), query.options());

    ACLNN_CMD(aclnnPrefixInferAttentionScore, query, key_list, value_list, atten_mask, seq_q, seq_kv,
              num_heads, scale, pre_tokens, next_tokens, num_kv_heads, sparse_mode, attention_out);
    return attention_out;
}

// Bind the C++ function to Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_prefix_infer_attention_score", &npu_prefix_infer_attention_score,
          "npu_prefix_infer_attention_score");
}
} // namespace op_api
