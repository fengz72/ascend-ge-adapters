/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prefix_infer_attention_score_proto.h
 * \brief PrefixInferAttentionScore算子原型定义, 供图模式构图/融合阶段识别算子。
 * 输入/输出/属性序与 op_host/prefix_infer_attention_score_def.cpp 严格镜像, 勿单独调整。
 */

#ifndef OPS_OP_PROTO_INC_PREFIX_INFER_ATTENTION_SCORE_OPS_H_
#define OPS_OP_PROTO_INC_PREFIX_INFER_ATTENTION_SCORE_OPS_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Prefix-in-Q prefill attention: TND布局, prefix Q/KV位于各张量头部只存一份,
* kernel侧batch 0'=[prefix;req0]合并为单batch计算, 其余batch为请求自身(可见prefix KV).

* @par Inputs:
* @li query: A Tensor. TND layout [P+sum(L), N_q, D], prefix Q at head. The type support float16, bf16.
* @li key: A Tensor. TND layout [P+sum(L), N_kv, D], prefix KV at head. The type support float16, bf16.
* @li value: A Tensor. TND layout [P+sum(L), N_kv, D], prefix KV at head. The type support float16, bf16.
* @li pse_shift: Optional. Not supported yet.
* @li atten_mask: Optional. A Tensor of bool. Compressed causal mask when sparse_mode=2
*     (shared mask only: 2D or batch dim == 1).
* @li actual_seq_lengths: Optional. A Tensor of int64, cumsum([P, L0, L1, ...]).
* @li actual_seq_lengths_kv: Optional. A Tensor of int64, identical to actual_seq_lengths.
* @li dequant_scale1/quant_scale1/dequant_scale2/quant_scale2/quant_offset2: Optional. Quant params, not supported yet.
* @li antiquant_scale/antiquant_offset: Optional. Antiquant params, not supported yet.
* @li block_table: Optional. Not supported yet.
* @li query_padding_size/kv_padding_size: Optional. Not supported yet.
* @li key/value_antiquant_scale/offset: Optional. Not supported yet.
* @li key_shared_prefix/value_shared_prefix/actual_shared_prefix_len: Optional. Reserved slots,
*     never connected (prefix KV must be placed at the head of key/value).
* @li query_rope/key_rope/key_rope_antiquant_scale: Optional. Not supported yet.
* @li dequant_scale_query: Optional. Not supported yet.
* @li learnable_sink: Optional. Not supported yet.
* @li q_start_idx/kv_start_idx: Optional. Not supported yet.

* @par Outputs:
* @li attention_out: A Tensor. Same shape/dtype as query [P+sum(L), N_q, D] (prefix segment at head).
* @li softmax_lse: A Tensor of float32, empty placeholder in current NoQuant scenario.

* @par Attributes:
* @li num_heads: Required. An int. The number of query heads.
* @li scale: A float. Default: 1.0.
* @li pre_tokens: An int. Default: 2147483647.
* @li next_tokens: An int. Default: 2147483647.
* @li input_layout: A string. Must be "TND". Default: "TND".
* @li num_key_value_heads: An int. Default: 0.
* @li sparse_mode: An int. Only 0 or 2 supported. Default: 0.
* @li inner_precise: An int. Default: 1.
* @li block_size/antiquant_mode/softmax_lse_flag/key_antiquant_mode/value_antiquant_mode/
*     query_quant_mode/pse_type/out_dtype: Reserved, keep consistent with FusedInferAttentionScore.

* @attention Constraints:
* @li Only TND layout, NoQuant dtype (float16/bfloat16) and sparse_mode 0/2 are supported.
* @li act arrays must be identical strictly increasing cumsum([P, L0, ...]) with size >= 2
*     (P > 0, at least one non-empty request).
* @li act[0] = P is the prefix length; per-batch q_len == kv_len.
*/
REG_OP(PrefixInferAttentionScore)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .DYNAMIC_INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_BOOL}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(key_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(value_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(value_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_shared_prefix, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(value_shared_prefix, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(actual_shared_prefix_len, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale_query, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(learnable_sink, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(softmax_lse, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale, Float, 1.0)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(input_layout, String, "TND")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(antiquant_mode, Int, 0)
    .ATTR(softmax_lse_flag, Bool, false)
    .ATTR(key_antiquant_mode, Int, 0)
    .ATTR(value_antiquant_mode, Int, 0)
    .ATTR(query_quant_mode, Int, 0)
    .ATTR(pse_type, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .OP_END_FACTORY_REG(PrefixInferAttentionScore)
}  // namespace ge

#endif // OPS_OP_PROTO_INC_PREFIX_INFER_ATTENTION_SCORE_OPS_H_
