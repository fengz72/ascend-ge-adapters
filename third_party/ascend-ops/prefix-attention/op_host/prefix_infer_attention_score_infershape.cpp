/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software and you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details.
 * You may not have this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prefix_infer_attention_score_infershape.cpp
 * \brief TND布局: attention_out形状 = query形状[T, N_q, D], softmax_lse为空占位
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "log/error_code.h"

using namespace ge;

namespace ops {
static constexpr uint32_t PIA_QUERY_INDEX = 0;
static constexpr uint32_t PIA_ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t PIA_SOFTMAX_LSE_INDEX = 1;
static constexpr uint32_t PIA_TND_DIM_NUMS = 3;
static constexpr uint32_t PIA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX = 5;
static constexpr uint32_t PIA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX = 6;

// 图模式: GE InferShapePass从OpDef挂接的函数取推导实现(SetInferShape/SetInferDataType, 见def.cpp),
// 非static供def.cpp引用; IMPL_OP_INFERSHAPE链仍保留(tiling侧InputsDataDependency使用)
ge::graphStatus InferShapePrefixInferAttentionScore(gert::InferShapeContext *context)
{
    const gert::Shape *queryShape = context->GetInputShape(PIA_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_IF(queryShape->GetDimNum() != PIA_TND_DIM_NUMS,
        OP_LOGE(context->GetNodeName(), "PrefixInferAttentionScore only supports TND layout, "
                "query dims must be 3, but got %zu.", queryShape->GetDimNum()),
        return ge::GRAPH_FAILED);
    gert::Shape *attentionOutShape = context->GetOutputShape(PIA_ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;
    gert::Shape *softmaxLseShape = context->GetOutputShape(PIA_SOFTMAX_LSE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);
    *softmaxLseShape = gert::Shape();
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypePrefixInferAttentionScore(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(PIA_ATTENTION_OUT_INDEX, context->GetInputDataType(PIA_QUERY_INDEX));
    context->SetOutputDataType(PIA_SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
} // namespace ops

IMPL_OP_INFERSHAPE(PrefixInferAttentionScore)
    .InferShape(ops::InferShapePrefixInferAttentionScore)
    .InferDataType(ops::InferDataTypePrefixInferAttentionScore)
    .InputsDataDependency({ops::PIA_INPUT_ACTUAL_SEQ_LENGTHS_INDEX, ops::PIA_INPUT_ACTUAL_SEQ_LENGTHS_KV_INDEX});
