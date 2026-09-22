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
 * \file prefix_infer_attention_score_base_tiling.h
 * \brief PrefixInferAttentionScore基础注册专用tiling结构
 * 字段布局与FusedInferAttentionScoreTilingData完全一致(二进制兼容,
 * kernel侧GET_TILING_DATA_WITH_STRUCT按FusedInferAttentionScoreTilingData解包);
 * 独立命名避免算子基础注册与per-key注册使用同名struct时生成的tiling_data.h出现类重定义。
 * 前置依赖: 本头必须在FusedInferAttentionScoreTilingData的子struct定义可见后包含
 * (即包含fia_tiling_impl/fused_infer_attention_score_tiling.h之后)。
 */
#ifndef PREFIX_INFER_ATTENTION_SCORE_BASE_TILING_H
#define PREFIX_INFER_ATTENTION_SCORE_BASE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PrefixInferAttentionScoreBaseTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPageAttentionParams, pageAttenParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionMaskParams, maskParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionWorkspaceParams, workspaceParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionOuterSplitParams, outerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionFlashDecodeParams, fdParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPrefixParams, prefixParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPseParams, pseParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionLeftPaddingParams, leftPaddingParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPostQuantParams, postquantParams);
END_TILING_DATA_DEF
} // namespace optiling

#endif // PREFIX_INFER_ATTENTION_SCORE_BASE_TILING_H
