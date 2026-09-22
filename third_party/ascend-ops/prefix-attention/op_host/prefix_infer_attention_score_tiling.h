/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software and you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may refer to the License for details.
 * You should not have this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prefix_infer_attention_score_tiling.h
 * \brief
 */
#ifndef PREFIX_INFER_ATTENTION_SCORE_TILING_H
#define PREFIX_INFER_ATTENTION_SCORE_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include "fia_tiling_impl/fused_infer_attention_score_tiling_index.h"

namespace optiling {
ge::graphStatus TilingPrefixInferAttentionScore(gert::TilingContext *context);
} // namespace optiling

#endif // PREFIX_INFER_ATTENTION_SCORE_TILING_H
