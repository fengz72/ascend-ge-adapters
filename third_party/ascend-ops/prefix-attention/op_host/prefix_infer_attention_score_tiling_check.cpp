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
 * \file prefix_infer_attention_score_tiling_check.cpp
 * \brief PrefixInferAttentionScore自有tiling约束校验:
 * 仅TND + NoQuant + 无PSE + sparse 0/2 + act_q与act_kv为相同严格递增cumsum数组([P, L0, L1, ...],
 * 且actLen>=2(至少一个请求)), kernel侧batch 0'=[prefix;req0]合并为单batch计算, 以及Q/K/V的
 * TND形状与N/D轴一致性校验。
 */
#include "prefix_infer_attention_score_tiling_check.h"

#include <algorithm>
#include <string>
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {
namespace {
constexpr uint32_t TND_DIM_NUM = 3U;
constexpr uint32_t TND_DIM_T = 0U;
constexpr uint32_t TND_DIM_N = 1U;
constexpr uint32_t TND_DIM_D = 2U;

bool IsTndDim(const gert::StorageShape *shape, uint32_t dim)
{
    return shape != nullptr && shape->GetStorageShape().GetDimNum() == TND_DIM_NUM &&
        shape->GetStorageShape().GetDim(dim) > 0;
}
} // namespace

ge::graphStatus PiaTilingCheck::CheckPrefixInQ(const FiaTilingInfo &fiaInfo)
{
    const char *opName = fiaInfo.opName != nullptr ? fiaInfo.opName : "PrefixInferAttentionScore";
    const FIAParaInfo &para = fiaInfo.opParamInfo;

    // 仅TND布局
    std::string layout = para.layOut != nullptr ? para.layOut : "";
    if (layout != "TND") {
        OP_LOGE(opName, "PrefixInferAttentionScore only supports TND layout, but got %s", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    // GQA: n1为n2的整数倍
    if (fiaInfo.n1Size == 0 || fiaInfo.n2Size == 0 || fiaInfo.n1Size % fiaInfo.n2Size != 0) {
        OP_LOGE(opName, "PrefixInferAttentionScore requires num_heads(%u) %% num_key_value_heads(%u) == 0",
                fiaInfo.n1Size, fiaInfo.n2Size);
        return ge::GRAPH_FAILED;
    }
    // 仅NoQuant(量化相关输入不参与计算, 存在即视为非法)
    if (fiaInfo.antiQuantFlag || para.deqScale1.tensor != nullptr || para.quantScale1.tensor != nullptr ||
        para.deqScale2.tensor != nullptr || para.quantScale2.tensor != nullptr ||
        para.quantOffset2.tensor != nullptr || para.antiquantScale.tensor != nullptr ||
        para.antiquantOffset.tensor != nullptr || para.keyAntiquantScale.tensor != nullptr ||
        para.keyAntiquantOffset.tensor != nullptr || para.valueAntiquantScale.tensor != nullptr ||
        para.valueAntiquantOffset.tensor != nullptr) {
        OP_LOGE(opName, "PrefixInferAttentionScore only supports NoQuant mode");
        return ge::GRAPH_FAILED;
    }
    // 不支持PSE/RoPE/PA
    if (fiaInfo.pseShiftFlag) {
        OP_LOGE(opName, "PrefixInferAttentionScore does not support pse_shift");
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.pageAttentionFlag) {
        OP_LOGE(opName, "PrefixInferAttentionScore does not support PAGE_ATTENTION");
        return ge::GRAPH_FAILED;
    }
    if (para.queryRope.tensor != nullptr || para.keyRope.tensor != nullptr) {
        OP_LOGE(opName, "PrefixInferAttentionScore does not support rope");
        return ge::GRAPH_FAILED;
    }
    // 不允许显式传入shared prefix输入(prefix KV已位于key/value头部)
    if (para.keySharedPrefix.tensor != nullptr || para.valueSharedPrefix.tensor != nullptr ||
        para.actualSharedPrefixLen.tensor != nullptr) {
        OP_LOGE(opName, "PrefixInferAttentionScore does not accept key/value_shared_prefix or "
                        "actual_shared_prefix_len; prefix KV must be placed at the head of key/value");
        return ge::GRAPH_FAILED;
    }
    // sparse_mode: 0(显式mask)或2(压缩causal)
    int32_t sparseMode = para.sparseMode != nullptr ? *para.sparseMode : 0;
    if (sparseMode != SPARSE_MODE_NO_MASK && sparseMode != SPARSE_MODE_LEFT_UP) {
        OP_LOGE(opName, "PrefixInferAttentionScore only supports sparse_mode=0 or 2, but got %d", sparseMode);
        return ge::GRAPH_FAILED;
    }
    // TND形状: query/key/value均为[T, N, D], N/D轴一致(prefix KV位于key/value头部, 不再是独立张量)
    if (!IsTndDim(para.query.shape, TND_DIM_T) || !IsTndDim(para.key.shape, TND_DIM_T) ||
        !IsTndDim(para.value.shape, TND_DIM_T)) {
        OP_LOGE(opName, "PrefixInferAttentionScore requires query/key/value to be 3D TND tensors [T, N, D]");
        return ge::GRAPH_FAILED;
    }
    if (para.key.shape->GetStorageShape().GetDim(TND_DIM_N) != static_cast<int64_t>(fiaInfo.n2Size) ||
        para.value.shape->GetStorageShape().GetDim(TND_DIM_N) != static_cast<int64_t>(fiaInfo.n2Size) ||
        para.key.shape->GetStorageShape().GetDim(TND_DIM_D) != static_cast<int64_t>(fiaInfo.qkHeadDim) ||
        para.value.shape->GetStorageShape().GetDim(TND_DIM_D) != static_cast<int64_t>(fiaInfo.vHeadDim)) {
        OP_LOGE(opName, "PrefixInferAttentionScore requires key/value TND shape to be [T, %u, %u/%u]",
                fiaInfo.n2Size, fiaInfo.qkHeadDim, fiaInfo.vHeadDim);
        return ge::GRAPH_FAILED;
    }
    // act契约: act_q与act_kv为相同的严格递增cumsum数组([P, L0, L1, ...]); P = act_kv[0] > 0;
    // kernel侧batch 0'=[prefix;req0]合并为单batch(q_len=kv_len=P+L0), batch j'>0为请求j'(q_len=kv_len=L_j)
    auto *qSeqTensor = para.actualSeqLengthsQ.tensor;
    auto *kvSeqTensor = para.actualSeqLengths.tensor;
    if (qSeqTensor == nullptr || kvSeqTensor == nullptr ||
        qSeqTensor->GetData<int64_t>() == nullptr || kvSeqTensor->GetData<int64_t>() == nullptr) {
        OP_LOGE(opName, "PrefixInferAttentionScore requires actual_seq_lengths and actual_seq_lengths_kv "
                        "to be non-empty");
        return ge::GRAPH_FAILED;
    }
    if (qSeqTensor->GetShapeSize() != kvSeqTensor->GetShapeSize() ||
        qSeqTensor->GetShapeSize() < static_cast<int64_t>(fiaInfo.bSize)) {
        OP_LOGE(opName, "PrefixInferAttentionScore requires actual_seq_lengths and actual_seq_lengths_kv "
                        "to have the same size (>= bSize %u)", fiaInfo.bSize);
        return ge::GRAPH_FAILED;
    }
    // prefix-merged: batch 0'=[prefix;req0]合并计算, 要求至少存在一个请求batch
    if (qSeqTensor->GetShapeSize() < 2) {
        OP_LOGE(opName, "PrefixInferAttentionScore requires at least one request batch "
                        "(actual_seq_lengths size >= 2, i.e. cumsum([P, L0, ...])), but got size %ld",
                qSeqTensor->GetShapeSize());
        return ge::GRAPH_FAILED;
    }
    const int64_t *qData = qSeqTensor->GetData<int64_t>();
    const int64_t *kvData = kvSeqTensor->GetData<int64_t>();
    int64_t batchSize = static_cast<int64_t>(fiaInfo.bSize);
    int64_t prefixLen = kvData[0];
    if (prefixLen <= 0) {
        OP_LOGE(opName, "PrefixInferAttentionScore requires prefix_len(act_kv[0]) > 0, but got %ld", prefixLen);
        return ge::GRAPH_FAILED;
    }
    int64_t prev = 0;
    int64_t maxReqLen = 0; // 请求batch(i>0)的kv长度最大值
    for (int64_t i = 0; i < batchSize; i++) {
        int64_t qLen = qData[i] - prev;
        int64_t kvLen = kvData[i] - prev;
        if (qData[i] != kvData[i] || qLen <= 0) {
            OP_LOGE(opName, "PrefixInferAttentionScore requires actual_seq_lengths and actual_seq_lengths_kv "
                            "to be identical strictly increasing cumsum arrays, all batches non-empty "
                            "(batch %ld: q=%ld, len=%ld)", i, qData[i], qLen);
            return ge::GRAPH_FAILED;
        }
        if (qLen != kvLen) {
            OP_LOGE(opName, "PrefixInferAttentionScore requires Q length == KV length for each batch "
                            "(batch %ld: q_len=%ld, kv_len=%ld)", i, qLen, kvLen);
            return ge::GRAPH_FAILED;
        }
        if (i > 0) {
            maxReqLen = std::max(maxReqLen, kvLen);
        }
        prev = kvData[i];
    }
    // sparse_mode=0的显式mask: 共享mask [1或2D, >=s1Size, >=prefixLen+maxReqLen]
    // (prefix-merged下kernel按合并后batch索引mask, 逐batch mask无法映射到合并结构, 仅支持共享mask;
    //  s1Size已在tiling入口改写为合并后最大Q长度max(P+L0, maxReqLen))
    if (fiaInfo.attenMaskFlag && sparseMode == SPARSE_MODE_NO_MASK) {
        auto *maskTensor = para.attenMask.tensor;
        if (maskTensor->GetStorageShape().GetDimNum() == 3U &&
            maskTensor->GetStorageShape().GetDim(0) != 1) {
            OP_LOGE(opName, "PrefixInferAttentionScore only supports shared atten_mask "
                            "(2D or batch dim == 1) when sparse_mode=0, but got batch dim %ld",
                    maskTensor->GetStorageShape().GetDim(0));
            return ge::GRAPH_FAILED;
        }
        uint32_t maskS1 = maskTensor->GetStorageShape().GetDim(maskTensor->GetStorageShape().GetDimNum() - 2);
        uint32_t maskS2 = maskTensor->GetStorageShape().GetDim(maskTensor->GetStorageShape().GetDimNum() - 1);
        if (maskS1 < fiaInfo.s1Size || maskS2 < static_cast<uint32_t>(prefixLen + maxReqLen)) {
            OP_LOGE(opName, "PrefixInferAttentionScore requires atten_mask shape [1 or 2D, >=%u, >=%ld], "
                            "but got S1=%u, S2=%u", fiaInfo.s1Size, prefixLen + maxReqLen, maskS1, maskS2);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PiaTilingCheck::Check(const FiaTilingInfo &fiaInfo)
{
    return CheckPrefixInQ(fiaInfo);
}
} // namespace optiling
