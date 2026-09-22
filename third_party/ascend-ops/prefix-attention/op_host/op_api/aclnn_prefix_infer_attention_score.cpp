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

#include "aclnn_prefix_infer_attention_score.h"

#include "opdev/common_types.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

/* vendored full参数plumbing(op_api/aclnn_..._full.cpp, 改名自仓内autogen inner)(def输入序/attr序与FusedInferAttentionScore一致) */
extern aclnnStatus aclnnFullPrefixInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
    const aclTensor *pseShiftOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *dequantScale1Optional, const aclTensor *quantScale1Optional,
    const aclTensor *dequantScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional,
    const aclTensor *dequantScaleQueryOptional, const aclTensor *learnableSinkOptional,
    const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    int64_t numHeads, double scale, int64_t preTokens, int64_t nextTokens, char *inputLayoutOptional,
    int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise, int64_t blockSize,
    int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode, int64_t valueAntiquantMode,
    int64_t queryQuantMode, int64_t pseType, int64_t outDtype,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnFullPrefixInferAttentionScore(void *workspace, uint64_t workspaceSize,
                                                       aclOpExecutor *executor, const aclrtStream stream);

namespace {

aclnnStatus CreateSoftmaxLsePlaceholder(const aclTensor *&placeHolder, aclTensor *&tempTensor)
{
    // softmax_lse_flag恒为false, 输出槽位传空占位tensor
    std::vector<int64_t> shape = {0};
    int64_t addr = 0xff;
    tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                 shape.data(), shape.size(), static_cast<void *>(&addr));
    if (tempTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "create softmax lse placeholder failed");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    placeHolder = tempTensor;
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnPrefixInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    int64_t numHeads, double scaleValue, int64_t preTokens, int64_t nextTokens, int64_t numKeyValueHeads,
    int64_t sparseMode, const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    const aclTensor *placeHolder = nullptr;
    aclTensor *tempTensor = nullptr;
    aclnnStatus ret = CreateSoftmaxLsePlaceholder(placeHolder, tempTensor);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    char inputLayout[] = "TND"; // 仅TND布局
    ret = aclnnFullPrefixInferAttentionScoreGetWorkspaceSize(
        query, key, value, nullptr, attenMaskOptional, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads, sparseMode,
        1,  // innerPrecise
        0,  // blockSize
        0,  // antiquantMode
        false,  // softmaxLseFlag
        0, 0, 0,  // keyAntiquantMode/valueAntiquantMode/queryQuantMode
        0,  // pseType
        0,  // outDtype
        attentionOut, placeHolder, workspaceSize, executor);
    aclDestroyTensor(tempTensor);
    return ret;
}

aclnnStatus aclnnPrefixInferAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            const aclrtStream stream)
{
    return aclnnFullPrefixInferAttentionScore(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
