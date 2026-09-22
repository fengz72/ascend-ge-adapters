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

#ifndef ACLNN_PREFIX_INFER_ATTENTION_SCORE_H_
#define ACLNN_PREFIX_INFER_ATTENTION_SCORE_H_
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief TND prefix-in-Q prefill attention:
 *   query [P + sum(L), N_q, D], key/value [P + sum(L), N_kv, D] (prefix KV在key/value头部, 只存一份)
 *   actual_seq_lengths与actual_seq_lengths_kv均为cumsum([P, L0, L1, ...])且逐元素相等,
 *   prefix长度P = actual_seq_lengths_kv[0]; kernel侧batch 0'=[prefix;req0]合并为单batch,
 *   其余batch为请求自身; 约束: 仅NoQuant fp16/bf16, sparse_mode 0(显式共享mask)或2(压缩causal),
 *   P>0, actLen>=2(至少一个请求), act严格递增, 逐batch q_len==kv_len
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnPrefixInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    int64_t numHeads, double scaleValue, int64_t preTokens, int64_t nextTokens, int64_t numKeyValueHeads,
    int64_t sparseMode, const aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnPrefixInferAttentionScore is used to perform calculations.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnPrefixInferAttentionScore(void *workspace,
                                                                                  uint64_t workspaceSize,
                                                                                  aclOpExecutor *executor,
                                                                                  const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_PREFIX_INFER_ATTENTION_SCORE_H_
