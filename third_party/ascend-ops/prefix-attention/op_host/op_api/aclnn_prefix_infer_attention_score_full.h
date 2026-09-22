
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FULL_PREFIX_INFER_ATTENTION_SCORE_H_
#define ACLNN_FULL_PREFIX_INFER_ATTENTION_SCORE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFullPrefixInferAttentionScoreGetWorkspaceSize
 * parameters :
 * query : required
 * key : dynamic
 * value : dynamic
 * pseShiftOptional : optional
 * attenMaskOptional : optional
 * actualSeqLengthsOptional : optional
 * actualSeqLengthsKvOptional : optional
 * dequantScale1Optional : optional
 * quantScale1Optional : optional
 * dequantScale2Optional : optional
 * quantScale2Optional : optional
 * quantOffset2Optional : optional
 * antiquantScaleOptional : optional
 * antiquantOffsetOptional : optional
 * blockTableOptional : optional
 * queryPaddingSizeOptional : optional
 * kvPaddingSizeOptional : optional
 * keyAntiquantScaleOptional : optional
 * keyAntiquantOffsetOptional : optional
 * valueAntiquantScaleOptional : optional
 * valueAntiquantOffsetOptional : optional
 * keySharedPrefixOptional : optional
 * valueSharedPrefixOptional : optional
 * actualSharedPrefixLenOptional : optional
 * queryRopeOptional : optional
 * keyRopeOptional : optional
 * keyRopeAntiquantScaleOptional : optional
 * dequantScaleQueryOptional : optional
 * learnableSinkOptional : optional
 * qStartIdxOptional : optional
 * kvStartIdxOptional : optional
 * numHeads : required
 * scale : optional
 * preTokens : optional
 * nextTokens : optional
 * inputLayoutOptional : optional
 * numKeyValueHeads : optional
 * sparseMode : optional
 * innerPrecise : optional
 * blockSize : optional
 * antiquantMode : optional
 * softmaxLseFlag : optional
 * keyAntiquantMode : optional
 * valueAntiquantMode : optional
 * queryQuantMode : optional
 * pseType : optional
 * outDtype : optional
 * attentionOutOut : required
 * softmaxLseOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFullPrefixInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *dequantScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *dequantScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional,
    const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional,
    const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional,
    const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional,
    const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *keyRopeAntiquantScaleOptional,
    const aclTensor *dequantScaleQueryOptional,
    const aclTensor *learnableSinkOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
    int64_t numHeads,
    double scale,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayoutOptional,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t antiquantMode,
    bool softmaxLseFlag,
    int64_t keyAntiquantMode,
    int64_t valueAntiquantMode,
    int64_t queryQuantMode,
    int64_t pseType,
    int64_t outDtype,
    const aclTensor *attentionOutOut,
    const aclTensor *softmaxLseOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnFullPrefixInferAttentionScoreTensorGetWorkspaceSize(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *dequantScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *dequantScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional,
    const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional,
    const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional,
    const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional,
    const aclTensor *valueSharedPrefixOptional,
    const aclTensor *actualSharedPrefixLenOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *keyRopeAntiquantScaleOptional,
    const aclTensor *dequantScaleQueryOptional,
    const aclTensor *learnableSinkOptional,
    const aclTensor *qStartIdxOptional,
    const aclTensor *kvStartIdxOptional,
    int64_t numHeads,
    double scale,
    int64_t preTokens,
    int64_t nextTokens,
    char *inputLayoutOptional,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t antiquantMode,
    bool softmaxLseFlag,
    int64_t keyAntiquantMode,
    int64_t valueAntiquantMode,
    int64_t queryQuantMode,
    int64_t pseType,
    int64_t outDtype,
    const aclTensor *attentionOutOut,
    const aclTensor *softmaxLseOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFullPrefixInferAttentionScore
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFullPrefixInferAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
