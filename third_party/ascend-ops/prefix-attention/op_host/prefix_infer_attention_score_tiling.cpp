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
 * \file prefix_infer_attention_score_tiling.cpp
 * \brief tiling入口: 复用FIA(9.0.0)的FiaInfoParser按索引解析输入/attr(def输入/attr序与FIA一致),
 * 随后从act_kv[0]取prefix长度P并硬编码prefix-in-Q+prefix-merged语义(prefix KV位于key/value头部,
 * batch 0'=[prefix;req0]合并为单batch, act数组按索引+1重解释, 生效bSize=B-1), 再走本算子自有checker
 * 校验, 最后复用common模板注册表分派(FiaTilingNonQuant)。
 * tiling data复用FusedInferAttentionScoreTilingData(prefixInQ/prefixMerged为kernel编译期模板标记, 不占tiling字段)。
 */
#include "prefix_infer_attention_score_tiling.h"
#include "prefix_infer_attention_score_tiling_check.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "common_impl/fia_tiling_templates_registry.h"
#include "common_impl/arch32/fia_tiling_nonquant.h"
#include "fia_tiling_impl/fused_infer_attention_score_tiling.h"
#include "fia_tiling_impl/fused_infer_attention_score_tiling_info_parser.h"
#include "fia_tiling_impl/prefix_infer_attention_score_base_tiling.h"

using namespace ge;

namespace optiling {
// 基础注册(自包含): kernel编译时框架按算子名查询tiling def, 基础注册将vendored tiling struct
// (FusedInferAttentionScoreTilingData等, common kernel模板依赖)带入生成的tiling_data.h;
// 基础注册使用独立命名的等价布局struct, 避免与per-key注册同名导致生成文件类重定义
REGISTER_TILING_DATA_CLASS(PrefixInferAttentionScore, PrefixInferAttentionScoreBaseTilingData)
// kernel侧实例化TND NoQuant GQA泛化路径的4个tiling key(fp16/bf16 × 非split/splitKv)
REGISTER_TILING_DATA_CLASS(PrefixInferAttentionScore_106000000030000003, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(PrefixInferAttentionScore_106000000030022223, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(PrefixInferAttentionScore_106000000030100003, FusedInferAttentionScoreTilingData)
REGISTER_TILING_DATA_CLASS(PrefixInferAttentionScore_106000000030122223, FusedInferAttentionScoreTilingData)

// 注册NoQuant GQA泛化tiling模板(与FIA同模板, 按算子名隔离); 不支持场景在IsCapable处自然拒绝
REGISTER_TILING_TEMPLATE_FIA(PrefixInferAttentionScore, FiaTilingNonQuant,
    std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 29);

static ge::graphStatus TilingPrepareForPrefixInferAttentionScore(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// 图模式二进制匹配: 生成与binary_info json(构建期)一致的simplifiedKey。
// 格式(与内置FIA及本包json对齐): {OpType}/d={det},p={implmode}/{query},{fmt}/{key},{fmt}/{value},{fmt}/{pse},{fmt}/0,2
// 未连接可选输入(pse_shift)用query信息代替(仓内matmul_v3同款约定); p固定1(包内全部为high_performance二进制)。
// 未注册此函数时GE图编译无法匹配预编译kernel, 将回退JIT源码编译(launch参数错误, 输出错乱)。
static ge::graphStatus PiaGenSimplifiedKey(gert::TilingContext *context, ge::char_t *simplifiedKey)
{
    static constexpr size_t KEY_MAX_LEN = 256;
    static constexpr int32_t IMPL_MODE_HIGH_PERFORMANCE = 1;
    const char *opType = "PrefixInferAttentionScore";
    const gert::CompileTimeTensorDesc *queryDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    auto queryDtype = queryDesc->GetDataType();
    auto queryFmt = queryDesc->GetStorageFormat();

    int32_t det = context->GetDeterministic() != 0 ? 1 : 0;
    std::string key;
    key.reserve(KEY_MAX_LEN);
    key.append(opType).append("/d=").append(std::to_string(det))
        .append(",p=").append(std::to_string(IMPL_MODE_HIGH_PERFORMANCE));
    constexpr int32_t KEY_INPUTS[] = {0, 1, 2, 3}; // query/key/value/pse_shift
    for (int32_t idx : KEY_INPUTS) {
        const gert::CompileTimeTensorDesc *desc = context->GetInputDesc(idx);
        auto dtype = (desc != nullptr) ? desc->GetDataType() : queryDtype;
        auto fmt = (desc != nullptr) ? desc->GetStorageFormat() : queryFmt;
        key.append("/").append(std::to_string(static_cast<int32_t>(dtype)))
            .append(",").append(std::to_string(static_cast<int32_t>(fmt)));
    }
    key.append("/0,2");
    OP_CHECK_IF(key.size() >= KEY_MAX_LEN,
        OP_LOGE(context->GetNodeName(), "simplifiedKey too long: %zu", key.size()),
        return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(simplifiedKey, KEY_MAX_LEN, key.c_str(), key.size() + 1);
    OP_CHECK_IF(ret != EOK,
        OP_LOGE(context->GetNodeName(), "memcpy_s simplifiedKey failed, ret=%d", static_cast<int>(ret)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrefixInferAttentionScore(gert::TilingContext *context)
{
    FiaTilingInfo fiaInfo;
    // 复用FIA parser: 输入/attr按索引读取, def序与FIA(9.0.0)一致
    FiaInfoParser fiaInfoParser(context);
    if (fiaInfoParser.Parse(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // prefix-in-Q语义: prefix KV位于key/value张量头部(batch 0区), prefix长度P=act_kv[0]
    // (key/value_shared_prefix与actual_shared_prefix_len输入槽位保留但不连接)
    fiaInfo.prefixInQFlag = true;
    const gert::Tensor *actKvTensor = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    if (actKvTensor != nullptr && actKvTensor->GetShapeSize() > 0 &&
        actKvTensor->GetData<int64_t>() != nullptr) {
        uint32_t prefixLen = static_cast<uint32_t>(actKvTensor->GetData<int64_t>()[0]);
        fiaInfo.sysPrefixFlag = true;
        fiaInfo.systemPrefixLen = prefixLen;
        fiaInfo.systemPrefixMaxLen = prefixLen;
    }

    // prefix-merged: batch 0'=[prefix;req0]合并为单batch(q=kv=[0,P+L0)方阵causal), 请求j'对应batch j'。
    // kernel/分核侧对act数组按"索引整体+1"重解释: batch 0'长度=act[1], batch j'基底/长度=act[j']/差分(act[j'+1],act[j'])。
    // 生效bSize=B-1(由FiaTilingNonQuant物化), s1Size改写为合并后最大Q长度(保证mm1/mm2 workspace的M轴尺寸);
    // actLen<2时保持false, 由PiaTilingCheck拒绝(纯prefix无请求为非法输入)
    fiaInfo.prefixMergedFlag = actKvTensor != nullptr && actKvTensor->GetShapeSize() >= 2 &&
        actKvTensor->GetData<int64_t>() != nullptr;
    if (fiaInfo.prefixMergedFlag) {
        const int64_t *actData = actKvTensor->GetData<int64_t>();
        int64_t actLen = actKvTensor->GetShapeSize();
        int64_t mergedS1Max = actData[1]; // batch 0'的q_len = P+L0
        for (int64_t i = 2; i < actLen; i++) {
            int64_t reqLen = actData[i] - actData[i - 1];
            mergedS1Max = reqLen > mergedS1Max ? reqLen : mergedS1Max;
        }
        fiaInfo.s1Size = static_cast<uint32_t>(mergedS1Max);
    }

    // 本算子自有约束校验(替代FIA checker: TND+prefix-in-Q专属约束)
    if (PiaTilingCheck::Check(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &fiaInfo);
}

IMPL_OP_OPTILING(PrefixInferAttentionScore)
    .GenSimplifiedKey(PiaGenSimplifiedKey)
    .TilingInputsDataDependency({ACTUAL_SEQ_Q_INDEX, ACTUAL_SEQ_KV_INDEX, QUERY_PADDING_SIZE_INDEX,
                                 KV_PADDING_SIZE_INDEX},
                                 {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .Tiling(TilingPrefixInferAttentionScore)
    .TilingParse<FusedInferAttentionScoreCompileInfo>(TilingPrepareForPrefixInferAttentionScore);
} // namespace optiling
