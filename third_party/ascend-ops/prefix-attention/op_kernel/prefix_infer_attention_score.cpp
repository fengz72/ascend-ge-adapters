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
 * \brief prefix-in-Q prefill attention kernel入口:
 * 仅实例化TND NoQuant GQA泛化模板(fp16/bf16两个tiling key), 复用attention/common的FIA kernel模板。
 * 输入序与FusedInferAttentionScore的op def一致(prefix_in_Q场景下量化/PA/RoPE/shared prefix等
 * 输入均为nullptr, prefix KV位于key/value张量头部)。
 * prefix-merged: batch 0'=[prefix;req0]合并为单batch(方阵causal), kernel对act数组按索引+1重解释。
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "adv_api/activation/softmax.h"
#include "adv_api/activation/softmaxflashv2.h"
#include "adv_api/math/log.h"
#include "adv_api/select/selectwithbytesmask.h"

// common kernel模板链依赖的全局标量工具(FIA入口经由IFA/PFA完整前导获得, 此处自包含定义,
// 闭包经全链扫描确定; SoftmaxConfig/SoftmaxMode由上方CANN softmax头提供)
constexpr float FLOAT_ZERO = 0;
constexpr float FLOAT_MAX = 3.402823466e+38F; // FLT_MAX
constexpr float FLOAT_EPS = 2.220446049250313e-16;

constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;

// common kernel模板链在fa_base_vector命名空间外引用的全局常量(与vector_common.h内同名常量作用域隔离)
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256;
constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256; // repeat stride不能超过256
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

// 全局标量Max/Min与FLOAT_INF(FIA入口经由IFA/PFA前导获得, 此处自包含定义)
template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

constexpr float FLOAT_INF = 3e+99; // 与AttentionCommon::ConstInfo::FLOAT_INF一致

// common kernel模板链在fa_base_vector命名空间外引用的全局向量工具(源自IFA公共前导, 自包含携带)
using AscendC::LocalTensor;
using AscendC::BinaryRepeatParams;
__aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2; // 32/4 * 2 = 16
    }

    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum],
                repeatElementNum, dLoop, repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum], src0Ub[i * columnCount + dLoop * repeatElementNum],
                    src1Ub[i * blockElementNum], dRemain, 1, repeatParams);
            }
        }
    }
}

__aicore__ inline void VecMulMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // dstUb[i, j] = src0Ub[j] * src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, actualColumnCount] dstUb:[dealRowCount, columnCount]
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;
        uint32_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            // offset = i * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            // offset = loopCount * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            Mul(dstUb[offset], src0Ub, src1Ub[offset], actualColumnCount);
            offset += columnCount;
        }
    }
}

#include "common/op_kernel/arch32/fia_kernel_nonquant.h"

using namespace AscendC;

// TND NoQuant GQA GENERAL 4个tiling key(非split + splitKv, fp16/bf16):
//   fp16 非split : 106000000030000003
//   fp16 splitKv : 106000000030100003
//   bf16 非split : 106000000030022223
//   bf16 splitKv : 106000000030122223
// cube块使用FiaBlockCubeNonQuantGqa(Q2V2KP3高性能tiling: Q常驻L1复用/K三buffer环/D=64时N切256,
// 与FIA生产包HIGH_PERFORMANCE_GQA(103类key)同款, 对齐FIA公平基线);
// FIAType第12个模板参数PREFIX_IN_Q=true(prefix-in-Q语义, FIA侧恒为false);
// 第13个模板参数PREFIX_MERGED=true(batch 0'=[prefix;req0]合并为单batch, act索引+1重解释)
#define INVOKE_PIA_OP_GENERAL_IMPL(templateClass, CubeBlockType, VecBlockType, FdBlockType, ...)                     \
    do {                                                                                                             \
        using CubeBlockTypeT = typename std::conditional<g_coreType == AscendC::AIC,                                 \
                               CubeBlockType<FIAType<__VA_ARGS__>>, CubeBlockType##Dummy<FIAType<__VA_ARGS__>>>::type; \
        using VecBlockTypeT = typename std::conditional<g_coreType == AscendC::AIC,                                  \
                              VecBlockType##Dummy<FIAType<__VA_ARGS__>>, VecBlockType<FIAType<__VA_ARGS__>>>::type;  \
        using FdBlockTypeT = typename std::conditional<g_coreType == AscendC::AIC,                                   \
                             FdBlockType##Dummy<FIAType<__VA_ARGS__>>, FdBlockType<FIAType<__VA_ARGS__>>>::type;     \
        templateClass<FIAType<__VA_ARGS__>, CubeBlockTypeT, VecBlockTypeT, FdBlockTypeT> op;                         \
        FIA_COPY_TILING_DATA(FusedInferAttentionScoreTilingData, tiling);                                            \
        op.Init(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, deqScale1, quantScale1, \
                deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blockTable, queryPaddingSize, \
                kvPaddingSize, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,     \
                keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, keyRopeAntiquantScale, \
                learnableSink, attentionOut, softmaxLse, user, tiling_data, tiling, &tPipe);                         \
        op.Process();                                                                                                \
    } while (0)

#define FIA_COPY_TILING_DATA(tilingDataStruct, tiling)                                                               \
    GET_TILING_DATA_WITH_STRUCT(tilingDataStruct, tiling_data_in, tiling);                                           \
    const tilingDataStruct *__restrict tiling_data = &tiling_data_in;

extern "C" __global__ __aicore__ void prefix_infer_attention_score(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask,
    __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1,
    __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
    __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale,
    __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset,
    __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
    __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
    __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *dequantScaleQuery,
    __gm__ uint8_t *learnableSink, __gm__ uint8_t *qStartIdx, __gm__ uint8_t *kvStartIdx,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    // 不支持310系列
#elif (__CCE_AICORE__ == 200)
    // 不支持dav200
#else
    // 编译系统扫描的tiling key列表(每key生成一份二进制), 用字面量与FIA保持一致
    TILING_KEY_IS(106000000030000003);
    TILING_KEY_IS(106000000030022223);
    TILING_KEY_IS(106000000030100003);
    TILING_KEY_IS(106000000030122223);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if TILING_KEY_VAR == 106000000030000003
    INVOKE_PIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND,
                               false, true, true);
#elif TILING_KEY_VAR == 106000000030022223
    INVOKE_PIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, false, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::TND, false, true, true);
#elif TILING_KEY_VAR == 106000000030100003
    INVOKE_PIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, true, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND,
                               false, true, true);
#elif TILING_KEY_VAR == 106000000030122223
    INVOKE_PIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, false, true, FIA_LAYOUT::TND, false,
                               false, FIA_LAYOUT::TND, false, true, true);
#endif
#endif
}
