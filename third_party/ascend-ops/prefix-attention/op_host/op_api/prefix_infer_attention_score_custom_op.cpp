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
 * \file prefix_infer_attention_score_custom_op.cpp
 * \brief GE自定义算子交付件: EagerExecuteOp + REG_AUTO_MAPPING_OP(V2入图机制)
 *
 * 机制(参考ge仓examples/custom_op/ascendc_add_custom): 本so经ASCEND_CUSTOM_OPP_PATH
 * 加载(指向so所在目录或vendor包根目录), REG_AUTO_MAPPING_OP将算子注册进
 * CustomOpFactory(注册键=类名=op type); 图编译期DNN_VM_CUSTOM引擎认领节点,
 * 运行期经LoweringCustomNode -> ExecuteCustomOp回调本Execute — 静态/动态
 * (符号化shape)图走同一路径。
 *
 * Execute内直调本算子的aclnn入口(Nnopbase executor: tiling+预编译kernel, 与eager
 * 同链路)。act_seq三输入为图Const/Data输入(device tensor), tiling需host值而
 * Nnopbase executor不做valueDepend输入的D2H, 故回调内D2H至pinned宿主缓冲后走
 * list变体(aclnnPrefixInferAttentionScore*), 勿改为Tensor变体直传device指针
 * (tiling会对GetData<int64_t>()做host解引用, SIGSEGV)。
 */

#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <vector>

#include "graph/custom_op.h"
#include "graph/tensor.h"
#include "exe_graph/runtime/eager_op_execution_context.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "graph/utils/type_utils.h"
#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include "log/log.h"

#include "op_api/aclnn_prefix_infer_attention_score.h"

using namespace ge;

namespace {

constexpr int64_t ACLNN_OK = 0;

// 图内连接输入的扁平序(与converter inputs dict序一致, 全部6个均连接):
// 0=query 1=key 2=value 3=atten_mask 4=act_q 5=act_kv
// (prefix KV位于key/value头部, P=act_kv[0], 无独立prefix输入)
constexpr size_t IN_QUERY = 0;
constexpr size_t IN_KEY = 1;
constexpr size_t IN_VALUE = 2;
constexpr size_t IN_MASK = 3;
constexpr size_t IN_ACT_Q = 4;
constexpr size_t IN_ACT_KV = 5;
constexpr size_t IN_NUM = 6;

// attr序(def.cpp声明序, 勿调整)
constexpr size_t ATTR_NUM_HEADS = 0;
constexpr size_t ATTR_SCALE = 1;
constexpr size_t ATTR_PRE_TOKENS = 2;
constexpr size_t ATTR_NEXT_TOKENS = 3;
constexpr size_t ATTR_NUM_KV_HEADS = 5;
constexpr size_t ATTR_SPARSE_MODE = 6;

aclTensor *MakeAclTensor(const gert::Tensor *t)
{
    const auto &storageShape = t->GetShape().GetStorageShape();
    const size_t dimNum = static_cast<size_t>(storageShape.GetDimNum());
    std::vector<int64_t> dims(dimNum);
    for (size_t i = 0; i < dimNum; i++) {
        dims[i] = storageShape.GetDim(i);
    }
    std::vector<int64_t> strides(dimNum);
    int64_t acc = 1;
    for (size_t i = dimNum; i > 0; i--) {
        strides[i - 1] = acc;
        acc *= dims[i - 1];
    }
    auto aclDtype = static_cast<aclDataType>(static_cast<int32_t>(t->GetDataType()));
    return aclCreateTensor(dims.data(), dimNum, aclDtype, strides.data(), 0, ACL_FORMAT_ND,
                           dims.data(), dimNum, const_cast<void *>(t->GetAddr()));
}

// Nnopbase线程局部大内存池生命周期(镜像eager扩展ACLNN_CMD约定, aclnn_common.h):
// InitHugeMemThreadLocal → GetWorkspaceSize → launch → ReleaseHugeMem → UnInitHugeMemThreadLocal。
// GE回调内无外层ACLNN_CMD包装, 缺失时同图第二次run的aclnn内部op::internal::Allocate
// 会在已释放的池上分配(段错误), 故每次Execute自管三件套。符号位于libnnopbase.so,
// 经本so的DT_NEEDED(libopapi.so)链dlsym可达。
using PiaInitHugeMemFn = int (*)(void *, bool);
using PiaUnInitHugeMemFn = void (*)(void *, bool);
using PiaReleaseHugeMemFn = void (*)(void *, bool);

PiaInitHugeMemFn PiaGetInitHugeMem()
{
    return reinterpret_cast<PiaInitHugeMemFn>(dlsym(RTLD_DEFAULT, "InitHugeMemThreadLocal"));
}

PiaUnInitHugeMemFn PiaGetUnInitHugeMem()
{
    return reinterpret_cast<PiaUnInitHugeMemFn>(dlsym(RTLD_DEFAULT, "UnInitHugeMemThreadLocal"));
}

PiaReleaseHugeMemFn PiaGetReleaseHugeMem()
{
    return reinterpret_cast<PiaReleaseHugeMemFn>(dlsym(RTLD_DEFAULT, "ReleaseHugeMem"));
}

// act_seq宿主缓冲必须为pinned内存(aclrtMallocHost): Nnopbase hostInput指针会被kernel
// 按GM直接寻址, 普通pageable堆内存会被device读到垃圾值(batch边界失效)。
// thread_local + 一次分配永不释放(168B量级, 生命周期覆盖异步launch)。
int64_t *PiaGetPinnedActBuf(size_t need)
{
    static thread_local int64_t *pinnedAct = nullptr;
    static thread_local size_t pinnedCap = 0;
    if (pinnedAct == nullptr || pinnedCap < need) {
        if (pinnedAct != nullptr) {
            (void)aclrtFreeHost(pinnedAct);
        }
        void *buf = nullptr;
        if (aclrtMallocHost(&buf, need) != ACL_SUCCESS || buf == nullptr) {
            return nullptr;
        }
        pinnedAct = static_cast<int64_t *>(buf);
        pinnedCap = need;
    }
    return pinnedAct;
}

} // namespace

// 类名必须与op type一致(REG_AUTO_MAPPING_OP以类名为注册键; 与def.cpp的ops::类
// 不同命名空间不同库, 无冲突)
class PrefixInferAttentionScore : public EagerExecuteOp {
public:
    graphStatus Execute(gert::EagerOpExecutionContext *ctx) override
    {
        // Nnopbase线程局部内存池生命周期自管(eager链路ACLNN_CMD同款)
        auto initMem = PiaGetInitHugeMem();
        auto unInitMem = PiaGetUnInitHugeMem();
        auto releaseMem = PiaGetReleaseHugeMem();
        if (initMem != nullptr) {
            (void)initMem(nullptr, false);
        }

        const gert::Tensor *in[IN_NUM];
        for (size_t i = 0; i < IN_NUM; i++) {
            in[i] = ctx->GetInputTensor(i);
            OP_CHECK_NULL_WITH_CONTEXT(ctx, in[i]);
        }
        const gert::RuntimeAttrs *attrs = ctx->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(ctx, attrs);
        const int64_t numHeads = *(attrs->GetInt(ATTR_NUM_HEADS));
        const double scale = static_cast<double>(*(attrs->GetFloat(ATTR_SCALE)));
        const int64_t preTokens = *(attrs->GetInt(ATTR_PRE_TOKENS));
        const int64_t nextTokens = *(attrs->GetInt(ATTR_NEXT_TOKENS));
        const int64_t numKvHeads = *(attrs->GetInt(ATTR_NUM_KV_HEADS));
        const int64_t sparseMode = *(attrs->GetInt(ATTR_SPARSE_MODE));
        // 其余attr(inner_precise/block_size/antiquant系/lse_flag/pse_type/out_dtype)由
        // aclnn wrapper内部按交付语义固化(与eager一致), 此处不透传

        // 输出0: attention_out (shape/dtype/format同query)
        gert::Tensor *outT = ctx->MallocOutputTensor(0, in[IN_QUERY]->GetShape(), in[IN_QUERY]->GetFormat(),
                                                     in[IN_QUERY]->GetDataType(),
                                                     static_cast<size_t>(in[IN_QUERY]->GetSize()));
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outT);

        // 构造acl描述子: key/value为单元素tensorlist
        aclTensor *qA = MakeAclTensor(in[IN_QUERY]);
        aclTensor *kA = MakeAclTensor(in[IN_KEY]);
        aclTensor *vA = MakeAclTensor(in[IN_VALUE]);
        aclTensor *maskA = MakeAclTensor(in[IN_MASK]);
        aclTensor *outA = MakeAclTensor(outT);
        const aclTensor *keyArr[1] = {kA};
        const aclTensor *valueArr[1] = {vA};
        aclTensorList *keyList = aclCreateTensorList(keyArr, 1);
        aclTensorList *valueList = aclCreateTensorList(valueArr, 1);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, qA);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, keyList);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, valueList);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outA);

        // act_seq两输入(Const/Data device int64): D2H读当前值后经pinned宿主缓冲构造
        // aclIntArray走list变体(tiling需host值)
        const size_t nQ = static_cast<size_t>(in[IN_ACT_Q]->GetShapeSize());
        const size_t nKv = static_cast<size_t>(in[IN_ACT_KV]->GetShapeSize());
        int64_t *pinnedAct = PiaGetPinnedActBuf((nQ + nKv + 2) * sizeof(int64_t));
        OP_CHECK_NULL_WITH_CONTEXT(ctx, pinnedAct);
        int64_t *hActQ = pinnedAct;
        int64_t *hActKv = pinnedAct + nQ + 1;
        auto d2h = [ctx](const gert::Tensor *t, int64_t *dst, size_t n) -> bool {
            if (n == 0 || t->GetAddr() == nullptr) {
                OP_LOGE(ctx->GetNodeName(), "PIA[V2] act_seq tensor invalid");
                return false;
            }
            return aclrtMemcpy(dst, n * sizeof(int64_t), t->GetAddr(), n * sizeof(int64_t),
                               ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
        };
        OP_CHECK_IF(!d2h(in[IN_ACT_Q], hActQ, nQ) || !d2h(in[IN_ACT_KV], hActKv, nKv),
            OP_LOGE(ctx->GetNodeName(), "PIA[V2] act_seq D2H failed"), return GRAPH_FAILED);
        aclIntArray *actQArr = aclCreateIntArray(hActQ, nQ);
        aclIntArray *actKvArr = aclCreateIntArray(hActKv, nKv);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, actQArr);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, actKvArr);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto ret = aclnnPrefixInferAttentionScoreGetWorkspaceSize(
            qA, keyList, valueList, maskA, actQArr, actKvArr,
            numHeads, scale, preTokens, nextTokens, numKvHeads, sparseMode,
            outA, &workspaceSize, &executor);
        if (ret != ACLNN_OK || executor == nullptr) {
            OP_LOGE(ctx->GetNodeName(), "PIA[V2] GetWorkspaceSize failed, ret=%d", static_cast<int>(ret));
            if (unInitMem != nullptr) {
                unInitMem(nullptr, false);
            }
            return GRAPH_FAILED;
        }
        void *workspace = nullptr;
        if (workspaceSize > 0) {
            workspace = ctx->MallocWorkSpace(static_cast<size_t>(workspaceSize));
            OP_CHECK_NULL_WITH_CONTEXT(ctx, workspace);
        }
        ret = aclnnPrefixInferAttentionScore(workspace, workspaceSize, executor,
                                             static_cast<aclrtStream>(ctx->GetStream()));
        // launch后释放/反初始化线程局部内存池(镜像eager: ReleaseHugeMem + UnInit)
        if (releaseMem != nullptr) {
            releaseMem(nullptr, false);
        }
        if (unInitMem != nullptr) {
            unInitMem(nullptr, false);
        }
        if (ret != ACLNN_OK) {
            OP_LOGE(ctx->GetNodeName(), "PIA[V2] run failed, ret=%d", static_cast<int>(ret));
            return GRAPH_FAILED;
        }

        aclDestroyTensor(qA);
        aclDestroyTensor(kA);
        aclDestroyTensor(vA);
        aclDestroyTensor(maskA);
        aclDestroyTensor(outA);
        aclDestroyTensorList(keyList);
        aclDestroyTensorList(valueList);
        aclDestroyIntArray(actQArr);
        aclDestroyIntArray(actKvArr);
        return GRAPH_SUCCESS;
    }
};

// 类名必须与op type一致(GE按op type名查找CustomOpFactory)
REG_AUTO_MAPPING_OP(PrefixInferAttentionScore);
