#include <string.h>
#include "graph/types.h"
#include "aclnn_prefix_infer_attention_score_full.h"

#if 0 // ACLNN_WITH_BINARY: 模板autogen不生成op_resource头, 独立工程禁用binary资源表
#include <vector>
#include <tuple>
#include <map>
#include "graph/ascend_string.h"
#include "PrefixInferAttentionScore_op_resource.h"
using OP_HOST_FUNC_HANDLE = std::vector<void *>;
using OP_RES = std::tuple<const uint8_t *, const uint8_t *>;
using OP_BINARY_RES = std::vector<OP_RES>;
using OP_RUNTIME_KB_RES = std::vector<OP_RES>;
using OP_RESOURCES = std::map<ge::AscendString,
    std::tuple<OP_HOST_FUNC_HANDLE, OP_BINARY_RES, OP_RUNTIME_KB_RES>>;
using OP_SOC_RESOURCES = std::map<ge::AscendString, std::tuple<OP_HOST_FUNC_HANDLE,
    std::map<ge::AscendString, OP_BINARY_RES>, OP_RUNTIME_KB_RES>>;
namespace op {
extern uint32_t GenOpTypeId(const char *op_name, const OP_RESOURCES &op_resources);
extern uint32_t GenOpTypeId(const char *op_name, const OP_SOC_RESOURCES &op_resources);
}
#endif

namespace {
typedef struct {
    uint32_t id;
    const char *funcName;
    bool hasReg;
} NnopbaseDfxId;
typedef struct {
    ge::DataType dtype;
    ge::Format format;
} TensorDesc;
typedef struct {
    TensorDesc *inputsDesc;
    size_t inputsNum;
    TensorDesc *outputsDesc;
    size_t outputsNum;
} SupportInfo;
typedef struct {
    SupportInfo *supportInfo;
    size_t num;
} OpSocSupportInfo;
typedef struct {
    OpSocSupportInfo *socSupportInfo;
    size_t num;
} OpSupportList;
enum SocType {
    SOC_VERSION_ASCEND910A = 1,
    SOC_VERSION_ASCEND910B = 2,
    SOC_VERSION_ASCEND910_93 = 3,
    SOC_VERSION_ASCEND950 = 4,
    SOC_VERSION_ASCEND310P = 5,
    SOC_VERSION_ASCEND310B = 6,
    SOC_VERSION_BS9SX1A = 7,
    SOC_VERSION_ASCEND610Lite = 8,
    SOC_VERSION_MC61AM21A = 10, // 9 is deprecated
    SOC_VERSION_MC62CM12A = 11,
    SOC_VERSION_BS9SX2A = 12,
    SOC_VERSION_ASCEND910_96 = 13,
    SOC_VERSION_KIRINX90 = 14,
    SOC_VERSION_KIRIN9030 = 15
};
enum NnopbaseAttrDtype {
    kNnopbaseBool = 0U,
    kNnopbaseFloat,
    kNnopbaseInt,
    kNnopbaseString,
    kNnopbaseAttrEnd
};
uint32_t socSupportList[] = {SOC_VERSION_ASCEND910_93,SOC_VERSION_ASCEND910B};
uint32_t socSupportListLen = 2;

TensorDesc inputDesc0_0[31] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_1[31] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_2[31] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc0_3[31] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc outputDesc0_0[2] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_1[2] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_2[2] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc0_3[2] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
SupportInfo list0_0 = {inputDesc0_0, 31, outputDesc0_0, 2};
SupportInfo list0_1 = {inputDesc0_1, 31, outputDesc0_1, 2};
SupportInfo list0_2 = {inputDesc0_2, 31, outputDesc0_2, 2};
SupportInfo list0_3 = {inputDesc0_3, 31, outputDesc0_3, 2};
SupportInfo supportInfo0[4] = {list0_0, list0_1, list0_2, list0_3};
OpSocSupportInfo socSupportInfo0= {supportInfo0, 4};

TensorDesc inputDesc1_0[31] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc1_1[31] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc1_2[31] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc inputDesc1_3[31] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BOOL, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_UINT64, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT32, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND},
     {ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND},
     {ge::DT_INT64, ge::FORMAT_ND}};
TensorDesc outputDesc1_0[2] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_1[2] =
    {{ge::DT_FLOAT16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_2[2] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
TensorDesc outputDesc1_3[2] =
    {{ge::DT_BF16, ge::FORMAT_ND},
     {ge::DT_FLOAT, ge::FORMAT_ND}};
SupportInfo list1_0 = {inputDesc1_0, 31, outputDesc1_0, 2};
SupportInfo list1_1 = {inputDesc1_1, 31, outputDesc1_1, 2};
SupportInfo list1_2 = {inputDesc1_2, 31, outputDesc1_2, 2};
SupportInfo list1_3 = {inputDesc1_3, 31, outputDesc1_3, 2};
SupportInfo supportInfo1[4] = {list1_0, list1_1, list1_2, list1_3};
OpSocSupportInfo socSupportInfo1= {supportInfo1, 4};

OpSocSupportInfo opSocSupportList[2] = {socSupportInfo0, socSupportInfo1};
OpSupportList supportList = {opSocSupportList, 2};

[[maybe_unused]] uint32_t NNOPBASE_PrefixInferAttentionScore = 0U;
} // namespace

extern void NnopbaseOpLogE(const aclnnStatus code, const char *const expr);

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus NnopbaseCreateExecutorSpace(void **space);
extern void *NnopbaseGetExecutor(void *space, const char *opType, char *inputsDesc, uint32_t inputNum,
                                 char *outputsDesc, uint32_t outputNum, char *attrsDesc, uint32_t attrsNum);
extern aclnnStatus NnopbaseAddInput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIgnoreContinuesInput(void *executor,
                                                   const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddIntArrayInput(void *executor, const aclIntArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddBoolArrayInput(void *executor, const aclBoolArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddFloatArrayInput(void *executor, const aclFloatArray *array, const uint32_t index);
extern aclnnStatus NnopbaseAddOutput(void *executor, const aclTensor *tensor, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicInput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddDynamicOutput(void *executor, const aclTensorList *tensor_list, const uint32_t index);
extern aclnnStatus NnopbaseAddAttrWithDtype(void *executor, void *attrAddr, size_t attrLen, const size_t index, const NnopbaseAttrDtype dtype);
extern aclnnStatus NnopbaseAddIntArrayAttr(void *executor, const aclIntArray* array, const size_t index);
extern aclnnStatus NnopbaseAddFloatArrayAttr(void *executor, const aclFloatArray* array, const size_t index);
extern aclnnStatus NnopbaseAddBoolArrayAttr(void *executor, const aclBoolArray* array, const size_t index);
extern aclnnStatus NnopbaseAddArrayAttrWithDtype(void *executor, void *array, const size_t len, const size_t elementSize, const size_t index, const NnopbaseAttrDtype dtype);
extern uint64_t NnopbaseMsprofSysTime();
extern aclnnStatus NnopbaseAddTilingId(void *executor, NnopbaseDfxId *tilingId);
extern void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);
extern aclnnStatus NnopbaseRunForWorkspace(void *executor, uint64_t *workspaceLen);
extern aclnnStatus NnopbaseRunWithWorkspace(void *executor, aclrtStream stream, void *workspace, uint64_t workspaceSize);
extern aclnnStatus NnopbaseAddSupportList(void *executor, OpSupportList *list, uint32_t *socSupportList, size_t socSupportListLen);
extern aclnnStatus NnopbaseAddScalarInput(void *executor, const aclScalar *scalar, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern aclnnStatus NnopbaseAddScalarListInput(void *executor, const aclScalarList *scalarList, const uint32_t index, const int32_t srcIndex, const ge::DataType dtype);
extern void NnopbaseAddOpTypeId(void *executor, const uint32_t opTypeId);
extern aclnnStatus __attribute__((weak)) NnopbaseAddParamName(void *executor, const uint32_t index, const char *name, const bool isInput);
extern aclnnStatus __attribute__((weak)) NnopbaseSetFormatMatchMode(void *executor, const uint32_t mode);
extern aclnnStatus NnopbaseSetRef(void *executor, const size_t inputIrIdx, const size_t outputIrIdx);
extern void __attribute__((weak)) NnopbaseSetMatchArgsFlag(void *executor);
extern bool __attribute__((weak)) NnopbaseMatchArgs(void *executor, uint64_t *workspaceLen);
extern aclnnStatus NnopbaseGetUnContiguousTensors(void *executor, const aclTensorList **inTensors);
extern aclnnStatus NnopbaseSetUnContExecutor(void *executor, aclOpExecutor *inExe, const size_t inWsSize);
extern aclnnStatus NnopbaseGetUnContExecutor(void *executor, aclOpExecutor **inExe, size_t *inWsSize);
extern aclnnStatus NnopbaseGetRefUnContiguousTensors(void *executor, const aclTensorList **unContTensors, const aclTensorList **contTensors);
extern aclnnStatus NnopbaseSetViewCopyExecutor(void *executor, aclOpExecutor *exe);
extern aclnnStatus NnopbaseGetViewCopyExecutor(void *executor, aclOpExecutor **exe);
extern aclnnStatus NnopbaseReleaseRefContiguousTensors(void *executor, const aclTensorList **tensors);
extern void *NnopbaseGetApiFunc(const char *funcName);
using AclnnContiguousGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensorList *, uint64_t *, aclOpExecutor **);
using AclnnViewCopyGetWorkspaceSizeFunc = aclnnStatus (*)(const aclTensorList *, const aclTensorList *, uint64_t *, aclOpExecutor **);
using AclnnFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

#define ACLNN_SUCCESS  0
#define ACLNN_ERR_PARAM_NULLPTR 161001
#define ACLNN_ERR_PARAM_INVALID 161002

#define NNOPBASE_ASSERT_OK_RETVAL(v)                                    \
    do {                                                                \
        const aclnnStatus _chk_stutus = (v);                            \
        if (_chk_stutus != ACLNN_SUCCESS) {                             \
            NnopbaseOpLogE(_chk_stutus, #v);                            \
            return _chk_stutus;                                         \
        }                                                               \
    } while (false)

#define NNOPBASE_ASSERT_NOTNULL_RETVAL(v)                               \
    do {                                                                \
        if ((v) == nullptr) {                                           \
            NnopbaseOpLogE(ACLNN_ERR_PARAM_NULLPTR, #v " != nullptr");  \
            return ACLNN_ERR_PARAM_NULLPTR;                             \
        }                                                               \
    } while (false)

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
    aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
#ifdef ACLNN_WITH_BINARY
    static uint32_t PrefixInferAttentionScoreOpTypeId = op::GenOpTypeId("PrefixInferAttentionScore", PrefixInferAttentionScore_RESOURCES);
#endif
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    static NnopbaseDfxId tilingId = {0x60000, "aclnnFullPrefixInferAttentionScoreTiling", false};
    void *nnopExecutor;
    static void *executorSpace = NULL;
    const char *opType = "PrefixInferAttentionScore";
    char inputDesc[] = {1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    char outputDesc[] = {1, 1};
    char attrDesc[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    NNOPBASE_ASSERT_NOTNULL_RETVAL(query);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(key);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(value);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(attentionOutOut);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(softmaxLseOut);

    if (!executorSpace) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCreateExecutorSpace(&executorSpace));
    }
    nnopExecutor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(nnopExecutor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    *executor = reinterpret_cast<aclOpExecutor *>(nnopExecutor);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddTilingId(*executor, &tilingId));
    if (NnopbaseSetMatchArgsFlag != NULL) {
        NnopbaseSetMatchArgsFlag(*executor);
    }
#ifdef ACLNN_WITH_BINARY
    NnopbaseAddOpTypeId(*executor, PrefixInferAttentionScoreOpTypeId);
#endif
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, query, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddDynamicInput(*executor, key, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddDynamicInput(*executor, value, 2));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, pseShiftOptional, 3));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, attenMaskOptional, 4));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayInput(*executor, actualSeqLengthsOptional, 5));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayInput(*executor, actualSeqLengthsKvOptional, 6));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, dequantScale1Optional, 7));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, quantScale1Optional, 8));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, dequantScale2Optional, 9));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, quantScale2Optional, 10));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, quantOffset2Optional, 11));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, antiquantScaleOptional, 12));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, antiquantOffsetOptional, 13));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, blockTableOptional, 14));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, queryPaddingSizeOptional, 15));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, kvPaddingSizeOptional, 16));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, keyAntiquantScaleOptional, 17));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, keyAntiquantOffsetOptional, 18));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, valueAntiquantScaleOptional, 19));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, valueAntiquantOffsetOptional, 20));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, keySharedPrefixOptional, 21));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, valueSharedPrefixOptional, 22));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayInput(*executor, actualSharedPrefixLenOptional, 23));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, queryRopeOptional, 24));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, keyRopeOptional, 25));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, keyRopeAntiquantScaleOptional, 26));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, dequantScaleQueryOptional, 27));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, learnableSinkOptional, 28));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayInput(*executor, qStartIdxOptional, 29));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddIntArrayInput(*executor, kvStartIdxOptional, 30));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&numHeads), sizeof(int64_t), 0, kNnopbaseInt));
    float tmp1 = static_cast<float>(scale);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp1), sizeof(float), 1, kNnopbaseFloat));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&preTokens), sizeof(int64_t), 2, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&nextTokens), sizeof(int64_t), 3, kNnopbaseInt));
    if (inputLayoutOptional) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(inputLayoutOptional), strlen(inputLayoutOptional) + 1, 4, kNnopbaseString));
    } else {
        static char *inputLayoutOptionalDef = "TND";
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(inputLayoutOptionalDef), strlen(inputLayoutOptionalDef) + 1, 4, kNnopbaseString));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&numKeyValueHeads), sizeof(int64_t), 5, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&sparseMode), sizeof(int64_t), 6, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&innerPrecise), sizeof(int64_t), 7, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&blockSize), sizeof(int64_t), 8, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&antiquantMode), sizeof(int64_t), 9, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&softmaxLseFlag), sizeof(bool), 10, kNnopbaseBool));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&keyAntiquantMode), sizeof(int64_t), 11, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&valueAntiquantMode), sizeof(int64_t), 12, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&queryQuantMode), sizeof(int64_t), 13, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&pseType), sizeof(int64_t), 14, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&outDtype), sizeof(int64_t), 15, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, attentionOutOut, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, softmaxLseOut, 1));
    if (NnopbaseMatchArgs != NULL) {
        if (NnopbaseMatchArgs(*executor, workspaceSize)) {
            NnopbaseReportApiInfo(timeStamp, dfxId);
            return ACLNN_SUCCESS;
        }
    }
    if (NnopbaseAddParamName != NULL) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "query", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "key", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 2, "value", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 3, "pseShiftOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 4, "attenMaskOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 5, "actualSeqLengthsOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 6, "actualSeqLengthsKvOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 7, "dequantScale1Optional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 8, "quantScale1Optional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 9, "dequantScale2Optional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 10, "quantScale2Optional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 11, "quantOffset2Optional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 12, "antiquantScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 13, "antiquantOffsetOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 14, "blockTableOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 15, "queryPaddingSizeOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 16, "kvPaddingSizeOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 17, "keyAntiquantScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 18, "keyAntiquantOffsetOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 19, "valueAntiquantScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 20, "valueAntiquantOffsetOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 21, "keySharedPrefixOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 22, "valueSharedPrefixOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 23, "actualSharedPrefixLenOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 24, "queryRopeOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 25, "keyRopeOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 26, "keyRopeAntiquantScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 27, "dequantScaleQueryOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 28, "learnableSinkOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 29, "qStartIdxOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 30, "kvStartIdxOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "attentionOutOut", false));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "softmaxLseOut", false));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddSupportList(*executor, &supportList, socSupportList, socSupportListLen));

    const aclTensorList *inUnContTensors = nullptr;
    NnopbaseGetUnContiguousTensors(*executor, &inUnContTensors);
    aclOpExecutor *aclInExecutor = nullptr;
    uint64_t inContWorkspaceSize = 0U;
    if (inUnContTensors != nullptr) {
        static AclnnContiguousGetWorkspaceSizeFunc aclnnContiguousGetWorkspaceSize = (AclnnContiguousGetWorkspaceSizeFunc)NnopbaseGetApiFunc("aclnnContiguousGetWorkspaceSize");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(aclnnContiguousGetWorkspaceSize);
        NNOPBASE_ASSERT_OK_RETVAL(aclnnContiguousGetWorkspaceSize(inUnContTensors, &inContWorkspaceSize, &aclInExecutor));
    }
    NnopbaseSetUnContExecutor(*executor, aclInExecutor, inContWorkspaceSize);

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunForWorkspace(*executor, workspaceSize));
    *workspaceSize += inContWorkspaceSize;
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

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
    aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
#ifdef ACLNN_WITH_BINARY
    static uint32_t PrefixInferAttentionScoreOpTypeId = op::GenOpTypeId("PrefixInferAttentionScore", PrefixInferAttentionScore_RESOURCES);
#endif
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    static NnopbaseDfxId tilingId = {0x60000, "aclnnFullPrefixInferAttentionScoreTiling", false};
    void *nnopExecutor;
    static void *executorSpace = NULL;
    const char *opType = "PrefixInferAttentionScore";
    char inputDesc[] = {1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    char outputDesc[] = {1, 1};
    char attrDesc[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    NNOPBASE_ASSERT_NOTNULL_RETVAL(query);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(key);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(value);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(attentionOutOut);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(softmaxLseOut);

    if (!executorSpace) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseCreateExecutorSpace(&executorSpace));
    }
    nnopExecutor = NnopbaseGetExecutor(executorSpace, opType, inputDesc, sizeof(inputDesc) / sizeof(char), outputDesc,
                                       sizeof(outputDesc) / sizeof(char), attrDesc, sizeof(attrDesc) / sizeof(char));
    NNOPBASE_ASSERT_NOTNULL_RETVAL(nnopExecutor);
    NNOPBASE_ASSERT_NOTNULL_RETVAL(executor);
    *executor = reinterpret_cast<aclOpExecutor *>(nnopExecutor);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddTilingId(*executor, &tilingId));
    if (NnopbaseSetMatchArgsFlag != NULL) {
        NnopbaseSetMatchArgsFlag(*executor);
    }
#ifdef ACLNN_WITH_BINARY
    NnopbaseAddOpTypeId(*executor, PrefixInferAttentionScoreOpTypeId);
#endif
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, query, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddDynamicInput(*executor, key, 1));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddDynamicInput(*executor, value, 2));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, pseShiftOptional, 3));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, attenMaskOptional, 4));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, actualSeqLengthsOptional, 5));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, actualSeqLengthsKvOptional, 6));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, dequantScale1Optional, 7));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, quantScale1Optional, 8));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, dequantScale2Optional, 9));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, quantScale2Optional, 10));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, quantOffset2Optional, 11));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, antiquantScaleOptional, 12));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, antiquantOffsetOptional, 13));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, blockTableOptional, 14));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, queryPaddingSizeOptional, 15));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, kvPaddingSizeOptional, 16));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, keyAntiquantScaleOptional, 17));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, keyAntiquantOffsetOptional, 18));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, valueAntiquantScaleOptional, 19));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, valueAntiquantOffsetOptional, 20));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, keySharedPrefixOptional, 21));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, valueSharedPrefixOptional, 22));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, actualSharedPrefixLenOptional, 23));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, queryRopeOptional, 24));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, keyRopeOptional, 25));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, keyRopeAntiquantScaleOptional, 26));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, dequantScaleQueryOptional, 27));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, learnableSinkOptional, 28));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, qStartIdxOptional, 29));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddInput(*executor, kvStartIdxOptional, 30));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&numHeads), sizeof(int64_t), 0, kNnopbaseInt));
    float tmp1 = static_cast<float>(scale);
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&tmp1), sizeof(float), 1, kNnopbaseFloat));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&preTokens), sizeof(int64_t), 2, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&nextTokens), sizeof(int64_t), 3, kNnopbaseInt));
    if (inputLayoutOptional) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(inputLayoutOptional), strlen(inputLayoutOptional) + 1, 4, kNnopbaseString));
    } else {
        static char *inputLayoutOptionalDef = "TND";
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(inputLayoutOptionalDef), strlen(inputLayoutOptionalDef) + 1, 4, kNnopbaseString));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&numKeyValueHeads), sizeof(int64_t), 5, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&sparseMode), sizeof(int64_t), 6, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&innerPrecise), sizeof(int64_t), 7, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&blockSize), sizeof(int64_t), 8, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&antiquantMode), sizeof(int64_t), 9, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&softmaxLseFlag), sizeof(bool), 10, kNnopbaseBool));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&keyAntiquantMode), sizeof(int64_t), 11, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&valueAntiquantMode), sizeof(int64_t), 12, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&queryQuantMode), sizeof(int64_t), 13, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&pseType), sizeof(int64_t), 14, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddAttrWithDtype(*executor, static_cast<void*>(&outDtype), sizeof(int64_t), 15, kNnopbaseInt));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, attentionOutOut, 0));
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddOutput(*executor, softmaxLseOut, 1));
    if (NnopbaseMatchArgs != NULL) {
        if (NnopbaseMatchArgs(*executor, workspaceSize)) {
            NnopbaseReportApiInfo(timeStamp, dfxId);
            return ACLNN_SUCCESS;
        }
    }
    if (NnopbaseAddParamName != NULL) {
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "query", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "key", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 2, "value", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 3, "pseShiftOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 4, "attenMaskOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 5, "actualSeqLengthsOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 6, "actualSeqLengthsKvOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 7, "dequantScale1Optional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 8, "quantScale1Optional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 9, "dequantScale2Optional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 10, "quantScale2Optional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 11, "quantOffset2Optional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 12, "antiquantScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 13, "antiquantOffsetOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 14, "blockTableOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 15, "queryPaddingSizeOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 16, "kvPaddingSizeOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 17, "keyAntiquantScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 18, "keyAntiquantOffsetOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 19, "valueAntiquantScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 20, "valueAntiquantOffsetOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 21, "keySharedPrefixOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 22, "valueSharedPrefixOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 23, "actualSharedPrefixLenOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 24, "queryRopeOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 25, "keyRopeOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 26, "keyRopeAntiquantScaleOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 27, "dequantScaleQueryOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 28, "learnableSinkOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 29, "qStartIdxOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 30, "kvStartIdxOptional", true));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 0, "attentionOutOut", false));
        NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddParamName(*executor, 1, "softmaxLseOut", false));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseAddSupportList(*executor, &supportList, socSupportList, socSupportListLen));

    const aclTensorList *inUnContTensors = nullptr;
    NnopbaseGetUnContiguousTensors(*executor, &inUnContTensors);
    aclOpExecutor *aclInExecutor = nullptr;
    uint64_t inContWorkspaceSize = 0U;
    if (inUnContTensors != nullptr) {
        static AclnnContiguousGetWorkspaceSizeFunc aclnnContiguousGetWorkspaceSize = (AclnnContiguousGetWorkspaceSizeFunc)NnopbaseGetApiFunc("aclnnContiguousGetWorkspaceSize");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(aclnnContiguousGetWorkspaceSize);
        NNOPBASE_ASSERT_OK_RETVAL(aclnnContiguousGetWorkspaceSize(inUnContTensors, &inContWorkspaceSize, &aclInExecutor));
    }
    NnopbaseSetUnContExecutor(*executor, aclInExecutor, inContWorkspaceSize);

    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunForWorkspace(*executor, workspaceSize));
    *workspaceSize += inContWorkspaceSize;
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFullPrefixInferAttentionScore(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    aclOpExecutor *aclInExecutor = nullptr;
    uint64_t inContWorkspaceSize = 0U;
    NnopbaseGetUnContExecutor(executor, &aclInExecutor, &inContWorkspaceSize);
    if (workspaceSize < inContWorkspaceSize) {
        NnopbaseOpLogE(ACLNN_ERR_PARAM_INVALID, "input workspaceSize must be larger than contiguous size!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    workspaceSize -= inContWorkspaceSize;
    void *inWorkspace = (char *)workspace + workspaceSize;
    if (aclInExecutor != nullptr) {
        static AclnnFunc aclnnContiguous = (AclnnFunc)NnopbaseGetApiFunc("aclnnContiguous");
        NNOPBASE_ASSERT_NOTNULL_RETVAL(aclnnContiguous);
        NNOPBASE_ASSERT_OK_RETVAL(aclnnContiguous(inWorkspace, inContWorkspaceSize, aclInExecutor, stream));
    }
    NNOPBASE_ASSERT_OK_RETVAL(NnopbaseRunWithWorkspace(executor, stream, workspace, workspaceSize));
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
