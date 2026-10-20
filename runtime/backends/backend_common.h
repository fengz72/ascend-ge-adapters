#pragma once
#include <acl/acl.h>
namespace ge_runtime {
// 整卡 HBM 占用 (MB)，查询失败返回 -1。
inline double HbmUsedMb() {
    size_t freeBytes=0,totalBytes=0;
    if(aclrtGetMemInfo(ACL_HBM_MEM,&freeBytes,&totalBytes)!=ACL_SUCCESS) return -1.0;
    return static_cast<double>(totalBytes-freeBytes)/(1024.0*1024.0);
}
} // namespace ge_runtime
