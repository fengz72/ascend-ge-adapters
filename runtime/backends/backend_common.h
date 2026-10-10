#pragma once

// 两个后端 (acl_backend / gesession_backend) 共用的 ACL 侧小工具。
//
// 为什么是自由函数而不是 Backend 基类: 两后端的差异只在"用哪套执行 API" (aclmdl* vs
// ge::Session), 抽基类要为一处分支设计虚接口, 收益为负 (docs §9 的结论, 这里遵循同一标准)。
// 但**确实相同**的那部分不该各抄一份 —— 原来 ElapsedMs 有三份、HbmUsedMb 有两份逐字节相同的
// 副本, LoadInputs 的"读 .bin + 校验字节数 + 打日志"两边约 90% 重合。
//
// 分工:
//     bench.h              计时原语 (Clock / ElapsedMs) — 不依赖 ACL, bench.cpp 也要用
//     backend_common.h     本文件 — 需要 acl.h 的共享件 (HBM 查询 / host 输入载入)
//
// **不合并** dtype 枚举表 (ToAclDtype / ToGeDtype): 那是两套不同的枚举 (aclDataType vs
// ge::DataType), 合起来只会得到一张"字符串 → 两个枚举"的表, 读的人还得记住哪一列给谁用。
// 但两侧接受的**字符串集合**必须与 io_spec.cpp 的 DtypeSize 一致 —— 否则一个 dtype 能通过
// ValidatePlan 却在后端转换时失败 (或更糟: 被静默当成别的类型)。

#include <acl/acl.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "../io_spec.h"

namespace ge_runtime {

// 整机 HBM 占用 (MB); 取不到返回 -1
inline double HbmUsedMb() {
    size_t freeBytes = 0, totalBytes = 0;
    if (aclrtGetMemInfo(ACL_HBM_MEM, &freeBytes, &totalBytes) != ACL_SUCCESS) {
        return -1.0;
    }
    return static_cast<double>(totalBytes - freeBytes) / (1024.0 * 1024.0);
}

// 按 plan 读回输入 .bin → host 缓冲, 并校验字节数 == shape×dtype。
//
// dtype **是否被本后端支持**由调用方先判 (两后端的枚举表不同, 见文件头): 本函数只负责
// "读得到、字节数对得上", 这两件事与后端无关。
inline bool LoadHostInputs(const std::vector<TensorPlan> &plans,
                           std::vector<std::vector<char>> &out) {
    out.clear();
    out.reserve(plans.size());
    for (size_t i = 0; i < plans.size(); i++) {
        const TensorPlan &p = plans[i];
        std::vector<char> host;
        if (!ReadBinFile(p.file, host)) {
            fprintf(stderr, "[ERROR] 读不到输入文件: %s\n", p.file.c_str());
            return false;
        }
        if (host.size() != p.Bytes()) {
            fprintf(stderr, "[ERROR] 输入 '%s' 文件字节数 %zu != %zu\n",
                    p.logical.c_str(), host.size(), p.Bytes());
            return false;
        }
        std::cout << "[INFO] Input[" << i << "] '" << p.logical << "' node=" << p.node
                  << " shape=" << ShapeToString(p.shape) << " dtype=" << p.dtype
                  << " bytes=" << p.Bytes() << std::endl;
        out.push_back(std::move(host));
    }
    return true;
}

}  // namespace ge_runtime
