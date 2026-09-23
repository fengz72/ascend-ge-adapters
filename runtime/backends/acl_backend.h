#pragma once

#include <string>
#include <vector>

#include "../bench.h"
#include "../io_spec.h"
#include "../pool.h"

namespace ge_runtime {

struct AclOptions {
    int device = 0;
    BenchOptions bench;             // 池模式用 bench.threads(=实例数)/requests/warmup
    size_t output_reserve = 256ull * 1024 * 1024;
    std::string aclConfigPath;      // 非空 → aclInit(configPath), 承载 dump/profiling 配置
    const RequestPool *pool = nullptr;   // 非空 → 变长负载 (请求池回放) 模式
    uint64_t seed = 0;
};

// perf 非空且为池模式时填入性能结果 (池模式不落盘输出 — 精度与性能分开跑)
bool RunAclBackend(const Manifest &manifest, const IoSpec &spec,
                   const std::vector<TensorPlan> &inputs, const AclOptions &opt,
                   std::vector<HostTensor> &outputs, PerfResult *perf = nullptr);

}  // namespace ge_runtime
