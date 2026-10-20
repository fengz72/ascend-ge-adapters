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
    std::string aclConfigPath;      // 非空 → aclInit(configPath), 承载 dump/profiling 配置
    const RequestPool *pool = nullptr;   // 必填；所有执行统一使用请求池
    std::function<bool(int, const Request &, const std::vector<HostTensor> &)> capture;
    uint64_t seed = 0;
};

// capture 非空时逐 case 收集输出，否则执行性能测试。
bool RunAclBackend(const ModelGraph &graph, const IoSpec &spec,
                   const AclOptions &opt, PerfResult *perf = nullptr);

}  // namespace ge_runtime
