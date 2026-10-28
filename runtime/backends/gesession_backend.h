#pragma once

#include <string>
#include <vector>

#include "../acl_json.h"
#include "../bench.h"
#include "../io_spec.h"
#include "../pool.h"

namespace ge_runtime {

struct GeSessionOptions {
    int device = 0;
    int graph_run_mode = 0;          // GE PREDICTION; inference is the project default.
    std::string precision_mode = "force_fp16";
    std::string aicore_num;
    BenchOptions bench;             // 池模式用 bench.threads(=实例数)/requests/warmup
    GeProfilingConfig profiling;    // GE 在线路径经 GEInitialize 全局选项开启 (不走 acl.json)
    const RequestPool *pool = nullptr;   // 必填；所有执行统一使用请求池
    std::function<bool(int, const Request &, const std::vector<HostTensor> &)> capture;
    std::function<bool(int, const Request &, const RequestRecord &, const std::vector<HostTensor> &)> replayCapture;
    uint64_t seed = 0;
};

// capture 非空时逐 case 收集输出，否则执行性能测试。
bool RunGeSessionBackend(const ModelGraph &graph, const IoSpec &spec,
                   const GeSessionOptions &opt, PerfResult *perf = nullptr);

}  // namespace ge_runtime
