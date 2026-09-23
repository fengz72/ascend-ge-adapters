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
    int graph_run_mode = 1;
    std::string precision_mode = "force_fp16";
    std::string aicore_num;
    BenchOptions bench;             // 池模式用 bench.threads(=实例数)/requests/warmup
    GeProfilingConfig profiling;    // GE 在线路径经 GEInitialize 全局选项开启 (不走 acl.json)
    const RequestPool *pool = nullptr;   // 非空 → 变长负载 (请求池回放) 模式
    uint64_t seed = 0;
};

// perf 非空且为池模式时填入性能结果 (池模式不落盘输出 — 精度与性能分开跑)
bool RunGeSessionBackend(const Manifest &manifest, const IoSpec &spec,
                         const std::vector<TensorPlan> &inputs, const GeSessionOptions &opt,
                         std::vector<HostTensor> &outputs, PerfResult *perf = nullptr);

}  // namespace ge_runtime
