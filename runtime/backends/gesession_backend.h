#pragma once

#include <string>
#include <vector>

#include "../acl_json.h"
#include "../bench.h"
#include "../io_spec.h"

namespace ge_runtime {

struct GeSessionOptions {
    int device = 0;
    int graph_run_mode = 1;
    std::string precision_mode = "force_fp16";
    std::string aicore_num;
    BenchOptions bench;
    GeProfilingConfig profiling;    // GE 在线路径经 GEInitialize 全局选项开启 (不走 acl.json)
};

bool RunGeSessionBackend(const Manifest &manifest, const IoSpec &spec,
                         const std::vector<TensorPlan> &inputs, const GeSessionOptions &opt,
                         std::vector<HostTensor> &outputs);

}  // namespace ge_runtime
