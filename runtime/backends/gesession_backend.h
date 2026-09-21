#pragma once

#include <string>
#include <vector>

#include "../bench.h"
#include "../io_spec.h"

namespace ge_runtime {

struct GeSessionOptions {
    int device = 0;
    int graph_run_mode = 1;
    std::string precision_mode = "force_fp16";
    std::string aicore_num;
    BenchOptions bench;
};

bool RunGeSessionBackend(const Manifest &manifest, const IoSpec &spec,
                         const std::vector<TensorPlan> &inputs, const GeSessionOptions &opt,
                         std::vector<HostTensor> &outputs);

}  // namespace ge_runtime
