#pragma once

#include <vector>

#include "../bench.h"
#include "../io_spec.h"

namespace ge_runtime {

struct AclOptions {
    int device = 0;
    BenchOptions bench;
    size_t output_reserve = 256ull * 1024 * 1024;
};

bool RunAclBackend(const Manifest &manifest, const IoSpec &spec,
                   const std::vector<TensorPlan> &inputs, const AclOptions &opt,
                   std::vector<HostTensor> &outputs);

}  // namespace ge_runtime
