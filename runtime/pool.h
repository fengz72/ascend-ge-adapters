#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "io_spec.h"

namespace ge_runtime {

// One request contains its concrete shapes and host input bytes.
struct Request {
    std::string name;                        // req_000
    std::vector<TensorPlan> plans;           // 图序 (与 io_spec.inputs 同序)
    std::vector<std::vector<char>> data;     // host 字节, 与 plans 同序 (启动时一次读入)
    int64_t samples = 0;                      // business samples declared by the input case
    std::string shapeKey;                    // 各输入 shape 拼接, 用于统计 distinct shape
};

// All instances share the input pool and the seeded global measurement sequence.
struct RequestPool {
    std::vector<Request> requests;
    std::vector<size_t> maxBytes;                 // 逐输入: 最大字节数 (预分配 device 缓冲用)
    std::vector<size_t> maxOutputBytes;          // maximum declared output capacity per output
    size_t distinctShapes = 0;

    bool Empty() const { return requests.empty(); }
};

// One packed binary input file; offsets and shapes arrive in memory over stdin.
RequestPool LoadPool(const IoSpec &spec, const nlohmann::json &inputs);

// Seeded sampling with replacement; warmup and measurement use separate samplers.
class Sampler {
public:
    Sampler(size_t poolSize, uint64_t seed)
        : poolSize_(poolSize), rng_(seed) {}

    size_t Next() {
        if (poolSize_ == 0) {
            return 0;
        }
        if (poolSize_ == 1) {
            return 0;
        }
        return static_cast<size_t>(rng_() % poolSize_);
    }

private:
    size_t poolSize_;
    std::mt19937_64 rng_;
};

}  // namespace ge_runtime
