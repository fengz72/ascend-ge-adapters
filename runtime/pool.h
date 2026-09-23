#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "io_spec.h"

namespace ge_runtime {

// 一个请求 = 一套具体输入 (shape + host 数据), 对应 pool 目录下的一个 req_*/bundle.json
struct Request {
    std::string name;                        // req_000
    std::vector<TensorPlan> plans;           // 图序 (与 io_spec.inputs 同序)
    std::vector<std::vector<char>> data;     // host 字节, 与 plans 同序 (启动时一次读入)
    int64_t tokens = 0;                      // 报告用: 各输入元素数的最大值 (varlen 模型即 T)
    std::string shapeKey;                    // 各输入 shape 拼接, 用于统计 distinct shape
};

// 请求池: 所有实例**共享**, 各实例用 seed+instance_id 独立随机抽样
struct RequestPool {
    std::vector<Request> requests;
    std::vector<std::vector<int64_t>> maxShape;   // 逐输入: 各维最大 (预分配 device 缓冲用)
    std::vector<size_t> maxBytes;                 // 逐输入: 最大字节数
    size_t maxOutputBytes = 0;                    // 输出缓冲: pool 内 golden 的最大字节数 (0=未知)
    size_t distinctShapes = 0;

    bool Empty() const { return requests.empty(); }
};

// 扫描 dir 下的 req_*/bundle.json (也接受 dir/bundle.json 单套) → 请求池
RequestPool LoadPool(const IoSpec &spec, const std::string &dir);

// 每实例一个: 可复现的随机抽样 (seed = base_seed + instance_id)
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
