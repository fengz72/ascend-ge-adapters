#pragma once

#include <cstddef>
#include <functional>
#include <string>

namespace ge_runtime {

struct BenchOptions {
    int warmup = 0;
    int runs = 1;
};

struct BenchStats {
    size_t runs = 0;
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double p50_ms = 0.0;
    double p99_ms = 0.0;
    double max_ms = 0.0;
};

bool BenchRun(const BenchOptions &opt, const std::function<bool()> &step, BenchStats &stats);
void PrintBenchStats(const std::string &label, const BenchStats &stats);

}  // namespace ge_runtime
