#include "bench.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <vector>

namespace ge_runtime {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
}

}  // namespace

bool BenchRun(const BenchOptions &opt, const std::function<bool()> &step, BenchStats &stats) {
    for (int i = 0; i < opt.warmup; i++) {
        if (!step()) {
            fprintf(stderr, "[ERROR] warmup run %d failed\n", i);
            return false;
        }
    }

    int runs = std::max(1, opt.runs);
    std::vector<double> times;
    times.reserve(static_cast<size_t>(runs));
    for (int i = 0; i < runs; i++) {
        auto begin = Clock::now();
        if (!step()) {
            fprintf(stderr, "[ERROR] bench run %d failed\n", i);
            return false;
        }
        times.push_back(ElapsedMs(begin, Clock::now()));
    }

    std::sort(times.begin(), times.end());
    stats.runs = times.size();
    stats.avg_ms = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    stats.min_ms = times.front();
    stats.max_ms = times.back();
    stats.p50_ms = times[times.size() / 2];
    size_t p99 = static_cast<size_t>(static_cast<double>(times.size()) * 0.99);
    stats.p99_ms = times[std::min(p99, times.size() - 1)];
    return true;
}

void PrintBenchStats(const std::string &label, const BenchStats &stats) {
    std::cout << "\n============================================================\n"
              << label << " (" << stats.runs << " runs)\n"
              << "============================================================\n"
              << "  avg: " << stats.avg_ms << " ms\n"
              << "  min: " << stats.min_ms << " ms\n"
              << "  p50: " << stats.p50_ms << " ms\n"
              << "  p99: " << stats.p99_ms << " ms\n"
              << "  max: " << stats.max_ms << " ms\n"
              << "============================================================\n"
              << std::endl;
}

}  // namespace ge_runtime
