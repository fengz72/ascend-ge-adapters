#include "bench.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

namespace ge_runtime {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
}

struct Percentiles {
    double avg = 0, min = 0, p50 = 0, p99 = 0, max = 0;
};

Percentiles Summarize(std::vector<double> times) {
    Percentiles p;
    if (times.empty()) {
        return p;
    }
    std::sort(times.begin(), times.end());
    p.avg = std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size());
    p.min = times.front();
    p.max = times.back();
    p.p50 = times[times.size() / 2];
    size_t idx = static_cast<size_t>(static_cast<double>(times.size()) * 0.99);
    p.p99 = times[std::min(idx, times.size() - 1)];
    return p;
}

void ReleaseAll(const ThreadResources &res, std::vector<bool> &created) {
    if (!res.release) {
        return;
    }
    for (size_t t = 0; t < created.size(); t++) {
        if (created[t]) {
            res.release(static_cast<int>(t));
            created[t] = false;
        }
    }
}

}  // namespace

bool IsThroughputMode(const BenchOptions &opt) {
    return opt.threads > 1 || opt.requests > 0 || !opt.sweep.empty();
}

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

    Percentiles p = Summarize(std::move(times));
    stats.runs = static_cast<size_t>(runs);
    stats.avg_ms = p.avg;
    stats.min_ms = p.min;
    stats.p50_ms = p.p50;
    stats.p99_ms = p.p99;
    stats.max_ms = p.max;
    return true;
}

bool BenchSweep(const BenchOptions &opt, const ThreadResources &res,
                std::vector<ThroughputStats> &all) {
    if (!res.setup || !res.step) {
        fprintf(stderr, "[ERROR] BenchSweep 需要 setup/step 钩子\n");
        return false;
    }
    std::vector<int> levels = opt.sweep.empty()
                                  ? std::vector<int>{std::max(1, opt.threads)}
                                  : opt.sweep;

    for (int threads : levels) {
        if (threads < 1) {
            continue;
        }
        ThroughputStats stats;
        stats.threads = threads;
        std::vector<bool> created(static_cast<size_t>(threads), false);

        for (int t = 0; t < threads; t++) {
            if (!res.setup(t)) {
                fprintf(stderr, "[ERROR] setup 线程资源失败 (tid=%d, threads=%d)\n", t, threads);
                ReleaseAll(res, created);
                return false;
            }
            created[static_cast<size_t>(t)] = true;
        }

        // warmup 轮转覆盖每份资源: GE 每个 stream 首次执行都要做 shape 特化
        int warm = std::max(opt.warmup, threads);
        bool ok = true;
        for (int i = 0; i < warm && ok; i++) {
            ok = res.step(i % threads);
        }
        if (!ok) {
            fprintf(stderr, "[ERROR] warmup 执行失败 (threads=%d)\n", threads);
            ReleaseAll(res, created);
            return false;
        }

        int perThread = opt.requests > 0 ? std::max(1, opt.requests / threads) : std::max(1, opt.runs);
        std::vector<std::vector<double>> perThreadTimes(static_cast<size_t>(threads));
        std::atomic<size_t> errors{0};

        auto wallBegin = Clock::now();
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(threads));
        for (int t = 0; t < threads; t++) {
            workers.emplace_back([&res, &perThreadTimes, &errors, t, perThread]() {
                if (res.threadEnter) {
                    res.threadEnter(t);
                }
                std::vector<double> &times = perThreadTimes[static_cast<size_t>(t)];
                times.reserve(static_cast<size_t>(perThread));
                for (int i = 0; i < perThread; i++) {
                    auto begin = Clock::now();
                    if (!res.step(t)) {
                        errors.fetch_add(1);
                        fprintf(stderr, "[ERROR] tid=%d 第 %d 个请求执行失败\n", t, i);
                        break;
                    }
                    times.push_back(ElapsedMs(begin, Clock::now()));
                }
                if (res.threadExit) {
                    res.threadExit(t);
                }
            });
        }
        for (auto &w : workers) {
            w.join();
        }
        double wallMs = ElapsedMs(wallBegin, Clock::now());

        std::vector<double> times;
        for (const auto &v : perThreadTimes) {
            times.insert(times.end(), v.begin(), v.end());
        }
        Percentiles p = Summarize(times);
        stats.requests = times.size();
        stats.errors = errors.load();
        stats.wall_ms = wallMs;
        stats.qps = wallMs > 0 ? static_cast<double>(stats.requests) / (wallMs / 1000.0) : 0.0;
        stats.avg_ms = p.avg;
        stats.min_ms = p.min;
        stats.p50_ms = p.p50;
        stats.p99_ms = p.p99;
        stats.max_ms = p.max;

        ReleaseAll(res, created);
        all.push_back(stats);
        if (stats.errors > 0) {
            return false;
        }
    }
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

void PrintThroughputStats(const std::string &label, const ThroughputStats &s) {
    std::cout << "\n============================================================\n"
              << label << " (threads=" << s.threads << ", requests=" << s.requests << ")\n"
              << "============================================================\n"
              << std::fixed << std::setprecision(3)
              << "  wall:    " << s.wall_ms << " ms\n"
              << "  QPS:     " << std::setprecision(2) << s.qps << " req/s\n"
              << std::setprecision(3)
              << "  e2e avg: " << s.avg_ms << " ms\n"
              << "  e2e min: " << s.min_ms << " ms\n"
              << "  e2e p50: " << s.p50_ms << " ms\n"
              << "  e2e p99: " << s.p99_ms << " ms\n"
              << "  e2e max: " << s.max_ms << " ms\n"
              << "  errors:  " << s.errors << "\n"
              << "============================================================\n"
              << std::endl;
}

}  // namespace ge_runtime
