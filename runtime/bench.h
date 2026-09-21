#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace ge_runtime {

struct BenchOptions {
    int warmup = 0;
    int runs = 1;
    int threads = 1;              // >1 → 吞吐模式 (闭环并发)
    int requests = 0;             // 吞吐模式总请求数 (0 → 每线程 runs 个)
    std::vector<int> sweep;       // 非空 → 依次跑各线程档, 每档独立建/销线程资源
};

struct BenchStats {
    size_t runs = 0;
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double p50_ms = 0.0;
    double p99_ms = 0.0;
    double max_ms = 0.0;
};

struct ThroughputStats {
    int threads = 0;
    size_t requests = 0;
    size_t errors = 0;
    double wall_ms = 0.0;
    double qps = 0.0;
    double avg_ms = 0.0;          // 单请求 e2e (含并发排队)
    double min_ms = 0.0;
    double p50_ms = 0.0;
    double p99_ms = 0.0;
    double max_ms = 0.0;
};

// 线程资源钩子 — bench 只管编排/计时, 资源生命周期归后端 (ACL dataset / GE stream 等)
struct ThreadResources {
    std::function<bool(int tid)> setup;         // 主线程串行调用: 建第 tid 份资源
    std::function<bool(int tid)> step;          // 工作线程调用: 执行一次请求
    std::function<void(int tid)> release;       // 主线程串行调用: 销毁第 tid 份资源
    std::function<void(int tid)> threadEnter;   // 工作线程启动时 (如 aclrtSetCurrentContext)
    std::function<void(int tid)> threadExit;    // 工作线程退出前
};

bool BenchRun(const BenchOptions &opt, const std::function<bool()> &step, BenchStats &stats);

// 吞吐: 按 opt.sweep (或 {opt.threads}) 逐档跑; 每档 setup→warmup→并发 requests→release
bool BenchSweep(const BenchOptions &opt, const ThreadResources &res,
                std::vector<ThroughputStats> &all);

bool IsThroughputMode(const BenchOptions &opt);

void PrintBenchStats(const std::string &label, const BenchStats &stats);
void PrintThroughputStats(const std::string &label, const ThroughputStats &stats);

}  // namespace ge_runtime
