#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pool.h"

namespace ge_runtime {

struct BenchOptions {
    int warmup = 0;
    int runs = 1;
    int threads = 1;              // >1 → 吞吐模式 (闭环并发)
    int requests = 0;             // 吞吐模式总请求数 (0 → 每线程 runs 个)
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

// ---------------------------------------------------------------- 变长负载 (请求池回放)

struct PhaseTime {
    double h2dMs = 0.0;
    double descMs = 0.0;      // ACL: 重设 tensor desc; GE: 重建 gert::Tensor
};

struct RequestRecord {
    int instance = 0;
    size_t req = 0;
    std::string inputSet;
    std::string shapeKey;
    int64_t tokens = 0;
    double h2dMs = 0.0, descMs = 0.0, execMs = 0.0, e2eMs = 0.0;
    bool firstHit = false;    // 该实例首次遇到这个 shape (GE 会在此付特化代价)
    int error = 0;
};

struct InstancePerf {
    std::string name;
    size_t requests = 0, errors = 0, distinctShapes = 0;
    double qps = 0.0;
    double avgMs = 0.0, minMs = 0.0, p50Ms = 0.0, p99Ms = 0.0, maxMs = 0.0;
    double execAvgMs = 0.0, execP99Ms = 0.0;
    double h2dAvgMs = 0.0, descAvgMs = 0.0;
    double loadMs = 0.0, specializeMs = 0.0;
};

struct PerfResult {
    std::vector<RequestRecord> records;
    std::vector<InstancePerf> instances;
    size_t requests = 0, errors = 0;
    double wallMs = 0.0, qps = 0.0;
    double avgMs = 0.0, minMs = 0.0, p50Ms = 0.0, p99Ms = 0.0, maxMs = 0.0;
    double execAvgMs = 0.0, execP99Ms = 0.0;   // 聚合 exec (execute+sync, 不含 h2d/desc)
    double tokAvg = 0.0, tokMin = 0.0, tokP50 = 0.0, tokP99 = 0.0, tokMax = 0.0;  // 每请求 packed token 数 T 的分布
    double hbmBaseMb = 0.0, hbmPeakMb = 0.0;   // 建实例前 / 建完实例后的整机 HBM 占用
    double warmupMs = 0.0;
    size_t warmupRuns = 0, distinctShapes = 0;
};

struct PoolResources {
    std::function<bool(int tid)> setup;                                  // 建第 tid 份实例资源
    std::function<bool(int tid, const Request &, PhaseTime &)> setInputs; // 每请求: H2D + desc
    std::function<bool(int tid)> execute;                                // execute + sync
    std::function<void(int tid)> release;
    std::function<void(int tid)> threadEnter;
    std::function<double(int tid)> loadMs;                               // 该实例加载/编译耗时
    std::function<double()> hbmUsedMb;                                   // 整机 HBM 占用 (不可用返回 -1)
};

// 闭环并发: instances 个 worker (1:1 绑实例), 共 requests 个请求, 每实例从共享池随机抽样
// (seed + tid, 可复现)。warmup 先覆盖 每实例 × 每个 distinct shape (GE 首次执行要特化)。
bool BenchPool(int instances, size_t requests, int warmup, uint64_t seed,
               const RequestPool &pool, const PoolResources &res, PerfResult &out);

void PrintPerfResult(const std::string &label, const PerfResult &perf);
nlohmann::json PerfToJson(const PerfResult &perf);            // 调用方合并 meta 后落盘
bool WriteRequestsCsv(const std::string &path, const PerfResult &perf);

// 吞吐: opt.threads 个 worker 闭环跑 (1 worker ↔ 1 份资源, 无锁); setup→warmup→并发→release。
// **不做档位扫描** — 扫描是"同一件事跑多遍", 归脚本 (tools/sweep.py 逐档起进程),
// C++ 只负责测准一档。
bool BenchThroughput(const BenchOptions &opt, const ThreadResources &res, ThroughputStats &stats);

bool IsThroughputMode(const BenchOptions &opt);

void PrintBenchStats(const std::string &label, const BenchStats &stats);
void PrintThroughputStats(const std::string &label, const ThroughputStats &stats);

}  // namespace ge_runtime
