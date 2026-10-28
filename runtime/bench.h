#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "pool.h"

namespace ge_runtime {

// 两个后端共用的计时原语，不依赖 ACL。
using Clock = std::chrono::steady_clock;

inline double ElapsedMs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
}

struct ReplayOptions {
    bool enabled = false;
    double minSeconds = 0, maxSeconds = 0;
    size_t minRequests = 0, maxRequests = 2000000;
    int warmupPerCase = 10;
    bool captureOutputs = false;
    std::string runId;
    std::function<Clock::time_point()> now; // Controlled clock for CPU contract tests.
};

struct BenchOptions {
    int warmup = 0;
    int threads = 1;              // >1 → 吞吐模式 (闭环并发)
    int requests = 1;             // 测量段总调用次数
    ReplayOptions replay;
};

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
    int64_t samples = 0;
    double h2dMs = 0.0, descMs = 0.0, execMs = 0.0, e2eMs = 0.0, d2hMs = 0.0;
    int64_t startedNs = 0, completedNs = 0;
    size_t h2dBytes = 0, d2hBytes = 0;
    std::string phase = "measure", errorStage;
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
    std::vector<RequestRecord> warmupRecords;
    bool replay = false, statisticsSufficient = false;
    std::string runId;
    int64_t windowStartNs = 0, windowEndNs = 0, lastCompletedNs = 0;
    size_t windowSuccesses = 0, inFlightAtEnd = 0;
    double drainMs = 0, d2hAvgMs = 0;
    std::vector<InstancePerf> instances;
    size_t requests = 0, errors = 0, samples = 0;
    double wallMs = 0.0, qps = 0.0;
    double avgMs = 0.0, minMs = 0.0, p50Ms = 0.0, p99Ms = 0.0, maxMs = 0.0;
    double execAvgMs = 0.0, execP99Ms = 0.0;   // 聚合 exec (execute+sync, 不含 h2d/desc)
    double tokAvg = 0.0, tokMin = 0.0, tokP50 = 0.0, tokP99 = 0.0, tokMax = 0.0;  // 每次调用的业务样本数
    double hbmBaseMb = 0.0, hbmPeakMb = 0.0;   // 建实例前 / 建完实例后的整机 HBM 占用
    double warmupMs = 0.0;
    size_t warmupRuns = 0, distinctShapes = 0;
};

struct PoolResources {
    std::function<bool(int tid)> setup;                                  // 建第 tid 份实例资源
    std::function<bool(int tid, const Request &, PhaseTime &)> setInputs; // 每请求: H2D + desc
    std::function<bool(int tid, const Request &, RequestRecord &)> finishOutputs; // Full D2H, in timed boundary.
    std::function<bool(int tid, const Request &, const RequestRecord &)> captureRequest; // After completion, pressure only.
    std::function<bool(int tid)> execute;                                // execute + sync
    std::function<void(int tid)> release;
    std::function<void(int tid)> threadEnter;
    std::function<double(int tid)> loadMs;                               // 该实例加载/编译耗时
    std::function<double()> hbmUsedMb;                                   // 整机 HBM 占用 (不可用返回 -1)
};

// Load each instance once, execute every selected case, collect each output.
bool CapturePool(int instances, const RequestPool &pool, const PoolResources &res,
                 const std::function<bool(int, const Request &)> &collect);

// 闭环并发: instances 个 worker (1:1 绑实例), 共 requests 个请求, 每实例从共享池随机抽样
// 全局 seed 先生成请求序列，再分给实例；warmup 不消耗测量序列。
// 每实例在同一工作线程中预热和测量，全部预热成功后统一开始计时。
bool BenchPool(int instances, size_t requests, int warmup, uint64_t seed,
               const RequestPool &pool, const PoolResources &res, PerfResult &out);

size_t NearestRankIndex(size_t count, double quantile);
bool ReplayPool(int instances, uint64_t seed, const ReplayOptions &options,
                const RequestPool &pool, const PoolResources &res, PerfResult &out);
bool WriteRequestEvents(const std::string &path, const PerfResult &perf);

void PrintPerfResult(const std::string &label, const PerfResult &perf);
nlohmann::json PerfToJson(const PerfResult &perf);            // 调用方合并 meta 后落盘
bool WriteRequestsCsv(const std::string &path, const PerfResult &perf);

}  // namespace ge_runtime
