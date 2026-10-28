#include "bench.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

namespace ge_runtime {
namespace {

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
    p.p50 = times[NearestRankIndex(times.size(), 0.50)];
    p.p99 = times[NearestRankIndex(times.size(), 0.99)];
    return p;
}

void ReleasePool(const PoolResources &res, std::vector<bool> &created) {
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

double Mean(const std::vector<double> &v) {
    return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// 每个 distinct shape 取一个代表请求 (warmup 要覆盖全部 shape: GE 首次执行才特化)
std::vector<size_t> ShapeRepresentatives(const RequestPool &pool) {
    std::vector<size_t> reps;
    std::set<std::string> seen;
    for (size_t i = 0; i < pool.requests.size(); i++) {
        if (seen.insert(pool.requests[i].shapeKey).second) {
            reps.push_back(i);
        }
    }
    return reps;
}

}  // namespace

bool CapturePool(int instances, const RequestPool &pool, const PoolResources &res,
                 const std::function<bool(int, const Request &)> &collect) {
    if (instances < 1 || pool.Empty() || !res.setup || !res.setInputs || !res.execute || !collect) return false;
    std::vector<bool> created(static_cast<size_t>(instances), false);
    for (int t=0; t<instances; ++t) {
        if (!res.setup(t)) { ReleasePool(res,created); return false; }
        created[static_cast<size_t>(t)] = true;
    }
    std::atomic<int> errors{0};
    std::vector<std::thread> workers;
    for (int t=0; t<instances; ++t) workers.emplace_back([&,t]() {
        if (res.threadEnter) res.threadEnter(t);
        for (const auto &request:pool.requests) {
            PhaseTime phase;
            if (!res.setInputs(t,request,phase) || !res.execute(t) || !collect(t,request)) {
                errors++; break;
            }
        }
    });
    for (auto &thread:workers) thread.join();
    ReleasePool(res,created);
    return errors.load()==0;
}

bool BenchPool(int instances, size_t requests, int warmup, uint64_t seed,
               const RequestPool &pool, const PoolResources &res, PerfResult &out) {
    if (instances < 1) {
        instances = 1;
    }
    if (pool.Empty()) {
        fprintf(stderr, "[ERROR] 请求池为空\n");
        return false;
    }
    if (!res.setup || !res.setInputs || !res.execute) {
        fprintf(stderr, "[ERROR] BenchPool 需要 setup/setInputs/execute 钩子\n");
        return false;
    }

    out.distinctShapes = pool.distinctShapes;
    out.hbmBaseMb = res.hbmUsedMb ? res.hbmUsedMb() : -1.0;

    std::vector<bool> created(static_cast<size_t>(instances), false);
    for (int t = 0; t < instances; t++) {
        if (!res.setup(t)) {
            fprintf(stderr, "[ERROR] setup 实例 %d 失败\n", t);
            ReleasePool(res, created);
            return false;
        }
        created[static_cast<size_t>(t)] = true;
    }
    out.hbmPeakMb = res.hbmUsedMb ? res.hbmUsedMb() : -1.0;
    if (out.hbmBaseMb >= 0 && out.hbmPeakMb >= 0) {
        std::cout << "[INFO] " << instances << " 个实例占用 HBM "
                  << (out.hbmPeakMb - out.hbmBaseMb) << " MB (整机 " << out.hbmBaseMb
                  << " → " << out.hbmPeakMb << " MB)" << std::endl;
    }

    // 预先分配预热请求，保持原有逐实例 shape 覆盖和随机补齐的采样顺序。
    std::vector<size_t> reps = ShapeRepresentatives(pool);
    std::vector<std::vector<size_t>> warmSequences(static_cast<size_t>(instances), reps);
    Sampler warmSampler(pool.requests.size(), seed);
    size_t plannedWarmRuns = reps.size() * static_cast<size_t>(instances);
    while (plannedWarmRuns < static_cast<size_t>(std::max(0, warmup)) * instances) {
        size_t t = plannedWarmRuns % static_cast<size_t>(instances);
        warmSequences[t].push_back(warmSampler.Next());
        ++plannedWarmRuns;
    }
    std::vector<std::set<std::string>> seenShapes(static_cast<size_t>(instances));
    std::vector<double> warmSpecialize(static_cast<size_t>(instances), 0.0);
    std::vector<size_t> warmCounts(static_cast<size_t>(instances), 0);

    // Generate one measurement sequence before timing; warmup does not consume it.
    Sampler sampler(pool.requests.size(), seed);
    std::vector<size_t> sequence(requests);
    for (auto &index:sequence) index=sampler.Next();
    size_t perInstance = requests / static_cast<size_t>(instances);
    size_t remainder = requests % static_cast<size_t>(instances);
    std::vector<std::vector<RequestRecord>> per(static_cast<size_t>(instances));
    std::atomic<size_t> errors{0};

    std::mutex phaseMutex;
    std::condition_variable phaseCv;
    size_t ready = 0;
    bool startMeasure = false, warmFailed = false;
    auto warmBegin = Clock::now();
    Clock::time_point wallBegin;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(instances));
    try {
        for (int t = 0; t < instances; t++) {
            workers.emplace_back([&, t]() {
                std::vector<RequestRecord> &recs = per[static_cast<size_t>(t)];
                size_t count = perInstance + (static_cast<size_t>(t) < remainder ? 1 : 0);
                bool warmOk = true;
                try {
                    if (res.threadEnter) res.threadEnter(t);
                    recs.reserve(count);
                    // 在同一工作线程预热和测量，覆盖线程首次使用 ACL/runtime 的开销。
                    for (size_t idx : warmSequences[static_cast<size_t>(t)]) {
                        PhaseTime phase;
                        if (!res.setInputs(t, pool.requests[idx], phase)) {
                            warmOk = false;
                            break;
                        }
                        auto execBegin = Clock::now();
                        if (!res.execute(t)) {
                            warmOk = false;
                            break;
                        }
                        double execMs = ElapsedMs(execBegin, Clock::now());
                        if (seenShapes[static_cast<size_t>(t)].insert(pool.requests[idx].shapeKey).second) {
                            warmSpecialize[static_cast<size_t>(t)] += execMs;
                        }
                        ++warmCounts[static_cast<size_t>(t)];
                    }
                } catch (const std::exception &e) {
                    fprintf(stderr, "[ERROR] warmup instance=%d: %s\n", t, e.what());
                    warmOk = false;
                } catch (...) {
                    fprintf(stderr, "[ERROR] warmup instance=%d: unknown exception\n", t);
                    warmOk = false;
                }
                if (!warmOk) fprintf(stderr, "[ERROR] warmup 失败 (instance=%d)\n", t);
                {
                    std::unique_lock<std::mutex> lock(phaseMutex);
                    warmFailed = warmFailed || !warmOk;
                    ++ready;
                    phaseCv.notify_all();
                    phaseCv.wait(lock, [&]() { return startMeasure; });
                    if (warmFailed) return;
                }
                for (size_t i = 0; i < count; i++) {
                    size_t global = static_cast<size_t>(t) + i*static_cast<size_t>(instances);
                    const Request &req = pool.requests[sequence[global]];
                    RequestRecord rec;
                    rec.instance = t;
                    rec.req = global;
                    rec.inputSet = req.name;
                    rec.shapeKey = req.shapeKey;
                    rec.samples = req.samples;

                    PhaseTime phase;
                    auto t0 = Clock::now();
                    bool ok = res.setInputs(t, req, phase);
                    rec.h2dMs = phase.h2dMs;
                    rec.descMs = phase.descMs;
                    auto t1 = Clock::now();
                    if (ok) ok = res.execute(t);
                    auto t2 = Clock::now();
                    rec.execMs = ok ? ElapsedMs(t1, t2) : 0.0;
                    rec.e2eMs = ElapsedMs(t0, t2);
                    rec.error = ok ? 0 : 1;
                    recs.push_back(rec);
                    if (!ok) {
                        errors.fetch_add(1);
                        fprintf(stderr, "[ERROR] instance=%d 第 %zu 个请求失败 (input=%s)\n", t, i,
                                req.name.c_str());
                        break;
                    }
                }
            });
        }
    } catch (const std::exception &e) {
        // 启动部分线程失败时也释放等待者，再统一回收实例资源。
        {
            std::lock_guard<std::mutex> lock(phaseMutex);
            warmFailed = true;
            startMeasure = true;
        }
        phaseCv.notify_all();
        for (auto &worker : workers) worker.join();
        fprintf(stderr, "[ERROR] 创建 benchmark 工作线程失败: %s\n", e.what());
        ReleasePool(res, created);
        return false;
    }
    {
        std::unique_lock<std::mutex> lock(phaseMutex);
        phaseCv.wait(lock, [&]() { return ready == static_cast<size_t>(instances); });
        out.warmupRuns = std::accumulate(warmCounts.begin(), warmCounts.end(), size_t{0});
        out.warmupMs = ElapsedMs(warmBegin, Clock::now());
        if (!warmFailed) {
            std::cout << "[INFO] warmup " << out.warmupRuns << " 次 (覆盖 " << reps.size()
                      << " 种 shape × " << instances << " 实例), cost " << out.warmupMs
                      << " ms; measurement uses the same workers" << std::endl;
        }
        // 所有工作线程完成预热后统一起跑，线程创建和预热不计入吞吐窗口。
        wallBegin = Clock::now();
        startMeasure = true;
    }
    phaseCv.notify_all();
    for (auto &worker : workers) worker.join();
    if (warmFailed) {
        ReleasePool(res, created);
        return false;
    }
    out.wallMs = ElapsedMs(wallBegin, Clock::now());

    // ---- 汇总: 每实例 (含首次命中该 shape 的特化耗时) + 聚合 ----
    std::vector<double> allE2e;
    std::vector<double> allExec;
    std::vector<double> allTokens;
    for (int t = 0; t < instances; t++) {
        InstancePerf ip;
        ip.name = "instance_" + std::to_string(t);
        ip.specializeMs = warmSpecialize[static_cast<size_t>(t)];   // 特化代价 (warmup 段付的)
        std::vector<double> e2e, exec, h2d, desc;
        std::set<std::string> seen = seenShapes[static_cast<size_t>(t)];
        size_t shapesInMeasure = 0;
        for (auto &rec : per[static_cast<size_t>(t)]) {
            if (seen.insert(rec.shapeKey).second) {
                rec.firstHit = true;               // warmup 没覆盖到的 shape (正常应为 0)
                shapesInMeasure++;
            }
            out.records.push_back(rec);
            if (rec.error != 0) {
                ip.errors++;
                continue;
            }
            e2e.push_back(rec.e2eMs);
            exec.push_back(rec.execMs);
            h2d.push_back(rec.h2dMs);
            desc.push_back(rec.descMs);
            allTokens.push_back(static_cast<double>(rec.samples));
        }
        ip.distinctShapes = seen.size();
        if (shapesInMeasure > 0) {
            fprintf(stderr, "[WARN] %s 在测量段遇到 %zu 种 warmup 未覆盖的 shape (特化会污染延迟)\n",
                    ip.name.c_str(), shapesInMeasure);
        }
        ip.requests = e2e.size();
        if (!e2e.empty()) {
            Percentiles p = Summarize(e2e);
            ip.avgMs = p.avg; ip.minMs = p.min; ip.p50Ms = p.p50; ip.p99Ms = p.p99; ip.maxMs = p.max;
            Percentiles pe = Summarize(exec);
            ip.execAvgMs = pe.avg; ip.execP99Ms = pe.p99;
            ip.h2dAvgMs = Mean(h2d);
            ip.descAvgMs = Mean(desc);
            allE2e.insert(allE2e.end(), e2e.begin(), e2e.end());
            allExec.insert(allExec.end(), exec.begin(), exec.end());
        }
        ip.qps = out.wallMs > 0 ? static_cast<double>(ip.requests) / (out.wallMs / 1000.0) : 0.0;
        if (res.loadMs) {
            ip.loadMs = res.loadMs(t);
        }
        out.instances.push_back(ip);
    }

    out.requests = allE2e.size();
    out.samples = static_cast<size_t>(std::accumulate(allTokens.begin(),allTokens.end(),0.0));
    out.errors = errors.load();
    out.qps = out.wallMs > 0 ? static_cast<double>(out.requests) / (out.wallMs / 1000.0) : 0.0;
    if (!allE2e.empty()) {
        Percentiles p = Summarize(allE2e);
        out.avgMs = p.avg; out.minMs = p.min; out.p50Ms = p.p50; out.p99Ms = p.p99; out.maxMs = p.max;
    }
    if (!allExec.empty()) {
        Percentiles pe = Summarize(allExec);
        out.execAvgMs = pe.avg; out.execP99Ms = pe.p99;
    }
    if (!allTokens.empty()) {
        Percentiles pt = Summarize(allTokens);
        out.tokAvg = pt.avg; out.tokMin = pt.min; out.tokP50 = pt.p50; out.tokP99 = pt.p99;
        out.tokMax = pt.max;
    }

    std::sort(out.records.begin(), out.records.end(), [](const RequestRecord &a, const RequestRecord &b) { return a.req < b.req; });
    ReleasePool(res, created);
    return out.errors == 0;
}

void PrintPerfResult(const std::string &label, const PerfResult &p) {
    std::cout << "\n============================================================\n"
              << label << "\n============================================================\n"
              << std::fixed << std::setprecision(2)
              << "  instances: " << p.instances.size()
              << "   requests: " << p.requests << "   errors: " << p.errors << "\n"
              << "  wall:      " << std::setprecision(3) << p.wallMs << " ms\n"
              << "  QPS:       " << std::setprecision(2) << p.qps << " req/s\n"
              << "  e2e  ms:   avg " << std::setprecision(3) << p.avgMs << "  min " << p.minMs
              << "  p50 " << p.p50Ms << "  p99 " << p.p99Ms << "  max " << p.maxMs << "\n"
              << "  exec ms:   avg " << std::setprecision(3) << p.execAvgMs
              << "  p99 " << p.execP99Ms << "  (execute+sync, 不含 h2d/desc)\n"
              << "  samples:    avg " << std::setprecision(1) << p.tokAvg << "  min "
              << std::setprecision(0) << p.tokMin << "  p50 " << p.tokP50 << "  p99 " << p.tokP99
              << "  max " << p.tokMax << "  (每次调用的业务样本数)\n"
              << "  warmup:    " << p.warmupRuns << " 次 / " << std::setprecision(1) << p.warmupMs
              << " ms   distinct shapes: " << p.distinctShapes << "\n";
    if (p.hbmBaseMb >= 0 && p.hbmPeakMb >= 0) {
        std::cout << "  HBM:       " << std::setprecision(0) << p.hbmBaseMb << " → " << p.hbmPeakMb
                  << " MB (实例占 " << (p.hbmPeakMb - p.hbmBaseMb) << " MB)\n";
    }
    std::cout << "  ---- 每实例 ----\n" << std::setprecision(3);
    for (const auto &ip : p.instances) {
        std::cout << "  " << ip.name << ": req=" << ip.requests << " qps=" << std::setprecision(2)
                  << ip.qps << std::setprecision(3) << " e2e avg=" << ip.avgMs << " p99=" << ip.p99Ms
                  << " | exec avg=" << ip.execAvgMs << " p99=" << ip.execP99Ms
                  << " | h2d=" << ip.h2dAvgMs << " desc=" << ip.descAvgMs
                  << " | load=" << std::setprecision(1) << ip.loadMs << "ms"
                  << " shapes=" << ip.distinctShapes
                  << " 特化=" << ip.specializeMs << "ms err=" << ip.errors << "\n";
    }
    std::cout << "============================================================\n" << std::endl;
}

nlohmann::json PerfToJson(const PerfResult &p) {
    using Json = nlohmann::json;
    auto pct = [](const InstancePerf &ip) {
        return Json{{"avg", ip.avgMs}, {"min", ip.minMs}, {"p50", ip.p50Ms},
                    {"p99", ip.p99Ms}, {"max", ip.maxMs}};
    };
    Json instances = Json::array();
    for (const auto &ip : p.instances) {
        instances.push_back({{"name", ip.name},
                             {"requests", ip.requests},
                             {"errors", ip.errors},
                             {"qps", ip.qps},
                             {"e2e_ms", pct(ip)},
                             {"exec_ms", {{"avg", ip.execAvgMs}, {"p99", ip.execP99Ms}}},
                             {"h2d_ms", {{"avg", ip.h2dAvgMs}}},
                             {"desc_ms", {{"avg", ip.descAvgMs}}},
                             {"load_ms", ip.loadMs},
                             {"distinct_shapes", ip.distinctShapes},
                             {"specialize_ms", ip.specializeMs}});
    }
    // Replay rates count only completions inside the common measurement window.
    size_t windowSamples=0;
    for(const auto &record:p.records)
        if(!record.error && record.completedNs<=p.windowEndNs) windowSamples+=record.samples;
    // Public metrics consumed by core/results.py.
    return Json{{"schema", "ge-performance/1"},
                {"requests", p.requests},
                {"errors", p.errors},
                {"wall_ms", p.wallMs},
                {"qps", p.qps},
                {"samples_per_second", p.wallMs>0 ? (p.replay ? windowSamples : p.samples)/(p.wallMs/1000.0) : 0.0},
                {"latency_scope", p.replay ? "h2d_desc_execute_sync_d2h" : "h2d_desc_execute_sync"},
                {"arrival_model", "closed_loop"},
                {"slo_status", "not_defined"},
                {"run_id", p.runId},
                {"statistics_sufficient", p.replay ? Json(p.statisticsSufficient) : Json(nullptr)},
                {"window_start_ns", p.windowStartNs}, {"window_end_ns", p.windowEndNs},
                {"last_completed_ns", p.lastCompletedNs},
                {"window_successes", p.windowSuccesses}, {"cohort_successes", p.requests},
                {"in_flight_at_end", p.inFlightAtEnd}, {"drain_ms", p.drainMs},
                {"cohort_qps", p.wallMs > 0 ? p.requests / (p.wallMs/1000) : 0},
                {"d2h_ms", {{"avg", p.d2hAvgMs}}},
                {"e2e_ms", {{"avg", p.avgMs}, {"min", p.minMs}, {"p50", p.p50Ms},
                            {"p99", p.p99Ms}, {"max", p.maxMs}}},
                {"exec_ms", {{"avg", p.execAvgMs}, {"p99", p.execP99Ms}}},
                {"samples", {{"avg", p.tokAvg}, {"min", p.tokMin}, {"p50", p.tokP50},
                            {"p99", p.tokP99}, {"max", p.tokMax}}},
                {"warmup", {{"runs", p.warmupRuns}, {"ms", p.warmupMs}}},
                {"distinct_shapes", p.distinctShapes},
                {"hbm_mb", {{"base", p.hbmBaseMb}, {"after_setup", p.hbmPeakMb}}},
                {"instances", instances}};
}

bool WriteRequestsCsv(const std::string &path, const PerfResult &perf) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[ERROR] 无法写 %s\n", path.c_str());
        return false;
    }
    f << "instance,req,input_set,samples,shape_key,first_hit,h2d_ms,desc_ms,exec_ms,e2e_ms,error\n";
    f << std::fixed << std::setprecision(4);
    for (const auto &r : perf.records) {
        f << r.instance << ',' << r.req << ',' << r.inputSet << ',' << r.samples << ','
          << '"' << r.shapeKey << '"' << ',' << (r.firstHit ? 1 : 0) << ','
          << r.h2dMs << ',' << r.descMs << ',' << r.execMs << ',' << r.e2eMs << ','
          << r.error << '\n';
    }
    return true;
}

}  // namespace ge_runtime
