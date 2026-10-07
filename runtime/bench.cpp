#include "bench.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
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
    return opt.threads > 1 || opt.requests > 0;
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

bool BenchThroughput(const BenchOptions &opt, const ThreadResources &res, ThroughputStats &stats) {
    if (!res.setup || !res.step) {
        fprintf(stderr, "[ERROR] BenchThroughput 需要 setup/step 钩子\n");
        return false;
    }
    const int threads = std::max(1, opt.threads);
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
    return stats.errors == 0;
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

// ---------------------------------------------------------------- 变长负载 (请求池回放)

namespace ge_runtime {
namespace {

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

    // ---- warmup: 每实例 × 每个 distinct shape, 不足 warmup 再随机补 ----
    std::vector<size_t> reps = ShapeRepresentatives(pool);
    auto warmBegin = Clock::now();
    PhaseTime ph;
    size_t warmRuns = 0;
    // 特化代价付在 warmup: 逐实例记录"首次命中某 shape"的 execute 耗时, 并把已见过的 shape
    // 带进测量段 (否则测量段会把 warmup 已经特化过的 shape 又算一次特化)
    std::vector<std::set<std::string>> seenShapes(static_cast<size_t>(instances));
    std::vector<double> warmSpecialize(static_cast<size_t>(instances), 0.0);
    for (int t = 0; t < instances; t++) {
        for (size_t idx : reps) {
            if (!res.setInputs(t, pool.requests[idx], ph)) {
                fprintf(stderr, "[ERROR] warmup setInputs 失败 (instance=%d, req=%s)\n", t,
                        pool.requests[idx].name.c_str());
                ReleasePool(res, created);
                return false;
            }
            auto execBegin = Clock::now();
            if (!res.execute(t)) {
                fprintf(stderr, "[ERROR] warmup execute 失败 (instance=%d, req=%s)\n", t,
                        pool.requests[idx].name.c_str());
                ReleasePool(res, created);
                return false;
            }
            double execMs = ElapsedMs(execBegin, Clock::now());
            if (seenShapes[static_cast<size_t>(t)].insert(pool.requests[idx].shapeKey).second) {
                warmSpecialize[static_cast<size_t>(t)] += execMs;
            }
            warmRuns++;
        }
    }
    Sampler warmSampler(pool.requests.size(), seed);
    while (warmRuns < static_cast<size_t>(std::max(0, warmup))) {
        int t = static_cast<int>(warmRuns % static_cast<size_t>(instances));
        if (!res.setInputs(t, pool.requests[warmSampler.Next()], ph) || !res.execute(t)) {
            fprintf(stderr, "[ERROR] warmup 失败 (instance=%d)\n", t);
            ReleasePool(res, created);
            return false;
        }
        warmRuns++;
    }
    out.warmupRuns = warmRuns;
    out.warmupMs = ElapsedMs(warmBegin, Clock::now());
    std::cout << "[INFO] warmup " << warmRuns << " 次 (覆盖 " << reps.size() << " 种 shape × "
              << instances << " 实例), cost " << out.warmupMs << " ms" << std::endl;

    // ---- 测量: 每 worker 闭环, 从共享池随机抽样 (seed + tid, 可复现) ----
    size_t perInstance = std::max<size_t>(1, requests / static_cast<size_t>(instances));
    std::vector<std::vector<RequestRecord>> per(static_cast<size_t>(instances));
    std::atomic<size_t> errors{0};

    auto wallBegin = Clock::now();
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(instances));
    for (int t = 0; t < instances; t++) {
        workers.emplace_back([&res, &pool, &per, &errors, t, perInstance, seed]() {
            if (res.threadEnter) {
                res.threadEnter(t);
            }
            Sampler sampler(pool.requests.size(), seed + static_cast<uint64_t>(t));
            std::vector<RequestRecord> &recs = per[static_cast<size_t>(t)];
            recs.reserve(perInstance);
            for (size_t i = 0; i < perInstance; i++) {
                const Request &req = pool.requests[sampler.Next()];
                RequestRecord rec;
                rec.instance = t;
                rec.req = i;
                rec.inputSet = req.name;
                rec.shapeKey = req.shapeKey;
                rec.tokens = req.tokens;

                PhaseTime phase;
                auto t0 = Clock::now();
                bool ok = res.setInputs(t, req, phase);
                rec.h2dMs = phase.h2dMs;
                rec.descMs = phase.descMs;
                auto t1 = Clock::now();
                if (ok) {
                    ok = res.execute(t);
                }
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
    for (auto &w : workers) {
        w.join();
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
            allTokens.push_back(static_cast<double>(rec.tokens));
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
              << "  tokens:    avg " << std::setprecision(1) << p.tokAvg << "  min "
              << std::setprecision(0) << p.tokMin << "  p50 " << p.tokP50 << "  p99 " << p.tokP99
              << "  max " << p.tokMax << "  (每请求 packed T)\n"
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
    return Json{{"schema", "ge-bench/1"},
                {"requests", p.requests},
                {"errors", p.errors},
                {"wall_ms", p.wallMs},
                {"qps", p.qps},
                {"e2e_ms", {{"avg", p.avgMs}, {"min", p.minMs}, {"p50", p.p50Ms},
                            {"p99", p.p99Ms}, {"max", p.maxMs}}},
                {"exec_ms", {{"avg", p.execAvgMs}, {"p99", p.execP99Ms}}},
                {"tokens", {{"avg", p.tokAvg}, {"min", p.tokMin}, {"p50", p.tokP50},
                            {"p99", p.tokP99}, {"max", p.tokMax}}},
                {"warmup", {{"runs", p.warmupRuns}, {"ms", p.warmupMs}}},
                {"distinct_shapes", p.distinctShapes},
                {"hbm_mb", {{"base", p.hbmBaseMb}, {"peak", p.hbmPeakMb}}},
                {"instances", instances}};
}

bool WriteRequestsCsv(const std::string &path, const PerfResult &perf) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[ERROR] 无法写 %s\n", path.c_str());
        return false;
    }
    f << "instance,req,input_set,tokens,shape_key,first_hit,h2d_ms,desc_ms,exec_ms,e2e_ms,error\n";
    f << std::fixed << std::setprecision(4);
    for (const auto &r : perf.records) {
        f << r.instance << ',' << r.req << ',' << r.inputSet << ',' << r.tokens << ','
          << '"' << r.shapeKey << '"' << ',' << (r.firstHit ? 1 : 0) << ','
          << r.h2dMs << ',' << r.descMs << ',' << r.execMs << ',' << r.e2eMs << ','
          << r.error << '\n';
    }
    return true;
}

}  // namespace ge_runtime
