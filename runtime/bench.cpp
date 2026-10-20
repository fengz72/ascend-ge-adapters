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
    while (warmRuns < static_cast<size_t>(std::max(0, warmup)*instances)) {
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

    // Generate one measurement sequence before timing; instance count changes only its partition.
    Sampler sampler(pool.requests.size(), seed);
    std::vector<size_t> sequence(requests);
    for (auto &index:sequence) index=sampler.Next();
    size_t perInstance = requests / static_cast<size_t>(instances);
    size_t remainder = requests % static_cast<size_t>(instances);
    std::vector<std::vector<RequestRecord>> per(static_cast<size_t>(instances));
    std::atomic<size_t> errors{0};

    auto wallBegin = Clock::now();
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(instances));
    for (int t = 0; t < instances; t++) {
        workers.emplace_back([&res, &pool, &per, &errors, &sequence, t, perInstance, remainder, instances]() {
            if (res.threadEnter) {
                res.threadEnter(t);
            }
            std::vector<RequestRecord> &recs = per[static_cast<size_t>(t)];
            size_t count=perInstance+(static_cast<size_t>(t)<remainder ? 1:0);
            recs.reserve(count);
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
    // Public metrics consumed by core/results.py.
    return Json{{"schema", "ge-performance/1"},
                {"requests", p.requests},
                {"errors", p.errors},
                {"wall_ms", p.wallMs},
                {"qps", p.qps},
                {"samples_per_second", p.wallMs>0 ? p.samples/(p.wallMs/1000.0) : 0.0},
                {"latency_scope", "h2d_desc_execute_sync"},
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
