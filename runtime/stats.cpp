// 抽样计划与计时聚合实现 (见 stats.h)。纯 CPU, 不依赖 GE / ACL。
#include "stats.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ge_execute {
namespace {

std::vector<size_t> DistinctRepresentatives(const std::vector<std::string> &shapeKeys) {
    std::vector<size_t> reps;
    std::vector<std::string> seen;
    for (size_t i = 0; i < shapeKeys.size(); ++i) {
        if (std::find(seen.begin(), seen.end(), shapeKeys[i]) == seen.end()) {
            seen.push_back(shapeKeys[i]);
            reps.push_back(i);
        }
    }
    return reps;
}

}  // namespace

size_t NearestRankIndex(size_t count, double quantile) {
    if (count == 0) {
        throw std::invalid_argument("NearestRankIndex: count must be > 0");
    }
    if (!(quantile > 0.0) || quantile > 1.0) {
        throw std::invalid_argument("NearestRankIndex: quantile must be in (0, 1]");
    }
    // ceil(q*count) 在 q=0.99,count=100 时因浮点误差可能变成 100;
    // 减一个极小量再向上取整, 保证 exact hundredth 不跳到下一档。
    double position = std::ceil(quantile * static_cast<double>(count) - 1e-9);
    if (position < 1.0) {
        position = 1.0;
    }
    const size_t rank = static_cast<size_t>(position);
    return std::min(count - 1, rank - 1);
}

double Percentile(std::vector<double> values, double quantile) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    return values[NearestRankIndex(values.size(), quantile)];
}

std::vector<size_t> DrawSequence(size_t poolSize, size_t count, uint64_t seed) {
    if (count == 0) {
        return {};
    }
    if (poolSize <= 1) {
        return std::vector<size_t>(count, 0);
    }
    std::mt19937_64 rng(seed);
    std::vector<size_t> sequence;
    sequence.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        sequence.push_back(static_cast<size_t>(rng() % poolSize));
    }
    return sequence;
}

std::vector<std::vector<size_t>> MeasurePlan(size_t poolSize, size_t requests,
                                             size_t instances, uint64_t seed) {
    if (instances < 1) {
        instances = 1;
    }
    if (requests == 0) {
        return {};
    }
    const std::vector<size_t> global = DrawSequence(poolSize, requests, seed);
    std::vector<std::vector<size_t>> plan(instances);
    for (size_t request = 0; request < global.size(); ++request) {
        plan[request % instances].push_back(global[request]);
    }
    return plan;
}

std::vector<std::vector<size_t>> WarmupPlan(const std::vector<std::string> &shapeKeys,
                                            size_t warmupPerInstance, size_t instances,
                                            uint64_t seed) {
    if (shapeKeys.empty()) {
        return {};
    }
    if (instances < 1) {
        instances = 1;
    }
    const std::vector<size_t> reps = DistinctRepresentatives(shapeKeys);
    const size_t target = std::max(warmupPerInstance, reps.size());
    const size_t extra = target - reps.size();
    const std::vector<size_t> fill = DrawSequence(shapeKeys.size(), extra, seed);

    std::vector<std::vector<size_t>> plan(instances);
    for (size_t instance = 0; instance < instances; ++instance) {
        plan[instance] = reps;
        plan[instance].insert(plan[instance].end(), fill.begin(), fill.end());
    }
    return plan;
}

void Summarize(RunStats &stats) {
    std::vector<double> e2e;
    std::vector<double> exec;
    e2e.reserve(stats.records.size());
    exec.reserve(stats.records.size());
    double d2hSum = 0.0;

    size_t instanceCount = stats.perInstance.size();
    for (const RequestTiming &record : stats.records) {
        if (record.instance >= 0) {
            const size_t index = static_cast<size_t>(record.instance);
            instanceCount = std::max(instanceCount, index + 1);
        }
    }
    stats.perInstance.assign(instanceCount, 0);

    stats.requests = 0;
    stats.errors = 0;
    int64_t samples = 0;
    for (const RequestTiming &record : stats.records) {
        if (record.instance >= 0) {
            ++stats.perInstance[static_cast<size_t>(record.instance)];
        }
        if (record.error) {
            ++stats.errors;
            continue;
        }
        ++stats.requests;
        samples += record.samples;   // 业务样本数由调用方从 Case::samples 填进每条记录
        e2e.push_back(record.e2eMs);
        exec.push_back(record.execMs);
        d2hSum += record.d2hMs;
    }
    stats.samples = samples;

    if (e2e.empty()) {
        stats.e2eAvg = stats.e2eMin = stats.e2eP50 = stats.e2eP99 = stats.e2eMax = 0.0;
        stats.execAvg = stats.execP99 = 0.0;
        stats.d2hAvg = 0.0;
    } else {
        const double sum = std::accumulate(e2e.begin(), e2e.end(), 0.0);
        stats.e2eAvg = sum / static_cast<double>(e2e.size());
        stats.e2eMin = *std::min_element(e2e.begin(), e2e.end());
        stats.e2eMax = *std::max_element(e2e.begin(), e2e.end());
        stats.e2eP50 = Percentile(e2e, 0.50);
        stats.e2eP99 = Percentile(e2e, 0.99);
        const double execSum = std::accumulate(exec.begin(), exec.end(), 0.0);
        stats.execAvg = execSum / static_cast<double>(exec.size());
        stats.execP99 = Percentile(exec, 0.99);
        stats.d2hAvg = d2hSum / static_cast<double>(e2e.size());
    }

    if (stats.wallMs <= 0.0) {
        stats.qps = 0.0;
        stats.samplesPerSecond = 0.0;
    } else {
        stats.qps = static_cast<double>(stats.requests) * 1000.0 / stats.wallMs;
        stats.samplesPerSecond = static_cast<double>(stats.samples) * 1000.0 / stats.wallMs;
    }
}

nlohmann::json StatsToJson(const RunStats &stats) {
    nlohmann::json out;
    out["status"] = stats.status;
    out["mode"] = stats.mode;
    out["requests"] = stats.requests;
    out["errors"] = stats.errors;
    out["samples"] = stats.samples;
    out["wall_ms"] = stats.wallMs;
    out["qps"] = stats.qps;
    out["samples_per_second"] = stats.samplesPerSecond;
    out["e2e_avg"] = stats.e2eAvg;
    out["e2e_min"] = stats.e2eMin;
    out["e2e_p50"] = stats.e2eP50;
    out["e2e_p99"] = stats.e2eP99;
    out["e2e_max"] = stats.e2eMax;
    out["exec_avg"] = stats.execAvg;
    out["exec_p99"] = stats.execP99;
    out["d2h_avg"] = stats.d2hAvg;
    out["warmup_runs"] = stats.warmupRuns;
    out["warmup_ms"] = stats.warmupMs;
    out["compile_ms"] = stats.compileMs;
    out["load_ms"] = stats.loadMs;
    out["per_instance"] = stats.perInstance;

    nlohmann::json records = nlohmann::json::array();
    for (const RequestTiming &record : stats.records) {
        records.push_back({{"instance", record.instance},
                           {"request", record.request},
                           {"case_id", record.caseId},
                           {"shape_key", record.shapeKey},
                           {"samples", record.samples},
                           {"h2d_ms", record.h2dMs},
                           {"exec_ms", record.execMs},
                           {"d2h_ms", record.d2hMs},
                           {"e2e_ms", record.e2eMs},
                           {"first_hit", record.firstHit},
                           {"error", record.error}});
    }
    out["records"] = std::move(records);
    return out;
}

void WriteJsonFile(const std::string &path, const nlohmann::json &value) {
    const std::filesystem::path target(path);
    if (target.has_parent_path() && !target.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(target.parent_path(), ec);
        if (ec) {
            throw std::runtime_error("WriteJsonFile '" + path + "': cannot create parent dir: " +
                                     ec.message());
        }
    }
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("WriteJsonFile '" + path + "': cannot open for write");
    }
    out << value.dump(2) << '\n';
    if (!out) {
        throw std::runtime_error("WriteJsonFile '" + path + "': write failed");
    }
}

}  // namespace ge_execute
