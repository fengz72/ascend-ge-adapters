#pragma once
// 抽样计划与计时聚合。纯 CPU, 不依赖 GE / ACL。
//
// 口径 (与项目既有约定一致):
//   * warmup 是**每实例**次数, 且每实例至少跑遍池内每个 distinct shape 一次
//     (GE 首次执行某 shape 才特化, 特化代价必须留在预热段)。
//   * iterations 是**所有实例合计**的测量次数; 全局序列先按 seed 生成, 再按实例
//     交错切分, 所以实例数只改分工, 不改序列本身。
//   * e2e = h2d + execute + sync + 全量 D2H (d2hMs 由调用方从 CollectOutputs 折进来)。
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ge_execute {

struct RequestTiming {
    int instance = 0;
    size_t request = 0;               // 全局测量序号 (0-based)
    std::string caseId;
    std::string shapeKey;
    int64_t samples = 0;              // 该 case 的业务样本数 (调用方从 Case::samples 填)
    double h2dMs = 0, execMs = 0, d2hMs = 0, e2eMs = 0;
    bool firstHit = false;            // 该实例在测量段遇到 warmup 未覆盖的 shape
    bool error = false;
};

struct RunStats {
    std::string status = "success";   // success | failed
    std::string mode = "bench";       // infer | bench
    size_t requests = 0, errors = 0;  // 测量段成功 / 失败次数
    int64_t samples = 0;              // 成功请求的业务样本合计
    double wallMs = 0, qps = 0, samplesPerSecond = 0;
    double e2eAvg = 0, e2eMin = 0, e2eP50 = 0, e2eP99 = 0, e2eMax = 0;
    double execAvg = 0, execP99 = 0;
    double d2hAvg = 0;                // 全量 D2H 平均耗时 (e2e 的一部分)
    size_t warmupRuns = 0;
    double warmupMs = 0, compileMs = 0;
    std::vector<double> loadMs;       // 每实例 LoadGraph + 缓冲分配
    std::vector<size_t> perInstance;  // 每实例测量请求数 (含失败)
    std::vector<RequestTiming> records;
};

// nearest-rank: min(count-1, ceil(quantile*count)-1)。
// count==0 或 quantile 不在 (0,1] 抛 std::invalid_argument。
size_t NearestRankIndex(size_t count, double quantile);

// 按值排序后取分位; values 为空返回 0。
double Percentile(std::vector<double> values, double quantile);

// mt19937_64(seed) 有放回抽样 count 次; poolSize<=1 时恒为 0。
std::vector<size_t> DrawSequence(size_t poolSize, size_t count, uint64_t seed);

// 测量计划: DrawSequence(poolSize, requests, seed) 生成全局序列后按实例交错切分
// (实例 t 取全局下标 t, t+K, t+2K, ...)。instances<1 视为 1; requests==0 返回空计划。
std::vector<std::vector<size_t>> MeasurePlan(size_t poolSize, size_t requests,
                                             size_t instances, uint64_t seed);

// 预热计划: shapeKeys 与 case 同序, 取每个 distinct shape 首次出现的下标作为代表,
// 每实例先跑全部代表, 再用 DrawSequence(poolSize, extra, seed) 轮转补齐到
// 每实例 warmupPerInstance 次。warmupPerInstance==0 时仍覆盖全部 shape。
// instances<1 视为 1; shapeKeys 为空返回空计划。
std::vector<std::vector<size_t>> WarmupPlan(const std::vector<std::string> &shapeKeys,
                                            size_t warmupPerInstance, size_t instances,
                                            uint64_t seed);

// 由 records 汇总 requests/errors/samples/e2e*/exec*/perInstance;
// samples 累加成功请求的 RequestTiming::samples (调用方必须先填好)。
// wall/qps/samplesPerSecond 由 wallMs 推出 (wallMs<=0 时两者为 0)。
// compile/load/warmup 字段由调用方填, Summarize 不改它们。
void Summarize(RunStats &stats);

// result.json 的完整内容 (含 records 数组)。
nlohmann::json StatsToJson(const RunStats &stats);

// 缩进 2 空格写文件, 父目录不存在则递归创建。失败抛 std::runtime_error。
void WriteJsonFile(const std::string &path, const nlohmann::json &value);

}  // namespace ge_execute
