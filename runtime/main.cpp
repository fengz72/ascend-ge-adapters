// ge_execute: 加载 AIR/ONNX 单图 -> GESession -> 按 cases.json 跑 infer 或 bench。
//
//   ge_execute --graph <model.air|model.onnx> --cases <cases.json> --output-dir <dir> --device <N>
//              [--format air|onnx] [--instances K=1] [--warmup M=0] [--iterations N=0] [--seed S=42]
//              [--aicore-num A|B] [--precision-mode force_fp16] [--graph-run-mode 0]
//              [--profiling] [--aic-metrics PipeUtilization] [--help]
//
// infer (iterations==0): 单实例按 cases 顺序各跑一次, 每个输出写
//   <output-dir>/<case_id>/output_<i>.npy。
// bench (iterations>0): K 个实例先各自预热 (必须覆盖池内每个 distinct shape),
//   再按种子序列跑 N 次全局请求。两种模式同口径:
//   e2e = h2d + execute + sync + 全量 D2H (bench 丢弃回读数据, 但传输照做)。
//
// 这里只做 CLI 编排; GE/ACL 细节全部隐藏在 session.cpp 后面。
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "npy.h"
#include "session.h"
#include "stats.h"

namespace ge_execute {
namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

// 与 session.cpp 同口径: steady_clock 先落到 us, 再换算 ms。
double ElapsedMs(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /
           1000.0;
}

std::string ShapeToString(const std::vector<int64_t> &shape) {
    std::string text = "[";
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            text += ",";
        }
        text += std::to_string(shape[index]);
    }
    return text + "]";
}

// CLI 用法错误统一 exit 2, 与运行期异常 (exit 1) 区分开, 方便脚本判断。
struct CliError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Options {
    std::string graphPath;
    std::string casesPath;
    std::string outputDir;
    std::string format;                       // 空 = 按扩展名推断
    std::string aicoreNum;
    std::string precisionMode = "force_fp16";
    std::string aicMetrics = "PipeUtilization";
    long long device = -1;
    long long instances = 1;
    long long warmup = 0;
    long long iterations = 0;
    unsigned long long seed = 42;
    long long graphRunMode = 0;
    bool profiling = false;
    bool help = false;
};

void Usage(std::ostream &os) {
    os << "usage: ge_execute --graph <model.air|model.onnx> --cases <cases.json>\n"
          "                   --output-dir <dir> --device <N>\n"
          "                   [--format air|onnx] [--instances K=1] [--warmup M=0]\n"
          "                   [--iterations N=0] [--seed S=42] [--aicore-num A|B]\n"
          "                   [--precision-mode force_fp16] [--graph-run-mode 0]\n"
          "                   [--profiling] [--aic-metrics PipeUtilization] [--help]\n"
          "  --iterations N > 0: bench over N global requests on K closed-loop instances\n"
          "  default:            infer, every case once in order -> per-case .npy outputs\n";
}

long long ToInt(const std::string &text, const char *name) {
    try {
        size_t used = 0;
        const long long value = std::stoll(text, &used);
        if (used != text.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception &) {
        throw CliError(std::string(name) + " expects an integer: '" + text + "'");
    }
}

unsigned long long ToU64(const std::string &text, const char *name) {
    if (text.empty() || text.front() == '-') {
        throw CliError(std::string(name) + " expects a non-negative integer: '" + text + "'");
    }
    try {
        size_t used = 0;
        const unsigned long long value = std::stoull(text, &used);
        if (used != text.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception &) {
        throw CliError(std::string(name) + " expects a non-negative integer: '" + text + "'");
    }
}

void ParseArgs(int argc, char **argv, Options &opt) {
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto value = [&](const char *name) -> std::string {
            // 缺值表现为越界, 或者下一个 token 已经是一个 flag。
            if (index + 1 >= argc || std::string(argv[index + 1]).rfind("--", 0) == 0) {
                throw CliError(std::string(name) + " needs a value");
            }
            return argv[++index];
        };
        if (arg == "--graph") {
            opt.graphPath = value("--graph");
        } else if (arg == "--cases") {
            opt.casesPath = value("--cases");
        } else if (arg == "--output-dir") {
            opt.outputDir = value("--output-dir");
        } else if (arg == "--device") {
            opt.device = ToInt(value("--device"), "--device");
        } else if (arg == "--format") {
            opt.format = value("--format");
        } else if (arg == "--instances") {
            opt.instances = ToInt(value("--instances"), "--instances");
        } else if (arg == "--warmup") {
            opt.warmup = ToInt(value("--warmup"), "--warmup");
        } else if (arg == "--iterations") {
            opt.iterations = ToInt(value("--iterations"), "--iterations");
        } else if (arg == "--seed") {
            opt.seed = ToU64(value("--seed"), "--seed");
        } else if (arg == "--aicore-num") {
            opt.aicoreNum = value("--aicore-num");
        } else if (arg == "--precision-mode") {
            opt.precisionMode = value("--precision-mode");
        } else if (arg == "--graph-run-mode") {
            opt.graphRunMode = ToInt(value("--graph-run-mode"), "--graph-run-mode");
        } else if (arg == "--aic-metrics") {
            opt.aicMetrics = value("--aic-metrics");
        } else if (arg == "--profiling") {
            opt.profiling = true;
        } else if (arg == "--help" || arg == "-h") {
            opt.help = true;
            return;
        } else {
            throw CliError("unknown argument: " + arg);
        }
    }
    if (opt.graphPath.empty() || opt.casesPath.empty() || opt.outputDir.empty() ||
        opt.device < 0) {
        throw CliError("--graph, --cases, --output-dir and --device are required");
    }
    if (opt.instances < 1) {
        throw CliError("--instances needs a value >= 1");
    }
    if (opt.warmup < 0) {
        throw CliError("--warmup needs a value >= 0");
    }
    if (opt.iterations < 0) {
        throw CliError("--iterations needs a value >= 0");
    }
    if (opt.graphRunMode != 0 && opt.graphRunMode != 1) {
        throw CliError("--graph-run-mode must be 0 or 1");
    }
    try {
        NormalizeAicoreNum(opt.aicoreNum);   // 值格式不对属于用法错误 -> exit 2
    } catch (const std::exception &error) {
        throw CliError(error.what());
    }
}

Json ConfigJson(const Options &opt, const std::string &format, int instances) {
    return Json{{"graph", opt.graphPath},
                {"cases", opt.casesPath},
                {"output_dir", opt.outputDir},
                {"device", opt.device},
                {"format", format},
                {"instances", instances},
                {"warmup", opt.warmup},
                {"iterations", opt.iterations},
                {"seed", opt.seed},
                {"aicore_num", opt.aicoreNum},
                {"precision_mode", opt.precisionMode},
                {"graph_run_mode", opt.graphRunMode},
                {"profiling", opt.profiling},
                {"aic_metrics", opt.aicMetrics},
                {"profiling_output", opt.profiling ? opt.outputDir + "/profiling" : ""}};
}

// result.json 的公共骨架: 固定头部 (status/mode/config/pool) + StatsToJson 的其余字段。
// 逐字段搬运是为了不依赖 StatsToJson 内部的键顺序, 新加字段也能自动透传。
Json ResultBase(const RunStats &stats, const Json &config, const CasePool &pool) {
    Json result = Json::object();
    result["status"] = stats.status;
    result["mode"] = stats.mode;
    result["config"] = config;
    result["pool"] = Json{{"cases", pool.cases.size()},
                          {"distinct_shapes", pool.distinctShapes}};
    Json tail = StatsToJson(stats);
    for (auto it = tail.begin(); it != tail.end(); ++it) {
        if (it.key() != "status" && it.key() != "mode") {
            result[it.key()] = it.value();
        }
    }
    return result;
}

// ---------------------------------------------------------------- infer

bool RunInfer(const Options &opt, const Json &config, const CasePool &pool, Session &session) {
    RunStats stats;
    stats.mode = "infer";
    Json cases = Json::array();
    size_t index = 0;
    for (const Case &request : pool.cases) {
        // infer 固定单实例, 按 cases 顺序各跑一次。计时覆盖 h2d + execute + sync + 全量
        // D2H; 写 npy 是交付动作, 不进 wall_ms。
        const RunTimings measured = session.Execute(0, request);
        const Readback readback = session.CollectOutputs(0);
        const std::vector<OutputTensor> &outputs = readback.outputs;
        Json entries = Json::array();
        for (size_t outputIndex = 0; outputIndex < outputs.size(); ++outputIndex) {
            const OutputTensor &output = outputs[outputIndex];
            const std::string name = "output_" + std::to_string(outputIndex);
            const std::string path =
                opt.outputDir + "/" + request.id + "/" + name + ".npy";
            WriteNpy(path, output.dtype, output.shape, output.data.data(), output.data.size());
            entries.push_back(Json{{"name", name},
                                   {"dtype", output.dtype},
                                   {"shape", output.shape},
                                   {"path", path},
                                   {"bytes", output.data.size()}});
            std::printf("  %s %s %s %zu bytes -> %s\n", request.id.c_str(), name.c_str(),
                        (output.dtype + ShapeToString(output.shape)).c_str(),
                        output.data.size(), path.c_str());
        }
        cases.push_back(Json{{"id", request.id}, {"outputs", entries}});

        RequestTiming timing;
        timing.instance = 0;
        timing.request = index++;
        timing.caseId = request.id;
        timing.shapeKey = request.shapeKey;
        timing.samples = request.samples;
        timing.h2dMs = measured.h2dMs;
        timing.execMs = measured.execMs;
        timing.d2hMs = readback.d2hMs;
        timing.e2eMs = measured.e2eMs + readback.d2hMs;
        stats.records.push_back(timing);
    }
    // wall_ms = 逐 case (h2d + execute + sync + D2H) 之和, 不含写 npy,
    // 所以 infer 的 qps 是执行侧吞吐。
    for (const RequestTiming &timing : stats.records) {
        stats.wallMs += timing.e2eMs;
    }
    Summarize(stats);

    Json result = ResultBase(stats, config, pool);
    result["cases"] = cases;
    result["timed_phases"] = Json::array({"h2d", "execute", "sync", "d2h"});
    const std::string resultPath = opt.outputDir + "/result.json";
    WriteJsonFile(resultPath, result);

    std::printf("infer: %zu case(s), e2e %.3f ms (avg %.3f, p50 %.3f, max %.3f, d2h avg %.3f)\n",
                pool.cases.size(), stats.wallMs, stats.e2eAvg, stats.e2eP50, stats.e2eMax,
                stats.d2hAvg);
    std::printf("  result: %s\n", resultPath.c_str());
    return true;
}

// ---------------------------------------------------------------- bench

struct WorkerOutcome {
    std::vector<RequestTiming> records;
    size_t warmupRuns = 0;
    std::string warmupError;
};

bool RunBench(const Options &opt, const Json &config, const CasePool &pool, Session &session) {
    const size_t instances = static_cast<size_t>(opt.instances);
    const std::vector<std::string> shapeKeys = pool.shapeKeys();
    // 预热是每实例次数, 但必须先覆盖池内每个 distinct shape: GE 首次执行某 shape 才特化,
    // 这笔代价必须留在预热段, 不能落进测量段的第一发请求。
    const std::vector<std::vector<size_t>> warmupPlans =
        WarmupPlan(shapeKeys, static_cast<size_t>(opt.warmup), instances, opt.seed);
    // iterations 是所有实例合计的测量次数; 全局序列按 seed 生成后交错切分。
    const std::vector<std::vector<size_t>> measurePlans =
        MeasurePlan(pool.cases.size(), static_cast<size_t>(opt.iterations), instances, opt.seed);

    std::vector<WorkerOutcome> outcomes(instances);
    std::atomic<bool> halt{false};
    std::mutex gateMutex;
    std::condition_variable gate;
    size_t warmupDone = 0;
    bool timedStart = false;
    std::mutex errorMutex;
    std::string firstError;

    auto recordError = [&](const std::string &what) {
        std::lock_guard<std::mutex> lock(errorMutex);
        if (firstError.empty()) {
            firstError = what;
        }
    };

    const auto warmupBegin = Clock::now();
    std::vector<std::thread> workers;
    workers.reserve(instances);
    for (size_t instance = 0; instance < instances; ++instance) {
        workers.emplace_back([&, instance] {
            WorkerOutcome &outcome = outcomes[instance];
            // 覆盖集合: 预热跑过的 shapeKey + 测量段已经见过的, 用来标 firstHit。
            std::unordered_map<std::string, bool> covered;
            try {
                session.BindThread();   // 工作线程共享默认 context
                for (size_t requestIndex : warmupPlans[instance]) {
                    session.Execute(static_cast<int>(instance), pool.cases[requestIndex]);
                    // 预热要走和测量段完全相同的路径: 含全量 D2H (discard), 这样
                    // scratch 缓冲的增长和 GE 输出缓冲的复用都在预热段付掉。
                    session.CollectOutputs(static_cast<int>(instance), /*discard=*/true);
                    covered[pool.cases[requestIndex].shapeKey] = true;
                    outcome.warmupRuns += 1;
                }
            } catch (const std::exception &error) {
                outcome.warmupError = error.what();
                halt.store(true);
            }

            {
                std::lock_guard<std::mutex> lock(gateMutex);
                warmupDone += 1;
            }
            gate.notify_all();
            {
                std::unique_lock<std::mutex> lock(gateMutex);
                gate.wait(lock, [&] { return timedStart; });
            }
            if (halt.load()) {
                return;   // 预热失败: 整体失败, 不再进入测量段
            }

            const std::vector<size_t> &plan = measurePlans[instance];
            outcome.records.reserve(plan.size());
            for (size_t position = 0; position < plan.size(); ++position) {
                if (halt.load()) {
                    break;
                }
                const Case &request = pool.cases[plan[position]];
                RequestTiming timing;
                timing.instance = static_cast<int>(instance);
                // MeasurePlan 把全局下标 j*K+t 分给实例 t, 这里还原全局序号用于排序。
                timing.request = position * instances + instance;
                timing.caseId = request.id;
                timing.shapeKey = request.shapeKey;
                timing.samples = request.samples;
                timing.firstHit = covered.emplace(request.shapeKey, true).second;
                try {
                    const RunTimings measured =
                        session.Execute(static_cast<int>(instance), request);
                    // bench 同样做全量 D2H (discard=true 只丢数据不省传输), 口径与 infer
                    // 一致: e2e = h2d + execute + sync + D2H。
                    const Readback readback =
                        session.CollectOutputs(static_cast<int>(instance), /*discard=*/true);
                    timing.h2dMs = measured.h2dMs;
                    timing.execMs = measured.execMs;
                    timing.d2hMs = readback.d2hMs;
                    timing.e2eMs = measured.e2eMs + readback.d2hMs;
                } catch (const std::exception &error) {
                    timing.error = true;
                    halt.store(true);
                    recordError(error.what());
                }
                outcome.records.push_back(timing);
            }
        });
    }

    // 全部实例预热完成后统一开始计时: wallMs 只覆盖测量段, 不含预热/编译/LoadGraph。
    {
        std::unique_lock<std::mutex> lock(gateMutex);
        gate.wait(lock, [&] { return warmupDone == instances; });
    }
    const bool warmupFailed = halt.load();
    const double warmupMs = ElapsedMs(warmupBegin, Clock::now());
    const auto measureBegin = Clock::now();
    {
        std::lock_guard<std::mutex> lock(gateMutex);
        timedStart = true;
    }
    gate.notify_all();
    for (std::thread &worker : workers) {
        worker.join();
    }
    const double wallMs = ElapsedMs(measureBegin, Clock::now());

    RunStats stats;
    stats.mode = "bench";
    for (const WorkerOutcome &outcome : outcomes) {
        stats.warmupRuns += outcome.warmupRuns;
        for (const RequestTiming &record : outcome.records) {
            stats.records.push_back(record);
        }
        if (!outcome.warmupError.empty()) {
            std::fprintf(stderr, "ge_execute: instance warmup failed: %s\n",
                         outcome.warmupError.c_str());
        }
    }
    std::sort(stats.records.begin(), stats.records.end(),
              [](const RequestTiming &left, const RequestTiming &right) {
                  return left.request < right.request;
              });
    stats.wallMs = warmupFailed ? 0.0 : wallMs;
    stats.warmupMs = warmupMs;
    stats.compileMs = session.compileMs();
    for (size_t instance = 0; instance < instances; ++instance) {
        stats.loadMs.push_back(session.loadMs(static_cast<int>(instance)));
    }
    Summarize(stats);   // requests/errors/samples/e2e*/exec*/perInstance/qps 全部由 records 汇总
    stats.status = (warmupFailed || stats.errors > 0) ? "failed" : "success";
    if (!firstError.empty()) {
        std::fprintf(stderr, "ge_execute: request failed: %s\n", firstError.c_str());
    }

    Json result = ResultBase(stats, config, pool);
    // 明示计时口径: bench 段不做 D2H; execMs 是 execute + sync 的合成值。
    result["timed_phases"] = Json::array({"h2d", "execute", "sync", "d2h"});
    const std::string resultPath = opt.outputDir + "/result.json";
    WriteJsonFile(resultPath, result);

    std::printf("bench: requests=%zu errors=%zu qps=%.3f samples/s=%.3f\n", stats.requests,
                stats.errors, stats.qps, stats.samplesPerSecond);
    std::printf("  e2e ms: p50=%.3f p99=%.3f avg=%.3f (exec avg %.3f, d2h avg %.3f)\n",
                stats.e2eP50, stats.e2eP99, stats.e2eAvg, stats.execAvg, stats.d2hAvg);
    std::printf("  warmup: %zu run(s), %.3f ms\n", stats.warmupRuns, stats.warmupMs);
    std::printf("  compile: %.3f ms\n", stats.compileMs);
    for (size_t instance = 0; instance < stats.loadMs.size(); ++instance) {
        std::printf("  load: instance %zu = %.3f ms\n", instance, stats.loadMs[instance]);
    }
    std::printf("  result: %s\n", resultPath.c_str());
    return stats.status == "success";
}

}  // namespace
}  // namespace ge_execute

int main(int argc, char **argv) {
    using namespace ge_execute;
    try {
        Options opt;
        ParseArgs(argc, argv, opt);
        if (opt.help) {
            Usage(std::cout);
            return 0;
        }

        std::string format = opt.format;
        if (format.empty()) {
            try {
                format = InferFormat(opt.graphPath);
            } catch (const std::exception &error) {
                throw CliError(error.what());   // 图路径后缀不对属于用法错误 -> exit 2
            }
        } else {
            std::transform(format.begin(), format.end(), format.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (format != "air" && format != "onnx") {
                throw CliError("--format must be air or onnx: '" + opt.format + "'");
            }
        }
        const bool bench = opt.iterations > 0;
        // infer 忽略 instances/warmup/seed: 固定单实例按 cases 顺序各跑一次。
        const int instances = bench ? static_cast<int>(opt.instances) : 1;

        const CasePool pool = LoadCases(opt.casesPath);
        SessionConfig config;
        config.graphPath = opt.graphPath;
        config.format = format;
        config.device = static_cast<int>(opt.device);
        config.instances = instances;
        config.aicoreNum = opt.aicoreNum;
        config.precisionMode = opt.precisionMode;
        config.graphRunMode = static_cast<int>(opt.graphRunMode);
        config.profiling = opt.profiling;
        config.profilingOutput = opt.profiling ? opt.outputDir + "/profiling" : "";
        config.aicMetrics = opt.aicMetrics;

        Session session(config);
        session.Init(pool);
        std::printf("graph: %s (%s), device %d, %zu case(s), %zu distinct shape(s)\n",
                    opt.graphPath.c_str(), format.c_str(), config.device, pool.cases.size(),
                    pool.distinctShapes);

        const Json configJson = ConfigJson(opt, format, instances);
        const bool ok = bench ? RunBench(opt, configJson, pool, session)
                              : RunInfer(opt, configJson, pool, session);
        return ok ? 0 : 1;
    } catch (const CliError &error) {
        std::fprintf(stderr, "ge_execute: %s\n", error.what());
        Usage(std::cerr);
        return 2;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "ge_execute: %s\n", error.what());
        return 1;
    }
}
