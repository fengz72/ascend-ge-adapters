#include "bench_plan.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "backends/acl_backend.h"
#include "backends/gesession_backend.h"
#include "bench.h"
#include "io_spec.h"
#include "pool.h"

namespace ge_runtime {
namespace {

using Json = nlohmann::json;

Json ParseJsonFile(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("打不开 bench plan: " + path);
    }
    return Json::parse(f);
}

bool WriteJson(const std::string &path, const Json &j) {
    if (path.empty()) {
        return true;
    }
    MakeDirs(DirName(path));
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[ERROR] 无法写 %s\n", path.c_str());
        return false;
    }
    f << j.dump(2) << std::endl;
    return true;
}

}  // namespace

int RunBenchPlan(const std::string &planPath) {
    try {
        Json plan = ParseJsonFile(planPath);
        std::string manifestPath = plan.at("manifest").get<std::string>();
        Manifest manifest = Manifest::Load(manifestPath);

        int device = plan.value("device", manifest.device);
        int instances = plan.value("instances", 1);
        size_t requests = plan.value("requests", static_cast<size_t>(0));
        int warmup = plan.value("warmup", 0);
        uint64_t seed = plan.value("seed", static_cast<uint64_t>(0));

        Json inputsCfg = plan.value("inputs", Json::object());
        std::string mode = inputsCfg.value("mode", "pool");
        std::string poolDir = inputsCfg.value("dir", "");
        if (mode != "pool") {
            fprintf(stderr, "[ERROR] bench plan 只支持 inputs.mode=pool (收到 %s); "
                            "固定输入直接用 ge_runtime <manifest> --bench N\n", mode.c_str());
            return 1;
        }
        if (poolDir.empty()) {
            fprintf(stderr, "[ERROR] bench plan 缺 inputs.dir (请求池目录)\n");
            return 1;
        }

        IoSpec spec = IoSpec::Load(manifest.Resolve(manifest.io_spec));
        RequestPool pool = LoadPool(spec, poolDir);
        std::cout << "[INFO] bench plan: " << planPath << "\n"
                  << "[INFO]   backend=" << manifest.backend << " device=" << device
                  << " instances=" << instances << " requests=" << requests
                  << " warmup=" << warmup << " seed=" << seed << "\n"
                  << "[INFO]   请求池 " << pool.requests.size() << " 套 / "
                  << pool.distinctShapes << " 种 shape, 最大输出 "
                  << pool.maxOutputBytes << " B" << std::endl;

        BenchOptions bench;
        bench.warmup = warmup;
        bench.runs = 1;
        bench.threads = instances;
        bench.requests = static_cast<int>(requests);

        Json backendCfg = plan.value("backend_options", Json::object());
        Json profCfg = plan.value("profiling", Json::object());
        bool profEnabled = profCfg.value("enabled", false);
        PerfResult perf;
        std::vector<HostTensor> outputs;      // 池模式不落盘输出 (精度与性能分开跑)
        bool ok = false;

        if (manifest.backend == "om_acl") {
            AclOptions opt;
            opt.device = device;
            opt.bench = bench;
            opt.pool = &pool;
            opt.seed = seed;
            opt.output_reserve = static_cast<size_t>(backendCfg.value("output_reserve_mb", 256)) *
                                 1024 * 1024;
            opt.aclConfigPath = backendCfg.value("acl_config", "");
            if (profEnabled && opt.aclConfigPath.empty()) {
                // ACL 的 profiling 只能经 aclInit(acl.json) 生效, 而生成 acl.json 是单请求路径
                // (main.cpp) 的职责 — 此处不重复一套生成逻辑, 明确告知而非静默忽略。
                std::cout << "[WARN] bench-plan 模式不为 om_acl 生成 acl.json → profiling 未开启 "
                             "(改用 ge_session, 或让 backend_options.acl_config 指向自备 acl.json; "
                             "单请求路径 ge_runtime <manifest> --profiling 会自动生成)" << std::endl;
            }
            ok = RunAclBackend(manifest, spec, {}, opt, outputs, &perf);
        } else if (manifest.backend == "ge_session") {
            GeSessionOptions opt;
            opt.device = device;
            opt.bench = bench;
            opt.pool = &pool;
            opt.seed = seed;
            opt.graph_run_mode = backendCfg.value("graph_run_mode", 1);
            opt.precision_mode = backendCfg.value("precision_mode", "force_fp16");
            opt.aicore_num = backendCfg.value("aicore_num", "");
            if (profEnabled) {
                opt.profiling.enabled = true;
                opt.profiling.outputPath = profCfg.value("output", "./profiling_data");
                std::string metrics = profCfg.value("aic_metrics", "");
                if (!metrics.empty()) {       // 空 → 保留 GeProfilingConfig 的默认指标
                    opt.profiling.aicMetrics = metrics;
                }
            }
            ok = RunGeSessionBackend(manifest, spec, {}, opt, outputs, &perf);
        } else {
            fprintf(stderr, "[ERROR] unknown backend: %s\n", manifest.backend.c_str());
            return 1;
        }
        if (!ok) {
            fprintf(stderr, "[ERROR] bench plan 执行失败 (详见上方)\n");
            return 1;
        }

        // ---- 落盘: perf.json (数据 + C++ 侧 meta) 与 perf_requests.csv ----
        Json report = plan.value("report", Json::object());
        Json result = PerfToJson(perf);
        result["backend"] = manifest.backend;
        result["device"] = device;
        result["manifest"] = manifestPath;
        result["graph_path"] = manifest.Resolve(manifest.graph_path);
        result["om_path"] = manifest.om_path.empty() ? Json() : Json(manifest.Resolve(manifest.om_path));
        result["load"] = {{"instances", instances}, {"requests", requests},
                          {"warmup", warmup}, {"seed", seed}};
        result["pool"] = {{"dir", poolDir}, {"size", pool.requests.size()},
                          {"distinct_shapes", pool.distinctShapes},
                          {"max_output_bytes", pool.maxOutputBytes}};
        for (auto it = report.begin(); it != report.end(); ++it) {
            if (it.key() != "perf_json" && it.key() != "requests_csv") {
                result[it.key()] = it.value();      // run_id / scenario 等由 Python 侧塞进来
            }
        }

        if (!WriteJson(report.value("perf_json", ""), result)) {
            return 1;
        }
        std::string csv = report.value("requests_csv", "");
        if (!csv.empty() && !WriteRequestsCsv(csv, perf)) {
            return 1;
        }
        std::cout << "[INFO] 性能数据已写: " << report.value("perf_json", "(未指定)")
                  << (csv.empty() ? "" : " + " + csv) << std::endl;
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "[ERROR] %s\n", e.what());
        return 1;
    }
}

}  // namespace ge_runtime
