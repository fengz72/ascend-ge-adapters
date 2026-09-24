#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "acl_json.h"
#include "bench_plan.h"
#include "backends/acl_backend.h"
#include "backends/gesession_backend.h"
#include "bench.h"
#include "io_spec.h"

namespace {

struct Options {
    std::string manifest;
    std::string benchPlan;              // 非空 → 多实例变长负载模式 (plan 里含 manifest 路径)
    std::string output_dir;
    int device = -1;
    int warmup = 0;
    int bench = 1;
    int graph_run_mode = 1;
    std::string precision_mode = "force_fp16";
    std::string aicore_num;
    size_t output_reserve_mb = 256;
    int threads = 1;
    int requests = 0;
    ge_runtime::DumpConfig dump;
    ge_runtime::ProfilingConfig profiling;
};

void PrintUsage(const char *prog) {
    std::cout
        << "Usage: " << prog << " <manifest.json> [options]\n"
        << "       " << prog << " --bench-plan <plan.json>      # 多实例 + 请求池 (性能测试)\n"
        << "\n配置驱动的 GE 运行时: 读 manifest → io_spec/bundle → 按 backend 分发执行 → 落盘输出。\n"
        << "\nOptions:\n"
        << "  --output_dir <dir>      输出目录 (默认 <manifest 根>/io/outputs)\n"
        << "  --device <id>           覆盖 manifest.device\n"
        << "  --warmup <N>            预热次数 (默认 0)\n"
        << "  --bench <N>             计时执行次数 (默认 1)\n"
        << "  --threads <N>           并发线程数 (>1 → 吞吐模式, 每线程独立 stream/dataset)\n"
        << "  --requests <M>          吞吐模式总请求数 (闭环, 均分到各线程; 默认每线程 --bench 个)\n"
        << "                          (并发档位扫描不在这里: 用 tools/sweep.py 逐档起进程)\n"
        << "  --bench-plan <json>     多实例变长负载: 共享请求池随机抽样 + 分阶段计时,\n"
        << "                          产出 perf.json / perf_requests.csv (由 core/bench.py 生成 plan)\n"
        << "  --graph_run_mode <m>    ge_session: 0=host 1=device (默认 1)\n"
        << "  --precision_mode <p>    ge_session: 默认 force_fp16\n"
        << "  --aicore_num <spec>     ge_session: 限核, N 或 'aic|aiv'\n"
        << "  --output_reserve <MB>   om_acl: 动态输出预留 (默认 256; 有 bundle 时按 golden 精确推导)\n"
        << "\nDump (仅 om_acl, 经 aclInit(acl.json)):\n"
        << "  --dump                  开启 dump (自动生成 acl.json)\n"
        << "  --dump_config <path>    用自备 acl.json (不再生成)\n"
        << "  --dump_path <dir>       dump 输出目录 (默认 ./dump_data)\n"
        << "  --dump_mode <m>         input|output|all (默认 output)\n"
        << "  --dump_level <l>        op|kernel|all (默认 op)\n"
        << "  --dump_data <d>         tensor|stats (默认 tensor)\n"
        << "  --dump_model_name <n>   dump 配置里的 model_name (可选)\n"
        << "  --dump_layer <a,b>      只 dump 指定层 (可选)\n"
        << "\nProfiling (om_acl 走 acl.json; ge_session 走 GEInitialize 选项):\n"
        << "  --profiling             开启 profiling\n"
        << "  --profiling_output <d>  PROF_* 输出目录 (默认 ./profiling_data)\n"
        << "  --profiling_aic_metrics <m>  如 PipeUtilization (ge_session 默认即此)\n"
        << "  --profiling_no_task_time / --profiling_no_runtime_api / --profiling_no_ascendcl\n"
        << "  解析: python3 tools/parse_profiling.py summary --profiling_dir <dir>\n"
        << "  -h, --help              显示帮助\n"
        << "\nExample:\n"
        << "  " << prog << " models/qwen2.5-0.5b/io/manifest.json --device 6 --bench 10\n"
        << std::endl;
}

bool ParseArgs(const std::vector<std::string> &args, const std::string &prog, Options &opt) {
    bool missingValue = false;
    for (size_t idx = 0; idx < args.size(); idx++) {
        const std::string &arg = args[idx];
        auto next = [&](const char *name) -> std::string {
            if (idx + 1 >= args.size()) {
                fprintf(stderr, "[ERROR] %s needs a value\n", name);
                missingValue = true;
                return "";
            }
            return args[++idx];
        };
        if (arg == "-h" || arg == "--help") {
            PrintUsage(prog.c_str());
            return false;
        } else if (arg == "--output_dir") {
            opt.output_dir = next("--output_dir");
        } else if (arg == "--device") {
            opt.device = std::atoi(next("--device").c_str());
        } else if (arg == "--warmup") {
            opt.warmup = std::atoi(next("--warmup").c_str());
        } else if (arg == "--bench") {
            opt.bench = std::atoi(next("--bench").c_str());
        } else if (arg == "--graph_run_mode") {
            opt.graph_run_mode = std::atoi(next("--graph_run_mode").c_str());
        } else if (arg == "--precision_mode") {
            opt.precision_mode = next("--precision_mode");
        } else if (arg == "--aicore_num") {
            opt.aicore_num = next("--aicore_num");
        } else if (arg == "--bench-plan") {
            opt.benchPlan = next("--bench-plan");
        } else if (arg == "--threads") {
            opt.threads = std::atoi(next("--threads").c_str());
        } else if (arg == "--requests") {
            opt.requests = std::atoi(next("--requests").c_str());
        } else if (arg == "--dump") {
            opt.dump.enabled = true;
        } else if (arg == "--dump_config") {
            opt.dump.configPath = next("--dump_config");
            opt.dump.enabled = true;
        } else if (arg == "--dump_path") {
            opt.dump.dumpPath = next("--dump_path");
        } else if (arg == "--dump_mode") {
            opt.dump.dumpMode = next("--dump_mode");
        } else if (arg == "--dump_level") {
            opt.dump.dumpLevel = next("--dump_level");
        } else if (arg == "--dump_data") {
            opt.dump.dumpData = next("--dump_data");
        } else if (arg == "--dump_model_name") {
            opt.dump.modelName = next("--dump_model_name");
        } else if (arg == "--dump_layer") {
            std::stringstream ss(next("--dump_layer"));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) {
                    opt.dump.layers.push_back(tok);
                }
            }
        } else if (arg == "--profiling") {
            opt.profiling.enabled = true;
        } else if (arg == "--profiling_output") {
            opt.profiling.outputPath = next("--profiling_output");
            opt.profiling.enabled = true;
        } else if (arg == "--profiling_aic_metrics") {
            opt.profiling.aicMetrics = next("--profiling_aic_metrics");
        } else if (arg == "--profiling_no_task_time") {
            opt.profiling.taskTime = false;
        } else if (arg == "--profiling_no_runtime_api") {
            opt.profiling.runtimeApi = false;
        } else if (arg == "--profiling_no_ascendcl") {
            opt.profiling.ascendcl = false;
        } else if (arg == "--output_reserve") {
            opt.output_reserve_mb = static_cast<size_t>(std::atoll(next("--output_reserve").c_str()));
        } else if (arg[0] == '-') {
            fprintf(stderr, "[ERROR] unknown option: %s\n", arg.c_str());
            PrintUsage(prog.c_str());
            return false;
        } else if (opt.manifest.empty()) {
            opt.manifest = arg;
        } else {
            fprintf(stderr, "[ERROR] unexpected positional argument: %s\n", arg.c_str());
            return false;
        }
    }
    if (missingValue || (opt.manifest.empty() && opt.benchPlan.empty())) {
        if (!missingValue) {
            PrintUsage(prog.c_str());
        }
        return false;
    }
    return true;
}

// argv → token 列表, 并把 --key=value 拆成 --key value (便于 Python 侧透传)
std::vector<std::string> Tokenize(int argc, char *argv[]) {
    std::vector<std::string> tokens;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        size_t eq = arg.find('=');
        if (arg.size() > 2 && arg[0] == '-' && arg[1] == '-' && eq != std::string::npos) {
            tokens.push_back(arg.substr(0, eq));
            tokens.push_back(arg.substr(eq + 1));
        } else {
            tokens.push_back(arg);
        }
    }
    return tokens;
}

}  // namespace

int main(int argc, char *argv[]) {
    using namespace ge_runtime;

    Options opt;
    if (!ParseArgs(Tokenize(argc, argv), argv[0], opt)) {
        return 1;
    }
    if (!opt.benchPlan.empty()) {
        return ge_runtime::RunBenchPlan(opt.benchPlan);
    }

    try {
        Manifest manifest = Manifest::Load(opt.manifest);
        int device = opt.device >= 0 ? opt.device : manifest.device;
        std::cout << "[INFO] manifest: " << opt.manifest << " (backend=" << manifest.backend
                  << ", device=" << device << ", base=" << manifest.base_dir << ")" << std::endl;

        IoSpec spec = IoSpec::Load(manifest.Resolve(manifest.io_spec));

        // 输入来源: bundle (验证态: 具体 shape + .bin 都在里面); 部署态 --input 已移除
        if (manifest.bundle.empty()) {
            fprintf(stderr, "[ERROR] manifest 无 bundle: 验证态需要 io/bundle.json; "
                            "部署态 --input 已移除\n");
            return 1;
        }
        std::string bundlePath = manifest.Resolve(manifest.bundle);
        Bundle bundle = Bundle::Load(bundlePath);
        std::vector<TensorPlan> plans = BuildInputPlans(spec, bundle, DirName(bundlePath));

        std::string outputDir = opt.output_dir.empty()
                                    ? manifest.Resolve("io/outputs")
                                    : opt.output_dir;

        // dump/profiling: om_acl 经 acl.json + aclInit(configPath); ge_session 经 GEInitialize
        std::string aclConfigPath;
        bool isAcl = (manifest.backend == "om_acl");
        if (opt.dump.enabled && !isAcl) {
            std::cout << "[WARN] dump 是 OM/ACL 专属能力, backend=" << manifest.backend
                      << " 下忽略 (ge_session 请用 --profiling)" << std::endl;
            opt.dump.enabled = false;
        }
        if (isAcl && (opt.dump.enabled || opt.profiling.enabled)) {
            if (!opt.dump.configPath.empty()) {
                aclConfigPath = opt.dump.configPath;
                std::cout << "[INFO] 使用自备 acl.json: " << aclConfigPath << std::endl;
            } else {
                MakeDirs(outputDir);
                if (opt.dump.enabled) {
                    MakeDirs(opt.dump.dumpPath);
                }
                if (opt.profiling.enabled) {
                    MakeDirs(opt.profiling.outputPath);
                }
                aclConfigPath = outputDir + "/acl.json";
                if (!GenerateAclJson(opt.dump, opt.profiling, aclConfigPath)) {
                    return 1;
                }
                std::cout << "[INFO] 生成 acl.json: " << aclConfigPath
                          << " (dump=" << (opt.dump.enabled ? "on" : "off")
                          << ", profiling=" << (opt.profiling.enabled ? "on" : "off") << ")"
                          << std::endl;
            }
            if (opt.dump.enabled && opt.profiling.enabled) {
                std::cout << "[WARN] dump 与 profiling 同开时 profiling 配置可能不生效 "
                             "(CANN 两套机制); 建议分两次跑" << std::endl;
            }
        }

        BenchOptions bench;
        bench.warmup = opt.warmup;
        bench.runs = opt.bench;
        bench.threads = opt.threads;
        bench.requests = opt.requests;
        std::vector<HostTensor> outputs;
        bool ok = false;
        if (manifest.backend == "om_acl") {
            AclOptions aclOpt;
            aclOpt.device = device;
            aclOpt.bench = bench;
            // 单输出时用 golden 的具体 shape 精确推导缓冲 (不靠预留猜);
            // 多输出才退回 --output_reserve
            size_t derived = spec.outputs.size() == 1 ? ExpectedOutputBytes(spec, bundle, 0) : 0;
            aclOpt.output_reserve = derived ? derived : opt.output_reserve_mb * 1024 * 1024;
            aclOpt.aclConfigPath = aclConfigPath;
            std::cout << "[INFO] 输出缓冲 " << aclOpt.output_reserve << " 字节 ("
                      << (derived ? "按 bundle golden shape 推导" : "--output_reserve 预留")
                      << ")" << std::endl;
            ok = RunAclBackend(manifest, spec, plans, aclOpt, outputs);
        } else if (manifest.backend == "ge_session") {
            GeSessionOptions geOpt;
            geOpt.device = device;
            geOpt.bench = bench;
            geOpt.graph_run_mode = opt.graph_run_mode;
            geOpt.precision_mode = opt.precision_mode;
            geOpt.aicore_num = opt.aicore_num;
            geOpt.profiling.enabled = opt.profiling.enabled;
            geOpt.profiling.outputPath = opt.profiling.outputPath;
            if (!opt.profiling.aicMetrics.empty()) {
                geOpt.profiling.aicMetrics = opt.profiling.aicMetrics;
            }
            ok = RunGeSessionBackend(manifest, spec, plans, geOpt, outputs);
        } else {
            fprintf(stderr, "[ERROR] unknown backend: %s\n", manifest.backend.c_str());
            return 1;
        }
        if (!ok) {
            fprintf(stderr, "[ERROR] backend '%s' execution failed\n", manifest.backend.c_str());
            return 1;
        }

        std::string indexPath;
        if (!SaveOutputs(outputDir, outputs, indexPath)) {
            return 1;
        }
        std::cout << "[INFO] " << outputs.size() << " output(s) saved to " << outputDir
                  << " (index: " << indexPath << ")" << std::endl;
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "[ERROR] %s\n", e.what());
        return 1;
    }
}
