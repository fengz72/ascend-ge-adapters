#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "backends/acl_backend.h"
#include "backends/gesession_backend.h"
#include "bench.h"
#include "io_spec.h"

namespace {

struct Options {
    std::string manifest;
    std::string output_dir;
    std::vector<std::string> inputs;      // 部署态 (无 bundle): logical:d0,d1,...:file
    int device = -1;
    int warmup = 0;
    int bench = 1;
    int graph_run_mode = 1;
    std::string precision_mode = "force_fp16";
    std::string aicore_num;
    size_t output_reserve_mb = 256;
};

void PrintUsage(const char *prog) {
    std::cout
        << "Usage: " << prog << " <manifest.json> [options]\n"
        << "\n配置驱动的 GE 运行时: 读 manifest → io_spec/bundle → 按 backend 分发执行 → 落盘输出。\n"
        << "\nOptions:\n"
        << "  --output_dir <dir>      输出目录 (默认 <manifest 根>/verification/outputs)\n"
        << "  --input <spec>          部署态 (manifest 无 bundle) 必填, 可重复:\n"
        << "                            logical:d0,d1,...:file.bin   (dtype/format/node 取自 io_spec)\n"
        << "  --device <id>           覆盖 manifest.device\n"
        << "  --warmup <N>            预热次数 (默认 0)\n"
        << "  --bench <N>             计时执行次数 (默认 1)\n"
        << "  --graph_run_mode <m>    ge_session: 0=host 1=device (默认 1)\n"
        << "  --precision_mode <p>    ge_session: 默认 force_fp16\n"
        << "  --aicore_num <spec>     ge_session: 限核, N 或 'aic|aiv'\n"
        << "  --output_reserve <MB>   om_acl: 动态输出预留 (默认 256)\n"
        << "  -h, --help              显示帮助\n"
        << "\nExample:\n"
        << "  " << prog << " models/qwen2.5-0.5b/deploy/manifest.json --device 6 --bench 10\n"
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
        } else if (arg == "--input") {
            std::string spec = next("--input");
            if (spec.empty()) {
                missingValue = true;
            } else {
                opt.inputs.push_back(spec);
            }
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
    if (missingValue || opt.manifest.empty()) {
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

    try {
        Manifest manifest = Manifest::Load(opt.manifest);
        int device = opt.device >= 0 ? opt.device : manifest.device;
        std::cout << "[INFO] manifest: " << opt.manifest << " (backend=" << manifest.backend
                  << ", device=" << device << ", base=" << manifest.base_dir << ")" << std::endl;

        IoSpec spec = IoSpec::Load(manifest.Resolve(manifest.io_spec));

        // 输入来源二选一: bundle (验证态: 具体 shape + .bin 都在里面) 或 --input (部署态)
        Bundle bundle;
        bool haveBundle = !manifest.bundle.empty();
        std::vector<TensorPlan> plans;
        if (haveBundle) {
            std::string bundlePath = manifest.Resolve(manifest.bundle);
            bundle = Bundle::Load(bundlePath);
            plans = BuildInputPlans(spec, bundle, DirName(bundlePath));
        } else if (!opt.inputs.empty()) {
            plans = BuildInputPlansFromArgs(spec, opt.inputs);
            std::cout << "[INFO] 部署态: manifest 无 bundle, 输入来自 --input ("
                      << plans.size() << " 项)" << std::endl;
        } else {
            fprintf(stderr, "[ERROR] 输入无来源: manifest 没有 bundle, 也没给 --input\n"
                            "        验证态: manifest.bundle → verification/bundle.json (docs §5.4)\n"
                            "        部署态: --input logical:d0,d1,...:file.bin (逐输入, 可重复)\n");
            return 1;
        }

        std::string outputDir = opt.output_dir.empty()
                                    ? manifest.Resolve("verification/outputs")
                                    : opt.output_dir;

        BenchOptions bench{opt.warmup, opt.bench};
        std::vector<HostTensor> outputs;
        bool ok = false;
        if (manifest.backend == "om_acl") {
            AclOptions aclOpt;
            aclOpt.device = device;
            aclOpt.bench = bench;
            // 单输出且有 bundle 时, 用 golden 的具体 shape 精确推导缓冲 (不靠预留猜);
            // 多输出/部署态才退回 --output_reserve
            size_t derived = (haveBundle && spec.outputs.size() == 1)
                                 ? ExpectedOutputBytes(spec, bundle, 0) : 0;
            aclOpt.output_reserve = derived ? derived : opt.output_reserve_mb * 1024 * 1024;
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
