#include "gesession_backend.h"

#include <acl/acl.h>
#include <ge/ge_api.h>
#include <graph/graph.h>
#include <exe_graph/runtime/tensor.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <nlohmann/json.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ge_runtime {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0;
}

bool ToGeDtype(const std::string &s, ge::DataType &dt) {
    if (s == "float16") { dt = ge::DT_FLOAT16; return true; }
    if (s == "bfloat16") { dt = ge::DT_BF16; return true; }
    if (s == "float32" || s == "float") { dt = ge::DT_FLOAT; return true; }
    if (s == "float64" || s == "double") { dt = ge::DT_DOUBLE; return true; }
    if (s == "int8") { dt = ge::DT_INT8; return true; }
    if (s == "int16") { dt = ge::DT_INT16; return true; }
    if (s == "int32") { dt = ge::DT_INT32; return true; }
    if (s == "int64") { dt = ge::DT_INT64; return true; }
    if (s == "uint8") { dt = ge::DT_UINT8; return true; }
    if (s == "uint16") { dt = ge::DT_UINT16; return true; }
    if (s == "uint32") { dt = ge::DT_UINT32; return true; }
    if (s == "uint64") { dt = ge::DT_UINT64; return true; }
    if (s == "bool") { dt = ge::DT_BOOL; return true; }
    return false;
}

std::string FromGeDtype(ge::DataType dt) {
    switch (dt) {
        case ge::DT_FLOAT16: return "float16";
        case ge::DT_BF16: return "bfloat16";
        case ge::DT_FLOAT: return "float32";
        case ge::DT_DOUBLE: return "float64";
        case ge::DT_INT8: return "int8";
        case ge::DT_INT16: return "int16";
        case ge::DT_INT32: return "int32";
        case ge::DT_INT64: return "int64";
        case ge::DT_UINT8: return "uint8";
        case ge::DT_UINT16: return "uint16";
        case ge::DT_UINT32: return "uint32";
        case ge::DT_UINT64: return "uint64";
        case ge::DT_BOOL: return "bool";
        default: return "unknown";
    }
}

std::string AicoreSpec(const std::string &raw) {
    if (raw.empty()) {
        return raw;
    }
    bool isDigit = raw.find_first_not_of("0123456789") == std::string::npos;
    if (!isDigit) {
        return raw;
    }
    int aic = std::stoi(raw);
    return raw + "|" + std::to_string(aic * 2);
}

// 每线程一份执行资源: 独立 stream + LoadGraph 绑定 + 输入 device 缓冲/gert::Tensor。
// Session 与已编译图全局共享 (单 Session 多 stream, 与 atb/bench_ge_latency 同构)。
struct GeContext {
    uint32_t graphId = 0;
    aclrtStream stream = nullptr;
    std::vector<gert::Tensor> devInputs;
    std::vector<void *> inPtrs;
    std::vector<gert::Tensor> devOutputs;
    std::set<void *> outAddrs;
    bool warnedAddrGrowth = false;
};

class GeRunner {
public:
    explicit GeRunner(const GeSessionOptions &opt) : opt_(opt) {}

    ~GeRunner() { Destroy(); }

    aclrtContext Context() const { return aclCtx_; }

    bool Init(const std::string &graphPath, int numGraphs) {
        std::map<ge::AscendString, ge::AscendString> globalOptions = {
            {ge::AscendString("ge.graphRunMode"),
             ge::AscendString(std::to_string(opt_.graph_run_mode).c_str())},
            {ge::AscendString("ge.exec.deviceId"),
             ge::AscendString(std::to_string(opt_.device).c_str())},
        };
        std::string aicore = AicoreSpec(opt_.aicore_num);
        if (!aicore.empty()) {
            globalOptions.emplace(ge::AscendString(ge::ir_option::AICORE_NUM),
                                  ge::AscendString(aicore.c_str()));
            std::cout << "[INFO] GEInitialize " << ge::ir_option::AICORE_NUM << "=" << aicore << std::endl;
        }
        if (opt_.profiling.enabled) {
            // GE profiling 经 GEInitialize 全局选项开启, 产出 msprof 可解析的 PROF_* 会话
            MakeDirs(opt_.profiling.outputPath);
            globalOptions.emplace(ge::AscendString(ge::OPTION_EXEC_PROFILING_MODE),
                                  ge::AscendString("1"));
            nlohmann::json profJson;
            profJson["output"] = opt_.profiling.outputPath;
            profJson["task_time"] = "on";
            profJson["runtime_api"] = "on";
            if (!opt_.profiling.aicMetrics.empty()) {
                profJson["aic_metrics"] = opt_.profiling.aicMetrics;
            }
            std::string profOpt = profJson.dump();
            globalOptions.emplace(ge::AscendString(ge::OPTION_EXEC_PROFILING_OPTIONS),
                                  ge::AscendString(profOpt.c_str()));
            std::cout << "[INFO] GE profiling enabled -> " << opt_.profiling.outputPath
                      << " (aic_metrics=" << opt_.profiling.aicMetrics << ")" << std::endl;
        }
        if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
            fprintf(stderr, "[ERROR] GEInitialize failed: %s\n", ge::GEGetErrorMsg().c_str());
            return false;
        }
        geInited_ = true;

        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            fprintf(stderr, "[ERROR] aclInit failed, ret=%d\n", ret);
            return false;
        }
        aclInited_ = true;

        std::map<ge::AscendString, ge::AscendString> sessionOptions = {
            {ge::AscendString("ge.session_device_id"),
             ge::AscendString(std::to_string(opt_.device).c_str())},
            {ge::AscendString("ge.exec.precision_mode"),
             ge::AscendString(opt_.precision_mode.c_str())},
        };
        session_ = std::make_shared<ge::Session>(sessionOptions);

        // 每线程一个图实例: LoadGraph 不支持对同一 graphId 重复加载 (ge_api.h 约束),
        // 故并发 N 路就要 N 份 AddGraph+CompileGraph (串行, 单份 ~10s — 在线后端的固有代价)。
        // CompileGraph 必须在 aclrtSetDevice **之前** (对齐 atb/bench_ge_latency 实测序列)。
        numGraphs_ = std::max(1, numGraphs);
        auto tAll = Clock::now();
        for (uint32_t gid = 1; gid <= static_cast<uint32_t>(numGraphs_); gid++) {
            auto t0 = Clock::now();
            ge::Graph graph;
            if (graph.LoadFromFile(graphPath.c_str()) != ge::GRAPH_SUCCESS) {
                fprintf(stderr, "[ERROR] Graph::LoadFromFile failed: %s\n", graphPath.c_str());
                return false;
            }
            ge::Status st = session_->AddGraph(gid, graph);
            if (st != ge::SUCCESS) {
                fprintf(stderr, "[ERROR] AddGraph(gid=%u) failed, ret=%d\n", gid, st);
                return false;
            }
            st = session_->CompileGraph(gid);
            if (st != ge::SUCCESS) {
                fprintf(stderr, "[ERROR] CompileGraph(gid=%u) failed, ret=%d\n", gid, st);
                return false;
            }
            std::cout << "[INFO] CompileGraph ok (gid=" << gid << "/" << numGraphs_ << "), cost "
                      << ElapsedMs(t0, Clock::now()) << " ms" << std::endl;
        }
        if (numGraphs_ > 1) {
            std::cout << "[INFO] " << numGraphs_ << " 份图实例编译总耗时 "
                      << ElapsedMs(tAll, Clock::now()) << " ms (" << graphPath << ")" << std::endl;
        }

        ret = aclrtSetDevice(opt_.device);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtSetDevice(%d) failed, ret=%d\n", opt_.device, ret);
            return false;
        }
        deviceSet_ = true;

        // stream 必须建在**默认 context** 上 (显式 CreateContext 会让 GE executor 报
        // "stream is not in current ctx"); 工作线程用 aclrtSetCurrentContext 共享它
        ret = aclrtGetCurrentContext(&aclCtx_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtGetCurrentContext failed, ret=%d\n", ret);
            return false;
        }
        return true;
    }

    bool LoadInputs(const std::vector<TensorPlan> &plans) {
        plans_ = plans;
        hostInputs_.clear();
        hostInputs_.reserve(plans.size());
        dtypes_.clear();
        for (size_t i = 0; i < plans.size(); i++) {
            const TensorPlan &p = plans[i];
            ge::DataType dtype;
            if (!ToGeDtype(p.dtype, dtype)) {
                fprintf(stderr, "[ERROR] 不支持的 dtype '%s' (输入 '%s')\n",
                        p.dtype.c_str(), p.logical.c_str());
                return false;
            }
            std::vector<char> host;
            if (!ReadBinFile(p.file, host)) {
                fprintf(stderr, "[ERROR] 读不到输入文件: %s\n", p.file.c_str());
                return false;
            }
            if (host.size() != p.Bytes()) {
                fprintf(stderr, "[ERROR] 输入 '%s' 文件字节数 %zu != %zu\n",
                        p.logical.c_str(), host.size(), p.Bytes());
                return false;
            }
            dtypes_.push_back(dtype);
            hostInputs_.push_back(std::move(host));
            std::cout << "[INFO] Input[" << i << "] '" << p.logical << "' node=" << p.node
                      << " shape=" << ShapeToString(p.shape) << " dtype=" << p.dtype
                      << " bytes=" << p.Bytes() << std::endl;
        }
        return true;
    }

    bool CreateContext(int tid) {
        if (FindContext(tid) != nullptr) {
            return true;                       // 幂等: sweep 各档复用同一份图绑定
        }
        if (tid + 1 > numGraphs_) {
            fprintf(stderr, "[ERROR] tid=%d 超出已编译图实例数 %d\n", tid, numGraphs_);
            return false;
        }
        auto ctx = std::make_unique<GeContext>();
        ctx->graphId = static_cast<uint32_t>(tid + 1);
        aclError ret = aclrtCreateStream(&ctx->stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtCreateStream failed (tid=%d), ret=%d\n", tid, ret);
            return false;
        }
        // 每个 stream 都要 LoadGraph 绑定一次 (图已 CompileGraph, 此处只加载)
        auto t0 = Clock::now();
        ge::Status st = session_->LoadGraph(ctx->graphId, {}, ctx->stream);
        if (st != ge::SUCCESS) {
            fprintf(stderr, "[ERROR] LoadGraph(gid=%u) failed (tid=%d), ret=%d\n",
                    ctx->graphId, tid, st);
            ReleaseContext(*ctx);
            return false;
        }
        std::cout << "[INFO] LoadGraph ok (gid=" << ctx->graphId << ", tid=" << tid << "), cost "
                  << ElapsedMs(t0, Clock::now()) << " ms" << std::endl;

        for (size_t i = 0; i < plans_.size(); i++) {
            const TensorPlan &p = plans_[i];
            size_t bytes = p.Bytes();
            void *dev = nullptr;
            ret = aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS || dev == nullptr) {
                fprintf(stderr, "[ERROR] aclrtMalloc(%zu) failed (tid=%d, input '%s'), ret=%d\n",
                        bytes, tid, p.logical.c_str(), ret);
                ReleaseContext(*ctx);
                return false;
            }
            ret = aclrtMemcpy(dev, bytes, hostInputs_[i].data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] H2D failed (tid=%d, input '%s'), ret=%d\n",
                        tid, p.logical.c_str(), ret);
                aclrtFree(dev);
                ReleaseContext(*ctx);
                return false;
            }
            ctx->inPtrs.push_back(dev);

            gert::Tensor t;
            gert::StorageShape ss;
            for (int64_t d : p.shape) {
                ss.MutableOriginShape().AppendDim(d);
                ss.MutableStorageShape().AppendDim(d);
            }
            t.GetShape() = ss;
            t.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
            t.SetDataType(dtypes_[i]);
            t.SetData(gert::TensorData(dev, nullptr, bytes, gert::kOnDeviceHbm));
            ctx->devInputs.emplace_back(std::move(t));
        }

        ctxs_[tid] = std::move(ctx);
        return true;
    }

    bool Execute(int tid) {
        GeContext *ctx = FindContext(tid);
        if (ctx == nullptr) {
            fprintf(stderr, "[ERROR] tid=%d 无执行资源\n", tid);
            return false;
        }
        ctx->devOutputs.clear();
        ge::Status st = session_->ExecuteGraphWithStreamAsync(ctx->graphId, ctx->stream,
                                                             ctx->devInputs, ctx->devOutputs);
        if (st != ge::SUCCESS) {
            fprintf(stderr, "[ERROR] ExecuteGraphWithStreamAsync failed (tid=%d), ret=%d\n", tid, st);
            return false;
        }
        aclError ret = aclrtSynchronizeStream(ctx->stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtSynchronizeStream failed (tid=%d), ret=%d\n", tid, ret);
            return false;
        }
        for (auto &t : ctx->devOutputs) {
            if (t.GetAddr() != nullptr) {
                ctx->outAddrs.insert(t.GetAddr());
            }
        }
        // 实测 GE 每轮复用同一输出地址 (set 去重后恒为 1); 若将来改成每轮新分配, 先告警
        if (!ctx->warnedAddrGrowth && ctx->outAddrs.size() > 64) {
            ctx->warnedAddrGrowth = true;
            fprintf(stderr, "[WARN] tid=%d 已累计 %zu 个不同输出 device 地址 (GE 未复用缓冲?), "
                            "HBM 会随请求数线性增长\n", tid, ctx->outAddrs.size());
        }
        return true;
    }

    bool CollectOutputs(int tid, const IoSpec &spec, std::vector<HostTensor> &outputs) {
        GeContext *ctx = FindContext(tid);
        if (ctx == nullptr) {
            fprintf(stderr, "[ERROR] tid=%d 无执行资源, 无法取输出\n", tid);
            return false;
        }
        for (size_t i = 0; i < ctx->devOutputs.size(); i++) {
            const gert::Tensor &t = ctx->devOutputs[i];
            size_t bytes = t.GetSize();
            const auto &shape = t.GetShape().GetStorageShape();

            HostTensor out;
            out.dtype = FromGeDtype(t.GetDataType());
            out.logical = i < spec.outputs.size() && !spec.outputs[i].logical.empty()
                              ? spec.outputs[i].logical
                              : "output_" + std::to_string(i);
            for (size_t d = 0; d < shape.GetDimNum(); d++) {
                out.shape.push_back(shape.GetDim(d));
            }
            out.data.resize(bytes);
            if (bytes > 0 && t.GetAddr() != nullptr) {
                aclError ret = aclrtMemcpy(out.data.data(), bytes, t.GetAddr(), bytes,
                                           ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_SUCCESS) {
                    fprintf(stderr, "[ERROR] D2H failed for output[%zu], ret=%d\n", i, ret);
                    return false;
                }
            }
            std::cout << "[INFO] Output[" << i << "] '" << out.logical << "' shape="
                      << ShapeToString(out.shape) << " dtype=" << out.dtype
                      << " bytes=" << bytes << std::endl;
            outputs.push_back(std::move(out));
        }
        return true;
    }

    // 不逐份释放: LoadGraph 对同一 graphId 只能调一次, sweep 各档必须复用图绑定;
    // 全部资源在 Destroy() 里统一释放。
    void ReleaseContext(int) {}

    void Destroy() {
        for (auto &kv : ctxs_) {
            ReleaseContext(*kv.second);
        }
        ctxs_.clear();
        hostInputs_.clear();
        session_.reset();
        if (deviceSet_) {
            aclrtResetDevice(opt_.device);
            deviceSet_ = false;
        }
        if (geInited_) {
            ge::GEFinalize();
            geInited_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

private:
    static void ReleaseContext(GeContext &ctx) {
        for (void *p : ctx.inPtrs) {
            if (p != nullptr) {
                aclrtFree(p);
            }
        }
        ctx.inPtrs.clear();
        ctx.devInputs.clear();
        ctx.devOutputs.clear();
        for (void *p : ctx.outAddrs) {
            if (p != nullptr) {
                aclrtFree(p);
            }
        }
        ctx.outAddrs.clear();
        if (ctx.stream != nullptr) {
            aclrtDestroyStream(ctx.stream);
            ctx.stream = nullptr;
        }
    }

    GeContext *FindContext(int tid) {
        auto it = ctxs_.find(tid);
        return it == ctxs_.end() ? nullptr : it->second.get();
    }

    GeSessionOptions opt_;
    int numGraphs_ = 1;
    bool geInited_ = false;
    bool aclInited_ = false;
    bool deviceSet_ = false;
    std::shared_ptr<ge::Session> session_;
    aclrtContext aclCtx_ = nullptr;
    std::vector<TensorPlan> plans_;
    std::vector<std::vector<char>> hostInputs_;
    std::vector<ge::DataType> dtypes_;
    std::map<int, std::unique_ptr<GeContext>> ctxs_;
};

}  // namespace

bool RunGeSessionBackend(const Manifest &manifest, const IoSpec &spec,
                         const std::vector<TensorPlan> &inputs, const GeSessionOptions &opt,
                         std::vector<HostTensor> &outputs) {
    if (manifest.graph_path.empty()) {
        fprintf(stderr, "[ERROR] manifest 无 graph_path (backend=ge_session 需要 AIR/ONNX)\n");
        return false;
    }
    std::string graphPath = manifest.Resolve(manifest.graph_path);
    if (graphPath.size() >= 5 &&
        graphPath.compare(graphPath.size() - 5, 5, ".onnx") == 0) {
        fprintf(stderr, "[ERROR] GeSession 在线后端只吃 GE 图 (.air/.pbtxt), 收到 ONNX: %s\n"
                        "        ONNX 请先经 ATC (--framework=5) 转 OM 走 om_acl 后端 (docs §1/§15)\n",
                graphPath.c_str());
        return false;
    }

    int numGraphs = opt.bench.sweep.empty() ? std::max(1, opt.bench.threads)
                                           : *std::max_element(opt.bench.sweep.begin(),
                                                               opt.bench.sweep.end());
    GeRunner runner(opt);
    if (!runner.Init(graphPath, numGraphs)) {
        return false;
    }
    if (!runner.LoadInputs(inputs)) {
        return false;
    }

    ThreadResources res;
    res.setup = [&runner](int tid) { return runner.CreateContext(tid); };
    res.step = [&runner](int tid) { return runner.Execute(tid); };
    res.release = [&runner](int tid) { runner.ReleaseContext(tid); };
    // 工作线程共享主线程的默认 context (CANN 无 reset 接口; 线程退出即释放)
    res.threadEnter = [&runner](int) { aclrtSetCurrentContext(runner.Context()); };

    if (IsThroughputMode(opt.bench)) {
        std::vector<ThroughputStats> all;
        if (!BenchSweep(opt.bench, res, all)) {
            return false;
        }
        for (const auto &s : all) {
            PrintThroughputStats("GeSession execute+sync throughput", s);
        }
        if (!runner.CreateContext(0) || !runner.Execute(0)) {
            return false;
        }
        bool ok = runner.CollectOutputs(0, spec, outputs);
        runner.ReleaseContext(0);
        return ok;
    }

    if (!runner.CreateContext(0)) {
        return false;
    }
    auto t0 = Clock::now();
    if (!runner.Execute(0)) {
        return false;
    }
    std::cout << "[INFO] First execute (incl. shape specialization), cost "
              << ElapsedMs(t0, Clock::now()) << " ms" << std::endl;

    BenchStats stats;
    if (!BenchRun(opt.bench, [&runner]() { return runner.Execute(0); }, stats)) {
        return false;
    }
    PrintBenchStats("GeSession execute+sync", stats);
    bool ok = runner.CollectOutputs(0, spec, outputs);
    runner.ReleaseContext(0);
    return ok;
}

}  // namespace ge_runtime
