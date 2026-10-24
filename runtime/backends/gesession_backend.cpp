#include "gesession_backend.h"
#include "graph_loader.h"

#include <acl/acl.h>
#include <ge/ge_api.h>
#include <ge/ge_graph_compile_summary.h>
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
#include <stdexcept>
#include <string>
#include <vector>

#include "backend_common.h"

namespace ge_runtime {
namespace {

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
    auto separator = raw.find('|');
    if (separator == std::string::npos || separator == 0 || separator + 1 == raw.size() ||
        raw.find('|',separator+1) != std::string::npos ||
        raw.substr(0,separator).find_first_not_of("0123456789") != std::string::npos ||
        raw.substr(separator+1).find_first_not_of("0123456789") != std::string::npos ||
        std::stoul(raw.substr(0,separator)) == 0 || std::stoul(raw.substr(separator+1)) == 0) {
        throw std::runtime_error("aicore_num requires an explicit positive AIC|AIV pair");
    }
    return raw;
}

// 每线程一份执行资源: 独立 stream + LoadGraph 绑定 + 输入 device 缓冲/gert::Tensor。
// Session 与已编译图全局共享 (单 Session 多 stream, 与 atb/bench_ge_latency 同构)。
struct GeContext {
    uint32_t graphId = 0;
    aclrtStream stream = nullptr;
    std::vector<gert::Tensor> devInputs;
    std::vector<void *> inPtrs;
    std::vector<size_t> inSizes;      // 每输入分配字节 (池模式 = pool 内最大)
    std::vector<gert::Tensor> devOutputs;
    std::vector<void *> staticOutputPtrs;
    std::set<void *> outAddrs;
    bool warnedAddrGrowth = false;
    double loadMs = 0.0;              // 本实例 LoadGraph 耗时 (CompileGraph 在 Init 里, 另报)
};

struct CompiledOutputs {
    bool callerAllocated = false;
    std::vector<TensorPlan> plans;
    std::vector<ge::DataType> dtypes;
};

class GeRunner {
public:
    GeRunner(const GeSessionOptions &opt, const IoSpec &spec) : opt_(opt), spec_(spec) {}

    ~GeRunner() { Destroy(); }

    aclrtContext Context() const { return aclCtx_; }

    bool Init(const std::string &graphPath, const std::string &format, int numGraphs) {
        std::map<ge::AscendString, ge::AscendString> globalOptions = {
            {ge::AscendString("ge.graphRunMode"),
             ge::AscendString(std::to_string(opt_.graph_run_mode).c_str())},
            {ge::AscendString("ge.exec.deviceId"),
             ge::AscendString(std::to_string(opt_.device).c_str())},
        };
        std::string aicore = AicoreSpec(opt_.aicore_num);
        std::cout << "[INFO] GEInitialize ge.graphRunMode=" << opt_.graph_run_mode << std::endl;
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
            if (!LoadGraphFile(graphPath,format,graph)) {
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
                fprintf(stderr, "[ERROR] CompileGraph(gid=%u) failed, ret=%d: %s\n",
                        gid, st, ge::GEGetErrorMsg().c_str());
                return false;
            }
            auto summary = session_->GetCompiledGraphSummary(gid);
            if (summary == nullptr) {
                fprintf(stderr, "[ERROR] GetCompiledGraphSummary(gid=%u) failed\n", gid);
                return false;
            }
            CompiledOutputs outputs;
            outputs.callerAllocated = summary->IsStatic();
            if (outputs.callerAllocated) {
                std::vector<ge::Shape> shapes;
                if (summary->GetOutputShapes(shapes) != ge::SUCCESS ||
                    summary->GetOutputDtypes(outputs.dtypes) != ge::SUCCESS ||
                    shapes.size() != spec_.outputs.size() || shapes.size() != outputs.dtypes.size()) {
                    fprintf(stderr, "[ERROR] Invalid compiled output metadata (gid=%u)\n", gid);
                    return false;
                }
                for (size_t i = 0; i < shapes.size(); ++i) {
                    TensorPlan plan;
                    plan.logical = spec_.outputs[i].logical;
                    plan.dtype = FromGeDtype(outputs.dtypes[i]);
                    plan.format = "ND";
                    plan.shape = shapes[i].GetDims();
                    if (plan.dtype != spec_.outputs[i].dtype ||
                        plan.shape.size() != spec_.outputs[i].shape.size()) {
                        fprintf(stderr, "[ERROR] Compiled output contract mismatch: %s\n", plan.logical.c_str());
                        return false;
                    }
                    for (size_t dim = 0; dim < plan.shape.size(); ++dim) {
                        auto expected = spec_.outputs[i].shape[dim];
                        if (expected != -1 && expected != plan.shape[dim]) {
                            fprintf(stderr, "[ERROR] Compiled output shape mismatch: %s\n", plan.logical.c_str());
                            return false;
                        }
                    }
                    plan.Bytes();  // Reject unresolved shapes or byte-size overflow before allocation.
                    outputs.plans.push_back(std::move(plan));
                }
            }
            std::cout << "[INFO] Graph " << gid << " output allocation="
                      << (outputs.callerAllocated ? "caller (static compiled)" : "GE (dynamic compiled)")
                      << std::endl;
            compiledOutputs_.emplace(gid, std::move(outputs));
            std::cout << "[INFO] CompileGraph ok (gid=" << gid << "/" << numGraphs_ << "), cost "
                      << ElapsedMs(t0, Clock::now()) << " ms" << std::endl;
        }
        compileMs_ = ElapsedMs(tAll, Clock::now());
        if (numGraphs_ > 1) {
            std::cout << "[INFO] " << numGraphs_ << " 份图实例编译总耗时 "
                      << compileMs_ << " ms (" << graphPath << ")" << std::endl;
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


    bool CreateContext(int tid) {
        if (FindContext(tid) != nullptr) {
            return true;                       // 幂等: 吞吐跑完后取输出会再调一次 (tid=0)
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

        // 按池内最大 shape 预分配，每请求 H2D 并重建 gert::Tensor。
        size_t nInputs = pool_->maxBytes.size();
        for (size_t i = 0; i < nInputs; i++) {
            size_t bytes = pool_->maxBytes[i];
            void *dev = nullptr;
            ret = aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS || dev == nullptr) {
                fprintf(stderr, "[ERROR] aclrtMalloc(%zu) failed (tid=%d, input %zu), ret=%d\n",
                        bytes, tid, i, ret);
                ReleaseContext(*ctx);
                return false;
            }
            // push_back 起 dev 的所有权归 ctx->inPtrs — 失败路径只调 ReleaseContext,
            // 再手动 aclrtFree(dev) 就是二次释放
            ctx->inPtrs.push_back(dev);
            ctx->inSizes.push_back(bytes);

        }

        const auto &outputs = compiledOutputs_.at(ctx->graphId);
        if (outputs.callerAllocated) {
            if (pool_->maxOutputBytes.size() != outputs.plans.size()) {
                fprintf(stderr, "[ERROR] Output capacity count differs from compiled outputs\n");
                ReleaseContext(*ctx);
                return false;
            }
            for (size_t i = 0; i < outputs.plans.size(); ++i) {
                const auto &plan = outputs.plans[i];
                size_t bytes = plan.Bytes();
                if (bytes > pool_->maxOutputBytes[i]) {
                    fprintf(stderr, "[ERROR] Compiled output '%s' requires %zu bytes > declared capacity %zu\n",
                            plan.logical.c_str(), bytes, pool_->maxOutputBytes[i]);
                    ReleaseContext(*ctx);
                    return false;
                }
                void *dev = nullptr;
                ret = aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS || dev == nullptr) {
                    fprintf(stderr, "[ERROR] Output aclrtMalloc(%zu) failed (tid=%d, output=%zu), ret=%d\n",
                            bytes, tid, i, ret);
                    ReleaseContext(*ctx);
                    return false;
                }
                ctx->staticOutputPtrs.push_back(dev);
                ctx->outAddrs.insert(dev);
                std::cout << "[INFO] Allocated output[" << i << "] '" << plan.logical << "' shape="
                          << ShapeToString(plan.shape) << " dtype=" << plan.dtype << " bytes=" << bytes
                          << " (tid=" << tid << ")" << std::endl;
            }
        }

        ctx->loadMs = ElapsedMs(t0, Clock::now());
        ctxs_[tid] = std::move(ctx);
        return true;
    }

    // 池模式: 每请求 H2D 实际字节 + 按该请求 shape 重建 gert::Tensor (device 缓冲复用)
    bool SetInputs(int tid, const Request &req, PhaseTime &phase) {
        GeContext *ctx = FindContext(tid);
        if (ctx == nullptr) {
            fprintf(stderr, "[ERROR] tid=%d 无执行资源\n", tid);
            return false;
        }
        if (req.plans.size() != ctx->inPtrs.size()) {
            fprintf(stderr, "[ERROR] 请求 %s 有 %zu 个输入, 实例有 %zu 个缓冲\n",
                    req.name.c_str(), req.plans.size(), ctx->inPtrs.size());
            return false;
        }
        auto h2dBegin = Clock::now();
        for (size_t i = 0; i < req.plans.size(); i++) {
            size_t bytes = req.plans[i].Bytes();
            if (bytes > ctx->inSizes[i]) {
                fprintf(stderr, "[ERROR] 请求 %s 输入 %zu 需要 %zu 字节 > 缓冲 %zu\n",
                        req.name.c_str(), i, bytes, ctx->inSizes[i]);
                return false;
            }
            aclError ret = aclrtMemcpy(ctx->inPtrs[i], ctx->inSizes[i], req.data[i].data(), bytes,
                                       ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] H2D failed (tid=%d, req=%s, input %zu), ret=%d\n",
                        tid, req.name.c_str(), i, ret);
                return false;
            }
        }
        phase.h2dMs = ElapsedMs(h2dBegin, Clock::now());

        auto descBegin = Clock::now();
        ctx->devInputs.clear();
        for (size_t i = 0; i < req.plans.size(); i++) {
            // TensorData 的 size 给**分配大小**(与 atb/bench_ge_latency 一致), shape 给该请求的实际 shape
            MakeTensor(ctx->devInputs, ctx->inPtrs[i], ctx->inSizes[i], req.plans[i].shape, dtypes_[i]);
        }
        phase.descMs = ElapsedMs(descBegin, Clock::now());
        return true;
    }

    double LoadMs(int tid) const {
        auto it = ctxs_.find(tid);
        return it == ctxs_.end() ? 0.0 : it->second->loadMs;
    }

    bool UsePool(const RequestPool *pool) {
        pool_ = pool;
        // dtype 从首个请求的 plan 取，各请求 dtype 一致。
        if (dtypes_.empty() && !pool->requests.empty()) {
            for (const auto &plan : pool->requests[0].plans) {
                ge::DataType dt;
                if (!ToGeDtype(plan.dtype, dt)) {
                    // 不检查就静默退成 DT_FLOAT: 字节宽错了, H2D 数据被按 float 解释
                    fprintf(stderr, "[ERROR] 不支持的 dtype '%s' (请求池输入 '%s')\n",
                            plan.dtype.c_str(), plan.logical.c_str());
                    return false;
                }
                dtypes_.push_back(dt);
            }
        }
        return true;
    }

    bool Execute(int tid) {
        GeContext *ctx = FindContext(tid);
        if (ctx == nullptr) {
            fprintf(stderr, "[ERROR] tid=%d 无执行资源\n", tid);
            return false;
        }
        ctx->devOutputs.clear();
        const auto &outputs = compiledOutputs_.at(ctx->graphId);
        if (outputs.callerAllocated) {
            for (size_t i = 0; i < outputs.plans.size(); ++i) {
                const auto &plan = outputs.plans[i];
                MakeTensor(ctx->devOutputs, ctx->staticOutputPtrs[i], plan.Bytes(), plan.shape, outputs.dtypes[i]);
            }
        }
        ge::Status st = session_->ExecuteGraphWithStreamAsync(ctx->graphId, ctx->stream,
                                                             ctx->devInputs, ctx->devOutputs);
        if (st != ge::SUCCESS) {
            fprintf(stderr, "[ERROR] ExecuteGraphWithStreamAsync failed (tid=%d), ret=%d: %s\n",
                    tid, st, ge::GEGetErrorMsg().c_str());
            return false;
        }
        aclError ret = aclrtSynchronizeStream(ctx->stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtSynchronizeStream failed (tid=%d), ret=%d\n", tid, ret);
            return false;
        }
        if (ctx->devOutputs.size() != spec_.outputs.size()) {
            fprintf(stderr, "[ERROR] Runtime output count differs from graph IO\n");
            return false;
        }
        for (size_t i = 0; i < ctx->devOutputs.size(); ++i) {
            auto &t = ctx->devOutputs[i];
            if (outputs.callerAllocated && t.GetAddr() != ctx->staticOutputPtrs[i]) {
                fprintf(stderr, "[ERROR] GE replaced caller-owned output buffer (output=%zu)\n", i);
                return false;
            }
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
            out.instance = tid;
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

    // 不逐份释放: LoadGraph 对同一 graphId 只能调一次, 取输出时还要复用同一份图绑定;
    // 全部资源在 Destroy() 里统一释放。

    void Destroy() {
        for (auto &kv : ctxs_) {
            ReleaseContext(*kv.second);
        }
        ctxs_.clear();
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
        ctx.staticOutputPtrs.clear();
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

    static void MakeTensor(std::vector<gert::Tensor> &out, void *dev, size_t allocBytes,
                           const std::vector<int64_t> &shape, ge::DataType dtype) {
        gert::Tensor t;
        gert::StorageShape ss;
        for (int64_t d : shape) {
            ss.MutableOriginShape().AppendDim(d);
            ss.MutableStorageShape().AppendDim(d);
        }
        t.GetShape() = ss;
        t.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
        t.SetDataType(dtype);
        t.SetData(gert::TensorData(dev, nullptr, allocBytes, gert::kOnDeviceHbm));
        out.emplace_back(std::move(t));
    }

    GeContext *FindContext(int tid) {
        auto it = ctxs_.find(tid);
        return it == ctxs_.end() ? nullptr : it->second.get();
    }

    GeSessionOptions opt_;
    const IoSpec &spec_;
    int numGraphs_ = 1;
    double compileMs_ = 0.0;
    bool geInited_ = false;
    bool aclInited_ = false;
    bool deviceSet_ = false;
    std::shared_ptr<ge::Session> session_;
    aclrtContext aclCtx_ = nullptr;
    std::vector<ge::DataType> dtypes_;
    const RequestPool *pool_ = nullptr;
    std::map<int, std::unique_ptr<GeContext>> ctxs_;
    std::map<uint32_t, CompiledOutputs> compiledOutputs_;
};

}  // namespace

bool RunGeSessionBackend(const ModelGraph &graph, const IoSpec &spec,
                   const GeSessionOptions &opt, PerfResult *perf) {
    if(!opt.pool || opt.pool->Empty()) { fprintf(stderr,"[ERROR] request pool is required\n");return false; }
    if(graph.graph_path.empty()) { fprintf(stderr,"[ERROR] missing graph path\n");return false; }
    GeRunner runner(opt, spec);
    if(!runner.Init(graph.graph_path,graph.graph_format,std::max(1,opt.bench.threads))) return false;
    // ---- 变长负载: 请求池回放 (单 Session 多图, 每实例一份 graphId+stream) ----
    if (!runner.UsePool(opt.pool)) {
        return false;
    }
    PoolResources pres;
    pres.setup = [&runner](int tid) { return runner.CreateContext(tid); };
    pres.setInputs = [&runner](int tid, const Request &req, PhaseTime &ph) {
        return runner.SetInputs(tid, req, ph);
    };
    pres.execute = [&runner](int tid) { return runner.Execute(tid); };
    pres.release = [&runner](int) {};      // 图绑定不可重复建立, 统一在 Destroy 释放
    pres.threadEnter = [&runner](int) { aclrtSetCurrentContext(runner.Context()); };
    pres.loadMs = [&runner](int tid) {
        return runner.LoadMs(tid);   // 该实例的 LoadGraph + 全部图编译
    };
    pres.hbmUsedMb = []() { return HbmUsedMb(); };

    if (opt.capture) return CapturePool(opt.bench.threads, *opt.pool, pres, [&](int tid, const Request &request) {
        std::vector<HostTensor> tensors;
        return runner.CollectOutputs(tid, spec, tensors) && opt.capture(tid, request, tensors);
    });

    PerfResult result;
    bool ok = BenchPool(opt.bench.threads, opt.bench.requests, opt.bench.warmup, opt.seed,
                        *opt.pool, pres, result);
    if (ok) {
        PrintPerfResult("GeSession 变长负载 (请求池回放)", result);
    }
    if (perf != nullptr) {
        *perf = std::move(result);
    }
    return ok;
}

} // namespace ge_runtime
