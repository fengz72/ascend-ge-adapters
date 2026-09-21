#include "acl_backend.h"

#include <acl/acl.h>

#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ge_runtime {
namespace {

aclDataType ToAclDtype(const std::string &s) {
    if (s == "float16") return ACL_FLOAT16;
    if (s == "bfloat16") return ACL_BF16;
    if (s == "float32" || s == "float") return ACL_FLOAT;
    if (s == "float64" || s == "double") return ACL_DOUBLE;
    if (s == "int8") return ACL_INT8;
    if (s == "int16") return ACL_INT16;
    if (s == "int32") return ACL_INT32;
    if (s == "int64") return ACL_INT64;
    if (s == "uint8") return ACL_UINT8;
    if (s == "uint16") return ACL_UINT16;
    if (s == "uint32") return ACL_UINT32;
    if (s == "uint64") return ACL_UINT64;
    if (s == "bool") return ACL_BOOL;
    return ACL_DT_UNDEFINED;
}

std::string FromAclDtype(aclDataType dt) {
    switch (dt) {
        case ACL_FLOAT16: return "float16";
        case ACL_BF16: return "bfloat16";
        case ACL_FLOAT: return "float32";
        case ACL_DOUBLE: return "float64";
        case ACL_INT8: return "int8";
        case ACL_INT16: return "int16";
        case ACL_INT32: return "int32";
        case ACL_INT64: return "int64";
        case ACL_UINT8: return "uint8";
        case ACL_UINT16: return "uint16";
        case ACL_UINT32: return "uint32";
        case ACL_UINT64: return "uint64";
        case ACL_BOOL: return "bool";
        default: return "unknown";
    }
}

aclFormat ToAclFormat(const std::string &s) {
    if (s == "NCHW") return ACL_FORMAT_NCHW;
    if (s == "NHWC") return ACL_FORMAT_NHWC;
    if (s == "ND") return ACL_FORMAT_ND;
    if (s == "NC1HWC0") return ACL_FORMAT_NC1HWC0;
    if (s == "NCDHW") return ACL_FORMAT_NCDHW;
    if (s == "NDC1HWC0") return ACL_FORMAT_NDC1HWC0;
    if (s == "FRACTAL_Z") return ACL_FORMAT_FRACTAL_Z;
    if (s == "FRACTAL_NZ") return ACL_FORMAT_FRACTAL_NZ;
    return ACL_FORMAT_UNDEFINED;
}

// 每线程一份执行资源 (stream + input/output dataset + device 缓冲); modelId/desc 全局共享 —
// ACL 多线程推理的标准形态 (每线程独立 dataset/stream, 共享已加载模型)。
struct AclContext {
    bool valid = false;
    bool modelLoaded = false;
    uint32_t modelId = 0;
    aclmdlDesc *desc = nullptr;
    aclrtStream stream = nullptr;
    aclmdlDataset *in = nullptr;
    aclmdlDataset *out = nullptr;
    std::vector<void *> inBufs;
    std::vector<void *> outBufs;
    std::vector<size_t> outSizes;
};

class AclRunner {
public:
    explicit AclRunner(const AclOptions &opt) : opt_(opt) {}

    ~AclRunner() { Destroy(); }

    aclrtContext Context() const { return aclCtx_; }

    bool Init(const std::string &omPath) {
        // dump/profiling 经 aclInit(configPath) 生效 (acl.json 由 main 生成, 见 acl_json.cpp)
        const char *cfgPath = opt_.aclConfigPath.empty() ? nullptr : opt_.aclConfigPath.c_str();
        if (cfgPath != nullptr) {
            std::cout << "[INFO] aclInit with config: " << cfgPath << std::endl;
        }
        aclError ret = aclInit(cfgPath);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            fprintf(stderr, "[ERROR] aclInit failed, ret=%d\n", ret);
            return false;
        }
        aclInited_ = true;

        ret = aclrtSetDevice(opt_.device);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtSetDevice(%d) failed, ret=%d\n", opt_.device, ret);
            return false;
        }
        deviceSet_ = true;

        // 工作线程要 aclrtSetCurrentContext 才能调 ACL API; 用**默认** context
        // (GE 在线路径要求默认 context, 两后端统一口径)
        ret = aclrtGetCurrentContext(&aclCtx_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtGetCurrentContext failed, ret=%d\n", ret);
            return false;
        }

        // 不在此加载模型: ACL 多线程要求每线程独立 modelId, 元数据 (输入/输出个数与名字)
        // 由首个 CreateContext 的 desc 顺带打印 — 省一次 ~2s 的重复加载与一份 HBM。
        omPath_ = omPath;
        return true;
    }

    // 读盘 + 校验一次, host 数据缓存供各线程 H2D 复用 (避免每线程重复读盘)
    bool LoadInputs(const std::vector<TensorPlan> &plans, bool dynamic) {
        plans_ = plans;
        dynamic_ = dynamic;
        hostInputs_.clear();
        hostInputs_.reserve(plans.size());
        for (size_t i = 0; i < plans.size(); i++) {
            const TensorPlan &p = plans[i];
            if (ToAclDtype(p.dtype) == ACL_DT_UNDEFINED) {
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
            hostInputs_.push_back(std::move(host));
            std::cout << "[INFO] Input[" << i << "] '" << p.logical << "' node=" << p.node
                      << " shape=" << ShapeToString(p.shape) << " dtype=" << p.dtype
                      << " bytes=" << p.Bytes() << std::endl;
        }
        return true;
    }

    bool CreateContext(int tid) {
        AclContext ctx;
        aclError ret = aclmdlLoadFromFile(omPath_.c_str(), &ctx.modelId);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclmdlLoadFromFile failed (tid=%d), ret=%d\n", tid, ret);
            return false;
        }
        ctx.modelLoaded = true;
        ctx.desc = aclmdlCreateDesc();
        if (ctx.desc == nullptr || aclmdlGetDesc(ctx.desc, ctx.modelId) != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclmdlGetDesc failed (tid=%d)\n", tid);
            ReleaseContext(ctx);
            return false;
        }
        if (inputCount_ == 0) {                     // 首个 context: 取元数据并校验输入个数
            inputCount_ = aclmdlGetNumInputs(ctx.desc);
            outputCount_ = aclmdlGetNumOutputs(ctx.desc);
            std::cout << "[INFO] OM loaded: " << omPath_ << " (inputs=" << inputCount_
                      << ", outputs=" << outputCount_ << ")" << std::endl;
            for (size_t i = 0; i < inputCount_; i++) {
                const char *name = aclmdlGetInputNameByIndex(ctx.desc, i);
                size_t size = aclmdlGetInputSizeByIndex(ctx.desc, i);
                std::cout << "[INFO]   OM input[" << i << "] name=" << (name ? name : "?")
                          << " staticSize=" << size << (size == 0 ? " (dynamic)" : "") << std::endl;
            }
            for (size_t i = 0; i < outputCount_; i++) {
                const char *name = aclmdlGetOutputNameByIndex(ctx.desc, i);
                size_t size = aclmdlGetOutputSizeByIndex(ctx.desc, i);
                std::cout << "[INFO]   OM output[" << i << "] name=" << (name ? name : "?")
                          << " staticSize=" << size << (size == 0 ? " (dynamic)" : "") << std::endl;
            }
            if (plans_.size() != inputCount_) {
                fprintf(stderr, "[ERROR] io_spec/bundle 给出 %zu 个输入, 但 OM 有 %zu 个\n",
                        plans_.size(), inputCount_);
                ReleaseContext(ctx);
                return false;
            }
        }
        ret = aclrtCreateStream(&ctx.stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtCreateStream failed (tid=%d), ret=%d\n", tid, ret);
            return false;
        }
        ctx.in = aclmdlCreateDataset();
        ctx.out = aclmdlCreateDataset();
        if (ctx.in == nullptr || ctx.out == nullptr) {
            fprintf(stderr, "[ERROR] aclmdlCreateDataset failed (tid=%d)\n", tid);
            ReleaseContext(ctx);
            return false;
        }

        for (size_t i = 0; i < plans_.size(); i++) {
            const TensorPlan &p = plans_[i];
            size_t bytes = p.Bytes();
            void *dev = nullptr;
            ret = aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS || dev == nullptr) {
                fprintf(stderr, "[ERROR] aclrtMalloc(%zu) failed for input '%s' (tid=%d), ret=%d\n",
                        bytes, p.logical.c_str(), tid, ret);
                ReleaseContext(ctx);
                return false;
            }
            ret = aclrtMemcpy(dev, bytes, hostInputs_[i].data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] H2D failed for input '%s' (tid=%d), ret=%d\n",
                        p.logical.c_str(), tid, ret);
                aclrtFree(dev);
                ReleaseContext(ctx);
                return false;
            }
            aclDataBuffer *buf = aclCreateDataBuffer(dev, bytes);
            if (buf == nullptr || aclmdlAddDatasetBuffer(ctx.in, buf) != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] add input dataset buffer failed (tid=%d, input %zu)\n", tid, i);
                if (buf != nullptr) {
                    aclDestroyDataBuffer(buf);
                }
                aclrtFree(dev);
                ReleaseContext(ctx);
                return false;
            }
            ctx.inBufs.push_back(dev);

            if (dynamic_) {
                aclTensorDesc *td = aclCreateTensorDesc(ToAclDtype(p.dtype),
                                                        static_cast<int32_t>(p.shape.size()),
                                                        p.shape.data(), ToAclFormat(p.format));
                if (td == nullptr) {
                    fprintf(stderr, "[ERROR] aclCreateTensorDesc failed (tid=%d, input '%s')\n",
                            tid, p.logical.c_str());
                    ReleaseContext(ctx);
                    return false;
                }
                ret = aclmdlSetDatasetTensorDesc(ctx.in, td, i);
                aclDestroyTensorDesc(td);
                if (ret != ACL_SUCCESS) {
                    fprintf(stderr, "[ERROR] aclmdlSetDatasetTensorDesc failed (tid=%d, input %zu), ret=%d\n",
                            tid, i, ret);
                    ReleaseContext(ctx);
                    return false;
                }
            }
        }

        for (size_t i = 0; i < outputCount_; i++) {
            size_t bytes = aclmdlGetOutputSizeByIndex(ctx.desc, i);
            if (bytes == 0) {
                bytes = opt_.output_reserve;
            }
            void *dev = nullptr;
            ret = aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS || dev == nullptr) {
                fprintf(stderr, "[ERROR] aclrtMalloc(%zu) failed for output[%zu] (tid=%d), ret=%d\n",
                        bytes, i, tid, ret);
                ReleaseContext(ctx);
                return false;
            }
            aclDataBuffer *buf = aclCreateDataBuffer(dev, bytes);
            if (buf == nullptr || aclmdlAddDatasetBuffer(ctx.out, buf) != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] add output dataset buffer failed (tid=%d, output %zu)\n", tid, i);
                if (buf != nullptr) {
                    aclDestroyDataBuffer(buf);
                }
                aclrtFree(dev);
                ReleaseContext(ctx);
                return false;
            }
            ctx.outBufs.push_back(dev);
            ctx.outSizes.push_back(bytes);
        }

        ctx.valid = true;
        StoreContext(tid, ctx);
        return true;
    }

    bool Execute(int tid) {
        AclContext *ctx = FindContext(tid);
        if (ctx == nullptr) {
            fprintf(stderr, "[ERROR] tid=%d 无执行资源\n", tid);
            return false;
        }
        aclError ret = aclmdlExecuteAsync(ctx->modelId, ctx->in, ctx->out, ctx->stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclmdlExecuteAsync failed (tid=%d), ret=%d\n", tid, ret);
            return false;
        }
        ret = aclrtSynchronizeStream(ctx->stream);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtSynchronizeStream failed (tid=%d), ret=%d\n", tid, ret);
            return false;
        }
        return true;
    }

    bool CollectOutputs(int tid, const IoSpec &spec, std::vector<HostTensor> &outputs) {
        AclContext *ctx = FindContext(tid);
        if (ctx == nullptr) {
            fprintf(stderr, "[ERROR] tid=%d 无执行资源, 无法取输出\n", tid);
            return false;
        }
        for (size_t i = 0; i < outputCount_; i++) {
            size_t bytes = 0;
            std::vector<int64_t> shape;
            aclDataType dtype = ACL_DT_UNDEFINED;

            // 动态图: 执行后 dataset 上挂实际 desc; 静态图: dataset 无 desc → 取 model desc
            aclTensorDesc *td = aclmdlGetDatasetTensorDesc(ctx->out, i);
            if (td != nullptr) {
                bytes = aclGetTensorDescSize(td);
                dtype = aclGetTensorDescType(td);
                int32_t ndim = aclGetTensorDescNumDims(td);
                for (int32_t d = 0; d < ndim; d++) {
                    int64_t dim = 0;
                    aclGetTensorDescDimV2(td, d, &dim);
                    shape.push_back(dim);
                }
            } else {
                bytes = aclmdlGetOutputSizeByIndex(ctx->desc, i);
                dtype = aclmdlGetOutputDataType(ctx->desc, i);
                aclmdlIODims dims;
                if (aclmdlGetOutputDims(ctx->desc, i, &dims) == ACL_SUCCESS) {
                    for (int32_t d = 0; d < dims.dimCount; d++) {
                        shape.push_back(dims.dims[d]);
                    }
                }
            }

            if (i < ctx->outSizes.size() && bytes > ctx->outSizes[i]) {
                fprintf(stderr, "[ERROR] output[%zu] 实际需要 %zu 字节 > 预留 %zu 字节 "
                                "(--output_reserve %zu MB); 输出已可能越界, 结果不可信\n",
                        i, bytes, ctx->outSizes[i], ctx->outSizes[i] / (1024 * 1024));
                return false;
            }

            HostTensor t;
            t.dtype = FromAclDtype(dtype);
            t.logical = i < spec.outputs.size() && !spec.outputs[i].logical.empty()
                            ? spec.outputs[i].logical
                            : "output_" + std::to_string(i);
            t.shape = shape;
            t.data.resize(bytes);

            aclDataBuffer *buf = aclmdlGetDatasetBuffer(ctx->out, i);
            void *dev = aclGetDataBufferAddr(buf);
            if (bytes > 0 && dev != nullptr) {
                aclError ret = aclrtMemcpy(t.data.data(), bytes, dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_SUCCESS) {
                    fprintf(stderr, "[ERROR] D2H failed for output[%zu], ret=%d\n", i, ret);
                    return false;
                }
            }
            std::cout << "[INFO] Output[" << i << "] '" << t.logical << "' shape="
                      << ShapeToString(t.shape) << " dtype=" << t.dtype << " bytes=" << bytes
                      << std::endl;
            outputs.push_back(std::move(t));
        }
        return true;
    }

    void ReleaseContext(int tid) {
        AclContext *ctx = FindContext(tid);
        if (ctx == nullptr) {
            return;
        }
        ReleaseContext(*ctx);
        ctx->valid = false;
    }

    void Destroy() {
        for (auto &kv : ctxs_) {
            if (kv.second.valid) {
                ReleaseContext(kv.second);
                kv.second.valid = false;
            }
        }
        ctxs_.clear();
        hostInputs_.clear();
        if (deviceSet_) {
            aclrtResetDevice(opt_.device);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

private:
    static void ReleaseContext(AclContext &ctx) {
        DestroyDataset(ctx.in, ctx.inBufs);
        DestroyDataset(ctx.out, ctx.outBufs);
        ctx.outSizes.clear();
        ctx.in = nullptr;
        ctx.out = nullptr;
        if (ctx.stream != nullptr) {
            aclrtSynchronizeStream(ctx.stream);
            aclrtDestroyStream(ctx.stream);
            ctx.stream = nullptr;
        }
        if (ctx.desc != nullptr) {
            aclmdlDestroyDesc(ctx.desc);
            ctx.desc = nullptr;
        }
        if (ctx.modelLoaded) {
            aclmdlUnload(ctx.modelId);
            ctx.modelLoaded = false;
            ctx.modelId = 0;
        }
    }

    static void DestroyDataset(aclmdlDataset *&ds, std::vector<void *> &devBuffers) {
        if (ds != nullptr) {
            for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(ds); i++) {
                aclDestroyDataBuffer(aclmdlGetDatasetBuffer(ds, i));
            }
            aclmdlDestroyDataset(ds);
        }
        for (void *p : devBuffers) {
            if (p != nullptr) {
                aclrtFree(p);
            }
        }
        devBuffers.clear();
    }

    void StoreContext(int tid, const AclContext &ctx) { ctxs_[tid] = ctx; }

    AclContext *FindContext(int tid) {
        auto it = ctxs_.find(tid);
        return (it == ctxs_.end() || !it->second.valid) ? nullptr : &it->second;
    }

    AclOptions opt_;
    bool aclInited_ = false;
    bool deviceSet_ = false;
    bool dynamic_ = false;
    std::string omPath_;
    aclrtContext aclCtx_ = nullptr;
    size_t inputCount_ = 0;
    size_t outputCount_ = 0;
    std::vector<TensorPlan> plans_;
    std::vector<std::vector<char>> hostInputs_;
    std::map<int, AclContext> ctxs_;
};

}  // namespace

bool RunAclBackend(const Manifest &manifest, const IoSpec &spec,
                   const std::vector<TensorPlan> &inputs, const AclOptions &opt,
                   std::vector<HostTensor> &outputs) {
    if (manifest.om_path.empty()) {
        fprintf(stderr, "[ERROR] manifest 无 om_path (backend=om_acl 需要 ATC 编译出的 OM)\n");
        return false;
    }
    std::string omPath = manifest.Resolve(manifest.om_path);

    AclRunner runner(opt);
    if (!runner.Init(omPath)) {
        return false;
    }
    if (!runner.LoadInputs(inputs, spec.HasDynamicInput())) {
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
            PrintThroughputStats("ACL OM execute+sync throughput", s);
        }
        // 吞吐跑完后资源已释放: 单开一份资源跑一次, 取输出落盘
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
    BenchStats stats;
    if (!BenchRun(opt.bench, [&runner]() { return runner.Execute(0); }, stats)) {
        return false;
    }
    PrintBenchStats("ACL OM execute+sync", stats);
    bool ok = runner.CollectOutputs(0, spec, outputs);
    runner.ReleaseContext(0);
    return ok;
}

}  // namespace ge_runtime
