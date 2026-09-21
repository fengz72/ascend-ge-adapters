#include "gesession_backend.h"

#include <acl/acl.h>
#include <ge/ge_api.h>
#include <graph/graph.h>
#include <exe_graph/runtime/tensor.h>

#include <chrono>
#include <cstdio>
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

class GeRunner {
public:
    explicit GeRunner(const GeSessionOptions &opt) : opt_(opt) {}

    ~GeRunner() { Destroy(); }

    bool Init(const std::string &graphPath) {
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

        auto t0 = Clock::now();
        ge::Graph graph;
        if (graph.LoadFromFile(graphPath.c_str()) != ge::GRAPH_SUCCESS) {
            fprintf(stderr, "[ERROR] Graph::LoadFromFile failed: %s\n", graphPath.c_str());
            return false;
        }
        ge::Status st = session_->AddGraph(graphId_, graph);
        if (st != ge::SUCCESS) {
            fprintf(stderr, "[ERROR] AddGraph failed, ret=%d\n", st);
            return false;
        }
        st = session_->CompileGraph(graphId_);
        if (st != ge::SUCCESS) {
            fprintf(stderr, "[ERROR] CompileGraph failed, ret=%d\n", st);
            return false;
        }
        std::cout << "[INFO] CompileGraph ok (" << graphPath << "), cost "
                  << ElapsedMs(t0, Clock::now()) << " ms" << std::endl;

        ret = aclrtSetDevice(opt_.device);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtSetDevice(%d) failed, ret=%d\n", opt_.device, ret);
            return false;
        }
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtCreateStream failed, ret=%d\n", ret);
            return false;
        }
        t0 = Clock::now();
        st = session_->LoadGraph(graphId_, {}, stream_);
        if (st != ge::SUCCESS) {
            fprintf(stderr, "[ERROR] LoadGraph failed, ret=%d\n", st);
            return false;
        }
        std::cout << "[INFO] LoadGraph ok, cost " << ElapsedMs(t0, Clock::now()) << " ms" << std::endl;
        return true;
    }

    bool PrepareInputs(const std::vector<TensorPlan> &plans) {
        for (size_t i = 0; i < plans.size(); i++) {
            const TensorPlan &p = plans[i];
            ge::DataType dtype;
            if (!ToGeDtype(p.dtype, dtype)) {
                fprintf(stderr, "[ERROR] unsupported dtype '%s' for input '%s'\n",
                        p.dtype.c_str(), p.logical.c_str());
                return false;
            }
            std::vector<char> host;
            if (!ReadBinFile(p.file, host)) {
                fprintf(stderr, "[ERROR] cannot read input file: %s\n", p.file.c_str());
                return false;
            }
            size_t bytes = p.Bytes();
            if (host.size() != bytes) {
                fprintf(stderr, "[ERROR] input '%s' file size %zu != %zu bytes\n",
                        p.logical.c_str(), host.size(), bytes);
                return false;
            }

            void *dev = nullptr;
            aclError ret = aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS || dev == nullptr) {
                fprintf(stderr, "[ERROR] aclrtMalloc(%zu) failed for input '%s', ret=%d\n",
                        bytes, p.logical.c_str(), ret);
                return false;
            }
            ret = aclrtMemcpy(dev, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] H2D failed for input '%s', ret=%d\n", p.logical.c_str(), ret);
                aclrtFree(dev);
                return false;
            }
            inputPtrs_.push_back(dev);

            gert::Tensor t;
            gert::StorageShape ss;
            for (int64_t d : p.shape) {
                ss.MutableOriginShape().AppendDim(d);
                ss.MutableStorageShape().AppendDim(d);
            }
            t.GetShape() = ss;
            t.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
            t.SetDataType(dtype);
            t.SetData(gert::TensorData(dev, nullptr, bytes, gert::kOnDeviceHbm));
            devInputs_.emplace_back(std::move(t));

            std::cout << "[INFO] Input[" << i << "] '" << p.logical << "' node=" << p.node
                      << " shape=" << ShapeToString(p.shape) << " dtype=" << p.dtype
                      << " bytes=" << bytes << std::endl;
        }
        return true;
    }

    bool ExecuteFirst() {
        auto t0 = Clock::now();
        ge::Status st = session_->ExecuteGraphWithStreamAsync(graphId_, stream_, devInputs_, devOutputs_);
        if (st != ge::SUCCESS) {
            fprintf(stderr, "[ERROR] ExecuteGraphWithStreamAsync failed, ret=%d\n", st);
            return false;
        }
        aclError ret = aclrtSynchronizeStream(stream_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtSynchronizeStream failed, ret=%d\n", ret);
            return false;
        }
        TrackOutputs(devOutputs_);
        std::cout << "[INFO] First execute (incl. shape specialization), cost "
                  << ElapsedMs(t0, Clock::now()) << " ms" << std::endl;
        return true;
    }

    bool Execute() {
        std::vector<gert::Tensor> tmp;
        ge::Status st = session_->ExecuteGraphWithStreamAsync(graphId_, stream_, devInputs_, tmp);
        if (st != ge::SUCCESS) {
            fprintf(stderr, "[ERROR] ExecuteGraphWithStreamAsync failed, ret=%d\n", st);
            return false;
        }
        aclError ret = aclrtSynchronizeStream(stream_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclrtSynchronizeStream failed, ret=%d\n", ret);
            return false;
        }
        TrackOutputs(tmp);
        return true;
    }

    bool CollectOutputs(const IoSpec &spec, std::vector<HostTensor> &outputs) {
        for (size_t i = 0; i < devOutputs_.size(); i++) {
            const gert::Tensor &t = devOutputs_[i];
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

    void Destroy() {
        for (void *p : inputPtrs_) {
            if (p != nullptr) {
                aclrtFree(p);
            }
        }
        inputPtrs_.clear();
        devInputs_.clear();
        devOutputs_.clear();
        for (void *p : outputAddrs_) {
            if (p != nullptr) {
                aclrtFree(p);
            }
        }
        outputAddrs_.clear();
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
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
    void TrackOutputs(std::vector<gert::Tensor> &tensors) {
        for (auto &t : tensors) {
            if (t.GetAddr() != nullptr) {
                outputAddrs_.insert(t.GetAddr());
            }
        }
        // 实测 GE 每轮复用同一输出地址 (set 去重后恒为 1, 20000 轮 HBM 不涨)。
        // 若将来 GE 改成每轮新分配, 这里会先告警而不是静默吃满 HBM。
        if (!warnedAddrGrowth_ && outputAddrs_.size() > 64) {
            warnedAddrGrowth_ = true;
            fprintf(stderr, "[WARN] 已累计 %zu 个不同的输出 device 地址 (GE 未复用缓冲?), "
                            "HBM 占用会随 --bench 线性增长\n", outputAddrs_.size());
        }
    }

    GeSessionOptions opt_;
    uint32_t graphId_ = 1;
    bool geInited_ = false;
    bool aclInited_ = false;
    bool deviceSet_ = false;
    std::shared_ptr<ge::Session> session_;
    aclrtStream stream_ = nullptr;
    std::vector<gert::Tensor> devInputs_;
    std::vector<gert::Tensor> devOutputs_;
    std::vector<void *> inputPtrs_;
    std::set<void *> outputAddrs_;
    bool warnedAddrGrowth_ = false;
};

}  // namespace

bool RunGeSessionBackend(const Manifest &manifest, const IoSpec &spec,
                         const std::vector<TensorPlan> &inputs, const GeSessionOptions &opt,
                         std::vector<HostTensor> &outputs) {
    if (manifest.graph_path.empty()) {
        fprintf(stderr, "[ERROR] manifest has no graph_path (backend=ge_session needs AIR/ONNX)\n");
        return false;
    }
    std::string graphPath = manifest.Resolve(manifest.graph_path);
    if (graphPath.size() >= 5 &&
        graphPath.compare(graphPath.size() - 5, 5, ".onnx") == 0) {
        fprintf(stderr, "[ERROR] GeSession 在线后端只吃 GE 图 (.air/.pbtxt), 收到 ONNX: %s\n"
                        "        ONNX 请先经 ATC (--framework=5) 转 OM 走 om_acl 后端 (docs §15 待定)\n",
                graphPath.c_str());
        return false;
    }

    GeRunner runner(opt);
    if (!runner.Init(graphPath)) {
        return false;
    }
    if (!runner.PrepareInputs(inputs)) {
        return false;
    }
    if (!runner.ExecuteFirst()) {
        return false;
    }

    BenchStats stats;
    if (!BenchRun(opt.bench, [&runner]() { return runner.Execute(); }, stats)) {
        return false;
    }
    PrintBenchStats("GeSession execute+sync", stats);

    return runner.CollectOutputs(spec, outputs);
}

}  // namespace ge_runtime
