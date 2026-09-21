#include "acl_backend.h"

#include <acl/acl.h>

#include <cstdio>
#include <iostream>
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

class AclRunner {
public:
    AclRunner(const AclOptions &opt) : opt_(opt) {}

    ~AclRunner() { Destroy(); }

    bool Init(const std::string &omPath) {
        aclError ret = aclInit(nullptr);
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

        ret = aclmdlLoadFromFile(omPath.c_str(), &modelId_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclmdlLoadFromFile failed: %s, ret=%d\n", omPath.c_str(), ret);
            return false;
        }
        modelLoaded_ = true;

        desc_ = aclmdlCreateDesc();
        if (desc_ == nullptr) {
            fprintf(stderr, "[ERROR] aclmdlCreateDesc returned nullptr\n");
            return false;
        }
        ret = aclmdlGetDesc(desc_, modelId_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclmdlGetDesc failed, ret=%d\n", ret);
            return false;
        }
        inputCount_ = aclmdlGetNumInputs(desc_);
        outputCount_ = aclmdlGetNumOutputs(desc_);
        std::cout << "[INFO] OM loaded: " << omPath << " (modelId=" << modelId_
                  << ", inputs=" << inputCount_ << ", outputs=" << outputCount_ << ")" << std::endl;
        for (size_t i = 0; i < inputCount_; i++) {
            const char *name = aclmdlGetInputNameByIndex(desc_, i);
            size_t size = aclmdlGetInputSizeByIndex(desc_, i);
            std::cout << "[INFO]   OM input[" << i << "] name=" << (name ? name : "?")
                      << " staticSize=" << size << (size == 0 ? " (dynamic)" : "") << std::endl;
        }
        for (size_t i = 0; i < outputCount_; i++) {
            const char *name = aclmdlGetOutputNameByIndex(desc_, i);
            size_t size = aclmdlGetOutputSizeByIndex(desc_, i);
            std::cout << "[INFO]   OM output[" << i << "] name=" << (name ? name : "?")
                      << " staticSize=" << size << (size == 0 ? " (dynamic)" : "") << std::endl;
        }
        return true;
    }

    bool PrepareInputs(const std::vector<TensorPlan> &plans, bool dynamic) {
        if (plans.size() != inputCount_) {
            fprintf(stderr, "[ERROR] io_spec/bundle give %zu inputs but OM has %zu\n",
                    plans.size(), inputCount_);
            return false;
        }
        input_ = aclmdlCreateDataset();
        if (input_ == nullptr) {
            fprintf(stderr, "[ERROR] aclmdlCreateDataset(input) failed\n");
            return false;
        }

        for (size_t i = 0; i < plans.size(); i++) {
            const TensorPlan &p = plans[i];
            aclDataType dtype = ToAclDtype(p.dtype);
            if (dtype == ACL_DT_UNDEFINED) {
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

            aclDataBuffer *buf = aclCreateDataBuffer(dev, bytes);
            if (buf == nullptr) {
                fprintf(stderr, "[ERROR] aclCreateDataBuffer failed for input '%s'\n", p.logical.c_str());
                aclrtFree(dev);
                return false;
            }
            ret = aclmdlAddDatasetBuffer(input_, buf);
            if (ret != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] aclmdlAddDatasetBuffer(input[%zu]) failed, ret=%d\n", i, ret);
                aclDestroyDataBuffer(buf);
                aclrtFree(dev);
                return false;
            }
            inputBuffers_.push_back(dev);

            if (dynamic) {
                aclTensorDesc *td = aclCreateTensorDesc(dtype, static_cast<int32_t>(p.shape.size()),
                                                        p.shape.data(), ToAclFormat(p.format));
                if (td == nullptr) {
                    fprintf(stderr, "[ERROR] aclCreateTensorDesc failed for input '%s'\n", p.logical.c_str());
                    return false;
                }
                ret = aclmdlSetDatasetTensorDesc(input_, td, i);
                aclDestroyTensorDesc(td);
                if (ret != ACL_SUCCESS) {
                    fprintf(stderr, "[ERROR] aclmdlSetDatasetTensorDesc(input[%zu]) failed, ret=%d\n", i, ret);
                    return false;
                }
            }
            std::cout << "[INFO] Input[" << i << "] '" << p.logical << "' node=" << p.node
                      << " shape=" << ShapeToString(p.shape) << " dtype=" << p.dtype
                      << " bytes=" << bytes << std::endl;
        }
        return true;
    }

    bool PrepareOutputs() {
        output_ = aclmdlCreateDataset();
        if (output_ == nullptr) {
            fprintf(stderr, "[ERROR] aclmdlCreateDataset(output) failed\n");
            return false;
        }
        for (size_t i = 0; i < outputCount_; i++) {
            size_t bytes = aclmdlGetOutputSizeByIndex(desc_, i);
            if (bytes == 0) {
                bytes = opt_.output_reserve;
                std::cout << "[INFO] Output[" << i << "] size unknown (dynamic), reserving "
                          << bytes << " bytes" << std::endl;
            }
            void *dev = nullptr;
            aclError ret = aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS || dev == nullptr) {
                fprintf(stderr, "[ERROR] aclrtMalloc(%zu) failed for output[%zu], ret=%d\n", bytes, i, ret);
                return false;
            }
            aclDataBuffer *buf = aclCreateDataBuffer(dev, bytes);
            if (buf == nullptr) {
                fprintf(stderr, "[ERROR] aclCreateDataBuffer failed for output[%zu]\n", i);
                aclrtFree(dev);
                return false;
            }
            ret = aclmdlAddDatasetBuffer(output_, buf);
            if (ret != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] aclmdlAddDatasetBuffer(output[%zu]) failed, ret=%d\n", i, ret);
                aclDestroyDataBuffer(buf);
                aclrtFree(dev);
                return false;
            }
            outputBuffers_.push_back(dev);
            outputSizes_.push_back(bytes);
        }
        return true;
    }

    bool Execute() {
        aclError ret = aclmdlExecute(modelId_, input_, output_);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] aclmdlExecute failed, ret=%d\n", ret);
            return false;
        }
        return true;
    }

    bool CollectOutputs(const IoSpec &spec, std::vector<HostTensor> &outputs) {
        for (size_t i = 0; i < outputCount_; i++) {
            size_t bytes = 0;
            std::vector<int64_t> shape;
            aclDataType dtype = ACL_DT_UNDEFINED;

            // 动态图: 执行后 dataset 上挂着实际 desc; 静态图: dataset 无 desc → 取 model desc
            aclTensorDesc *td = aclmdlGetDatasetTensorDesc(output_, i);
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
                bytes = aclmdlGetOutputSizeByIndex(desc_, i);
                dtype = aclmdlGetOutputDataType(desc_, i);
                aclmdlIODims dims;
                if (aclmdlGetOutputDims(desc_, i, &dims) == ACL_SUCCESS) {
                    for (int32_t d = 0; d < dims.dimCount; d++) {
                        shape.push_back(dims.dims[d]);
                    }
                }
            }

            // 动态输出按 reserve 预分配, 实际 size 超出即越界 (执行时已写坏 HBM) → 硬失败
            if (i < outputSizes_.size() && bytes > outputSizes_[i]) {
                fprintf(stderr, "[ERROR] output[%zu] 实际需要 %zu 字节 > 预留 %zu 字节 "
                                "(--output_reserve %zu MB); 输出已可能越界, 结果不可信\n",
                        i, bytes, outputSizes_[i], outputSizes_[i] / (1024 * 1024));
                return false;
            }

            HostTensor t;
            t.dtype = FromAclDtype(dtype);
            t.logical = i < spec.outputs.size() && !spec.outputs[i].logical.empty()
                            ? spec.outputs[i].logical
                            : "output_" + std::to_string(i);
            t.shape = shape;
            t.data.resize(bytes);

            aclDataBuffer *buf = aclmdlGetDatasetBuffer(output_, i);
            void *dev = aclGetDataBufferAddr(buf);
            if (bytes > 0 && dev != nullptr) {
                aclError ret = aclrtMemcpy(t.data.data(), bytes, dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
                if (ret != ACL_SUCCESS) {
                    fprintf(stderr, "[ERROR] D2H failed for output[%zu], ret=%d\n", i, ret);
                    return false;
                }
            }
            std::cout << "[INFO] Output[" << i << "] '" << t.logical << "' shape="
                      << ShapeToString(t.shape) << " dtype=" << t.dtype << " bytes=" << bytes << std::endl;
            outputs.push_back(std::move(t));
        }
        return true;
    }

    void Destroy() {
        DestroyDataset(input_, inputBuffers_);
        DestroyDataset(output_, outputBuffers_);
        outputSizes_.clear();
        input_ = nullptr;
        output_ = nullptr;
        if (desc_ != nullptr) {
            aclmdlDestroyDesc(desc_);
            desc_ = nullptr;
        }
        if (modelLoaded_) {
            aclmdlUnload(modelId_);
            modelLoaded_ = false;
        }
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

    AclOptions opt_;
    bool aclInited_ = false;
    bool deviceSet_ = false;
    bool modelLoaded_ = false;
    uint32_t modelId_ = 0;
    aclmdlDesc *desc_ = nullptr;
    aclmdlDataset *input_ = nullptr;
    aclmdlDataset *output_ = nullptr;
    size_t inputCount_ = 0;
    size_t outputCount_ = 0;
    std::vector<void *> inputBuffers_;
    std::vector<void *> outputBuffers_;
    std::vector<size_t> outputSizes_;
};

}  // namespace

bool RunAclBackend(const Manifest &manifest, const IoSpec &spec,
                   const std::vector<TensorPlan> &inputs, const AclOptions &opt,
                   std::vector<HostTensor> &outputs) {
    std::string omPath = manifest.Resolve(manifest.om_path);
    if (manifest.om_path.empty()) {
        fprintf(stderr, "[ERROR] manifest has no om_path (backend=om_acl needs ATC-compiled OM)\n");
        return false;
    }

    AclRunner runner(opt);
    if (!runner.Init(omPath)) {
        return false;
    }
    if (!runner.PrepareInputs(inputs, spec.HasDynamicInput())) {
        return false;
    }
    if (!runner.PrepareOutputs()) {
        return false;
    }

    BenchStats stats;
    if (!BenchRun(opt.bench, [&runner]() { return runner.Execute(); }, stats)) {
        return false;
    }
    PrintBenchStats("ACL OM execute", stats);

    return runner.CollectOutputs(spec, outputs);
}

}  // namespace ge_runtime
