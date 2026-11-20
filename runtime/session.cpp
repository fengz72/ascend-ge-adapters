// GE Session 单图执行器 (AIR / ONNX) 的实现。
// GE / ACL 头只在这个 .cpp 里 (pimpl), 其余翻译单元保持与 CANN 无关。
#include "session.h"

#include <acl/acl.h>
#include <exe_graph/runtime/tensor.h>
#include <ge/ge_api.h>
#include <ge/ge_graph_compile_summary.h>
#include <graph/graph.h>
#include <parser/onnx_parser.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "npy.h"

namespace ge_execute {
namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void Fail(const std::string &what) { throw std::runtime_error(what); }

// steady_clock 的原始周期在不同平台上不一致, 先固定到 us 再换算 ms, 保证
// "微秒精度" 这个口径不随实现漂移。
double ElapsedMs(Clock::time_point begin, Clock::time_point end) {
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /
           1000.0;
}

void CheckAcl(aclError ret, const std::string &what) {
    if (ret != ACL_SUCCESS) {
        Fail(what + " failed, aclError=" + std::to_string(static_cast<int>(ret)));
    }
}

std::string Lower(std::string value) {
    for (char &c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

void MakeDirs(const std::string &path) {
    if (path.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
        Fail("cannot create directory " + path + ": " + ec.message());
    }
}

// npy.h 白名单规范名 -> ge::DataType。未知名字必须抛: 静默退成 DT_FLOAT 会让
// H2D 的字节被按 4 字节元素解释 —— 形状照样通过, 数值全错。
ge::DataType ToGeDtype(const std::string &name) {
    if (name == "float16") return ge::DT_FLOAT16;
    if (name == "bfloat16") return ge::DT_BF16;
    if (name == "float32") return ge::DT_FLOAT;
    if (name == "float64") return ge::DT_DOUBLE;
    if (name == "int8") return ge::DT_INT8;
    if (name == "int16") return ge::DT_INT16;
    if (name == "int32") return ge::DT_INT32;
    if (name == "int64") return ge::DT_INT64;
    if (name == "uint8") return ge::DT_UINT8;
    if (name == "uint16") return ge::DT_UINT16;
    if (name == "uint32") return ge::DT_UINT32;
    if (name == "uint64") return ge::DT_UINT64;
    if (name == "bool") return ge::DT_BOOL;
    Fail("unsupported dtype: '" + name + "'");
}

// 反向映射: 输出 dtype 一律报规范名, bfloat16 原样报 (npy 里按 uint16 存)。
std::string FromGeDtype(ge::DataType dtype) {
    switch (dtype) {
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
        default:
            Fail("unsupported ge::DataType: " + std::to_string(static_cast<int>(dtype)));
    }
}

size_t Elements(const std::vector<int64_t> &shape) {
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            Fail("unresolved dimension in compiled output shape");
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

// ONNX 走 parser 库, AIR 直接反序列化 (graph_loader 同款结论: 不能把 ONNX 当 AIR 读)。
bool LoadGraphFile(const std::string &path, const std::string &format, ge::Graph &graph) {
    if (format == "air") {
        return graph.LoadFromFile(path.c_str()) == ge::GRAPH_SUCCESS;
    }
    std::map<ge::AscendString, ge::AscendString> options;
    return ge::aclgrphParseONNX(path.c_str(), options, graph) == ge::GRAPH_SUCCESS;
}

// 每请求重建张量描述, 但复用预分配的 device 缓冲。
// TensorData 的 size 必须给**分配大小**而不是该请求的逻辑字节: GE 会按
// (size, shape) 判断缓冲能力, 给逻辑字节会让"小请求先跑、大请求后跑"的
// 变长场景被 GE 当成缓冲不足。
void EmplaceTensor(std::vector<gert::Tensor> &out, void *device, size_t allocBytes,
                   const std::vector<int64_t> &shape, ge::DataType dtype) {
    gert::Tensor tensor;
    gert::StorageShape storage;
    for (int64_t dim : shape) {
        storage.MutableOriginShape().AppendDim(dim);
        storage.MutableStorageShape().AppendDim(dim);
    }
    tensor.GetShape() = storage;
    tensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    tensor.SetDataType(dtype);
    tensor.SetData(gert::TensorData(device, nullptr, allocBytes, gert::kOnDeviceHbm));
    out.emplace_back(std::move(tensor));
}

// 静态编译图的输出形状/dtype 在编译期就固定; 这类输出由调用方分配并持有。
struct OutputSlot {
    ge::DataType dtype = ge::DT_UNDEFINED;
    std::vector<int64_t> shape;
    size_t bytes = 0;                 // 该 shape/dtype 的精确字节数
};

// 一个闭环路 = 一份 graphId + 一个 stream + 一套输入缓冲 (可选一套输出缓冲)。
struct Instance {
    uint32_t graphId = 0;
    aclrtStream stream = nullptr;
    double loadMs = 0.0;
    std::vector<void *> inputPtrs;
    std::vector<size_t> inputCap;     // 每输入分配字节 (= pool.maxBytes[i])
    bool callerOutputs = false;       // summary->IsStatic()
    std::vector<OutputSlot> outSlots;
    std::vector<void *> outputPtrs;   // 自持输出缓冲 (仅 callerOutputs)
    std::vector<gert::Tensor> lastOutputs;   // 上一次 Execute 传给 GE 的输出张量
    std::vector<char> scratch;               // discard 模式复用的主机落地缓冲 (按最大输出增长)
    bool executed = false;            // 至少成功执行过一次 (CollectOutputs 的前置条件)
};

void ReleaseInstance(Instance &instance) {
    for (void *ptr : instance.inputPtrs) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
    instance.inputPtrs.clear();
    instance.inputCap.clear();
    // 只释放自己 aclrtMalloc 的输出; 动态图的输出地址归 GE, 不能在这里 free。
    for (void *ptr : instance.outputPtrs) {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
    instance.outputPtrs.clear();
    instance.outSlots.clear();
    instance.lastOutputs.clear();
    instance.scratch.clear();
    instance.scratch.shrink_to_fit();
    instance.executed = false;
    if (instance.stream != nullptr) {
        aclrtDestroyStream(instance.stream);
        instance.stream = nullptr;
    }
}

}  // namespace

struct Session::Impl {
    explicit Impl(SessionConfig value) : config(std::move(value)) {}

    SessionConfig config;
    std::string format;
    std::shared_ptr<ge::Session> session;
    aclrtContext context = nullptr;   // 默认 context, 工作线程共享
    std::vector<std::unique_ptr<Instance>> instances;
    double compileMs = 0.0;
    bool geInited = false;
    bool aclInited = false;
    bool deviceSet = false;
};

Session::Session(SessionConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {}

Session::~Session() { Destroy(); }

std::string NormalizeAicoreNum(const std::string &raw) {
    if (raw.empty()) {
        return raw;
    }
    const auto bar = raw.find('|');
    if (bar == std::string::npos || bar == 0 || bar + 1 == raw.size() ||
        raw.find('|', bar + 1) != std::string::npos) {
        Fail("aicore_num requires an explicit positive AIC|AIV pair: '" + raw + "'");
    }
    const std::string aic = raw.substr(0, bar);
    const std::string aiv = raw.substr(bar + 1);
    if (aic.find_first_not_of("0123456789") != std::string::npos ||
        aiv.find_first_not_of("0123456789") != std::string::npos) {
        Fail("aicore_num requires an explicit positive AIC|AIV pair: '" + raw + "'");
    }
    try {
        if (std::stoull(aic) == 0 || std::stoull(aiv) == 0) {
            Fail("aicore_num values must be positive: '" + raw + "'");
        }
    } catch (const std::runtime_error &) {
        throw;   // Fail 抛出的 runtime_error 原样上抛
    } catch (const std::exception &) {
        Fail("aicore_num is out of range: '" + raw + "'");
    }
    return raw;
}

std::string InferFormat(const std::string &graphPath) {
    const auto dot = graphPath.find_last_of('.');
    const std::string ext = dot == std::string::npos ? std::string() : Lower(graphPath.substr(dot));
    if (ext == ".air") {
        return "air";
    }
    if (ext == ".onnx") {
        return "onnx";
    }
    Fail("cannot infer graph format from '" + graphPath + "' (expect .air or .onnx)");
}

void Session::Init(const CasePool &pool) {
    Impl &impl = *impl_;
    if (impl.session != nullptr || !impl.instances.empty()) {
        Fail("Session::Init called twice");
    }
    if (pool.empty()) {
        Fail("case pool is empty");
    }
    if (pool.maxBytes.empty()) {
        Fail("case pool has no inputs");
    }
    impl.format = impl.config.format.empty() ? InferFormat(impl.config.graphPath)
                                             : Lower(impl.config.format);
    if (impl.format != "air" && impl.format != "onnx") {
        Fail("unsupported graph format: '" + impl.config.format + "'");
    }
    const int instanceCount = std::max(1, impl.config.instances);

    // ---- 1. GEInitialize: 全局选项在进程内只生效一次, 之后再也改不了 ----
    std::map<ge::AscendString, ge::AscendString> globalOptions;
    globalOptions.emplace(ge::AscendString("ge.exec.deviceId"),
                          ge::AscendString(std::to_string(impl.config.device).c_str()));
    const std::string aicore = NormalizeAicoreNum(impl.config.aicoreNum);
    if (!aicore.empty()) {
        globalOptions.emplace(ge::AscendString(ge::ir_option::AICORE_NUM),
                              ge::AscendString(aicore.c_str()));
    }
    if (impl.config.profiling) {
        // GE profiling 只有 GEInitialize 这一个开关 (执行期没有 API), 产出 msprof 可解析的
        // PROF_* 会话; 输出目录必须在初始化前就存在。
        if (impl.config.profilingOutput.empty()) {
            Fail("profiling enabled but profilingOutput is empty");
        }
        MakeDirs(impl.config.profilingOutput);
        globalOptions.emplace(ge::AscendString(ge::OPTION_EXEC_PROFILING_MODE),
                              ge::AscendString("1"));
        nlohmann::json options;
        options["output"] = impl.config.profilingOutput;
        options["task_time"] = "on";
        options["runtime_api"] = "on";
        if (!impl.config.aicMetrics.empty()) {
            options["aic_metrics"] = impl.config.aicMetrics;
        }
        const std::string encoded = options.dump();
        globalOptions.emplace(ge::AscendString(ge::OPTION_EXEC_PROFILING_OPTIONS),
                              ge::AscendString(encoded.c_str()));
    }
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        Fail("GEInitialize failed: " + ge::GEGetErrorMsg());
    }
    impl.geInited = true;

    // ---- 2. aclInit: 与 GEInitialize 是两个独立子系统, 重复初始化算成功 ----
    const aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS && aclRet != ACL_ERROR_REPEAT_INITIALIZE) {
        Fail("aclInit failed, aclError=" + std::to_string(static_cast<int>(aclRet)));
    }
    impl.aclInited = true;

    // ---- 3. ge::Session: device / precision 是 session 级选项, 不能提到 GEInitialize ----
    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    sessionOptions.emplace(ge::AscendString("ge.session_device_id"),
                           ge::AscendString(std::to_string(impl.config.device).c_str()));
    sessionOptions.emplace(ge::AscendString("ge.exec.precision_mode"),
                           ge::AscendString(impl.config.precisionMode.c_str()));
    impl.session = std::make_shared<ge::Session>(sessionOptions);

    // ---- 4. 每实例 AddGraph + CompileGraph, 全部编译完才能进第 5 步 ----
    // 同一 graphId 不能 LoadGraph 两次 (ge_api.h 约束), 所以 K 路并发必须是 K 份
    // AddGraph+CompileGraph; 串行执行, 单份 ~10s 是这套在线图的固有代价。
    // CompileGraph 必须在 aclrtSetDevice **之前** (对齐 atb/bench_ge_latency 的实测序列,
    // 反过来会在编译期触发设备上下文相关的初始化差异)。
    const auto compileBegin = Clock::now();
    for (int index = 0; index < instanceCount; ++index) {
        const auto instanceBegin = Clock::now();
        auto instance = std::make_unique<Instance>();
        instance->graphId = static_cast<uint32_t>(index + 1);
        const std::string gid = std::to_string(instance->graphId);
        ge::Graph graph;
        if (!LoadGraphFile(impl.config.graphPath, impl.format, graph)) {
            Fail("cannot load graph '" + impl.config.graphPath + "' as " + impl.format);
        }
        if (impl.session->AddGraph(instance->graphId, graph) != ge::SUCCESS) {
            Fail("AddGraph(gid=" + gid + ") failed: " + ge::GEGetErrorMsg());
        }
        if (impl.session->CompileGraph(instance->graphId) != ge::SUCCESS) {
            Fail("CompileGraph(gid=" + gid + ") failed: " + ge::GEGetErrorMsg());
        }
        auto summary = impl.session->GetCompiledGraphSummary(instance->graphId);
        if (summary == nullptr) {
            Fail("GetCompiledGraphSummary(gid=" + gid + ") returned null");
        }
        // 静态图: 输出张量由调用方分配, shape/dtype 取编译期 summary; 动态图由 GE 分配。
        instance->callerOutputs = summary->IsStatic();
        if (instance->callerOutputs) {
            std::vector<ge::Shape> shapes;
            std::vector<ge::DataType> dtypes;
            if (summary->GetOutputShapes(shapes) != ge::SUCCESS ||
                summary->GetOutputDtypes(dtypes) != ge::SUCCESS ||
                shapes.size() != dtypes.size()) {
                Fail("invalid compiled output metadata (gid=" + gid + ")");
            }
            for (size_t slotIndex = 0; slotIndex < shapes.size(); ++slotIndex) {
                OutputSlot slot;
                slot.dtype = dtypes[slotIndex];
                slot.shape = shapes[slotIndex].GetDims();
                const size_t itemBytes = DtypeBytes(FromGeDtype(slot.dtype));
                if (itemBytes == 0) {
                    Fail("unsupported compiled output dtype (gid=" + gid + ")");
                }
                slot.bytes = Elements(slot.shape) * itemBytes;
                instance->outSlots.push_back(std::move(slot));
            }
        }
        std::printf("[INFO] CompileGraph ok (gid=%s/%d), cost %.1f ms, outputs=%zu, "
                    "allocation=%s\n",
                    gid.c_str(), instanceCount,
                    ElapsedMs(instanceBegin, Clock::now()), instance->outSlots.size(),
                    instance->callerOutputs ? "caller (static compiled)"
                                            : "GE (dynamic compiled)");
        impl.instances.push_back(std::move(instance));
    }
    impl.compileMs = ElapsedMs(compileBegin, Clock::now());

    // 静态编译图的 shape 在编译期就钉死了: 池里出现第二种 shape 时 GE 不报错, 只按
    // 编译期 shape 解释字节 (实测静态 [2,4] 图喂 [3,4] 输入, 输出照样是 [2,3]),
    // 所以在 Init 就拦掉。GE 没有查询编译期**输入** shape 的 API, 因此"池内 shape
    // 与图自身静态 shape 不一致"这种查不出来 —— 那需要 io 契约, 本项目明确不要。
    if (impl.instances.front()->callerOutputs && pool.distinctShapes > 1) {
        Fail("'" + impl.config.graphPath + "' compiles to a static shape but the case pool has " +
             std::to_string(pool.distinctShapes) + " distinct shapes; a static graph takes " +
             "exactly one shape");
    }

    // ---- 5. aclrtSetDevice + 取默认 context ----
    // stream 必须建在**默认 context** 上; 显式 aclrtCreateContext 会让 GE executor 报
    // "stream is not in current ctx"。工作线程用 aclrtSetCurrentContext 共享这个 context。
    CheckAcl(aclrtSetDevice(impl.config.device),
             "aclrtSetDevice(" + std::to_string(impl.config.device) + ")");
    impl.deviceSet = true;
    CheckAcl(aclrtGetCurrentContext(&impl.context), "aclrtGetCurrentContext");
    if (impl.context == nullptr) {
        Fail("aclrtGetCurrentContext returned a null context");
    }

    // ---- 6. 每实例: stream -> LoadGraph -> 输入/输出缓冲 ----
    for (auto &owned : impl.instances) {
        Instance &instance = *owned;
        const std::string gid = std::to_string(instance.graphId);
        CheckAcl(aclrtCreateStream(&instance.stream), "aclrtCreateStream(gid=" + gid + ")");

        const auto loadBegin = Clock::now();
        // 同一个 graphId 只能 LoadGraph 一次; 每实例一份 gid, 所以这里恰好每图一次。
        if (impl.session->LoadGraph(instance.graphId, {}, instance.stream) != ge::SUCCESS) {
            Fail("LoadGraph(gid=" + gid + ") failed: " + ge::GEGetErrorMsg());
        }

        // 输入缓冲按池内最大字节预分配: 每请求只 H2D 实际字节, 避免每请求 malloc。
        for (size_t inputIndex = 0; inputIndex < pool.maxBytes.size(); ++inputIndex) {
            const size_t capacity = pool.maxBytes[inputIndex];
            void *device = nullptr;
            CheckAcl(aclrtMalloc(&device, capacity, ACL_MEM_MALLOC_HUGE_FIRST),
                     "aclrtMalloc(" + std::to_string(capacity) + ", gid=" + gid + ")");
            instance.inputPtrs.push_back(device);
            instance.inputCap.push_back(capacity);
        }

        if (instance.callerOutputs) {
            for (size_t slotIndex = 0; slotIndex < instance.outSlots.size(); ++slotIndex) {
                void *device = nullptr;
                CheckAcl(aclrtMalloc(&device, instance.outSlots[slotIndex].bytes,
                                     ACL_MEM_MALLOC_HUGE_FIRST),
                         "aclrtMalloc(output " + std::to_string(slotIndex) + ", gid=" + gid + ")");
                instance.outputPtrs.push_back(device);
            }
        }
        instance.loadMs = ElapsedMs(loadBegin, Clock::now());
        std::printf("[INFO] LoadGraph ok (gid=%s), cost %.1f ms, %zu input buffer(s)%s\n",
                    gid.c_str(), instance.loadMs, instance.inputPtrs.size(),
                    instance.callerOutputs ? ", caller-owned outputs" : "");
    }
}

void Session::BindThread() const {
    const Impl &impl = *impl_;
    if (impl.context == nullptr) {
        Fail("Session::BindThread before Init");
    }
    CheckAcl(aclrtSetCurrentContext(impl.context), "aclrtSetCurrentContext");
}

double Session::compileMs() const { return impl_->compileMs; }

double Session::loadMs(int instance) const {
    if (instance < 0 || static_cast<size_t>(instance) >= impl_->instances.size()) {
        return 0.0;
    }
    return impl_->instances[static_cast<size_t>(instance)]->loadMs;
}

RunTimings Session::Execute(int instance, const Case &request) {
    Impl &impl = *impl_;
    if (instance < 0 || static_cast<size_t>(instance) >= impl.instances.size()) {
        Fail("Execute: instance " + std::to_string(instance) + " out of range (have " +
             std::to_string(impl.instances.size()) + ")");
    }
    Instance &slot = *impl.instances[static_cast<size_t>(instance)];
    if (request.inputs.size() != slot.inputPtrs.size()) {
        Fail("Execute(" + request.id + "): " + std::to_string(request.inputs.size()) +
             " inputs but instance has " + std::to_string(slot.inputPtrs.size()) + " buffers");
    }

    RunTimings timing;

    // H2D 只拷贝该 case 的实际字节 (缓冲按池内最大预分配)。
    const auto h2dBegin = Clock::now();
    for (size_t inputIndex = 0; inputIndex < request.inputs.size(); ++inputIndex) {
        const CaseInput &input = request.inputs[inputIndex];
        if (input.data.size() > slot.inputCap[inputIndex]) {
            Fail("Execute(" + request.id + "): input " + std::to_string(inputIndex) + " needs " +
                 std::to_string(input.data.size()) + " bytes > buffer " +
                 std::to_string(slot.inputCap[inputIndex]));
        }
        if (!input.data.empty()) {
            CheckAcl(aclrtMemcpy(slot.inputPtrs[inputIndex], slot.inputCap[inputIndex],
                                 input.data.data(), input.data.size(),
                                 ACL_MEMCPY_HOST_TO_DEVICE),
                     "H2D(" + request.id + ", input " + std::to_string(inputIndex) + ")");
        }
    }
    timing.h2dMs = ElapsedMs(h2dBegin, Clock::now());

    // 重建张量描述是纯主机开销, 不计入 h2dMs/execMs (口径: e2e = h2d + execute + sync)。
    std::vector<gert::Tensor> inputs;
    inputs.reserve(request.inputs.size());
    for (size_t inputIndex = 0; inputIndex < request.inputs.size(); ++inputIndex) {
        EmplaceTensor(inputs, slot.inputPtrs[inputIndex], slot.inputCap[inputIndex],
                      request.inputs[inputIndex].shape,
                      ToGeDtype(request.inputs[inputIndex].dtype));
    }

    // 静态图: 每轮用同一批自持地址重建输出张量传出去; 动态图: 传空 vector 让 GE 分配。
    slot.lastOutputs.clear();
    if (slot.callerOutputs) {
        for (size_t slotIndex = 0; slotIndex < slot.outSlots.size(); ++slotIndex) {
            const OutputSlot &plan = slot.outSlots[slotIndex];
            EmplaceTensor(slot.lastOutputs, slot.outputPtrs[slotIndex], plan.bytes, plan.shape,
                          plan.dtype);
        }
    }

    const auto execBegin = Clock::now();
    const ge::Status status = impl.session->ExecuteGraphWithStreamAsync(
        slot.graphId, slot.stream, inputs, slot.lastOutputs);
    if (status != ge::SUCCESS) {
        Fail("ExecuteGraphWithStreamAsync(gid=" + std::to_string(slot.graphId) +
             ") failed: " + ge::GEGetErrorMsg());
    }
    CheckAcl(aclrtSynchronizeStream(slot.stream),
             "aclrtSynchronizeStream(gid=" + std::to_string(slot.graphId) + ")");
    timing.execMs = ElapsedMs(execBegin, Clock::now());
    timing.e2eMs = timing.h2dMs + timing.execMs;

    if (slot.callerOutputs) {
        if (slot.lastOutputs.size() != slot.outSlots.size()) {
            Fail("GE returned " + std::to_string(slot.lastOutputs.size()) +
                 " outputs for gid=" + std::to_string(slot.graphId) + ", expected " +
                 std::to_string(slot.outSlots.size()));
        }
        for (size_t slotIndex = 0; slotIndex < slot.outSlots.size(); ++slotIndex) {
            if (slot.lastOutputs[slotIndex].GetAddr() != slot.outputPtrs[slotIndex]) {
                Fail("GE replaced caller-owned output buffer (gid=" +
                     std::to_string(slot.graphId) + ", output=" + std::to_string(slotIndex) + ")");
            }
        }
    }
    slot.executed = true;
    return timing;
}

Readback Session::CollectOutputs(int instance, bool discard) {
    Impl &impl = *impl_;
    if (instance < 0 || static_cast<size_t>(instance) >= impl.instances.size()) {
        Fail("CollectOutputs: instance " + std::to_string(instance) + " out of range");
    }
    Instance &slot = *impl.instances[static_cast<size_t>(instance)];
    if (!slot.executed || slot.lastOutputs.empty()) {
        Fail("CollectOutputs: instance " + std::to_string(instance) +
             " has no completed Execute");
    }
    Readback readback;
    if (!discard) {
        readback.outputs.reserve(slot.lastOutputs.size());
    }
    const auto begin = Clock::now();
    for (size_t index = 0; index < slot.lastOutputs.size(); ++index) {
        const gert::Tensor &tensor = slot.lastOutputs[index];
        const std::string dtype = FromGeDtype(tensor.GetDataType());
        std::vector<int64_t> shape;
        const auto &storage = tensor.GetShape().GetStorageShape();
        shape.reserve(storage.GetDimNum());
        for (size_t dim = 0; dim < storage.GetDimNum(); ++dim) {
            shape.push_back(storage.GetDim(dim));
        }
        // 该拷的字节数一律以 shape*dtype 反算: 静态自持缓冲的 GetSize() 是**分配大小**
        // (按池内最大预分配时会大于逻辑字节)。反过来 GE 给的缓冲比 shape 所需还小就是
        // 真不一致, 必须报错 —— 静默按小值截断会产出一个短 npy, 下游比较才发现不了。
        const size_t logical = Elements(shape) * DtypeBytes(dtype);
        const size_t reported = tensor.GetSize();
        if (reported < logical) {
            Fail("CollectOutputs: output " + std::to_string(index) + " (" + dtype + ") needs " +
                 std::to_string(logical) + " bytes but GE returned a " +
                 std::to_string(reported) + "-byte buffer");
        }
        const void *address = tensor.GetAddr();
        if (logical > 0 && address == nullptr) {
            Fail("CollectOutputs: output address is null");
        }
        OutputTensor output;
        char *destination = nullptr;
        if (discard) {
            // bench 只要传输代价, 不要数据: 落到每实例复用的 scratch, 稳态零分配,
            // 免得测到的是分配器噪声。scratch 的增长只发生在第一次 (预热段就付掉)。
            if (slot.scratch.size() < logical) {
                slot.scratch.resize(logical);
            }
            destination = slot.scratch.data();
        } else {
            output.dtype = dtype;
            output.shape = shape;
            output.data.resize(logical);
            destination = output.data.data();
        }
        if (logical > 0) {
            CheckAcl(aclrtMemcpy(destination, logical, address, logical,
                                 ACL_MEMCPY_DEVICE_TO_HOST),
                     "D2H(" + dtype + ", output " + std::to_string(index) + ")");
        }
        if (!discard) {
            readback.outputs.push_back(std::move(output));
        }
    }
    readback.d2hMs = ElapsedMs(begin, Clock::now());
    return readback;
}

void Session::Destroy() {
    Impl &impl = *impl_;
    // 顺序是实测约束: 先放实例资源 (缓冲 + stream), 再拆 Session, 再退设备,
    // 最后 GEFinalize -> aclFinalize。任一步都只是尽力而为, 析构路径不能抛。
    for (auto &instance : impl.instances) {
        ReleaseInstance(*instance);
    }
    impl.instances.clear();
    impl.session.reset();
    if (impl.deviceSet) {
        const aclError ret = aclrtResetDevice(impl.config.device);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "ge_execute: aclrtResetDevice(%d) failed, aclError=%d\n",
                         impl.config.device, static_cast<int>(ret));
        }
        impl.deviceSet = false;
    }
    if (impl.geInited) {
        if (ge::GEFinalize() != ge::SUCCESS) {
            std::fprintf(stderr, "ge_execute: GEFinalize failed: %s\n", ge::GEGetErrorMsg().c_str());
        }
        impl.geInited = false;
    }
    if (impl.aclInited) {
        const aclError ret = aclFinalize();
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "ge_execute: aclFinalize failed, aclError=%d\n",
                         static_cast<int>(ret));
        }
        impl.aclInited = false;
    }
}

}  // namespace ge_execute
