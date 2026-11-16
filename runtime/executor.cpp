// Minimal executor for the text-only pipeline: loads the shared text AIR,
// feeds the 2 positional inputs (input_ids, position_ids) of every case from
// the packed pool written by generate_inputs.py, and writes the scores.
//
// Usage (all paths default relative to the current directory; run from the
// adapter root for the canonical defaults):
//   executor [--air work/build/text.air] [--records work/inputs/records.json]
//            [--output-dir work/output] [--case-filter <id>]...
//            [--iterations N] [--warmup M] [--seed S]
//            [--instances K] [--aicore-num A|B] [--readout <readout.json>]
//
// Without --iterations the run is INFER: every case (or the --case-filter
// subset) executes once, in order, and work/output gets outputs.bin +
// result.json (the ge-compare/1 layout tools/compare.py reads).  With
// --iterations N the run is BENCH: closed-loop concurrency on --instances K
// workers (default 1; K graphs are compiled, one per instance), M warmup
// requests per instance, then N measured requests in total -- the seeded
// request sequence is generated once before timing and split across
// instances interleaved, every phase timed per request (h2d / exec =
// execute+sync / d2h / decider / e2e; model_ms = h2d+exec+d2h, e2e spans
// H2D through the read-out).  --aicore-num sets the per-operator AIC|AIV
// core cap via ge::ir_option::AICORE_NUM at GEInitialize: it is a
// compile-time cap and neither reserves nor isolates cores for an
// instance.  --readout enables the CPU read-out decider phase (official
// semantics: softmax over the first K letter columns at the case
// temperature, the readout.json schema startlux-text-readout/1 written by
// generate_inputs.py); without it decider_ms stays 0 and e2e ends at D2H.
//
// records.json (schema startlux-text-records/2): {"file": blob,
// "cases": [{"id", "text_tensors": [input_ids, position_ids]}]}.  B=1 is
// pinned by the graph (io-spec axis 0 static) and re-validated here; the
// laid-out width IS the true length -- the records cannot express padding,
// there is no actual_seq_lengths.  The text AIR carries
// ChunkGatedDeltaRule: the compat prototype .so must be LD_PRELOADed into
// this process (the parent adapter's work/deps artifact); the GDN causal
// conv runs on the standard Pad -> Conv2D -> SiLU route, so no
// causal-conv1d vendor op is involved.
//
// Profiling: wrap the binary with msprof (it injects the PROF environment and
// exports the analysis CSVs itself); no in-process profiling code.
#include <acl/acl.h>
#include <ge/ge_api.h>
#include <graph/graph.h>
#include <exe_graph/runtime/tensor.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

double ElapsedMs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void MakeDirs(const std::string &path) {
    if (!path.empty() && mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("cannot create directory: " + path);
    }
}

bool ToGeDtype(const std::string &s, ge::DataType &dt) {
    if (s == "float32" || s == "float") { dt = ge::DT_FLOAT; return true; }
    if (s == "float16") { dt = ge::DT_FLOAT16; return true; }
    if (s == "bfloat16") { dt = ge::DT_BF16; return true; }
    if (s == "int32") { dt = ge::DT_INT32; return true; }
    if (s == "int64") { dt = ge::DT_INT64; return true; }
    if (s == "bool") { dt = ge::DT_BOOL; return true; }
    if (s == "uint8") { dt = ge::DT_UINT8; return true; }
    return false;
}

std::string FromGeDtype(ge::DataType dt) {
    switch (dt) {
        case ge::DT_FLOAT: return "float32";
        case ge::DT_FLOAT16: return "float16";
        case ge::DT_BF16: return "bfloat16";
        case ge::DT_INT32: return "int32";
        case ge::DT_INT64: return "int64";
        case ge::DT_BOOL: return "bool";
        case ge::DT_UINT8: return "uint8";
        default: return "unknown";
    }
}

struct TensorRef {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t offset = 0, bytes = 0;
    ge::DataType geDtype = ge::DT_UNDEFINED;
};

struct Case {
    std::string id;
    size_t batch = 0;                    // input_ids shape[0]
    std::vector<TensorRef> text;         // 2: input_ids, position_ids
};

// 官方 read-out 语义的逐 case 元数据 (verify_dev._teacher_row_report 同式):
// p = softmax(scores[0..K-1] / temperature), 前 K 列即选项 letters。
struct ReadoutMeta {
    size_t lettersAsked = 0;
    double temperature = 0.0;
    std::vector<std::string> labels;
};

TensorRef ParseTensorRef(const Json &node) {
    TensorRef ref;
    ref.dtype = node.at("dtype").get<std::string>();
    if (!ToGeDtype(ref.dtype, ref.geDtype)) {
        throw std::runtime_error("unsupported records dtype: " + ref.dtype);
    }
    for (const auto &dim : node.at("shape")) ref.shape.push_back(dim.get<int64_t>());
    ref.offset = node.at("offset").get<size_t>();
    ref.bytes = node.at("bytes").get<size_t>();
    return ref;
}

void MakeTensor(std::vector<gert::Tensor> &out, void *dev, size_t allocBytes,
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
    tensor.SetData(gert::TensorData(dev, nullptr, allocBytes, gert::kOnDeviceHbm));
    out.emplace_back(std::move(tensor));
}

std::vector<char> ReadFile(const std::string &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open file: " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(stream)),
                             std::istreambuf_iterator<char>());
}

double Percentile(std::vector<double> &sorted, double quantile) {
    if (sorted.empty()) return 0.0;
    double position = quantile * static_cast<double>(sorted.size() - 1);
    size_t lower = static_cast<size_t>(position);
    size_t upper = std::min(lower + 1, sorted.size() - 1);
    double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

// 分相统计块: avg/p50/p99 (nlohmann 初始化列表按字典序落盘)。
Json PhaseStats(std::vector<double> values) {
    Json out = {{"avg", 0.0}, {"p50", 0.0}, {"p99", 0.0}};
    if (values.empty()) return out;
    double avg = std::accumulate(values.begin(), values.end(), 0.0) /
                 static_cast<double>(values.size());
    std::sort(values.begin(), values.end());
    out["avg"] = avg;
    out["p50"] = Percentile(values, 0.5);
    out["p99"] = Percentile(values, 0.99);
    return out;
}

// 与 runtime/backends/gesession_backend.cpp 的 AicoreSpec 相同的 "AIC|AIV"
// 校验 (executor 是自包含单文件目标, 不链共享后端)。
std::string AicoreSpec(const std::string &raw) {
    if (raw.empty()) return raw;
    auto separator = raw.find('|');
    if (separator == std::string::npos || separator == 0 || separator + 1 == raw.size() ||
        raw.find('|', separator + 1) != std::string::npos ||
        raw.substr(0, separator).find_first_not_of("0123456789") != std::string::npos ||
        raw.substr(separator + 1).find_first_not_of("0123456789") != std::string::npos ||
        std::stoul(raw.substr(0, separator)) == 0 ||
        std::stoul(raw.substr(separator + 1)) == 0) {
        throw std::runtime_error("aicore_num requires an explicit positive AIC|AIV pair");
    }
    return raw;
}

// ---------------------------------------------------------------- the executor

// 每实例一份执行资源: 独立 stream + LoadGraph 绑定 + 输入 device 缓冲, 以及
// gert::Tensor 不可拷贝、故挂在实例内的请求 scratch (工作线程只触达自己的
// 实例, 无共享竞争)。LoadGraph 不支持对同一 graphId 重复加载 (ge_api.h
// 约束), 故 K 路并发就是 K 份 AddGraph+CompileGraph; Session 全局共享
// (与共享后端同构)。
struct Instance {
    uint32_t graphId = 0;
    aclrtStream stream = nullptr;
    std::vector<void *> inputBuffers;
    double loadMs = 0.0;
    std::vector<gert::Tensor> inputs;
    std::vector<gert::Tensor> outputs;
    std::vector<char> hostBytes;
};

class Executor {
public:
    ~Executor() { Destroy(); }

    void Init(int device, const std::string &airPath, int instances,
              const std::string &aicoreNum) {
        std::map<ge::AscendString, ge::AscendString> globalOptions = {
            {ge::AscendString("ge.graphRunMode"), ge::AscendString("0")},
            {ge::AscendString("ge.exec.deviceId"),
             ge::AscendString(std::to_string(device).c_str())},
        };
        std::string aicore = AicoreSpec(aicoreNum);
        std::cout << "[INFO] GEInitialize ge.graphRunMode=0" << std::endl;
        if (!aicore.empty()) {
            globalOptions.emplace(ge::AscendString(ge::ir_option::AICORE_NUM),
                                  ge::AscendString(aicore.c_str()));
            std::cout << "[INFO] GEInitialize " << ge::ir_option::AICORE_NUM
                      << "=" << aicore << std::endl;
        }
        if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
            throw std::runtime_error(std::string("GEInitialize failed: ") +
                                     ge::GEGetErrorMsg().c_str());
        }
        geInited_ = true;
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            throw std::runtime_error("aclInit failed");
        }
        aclInited_ = true;

        std::map<ge::AscendString, ge::AscendString> sessionOptions = {
            {ge::AscendString("ge.session_device_id"),
             ge::AscendString(std::to_string(device).c_str())},
            {ge::AscendString("ge.exec.precision_mode"),
             ge::AscendString("must_keep_origin_dtype")},
        };
        session_ = std::make_unique<ge::Session>(sessionOptions);

        // K 份图: AddGraph+CompileGraph 逐份串行, 且必须全部在 aclrtSetDevice
        // 之前 (与共享后端实测序列一致)。
        instances_.resize(static_cast<size_t>(instances));
        auto compileBegin = Clock::now();
        for (uint32_t gid = 1; gid <= static_cast<uint32_t>(instances); gid++) {
            auto begin = Clock::now();
            ge::Graph graph;
            if (graph.LoadFromFile(airPath.c_str()) != ge::GRAPH_SUCCESS) {
                throw std::runtime_error("cannot load AIR: " + airPath);
            }
            ge::Status st = session_->AddGraph(gid, graph);
            if (st != ge::SUCCESS) {
                throw std::runtime_error("AddGraph failed (gid=" + std::to_string(gid) + ")");
            }
            st = session_->CompileGraph(gid);
            if (st != ge::SUCCESS) {
                throw std::runtime_error(std::string("CompileGraph failed (gid=" +
                                     std::to_string(gid) + "): ") + ge::GEGetErrorMsg().c_str());
            }
            instances_[gid - 1].graphId = gid;
            std::cout << "[INFO] CompileGraph ok (gid=" << gid << "/" << instances
                      << "), cost " << ElapsedMs(begin, Clock::now()) << " ms" << std::endl;
        }
        if (instances > 1) {
            std::cout << "[INFO] " << instances << " 份图实例编译总耗时 "
                      << ElapsedMs(compileBegin, Clock::now()) << " ms" << std::endl;
        }

        if (aclrtSetDevice(device) != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSetDevice failed");
        }
        deviceSet_ = true;
        if (aclrtGetCurrentContext(&context_) != ACL_SUCCESS) {
            throw std::runtime_error("aclrtGetCurrentContext failed");
        }
        hbmBaseFreeMb_ = HbmFreeMb();

        for (size_t i = 0; i < instances_.size(); i++) {
            Instance &ctx = instances_[i];
            if (aclrtCreateStream(&ctx.stream) != ACL_SUCCESS) {
                throw std::runtime_error("aclrtCreateStream failed");
            }
            auto begin = Clock::now();
            ge::Status st = session_->LoadGraph(ctx.graphId, {}, ctx.stream);
            if (st != ge::SUCCESS) {
                throw std::runtime_error(std::string("LoadGraph failed (gid=" +
                                     std::to_string(ctx.graphId) + "): ") +
                                     ge::GEGetErrorMsg().c_str());
            }
            ctx.loadMs = ElapsedMs(begin, Clock::now());
            std::cout << "[INFO] LoadGraph ok (gid=" << ctx.graphId << ", instance=" << i
                      << "), cost " << ctx.loadMs << " ms" << std::endl;
        }
    }

    // Pre-allocate the 2 host-fed input buffers per instance at the max byte
    // size across cases; the graph reads them in the fixed order input_ids,
    // position_ids.
    void Allocate(const std::vector<Case> &cases) {
        inputSizes_.assign(2, 0);
        for (const auto &one : cases) {
            for (size_t i = 0; i < one.text.size() && i < 2; i++) {
                inputSizes_[i] = std::max(inputSizes_[i], one.text[i].bytes);
            }
        }
        for (auto &ctx : instances_) {
            ctx.inputBuffers.clear();
            for (size_t i = 0; i < 2; i++) {
                void *dev = nullptr;
                if (inputSizes_[i] > 0 &&
                    aclrtMalloc(&dev, inputSizes_[i],
                                ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                    throw std::runtime_error("aclrtMalloc failed for input buffer");
                }
                ctx.inputBuffers.push_back(dev);
            }
        }
        hbmAfterFreeMb_ = HbmFreeMb();
    }

    // 一次完整请求, 全部相位在调用线程内计时。meta 非空时执行 CPU read-out
    // decider 并计入 e2e; 为空时 e2e 边界到 D2H 完成, decider 记 0。
    bool Run(size_t inst, const Case &one, const std::vector<char> &blob,
             const ReadoutMeta *meta, double *h2dMs, double *execMs, double *d2hMs,
             double *deciderMs, double *e2eMs, double *sink) {
        Instance &ctx = instances_[inst];
        auto t0 = Clock::now();
        std::vector<gert::Tensor> &inputs = ctx.inputs;
        inputs.clear();
        for (size_t i = 0; i < one.text.size() && i < ctx.inputBuffers.size(); i++) {
            const TensorRef &ref = one.text[i];
            aclError copyRet = aclrtMemcpy(ctx.inputBuffers[i], inputSizes_[i],
                                           blob.data() + ref.offset, ref.bytes,
                                           ACL_MEMCPY_HOST_TO_DEVICE);
            if (copyRet != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] text H2D failed (input %zu, ret=%d)\n", i,
                        static_cast<int>(copyRet));
                return false;
            }
            MakeTensor(inputs, ctx.inputBuffers[i], inputSizes_[i], ref.shape, ref.geDtype);
        }
        auto t1 = Clock::now();
        std::vector<gert::Tensor> &outputs = ctx.outputs;
        outputs.clear();
        if (session_->ExecuteGraphWithStreamAsync(ctx.graphId, ctx.stream, inputs,
                                                   outputs) != ge::SUCCESS ||
            aclrtSynchronizeStream(ctx.stream) != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] ExecuteGraphWithStreamAsync failed\n");
            return false;
        }
        auto t2 = Clock::now();
        const gert::Tensor &scores = ctx.outputs[0];
        size_t bytes = scores.GetSize();
        std::vector<char> &host = ctx.hostBytes;
        host.assign(bytes, 0);
        if (bytes > 0 &&
            aclrtMemcpy(host.data(), bytes, scores.GetAddr(), bytes,
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] scores D2H failed\n");
            return false;
        }
        auto t3 = Clock::now();
        double topProbability = 0.0;
        if (meta != nullptr) {
            // decider 契约: scores 必须是 [1,26] float32 的 letter logits。
            if (bytes != 26 * sizeof(float) ||
                scores.GetDataType() != ge::DT_FLOAT) {
                fprintf(stderr, "[ERROR] decider needs [1,26] float32 scores\n");
                return false;
            }
            const float *row = reinterpret_cast<const float *>(host.data());
            size_t k = std::min(meta->lettersAsked, size_t{26});
            double scale = 1.0 / meta->temperature;
            double top = row[0] * scale;
            for (size_t j = 1; j < k; j++) top = std::max(top, row[j] * scale);
            double denominator = 0.0;
            std::vector<double> probabilities(k);
            for (size_t j = 0; j < k; j++) {
                probabilities[j] = std::exp(row[j] * scale - top);
                denominator += probabilities[j];
            }
            size_t answer = 0;
            for (size_t j = 0; j < k; j++) {
                probabilities[j] /= denominator;
                if (probabilities[j] > probabilities[answer]) answer = j;
            }
            topProbability = probabilities[answer];
        }
        auto t4 = Clock::now();
        *h2dMs = ElapsedMs(t0, t1);
        *execMs = ElapsedMs(t1, t2);
        *d2hMs = ElapsedMs(t2, t3);
        *deciderMs = ElapsedMs(t3, t4);
        *e2eMs = ElapsedMs(t0, t4);
        if (meta != nullptr) *sink = topProbability;
        return true;
    }

    // 工作线程入口: ACL 上下文创建于主线程, 每个实例线程须先绑定才能
    // memcpy/execute (与共享后端的 threadEnter 同构)。
    bool BindThread() const {
        return aclrtSetCurrentContext(context_) == ACL_SUCCESS;
    }

    const gert::Tensor &Scores(size_t inst) const { return instances_[inst].outputs[0]; }
    const std::vector<char> &HostScores(size_t inst) const {
        return instances_[inst].hostBytes;
    }
    double HbmBaseFreeMb() const { return hbmBaseFreeMb_; }
    double HbmAfterFreeMb() const { return hbmAfterFreeMb_; }

    double HbmFreeMb() const {
        size_t freeBytes = 0, totalBytes = 0;
        if (aclrtGetMemInfo(ACL_HBM_MEM, &freeBytes, &totalBytes) != ACL_SUCCESS) {
            return -1.0;
        }
        return static_cast<double>(freeBytes) / (1024.0 * 1024.0);
    }

    void Destroy() {
        for (auto &ctx : instances_) {
            for (void *buffer : ctx.inputBuffers) {
                if (buffer != nullptr) aclrtFree(buffer);
            }
            ctx.inputBuffers.clear();
            if (ctx.stream != nullptr) {
                aclrtDestroyStream(ctx.stream);
                ctx.stream = nullptr;
            }
            ctx.inputs.clear();
            ctx.outputs.clear();
            ctx.hostBytes.clear();
        }
        instances_.clear();
        session_.reset();
        if (deviceSet_) {
            aclrtResetDevice(device_);
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
    int device_ = 0;
    bool geInited_ = false, aclInited_ = false, deviceSet_ = false;
    aclrtContext context_ = nullptr;
    std::unique_ptr<ge::Session> session_;
    std::vector<Instance> instances_;
    std::vector<size_t> inputSizes_;
    double hbmBaseFreeMb_ = -1.0, hbmAfterFreeMb_ = -1.0;
};

void Usage() {
    std::cout <<
        "usage: executor [--air work/build/text.air] [--records work/inputs/records.json]\n"
        "                [--output-dir work/output] [--case-filter <id>]...\n"
        "                [--iterations N] [--warmup M] [--seed S]\n"
        "                [--instances K] [--aicore-num A|B] [--readout <readout.json>]\n"
        "  default (no --iterations): infer over every case, outputs.bin + result.json\n"
        "  with --iterations N:        bench, N sampled requests over K closed-loop\n"
        "                             instances, per-phase performance + requests.csv\n";
}

struct Sample {
    int instance = 0;
    size_t req = 0;               // 全局测量序列下标
    size_t caseIndex = 0;
    double h2d = 0.0, exec = 0.0, d2h = 0.0, decider = 0.0, model = 0.0, e2e = 0.0;
    double sink = 0.0;            // read-out top 概率累加项 (证明 decider 真实执行)
};

}  // namespace

int main(int argc, char **argv) {
    try {
        std::string airPath = "work/build/text.air";
        std::string recordsPath = "work/inputs/records.json";
        std::string outputDir = "work/output";
        std::string aicoreNum, readoutPath;
        std::set<std::string> caseFilter;
        bool bench = false, warmupGiven = false, seedGiven = false;
        bool instancesGiven = false, readoutGiven = false;
        long iterations = 0, warmup = 0, instances = 1;
        unsigned long long seed = 42;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            auto value = [&](const char *name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string(name) + " needs a value");
                }
                return argv[++i];
            };
            if (arg == "--air") {
                airPath = value("--air");
            } else if (arg == "--records") {
                recordsPath = value("--records");
            } else if (arg == "--output-dir") {
                outputDir = value("--output-dir");
            } else if (arg == "--case-filter") {
                caseFilter.insert(value("--case-filter"));
            } else if (arg == "--iterations") {
                iterations = std::stol(value("--iterations"));
                bench = true;
            } else if (arg == "--warmup") {
                warmup = std::stol(value("--warmup"));
                warmupGiven = true;
            } else if (arg == "--seed") {
                seed = std::stoull(value("--seed"));
                seedGiven = true;
            } else if (arg == "--instances") {
                instances = std::stol(value("--instances"));
                instancesGiven = true;
            } else if (arg == "--aicore-num") {
                aicoreNum = value("--aicore-num");
            } else if (arg == "--readout") {
                readoutPath = value("--readout");
                readoutGiven = true;
            } else if (arg == "--help" || arg == "-h") {
                Usage();
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }
        if (bench && iterations < 1) {
            throw std::runtime_error("--iterations needs a value >= 1");
        }
        if (!bench && (warmupGiven || seedGiven)) {
            throw std::runtime_error("--warmup/--seed require --iterations");
        }
        if (!bench && instancesGiven) {
            throw std::runtime_error("--instances requires --iterations");
        }
        if (!bench && readoutGiven) {
            throw std::runtime_error("--readout requires --iterations");
        }
        if (instances < 1) {
            throw std::runtime_error("--instances needs a value >= 1");
        }
        AicoreSpec(aicoreNum);

        Json records = Json::parse(ReadFile(recordsPath));
        if (records.at("schema").get<std::string>() != "startlux-text-records/2") {
            throw std::runtime_error(recordsPath + " is not the text record index "
                                     "(startlux-text-records/2; regenerate "
                                     "work/inputs with generate_inputs.py)");
        }
        std::vector<char> blob = ReadFile(records.at("file").get<std::string>());
        static const char *kTensorNames[2] = {"input_ids", "position_ids"};
        std::vector<Case> cases;
        for (const auto &entry : records.at("cases")) {
            Case one;
            one.id = entry.at("id").get<std::string>();
            const auto &tensors = entry.at("text_tensors");
            if (tensors.size() != 2) {
                throw std::runtime_error("case " + one.id + " must carry exactly "
                                         "2 text tensors");
            }
            for (size_t i = 0; i < 2; i++) {
                if (tensors[i].at("name").get<std::string>() != kTensorNames[i]) {
                    throw std::runtime_error("case " + one.id + " tensor " +
                                             std::to_string(i) + " must be " + kTensorNames[i]);
                }
                one.text.push_back(ParseTensorRef(tensors[i]));
            }
            if (one.text[0].geDtype != ge::DT_INT64 || one.text[1].geDtype != ge::DT_INT64) {
                throw std::runtime_error("case " + one.id + " carries unexpected "
                                         "dtypes (expected int64/int64)");
            }
            const std::vector<int64_t> &ids = one.text[0].shape;
            const std::vector<int64_t> &pos = one.text[1].shape;
            if (ids.size() != 2 || ids[0] != 1 || ids[1] < 1) {
                throw std::runtime_error("case " + one.id + " input_ids must be "
                                         "[1, S] (B is pinned to 1 by the graph)");
            }
            one.batch = 1;
            if (pos.size() != 3 || pos[0] != 3 || pos[1] != 1 || pos[2] != ids[1]) {
                throw std::runtime_error("case " + one.id + " position_ids must be "
                                         "[3, 1, S] of the input_ids width");
            }
            for (const auto &ref : one.text) {
                if (ref.offset + ref.bytes > blob.size()) {
                    throw std::runtime_error("case " + one.id + " tensor " + ref.dtype +
                                             " exceeds the inputs blob");
                }
            }
            cases.push_back(std::move(one));
        }
        if (cases.empty()) throw std::runtime_error("no cases in the record pool");
        if (!caseFilter.empty()) {
            std::vector<Case> filtered;
            for (auto &one : cases) {
                if (caseFilter.count(one.id)) {
                    filtered.push_back(std::move(one));
                }
            }
            if (filtered.empty()) {
                throw std::runtime_error("case_filter selected no cases");
            }
            cases = std::move(filtered);
        }

        // readout 元数据: 只对过滤后的 case 集合对齐, 缺失直接报错。
        bool readoutEnabled = false;
        std::vector<ReadoutMeta> readouts;
        if (readoutGiven) {
            Json readoutJson = Json::parse(ReadFile(readoutPath));
            if (readoutJson.at("schema").get<std::string>() != "startlux-text-readout/1") {
                throw std::runtime_error(readoutPath + " is not the readout metadata "
                                         "(startlux-text-readout/1; regenerate "
                                         "work/inputs with generate_inputs.py)");
            }
            std::map<std::string, ReadoutMeta> byId;
            for (const auto &entry : readoutJson.at("cases")) {
                ReadoutMeta meta;
                meta.temperature = entry.at("temperature").get<double>();
                meta.lettersAsked = entry.at("letters_asked").get<size_t>();
                for (const auto &label : entry.at("labels")) {
                    meta.labels.push_back(label.get<std::string>());
                }
                if (!(meta.temperature > 0.0) || !std::isfinite(meta.temperature) ||
                    meta.labels.empty() || meta.labels.size() != meta.lettersAsked ||
                    meta.lettersAsked > 26 ||
                    meta.lettersAsked < 2) {
                    throw std::runtime_error("invalid readout metadata for case " +
                                             entry.at("id").get<std::string>());
                }
                byId[entry.at("id").get<std::string>()] = std::move(meta);
            }
            for (const auto &one : cases) {
                auto found = byId.find(one.id);
                if (found == byId.end()) {
                    throw std::runtime_error("readout metadata misses case " + one.id);
                }
                readouts.push_back(std::move(found->second));
            }
            readoutEnabled = true;
            std::cout << "[INFO] readout decider enabled: " << readoutPath
                      << " (" << cases.size() << " cases)" << std::endl;
        }

        MakeDirs(outputDir);
        Executor executor;
        executor.Init(0, airPath, static_cast<int>(instances), aicoreNum);
        executor.Allocate(cases);

        Json result = {{"status", "success"}, {"backend", "executor"}};
        if (!bench) {
            result["mode"] = "infer";
            std::ofstream binary(outputDir + "/outputs.bin", std::ios::binary);
            if (!binary) throw std::runtime_error("cannot open output binary");
            Json outputs = Json::object();
            for (const auto &one : cases) {
                double h2dMs = 0.0, execMs = 0.0, d2hMs = 0.0, deciderMs = 0.0, e2eMs = 0.0,
                       sink = 0.0;
                if (!executor.Run(0, one, blob, nullptr, &h2dMs, &execMs, &d2hMs,
                                  &deciderMs, &e2eMs, &sink)) {
                    throw std::runtime_error("execution failed on case " + one.id);
                }
                const gert::Tensor &scores = executor.Scores(0);
                size_t bytes = scores.GetSize();
                const auto &shape = scores.GetShape().GetStorageShape();
                std::vector<int64_t> dims;
                for (size_t d = 0; d < shape.GetDimNum(); d++) dims.push_back(shape.GetDim(d));
                size_t offset = static_cast<size_t>(binary.tellp());
                const std::vector<char> &host = executor.HostScores(0);
                binary.write(host.data(), host.size());
                outputs[one.id] = Json::array({Json({
                    {"instance", 0},
                    {"tensors", Json::array({{{"name", "scores"},
                                              {"dtype", FromGeDtype(scores.GetDataType())},
                                              {"shape", dims},
                                              {"offset", offset},
                                              {"bytes", bytes}}})},
                })});
                std::cout << "[INFO] case " << one.id << " scores shape=["
                          << (dims.size() > 0 ? std::to_string(dims[0]) : "") << ","
                          << (dims.size() > 1 ? std::to_string(dims[1]) : "") << "] text "
                          << (h2dMs + execMs + d2hMs) << " ms" << std::endl;
            }
            binary.close();
            result["outputs"] = outputs;
        } else {
            result["mode"] = "bench";

            // 全局 seed 先生成测量序列再分发; warmup 用独立同 seed 采样器,
            // 不消耗测量序列。每实例预热先覆盖全部 distinct shape 再随机补齐。
            std::vector<size_t> reps;
            std::set<int64_t> seenShapes;
            for (size_t i = 0; i < cases.size(); i++) {
                if (seenShapes.insert(cases[i].text[0].shape[1]).second) {
                    reps.push_back(i);
                }
            }
            std::vector<std::vector<size_t>> warmSequences(static_cast<size_t>(instances),
                                                           reps);
            {
                std::mt19937_64 warmRng(seed);
                std::uniform_int_distribution<size_t> warmPick(0, cases.size() - 1);
                for (auto &sequence : warmSequences) {
                    while (sequence.size() < static_cast<size_t>(warmup)) {
                        sequence.push_back(warmPick(warmRng));
                    }
                }
            }
            std::vector<size_t> sequence(static_cast<size_t>(iterations));
            {
                std::mt19937_64 rng(seed);
                std::uniform_int_distribution<size_t> pick(0, cases.size() - 1);
                for (auto &index : sequence) index = pick(rng);
            }
            size_t perInstance = static_cast<size_t>(iterations) /
                                 static_cast<size_t>(instances);
            size_t remainder = static_cast<size_t>(iterations) %
                               static_cast<size_t>(instances);

            std::atomic<size_t> errors{0};
            std::mutex gateMutex;
            std::condition_variable gateCv;
            size_t ready = 0;
            bool startMeasure = false, warmFailed = false;
            std::vector<std::vector<Sample>> per(static_cast<size_t>(instances));
            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(instances));
            for (int t = 0; t < instances; t++) {
                workers.emplace_back([&, t]() {
                    std::vector<Sample> &mine = per[static_cast<size_t>(t)];
                    size_t count = perInstance +
                                   (static_cast<size_t>(t) < remainder ? 1 : 0);
                    mine.reserve(count);
                    bool warmOk = executor.BindThread();
                    if (!warmOk) {
                        fprintf(stderr, "[ERROR] bind context failed (instance=%d)\n", t);
                    }
                    try {
                        for (size_t idx : warmSequences[static_cast<size_t>(t)]) {
                            double h2dMs = 0.0, execMs = 0.0, d2hMs = 0.0, deciderMs = 0.0,
                                   e2eMs = 0.0, sink = 0.0;
                            if (!executor.Run(static_cast<size_t>(t), cases[idx], blob,
                                              readoutEnabled ? &readouts[idx] : nullptr,
                                              &h2dMs, &execMs, &d2hMs, &deciderMs,
                                              &e2eMs, &sink)) {
                                warmOk = false;
                                break;
                            }
                        }
                    } catch (const std::exception &error) {
                        fprintf(stderr, "[ERROR] warmup instance=%d: %s\n", t, error.what());
                        warmOk = false;
                    } catch (...) {
                        fprintf(stderr, "[ERROR] warmup instance=%d: unknown exception\n", t);
                        warmOk = false;
                    }
                    // 全部实例预热完成后统一开表 (同一工作线程预热与测量)。
                    {
                        std::unique_lock<std::mutex> lock(gateMutex);
                        warmFailed = warmFailed || !warmOk;
                        ++ready;
                        gateCv.notify_all();
                        gateCv.wait(lock, [&]() { return startMeasure; });
                    }
                    if (warmFailed) return;
                    for (size_t i = 0; i < count; i++) {
                        size_t global = static_cast<size_t>(t) +
                                        i * static_cast<size_t>(instances);
                        size_t caseIndex = sequence[global];
                        double h2dMs = 0.0, execMs = 0.0, d2hMs = 0.0, deciderMs = 0.0,
                               e2eMs = 0.0, sink = 0.0;
                        if (executor.Run(static_cast<size_t>(t), cases[caseIndex], blob,
                                         readoutEnabled ? &readouts[caseIndex] : nullptr,
                                         &h2dMs, &execMs, &d2hMs, &deciderMs, &e2eMs,
                                         &sink)) {
                            mine.push_back({t, global, caseIndex, h2dMs, execMs, d2hMs,
                                            deciderMs, h2dMs + execMs + d2hMs, e2eMs, sink});
                        } else {
                            errors.fetch_add(1);
                            fprintf(stderr, "[ERROR] instance=%d request %zu failed\n",
                                    t, global);
                            break;
                        }
                    }
                });
            }
            Clock::time_point measureBegin;
            {
                std::unique_lock<std::mutex> lock(gateMutex);
                gateCv.wait(lock, [&]() {
                    return ready == static_cast<size_t>(instances);
                });
                if (warmFailed) {
                    startMeasure = true;
                    gateCv.notify_all();
                    lock.unlock();
                    for (auto &worker : workers) worker.join();
                    throw std::runtime_error("warmup execution failed");
                }
                measureBegin = Clock::now();
                startMeasure = true;
                gateCv.notify_all();
            }
            for (auto &worker : workers) worker.join();
            double wallMs = ElapsedMs(measureBegin, Clock::now());
            double seconds = wallMs / 1000.0;

            // 汇总: 成功样本逐相位统计, 失败只计数。
            std::vector<Sample> all;
            all.reserve(perInstance * static_cast<size_t>(instances) + remainder);
            size_t rows = 0;
            double sinkSum = 0.0;
            Json perInstanceStats = Json::array();
            for (int t = 0; t < instances; t++) {
                const std::vector<Sample> &mine = per[static_cast<size_t>(t)];
                std::vector<double> e2e, exec;
                e2e.reserve(mine.size());
                exec.reserve(mine.size());
                for (const auto &sample : mine) {
                    e2e.push_back(sample.e2e);
                    exec.push_back(sample.exec);
                    rows += cases[sample.caseIndex].batch;
                    sinkSum += sample.sink;
                }
                perInstanceStats.push_back({
                    {"instance", t},
                    {"requests", mine.size()},
                    {"errors", perInstance + (static_cast<size_t>(t) < remainder ? 1 : 0) -
                               mine.size()},
                    {"qps", seconds > 0 ? mine.size() / seconds : 0.0},
                    {"e2e_ms", PhaseStats(e2e)},
                    {"exec_ms", PhaseStats(exec)},
                });
                all.insert(all.end(), mine.begin(), mine.end());
            }
            std::sort(all.begin(), all.end(), [](const Sample &a, const Sample &b) {
                return a.req < b.req;
            });
            std::vector<double> e2e, model, exec, h2d, d2h, decider;
            e2e.reserve(all.size());
            model.reserve(all.size());
            exec.reserve(all.size());
            h2d.reserve(all.size());
            d2h.reserve(all.size());
            decider.reserve(all.size());
            for (const auto &sample : all) {
                e2e.push_back(sample.e2e);
                model.push_back(sample.model);
                exec.push_back(sample.exec);
                h2d.push_back(sample.h2d);
                d2h.push_back(sample.d2h);
                decider.push_back(sample.decider);
            }
            Json perCase = Json::object();
            std::vector<std::vector<double>> byCase(cases.size());
            for (const auto &sample : all) byCase[sample.caseIndex].push_back(sample.e2e);
            for (size_t i = 0; i < cases.size(); i++) {
                if (byCase[i].empty()) continue;
                std::sort(byCase[i].begin(), byCase[i].end());
                perCase[cases[i].id] = {{"p50", Percentile(byCase[i], 0.5)},
                                        {"p99", Percentile(byCase[i], 0.99)},
                                        {"requests", byCase[i].size()}};
            }

            result["performance"] = {
                {"qps", seconds > 0 ? all.size() / seconds : 0.0},
                {"requests", all.size()},
                {"rows", rows},
                {"samples_per_second", seconds > 0 ? rows / seconds : 0.0},
                {"errors", errors.load()},
                {"wall_seconds", seconds},
                {"instances", instances},
                {"aicore_num", aicoreNum.empty() ? Json(nullptr) : Json(aicoreNum)},
                {"seed", seed},
                {"warmup", warmup},
                {"iterations", iterations},
                {"decider", readoutEnabled},
                {"arrival_model", "closed_loop"},
                {"e2e_ms", PhaseStats(e2e)},
                {"model_ms", PhaseStats(model)},
                {"exec_ms", PhaseStats(exec)},
                {"h2d_ms", PhaseStats(h2d)},
                {"d2h_ms", PhaseStats(d2h)},
                {"decider_ms", PhaseStats(decider)},
                {"per_case", perCase},
                {"per_instance", perInstanceStats},
                {"hbm_base_free_mb", executor.HbmBaseFreeMb()},
                {"hbm_after_load_free_mb", executor.HbmAfterFreeMb()},
                {"decider_sink_sum", sinkSum},
            };
            std::ofstream csv(outputDir + "/requests.csv");
            csv << "instance,req,case,samples,h2d_ms,exec_ms,d2h_ms,decider_ms,"
                   "model_ms,e2e_ms,error\n";
            for (const auto &sample : all) {
                csv << sample.instance << "," << sample.req << ","
                    << cases[sample.caseIndex].id << ","
                    << cases[sample.caseIndex].batch << ","
                    << sample.h2d << "," << sample.exec << "," << sample.d2h << ","
                    << sample.decider << "," << sample.model << "," << sample.e2e
                    << ",0\n";
            }
            csv.close();
        }
        std::ofstream jsonOut(outputDir + "/result.json");
        if (!jsonOut) throw std::runtime_error("cannot write result.json");
        jsonOut << result.dump(1) << "\n" << std::endl;
        jsonOut.close();
        std::cout << "GE_RESULT " << result.dump() << std::endl;
        // Leave via _exit after the explicit teardown: everything real
        // (acl/GE finalize, output writes) is done by then, and the remaining
        // C++ static destruction (dlopened CANN vendor libs + GE registry)
        // reliably corrupts the heap on this build.  _exit skips stack
        // unwinding, so run the destructor by hand once here; the error path
        // keeps the normal exit.
        std::cout.flush();
        std::cerr.flush();
        executor.~Executor();
        std::fflush(nullptr);
        _exit(0);
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
