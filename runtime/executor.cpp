// Minimal single-graph executor for the text-only pipeline: loads ONE text
// AIR, feeds the 2 positional inputs (input_ids, position_ids) of every
// case from the packed pool written by generate_inputs.py, and writes the
// scores.
//
// Usage (all paths default relative to the current directory; run from the
// adapter root for the canonical defaults):
//   executor [--air work/build/text.air] [--records work/inputs/records.json]
//            [--output-dir work/output] [--case-filter <id>]...
//            [--iterations N] [--warmup M] [--seed S]
//
// Without --iterations the run is INFER: every case (or the --case-filter
// subset) executes once, in order, and work/output gets outputs.bin +
// result.json (the ge-compare/1 layout tools/compare.py reads).  With
// --iterations N the run is BENCH: M warmup requests, then N measured
// requests uniformly sampled over the pool (seeded), reported as the
// performance block + requests.csv; --warmup/--seed require --iterations.
//
// records.json (schema startlux-text-records/2): {"file": blob,
// "cases": [{"id", "text_tensors": [input_ids, position_ids]}]}.  B=1 is
// pinned by the graph (io-spec axis 0 static) and re-validated here; the
// laid-out width IS the true length -- the records cannot express padding,
// there is no actual_seq_lengths.  The text AIR carries
// ChunkGatedDeltaRule: the
// compat prototype .so must be LD_PRELOADed into this process (the parent
// adapter's work/deps artifact); the GDN causal conv runs on the standard
// Pad -> Conv2D -> SiLU route, so no causal-conv1d vendor op is involved.
//
// Profiling: wrap the binary with msprof (it injects the PROF environment and
// exports the analysis CSVs itself); no in-process profiling code.
#include <acl/acl.h>
#include <ge/ge_api.h>
#include <graph/graph.h>
#include <exe_graph/runtime/tensor.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
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

// ---------------------------------------------------------------- the executor

class Executor {
public:
    ~Executor() { Destroy(); }

    void Init(int device, const std::string &airPath) {
        std::map<ge::AscendString, ge::AscendString> globalOptions = {
            {ge::AscendString("ge.graphRunMode"), ge::AscendString("0")},
            {ge::AscendString("ge.exec.deviceId"),
             ge::AscendString(std::to_string(device).c_str())},
        };
        std::cout << "[INFO] GEInitialize ge.graphRunMode=0" << std::endl;
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

        // CompileGraph must run before aclrtSetDevice (same sequence as the
        // reference runtime).
        auto begin = Clock::now();
        ge::Graph graph;
        if (graph.LoadFromFile(airPath.c_str()) != ge::GRAPH_SUCCESS) {
            throw std::runtime_error("cannot load AIR: " + airPath);
        }
        ge::Status st = session_->AddGraph(graphId_, graph);
        if (st != ge::SUCCESS) {
            throw std::runtime_error("AddGraph failed");
        }
        st = session_->CompileGraph(graphId_);
        if (st != ge::SUCCESS) {
            throw std::runtime_error(std::string("CompileGraph failed: ") +
                                     ge::GEGetErrorMsg().c_str());
        }
        std::cout << "[INFO] CompileGraph ok (gid=" << graphId_ << "), cost "
                  << ElapsedMs(begin, Clock::now()) << " ms" << std::endl;

        if (aclrtSetDevice(device) != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSetDevice failed");
        }
        deviceSet_ = true;
        if (aclrtCreateStream(&stream_) != ACL_SUCCESS) {
            throw std::runtime_error("aclrtCreateStream failed");
        }
        begin = Clock::now();
        st = session_->LoadGraph(graphId_, {}, stream_);
        if (st != ge::SUCCESS) {
            throw std::runtime_error(std::string("LoadGraph failed: ") +
                                     ge::GEGetErrorMsg().c_str());
        }
        std::cout << "[INFO] LoadGraph ok (gid=" << graphId_ << "), cost "
                  << ElapsedMs(begin, Clock::now()) << " ms" << std::endl;
    }

    // Pre-allocate the 2 host-fed input buffers at the max byte size across
    // cases; the graph reads them in the fixed order input_ids,
    // position_ids.
    void Allocate(const std::vector<Case> &cases) {
        inputSizes_.assign(2, 0);
        for (const auto &one : cases) {
            for (size_t i = 0; i < one.text.size() && i < 2; i++) {
                inputSizes_[i] = std::max(inputSizes_[i], one.text[i].bytes);
            }
        }
        for (size_t i = 0; i < 2; i++) {
            void *dev = nullptr;
            if (inputSizes_[i] > 0 &&
                aclrtMalloc(&dev, inputSizes_[i],
                            ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMalloc failed for input buffer");
            }
            inputBuffers_.push_back(dev);
        }
    }

    bool Run(const Case &one, const std::vector<char> &blob, double *textMs) {
        auto begin = Clock::now();
        inputs_.clear();
        for (size_t i = 0; i < one.text.size(); i++) {
            const TensorRef &ref = one.text[i];
            if (aclrtMemcpy(inputBuffers_[i], inputSizes_[i],
                            blob.data() + ref.offset, ref.bytes,
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] text H2D failed (input %zu)\n", i);
                return false;
            }
            MakeTensor(inputs_, inputBuffers_[i], inputSizes_[i], ref.shape, ref.geDtype);
        }
        outputs_.clear();
        if (session_->ExecuteGraphWithStreamAsync(graphId_, stream_, inputs_,
                                                  outputs_) != ge::SUCCESS ||
            aclrtSynchronizeStream(stream_) != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] ExecuteGraphWithStreamAsync failed\n");
            return false;
        }
        *textMs = ElapsedMs(begin, Clock::now());
        return true;
    }

    const gert::Tensor &Scores() const { return outputs_[0]; }

    void Destroy() {
        for (void *buffer : inputBuffers_) {
            if (buffer != nullptr) aclrtFree(buffer);
        }
        inputBuffers_.clear();
        inputs_.clear();
        outputs_.clear();
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
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
    std::unique_ptr<ge::Session> session_;
    uint32_t graphId_ = 1;
    aclrtStream stream_ = nullptr;
    std::vector<void *> inputBuffers_;
    std::vector<size_t> inputSizes_;
    std::vector<gert::Tensor> inputs_;
    std::vector<gert::Tensor> outputs_;
};

void Usage() {
    std::cout <<
        "usage: executor [--air work/build/text.air] [--records work/inputs/records.json]\n"
        "                [--output-dir work/output] [--case-filter <id>]...\n"
        "                [--iterations N] [--warmup M] [--seed S]\n"
        "  default (no --iterations): infer over every case, outputs.bin + result.json\n"
        "  with --iterations N:        bench, N sampled requests, performance + requests.csv\n";
}

}  // namespace

int main(int argc, char **argv) {
    try {
        std::string airPath = "work/build/text.air";
        std::string recordsPath = "work/inputs/records.json";
        std::string outputDir = "work/output";
        std::set<std::string> caseFilter;
        bool bench = false, warmupGiven = false, seedGiven = false;
        long iterations = 0, warmup = 0;
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
                                             std::to_string(i) + " must be " +
                                             kTensorNames[i]);
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

        MakeDirs(outputDir);
        Executor executor;
        executor.Init(0, airPath);
        executor.Allocate(cases);

        Json result = {{"status", "success"}, {"backend", "executor"}};
        if (!bench) {
            result["mode"] = "infer";
            std::ofstream binary(outputDir + "/outputs.bin", std::ios::binary);
            if (!binary) throw std::runtime_error("cannot open output binary");
            Json outputs = Json::object();
            for (const auto &one : cases) {
                double textMs = 0.0;
                if (!executor.Run(one, blob, &textMs)) {
                    throw std::runtime_error("execution failed on case " + one.id);
                }
                const gert::Tensor &scores = executor.Scores();
                size_t bytes = scores.GetSize();
                const auto &shape = scores.GetShape().GetStorageShape();
                std::vector<int64_t> dims;
                for (size_t d = 0; d < shape.GetDimNum(); d++) dims.push_back(shape.GetDim(d));
                size_t offset = static_cast<size_t>(binary.tellp());
                std::vector<char> host(bytes);
                if (bytes > 0 &&
                    aclrtMemcpy(host.data(), bytes, scores.GetAddr(), bytes,
                                ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                    throw std::runtime_error("scores D2H failed on case " + one.id);
                }
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
                          << textMs << " ms" << std::endl;
            }
            binary.close();
            result["outputs"] = outputs;
        } else {
            result["mode"] = "bench";
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<size_t> pick(0, cases.size() - 1);
            for (long i = 0; i < warmup; i++) {
                double textMs = 0.0;
                if (!executor.Run(cases[pick(rng)], blob, &textMs)) {
                    throw std::runtime_error("warmup execution failed");
                }
            }
            struct Sample { size_t caseIndex; double e2e, text; };
            std::vector<Sample> samples;
            samples.reserve(iterations);
            size_t rows = 0, errors = 0;
            auto measureBegin = Clock::now();
            for (long i = 0; i < iterations; i++) {
                size_t caseIndex = pick(rng);
                double textMs = 0.0;
                if (executor.Run(cases[caseIndex], blob, &textMs)) {
                    samples.push_back({caseIndex, textMs, textMs});
                    rows += cases[caseIndex].batch;
                } else {
                    errors++;
                }
            }
            double wallMs = ElapsedMs(measureBegin, Clock::now());

            std::vector<double> e2e;
            for (const auto &sample : samples) e2e.push_back(sample.e2e);
            std::sort(e2e.begin(), e2e.end());
            Json perCase = Json::object();
            std::vector<std::vector<double>> byCase(cases.size());
            for (const auto &sample : samples) byCase[sample.caseIndex].push_back(sample.e2e);
            for (size_t i = 0; i < cases.size(); i++) {
                if (byCase[i].empty()) continue;
                std::sort(byCase[i].begin(), byCase[i].end());
                perCase[cases[i].id] = {{"p50", Percentile(byCase[i], 0.5)},
                                        {"p99", Percentile(byCase[i], 0.99)},
                                        {"requests", byCase[i].size()}};
            }
            double seconds = wallMs / 1000.0;
            result["performance"] = {
                {"qps", samples.empty() ? 0.0 : samples.size() / seconds},
                {"requests", samples.size()},
                {"rows", rows},
                {"samples_per_second", seconds > 0 ? rows / seconds : 0.0},
                {"errors", errors},
                {"e2e_ms", {{"p50", Percentile(e2e, 0.5)}, {"p99", Percentile(e2e, 0.99)},
                            {"avg", e2e.empty() ? 0.0
                                 : std::accumulate(e2e.begin(), e2e.end(), 0.0) / e2e.size()}}},
                {"wall_seconds", seconds},
                {"per_case", perCase},
                {"arrival_model", "closed_loop"},
            };
            std::ofstream csv(outputDir + "/requests.csv");
            csv << "instance,req,case,samples,text_ms,e2e_ms,error\n";
            for (size_t i = 0; i < samples.size(); i++) {
                csv << 0 << "," << i << "," << cases[samples[i].caseIndex].id << ","
                    << cases[samples[i].caseIndex].batch << ","
                    << samples[i].text << "," << samples[i].e2e << ",0\n";
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
