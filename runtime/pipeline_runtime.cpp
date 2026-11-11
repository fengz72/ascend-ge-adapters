// Minimal two-stage GE pipeline runner: vision AIR -> (device-resident) image
// features -> text AIR -> scores.  Positional input binding in the fixed order
// of the exported io specs; no io_spec embedding, no pool framework.
//
// Request (stdin JSON):
//   {"mode": "infer"|"bench", "device": 0,
//    "vision_air": "...", "text_air": "...",
//    "records": "path/to/records.json",     // {"file": blob, "cases": [...]}
//    "warmup": 32, "iterations": 2048, "seed": 42,   // bench only
//    "output_dir": "..."}
// records.json cases: {"id": str, "vision": [{dtype, shape, offset, bytes} x5],
//                      "text":   [{dtype, shape, offset, bytes} x4]}
// Fixed positional orders:
//   vision: pixel_values, vision_pos_embeds, vision_rotary_cos,
//           vision_rotary_sin, vision_cu_seqlens
//   text:   input_ids, position_ids, actual_seq_lengths, [image_features from
//           the vision output], image_feature_index
//
// Output: one "GE_RESULT {json}" line.  infer writes outputs.bin + result.json
// in the ge-compare/1 shape (tools/compare.py / verify_business.py compatible);
// bench writes performance json + requests.csv.
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
#include <stdexcept>
#include <string>
#include <sys/stat.h>
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
    size_t batch = 0;                    // text input_ids shape[0]
    std::vector<TensorRef> vision, text; // 5 + 4 (features comes from stage 1)
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

// ---------------------------------------------------------------- the pipeline

struct Stage {
    uint32_t graphId = 0;
    aclrtStream stream = nullptr;        // one stream PER GRAPH: two graphs
                                         // alternating on one stream forces GE
                                         // to re-bind the execution plan
    std::vector<void *> inputBuffers;    // host-fed inputs only
    std::vector<size_t> inputSizes;      // allocated bytes per input
    std::vector<gert::Tensor> inputs;    // rebuilt per case
    std::vector<gert::Tensor> outputs;   // filled by GE (dynamic graphs)
};

class Pipeline {
public:
    ~Pipeline() { Destroy(); }

    void Init(int device, const std::string &visionAir, const std::string &textAir) {
        std::map<ge::AscendString, ge::AscendString> globalOptions = {
            {ge::AscendString("ge.graphRunMode"), ge::AscendString("0")},
            {ge::AscendString("ge.exec.deviceId"),
             ge::AscendString(std::to_string(device).c_str())},
        };
        std::cout << "[INFO] GEInitialize ge.graphRunMode=0" << std::endl;
        if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
            throw std::runtime_error(std::string("GEInitialize failed: ") + ge::GEGetErrorMsg().c_str());
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
        const uint32_t ids[2] = {1, 2};
        const std::string *paths[2] = {&visionAir, &textAir};
        for (int stage = 0; stage < 2; stage++) {
            auto begin = Clock::now();
            ge::Graph graph;
            if (graph.LoadFromFile(paths[stage]->c_str()) != ge::GRAPH_SUCCESS) {
                throw std::runtime_error("cannot load AIR: " + *paths[stage]);
            }
            ge::Status st = session_->AddGraph(ids[stage], graph);
            if (st != ge::SUCCESS) {
                throw std::runtime_error("AddGraph(" + std::to_string(ids[stage]) + ") failed");
            }
            st = session_->CompileGraph(ids[stage]);
            if (st != ge::SUCCESS) {
                throw std::runtime_error(std::string("CompileGraph(") +
                                         std::to_string(ids[stage]) + ") failed: " +
                                         ge::GEGetErrorMsg().c_str());
            }
            std::cout << "[INFO] CompileGraph ok (gid=" << ids[stage] << "), cost "
                      << ElapsedMs(begin, Clock::now()) << " ms" << std::endl;
        }

        if (aclrtSetDevice(device) != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSetDevice failed");
        }
        deviceSet_ = true;
        for (int stage = 0; stage < 2; stage++) {
            if (aclrtCreateStream(&stages_[stage].stream) != ACL_SUCCESS) {
                throw std::runtime_error("aclrtCreateStream failed");
            }
            auto begin = Clock::now();
            ge::Status st = session_->LoadGraph(ids[stage], {}, stages_[stage].stream);
            if (st != ge::SUCCESS) {
                throw std::runtime_error(std::string("LoadGraph(") +
                                         std::to_string(ids[stage]) + ") failed");
            }
            stages_[stage].graphId = ids[stage];
            std::cout << "[INFO] LoadGraph ok (gid=" << ids[stage] << "), cost "
                      << ElapsedMs(begin, Clock::now()) << " ms" << std::endl;
        }
    }

    // Pre-allocate host-fed input buffers at the max byte size across cases.
    // Vision: 5 inputs; text: 4 host inputs (indices 0,1,2,4 -- index 3 is the
    // device-resident features from the vision output).
    void Allocate(const std::vector<Case> &cases) {
        const size_t counts[2] = {5, 4};
        for (int stage = 0; stage < 2; stage++) {
            stages_[stage].inputSizes.assign(counts[stage], 0);
            for (const auto &one : cases) {
                const auto &tensors = stage == 0 ? one.vision : one.text;
                for (size_t i = 0; i < tensors.size() && i < counts[stage]; i++) {
                    stages_[stage].inputSizes[i] =
                        std::max(stages_[stage].inputSizes[i], tensors[i].bytes);
                }
            }
            for (size_t i = 0; i < counts[stage]; i++) {
                void *dev = nullptr;
                if (stages_[stage].inputSizes[i] > 0 &&
                    aclrtMalloc(&dev, stages_[stage].inputSizes[i],
                                ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                    throw std::runtime_error("aclrtMalloc failed for stage input buffer");
                }
                stages_[stage].inputBuffers.push_back(dev);
            }
        }
    }

    // One full pipeline execution; features never touch the host.
    bool Run(const Case &one, const std::vector<char> &blob, double *visionMs,
             double *textMs) {
        Stage &vision = stages_[0];
        Stage &text = stages_[1];

        auto visionBegin = Clock::now();
        vision.inputs.clear();
        for (size_t i = 0; i < one.vision.size(); i++) {
            const TensorRef &ref = one.vision[i];
            if (aclrtMemcpy(vision.inputBuffers[i], vision.inputSizes[i],
                            blob.data() + ref.offset, ref.bytes,
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] vision H2D failed (input %zu)\n", i);
                return false;
            }
            MakeTensor(vision.inputs, vision.inputBuffers[i],
                       vision.inputSizes[i], ref.shape, ref.geDtype);
        }
        vision.outputs.clear();
        if (session_->ExecuteGraphWithStreamAsync(vision.graphId, vision.stream,
                                                  vision.inputs,
                                                  vision.outputs) != ge::SUCCESS ||
            aclrtSynchronizeStream(vision.stream) != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] vision ExecuteGraphWithStreamAsync failed\n");
            return false;
        }
        if (vision.outputs.size() != 1) {
            fprintf(stderr, "[ERROR] vision stage returned %zu outputs\n",
                    vision.outputs.size());
            return false;
        }
        *visionMs = ElapsedMs(visionBegin, Clock::now());

        auto textBegin = Clock::now();
        text.inputs.clear();
        // positional slots: 0 input_ids, 1 position_ids, 2 actual_seq_lengths,
        // 3 image_features (the vision output, device-resident),
        // 4 image_feature_index; one.text carries the four HOST tensors in
        // the order (0, 1, 2, 4)
        for (size_t i = 0; i < one.text.size(); i++) {
            const TensorRef &ref = one.text[i];
            if (aclrtMemcpy(text.inputBuffers[i], text.inputSizes[i],
                            blob.data() + ref.offset, ref.bytes,
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                fprintf(stderr, "[ERROR] text H2D failed (input %zu)\n", i);
                return false;
            }
            if (i == 3) {
                MakeTensor(text.inputs, nullptr, 0, {1}, ge::DT_FLOAT);  // slot 3
            }
            MakeTensor(text.inputs, text.inputBuffers[i], text.inputSizes[i],
                       ref.shape, ref.geDtype);
        }
        if (text.inputs.size() != 5) {
            fprintf(stderr, "[ERROR] text input count is %zu, expected 5\n",
                    text.inputs.size());
            return false;
        }
        // slot 3 adopts the vision output's device buffer: gert::Tensor copy
        // assignment is deleted, so the descriptor is rebuilt over the same
        // storage (shape + dtype + a fresh TensorData view of the address)
        gert::Tensor &features = text.inputs[3];
        gert::Tensor &source = vision.outputs[0];
        features.GetShape() = source.GetShape();
        features.SetDataType(source.GetDataType());
        features.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
        features.SetData(gert::TensorData(source.GetAddr(), nullptr, source.GetSize(),
                                          gert::kOnDeviceHbm));
        text.outputs.clear();
        if (session_->ExecuteGraphWithStreamAsync(text.graphId, text.stream,
                                                  text.inputs,
                                                  text.outputs) != ge::SUCCESS ||
            aclrtSynchronizeStream(text.stream) != ACL_SUCCESS) {
            fprintf(stderr, "[ERROR] text ExecuteGraphWithStreamAsync failed\n");
            return false;
        }
        *textMs = ElapsedMs(textBegin, Clock::now());
        return true;
    }

    const gert::Tensor &Scores() const { return stages_[1].outputs[0]; }

    void Destroy() {
        for (auto &stage : stages_) {
            for (void *buffer : stage.inputBuffers) {
                if (buffer != nullptr) aclrtFree(buffer);
            }
            stage.inputBuffers.clear();
            stage.inputs.clear();
            stage.outputs.clear();
        }
        for (auto &stage : stages_) {
            if (stage.stream != nullptr) {
                aclrtDestroyStream(stage.stream);
                stage.stream = nullptr;
            }
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
    Stage stages_[2];
};

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

}  // namespace

int main() {
    try {
        Json request = Json::parse(std::cin);
        std::string mode = request.at("mode").get<std::string>();
        if (mode != "infer" && mode != "bench") {
            throw std::runtime_error("mode must be infer or bench");
        }
        int device = request.value("device", 0);
        std::string outputDir = request.at("output_dir").get<std::string>();
        MakeDirs(outputDir);

        Json records = Json::parse(ReadFile(request.at("records").get<std::string>()));
        if (records.at("schema").get<std::string>() != "startlux-split-records/1") {
            throw std::runtime_error("records.json is not the split record index");
        }
        std::vector<char> blob = ReadFile(records.at("file").get<std::string>());
        std::vector<Case> cases;
        for (const auto &entry : records.at("cases")) {
            Case one;
            one.id = entry.at("id").get<std::string>();
            for (const auto &tensor : entry.at("vision_tensors")) {
                one.vision.push_back(ParseTensorRef(tensor));
            }
            for (const auto &tensor : entry.at("text_tensors")) {
                one.text.push_back(ParseTensorRef(tensor));
            }
            if (one.vision.size() != 5 || one.text.size() != 4) {
                throw std::runtime_error("case " + one.id + " must carry 5 vision + 4 text tensors");
            }
            one.batch = one.text[0].shape.at(0);
            cases.push_back(std::move(one));
        }
        if (cases.empty()) throw std::runtime_error("no cases in the record pool");

        Pipeline pipeline;
        pipeline.Init(device, request.at("vision_air").get<std::string>(),
                      request.at("text_air").get<std::string>());
        pipeline.Allocate(cases);

        Json result = {{"status", "success"}, {"mode", mode}, {"backend", "ge_pipeline"}};
        if (mode == "infer") {
            std::ofstream binary(outputDir + "/outputs.bin", std::ios::binary);
            if (!binary) throw std::runtime_error("cannot open output binary");
            Json outputs = Json::object();
            for (const auto &one : cases) {
                double visionMs = 0.0, textMs = 0.0;
                if (!pipeline.Run(one, blob, &visionMs, &textMs)) {
                    throw std::runtime_error("pipeline execution failed on case " + one.id);
                }
                const gert::Tensor &scores = pipeline.Scores();
                size_t bytes = scores.GetSize();
                const auto &shape = scores.GetShape().GetStorageShape();
                std::vector<int64_t> dims;
                for (size_t d = 0; d < shape.GetDimNum(); d++) dims.push_back(shape.GetDim(d));
                Json tensors = Json::array();
                size_t offset = static_cast<size_t>(binary.tellp());
                std::vector<char> host(bytes);
                if (bytes > 0 &&
                    aclrtMemcpy(host.data(), bytes, scores.GetAddr(), bytes,
                                ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                    throw std::runtime_error("scores D2H failed on case " + one.id);
                }
                binary.write(host.data(), host.size());
                tensors.push_back({{"name", "scores"}, {"dtype", FromGeDtype(scores.GetDataType())},
                                   {"shape", dims}, {"offset", offset}, {"bytes", bytes}});
                std::cout << "[INFO] case " << one.id << " scores shape=["
                          << (dims.size() > 0 ? std::to_string(dims[0]) : "") << ","
                          << (dims.size() > 1 ? std::to_string(dims[1]) : "") << "] vision "
                          << visionMs << " ms, text " << textMs << " ms" << std::endl;
                outputs[one.id] = Json::array({Json({{"instance", 0}, {"tensors", tensors}})});
            }
            binary.close();
            result["outputs"] = outputs;
        } else {
            int warmup = request.value("warmup", 0);
            int iterations = request.at("iterations").get<int>();
            uint64_t seed = request.at("seed").get<uint64_t>();
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<size_t> pick(0, cases.size() - 1);
            for (int i = 0; i < warmup; i++) {
                double visionMs = 0.0, textMs = 0.0;
                if (!pipeline.Run(cases[pick(rng)], blob, &visionMs, &textMs)) {
                    throw std::runtime_error("warmup execution failed");
                }
            }
            struct Sample { size_t caseIndex; double e2e, vision, text; };
            std::vector<Sample> samples;
            samples.reserve(iterations);
            size_t rows = 0, errors = 0;
            auto measureBegin = Clock::now();
            for (int i = 0; i < iterations; i++) {
                size_t caseIndex = pick(rng);
                double visionMs = 0.0, textMs = 0.0;
                if (pipeline.Run(cases[caseIndex], blob, &visionMs, &textMs)) {
                    samples.push_back({caseIndex, visionMs + textMs, visionMs, textMs});
                    rows += cases[caseIndex].batch;
                } else {
                    errors++;
                }
            }
            double wallMs = ElapsedMs(measureBegin, Clock::now());

            std::vector<double> e2e, perVision, perText;
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
            csv << "instance,req,case,samples,vision_ms,text_ms,e2e_ms,error\n";
            for (size_t i = 0; i < samples.size(); i++) {
                csv << 0 << "," << i << "," << cases[samples[i].caseIndex].id << ","
                    << cases[samples[i].caseIndex].batch << ","
                    << samples[i].vision << "," << samples[i].text << ","
                    << samples[i].e2e << ",0\n";
            }
            csv.close();
        }
        std::cout << "GE_RESULT " << result.dump() << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
