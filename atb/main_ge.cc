// =============================================================================
// ge_infer — GESession 在线加载并执行 AIR 模型 (跳过 ATC 离线编译)
//
// 流程 (参照 cann/ge PR#743 RunGraphAsync 样例):
//   GEInitialize(graphRunMode) → aclInit → ge::Session → Graph::LoadFromFile(air)
//   → AddGraph → CompileGraph → aclrtSetDevice → CreateStream → LoadGraph
//   → 构造 gert::Tensor (实际 shape + H2D) → ExecuteGraphWithStreamAsync
//   → SyncStream → 动态输出自动回填 → D2H 保存
//
// 输入格式与 acl_infer 一致: name:shape:dtype:format:file (可重复)
// 注意: --input 顺序必须与图 Data 节点顺序一致 (GE 按 inputs 下标映射, 不校验名字)
//   基线 qwen2.5-0.5b.air:        asl, input_ids, position_ids
//   prefix qwen2.5-0.5b-prefix.air: act_q, act_kv, input_ids, position_ids
//
// 运行环境 (关键):
//   source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
//   prefix AIR 还需: source $ASCEND_HOME_PATH/opp/vendors/custom_prefix_attn/bin/set_env.bash
//   CANN tbe pywrapper 使用内嵌 /usr/bin/python3 (无 numpy), 需将本地 python3.11
//   site-packages 加入 PYTHONPATH, 否则 GEInitialize 因 tbe 初始化失败而返回 -1:
//   export PYTHONPATH=/usr/local/python3.11.15/lib/python3.11/site-packages:$PYTHONPATH
// =============================================================================

#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "ge/ge_api.h"
#include "graph/graph.h"
#include "exe_graph/runtime/tensor.h"

using Clock = std::chrono::high_resolution_clock;

static double ElapsedMs(Clock::time_point s, Clock::time_point e) {
    return std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() / 1000.0;
}

struct InputSpec {
    std::string name;
    std::vector<int64_t> shape;
    ge::DataType dtype;
    std::string dataFile;
};

static bool ParseDtype(const std::string &s, ge::DataType &dt) {
    if (s == "int64") { dt = ge::DT_INT64; return true; }
    if (s == "int32") { dt = ge::DT_INT32; return true; }
    if (s == "float16") { dt = ge::DT_FLOAT16; return true; }
    if (s == "float") { dt = ge::DT_FLOAT; return true; }
    if (s == "bool") { dt = ge::DT_BOOL; return true; }
    return false;
}

static size_t DtypeSize(ge::DataType dt) {
    switch (dt) {
        case ge::DT_INT64: return 8;
        case ge::DT_INT32: return 4;
        case ge::DT_FLOAT: return 4;
        case ge::DT_FLOAT16: return 2;
        case ge::DT_BOOL: return 1;
        default: return 0;
    }
}

static bool ParseInputSpec(const std::string &spec, InputSpec &out) {
    // name:shape:dtype:format:file  (format 仅校验为 ND, GE 路径固定 ND)
    std::vector<std::string> parts;
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ':')) parts.push_back(tok);
    if (parts.size() != 5) {
        std::cerr << "[ERROR] Invalid --input (need name:shape:dtype:format:file): " << spec << std::endl;
        return false;
    }
    out.name = parts[0];
    out.dataFile = parts[4];
    if (parts[3] != "ND") {
        std::cerr << "[ERROR] Only ND format supported, got: " << parts[3] << std::endl;
        return false;
    }
    if (!ParseDtype(parts[2], out.dtype)) {
        std::cerr << "[ERROR] Unsupported dtype: " << parts[2] << std::endl;
        return false;
    }
    std::stringstream sshape(parts[1]);
    out.shape.clear();
    while (std::getline(sshape, tok, ',')) {
        if (tok.empty()) continue;
        out.shape.push_back(std::stoll(tok));
    }
    if (out.shape.empty()) {
        std::cerr << "[ERROR] Empty shape: " << spec << std::endl;
        return false;
    }
    return true;
}

static bool ReadBinFile(const std::string &path, void *&data, size_t &size) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << path << std::endl;
        return false;
    }
    size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    data = malloc(size);
    if (data == nullptr) {
        std::cerr << "[ERROR] malloc failed for " << path << std::endl;
        return false;
    }
    f.read(static_cast<char *>(data), size);
    f.close();
    return true;
}

static void MakeStorageShape(const std::vector<int64_t> &dims, gert::StorageShape &ss) {
    auto &origin = ss.MutableOriginShape();
    auto &storage = ss.MutableStorageShape();
    for (int64_t d : dims) {
        origin.AppendDim(d);
        storage.AppendDim(d);
    }
}

static bool BuildDeviceTensor(const InputSpec &spec, gert::Tensor &devTensor, void *&devPtr) {
    size_t expected = DtypeSize(spec.dtype);
    int64_t numel = 1;
    for (int64_t d : spec.shape) numel *= d;
    size_t bytes = static_cast<size_t>(numel) * expected;

    void *hostData = nullptr;
    size_t fileSize = 0;
    if (!ReadBinFile(spec.dataFile, hostData, fileSize)) return false;
    if (fileSize != bytes) {
        std::cerr << "[ERROR] Input '" << spec.name << "' file size " << fileSize
                  << " != shape bytes " << bytes << " (" << spec.dataFile << ")" << std::endl;
        free(hostData);
        return false;
    }

    aclError ret = aclrtMalloc(&devPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtMalloc failed for input '" << spec.name << "', ret=" << ret << std::endl;
        free(hostData);
        return false;
    }
    ret = aclrtMemcpy(devPtr, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    free(hostData);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] H2D failed for input '" << spec.name << "', ret=" << ret << std::endl;
        aclrtFree(devPtr);
        return false;
    }

    gert::StorageShape ss;
    MakeStorageShape(spec.shape, ss);
    devTensor.GetShape() = ss;
    devTensor.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
    devTensor.SetDataType(spec.dtype);
    gert::TensorData td(devPtr, nullptr, bytes, gert::kOnDeviceHbm);
    devTensor.SetData(std::move(td));
    return true;
}

static std::string ShapeToString(const gert::Tensor &t) {
    auto shape = t.GetShape().GetStorageShape();
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        os << shape.GetDim(i) << (i + 1 < shape.GetDimNum() ? "," : "");
    }
    os << "]";
    return os.str();
}

static void PrintUsage(const char *prog) {
    std::cout << "Usage: " << prog << " --model <air> --input <name:shape:dtype:ND:file> [options]\n"
              << "\nGESession online inference for dynamic AIR models (no ATC).\n"
              << "\nOptions:\n"
              << "  --model <path>        AIR model path (required)\n"
              << "  --input <spec>        Input spec, repeatable. ORDER MUST match graph Data nodes:\n"
              << "                          baseline:  asl, input_ids, position_ids\n"
              << "                          prefix:    act_q, act_kv, input_ids, position_ids\n"
              << "  --output_dir <dir>    Output directory (default: ./output_ge)\n"
              << "  --device_id <id>      NPU device (default: 0)\n"
              << "  --graph_run_mode <m>  GE graphRunMode, 0=host 1=device (default: 1)\n"
              << "  --precision_mode <p>  e.g. force_fp16 / allow_fp32_to_fp16 (default: force_fp16)\n"
              << "  --warmup <N>          Warmup executions before timing (default: 10)\n"
              << "  --bench <N>           Timed executions for stats (default: 100)\n"
              << "  --aicore-num <spec>   Limit AI cores, e.g. 12 (=12|24, AIC|AIV 1:2) or '12|24';\n"
              << "                        empty = all cores (injected via AddGraph options)\n"
              << "  -h, --help            Show this help\n"
              << "\nExample (baseline):\n"
              << "  " << prog << " --model air/qwen2.5-0.5b.air --device_id 14 \\\n"
              << "    --input \"arg1_1:10:int64:ND:input_data/actual_seq_lengths.bin\" \\\n"
              << "    --input \"arg4_1:2080:int64:ND:input_data/input_ids.bin\" \\\n"
              << "    --input \"arg7_1:2080:int64:ND:input_data/position_ids.bin\"" << std::endl;
}

int main(int argc, char *argv[]) {
    std::string modelPath;
    std::vector<InputSpec> inputSpecs;
    std::string outputDir = "./output_ge";
    int32_t deviceId = 0;
    int32_t graphRunMode = 1;
    std::string precisionMode = "force_fp16";
    int32_t warmupRuns = 10;
    int32_t benchRuns = 100;
    std::string aicoreNum;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            InputSpec spec;
            if (!ParseInputSpec(argv[++i], spec)) return 1;
            inputSpecs.push_back(spec);
        } else if (arg == "--output_dir" && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--device_id" && i + 1 < argc) {
            deviceId = std::stoi(argv[++i]);
        } else if (arg == "--graph_run_mode" && i + 1 < argc) {
            graphRunMode = std::stoi(argv[++i]);
        } else if (arg == "--precision_mode" && i + 1 < argc) {
            precisionMode = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmupRuns = std::stoi(argv[++i]);
        } else if (arg == "--bench" && i + 1 < argc) {
            benchRuns = std::stoi(argv[++i]);
        } else if (arg == "--aicore-num" && i + 1 < argc) {
            aicoreNum = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "[ERROR] Unknown option: " << arg << std::endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }
    if (modelPath.empty() || inputSpecs.empty()) {
        PrintUsage(argv[0]);
        return 1;
    }

    // ---- 1. GEInitialize (global) + aclInit ----
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {ge::AscendString("ge.graphRunMode"), ge::AscendString(std::to_string(graphRunMode).c_str())},
        {ge::AscendString("ge.exec.deviceId"), ge::AscendString(std::to_string(deviceId).c_str())},
    };
    if (!aicoreNum.empty()) {
        // 整数 N → N|2N (AIC|AIV 1:2), "aic|aiv" 原样透传 (约定同 run.sh atc)
        std::string spec = aicoreNum;
        bool isDigit = !spec.empty() && spec.find_first_not_of("0123456789") == std::string::npos;
        if (isDigit) spec = spec + "|" + std::to_string(std::stoi(spec) * 2);
        globalOptions.emplace(ge::AscendString(ge::ir_option::AICORE_NUM), ge::AscendString(spec.c_str()));
        std::cout << "[INFO] GEInitialize with " << ge::ir_option::AICORE_NUM << "=" << spec << std::endl;
    }
    ge::Status ret = ge::GEInitialize(globalOptions);
    if (ret != ge::SUCCESS) {
        std::cerr << "[ERROR] GEInitialize failed, ret=" << ret << std::endl;
        return 1;
    }
    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS && aclRet != ACL_ERROR_REPEAT_INITIALIZE) {
        std::cerr << "[ERROR] aclInit failed, ret=" << aclRet << std::endl;
        return 1;
    }

    // ---- 2. Session ----
    std::map<ge::AscendString, ge::AscendString> sessionOptions = {
        {ge::AscendString("ge.session_device_id"), ge::AscendString(std::to_string(deviceId).c_str())},
        {ge::AscendString("ge.exec.precision_mode"), ge::AscendString(precisionMode.c_str())},
    };
    auto session = std::make_shared<ge::Session>(sessionOptions);
    if (session == nullptr) {
        std::cerr << "[ERROR] Create Session failed" << std::endl;
        return 1;
    }

    // ---- 3. 解析 AIR + AddGraph + CompileGraph ----
    auto t0 = Clock::now();
    ge::Graph graph;
    if (graph.LoadFromFile(modelPath.c_str()) != ge::GRAPH_SUCCESS) {
        std::cerr << "[ERROR] Graph::LoadFromFile failed: " << modelPath << std::endl;
        return 1;
    }
    const uint32_t graphId = 1;
    ret = session->AddGraph(graphId, graph);
    if (ret != ge::SUCCESS) {
        std::cerr << "[ERROR] AddGraph failed, ret=" << ret << std::endl;
        return 1;
    }
    ret = session->CompileGraph(graphId);
    auto t1 = Clock::now();
    if (ret != ge::SUCCESS) {
        std::cerr << "[ERROR] CompileGraph failed, ret=" << ret << std::endl;
        return 1;
    }
    std::cout << "[INFO] CompileGraph success, cost " << ElapsedMs(t0, t1) << " ms" << std::endl;

    // ---- 4. SetDevice + Stream + LoadGraph ----
    aclRet = aclrtSetDevice(deviceId);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtSetDevice failed, ret=" << aclRet << std::endl;
        return 1;
    }
    aclrtStream stream = nullptr;
    aclRet = aclrtCreateStream(&stream);
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclrtCreateStream failed, ret=" << aclRet << std::endl;
        return 1;
    }
    t0 = Clock::now();
    ret = session->LoadGraph(graphId, {}, stream);
    t1 = Clock::now();
    if (ret != ge::SUCCESS) {
        std::cerr << "[ERROR] LoadGraph failed, ret=" << ret << std::endl;
        return 1;
    }
    std::cout << "[INFO] LoadGraph success, cost " << ElapsedMs(t0, t1) << " ms" << std::endl;

    // ---- 5. 构造输入 device tensor (真实数据 + 实际 shape) ----
    std::vector<gert::Tensor> devInputs;
    std::vector<void *> inputPtrs;
    for (size_t i = 0; i < inputSpecs.size(); i++) {
        gert::Tensor t;
        void *ptr = nullptr;
        if (!BuildDeviceTensor(inputSpecs[i], t, ptr)) return 1;
        std::cout << "[INFO] Input[" << i << "] '" << inputSpecs[i].name << "' shape="
                  << ShapeToString(t) << " ready" << std::endl;
        devInputs.emplace_back(std::move(t));
        inputPtrs.push_back(ptr);
    }

    // ---- 6. 首次执行 (含 shape 特化, 确定输出 shape) ----
    std::vector<gert::Tensor> devOutputs;
    t0 = Clock::now();
    ret = session->ExecuteGraphWithStreamAsync(graphId, stream, devInputs, devOutputs);
    if (ret != ge::SUCCESS) {
        std::cerr << "[ERROR] ExecuteGraphWithStreamAsync failed, ret=" << ret << std::endl;
        return 1;
    }
    aclRet = aclrtSynchronizeStream(stream);
    t1 = Clock::now();
    if (aclRet != ACL_SUCCESS) {
        std::cerr << "[ERROR] SynchronizeStream failed, ret=" << aclRet << std::endl;
        return 1;
    }
    std::cout << "[INFO] First execute (incl. shape specialization), cost " << ElapsedMs(t0, t1) << " ms" << std::endl;

    // 输出 device 地址登记 (GE 可能复用或新分配, 统一去重后释放)
    std::set<void *> outputAddrs;
    for (auto &out : devOutputs) {
        if (out.GetAddr() != nullptr) outputAddrs.insert(out.GetAddr());
    }

    // ---- 6.1 warmup (不计入) ----
    if (warmupRuns > 0) {
        std::cout << "[INFO] Running " << warmupRuns << " warmup iteration(s)..." << std::endl;
        for (int32_t w = 0; w < warmupRuns; w++) {
            std::vector<gert::Tensor> tmp;
            ret = session->ExecuteGraphWithStreamAsync(graphId, stream, devInputs, tmp);
            if (ret != ge::SUCCESS) {
                std::cerr << "[ERROR] Warmup run " << w << " failed, ret=" << ret << std::endl;
                return 1;
            }
            aclRet = aclrtSynchronizeStream(stream);
            if (aclRet != ACL_SUCCESS) {
                std::cerr << "[ERROR] Warmup sync failed, ret=" << aclRet << std::endl;
                return 1;
            }
            for (auto &out : tmp) {
                if (out.GetAddr() != nullptr) outputAddrs.insert(out.GetAddr());
            }
        }
        std::cout << "[INFO] Warmup completed" << std::endl;
    }

    // ---- 6.2 bench (计时, 口径对齐 acl_infer: 纯 execute + sync) ----
    std::vector<double> inferTimes;
    inferTimes.reserve(benchRuns);
    for (int32_t b = 0; b < benchRuns; b++) {
        std::vector<gert::Tensor> tmp;
        auto ts = Clock::now();
        ret = session->ExecuteGraphWithStreamAsync(graphId, stream, devInputs, tmp);
        if (ret != ge::SUCCESS) {
            std::cerr << "[ERROR] Bench run " << b << " failed, ret=" << ret << std::endl;
            return 1;
        }
        aclRet = aclrtSynchronizeStream(stream);
        auto te = Clock::now();
        if (aclRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] Bench sync failed, ret=" << aclRet << std::endl;
            return 1;
        }
        inferTimes.push_back(ElapsedMs(ts, te));
        for (auto &out : tmp) {
            if (out.GetAddr() != nullptr) outputAddrs.insert(out.GetAddr());
        }
    }
    std::sort(inferTimes.begin(), inferTimes.end());
    double avg = 0.0;
    for (double t : inferTimes) avg += t;
    avg /= inferTimes.size();
    std::cout << "\n============================================================\n"
              << "GE Execute Benchmark (execute+sync, " << benchRuns << " runs, warmup=" << warmupRuns << ")\n"
              << "============================================================\n"
              << "  avg: " << avg << " ms\n"
              << "  min: " << inferTimes.front() << " ms\n"
              << "  p50: " << inferTimes[inferTimes.size() / 2] << " ms\n"
              << "  p99: " << inferTimes[static_cast<size_t>(inferTimes.size() * 0.99)] << " ms\n"
              << "  max: " << inferTimes.back() << " ms\n"
              << "============================================================\n" << std::endl;

    // ---- 7. 动态输出: GE 自动分配, D2H 保存 ----
    mkdir(outputDir.c_str(), 0755);  // 已存在时忽略 EEXIST
    std::cout << "[INFO] Outputs: " << devOutputs.size() << std::endl;
    for (size_t i = 0; i < devOutputs.size(); i++) {
        const auto &out = devOutputs[i];
        size_t bytes = out.GetSize();
        void *hostBuf = malloc(bytes);
        if (hostBuf == nullptr) {
            std::cerr << "[ERROR] malloc output host buffer failed" << std::endl;
            return 1;
        }
        aclRet = aclrtMemcpy(hostBuf, bytes, out.GetAddr(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (aclRet != ACL_SUCCESS) {
            std::cerr << "[ERROR] D2H failed for output[" << i << "], ret=" << aclRet << std::endl;
            free(hostBuf);
            return 1;
        }
        std::ostringstream os;
        os << outputDir << "/output_" << i << ".bin";
        std::ofstream f(os.str(), std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "[ERROR] Cannot open output file (dir exists?): " << os.str() << std::endl;
            return 1;
        }
        f.write(static_cast<char *>(hostBuf), bytes);
        f.close();
        free(hostBuf);
        std::cout << "[INFO] Output[" << i << "]: shape=" << ShapeToString(out)
                  << " bytes=" << bytes << " saved to " << os.str() << std::endl;
    }

    // ---- 8. 清理 ----
    for (void *p : inputPtrs) aclrtFree(p);
    for (void *p : outputAddrs) aclrtFree(p);
    aclrtDestroyStream(stream);
    session.reset();
    aclrtResetDevice(deviceId);
    ge::GEFinalize();
    aclFinalize();
    std::cout << "[INFO] GE inference completed successfully" << std::endl;
    return 0;
}
