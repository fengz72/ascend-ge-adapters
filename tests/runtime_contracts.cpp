// CPU 契约测试: npy 读写 / cases.json 解析 / 抽样与计时聚合。
// 只依赖 runtime/{npy,cases,stats}.{h,cpp} + vendored nlohmann, 不需要 GE / ACL。
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cases.h"
#include "npy.h"
#include "stats.h"

namespace fs = std::filesystem;
using namespace ge_execute;

#define CHECK(condition)                                                          \
    do {                                                                          \
        if (!(condition)) {                                                       \
            throw std::runtime_error(std::string("check failed: ") + #condition + \
                                     " (line " + std::to_string(__LINE__) + ")"); \
        }                                                                         \
    } while (false)

namespace {

constexpr char kMagic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};

void WriteBytes(const std::string &path, const void *data, size_t bytes) {
    const fs::path target(path);
    if (target.has_parent_path()) {
        fs::create_directories(target.parent_path());
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    if (!out) {
        throw std::runtime_error("short write " + path);
    }
}

void WriteText(const std::string &path, const std::string &text) {
    WriteBytes(path, text.data(), text.size());
}

std::vector<char> ReadBytes(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }
    in.seekg(0, std::ios::end);
    const std::streamoff length = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<char> out(static_cast<size_t>(length));
    if (length > 0 && !in.read(out.data(), length)) {
        throw std::runtime_error("short read " + path);
    }
    return out;
}

void WriteRawNpy(const std::string &path, const std::string &dictBody,
                 const std::vector<char> &payload, int major = 1) {
    std::string header = dictBody;
    if (header.empty() || header.back() != '\n') {
        header.push_back('\n');
    }
    std::vector<char> raw(kMagic, kMagic + sizeof(kMagic));
    raw.push_back(static_cast<char>(major));
    raw.push_back(0);
    if (major == 1) {
        const uint16_t len = static_cast<uint16_t>(header.size());
        raw.push_back(static_cast<char>(len & 0xFF));
        raw.push_back(static_cast<char>((len >> 8) & 0xFF));
    } else {
        const uint32_t len = static_cast<uint32_t>(header.size());
        for (int i = 0; i < 4; ++i) {
            raw.push_back(static_cast<char>((len >> (8 * i)) & 0xFF));
        }
    }
    raw.insert(raw.end(), header.begin(), header.end());
    raw.insert(raw.end(), payload.begin(), payload.end());
    WriteBytes(path, raw.data(), raw.size());
}

template <typename Fn>
bool ThrowsRuntime(Fn fn) {
    try {
        fn();
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

template <typename Fn>
bool ThrowsInvalidArgument(Fn fn) {
    try {
        fn();
    } catch (const std::invalid_argument &) {
        return true;
    }
    return false;
}

void TestNpyRoundTrip(const fs::path &dir) {
    const fs::path root = dir / "npy";
    fs::create_directories(root);

    {
        const std::string path = (root / "f32_2d.npy").string();
        const float values[6] = {1.0f, -2.5f, 3.25f, 4.0f, 5.5f, 6.125f};
        WriteNpy(path, "float32", {2, 3}, values, sizeof(values));
        const NpyArray array = ReadNpy(path);
        CHECK(array.dtype == "float32");
        CHECK(array.shape == (std::vector<int64_t>{2, 3}));
        CHECK(array.data.size() == sizeof(values));
        CHECK(std::memcmp(array.data.data(), values, sizeof(values)) == 0);
    }
    {
        const std::string path = (root / "i64_1d.npy").string();
        const int64_t values[3] = {-1, 0, 9000000000LL};
        WriteNpy(path, "int64", {3}, values, sizeof(values));
        const NpyArray array = ReadNpy(path);
        CHECK(array.dtype == "int64");
        CHECK(array.shape == (std::vector<int64_t>{3}));
        CHECK(array.data.size() == sizeof(values));
        CHECK(std::memcmp(array.data.data(), values, sizeof(values)) == 0);
    }
    {
        const std::string path = (root / "bool_1d.npy").string();
        const unsigned char values[3] = {0, 1, 1};
        WriteNpy(path, "bool", {3}, values, sizeof(values));
        const NpyArray array = ReadNpy(path);
        CHECK(array.dtype == "bool");
        CHECK(array.shape == (std::vector<int64_t>{3}));
        CHECK(array.data.size() == sizeof(values));
        CHECK(std::memcmp(array.data.data(), values, sizeof(values)) == 0);
    }
    {
        const std::string path = (root / "scalar.npy").string();
        const int32_t value = 42;
        WriteNpy(path, "int32", {}, &value, sizeof(value));
        const NpyArray array = ReadNpy(path);
        CHECK(array.dtype == "int32");
        CHECK(array.shape.empty());
        CHECK(array.data.size() == sizeof(value));
        CHECK(std::memcmp(array.data.data(), &value, sizeof(value)) == 0);
    }
    {
        const std::string path = (root / "u8_1d.npy").string();
        const uint8_t values[4] = {0, 255, 1, 2};
        WriteNpy(path, "uint8", {4}, values, sizeof(values));
        const NpyArray array = ReadNpy(path);
        CHECK(array.dtype == "uint8");
        CHECK(array.shape == (std::vector<int64_t>{4}));
        CHECK(array.data.size() == sizeof(values));
        CHECK(std::memcmp(array.data.data(), values, sizeof(values)) == 0);
    }
    // dtype <-> descr 表
    CHECK(DtypeBytes("bfloat16") == 2 && DtypeBytes("float32") == 4 &&
          DtypeBytes("bool") == 1 && DtypeBytes("nope") == 0);
    CHECK(DtypeToDescr("float32") == "<f4" && DtypeToDescr("uint8") == "|u1" &&
          DtypeToDescr("bfloat16") == "|u2" && DtypeToDescr("nope").empty());
    CHECK(DescrToDtype("<f4") == "float32" && DescrToDtype("|b1") == "bool" &&
          DescrToDtype("|u2") == "bfloat16" && DescrToDtype("<u2") == "uint16" &&
          DescrToDtype(">f4").empty() && DescrToDtype("|O").empty() &&
          DescrToDtype("|U2").empty());
}

void TestNpyRejects(const fs::path &dir) {
    const fs::path root = dir / "npy_rejects";
    fs::create_directories(root);
    const std::vector<char> payload8(8, '\x01');

    {
        const std::string path = (root / "fortran.npy").string();
        WriteRawNpy(path, "{'descr': '<f4', 'fortran_order': True, 'shape': (2,), }",
                    payload8);
        CHECK(ThrowsRuntime([&] { ReadNpy(path); }));
    }
    {
        const std::string path = (root / "big_endian.npy").string();
        WriteRawNpy(path, "{'descr': '>f4', 'fortran_order': False, 'shape': (2,), }",
                    payload8);
        CHECK(ThrowsRuntime([&] { ReadNpy(path); }));
    }
    {
        const std::string objectPath = (root / "object.npy").string();
        WriteRawNpy(objectPath,
                    "{'descr': '|O', 'fortran_order': False, 'shape': (1,), }", payload8);
        CHECK(ThrowsRuntime([&] { ReadNpy(objectPath); }));
        const std::string unicodePath = (root / "unicode.npy").string();
        WriteRawNpy(unicodePath,
                    "{'descr': '|U2', 'fortran_order': False, 'shape': (2,), }", payload8);
        CHECK(ThrowsRuntime([&] { ReadNpy(unicodePath); }));
    }
    {
        // 先写合法文件, 再截断 2 字节。
        const std::string valid = (root / "valid.npy").string();
        const int32_t values[2] = {1, 2};
        WriteNpy(valid, "int32", {2}, values, sizeof(values));
        std::vector<char> raw = ReadBytes(valid);
        raw.resize(raw.size() - 2);
        const std::string truncated = (root / "truncated.npy").string();
        WriteBytes(truncated, raw.data(), raw.size());
        CHECK(ThrowsRuntime([&] { ReadNpy(truncated); }));
    }
    {
        const std::string path = (root / "bad_magic.npy").string();
        std::vector<char> raw(kMagic, kMagic + sizeof(kMagic));
        raw[0] = 'X';
        raw.push_back(1);
        raw.push_back(0);
        raw.push_back(0);
        raw.push_back(0);
        WriteBytes(path, raw.data(), raw.size());
        CHECK(ThrowsRuntime([&] { ReadNpy(path); }));
    }
    {
        const std::string path = (root / "short_data.npy").string();
        WriteRawNpy(path, "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
                    payload8);
        CHECK(ThrowsRuntime([&] { ReadNpy(path); }));
    }
    {
        // bfloat16 以 '|u2' 存储; override 只做标注, 字节数必须一致。
        const std::string path = (root / "bf16.npy").string();
        WriteRawNpy(path, "{'descr': '|u2', 'fortran_order': False, 'shape': (2,), }",
                    std::vector<char>{'\x80', '\x3f', '\x00', '\x40'});
        const NpyArray native = ReadNpy(path);
        CHECK(native.dtype == "bfloat16" && native.shape.size() == 1 &&
              native.shape[0] == 2 && native.data.size() == 4);
        const NpyArray overridden = ReadNpy(path, "bfloat16");
        CHECK(overridden.dtype == "bfloat16" && overridden.data == native.data);
    }
    {
        const std::string path = (root / "override_mismatch.npy").string();
        WriteRawNpy(path, "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }",
                    payload8);
        CHECK(ThrowsRuntime([&] { ReadNpy(path, "float16"); }));
        CHECK(ThrowsRuntime([&] { ReadNpy(path, "not_a_dtype"); }));
    }
    {
        // v2.0 头 (4 字节 header_len) 应能读取。
        const std::string path = (root / "v2.npy").string();
        const float values[2] = {1.5f, 2.5f};
        std::vector<char> payload(sizeof(values));
        std::memcpy(payload.data(), values, sizeof(values));
        WriteRawNpy(path, "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }",
                    payload, 2);
        const NpyArray array = ReadNpy(path);
        CHECK(array.dtype == "float32" && array.shape == (std::vector<int64_t>{2}));
        CHECK(std::memcmp(array.data.data(), values, sizeof(values)) == 0);
    }
}

nlohmann::json GoodCases() {
    return nlohmann::json::parse(R"json({
        "cases": [
            {"id": "case-0000",
             "inputs": [
                 {"name": "x", "path": "a/x.npy"},
                 {"name": "y", "path": "a/y.npy"}]},
            {"id": "case-0001",
             "samples": 3,
             "inputs": [
                 {"name": "x", "path": "b/x.npy"},
                 {"name": "y", "path": "a/y.npy"}]}
        ]
    })json");
}

void WriteJson(const std::string &path, const nlohmann::json &value) {
    WriteText(path, value.dump(2));
}

void TestCases(const fs::path &dir) {
    const fs::path root = dir / "cases";
    fs::create_directories(root / "a");
    fs::create_directories(root / "b");

    const float xA[6] = {1, 2, 3, 4, 5, 6};
    const float xB[15] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const int64_t yA[4] = {7, 8, 9, 10};
    WriteNpy((root / "a" / "x.npy").string(), "float32", {2, 3}, xA, sizeof(xA));
    WriteNpy((root / "b" / "x.npy").string(), "float32", {5, 3}, xB, sizeof(xB));
    WriteNpy((root / "a" / "y.npy").string(), "int64", {4}, yA, sizeof(yA));

    const std::string casesPath = (root / "cases.json").string();
    const nlohmann::json good = GoodCases();
    WriteJson(casesPath, good);

    const CasePool pool = LoadCases(casesPath);
    CHECK(pool.cases.size() == 2);
    CHECK(!pool.empty());
    CHECK(pool.cases[0].id == "case-0000");
    CHECK(pool.cases[0].samples == 1);  // 缺省 samples
    CHECK(pool.cases[1].samples == 3);
    CHECK(pool.cases[0].inputs.size() == 2 && pool.cases[1].inputs.size() == 2);
    CHECK(pool.cases[0].inputs[0].name == "x");
    CHECK(pool.cases[0].inputs[0].dtype == "float32");
    CHECK(pool.cases[0].inputs[0].shape == (std::vector<int64_t>{2, 3}));
    CHECK(pool.cases[0].inputs[0].data.size() == sizeof(xA));
    CHECK(std::memcmp(pool.cases[0].inputs[0].data.data(), xA, sizeof(xA)) == 0);
    CHECK(pool.cases[0].inputs[1].dtype == "int64");
    CHECK(pool.cases[0].inputs[1].data.size() == sizeof(yA));
    CHECK(pool.cases[1].inputs[0].shape == (std::vector<int64_t>{5, 3}));
    CHECK(pool.cases[0].shapeKey == "[2, 3];[4];");
    CHECK(pool.cases[1].shapeKey == "[5, 3];[4];");
    CHECK(pool.distinctShapes == 2);
    CHECK(pool.maxBytes.size() == 2);
    CHECK(pool.maxBytes[0] == sizeof(xB));
    CHECK(pool.maxBytes[1] == sizeof(yA));
    const std::vector<std::string> keys = pool.shapeKeys();
    CHECK(keys == (std::vector<std::string>{"[2, 3];[4];", "[5, 3];[4];"}));

    // 每个拒绝路径各写一份坏 JSON。
    const auto expectReject = [&](const nlohmann::json &bad, const std::string &name) {
        const std::string path = (root / ("bad_" + name + ".json")).string();
        WriteJson(path, bad);
        CHECK(ThrowsRuntime([&] { LoadCases(path); }));
    };

    {
        nlohmann::json bad = good;
        bad["cases"][1]["id"] = bad["cases"][0]["id"];
        expectReject(bad, "duplicate_id");
    }
    {
        nlohmann::json bad = good;
        bad["cases"] = nlohmann::json::array();
        expectReject(bad, "empty_cases");
    }
    {
        nlohmann::json bad = good;
        bad["cases"][0]["inputs"] = nlohmann::json::array();
        expectReject(bad, "empty_inputs");
    }
    {
        nlohmann::json bad = good;
        bad["cases"][1]["inputs"].erase(bad["cases"][1]["inputs"].begin());
        expectReject(bad, "input_count_mismatch");
    }
    {
        nlohmann::json bad = good;
        bad["cases"][0]["samples"] = 0;
        expectReject(bad, "samples_zero");
    }
    {
        nlohmann::json bad = good;
        bad["cases"][0]["inputs"][0]["path"] = "a/missing.npy";
        expectReject(bad, "missing_npy");
    }
    {
        nlohmann::json bad = good;
        bad["cases"][0].erase("id");
        expectReject(bad, "missing_id");
    }
    {
        const std::string missing = (root / "does-not-exist.json").string();
        CHECK(ThrowsRuntime([&] { LoadCases(missing); }));
    }
}

void TestStats() {
    // Nearest-rank 边界。
    CHECK(NearestRankIndex(1, 0.5) == 0);
    CHECK(NearestRankIndex(1, 0.99) == 0);
    CHECK(NearestRankIndex(1, 1.0) == 0);
    CHECK(NearestRankIndex(100, 0.99) == 98);
    CHECK(NearestRankIndex(1000, 0.99) == 989);
    CHECK(NearestRankIndex(4, 0.5) == 1);
    CHECK(NearestRankIndex(10, 1.0) == 9);
    CHECK(ThrowsInvalidArgument([] { return NearestRankIndex(0, 0.5); }));
    CHECK(ThrowsInvalidArgument([] { return NearestRankIndex(5, 0.0); }));
    CHECK(ThrowsInvalidArgument([] { return NearestRankIndex(5, 1.5); }));
    CHECK(ThrowsInvalidArgument([] { return NearestRankIndex(5, -0.1); }));

    CHECK(Percentile({}, 0.5) == 0.0);
    const std::vector<double> sample = {5, 1, 4, 2, 3};
    CHECK(Percentile(sample, 0.50) == 3.0);  // nearest-rank 第 3 小
    CHECK(Percentile(sample, 0.99) == 5.0);

    // DrawSequence: 可复现, 不同 seed 不同, poolSize<=1 恒 0。
    const std::vector<size_t> drawA = DrawSequence(10, 100, 42);
    const std::vector<size_t> drawB = DrawSequence(10, 100, 42);
    CHECK(drawA == drawB);
    const std::vector<size_t> drawC = DrawSequence(10, 100, 43);
    CHECK(drawA != drawC);
    bool inRange = true;
    for (size_t value : drawA) {
        inRange = inRange && value < 10;
    }
    CHECK(inRange);
    CHECK(DrawSequence(1, 5, 42) == std::vector<size_t>(5, 0));
    CHECK(DrawSequence(0, 3, 42) == std::vector<size_t>(3, 0));
    CHECK(DrawSequence(10, 0, 42).empty());

    // MeasurePlan: 全局序列交错切分。
    const size_t requests = 101;
    const size_t instances = 3;
    const std::vector<size_t> global = DrawSequence(7, requests, 99);
    const std::vector<std::vector<size_t>> plan =
        MeasurePlan(7, requests, instances, 99);
    CHECK(plan.size() == instances);
    size_t total = 0;
    for (const auto &lane : plan) {
        total += lane.size();
    }
    CHECK(total == requests);
    bool interleaved = true;
    for (size_t t = 0; t < instances; ++t) {
        for (size_t k = 0; k < plan[t].size(); ++k) {
            interleaved = interleaved && plan[t][k] == global[t + k * instances];
        }
    }
    CHECK(interleaved);
    CHECK(plan[0][0] == global[0]);
    CHECK(MeasurePlan(7, 0, 3, 99).empty());
    CHECK(MeasurePlan(7, 4, 0, 99).size() == 1);

    // WarmupPlan: 覆盖全部 distinct shape, 再补齐。
    const std::vector<std::string> shapeKeys = {"a", "b", "a", "c", "b"};
    const std::vector<std::vector<size_t>> warm =
        WarmupPlan(shapeKeys, 0, 2, 42);
    CHECK(warm.size() == 2);
    for (const auto &lane : warm) {
        CHECK(lane.size() == 3);
        CHECK(lane == (std::vector<size_t>{0, 1, 3}));
    }
    const std::vector<std::vector<size_t>> warm2 = WarmupPlan(shapeKeys, 7, 2, 42);
    CHECK(warm2.size() == 2);
    for (const auto &lane : warm2) {
        CHECK(lane.size() == 7);
        CHECK(lane[0] == 0 && lane[1] == 1 && lane[2] == 3);
    }
    CHECK(WarmupPlan({}, 5, 2, 42).empty());
    CHECK(WarmupPlan(shapeKeys, 2, 0, 42).size() == 1);

    // Summarize: 只把 success 计入 requests/samples/latency, errors/perInstance 含失败。
    RunStats stats;
    stats.wallMs = 1000.0;
    stats.warmupRuns = 4;
    stats.warmupMs = 12.5;
    stats.compileMs = 3.0;
    stats.loadMs = {1.0, 2.0};

    auto addRecord = [&stats](int instance, size_t request, const std::string &caseId,
                              double e2e, double exec, double d2h, int64_t samples, bool error) {
        RequestTiming record;
        record.instance = instance;
        record.request = request;
        record.caseId = caseId;
        record.shapeKey = "shape";
        record.samples = samples;
        record.h2dMs = 1.0;
        record.execMs = exec;
        record.d2hMs = d2h;
        record.e2eMs = e2e;
        record.error = error;
        stats.records.push_back(record);
    };
    addRecord(0, 0, "case-a", 3.0, 2.0, 0.5, 16, false);
    addRecord(0, 1, "case-b", 5.0, 4.0, 1.0, 7, false);
    addRecord(1, 2, "case-a", 10.0, 6.0, 1.5, 5, false);
    addRecord(1, 3, "case-a", 99.0, 90.0, 8.0, 11, true);   // 失败请求的样本/D2H 不计入

    Summarize(stats);
    CHECK(stats.requests == 3);
    CHECK(stats.errors == 1);
    CHECK(stats.samples == 28);   // 16 + 7 + 5, 失败那条的 11 不算
    CHECK(stats.perInstance == (std::vector<size_t>{2, 2}));
    CHECK(stats.e2eAvg == 6.0);
    CHECK(stats.e2eMin == 3.0);
    CHECK(stats.e2eMax == 10.0);
    CHECK(stats.e2eP50 == 5.0);   // {3,5,10} 的 nearest-rank p50
    CHECK(stats.e2eP99 == 10.0);
    CHECK(stats.execAvg == 4.0);
    CHECK(stats.execP99 == 6.0);
    CHECK(stats.d2hAvg == 1.0);   // (0.5 + 1.0 + 1.5) / 3, 失败那条的 8.0 不算
    CHECK(stats.qps == 3.0);      // 3 次成功 / 1s
    CHECK(stats.samplesPerSecond == 28.0);

    RunStats zeroWall;
    zeroWall.wallMs = 0.0;
    zeroWall.records = stats.records;
    Summarize(zeroWall);
    CHECK(zeroWall.qps == 0.0 && zeroWall.samplesPerSecond == 0.0);
}

void TestJsonOutput(const fs::path &dir) {
    const fs::path root = dir / "json";
    fs::create_directories(root);

    RunStats stats;
    stats.status = "success";
    stats.mode = "bench";
    stats.wallMs = 500.0;
    stats.warmupRuns = 2;
    stats.loadMs = {0.5};
    RequestTiming record;
    record.instance = 0;
    record.request = 0;
    record.caseId = "case-a";
    record.shapeKey = "[1];";
    record.samples = 7;
    record.h2dMs = 0.1;
    record.execMs = 0.2;
    record.d2hMs = 0.05;
    record.e2eMs = 0.35;   // h2d + exec + d2h, 与运行时口径一致
    record.firstHit = true;
    stats.records.push_back(record);
    Summarize(stats);

    const nlohmann::json json = StatsToJson(stats);
    CHECK(json["status"] == "success");
    CHECK(json["mode"] == "bench");
    CHECK(json["requests"] == 1);
    CHECK(json["errors"] == 0);
    CHECK(json["samples"] == 7);
    CHECK(json["wall_ms"] == 500.0);
    CHECK(json["qps"] == 2.0);
    CHECK(json["d2h_avg"] == 0.05);
    CHECK(json["warmup_runs"] == 2);
    CHECK(json["load_ms"].size() == 1);
    CHECK(json["per_instance"].size() == 1);
    CHECK(json["records"].size() == 1);
    const nlohmann::json &entry = json["records"][0];
    CHECK(entry["instance"] == 0);
    CHECK(entry["request"] == 0);
    CHECK(entry["case_id"] == "case-a");
    CHECK(entry["shape_key"] == "[1];");
    CHECK(entry["samples"] == 7);
    CHECK(entry["h2d_ms"] == 0.1);
    CHECK(entry["exec_ms"] == 0.2);
    CHECK(entry["d2h_ms"] == 0.05);
    CHECK(entry["e2e_ms"] == 0.35);
    CHECK(entry["first_hit"] == true);
    CHECK(entry["error"] == false);

    // 父目录不存在时递归创建, 且落盘内容可被重新解析。
    const std::string nested = (root / "nested" / "deeper" / "result.json").string();
    WriteJsonFile(nested, json);
    CHECK(fs::exists(nested));
    std::ifstream in(nested);
    CHECK(static_cast<bool>(in));
    nlohmann::json parsed;
    in >> parsed;
    CHECK(parsed["records"].size() == 1);
    CHECK(parsed["records"][0]["case_id"] == "case-a");
    CHECK(parsed["requests"] == 1);
}

}  // namespace

int main(int argc, char **argv) {
    try {
        CHECK(argc == 2);
        const fs::path dir = argv[1];
        fs::create_directories(dir);
        TestNpyRoundTrip(dir);
        TestNpyRejects(dir);
        TestCases(dir);
        TestStats();
        TestJsonOutput(dir);
        std::cout << "runtime_contracts: OK\n";
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "runtime_contracts: FAILED: " << ex.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "runtime_contracts: FAILED: unknown exception\n";
        return 1;
    }
}
