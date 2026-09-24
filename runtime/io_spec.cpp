#include "io_spec.h"

#include <sys/stat.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace ge_runtime {
namespace {

using Json = nlohmann::json;

Json ParseFile(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open json file: " + path);
    }
    try {
        return Json::parse(f);
    } catch (const std::exception &e) {
        throw std::runtime_error("invalid json in " + path + ": " + e.what());
    }
}

std::vector<int64_t> ToShape(const Json &j) {
    std::vector<int64_t> shape;
    if (!j.is_array()) {
        return shape;
    }
    for (const auto &d : j) {
        shape.push_back(d.get<int64_t>());
    }
    return shape;
}

// 字段缺失或为 null 都视作空串 (manifest 的 om_path 在 ge_session 下即为 null)
std::string Str(const Json &j, const char *key) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        return "";
    }
    return it->get<std::string>();
}

IoNode ToIoNode(const Json &j) {
    IoNode n;
    n.node = Str(j, "node");
    n.logical = Str(j, "logical");
    n.dtype = Str(j, "dtype");
    std::string format = Str(j, "format");
    n.format = format.empty() ? "ND" : format;
    if (j.contains("shape")) {
        n.shape = ToShape(j["shape"]);
    }
    if (j.contains("dynamic_dims")) {
        n.dynamic_dims = ToShape(j["dynamic_dims"]);
    }
    return n;
}

BundleEntry ToBundleEntry(const Json &j) {
    BundleEntry e;
    e.logical = Str(j, "logical");
    e.file = Str(j, "file");
    if (j.contains("shape")) {
        e.shape = ToShape(j["shape"]);
    }
    return e;
}

// bundle 路径的 plan 校验: dtype 可识别、shape 具体、
// .bin 存在且字节数 == shape×dtype。只 stat 不读盘 — 真正的读发生在 backend 的 H2D。
void ValidatePlan(const TensorPlan &p, const char *shapeFrom) {
    if (p.dtype.empty()) {
        throw std::runtime_error("io_spec input '" + p.logical + "' has no dtype");
    }
    if (DtypeSize(p.dtype) == 0) {
        throw std::runtime_error("unsupported dtype '" + p.dtype + "' for input '" + p.logical + "'");
    }
    if (p.shape.empty()) {
        throw std::runtime_error(std::string(shapeFrom) + " gives no concrete shape for input '" +
                                 p.logical + "'");
    }
    for (int64_t d : p.shape) {
        if (d <= 0) {
            throw std::runtime_error("input '" + p.logical + "' shape must be positive, got " +
                                     ShapeToString(p.shape) + " (from " + shapeFrom + ")");
        }
    }
    struct stat st {};
    if (stat(p.file.c_str(), &st) != 0) {
        throw std::runtime_error("cannot stat input file: " + p.file + " (" + strerror(errno) + ")");
    }
    if (!S_ISREG(st.st_mode)) {
        throw std::runtime_error("input file is not a regular file: " + p.file);
    }
    size_t fileSize = static_cast<size_t>(st.st_size);
    if (fileSize != p.Bytes()) {
        throw std::runtime_error("input '" + p.logical + "' file " + p.file + " size " +
                                 std::to_string(fileSize) + " != shape bytes " +
                                 std::to_string(p.Bytes()) + " (shape=" + ShapeToString(p.shape) +
                                 ", dtype=" + p.dtype + ")");
    }
}

}  // namespace

bool IoSpec::HasDynamicInput() const {
    for (const auto &n : inputs) {
        if (!n.dynamic_dims.empty()) {
            return true;
        }
        for (int64_t d : n.shape) {
            if (d < 0) {
                return true;
            }
        }
    }
    return false;
}

IoSpec IoSpec::Load(const std::string &path) {
    Json j = ParseFile(path);
    IoSpec spec;
    if (j.contains("inputs")) {
        for (const auto &e : j["inputs"]) {
            spec.inputs.push_back(ToIoNode(e));
        }
    }
    if (j.contains("outputs")) {
        for (const auto &e : j["outputs"]) {
            spec.outputs.push_back(ToIoNode(e));
        }
    }
    if (spec.inputs.empty()) {
        throw std::runtime_error("io_spec has no inputs: " + path);
    }
    return spec;
}

Bundle Bundle::Load(const std::string &path) {
    Json j = ParseFile(path);
    Bundle b;
    if (j.contains("inputs")) {
        for (const auto &e : j["inputs"]) {
            b.inputs.push_back(ToBundleEntry(e));
        }
    }
    if (j.contains("golden") && j["golden"].is_object()) {
        b.golden = ToBundleEntry(j["golden"]);
    }
    if (b.inputs.empty()) {
        throw std::runtime_error("bundle has no inputs: " + path);
    }
    return b;
}

std::string Manifest::Resolve(const std::string &rel) const {
    return JoinPath(base_dir, rel);
}

Manifest Manifest::Load(const std::string &path) {
    Json j = ParseFile(path);
    Manifest m;
    m.backend = Str(j, "backend");
    m.graph_path = Str(j, "graph_path");
    m.om_path = Str(j, "om_path");
    m.io_spec = Str(j, "io_spec");
    m.bundle = Str(j, "bundle");
    if (j.contains("device") && j["device"].is_number_integer()) {
        m.device = j["device"].get<int>();
    }
    // manifest 内路径相对 base_dir (= io/ 的父目录, 见 docs §5.5)
    m.base_dir = DirName(DirName(path));
    if (m.backend.empty()) {
        throw std::runtime_error("manifest missing 'backend': " + path);
    }
    return m;
}

size_t TensorPlan::Bytes() const {
    return static_cast<size_t>(NumElements(shape)) * DtypeSize(dtype);
}

std::vector<TensorPlan> BuildInputPlans(const IoSpec &spec, const Bundle &bundle,
                                       const std::string &bundle_dir) {
    std::vector<TensorPlan> plans;
    for (size_t i = 0; i < spec.inputs.size(); i++) {
        const IoNode &node = spec.inputs[i];
        const BundleEntry *entry = nullptr;
        for (const auto &e : bundle.inputs) {
            if (!e.logical.empty() && e.logical == node.logical) {
                entry = &e;
                break;
            }
        }
        if (entry == nullptr) {
            if (i >= bundle.inputs.size()) {
                throw std::runtime_error("bundle lacks input #" + std::to_string(i) +
                                         " (logical=" + node.logical + ")");
            }
            entry = &bundle.inputs[i];
        }

        TensorPlan p;
        p.node = node.node.empty() ? node.logical : node.node;
        p.logical = node.logical;
        p.dtype = node.dtype;
        p.format = node.format.empty() ? "ND" : node.format;
        p.shape = entry->shape;
        p.file = JoinPath(bundle_dir, entry->file);

        ValidatePlan(p, "bundle");
        plans.push_back(p);
    }
    return plans;
}

size_t ExpectedOutputBytes(const IoSpec &spec, const Bundle &bundle, size_t index) {
    if (index >= spec.outputs.size()) {
        return 0;
    }
    const IoNode &node = spec.outputs[index];
    const std::vector<int64_t> &concrete = bundle.golden.shape;
    size_t elem = DtypeSize(node.dtype);
    if (elem == 0 || concrete.size() != node.shape.size()) {
        return 0;
    }
    int64_t numel = 1;
    for (size_t i = 0; i < node.shape.size(); i++) {
        int64_t d = node.shape[i];
        if (d < 0) {
            d = concrete[i];                 // 动态维取 golden 的具体值
        } else if (d != concrete[i]) {
            return 0;                        // 静态维与 golden 不符 → 不敢推导
        }
        if (d <= 0) {
            return 0;
        }
        numel *= d;
    }
    return static_cast<size_t>(numel) * elem;
}

size_t DtypeSize(const std::string &dtype) {
    if (dtype == "float16" || dtype == "bfloat16" || dtype == "int16" || dtype == "uint16") {
        return 2;
    }
    if (dtype == "float32" || dtype == "float" || dtype == "int32" || dtype == "uint32") {
        return 4;
    }
    if (dtype == "float64" || dtype == "double" || dtype == "int64" || dtype == "uint64") {
        return 8;
    }
    if (dtype == "int8" || dtype == "uint8" || dtype == "bool") {
        return 1;
    }
    return 0;
}

int64_t NumElements(const std::vector<int64_t> &shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= d;
    }
    return n;
}

std::string ShapeToString(const std::vector<int64_t> &shape) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        os << (i ? ", " : "") << shape[i];
    }
    os << "]";
    return os.str();
}

bool ReadBinFile(const std::string &path, std::vector<char> &out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return false;
    }
    std::streamsize size = f.tellg();
    if (size < 0) {
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    if (size > 0 && !f.read(out.data(), size)) {
        return false;
    }
    return true;
}

bool WriteBinFile(const std::string &path, const void *data, size_t bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
        return false;
    }
    if (bytes > 0) {
        f.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    }
    bool ok = f.good();
    f.close();
    return ok;
}

bool MakeDirs(const std::string &path) {
    if (path.empty()) {
        return false;
    }
    std::string cur;
    std::istringstream ss(path);
    std::string token;
    if (path[0] == '/') {
        cur = "/";
    }
    while (std::getline(ss, token, '/')) {
        if (token.empty()) {
            continue;
        }
        cur = cur.empty() ? token : (cur.back() == '/' ? cur + token : cur + "/" + token);
        if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
    }
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool SaveOutputs(const std::string &dir, const std::vector<HostTensor> &outputs,
                 std::string &indexPath) {
    if (!MakeDirs(dir)) {
        fprintf(stderr, "[ERROR] cannot create output dir: %s\n", dir.c_str());
        return false;
    }
    Json items = Json::array();
    for (size_t i = 0; i < outputs.size(); i++) {
        const HostTensor &t = outputs[i];
        std::string rel = "output_" + std::to_string(i) + ".bin";
        if (!WriteBinFile(JoinPath(dir, rel), t.data.data(), t.data.size())) {
            fprintf(stderr, "[ERROR] cannot write %s\n", JoinPath(dir, rel).c_str());
            return false;
        }
        items.push_back({{"logical", t.logical},
                         {"dtype", t.dtype},
                         {"shape", t.shape},
                         {"bytes", t.data.size()},
                         {"file", rel}});
    }
    indexPath = JoinPath(dir, "outputs.json");
    std::ofstream f(indexPath);
    if (!f.is_open()) {
        fprintf(stderr, "[ERROR] cannot write %s\n", indexPath.c_str());
        return false;
    }
    f << Json({{"outputs", items}}).dump(2) << std::endl;
    return true;
}

std::string DirName(const std::string &path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

std::string JoinPath(const std::string &dir, const std::string &rel) {
    if (rel.empty()) {
        return rel;
    }
    if (rel[0] == '/') {
        return rel;
    }
    if (dir.empty() || dir == ".") {
        return rel;
    }
    return dir + "/" + rel;
}

}  // namespace ge_runtime
