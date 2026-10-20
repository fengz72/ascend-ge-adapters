#include "io_spec.h"
#include <set>
#include <limits>

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

// 可选字段缺失或为 null 时使用空串。
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
    n.logical = Str(j, "name");
    n.max_bytes = j.value("max_bytes", size_t{0});
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

}  // namespace

IoSpec IoSpec::FromJson(const Json &j) {
    if (j.value("schema", "") != "ge-io/1") throw std::runtime_error("unsupported IOSpec schema");
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
        throw std::runtime_error("io_spec has no inputs");
    }
    for (const auto *nodes : {&spec.inputs, &spec.outputs}) {
        std::set<std::string> names;
        for (const auto &node : *nodes) {
            if (node.logical.empty() || !names.insert(node.logical).second || DtypeSize(node.dtype)==0)
                throw std::runtime_error("invalid/duplicate IOSpec name or dtype");
        }
    }
    if (spec.outputs.empty()) throw std::runtime_error("IOSpec requires outputs");
    return spec;
}

size_t TensorPlan::Bytes() const {
    auto count = static_cast<size_t>(NumElements(shape));
    auto width = DtypeSize(dtype);
    if (width==0 || count>std::numeric_limits<size_t>::max()/width) throw std::runtime_error("tensor byte size overflow");
    return count * width;
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
        if (d<1 || n>std::numeric_limits<int64_t>::max()/d) throw std::runtime_error("tensor element count overflow");
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

}  // namespace ge_runtime
