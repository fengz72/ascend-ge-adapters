#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace ge_runtime {

struct IoNode {
    std::string node;
    std::string logical;
    std::string dtype;
    std::string format = "ND";
    std::vector<int64_t> shape;
    std::vector<int64_t> dynamic_dims;
    size_t max_bytes = 0;
};

struct IoSpec {
    std::vector<IoNode> inputs;
    std::vector<IoNode> outputs;

    static IoSpec FromJson(const nlohmann::json &value);
};

struct ModelGraph {
    std::string backend;
    std::string graph_path;
    std::string graph_format;
    std::string om_path;
    int device = 0;
};

struct TensorPlan {
    std::string node;
    std::string logical;
    std::string dtype;
    std::string format;
    std::vector<int64_t> shape;

    size_t Bytes() const;
};

struct HostTensor {
    int instance = 0;
    std::string logical;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<char> data;
};

size_t DtypeSize(const std::string &dtype);
bool SameDtype(const std::string &left, const std::string &right);
int64_t NumElements(const std::vector<int64_t> &shape);
std::string ShapeToString(const std::vector<int64_t> &shape);
bool ReadBinFile(const std::string &path, std::vector<char> &out);
bool MakeDirs(const std::string &path);

}  // namespace ge_runtime
