#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ge_runtime {

struct IoNode {
    std::string node;
    std::string logical;
    std::string dtype;
    std::string format = "ND";
    std::vector<int64_t> shape;
    std::vector<int64_t> dynamic_dims;
};

struct IoSpec {
    std::vector<IoNode> inputs;
    std::vector<IoNode> outputs;

    bool HasDynamicInput() const;
    static IoSpec Load(const std::string &path);
};

struct BundleEntry {
    std::string logical;
    std::vector<int64_t> shape;
    std::string file;
};

struct Bundle {
    std::vector<BundleEntry> inputs;
    BundleEntry golden;

    static Bundle Load(const std::string &path);
};

struct Manifest {
    std::string backend;
    std::string graph_path;
    std::string om_path;
    std::string io_spec;
    std::string bundle;
    std::string base_dir;
    int device = 0;

    std::string Resolve(const std::string &rel) const;
    static Manifest Load(const std::string &path);
};

struct TensorPlan {
    std::string node;
    std::string logical;
    std::string dtype;
    std::string format;
    std::string file;
    std::vector<int64_t> shape;

    size_t Bytes() const;
};

struct HostTensor {
    std::string logical;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<char> data;
};

std::vector<TensorPlan> BuildInputPlans(const IoSpec &spec, const Bundle &bundle,
                                       const std::string &bundle_dir);

// 用 bundle.golden 的具体 shape 解析 io_spec 输出的动态维 → 精确字节数; 解析不出返回 0
size_t ExpectedOutputBytes(const IoSpec &spec, const Bundle &bundle, size_t index);

size_t DtypeSize(const std::string &dtype);
int64_t NumElements(const std::vector<int64_t> &shape);
std::string ShapeToString(const std::vector<int64_t> &shape);
std::string DirName(const std::string &path);
std::string JoinPath(const std::string &dir, const std::string &rel);

bool ReadBinFile(const std::string &path, std::vector<char> &out);
bool WriteBinFile(const std::string &path, const void *data, size_t bytes);
bool MakeDirs(const std::string &path);
bool SaveOutputs(const std::string &dir, const std::vector<HostTensor> &outputs,
                 std::string &indexPath);

}  // namespace ge_runtime
