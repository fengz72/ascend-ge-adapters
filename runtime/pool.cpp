#include "pool.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ge_runtime {
namespace {

std::string BaseName(const std::string &path) {
    size_t pos = path.find_last_of('/');
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

bool HasBundle(const std::string &dir) {
    struct stat st {};
    return stat(JoinPath(dir, "bundle.json").c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// <dir>/bundle.json 存在 → 单套; 否则收 <dir>/*/bundle.json (按名字排序, 生成器用零填充编号)
std::vector<std::string> FindBundleDirs(const std::string &dir) {
    std::vector<std::string> dirs;
    if (HasBundle(dir)) {
        dirs.push_back(dir);
        return dirs;
    }
    DIR *dp = opendir(dir.c_str());
    if (dp == nullptr) {
        return dirs;
    }
    while (struct dirent *ent = readdir(dp)) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        std::string sub = JoinPath(dir, name);
        struct stat st {};
        if (stat(sub.c_str(), &st) == 0 && S_ISDIR(st.st_mode) && HasBundle(sub)) {
            dirs.push_back(sub);
        }
    }
    closedir(dp);
    std::sort(dirs.begin(), dirs.end());
    return dirs;
}

}  // namespace

RequestPool LoadPool(const IoSpec &spec, const std::string &dir) {
    std::vector<std::string> dirs = FindBundleDirs(dir);
    if (dirs.empty()) {
        throw std::runtime_error("请求池为空: " + dir +
                                 " (期望 <dir>/req_*/bundle.json, 或 <dir>/bundle.json 单套)");
    }

    RequestPool pool;
    std::set<std::string> shapeKeys;
    for (const auto &d : dirs) {
        Bundle bundle = Bundle::Load(JoinPath(d, "bundle.json"));
        Request req;
        req.name = (d == dir) ? "bundle" : BaseName(d);
        req.plans = BuildInputPlans(spec, bundle, d);
        for (const auto &p : req.plans) {
            std::vector<char> bytes;
            if (!ReadBinFile(p.file, bytes)) {
                throw std::runtime_error("读不到请求 " + req.name + " 的输入: " + p.file);
            }
            if (bytes.size() != p.Bytes()) {
                throw std::runtime_error("请求 " + req.name + " 输入 '" + p.logical + "' 字节数 " +
                                         std::to_string(bytes.size()) + " != " +
                                         std::to_string(p.Bytes()));
            }
            req.data.push_back(std::move(bytes));
        }

        std::ostringstream key;
        for (size_t i = 0; i < req.plans.size(); i++) {
            req.tokens = std::max(req.tokens, NumElements(req.plans[i].shape));
            key << ShapeToString(req.plans[i].shape) << (i + 1 < req.plans.size() ? ";" : "");
        }
        req.shapeKey = key.str();
        shapeKeys.insert(req.shapeKey);
        pool.maxOutputBytes = std::max(pool.maxOutputBytes, ExpectedOutputBytes(spec, bundle, 0));
        pool.requests.push_back(std::move(req));
    }

    pool.maxShape.assign(spec.inputs.size(), {});
    pool.maxBytes.assign(spec.inputs.size(), 0);
    for (const auto &req : pool.requests) {
        for (size_t i = 0; i < req.plans.size() && i < pool.maxShape.size(); i++) {
            const std::vector<int64_t> &shape = req.plans[i].shape;
            if (pool.maxShape[i].size() < shape.size()) {
                pool.maxShape[i].resize(shape.size(), 0);
            }
            for (size_t d = 0; d < shape.size(); d++) {
                pool.maxShape[i][d] = std::max(pool.maxShape[i][d], shape[d]);
            }
            pool.maxBytes[i] = std::max(pool.maxBytes[i], req.plans[i].Bytes());
        }
    }
    pool.distinctShapes = shapeKeys.size();
    return pool;
}

}  // namespace ge_runtime
