// cases.json (schema ge-cases/1) 解析实现。不依赖 GE / ACL。
#include "cases.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "npy.h"

namespace ge_execute {
namespace {

std::string ShapeKeyOf(const Case &item) {
    // 逐输入 "[d0, d1];" 拼接。格式稳定, 供 distinct shape 统计与 warmup 覆盖使用。
    std::string key;
    for (const CaseInput &input : item.inputs) {
        key += '[';
        for (size_t i = 0; i < input.shape.size(); ++i) {
            if (i != 0) {
                key += ", ";
            }
            key += std::to_string(input.shape[i]);
        }
        key += "];";
    }
    return key;
}

std::filesystem::path ResolveInputPath(const std::filesystem::path &baseDir,
                                       const std::string &raw) {
    const std::filesystem::path candidate(raw);
    if (candidate.is_absolute()) {
        return candidate;
    }
    return baseDir / candidate;
}

}  // namespace

std::vector<std::string> CasePool::shapeKeys() const {
    std::vector<std::string> keys;
    keys.reserve(cases.size());
    for (const Case &item : cases) {
        keys.push_back(item.shapeKey);
    }
    return keys;
}

CasePool LoadCases(const std::string &casesJsonPath) {
    const auto fail = [&casesJsonPath](const std::string &reason) {
        throw std::runtime_error("LoadCases '" + casesJsonPath + "': " + reason);
    };

    std::ifstream in(casesJsonPath);
    if (!in) {
        fail("cannot open file");
    }
    nlohmann::json root;
    try {
        in >> root;
    } catch (const std::exception &ex) {
        fail(std::string("JSON parse failed: ") + ex.what());
    }

    if (!root.contains("cases") || !root["cases"].is_array()) {
        fail("'cases' missing or not an array");
    }
    const nlohmann::json &caseArray = root["cases"];
    if (caseArray.empty()) {
        fail("'cases' is empty");
    }

    const std::filesystem::path baseDir =
        std::filesystem::path(casesJsonPath).parent_path();

    CasePool pool;
    std::unordered_set<std::string> seenIds;
    size_t inputCount = 0;

    for (size_t ci = 0; ci < caseArray.size(); ++ci) {
        const nlohmann::json &entry = caseArray[ci];
        const std::string where = "cases[" + std::to_string(ci) + "]";
        if (!entry.is_object()) {
            fail(where + " is not an object");
        }

        Case item;
        if (!entry.contains("id") || !entry["id"].is_string() ||
            entry["id"].get<std::string>().empty()) {
            fail(where + ".id missing or empty");
        }
        item.id = entry["id"].get<std::string>();
        if (!seenIds.insert(item.id).second) {
            fail("duplicate case id '" + item.id + "'");
        }

        item.samples = 1;
        if (entry.contains("samples")) {
            if (!entry["samples"].is_number_integer()) {
                fail("case '" + item.id + "': samples must be a positive integer");
            }
            item.samples = entry["samples"].get<int64_t>();
            if (item.samples <= 0) {
                fail("case '" + item.id + "': samples must be a positive integer");
            }
        }

        if (!entry.contains("inputs") || !entry["inputs"].is_array() ||
            entry["inputs"].empty()) {
            fail("case '" + item.id + "': inputs missing or empty");
        }
        const nlohmann::json &inputs = entry["inputs"];
        if (ci == 0) {
            inputCount = inputs.size();
        } else if (inputs.size() != inputCount) {
            fail("case '" + item.id + "': input count " + std::to_string(inputs.size()) +
                 " != first case's " + std::to_string(inputCount));
        }

        for (size_t ii = 0; ii < inputs.size(); ++ii) {
            const nlohmann::json &rawInput = inputs[ii];
            const std::string inWhere =
                "case '" + item.id + "' input[" + std::to_string(ii) + "]";
            if (!rawInput.is_object()) {
                fail(inWhere + " is not an object");
            }

            CaseInput input;
            if (rawInput.contains("name")) {
                if (!rawInput["name"].is_string()) {
                    fail(inWhere + ": name must be a string");
                }
                input.name = rawInput["name"].get<std::string>();
            }
            if (!rawInput.contains("path") || !rawInput["path"].is_string() ||
                rawInput["path"].get<std::string>().empty()) {
                fail(inWhere + ": path missing or empty");
            }
            if (rawInput.contains("dtype")) {
                if (!rawInput["dtype"].is_string()) {
                    fail(inWhere + ": dtype must be a string");
                }
                input.dtype = rawInput["dtype"].get<std::string>();
            }

            const std::string rawPath = rawInput["path"].get<std::string>();
            const std::filesystem::path resolved = ResolveInputPath(baseDir, rawPath);
            NpyArray array = ReadNpy(resolved.string(), input.dtype);
            input.dtype = array.dtype;
            input.shape = std::move(array.shape);
            input.data = std::move(array.data);
            item.inputs.push_back(std::move(input));
        }

        item.shapeKey = ShapeKeyOf(item);
        pool.cases.push_back(std::move(item));
    }

    pool.maxBytes.assign(inputCount, 0);
    std::set<std::string> distinct;
    for (const Case &item : pool.cases) {
        distinct.insert(item.shapeKey);
        for (size_t ii = 0; ii < item.inputs.size(); ++ii) {
            pool.maxBytes[ii] = std::max(pool.maxBytes[ii], item.inputs[ii].data.size());
        }
    }
    pool.distinctShapes = distinct.size();
    return pool;
}

}  // namespace ge_execute
