#include "acl_json.h"

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

namespace ge_runtime {

bool GenerateAclJson(const DumpConfig &dump, const ProfilingConfig &prof, const std::string &path) {
    using Json = nlohmann::json;
    Json root = Json::object();

    if (dump.enabled) {
        Json item = Json::object();
        if (!dump.modelName.empty()) {
            item["model_name"] = dump.modelName;
        }
        if (!dump.layers.empty()) {
            item["layer"] = dump.layers;
        }
        Json dumpList = Json::array();
        dumpList.push_back(item.empty() ? Json::object() : item);
        root["dump"] = {{"dump_list", dumpList},
                        {"dump_path", dump.dumpPath},
                        {"dump_mode", dump.dumpMode},
                        {"dump_level", dump.dumpLevel},
                        {"dump_data", dump.dumpData},
                        {"dump_op_switch", "off"}};
    }

    if (prof.enabled) {
        Json p = {{"switch", "on"},
                  {"output", prof.outputPath},
                  {"task_time", prof.taskTime ? "on" : "off"},
                  {"runtime_api", prof.runtimeApi ? "on" : "off"},
                  {"ascendcl", prof.ascendcl ? "on" : "off"}};
        if (!prof.aicMetrics.empty()) {
            p["aic_metrics"] = prof.aicMetrics;
        }
        root["profiler"] = p;
    }

    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ERROR] 无法写 acl.json: " << path << std::endl;
        return false;
    }
    f << root.dump(4) << std::endl;
    return true;
}

}  // namespace ge_runtime
