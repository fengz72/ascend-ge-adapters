#pragma once

#include <string>
#include <vector>

namespace ge_runtime {

// OM/ACL 路径的 dump 配置 (写进 acl.json 的 dump 段, 由 aclInit(configPath) 生效)
struct DumpConfig {
    bool enabled = false;
    std::string configPath;                       // 用户自备 acl.json 时优先, 不再生成
    std::string dumpPath = "./dump_data";
    std::string dumpMode = "output";              // input | output | all
    std::string dumpLevel = "op";                 // op | kernel | all
    std::string dumpData = "tensor";              // tensor | stats
    std::string modelName;
    std::vector<std::string> layers;
};

// OM/ACL 路径的 profiling 配置 (acl.json 的 profiler 段 → PROF_* 会话, 用 tools/parse_profiling.py 解析)
struct ProfilingConfig {
    bool enabled = false;
    std::string outputPath = "./profiling_data";
    std::string aicMetrics;                       // 如 PipeUtilization
    bool taskTime = true;
    bool runtimeApi = true;
    bool ascendcl = true;
};

// GeSession 在线路径的 profiling (经 GEInitialize 全局选项, 不走 acl.json)
struct GeProfilingConfig {
    bool enabled = false;
    std::string outputPath = "./profiling_data";
    std::string aicMetrics = "PipeUtilization";
};

bool GenerateAclJson(const DumpConfig &dump, const ProfilingConfig &prof, const std::string &path);

}  // namespace ge_runtime
