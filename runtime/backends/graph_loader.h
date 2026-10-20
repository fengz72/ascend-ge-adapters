#pragma once
#include <string>
#include <graph/graph.h>
namespace ge_runtime {
bool LoadGraphFile(const std::string &path, const std::string &format, ge::Graph &graph);
}
