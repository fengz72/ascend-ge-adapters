#include "graph_loader.h"
#include <parser/onnx_parser.h>
#include <map>
namespace ge_runtime {
bool LoadGraphFile(const std::string &path, const std::string &format, ge::Graph &graph) {
    if(format=="air") return graph.LoadFromFile(path.c_str())==ge::GRAPH_SUCCESS;
    if(format=="onnx") {
        std::map<ge::AscendString,ge::AscendString> options;
        return ge::aclgrphParseONNX(path.c_str(),options,graph)==ge::GRAPH_SUCCESS;
    }
    return false;
}
}
