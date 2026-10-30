// Exercise ONNX parsing and GE CompileGraph with the custom passes, without
// aclrtSetDevice, LoadGraph, streams, inputs, or ExecuteGraph.
#include <acl/acl.h>
#include <ge/ge_api.h>
#include <graph/graph.h>
#include <parser/onnx_parser.h>

#include <cstdlib>
#include <iostream>
#include <map>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: compile_only exported-model.onnx logical-device\n";
    return 2;
  }
  // The pass library was loaded at process start. Its preload must not leak
  // into compiler workers started by CANN.
  if (std::getenv("BGE_FIAS_PASS_SO") != nullptr && unsetenv("LD_PRELOAD") != 0)
    return 3;
  std::map<ge::AscendString, ge::AscendString> global = {
      {ge::AscendString("ge.graphRunMode"), ge::AscendString("0")},
      {ge::AscendString("ge.exec.deviceId"), ge::AscendString(argv[2])},
      {ge::AscendString(ge::ir_option::AICORE_NUM), ge::AscendString("12|24")},
  };
  if (ge::GEInitialize(global) != ge::SUCCESS) return 4;
  if (aclInit(nullptr) != ACL_SUCCESS) return 5;
  ge::Graph graph;
  std::map<ge::AscendString, ge::AscendString> parse_options;
  if (ge::aclgrphParseONNX(argv[1], parse_options, graph) != ge::GRAPH_SUCCESS)
    return 6;
  std::map<ge::AscendString, ge::AscendString> session_options = {
      {ge::AscendString("ge.session_device_id"), ge::AscendString(argv[2])},
      {ge::AscendString("ge.exec.precision_mode"), ge::AscendString("force_fp16")},
  };
  int compile_status = 0;
  {
    ge::Session session(session_options);
    if (session.AddGraph(1, graph) != ge::SUCCESS) compile_status = 7;
    else if (session.CompileGraph(1) != ge::SUCCESS) {
      std::cerr << ge::GEGetErrorMsg() << '\n';
      compile_status = 8;
    }
  }
  if (aclFinalize() != ACL_SUCCESS || ge::GEFinalize() != ge::SUCCESS) return 9;
  if (compile_status != 0) return compile_status;
  std::cout << "[BGE_PRECOMPUTE] compile_only_passed\n";
  return 0;
}
