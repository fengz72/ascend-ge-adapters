#include <graph/graph.h>
#include <graph/operator_factory.h>
#include <register/register_custom_pass.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace bge_fusion {
std::string Name(const ge::GNode &n) { ge::AscendString s; n.GetName(s); return s.GetString(); }
std::string Type(const ge::GNode &n) { ge::AscendString s; n.GetType(s); return s.GetString(); }
void Check(ge::graphStatus status, const char *what) {
  if (status != ge::GRAPH_SUCCESS) throw std::runtime_error(what);
}
bool NoControl(const ge::GNode &n) {
  return n.GetInControlNodes().empty() && n.GetOutControlNodes().empty();
}
bool SameShape(const ge::TensorDesc &a, const ge::TensorDesc &b) {
  return a.GetShape().GetDims() == b.GetShape().GetDims() &&
         a.GetDataType() == ge::DT_FLOAT && b.GetDataType() == ge::DT_FLOAT;
}

// Deliberately BGE-specific: never rewrite an arbitrary LayerNorm in another model.
int Fuse(ge::Graph &graph) {
  int fused = 0;
  for (auto ln : graph.GetDirectNode()) {
    const auto name = Name(ln);
    if (Type(ln) != "LayerNorm" ||
        name.find("/encoder/backbone/encoder/layer.") == std::string::npos ||
        !NoControl(ln) || ln.GetInputsSize() != 3 || ln.GetOutputsSize() != 3)
      continue;
    if (!ln.GetOutDataNodesAndPortIndexs(1).empty() ||
        !ln.GetOutDataNodesAndPortIndexs(2).empty()) continue;
    auto add_edge = ln.GetInDataNodesAndPortIndexs(0);
    if (!add_edge.first || add_edge.second != 0) continue;
    auto add = *add_edge.first;
    if (Type(add) != "Add" || add.GetInputsSize() != 2 ||
        add.GetOutputsSize() != 1 || !NoControl(add)) continue;
    auto users = add.GetOutDataNodesAndPortIndexs(0);
    if (users.size() != 1 || Name(*users[0].first) != name || users[0].second != 0)
      continue;
    ge::TensorDesc x1, x2, gamma, beta, y, mean, variance;
    Check(add.GetInputDesc(0, x1), "read residual input 0");
    Check(add.GetInputDesc(1, x2), "read residual input 1");
    Check(ln.GetInputDesc(1, gamma), "read gamma");
    Check(ln.GetInputDesc(2, beta), "read beta");
    Check(ln.GetOutputDesc(0, y), "read y");
    Check(ln.GetOutputDesc(1, mean), "read mean");
    Check(ln.GetOutputDesc(2, variance), "read variance");
    const auto dims = x1.GetShape().GetDims();
    if (!SameShape(x1, x2) || dims.size() != 3 || dims.back() != 1024 ||
        y.GetDataType() != ge::DT_FLOAT || gamma.GetDataType() != ge::DT_FLOAT ||
        beta.GetDataType() != ge::DT_FLOAT ||
        gamma.GetShape().GetDims() != std::vector<int64_t>{1024} ||
        beta.GetShape().GetDims() != std::vector<int64_t>{1024}) continue;
    int64_t norm_axis = 0, params_axis = 0;
    float epsilon = 0;
    if (ln.GetAttr("begin_norm_axis", norm_axis) != ge::GRAPH_SUCCESS ||
        ln.GetAttr("begin_params_axis", params_axis) != ge::GRAPH_SUCCESS ||
        ln.GetAttr("epsilon", epsilon) != ge::GRAPH_SUCCESS ||
        (norm_axis != -1 && norm_axis != 2) ||
        (params_axis != -1 && params_axis != 2) ||
        epsilon != static_cast<float>(1e-12)) continue;
    std::vector<std::pair<ge::GNodePtr, int32_t>> inputs = {
        add.GetInDataNodesAndPortIndexs(0), add.GetInDataNodesAndPortIndexs(1),
        ln.GetInDataNodesAndPortIndexs(1), ln.GetInDataNodesAndPortIndexs(2)};
    if (std::any_of(inputs.begin(), inputs.end(), [](const auto &e) { return !e.first; }))
      continue;
    auto op = ge::OperatorFactory::CreateOperator((name + "/BgeAddLayerNorm").c_str(),
                                                  "AddLayerNorm");
    if (op.GetInputsSize() < 4 || op.GetOutputsSize() != 4)
      throw std::runtime_error("CANN AddLayerNorm schema is not registered or has unexpected ports");
    op.SetAttr("epsilon", epsilon);
    Check(op.UpdateInputDesc(0U, x1), "set x1");
    Check(op.UpdateInputDesc(1, x2), "set x2");
    Check(op.UpdateInputDesc(2, gamma), "set gamma");
    Check(op.UpdateInputDesc(3, beta), "set beta");
    Check(op.UpdateOutputDesc(0U, y), "set y");
    Check(op.UpdateOutputDesc(1, mean), "set mean");
    // The statistics have identical shape/dtype; AddLayerNorm's third output is
    // rstd rather than variance. Neither statistic has a consumer in this pattern.
    Check(op.UpdateOutputDesc(2, variance), "set rstd descriptor");
    Check(op.UpdateOutputDesc(3, x1), "set residual-sum descriptor");
    auto replacement = graph.AddNodeByOp(op);
    for (size_t i = 0; i < inputs.size(); ++i)
      Check(graph.AddDataEdge(*inputs[i].first, inputs[i].second, replacement, i),
            "connect AddLayerNorm input");
    const auto downstream = ln.GetOutDataNodesAndPortIndexs(0);
    for (const auto &edge : downstream) {
      Check(graph.RemoveEdge(ln, 0, *edge.first, edge.second), "remove original LN output");
      Check(graph.AddDataEdge(replacement, 0, *edge.first, edge.second), "replace LN output");
    }
    Check(graph.RemoveNode(ln), "remove original LN");
    Check(graph.RemoveNode(add), "remove residual Add");
    ++fused;
  }
  return fused;
}

ge::Status Run(ge::GraphPtr &graph, ge::CustomPassContext &context) {
  try {
    const auto count = Fuse(*graph);
    std::cout << "[BGE_FUSION] AddLayerNorm replacements=" << count << std::endl;
    return ge::SUCCESS;
  } catch (const std::exception &e) {
    context.SetErrorMessage(ge::AscendString(e.what()));
    std::cerr << "[BGE_FUSION] failed: " << e.what() << std::endl;
    return ge::FAILED;
  }
}
}

#ifndef BGE_PASS_NO_REGISTER
REGISTER_CUSTOM_PASS("BgeResidualAddLayerNormFP32")
    .CustomPassFn(bge_fusion::Run)
    .Stage(ge::CustomPassStage::kAfterBuiltinFusionPass);
#endif
