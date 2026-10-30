#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "all_ops.h"
#include "register/register_custom_pass.h"

// First-layer experiment for the original BGE ONNX graph.  Deliberately match
// its complete attention boundary; do not apply this pass to unrelated graphs.
namespace {

constexpr char kTail[] =
    "/encoder/backbone/encoder/layer.0/attention/self/Transpose_3";
constexpr char kFusedTailPrefix[] =
    "PartitionedCall_/encoder/backbone/encoder/layer.0/attention/self/Transpose_3_Transpose_";
constexpr char kMaskSelect[] = "/encoder/backbone/Where_2";
constexpr char kSelf[] = "/encoder/backbone/encoder/layer.0/attention/self/";

bool TailName(const std::string &name) {
  return name == kTail || name.rfind(kFusedTailPrefix, 0) == 0;
}

bool SelfName(const ge::GNode &node, const char *suffix);

std::string Name(const ge::GNode &n) {
  ge::AscendString s;
  if (n.GetName(s) != ge::GRAPH_SUCCESS || s.GetString() == nullptr) return {};
  return s.GetString();
}

bool Type(const ge::GNode &n, const char *expected) {
  ge::AscendString s;
  return n.GetType(s) == ge::GRAPH_SUCCESS && s == expected;
}

bool SelfName(const ge::GNode &node, const char *suffix) {
  return Name(node) == std::string(kSelf) + suffix;
}

bool KnownOrDynamic(int64_t actual, int64_t expected) {
  return actual == expected || actual == -1;
}

bool Input(const ge::GNode &n, int index, ge::GNode &src) {
  auto edge = n.GetInDataNodesAndPortIndexs(index);
  if (edge.first == nullptr || edge.second != 0) return false;
  src = *edge.first;
  return true;
}

bool TypedInput(const ge::GNode &n, int index, const char *type,
                ge::GNode &src) {
  return Input(n, index, src) && Type(src, type);
}

bool SingleConsumer(const ge::GNode &n) {
  return n.GetOutDataNodesAndPortIndexs(0).size() == 1;
}

bool Desc(const ge::GNode &n, ge::TensorDesc &out) {
  return n.GetOutputDesc(0, out) == ge::GRAPH_SUCCESS;
}

void DumpCandidate(ge::Graph &graph) {
  for (const auto &node : graph.GetAllNodes()) {
    const auto name = Name(node);
    if (name.find("layer.0/attention/self/") == std::string::npos &&
        name != kMaskSelect) continue;
    ge::AscendString type;
    node.GetType(type);
    std::cerr << "[BGE_FIAS] node " << name << " type="
              << (type.GetString() ? type.GetString() : "?");
    for (int i = 0; i < 3; ++i) {
      auto edge = node.GetInDataNodesAndPortIndexs(i);
      if (edge.first != nullptr) {
        std::cerr << " in" << i << "=" << Name(*edge.first)
                  << ":" << edge.second;
      }
    }
    ge::TensorDesc desc;
    if (Desc(node, desc)) {
      std::cerr << " out_dtype=" << static_cast<int>(desc.GetDataType())
                << " out_shape=";
      for (auto dim : desc.GetShape().GetDims()) std::cerr << dim << ',';
    }
    std::cerr << '\n';
  }
}

struct Match {
  ge::GNode tail, bmm2, softmax, score_add, bmm1;
  ge::GNode q_mul, k_mul, q_transpose, k_transpose, v_transpose;
  ge::GNode query, key, value, mask_select, mask;
  ge::TensorDesc q_desc, k_desc, v_desc, mask_desc, out_desc;
};

bool MatchFirstLayer(const ge::GNode &tail, Match &m) {
  if (!TailName(Name(tail)) || !Type(tail, "Transpose")) return false;
  m.tail = tail;
  if (!TypedInput(tail, 0, "BatchMatMulV2", m.bmm2) ||
      !TypedInput(m.bmm2, 0, "SoftmaxV2", m.softmax) ||
      !TypedInput(m.bmm2, 1, "Transpose", m.v_transpose) ||
      !TypedInput(m.softmax, 0, "Add", m.score_add) ||
      !TypedInput(m.score_add, 0, "BatchMatMulV2", m.bmm1) ||
      !TypedInput(m.score_add, 1, "SelectV2", m.mask_select) ||
      Name(m.mask_select) != kMaskSelect ||
      !Input(m.mask_select, 0, m.mask) ||
      !TypedInput(m.bmm1, 0, "Mul", m.q_mul) ||
      !TypedInput(m.bmm1, 1, "Mul", m.k_mul) ||
      !TypedInput(m.q_mul, 0, "Transpose", m.q_transpose) ||
      !TypedInput(m.k_mul, 0, "Transpose", m.k_transpose) ||
      !TypedInput(m.q_transpose, 0, "Reshape", m.query) ||
      !TypedInput(m.k_transpose, 0, "Reshape", m.key) ||
      !TypedInput(m.v_transpose, 0, "Reshape", m.value)) return false;

  if (!SelfName(m.bmm2, "MatMul_1") ||
      !SelfName(m.softmax, "Softmax") ||
      !SelfName(m.score_add, "Add") ||
      !SelfName(m.bmm1, "MatMul") ||
      !SelfName(m.q_mul, "Mul") ||
      !SelfName(m.k_mul, "Mul_1") ||
      !SelfName(m.query, "Reshape") ||
      !SelfName(m.key, "Reshape_1") ||
      !SelfName(m.value, "Reshape_2")) return false;

  // The three reshapes are BSND.  All dynamic B/S values are retained; only
  // head count and width are fixed by this BGE model.
  if (!Desc(m.query, m.q_desc) || !Desc(m.key, m.k_desc) ||
      !Desc(m.value, m.v_desc) || !Desc(m.mask, m.mask_desc) ||
      !Desc(m.tail, m.out_desc)) return false;
  auto q = m.q_desc.GetShape();
  auto k = m.k_desc.GetShape();
  auto v = m.v_desc.GetShape();
  auto a = m.mask_desc.GetShape();
  auto o = m.out_desc.GetShape();
  if (q.GetDimNum() != 4 || k.GetDimNum() != 4 || v.GetDimNum() != 4 ||
      a.GetDimNum() != 4 || o.GetDimNum() != 4 ||
      !KnownOrDynamic(q.GetDim(2), 16) || !KnownOrDynamic(q.GetDim(3), 64) ||
      !KnownOrDynamic(k.GetDim(2), 16) || !KnownOrDynamic(k.GetDim(3), 64) ||
      !KnownOrDynamic(v.GetDim(2), 16) || !KnownOrDynamic(v.GetDim(3), 64) ||
      !KnownOrDynamic(a.GetDim(1), 1) ||
      m.mask_desc.GetDataType() != ge::DT_BOOL) return false;
  // This pass runs before GE's force_fp16 precision conversion. FIAS is
  // initially wired with the model's FP32 descriptors; GE converts its Q/K/V
  // and output to FP16 in the next optimization stage (checked below).
  if (m.q_desc.GetDataType() != ge::DT_FLOAT ||
      m.k_desc.GetDataType() != ge::DT_FLOAT ||
      m.v_desc.GetDataType() != ge::DT_FLOAT ||
      m.out_desc.GetDataType() != ge::DT_FLOAT) return false;

  // Rewriting shared or externally consumed intermediates would change graph
  // semantics. The mask remains shared by all 24 layers.
  for (const auto &n : {m.bmm2, m.softmax, m.score_add, m.bmm1,
                        m.q_mul, m.k_mul}) {
    if (!SingleConsumer(n)) return false;
  }
  return true;
}

ge::Status Fuse(ge::Graph &graph, Match &m) {
  ge::op::FusedInferAttentionScore op("BgeFusedInferAttentionLayer0");
  // The default create_dynamic_input_* appends after all optional inputs.
  // FIAS lowering expects key/value at their IR positions immediately after Q.
  op.create_dynamic_input_byindex_key(1, 1);
  op.create_dynamic_input_byindex_value(1, 2);
  op.set_attr_num_heads(16);
  op.set_attr_num_key_value_heads(16);
  op.set_attr_input_layout("BSND");
  op.set_attr_scale(0.125f);
  op.set_attr_sparse_mode(0);  // full bidirectional padding mask
  op.set_attr_softmax_lse_flag(false);
  if (op.update_input_desc_query(m.q_desc) != ge::GRAPH_SUCCESS ||
      op.UpdateDynamicInputDesc("key", 0, m.k_desc) != ge::GRAPH_SUCCESS ||
      op.UpdateDynamicInputDesc("value", 0, m.v_desc) != ge::GRAPH_SUCCESS ||
      op.update_input_desc_atten_mask(m.mask_desc) != ge::GRAPH_SUCCESS ||
      op.update_output_desc_attention_out(m.out_desc) != ge::GRAPH_SUCCESS) {
    return ge::FAILED;
  }
  ge::TensorDesc lse(ge::Shape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
  if (op.update_output_desc_softmax_lse(lse) != ge::GRAPH_SUCCESS) {
    return ge::FAILED;
  }

  ge::GNode fused = graph.AddNodeByOp(op);
  int32_t query_port = -1, mask_port = -1;
  std::vector<int32_t> key_ports, value_ports;
  if (fused.GetInputIndexByName("query", query_port) != ge::GRAPH_SUCCESS ||
      fused.GetDynamicInputIndexesByName("key", key_ports) != ge::GRAPH_SUCCESS ||
      fused.GetDynamicInputIndexesByName("value", value_ports) != ge::GRAPH_SUCCESS ||
      fused.GetInputIndexByName("atten_mask", mask_port) != ge::GRAPH_SUCCESS ||
      query_port < 0 || key_ports.size() != 1 || value_ports.size() != 1 ||
      mask_port < 0) {
    std::cerr << "[BGE_FIAS] cannot resolve FIAS input ports\n";
    return ge::FAILED;
  }
  std::cerr << "[BGE_FIAS] ports query=" << query_port
            << " key=" << key_ports[0] << " value=" << value_ports[0]
            << " mask=" << mask_port << '\n';
  if (fused.UpdateInputDesc(query_port, m.q_desc) != ge::GRAPH_SUCCESS ||
      fused.UpdateInputDesc(key_ports[0], m.k_desc) != ge::GRAPH_SUCCESS ||
      fused.UpdateInputDesc(value_ports[0], m.v_desc) != ge::GRAPH_SUCCESS ||
      fused.UpdateInputDesc(mask_port, m.mask_desc) != ge::GRAPH_SUCCESS) {
    std::cerr << "[BGE_FIAS] cannot set FIAS input descriptors\n";
    return ge::FAILED;
  }
  for (int port : {query_port, key_ports[0], value_ports[0], mask_port}) {
    ge::TensorDesc desc;
    auto status = fused.GetInputDesc(port, desc);
    std::cerr << "[BGE_FIAS] input_desc port=" << port
              << " status=" << status;
    if (status == ge::GRAPH_SUCCESS) {
      std::cerr << " dtype=" << static_cast<int>(desc.GetDataType())
                << " shape=";
      for (auto dim : desc.GetShape().GetDims()) std::cerr << dim << ',';
    }
    std::cerr << '\n';
  }
  if (graph.AddDataEdge(m.query, 0, fused, query_port) != ge::GRAPH_SUCCESS ||
      graph.AddDataEdge(m.key, 0, fused, key_ports[0]) != ge::GRAPH_SUCCESS ||
      graph.AddDataEdge(m.value, 0, fused, value_ports[0]) != ge::GRAPH_SUCCESS ||
      graph.AddDataEdge(m.mask, 0, fused, mask_port) != ge::GRAPH_SUCCESS) {
    return ge::FAILED;
  }

  // FIAS output 0 is BSND, exactly the shape of the original tail Transpose.
  const auto consumers = m.tail.GetOutDataNodesAndPortIndexs(0);
  std::cerr << "[BGE_FIAS] tail_consumers=" << consumers.size();
  for (const auto &edge : consumers) {
    if (edge.first != nullptr) {
      ge::AscendString type;
      edge.first->GetType(type);
      std::cerr << " " << Name(*edge.first) << ":" << edge.second
                << "(" << (type.GetString() ? type.GetString() : "?") << ")";
    }
  }
  std::cerr << '\n';
  for (const auto &edge : consumers) {
    if (edge.first == nullptr ||
        graph.RemoveEdge(m.tail, 0, *edge.first, edge.second) != ge::GRAPH_SUCCESS ||
        graph.AddDataEdge(fused, 0, *edge.first, edge.second) != ge::GRAPH_SUCCESS) {
      return ge::FAILED;
    }
  }
  for (const auto &n : {m.tail, m.bmm2, m.softmax, m.score_add, m.bmm1,
                        m.q_mul, m.k_mul}) {
    ge::GNode copy = n;
    if (graph.RemoveNode(copy) != ge::GRAPH_SUCCESS) return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::GraphPtr FindOwner(const ge::GraphPtr &graph, ge::GNode &tail) {
  for (const auto &node : graph->GetDirectNode()) {
    if (TailName(Name(node))) {
      tail = node;
      return graph;
    }
  }
  for (const auto &child : graph->GetAllSubgraphs()) {
    if (child == nullptr) continue;
    auto owner = FindOwner(child, tail);
    if (owner != nullptr) return owner;
  }
  return nullptr;
}

ge::Status Run(ge::GraphPtr &graph, ge::CustomPassContext &context) {
  if (graph == nullptr) return ge::FAILED;
  ge::GNode node;
  auto owner = FindOwner(graph, node);
  if (owner != nullptr) {
    auto inspect = [&](const char *phase) {
      size_t netoutputs = 0;
      for (const auto &direct : owner->GetDirectNode()) {
        if (Type(direct, "NetOutput")) ++netoutputs;
      }
      std::cerr << "[BGE_FIAS] " << phase << " direct_nodes="
                << owner->GetDirectNode().size() << " netoutputs="
                << netoutputs << '\n';
    };
    std::cerr << "[BGE_FIAS] root=" << graph->GetName()
              << " owner=" << owner->GetName() << '\n';
    inspect("before");
    Match m;
    if (!MatchFirstLayer(node, m)) {
      DumpCandidate(*owner);
      context.SetErrorMessage(ge::AscendString("BGE FIAS layer0 pattern or FP16 descriptor mismatch"));
      std::cerr << "[BGE_FIAS] layer0 pattern or FP16 descriptor mismatch\n";
      return ge::FAILED;
    }
    auto result = Fuse(*owner, m);
    inspect("after");
    if (std::getenv("BGE_FIAS_DUMP_STRUCTURE") != nullptr) {
      std::ofstream out("/tmp/bge-fias-graph.txt");
      if (out) owner->Dump(ge::Graph::DumpFormat::kReadable, out);
    }
    std::cerr << "[BGE_FIAS] layer0 replacement="
              << (result == ge::SUCCESS ? "success" : "failed") << '\n';
    return result;
  }
  return ge::SUCCESS;  // Other graphs from the same GE Session are untouched.
}

ge::Status CheckPostOptimize(ge::GraphPtr &graph, ge::CustomPassContext &) {
  if (graph == nullptr) return ge::FAILED;
  for (const auto &node : graph->GetAllNodes()) {
    if (Name(node) != "BgeFusedInferAttentionLayer0") continue;
    std::cerr << "[BGE_FIAS] post_opt type=FusedInferAttentionScore";
    for (int port : {0, 1, 2, 4}) {
      ge::TensorDesc desc;
      if (node.GetInputDesc(port, desc) == ge::GRAPH_SUCCESS)
        std::cerr << " input" << port << "_dtype="
                  << static_cast<int>(desc.GetDataType());
    }
    ge::TensorDesc desc;
    if (node.GetOutputDesc(0, desc) == ge::GRAPH_SUCCESS)
      std::cerr << " output_dtype=" << static_cast<int>(desc.GetDataType());
    std::cerr << '\n';
  }
  return ge::SUCCESS;
}
}  // namespace

REGISTER_CUSTOM_PASS("BgeFusedInferAttentionLayer0")
    .CustomPassFn(Run)
    .Stage(ge::CustomPassStage::kAfterBuiltinFusionPass);

REGISTER_CUSTOM_PASS("BgeFiasPostOptimizeCheck")
    .CustomPassFn(CheckPostOptimize)
    .Stage(ge::CustomPassStage::kAfterOriginGraphOptimize);
