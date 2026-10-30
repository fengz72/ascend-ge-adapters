#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "all_ops.h"
#include "register/register_custom_pass.h"

// Match the validated bidirectional attention structure, regardless of layer
// names or count. Unsupported attention structures remain untouched.
namespace {

constexpr char kFusedPrefix[] = "BgeFusedInferAttentionCandidate";

std::string Name(const ge::GNode &n) {
  ge::AscendString s;
  if (n.GetName(s) != ge::GRAPH_SUCCESS || s.GetString() == nullptr) return {};
  return s.GetString();
}

bool Type(const ge::GNode &n, const char *expected) {
  ge::AscendString s;
  return n.GetType(s) == ge::GRAPH_SUCCESS && s == expected;
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

bool ConstInts(const ge::GNode &node, std::vector<int64_t> &values) {
  if (!Type(node, "Const")) return false;
  ge::Tensor tensor;
  if (node.GetAttr("value", tensor) != ge::GRAPH_SUCCESS ||
      tensor.GetData() == nullptr) return false;
  const auto dtype = tensor.GetDataType();
  const size_t width = dtype == ge::DT_INT32 ? sizeof(int32_t)
                                            : dtype == ge::DT_INT64 ? sizeof(int64_t) : 0;
  if (width == 0 || tensor.GetSize() == 0 || tensor.GetSize() % width != 0)
    return false;
  const auto *bytes = static_cast<const uint8_t *>(tensor.GetData());
  for (size_t offset = 0; offset < tensor.GetSize(); offset += width) {
    int64_t value = 0;
    if (dtype == ge::DT_INT32) {
      int32_t narrow = 0;
      std::memcpy(&narrow, bytes + offset, width);
      value = narrow;
    } else {
      std::memcpy(&value, bytes + offset, width);
    }
    values.push_back(value);
  }
  return true;
}

bool ConstFloat(const ge::GNode &node, float &value) {
  if (!Type(node, "Const")) return false;
  ge::Tensor tensor;
  if (node.GetAttr("value", tensor) != ge::GRAPH_SUCCESS ||
      tensor.GetDataType() != ge::DT_FLOAT ||
      tensor.GetSize() != sizeof(float) || tensor.GetData() == nullptr)
    return false;
  std::memcpy(&value, tensor.GetData(), sizeof(float));
  return std::isfinite(value);
}

bool Permutation(const ge::GNode &node, const std::vector<int64_t> &expected) {
  ge::GNode constant;
  std::vector<int64_t> actual;
  return Type(node, "Transpose") && Input(node, 1, constant) &&
         ConstInts(constant, actual) && actual == expected;
}

bool HeadsAndWidth(const ge::GNode &reshape, int64_t &heads, int64_t &width) {
  ge::GNode concat, head_const, width_const;
  std::vector<int64_t> h, d;
  if (!TypedInput(reshape, 1, "ConcatD", concat) ||
      !Input(concat, 2, head_const) || !Input(concat, 3, width_const) ||
      !ConstInts(head_const, h) || !ConstInts(width_const, d) ||
      h.size() != 1 || d.size() != 1 || h[0] <= 0 || d[0] <= 0)
    return false;
  heads = h[0];
  width = d[0];
  return true;
}

bool SameNode(const ge::GNode &a, const ge::GNode &b) {
  return !Name(a).empty() && Name(a) == Name(b);
}

bool PlainMatMul(const ge::GNode &node) {
  if (node.GetInDataNodesAndPortIndexs(2).first != nullptr ||
      node.GetInDataNodesAndPortIndexs(3).first != nullptr)
    return false;
  bool adj = false;
  if (node.GetAttr("adj_x1", adj) == ge::GRAPH_SUCCESS && adj) return false;
  if (node.GetAttr("adj_x2", adj) == ge::GRAPH_SUCCESS && adj) return false;
  int64_t offset = 0;
  return node.GetAttr("offset_x", offset) != ge::GRAPH_SUCCESS || offset == 0;
}

bool LastAxisSoftmax(const ge::GNode &node) {
  std::vector<int64_t> axes;
  if (node.GetAttr("axes", axes) == ge::GRAPH_SUCCESS &&
      axes != std::vector<int64_t>{-1} &&
      axes != std::vector<int64_t>{3}) return false;
  bool half_to_float = false;
  return node.GetAttr("half_to_float", half_to_float) != ge::GRAPH_SUCCESS ||
         !half_to_float;
}

struct Match {
  ge::GNode tail, bmm2, softmax, score_add, bmm1;
  ge::GNode q_mul, k_mul, q_transpose, k_transpose, v_transpose;
  ge::GNode query, key, value, mask_select, mask;
  ge::TensorDesc q_desc, k_desc, v_desc, mask_desc, out_desc;
  int64_t heads = 0, width = 0;
};

bool MatchAttention(const ge::GNode &tail, Match &m) {
  if (!Type(tail, "Transpose")) return false;
  m.tail = tail;
  if (!TypedInput(tail, 0, "BatchMatMulV2", m.bmm2) ||
      !TypedInput(m.bmm2, 0, "SoftmaxV2", m.softmax) ||
      !TypedInput(m.bmm2, 1, "Transpose", m.v_transpose) ||
      !TypedInput(m.softmax, 0, "Add", m.score_add) ||
      !TypedInput(m.score_add, 0, "BatchMatMulV2", m.bmm1) ||
      !TypedInput(m.score_add, 1, "SelectV2", m.mask_select) ||
      !Input(m.mask_select, 0, m.mask) ||
      !TypedInput(m.bmm1, 0, "Mul", m.q_mul) ||
      !TypedInput(m.bmm1, 1, "Mul", m.k_mul) ||
      !TypedInput(m.q_mul, 0, "Transpose", m.q_transpose) ||
      !TypedInput(m.k_mul, 0, "Transpose", m.k_transpose) ||
      !TypedInput(m.q_transpose, 0, "Reshape", m.query) ||
      !TypedInput(m.k_transpose, 0, "Reshape", m.key) ||
      !TypedInput(m.v_transpose, 0, "Reshape", m.value)) return false;
  if (!PlainMatMul(m.bmm1) || !PlainMatMul(m.bmm2) ||
      !LastAxisSoftmax(m.softmax)) return false;

  // Do not infer BSND from four-dimensional descriptors alone: transposes
  // with different permutations can otherwise match the same operator chain.
  if (!Permutation(m.q_transpose, {0, 2, 1, 3}) ||
      !Permutation(m.k_transpose, {0, 2, 3, 1}) ||
      !Permutation(m.v_transpose, {0, 2, 1, 3}) ||
      !Permutation(m.tail, {0, 2, 1, 3})) return false;

  int64_t kh = 0, kd = 0, vh = 0, vd = 0;
  if (!HeadsAndWidth(m.query, m.heads, m.width) ||
      !HeadsAndWidth(m.key, kh, kd) ||
      !HeadsAndWidth(m.value, vh, vd) ||
      kh != m.heads || vh != m.heads ||
      kd != m.width || vd != m.width) return false;

  // Original score scale is split across Q and K:
  //   Q * sqrt(1/sqrt(D)) @ K * sqrt(1/sqrt(D)).
  ge::GNode q_sqrt, k_sqrt, q_div, k_div, numerator, denom_sqrt;
  ge::GNode denom_cast, denom_slice, denom_shape, shape_source;
  float one = 0.0f;
  if (!TypedInput(m.q_mul, 1, "Sqrt", q_sqrt) ||
      !TypedInput(m.k_mul, 1, "Sqrt", k_sqrt) ||
      !TypedInput(q_sqrt, 0, "RealDiv", q_div) ||
      !TypedInput(k_sqrt, 0, "RealDiv", k_div) ||
      !SameNode(q_div, k_div) ||
      !Input(q_div, 0, numerator) || !ConstFloat(numerator, one) ||
      one != 1.0f ||
      !TypedInput(q_div, 1, "Sqrt", denom_sqrt) ||
      !TypedInput(denom_sqrt, 0, "Cast", denom_cast) ||
      !Input(denom_cast, 0, denom_slice) ||
      !TypedInput(denom_slice, 0, "Shape", denom_shape) ||
      !Input(denom_shape, 0, shape_source) ||
      !SameNode(shape_source, m.q_transpose)) return false;
  std::vector<int64_t> begin, end, strides;
  if (!Type(denom_slice, "StridedSliceD") ||
      denom_slice.GetAttr("begin", begin) != ge::GRAPH_SUCCESS ||
      denom_slice.GetAttr("end", end) != ge::GRAPH_SUCCESS ||
      denom_slice.GetAttr("strides", strides) != ge::GRAPH_SUCCESS ||
      begin != std::vector<int64_t>{-1} ||
      end != std::vector<int64_t>{std::numeric_limits<int64_t>::max()} ||
      strides != std::vector<int64_t>{1}) return false;

  // SelectV2(true, -FLT_MAX, 1 - binary_keep_mask): the same Sub feeds its
  // BOOL condition. FIAS expects true for blocked positions.
  ge::GNode blocked_cast, true_value, unblocked_bias, bias_one, raw_mask_cast;
  float minus_inf = 0.0f, bias_base = 0.0f;
  if (!TypedInput(m.mask_select, 0, "Cast", blocked_cast) ||
      !Input(m.mask_select, 1, true_value) ||
      !TypedInput(m.mask_select, 2, "Sub", unblocked_bias) ||
      !Input(blocked_cast, 0, m.mask) ||
      !SameNode(m.mask, unblocked_bias) ||
      !ConstFloat(true_value, minus_inf) || minus_inf > -1.0e30f ||
      !Input(unblocked_bias, 0, bias_one) ||
      !ConstFloat(bias_one, bias_base) || bias_base != 1.0f ||
      !Input(unblocked_bias, 1, raw_mask_cast)) return false;
  m.mask = blocked_cast;

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
      !KnownOrDynamic(q.GetDim(2), m.heads) || !KnownOrDynamic(q.GetDim(3), m.width) ||
      !KnownOrDynamic(k.GetDim(2), m.heads) || !KnownOrDynamic(k.GetDim(3), m.width) ||
      !KnownOrDynamic(v.GetDim(2), m.heads) || !KnownOrDynamic(v.GetDim(3), m.width) ||
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

ge::Status Fuse(ge::Graph &graph, size_t index, Match &m) {
  ge::op::FusedInferAttentionScore op(std::string(kFusedPrefix) +
                                     std::to_string(index));
  // The default create_dynamic_input_* appends after all optional inputs.
  // FIAS lowering expects key/value at their IR positions immediately after Q.
  op.create_dynamic_input_byindex_key(1, 1);
  op.create_dynamic_input_byindex_value(1, 2);
  op.set_attr_num_heads(m.heads);
  op.set_attr_num_key_value_heads(m.heads);
  op.set_attr_input_layout("BSND");
  op.set_attr_scale(1.0f / std::sqrt(static_cast<float>(m.width)));
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

ge::GraphPtr FindOwner(const ge::GraphPtr &graph, const std::string &name) {
  for (const auto &node : graph->GetDirectNode()) {
    if (Name(node) == name) return graph;
  }
  for (const auto &child : graph->GetAllSubgraphs()) {
    if (child == nullptr) continue;
    auto owner = FindOwner(child, name);
    if (owner != nullptr) return owner;
  }
  return nullptr;
}

ge::Status Run(ge::GraphPtr &graph, ge::CustomPassContext &context) {
  if (graph == nullptr) return ge::FAILED;
  struct Planned { ge::GraphPtr owner; Match match; };
  std::vector<Planned> planned;
  std::set<std::string> seen;
  // Snapshot candidates before editing the graph, as in AttentionFusionPass.
  for (const auto &node : graph->GetAllNodes()) {
    if (!Type(node, "Transpose")) continue;
    Match match;
    if (!MatchAttention(node, match)) continue;
    const std::string name = Name(node);
    if (!seen.insert(name).second) continue;
    auto owner = FindOwner(graph, name);
    if (owner == nullptr) {
      const std::string error = "BGE FIAS cannot find owner for " + name;
      context.SetErrorMessage(ge::AscendString(error.c_str()));
      return ge::FAILED;
    }
    planned.push_back({owner, match});
  }
  std::cerr << "[BGE_FIAS] matched_subgraphs=" << planned.size() << '\n';
  for (size_t index = 0; index < planned.size(); ++index) {
    auto result = Fuse(*planned[index].owner, index, planned[index].match);
    if (result != ge::SUCCESS) {
      std::cerr << "[BGE_FIAS] replacement_failed_index=" << index << '\n';
      return result;
    }
  }
  if (!planned.empty() && std::getenv("BGE_FIAS_DUMP_STRUCTURE") != nullptr) {
    std::ofstream out("/tmp/bge-fias-graph.txt");
    if (out) planned.front().owner->Dump(ge::Graph::DumpFormat::kReadable, out);
  }
  std::cerr << "[BGE_FIAS] replaced_subgraphs=" << planned.size() << '\n';
  return ge::SUCCESS;
}

ge::Status CheckPostOptimize(ge::GraphPtr &graph, ge::CustomPassContext &) {
  if (graph == nullptr) return ge::FAILED;
  std::set<std::string> seen;
  bool descriptors_ok = true;
  for (const auto &node : graph->GetAllNodes()) {
    const std::string name = Name(node);
    if (name.rfind(kFusedPrefix, 0) != 0) continue;
    if (!seen.insert(name).second ||
        !Type(node, "FusedInferAttentionScore")) return ge::FAILED;
    for (int port : {0, 1, 2, 4}) {
      ge::TensorDesc desc;
      const auto expected = port == 4 ? ge::DT_BOOL : ge::DT_FLOAT16;
      if (node.GetInputDesc(port, desc) != ge::GRAPH_SUCCESS ||
          desc.GetDataType() != expected) descriptors_ok = false;
    }
    ge::TensorDesc desc;
    if (node.GetOutputDesc(0, desc) != ge::GRAPH_SUCCESS ||
        desc.GetDataType() != ge::DT_FLOAT16) descriptors_ok = false;
  }
  std::cerr << "[BGE_FIAS] post_opt_subgraphs=" << seen.size()
            << " descriptors_ok=" << descriptors_ok << '\n';
  return descriptors_ok ? ge::SUCCESS : ge::FAILED;
}
}  // namespace

REGISTER_CUSTOM_PASS("BgeFusedInferAttentionStructure")
    .CustomPassFn(Run)
    .Stage(ge::CustomPassStage::kAfterBuiltinFusionPass);

REGISTER_CUSTOM_PASS("BgeFiasPostOptimizeCheck")
    .CustomPassFn(CheckPostOptimize)
    .Stage(ge::CustomPassStage::kAfterOriginGraphOptimize);
