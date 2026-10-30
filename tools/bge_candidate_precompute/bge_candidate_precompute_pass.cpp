#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "all_ops.h"
#include "register/register_custom_pass.h"

// This pass runs once for every GE graph build. Deliberately no cross-graph or
// persistent cache: a cache would obscure whether the graph replacement works.
namespace {

std::string Name(const ge::GNode &node) {
  ge::AscendString name;
  if (node.GetName(name) != ge::GRAPH_SUCCESS || name.GetString() == nullptr) return {};
  return name.GetString();
}

bool Type(const ge::GNode &node, const char *type) {
  ge::AscendString actual;
  return node.GetType(actual) == ge::GRAPH_SUCCESS && actual == type;
}

bool Input(const ge::GNode &node, int index, const char *type, ge::GNode &source) {
  const auto edge = node.GetInDataNodesAndPortIndexs(index);
  if (edge.first == nullptr || edge.second != 0) return false;
  source = *edge.first;
  return Type(source, type);
}

bool Dims(const ge::GNode &node, std::vector<int64_t> &dims) {
  ge::TensorDesc desc;
  if (node.GetOutputDesc(0, desc) != ge::GRAPH_SUCCESS ||
      desc.GetDataType() != ge::DT_FLOAT) return false;
  dims = desc.GetShape().GetDims();
  return dims.size() == 2 && dims[0] > 0 && dims[1] > 0;
}

void DebugDesc(const ge::GNode &node) {
  ge::TensorDesc desc;
  const auto status = node.GetOutputDesc(0, desc);
  std::cerr << "[BGE_PRECOMPUTE] desc " << Name(node) << " status=" << status;
  if (status == ge::GRAPH_SUCCESS) {
    std::cerr << " dtype=" << static_cast<int>(desc.GetDataType())
              << " format=" << static_cast<int>(desc.GetFormat()) << " dims=";
    for (const auto n : desc.GetShape().GetDims()) std::cerr << n << ',';
  }
  std::cerr << '\n';
}

bool FloatData(const ge::GNode &node, std::vector<float> &data,
               std::vector<int64_t> &dims) {
  if (!Type(node, "Const") || !Dims(node, dims)) return false;
  ge::TensorDesc desc;
  if (node.GetOutputDesc(0, desc) != ge::GRAPH_SUCCESS ||
      desc.GetFormat() != ge::FORMAT_ND) return false;
  ge::Tensor value;
  if (node.GetAttr("value", value) != ge::GRAPH_SUCCESS ||
      value.GetDataType() != ge::DT_FLOAT || value.GetData() == nullptr) return false;
  const uint64_t count = static_cast<uint64_t>(dims[0]) * dims[1];
  if (count > std::numeric_limits<size_t>::max() / sizeof(float) ||
      value.GetSize() != count * sizeof(float)) return false;
  data.resize(static_cast<size_t>(count));
  std::memcpy(data.data(), value.GetData(), value.GetSize());
  return true;
}

bool ScalarFloat(const ge::GNode &node, float &result) {
  if (!Type(node, "Const")) return false;
  ge::Tensor value;
  if (node.GetAttr("value", value) != ge::GRAPH_SUCCESS ||
      value.GetDataType() != ge::DT_FLOAT || value.GetSize() != sizeof(float) ||
      value.GetData() == nullptr) return false;
  std::memcpy(&result, value.GetData(), sizeof(float));
  return std::isfinite(result) && result > 0.0f;
}

bool NoTranspose(const ge::GNode &node) {
  bool left = true, right = true;
  return node.GetAttr("adj_x1", left) == ge::GRAPH_SUCCESS &&
         node.GetAttr("adj_x2", right) == ge::GRAPH_SUCCESS &&
         !left && !right;
}

bool SingleUse(const ge::GNode &node) {
  return node.GetOutDataNodesAndPortIndexs(0).size() == 1;
}

bool HasEdge(const ge::GNode &source, const ge::GNode &target, int input) {
  for (const auto &edge : source.GetOutDataNodesAndPortIndexs(0)) {
    if (edge.first != nullptr && Name(*edge.first) == Name(target) &&
        edge.second == input) return true;
  }
  return false;
}

struct Match {
  ge::GNode first, norm, epsilon, clip, shape, expand, div, post;
  ge::GNode embedding, projection, post_projection;
  std::vector<float> embedding_data, projection_data, post_projection_data;
  int64_t rows = 0, inner = 0, width = 0;
  float floor = 0.0f;
};

bool MatchCandidate(const ge::GNode &post, Match &m) {
  const bool debug = std::getenv("BGE_PRECOMPUTE_DEBUG") != nullptr &&
                     Name(post) == "/MatMul";
  const auto fail = [&](const char *stage) {
    if (debug) std::cerr << "[BGE_PRECOMPUTE] candidate_rejected=" << stage << '\n';
    return false;
  };
  if (!Type(post, "BatchMatMulV2") ||
      !Input(post, 0, "RealDiv", m.div) ||
      !Input(post, 1, "Const", m.post_projection) ||
      !Input(m.div, 0, "BatchMatMulV2", m.first) ||
      !Input(m.div, 1, "Expand", m.expand) ||
      !Input(m.first, 0, "Const", m.embedding) ||
      !Input(m.first, 1, "Const", m.projection) ||
      !Input(m.expand, 0, "Maximum", m.clip) ||
      !Input(m.expand, 1, "Const", m.shape) ||
      !Input(m.clip, 0, "LpNormV2", m.norm) ||
      !Input(m.clip, 1, "Const", m.epsilon) ||
      !Input(m.norm, 0, "BatchMatMulV2", m.first) ||
      !ScalarFloat(m.epsilon, m.floor)) return fail("structure");

  // The match is deliberately strict: only the fixed-weight two-table branch
  // with the two dynamic lookups and final score consumer is rewritten.
  std::vector<int64_t> a, b, c, first_dims, div_dims, post_dims;
  std::vector<int64_t> norm_dims, clip_dims, expand_dims;
  if (!FloatData(m.embedding, m.embedding_data, a) ||
      !FloatData(m.projection, m.projection_data, b) ||
      !FloatData(m.post_projection, m.post_projection_data, c) ||
      !Dims(m.first, first_dims) || !Dims(m.norm, norm_dims) ||
      !Dims(m.clip, clip_dims) || !Dims(m.expand, expand_dims) ||
      !Dims(m.div, div_dims) || !Dims(post, post_dims) ||
      !NoTranspose(m.first) || !NoTranspose(post) ||
      a[1] != b[0] || b[1] != c[0] ||
      c[0] != c[1] || first_dims != std::vector<int64_t>({a[0], b[1]}) ||
      norm_dims != std::vector<int64_t>({a[0], 1}) ||
      clip_dims != norm_dims || expand_dims != first_dims ||
      div_dims != first_dims || post_dims != first_dims ||
      !SingleUse(m.embedding) || !SingleUse(m.projection) ||
      !SingleUse(m.post_projection) || !SingleUse(m.norm) ||
      !SingleUse(m.clip) || !SingleUse(m.expand) || !SingleUse(post) ||
      m.first.GetOutDataNodesAndPortIndexs(0).size() != 2 ||
      m.div.GetOutDataNodesAndPortIndexs(0).size() != 3 ||
      !HasEdge(m.first, m.norm, 0) || !HasEdge(m.first, m.div, 0) ||
      !HasEdge(m.div, post, 0)) {
    if (debug) {
      for (const auto &node : {m.first, m.norm, m.clip, m.expand, m.div, post})
        DebugDesc(node);
      std::cerr << "[BGE_PRECOMPUTE] metadata a=";
      for (auto n : a) std::cerr << n << ',';
      std::cerr << " b=";
      for (auto n : b) std::cerr << n << ',';
      std::cerr << " c=";
      for (auto n : c) std::cerr << n << ',';
      std::cerr << " norm=";
      for (auto n : norm_dims) std::cerr << n << ',';
      std::cerr << " clip=";
      for (auto n : clip_dims) std::cerr << n << ',';
      std::cerr << " expand=";
      for (auto n : expand_dims) std::cerr << n << ',';
      std::cerr << " first_users=" << m.first.GetOutDataNodesAndPortIndexs(0).size()
                << " div_users=" << m.div.GetOutDataNodesAndPortIndexs(0).size()
                << " post_users=" << post.GetOutDataNodesAndPortIndexs(0).size()
                << " no_transpose=" << NoTranspose(m.first) << ','
                << NoTranspose(post) << '\n';
    }
    return fail("metadata");
  }

  const auto post_users = post.GetOutDataNodesAndPortIndexs(0);
  if (post_users[0].first == nullptr || post_users[0].second != 0 ||
      !Type(*post_users[0].first, "GatherV2")) return fail("post_user");
  bool gather = false, score = false;
  for (const auto &edge : m.div.GetOutDataNodesAndPortIndexs(0)) {
    if (edge.first == nullptr) return fail("div_null_user");
    if (Name(*edge.first) == Name(post) && edge.second == 0) continue;
    if (Type(*edge.first, "GatherV2") && edge.second == 0 && !gather) {
      gather = true;
    } else if (Type(*edge.first, "BatchMatMulV2") && edge.second == 1 && !score) {
      bool adj_x2 = false;
      score = edge.first->GetAttr("adj_x2", adj_x2) == ge::GRAPH_SUCCESS &&
              adj_x2;
    } else return fail("div_user_type");
  }
  if (!gather || !score) return fail("div_users");

  std::vector<int64_t> axes;
  bool keepdim = false;
  float p = 0.0f, norm_epsilon = -1.0f;
  if (m.norm.GetAttr("axes", axes) != ge::GRAPH_SUCCESS ||
      m.norm.GetAttr("keepdim", keepdim) != ge::GRAPH_SUCCESS ||
      m.norm.GetAttr("p", p) != ge::GRAPH_SUCCESS ||
      m.norm.GetAttr("epsilon", norm_epsilon) != ge::GRAPH_SUCCESS ||
      axes != std::vector<int64_t>({-1}) || !keepdim || p != 2.0f ||
      !std::isfinite(norm_epsilon) || norm_epsilon < 0.0f ||
      norm_epsilon > 1.0e-12f) {
    if (debug) std::cerr << "[BGE_PRECOMPUTE] norm p=" << p
                         << " keepdim=" << keepdim << " epsilon="
                         << norm_epsilon << " axes_size=" << axes.size() << '\n';
    return fail("norm_attributes");
  }
  m.rows = a[0];
  m.inner = a[1];
  m.width = b[1];
  m.post = post;
  return true;
}

void MatMul(const std::vector<float> &a, const std::vector<float> &b,
            int64_t rows, int64_t inner, int64_t width,
            std::vector<float> &out) {
  out.assign(static_cast<size_t>(rows * width), 0.0f);
#pragma omp parallel for schedule(static) num_threads(8)
  for (int64_t row = 0; row < rows; ++row) {
    float *dst = out.data() + row * width;
    const float *left = a.data() + row * inner;
    for (int64_t k = 0; k < inner; ++k) {
      const float x = left[k];
      const float *right = b.data() + k * width;
      for (int64_t col = 0; col < width; ++col) dst[col] += x * right[col];
    }
  }
}

bool Precompute(const Match &m, std::vector<float> &normalized,
                std::vector<float> &projected) {
  MatMul(m.embedding_data, m.projection_data, m.rows, m.inner, m.width,
         normalized);
  for (int64_t row = 0; row < m.rows; ++row) {
    float *values = normalized.data() + row * m.width;
    double squared = 0.0;
    for (int64_t col = 0; col < m.width; ++col)
      squared += static_cast<double>(values[col]) * values[col];
    const double divisor = std::max(std::sqrt(squared), static_cast<double>(m.floor));
    if (!std::isfinite(divisor) || divisor <= 0.0) return false;
    for (int64_t col = 0; col < m.width; ++col) {
      values[col] = static_cast<float>(values[col] / divisor);
      if (!std::isfinite(values[col])) return false;
    }
  }
  MatMul(normalized, m.post_projection_data, m.rows, m.width, m.width,
         projected);
  return std::all_of(projected.begin(), projected.end(), [](float x) {
    return std::isfinite(x);
  });
}

bool AddConst(ge::Graph &graph, const std::string &name,
              const std::vector<float> &values, ge::TensorDesc desc,
              ge::GNode &node) {
  if (desc.GetDataType() != ge::DT_FLOAT ||
      desc.GetShape().GetDims().size() != 2 ||
      static_cast<uint64_t>(desc.GetShape().GetDims()[0]) *
          desc.GetShape().GetDims()[1] != values.size()) {
    std::cerr << "[BGE_PRECOMPUTE] add_const_bad_desc=" << name
              << " dtype=" << static_cast<int>(desc.GetDataType()) << " dims=";
    for (auto dim : desc.GetShape().GetDims()) std::cerr << dim << ',';
    std::cerr << " values=" << values.size() << '\n';
    return false;
  }
  desc.SetOriginShape(desc.GetShape());
  desc.SetOriginFormat(desc.GetFormat());
  ge::Tensor value(desc);
  if (value.SetData(reinterpret_cast<const uint8_t *>(values.data()),
                    values.size() * sizeof(float)) != ge::GRAPH_SUCCESS) {
    std::cerr << "[BGE_PRECOMPUTE] add_const_set_data_failed=" << name << '\n';
    return false;
  }
  ge::op::Const op(name.c_str());
  op.set_attr_value(value);
  if (op.update_output_desc_y(desc) != ge::GRAPH_SUCCESS) {
    std::cerr << "[BGE_PRECOMPUTE] add_const_output_desc_failed=" << name << '\n';
    return false;
  }
  node = graph.AddNodeByOp(op);
  if (!Type(node, "Const")) {
    std::cerr << "[BGE_PRECOMPUTE] add_const_node_failed=" << name << '\n';
    return false;
  }
  return true;
}

bool ReplaceEdges(ge::Graph &graph, const ge::GNode &old_source,
                  ge::GNode &new_source) {
  ge::GNode old = old_source;
  const auto users = old_source.GetOutDataNodesAndPortIndexs(0);
  for (const auto &edge : users) {
    if (edge.first == nullptr ||
        graph.RemoveEdge(old, 0, *edge.first,
                         edge.second) != ge::GRAPH_SUCCESS) {
      std::cerr << "[BGE_PRECOMPUTE] remove_edge_failed source="
                << Name(old_source) << " dest="
                << (edge.first == nullptr ? "null" : Name(*edge.first)) << '\n';
      return false;
    }
    if (graph.AddDataEdge(new_source, 0, *edge.first, edge.second) !=
            ge::GRAPH_SUCCESS) {
      std::cerr << "[BGE_PRECOMPUTE] add_edge_failed source="
                << Name(new_source) << " dest=" << Name(*edge.first)
                << " port=" << edge.second << '\n';
      return false;
    }
  }
  return true;
}

bool Remove(ge::Graph &graph, ge::GNode node) {
  return graph.RemoveNode(node) == ge::GRAPH_SUCCESS;
}

bool Find(ge::Graph &graph, const std::string &name, ge::GNode &found) {
  for (const auto &node : graph.GetDirectNode()) {
    if (Name(node) != name) continue;
    found = node;
    return true;
  }
  return false;
}

ge::Status Run(ge::GraphPtr &graph, ge::CustomPassContext &context) {
  if (graph == nullptr) return ge::FAILED;
  std::vector<Match> matches;
  for (const auto &node : graph->GetDirectNode()) {
    Match match;
    if (MatchCandidate(node, match)) matches.push_back(std::move(match));
  }
  std::cerr << "[BGE_PRECOMPUTE] matched=" << matches.size() << '\n';
  if (matches.size() != 1) {
    context.SetErrorMessage("BGE candidate precompute requires exactly one matching chain");
    return ge::FAILED;
  }
  Match &m = matches.front();
  std::vector<float> normalized, projected;
  if (!Precompute(m, normalized, projected)) {
    context.SetErrorMessage("BGE candidate precompute produced nonfinite values");
    return ge::FAILED;
  }
  ge::TensorDesc c_desc, p_desc;
  if (m.div.GetOutputDesc(0, c_desc) != ge::GRAPH_SUCCESS ||
      m.post.GetOutputDesc(0, p_desc) != ge::GRAPH_SUCCESS) return ge::FAILED;
  const std::string div_name = Name(m.div), post_name = Name(m.post);
  std::vector<std::string> old_names;
  for (const auto &node : {m.post, m.div, m.expand, m.clip, m.norm, m.first,
                           m.embedding, m.projection, m.post_projection})
    old_names.push_back(Name(node));
  ge::GNode c, p;
  if (!AddConst(*graph, "BgeCandidateNormalizedPrecomputed", normalized,
                c_desc, c) ||
      !AddConst(*graph, "BgeCandidateProjectedPrecomputed", projected,
                p_desc, p)) {
    context.SetErrorMessage("BGE candidate precompute could not replace graph edges");
    return ge::FAILED;
  }
  ge::GNode current_div, current_post;
  if (!Find(*graph, "BgeCandidateNormalizedPrecomputed", c) ||
      !Find(*graph, "BgeCandidateProjectedPrecomputed", p) ||
      !Find(*graph, post_name, current_post) ||
      !ReplaceEdges(*graph, current_post, p)) {
    context.SetErrorMessage("BGE candidate precompute could not rewire projected table");
    return ge::FAILED;
  }
  if (!Find(*graph, "BgeCandidateNormalizedPrecomputed", c) ||
      !Find(*graph, div_name, current_div) ||
      !ReplaceEdges(*graph, current_div, c)) {
    context.SetErrorMessage("BGE candidate precompute could not rewire consumers");
    return ge::FAILED;
  }
  for (const auto &name : old_names) {
    ge::GNode node;
    if (!Find(*graph, name, node) || !Remove(*graph, node)) {
      context.SetErrorMessage("BGE candidate precompute could not remove old nodes");
      return ge::FAILED;
    }
  }
  std::cerr << "[BGE_PRECOMPUTE] replaced=1 rows=" << m.rows
            << " width=" << m.width << " tables_bytes="
            << (normalized.size() + projected.size()) * sizeof(float) << '\n';
  return ge::SUCCESS;
}
}  // namespace

REGISTER_CUSTOM_PASS("BgeCandidateBuildPrecompute")
    .CustomPassFn(Run)
    .Stage(ge::CustomPassStage::kAfterBuiltinFusionPass);
