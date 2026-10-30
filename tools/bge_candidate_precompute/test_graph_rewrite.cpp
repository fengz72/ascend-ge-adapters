// A CPU-only GE Graph API test for the structural match and the three
// consumer rewires. No GE Session, CompileGraph, or NPU is used.
#include <cstdlib>
#include "bge_candidate_precompute_pass.cpp"

namespace {

ge::GNode FloatConst(ge::Graph &graph, const char *name,
                     const std::vector<int64_t> &dims,
                     const std::vector<float> &values) {
  ge::TensorDesc desc(ge::Shape(dims), ge::FORMAT_ND, ge::DT_FLOAT);
  desc.SetOriginShape(ge::Shape(dims));
  ge::Tensor value(desc);
  if (value.SetData(reinterpret_cast<const uint8_t *>(values.data()),
                    values.size() * sizeof(float)) != ge::GRAPH_SUCCESS)
    std::abort();
  ge::op::Const op(name);
  op.set_attr_value(value);
  op.update_output_desc_y(desc);
  return graph.AddNodeByOp(op);
}

ge::GNode ShapeConst(ge::Graph &graph, const char *name,
                     const std::vector<int64_t> &values) {
  ge::TensorDesc desc(ge::Shape({static_cast<int64_t>(values.size())}),
                      ge::FORMAT_ND, ge::DT_INT64);
  desc.SetOriginShape(desc.GetShape());
  ge::Tensor value(desc);
  if (value.SetData(reinterpret_cast<const uint8_t *>(values.data()),
                    values.size() * sizeof(int64_t)) != ge::GRAPH_SUCCESS)
    std::abort();
  ge::op::Const op(name);
  op.set_attr_value(value);
  op.update_output_desc_y(desc);
  return graph.AddNodeByOp(op);
}

ge::GNode Bmm(ge::Graph &graph, const char *name, int64_t rows,
              int64_t width, bool adj_x2 = false) {
  ge::op::BatchMatMulV2 op(name);
  op.set_attr_adj_x1(false);
  op.set_attr_adj_x2(adj_x2);
  ge::TensorDesc desc(ge::Shape({rows, width}), ge::FORMAT_ND, ge::DT_FLOAT);
  desc.SetOriginShape(desc.GetShape());
  op.update_output_desc_y(desc);
  return graph.AddNodeByOp(op);
}

void Edge(ge::Graph &graph, ge::GNode &from, ge::GNode &to, int port) {
  if (graph.AddDataEdge(from, 0, to, port) != ge::GRAPH_SUCCESS) std::abort();
}
}  // namespace

int main() {
  ge::Graph graph("candidate-test");
  auto e = FloatConst(graph, "embedding", {3, 3},
                      {1, 0, 0, 0, 3, 4, 0, 0, 0});
  auto w = FloatConst(graph, "projection", {3, 2},
                      {1, 0, 0, 1, 0, 0});
  auto v = FloatConst(graph, "post-projection", {2, 2}, {2, 0, 0, 3});
  auto eps = FloatConst(graph, "epsilon", {}, {1.0e-12f});
  auto shape = ShapeConst(graph, "shape", {3, 2});
  auto first = Bmm(graph, "first", 3, 2);
  ge::op::LpNormV2 norm_op("norm");
  norm_op.set_attr_p(2.0f);
  norm_op.set_attr_axes({-1});
  norm_op.set_attr_keepdim(true);
  norm_op.set_attr_epsilon(0.0f);
  auto norm = graph.AddNodeByOp(norm_op);
  auto clip = graph.AddNodeByOp(ge::op::Maximum("clip"));
  auto expand = graph.AddNodeByOp(ge::op::Expand("expand"));
  auto div = graph.AddNodeByOp(ge::op::RealDiv("div"));
  ge::TensorDesc norm_table(ge::Shape({3, 1}), ge::FORMAT_ND, ge::DT_FLOAT);
  norm_table.SetOriginShape(norm_table.GetShape());
  ge::TensorDesc table(ge::Shape({3, 2}), ge::FORMAT_ND, ge::DT_FLOAT);
  table.SetOriginShape(table.GetShape());
  if (norm.UpdateOutputDesc(0, norm_table) != ge::GRAPH_SUCCESS ||
      clip.UpdateOutputDesc(0, norm_table) != ge::GRAPH_SUCCESS ||
      expand.UpdateOutputDesc(0, table) != ge::GRAPH_SUCCESS ||
      div.UpdateOutputDesc(0, table) != ge::GRAPH_SUCCESS) return 1;
  auto post = Bmm(graph, "post", 3, 2);
  auto gather_p = graph.AddNodeByOp(ge::op::GatherV2("gather-p"));
  auto gather_c = graph.AddNodeByOp(ge::op::GatherV2("gather-c"));
  auto score = Bmm(graph, "score", 1, 3, true);
  Edge(graph, e, first, 0);
  Edge(graph, w, first, 1);
  Edge(graph, first, norm, 0);
  Edge(graph, norm, clip, 0);
  Edge(graph, eps, clip, 1);
  Edge(graph, clip, expand, 0);
  Edge(graph, shape, expand, 1);
  Edge(graph, first, div, 0);
  Edge(graph, expand, div, 1);
  Edge(graph, div, post, 0);
  Edge(graph, v, post, 1);
  Edge(graph, post, gather_p, 0);
  Edge(graph, div, gather_c, 0);
  Edge(graph, div, score, 1);

  Match match;
  if (!MatchCandidate(post, match) || Name(match.post) != "post") return 2;
  std::vector<float> c_values, p_values;
  if (!Precompute(match, c_values, p_values)) return 3;
  ge::TensorDesc c_desc, p_desc;
  if (div.GetOutputDesc(0, c_desc) != ge::GRAPH_SUCCESS ||
      post.GetOutputDesc(0, p_desc) != ge::GRAPH_SUCCESS) return 4;
  ge::GNode c, p;
  if (!AddConst(graph, "C", c_values, c_desc, c) ||
      !AddConst(graph, "P", p_values, p_desc, p) ||
      !Find(graph, "C", c) || !Find(graph, "P", p) ||
      !Find(graph, "post", post) || !Find(graph, "div", div) ||
      !ReplaceEdges(graph, post, p) || !Find(graph, "C", c) ||
      !Find(graph, "div", div) || !ReplaceEdges(graph, div, c)) return 4;
  if (!HasEdge(p, gather_p, 0) || !HasEdge(c, gather_c, 0) ||
      !HasEdge(c, score, 1) ||
      !post.GetOutDataNodesAndPortIndexs(0).empty() ||
      !div.GetOutDataNodesAndPortIndexs(0).empty()) return 5;
  for (const char *name : {"post", "div", "expand", "clip", "norm", "first",
                           "embedding", "projection", "post-projection"}) {
    ge::GNode node;
    if (!Find(graph, name, node) || !Remove(graph, node)) return 6;
  }
  return 0;
}
