#define BGE_PASS_NO_REGISTER
#include "add_layernorm_attention_pass.cpp"
#include <dlfcn.h>

ge::TensorDesc Desc(std::vector<int64_t> shape) {
  return ge::TensorDesc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_FLOAT);
}
ge::GNode Make(ge::Graph &g, const std::string &name, const char *type,
              const std::vector<ge::TensorDesc> &ins, const std::vector<ge::TensorDesc> &outs) {
  auto op = ge::OperatorFactory::CreateOperator(name.c_str(), type);
  if (op.GetOutputsSize() < outs.size()) throw std::runtime_error(std::string("missing schema ") + type);
  for (uint32_t i = 0; i < ins.size(); ++i) bge_fusion::Check(op.UpdateInputDesc(i, ins[i]), "test input desc");
  for (uint32_t i = 0; i < outs.size(); ++i) bge_fusion::Check(op.UpdateOutputDesc(i, outs[i]), "test output desc");
  return g.AddNodeByOp(op);
}
int Test(const std::string &variant) {
  ge::Graph g("synthetic_cpu_only");
  auto x = Desc({1, 6, 1024}), params = Desc({1024}), stats = Desc({1, 6, 1});
  auto a = Make(g, "x1", "Data", {}, {x}), b = Make(g, "x2", "Data", {}, {x});
  auto gamma = Make(g, "gamma", "Data", {}, {params}), beta = Make(g, "beta", "Data", {}, {params});
  auto add = Make(g, "residual", "Add", {x, x}, {x});
  std::string name = "/encoder/backbone/encoder/layer.0/attention/output/LayerNorm/fused";
  if (variant == "ffn-output") name = "/encoder/backbone/encoder/layer.0/output/LayerNorm/fused";
  if (variant == "other-model") name = "/other/encoder/layer.0/attention/output/LayerNorm/fused";
  if (variant == "layer24") name = "/encoder/backbone/encoder/layer.24/attention/output/LayerNorm/fused";
  if (variant == "embeddings") name = "/encoder/backbone/embeddings/LayerNorm/fused";
  if (variant == "leading-zero-layer") name = "/encoder/backbone/encoder/layer.01/attention/output/LayerNorm/fused";
  if (variant == "valid-last-layer") name = "/encoder/backbone/encoder/layer.23/attention/output/LayerNorm/fused";
  auto ln = Make(g, name, "LayerNorm", {x, params, params}, {x, stats, stats});
  auto sink = Make(g, "sink", "Identity", {x}, {x});
  int64_t axis = variant == "nonlast-axis" ? 1 : 2;
  float epsilon = variant == "wrong-epsilon" ? 1e-5f : 1e-12f;
  bge_fusion::Check(ln.SetAttr("begin_norm_axis", axis), "test norm axis");
  bge_fusion::Check(ln.SetAttr("begin_params_axis", axis), "test param axis");
  bge_fusion::Check(ln.SetAttr("epsilon", epsilon), "test epsilon");
  bge_fusion::Check(g.AddDataEdge(a, 0, add, 0), "test add 0");
  bge_fusion::Check(g.AddDataEdge(b, 0, add, 1), "test add 1");
  bge_fusion::Check(g.AddDataEdge(add, 0, ln, 0), "test ln x");
  bge_fusion::Check(g.AddDataEdge(gamma, 0, ln, 1), "test ln gamma");
  bge_fusion::Check(g.AddDataEdge(beta, 0, ln, 2), "test ln beta");
  bge_fusion::Check(g.AddDataEdge(ln, 0, sink, 0), "test sink");
  if (variant == "shared-residual") {
    auto other = Make(g, "other", "Identity", {x}, {x});
    bge_fusion::Check(g.AddDataEdge(add, 0, other, 0), "test external Add consumer");
  }
  if (variant == "statistics-used") {
    auto other = Make(g, "other", "Identity", {stats}, {stats});
    bge_fusion::Check(g.AddDataEdge(ln, 2, other, 0), "test external variance consumer");
  }
  if (variant == "control-edge") bge_fusion::Check(g.AddControlEdge(a, ln), "test control edge");
  if (variant == "mixed-input-dtype") {
    auto half = x; half.SetDataType(ge::DT_FLOAT16);
    bge_fusion::Check(add.UpdateInputDesc(1, half), "test half descriptor");
  }
  const auto count = bge_fusion::Fuse(g);
  const int expected = (variant == "valid" || variant == "valid-last-layer") ? 1 : 0;
  if (count != expected) throw std::runtime_error("unexpected fusion result for " + variant);
  if (count) {
    auto source = sink.GetInDataNodesAndPortIndexs(0);
    if (!source.first || bge_fusion::Type(*source.first) != "AddLayerNorm" || source.second != 0)
      throw std::runtime_error("consumer was not rewired to fused y");
    auto fused = *source.first;
    float actual = 0;
    bge_fusion::Check(fused.GetAttr("epsilon", actual), "read fused epsilon");
    if (actual != epsilon) throw std::runtime_error("epsilon changed");
    const std::vector<std::string> expected_sources = {"x1", "x2", "gamma", "beta"};
    for (int i = 0; i < 4; ++i) {
      auto input = fused.GetInDataNodesAndPortIndexs(i);
      if (!input.first || bge_fusion::Name(*input.first) != expected_sources[i] || input.second != 0)
        throw std::runtime_error("fused input order changed");
    }
  }
  std::cout << "PASS " << variant << std::endl;
  return count;
}
int main() {
  // Loading schemas does not call GEInitialize, aclInit or open an NPU device.
  void *schema = dlopen("/usr/local/Ascend/cann-9.0.0/opp/built-in/op_proto/lib/linux/aarch64/libopsproto.so", RTLD_NOW | RTLD_GLOBAL);
  if (!schema) { std::cerr << dlerror() << std::endl; return 1; }
  try {
    for (const auto *test : {"valid", "wrong-epsilon", "nonlast-axis", "shared-residual", "statistics-used", "control-edge", "mixed-input-dtype", "ffn-output", "other-model", "layer24", "embeddings", "leading-zero-layer", "valid-last-layer"}) Test(test);
    std::cout << "13 CPU graph contract tests passed; no device execution" << std::endl;
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
