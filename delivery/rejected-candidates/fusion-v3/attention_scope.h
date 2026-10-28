#pragma once
#include <string>

namespace bge_fusion {
// This candidate is deliberately limited to the 24 attention/output residuals
// measured in the unchanged BGE encoder. FFN/output and embeddings stay separate.
inline bool IsAttentionOutputLayerNorm(const std::string &name) {
  const std::string prefix = "/encoder/backbone/encoder/layer.";
  const std::string tail = "/attention/output/LayerNorm/";
  if (name.compare(0, prefix.size(), prefix) != 0) return false;
  const auto slash = name.find('/', prefix.size());
  if (slash == std::string::npos) return false;
  const auto digits = name.substr(prefix.size(), slash - prefix.size());
  if (digits.empty() || digits.size() > 2 || (digits.size() > 1 && digits[0] == '0')) return false;
  unsigned layer = 0;
  for (const char ch : digits) {
    if (ch < '0' || ch > '9') return false;
    layer = layer * 10 + static_cast<unsigned>(ch - '0');
  }
  return layer < 24 && name.compare(slash, tail.size(), tail) == 0 && name.size() > slash + tail.size();
}
}
