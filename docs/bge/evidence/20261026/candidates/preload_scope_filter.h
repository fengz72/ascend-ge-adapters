#pragma once
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

namespace bge_preload_scope {
inline std::string Canonical(const std::string &path) {
  char *resolved = realpath(path.c_str(), nullptr);
  if (!resolved) return {};
  std::string result(resolved);
  free(resolved);
  return result;
}
inline std::string Basename(const std::string &path) {
  const auto p = path.find_last_of('/');
  return p == std::string::npos ? path : path.substr(p + 1);
}
struct Filtered {
  std::string value;
  bool changed = false;
};
inline Filtered Filter(const std::string &raw, const std::string &self,
                       const std::string &executable) {
  if (Basename(executable) != "ge_runtime") return {raw, false};
  const auto canonical_self = Canonical(self);
  if (canonical_self.empty()) return {raw, false};
  std::vector<std::string> kept;
  bool changed = false;
  size_t i = 0;
  while (i < raw.size()) {
    while (i < raw.size() && (raw[i] == ':' || std::isspace(static_cast<unsigned char>(raw[i])))) ++i;
    const auto start = i;
    while (i < raw.size() && raw[i] != ':' && !std::isspace(static_cast<unsigned char>(raw[i]))) ++i;
    if (start == i) continue;
    const auto token = raw.substr(start, i - start);
    // Never remove a basename merely because it resembles this library. An
    // unresolved token is preserved; task plans use an absolute plugin path.
    const auto candidate = Canonical(token);
    if (!candidate.empty() && candidate == canonical_self) changed = true;
    else kept.push_back(token);
  }
  if (!changed) return {raw, false};
  std::string value;
  for (const auto &token : kept) {
    if (!value.empty()) value += ':';
    value += token;
  }
  return {value, true};
}
}
