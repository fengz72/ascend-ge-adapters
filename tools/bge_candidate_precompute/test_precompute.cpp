// CPU-only numerical smoke test; includes the implementation to exercise its
// actual host computation without starting a GE Session or using an NPU.
#include "bge_candidate_precompute_pass.cpp"

int main() {
  Match match;
  match.rows = 3;
  match.inner = 3;
  match.width = 2;
  match.floor = 1.0e-12f;
  match.embedding_data = {1, 0, 0, 0, 3, 4, 0, 0, 0};
  match.projection_data = {1, 0, 0, 1, 0, 0};
  match.post_projection_data = {2, 0, 0, 3};
  std::vector<float> normalized, projected;
  if (!Precompute(match, normalized, projected)) return 1;
  const std::vector<float> expected_normalized = {1, 0, 0, 1, 0, 0};
  const std::vector<float> expected_projected = {2, 0, 0, 3, 0, 0};
  if (normalized != expected_normalized || projected != expected_projected)
    return 2;
  match.embedding_data[0] = std::numeric_limits<float>::infinity();
  if (Precompute(match, normalized, projected)) return 3;
  return 0;
}
