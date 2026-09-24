#pragma once

#include <string>

namespace ge_runtime {

// 跑一份 bench plan (由 Python 侧 core/bench.py 从 model.yaml 的 bench 段生成):
// 多实例 + 共享请求池随机抽样 + 分阶段计时 → 写 perf.json 与 perf_requests.csv。
// 返回进程退出码 (0 成功)。
int RunBenchPlan(const std::string &planPath);

}  // namespace ge_runtime
