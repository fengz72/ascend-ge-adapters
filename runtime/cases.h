#pragma once
// cases.json (schema ge-cases/1) -> 内存 case 池。不依赖 GE / ACL。
//
//   {"cases": [
//     {"id": "case-0000",
//      "samples": 16,
//      "inputs": [{"name": "input_ids",
//                  "path": "case-0000/input_ids.npy",
//                  "dtype": "bfloat16"}]}]}
//
// 约定:
//   * inputs 数组顺序即图输入顺序 (位置绑定)。不做名字匹配, 不校验 dtype/shape
//     是否与图一致 —— 那是 GE 的事。
//   * path 相对 cases.json 所在目录 (绝对路径原样用)。
//   * dtype / shape 一律取 npy 头, JSON 里的 "dtype" 只是 bfloat16 这类
//     numpy 表示不了的类型的标注 (透传给 ReadNpy 的 dtype_override)。
//   * samples 可选, 缺省 1, 只用于 result.json 的 samples/秒。
//   * id 仅用于日志与 infer 输出目录名。
#include <cstdint>
#include <string>
#include <vector>

namespace ge_execute {

struct CaseInput {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<char> data;         // 主机字节, 启动时一次读入
};

struct Case {
    std::string id;
    int64_t samples = 1;
    std::vector<CaseInput> inputs;
    std::string shapeKey;           // 各输入 shape 以 ';' 拼接, 用于 warmup 覆盖 distinct shape
};

struct CasePool {
    std::vector<Case> cases;
    std::vector<size_t> maxBytes;   // 逐输入位置: 池内最大字节数 (device 缓冲预分配用)
    size_t distinctShapes = 0;
    bool empty() const { return cases.empty(); }
    std::vector<std::string> shapeKeys() const;   // 与 cases 同序
};

// 解析 cases.json 并把全部 npy 一次读进内存。失败抛 std::runtime_error:
//   文件打不开、JSON 解析失败、cases 缺失/不是数组/为空、
//   id 缺失/为空/重复、inputs 缺失/为空、input 缺 path、
//   各 case 的输入个数不一致、samples 不是正整数、npy 读取失败 (透传 ReadNpy 消息)。
CasePool LoadCases(const std::string &casesJsonPath);

}  // namespace ge_execute
