#pragma once
// 最小 .npy 读写 (NumPy v1.0 / v2.0 头, C-order, 白名单 dtype)。
// 不依赖 GE / ACL, 可单独进 CPU 契约测试。
//
// 白名单 dtype (规范名 -> 字节数):
//   float16 2, bfloat16 2, float32 4, float64 8,
//   int8 1, int16 2, int32 4, int64 8,
//   uint8 1, uint16 2, uint32 4, uint64 8, bool 1
// numpy 没有 bfloat16: 模型侧按 uint16 存, 由调用方传 dtype_override="bfloat16" 标注。
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ge_execute {

struct NpyArray {
    std::string dtype;              // 规范名 (见上)
    std::vector<int64_t> shape;
    std::vector<char> data;         // C-order 原始字节, size == 元素数 * DtypeBytes(dtype)
};

// 白名单外返回 0。
size_t DtypeBytes(const std::string &dtype);

// 规范 dtype -> numpy descr ("bfloat16" 写 "|u2"; 单字节类型 numpy 用 '|')。
// 白名单外返回 ""。
std::string DtypeToDescr(const std::string &dtype);

// numpy descr -> 规范 dtype。只接受小端 '<' 与不分字节序 '|' 的白名单类型;
// 大端 '>'、对象 'O'、字符串 'S'/'U'、复数 'c'、datetime、白名单外一律返回 ""。
std::string DescrToDtype(const std::string &descr);

// 读取 path。dtype_override 非空时用它替代 npy 头里的 dtype 解释 (字节数必须一致)。
// 失败抛 std::runtime_error, 消息含 path 与原因:
//   打不开 / 截断、magic 错、版本不是 1.0 或 2.0、header 长度越界、
//   header 字典解析失败、fortran_order=True、descr 不支持、
//   dtype_override 不支持或与头部字节数不符、数据字节数 != 元素数 * itemsize。
NpyArray ReadNpy(const std::string &path, const std::string &dtype_override = "");

// 写 v1.0 头: magic "\x93NUMPY" + 1 字节主版本 + 1 字节次版本 + 2 字节小端 header_len
// + 字典字面量 "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }" + '\n',
// 从文件头起补齐到 64 字节对齐 (用空格填充)。父目录不存在则递归创建。
// shape 为空视为标量 (0 维)。bytes 与 shape * DtypeBytes(dtype) 不符抛 runtime_error。
void WriteNpy(const std::string &path, const std::string &dtype,
              const std::vector<int64_t> &shape, const void *data, size_t bytes);

}  // namespace ge_execute
