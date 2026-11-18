// 最小 .npy 读写实现 (见 npy.h)。不依赖 GE / ACL, 可单独进 CPU 契约测试。
#include "npy.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ge_execute {
namespace {

constexpr char kMagic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};

size_t ProductOrThrow(const std::vector<int64_t> &shape, size_t itemsize,
                      const std::string &path) {
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("ReadNpy '" + path + "': negative shape dimension " +
                                     std::to_string(dim));
        }
        if (dim != 0 && count > std::numeric_limits<size_t>::max() /
                                     static_cast<size_t>(dim)) {
            throw std::runtime_error("ReadNpy '" + path + "': shape element count overflow");
        }
        count *= static_cast<size_t>(dim);
    }
    if (itemsize != 0 && count > std::numeric_limits<size_t>::max() / itemsize) {
        throw std::runtime_error("ReadNpy '" + path + "': byte size overflow");
    }
    return count * itemsize;
}

uint32_t ReadLE(const std::vector<char> &buf, size_t offset, size_t width) {
    uint32_t value = 0;
    for (size_t i = 0; i < width; ++i) {
        value |= static_cast<uint32_t>(static_cast<unsigned char>(buf[offset + i])) << (8 * i);
    }
    return value;
}

std::vector<char> ReadWholeFile(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("ReadNpy '" + path + "': cannot open file");
    }
    in.seekg(0, std::ios::end);
    std::streamoff length = in.tellg();
    if (length < 0) {
        throw std::runtime_error("ReadNpy '" + path + "': cannot determine file size");
    }
    in.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(length));
    if (length > 0 && !in.read(buf.data(), length)) {
        throw std::runtime_error("ReadNpy '" + path + "': truncated file");
    }
    return buf;
}

// 头部字典是 Python 字面量。只解析 descr/fortran_order/shape, 其余键跳过。
struct NpyHeader {
    bool hasDescr = false;
    bool hasFortran = false;
    bool hasShape = false;
    std::string descr;
    bool fortranOrder = false;
    std::vector<int64_t> shape;
};

class HeaderParser {
public:
    explicit HeaderParser(const std::string &text) : text_(text) {}

    NpyHeader Parse() {
        NpyHeader header;
        SkipSpace();
        Expect('{');
        SkipSpace();
        if (Consume('}')) {
            Fail("empty header dict");
        }
        while (true) {
            SkipSpace();
            const std::string key = ParseString();
            SkipSpace();
            Expect(':');
            SkipSpace();
            if (key == "descr") {
                header.descr = ParseString();
                header.hasDescr = true;
            } else if (key == "fortran_order") {
                header.fortranOrder = ParseBool();
                header.hasFortran = true;
            } else if (key == "shape") {
                header.shape = ParseShape();
                header.hasShape = true;
            } else {
                SkipValue();
            }
            SkipSpace();
            if (Consume(',')) {
                SkipSpace();
                if (Consume('}')) {
                    break;
                }
                continue;
            }
            if (Consume('}')) {
                break;
            }
            Fail("expected ',' or '}' in header dict");
        }
        SkipSpace();
        if (pos_ != text_.size()) {
            Fail("trailing characters after header dict");
        }
        if (!header.hasDescr || !header.hasFortran || !header.hasShape) {
            Fail("header dict missing descr/fortran_order/shape");
        }
        return header;
    }

private:
    [[noreturn]] void Fail(const std::string &reason) const {
        throw std::runtime_error(reason);
    }

    bool AtEnd() const { return pos_ >= text_.size(); }
    char Peek() const { return AtEnd() ? '\0' : text_[pos_]; }
    char Get() {
        if (AtEnd()) {
            Fail("unexpected end of header");
        }
        return text_[pos_++];
    }
    bool Consume(char expected) {
        if (Peek() != expected) {
            return false;
        }
        ++pos_;
        return true;
    }
    void Expect(char expected) {
        if (!Consume(expected)) {
            Fail(std::string("expected '") + expected + "' in header");
        }
    }
    void SkipSpace() {
        while (!AtEnd() && std::isspace(static_cast<unsigned char>(text_[pos_])) != 0) {
            ++pos_;
        }
    }

    std::string ParseString() {
        const char quote = Peek();
        if (quote != '\'' && quote != '"') {
            Fail("expected quoted string in header");
        }
        ++pos_;
        std::string out;
        while (true) {
            const char c = Get();
            if (c == quote) {
                break;
            }
            if (c == '\\') {
                out.push_back(Get());
                continue;
            }
            out.push_back(c);
        }
        return out;
    }

    bool ParseBool() {
        if (text_.compare(pos_, 4, "True") == 0) {
            pos_ += 4;
            return true;
        }
        if (text_.compare(pos_, 5, "False") == 0) {
            pos_ += 5;
            return false;
        }
        Fail("fortran_order must be True or False");
    }

    int64_t ParseNonNegativeInt() {
        SkipSpace();
        if (Consume('+')) {
            SkipSpace();
        }
        if (Peek() < '0' || Peek() > '9') {
            Fail("shape dimension is not a non-negative integer");
        }
        int64_t value = 0;
        while (!AtEnd() && Peek() >= '0' && Peek() <= '9') {
            const int digit = Get() - '0';
            if (value > (std::numeric_limits<int64_t>::max() - digit) / 10) {
                Fail("shape dimension overflow");
            }
            value = value * 10 + digit;
        }
        return value;
    }

    std::vector<int64_t> ParseShape() {
        Expect('(');
        SkipSpace();
        std::vector<int64_t> shape;
        if (Consume(')')) {
            return shape;  // 0 维标量
        }
        while (true) {
            shape.push_back(ParseNonNegativeInt());
            SkipSpace();
            if (Consume(',')) {
                SkipSpace();
                if (Consume(')')) {
                    break;  // 单元素元组尾逗号 (3,)
                }
                continue;
            }
            if (Consume(')')) {
                break;
            }
            Fail("expected ',' or ')' in shape tuple");
        }
        return shape;
    }

    // 未知键: 接受字符串/数字/布尔/None/嵌套 tuple/list, 只做括号配平。
    void SkipValue() {
        SkipSpace();
        const char c = Peek();
        if (c == '\'' || c == '"') {
            ParseString();
            return;
        }
        if (c == '(' || c == '[') {
            const char close = (c == '(') ? ')' : ']';
            ++pos_;
            SkipSpace();
            if (Consume(close)) {
                return;
            }
            while (true) {
                SkipValue();
                SkipSpace();
                if (Consume(',')) {
                    SkipSpace();
                    if (Consume(close)) {
                        return;
                    }
                    continue;
                }
                if (Consume(close)) {
                    return;
                }
                Fail("unbalanced value in header dict");
            }
        }
        const size_t start = pos_;
        while (!AtEnd()) {
            const char ch = Peek();
            if (ch == ',' || ch == '}' || ch == ')' || ch == ']' ||
                std::isspace(static_cast<unsigned char>(ch)) != 0) {
                break;
            }
            ++pos_;
        }
        if (pos_ == start) {
            Fail("invalid value in header dict");
        }
    }

    const std::string &text_;
    size_t pos_ = 0;
};

std::string ShapeTuple(const std::vector<int64_t> &shape) {
    std::ostringstream out;
    out << '(';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << shape[i];
    }
    if (shape.size() == 1) {
        out << ',';
    }
    out << ')';
    return out.str();
}

}  // namespace

size_t DtypeBytes(const std::string &dtype) {
    if (dtype == "float16" || dtype == "bfloat16" || dtype == "int16" ||
        dtype == "uint16") {
        return 2;
    }
    if (dtype == "float32" || dtype == "int32" || dtype == "uint32") {
        return 4;
    }
    if (dtype == "float64" || dtype == "int64" || dtype == "uint64") {
        return 8;
    }
    if (dtype == "int8" || dtype == "uint8" || dtype == "bool") {
        return 1;
    }
    return 0;
}

std::string DtypeToDescr(const std::string &dtype) {
    // 单字节类型 numpy 用 '|'; 多字节用 '<' (C-order 小端)。
    if (dtype == "bool") return "|b1";
    if (dtype == "int8") return "|i1";
    if (dtype == "uint8") return "|u1";
    if (dtype == "bfloat16") return "|u2";  // numpy 无 bfloat16, 按 uint16 存
    if (dtype == "float16") return "<f2";
    if (dtype == "float32") return "<f4";
    if (dtype == "float64") return "<f8";
    if (dtype == "int16") return "<i2";
    if (dtype == "int32") return "<i4";
    if (dtype == "int64") return "<i8";
    if (dtype == "uint16") return "<u2";
    if (dtype == "uint32") return "<u4";
    if (dtype == "uint64") return "<u8";
    return "";
}

std::string DescrToDtype(const std::string &descr) {
    // 只接受小端 '<' 与不分字节序 '|'。'|u2' 特指 bfloat16 (详见 npy.h)。
    if (descr.size() == 2 && descr[1] == '?' && (descr[0] == '<' || descr[0] == '|')) {
        return "bool";
    }
    if (descr.size() < 3) {
        return "";
    }
    const char order = descr[0];
    if (order != '<' && order != '|') {
        return "";
    }
    const char kind = descr[1];
    size_t width = 0;
    for (size_t i = 2; i < descr.size(); ++i) {
        if (descr[i] < '0' || descr[i] > '9') {
            return "";
        }
        width = width * 10 + static_cast<size_t>(descr[i] - '0');
        if (width > 64) {
            return "";
        }
    }
    if (kind == 'f') {
        if (width == 2) return "float16";
        if (width == 4) return "float32";
        if (width == 8) return "float64";
        return "";
    }
    if (kind == 'i') {
        if (width == 1) return "int8";
        if (width == 2) return "int16";
        if (width == 4) return "int32";
        if (width == 8) return "int64";
        return "";
    }
    if (kind == 'u') {
        if (width == 1) return "uint8";
        if (width == 2) return order == '|' ? "bfloat16" : "uint16";
        if (width == 4) return "uint32";
        if (width == 8) return "uint64";
        return "";
    }
    if (kind == 'b' && width == 1) {
        return "bool";
    }
    return "";
}

NpyArray ReadNpy(const std::string &path, const std::string &dtype_override) {
    const std::vector<char> raw = ReadWholeFile(path);
    if (raw.size() < 10) {
        throw std::runtime_error("ReadNpy '" + path + "': truncated header");
    }
    if (std::memcmp(raw.data(), kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("ReadNpy '" + path + "': bad magic");
    }
    const uint8_t major = static_cast<uint8_t>(raw[6]);
    const uint8_t minor = static_cast<uint8_t>(raw[7]);
    if (minor != 0 || (major != 1 && major != 2)) {
        throw std::runtime_error("ReadNpy '" + path + "': unsupported version " +
                                 std::to_string(major) + "." + std::to_string(minor) +
                                 " (only 1.0 / 2.0)");
    }
    size_t headerStart = 0;
    uint32_t headerLen = 0;
    if (major == 1) {
        headerStart = 10;
        headerLen = ReadLE(raw, 8, 2);
    } else {
        if (raw.size() < 12) {
            throw std::runtime_error("ReadNpy '" + path + "': truncated v2 header");
        }
        headerStart = 12;
        headerLen = ReadLE(raw, 8, 4);
    }
    if (headerLen == 0) {
        throw std::runtime_error("ReadNpy '" + path + "': empty header");
    }
    if (headerStart + headerLen > raw.size()) {
        throw std::runtime_error("ReadNpy '" + path + "': header length " +
                                 std::to_string(headerLen) + " out of range");
    }

    NpyHeader header;
    try {
        header = HeaderParser(std::string(raw.data() + headerStart, headerLen)).Parse();
    } catch (const std::runtime_error &ex) {
        throw std::runtime_error("ReadNpy '" + path + "': header dict parse failed: " +
                                 ex.what());
    }
    if (header.fortranOrder) {
        throw std::runtime_error("ReadNpy '" + path +
                                 "': fortran_order=True is not supported");
    }

    const std::string headerDtype = DescrToDtype(header.descr);
    if (headerDtype.empty()) {
        throw std::runtime_error("ReadNpy '" + path + "': unsupported descr '" +
                                 header.descr + "'");
    }
    std::string dtype = headerDtype;
    size_t itemsize = DtypeBytes(headerDtype);
    if (!dtype_override.empty()) {
        const size_t overrideBytes = DtypeBytes(dtype_override);
        if (overrideBytes == 0) {
            throw std::runtime_error("ReadNpy '" + path + "': unsupported dtype_override '" +
                                     dtype_override + "'");
        }
        if (overrideBytes != itemsize) {
            throw std::runtime_error("ReadNpy '" + path + "': dtype_override '" +
                                     dtype_override + "' size " +
                                     std::to_string(overrideBytes) +
                                     " != header dtype '" + headerDtype + "' size " +
                                     std::to_string(itemsize));
        }
        dtype = dtype_override;
        itemsize = overrideBytes;
    }

    const size_t expected = ProductOrThrow(header.shape, itemsize, path);
    const size_t dataStart = headerStart + headerLen;
    const size_t actual = raw.size() - dataStart;
    if (actual != expected) {
        throw std::runtime_error("ReadNpy '" + path + "': data size " +
                                 std::to_string(actual) + " != expected " +
                                 std::to_string(expected));
    }

    NpyArray out;
    out.dtype = dtype;
    out.shape = header.shape;
    out.data.assign(raw.begin() + static_cast<std::ptrdiff_t>(dataStart), raw.end());
    return out;
}

void WriteNpy(const std::string &path, const std::string &dtype,
              const std::vector<int64_t> &shape, const void *data, size_t bytes) {
    const size_t itemsize = DtypeBytes(dtype);
    if (itemsize == 0) {
        throw std::runtime_error("WriteNpy '" + path + "': unsupported dtype '" + dtype + "'");
    }
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::runtime_error("WriteNpy '" + path + "': negative shape dimension " +
                                     std::to_string(dim));
        }
        if (dim != 0 && count > std::numeric_limits<size_t>::max() /
                                     static_cast<size_t>(dim)) {
            throw std::runtime_error("WriteNpy '" + path + "': shape element count overflow");
        }
        count *= static_cast<size_t>(dim);
    }
    if (itemsize != 0 && count > std::numeric_limits<size_t>::max() / itemsize) {
        throw std::runtime_error("WriteNpy '" + path + "': byte size overflow");
    }
    const size_t expected = count * itemsize;
    if (bytes != expected) {
        throw std::runtime_error("WriteNpy '" + path + "': data size " +
                                 std::to_string(bytes) + " != shape*dtype " +
                                 std::to_string(expected));
    }
    if (bytes != 0 && data == nullptr) {
        throw std::runtime_error("WriteNpy '" + path + "': null data pointer");
    }

    const std::string descr = DtypeToDescr(dtype);
    const std::string dict = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " +
                             ShapeTuple(shape) + ", }";
    // 从文件起点补齐到 64 字节对齐, 末字节必须是 '\n'。
    const size_t pad = (64 - ((10 + dict.size() + 1) % 64)) % 64;
    std::string header = dict;
    header.append(pad, ' ');
    header.push_back('\n');
    if (header.size() > 0xFFFF) {
        throw std::runtime_error("WriteNpy '" + path + "': v1.0 header exceeds 65535 bytes");
    }

    const std::filesystem::path target(path);
    if (target.has_parent_path() && !target.parent_path().empty()) {
        std::error_code ec;
        std::filesystem::create_directories(target.parent_path(), ec);
        if (ec) {
            throw std::runtime_error("WriteNpy '" + path + "': cannot create parent dir: " +
                                     ec.message());
        }
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("WriteNpy '" + path + "': cannot open for write");
    }
    out.write(kMagic, sizeof(kMagic));
    const char version[2] = {1, 0};
    out.write(version, sizeof(version));
    const char length[2] = {static_cast<char>(header.size() & 0xFF),
                            static_cast<char>((header.size() >> 8) & 0xFF)};
    out.write(length, sizeof(length));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (bytes != 0) {
        out.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
    }
    if (!out) {
        throw std::runtime_error("WriteNpy '" + path + "': write failed");
    }
}

}  // namespace ge_execute
