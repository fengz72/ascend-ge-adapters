#pragma once
// GE Session 单图执行封装 (AIR / ONNX)。
// GE 与 ACL 头只出现在 session.cpp 里 (pimpl), 所以 main.cpp 和 CPU 契约测试
// 都不需要 GE 环境。所有失败都抛 std::runtime_error (消息含 GEGetErrorMsg)。
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cases.h"

namespace ge_execute {

struct SessionConfig {
    std::string graphPath;
    std::string format;                        // "air" | "onnx"; 空则按扩展名推断
    int device = 0;
    int instances = 1;                         // 图实例数 = 闭环并发数
    std::string aicoreNum;                     // "" = 不限核; 否则 "正整数|正整数"
    std::string precisionMode = "force_fp16";
    bool profiling = false;
    std::string profilingOutput;               // profiling=true 时的输出目录
    std::string aicMetrics = "PipeUtilization";
};

struct OutputTensor {
    std::string dtype;                         // 规范名 (bfloat16 原样报)
    std::vector<int64_t> shape;                // 运行时真实 shape
    std::vector<char> data;                    // D2H 后的主机字节
};

// 一次全量 D2H 的结果。discard=true 时 outputs 为空, 只回传耗时。
struct Readback {
    std::vector<OutputTensor> outputs;
    double d2hMs = 0;
};

struct RunTimings {
    double h2dMs = 0, execMs = 0, d2hMs = 0, e2eMs = 0;   // execMs 含 execute + sync
};

// "正整数|正整数" 之外的任何非空形式抛 std::runtime_error; 空串原样返回。
std::string NormalizeAicoreNum(const std::string &raw);

// 按扩展名推断 format: .air -> "air", .onnx -> "onnx", 其他抛 runtime_error。
std::string InferFormat(const std::string &graphPath);

class Session {
public:
    explicit Session(SessionConfig config);
    ~Session();
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;

    // 初始化顺序是实测约束, 不可调换:
    //   1. GEInitialize{ge.exec.deviceId,
    //                   [ge::ir_option::AICORE_NUM], [profiling 两个选项]}
    //      (不设 ge.graphRunMode: 它是可选项, GE 默认 0 = 在线推理)
    //   2. aclInit (ACL_ERROR_REPEAT_INITIALIZE 视为成功)
    //   3. ge::Session{ge.session_device_id, ge.exec.precision_mode}
    //   4. 每实例: 读图 (air: Graph::LoadFromFile / onnx: aclgrphParseONNX)
    //      -> AddGraph(gid = i+1) -> CompileGraph(gid) -> GetCompiledGraphSummary(gid)
    //      全部实例编译完才进第 5 步 (CompileGraph 必须在 aclrtSetDevice 之前)
    //   5. aclrtSetDevice -> aclrtGetCurrentContext
    //      (stream 必须建在**默认 context** 上; 显式 CreateContext 会让 GE executor
    //       报 "stream is not in current ctx")
    //   6. 每实例: aclrtCreateStream -> Session::LoadGraph(gid, {}, stream)
    //      (同一 gid 只能 LoadGraph 一次) -> 按 pool.maxBytes 预分配输入 device 缓冲
    //      -> 静态编译图 (summary->IsStatic()) 时按 summary 的 shape/dtype 精确分配
    //         输出缓冲并由调用方持有; 动态图不分配, 用 GE 返回的张量地址
    // profiling 输出目录在 GEInitialize 前创建。
    void Init(const CasePool &pool);

    // 工作线程进入时调用一次: aclrtSetCurrentContext(默认 context)。
    void BindThread() const;

    double compileMs() const;                  // 全部图实例编译总耗时
    double loadMs(int instance) const;         // 该实例 LoadGraph + 缓冲分配

    // H2D 该 case 的实际字节 -> 按该 case 的 shape 重建 gert::Tensor
    // (TensorData 的 size 给**分配大小**, shape 给该请求实际 shape)
    // -> ExecuteGraphWithStreamAsync -> aclrtSynchronizeStream。
    // 返回的 e2eMs = h2dMs + execMs, **不含 D2H**; 取输出另调 CollectOutputs,
    // 由调用方把 d2hMs 折进 e2e (口径见 README)。
    // instance 超出已建实例数、输入个数与缓冲不符、字节数超缓冲都抛 runtime_error。
    RunTimings Execute(int instance, const Case &request);

    // 把该实例上一次 Execute 的输出全量 D2H。
    // discard=false (infer): 返回主机字节, 每输出一个 OutputTensor。
    // discard=true  (bench): 付同样的传输代价但丢弃数据, outputs 为空 —— 主机侧
    //   复用每实例的 scratch 缓冲, 稳态不分配, 因此测到的是纯传输而不是分配器噪声。
    Readback CollectOutputs(int instance, bool discard = false);

    void Destroy();                            // 幂等; 析构会调用
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ge_execute
