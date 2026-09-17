// =============================================================================
// bench_ge_latency — GESession 多线程闭环延迟/吞吐 benchmark (thread sweep)
//
// 镜像 bench_latency (ATC OM 路径) 的结构与口径, 图执行引擎换成 GESession 在线
// 加载的 AIR 图, 输出相同格式的 sweep 表, 供 GE 在线 vs ATC OM 直接对比。
//
// 架构 (单 Session 多图):
//   GEInitialize(graphRunMode=1, 进程级一次)
//   → ge::Session (一次)
//   → 启动阶段: 对每线程 i: Graph::LoadFromFile(air) → AddGraph(gid=i)
//      → CompileGraph (串行, ~10s×N, 耗时单独报告) → LoadGraph(gid=i, stream_i)
//   → 每线程独立闭环: DataGen → H2D → ExecuteGraphWithStreamAsync → Sync → D2H
//   → warmup 轮流打满所有图 (触发各自的 shape 特化, 不计入)
//
// 输入顺序与图 Data 节点一致:
//   基线: [asl, input_ids, position_ids];  prefix: [act_q, act_kv, input_ids, position_ids]
//
// 运行环境: source set_env.sh; prefix AIR 需 source vendor set_env.bash;
//   CANN tbe pywrapper 内嵌 /usr/bin/python3 无 numpy, 需
//   export PYTHONPATH=/usr/local/python3.11.15/lib/python3.11/site-packages:$PYTHONPATH
// =============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>
#include <atomic>
#include <random>
#include <cmath>
#include <map>
#include <sys/stat.h>
#include <set>

#include "acl/acl.h"
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "exe_graph/runtime/tensor.h"

// ============================================================================
// Constants (与 bench_latency 完全一致, 保证同口径)
// ============================================================================

#define ACL_CHECK(ret, msg) \
    do { \
        if ((ret) != ACL_SUCCESS) { \
            std::cerr << "[ERROR] " << msg << ", ret=" << ret << std::endl; \
            return ret; \
        } \
    } while (0)

#define GE_CHECK(ret, msg) \
    do { \
        if ((ret) != ge::SUCCESS) { \
            std::cerr << "[ERROR] " << msg << ", ret=" << ret << std::endl; \
            return 1; \
        } \
    } while (0)

static const int    VOCAB_SIZE         = 151936;
static const int    MAX_BATCH_SIZE     = 40;   // --batch-size 上限 (batch+1 的 act 数组也受此约束)
static const int    MAX_SEQ_LEN        = 218;
static const int    MAX_TOTAL_TOKENS   = MAX_BATCH_SIZE * MAX_SEQ_LEN;
static const int    MAX_PREFIX_LEN     = 32;   // prefix 长度上限 (参数校验用)

static const double BATCH_AVG   = 9.8;
static const double BATCH_STD   = 0.35;
static const bool   BATCH_FIXED = true;
static int          BATCH_FIXED_VAL = 10;   // --batch-size 可覆盖
static const double SEQ_LOG_MEAN = 4.997;
static const double SEQ_LOG_STD  = 0.167;

static bool   g_seq_fixed = false;
static int    g_seq_fixed_val = 208;

// prefix 模式: packed prefix-in-Q 布局 [prefix(P), req0, req1, ...]
// 每请求总长 = seq_len (prefix + own), 3 输入 (与基线图同构, 图 Data 节点顺序):
//   act/input_ids/position_ids, act = cumsum([P, L0, ...]) (prefix 独立成 batch 0)
// --prefix <P> 固定 P; --prefix <P1>-<P2> 每请求均匀随机 P∈[P1,P2]
static int    g_prefix_len = 0;
static int    g_prefix_max = 0;   // 随机上界 (固定时 = g_prefix_len)

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

static inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

// ============================================================================
// Request (与 bench_latency 一致)
// ============================================================================

struct Request {
    int req_id;
    std::vector<int64_t> input_ids;
    std::vector<int64_t> actual_seq_lengths;   // 基线: cumsum(seq_lens)
    std::vector<int64_t> act;                  // prefix: cumsum([P, L0, ...]) (batch+1 元素)
    std::vector<int64_t> position_ids;
    int total_tokens;
    int batch_size;

    TimePoint arrive;
    TimePoint gen_done;
    TimePoint h2d_done;
    TimePoint execute_done;
    TimePoint d2h_done;
};

class RequestGenerator {
public:
    RequestGenerator(unsigned seed)
        : rng_(seed),
          batch_dist_(BATCH_AVG, BATCH_STD),
          seq_log_dist_(SEQ_LOG_MEAN, SEQ_LOG_STD) {}

    Request generate(int req_id, TimePoint arrive_time) {
        Request req;
        req.req_id = req_id;
        req.arrive = arrive_time;

        int bs;
        if (BATCH_FIXED) {
            bs = BATCH_FIXED_VAL;
        } else {
            bs = (int)std::round(batch_dist_(rng_));
            bs = std::max(1, std::min(MAX_BATCH_SIZE, bs));
        }
        req.batch_size = bs;

        std::vector<int> seq_lens(bs);
        int total = 0;
        for (int i = 0; i < bs; i++) {
            int sl;
            if (g_seq_fixed) {
                sl = g_seq_fixed_val;
            } else {
                sl = (int)std::round(std::exp(seq_log_dist_(rng_)));
                sl = std::max(1, std::min(MAX_SEQ_LEN, sl));
            }
            seq_lens[i] = sl;
            total += sl;
        }

        if (g_prefix_len > 0) {
            // 每请求 P: 固定值或 [g_prefix_len, g_prefix_max] 均匀随机
            const int P = (g_prefix_max > g_prefix_len)
                              ? (g_prefix_len + (int)(rng_() % (unsigned)(g_prefix_max - g_prefix_len + 1)))
                              : g_prefix_len;
            std::vector<int> own_lens(bs);
            int own_total = 0;
            for (int i = 0; i < bs; i++) {
                own_lens[i] = std::max(1, seq_lens[i] - P);
                own_total += own_lens[i];
            }
            req.total_tokens = P + own_total;

            req.input_ids.assign(req.total_tokens, 0);

            std::vector<int64_t> pos_ids(req.total_tokens);
            int idx = 0;
            for (int p = 0; p < P; p++) pos_ids[idx++] = p;
            for (int i = 0; i < bs; i++)
                for (int p = 0; p < own_lens[i]; p++) pos_ids[idx++] = P + p;
            req.position_ids = std::move(pos_ids);

            // act = cumsum([P, L0, L1, ...]) — prefix 独立成 batch 0 (KV 内嵌版算子)
            req.act.resize(bs + 1);
            int acc = 0;
            req.act[0] = (acc += P);
            for (int i = 0; i < bs; i++) {
                acc += own_lens[i];
                req.act[i + 1] = acc;
            }

            req.gen_done = Clock::now();
            return req;
        }

        req.total_tokens = total;
        req.input_ids.resize(total, 0);

        req.actual_seq_lengths.resize(bs);
        int acc = 0;
        for (int i = 0; i < bs; i++) {
            acc += seq_lens[i];
            req.actual_seq_lengths[i] = acc;
        }

        std::vector<int64_t> pos_ids(total);
        int idx = 0;
        for (int i = 0; i < bs; i++) {
            for (int p = 0; p < seq_lens[i]; p++) {
                pos_ids[idx++] = p;
            }
        }
        req.position_ids = std::move(pos_ids);
        req.gen_done = Clock::now();
        return req;
    }

private:
    std::mt19937 rng_;
    std::normal_distribution<double> batch_dist_;
    std::normal_distribution<double> seq_log_dist_;
};

// ============================================================================
// Latency stats (与 bench_latency 一致)
// ============================================================================

struct LatencyStats {
    std::vector<double> gen_times;
    std::vector<double> h2d_times;
    std::vector<double> exec_times;
    std::vector<double> d2h_times;
    std::vector<double> e2e_times;
    long total_tokens = 0;
    std::map<int, std::vector<double>> per_thread;
    std::mutex mtx;

    void record(const Request& req, int thread_id) {
        std::lock_guard<std::mutex> lock(mtx);
        gen_times.push_back(elapsed_ms(req.arrive, req.gen_done));
        h2d_times.push_back(elapsed_ms(req.gen_done, req.h2d_done));
        exec_times.push_back(elapsed_ms(req.h2d_done, req.execute_done));
        d2h_times.push_back(elapsed_ms(req.execute_done, req.d2h_done));
        e2e_times.push_back(elapsed_ms(req.arrive, req.d2h_done));
        total_tokens += req.total_tokens;
        per_thread[thread_id].push_back(elapsed_ms(req.arrive, req.d2h_done));
    }

    static double percentile(std::vector<double> data, double p) {
        if (data.empty()) return 0.0;
        std::sort(data.begin(), data.end());
        size_t idx = (size_t)(data.size() * p / 100.0);
        if (idx >= data.size()) idx = data.size() - 1;
        return data[idx];
    }

    static double mean(const std::vector<double>& data) {
        if (data.empty()) return 0.0;
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }

    static void print_row(const std::string& name, const std::vector<double>& data) {
        std::cout << "  " << std::left << std::setw(10) << name
                  << " avg=" << std::fixed << std::setprecision(3) << mean(data)
                  << " p50=" << percentile(data, 50)
                  << " p90=" << percentile(data, 90)
                  << " p99=" << percentile(data, 99)
                  << " max=" << percentile(data, 100) << "\n";
    }

    void report(double total_ms, int num_threads, int warmup) const {
        int n = e2e_times.size();
        if (n == 0) {
            std::cout << "[WARN] No requests completed" << std::endl;
            return;
        }
        double qps = n / (total_ms / 1000.0);
        double tps = total_tokens / (total_ms / 1000.0);

        std::cout << "\n" << std::string(72, '=') << "\n";
        std::cout << "GE Latency Benchmark Results (Independent Threads, AIR online)\n";
        std::cout << std::string(72, '=') << "\n";
        std::cout << "  Threads:          " << num_threads << "\n";
        std::cout << "  Requests:         " << n << " (warmup=" << warmup << ")\n";
        std::cout << "  Total Time:       " << std::fixed << std::setprecision(2) << total_ms << " ms\n";
        std::cout << "  Total Tokens:     " << total_tokens << "\n";
        std::cout << "  ----------------------------------------------------------------\n";
        std::cout << "  QPS:              " << std::fixed << std::setprecision(2) << qps << " req/s\n";
        std::cout << "  Token Throughput: " << std::setprecision(0) << tps << " tokens/s\n";
        std::cout << "  ----------------------------------------------------------------\n";
        print_row("E2E", e2e_times);
        print_row("  DataGen", gen_times);
        print_row("  H2D", h2d_times);
        print_row("  Execute", exec_times);
        print_row("  D2H", d2h_times);

        std::cout << "  " << std::string(68, '-') << "\n";
        std::cout << "  Per-Thread E2E:\n";
        for (auto& kv : per_thread) {
            std::cout << "    Thread " << kv.first << ": " << kv.second.size()
                      << " reqs, avg_e2e=" << std::fixed << std::setprecision(3)
                      << mean(kv.second) << "ms\n";
        }
        std::cout << std::string(72, '=') << "\n\n";
    }

    struct Summary {
        double qps;
        double tps;
        double e2e_avg, e2e_p99;
        double gen_avg;
        double h2d_avg;
        double exec_avg, exec_p99;
        double d2h_avg;
    };

    Summary get_summary(double total_ms) const {
        double qps = e2e_times.size() / (total_ms / 1000.0);
        double tps = total_tokens / (total_ms / 1000.0);
        return {qps, tps,
                mean(e2e_times), percentile(e2e_times, 99),
                mean(gen_times),
                mean(h2d_times),
                mean(exec_times), percentile(exec_times, 99),
                mean(d2h_times)};
    }
};

// ============================================================================
// GeContext: 每线程一个图实例 (graph_id + stream + 输入输出缓冲)
// ============================================================================

class GeContext {
public:
    int thread_id;
    uint32_t graph_id;
    aclrtStream stream;
    std::vector<void*> input_buffers;      // 持久 device 缓冲
    std::vector<size_t> input_max_sizes;
    std::set<void*> output_addrs;          // GE 分配的输出缓冲 (去重, cleanup 释放)
    std::vector<uint8_t> host_output;

    static constexpr size_t kMaxInputs = 4;

    GeContext(int tid, uint32_t gid) : thread_id(tid), graph_id(gid), stream(nullptr) {
        // 图输入顺序 (Data 节点): act, input_ids, position_ids (prefix 与基线同构)
        if (g_prefix_len > 0) {
            input_max_sizes = {
                (size_t)(MAX_BATCH_SIZE + 1) * 8,  // act cumsum([P, L0, ...]) (batch+1)
                (size_t)MAX_TOTAL_TOKENS * 8,      // input_ids
                (size_t)MAX_TOTAL_TOKENS * 8,      // position_ids
            };
        } else {
            input_max_sizes = {
                (size_t)MAX_BATCH_SIZE * 8,     // actual_seq_lengths
                (size_t)MAX_TOTAL_TOKENS * 8,   // input_ids
                (size_t)MAX_TOTAL_TOKENS * 8,   // position_ids
            };
        }
        host_output.resize((size_t)MAX_BATCH_SIZE * VOCAB_SIZE * 2);
    }

    int init(const std::string& model_path_ununsed, ge::Session* session) {
        // 主线程已 aclrtSetDevice (默认 context); stream 建在默认 context,
        // 与 GE 内部执行线程的 context 绑定一致 (显式 CreateContext 会导致
        // GE executor 线程报 "stream is not in current ctx")
        ACL_CHECK(aclrtCreateStream(&stream), "create stream " << thread_id);

        for (size_t i = 0; i < input_max_sizes.size(); i++) {
            void* ptr = nullptr;
            ACL_CHECK(aclrtMalloc(&ptr, input_max_sizes[i], ACL_MEM_MALLOC_HUGE_FIRST),
                      "malloc input " << thread_id);
            input_buffers.push_back(ptr);
        }
        // 图已由外部 CompileGraph; 此处绑定 stream
        GE_CHECK(session->LoadGraph(graph_id, {}, stream), "LoadGraph " << graph_id);
        return ACL_SUCCESS;
    }

    // H2D: 拷贝请求各输入到持久缓冲 + 构造 gert::Tensor (顺序 = 图 Data 节点)
    int set_inputs(const Request& req) {
        struct In {
            const void* ptr;
            size_t bytes;
            int64_t dim;
        };
        In ins[kMaxInputs];
        int n = 0;
        if (g_prefix_len > 0) {
            ins[n++] = {req.act.data(), req.act.size() * 8, (int64_t)req.act.size()};
            ins[n++] = {req.input_ids.data(), req.input_ids.size() * 8, (int64_t)req.total_tokens};
            ins[n++] = {req.position_ids.data(), req.position_ids.size() * 8, (int64_t)req.total_tokens};
        } else {
            ins[n++] = {req.actual_seq_lengths.data(), req.actual_seq_lengths.size() * 8,
                        (int64_t)req.batch_size};
            ins[n++] = {req.input_ids.data(), req.input_ids.size() * 8, (int64_t)req.total_tokens};
            ins[n++] = {req.position_ids.data(), req.position_ids.size() * 8, (int64_t)req.total_tokens};
        }

        dev_inputs.clear();
        for (int i = 0; i < n; i++) {
            if (ins[i].bytes > input_max_sizes[i]) {
                std::cerr << "[ERROR] Input " << i << " size " << ins[i].bytes
                          << " > buffer " << input_max_sizes[i] << std::endl;
                return ACL_ERROR_INVALID_PARAM;
            }
            ACL_CHECK(aclrtMemcpy(input_buffers[i], input_max_sizes[i], ins[i].ptr, ins[i].bytes,
                                  ACL_MEMCPY_HOST_TO_DEVICE),
                      "H2D input[" << i << "]");

            gert::Tensor t;
            gert::StorageShape ss;
            ss.MutableOriginShape().AppendDim(ins[i].dim);
            ss.MutableStorageShape().AppendDim(ins[i].dim);
            t.GetShape() = ss;
            t.MutableFormat() = gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {});
            t.SetDataType(ge::DT_INT64);
            gert::TensorData td(input_buffers[i], nullptr, input_max_sizes[i], gert::kOnDeviceHbm);
            t.SetData(std::move(td));
            dev_inputs.emplace_back(std::move(t));
        }
        return ACL_SUCCESS;
    }

    int execute(ge::Session* session) {
        dev_outputs.clear();
        GE_CHECK(session->ExecuteGraphWithStreamAsync(graph_id, stream, dev_inputs, dev_outputs),
                 "ExecuteGraphWithStreamAsync " << graph_id);
        ACL_CHECK(aclrtSynchronizeStream(stream), "sync stream " << thread_id);
        for (auto& out : dev_outputs) {
            if (out.GetAddr() != nullptr) output_addrs.insert(out.GetAddr());
        }
        return ACL_SUCCESS;
    }

    int d2h(int batch_size) {
        if (dev_outputs.empty()) return ACL_ERROR_INTERNAL_ERROR;
        size_t out_bytes = (size_t)batch_size * VOCAB_SIZE * 2;
        size_t actual = dev_outputs[0].GetSize();
        if (actual < out_bytes) out_bytes = actual;
        ACL_CHECK(aclrtMemcpy(host_output.data(), host_output.size(),
                              dev_outputs[0].GetAddr(), out_bytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "D2H output thread " << thread_id);
        return ACL_SUCCESS;
    }

    void cleanup() {
        for (void* p : input_buffers) {
            if (p) aclrtFree(p);
        }
        input_buffers.clear();
        for (void* p : output_addrs) {
            if (p) aclrtFree(p);
        }
        output_addrs.clear();
        if (stream) {
            aclrtDestroyStream(stream);
            stream = nullptr;
        }
    }

private:
    std::vector<gert::Tensor> dev_inputs;
    std::vector<gert::Tensor> dev_outputs;
};

// ============================================================================
// Benchmark runner (单 Session 多图, 图预先编译加载, sweep 各档复用)
// ============================================================================

struct BenchResult {
    int num_threads;
    double total_ms;
    LatencyStats::Summary summary;
};

static int g_device_id = 0;
static aclrtContext g_default_ctx = nullptr;

static int run_benchmark(ge::Session* session, const std::vector<std::unique_ptr<GeContext>>& ctxs,
                         int num_threads, int total_requests, int warmup, LatencyStats& stats,
                         double& total_ms_out) {
    int requests_per_thread = total_requests / num_threads;
    std::atomic<int> errors(0);

    // warmup: 轮流打满所有图 (触发各自 shape 特化), 不计入
    if (warmup > 0) {
        std::cout << "[INFO] Warmup (" << warmup << " requests, round-robin over "
                  << num_threads << " graphs)..." << std::endl;
        RequestGenerator gen(0);
        for (int i = 0; i < warmup; i++) {
            Request req = gen.generate(i, Clock::now());
            GeContext* c = ctxs[i % num_threads].get();
            if (c->set_inputs(req) != ACL_SUCCESS) return 1;
            if (c->execute(session) != ACL_SUCCESS) return 1;
            if (c->d2h(req.batch_size) != ACL_SUCCESS) return 1;
        }
        std::cout << "[INFO] Warmup done" << std::endl;
    }

    TimePoint bench_start = Clock::now();
    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t]() {
            aclrtSetCurrentContext(g_default_ctx);  // 共享默认 context (与 GE executor 一致)
            GeContext* ctx = ctxs[t].get();
            RequestGenerator gen(42 + t * 1000);
            for (int i = 0; i < requests_per_thread; i++) {
                int req_id = t * requests_per_thread + i;
                Request req = gen.generate(req_id, Clock::now());

                req.gen_done = Clock::now();
                if (ctx->set_inputs(req) != ACL_SUCCESS) {
                    errors.fetch_add(1);
                    return;
                }
                req.h2d_done = Clock::now();

                if (ctx->execute(session) != ACL_SUCCESS) {
                    errors.fetch_add(1);
                    return;
                }
                req.execute_done = Clock::now();

                if (ctx->d2h(req.batch_size) != ACL_SUCCESS) {
                    errors.fetch_add(1);
                    return;
                }
                req.d2h_done = Clock::now();

                stats.record(req, t);
            }
        });
    }
    for (auto& w : workers) w.join();
    TimePoint bench_end = Clock::now();

    double total_ms = elapsed_ms(bench_start, bench_end);
    total_ms_out = total_ms;
    stats.report(total_ms, num_threads, warmup);
    if (errors.load() > 0) {
        std::cout << "[WARN] " << errors.load() << " errors occurred" << std::endl;
        return 1;
    }
    return 0;
}

// ============================================================================
// Sweep table (与 bench_latency 一致)
// ============================================================================

static void print_sweep_table(const std::vector<BenchResult>& results) {
    std::cout << "\n" << std::string(130, '=') << "\n";
    std::cout << "GE Sweep Results: Latency & Throughput vs Thread Count (AIR online, single session multi-graph)\n";
    std::cout << std::string(130, '=') << "\n";
    std::cout << std::right
              << std::setw(7) << "Threads" << " | "
              << std::setw(10) << "QPS" << " | "
              << std::setw(12) << "Token/s" << " | "
              << std::setw(10) << "Gen avg" << " | "
              << std::setw(10) << "H2D avg" << " | "
              << std::setw(10) << "Exec avg" << " | "
              << std::setw(10) << "Exec p99" << " | "
              << std::setw(10) << "D2H avg" << " | "
              << std::setw(10) << "E2E avg" << " | "
              << std::setw(10) << "E2E p99" << "\n";
    std::cout << std::string(130, '-') << "\n";
    for (auto& r : results) {
        std::cout << std::right
                  << std::setw(7) << r.num_threads << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.summary.qps << " | "
                  << std::setw(12) << std::setprecision(0) << r.summary.tps << " | "
                  << std::setw(10) << std::setprecision(3) << r.summary.gen_avg << " | "
                  << std::setw(10) << r.summary.h2d_avg << " | "
                  << std::setw(10) << std::setprecision(2) << r.summary.exec_avg << " | "
                  << std::setw(10) << r.summary.exec_p99 << " | "
                  << std::setw(10) << std::setprecision(3) << r.summary.d2h_avg << " | "
                  << std::setw(10) << std::setprecision(2) << r.summary.e2e_avg << " | "
                  << std::setw(10) << r.summary.e2e_p99 << "\n";
    }
    std::cout << std::string(130, '=') << "\n\n";
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\nGE (GESession, AIR online) latency & throughput benchmark.\n"
              << "Mirrors bench_latency (ATC OM path) for direct comparison.\n"
              << "\nOptions:\n"
              << "  --model <path>       Path to .air model file (required)\n"
              << "  --threads <N>        Number of independent threads (default: 1)\n"
              << "  --sweep <list>       Comma-separated thread counts (e.g. 1,2,3,4,5,6)\n"
              << "  --requests <N>       Total requests across all threads (default: 10000)\n"
              << "  --warmup <N>         Warmup requests, round-robin over graphs (default: 50)\n"
              << "  --device-id <id>     NPU device ID (default: 0)\n"
              << "  --fixed-seq <len>    Fix all sequence lengths to <len> (default: random)\n"
              << "  --prefix <P|P1-P2>   Packed prefix-in-Q mode (dynamic-P graph, 7 inputs);\n"
              << "                       fixed P or per-request uniform random in [P1,P2]\n"
              << "  --batch-size <N>     Fixed batch size per request (default 10, max "
              << MAX_BATCH_SIZE << ")\n"
              << "  --aicore-num <spec>  Limit AI cores: int N (=N|2N, AIC|AIV 1:2) or 'aic|aiv';\n"
              << "                       empty = all cores (injected via AddGraph options)\n"
              << "  --profiling          Enable GE profiling (ge.exec.profilingMode via GEInitialize)\n"
              << "  --profiling_output <dir>  Profiling output dir (default: ./profiling_data)\n"
              << "  -h, --help           Show this help\n"
              << "\nExamples:\n"
              << "  " << prog << " --model m.air --sweep 1,2,3,4,5,6 --requests 8000 --device-id 14 --fixed-seq 208\n"
              << "  " << prog << " --model m-prefix.air --sweep 1,2,3,4,5,6 --requests 8000 --device-id 14 --fixed-seq 208 --prefix 20\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::string model_path;
    int num_threads = 1;
    std::string sweep;
    int total_requests = 10000;
    int warmup = 50;
    int device_id = 0;
    std::string aicore_num;
    bool profiling = false;
    std::string profiling_output = "./profiling_data";
    int batch_size = 0;  // 0 = 默认 10

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--sweep" && i + 1 < argc) {
            sweep = argv[++i];
        } else if (arg == "--requests" && i + 1 < argc) {
            total_requests = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--device-id" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else if (arg == "--fixed-seq" && i + 1 < argc) {
            g_seq_fixed = true;
            g_seq_fixed_val = std::stoi(argv[++i]);
        } else if (arg == "--prefix" && i + 1 < argc) {
            // <P> 固定; <P1>-<P2> 每请求均匀随机
            std::string spec = argv[++i];
            auto dash = spec.find('-');
            try {
                if (dash != std::string::npos) {
                    g_prefix_len = std::stoi(spec.substr(0, dash));
                    g_prefix_max = std::stoi(spec.substr(dash + 1));
                } else {
                    g_prefix_len = std::stoi(spec);
                    g_prefix_max = g_prefix_len;
                }
            } catch (const std::exception&) {
                std::cerr << "[ERROR] Invalid --prefix spec: " << spec << std::endl;
                return 1;
            }
            if (g_prefix_len <= 0 || g_prefix_max < g_prefix_len || g_prefix_max > MAX_PREFIX_LEN) {
                std::cerr << "[ERROR] --prefix need 0 < P <= " << MAX_PREFIX_LEN
                          << " and P1 <= P2, got: " << spec << std::endl;
                return 1;
            }
        } else if (arg == "--aicore-num" && i + 1 < argc) {
            aicore_num = argv[++i];
        } else if (arg == "--profiling") {
            profiling = true;
        } else if (arg == "--profiling_output" && i + 1 < argc) {
            profiling_output = argv[++i];
            profiling = true;
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
            if (batch_size < 1 || batch_size > MAX_BATCH_SIZE) {
                std::cerr << "[ERROR] --batch-size need 1.." << MAX_BATCH_SIZE
                          << ", got " << batch_size << std::endl;
                return 1;
            }
            BATCH_FIXED_VAL = batch_size;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<int> thread_counts;
    if (!sweep.empty()) {
        std::stringstream ss(sweep);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            thread_counts.push_back(std::stoi(tok));
        }
    } else {
        thread_counts.push_back(num_threads);
    }
    int max_threads = *std::max_element(thread_counts.begin(), thread_counts.end());

    std::cout << "========================================================================\n";
    std::cout << "GE Benchmark Config: model=" << model_path << ", device=" << device_id
              << ", requests=" << total_requests << ", warmup=" << warmup
              << ", batch=" << BATCH_FIXED_VAL
              << ", fixed_seq=" << (g_seq_fixed ? std::to_string(g_seq_fixed_val) : "random")
              << ", prefix=" << (g_prefix_len > 0
                                     ? (g_prefix_max > g_prefix_len
                                            ? std::to_string(g_prefix_len) + "-" + std::to_string(g_prefix_max) + " (random)"
                                            : std::to_string(g_prefix_len))
                                     : std::string("off"))
              << ", aicore=" << (aicore_num.empty() ? "all" : aicore_num)
              << ", sweep_max_threads=" << max_threads << "\n";
    std::cout << "========================================================================\n";

    // ---- 1. GEInitialize + aclInit ----
    // 限核 {ge::ir_option::AICORE_NUM, "aic|aiv"} 必须挂在 GEInitialize 全局选项:
    // 挂 AddGraph 图选项时 CompileGraph 正常但 LoadGraph 报 GetPlatformInfo failed
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {ge::AscendString("ge.graphRunMode"), ge::AscendString("1")},
        {ge::AscendString("ge.exec.deviceId"), ge::AscendString(std::to_string(device_id).c_str())},
    };
    if (!aicore_num.empty()) {
        std::string spec = aicore_num;
        bool isDigit = !spec.empty() && spec.find_first_not_of("0123456789") == std::string::npos;
        if (isDigit) spec = spec + "|" + std::to_string(std::stoi(spec) * 2);
        globalOptions.emplace(ge::AscendString(ge::ir_option::AICORE_NUM), ge::AscendString(spec.c_str()));
        std::cout << "[INFO] GEInitialize with " << ge::ir_option::AICORE_NUM << "=" << spec << std::endl;
    }
    if (profiling) {
        // GE profiling 经 GEInitialize 全局选项开启, 产出 msprof 可解析的 PROF_* 会话
        // aic_metrics=PipeUtilization: 采集算子级 pipeline (cube/vector/mte 等) 利用率
        mkdir(profiling_output.c_str(), 0755);
        globalOptions.emplace(ge::AscendString(ge::OPTION_EXEC_PROFILING_MODE), ge::AscendString("1"));
        std::string opt = "{\"output\":\"" + profiling_output +
                          "\",\"task_time\":\"on\",\"runtime_api\":\"on\","
                          "\"aic_metrics\":\"PipeUtilization\"}";
        globalOptions.emplace(ge::AscendString(ge::OPTION_EXEC_PROFILING_OPTIONS),
                              ge::AscendString(opt.c_str()));
        std::cout << "[INFO] GE profiling enabled (PipeUtilization) -> " << profiling_output << std::endl;
    }
    GE_CHECK(ge::GEInitialize(globalOptions), "GEInitialize");
    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS && aclRet != ACL_ERROR_REPEAT_INITIALIZE) {
        std::cerr << "[ERROR] aclInit failed, ret=" << aclRet << std::endl;
        return 1;
    }

    std::map<ge::AscendString, ge::AscendString> sessionOptions = {
        {ge::AscendString("ge.session_device_id"), ge::AscendString(std::to_string(device_id).c_str())},
        {ge::AscendString("ge.exec.precision_mode"), ge::AscendString("force_fp16")},
    };
    auto session = std::make_shared<ge::Session>(sessionOptions);
    if (session == nullptr) {
        std::cerr << "[ERROR] Create Session failed" << std::endl;
        return 1;
    }

    // ---- 2. 预编译 max_threads 个图 (串行, sweep 各档复用) ----
    // 注意: CompileGraph 必须在 aclrtSetDevice 之前 (对齐 ge_infer 成功序列:
    // GE 编译期内部线程先建立 context 绑定, 之后再 SetDevice 建默认 context,
    // stream 与 GE executor 的 context 才能一致)
    // 限核已在 GEInitialize 全局设置 (aicoreNum), 图选项无需重复注入
    std::cout << "[INFO] Compiling " << max_threads << " graphs (serial, one per thread)..." << std::endl;
    double compile_total_ms = 0.0;
    for (int i = 0; i < max_threads; i++) {
        auto t0 = Clock::now();
        ge::Graph graph;
        if (graph.LoadFromFile(model_path.c_str()) != ge::GRAPH_SUCCESS) {
            std::cerr << "[ERROR] Graph::LoadFromFile failed (graph " << i << "): " << model_path << std::endl;
            return 1;
        }
        GE_CHECK(session->AddGraph(static_cast<uint32_t>(i + 1), graph), "AddGraph " << i + 1);
        GE_CHECK(session->CompileGraph(static_cast<uint32_t>(i + 1)), "CompileGraph " << i + 1);
        auto t1 = Clock::now();
        double ms = elapsed_ms(t0, t1);
        compile_total_ms += ms;
        std::cout << "[INFO] Graph[" << (i + 1) << "] compiled in " << ms << " ms" << std::endl;
    }
    std::cout << "[INFO] All graphs compiled, total " << compile_total_ms << " ms" << std::endl;

    // ---- 2.1 编译完成后再绑定 device (默认 context), 建 stream + LoadGraph ----
    ACL_CHECK(aclrtSetDevice(device_id), "aclrtSetDevice");
    aclrtContext acl_ctx = nullptr;
    ACL_CHECK(aclrtGetCurrentContext(&acl_ctx), "aclrtGetCurrentContext");
    g_default_ctx = acl_ctx;
    g_device_id = device_id;

    std::vector<std::unique_ptr<GeContext>> ctxs;
    for (int t = 0; t < max_threads; t++) {
        auto ctx = std::make_unique<GeContext>(t, static_cast<uint32_t>(t + 1));
        int r = ctx->init(model_path, session.get());
        if (r != ACL_SUCCESS) {
            std::cerr << "[ERROR] Failed to init GeContext " << t << std::endl;
            return r;
        }
        ctxs.push_back(std::move(ctx));
    }
    std::cout << "[INFO] Created " << max_threads << " graph contexts (each with own graph_id + stream)" << std::endl;

    // ---- 3. sweep ----
    std::vector<BenchResult> results;
    for (int n : thread_counts) {
        std::cout << "\n>>> Running benchmark with " << n << " thread(s)...\n";
        LatencyStats stats;
        double total_ms = 0.0;
        if (run_benchmark(session.get(), ctxs, n, total_requests, warmup, stats, total_ms) != 0) {
            std::cerr << "[ERROR] Benchmark failed at " << n << " threads" << std::endl;
            return 1;
        }
        BenchResult r;
        r.num_threads = n;
        r.total_ms = total_ms;
        r.summary = stats.get_summary(total_ms);
        results.push_back(r);
    }

    // ---- 4. 汇总 ----
    print_sweep_table(results);

    for (auto& ctx : ctxs) ctx->cleanup();
    ctxs.clear();
    session.reset();
    aclrtResetDevice(device_id);
    ge::GEFinalize();
    aclFinalize();
    std::cout << "[INFO] GE benchmark completed" << std::endl;
    return 0;
}
