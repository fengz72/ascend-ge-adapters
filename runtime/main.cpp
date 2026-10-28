#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include "io_spec.h"
#include "pool.h"
#include <graph/attr_value.h>
#include <graph/graph.h>
#include "bench.h"
#include "acl_json.h"
#include "backends/acl_backend.h"
#include "backends/gesession_backend.h"

using Json=nlohmann::json;
using namespace ge_runtime;

int main(int argc,char **argv) {
    try {
        if(argc!=1) throw std::runtime_error("ge_runtime takes a JSON execution request on stdin");
        Json plan=Json::parse(std::cin);
        std::string mode=plan.at("mode").get<std::string>();
        if(mode=="write_io" || mode=="read_io") {
            std::string path=plan.at("graph").at("path").get<std::string>();
            ge::Graph graph;
            if(graph.LoadFromFile(path.c_str())!=ge::GRAPH_SUCCESS) throw std::runtime_error("cannot load AIR: "+path);
            ge::AttrValue value;
            Json io;
            if(mode=="write_io") {
                io=plan.at("io");IoSpec::FromJson(io);
                std::string body=io.dump();
                if(value.SetAttrValue(ge::AscendString(body.c_str()))!=ge::GRAPH_SUCCESS ||
                   graph.SetAttr(ge::AscendString("_ge_adapters_io"),value)!=ge::GRAPH_SUCCESS ||
                   graph.SaveToFile(path.c_str())!=ge::GRAPH_SUCCESS) throw std::runtime_error("cannot embed IO in AIR: "+path);
            } else {
                ge::AscendString body;
                if(graph.GetAttr(ge::AscendString("_ge_adapters_io"),value)!=ge::GRAPH_SUCCESS ||
                   value.GetAttrValue(body)!=ge::GRAPH_SUCCESS) throw std::runtime_error("AIR has no embedded IO; run build: "+path);
                io=Json::parse(body.GetString());IoSpec::FromJson(io);
            }
            std::cout<<"GE_RESULT "<<Json({{"io",io}}).dump()<<std::endl;return 0;
        }
        std::string output=plan.at("output_dir").get<std::string>();MakeDirs(output);
        if(mode!="infer" && mode!="bench") throw std::runtime_error("unknown mode");
        ModelGraph graph;graph.backend=plan.at("backend").get<std::string>();
        graph.graph_path=plan.at("graph").at("path").get<std::string>();
        graph.graph_format=plan.at("graph").at("format").get<std::string>();
        if(!plan.at("graph").at("om_path").is_null()) graph.om_path=plan.at("graph").at("om_path").get<std::string>();
        graph.device=plan.at("device").get<int>();
        IoSpec io=IoSpec::FromJson(plan.at("io"));
        RequestPool pool=LoadPool(io,plan.at("inputs"));
        BenchOptions bench;bench.threads=plan.at("instances").get<int>();
        bench.warmup=mode=="infer"?0:plan.at("warmup").get<int>();
        bench.requests=plan.at("iterations").get<int>();
        if(bench.threads<1 || bench.requests<1 || bench.warmup<0) throw std::runtime_error("invalid execution counts");
        if(plan.contains("replay")) {
            if(mode!="bench" || graph.backend!="ge_session") throw std::runtime_error("replay requires GE Session bench");
            const auto &replay=plan.at("replay"); bench.replay.enabled=true;
            bench.replay.runId=replay.at("run_id").get<std::string>();
            bench.replay.minSeconds=replay.at("min_seconds").get<double>();
            bench.replay.maxSeconds=replay.at("max_seconds").get<double>();
            bench.replay.minRequests=replay.at("min_requests").get<size_t>();
            bench.replay.maxRequests=replay.value("max_requests",size_t{2000000});
            bench.replay.warmupPerCase=replay.value("warmup_per_case",10);
            bench.replay.captureOutputs=replay.value("capture_outputs",false);
            if(bench.replay.runId.empty() || !std::isfinite(bench.replay.minSeconds) || !std::isfinite(bench.replay.maxSeconds) ||
                bench.replay.minSeconds<=0 || bench.replay.maxSeconds<bench.replay.minSeconds || !bench.replay.minRequests ||
                bench.replay.maxRequests<bench.replay.minRequests || bench.replay.maxRequests>10000000 || bench.replay.warmupPerCase<1)
                throw std::runtime_error("invalid replay timing/counts");
        }
        const auto &options=plan.at("options");const auto &observe=plan.at("observability");
        bool profile=observe.value("profiling",false),dump=observe.value("dump",false);
        if(profile&&dump) throw std::runtime_error("collect profiling and dump separately");
        if(graph.backend=="ge_session" && dump) throw std::runtime_error("GE Session dump is unsupported");
        Json captured=Json::object();std::mutex captureMutex;std::ofstream binary;
        std::ofstream manifest;
        if(mode!="bench" || bench.replay.captureOutputs) {
            binary.open(output+"/outputs.bin",std::ios::binary);
            if(!binary) throw std::runtime_error("cannot open output binary");
        }
        if(bench.replay.captureOutputs) {
            manifest.open(output+"/manifest.jsonl");
            if(!manifest) throw std::runtime_error("cannot open capture manifest");
        }
        auto replayCapture=[&](int instance,const Request &request,const RequestRecord &record,const std::vector<HostTensor> &tensors) {
            std::lock_guard<std::mutex> guard(captureMutex);
            if(tensors.size()!=io.outputs.size() || !binary || !manifest) return false;
            Json row={{"run_id",bench.replay.runId},{"request_id",record.req},{"instance",instance},
                {"case_id",request.name},{"phase","measure"},{"tensors",Json::array()}};
            for(const auto &tensor:tensors) {
                auto offset=binary.tellp();if(offset<0) return false;
                binary.write(tensor.data.data(),tensor.data.size());
                row["tensors"].push_back({{"name",tensor.logical},{"dtype",tensor.dtype},{"shape",tensor.shape},
                    {"offset",static_cast<size_t>(offset)},{"bytes",tensor.data.size()}});
            }
            binary.flush();if(!binary) return false;
            manifest<<row.dump()<<'\n';manifest.flush();return static_cast<bool>(manifest);
        };
        auto capture=[&](int instance,const Request &request,const std::vector<HostTensor> &tensors) {
            std::lock_guard<std::mutex> guard(captureMutex);
            if(tensors.size()!=io.outputs.size() || !binary) return false;
            Json row={{"instance",instance},{"tensors",Json::array()}};
            for(const auto &tensor:tensors) {
                size_t offset=static_cast<size_t>(binary.tellp());
                binary.write(tensor.data.data(),tensor.data.size());
                row["tensors"].push_back({{"name",tensor.logical},{"dtype",tensor.dtype},{"shape",tensor.shape},{"offset",offset},{"bytes",tensor.data.size()}});
            }
            if(!binary) return false;
            if(!captured.contains(request.name)) captured[request.name]=Json::array();
            captured[request.name].push_back(row);return true;
        };
        PerfResult perf;bool ok=false;
        if(graph.backend=="ge_session") {
            GeSessionOptions opt;opt.device=graph.device;opt.bench=bench;opt.pool=&pool;
            opt.seed=plan.at("seed").get<uint64_t>();
            opt.graph_run_mode=options.value("graph_run_mode",opt.graph_run_mode);
            opt.precision_mode=options.value("precision_mode",std::string("force_fp16"));
            opt.aicore_num=options.value("aicore_num",std::string());
            opt.profiling.enabled=profile;opt.profiling.outputPath=output+"/"+mode+"-profiling";
            opt.profiling.aicMetrics=observe.value("aic_metrics",std::string("PipeUtilization"));
            if(mode!="bench") opt.capture=capture;
            if(bench.replay.captureOutputs) opt.replayCapture=replayCapture;
            ok=RunGeSessionBackend(graph,io,opt,&perf);
        } else if(graph.backend=="atc_om") {
            AclOptions opt;opt.device=graph.device;opt.bench=bench;opt.pool=&pool;opt.seed=plan.at("seed").get<uint64_t>();
            if(mode!="bench") opt.capture=capture;
            if(profile||dump) {
                DumpConfig d;ProfilingConfig p;d.enabled=dump;d.dumpPath=output+"/dump";
                p.enabled=profile;p.outputPath=output+"/"+mode+"-profiling";p.aicMetrics=observe.value("aic_metrics","");
                MakeDirs(d.dumpPath);MakeDirs(p.outputPath);opt.aclConfigPath=output+"/acl.json";
                if(!GenerateAclJson(d,p,opt.aclConfigPath)) throw std::runtime_error("cannot create ACL profiler options");
            }
            ok=RunAclBackend(graph,io,opt,&perf);
        } else throw std::runtime_error("unknown backend");
        if(!ok && (mode!="bench" || (perf.records.empty() && perf.warmupRecords.empty()))) throw std::runtime_error("backend execution failed");
        Json result={{"status",ok?"success":"failed"},{"mode",mode},{"backend",graph.backend}};
        if(!ok) result["error"]="backend execution failed; partial measurements retained";
        if(mode=="bench") {
            binary.close();manifest.close();
            result["performance"]=PerfToJson(perf);
            if(bench.replay.enabled) {
                result["replay"]=plan.at("replay");
                if(!WriteRequestEvents(output+"/events.jsonl",perf)) throw std::runtime_error("cannot save replay events");
                result["events"]=output+"/events.jsonl";
                if(bench.replay.captureOutputs) result["manifest"]=output+"/manifest.jsonl";
            }
            result["performance"]["profiling_enabled"]=profile;
            result["performance"]["capture_enabled"]=bench.replay.captureOutputs;
            result["performance"]["sampling"]={{"seed",plan.at("seed")},{"replacement",true},{"pool_size",pool.requests.size()},{"order","global_request_index"}};
            if(!WriteRequestsCsv(output+"/requests.csv",perf)) throw std::runtime_error("cannot save request metrics");
        } else {binary.close();result["outputs"]=captured;}
        if(profile) result["profiling"]=output+"/"+mode+"-profiling";
        if(dump) result["dump"]=output+"/dump";
        std::cout<<"GE_RESULT "<<result.dump()<<std::endl;return ok?0:1;
    } catch(const std::exception &error) {std::cerr<<error.what()<<std::endl;return 1;}
}
