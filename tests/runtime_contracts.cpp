#include <stdexcept>
#include <atomic>
#include <thread>
#include <fstream>
#include <iostream>
#include "io_spec.h"
#include "bench.h"
using namespace ge_runtime;
#define CHECK(condition) do { if(!(condition)) throw std::runtime_error("check failed: " #condition); } while(false)

int main(int argc,char **argv) {
    CHECK(argc==2);
    std::string dir=argv[1];MakeDirs(dir);
    IoNode a;a.logical="encoder/output:0";a.dtype="float32";a.shape={-1};a.dynamic_dims={0};
    CHECK(SameDtype("float", "float32") && SameDtype("double", "float64"));
    CHECK(SameDtype("int64", "int64") && !SameDtype("float16", "float32"));
    CHECK(!SameDtype("unknown", "unknown"));
    {
        IoSpec packedSpec;IoNode input;input.logical="features";input.node="graph_data_2";
        input.dtype="float32";input.shape={-1,2};packedSpec.inputs={input};packedSpec.outputs={a};
        float values[]={1,2,3,4,5,6};
        {std::ofstream file(dir+"/packed.bin",std::ios::binary);file.write(reinterpret_cast<const char*>(values),sizeof(values));CHECK(file.good());}
        nlohmann::json inputs={{"file",dir+"/packed.bin"},{"cases",nlohmann::json::array({
            {{"id","first"},{"samples",1},{"tensors",nlohmann::json::array({
                {{"name","features"},{"dtype","float32"},{"shape",{1,2}},{"offset",0},{"bytes",8}}
            })},{"output_capacities",{{a.logical,8}}}},
            {{"id","second"},{"samples",2},{"tensors",nlohmann::json::array({
                {{"name","features"},{"dtype","float32"},{"shape",{2,2}},{"offset",8},{"bytes",16}}
            })},{"output_capacities",{{a.logical,16}}}}
        })}};
        auto packed=LoadPool(packedSpec,inputs);
        CHECK(packed.requests.size()==2 && packed.distinctShapes==2);
        CHECK(packed.maxBytes[0]==16 && packed.maxOutputBytes[0]==16);
        CHECK(packed.requests[0].plans[0].node=="graph_data_2");
        CHECK(packed.requests[1].data[0].size()==16);
        auto invalid=inputs;invalid["cases"][1]["tensors"][0]["offset"]=999;
        bool rejected=false;try{LoadPool(packedSpec,invalid);}catch(...){rejected=true;}
        CHECK(rejected);
        invalid=inputs;invalid["cases"][0]["tensors"][0]["name"]="wrong";
        rejected=false;try{LoadPool(packedSpec,invalid);}catch(...){rejected=true;}
        CHECK(rejected);
    }
    RequestPool pool;pool.distinctShapes=1;
    for(int i=0;i<100;i++) { Request r;r.name="case-"+std::to_string(i);r.shapeKey="fixed";r.samples=2;pool.requests.push_back(r); }
    PoolResources resources;
    resources.setup=[](int){return true;};
    resources.setInputs=[](int,const Request &,PhaseTime &){return true;};
    resources.execute=[](int){return true;};
    resources.release=[](int){};
    PerfResult one,two,again,different;
    CHECK(BenchPool(1,1003,2,42,pool,resources,one));
    CHECK(BenchPool(2,1003,5,42,pool,resources,two));
    CHECK(BenchPool(2,1003,0,42,pool,resources,again));
    CHECK(BenchPool(2,1003,0,43,pool,resources,different));
    CHECK(two.requests==1003 && two.samples==2006 && two.errors==0);
    CHECK(two.instances[0].requests==502 && two.instances[1].requests==501);
    bool changed=false;
    for(size_t i=0;i<one.records.size();i++) {
        CHECK(one.records[i].req==i && two.records[i].req==i);
        CHECK(one.records[i].inputSet==two.records[i].inputSet);
        CHECK(two.records[i].inputSet==again.records[i].inputSet);
        changed=changed || two.records[i].inputSet!=different.records[i].inputSet;
    }
    CHECK(changed);
    std::vector<std::vector<std::string>> captures(2);
    CHECK(CapturePool(2,pool,resources,[&](int tid,const Request &request) { captures[tid].push_back(request.name);return true; }));
    CHECK(captures[0].size()==100 && captures[0]==captures[1]);
    CHECK(!CapturePool(2,pool,resources,[](int,const Request &){return false;}));
    // Warmup succeeds, then the fourth measured call fails. Keep all completed records.
    size_t calls=0,released=0;
    resources.execute=[&](int){return ++calls!=5;};
    resources.release=[&](int){++released;};
    PerfResult partial;
    CHECK(!BenchPool(1,10,0,42,pool,resources,partial));
    CHECK(partial.requests==3 && partial.samples==6 && partial.errors==1);
    CHECK(partial.records.size()==4 && partial.records.back().error==1 && released==1);
    CHECK(PerfToJson(partial)["errors"]==1);
    CHECK(WriteRequestsCsv(dir+"/partial.csv",partial));
    {std::ifstream file(dir+"/partial.csv");size_t lines=0;std::string line;while(std::getline(file,line))++lines;CHECK(lines==5);}
    // Warmup and measurement must share each instance's entered worker thread.
    // A slow worker must finish warming before any worker measures a request.
    {
        PoolResources sameWorker;
        std::thread::id owners[2];
        std::atomic<size_t> completed[2]{};
        size_t releasedWorkers=0;
        sameWorker.setup=[](int){return true;};
        sameWorker.threadEnter=[&](int tid){owners[tid]=std::this_thread::get_id();};
        sameWorker.setInputs=[&](int tid,const Request &,PhaseTime &) {
            if(owners[tid]!=std::this_thread::get_id()) return false;
            if(completed[tid].load()>=5 && completed[1-tid].load()<5) return false;
            return true;
        };
        sameWorker.execute=[&](int tid) {
            if(owners[tid]!=std::this_thread::get_id()) return false;
            if(tid==1 && completed[tid].load()==0)
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            ++completed[tid];
            return true;
        };
        sameWorker.release=[&](int){++releasedWorkers;};
        PerfResult warmed;
        CHECK(BenchPool(2,7,5,42,pool,sameWorker,warmed));
        CHECK(warmed.warmupRuns==10 && warmed.requests==7 && warmed.errors==0);
        CHECK(completed[0]==9 && completed[1]==8 && releasedWorkers==2);
        for(const auto &rec:warmed.records) CHECK(!rec.firstHit);
    }
    // One worker's warmup failure must release all waiting workers and resources.
    {
        PoolResources failWarm;
        size_t releasedWorkers=0;
        failWarm.setup=[](int){return true;};
        failWarm.setInputs=[](int,const Request &,PhaseTime &){return true;};
        failWarm.execute=[](int tid){return tid!=0;};
        failWarm.release=[&](int){++releasedWorkers;};
        PerfResult failedWarm;
        CHECK(!BenchPool(2,7,5,42,pool,failWarm,failedWarm));
        CHECK(failedWarm.records.empty() && releasedWorkers==2);
        releasedWorkers=0;
        failWarm.threadEnter=[](int tid){if(tid==0) throw std::runtime_error("thread enter failure");};
        failWarm.execute=[](int){return true;};
        PerfResult failedEnter;
        CHECK(!BenchPool(2,7,5,42,pool,failWarm,failedEnter));
        CHECK(failedEnter.records.empty() && releasedWorkers==2);
    }
    resources.setup=[](int){return false;};
    PerfResult unmeasured;
    CHECK(!BenchPool(1,10,0,42,pool,resources,unmeasured));
    CHECK(unmeasured.records.empty());
    // Nearest rank must not move p99 up one sample at exact hundredths.
    CHECK(NearestRankIndex(100,.99)==98 && NearestRankIndex(1000,.99)==989);
    CHECK(NearestRankIndex(1,.99)==0);
    {
        RequestPool cases;cases.distinctShapes=2;
        for(int i=0;i<5;++i){Request req;req.name="sample-"+std::to_string(i);req.shapeKey=std::to_string(i%2);req.samples=1;cases.requests.push_back(req);}
        PoolResources fake;fake.setup=[](int){return true;};fake.release=[](int){};
        fake.setInputs=[](int,const Request &,PhaseTime &){return true;};fake.execute=[](int){return true;};
        std::atomic<int64_t> ticks{0};size_t finished=0,captured=0;
        ReplayOptions options;options.runId="cpu";options.enabled=true;options.minSeconds=.010;options.maxSeconds=.100;
        options.minRequests=3;options.maxRequests=100;options.warmupPerCase=1;options.captureOutputs=true;
        options.now=[&](){return Clock::time_point(std::chrono::milliseconds(ticks.fetch_add(1)));};
        fake.finishOutputs=[&](int,const Request &,RequestRecord &r){++finished;r.d2hBytes=40448;ticks.fetch_add(7);return true;};
        fake.captureRequest=[&](int,const Request &req,const RequestRecord &r){
            CHECK(r.inputSet==req.name && r.completedNs>r.startedNs && r.d2hBytes==40448);++captured;return true;
        };
        PerfResult replay;CHECK(ReplayPool(1,42,options,cases,fake,replay));
        CHECK(replay.statisticsSufficient && replay.warmupRuns==5 && replay.requests>=3);
        CHECK(finished==replay.requests+5 && captured==replay.requests);
        CHECK(replay.records.size()==replay.requests && replay.windowSuccesses<=replay.requests);
        for(size_t i=0;i<replay.records.size();++i){const auto &r=replay.records[i];CHECK(r.req==i && r.d2hMs>=7 && r.e2eMs>=r.d2hMs && !r.error);}
        CHECK(PerfToJson(replay)["latency_scope"]=="h2d_desc_execute_sync_d2h");
        CHECK(WriteRequestEvents(dir+"/events.jsonl",replay));
        {std::ifstream f(dir+"/events.jsonl");std::string line;size_t lines=0;while(std::getline(f,line)){auto e=nlohmann::json::parse(line);CHECK(e["d2h_bytes"]==40448);++lines;}CHECK(lines==replay.requests+5);}
        // Multiple independent workers must retain case/request/output association, including equal shapes.
        {
            ReplayOptions parallel=options;parallel.now={};parallel.minSeconds=.001;parallel.maxSeconds=1;
            parallel.minRequests=100;parallel.maxRequests=200;
            PoolResources isolated;std::string current[2];size_t captures[2]={0,0};
            isolated.setup=[](int){return true;};isolated.release=[](int){};
            isolated.setInputs=[&](int tid,const Request &req,PhaseTime &){current[tid]=req.name;return true;};
            isolated.execute=[](int){return true;};
            isolated.finishOutputs=[&](int tid,const Request &req,RequestRecord &r){CHECK(current[tid]==req.name);r.d2hBytes=40448;return true;};
            isolated.captureRequest=[&](int tid,const Request &req,const RequestRecord &r){CHECK(current[tid]==req.name && r.inputSet==req.name && r.instance==tid);++captures[tid];return true;};
            PerfResult concurrent;CHECK(ReplayPool(2,42,parallel,cases,isolated,concurrent));
            CHECK(concurrent.requests>=100 && concurrent.requests<=200 && captures[0]+captures[1]==concurrent.requests);
            Sampler expected(cases.requests.size(),42);
            for(size_t i=0;i<concurrent.records.size();++i){CHECK(concurrent.records[i].req==i);CHECK(concurrent.records[i].inputSet==cases.requests[expected.Next()].name);}
        }
        // A copy failure must retain its stage/terminal record, stop dispatch, and never call capture.
        captured=0;finished=0;ticks=0;
        fake.finishOutputs=[&](int,const Request &,RequestRecord &r){r.d2hBytes=40448;return ++finished!=7;};
        PerfResult failed;CHECK(!ReplayPool(1,42,options,cases,fake,failed));
        CHECK(failed.requests==1 && failed.errors==1 && failed.records.size()==2 && captured==1);
        CHECK(failed.records.back().error && failed.records.back().errorStage=="d2h");
        CHECK(!failed.statisticsSufficient);
        // A bounded window drains a slow request but does not count its late completion as window QPS.
        ticks=0;options.minRequests=1;options.maxRequests=50;options.maxSeconds=.025;options.minSeconds=.010;options.captureOutputs=false;
        fake.finishOutputs=[&](int,const Request &,RequestRecord &r){r.d2hBytes=40448;ticks.fetch_add(20);return true;};
        PerfResult limited;CHECK(ReplayPool(1,42,options,cases,fake,limited));
        CHECK(!limited.statisticsSufficient && limited.windowSuccesses==0 && limited.requests==1 && limited.inFlightAtEnd==1 && limited.drainMs>0);
        CHECK(PerfToJson(limited)["samples_per_second"]==0 && PerfToJson(limited)["cohort_qps"].get<double>()>0);
        // Sink failures stay terminal failures, rather than a successful D2H turning into a claimed successful response.
        ticks=0;options.captureOutputs=true;options.minRequests=1;options.maxRequests=10;
        fake.captureRequest=[](int,const Request &,const RequestRecord &){throw std::runtime_error("sink failure");return true;};
        PerfResult sinkFailed;CHECK(!ReplayPool(1,42,options,cases,fake,sinkFailed));
        CHECK(sinkFailed.errors==1 && sinkFailed.records.size()==1 && sinkFailed.records[0].errorStage=="capture");
    }
    std::cout<<"runtime contract checks passed\n";
}
