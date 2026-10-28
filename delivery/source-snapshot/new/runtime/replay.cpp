#include "bench.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <fstream>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>

namespace ge_runtime {
size_t NearestRankIndex(size_t count, double quantile) {
    if (!count || !(quantile > 0 && quantile <= 1)) throw std::invalid_argument("invalid percentile");
    return std::min(count-1, static_cast<size_t>(std::ceil(quantile*count))-1);
}
namespace {
int64_t Ns(Clock::time_point point) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(point.time_since_epoch()).count();
}
void Summarize(const std::vector<double> &values, double &avg, double &lo, double &p50, double &p99, double &hi) {
    if (values.empty()) return;
    auto sorted=values; std::sort(sorted.begin(),sorted.end());
    avg=std::accumulate(sorted.begin(),sorted.end(),0.0)/sorted.size();
    lo=sorted.front(); hi=sorted.back();
    p50=sorted[NearestRankIndex(sorted.size(),.50)]; p99=sorted[NearestRankIndex(sorted.size(),.99)];
}
}
bool ReplayPool(int instances, uint64_t seed, const ReplayOptions &opt,
                const RequestPool &pool, const PoolResources &res, PerfResult &out) {
    if (instances<1 || pool.Empty() || !res.setup || !res.setInputs || !res.execute || !res.finishOutputs ||
        !std::isfinite(opt.minSeconds) || !std::isfinite(opt.maxSeconds) ||
        opt.minSeconds<=0 || opt.maxSeconds<opt.minSeconds || !opt.minRequests ||
        opt.maxRequests<opt.minRequests || opt.maxRequests>10000000 || opt.warmupPerCase<1 ||
        opt.runId.empty() || (opt.captureOutputs && !res.captureRequest)) return false;
    out=PerfResult{}; out.replay=true; out.runId=opt.runId; out.distinctShapes=pool.distinctShapes;
    auto now=opt.now ? opt.now : [](){return Clock::now();};
    std::vector<bool> created(static_cast<size_t>(instances));
    auto release=[&](){for(int t=0;t<instances;++t) if(created[t] && res.release) {res.release(t);created[t]=false;}};
    try {
        out.hbmBaseMb=res.hbmUsedMb ? res.hbmUsedMb() : -1;
        for(int t=0;t<instances;++t) {if(!res.setup(t)){release();return false;}created[t]=true;}
        out.hbmPeakMb=res.hbmUsedMb ? res.hbmUsedMb() : -1;
    } catch(...) {release();throw;}
    Sampler sampler(pool.requests.size(),seed);
    std::vector<size_t> sequence(opt.maxRequests);
    for(auto &index:sequence) index=sampler.Next();
    std::vector<std::vector<RequestRecord>> records(instances),warmRecords(instances);
    std::atomic<bool> stop{false}; std::atomic<size_t> completed{0},errors{0};
    std::mutex barrier,dispatch; std::condition_variable cv;
    size_t ready=0,next=0; bool started=false,warmFailed=false,closed=false;
    Clock::time_point windowBegin,windowEnd,deadline;
    auto warmBegin=now();
    auto run=[&](int tid,const Request &req,RequestRecord &rec,bool capture) {
        PhaseTime phase;
        // Measurement requests are admitted and timestamped under the dispatcher lock.
        // This includes the short dispatch-to-H2D gap and keeps tail requests in their send cohort.
        if(!rec.startedNs) rec.startedNs=Ns(now());
        bool ok=false;
        try {
            rec.errorStage="h2d_desc";
            for(const auto &plan:req.plans) rec.h2dBytes+=plan.Bytes();
            ok=res.setInputs(tid,req,phase); rec.h2dMs=phase.h2dMs;rec.descMs=phase.descMs;
            if(ok) {
                rec.errorStage="execute_sync";auto execBegin=now();ok=res.execute(tid);
                rec.execMs=std::chrono::duration<double,std::milli>(now()-execBegin).count();
            }
            if(ok) {
                rec.errorStage="d2h";auto copyBegin=now();ok=res.finishOutputs(tid,req,rec);
                rec.d2hMs=std::chrono::duration<double,std::milli>(now()-copyBegin).count();
            }
        } catch(const std::exception &e) {fprintf(stderr,"[ERROR] replay instance=%d stage=%s: %s\n",tid,rec.errorStage.c_str(),e.what());ok=false;}
          catch(...) {ok=false;}
        rec.completedNs=Ns(now());rec.e2eMs=(rec.completedNs-rec.startedNs)/1e6;
        if(ok && capture) {
            rec.errorStage="capture";
            try {ok=res.captureRequest(tid,req,rec);}catch(...) {ok=false;}
        }
        rec.error=ok ? 0 : 1;
        if(ok) rec.errorStage.clear(); else {++errors;stop=true;}
        return ok;
    };
    std::vector<std::thread> workers;
    try {
        for(int tid=0;tid<instances;++tid) workers.emplace_back([&,tid](){
            bool warmOk=true;
            try {
                if(res.threadEnter) res.threadEnter(tid);
                for(int round=0;round<opt.warmupPerCase && warmOk;++round) for(size_t i=0;i<pool.requests.size();++i) {
                    const auto &req=pool.requests[i];RequestRecord rec;
                    rec.instance=tid;rec.req=static_cast<size_t>(round)*pool.requests.size()+i;
                    rec.inputSet=req.name;rec.shapeKey=req.shapeKey;rec.samples=req.samples;rec.phase="warmup";
                    warmOk=run(tid,req,rec,false);warmRecords[tid].push_back(std::move(rec));
                    if(!warmOk) break;
                }
            }catch(...) {warmOk=false;stop=true;++errors;}
            {
                std::unique_lock<std::mutex> lock(barrier);warmFailed=warmFailed||!warmOk;++ready;cv.notify_all();
                cv.wait(lock,[&](){return started;});if(warmFailed) return;
            }
            while(!stop.load()) {
                size_t index=0;int64_t admittedNs=0;
                {
                    std::lock_guard<std::mutex> lock(dispatch);auto time=now();
                    if(closed || stop.load()) break;
                    double elapsed=std::chrono::duration<double>(time-windowBegin).count();
                    if(time>=deadline || (elapsed>=opt.minSeconds && completed.load()>=opt.minRequests) || next>=sequence.size()) {
                        closed=true;windowEnd=std::min(time,deadline);break;
                    }
                    index=next++;admittedNs=Ns(time);
                }
                const auto &req=pool.requests[sequence[index]];RequestRecord rec;
                rec.instance=tid;rec.req=index;rec.startedNs=admittedNs;rec.inputSet=req.name;rec.shapeKey=req.shapeKey;rec.samples=req.samples;
                bool ok=run(tid,req,rec,opt.captureOutputs);
                records[tid].push_back(std::move(rec));if(ok) ++completed;
            }
        });
    }catch(...) {
        {std::lock_guard<std::mutex> lock(barrier);warmFailed=true;started=true;}
        stop=true;cv.notify_all();for(auto &worker:workers) worker.join();release();throw;
    }
    {
        std::unique_lock<std::mutex> lock(barrier);cv.wait(lock,[&](){return ready==static_cast<size_t>(instances);});
        out.warmupMs=std::chrono::duration<double,std::milli>(now()-warmBegin).count();
        windowBegin=now();windowEnd=windowBegin;
        deadline=windowBegin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(opt.maxSeconds));started=true;
    }
    cv.notify_all();for(auto &worker:workers) worker.join();
    if(!closed) windowEnd=std::min(now(),deadline);
    out.windowStartNs=Ns(windowBegin);out.windowEndNs=Ns(windowEnd);
    out.wallMs=std::max(0.0,(out.windowEndNs-out.windowStartNs)/1e6);
    std::vector<double> latency,exec,d2h,samples;
    for(int tid=0;tid<instances;++tid) {
        for(auto &rec:warmRecords[tid]) out.warmupRecords.push_back(std::move(rec));
        InstancePerf ip;ip.name="instance_"+std::to_string(tid);ip.loadMs=res.loadMs ? res.loadMs(tid) : 0;
        std::vector<double> local,localExec;std::set<std::string> shapes;
        for(const auto &rec:records[tid]) {
            out.records.push_back(rec);shapes.insert(rec.shapeKey);
            out.lastCompletedNs=std::max(out.lastCompletedNs,rec.completedNs);
            if(rec.startedNs<=out.windowEndNs && rec.completedNs>out.windowEndNs) ++out.inFlightAtEnd;
            if(rec.error) {++ip.errors;continue;}
            if(rec.completedNs<=out.windowEndNs) {++out.windowSuccesses;ip.qps+=1;}
            ++ip.requests;out.samples+=rec.samples;
            latency.push_back(rec.e2eMs);exec.push_back(rec.execMs);d2h.push_back(rec.d2hMs);samples.push_back(rec.samples);
            local.push_back(rec.e2eMs);localExec.push_back(rec.execMs);
            ip.h2dAvgMs+=rec.h2dMs;ip.descAvgMs+=rec.descMs;
        }
        Summarize(local,ip.avgMs,ip.minMs,ip.p50Ms,ip.p99Ms,ip.maxMs);
        double unused=0;Summarize(localExec,ip.execAvgMs,unused,unused,ip.execP99Ms,unused);
        if(ip.requests){ip.h2dAvgMs/=ip.requests;ip.descAvgMs/=ip.requests;}
        ip.qps=out.wallMs>0 ? ip.qps/(out.wallMs/1000) : 0;ip.distinctShapes=shapes.size();out.instances.push_back(ip);
    }
    out.warmupRuns=out.warmupRecords.size();out.requests=latency.size();out.errors=errors.load();
    out.qps=out.wallMs>0 ? out.windowSuccesses/(out.wallMs/1000) : 0;
    out.drainMs=std::max(0.0,(out.lastCompletedNs-out.windowEndNs)/1e6);
    out.statisticsSufficient=!warmFailed && !out.errors && out.wallMs/1000>=opt.minSeconds && out.windowSuccesses>=opt.minRequests;
    Summarize(latency,out.avgMs,out.minMs,out.p50Ms,out.p99Ms,out.maxMs);
    double unused=0;Summarize(exec,out.execAvgMs,unused,unused,out.execP99Ms,unused);
    Summarize(samples,out.tokAvg,out.tokMin,out.tokP50,out.tokP99,out.tokMax);
    out.d2hAvgMs=d2h.empty()?0:std::accumulate(d2h.begin(),d2h.end(),0.0)/d2h.size();
    std::sort(out.records.begin(),out.records.end(),[](const auto &a,const auto &b){return a.req<b.req;});
    release();return !warmFailed && !out.errors;
}
bool WriteRequestEvents(const std::string &path,const PerfResult &perf) {
    std::ofstream file(path);if(!file) return false;
    auto write=[&](const RequestRecord &r){
        nlohmann::json event={{"run_id",perf.runId},{"request_id",r.phase=="warmup" ? nlohmann::json("warmup-"+std::to_string(r.instance)+"-"+std::to_string(r.req)) : nlohmann::json(r.req)},{"instance",r.instance},{"case_id",r.inputSet},
            {"shape_key",r.shapeKey},{"phase",r.phase},{"status",r.error?"failed":"success"},
            {"error_stage",r.errorStage.empty()?nlohmann::json(nullptr):nlohmann::json(r.errorStage)},
            {"started_ns",r.startedNs},{"scheduled_ns",r.startedNs},{"sent_ns",r.startedNs},{"received_ns",r.startedNs},
            {"completed_ns",r.completedNs},{"h2d_ms",r.h2dMs},{"desc_ms",r.descMs},{"execute_ms",r.execMs},
            {"d2h_ms",r.d2hMs},{"e2e_ms",r.e2eMs},{"h2d_bytes",r.h2dBytes},{"d2h_bytes",r.d2hBytes}};
        file<<event.dump()<<'\n';
    };
    for(const auto &record:perf.warmupRecords) write(record);
    for(const auto &record:perf.records) write(record);
    return static_cast<bool>(file);
}
} // namespace ge_runtime
