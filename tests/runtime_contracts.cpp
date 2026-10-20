#include <stdexcept>
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
    resources.setup=[](int){return false;};
    PerfResult unmeasured;
    CHECK(!BenchPool(1,10,0,42,pool,resources,unmeasured));
    CHECK(unmeasured.records.empty());
    std::cout<<"runtime contract checks passed\n";
}
