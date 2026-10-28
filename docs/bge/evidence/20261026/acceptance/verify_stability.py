#!/usr/bin/env python3
"""Independent 10-minute stability/resource and per-process lifecycle evidence."""
import argparse,datetime,json,math,re,statistics
from pathlib import Path
import verify_outputs as v
ROOT=Path(__file__).resolve().parent.parent
def lifecycle(names):
    out={}
    task_ids=set()
    for name in names:
        sp=ROOT/"jobs"/name/"status.json";status=json.loads(sp.read_text())
        assert status["state"]=="finished" and status["exit_code"]==0 and status["task_returncode"]==0
        assert status["reason"]=="exited" and status["process_group_stopped"]
        task_ids.add(status["task_id"])
        identity_path=ROOT/(name+"-identity.json");identity=json.loads(identity_path.read_text())
        assert identity["device"]==10 and identity["pci"]=="0000:89:00.0"
        assert identity["binary_sha256"]=="8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d"
        plan=json.loads(Path(identity["plan"]).read_text())
        assert identity["plan_sha256"]==v.sha(identity["plan"]) and plan["device"]==10
        bp=ROOT/(name+"-before.json");ap=ROOT/(name+"-after.json")
        before=json.loads(bp.read_text());after=json.loads(ap.read_text())
        for gate in [before,after]:
            assert gate["code"]==0 and gate["stdout"].strip().lower()=="no process in device."
        parse=datetime.datetime.fromisoformat
        assert parse(before["at"])<=parse(status["started_at"])<=parse(status["finished_at"])<=parse(after["at"])
        result=json.loads((ROOT/name/"result.json").read_text())
        assert result["returncode"]==0 and result["runtime"]["status"]=="success"
        out[name]={"passed":True,"task_id":status["task_id"],"task_pid":status["task"]["pid"],
          "started_at":status["started_at"],"finished_at":status["finished_at"],
          "outer_exit_code":status["exit_code"],"runtime_returncode":result["returncode"],
          "process_group_stopped":status["process_group_stopped"],"before_device_empty":True,"after_device_empty":True,
          "status_sha256":v.sha(sp),"before_gate_sha256":v.sha(bp),"after_gate_sha256":v.sha(ap),
          "identity_sha256":v.sha(identity_path),"plan_sha256":identity["plan_sha256"]}
    assert len(task_ids)==len(names),"lifecycle task IDs repeated"
    return {"passed":True,"independent_process_cycles":len(task_ids),"runs":out}
def slope_per_minute(points):
    xs=[x[0] for x in points];ys=[x[1] for x in points];xm=statistics.mean(xs);ym=statistics.mean(ys)
    denom=sum((x-xm)**2 for x in xs)
    return 60*sum((x-xm)*(y-ym) for x,y in points)/denom if denom else 0
def stability(name):
    folder=ROOT/name;result=json.loads((folder/"result.json").read_text());perf=result["runtime"]["performance"]
    assert not perf["capture_enabled"],"stability performance must not capture every output"
    check=v.perf_check(folder/"result.json",folder/"events.jsonl",len(perf["instances"]),600,1,10)
    assert check["passed"],check
    rows=v.read_jsonl(folder/"resources.jsonl");start=perf["window_start_ns"];end=perf["window_end_ns"]
    duration=(end-start)/1e9;data=[]
    for row in rows:
        if not start<=row["monotonic_ns"]<=end:continue
        assert row["npu_smi_code"]==0
        matches=[line for line in row["npu_smi_stdout"].splitlines() if "0000:89:00.0" in line]
        assert len(matches)==1
        hbm=re.search(r"(\d+)\s*/\s*65536",matches[0]);assert hbm
        pids={int(m.group(1)) for line in row["npu_smi_stdout"].splitlines()
              if (m:=re.match(r"^\|\s*5\s+0\s*\|\s*(\d+)\s*\|",line))}
        data.append({"seconds":(row["monotonic_ns"]-start)/1e9,"rss_kib":row["rss_kib"],"hbm_mb":int(hbm.group(1)),"host_pids":sorted(pids)})
    assert len(data)>=12,"too few in-window resource samples"
    head=[r for r in data if r["seconds"]<=120];tail=[r for r in data if r["seconds"]>=duration-120]
    assert len(head)>=3 and len(tail)>=3,"missing first/last 2-minute resource evidence"
    gaps=[b["seconds"]-a["seconds"] for a,b in zip(data,data[1:])]
    assert all(g>0 for g in gaps)
    metrics={}
    for key,floor,fraction,unit in [("rss_kib",128*1024,.05,"KiB"),("hbm_mb",64,.01,"MB")]:
        initial=statistics.median(r[key] for r in head);final=statistics.median(r[key] for r in tail)
        limit=max(floor,initial*fraction);increment=final-initial
        steps=[b[key]-a[key] for a,b in zip(data,data[1:])]
        metrics[key]={"unit":unit,"initial_2min_median":initial,"final_2min_median":final,
          "increment":increment,"allowed_increment":limit,"within_increment_limit":increment<=limit,
          "min":min(r[key] for r in data),"max":max(r[key] for r in data),
          "linear_slope_per_minute":slope_per_minute([(r["seconds"],r[key]) for r in data]),
          "positive_steps":sum(x>0 for x in steps),"negative_steps":sum(x<0 for x in steps),"zero_steps":sum(x==0 for x in steps)}
    events=[e for e in v.read_jsonl(folder/"events.jsonl") if e["phase"]=="measure"]
    buckets={}
    for i in range(math.ceil(duration/10)):
        seconds=min(10,duration-10*i)
        if seconds>0:buckets[i]={"seconds":seconds,"successes":0}
    for e in events:
        if e["completed_ns"]<=end:buckets[(e["completed_ns"]-start)//10_000_000_000]["successes"]+=1
    for i,row in buckets.items():row["start_seconds"]=10*i;row["qps"]=row["successes"]/row["seconds"]
    cycles=lifecycle([name])
    return {"schema":"bge-independent-stability/1","passed_numeric_contract":all(x["within_increment_limit"] for x in metrics.values()),
      "trend_assessment":"requires independent review of series and retained-request logging; no invented trend threshold",
      "run_id":name,"performance":check,"lifecycle":cycles,"duration_seconds":duration,
      "resource_samples":len(data),"head_samples":len(head),"tail_samples":len(tail),
      "first_resource_sample_seconds":data[0]["seconds"],"last_resource_sample_seconds":data[-1]["seconds"],
      "resource_interval_median_seconds":statistics.median(gaps),"resource_interval_max_seconds":max(gaps),
      "resources":metrics,"resource_series":data,"ten_second_throughput":list(buckets.values()),
      "resource_sha256":v.sha(folder/"resources.jsonl")}
def main():
    p=argparse.ArgumentParser();p.add_argument("--stability");p.add_argument("--lifecycle-runs",nargs="+")
    p.add_argument("--report",required=True);a=p.parse_args()
    out=stability(a.stability) if a.stability else lifecycle(a.lifecycle_runs)
    Path(a.report).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k not in ["resource_series","ten_second_throughput","runs"]}),flush=True)
if __name__=="__main__":main()
