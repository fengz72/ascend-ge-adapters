#!/usr/bin/env python3
"""Independent multi-round aggregation and device-10 resource evidence audit."""
import argparse,collections,json,math,re,statistics
from pathlib import Path
import verify_outputs as v
p=argparse.ArgumentParser();p.add_argument("--runs",nargs="+",required=True);p.add_argument("--report",required=True)
p.add_argument("--instances",type=int,default=1);p.add_argument("--min-seconds",type=float,default=60)
p.add_argument("--min-requests",type=int,default=10000)
a=p.parse_args();root=Path(__file__).resolve().parent.parent
witnesses=[]
wp=root/"acceptance/host-pid-witnesses.jsonl"
if wp.exists():witnesses=v.read_jsonl(wp)
def known_host_pids(name):
    found=set()
    for w in witnesses:
        for row in w.get("processes",[]):
            if row["run_id"]==name and row["role"]=="ge_runtime":found.add(row["host_pid"])
        for row in w.get("matches",[]):
            if str(root/"plans"/(name+".json")) in row:
                continue
            if str(root/"runner/runtime/build/ge_runtime") in row:
                # Legacy first witness lists its launcher's exact plan alongside runtime children.
                if any(str(root/"plans"/(name+".json")) in x for x in w.get("matches",[])):
                    found.add(int(row.split()[0]))
    return found
def resources(name):
    path=root/name/"resources.jsonl";rows=v.read_jsonl(path)
    assert rows and all(r["npu_smi_code"]==0 for r in rows),"missing/failed device samples"
    stamps=[r["monotonic_ns"] for r in rows];assert all(x<y for x,y in zip(stamps,stamps[1:]))
    seen=set();hbm=[];rss=[];per=[]
    for row in rows:
        pids={int(m.group(1)) for line in row["npu_smi_stdout"].splitlines()
              if (m:=re.match(r"^\|\s*5\s+0\s*\|\s*(\d+)\s*\|",line))}
        seen.update(pids)
        for line in row["npu_smi_stdout"].splitlines():
            if "0000:89:00.0" in line:
                m=re.search(r"(\d+)\s*/\s*65536",line);assert m,line;hbm.append(int(m.group(1)))
        if row.get("rss_kib") is not None:rss.append(row["rss_kib"])
        per.append({"monotonic_ns":row["monotonic_ns"],"runtime_container_pid":row["runtime_pid"],"device10_host_pids":sorted(pids)})
    known=known_host_pids(name);unmapped=seen-known
    assert not (known and unmapped),(name,"unmapped device10 processes alongside witnessed own process; performance needs investigation",sorted(unmapped))
    return {"sample_count":len(rows),"device":10,"npu":5,"chip":0,"BDF":"0000:89:00.0",
      "observed_host_pids":sorted(seen),"directly_witnessed_own_host_pids":sorted(known),
      "unmapped_host_pids":sorted(unmapped),"external_process_observed":None if unmapped else False,
      "identity_limitation":"No contemporaneous direct host/container PID witness for these PIDs; use recorded pre/post exclusive gates as bounded evidence." if unmapped else None,
      "hbm_mb_min":min(hbm),"hbm_mb_max":max(hbm),"rss_kib_min":min(rss) if rss else None,
      "rss_kib_max":max(rss) if rss else None,"resource_sha256":v.sha(path),"samples":per}
rounds=[];merged=[];case_lat=collections.defaultdict(list);plan_ids=[];resource_reports={}
for name in a.runs:
    folder=root/name;job_status=json.loads((root/"jobs"/name/"status.json").read_text());assert job_status["state"]=="finished" and job_status["exit_code"]==0 and job_status["process_group_stopped"],(name,"outer device job failed",job_status)
    result=json.loads((folder/"result.json").read_text());plan=json.loads(Path(result["plan"]).read_text())
    assert plan["device"]==10 and plan["instances"]==a.instances and plan["options"]["graph_run_mode"]==0
    assert not plan["observability"]["profiling"] and not plan["observability"]["dump"]
    perf=result["runtime"]["performance"];assert not perf["capture_enabled"],"capture is pressure, not formal performance"
    assert perf["latency_scope"]=="h2d_desc_execute_sync_d2h" and perf["arrival_model"]=="closed_loop"
    assert result["returncode"]==0 and result["runtime"]["status"]=="success"
    own=v.perf_check(folder/"result.json",folder/"events.jsonl",a.instances,a.min_seconds,a.min_requests,10)
    assert own["passed"],own
    events=[e for e in v.read_jsonl(folder/"events.jsonl") if e["phase"]=="measure"]
    buckets=collections.Counter();count_cases=collections.Counter()
    for e in events:
        latency=(e["completed_ns"]-e["started_ns"])/1e6;merged.append(latency);case_lat[e["case_id"]].append(latency)
        count_cases[e["case_id"]]+=1
        if e["completed_ns"]<=perf["window_end_ns"]:
            buckets[(e["completed_ns"]-perf["window_start_ns"])//10_000_000_000]+=1
    window=own["window_seconds"];seg=[]
    for i in range(math.ceil(window/10)):
        seconds=min(10.0,window-10*i)
        if seconds>0:seg.append({"start_seconds":i*10,"seconds":seconds,"successes":buckets[i],"qps":buckets[i]/seconds})
    own.update(run_id=name,seed=plan["seed"],plan_sha256=v.sha(result["plan"]),
      per_case_counts=dict(sorted(count_cases.items())),ten_second_windows=seg)
    (root/"acceptance"/(name+"-performance.json")).write_text(json.dumps(own,indent=2)+"\n")
    resource_reports[name]=resources(name)
    plan_ids.append({"run_id":name,"seed":plan["seed"],"options":plan["options"],"graph":plan["graph"],
      "input_binary_sha256":v.sha(plan["inputs"]["file"]),"binary_sha256":v.sha(result["binary"])})
    rounds.append(own)
qps=[x["qps"] for x in rounds];median=statistics.median(qps)
report={"schema":"bge-independent-rounds/1","passed":True,"runs":a.runs,"instances":a.instances,
"device":10,"rounds":rounds,"median_qps":median,"min_qps":min(qps),"max_qps":max(qps),
"qps_relative_range_percent":100*(max(qps)-min(qps))/median,"merged_request_count":len(merged),
"merged_p50_ms":v.quantile(merged,.5),"merged_p99_ms":v.quantile(merged,.99),
"merged_per_case":{c:{"count":len(x),"p50_ms":v.quantile(x,.5),"p99_ms":v.quantile(x,.99)} for c,x in sorted(case_lat.items())},
"resource_evidence":resource_reports,"plan_identities":plan_ids,"p99_limit":None,"SLO_acceptance":"not_configured",
"resource_identity_limitations":[name for name,r in resource_reports.items() if r["unmapped_host_pids"]]}
Path(a.report).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:report[k] for k in ["passed","median_qps","min_qps","max_qps","qps_relative_range_percent",
"merged_request_count","merged_p50_ms","merged_p99_ms","merged_per_case","resource_identity_limitations"]}),flush=True)
print("ROUND_SUMMARY",json.dumps([{"run_id":r["run_id"],"qps":r["qps"],"p99_ms":r["p99_ms"],
"window_seconds":r["window_seconds"],"requests":r["cohort_successes"],"per_case_counts":r["per_case_counts"]} for r in rounds]),flush=True)
print("RESOURCES",json.dumps({n:{k:v for k,v in row.items() if k!="samples"} for n,row in resource_reports.items()}),flush=True)
