#!/usr/bin/env python3
"""Bind completed run artifacts to frozen executable, original graph, input bytes and job."""
import argparse,json,re
from pathlib import Path
import verify_outputs as v
import workload_identity as w
p=argparse.ArgumentParser();p.add_argument("--runs",nargs="+",required=True);p.add_argument("--report",required=True)
p.add_argument("--root",type=Path,default=Path(__file__).resolve().parent.parent)
a=p.parse_args();root=a.root;manifest=w.metadata();checks={}
for name in a.runs:
    folder=root/name;status_path=root/"jobs"/name/"status.json";status=json.loads(status_path.read_text())
    assert status["state"]=="finished" and status["exit_code"]==0 and status["process_group_stopped"],(name,"outer device job failed",status)
    result=json.loads((folder/"result.json").read_text());assert result["returncode"]==0
    runtime=result["runtime"];assert runtime["status"]=="success" and runtime["backend"]=="ge_session" and runtime["mode"]=="bench"
    plan_path=Path(result["plan"]);plan=json.loads(plan_path.read_text());job_path=root/"jobs"/name/"request.json";job=json.loads(job_path.read_text())
    assert plan["device"]==10 and plan["options"]["graph_run_mode"]==0
    assert plan["replay"]["run_id"]==name==runtime["performance"]["run_id"]
    assert plan["replay"]==runtime["replay"]
    assert Path(plan["output_dir"])==folder
    input_identity=w.check_plan_inputs(plan)
    assert runtime["performance"]["sampling"]["seed"]==plan["seed"] and runtime["performance"]["sampling"]["pool_size"]==len(input_identity["workload"]["case_ids"])
    assert runtime["performance"]["capture_enabled"]==plan["replay"]["capture_outputs"]
    assert len(runtime["performance"]["instances"])==plan["instances"]
    assert v.sha(result["binary"])=="8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d"
    assert v.sha(plan["graph"]["path"])=="5e641362d99c9a1afb6dfce70f47b4ac07f1a7c88272fdfc791ea04ae84f4822"
    assert v.sha(plan["inputs"]["file"])==input_identity["input_binary_sha256"]
    assert str(plan_path) in job["argv"] and result["binary"] in job["argv"]
    assert job["devices"]==["A03-R40-I191-99-4100006.JD.LOCAL/0000:89:00.0"]
    events=v.read_jsonl(folder/"events.jsonl");assert events and {e["run_id"] for e in events}=={name}
    if plan["replay"]["capture_outputs"]:
        rows=v.read_jsonl(folder/"manifest.jsonl");assert rows and {row["run_id"] for row in rows}=={name}
    plugins={}
    runtime_env_tokens=[]
    for key,value in plan.get("runtime_env",{}).items():
        if key=="LD_PRELOAD":runtime_env_tokens=["LD_PRELOAD="+value]
    for argv in job["argv"]+runtime_env_tokens:
        if argv.startswith("LD_PRELOAD="):
            for path in re.split(r"[:\s]+",argv.split("=",1)[1]):
                if path and Path(path).name.startswith("libbge_add_layernorm_pass"):
                    plugins[path]=v.sha(path)
                    assert plugins[path] in ["8767095f04348ed6097c769d71909421409397cedef534c4c7650d61df6cfff5","76ea9fb0f1800369b023e22d71298e508dcfaa6f69f34bcf952af4a1b5fa48ed"]
                    if path.endswith("_v2.so"):
                        assert plan["runtime_env"]["LD_PRELOAD"].startswith("/usr/lib64/libtcmalloc.so.4.5.16:")
                        assert not any(path in token for token in job["argv"]),"V2 plugin must not preload Python launcher"
    checks[name]={"passed":True,"result_sha256":v.sha(folder/"result.json"),"plan_sha256":v.sha(plan_path),
      "job_status_sha256":v.sha(status_path),"outer_exit_code":status["exit_code"],"job_request_sha256":v.sha(job_path),"binary_sha256":v.sha(result["binary"]),
      "packed_graph_sha256":v.sha(plan["graph"]["path"]),"input_binary_sha256":v.sha(plan["inputs"]["file"]),
      "workload":input_identity["workload"],"precision_mode":plan["options"]["precision_mode"],"plugins":plugins,"seed":plan["seed"],
      "capture_enabled":plan["replay"]["capture_outputs"]}
out={"schema":"bge-run-artifact-identity/1","passed":True,"runs":checks}
Path(a.report).write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out),flush=True)
