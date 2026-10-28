#!/usr/bin/env python3
"""CPU-only adversarial tests for independent acceptance helpers."""
import contextlib,copy,json,tempfile
from pathlib import Path
import numpy as np
import verify_outputs as v
HERE=Path(__file__).resolve().parent
checks={}
def reject(name,fn):
    try:
        result=fn()
        checks[name]=not result["passed"]
    except (AssertionError,ValueError,KeyError):
        checks[name]=True
def jsonl(path,rows):path.write_text("".join(json.dumps(r)+"\n" for r in rows))
meta,gold,expected=v.reference()
with tempfile.TemporaryDirectory(prefix="verifier-selftest-",dir=HERE) as td:
    root=Path(td);data=root/"outputs.bin";manifest=root/"manifest.jsonl";events=root/"events.jsonl";perf=root/"perf.json"
    rows=[];ev=[];blob=bytearray()
    for request in range(10):
        case=f"case-{request%5:04d}";offset=len(blob);blob.extend(gold[case].tobytes())
        rows.append({"run_id":"synthetic","request_id":request,"instance":0,"case_id":case,"phase":"measure",
          "tensors":[{"name":"scores","dtype":"float32","shape":[1,10112],"offset":offset,"bytes":40448}]})
        ev.append({"run_id":"synthetic","request_id":request,"instance":0,"case_id":case,"phase":"measure","status":"success"})
    data.write_bytes(blob);jsonl(manifest,rows);jsonl(events,ev)
    p=lambda:v.pressure_check(manifest,data,events,1,10,2)
    checks["pressure_valid_replays"]=p()["passed"]
    bad=copy.deepcopy(rows);bad[4]["case_id"]="case-0002";jsonl(manifest,bad)
    reject("pressure_reject_event_case_mismatch",p)
    bad_ev=copy.deepcopy(ev);bad_ev[4]["case_id"]="case-0002";jsonl(events,bad_ev)
    reject("pressure_reject_same_shape_wrong_scores",p)
    jsonl(manifest,rows);jsonl(events,ev)
    bad=copy.deepcopy(rows);bad[-1]["request_id"]=0;jsonl(manifest,bad)
    reject("pressure_reject_duplicate_id",p)
    jsonl(manifest,rows[:-1]);reject("pressure_reject_missing_request",p)
    bad=copy.deepcopy(rows);bad[1]["tensors"][0]["offset"]=0;jsonl(manifest,bad)
    reject("pressure_reject_binary_overlap",p)
    jsonl(manifest,rows);data.write_bytes(blob+b"trailing");reject("pressure_reject_unreferenced_bytes",p)
    base=100_000_000;measure=[];warm=[]
    for i in range(50):
        case=f"case-{i%5:04d}";nbytes=sum(x["bytes"] for x in meta["cases"][case]["inputs"].values())
        warm.append({"run_id":"timing-synthetic","request_id":f"warmup-0-{i}","instance":0,"case_id":case,
          "phase":"warmup","status":"success","started_ns":i*1_000_000,"completed_ns":(i+1)*1_000_000,
          "h2d_ms":.1,"desc_ms":.1,"execute_ms":.7,"d2h_ms":.1,"h2d_bytes":nbytes,"d2h_bytes":40448})
    for i in range(100):
        row=copy.deepcopy(warm[i%50]);row.update(request_id=i,phase="measure",
          started_ns=base+i*1_000_000,completed_ns=base+(i+1)*1_000_000)
        measure.append(row)
    timing={"window_start_ns":base,"window_end_ns":base+99_500_000,"window_successes":99,
      "cohort_successes":100,"drain_ms":.5}
    perf.write_text(json.dumps(timing));jsonl(events,warm+measure)
    p=lambda:v.perf_check(perf,events,1,0,99,10)
    ok=p();checks["performance_valid_window_and_drain"]=ok["passed"] and abs(ok["qps"]-99/.0995)<1e-8
    bad=copy.deepcopy(warm+measure);bad[-1]["d2h_bytes"]=4044;jsonl(events,bad)
    reject("performance_reject_incomplete_d2h",p)
    bad=copy.deepcopy(warm+measure);bad[-1]["status"]="failure";jsonl(events,bad)
    reject("performance_reject_failed_request",p)
    bad=copy.deepcopy(warm+measure);bad[-1]["d2h_ms"]=float("nan");jsonl(events,bad)
    reject("performance_reject_nan_segment",p)
    bad=copy.deepcopy(warm+measure);bad[-1]["started_ns"]-=500_000;bad[-1]["execute_ms"]+=.5;jsonl(events,bad)
    reject("performance_reject_inflight_overlap",p)
    jsonl(events,warm[:-1]+measure);reject("performance_reject_missing_warmup_case",p)
    jsonl(events,warm+measure);bad=copy.deepcopy(timing);bad["window_successes"]=100;perf.write_text(json.dumps(bad))
    reject("performance_reject_cohort_divided_by_window",p)
    perf.write_text(json.dumps(timing));reject("performance_reject_insufficient_count",lambda:v.perf_check(perf,events,1,0,100,10))
result={"schema":"bge-independent-verifier-cpu-tests/1","passed":all(checks.values()),"checks":checks}
(HERE/"verifier-adversarial-tests.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
assert result["passed"]
