#!/usr/bin/env python3
"""Independent, read-only BGE output/event acceptance; fixed thresholds."""
import argparse,collections,hashlib,json,math,sys,traceback
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
REFERENCE=HERE/"cpu-golden.npz"
META=HERE/"cpu-golden-manifest.json"
EXPECTED=Path("/export/home/ext.hejun23/workspace/bge/expected_outputs.json")
THRESHOLDS={"cosine_min":0.9999,"relative_l2_max":0.001,"max_abs_error":0.001,
            "top10_atol":0.001,"top10_rtol":0}
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read_jsonl(path):
    rows=[]
    with Path(path).open() as f:
        for n,line in enumerate(f,1):
            if not line.strip():raise ValueError(f"{path}:{n}: blank JSONL row")
            row=json.loads(line)
            if not isinstance(row,dict):raise ValueError(f"{path}:{n}: expected object")
            rows.append(row)
    return rows
def reference():
    meta=json.loads(META.read_text())
    assert sha(REFERENCE)==meta["golden"]["sha256"],"frozen CPU golden hash differs"
    assert sha(EXPECTED)==meta["files"]["expected_outputs.json"]["sha256"],"expected hash differs"
    with np.load(REFERENCE,allow_pickle=False) as z:
        gold={f"case-{i:04d}":z[f"case-{i:04d}/outputs/scores"].copy() for i in range(5)}
    expected={f"case-{r['case_id']:04d}":r["expected"] for r in json.loads(EXPECTED.read_text())}
    return meta,gold,expected
def compare_one(actual,golden,expected):
    reasons=[]
    if actual.dtype!=np.float32:reasons.append("output_dtype_not_float32")
    if actual.shape!=(1,10112):reasons.append("output_shape_not_1x10112")
    if reasons:return {"passed":False,"reasons":reasons,"dtype":str(actual.dtype),"shape":list(actual.shape)}
    if not np.isfinite(actual).all():return {"passed":False,"reasons":["output_nonfinite"]}
    a=actual.astype(np.float64).ravel();g=golden.astype(np.float64).ravel()
    normg=np.linalg.norm(g);norma=np.linalg.norm(a)
    cos=float(np.dot(a,g)/(norma*normg)) if norma*normg else float(norma==normg)
    rel=float(np.linalg.norm(a-g)/normg) if normg else (0.0 if norma==0 else None)
    maxabs=float(np.max(np.abs(a-g)))
    if cos<0.9999:reasons.append("cosine_below_minimum")
    if rel is None or rel>0.001:reasons.append("relative_l2_exceeds_maximum")
    if maxabs>0.001:reasons.append("full_score_max_abs_exceeds_maximum")
    idx=np.argsort(-actual[0],kind="stable")[:10];ids=(idx+1).tolist()
    scores=actual[0,idx].astype(np.float64);wanted=np.array(expected["top10_score"],dtype=np.float64)
    top_errors=np.abs(scores-wanted)
    if ids!=expected["top10_dense_id"]:reasons.append("top10_ids_or_order_mismatch")
    if np.any(top_errors>0.001):reasons.append("top10_score_absolute_error_exceeds_maximum")
    return {"passed":not reasons,"reasons":reasons,"cosine":cos,"relative_l2":rel,
      "max_abs_error":maxabs,"top10_ids_match":set(ids)==set(expected["top10_dense_id"]),
      "top10_order_match":ids==expected["top10_dense_id"],"top10_max_abs_error":float(top_errors.max()),
      "top10_dense_id":ids,"top10_score":scores.tolist(),
      "expected_top10_dense_id":expected["top10_dense_id"],"expected_top10_score":wanted.tolist()}
def npz_check(path):
    meta,gold,expected=reference()
    with np.load(path,allow_pickle=False) as z:
        keys={k for k in z.files if "/outputs/" in k}
        wanted={c+"/outputs/scores" for c in gold}
        assert keys==wanted,("case/output key mismatch",sorted(keys),sorted(wanted))
        reports={c:compare_one(z[c+"/outputs/scores"],gold[c],expected[c]) for c in gold}
    return {"kind":"five_case_npz","passed":all(x["passed"] for x in reports.values()),
      "output_path":str(path),"output_sha256":sha(path),"reference_sha256":meta["golden"]["sha256"],
      "thresholds":THRESHOLDS,"cases":reports}
def pressure_check(manifest,binary,events,instances,min_requests,min_case):
    meta,gold,expected=reference();rows=read_jsonl(manifest);ev=read_jsonl(events)
    measured=[e for e in ev if e.get("phase")=="measure"]
    assert len({e["run_id"] for e in ev})==1,"mixed run IDs in pressure events"
    keyed={}
    for e in measured:
        assert type(e.get("request_id")) is int,"measure request ID must be integer"
        key=(e["run_id"],e["request_id"])
        assert key not in keyed,("duplicate event request ID",key)
        keyed[key]=e
    assert len(rows)>=min_requests,("pressure count too small",len(rows),min_requests)
    seen=set();counts=collections.Counter();failures=[];intervals=[];worst=collections.defaultdict(float)
    per_case={};first_per_case={};first_failed_per_case={}
    binary=Path(binary);size=binary.stat().st_size
    with binary.open("rb") as f:
        for row in rows:
            assert row.get("phase")=="measure","manifest phase must be measure"
            key=(row["run_id"],row["request_id"])
            assert key not in seen,("duplicate manifest request",key);seen.add(key)
            assert key in keyed,("manifest request has no event",key)
            event=keyed[key]
            for k in ["instance","case_id","phase"]:assert row[k]==event[k],("event/manifest identity differs",key,k)
            assert event["status"] in ["success","ok"],("failed pressure request",key,event)
            inst=row["instance"];case=row["case_id"]
            assert type(inst) is int and 0<=inst<instances,("invalid instance",inst)
            assert case in gold,("unknown case",case)
            tensors=row["tensors"];assert len(tensors)==1 and tensors[0]["name"]=="scores"
            t=tensors[0];assert t["dtype"]=="float32" and t["shape"]==[1,10112],("tensor dtype/shape",t)
            offset=t["offset"];length=t["bytes"]
            assert type(offset) is int and type(length) is int and offset>=0 and length==40448 and offset+length<=size
            intervals.append((offset,offset+length));f.seek(offset);data=f.read(length);assert len(data)==length
            result=compare_one(np.frombuffer(data,dtype="<f4").reshape(1,10112),gold[case],expected[case])
            counts[(inst,case)]+=1
            summary=per_case.setdefault(case,{"requests":0,"failed_requests":0,"cosine_min":1.0,"max_abs_error":0.0,"relative_l2_max":0.0,"top10_max_abs_error":0.0})
            summary["requests"]+=1
            summary["failed_requests"]+=int(not result["passed"])
            if "cosine" in result:summary["cosine_min"]=min(summary["cosine_min"],result["cosine"])
            for target,source in [("max_abs_error","max_abs_error"),("relative_l2_max","relative_l2"),("top10_max_abs_error","top10_max_abs_error")]:
                summary[target]=max(summary[target],result.get(source) or 0.0)
            if case not in first_per_case:first_per_case[case]={"request_id":row["request_id"],"instance":inst,**result}
            if not result["passed"] and case not in first_failed_per_case:first_failed_per_case[case]={"request_id":row["request_id"],"instance":inst,**result}
            for name in ["max_abs_error","relative_l2","top10_max_abs_error"]:worst[name]=max(worst[name],result.get(name) or 0.0)
            if not result["passed"]:failures.append({"request_id":row["request_id"],"instance":inst,"case_id":case,**result})
    assert seen==set(keyed),("missing request outputs",len(seen),len(keyed))
    end=0
    for start,stop in sorted(intervals):assert start==end,("output binary gap/overlap",start,end);end=stop
    assert end==size,("unreferenced/trailing output bytes",end,size)
    for inst in range(instances):
        for case in gold:assert counts[(inst,case)]>=min_case,("instance/case coverage too small",inst,case,counts[(inst,case)])
    return {"kind":"pressure","passed":not failures,"requests":len(rows),"instances":instances,
      "per_instance_case_counts":{f"{i}/{c}":n for (i,c),n in sorted(counts.items())},
      "worst":dict(worst),"per_case_metrics":per_case,"first_result_per_case":first_per_case,"first_failure_per_case":first_failed_per_case,
      "failed_requests":len(failures),"failures":failures[:100],
      "manifest_sha256":sha(manifest),"binary_sha256":sha(binary),"events_sha256":sha(events),
      "reference_sha256":meta["golden"]["sha256"],"thresholds":THRESHOLDS}
def quantile(values,p):
    assert values
    return sorted(values)[max(0,math.ceil(p*len(values))-1)]
def perf_check(perf_path,event_path,instances,min_seconds,min_requests,warmup_min):
    meta,gold,_=reference();perf=json.loads(Path(perf_path).read_text())
    if "returncode" in perf:assert perf["returncode"]==0,"runtime command failed"
    if "runtime" in perf:perf=perf["runtime"]
    if "status" in perf:assert perf["status"]=="success","runtime status not success"
    if "performance" in perf:perf=perf["performance"]
    rows=read_jsonl(event_path);measured=[e for e in rows if e.get("phase")=="measure"]
    assert measured,"no measured requests"
    assert len({e["run_id"] for e in rows})==1,"mixed run IDs in performance events"
    start=perf["window_start_ns"];end=perf["window_end_ns"];assert type(start) is int and type(end) is int and end>start
    keys=set();byinst=collections.defaultdict(list);bycase=collections.defaultdict(list);warm=collections.Counter()
    window=0;latencies=[];segments=collections.defaultdict(list);max_end=end;error_count=0
    for e in rows:
        assert e.get("phase") in ["warmup","measure"],("unknown phase",e.get("phase"))
        key=(e["run_id"],e["request_id"]);assert key not in keys,("duplicate request ID",key);keys.add(key)
        assert type(e["instance"]) is int and 0<=e["instance"]<instances
        case=e["case_id"];assert case in gold,("unknown case",case)
        s=e["started_ns"];c=e["completed_ns"];assert type(s) is int and type(c) is int and c>=s
        assert e["status"] in ["success","ok"],("failed request makes performance invalid",e)
        for k in ["h2d_ms","desc_ms","execute_ms","d2h_ms"]:
            assert type(e[k]) in [int,float] and math.isfinite(e[k]) and e[k]>=0,(k,e[k])
        assert e["d2h_bytes"]==40448,("incomplete output D2H",e["d2h_bytes"])
        want=sum(v["bytes"] for v in meta["cases"][case]["inputs"].values())
        assert e["h2d_bytes"]==want,("H2D bytes differ",case,e["h2d_bytes"],want)
        total=(c-s)/1e6
        segment_sum=sum(e[k] for k in ["h2d_ms","desc_ms","execute_ms","d2h_ms"])
        assert segment_sum<=total+max(0.05,total*0.01),("segment sum exceeds full request time",total,e)
        if e["phase"]=="warmup":
            assert c<=start,"warmup overlaps measured window"
            warm[(e["instance"],case)]+=1;continue
        assert type(e["request_id"]) is int and e["request_id"]>=0
        assert start<=s<end,("request sent outside window",s,start,end)
        byinst[e["instance"]].append((s,c));bycase[case].append(total);latencies.append(total);max_end=max(max_end,c)
        if c<=end:window+=1
        for k in ["h2d_ms","desc_ms","execute_ms","d2h_ms"]:segments[k].append(e[k])
        segments["unattributed_host_ms"].append(max(0.0,total-segment_sum))
    for inst in range(instances):
        seq=sorted(byinst[inst]);assert seq,("instance has no requests",inst)
        assert all(a[1]<=b[0] for a,b in zip(seq,seq[1:])),("closed loop request overlap",inst)
        for case in gold:assert warm[(inst,case)]>=warmup_min,("insufficient per-case warmup",inst,case,warm[(inst,case)])
    seconds=(end-start)/1e9;qps=window/seconds;cohort=len(measured);drain=(max_end-end)/1e6
    assert {e["request_id"] for e in measured}==set(range(cohort)),"missing/global request ID gaps"
    if "qps" in perf:assert math.isclose(perf["qps"],qps,rel_tol=1e-10,abs_tol=1e-9),"published QPS differs"
    if "e2e_ms" in perf:
        assert math.isclose(perf["e2e_ms"]["p99"],quantile(latencies,.99),rel_tol=1e-10,abs_tol=1e-9),"published p99 differs"
        assert math.isclose(perf["e2e_ms"]["p50"],quantile(latencies,.5),rel_tol=1e-10,abs_tol=1e-9),"published p50 differs"
    assert perf["window_successes"]==window and perf["cohort_successes"]==cohort,("completion counts inconsistent",window,cohort,perf)
    assert abs(perf["drain_ms"]-drain)<=0.1,("drain inconsistent",perf["drain_ms"],drain)
    sufficient=seconds>=min_seconds and window>=min_requests
    return {"kind":"performance","passed":sufficient,"statistics_sufficient":sufficient,"errors":0,
      "minimum_seconds":min_seconds,"minimum_window_successes":min_requests,"window_seconds":seconds,
      "window_successes":window,"cohort_successes":cohort,"drain_ms":drain,"qps":qps,
      "p50_ms":quantile(latencies,.5),"p99_ms":quantile(latencies,.99),"max_ms":max(latencies),
      "p99_limit":None,"SLO_acceptance":"not_configured",
      "per_case":{c:{"count":len(v),"p50_ms":quantile(v,.5),"p99_ms":quantile(v,.99)} for c,v in bycase.items()},
      "segments":{k:{"mean":float(np.mean(v)),"p99":quantile(v,.99)} for k,v in segments.items()},
      "events_sha256":sha(event_path),"performance_sha256":sha(perf_path)}
def selftest():
    _,g,e=reference();a=g["case-0000"].copy()
    checks={}
    checks["golden_passes"]=compare_one(a,g["case-0000"],e["case-0000"])["passed"]
    bad=a.astype(np.float16);checks["reject_fp16_output"]=not compare_one(bad,g["case-0000"],e["case-0000"])["passed"]
    bad=a.copy();bad[0,12]=np.nan;checks["reject_nan"]=not compare_one(bad,g["case-0000"],e["case-0000"])["passed"]
    bad=a.copy();bad[0,12]+=np.float32(.01);checks["reject_full_vector_error"]=not compare_one(bad,g["case-0000"],e["case-0000"])["passed"]
    checks["reject_same_shape_wrong_case"]=not compare_one(g["case-0004"],g["case-0002"],e["case-0002"])["passed"]
    bad=a.copy();ix=np.argsort(-a[0],kind="stable")[:2];bad[0,ix]=bad[0,ix[::-1]]
    checks["reject_top10_order_swap"]="top10_ids_or_order_mismatch" in compare_one(bad,g["case-0000"],e["case-0000"])["reasons"]
    checks["nearest_rank_100"]=quantile(list(range(1,101)),.99)==99
    checks["nearest_rank_1000"]=quantile(list(range(1,1001)),.99)==990
    assert all(checks.values()),checks
    return {"kind":"selftest","passed":True,"checks":checks}
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest="mode",required=True)
    for mode in ["npz","pressure","perf","selftest"]:
        q=sub.add_parser(mode);q.add_argument("--report",required=True)
        if mode=="npz":q.add_argument("--outputs",required=True)
        if mode in ["pressure","perf"]:
            q.add_argument("--events",required=True);q.add_argument("--instances",type=int,required=True)
            q.add_argument("--min-requests",type=int,default=2000 if mode=="pressure" else 10000)
        if mode=="pressure":
            q.add_argument("--manifest",required=True);q.add_argument("--binary",required=True);q.add_argument("--min-per-case",type=int,default=20)
        if mode=="perf":
            q.add_argument("--performance",required=True);q.add_argument("--min-seconds",type=float,default=60);q.add_argument("--warmup-min",type=int,default=10)
    a=p.parse_args()
    try:
        if a.mode=="npz":result=npz_check(a.outputs)
        elif a.mode=="pressure":result=pressure_check(a.manifest,a.binary,a.events,a.instances,a.min_requests,a.min_per_case)
        elif a.mode=="perf":result=perf_check(a.performance,a.events,a.instances,a.min_seconds,a.min_requests,a.warmup_min)
        else:result=selftest()
    except Exception as exc:
        result={"passed":False,"kind":a.mode,"exception":str(exc),"traceback":traceback.format_exc()}
    target=Path(a.report);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["cases","traceback","failures","per_case","first_result_per_case","first_failure_per_case"]},allow_nan=False))
    return 0 if result["passed"] else 1
if __name__=="__main__":sys.exit(main())
