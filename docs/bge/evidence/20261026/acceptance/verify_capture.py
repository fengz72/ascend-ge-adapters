#!/usr/bin/env python3
"""Read-only independent five-case validation of GE capture result and raw binary."""
import argparse,json
from pathlib import Path
import numpy as np
from verify_outputs import reference,compare_one,sha,THRESHOLDS
p=argparse.ArgumentParser();p.add_argument("--result",required=True);p.add_argument("--binary",required=True);p.add_argument("--report",required=True);a=p.parse_args()
raw_result=json.loads(Path(a.result).read_text())
if "returncode" in raw_result:assert raw_result["returncode"]==0
runtime=raw_result.get("runtime",raw_result);assert runtime["status"]=="success"
meta,gold,expected=reference();outs=runtime["outputs"];assert set(outs)==set(gold)
raw=Path(a.binary).read_bytes();assert len(raw)==5*40448
reports={};intervals=[]
for case,rows in outs.items():
    assert len(rows)==1 and rows[0]["instance"]==0
    tensors=rows[0]["tensors"];assert len(tensors)==1
    t=tensors[0]
    assert t["name"]=="scores" and t["dtype"]=="float32" and t["shape"]==[1,10112] and t["bytes"]==40448
    assert type(t["offset"]) is int and 0<=t["offset"]<=len(raw)-40448
    intervals.append((t["offset"],t["offset"]+t["bytes"]))
    value=np.frombuffer(raw,dtype="<f4",count=10112,offset=t["offset"]).reshape(1,10112)
    reports[case]=compare_one(value,gold[case],expected[case])
end=0
for start,stop in sorted(intervals):assert start==end;end=stop
assert end==len(raw)
out={"schema":"bge-independent-acceptance/1","kind":"five-case-capture",
"passed":all(v["passed"] for v in reports.values()),"golden_sha256":meta["golden"]["sha256"],
"result_path":a.result,"result_sha256":sha(a.result),"output_path":a.binary,
"output_sha256":sha(a.binary),"thresholds":THRESHOLDS,"cases":reports}
Path(a.report).write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"passed":out["passed"],"cases":{k:{m:v[m] for m in [
"passed","cosine","relative_l2","max_abs_error","top10_order_match","top10_max_abs_error"]} for k,v in reports.items()}}))
raise SystemExit(0 if out["passed"] else 1)
