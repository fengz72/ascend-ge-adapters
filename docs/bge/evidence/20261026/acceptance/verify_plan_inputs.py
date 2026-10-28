#!/usr/bin/env python3
"""Bind packed input bytes to the independent original CPU manifest."""
import argparse,hashlib,json
from pathlib import Path
from verify_outputs import reference,sha
p=argparse.ArgumentParser();p.add_argument("--plan",required=True);p.add_argument("--report",required=True)
a=p.parse_args();plan=json.loads(Path(a.plan).read_text());meta,gold,_=reference()
assert plan["backend"]=="ge_session" and plan["options"]["graph_run_mode"]==0
cases=plan["inputs"]["cases"];assert {c["id"] for c in cases}==set(gold) and len(cases)==5
binary=Path(plan["inputs"]["file"]);raw=binary.read_bytes();intervals=[];checks={}
for case in cases:
    want=meta["cases"][case["id"]]["inputs"]
    assert case["samples"]==1 and case["output_capacities"]["scores"]==40448
    assert {t["name"] for t in case["tensors"]}==set(want) and len(case["tensors"])==len(want)
    checks[case["id"]]={}
    for t in case["tensors"]:
        ref=want[t["name"]];offset=t["offset"];nbytes=t["bytes"]
        assert t["dtype"]==ref["dtype"] and t["shape"]==ref["shape"] and nbytes==ref["bytes"]
        assert type(offset) is int and offset>=0 and offset+nbytes<=len(raw)
        got=hashlib.sha256(raw[offset:offset+nbytes]).hexdigest()
        assert got==ref["content_sha256"],(case["id"],t["name"],"input bytes differ")
        intervals.append((offset,offset+nbytes));checks[case["id"]][t["name"]]={"passed":True,"content_sha256":got}
end=0
for start,stop in sorted(intervals):assert start==end;end=stop
assert end==len(raw)
out={"schema":"bge-plan-input-identity/1","passed":True,"plan":a.plan,"plan_sha256":sha(a.plan),
"input_binary":str(binary),"input_binary_sha256":sha(binary),"input_bytes":len(raw),"device":plan["device"],
"instances":plan["instances"],"options":plan["options"],"cases":checks}
Path(a.report).write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:v for k,v in out.items() if k!="cases"}))
