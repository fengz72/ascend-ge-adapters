#!/usr/bin/env python3
"""Bind the actual fixed/random plan to frozen original BGE input bytes."""
import argparse,json
from pathlib import Path
import workload_identity as w

def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",required=True);p.add_argument("--report",required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());out=w.check_plan_inputs(plan)
    out.update(schema="bge-plan-input-identity/2",plan=a.plan,plan_sha256=w.sha(a.plan),device=plan["device"],instances=plan["instances"],options=plan["options"])
    Path(a.report).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="cases"}))
if __name__=="__main__":main()
