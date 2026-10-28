#!/usr/bin/env python3
"""Compare canonical original/GE-packed ONNX topology, IO, all initializer bytes."""
import collections,gc,hashlib,json,time
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper
ROOT=Path("/export/home/ext.hejun23/workspace/bge")
OUT=ROOT/"work/bge-opt-20261026/acceptance/graph-integrity.json"
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def sig(path):
    model=onnx.load(str(path),load_external_data=True)
    h=hashlib.sha256()
    for n in model.graph.node:
        blob=n.SerializeToString(deterministic=True);h.update(len(blob).to_bytes(8,"little"));h.update(blob)
    tensors={}
    for t in model.graph.initializer:
        a=np.ascontiguousarray(numpy_helper.to_array(t))
        tensors[t.name]={"data_type":t.data_type,"dtype":str(a.dtype),"shape":list(a.shape),
           "content_sha256":hashlib.sha256(memoryview(a)).hexdigest()}
    interfaces=[]
    for side,items in [("input",model.graph.input),("output",model.graph.output)]:
        for v in items:
            tp=v.type.tensor_type
            interfaces.append({"kind":side,"name":v.name,"elem_type":tp.elem_type,
              "shape":[d.dim_value if d.HasField("dim_value") else d.dim_param for d in tp.shape.dim]})
    result={"path":str(path),"file_sha256":sha(path),"nodes":len(model.graph.node),
       "operators":dict(collections.Counter(n.op_type for n in model.graph.node)),
       "node_content_sha256":h.hexdigest(),"initializers":tensors,
       "interface":interfaces,"opsets":[[o.domain,o.version] for o in model.opset_import],
       "sparse_initializers":len(model.graph.sparse_initializer),"functions":len(model.functions)}
    del model;gc.collect();return result
start=time.monotonic()
source=sig(ROOT/"model.onnx")
assert source["file_sha256"]=="2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0"
packed=sig(ROOT/"ascend-ge-adapters/work/bge/build/model.onnx")
checks={key:source[key]==packed[key] for key in ["nodes","operators","node_content_sha256",
"initializers","interface","opsets","sparse_initializers","functions"]}
checks["source_30_slice_zero_range"]=source["operators"].get("Slice",0)==30 and source["operators"].get("Range",0)==0
checks["packed_30_slice_zero_range"]=packed["operators"].get("Slice",0)==30 and packed["operators"].get("Range",0)==0
result={"schema":"bge-original-packed-integrity/1","passed":all(checks.values()),
"checks":checks,"original":source,"packed":packed,"duration_seconds":time.monotonic()-start}
OUT.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"passed":result["passed"],"checks":checks,"duration_seconds":result["duration_seconds"],
"nodes":source["nodes"],"initializers":len(source["initializers"]),"source_sha256":source["file_sha256"],
"packed_sha256":packed["file_sha256"]}),flush=True)
assert result["passed"]
