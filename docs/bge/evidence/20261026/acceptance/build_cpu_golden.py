#!/usr/bin/env python3
"""Independent CPU reference for original BGE; never invokes Ascend/GE."""
import datetime,hashlib,json,os,platform,sys,time
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT=Path("/export/home/ext.hejun23/workspace/bge")
OUT=ROOT/"work/bge-opt-20261026/acceptance"
EXPECTED_SHA={
"model.onnx":"2080f8762d9f8de50f8ec22f6a560453d3052d13012ea6fe7c199197962b3ba0",
"sample_inputs.json":"66571de2fa8c6a97e52a196a957801ab09b429751ba9cfb94712bb32c660b194",
"expected_outputs.json":"4765cc2b9dac9d2f523f9a684a81082d3b943caaff1fd2c4db90cfdf1b375384"}
def sha(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def array_info(x):
    return {"dtype":str(x.dtype),"shape":list(x.shape),"bytes":x.nbytes,"content_sha256":hashlib.sha256(x.tobytes(order="C")).hexdigest()}
def top_check(scores,expected):
    assert scores.dtype==np.float32 and scores.shape==(1,10112)
    assert np.isfinite(scores).all()
    idx=np.argsort(-scores[0],kind="stable")[:10]
    actual=scores[0,idx].astype(np.float64)
    ref=np.array(expected["top10_score"],dtype=np.float64)
    ids=(idx+1).tolist()
    errors=np.abs(actual-ref)
    return {"passed":ids==expected["top10_dense_id"] and bool(np.all(errors<=0.001)),
            "ids_match":set(ids)==set(expected["top10_dense_id"]),
            "order_match":ids==expected["top10_dense_id"],
            "max_abs_error":float(errors.max()),
            "dense_id":ids,"score":actual.tolist(),"atol":0.001,"rtol":0}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/"cpu-golden.npz"
    if target.exists():raise RuntimeError("Reference exists; do not overwrite frozen CPU golden")
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    files={}
    for name,want in EXPECTED_SHA.items():
        p=ROOT/name;got=sha(p);assert got==want,(name,got,want)
        files[name]={"path":str(p),"sha256":got,"bytes":p.stat().st_size}
    samples=json.loads((ROOT/"sample_inputs.json").read_text())
    expected={x["case_id"]:x["expected"] for x in json.loads((ROOT/"expected_outputs.json").read_text())}
    assert {x["case_id"] for x in samples}==set(range(5))==set(expected)
    options=ort.SessionOptions()
    options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level=3
    t=time.monotonic()
    session=ort.InferenceSession(str(ROOT/"model.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
    assert session.get_providers()==["CPUExecutionProvider"],session.get_providers()
    inputs=session.get_inputs()
    assert [o.name for o in session.get_outputs()]==["scores"]
    manifest={"schema":"bge-independent-cpu-golden/1","started_at":started,"pid":os.getpid(),
      "host":platform.node(),"python":sys.version,"numpy":np.__version__,"onnxruntime":ort.__version__,
      "providers":session.get_providers(),"session_options":{"intra_op_num_threads":2,"inter_op_num_threads":1,
      "execution_mode":"ORT_SEQUENTIAL","graph_optimization_level":"ORT_ENABLE_ALL"},
      "files":files,"model_io":{"inputs":[{"name":x.name,"type":x.type,"shape":x.shape} for x in inputs],
      "outputs":[{"name":x.name,"type":x.type,"shape":x.shape} for x in session.get_outputs()]},
      "cases":{}}
    arrays={}
    pool=ROOT/"ascend-ge-adapters/models/bge/io/inputs"
    for sample in sorted(samples,key=lambda x:x["case_id"]):
        case=f"case-{sample['case_id']:04d}";feed={};input_info={}
        for item in inputs:
            src="query_token_ids" if item.name=="input_ids" else item.name
            dtype=np.float32 if item.name=="stat_ratio" else np.int64
            value=np.asarray([sample["input"][src]],dtype=dtype)
            existing=np.load(pool/case/(item.name+".npy"),allow_pickle=False)
            assert existing.dtype==value.dtype and np.array_equal(existing,value),(case,item.name,"pool mismatch")
            feed[item.name]=value;input_info[item.name]=array_info(value)
        cstart=time.monotonic()
        scores=session.run(["scores"],feed)[0]
        business=top_check(scores,expected[sample["case_id"]])
        assert business["passed"],(case,business)
        arrays[case+"/outputs/scores"]=scores
        arrays[case+"/metadata/alignment"]=json.dumps({"rows":[0]})
        manifest["cases"][case]={"inputs":input_info,"output":array_info(scores),
          "duration_seconds":time.monotonic()-cstart,"business_top10":business}
        print("CPU_GOLDEN_CASE",case,"PASS","top10_maxabs",business["max_abs_error"],flush=True)
    np.savez(target,**arrays)
    manifest["golden"]={"path":str(target),"sha256":sha(target),"bytes":target.stat().st_size}
    manifest["historical_golden"]={}
    old=ROOT/"ascend-ge-adapters/work/bge/golden.npz"
    z=np.load(old,allow_pickle=False)
    for case in manifest["cases"]:
        a=arrays[case+"/outputs/scores"];b=z[case+"/outputs/scores"]
        manifest["historical_golden"][case]={"bitwise_equal":bool(np.array_equal(a,b)),
          "max_abs_error":float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max())}
    manifest["historical_golden_sha256"]=sha(old)
    manifest["duration_seconds"]=time.monotonic()-t
    manifest["finished_at"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest["passed"]=True
    (OUT/"cpu-golden-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"passed":True,"golden":manifest["golden"],"seconds":manifest["duration_seconds"],
      "historical_golden":manifest["historical_golden"]}),flush=True)
if __name__=="__main__":main()
