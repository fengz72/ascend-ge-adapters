from pathlib import Path
import json
from core.contracts import validate_io


def export(spec, inputs, directory):
    import onnx
    model = onnx.load(spec["model"])
    onnx.checker.check_model(model)
    io = validate_io(spec["io"])
    initializer = {x.name for x in model.graph.initializer}
    names = [x.name for x in model.graph.input if x.name not in initializer]
    by_name = {x["name"]: x for x in io["inputs"]}
    if set(names) != set(by_name) or len(names) != len(by_name):
        raise ValueError("ONNX inputs do not match logical interface")
    io["inputs"] = [dict(by_name[n], node=n) for n in names]
    if [x.name for x in model.graph.output] != [x["name"] for x in io["outputs"]]:
        raise ValueError("ONNX output binding mismatch")
    # Runtime IO belongs to the graph artifact, not a sidecar manifest.
    props={entry.key:entry.value for entry in model.metadata_props}
    props['ge_adapters_io']=json.dumps(io)
    onnx.helper.set_model_props(model,props)
    path = Path(directory) / "model.onnx"
    # Include external tensor data in the build rather than retaining source-relative references.
    (Path(directory) / "weights.data").unlink(missing_ok=True)
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.data", size_threshold=1024)
    return {"format": "onnx", "path": str(path)}, io
