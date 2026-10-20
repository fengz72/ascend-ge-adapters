import os
import subprocess
from pathlib import Path


def build(graph, io, options, target, directory):
    prefix = Path(directory) / "model"
    argv = ["atc", f"--framework={1 if graph['format']=='air' else 5}", f"--model={graph['path']}",
            f"--output={prefix}", f"--soc_version={target['soc']}"]
    for name in ("aicore_num", "precision_mode"):
        if name in options:
            argv.append(f"--{name}={options[name]}")
    if graph["format"] == "onnx" or not any(x.get("dynamic_dims") for x in io["inputs"]):
        argv.append("--input_shape=" + ";".join(f"{x['node']}:{','.join(map(str,x['shape']))}" for x in io["inputs"]))
    with open(Path(directory) / "atc.log", "w") as log:
        log.write("COMMAND " + repr(argv) + "\n")
        log.flush()
        subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
    candidates = sorted(Path(directory).glob("model*.om"))
    if len(candidates) != 1:
        raise RuntimeError(f"ATC must produce exactly one OM, got {candidates}")
    return candidates[0].name
