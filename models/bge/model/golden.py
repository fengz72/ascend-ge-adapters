import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

ADAPTER_DIR = Path(__file__).resolve().parents[1]

# The runtime reports outputs positionally and tools/compare.py aligns the two sides
# by name, so the golden mirrors the runtime layout down to this name.
OUTPUT_NAME = "output_0"
GRAPH_OUTPUT = "scores"

# Frozen recipe: with the ONNX bytes and the pool bytes it is what makes the golden
# reproducible.  A different onnxruntime build may differ in the last bits.
INTRA_OP_THREADS = 2
INTER_OP_THREADS = 1
PROVIDERS = ["CPUExecutionProvider"]


def session_options():
    options = ort.SessionOptions()
    options.intra_op_num_threads = INTRA_OP_THREADS
    options.inter_op_num_threads = INTER_OP_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return options


def load_pool(inputs_dir):
    """``cases.json`` + the NPY pool -> ``[(case_id, {input name: array})]``.

    ``cases.json`` is the index ``ge_execute`` itself consumes, so both sides are fed
    from the same files.
    """
    inputs_dir = Path(inputs_dir)
    index = json.loads((inputs_dir / "cases.json").read_text())
    pool = []
    for case in index["cases"]:
        feed = {item["name"]: np.load(inputs_dir / item["path"], allow_pickle=False)
                for item in case["inputs"]}
        pool.append((case["id"], feed))
    return pool


def write_case(out, case_id, tensor):
    """Write ``<out>/<case_id>/output_0.npy``; return its result.json entry.

    ``path`` is relative to result.json's own directory -- the convention
    ``tools/compare.py`` resolves.
    """
    relative = f"{case_id}/{OUTPUT_NAME}.npy"
    (out / case_id).mkdir(parents=True, exist_ok=True)
    np.save(out / relative, tensor)
    return {"name": OUTPUT_NAME, "dtype": str(tensor.dtype),
            "shape": [int(dim) for dim in tensor.shape], "path": relative,
            "bytes": int(tensor.nbytes)}


def run(model, inputs_dir, output_dir):
    session = ort.InferenceSession(str(model), sess_options=session_options(),
                                   providers=PROVIDERS)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"status": "success", "cases": []}
    for case_id, feed in load_pool(inputs_dir):
        scores = session.run([GRAPH_OUTPUT], feed)[0]
        entry = write_case(out, case_id, scores)
        result["cases"].append({"id": case_id, "outputs": [entry]})
        print(f"golden: {case_id} {entry['shape']} {entry['dtype']}", flush=True)
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the BGE CPU FP32 golden (onnxruntime) from the input pool."
    )
    parser.add_argument("--model", required=True,
                        help="raw ONNX export (inline weights, ~1.35 GB); the same file "
                             "ge_execute is given, e.g. $BGE_MODEL_PATH")
    parser.add_argument("--inputs-dir", default=str(ADAPTER_DIR / "work" / "inputs"),
                        help="input pool from generate_cases.py (cases.json + case-NNNN/*.npy)")
    parser.add_argument("--output-dir", default=str(ADAPTER_DIR / "work" / "golden"),
                        help="golden directory: result.json + case-NNNN/output_0.npy")
    args = parser.parse_args(argv)
    result = run(args.model, args.inputs_dir, args.output_dir)
    print(json.dumps({"status": "ok", "cases": len(result["cases"]),
                      "onnxruntime": ort.__version__, "numpy": np.__version__,
                      "python": sys.version.split()[0],
                      "model": str(Path(args.model).resolve()),
                      "output": str(Path(args.output_dir).resolve())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
