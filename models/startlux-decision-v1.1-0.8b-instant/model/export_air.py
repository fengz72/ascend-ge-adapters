"""Standalone AIR export and request generation for the StartLux unified route.

The export stage of the standalone shell flow
(docs/standalone-runtime-plan.md): loads model.py's composed wrapper, exports
the ONE dynamic AIR through TorchAir (the bind_input identity markers, the
Data-node re-partition, the dynamic/static axis pinning, the frozen-parameter +
specialize_float dynamo context -- no ``core`` imports), embeds the IO contract
into the AIR through the runtime's ``write_io`` mode, and generates both
execution requests automatically:

* ``work/infer/request.json``  -- mode infer, the full precision pool
  (all official example cases), one execution per case;
* ``work/bench/request.json``  -- mode bench, the bench pool, with the
  instance / warmup / iterations / seed parameters run.sh passes in (warmup
  requests are per-instance preheat, not consumed from the measurement
  sequence).

Node names, order and dtypes are taken from the ACTUAL export result
(parse_bindings on the emitted pbtxt), never hand-written.  The graph IO is
embedded into the AIR with the runtime's write_io mode and the echoed IO (the
authoritative embedded contract) is what the requests carry.  The compat
prototype (chunk_gated_delta_rule_proto) is loaded by LD_PRELOAD alone: the
process must be started with the prototype .so preloaded (run.sh injects it),
which registers the op before any torchair/GE import and is inherited by the
ge_runtime children spawned here (write_io); _require_proto_preloaded asserts it.

Both requests use the same AIR.  The GESession options are explicit
(graph_run_mode 0, precision_mode must_keep_origin_dtype) -- the runtime's
default precision policy is never relied upon.

NOTE: no `from __future__ import annotations` here on purpose -- PEP 563 would
keep the bind_input custom-op annotations as strings that torch's
infer_schema cannot resolve in this nested-scope export path.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root
ROOT = MODEL_DIR.parents[1]
WORK = MODEL_DIR / "work"
BUILD = WORK / "build"
INFER = WORK / "infer"
BENCH = WORK / "bench"
INPUTS = WORK / "inputs"
RUNTIME = ROOT / "runtime" / "build" / "ge_runtime"
PROTO_ID = "chunk_gated_delta_rule_proto"
PROTO_INSTALL = WORK / "deps" / PROTO_ID / "install"
PROTO_LIB_NAME = "libchunk_gated_delta_rule_compat_proto.so"
PROTO_LIB = PROTO_INSTALL / "lib" / PROTO_LIB_NAME


def _load_module(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(MODEL_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _require_proto_preloaded():
    """Fail loudly unless the compat prototype .so is LD_PRELOADed into THIS process.

    LD_PRELOAD is the single loading contract: the dynamic linker maps the .so and
    runs its REG_OP static registration before the interpreter starts, so the op is
    registered ahead of any torchair/GE import, and the ge_runtime children spawned
    here inherit it through os.environ.  Auto vendor loading via
    ASCEND_CUSTOM_OPP_PATH is not used: it reproducibly crashes GE initialization
    with SIGSEGV/SIGILL on this build.  run.sh injects LD_PRELOAD; a manual run must.
    """
    preloaded = any(Path(entry).name == PROTO_LIB_NAME
                    for entry in os.environ.get("LD_PRELOAD", "").split(os.pathsep) if entry)
    if preloaded:
        return
    if not PROTO_LIB.is_file():
        raise SystemExit(f"export_air: {PROTO_LIB} is not built; run prepare.sh first, "
                         f"then start this process with LD_PRELOAD={PROTO_LIB}")
    raise SystemExit(f"export_air: the ChunkGatedDeltaRule GE prototype is not preloaded; "
                     f"start with LD_PRELOAD={PROTO_LIB} (run.sh does this automatically)")


def _ge_call(plan):
    """One ge_runtime subprocess on stdin JSON; returns the parsed GE_RESULT."""
    if not RUNTIME.is_file():
        raise SystemExit(f"export_air: runtime binary missing: {RUNTIME}; "
                         "build it: bash runtime/build.sh")
    process = subprocess.run([str(RUNTIME)], input=json.dumps(plan), text=True,
                             capture_output=True)
    if process.returncode:
        tail = (process.stdout + process.stderr)[-4000:]
        raise SystemExit(f"export_air: ge_runtime exited {process.returncode} "
                         f"(mode {plan.get('mode')}); tail:\n{tail}")
    messages = [line[len("GE_RESULT "):] for line in process.stdout.splitlines()
                if line.startswith("GE_RESULT ")]
    if len(messages) != 1:
        tail = (process.stdout + process.stderr)[-4000:]
        raise SystemExit(f"export_air: ge_runtime produced {len(messages)} GE_RESULT lines; "
                         f"tail:\n{tail}")
    return json.loads(messages[0])


def _validate_io(io):
    """The IOSpec contract checks of the framework validator, inlined."""
    if not isinstance(io, dict) or set(io) != {"schema", "inputs", "outputs"}:
        raise ValueError(f"IOSpec must carry exactly schema/inputs/outputs: {sorted(io)}")
    if io["schema"] != "ge-io/1":
        raise ValueError(f"unsupported IOSpec schema {io['schema']!r}")
    for side in ("inputs", "outputs"):
        if not io[side]:
            raise ValueError(f"IOSpec {side} is empty")
        names = []
        for item in io[side]:
            unknown = set(item) - {"name", "dtype", "shape", "dynamic_dims", "node",
                                   "format", "max_bytes"}
            missing = {"name", "dtype", "shape"} - set(item)
            if unknown or missing:
                raise ValueError(f"IOSpec.{side} item keys invalid: "
                                 f"unknown={sorted(unknown)}, missing={sorted(missing)}")
            name = item["name"]
            if not isinstance(name, str) or not name or "\0" in name:
                raise ValueError(f"invalid tensor name: {name!r}")
            names.append(name)
            dynamic = item.get("dynamic_dims", [])
            if sorted(set(dynamic)) != dynamic \
                    or any(axis < 0 or axis >= len(item["shape"]) for axis in dynamic):
                raise ValueError(f"invalid dynamic axes: {item}")
            if any(dim == 0 or dim < -1 for dim in item["shape"]):
                raise ValueError(f"invalid shape: {item}")
            if {axis for axis, dim in enumerate(item["shape"]) if dim == -1} != set(dynamic):
                raise ValueError(f"shape and dynamic_dims disagree: {item}")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate {side}: {names}")
    return io


def _reorder_data_nodes_first():
    """Stable in-place partition moving root Data nodes to the op-list front.

    The installed torchair 2.12.0 export path bakes weight inputs POSITIONALLY
    (runtime input i replaces export_graph.op[i] with a weight Const). With
    dynamic=True the SymInt extraction chain is emitted right after
    Data(input_ids), shifting the Data block, so that bake clobbers the wrong
    ops and erases real input Data nodes. Root Data nodes have no inputs, so
    moving them to the front ordered by their index attribute is topologically
    equivalent and restores the position<->input alignment the bake assumes.
    Static exports are a no-op.
    """
    from torchair.ge._ge_graph import get_default_ge_graph
    ops = get_default_ge_graph().op

    def order(op):
        if op.type == "Data":
            return (0, op.attr["index"].i if "index" in op.attr else 0)
        return (1, 0)

    ops.sort(key=order)


def parse_bindings(pbtxt):
    """TorchAir emits TF GraphDef text: root-level node blocks, with serialized
    GE AttrDef values inside strings. Read bounded chunks to skip huge Const
    lines without buffering frozen weights or counting braces inside strings."""
    result, block, keep, line_start = [], None, False, True
    with open(pbtxt) as stream:
        while True:
            piece = stream.readline(65536)
            if not piece:
                break
            at_start, line_start = line_start, piece.endswith("\n")
            if at_start and piece.startswith("node {"):
                block, keep = [], True
                continue
            if block is None:
                continue
            if at_start and piece.startswith("}"):
                if keep:
                    text = "".join(block)
                    if re.search(r'op:\s*"Data"', text):
                        name = re.search(r'name:\s*"([^"]+)"', text)
                        index = re.search(r'key:\s*"index".*?i:\s*(\d+)', text, re.S)
                        logical = re.search(r'key:\s*"_ge_adapter_name".*?s:\s*"([^"]+)"',
                                            text, re.S)
                        if not all((name, index, logical)):
                            raise ValueError("incomplete exported Data binding")
                        result.append((int(index.group(1)), logical.group(1), name.group(1)))
                block = None
                continue
            if keep:
                block.append(piece)
                match = re.match(r'\s*op:\s*"([^"]+)"', piece) if at_start else None
                if match and match.group(1) != "Data":
                    keep = False
                    block.clear()
    return sorted(result)


def _example_values(io, examples):
    """Strictly validated tracing examples, ordered like the IOSpec inputs."""
    import torch

    if not isinstance(examples, dict):
        raise ValueError("example_inputs must map logical input names to tensors")
    expected = {node["name"]: node for node in io["inputs"]}
    if set(examples) != set(expected):
        raise ValueError(f"example_inputs names {sorted(examples)} do not match the "
                         f"IOSpec inputs {sorted(expected)}")
    values = []
    for name, node in expected.items():
        value = examples[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"example input {name} must be a torch tensor")
        dtype = str(value.dtype).removeprefix("torch.")
        if dtype != node["dtype"]:
            raise ValueError(f"example input {name} dtype {dtype} does not match the "
                             f"declared {node['dtype']}")
        shape = tuple(int(size) for size in value.shape)
        if len(shape) != len(node["shape"]):
            raise ValueError(f"example input {name} rank {len(shape)} does not match the "
                             f"declared shape {node['shape']}")
        for axis, (size, declared) in enumerate(zip(shape, node["shape"])):
            if declared == -1:
                if size < 2:
                    raise ValueError(f"example input {name} axis {axis} traces size "
                                     f"{size}; a dynamic axis needs an example size >= 2 "
                                     "or dynamo specializes it into a static graph")
            elif size != declared:
                raise ValueError(f"example input {name} axis {axis} size {size} does not "
                                 f"match the static shape {node['shape']}")
        values.append(value)
    return values


def export_torchair(spec, directory):
    """Export ``spec`` to a TorchAir AIR under ``directory``; returns
    (graph, io) with io's inputs carrying the real exported node names."""
    import torch
    import torch_npu  # noqa: F401  (the NPU backend must be initialized)
    from torch_npu.dynamo.torchair import dynamo_export, CompilerConfig
    from torchair import register_fx_node_ge_converter

    io = _validate_io(spec["io"])
    names = [x["name"] for x in io["inputs"]]

    @torch.library.custom_op("ge_adapters::bind_input", mutates_args=())
    def bind_input(x: torch.Tensor, logical: str) -> torch.Tensor:
        return x.clone()

    @bind_input.register_fake
    def bind_fake(x, logical):
        return torch.empty_like(x)

    @register_fx_node_ge_converter(torch.ops.ge_adapters.bind_input.default)
    def bind_converter(x, logical: str, meta_outputs=None):
        if x.node.type != "Data":
            raise ValueError(f"{logical}: expected direct graph Data")
        x.node.attr["_ge_adapter_name"].s = logical.encode()
        # All Data nodes are created during input processing before the body
        # conversion reaches the bind_input nodes; re-partitioning here (after
        # every logical-name mark) leaves the graph aligned for torchair's
        # positional weight bake.
        _reorder_data_nodes_first()
        return x

    class BoundModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args):
            bound = [bind_input(x, name) for x, name in zip(args, names)]
            return self.model(*bound)

    values = _example_values(io, spec["example_inputs"])
    for node, value in zip(io["inputs"], values):
        # dynamic=True symbolizes every UNMARKED axis as well, so axes the
        # IOSpec declares static (position_ids leading 3, pixel_values
        # [512,1536]) must be pinned with mark_static or the shipped AIR Data
        # shapes degrade to numel-product-only symbolic dims.
        dynamic = set(node.get("dynamic_dims", []))
        for axis in range(len(node["shape"])):
            if axis in dynamic:
                torch._dynamo.mark_dynamic(value, axis)
            else:
                torch._dynamo.mark_static(value, axis)
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    # dynamic=True otherwise lifts every Python float the model closes over
    # (norm eps, attn scaling, mask min_score, ...) into implicit f64 graph
    # INPUTS that the declared tensor IO contract does not include;
    # specialize_float keeps them as baked constants exactly like the
    # dynamic=False route. This is a documented dynamo compile option applied
    # for this export call only and restored afterwards.
    dynamo_context = getattr(torch._dynamo.config, "patch", None)
    if dynamo_context is None or not hasattr(torch._dynamo.config, "specialize_float"):
        raise RuntimeError("torch dynamo config.patch/specialize_float unavailable; "
                           "cannot export dynamic graphs with baked scalar attributes")
    with dynamo_context(specialize_float=True):
        dynamo_export(*values, model=BoundModel(spec["model"]).eval(),
                      export_path=str(directory), export_name="model",
                      dynamic=any(x.get("dynamic_dims") for x in io["inputs"]),
                      config=config)
    torch.npu.synchronize()
    path = Path(directory) / "model.air"
    if not path.is_file():
        raise RuntimeError("TorchAir did not generate AIR")
    bindings = parse_bindings(Path(directory) / "dynamo.pbtxt")
    if [x[0] for x in bindings] != list(range(len(names))) \
            or {x[1] for x in bindings} != set(names):
        raise ValueError(f"non-bijective graph input binding: {bindings}")
    by_name = {x["name"]: x for x in io["inputs"]}
    io["inputs"] = [dict(by_name[name], node=node) for _, name, node in bindings]
    # The protobuf text duplicates frozen weights; AIR plus IOSpec is the
    # delivery artifact.
    (Path(directory) / "dynamo.pbtxt").unlink()
    return {"format": "air", "path": str(path)}, io


def build_request(mode, graph, io, records, case_ids, inputs_file, output_dir, device,
                  instances, warmup, iterations, seed):
    wanted = set(case_ids)
    selected = [record for record in records if record["id"] in wanted]
    missing = wanted - {record["id"] for record in selected}
    if missing:
        raise SystemExit(f"export_air: records.json misses pool cases {sorted(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return {"mode": mode, "backend": "ge_session",
            "graph": {"format": "air", "path": graph["path"], "om_path": None},
            "io": io,
            "inputs": {"file": inputs_file, "cases": selected},
            "device": device, "instances": instances,
            "warmup": warmup, "iterations": iterations, "seed": seed,
            "options": {"graph_run_mode": 0, "precision_mode": "must_keep_origin_dtype"},
            "observability": {"profiling": False, "dump": False},
            "output_dir": str(output_dir.resolve())}


def main():
    parser = argparse.ArgumentParser(description="Export the StartLux unified AIR and "
                                                 "generate the execution requests")
    parser.add_argument("--device", type=int, default=0,
                        help="in-process device index (ASCEND_RT_VISIBLE_DEVICES maps it)")
    parser.add_argument("--instances", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    _require_proto_preloaded()
    model = _load_module("startlux_model_for_export", "model/model.py")
    unified, _placement = model.load(args.device)
    spec = model.export_example(unified, args.device)

    BUILD.mkdir(parents=True, exist_ok=True)
    air = BUILD / "model.air"
    if air.exists():
        air.unlink()
    graph, io = export_torchair(spec, BUILD)

    embedded = _ge_call({"mode": "write_io",
                         "graph": {"format": "air", "path": graph["path"]},
                         "io": io})["io"]
    _validate_io(embedded)

    records_doc = json.loads((INPUTS / "records.json").read_text())
    if records_doc.get("schema") != "startlux-standalone-records/1":
        raise SystemExit("export_air: work/inputs/records.json is not the standalone "
                         "record index; run generate_inputs.py first")
    records = records_doc["cases"]
    cases_doc = json.loads((INPUTS / "cases.json").read_text())
    pools = cases_doc["pools"]

    infer_plan = build_request(
        "infer", graph, embedded, records, pools["precision"], records_doc["file"],
        INFER, args.device, instances=1, warmup=0, iterations=1, seed=args.seed)
    (INFER / "request.json").write_text(json.dumps(infer_plan, indent=1) + "\n")

    bench_plan = build_request(
        "bench", graph, embedded, records, pools["bench"], records_doc["file"],
        BENCH, args.device, instances=args.instances, warmup=args.warmup,
        iterations=args.iterations, seed=args.seed)
    (BENCH / "request.json").write_text(json.dumps(bench_plan, indent=1) + "\n")

    print(f"export_air: AIR {graph['path']} ({Path(graph['path']).stat().st_size} bytes) "
          f"with embedded IO; requests: infer ({len(infer_plan['inputs']['cases'])} cases) -> "
          f"{INFER / 'request.json'}, bench ({len(bench_plan['inputs']['cases'])} cases, "
          f"warmup {args.warmup}, {args.iterations} requests, seed {args.seed}) -> "
          f"{BENCH / 'request.json'}")


if __name__ == "__main__":
    main()
