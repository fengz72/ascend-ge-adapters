"""Two-stage split AIR export: dynamic vision AIR + text AIR.

The export stage of the standalone shell flow: exports the StartLux delivery
as TWO dynamic AIRs (docs: the split decision, 2026-11-10) -- the vision
stage (any image count / any official grid; the official position tables as
inputs) and the text stage (image_features as an input, injected like the
official masked_scatter) -- through TorchAir (the bind_input identity
markers, the Data-node re-partition, the dynamic/static axis pinning, the
frozen-parameter + specialize_float dynamo context).

The text graph contains ChunkGatedDeltaRule, so the process must run with
the compat prototype .so LD_PRELOADed (run.sh injects it;
_require_proto_preloaded asserts it).  No io is embedded into the AIRs: the
pipeline runtime (runtime/pipeline_runtime.cpp) binds inputs positionally in
the fixed order of vision_io_spec / text_io_spec, and parse_bindings still
validates the exported Data binding is bijective before delivery.

NOTE: no `from __future__ import annotations` here on purpose -- PEP 563 would
keep the bind_input custom-op annotations as strings that torch's
infer_schema cannot resolve in this nested-scope export path.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root
WORK = MODEL_DIR / "work"
BUILD = WORK / "build"
PROTO_ID = "chunk_gated_delta_rule_proto"
PROTO_INSTALL = WORK / "deps" / PROTO_ID / "install"
PROTO_LIB_NAME = "libchunk_gated_delta_rule_compat_proto.so"
PROTO_LIB = PROTO_INSTALL / "lib" / PROTO_LIB_NAME

# this file runs as a script from inside the model package; make the package
# itself importable (the model root is the package's parent)
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from model import modeling_multimodal                            # noqa: E402
import causal_conv1d  # noqa: E401,F401  activator: dlopen libcust_opapi.so, mount
# the vendor opp, register torch.ops.ascend_ops.causal_conv1d (Meta +
# PrivateUse1) and the CausalConv1dGE fx converter -- must precede
# dynamo_export; fail-loud until third_party/ascend-ops/causal-conv1d is
# installed (prepare.sh runs its install.sh).


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


_BIND_REGISTERED = False


def _ensure_bind_input():
    """Register ge_adapters::bind_input once per process (a second export in the
    same process must not re-register the custom op or its converter)."""
    global _BIND_REGISTERED
    if _BIND_REGISTERED:
        return
    import torch
    from torchair import register_fx_node_ge_converter

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

    _BIND_REGISTERED = True


def export_torchair(spec, directory, export_name="model"):
    """Export ``spec`` to a TorchAir AIR under ``directory`` (file
    ``<export_name>.air``); returns (graph, io) with io's inputs carrying the
    real exported node names."""
    import torch
    import torch_npu  # noqa: F401  (the NPU backend must be initialized)
    from torch_npu.dynamo.torchair import dynamo_export, CompilerConfig

    _ensure_bind_input()
    io = _validate_io(spec["io"])
    names = [x["name"] for x in io["inputs"]]

    class BoundModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args):
            bound = [torch.ops.ge_adapters.bind_input(x, name)
                     for x, name in zip(args, names)]
            return self.model(*bound)

    values = _example_values(io, spec["example_inputs"])
    for node, value in zip(io["inputs"], values):
        # dynamic=True symbolizes every UNMARKED axis as well, so axes the
        # IOSpec declares static must be pinned with mark_static or the
        # shipped AIR Data shapes degrade to numel-product-only symbolic dims.
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
                      export_path=str(directory), export_name=export_name,
                      dynamic=any(x.get("dynamic_dims") for x in io["inputs"]),
                      config=config)
    torch.npu.synchronize()
    path = Path(directory) / f"{export_name}.air"
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


def main():
    parser = argparse.ArgumentParser(description="Export the StartLux split AIRs "
                                                 "(vision + text)")
    parser.add_argument("--device", type=int, default=0,
                        help="in-process device index (ASCEND_RT_VISIBLE_DEVICES maps it)")
    args = parser.parse_args()

    _require_proto_preloaded()
    unified, _placement = modeling_multimodal.load(args.device)

    BUILD.mkdir(parents=True, exist_ok=True)
    for name in ("vision", "text"):
        stale = BUILD / f"{name}.air"
        if stale.exists():
            stale.unlink()

    import torch

    vision_spec = {
        "format": "torch", "model": unified.model.visual,
        "io": modeling_multimodal.vision_io_spec(),
        "example_inputs": {
            key: torch.from_numpy(value).to(f"npu:{args.device}")
            for key, value in modeling_multimodal.vision_example_inputs().items()},
    }
    graph, _io = export_torchair(vision_spec, BUILD, export_name="vision")
    vision_path = graph["path"]

    text_spec = {
        "format": "torch", "model": unified,
        "io": modeling_multimodal.text_io_spec(),
        "example_inputs": {
            key: torch.from_numpy(value).to(f"npu:{args.device}")
            for key, value in modeling_multimodal.text_example_inputs().items()},
    }
    graph, _io = export_torchair(text_spec, BUILD, export_name="text")
    text_path = graph["path"]

    manifest = {"schema": "startlux-split-air/1",
                "vision": {"path": vision_path,
                           "bytes": Path(vision_path).stat().st_size},
                "text": {"path": text_path,
                         "bytes": Path(text_path).stat().st_size},
                "vision_io": modeling_multimodal.vision_io_spec(),
                "text_io": modeling_multimodal.text_io_spec()}
    (BUILD / "split.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"export_air: vision {vision_path} ({manifest['vision']['bytes']} bytes), "
          f"text {text_path} ({manifest['text']['bytes']} bytes); manifest: "
          f"{BUILD / 'split.json'}")


if __name__ == "__main__":
    main()
