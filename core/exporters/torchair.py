"""Export-owned identity markers bind logical names without patching TorchAir."""
from pathlib import Path
import re
from core.contracts import validate_io


def _reorder_data_nodes_first():
    """Stable in-place partition moving root Data nodes to the op-list front.

    The installed torchair 2.12.0 export path
    (torchair/_utils/export_utils.py::_convert_data_to_const) bakes weight
    inputs POSITIONALLY: runtime input i replaces export_graph.op[i] with a
    weight Const. With dynamic=True the SymInt extraction chain
    (Shape/Const/Gather) is emitted right after Data(input_ids), shifting the
    Data block, so that bake clobbers the wrong ops and erases real input Data
    nodes (minimal repro + instrumented evidence:
    work/startlux-validation/p4-dynamic/repro/repro-both.json). Root Data
    nodes have no inputs, so moving them to the front ordered by their index
    attribute is topologically equivalent and restores the position<->input
    alignment the bake assumes. RepeatedCompositeFieldContainer.sort is
    in-place; pre-sort Tensor.node references stay valid (verified). Static
    exports already have Data first in input order, making this a no-op.
    """
    from torchair.ge._ge_graph import get_default_ge_graph
    ops = get_default_ge_graph().op

    def order(op):
        if op.type == "Data":
            return (0, op.attr["index"].i if "index" in op.attr else 0)
        return (1, 0)

    ops.sort(key=order)


def parse_bindings(pbtxt):
    # TorchAir emits TF GraphDef text: root-level node blocks, with serialized
    # GE AttrDef values inside strings. Read bounded chunks to skip huge Const
    # lines without buffering frozen weights or counting braces inside strings.
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
                        logical = re.search(r'key:\s*"_ge_adapter_name".*?s:\s*"([^"]+)"', text, re.S)
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


def _example_values(torch, io, examples):
    """Strictly validated tracing examples, ordered like the IOSpec inputs.

    ``export_spec['example_inputs']`` is the adapter-owned deterministic trace
    example. The regular CLI build passes the first fixture case as ``inputs``
    and dynamo specializes size-1 dimensions, so a B1 first case can never
    trace a dynamic batch; adapters that need a dynamic graph provide this
    explicit example instead. Names must exactly cover the logical inputs;
    every tensor's dtype, rank and static dims must match the IOSpec; every
    declared dynamic axis must trace with size >= 2 so dynamo keeps it
    symbolic and the exported Data nodes really are the declared dynamic
    shapes instead of a graph the io merely claims dynamic.
    """
    if not isinstance(examples, dict):
        raise ValueError("export_spec example_inputs must map logical input names to tensors")
    expected = {node["name"]: node for node in io["inputs"]}
    if set(examples) != set(expected):
        raise ValueError(f"example_inputs names {sorted(examples)} do not match the IOSpec "
                         f"inputs {sorted(expected)}")
    values = []
    for name, node in expected.items():
        value = examples[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"example input {name} must be a torch tensor, "
                            f"got {type(value).__name__}")
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
                    raise ValueError(f"example input {name} axis {axis} traces size {size}; a "
                                    f"dynamic axis needs an example size >= 2 or dynamo "
                                    f"specializes it into a static graph")
            elif size != declared:
                raise ValueError(f"example input {name} axis {axis} size {size} does not match "
                                 f"the static shape {node['shape']}")
        values.append(value)
    return values


def export(spec, inputs, directory):
    import torch
    import torch_npu
    from torch_npu.dynamo.torchair import dynamo_export, CompilerConfig
    from torchair import register_fx_node_ge_converter
    io = validate_io(spec["io"])
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
        # positional weight bake. See _reorder_data_nodes_first above.
        _reorder_data_nodes_first()
        return x

    class BoundModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args):
            bound = [bind_input(x, name) for x, name in zip(args, names)]
            return self.model(*bound)

    examples = spec.get("example_inputs")
    if examples is not None:
        # Prefer the adapter-owned deterministic trace example over the passed
        # inputs: the regular CLI build passes the first fixture case, and a
        # dynamic graph cannot be traced from a size-1 example. Without
        # example_inputs the legacy behavior is unchanged.
        values = _example_values(torch, io, examples)
    else:
        if set(names) != set(inputs):
            raise ValueError("export example inputs do not match IOSpec")
        values = [inputs[name] for name in names]
    for node, value in zip(io["inputs"], values):
        # dynamic=True symbolizes every UNMARKED axis as well, so axes the
        # IOSpec declares static (unified mixed IO: position_ids leading 3,
        # pixel_values [512,1536]) must be pinned with mark_static or the
        # shipped AIR Data shapes degrade to numel-product-only symbolic dims
        # (reproduced: position_ids [-1,-1,-1], pixel_values [512,-1] in
        # work/startlux-validation/p6-static-input-contract/before-fix).
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
    # INPUTS (86 scalar Data nodes on the StartLux full model) that the
    # declared tensor IO contract does not include; specialize_float keeps
    # them as baked constants exactly like the dynamic=False route. This is a
    # documented dynamo compile option applied for this export call only and
    # restored afterwards (torch._dynamo.config.patch); the build step runs
    # in a single-purpose worker process.
    with torch._dynamo.config.patch(specialize_float=True):
        dynamo_export(*values, model=BoundModel(spec["model"]).eval(), export_path=str(directory),
                      export_name="model", dynamic=any(x.get("dynamic_dims") for x in io["inputs"]), config=config)
    torch.npu.synchronize()
    path = Path(directory) / "model.air"
    if not path.is_file():
        raise RuntimeError("TorchAir did not generate AIR")
    bindings = parse_bindings(Path(directory) / "dynamo.pbtxt")
    if [x[0] for x in bindings] != list(range(len(names))) or {x[1] for x in bindings} != set(names) or len(bindings) != len(names):
        raise ValueError(f"non-bijective graph input binding: {bindings}")
    by_name = {x["name"]: x for x in io["inputs"]}
    io["inputs"] = [dict(by_name[name], node=node) for _, name, node in bindings]
    # The protobuf text duplicates frozen weights; AIR plus IOSpec is the delivery artifact.
    (Path(directory) / "dynamo.pbtxt").unlink()
    return {"format": "air", "path": str(path)}, io
