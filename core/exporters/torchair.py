"""Export-owned identity markers bind logical names without patching TorchAir."""
from pathlib import Path
import re
from core.contracts import validate_io


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


def export(spec, inputs, directory):
    import torch
    import torch_npu
    from torch_npu.dynamo.torchair import dynamo_export, CompilerConfig
    from torchair import register_fx_node_ge_converter
    io = validate_io(spec["io"])
    names = [x["name"] for x in io["inputs"]]
    if set(names) != set(inputs):
        raise ValueError("export example inputs do not match IOSpec")

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
        return x

    class BoundModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args):
            bound = [bind_input(x, name) for x, name in zip(args, names)]
            return self.model(*bound)

    values = [inputs[name] for name in names]
    for node, value in zip(io["inputs"], values):
        for axis in node.get("dynamic_dims", []):
            torch._dynamo.mark_dynamic(value, axis)
    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
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
