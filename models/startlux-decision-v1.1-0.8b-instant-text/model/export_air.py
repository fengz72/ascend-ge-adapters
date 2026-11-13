"""Export the 2-input dynamic text AIR (default work/build/text.air).

``io_spec()`` pins B=1 statically and leaves S as the only dynamic axis; the
trace example comes from ``example_inputs()``.  The graph contains
ChunkGatedDeltaRule, so the process must start with the compat proto
LD_PRELOADed (``run.sh`` injects it).  Runs as a script
(``python3 model/export_air.py``) or as a module (``python3 -m model.export_air``).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# Pin the OMP pool BEFORE torch import: an unpinned all-core pool pays
# pool-sized sync overhead on every small parallel region (~40x slower).
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

ADAPTER_DIR = Path(__file__).resolve().parents[1]

# make the model package importable whether this file runs as a script
# (python3 model/export_air.py) or as a module (python3 -m model.export_air)
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

import torch
from model import modeling_text  # noqa: E402


def weights_dir():
    """STARTLUX_MODEL_DIR resolved once, loudly."""
    value = os.environ.get("STARTLUX_MODEL_DIR")
    if not value or not value.strip():
        raise ValueError("STARTLUX_MODEL_DIR is not set or is empty")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"STARTLUX_MODEL_DIR {path} is not a directory")
    return path


def _reorder_data_nodes_first():
    # torchair 2.12.0 bakes weight inputs POSITIONALLY; with dynamic=True the
    # SymInt chain shifts the Data block, so root Data nodes (no inputs) are
    # stably moved to the front by index to restore the bake's assumed
    # position<->input alignment.
    from torchair.ge._ge_graph import get_default_ge_graph

    ops = get_default_ge_graph().op

    def order(op):
        if op.type == "Data":
            return (0, op.attr["index"].i if "index" in op.attr else 0)
        return (1, 0)

    ops.sort(key=order)


def parse_bindings(pbtxt):
    """(index, logical name) pairs of the exported Data nodes, sorted."""
    result, node_lines, is_data_node, line_start = [], None, False, True
    with open(pbtxt, encoding="utf-8") as stream:
        while True:
            piece = stream.readline(65536)
            if not piece:
                break
            at_start, line_start = line_start, piece.endswith("\n")
            if at_start and piece.startswith("node {"):
                node_lines, is_data_node = [], True
                continue
            if node_lines is None:
                continue
            if at_start and piece.startswith("}"):
                if is_data_node:
                    text = "".join(node_lines)
                    if re.search(r'op:\s*"Data"', text):
                        name = re.search(r'name:\s*"([^"]+)"', text)
                        index = re.search(r'key:\s*"index".*?i:\s*(\d+)', text, re.S)
                        logical = re.search(
                            r'key:\s*"_ge_adapter_name".*?s:\s*"([^"]+)"', text, re.S
                        )
                        if not all((name, index, logical)):
                            raise ValueError("incomplete exported Data binding")
                        result.append(
                            (int(index.group(1)), logical.group(1), name.group(1))
                        )
                node_lines = None
                continue
            if is_data_node:
                node_lines.append(piece)
                match = re.match(r'\s*op:\s*"([^"]+)"', piece) if at_start else None
                if match and match.group(1) != "Data":
                    is_data_node = False
                    node_lines.clear()
    return sorted(result)


def _register_bind_input():
    # identity op that tags each graph Data node with its logical IO name; the
    # converter asserts the node really is a root Data before renaming it.
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


def export_torchair(spec, directory, export_name):
    from torch_npu.dynamo.torchair import CompilerConfig, dynamo_export

    directory = Path(directory)
    _register_bind_input()
    io = spec["io"]
    names = [x["name"] for x in io["inputs"]]

    class BoundModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args):
            bound = [
                torch.ops.ge_adapters.bind_input(x, name)
                for x, name in zip(args, names)
            ]
            return self.model(*bound)

    values = [spec["example_inputs"][node["name"]] for node in io["inputs"]]
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
    with torch._dynamo.config.patch(specialize_float=True):
        dynamo_export(
            *values,
            model=BoundModel(spec["model"]).eval(),
            export_path=str(directory),
            export_name=export_name,
            dynamic=any(x.get("dynamic_dims") for x in io["inputs"]),
            config=config,
        )
    torch.npu.synchronize()
    path = directory / f"{export_name}.air"
    if not path.is_file():
        raise RuntimeError("TorchAir did not generate AIR")
    bindings = parse_bindings(directory / "dynamo.pbtxt")
    if [x[0] for x in bindings] != list(range(len(names))) or {
        x[1] for x in bindings
    } != set(names):
        raise ValueError(f"non-bijective graph input binding: {bindings}")
    by_name = {x["name"]: x for x in io["inputs"]}
    io["inputs"] = [dict(by_name[name], node=node) for _, name, node in bindings]
    # The protobuf text duplicates frozen weights; AIR plus IOSpec is the
    # delivery artifact.
    (directory / "dynamo.pbtxt").unlink()
    return {"format": "air", "path": str(path)}, io


# ---------------------------------------------------- loading the model


def load_text_model(folder, device):
    # Only model.language_model.* is consumed (strict load); model.visual.*/mtp.*
    # are recognized skips, anything else is an error.
    from transformers import Qwen3_5Config
    from safetensors.torch import load_file

    config = Qwen3_5Config.from_pretrained(str(folder))
    text_config = config.text_config
    if not getattr(text_config, "tie_word_embeddings", False):
        raise ValueError(
            "decision_head is derived from embed_tokens; this "
            "checkpoint must have tie_word_embeddings=True"
        )
    letters = json.loads((folder / "decision_config.json").read_text())[
        "letter_token_ids"
    ]
    if len(letters) != 26 or len(set(letters)) != 26:
        raise ValueError("decision_config letter_token_ids must be 26 unique ids")

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = modeling_text.StartLuxTextModel(text_config)
    finally:
        torch.set_default_dtype(previous)

    state = {}
    for shard in sorted(folder.glob("*.safetensors")):
        state.update(load_file(str(shard)))
    prefix = "model.language_model."
    allowed_skips = ("model.visual.", "mtp.")
    text_state = {}
    for key in state:
        if key.startswith(prefix):
            text_state[key.removeprefix(prefix)] = state[key]
        elif not key.startswith(allowed_skips):
            raise RuntimeError(
                f"unexpected checkpoint tensor {key!r}; expected "
                f"{prefix}* plus the skipped {allowed_skips}"
            )
    model.load_state_dict(text_state, strict=True)
    model.install_decision_head(letters)

    model = model.to(device).requires_grad_(False).eval()
    return model


def io_spec():
    # B is pinned STATIC (axis 0 declared 1): a B!=1 request fails the graph
    # binding loudly instead of computing silently wrong scores; the only
    # dynamic axis left is S.  actual_seq_lengths is not an input -- under
    # the B=1 true-length contract the laid-out width IS the true length,
    # and every length consumer inside the model is shape-derived.
    return {
        "schema": "ge-io/1",
        "inputs": [
            {
                "name": "input_ids",
                "dtype": "int64",
                "shape": [1, -1],
                "dynamic_dims": [1],
            },
            {
                "name": "position_ids",
                "dtype": "int64",
                "shape": [3, 1, -1],
                "dynamic_dims": [2],
            },
        ],
        "outputs": [
            {"name": "scores", "dtype": "float32", "shape": [1, 26]},
        ],
    }


def example_inputs(device):
    # width must stay >= 2: a size-1 example makes dynamo specialize the S
    # axis into a static graph; the values just need to be legal for tracing.
    width = 65
    return {
        "input_ids": torch.zeros((1, width), dtype=torch.int64, device=device),
        "position_ids": torch.tile(
            torch.arange(width, dtype=torch.int64, device=device)[None, None, :],
            (3, 1, 1),
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export the text-only AIR (work/build/text.air)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="in-process device index (ASCEND_RT_VISIBLE_DEVICES maps it)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=ADAPTER_DIR / "work" / "build",
        help="output directory for the AIR (default: work/build)",
    )
    parser.add_argument(
        "--export-name",
        default="text",
        help="AIR file stem, written as <name>.air (default: text)",
    )
    args = parser.parse_args()

    import torch_npu

    device = torch.device(f"npu:{args.device}")
    torch_npu.npu.set_device(device)
    model = load_text_model(weights_dir(), device)

    spec = {
        "model": model,
        "io": io_spec(),
        "example_inputs": example_inputs(device),
    }
    args.export_dir.mkdir(parents=True, exist_ok=True)
    (args.export_dir / f"{args.export_name}.air").unlink(missing_ok=True)
    graph, _io = export_torchair(spec, args.export_dir, args.export_name)
    print(
        f"export_air: {args.export_name} {graph['path']} "
        f"({Path(graph['path']).stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
