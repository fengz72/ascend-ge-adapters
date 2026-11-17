"""Export the 9-input dynamic multimodal AIR (default work/build/onejev.air).

``io_spec()`` pins B=1 (and every shape-fixed axis) statically and leaves the
sequence width S and the patch count P as the only dynamic axes; the trace
example comes from ``example_inputs()`` (a tiny legal 1x4x4-grid request).
The graph contains ChunkGatedDeltaRule, so the process must start with the
compat proto LD_PRELOADed (``run.sh`` injects it).  Runs as a script
(``python3 model/export_air.py``) or as a module (``python3 -m model.export_air``).
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Pin the OMP pool BEFORE torch import: an unpinned all-core pool pays
# pool-sized sync overhead on every small parallel region (~40x slower).
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

ADAPTER_DIR = Path(__file__).resolve().parents[1]
DEPS_DIR = ADAPTER_DIR / "work" / "deps" / "py"

# Make the `model` PACKAGE importable whether this file runs as a script
# (python3 model/export_air.py) or as a module (python3 -m model.export_air).
# The order is forced deterministically: deps first, then the adapter root,
# and the script dir (model/) plus any CWD entry are dropped -- model/ holds
# the prototype model.py, which shadows the `model` package whenever the
# script dir is searched first (a stale ADAPTER_DIR entry deeper in sys.path
# once made the conditional insert skip and the prototype ran at import).
_this_dir = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path
               if p not in (_this_dir, "", str(ADAPTER_DIR), str(DEPS_DIR))]
sys.path[:0] = [str(DEPS_DIR), str(ADAPTER_DIR)]

import transformers

if transformers.__version__ != "5.17.0":
    raise RuntimeError(f"OneJev needs the task-local transformers 5.17.0 from {DEPS_DIR}, "
                       f"got {transformers.__version__} from {transformers.__file__}")

import torch

from model import modeling_onejev


def weights_dir():
    """ONEJEV_MODEL_DIR resolved once, loudly."""
    value = os.environ.get("ONEJEV_MODEL_DIR")
    if not value or not value.strip():
        raise ValueError("ONEJEV_MODEL_DIR is not set or is empty")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ValueError(f"ONEJEV_MODEL_DIR {path} is not a directory")
    return path


def _require_proto_preloaded():
    preloaded = os.environ.get("LD_PRELOAD", "")
    if "libchunk_gated_delta_rule_compat_proto.so" not in preloaded:
        raise RuntimeError(
            "the ChunkGatedDeltaRule compat proto is not LD_PRELOADed; start this "
            "process with LD_PRELOAD=work/deps/chunk_gated_delta_rule_proto/"
            "install/lib/libchunk_gated_delta_rule_compat_proto.so (run.sh injects it)")


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
    """(index, logical name, node name) of the exported Data nodes, sorted."""
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


_BIND_REGISTERED = False


def _register_bind_input():
    # identity op that tags each graph Data node with its logical IO name; the
    # converter asserts the node really is a root Data before renaming it.
    global _BIND_REGISTERED
    if _BIND_REGISTERED:
        return
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
    # (norm eps, attn scaling, ...) into implicit f64 graph INPUTS that the
    # declared tensor IO contract does not include; specialize_float keeps
    # them as baked constants exactly like the dynamic=False route.
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


def io_spec():
    # B is pinned STATIC on every [1, ...] input: a B!=1 request fails the
    # graph binding loudly instead of computing silently wrong scores.  The
    # only dynamic axes are the sequence width S and the patch count P; their
    # cross-tensor consistency (image-token span == P/4, feature_index range)
    # is a records-level contract enforced by generate_inputs.py.
    # image_grid_thw / cu_seqlens are NOT inputs: dead under the v1 patches
    # (see modeling_onejev.Decision), dynamo DCEs them out of the graph.
    return {
        "schema": "ge-io/1",
        "inputs": [
            {"name": "input_ids", "dtype": "int64", "shape": [1, -1],
             "dynamic_dims": [1]},
            {"name": "pixel_values", "dtype": "bfloat16", "shape": [-1, 1536],
             "dynamic_dims": [0]},
            {"name": "position_ids", "dtype": "int64", "shape": [3, 1, -1],
             "dynamic_dims": [2]},
            {"name": "interp_indices", "dtype": "int64", "shape": [-1, 4],
             "dynamic_dims": [0]},
            {"name": "interp_weights", "dtype": "float32", "shape": [-1, 4],
             "dynamic_dims": [0]},
            {"name": "vision_position_ids", "dtype": "int64", "shape": [-1, 2],
             "dynamic_dims": [0]},
            {"name": "feature_index", "dtype": "int64", "shape": [1, -1],
             "dynamic_dims": [1]},
        ],
        "outputs": [
            {"name": "scores", "dtype": "float32", "shape": [1, 26]},
        ],
    }


def example_inputs(folder, device):
    """A tiny legal single-image request (grid 1x4x8 -> P=32, 8 image tokens).

    Widths must stay >= 2 on every dynamic axis: a size-1 example makes dynamo
    specialize the axis into a static graph.  The example S must ALSO not equal
    any concrete dim a transposed view compares against (num_heads=8, kv=2,
    vision heads=12, head dims, ...): torchair's optimize_view evaluates
    meta_shape == srcshape elementwise, and an Eq(sym, hint) that comes out
    TRUE silently specializes the dynamic axis (S=8 == num_heads once baked S
    into the graph).  S=13 / P=32 collide with nothing.  The values are
    computed exactly like generate_inputs.py does.
    """
    from transformers import AutoConfig, Qwen3_5Model
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    config = AutoConfig.from_pretrained(str(folder), local_files_only=True)
    vision_start = config.vision_start_token_id
    image_token = config.image_token_id
    vision_end = config.vision_end_token_id
    merge = config.vision_config.spatial_merge_size
    side = int(config.vision_config.num_position_embeddings ** 0.5)

    grid = torch.tensor([[1, 4, 8]], dtype=torch.int64)
    patches = int(grid.prod())
    count = patches // merge ** 2
    ids = [vision_start] + [image_token] * count + [vision_end, 32, 33, 34]
    input_ids = torch.tensor([ids], dtype=torch.int64)
    width = input_ids.shape[1]
    feature_index = torch.tensor(
        [[-1] + list(range(count)) + [-1] * (width - 1 - count)], dtype=torch.int64)

    with torch.device("meta"):
        rope_model = Qwen3_5Model(config)
    position_ids, _ = rope_model.get_rope_index(
        input_ids, (input_ids == image_token).to(torch.int32), image_grid_thw=grid)

    interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=side, mode="bilinear", align_corners=True,
        spatial_merge_size=merge)
    vision_position_ids = get_vision_position_ids(grid, merge)

    pixel_values = torch.zeros((patches, 1536), dtype=torch.bfloat16)
    values = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "position_ids": position_ids,
        "interp_indices": interp_indices,
        "interp_weights": interp_weights.to(torch.float32),
        "vision_position_ids": vision_position_ids,
        "feature_index": feature_index,
    }
    return {name: value.to(device) for name, value in values.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Export the multimodal AIR (work/build/onejev.air)")
    parser.add_argument("--device", type=int, default=0,
                        help="in-process device index (ASCEND_RT_VISIBLE_DEVICES maps it)")
    parser.add_argument("--export-dir", type=Path,
                        default=ADAPTER_DIR / "work" / "build",
                        help="output directory for the AIR (default: work/build)")
    parser.add_argument("--export-name", default="onejev",
                        help="AIR file stem, written as <name>.air (default: onejev)")
    args = parser.parse_args()

    _require_proto_preloaded()
    import torch_npu

    device = torch.device(f"npu:{args.device}")
    torch_npu.npu.set_device(device)
    folder = weights_dir()
    model, ids = modeling_onejev.load_decision_model(folder, device)
    print(f"export_air: slot_token_ids {ids}")

    spec = {
        "model": model,
        "io": io_spec(),
        "example_inputs": example_inputs(folder, device),
    }
    args.export_dir.mkdir(parents=True, exist_ok=True)
    (args.export_dir / f"{args.export_name}.air").unlink(missing_ok=True)
    graph, _io = export_torchair(spec, args.export_dir, args.export_name)
    print(f"export_air: {args.export_name} {graph['path']} "
          f"({Path(graph['path']).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
