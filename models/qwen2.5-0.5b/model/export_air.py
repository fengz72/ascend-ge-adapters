"""Export the 3-input dynamic packed AIR (work/build/packed.air).

Every graph input is one-dimensional with axis 0 dynamic: ``input_ids [T]``
int64, ``position_ids [T]`` int64 and ``actual_seq_lengths [N+1]`` (int64 on a3,
int32 on a5), so one AIR serves the whole variable-length pool and the score
rows ``[N, 8]`` stay dynamic with it.  ``dynamic=True`` symbolizes every axis on
its own and ``frozen_parameter`` bakes the FP16 weights into the graph, so the
exported Data nodes are exactly the three example inputs, in positional order --
which is how the executor binds the records.  The platform prefix operator must
be importable, i.e. the process runs with work/deps/env.sh sourced (``run.sh``
injects it).
"""

import argparse
import json
import os
from pathlib import Path

import torch

import modeling_qwen

ADAPTER_DIR = Path(__file__).resolve().parents[1]

BOUNDS_DTYPE = {"a3": torch.int64, "a5": torch.int32}


def load_model(folder, device, platform):
    # The checkpoint stores the trunk under model.* and ties lm_head to
    # embed_tokens, so the score head is derived after a strict trunk load.
    from safetensors.torch import load_file

    config = json.loads((folder / "config.json").read_text())
    torch.set_default_dtype(torch.float16)
    model = modeling_qwen.QwenPackedModel(config, platform)

    state = {
        key.removeprefix("model."): value
        for key, value in load_file(str(folder / "model.safetensors")).items()
    }
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    tokens = json.loads((ADAPTER_DIR / "config" / "target_tokens.json").read_text())
    model.install_score_head(tokens["token_ids"])
    return model.to(device)


def example_inputs(device, bounds_dtype):
    # Every dynamic axis must trace at size >= 2: a size-1 example makes dynamo
    # specialize the axis into a static graph.  A 4-token prefix followed by two
    # 8-token requests gives T=20 over 3 segment bounds; the values only need to
    # be legal for tracing.
    positions = torch.cat((torch.arange(4), torch.arange(4, 12), torch.arange(4, 12)))
    return (
        torch.arange(20, dtype=torch.int64, device=device),
        positions.to(device=device, dtype=torch.int64),
        torch.tensor((4, 12, 20), dtype=bounds_dtype, device=device),
    )


def main():
    parser = argparse.ArgumentParser(description="Export work/build/packed.air")
    parser.add_argument(
        "--platform",
        choices=sorted(BOUNDS_DTYPE),
        default="a3",
        help="prefix operator and bounds dtype: a3 -> int64, a5 -> int32 "
        "(default: %(default)s)",
    )
    args = parser.parse_args()

    import torch_npu
    from torch_npu.dynamo.torchair import CompilerConfig, dynamo_export

    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    model = load_model(Path(os.environ["QWEN_MODEL_DIR"]), device, args.platform)

    config = CompilerConfig()
    config.experimental_config.frozen_parameter = True
    directory = ADAPTER_DIR / "work" / "build"
    directory.mkdir(parents=True, exist_ok=True)
    dynamo_export(
        *example_inputs(device, BOUNDS_DTYPE[args.platform]),
        model=model,
        export_path=str(directory),
        export_name="packed",
        dynamic=True,
        config=config,
    )
    # The protobuf text duplicates the frozen weights; the AIR is the artifact.
    (directory / "dynamo.pbtxt").unlink()
    path = directory / "packed.air"
    print(f"export_air: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
