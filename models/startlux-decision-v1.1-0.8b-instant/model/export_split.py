"""Two-stage split export: dynamic vision AIR + text AIR.

Exports the StartLux delivery as TWO dynamic AIRs (docs: the split decision,
2026-11-10): the vision stage (any image count / any official grid; tables as
inputs) and the text stage (image_features as an input, injected like the
official masked_scatter).  Reuses export_air's proven machinery (bind_input,
Data re-partition, dynamic-axis pinning, frozen parameters).  The text graph
contains ChunkGatedDeltaRule, so the process must run with the prototype
.so LD_PRELOADed (run.sh does this; _require_proto_preloaded asserts it).
No io is embedded into the AIRs: the pipeline runtime binds inputs
positionally in the fixed order of vision_io_spec / text_io_spec.
"""
import argparse
import json
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]   # the model root

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model import export_air, modeling_multimodal as mm               # noqa: E402

import torch                                                           # noqa: E402

WORK = MODEL_DIR / "work"
BUILD = WORK / "build"


def main():
    parser = argparse.ArgumentParser(description="Export the StartLux split AIRs "
                                                 "(vision + text)")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    export_air._require_proto_preloaded()
    unified, _placement = mm.load(args.device)

    BUILD.mkdir(parents=True, exist_ok=True)
    for name in ("vision", "text"):
        stale = BUILD / f"{name}.air"
        if stale.exists():
            stale.unlink()

    vision_spec = {
        "format": "torch", "model": unified.model.visual,
        "io": mm.vision_io_spec(),
        "example_inputs": {key: torch.from_numpy(value).to(f"npu:{args.device}")
                           for key, value in mm.vision_example_inputs().items()},
    }
    graph, _io = export_air.export_torchair(vision_spec, BUILD,
                                            export_name="vision")
    vision_path = graph["path"]

    text_spec = {
        "format": "torch", "model": unified,
        "io": mm.text_io_spec(),
        "example_inputs": {key: torch.from_numpy(value).to(f"npu:{args.device}")
                           for key, value in mm.text_example_inputs().items()},
    }
    graph, _io = export_air.export_torchair(text_spec, BUILD,
                                            export_name="text")
    text_path = graph["path"]

    manifest = {"schema": "startlux-split-air/1",
                "vision": {"path": vision_path, "bytes": Path(vision_path).stat().st_size},
                "text": {"path": text_path, "bytes": Path(text_path).stat().st_size},
                "vision_io": mm.vision_io_spec(), "text_io": mm.text_io_spec()}
    (BUILD / "split.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"export_split: vision {vision_path} ({manifest['vision']['bytes']} bytes), "
          f"text {text_path} ({manifest['text']['bytes']} bytes); manifest: "
          f"{BUILD / 'split.json'}")


if __name__ == "__main__":
    main()
