"""Frozen BGE input identity and explicit performance-scenario checks; no device APIs."""
import hashlib
import json
import os
from pathlib import Path

CASE_IDS = tuple(f"case-{i:04d}" for i in range(5))
META_SHA = "ba2afe4a2ab5c867174b5f515966f77413e79409d113c5a305fff7130bb3561f"
INPUT_SHA = "14ccd8bc7e4fb3f6437fbb7c1b47fc710c2cd2eba9ce220d1a1715a8936f7a44"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


PROJECT_ROOT = Path("/export/home/ext.hejun23/workspace/bge-opt-100")
_REFERENCE_OVERRIDE = None
_ORIGINAL_OVERRIDE = None


def add_reference_arguments(parser):
    parser.add_argument("--reference-dir", type=Path, help="Directory containing frozen cpu-golden-manifest.json")
    parser.add_argument("--original-inputs", type=Path, help="Frozen original inputs.bin or its directory")


def configure(args):
    global _REFERENCE_OVERRIDE, _ORIGINAL_OVERRIDE
    _REFERENCE_OVERRIDE = getattr(args, "reference_dir", None)
    _ORIGINAL_OVERRIDE = getattr(args, "original_inputs", None)


def reference_dir(path=None):
    return Path(path or _REFERENCE_OVERRIDE or os.environ.get("BGE_REFERENCE_DIR") or PROJECT_ROOT / "materials")


def original_inputs():
    path = Path(_ORIGINAL_OVERRIDE or os.environ.get("BGE_ORIGINAL_INPUTS") or PROJECT_ROOT / "materials/original-inputs/inputs.bin")
    return path / "inputs.bin" if path.is_dir() else path


def metadata(path=None):
    directory = reference_dir(path)
    file = directory / "cpu-golden-manifest.json"
    assert sha(file) == META_SHA, "independent CPU manifest differs"
    return json.loads(file.read_text())


def workload(plan):
    ids = [c["id"] for c in plan["inputs"]["cases"]]
    assert ids and len(ids) == len(set(ids)), "empty or duplicate case pool"
    assert set(ids) <= set(CASE_IDS), "unknown original BGE case ID"
    declared = plan.get("workload")
    if declared is None:
        assert ids == list(CASE_IDS), "legacy plan must retain the full original random-five pool"
        return {"kind": "random_cases", "case_ids": ids, "sampling": "uniform_with_replacement"}
    assert declared.get("case_ids") == ids, "workload metadata and actual pool differ"
    if declared.get("kind") == "fixed_case":
        assert len(ids) == 1 and declared.get("sampling") == "fixed", "fixed workload requires an actual one-case pool"
    else:
        assert declared.get("kind") == "random_cases", "unknown workload kind"
        assert ids == list(CASE_IDS) and declared.get("sampling") == "uniform_with_replacement", "random workload must retain the original five cases"
    return dict(declared)


def check_plan_inputs(plan, path=None):
    """Bind each tensor and the complete actual packed pool to the original bytes."""
    assert plan["backend"] == "ge_session" and plan["options"]["graph_run_mode"] == 0
    directory = reference_dir(path)
    meta = metadata(directory)
    original_path = original_inputs()
    assert sha(original_path) == INPUT_SHA, "original packed input identity differs"
    original = original_path.read_bytes()
    names = [t["name"] for t in meta["model_io"]["inputs"]]
    originals = {}
    position = 0
    for cid in CASE_IDS:
        size = sum(meta["cases"][cid]["inputs"][name]["bytes"] for name in names)
        originals[cid] = original[position:position + size]
        position += size
    assert position == len(original)
    scenario = workload(plan)
    actual_path = Path(plan["inputs"]["file"])
    raw = actual_path.read_bytes()
    offset = 0
    checks = {}
    for case in plan["inputs"]["cases"]:
        cid = case["id"]
        assert case["samples"] == 1 and case["output_capacities"] == {"scores": 40448}
        assert [t["name"] for t in case["tensors"]] == names, "tensor order/names differ from original IO"
        checks[cid] = {}
        for tensor in case["tensors"]:
            ref = meta["cases"][cid]["inputs"][tensor["name"]]
            assert tensor["dtype"] == ref["dtype"] and tensor["shape"] == ref["shape"]
            assert tensor["bytes"] == ref["bytes"] and type(tensor["offset"]) is int and tensor["offset"] == offset
            data = raw[offset:offset + tensor["bytes"]]
            digest = hashlib.sha256(data).hexdigest()
            assert len(data) == tensor["bytes"] and digest == ref["content_sha256"], (cid, tensor["name"], "input bytes differ")
            checks[cid][tensor["name"]] = {"passed": True, "content_sha256": digest}
            offset += len(data)
    assert offset == len(raw), "unreferenced packed input bytes"
    assert raw == b"".join(originals[cid] for cid in scenario["case_ids"]), "packed selected input differs from the original case bytes"
    return {"passed": True, "workload": scenario, "input_binary": str(actual_path),
            "input_binary_sha256": sha(actual_path), "input_bytes": len(raw), "cases": checks,
            "reference_manifest_sha256": META_SHA, "original_input_sha256": INPUT_SHA}
