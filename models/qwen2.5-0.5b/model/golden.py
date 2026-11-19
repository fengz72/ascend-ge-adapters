"""Independent CPU FP32 golden for the Qwen2.5-0.5B prefix pipeline.

The independent reference stage: loads the SAME weights folder (QWEN_MODEL_DIR)
as an original ``AutoModelForCausalLM`` float32 CPU model -- no method patches,
no model-package import -- and scores every case of ``work/inputs``.  The
packed pool holds one shared prefix followed by N request segments; the N
business rows are reconstructed as ``prefix + request_i`` from ``input_ids``
and ``actual_seq_lengths`` (no re-tokenization).  Each case's rows are padded
into one ``[N, W]`` batch with ``attention_mask`` / ``position_ids``; the
scoring positions are the unique true last positions
``(attention_mask.sum(dim=1) - 1).unique(sorted=True)``, so only those logits
are materialized, and ``logits[i, position(len(prefix)+len(tokens_i)-1),
token_ids]`` gives the ``[N, 8]`` score rows.  ``token_ids`` come from the
adapter's ``config/target_tokens.json``.

Output mirrors the runtime infer layout (``tools/compare.py`` reads both sides):
``work/golden/<case_id>/output_0.npy`` with the ``[N, 8]`` float32 scores + a
``work/golden/result.json`` case index whose output entries are position-named
``output_0`` (the runtime never forwards GE's semantic output name).

This stage needs no NPU and no LD_PRELOAD; its only inputs are the files written
by generate_inputs.py.
"""

import json
import os
from pathlib import Path

# Pin the CPU thread pools BEFORE torch imports: OMP sizes its pool at library
# init, and an unpinned pool makes every small parallel region pay pool-sized
# sync overhead (the reference delivery measured ~40x on a many-core host).
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import torch

ADAPTER_DIR = Path(__file__).resolve().parents[1]


def _load_cases(inputs):
    """cases.json + per-case npy inputs -> [(case_id, [prefix+request_i rows])]."""
    index = json.loads((inputs / "cases.json").read_text())
    cases = []
    for case in index["cases"]:
        paths = {item["name"]: item["path"] for item in case["inputs"]}
        ids = np.load(inputs / paths["input_ids"]).astype(np.int64)
        bounds = np.load(inputs / paths["actual_seq_lengths"]).astype(np.int64)
        prefix = ids[:bounds[0]]
        rows = [np.concatenate([prefix, ids[bounds[i]:bounds[i + 1]]])
                for i in range(len(bounds) - 1)]
        cases.append((case["id"], rows))
    return cases


def _score(model, rows, ids_wanted):
    """One case -> [N, len(ids_wanted)] float32 scores (HF padded-batch path)."""
    width = max(len(row) for row in rows)
    batch = torch.zeros((len(rows), width), dtype=torch.int64)
    mask = torch.zeros((len(rows), width), dtype=torch.int64)
    positions_in = torch.zeros((len(rows), width), dtype=torch.int64)
    for i, row in enumerate(rows):
        batch[i, :len(row)] = torch.from_numpy(row)
        mask[i, :len(row)] = 1
        positions_in[i, :len(row)] = torch.arange(len(row))
    keep = (mask.sum(dim=1) - 1).unique(sorted=True)
    with torch.inference_mode():
        logits = model(input_ids=batch, attention_mask=mask, position_ids=positions_in,
                       use_cache=False, logits_to_keep=keep).logits
    column = {int(position): index for index, position in enumerate(keep.tolist())}
    wanted = torch.tensor(ids_wanted)
    return torch.stack([logits[i, column[len(row) - 1], wanted]
                        for i, row in enumerate(rows)]).numpy()


def write_case_scores(out, case_id, tensor):
    """Write ``<out>/<case_id>/output_0.npy`` and return its result.json entry.

    The position name ``output_0`` mirrors the runtime infer layout; the path is
    relative to the directory holding result.json.  numpy's ``np.save`` does not
    create parent directories, so the per-case directory is made here.
    """
    relative = f"{case_id}/output_0.npy"
    (out / case_id).mkdir(parents=True, exist_ok=True)
    np.save(out / relative, tensor)
    return {"name": "output_0", "dtype": str(tensor.dtype),
            "shape": list(tensor.shape), "path": relative, "bytes": tensor.nbytes}


def main():
    inputs = ADAPTER_DIR / "work" / "inputs"
    out = ADAPTER_DIR / "work" / "golden"
    cases = _load_cases(inputs)
    ids_wanted = json.loads(
        (ADAPTER_DIR / "config" / "target_tokens.json").read_text())["token_ids"]

    from transformers import AutoModelForCausalLM

    # from_pretrained already yields a CPU float32 model in eval mode.
    model = AutoModelForCausalLM.from_pretrained(
        os.environ["QWEN_MODEL_DIR"], dtype=torch.float32, local_files_only=True)

    out.mkdir(parents=True, exist_ok=True)
    # The previous ge-compare/1 blob must not coexist with the npy layout.
    (out / "outputs.bin").unlink(missing_ok=True)
    result = {"status": "success", "cases": []}
    for case_id, rows in cases:
        tensor = _score(model, rows, ids_wanted)
        entry = write_case_scores(out, case_id, tensor)
        result["cases"].append({"id": case_id, "outputs": [entry]})
        print(f"golden: {case_id} N={tensor.shape[0]} scores {list(tensor.shape)}")

    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"golden: {len(result['cases'])} cases; output: {out}")


if __name__ == "__main__":
    main()
