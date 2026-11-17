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

Output (the ge-compare/1 layout ``tools/compare.py`` reads): ``work/golden/
outputs.bin`` (concatenated ``[N, 8]`` float32 score tensors) + ``work/golden/
result.json`` with the outputs index.

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
    """records.json + inputs.bin -> [(case_id, [prefix+request_i int64 rows])]."""
    records = json.loads((inputs / "records.json").read_text())
    blob = (inputs / "inputs.bin").read_bytes()
    cases = []
    for case in records["cases"]:
        tensors = {t["name"]: t for t in case["tensors"]}
        ids_ref, bounds_ref = tensors["input_ids"], tensors["actual_seq_lengths"]
        ids = np.frombuffer(blob, np.int64, count=ids_ref["shape"][0],
                            offset=ids_ref["offset"])
        bounds = np.frombuffer(blob, np.dtype(bounds_ref["dtype"]),
                               count=bounds_ref["shape"][0],
                               offset=bounds_ref["offset"]).astype(np.int64)
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
    result = {"status": "success", "outputs": {}}
    with (out / "outputs.bin").open("wb") as stream:
        for case_id, rows in cases:
            tensor = _score(model, rows, ids_wanted)
            data = tensor.tobytes(order="C")
            result["outputs"][case_id] = [{"instance": 0, "tensors": [{
                "name": "scores", "dtype": "float32", "shape": list(tensor.shape),
                "offset": stream.tell(), "bytes": len(data)}]}]
            stream.write(data)
            print(f"golden: {case_id} N={tensor.shape[0]} scores {list(tensor.shape)}")

    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"golden: {len(result['outputs'])} cases; output: {out}")


if __name__ == "__main__":
    main()
