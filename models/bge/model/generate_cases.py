import argparse
import json
from pathlib import Path
import sys

import numpy as np


# Not re-derived from the graph: reading its header costs ~3 GB RSS (the weights
# are inline in the protobuf).  Do not sort or reorder.
DTYPES = {
    "input_ids": np.int64,
    "attention_mask": np.int64,
    "long_cid3_raw": np.int64,
    "long_action": np.int64,
    "long_time_bucket": np.int64,
    "stat_dense_raw": np.int64,
    "stat_ratio": np.float32,
    "stat_count": np.int64,
}

BATCH = 1
TEXT_LEN_MIN = 6
TEXT_LEN_MAX = 13
TOKEN_ID_MIN = 101
TOKEN_ID_MAX = 12419
HISTORY_LEN = 128         # static in the graph, so not random
CID3_MIN = 1              # 0 means padding
CID3_MAX = 42689
ACTION_MIN = 0
ACTION_MAX = 2
STAT_WIDTH = 20
STAT_COUNT_MIN = 1
STAT_COUNT_MAX = 4
STAT_DENSE_MIN = 1
STAT_DENSE_MAX = 43887
STAT_RATIO_MAX = 0.973


def make_case(rng):
    """One case -> ``{feature name: array}``, keys in ``DTYPES`` order.

    Draw order is part of the seed contract.  Structural invariants matter: the
    graph runs FP16 internally, and an all-zero mask row has no unmasked position.
    """
    text_len = int(rng.integers(TEXT_LEN_MIN, TEXT_LEN_MAX + 1))
    history_valid = int(rng.integers(1, HISTORY_LEN + 1))
    stat_count = int(rng.integers(STAT_COUNT_MIN, STAT_COUNT_MAX + 1))

    long_cid3_raw = np.zeros((BATCH, HISTORY_LEN), dtype=np.int64)
    long_cid3_raw[0, :history_valid] = rng.integers(
        CID3_MIN, CID3_MAX + 1, history_valid, dtype=np.int64)
    long_action = np.zeros((BATCH, HISTORY_LEN), dtype=np.int64)
    long_action[0, :history_valid] = rng.integers(
        ACTION_MIN, ACTION_MAX + 1, history_valid, dtype=np.int64)
    # A recency index over the valid prefix, not a free value.
    long_time_bucket = np.zeros((BATCH, HISTORY_LEN), dtype=np.int64)
    long_time_bucket[0, :history_valid] = np.arange(1, history_valid + 1, dtype=np.int64)
    stat_dense_raw = np.zeros((BATCH, STAT_WIDTH), dtype=np.int64)
    stat_dense_raw[0, :stat_count] = rng.integers(
        STAT_DENSE_MIN, STAT_DENSE_MAX + 1, stat_count, dtype=np.int64)
    stat_ratio = np.zeros((BATCH, STAT_WIDTH), dtype=np.float32)
    stat_ratio[0, :stat_count] = rng.random(stat_count) * STAT_RATIO_MAX

    return {
        "input_ids": rng.integers(TOKEN_ID_MIN, TOKEN_ID_MAX + 1, (BATCH, text_len),
                                  dtype=np.int64),
        "attention_mask": np.ones((BATCH, text_len), dtype=np.int64),
        "long_cid3_raw": long_cid3_raw,
        "long_action": long_action,
        "long_time_bucket": long_time_bucket,
        "stat_dense_raw": stat_dense_raw,
        "stat_ratio": stat_ratio,
        "stat_count": np.full((BATCH,), stat_count, dtype=np.int64),
    }


def write_pool(output_dir, count, seed):
    """Write the pool; return the case id list.

    ``cases.json`` omits dtype and shape because ``ge_execute`` reads them from each
    NPY header, so they cannot drift.
    """
    rng = np.random.default_rng(seed)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    case_ids = []
    for index in range(count):
        case_id = f"case-{index:04d}"
        case_ids.append(case_id)
        folder = root / case_id
        folder.mkdir(exist_ok=True)
        for name, value in make_case(rng).items():
            np.save(folder / f"{name}.npy", value)
    payload = {"cases": [
        {"id": case_id, "samples": BATCH,
         "inputs": [{"name": name, "path": f"{case_id}/{name}.npy"} for name in DTYPES]}
        for case_id in case_ids
    ]}
    (root / "cases.json").write_text(json.dumps(payload, indent=1) + "\n")
    return case_ids


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a reproducible random BGE input pool (case-NNNN NPY + cases.json)."
    )
    parser.add_argument("--output-dir", required=True,
                        help="input pool directory; case-NNNN/ folders are written into it")
    parser.add_argument("--cases", type=int, default=5,
                        help="number of cases to generate (default 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed; same seed + same numpy version = same bytes (default 42)")
    args = parser.parse_args(argv)
    case_ids = write_pool(args.output_dir, args.cases, args.seed)
    print(json.dumps({"status": "ok", "cases": len(case_ids), "case_ids": case_ids,
                      "seed": args.seed,
                      "directory": str(Path(args.output_dir).resolve()),
                      "cases_json": str((Path(args.output_dir) / "cases.json").resolve())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
