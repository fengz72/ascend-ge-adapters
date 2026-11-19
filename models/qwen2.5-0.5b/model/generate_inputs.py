"""Input pool generation for the Qwen2.5-0.5B prefix pipeline.

Draws the historical 100-case lognormal request pool (case -> shared prefix +
BATCH_SIZE private requests) with the EXACT NumPy RNG sequence of the old
generate_cases.py, so all historical workload records stay valid, and writes
the per-case graph inputs under work/inputs/:

* ``cases.json``           the case index (schema ``ge-cases/1``);
* ``case-XXXX/<name>.npy`` one npy per graph input (3 per case).

The layout is the shared-prefix one both platform prefix operators consume:
``segments = [prefix] + requests``, ``input_ids = concat(segments)``,
``position_ids`` restarts at ``len(prefix)`` per request, and
``actual_seq_lengths = cumsum(len(segment))`` (N+1 entries).  The bounds dtype
follows the platform: ``a3`` -> int64, ``a5`` -> int32.

Steps of this pipeline (input generation -> CPU golden -> graph export ->
runtime) never import each other; they interact only through the files under
work/.  This stage needs no NPU, no weights beyond config.json, and no
orchestration framework.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ADAPTER_DIR = Path(__file__).resolve().parents[1]

# Historical pool parameters (former config/model.yaml data.params): the request
# lengths are round(exp(normal(MU, SIGMA))) clipped to [prefix+1, MAX_LEN].
COUNT = 100
BATCH_SIZE = 16
MU = 4.997
SIGMA = 0.167
MIN_LEN = 1
MAX_LEN = 218
PREFIX_MIN = 20
PREFIX_MAX = 25
SEED = 0

BOUNDS_DTYPE = {"a3": np.int64, "a5": np.int32}

INPUT_NAMES = ("input_ids", "position_ids", "actual_seq_lengths")


def extra_rng(index, kind):
    """Independent extension RNG (seed*100003+index token stream stays untouched)."""
    key = f"qwen-batch-extension:{SEED}:{index}:{kind}".encode()
    return np.random.RandomState(int.from_bytes(hashlib.sha256(key).digest()[:4], "little"))


def generate_cases(vocab):
    """Yield (case_id, prefix, requests) with the historical RNG sequence.

    The first ten requests of every case come from the shared batch=10 stream
    (prefix length, lengths, tokens); rows 10..BATCH_SIZE-1 extend it with the
    sha256-derived per-case RNGs.  Token/prefix lengths are its exact draws.
    """
    rng = np.random.RandomState(SEED)
    for index in range(COUNT):
        prefix_len = int(rng.randint(PREFIX_MIN, PREFIX_MAX + 1))
        lengths = np.clip(np.round(np.exp(rng.normal(MU, SIGMA, size=10))),
                          max(MIN_LEN, prefix_len + 1), MAX_LEN).astype(int).tolist()
        token_rng = np.random.RandomState(SEED * 100003 + index)
        prefix = token_rng.randint(0, vocab, size=prefix_len).astype(np.int64).tolist()
        requests = [token_rng.randint(0, vocab, size=length - prefix_len).astype(np.int64).tolist()
                    for length in lengths]
        length_rng = extra_rng(index, "lengths")
        extra_tokens = extra_rng(index, "tokens")
        for _ in range(10, BATCH_SIZE):
            length = int(np.clip(np.round(np.exp(length_rng.normal(MU, SIGMA))),
                                 max(MIN_LEN, prefix_len + 1), MAX_LEN))
            requests.append(extra_tokens.randint(0, vocab, size=length - prefix_len)
                            .astype(np.int64).tolist())
        yield f"case-{index:04d}", prefix, requests


def packed_inputs(prefix, requests, dtype):
    """Shared-prefix input layout: IDs/positions int64, bounds per platform."""
    segments = [prefix] + requests
    positions = [list(range(len(prefix)))] + [
        list(range(len(prefix), len(prefix) + len(tokens))) for tokens in requests]
    return (np.asarray(sum(segments, []), dtype=np.int64),
            np.asarray(sum(positions, []), dtype=np.int64),
            np.cumsum([len(tokens) for tokens in segments], dtype=np.int64).astype(dtype))


def write_pool(out_dir, vocab, platform):
    """Write cases.json + per-case npy inputs under ``out_dir``.

    Returns ``(case_count, business_samples, sha256_hexdigest)``.  The digest is
    the historical pool fingerprint: per case, in case order, update with the
    C-order bytes of (input_ids, position_ids, actual_seq_lengths) in turn.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Old ge-records/3 artifacts must not coexist with the ge-cases/1 layout.
    for stale in ("inputs.bin", "records.json"):
        (out / stale).unlink(missing_ok=True)

    bounds_dtype = BOUNDS_DTYPE[platform]
    cases, digest, total = [], hashlib.sha256(), 0
    for case_id, prefix, requests in generate_cases(vocab):
        arrays = packed_inputs(prefix, requests, bounds_dtype)
        case_dir = out / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        inputs = []
        for name, array in zip(INPUT_NAMES, arrays):
            np.save(case_dir / f"{name}.npy", array)
            digest.update(array.tobytes(order="C"))
            inputs.append({"name": name, "path": f"{case_id}/{name}.npy"})
        cases.append({"id": case_id, "samples": len(requests), "inputs": inputs})
        total += len(requests)
        print(f"generate_inputs: {case_id} T={len(arrays[0])} N={len(requests)}")

    (out / "cases.json").write_text(json.dumps({"cases": cases}, indent=1) + "\n")
    return len(cases), total, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Generate the prefix pipeline inputs "
                                                 "(work/inputs)")
    parser.add_argument("--platform", choices=sorted(BOUNDS_DTYPE), default="a3",
                        help="bounds dtype target: a3 -> int64, a5 -> int32 "
                             "(default: %(default)s)")
    args = parser.parse_args()

    config = Path(os.environ["QWEN_MODEL_DIR"], "config.json").read_text()
    vocab = json.loads(config)["vocab_size"]

    out = ADAPTER_DIR / "work" / "inputs"
    count, total, sha256 = write_pool(out, vocab, args.platform)
    print(f"generate_inputs: {count} cases, {total} business samples; pool "
          f"sha256 {sha256[:12]}; inputs: {out}")


if __name__ == "__main__":
    main()
