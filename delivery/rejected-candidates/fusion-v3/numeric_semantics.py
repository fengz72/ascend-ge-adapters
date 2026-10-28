#!/usr/bin/env python3
"""CPU NumPy FP32 residual/LN semantic check, not CANN kernel execution."""
from pathlib import Path
import json
import numpy as np


def main():
    rng = np.random.default_rng(20261026)
    results = []
    for sequence, scale in ((6, 1.0), (8, 1e-7), (13, 10.0)):
        x = (rng.normal(size=(1, sequence, 1024)) * scale).astype(np.float32)
        y = (rng.normal(size=x.shape) * scale).astype(np.float32)
        gamma = rng.uniform(.5, 1.5, size=1024).astype(np.float32)
        beta = rng.normal(size=1024).astype(np.float32)
        residual = np.add(x, y, dtype=np.float32)
        centered = residual - residual.mean(axis=-1, keepdims=True, dtype=np.float32)
        variance = np.square(centered).mean(axis=-1, keepdims=True, dtype=np.float32)
        actual = centered / np.sqrt(variance + np.float32(1e-12)) * gamma + beta
        reference = residual.astype(np.float64)
        reference = (reference - reference.mean(axis=-1, keepdims=True)) / np.sqrt(reference.var(axis=-1, keepdims=True) + float(np.float32(1e-12)))
        reference = reference * gamma.astype(np.float64) + beta.astype(np.float64)
        error = float(np.max(np.abs(actual.astype(np.float64) - reference)))
        assert actual.dtype == np.float32 and np.isfinite(actual).all() and error < 1e-5
        results.append({'sequence': sequence, 'scale': scale, 'max_abs_vs_fp64': error})
    x = np.linspace(-1e-7, 1e-7, 1024, dtype=np.float32)
    good = x / np.sqrt(x.var(dtype=np.float32) + np.float32(1e-12))
    bad = x / np.sqrt(x.var(dtype=np.float32) + np.float32(1e-5))
    wrong_epsilon_difference = float(np.max(np.abs(good - bad)))
    assert wrong_epsilon_difference > .09
    report = {'passed': True, 'numpy': np.__version__, 'npu_initialized': False,
              'scope': 'NumPy semantic check only, not CANN AddLayerNorm or full BGE',
              'numeric_cases': results, 'wrong_epsilon_detectable_max_diff': wrong_epsilon_difference}
    Path(__file__).with_name('numeric-semantics-result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
