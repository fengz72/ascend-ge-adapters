#!/usr/bin/env python3
"""Read-only evidence of GE tensor descriptors in a compiled final graph."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from inspect_ge_precision import blocks


def quoted(text, field):
    m = re.search(r'(?m)^\s*' + field + r':\s*"((?:\\.|[^"\\])*)"', text)
    return m.group(1) if m else None


def descriptor(text):
    dtype = re.search(r'(?m)^\s*dtype:\s*(\S+)', text)
    shape = next(blocks(text, 'shape'), '')
    return {'dtype': dtype.group(1) if dtype else None,
            'dims': [int(x) for x in re.findall(r'(?m)^\s*dim:\s*(-?\d+)', shape)]}


def parse(path):
    text = path.read_text(errors='replace')
    result = []
    for op in blocks(text, 'op'):
        attrs = {}
        for attr in blocks(op, 'attr'):
            key = quoted(attr, 'key')
            if not key:
                continue
            if key in ['epsilon', 'begin_norm_axis', 'begin_params_axis', 'src_type', 'dst_type', 'axis'] or any(
                    x in key.lower() for x in ['kernel', 'precision', 'lowering', 'impl_mode', 'core_type']):
                attrs[key] = attr[:6000]
        result.append({'name': quoted(op, 'name'), 'type': quoted(op, 'type'),
                       'input_refs': re.findall(r'(?m)^    input:\s*"([^"]*)"', op),
                       'inputs': [descriptor(d) for d in blocks(op, 'input_desc')],
                       'outputs': [descriptor(d) for d in blocks(op, 'output_desc')],
                       'selected_attrs': attrs})
    return result


def analyze(path):
    ops = parse(path)
    groups = defaultdict(Counter)
    for op in ops:
        key = (tuple(x['dtype'] for x in op['inputs']), tuple(x['dtype'] for x in op['outputs']))
        groups[op['type']][key] += 1
    summary = {kind: [{'input_dtypes': list(i), 'output_dtypes': list(o), 'count': n}
                      for (i, o), n in stats.most_common()] for kind, stats in groups.items()}
    interesting = {'MatMulV2', 'BatchMatMulV2', 'LayerNorm', 'AddLayerNorm', 'SoftmaxV2',
                   'LpNormV2', 'ClipByValue', 'RealDiv', 'Sqrt', 'Gelu', 'Cast'}
    sensitive = [op for op in ops if op['type'] in interesting or
                 any(x in (op['name'] or '').lower() for x in ['mask', 'norm', 'epsilon'])]
    matmul = [op for op in ops if op['type'] in ['MatMulV2','BatchMatMulV2']]
    half_cube = [op for op in matmul if len(op['inputs']) >= 2 and op['inputs'][0]['dtype'] == 'DT_FLOAT16'
                 and op['inputs'][1]['dtype'] == 'DT_FLOAT16' and op['outputs']
                 and op['outputs'][0]['dtype'] == 'DT_FLOAT']
    return {'graph': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'operator_count': len(ops), 'op_type_counts': dict(Counter(x['type'] for x in ops)),
            'dtype_distributions_by_type': summary,
            'fp16_inputs_fp32_output_matmuls': len(half_cube),
            'half_cube_examples': half_cube[:4], 'sensitive_operators': sensitive,
            'limitations': ['Descriptors prove declared kernel tensor types, not internal accumulation precision.',
                            'No accuracy or performance conclusion is inferred from this graph.',
                            'Root/subgraphs contain Data and shape nodes; counts are not executed kernel totals.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('graph', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = analyze(a.graph)
    a.output.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ['sensitive_operators','half_cube_examples']}, indent=2))
