#!/usr/bin/env python3
"""Summarize actual GE text graph descriptors, with bounded per-op evidence.

Both GE proto op blocks and ONNX-style node blocks are accepted. Unrecognized
descriptors are retained as text evidence rather than guessed from IO dtype.
This inspects graph-declared kernel tensor types, not hidden accumulation types.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|[{}]|[^{}"]+', re.S)


def blocks(text, field):
    # Match braces while respecting quoted strings, which can contain JSON.
    for match in re.finditer(r'(?m)^\s*' + field + r'\s*\{', text):
        start = text.index('{', match.start())
        depth = 0
        for token in TOKENS.finditer(text, start):
            if token.group() == '{':
                depth += 1
            elif token.group() == '}':
                depth -= 1
                if depth == 0:
                    yield text[match.start():token.end()]
                    break


def analyze(path):
    text = path.read_text(errors='replace')
    ops = list(blocks(text, 'op')) or list(blocks(text, 'node'))
    rows = []
    for block in ops:
        name = re.search(r'(?m)^\s*name:\s*"([^"]*)"', block)
        kind = re.search(r'(?m)^\s*(?:type|op_type):\s*"([^"]*)"', block)
        if not kind:
            continue
        descriptors = []
        for field in ['input_desc', 'output_desc']:
            for desc in blocks(block, field):
                dtype = re.findall(r'(?m)^\s*(?:dtype|data_type):\s*([^\n]+)', desc)
                descriptors.append({'direction': field, 'dtype_tokens': dtype,
                                    'excerpt': desc[:2000]})
        attrs = []
        for attr in list(blocks(block, 'attr')) + list(blocks(block, 'attribute')):
            if re.search(r'dtype|data_type|input_desc|output_desc|precision|kernel', attr, re.I):
                attrs.append(attr[:3000])
        rows.append({'name': name.group(1) if name else None, 'type': kind.group(1),
                     'descriptors': descriptors, 'dtype_attributes': attrs})
    counts = Counter(x['type'] for x in rows)
    interesting = [x for x in rows if any(s in x['type'].lower() for s in
                    ['matmul','gemm','layernorm','gelu','softmax','cast','reduce','sqrt','l2normal'])]
    fp16 = [x for x in interesting if re.search(r'DT_FLOAT16|float16|FLOAT16', json.dumps(x))]
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'operator_count': len(rows), 'op_types': dict(counts.most_common()),
            'fp16_evidence_nodes': fp16, 'interesting_nodes': interesting,
            'limitations': ['Graph tensor dtype is not proof of accumulation dtype.',
                            'No FP16 descriptor found does not prove absence if parser has unrecognized format.',
                            'Check compile logs/kernel metadata together with this graph.']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('graphs', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = [analyze(path) for path in args.graphs]
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{'path': x['path'], 'operators': x['operator_count'],
                       'op_types': x['op_types'], 'nodes_with_explicit_fp16_tokens': len(x['fp16_evidence_nodes'])}
                      for x in result], indent=2))


if __name__ == '__main__':
    main()
