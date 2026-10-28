"""Summarize actual GE build-graph tensor types for the force_fp16 job."""

import collections
import hashlib
import json
from pathlib import Path

from analyze_fp16_final_graph import parse

root = Path('/export/home/ext.hejun23/workspace/bge-opt-100')
manifest = json.loads((root/'work/fp16-graph-d10-r2-20261027-dump-manifest.json').read_text())
graphs = sorted(manifest['build_graphs'], key=lambda entry: entry['bytes'], reverse=True)
path = Path(graphs[0]['path'])
ops = parse(path)
summary = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
for op in ops:
    kind = op['type'] or 'UNKNOWN'
    ins = tuple(x['dtype'] for x in op['inputs'])
    outs = tuple(x['dtype'] for x in op['outputs'])
    signature = (ins, outs)
    summary[kind][signature] += 1
    if len(examples[kind]) < 3:
        examples[kind].append({'name': op['name'], 'inputs': ins, 'outputs': outs})
selected = ['MatMulV2', 'BatchMatMulV2', 'LayerNorm', 'SoftmaxV2', 'Gelu', 'Cast', 'Data', 'NetOutput']
result = {
    'job': manifest['job'], 'plan': manifest['plan'],
    'precision_mode': manifest['precision_mode'], 'dump_env': manifest['env'],
    'graph': str(path), 'graph_bytes': path.stat().st_size,
    'graph_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'operator_count': len(ops),
    'dtype_counts': dict(collections.Counter(dtype for op in ops for d in op['outputs']
                                             if (dtype := d['dtype']))),
    'selected_ops': {
        kind: {
            'count': sum(summary[kind].values()),
            'signatures': [{'input_dtypes': list(ins), 'output_dtypes': list(outs), 'count': count}
                           for (ins, outs), count in summary[kind].most_common()],
            'examples': examples[kind],
        } for kind in selected
    },
    'limitations': ['GE graph descriptors do not disclose internal accumulation precision.',
                    'This dump uses one fixed case and one instance; the eight-instance five-case pressure is separate.'],
}
out = root/'work/force-fp16-graph-analysis-20261027.json'
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
