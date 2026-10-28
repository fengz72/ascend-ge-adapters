"""Collect immutable 100-host force_fp16 comparison reports."""

import hashlib
import json
from pathlib import Path
from statistics import median

root = Path('/export/home/ext.hejun23/workspace/bge-opt-100')
names = {
    'search': {
        'fp32_case0': 'fp16-parent8-d10-case0-s1-20261027',
        'fp32_random': 'fp16-parent8-d10-random-s1-20261027',
        'fp16_case0': 'fp16-half8-d10-case0-s1-20261027',
        'fp16_random': 'fp16-half8-d10-random-s1-20261027',
    },
    'pressure': 'fp16-half8-d10-pressure-20261027',
    'graph': 'fp16-graph-d10-r2-20261027',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def performance(name):
    report = root/'work'/f'{name}-performance.json'
    plan = root/'work/plans'/f'{name}.json'
    status = root/'work/jobs'/name/'status.json'
    result = read(report)
    p = read(plan)
    s = read(status)
    assert result['passed'] and s['exit_code'] == 0
    assert p['options']['graph_run_mode'] == 0
    return {'name': name, 'qps': result['qps'], 'p99_ms': result['p99_ms'],
            'successes': result['window_successes'], 'seconds': result['window_seconds'],
            'precision_mode': p['options']['precision_mode'],
            'workload': result['workload'], 'report_sha256': sha(report),
            'plan_sha256': sha(plan), 'status_sha256': sha(status),
            'before_gate_sha256': sha(root/'work'/f'{name}-before-gate.json')
            if (root/'work'/f'{name}-before-gate.json').is_file() else None}


summary = {
    'schema': 'bge-force-fp16-results/1',
    'host': '100', 'device': 10, 'config': {'instances': 8, 'aicore_num': '12|24'},
    'search': {key: performance(name) for key, name in names['search'].items()},
}
summary['pressure'] = read(root/'work'/f"{names['pressure']}-accuracy.json")
summary['pressure']['job'] = names['pressure']
summary['graph'] = read(root/'work/force-fp16-graph-analysis-20261027.json')
summary['graph'] = {key:summary['graph'][key] for key in
                    ['job','precision_mode','graph_sha256','operator_count','dtype_counts','selected_ops']}
summary['graph']['selected_ops'].pop('Data', None)
summary['graph']['selected_ops'].pop('NetOutput', None)

formal = {}
for kind in ('case0','random'):
    for mode in ('fp32','fp16'):
        key = f'{mode}_{kind}'
        formal[key] = []
        for i in (1,2,3):
            name = f'fp16-formal-d10-{mode}-{kind}-r{i}-20261027'
            report = root/'work'/f'{name}-performance.json'
            if report.is_file():
                formal[key].append(performance(name))
summary['formal_rounds'] = formal
summary['formal_complete'] = all(len(group) == 3 for group in formal.values())
summary['comparison'] = {}
for kind in ('case0','random'):
    base = formal[f'fp32_{kind}']
    candidate = formal[f'fp16_{kind}']
    base_qps = median(x['qps'] for x in base)
    candidate_qps = median(x['qps'] for x in candidate)
    base_spread = 100*(max(x['qps'] for x in base)-min(x['qps'] for x in base))/base_qps
    candidate_spread = 100*(max(x['qps'] for x in candidate)-min(x['qps'] for x in candidate))/candidate_qps
    summary['comparison'][kind] = {
        'fp32_median_qps': base_qps, 'fp16_median_qps': candidate_qps,
        'qps_change_percent': 100*(candidate_qps/base_qps-1),
        'fp32_median_p99_ms': median(x['p99_ms'] for x in base),
        'fp16_median_p99_ms': median(x['p99_ms'] for x in candidate),
        'fp32_qps_spread_percent': base_spread,
        'fp16_qps_spread_percent': candidate_spread,
        'passes_significance_gate':
            100*(candidate_qps/base_qps-1) >= max(3, base_spread, candidate_spread)
            if summary['formal_complete'] else None,
    }
failed = 'fp16-formal-d10-fp32-case0-r3-20261027'
status = root/'work/jobs'/failed/'status.json'
if status.is_file():
    summary['interrupted_round'] = {
        'job': failed, 'status': read(status),
        'before_gate': read(root/'work'/f'{failed}-before-gate.json'),
    }
out = root/'work/force-fp16-results-20261027.json'
out.write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps({key:summary[key] for key in
                  ('schema','host','device','config','formal_complete','comparison')}, indent=2))
