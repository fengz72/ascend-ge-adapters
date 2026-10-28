"""Read-only post-run audit of the requested BGE FP16 10-instance profile."""
import collections
import json
import math
from pathlib import Path
import re
import statistics

root = Path('/export/home/ext.hejun23/workspace/bge-opt-100')
names = {
    'pressure': 'fp16-ten-d2-pressure-20261027',
    'fixed': 'fp16-ten-d2-fixed-formal-r1-20261027',
    'random': 'fp16-ten-d2-random-formal-r1-20261027',
    'stability': 'fp16-ten-d2-stability-20261027',
}
reports = {}
task_ids = set()
for role, name in names.items():
    status = json.loads((root / 'work/jobs' / name / 'status.json').read_text())
    identity = json.loads((root / 'work' / f'{name}-run-identity.json').read_text())
    plan = json.loads((root / 'work/plans' / f'{name}.json').read_text())
    before = json.loads((root / 'work' / f'{name}-before-gate.json').read_text())
    after = json.loads((root / 'work' / f'{name}-after-gate.json').read_text())
    assert status['state'] == 'finished' and status['exit_code'] == 0 and status['process_group_stopped']
    assert status['task_id'] not in task_ids
    task_ids.add(status['task_id'])
    assert identity['passed'] and before['clear'] and after['clear']
    assert plan['device'] == 2 and plan['instances'] == 10
    assert plan['options'] == {'precision_mode': 'force_fp16', 'graph_run_mode': 0, 'aicore_num': '12|24'}
    assert plan['graph']['path'] == str(root / 'materials/exported/model.onnx')
    reports[role] = {'name': name, 'status': 'passed', 'device': 2,
                     'before_after_gate': 'clear', 'identity': 'passed',
                     'options': plan['options']}
    if role == 'pressure':
        accuracy = json.loads((root / 'work' / f'{name}-accuracy.json').read_text())
        assert accuracy['passed'] and accuracy['failed_requests'] == 0
        assert accuracy['instances'] == 10 and accuracy['requests'] >= 2000
        assert len(accuracy['per_instance_case_counts']) == 50
        assert min(accuracy['per_instance_case_counts'].values()) >= 20
        reports[role].update(requests=accuracy['requests'], worst=accuracy['worst'],
                             minimum_per_instance_case=min(accuracy['per_instance_case_counts'].values()))
    else:
        perf = json.loads((root / 'work' / f'{name}-performance.json').read_text())
        assert perf['passed'] and perf['errors'] == 0 and perf['statistics_sufficient']
        assert perf['window_successes'] >= 10000
        assert perf['window_seconds'] >= (600 if role == 'stability' else 60)
        assert perf['workload']['kind'] == ('fixed_case' if role == 'fixed' else 'random_cases')
        reports[role].update(window_seconds=perf['window_seconds'],
                             window_successes=perf['window_successes'],
                             qps=perf['qps'], p99_ms=perf['p99_ms'],
                             max_ms=perf['max_ms'], errors=perf['errors'])

name = names['stability']
result = json.loads((root / 'work' / name / 'result.json').read_text())
perf = result['runtime']['performance']
start, end = perf['window_start_ns'], perf['window_end_ns']
buckets = collections.Counter()
with (root / 'work' / name / 'events.jsonl').open() as f:
    for line in f:
        e = json.loads(line)
        if e['phase'] == 'measure' and start <= e['completed_ns'] <= end:
            index = min((e['completed_ns'] - start) // 10_000_000_000, 59)
            buckets[index] += 1
assert len(buckets) == 60 and set(buckets) == set(range(60))
bucket_qps = [buckets[i] / 10 for i in range(60)]
rows = [json.loads(line) for line in (root / 'work' / name / 'resources.jsonl').read_text().splitlines()]
rows = [row for row in rows if start <= row['monotonic_ns'] <= end]
assert len(rows) >= 12
series = []
for row in rows:
    assert row['npu_smi_code'] == 0 and row['rss_kib'] >= 0
    matching = [line for line in row['npu_smi_stdout'].splitlines() if '0000:99:00.0' in line]
    assert len(matching) == 1
    hbm = re.findall(r'(\d+)\s*/\s*(\d+)', matching[0])
    assert hbm and int(hbm[-1][1]) == 65536
    series.append({'seconds': (row['monotonic_ns'] - start) / 1e9,
                   'rss_kib': row['rss_kib'], 'hbm_mb': int(hbm[-1][0])})
duration = (end - start) / 1e9
head = [row for row in series if row['seconds'] <= 120]
tail = [row for row in series if row['seconds'] >= duration - 120]
assert len(head) >= 3 and len(tail) >= 3
limits = {}
for key, floor, ratio in (('rss_kib', 128 * 1024, 0.05), ('hbm_mb', 64, 0.01)):
    initial = statistics.median(row[key] for row in head)
    final = statistics.median(row[key] for row in tail)
    limit = max(floor, initial * ratio)
    limits[key] = {'first_2min_median': initial, 'last_2min_median': final,
                   'increase': final - initial, 'allowed_increase': limit,
                   'within_limit': final - initial <= limit,
                   'peak': max(row[key] for row in series)}
assert all(x['within_limit'] for x in limits.values())
gaps = [b['seconds'] - a['seconds'] for a, b in zip(series, series[1:])]
reports['stability'].update(ten_second_qps_min=min(bucket_qps),
                            ten_second_qps_max=max(bucket_qps),
                            complete_ten_second_buckets=len(bucket_qps),
                            resource_samples=len(series),
                            sample_gap_median_seconds=statistics.median(gaps),
                            sample_gap_max_seconds=max(gaps),
                            resource_growth=limits)
report = {'schema': 'bge-fp16-ten-selection-audit/1', 'passed': True,
          'model': 'original external-weight ONNX',
          'precision': 'GE force_fp16, FP32 external floating IO',
          'latency_scope': 'h2d_desc_execute_sync_d2h',
          'no_p99_slo': True, 'runs': reports,
          'limits': ['Five supplied samples only', 'No production traffic distribution',
                     'Ten instances are user-selected; superiority over eight FP16 instances was not established']}
path = root / 'work/fp16-ten-d2-selection-audit-20261027.json'
path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'passed': True, 'report': str(path), 'runs': {k: {kk: vv for kk, vv in v.items() if kk in ('requests', 'window_successes', 'qps', 'p99_ms')} for k, v in reports.items()},
                  'resource_growth': limits}, ensure_ascii=False))
