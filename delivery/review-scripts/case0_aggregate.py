import hashlib
import json
import math
import pathlib
import re
import statistics
import subprocess

root = pathlib.Path('/export/home/ext.hejun23/workspace/bge-opt-100')
names = [f'case0-best-r{i}-20261027' for i in (1, 2, 3)]


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


rounds = []
latencies = []
for name in names:
    folder = root / 'work' / name
    plan = load(root / 'work/plans' / f'{name}.json')
    perf = load(root / 'work' / f'{name}-performance.json')
    identity = load(root / 'work' / f'{name}-run-identity.json')
    status = load(root / 'work/jobs' / name / 'status.json')
    before = load(root / 'work' / f'{name}-before-gate.json')
    after = load(root / 'work' / f'{name}-after-gate.json')
    assert status['state'] == 'finished' and status['exit_code'] == 0
    assert before['clear'] and after['clear']
    assert perf['passed'] and perf['errors'] == 0
    assert identity['passed']
    assert plan['workload'] == {'kind': 'fixed_case', 'case_ids': ['case-0000'], 'sampling': 'fixed'}
    assert plan['instances'] == 8
    assert plan['options'] == {'precision_mode': 'must_keep_origin_dtype',
                               'graph_run_mode': 0, 'aicore_num': '12|24'}
    assert perf['window_seconds'] >= 60 and perf['window_successes'] >= 10000
    assert perf['input_identity']['passed']
    assert perf['input_identity']['input_binary_sha256'] == '6a6fe4cbd2e443001ed12a34bfca48483b3d7dfa7c0c02c11361f30db501d5c7'
    assert perf['workload'] == plan['workload']
    assert identity['runs'][name]['plugins'] == {}
    assert identity['runs'][name]['binary_sha256'] == '8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d'
    assert identity['runs'][name]['packed_graph_sha256'] == '5e641362d99c9a1afb6dfce70f47b4ac07f1a7c88272fdfc791ea04ae84f4822'

    result = load(folder / 'result.json')['runtime']['performance']
    start, end = result['window_start_ns'], result['window_end_ns']
    current = []
    for line in (folder / 'events.jsonl').open():
        event = json.loads(line)
        if event['phase'] != 'measure':
            continue
        assert event['case_id'] == 'case-0000' and event['status'] == 'success'
        assert event['h2d_bytes'] == 3416 and event['d2h_bytes'] == 40448
        current.append((event['completed_ns'] - event['started_ns']) / 1e6)
    assert len(current) == perf['cohort_successes']
    assert math.isclose(quantile(current, .99), perf['p99_ms'], abs_tol=1e-6)
    latencies.extend(current)

    resource_rows = [json.loads(line) for line in (folder / 'resources.jsonl').open()]
    in_window = [row for row in resource_rows if start <= row['monotonic_ns'] <= end]
    target_pids = set()
    sibling_pids = set()
    for row in in_window:
        assert row['npu_smi_code'] == 0
        for line in row['npu_smi_stdout'].splitlines():
            for chip, destination in ((0, target_pids), (1, sibling_pids)):
                match = re.match(r'^\|\s*1\s+' + str(chip) + r'\s*\|\s*(\d+)\s*\|', line)
                if match:
                    destination.add(int(match.group(1)))
    assert in_window and len(target_pids) == 1 and not sibling_pids
    rounds.append({'run_id': name, 'qps': perf['qps'], 'p50_ms': perf['p50_ms'],
                   'p99_ms': perf['p99_ms'], 'max_ms': perf['max_ms'],
                   'window_seconds': perf['window_seconds'],
                   'window_successes': perf['window_successes'],
                   'cohort_successes': perf['cohort_successes'],
                   'errors': perf['errors'], 'resource_samples_in_window': len(in_window),
                   'target_host_pids': sorted(target_pids), 'sibling_host_pids': sorted(sibling_pids),
                   'job_status_sha256': digest(root / 'work/jobs' / name / 'status.json'),
                   'plan_sha256': digest(root / 'work/plans' / f'{name}.json'),
                   'events_sha256': digest(folder / 'events.jsonl'),
                   'resources_sha256': digest(folder / 'resources.jsonl'),
                   'before_gate_sha256': digest(root / 'work' / f'{name}-before-gate.json'),
                   'after_gate_sha256': digest(root / 'work' / f'{name}-after-gate.json')})

qps_values = [item['qps'] for item in rounds]
budget = load(root / 'work/budget.json')
assert budget['charged_seconds'] <= budget['max_seconds']
device_clear = {}
for chip in (0, 1):
    process = subprocess.run(['npu-smi', 'info', '-t', 'proc-mem', '-i', '1', '-c', str(chip)],
                             capture_output=True, text=True, check=True, timeout=20)
    device_clear[str(chip)] = process.stdout.strip() == 'No process in device.'
assert all(device_clear.values())

report = {'schema': 'bge100-case0-formal-summary/1', 'passed': True,
          'source': 'three independent fixed case-0000 GE mode=0 rounds',
          'rounds': rounds, 'median_qps': statistics.median(qps_values),
          'relative_qps_range_percent': 100 * (max(qps_values) - min(qps_values)) / statistics.median(qps_values),
          'pooled_request_count': len(latencies),
          'pooled_p50_ms': quantile(latencies, .5),
          'pooled_p99_ms': quantile(latencies, .99),
          'pooled_max_ms': max(latencies),
          'budget': {'charged_seconds': budget['charged_seconds'],
                     'remaining_seconds': budget['remaining_seconds'],
                     'max_seconds': budget['max_seconds']},
          'post_run_device_clear': device_clear}
path = root / 'work/case0-best-summary-20261027.json'
path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'passed': report['passed'], 'median_qps': report['median_qps'],
                  'pooled_p99_ms': report['pooled_p99_ms'],
                  'pooled_request_count': report['pooled_request_count'],
                  'relative_qps_range_percent': report['relative_qps_range_percent'],
                  'post_run_device_clear': device_clear, 'report': str(path)}))
