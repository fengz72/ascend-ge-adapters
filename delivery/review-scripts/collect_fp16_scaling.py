"""Collect the six completed same-device force_fp16 instance-search runs."""

import datetime
import hashlib
import json
import pathlib
import re
import statistics

root = pathlib.Path('/export/home/ext.hejun23/workspace/bge-opt-100')
rows = []


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for instances in (8, 10, 12):
    for round_number in (1, 2):
        name = f'fp16-scale{instances}-d10-case0-s{round_number}-20261027'
        folder = root / 'work' / name
        plan_path = root / 'work/plans' / f'{name}.json'
        perf_path = root / 'work' / f'{name}-performance.json'
        status_path = root / 'work/jobs' / name / 'status.json'
        resources_path = folder / 'resources.jsonl'
        plan, perf, status = map(read, (plan_path, perf_path, status_path))
        gates = [read(root / 'work' / f'{name}-{stage}-gate.json')
                 for stage in ('before', 'after')]
        identity = read(root / 'work' / f'{name}-run-identity.json')
        assert plan['device'] == 10 and plan['instances'] == instances
        assert plan['options'] == {'precision_mode': 'force_fp16',
                                   'graph_run_mode': 0, 'aicore_num': '12|24'}
        assert plan['workload'] == {'kind': 'fixed_case', 'case_ids': ['case-0000'],
                                    'sampling': 'fixed'}
        assert plan['replay']['capture_outputs'] is False
        assert status['state'] == 'finished' and status['exit_code'] == 0
        assert all(gate['clear'] for gate in gates)
        assert identity['passed'] and perf['passed'] and perf['errors'] == 0
        assert perf['window_seconds'] >= 30 and perf['window_successes'] >= 2000
        assert perf['input_identity']['passed']
        samples = [json.loads(line) for line in resources_path.read_text().splitlines()]
        hbm = []
        target_pids, sibling_pids = set(), set()
        for sample in samples:
            for line in sample.get('npu_smi_stdout', '').splitlines():
                if '0000:89:00.0' in line:
                    match = re.search(r'(\d+)\s*/\s*(\d+)\s*\|', line)
                    if match:
                        hbm.append(tuple(map(int, match.groups())))
                match = re.match(r'^\|\s*5\s+(0|1)\s*\|\s*(\d+)\s*\|', line)
                if match:
                    (target_pids if match.group(1) == '0' else sibling_pids).add(int(match.group(2)))
        rss = [sample['rss_kib'] for sample in samples
               if sample.get('rss_kib') is not None]
        assert hbm and not sibling_pids and len(target_pids) == 1
        rows.append({
            'name': name, 'instances': instances, 'round': round_number,
            'seed': plan['seed'], 'qps': perf['qps'],
            'p50_ms': perf['p50_ms'], 'p99_ms': perf['p99_ms'],
            'window_seconds': perf['window_seconds'],
            'window_successes': perf['window_successes'], 'errors': perf['errors'],
            'hbm_peak_mb': max(used for used, _ in hbm),
            'hbm_capacity_mb': hbm[0][1],
            'host_rss_peak_kib': max(rss), 'target_pids': sorted(target_pids),
            'job_duration_seconds': (datetime.datetime.fromisoformat(status['finished_at'])
                                     - datetime.datetime.fromisoformat(status['started_at'])).total_seconds(),
            'plan_sha256': digest(plan_path),
            'performance_sha256': digest(perf_path),
            'events_sha256': digest(folder / 'events.jsonl'),
            'resources_sha256': digest(resources_path),
            'job_status_sha256': digest(status_path),
            'remote_run_dir': str(folder),
        })

summary = {}
for instances in (8, 10, 12):
    group = [row for row in rows if row['instances'] == instances]
    summary[str(instances)] = {
        'qps_median': statistics.median(row['qps'] for row in group),
        'qps_spread_percent': 100 * (max(row['qps'] for row in group)
                                     - min(row['qps'] for row in group))
                              / statistics.median(row['qps'] for row in group),
        'p99_median_ms': statistics.median(row['p99_ms'] for row in group),
        'hbm_peak_mb': max(row['hbm_peak_mb'] for row in group),
    }

print(json.dumps({'schema': 'bge100-force-fp16-instance-scaling/1',
                  'host': '100', 'container': 'hejun-cann9-bge',
                  'device': 10, 'pci': '0000:89:00.0',
                  'measurement_level': 'two 30-second search rounds per configuration',
                  'correctness_at_new_instance_counts': 'not_tested',
                  'rows': rows, 'summary': summary}, indent=2))
