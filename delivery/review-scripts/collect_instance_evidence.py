import datetime
import hashlib
import json
import pathlib
import re

root = pathlib.Path('/export/home/ext.hejun23/workspace/bge-opt-100')
names = (
    'scale10-case0-s1-20261027',
    'scale12-case0-s1-20261027',
    'scale16-case0-s1-20261027',
    'scale8-case0-s2-20261027',
    'scale12-case0-s2-20261027',
    'scale16-case0-s2-20261027',
)

def load(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

rows = []
for name in names:
    folder = root / 'work' / name
    plan_path = root / 'work/plans' / (name + '.json')
    perf_path = root / 'work' / (name + '-performance.json')
    status_path = root / 'work/jobs' / name / 'status.json'
    resources_path = folder / 'resources.jsonl'
    plan, perf, status = map(load, (plan_path, perf_path, status_path))
    gates = [load(root / 'work' / (name + '-' + stage + '-gate.json'))
             for stage in ('before', 'after')]
    samples = [json.loads(line) for line in resources_path.read_text().splitlines()]
    hbm = []
    for sample in samples:
        for line in sample.get('npu_smi_stdout', '').splitlines():
            if '0000:99:00.0' in line:
                match = re.search(r'(\d+)\s*/\s*(\d+)\s*\|', line)
                if match:
                    hbm.append(tuple(map(int, match.groups())))
    rss = [sample['rss_kib'] for sample in samples if sample.get('rss_kib') is not None]
    rows.append({
        'name': name,
        'instances': plan['instances'],
        'options': plan['options'],
        'workload': perf['workload'],
        'window_seconds': perf['window_seconds'],
        'window_successes': perf['window_successes'],
        'errors': perf['errors'],
        'qps': perf['qps'],
        'p99_ms': perf['p99_ms'],
        'hbm_peak_mb': max((used for used, _ in hbm), default=None),
        'hbm_capacity_mb': hbm[0][1] if hbm else None,
        'host_rss_peak_kib': max(rss, default=None),
        'resource_samples': len(samples),
        'job_state': status['state'],
        'exit_code': status['exit_code'],
        'duration_seconds': (
            datetime.datetime.fromisoformat(status['finished_at'])
            - datetime.datetime.fromisoformat(status['started_at'])
        ).total_seconds(),
        'gates_clear': [gate['clear'] for gate in gates],
        'input_identity_passed': load(root / 'work' / (name + '-inputs.json'))['passed'],
        'run_identity_passed': load(root / 'work' / (name + '-run-identity.json'))['passed'],
        'performance_passed': perf['passed'],
        'plan_sha256': sha(plan_path),
        'performance_sha256': sha(perf_path),
        'resources_sha256': sha(resources_path),
        'remote_work_dir': str(folder),
        'remote_status': str(status_path),
    })
assert all(row['job_state'] == 'finished' and row['exit_code'] == 0
           and all(row['gates_clear']) and row['input_identity_passed']
           and row['run_identity_passed'] and row['performance_passed'] for row in rows)
print(json.dumps({'schema': 'bge-instance-scaling-evidence/1',
                  'device': 2, 'pci': '0000:99:00.0', 'rows': rows}, indent=2))
