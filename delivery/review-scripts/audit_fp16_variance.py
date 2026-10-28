"""Read archived force_fp16 performance and in-window resource evidence."""

import hashlib
import json
import pathlib
import re

root = pathlib.Path('/export/home/ext.hejun23/workspace/bge-opt-100')
names = [
    'fp16-formal-d10-fp16-case0-r1-20261027',
    'fp16-formal-d10-fp16-case0-r2-20261027',
    *[f'fp16-scale{i}-d10-case0-s{s}-20261027'
      for i, s in ((8, 1), (10, 1), (12, 1), (12, 2), (10, 2), (8, 2))],
]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


jobs = []
for name in names:
    folder = root / 'work' / name
    perf_path = root / 'work' / f'{name}-performance.json'
    resources_path = folder / 'resources.jsonl'
    perf = read(perf_path)
    result = read(folder / 'result.json')['runtime']['performance']
    status = read(root / 'work/jobs' / name / 'status.json')
    rows = [json.loads(line) for line in resources_path.read_text().splitlines()]
    start, end = result['window_start_ns'], result['window_end_ns']
    window = [row for row in rows if start <= row['monotonic_ns'] <= end]
    target_pids, sibling_pids = set(), set()
    memory_mb = []
    for row in window:
        assert row['npu_smi_code'] == 0
        for line in row['npu_smi_stdout'].splitlines():
            match = re.match(r'^\|\s*5\s+(0|1)\s*\|\s*(\d+)\s*\|', line)
            if match:
                (target_pids if match.group(1) == '0' else sibling_pids).add(int(match.group(2)))
            if '0000:89:00.0' in line:
                match = re.search(r'(\d+)\s*/\s*(\d+)\s*\|', line)
                if match:
                    memory_mb.append(tuple(map(int, match.groups())))
    before = read(root / 'work' / f'{name}-before-gate.json')
    after = read(root / 'work' / f'{name}-after-gate.json')
    assert status['state'] == 'finished' and status['exit_code'] == 0
    assert before['clear'] and after['clear']
    assert perf['passed'] and perf['errors'] == 0
    assert window and memory_mb and len(target_pids) == 1 and not sibling_pids
    jobs.append({
        'name': name, 'started_at': status['started_at'],
        'qps': perf['qps'], 'p50_ms': perf['p50_ms'], 'p99_ms': perf['p99_ms'],
        'segment_mean_ms': {key: value['mean'] for key, value in perf['segments'].items()},
        'window_seconds': perf['window_seconds'],
        'window_successes': perf['window_successes'],
        'in_window_resource_samples': len(window),
        'in_window_sample_offsets_seconds': [(row['monotonic_ns'] - start) / 1e9
                                             for row in window],
        'host_loadavg_1min': [row['host_loadavg'][0] for row in window],
        'target_process_pids': sorted(target_pids),
        'sibling_process_pids': sorted(sibling_pids),
        'target_hbm_mb': [used for used, _ in memory_mb],
        'before_gate_clear': before['clear'], 'after_gate_clear': after['clear'],
        'performance_sha256': digest(perf_path),
        'resources_sha256': digest(resources_path),
        'events_sha256': digest(folder / 'events.jsonl'),
    })

print(json.dumps({'schema': 'bge100-force-fp16-variance-audit/1',
                  'device': 10, 'pci': '0000:89:00.0',
                  'host_loadavg_scope': 'host-wide; not per container or NPU',
                  'jobs': jobs}, indent=2))
