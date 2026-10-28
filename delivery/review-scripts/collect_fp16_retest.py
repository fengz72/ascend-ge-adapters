"""Audit nine no-profiling force_fp16 performance retest rounds."""

import datetime
import hashlib
import json
import pathlib
import re
import statistics

root = pathlib.Path('/export/home/ext.hejun23/workspace/bge-opt-100')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = []
for instances in (8, 10, 12):
    for round_number in (1, 2, 3):
        name = f'fp16-retest{instances}-d10-case0-r{round_number}-20261027'
        folder = root / 'work' / name
        plan_path = root / 'work/plans' / f'{name}.json'
        perf_path = root / 'work' / f'{name}-performance.json'
        resources_path = folder / 'resources.jsonl'
        monitor_path = root / 'work' / f'{name}-host-monitor.jsonl'
        status_path = root / 'work/jobs' / name / 'status.json'
        plan, perf, status = map(read, (plan_path, perf_path, status_path))
        result = read(folder / 'result.json')['runtime']['performance']
        before, after = [read(root / 'work' / f'{name}-{phase}-gate.json')
                         for phase in ('before', 'after')]
        identity = read(root / 'work' / f'{name}-run-identity.json')
        assert plan['device'] == 10 and plan['instances'] == instances
        assert plan['options'] == {'precision_mode': 'force_fp16',
                                   'graph_run_mode': 0, 'aicore_num': '12|24'}
        assert plan['workload'] == {'kind': 'fixed_case', 'case_ids': ['case-0000'],
                                    'sampling': 'fixed'}
        assert plan['observability']['profiling'] is False
        assert plan['replay']['capture_outputs'] is False
        assert status['state'] == 'finished' and status['exit_code'] == 0
        assert before['clear'] and after['clear'] and identity['passed']
        assert perf['passed'] and perf['acceptance_level'] == 'formal_round'
        assert perf['errors'] == 0 and perf['window_seconds'] >= 60
        assert perf['window_successes'] >= 10000 and perf['input_identity']['passed']
        assert perf['input_identity']['input_binary_sha256'] == (
            '6a6fe4cbd2e443001ed12a34bfca48483b3d7dfa7c0c02c11361f30db501d5c7')
        bound = identity['runs'][name]
        assert bound['binary_sha256'] == '8efca50a62a7f569032345259d6e7813da428023903e4674d77de92bc7ac630d'
        assert bound['packed_graph_sha256'] == '5e641362d99c9a1afb6dfce70f47b4ac07f1a7c88272fdfc791ea04ae84f4822'
        assert bound['plugins'] == {}
        monitor_rows = [json.loads(line) for line in monitor_path.read_text().splitlines()]
        start, end = result['window_start_ns'], result['window_end_ns']
        window = [row for row in monitor_rows if start <= row['monotonic_ns'] <= end]
        assert len(window) >= 12
        gaps = [(b['monotonic_ns'] - a['monotonic_ns']) / 1e9
                for a, b in zip(window, window[1:])]
        assert max(gaps) <= 5.0
        assert all(row['chip0']['returncode'] == row['chip1']['returncode'] == 0
                   and len(row['chip0']['pids']) == 1
                   and row['chip0']['pids'][0]['in_target_container']
                   and row['chip0']['pids'][0]['comm'] == 'ge_runtime'
                   and row['chip1']['empty'] for row in window)
        host_pids = {row['chip0']['pids'][0]['host_pid'] for row in window}
        assert len(host_pids) == 1
        samples = [json.loads(line) for line in resources_path.read_text().splitlines()]
        hbm = []
        for sample in samples:
            for line in sample.get('npu_smi_stdout', '').splitlines():
                if '0000:89:00.0' in line:
                    match = re.search(r'(\d+)\s*/\s*(\d+)\s*\|', line)
                    if match:
                        hbm.append(tuple(map(int, match.groups())))
        rss = [sample['rss_kib'] for sample in samples
               if sample.get('rss_kib') is not None]
        assert hbm and rss
        rows.append({
            'name': name, 'instances': instances, 'round': round_number,
            'seed': plan['seed'], 'qps': perf['qps'], 'p50_ms': perf['p50_ms'],
            'p99_ms': perf['p99_ms'], 'window_seconds': perf['window_seconds'],
            'window_successes': perf['window_successes'], 'errors': perf['errors'],
            'host_loadavg_1min_min': min(row['host_loadavg'][0] for row in window),
            'host_loadavg_1min_max': max(row['host_loadavg'][0] for row in window),
            'target_host_pid': host_pids.pop(),
            'host_monitor_samples_in_window': len(window),
            'host_monitor_max_gap_seconds': max(gaps),
            'target_hbm_peak_mb': max(used for used, _ in hbm),
            'target_hbm_capacity_mb': hbm[0][1],
            'host_rss_peak_kib': max(rss),
            'job_duration_seconds': (datetime.datetime.fromisoformat(status['finished_at'])
                                     - datetime.datetime.fromisoformat(status['started_at'])).total_seconds(),
            'plan_sha256': digest(plan_path), 'performance_sha256': digest(perf_path),
            'events_sha256': digest(folder / 'events.jsonl'),
            'resources_sha256': digest(resources_path),
            'host_monitor_sha256': digest(monitor_path),
            'status_sha256': digest(status_path),
            'remote_run_dir': str(folder),
        })

summary = {}
for instances in (8, 10, 12):
    group = [row for row in rows if row['instances'] == instances]
    qps = [row['qps'] for row in group]
    summary[str(instances)] = {
        'qps_median': statistics.median(qps),
        'qps_spread_percent': 100 * (max(qps) - min(qps)) / statistics.median(qps),
        'p99_median_ms': statistics.median(row['p99_ms'] for row in group),
        'hbm_peak_mb': max(row['target_hbm_peak_mb'] for row in group),
    }
baseline = summary['8']
for instances in (10, 12):
    candidate = summary[str(instances)]
    candidate['qps_change_vs_8_percent'] = 100 * (
        candidate['qps_median'] / baseline['qps_median'] - 1)
    candidate['significance_gate_percent'] = max(
        3, baseline['qps_spread_percent'], candidate['qps_spread_percent'])
    candidate['passes_qps_gate'] = (candidate['qps_change_vs_8_percent']
                                    >= candidate['significance_gate_percent'])

print(json.dumps({'schema': 'bge100-force-fp16-retest/1',
                  'host': '100', 'container': 'hejun-cann9-bge',
                  'device': 10, 'pci': '0000:89:00.0',
                  'workload': 'fixed case-0000; includes H2D, GE completion, full D2H',
                  'profiling': False,
                  'host_monitor': 'npu-smi proc-mem and host load only; not profiling',
                  'accuracy_at_10_or_12_instances': 'not_tested',
                  'rows': rows, 'summary': summary}, indent=2))
