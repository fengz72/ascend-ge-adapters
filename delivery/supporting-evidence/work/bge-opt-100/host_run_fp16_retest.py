"""Run one managed BGE benchmark with host-side NPU process monitoring.

This script runs on host 100 and performs no profiling. The existing project
control.py still owns the device lock and before/after occupancy gates.
"""

import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

instances = int(sys.argv[1])
round_number = int(sys.argv[2])
assert instances in (8, 10, 12) and round_number in (1, 2, 3)
root = pathlib.Path('/export/home/ext.hejun23/workspace/bge-opt-100')
name = f'fp16-retest{instances}-d10-case0-r{round_number}-20261027'
container = 'hejun-cann9-bge'
assert subprocess.run(['docker', 'exec', container, 'test', '!', '-e',
                       str(root / 'work/jobs' / name)], timeout=15).returncode == 0
inspect = subprocess.run(['docker', 'inspect', '--format', '{{.Id}}', container],
                         text=True, capture_output=True, check=True, timeout=15)
container_id = inspect.stdout.strip()
assert len(container_id) == 64
before = [subprocess.run(['docker', 'exec', container, 'npu-smi', 'info', '-t',
                          'proc-mem', '-i', '5', '-c', str(chip)],
                         text=True, capture_output=True, check=True, timeout=15)
          for chip in (0, 1)]
assert all('No process in device.' in x.stdout for x in before)
monitor_path = pathlib.Path('/tmp') / f'{name}-host-monitor.jsonl'
launcher_path = pathlib.Path('/tmp') / f'{name}-host-launcher.log'
assert not monitor_path.exists() and not launcher_path.exists()
stop = threading.Event()


def sample(chip):
    result = subprocess.run(['docker', 'exec', container, 'npu-smi', 'info', '-t',
                             'proc-mem', '-i', '5', '-c', str(chip)],
                            text=True, capture_output=True, timeout=15)
    pids = [int(x) for x in re.findall(r'Process id:\s*(\d+)', result.stdout)]
    processes = []
    for pid in pids:
        path = pathlib.Path('/proc') / str(pid)
        try:
            cgroup = (path / 'cgroup').read_text()
            comm = (path / 'comm').read_text().strip()
        except (OSError, UnicodeError):
            cgroup, comm = '', None
        processes.append({'host_pid': pid, 'comm': comm,
                          'in_target_container': container_id in cgroup})
    return {'returncode': result.returncode, 'pids': processes,
            'empty': 'No process in device.' in result.stdout,
            'stderr': result.stderr[-500:]}


def monitor():
    with monitor_path.open('x') as out:
        while not stop.is_set():
            record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'monotonic_ns': time.monotonic_ns(),
                      'host_loadavg': os.getloadavg(),
                      'chip0': sample(0), 'chip1': sample(1)}
            out.write(json.dumps(record) + '\n')
            out.flush()
            stop.wait(2)


command = ['docker', 'exec', container, 'python3', str(root / 'tools/run_one.py'),
           '--formal', '--config', f'forcefp16-half{instances}', '--timeout', '700',
           name, '--', '--instances', str(instances), '--cores', '12|24',
           '--precision-mode', 'force_fp16', '--seconds', '60',
           '--max-seconds', '300', '--requests', '10000',
           '--seed', str(2040 + round_number), '--case-id', 'case-0000']
thread = threading.Thread(target=monitor, daemon=False)
thread.start()
print('START', name, 'host_loadavg', os.getloadavg(), flush=True)
try:
    with launcher_path.open('x') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
finally:
    stop.set()
    thread.join(timeout=20)
    if thread.is_alive():
        raise RuntimeError('Host NPU monitor did not stop')
print('EXIT', completed.returncode, 'monitor', str(monitor_path), flush=True)
for host_path in (monitor_path, launcher_path):
    subprocess.run(['docker', 'cp', str(host_path),
                    f'{container}:{root / "work" / host_path.name}'],
                   check=True, timeout=30)
if completed.returncode:
    print('LOG_TAIL', '\n'.join(launcher_path.read_text(errors='replace').splitlines()[-12:]),
          flush=True)
    raise SystemExit(completed.returncode)
report = subprocess.run(['docker', 'exec', container, 'cat',
                         str(root / 'work' / f'{name}-performance.json')],
                        capture_output=True, text=True, check=True, timeout=15)
performance = json.loads(report.stdout)
print('PERF', json.dumps({key: performance[key]
                          for key in ('qps', 'p99_ms', 'window_seconds',
                                      'window_successes', 'errors', 'passed')}), flush=True)
