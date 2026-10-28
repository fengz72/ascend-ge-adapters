"""Interleaved, same-device formal comparison of FP32 and force_fp16."""

import json
import pathlib
import subprocess
import sys

root = pathlib.Path('/export/home/ext.hejun23/workspace/bge-opt-100')
round_number = int(sys.argv[1])
assert round_number in (1, 2, 3)
device_tag = sys.argv[2] if len(sys.argv) > 2 else 'd10'
assert device_tag in ('d10', 'd13')
seed = 2031 + round_number


def run(mode, kind):
    name = f'fp16-formal-{device_tag}-{mode}-{kind}-r{round_number}-20261027'
    args = [
        sys.executable, str(root / 'tools/run_one.py'), '--formal',
        '--config', 'half8' if mode == 'fp32' else 'forcefp16-half8',
        '--timeout', '350', name, '--', '--instances', '8',
        '--cores', '12|24', '--precision-mode',
        'must_keep_origin_dtype' if mode == 'fp32' else 'force_fp16',
        '--seconds', '60', '--max-seconds', '300', '--requests', '10000',
        '--seed', str(seed),
    ]
    if kind == 'case0':
        args += ['--case-id', 'case-0000']
    log = root / 'work' / f'{name}-launcher.log'
    print('FORMAL_START', name, flush=True)
    with log.open('w') as output:
        result = subprocess.run(args, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        print('FORMAL_FAILED', name, result.returncode, '\n'.join(log.read_text(errors='replace').splitlines()[-25:]), flush=True)
        raise SystemExit(result.returncode)
    report = json.loads((root / 'work' / f'{name}-performance.json').read_text())
    assert report['passed'] and report['acceptance_level'] == 'formal_round'
    print('FORMAL_ACCEPTED', json.dumps({
        'name': name, 'qps': report['qps'], 'p99_ms': report['p99_ms'],
        'successes': report['window_successes'], 'seconds': report['window_seconds'],
    }), flush=True)


order = [('fp32', 'case0'), ('fp16', 'case0'), ('fp32', 'random'), ('fp16', 'random')]
if round_number == 2:
    order = [('fp16', 'case0'), ('fp32', 'case0'), ('fp16', 'random'), ('fp32', 'random')]
for mode, kind in order:
    run(mode, kind)
print('FORMAL_ROUND_DONE', round_number, flush=True)
