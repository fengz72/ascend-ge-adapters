"""Run a short managed force_fp16 job with GE build-graph dumping enabled."""

import json
import os
import pathlib
import subprocess
import sys

root = pathlib.Path('/export/home/ext.hejun23/workspace/bge-opt-100')
name = 'fp16-graph-d10-r2-20261027'
dump = root / 'work' / 'ge-graph-dumps' / name
dump.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env.update(DUMP_GE_GRAPH='2', DUMP_GRAPH_LEVEL='3', DUMP_GRAPH_PATH=str(dump))
subprocess.run([
    sys.executable, str(root / 'tools/make_plan.py'), name,
    '--instances', '1', '--cores', '12|24', '--precision-mode', 'force_fp16',
    '--seconds', '1', '--max-seconds', '60', '--requests', '1',
    '--seed', '2032', '--case-id', 'case-0000',
], cwd=root, env=env, check=True)
plan = root / 'work' / 'plans' / f'{name}.json'
subprocess.run([
    sys.executable, str(root / 'tools/control.py'), 'run', str(plan),
    '--job', name, '--config', 'forcefp16-graph-one', '--timeout', '180',
], cwd=root, env=env, check=True)
subprocess.run([
    sys.executable, str(root / 'tools/checks/validate_run_identity.py'),
    '--runs', name, '--report', str(root / 'work' / f'{name}-run-identity.json'),
], cwd=root, check=True)
builds = sorted(dump.rglob('ge_proto_*_Build.txt'))
assert builds, 'GE build graph was not dumped'
manifest = {
    'job': name, 'plan': str(plan),
    'precision_mode': json.loads(plan.read_text())['options']['precision_mode'],
    'env': {key: env[key] for key in ('DUMP_GE_GRAPH', 'DUMP_GRAPH_LEVEL', 'DUMP_GRAPH_PATH')},
    'build_graphs': [{'path': str(path), 'bytes': path.stat().st_size} for path in builds],
}
out = root / 'work' / f'{name}-dump-manifest.json'
out.write_text(json.dumps(manifest, indent=2) + '\n')
print('GE_GRAPH_DUMP_ACCEPTED', json.dumps(manifest), flush=True)
