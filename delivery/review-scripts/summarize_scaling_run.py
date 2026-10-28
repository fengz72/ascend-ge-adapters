import json
import pathlib
import re

root = pathlib.Path('/export/home/ext.hejun23/workspace/bge-opt-100')
for name in ('scale10-case0-s1-20261027', 'scale12-case0-s1-20261027',
             'scale16-case0-s1-20261027'):
    folder = root / 'work' / name
    samples = [json.loads(line) for line in (folder / 'resources.jsonl').read_text().splitlines()]
    rss = [sample['rss_kib'] for sample in samples if sample.get('rss_kib') is not None]
    hbm = []
    for sample in samples:
        for line in sample.get('npu_smi_stdout', '').splitlines():
            if '0000:99:00.0' in line:
                match = re.search(r'(\d+)\s*/\s*(\d+)\s*\|', line)
                if match:
                    hbm.append(tuple(map(int, match.groups())))
    perf = json.loads((root / 'work' / (name + '-performance.json')).read_text())
    print(json.dumps({'name': name, 'qps': perf['qps'], 'p99_ms': perf['p99_ms'],
                      'window_seconds': perf['window_seconds'],
                      'window_successes': perf['window_successes'],
                      'samples': len(samples), 'hbm_samples': len(hbm),
                      'target_hbm_peak_mb': max((used for used, _ in hbm), default=None),
                      'target_hbm_capacity_mb': hbm[0][1] if hbm else None,
                      'host_rss_peak_kib': max(rss, default=None)}, indent=2))
