#!/usr/bin/env python3
"""Prepare isolated GE compile-policy candidates; never initializes Ascend."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path('/export/home/ext.hejun23/workspace/bge/ascend-ge-adapters'))
    parser.add_argument('--out', type=Path, default=Path('/export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026/candidates'))
    args = parser.parse_args()
    base = yaml.safe_load((args.repo / 'models/bge/config.yaml').read_text())
    base['source']['ref'] = str(args.repo.parent / 'model.onnx')
    base['data']['script'] = str(args.repo / base['data']['script'])
    base['data']['directory'] = str(args.repo / base['data']['directory'])
    base['data']['params']['samples_file'] = str(args.repo.parent / 'sample_inputs.json')
    base['verification']['script'] = str(args.repo / base['verification']['script'])
    base['verification']['params']['expected_outputs'] = str(args.repo.parent / 'expected_outputs.json')
    # The floating external input remains FP32; this is not the compute policy.
    base['build']['dtype'] = 'float32'
    # All cores is the agreed baseline. Do not inherit historical 4|8 limit.
    base['build']['options'].pop('aicore_num', None)
    policies = {
        'cube-fp16in-fp32out': 'cube_fp16in_fp32out',
        'auto-mixed-fp16': 'allow_mix_precision_fp16',
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    import copy
    for name, mode in policies.items():
        value = copy.deepcopy(base)
        value['build']['options']['precision_mode'] = mode
        path = args.out / (name + '.yaml')
        path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
        manifest.append({'name': name, 'precision_mode': mode, 'config': str(path),
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'source_unchanged': True, 'build_dtype_is_external_io_dtype': True,
                         'actual_kernel_dtype': 'pending GE final graph and kernel evidence',
                         'device_validation': 'not run by preparation script'})
    (args.out / 'precision-candidates.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
