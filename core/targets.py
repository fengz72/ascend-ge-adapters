"""Exact chip catalog and device identity; no model or optimization policy."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import yaml
from core.contracts import exact_keys

CATALOG = Path(__file__).resolve().parent.parent / 'targets/targets.yaml'
TARGET_KEYS = {'id', 'family', 'soc', 'sku', 'expected_capacity'}


def validate_target(target):
    exact_keys(target, TARGET_KEYS, 'target', ('id', 'family', 'soc'))
    for key in ('id', 'family', 'soc'):
        if not isinstance(target[key], str) or not target[key].strip():
            raise ValueError(f'target.{key} must be a nonempty string')
    capacity = target.get('expected_capacity', {})
    exact_keys(capacity, ('aic', 'aiv'), 'target.expected_capacity')
    if any(type(value) is not int or value < 1 for value in capacity.values()):
        raise ValueError('target.expected_capacity values must be positive integers; omit unknown values')
    return target


def read_catalog(path=CATALOG):
    with open(path) as stream:
        catalog = yaml.safe_load(stream)
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError('target catalog must map platform families to exact chip names')
    entries = {}
    for family, chips in catalog.items():
        if not isinstance(family, str) or not family or not isinstance(chips, dict) or not chips:
            raise ValueError('each target family must contain a nonempty chip mapping')
        for soc, values in chips.items():
            exact_keys(values, ('sku', 'expected_capacity'), f'catalog.{family}.{soc}')
            target = validate_target(dict(values, id=soc, soc=soc, family=family))
            if soc in entries:
                raise ValueError(f'duplicate chip in target catalog: {soc}')
            entries[soc] = target
    return entries


def require_device(device):
    if type(device) is not int or device < 0:
        raise ValueError('an explicit nonnegative NPU index is required (--device)')
    return device


def inspect_device(device):
    device = require_device(device)
    import torch
    import torch_npu
    torch_npu.npu.set_device(device)
    props = torch.npu.get_device_properties(device)
    capacity = {}
    for kind, names in {'aic': ('cube_core_num', 'aicore_num', 'ai_core_count'),
                        'aiv': ('vector_core_num', 'vector_core_count')}.items():
        value = next((getattr(props, name) for name in names if hasattr(props, name)), None)
        if value is not None:
            capacity[kind] = int(value)
    return {'device': device, 'soc': props.name, 'capacity': capacity}


def inspect_device_isolated(device):
    """Keep device contexts out of the CLI/orchestrator process."""
    device = require_device(device)
    code = ('import json,sys; from core.targets import inspect_device; '
            'print(json.dumps(inspect_device(int(sys.argv[1]))))')
    result = subprocess.run([sys.executable, '-c', code, str(device)],
                            cwd=CATALOG.parent.parent, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'NPU {device} detection failed: {result.stderr.strip()}')
    return json.loads(result.stdout)


def confirm_target(target, hardware):
    if hardware['soc'] != target['soc']:
        raise ValueError(f"target SoC {target['soc']} != device {hardware['soc']}")
    for key, value in target.get('expected_capacity', {}).items():
        if hardware['capacity'].get(key) != value:
            raise ValueError(f"cannot confirm target capacity {key}={value}; detected {hardware['capacity']}")


def select_target(selector='auto', *, device=None, catalog=CATALOG):
    # Inline identities remain useful for private chips, without catalog mutation.
    if isinstance(selector, dict):
        target = validate_target(copy.deepcopy(selector))
        if device is not None:
            confirm_target(target, inspect_device_isolated(device))
        return target
    entries = read_catalog(catalog)
    families = {target['family'] for target in entries.values()}
    if not isinstance(selector, str) or selector not in {'auto', *entries, *families}:
        raise ValueError(f'unknown target {selector!r}; use an exact SoC from {catalog}')
    if device is not None:
        hardware = inspect_device_isolated(device)
        if hardware['soc'] not in entries:
            raise ValueError(f"device SoC {hardware['soc']} is not registered in {catalog}; add its exact identity")
        target = entries[hardware['soc']]
        if selector in families and target['family'] != selector:
            raise ValueError(f"requested platform {selector} != detected {target['family']}")
        if selector in entries:
            confirm_target(entries[selector], hardware)
        confirm_target(target, hardware)
    elif selector in entries:
        target = entries[selector]
    else:
        raise ValueError(f'target {selector!r} requires --device for chip detection; use --target with an exact SoC for offline configuration')
    return copy.deepcopy(target)
