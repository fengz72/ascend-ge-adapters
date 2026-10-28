#!/usr/bin/env python3
"""Build V2 and perform only CPU contracts, preload and child-env probes."""
import argparse
from pathlib import Path
import hashlib
import json
import os
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cann', type=Path, default=Path('/usr/local/Ascend/cann-9.0.0'))
    p.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args()
    b = a.directory
    base = ['g++', '-std=c++17', '-O2', '-D_GLIBCXX_USE_CXX11_ABI=0', '-I' + str(a.cann/'include')]
    libs = ['-L' + str(a.cann/'lib64'), '-Wl,-rpath,' + str(a.cann/'lib64'),
            '-lge_compiler', '-lge_common', '-lge_common_base', '-lgraph', '-lgraph_base', '-lregister', '-ldl']
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(a.cann/'lib64') + ':' + env.get('LD_LIBRARY_PATH', '')
    logs = []
    def run(cmd, **kwargs):
        r = subprocess.run([str(x) for x in cmd], text=True, capture_output=True, timeout=120, **kwargs)
        logs.append({'argv': [str(x) for x in cmd], 'returncode': r.returncode,
                     'stdout': r.stdout, 'stderr': r.stderr})
        if r.returncode:
            raise RuntimeError('CPU command failed: ' + repr(cmd) + '\n' + r.stdout + r.stderr)
        return r
    # Replace the newly built library atomically; never truncate an existing map.
    target = b/'libbge_add_layernorm_pass_v2.so'
    temporary = b/'libbge_add_layernorm_pass_v2.build.so'
    run(base + ['-fPIC','-shared',str(b/'add_layernorm_pass_v2.cpp')] + libs + ['-o',str(temporary)])
    temporary.replace(target)
    run(base + [str(b/'test_add_layernorm_pass.cpp')] + libs + ['-o',str(b/'test_add_layernorm_pass_v2')])
    run([b/'test_add_layernorm_pass_v2'], env=env)
    run(base + [str(b/'test_preload_scope.cpp'), '-o', str(b/'test_preload_scope')])
    run([b/'test_preload_scope', b/'v2-filter-fixtures'], env=env)
    probes = b/'v2-cpu-probes'; probes.mkdir(exist_ok=True)
    run(base + [str(b/'preload_probe.cpp'), '-o',str(probes/'ge_runtime')])
    run(base + [str(b/'preload_probe.cpp'), '-o',str(probes/'cpu_probe')])
    # Preserve one real pre-existing library and place self in a mixed-delimiter
    # value. The native runner remains mapped; only future child inheritance changes.
    other = '/usr/lib64/libtcmalloc.so.4.5.16'
    if not Path(other).exists():
        raise RuntimeError('CPU integration probe requires the actual installed malloc preload')
    probe_env = dict(env, LD_PRELOAD=other + ' ' + str(target))
    native = run([probes/'ge_runtime'], env=probe_env)
    assert 'LD_PRELOAD='+other+'\n' in native.stdout and 'SELF_MAPPED=1' in native.stdout
    assert '[BGE_PRELOAD_SCOPE] removed self' in native.stderr
    child = run([probes/'ge_runtime', '--child-python'], env=probe_env)
    assert 'CHILD_LD_PRELOAD='+other+'\n' in child.stdout and 'CHILD_SELF_MAPPED=False' in child.stdout
    non_native = run([probes/'cpu_probe'], env=probe_env)
    assert 'LD_PRELOAD='+probe_env['LD_PRELOAD']+'\n' in non_native.stdout
    assert 'SELF_MAPPED=1' in non_native.stdout and '[BGE_PRELOAD_SCOPE]' not in non_native.stderr
    reloc = run(['ldd','-r',target], env=env)
    assert 'undefined symbol' not in reloc.stdout+reloc.stderr and 'not found' not in reloc.stdout+reloc.stderr
    names = ['add_layernorm_pass_v2.cpp','preload_scope_filter.h','test_preload_scope.cpp',
             'preload_probe.cpp','build_add_layernorm_v2.py','libbge_add_layernorm_pass_v2.so']
    artifacts = [{'path': str(b/name), 'bytes': (b/name).stat().st_size,
                  'sha256': hashlib.sha256((b/name).read_bytes()).hexdigest()} for name in names]
    report = {'status':'pass','npu_initialized':False,'fusion_graph_contract_tests':7,
              'environment_filter_tests':11,'integration_checks':['native maps self but child env excludes self',
                    'native child Python does not map self','other executable preserves original preload'],
              'ldd_r_no_missing_or_undefined':True, 'commands':logs,'artifacts':artifacts}
    (b/'v2-cpu-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='commands'}, indent=2))


if __name__=='__main__':
    main()
