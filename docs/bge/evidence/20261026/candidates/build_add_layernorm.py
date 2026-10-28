#!/usr/bin/env python3
"""Build the isolated CANN9 pass and run CPU graph contracts only."""
import argparse
from pathlib import Path
import os
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cann', type=Path, default=Path('/usr/local/Ascend/cann-9.0.0'))
    p.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args()
    base = ['g++', '-std=c++17', '-O2', '-D_GLIBCXX_USE_CXX11_ABI=0',
            '-I' + str(a.cann / 'include')]
    libs = ['-L' + str(a.cann / 'lib64'), '-Wl,-rpath,' + str(a.cann / 'lib64'),
            '-lge_compiler', '-lge_common', '-lge_common_base', '-lgraph', '-lgraph_base', '-lregister']
    for source, output, flags in [
            ('add_layernorm_pass.cpp', 'libbge_add_layernorm_pass.so', ['-fPIC', '-shared']),
            ('test_add_layernorm_pass.cpp', 'test_add_layernorm_pass', [])]:
        subprocess.run(base + flags + [str(a.directory / source)] + libs + ['-ldl'] +
                       ['-o', str(a.directory / output)], check=True, timeout=120)
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(a.cann / 'lib64') + ':' + env.get('LD_LIBRARY_PATH', '')
    subprocess.run([str(a.directory / 'test_add_layernorm_pass')], env=env, check=True, timeout=60)


if __name__ == '__main__':
    main()
