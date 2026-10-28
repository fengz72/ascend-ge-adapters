import pathlib
import subprocess
import sys

root = pathlib.Path('/export/home/ext.hejun23/workspace/bge-opt-100')
round_number = int(sys.argv[1])
assert round_number in (1, 2, 3)
seed = 2026 + round_number


def run(label, kind):
    is_best = label == 'best'
    name = f'final-{label}-{kind}-r{round_number}'
    arguments = ['--instances', '8' if is_best else '1',
                 '--seconds', '60', '--max-seconds', '300',
                 '--requests', '10000', '--seed', str(seed)]
    if is_best:
        arguments.extend(['--cores', '12|24'])
    if kind == 'fixed':
        arguments.append('--fixed')
    command = [sys.executable, str(root / 'tools/run_one.py'),
               '--formal', '--config', 'half8' if is_best else 'base',
               '--timeout', '350', name, '--', *arguments]
    print('FORMAL_START', name, flush=True)
    subprocess.run(command, check=True)
    print('FORMAL_ACCEPTED', name, flush=True)


order = [('base', 'fixed'), ('best', 'fixed'),
         ('base', 'random'), ('best', 'random')]
if round_number == 2:
    order = [('best', 'fixed'), ('base', 'fixed'),
             ('best', 'random'), ('base', 'random')]
for item in order:
    run(*item)
print('FORMAL_ROUND_DONE', round_number, flush=True)
