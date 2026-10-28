import json
import pathlib
import shlex
import subprocess
import sys

code = sys.stdin.read()
result = subprocess.run(
    ['ssh', '-o', 'BatchMode=yes', '99', shlex.join(['docker', 'exec', 'hejun-cann9-bge', 'python3', '-c', code])],
    capture_output=True, text=True, timeout=150,
)
if result.returncode:
    print(result.stderr, file=sys.stderr)
    raise SystemExit(result.returncode)
print(result.stdout, end='')
