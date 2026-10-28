import argparse,shlex,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--host',default='100');p.add_argument('--script');p.add_argument('--timeout',type=int,default=600);p.add_argument('command',nargs='?');a=p.parse_args()
argv=['docker','exec','-i','hejun-cann9-bge','python3','-'] if a.script else ['docker','exec','hejun-cann9-bge','bash','-lc',a.command]
data=open(a.script,'rb').read() if a.script else None
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
                  '-o','ControlMaster=auto','-o','ControlPersist=1800',
                  '-o','ControlPath=/tmp/codex-bge100-stability',
                  a.host,shlex.join(argv)],input=data,timeout=a.timeout)
sys.exit(r.returncode)
