from pathlib import Path
import subprocess,sys,json,argparse
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--config',default='forcefp16-half10');p.add_argument('--timeout',type=int,default=350);p.add_argument('--formal',action='store_true');a,args=p.parse_known_args()
args=args[1:] if args[:1]==['--'] else args
subprocess.run([sys.executable,str(r/'tools/make_plan.py'),a.name,*args],check=True)
plan=r/'work/plans'/(a.name+'.json')
subprocess.run([sys.executable,str(r/'tools/control.py'),'run',str(plan),'--job',a.name,'--config',a.config,'--timeout',str(a.timeout)],check=True)
subprocess.run([sys.executable,str(r/'tools/checks/validate_run_identity.py'),'--runs',a.name,'--report',str(r/'work'/(a.name+'-run-identity.json'))],check=True)
pv=json.loads(plan.read_text())
if pv['replay']['capture_outputs']:
 folder=r/'work'/a.name
 cmd=[sys.executable,str(r/'tools/checks/verify_outputs.py'),'pressure','--manifest',str(folder/'manifest.jsonl'),'--binary',str(folder/'outputs.bin'),'--events',str(folder/'events.jsonl'),'--instances',str(pv['instances']),'--report',str(r/'work'/(a.name+'-accuracy.json'))]
 subprocess.run(cmd,check=True)
 accuracy=json.loads((r/'work'/(a.name+'-accuracy.json')).read_text())
 result=json.loads((folder/'result.json').read_text())
 assert accuracy['requests']==result['runtime']['performance']['requests'], 'pressure requests differ from runtime'
elif not pv['observability'].get('profiling'):
 cmd=[sys.executable,str(r/'tools/checks/validate_performance.py'),'--performance',str(r/'work'/a.name/'result.json'),'--events',str(r/'work'/a.name/'events.jsonl'),'--plan',str(plan),'--instances',str(pv['instances']),'--report',str(r/'work'/(a.name+'-performance.json'))]
 if not a.formal:cmd+=['--search','--min-seconds','30','--min-requests','2000']
 subprocess.run(cmd,check=True)
print('RUN_ACCEPTED',a.name,flush=True)
