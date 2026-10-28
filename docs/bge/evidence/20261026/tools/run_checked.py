import argparse,pathlib,json,subprocess,sys,datetime,hashlib,time,os
ROOT=pathlib.Path('/export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026')
p=argparse.ArgumentParser();p.add_argument('plan');p.add_argument('--job',required=True);p.add_argument('--binary',default=str(ROOT/'runner/runtime/build/ge_runtime'));p.add_argument('--timeout',type=int,default=650);p.add_argument('--env',action='append',default=[]);a=p.parse_args()
plan=json.loads(pathlib.Path(a.plan).read_text());assert plan['device']==10
job=ROOT/'jobs'/a.job
if job.exists():raise SystemExit('refusing to replace existing job')
used=0
for f in (ROOT/'jobs').glob('*/status.json'):
 s=json.loads(f.read_text())
 if s['state']!='finished':raise SystemExit('previous job not confirmed finished: '+str(f))
 used+=(datetime.datetime.fromisoformat(s['finished_at'])-datetime.datetime.fromisoformat(s['started_at'])).total_seconds()
if used+a.timeout>7200:raise SystemExit('finite device budget insufficient')
def gate(label):
 q=subprocess.run(['npu-smi','info','-t','proc-mem','-i','5','-c','0'],text=True,capture_output=True,timeout=30)
 row={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
 (ROOT/(a.job+'-'+label+'.json')).write_text(json.dumps(row,indent=2))
 return q.returncode==0 and 'No process in device.' in q.stdout
if not gate('before'):raise SystemExit('device10 occupied; no job launched')
manifest={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'binary_sha256':hashlib.sha256(pathlib.Path(a.binary).read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(pathlib.Path(a.plan).read_bytes()).hexdigest(),'plan':a.plan,'device':10,'pci':'0000:89:00.0','env':a.env,'runtime_env':plan.get('runtime_env',{}),'charged_before_seconds':used,'launcher_sha256':hashlib.sha256((ROOT/'tools/run_runtime.py').read_bytes()).hexdigest()}
(ROOT/(a.job+'-identity.json')).write_text(json.dumps(manifest,indent=2))
cmd=[sys.executable,str(ROOT/'tools/device_job.py'),'run','--cwd',str(ROOT/'runner'),'--run-dir',str(job),'--lock-root','/export/home/ext.hejun23/workspace/bge/.npu-workflow-locks','--device','A03-R40-I191-99-4100006.JD.LOCAL/0000:89:00.0','--timeout',str(a.timeout),'--grace','5','--','bash',str(ROOT/'tools/env_run.sh'),'env',*a.env,sys.executable,str(ROOT/'tools/run_runtime.py'),a.binary,a.plan]
r=subprocess.run(cmd)
print('CHECKED_JOB_EXIT',r.returncode,flush=True)
clean=gate('after');print('DEVICE10_RELEASED',clean,flush=True)
if not clean:raise SystemExit(75)
raise SystemExit(r.returncode)
