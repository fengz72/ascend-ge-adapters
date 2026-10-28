import pathlib,sys,json,subprocess,time,os,threading
binary,planfile=sys.argv[1:];plan=json.loads(pathlib.Path(planfile).read_text());out=pathlib.Path(plan['output_dir']);out.mkdir(parents=True,exist_ok=False)
beg=time.monotonic();print('EXECUTING',binary,planfile,flush=True)
stop=threading.Event();samples=[]
def monitor(pid):
 while not stop.is_set():
  stamp=time.monotonic_ns();row={'monotonic_ns':stamp,'elapsed_seconds':time.monotonic()-beg,'runtime_pid':pid}
  try:
   status=pathlib.Path('/proc/'+str(pid)+'/status').read_text();row['rss_kib']=int(next(x for x in status.splitlines() if x.startswith('VmRSS:')).split()[1])
  except (OSError,StopIteration,ValueError):row['rss_kib']=None
  if plan.get('monitor_resources',False):
   try:
    q=subprocess.run(['npu-smi','info'],text=True,capture_output=True,timeout=8)
    row['npu_smi_code']=q.returncode;row['npu_smi_stdout']=q.stdout;row['npu_smi_stderr']=q.stderr
   except subprocess.TimeoutExpired:row['npu_smi_code']='timeout'
  samples.append(row)
  if stop.wait(10):break
with (out/'runtime.log').open('w') as log:
 binary_env=os.environ.copy();binary_env.update(plan.get('runtime_env',{}))
 process=subprocess.Popen([binary],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True,env=binary_env)
 watcher=threading.Thread(target=monitor,args=(process.pid,),daemon=True);watcher.start()
 try:
  process.communicate(json.dumps(plan));code=process.returncode
 finally:
  stop.set();watcher.join(timeout=10)
(out/'resources.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in samples))
lines=(out/'runtime.log').read_text(errors='replace').splitlines();rows=[x[10:] for x in lines if x.startswith('GE_RESULT ')]
result={'returncode':code,'elapsed_seconds':time.monotonic()-beg,'plan':planfile,'binary':binary}
if len(rows)==1:result['runtime']=json.loads(rows[0])
(out/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='runtime'}),flush=True);print('LOG_TAIL','\n'.join(lines[-16:]),flush=True)
sys.exit(code if code else (0 if len(rows)==1 else 70))
