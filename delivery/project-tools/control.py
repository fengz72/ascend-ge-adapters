import argparse, datetime, hashlib, json, pathlib, subprocess, sys, os

ROOT = pathlib.Path(__file__).resolve().parents[1]
def load(p): return json.loads(pathlib.Path(p).read_text())
def save(p, x): pathlib.Path(p).write_text(json.dumps(x, indent=2))
def digest(p):
    with pathlib.Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def budget():
    registry=load(ROOT/'tools/config-registry.json')
    limit=registry.get('max_device_seconds',7200)
    assert limit is None or (type(limit) is int and limit>0), 'invalid device-time limit'
    jobs=[]
    for folder in sorted((ROOT/'work/jobs').iterdir()):
        if not folder.is_dir(): continue
        p=folder/'status.json'
        if not p.is_file(): raise RuntimeError('Unresolved job without status: '+str(folder))
        s=load(p)
        if s['state']!='finished': raise RuntimeError('Unresolved job: '+str(p))
        seconds=(datetime.datetime.fromisoformat(s['finished_at'])-datetime.datetime.fromisoformat(s['started_at'])).total_seconds()
        jobs.append({'name':p.parent.name,'seconds':seconds,'exit_code':s['exit_code']})
    used=1616.67991+sum(j['seconds'] for j in jobs)
    result={'historical_seconds':1616.67991,'charged_seconds':used,
            'remaining_seconds':None if limit is None else limit-used,
            'max_seconds':limit,'previous_limit_seconds':7200,'jobs':jobs}
    save(ROOT/'work/budget.json',result)
    return result
def gate(name):
    identity=load(ROOT/'tools/identity.json')
    required_chips=identity.get('gate_chips',[0,1])
    assert required_chips and set(required_chips)<={0,1}
    assert identity['logical_device']%2 in required_chips
    rows=[]
    for chip in [0,1]:
        r=subprocess.run(['npu-smi','info','-t','proc-mem','-i',str(identity.get('npu_id',7)),'-c',str(chip)],text=True,capture_output=True,timeout=20)
        rows.append({'chip':chip,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    full=subprocess.run(['npu-smi','info'],text=True,capture_output=True,timeout=20)
    result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'logical_device':identity['logical_device'],'pci':identity['physical_devices'][0].split('/',1)[1],'required_chips':required_chips,'rows':rows,'full_returncode':full.returncode,'full_stdout':full.stdout}
    result['clear']=all(rows[chip]['returncode']==0 and 'No process in device.' in rows[chip]['stdout'] for chip in required_chips)
    save(ROOT/'work'/f'{name}-gate.json',result)
    return result['clear']
def guarded(planfile,name,binary):
    if not gate(name+'-before'): return 76
    try:
        return subprocess.run([sys.executable,str(ROOT/'tools/run_runtime.py'),binary,planfile]).returncode
    finally:
        if not gate(name+'-after'): raise RuntimeError('Device release / exclusive occupancy could not be confirmed')
def run(a):
    plan=load(a.plan); identity=load(ROOT/'tools/identity.json')
    assert plan['device']==identity['logical_device']
    assert plan['options']['graph_run_mode']==0
    assert digest(identity['binary']['path'])==identity['binary']['sha256']
    assert pathlib.Path(plan['output_dir'])==ROOT/'work'/a.job
    assert not (ROOT/'work/jobs'/a.job).exists()
    b=budget()
    if b['remaining_seconds'] is not None and b['remaining_seconds']<a.timeout+35:
        raise RuntimeError('Insufficient finite device budget')
    for key in ['binary','graph','weights']:
        item=identity[key]
        assert digest(item['path'])==item['sha256'], key+' content changed'
    assert plan['graph']['path']==identity['graph']['path']
    env=plan.get('runtime_env',{})
    assert set(env)<= {'LD_PRELOAD'}, 'unregistered native environment'
    preloads=env.get('LD_PRELOAD',os.environ.get('LD_PRELOAD','')).split(':')
    assert all(x in preloads for x in identity['required_preloads'])
    for path in preloads:
        assert pathlib.Path(path).is_file()
        if path in identity['plugins']: assert digest(path)==identity['plugins'][path]
        else: assert path in identity['required_preloads'], 'unregistered preload'
    sig={'binary':identity['binary']['sha256'],'graph':identity['graph']['sha256'],'weights':identity['weights']['sha256'],'options':plan['options'],'instances':plan['instances'],'preloads':[digest(p) for p in preloads]}
    subprocess.run([sys.executable,str(ROOT/'tools/checks/verify_plan_inputs.py'),'--plan',a.plan,'--report',str(ROOT/'work'/f'{a.job}-inputs.json')],check=True)
    registry=load(ROOT/'tools/config-registry.json')
    if a.config in registry['known']:
        assert sig==registry['known'][a.config], 'known configuration identity changed'
    elif a.config in registry['new']:
        assert sig==registry['new'][a.config], 'candidate identity changed'
    else:
        max_new=registry.get('max_new_configs',3)
        assert max_new is None or (type(max_new) is int and max_new>=3), 'invalid configuration budget'
        if max_new is not None and len(registry['new'])>=max_new:
            raise RuntimeError('Configuration budget exhausted')
        registry['new'][a.config]=sig
        save(ROOT/'tools/config-registry.json',registry)
    manifest={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'binary_sha256':identity['binary']['sha256'],'plan_sha256':digest(a.plan),'plan':str(pathlib.Path(a.plan).resolve()),'device':identity['logical_device'],'pci':identity['physical_devices'][0].split('/',1)[1],'env':[],'runtime_env':plan.get('runtime_env',{}),'charged_before_seconds':b['charged_seconds'],'launcher_sha256':digest(ROOT/'tools/run_runtime.py'),'config_key':a.config}
    save(ROOT/'work'/f'{a.job}-identity.json',manifest)
    cmd=[sys.executable,str(ROOT/'tools/device_job.py'),'run','--cwd',str(ROOT/'runner'),'--run-dir',str(ROOT/'work/jobs'/a.job),'--lock-root','/export/home/ext.hejun23/workspace/.npu-workflow-locks',*[v for d in identity['physical_devices'] for v in ['--device',d]],'--timeout',str(a.timeout),'--grace','5','--','bash',str(ROOT/'tools/env_run.sh'),sys.executable,str(ROOT/'tools/control.py'),'guarded',str(pathlib.Path(a.plan).resolve()),'--job',a.job,'--binary',identity['binary']['path']]
    r=subprocess.run(cmd)
    print('BUDGET',json.dumps(budget()),flush=True)
    return r.returncode
def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='op',required=True)
    r=s.add_parser('run');r.add_argument('plan');r.add_argument('--job',required=True);r.add_argument('--config',required=True);r.add_argument('--timeout',type=int,default=350)
    r=s.add_parser('guarded');r.add_argument('plan');r.add_argument('--job',required=True);r.add_argument('--binary',required=True)
    s.add_parser('budget');a=p.parse_args()
    if a.op=='budget':print(json.dumps(budget(),indent=2));return 0
    if a.op=='guarded':return guarded(a.plan,a.job,a.binary)
    return run(a)
if __name__=='__main__':sys.exit(main())
