"""Python and command-line entry points for seven independent operations."""
import argparse
import datetime
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from core.config import ROOT, load_authoring, validate
from core.contracts import write_json, digest
from core import data, dependencies

STEPS=('data','baseline','prepare','build','infer','compare','bench')


def require_success(stage,result,directory):
    if result.get('status')=='failed':
        detail=result.get('error') or f"accuracy failed for {result.get('accuracy',{}).get('failed_cases',[])}"
        raise ValueError(f"{stage}: {detail}; see {directory/f'{stage}.log'}")


def worker(stage,config,cases,directory,device):
    job={'stage':stage,'config':config,'cases':cases,'output':str(directory),'device':device}
    path=directory/f'{stage}.log'
    env=dependencies.environment(config) if stage in ('build','infer','bench') else None
    with path.open('w') as log:
        process=subprocess.run([sys.executable,'-m','core.worker'],input=json.dumps(job),text=True,
                               stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
    lines=path.read_text(errors='replace').splitlines()
    messages=[line[len('GE_RESULT '):] for line in lines if line.startswith('GE_RESULT ')]
    if process.returncode or len(messages)!=1:
        errors=[line[len('GE_ERROR '):] for line in lines if line.startswith('GE_ERROR ')]
        detail=json.loads(errors[-1]) if errors else '\n'.join(lines[-20:])
        raise RuntimeError(f'{stage}: {detail}\nlog: {path}')
    return json.loads(messages[0])


def perform(stage,config,directory,device):
    if stage=='data':
        cases=data.generate(config)
        return {'status':'success','cases':len(cases),'samples':sum(c['samples'] for c in cases),
                'directory':config['data']['directory'],'pool_sha256':digest(cases)}
    if stage=='prepare':return {'status':'success','dependencies':dependencies.prepare(config)}
    if stage=='compare':return worker(stage,config,[],directory,device)
    cases=data.load(config)
    if stage in ('baseline','infer','compare'):cases=data.select_cases(cases,config['verification']['case_ids'])
    elif stage=='build':cases=cases[:1]
    return worker(stage,config,cases,directory,device)


def execute(stage,config,*,device=None,target=None,targets=None,case_ids=None,instances=None,profiling=None):
    """Read one current model YAML and execute a step, or sequence all seven."""
    if stage not in (*STEPS,'run'):raise ValueError(f'unknown step: {stage}')
    if stage in ('build','infer','bench','run'):
        from core.targets import require_device
        device=require_device(device)
    options={'target':target,'device':device,'source_paths':stage in ('data','baseline','build','run')}
    if targets is not None:options['catalog']=targets
    config=load_authoring(Path(config),**options)
    if case_ids is not None:config['verification']['case_ids']='all' if case_ids==['all'] else case_ids
    if instances is not None:config['benchmark']['instances']=instances
    if profiling is not None:config['observability']['profiling']=profiling
    validate(config)
    if stage=='baseline' and config['reference']['device']=='npu':
        from core.targets import require_device
        device=require_device(device)
    started=datetime.datetime.now(datetime.timezone.utc)
    run_id=started.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    directory=Path(config['work_dir'])/'results'/run_id;directory.mkdir(parents=True)
    began=time.monotonic()
    record={'schema':'ge-run/1','run_id':run_id,'started_at':started.isoformat(),
            'status':'running','stage':stage,'device':device,'config':config}
    try:
        if stage=='run':
            record['steps']={}
            for step in STEPS:
                result=perform(step,config,directory,device);record['steps'][step]=result
                require_success(step,result,directory)
        else:
            record.update(perform(stage,config,directory,device))
            require_success(stage,record,directory)
        record['status']='success'
    except Exception as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        record['duration_seconds']=time.monotonic()-began
        write_json(directory/'result.json',record)
    return {'status':record['status'],'result':str(directory/'result.json'),
            **{k:record[k] for k in ('steps','performance','accuracy','graph','golden','outputs','cases','samples','directory','pool_sha256','profiling','dump') if k in record}}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Independent model adaptation steps')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in (*STEPS,'run'):
        p=sub.add_parser(command)
        p.add_argument('--config',required=True,help='model YAML')
        p.add_argument('--device',type=int,required=command in ('build','infer','bench','run'))
        p.add_argument('--target');p.add_argument('--targets')
        if command in ('baseline','infer','compare','run'):p.add_argument('--case-ids',nargs='+')
        if command in ('bench','run'):p.add_argument('--instances',type=int)
        if command in ('infer','bench','run'):p.add_argument('--profiling',action='store_true',default=None)
    p=sub.add_parser('report');p.add_argument('results',nargs='+');p.add_argument('--output')
    args=vars(parser.parse_args(argv));command=args.pop('command')
    try:
        if command=='report':
            from core.results import report
            result=report(args['results'])
            if args['output']:Path(args['output']).write_text(result+'\n')
            print(result);return
        print(json.dumps(execute(command,**args),ensure_ascii=False,indent=2))
    except Exception as error:
        print(f'{command}: {type(error).__name__}: {error}',file=sys.stderr)
        raise SystemExit(1)


if __name__=='__main__':main()
