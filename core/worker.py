"""Each worker executes exactly one stage; original models stay unpatched."""
import importlib
import json
import sys
import traceback
from pathlib import Path
import numpy as np
from core.contracts import tensor_array,validate_io
from core.model_api import load_adapter,validate_handle,canonical
from core.sources import load_source
from core.data import select_cases
from core import dependencies

def encode(handle,cases,io,directory):
    expected={x['name']:x for x in io['inputs']};directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    records=[];example=None;path=directory/'inputs.bin'
    with path.open('wb') as stream:
        for case in cases:
            inputs=handle.encode(case)
            if set(inputs)!=set(expected):raise ValueError('encoded input names differ from graph')
            tensors=[]
            for name,value in inputs.items():
                array,dtype=tensor_array(value);node=expected[name]
                if dtype!=node['dtype'] or len(array.shape)!=len(node['shape']) or any(a!=-1 and a!=b for a,b in zip(node['shape'],array.shape)):raise ValueError(f'input contract mismatch: {name}')
                if any(n<1 for n in array.shape):raise ValueError('empty input dimension')
                data=array.tobytes(order='C');offset=stream.tell();stream.write(data)
                tensors.append({'name':name,'dtype':dtype,'shape':list(array.shape),'offset':offset,'bytes':len(data)})
            capacities=handle.output_capacities(case)
            if set(capacities)!={x['name'] for x in io['outputs']} or any(type(n) is not int or n<=0 for n in capacities.values()):raise ValueError('positive per-output capacities are required')
            records.append({'id':case['id'],'samples':case['samples'],'tensors':tensors,'output_capacities':capacities})
            if example is None:example=inputs
    return example,records

def packed(directory,records):return {'file':str(Path(directory)/'inputs.bin'),'cases':records}

def save_rows(path,rows):
    arrays={}
    for case_id,value in rows.items():
        value=canonical(value)
        arrays[case_id+'/metadata/alignment']=np.asarray(json.dumps(value['alignment']))
        for name,array in value['outputs'].items():arrays[case_id+'/outputs/'+name]=np.asarray(array)
    np.savez(path,**arrays)
    return str(path)


def read_row(archive,case_id):
    prefix=case_id+'/outputs/'
    return {'alignment':json.loads(str(archive[case_id+'/metadata/alignment'])),
            'outputs':{key[len(prefix):]:archive[key] for key in archive.files if key.startswith(prefix)}}


def baseline(config,cases,device):
    handle=validate_handle(load_adapter(config['model']['adapter_file']).prepare_reference(load_source(config,reference=True,device=device),config))
    rows={case['id']:handle.decode(handle.run(handle.encode(case)),case) for case in cases}
    path=save_rows(Path(config['work_dir'])/'golden.npz',rows)
    return {'status':'success','golden':{'path':path,'cases':len(rows)}}


def build(config,cases,device):
    from core.backends.runtime import attach_io
    handle=validate_handle(load_adapter(config['model']['adapter_file']).prepare_candidate(load_source(config,device=device),config),candidate=True)
    spec=handle.export_spec();validate_io(spec['io'])
    directory=Path(config['work_dir'])/'build';directory.mkdir(parents=True,exist_ok=True)
    exporter=importlib.import_module('core.exporters.'+('torchair' if spec['format']=='torch' else 'onnx'))
    graph,io=exporter.export(spec,handle.encode(cases[0]),directory)
    if graph['format']=='air':attach_io(graph,io)
    backend=importlib.import_module('core.backends.'+config['build']['backend'])
    om=backend.build(graph,io,config['build']['options'],config['target'],directory)
    return {'status':'success','graph':{**graph,'om_path':str(directory/om) if om else None}}


def infer_or_bench(stage,config,cases,output,device):
    from core.backends.runtime import graph_files,read_io,run
    graph=graph_files(config);io=read_io(graph)
    # Runtime recipes encode/decode arrays and size outputs without loading weights.
    handle=load_adapter(config['model']['adapter_file']).prepare_runtime(config,io)
    try:
        _,records=encode(handle,cases,io,output)
        result=run(config,graph,io,packed(output,records),output,stage,device)
        if stage=='bench':
            return {'status':result['status'],'graph':graph,
                    **{key:result[key] for key in ('error','performance','profiling','dump') if key in result}}
        rows={};raw=Path(output)/'outputs.bin'
        for case in cases:
            actuals=result['outputs'][case['id']]
            if len(actuals)!=1 or actuals[0]['instance']!=0:raise ValueError('infer requires one output per case')
            values={}
            for node in actuals[0]['tensors']:
                dtype='uint16' if node['dtype']=='bfloat16' else node['dtype']
                value=np.fromfile(raw,dtype=dtype,count=int(np.prod(node['shape'])),offset=node['offset']).reshape(node['shape'])
                if node['dtype']=='bfloat16':value=(value.astype(np.uint32)<<16).view(np.float32)
                values[node['name']]=value
            rows[case['id']]=handle.decode(values,case)
        path=save_rows(Path(config['work_dir'])/'outputs.npz',rows)
        return {'status':'success','graph':graph,'outputs':{'path':path,'cases':len(rows)},
                **{key:result[key] for key in ('profiling','dump') if key in result}}
    finally:
        (Path(output)/'inputs.bin').unlink(missing_ok=True)
        (Path(output)/'outputs.bin').unlink(missing_ok=True)


def compare_outputs(config):
    from core.verify import compare,compare_business
    directory=Path(config['work_dir']);policies=config['verification']['outputs'];reports={}
    if not policies:raise ValueError('verification.outputs must configure every business output')
    def case_ids(archive):
        return {key.partition('/')[0] for key in archive.files if key.partition('/')[2]=='metadata/alignment'}
    with np.load(directory/'golden.npz',allow_pickle=False) as golden, np.load(directory/'outputs.npz',allow_pickle=False) as outputs:
        available=sorted(case_ids(golden))
        selection=config['verification']['case_ids']
        actual=case_ids(outputs)
        if selection=='all' and set(available)!=actual:
            raise ValueError(f'golden/output case IDs differ: missing outputs={sorted(set(available)-actual)}, missing golden={sorted(actual-set(available))}')
        ids=available if selection=='all' else [case['id'] for case in select_cases([{'id':name} for name in available],selection)]
        if not ids:raise ValueError('golden contains no cases')
        business_rows={}
        for case_id in ids:
            candidate=read_row(outputs,case_id)
            reports[case_id]=compare(read_row(golden,case_id),candidate,policies)
            if config['verification'].get('script'):business_rows[case_id]=candidate
        for case_id,business in compare_business(config,business_rows).items():
            reports[case_id]['business']=business
            reports[case_id]['passed']=reports[case_id]['passed'] and business['passed']
    passed=all(report['passed'] for report in reports.values())
    return {'status':'success' if passed else 'failed',
            'accuracy':{'status':'passed' if passed else 'failed','case_ids':ids,
                        'failed_cases':[name for name,report in reports.items() if not report['passed']], 'cases':reports}}


def execute(job):
    config=job['config'];cases=job['cases'];stage=job['stage'];output=Path(job['output']);device=job['device']
    if stage=='baseline':return baseline(config,cases,device)
    if stage=='compare':return compare_outputs(config)
    if stage=='build':
        dependencies.activate(config)
        return build(config,cases,device)
    if stage in ('infer','bench'):return infer_or_bench(stage,config,cases,output,device)
    raise ValueError(f'unknown worker step: {stage}')


def main():
    job=json.load(sys.stdin)
    try:print('GE_RESULT '+json.dumps(execute(job),ensure_ascii=False))
    except Exception as error:
        traceback.print_exc()
        print('GE_ERROR '+json.dumps(f'{type(error).__name__}: {error}'),flush=True)
        raise SystemExit(1)

if __name__=='__main__':main()
