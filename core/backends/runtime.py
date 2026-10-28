"""Native execution and self-contained graph IO; no build or data generation."""
import json
import subprocess
import tempfile
from pathlib import Path
from core.config import ROOT
from core.contracts import validate_io
from core.targets import require_device

RUNTIME=ROOT/'runtime/build/ge_runtime'


def call(plan,path=None):
    if not RUNTIME.is_file():raise FileNotFoundError(f'{RUNTIME}; build runtime first: bash runtime/build.sh')
    # CANN compiler workers can inherit stdout and outlive a crashed runtime.
    # A regular file lets us report the parent's failure without waiting for pipe EOF.
    with (Path(path).open('w+',encoding='utf-8') if path else tempfile.TemporaryFile(mode='w+',encoding='utf-8')) as log:
        process=subprocess.run([str(RUNTIME)],input=json.dumps(plan),text=True,stdout=log,stderr=subprocess.STDOUT)
        log.seek(0);output=log.read()
    messages=[line[len('GE_RESULT '):] for line in output.splitlines() if line.startswith('GE_RESULT ')]
    if len(messages)!=1:
        raise RuntimeError(f'GE execution (exit {process.returncode}): '+output[-12000:]+(f'\nlog: {path}' if path else ''))
    result=json.loads(messages[0])
    if process.returncode:
        detail=f'GE execution (exit {process.returncode}): '+output[-12000:]+(f'\nlog: {path}' if path else '')
        if result.get('status')!='failed':raise RuntimeError(detail)
        result['error']=detail
    return result


def attach_io(graph,io):
    return call({'mode':'write_io','graph':graph,'io':validate_io(io)})['io']


def read_io(graph):
    if graph['format']=='onnx':
        import onnx
        model=onnx.load(graph['path'],load_external_data=False)
        values={item.key:item.value for item in model.metadata_props}
        if 'ge_adapters_io' not in values:raise ValueError(f"{graph['path']}: missing embedded IO; run build")
        return validate_io(json.loads(values['ge_adapters_io']))
    return validate_io(call({'mode':'read_io','graph':graph})['io'])


def graph_files(config):
    kind='onnx' if config['source']['kind']=='onnx' else 'air'
    path=Path(config['work_dir'])/'build'/f'model.{kind}'
    if not path.is_file():raise FileNotFoundError(f'{path}; run build first')
    om=path.with_suffix('.om') if config['build']['backend']=='atc_om' else None
    if om and not om.is_file():raise FileNotFoundError(f'{om}; run build first')
    return {'format':kind,'path':str(path),'om_path':str(om) if om else None}


def run(config,graph,io,inputs,output,mode,device):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plan={'backend':config['build']['backend'],'graph':graph,'io':io,'inputs':inputs,
          'device':require_device(device),'instances':config['benchmark']['instances'] if mode=='bench' else 1,
          'warmup':config['benchmark']['warmup'],'iterations':config['benchmark']['iterations'],
          'seed':config['benchmark']['seed'],'options':config['build']['options'],
          'observability':config['observability'],'output_dir':str(output.resolve()),'mode':mode}
    return call(plan,output/f'{mode}-runtime.log')
