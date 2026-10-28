"""Load one effective YAML mapping per invocation; no config snapshots."""
import copy
import os
import re
from pathlib import Path
import yaml
from core.contracts import exact_keys, safe_name, resolve
from core.data import validate_case_ids
from core.targets import CATALOG, select_target, validate_target

ROOT=Path(__file__).resolve().parent.parent
SECTIONS={
    'model': {'name','adapter_file','params'},
    'source': {'kind','ref','loader','factory','params','files'},
    'reference': {'device','dtype'},
    'data': {'script','directory','params'},
    'build': {'backend','dtype','options'},
    'verification': {'case_ids','outputs','script','params'},
    'benchmark': {'instances','warmup','iterations','seed'},
    'observability': {'profiling','dump','aic_metrics'},
}
DEFAULTS={
    'model': {'params': {}}, 'reference': {'device':'cpu','dtype':'float32'},
    'data': {'params': {}}, 'build': {'backend':'ge_session','dtype':'float32','options':{}},
    'verification': {'case_ids':'all','outputs':{}},
    'benchmark': {'instances':1,'warmup':10,'iterations':1000,'seed':42},
    'dependencies': {'custom_ops':[],'ge_passes':[],'parser_plugins':[]},
    'observability': {'profiling':False,'dump':False},
}
def validate_dtype(value,field,*,floating_only=False):
    import numpy as np
    if not isinstance(value,str) or not value:
        raise ValueError(f'{field} must name a concrete numeric or boolean dtype')
    # BF16 has native GE/ACL support but is not a built-in NumPy dtype.
    if value=='bfloat16':return value
    try:dtype=np.dtype(value)
    except (TypeError,ValueError) as error:
        raise ValueError(f'{field} is not a concrete dtype: {value!r}') from error
    if dtype.name!=value:
        raise ValueError(f'{field} must use a concrete canonical dtype name: {dtype.name}')
    supported=(dtype.kind=='f' and dtype.itemsize in (2,4,8) or
               dtype.kind in 'iu' and dtype.itemsize in (1,2,4,8) or
               dtype.kind=='b')
    if not supported:
        raise ValueError(f'{field}={value} is unsupported by native tensor IO')
    if floating_only and dtype.kind!='f':
        raise ValueError(f'{field}={value} cannot cast Torch model parameters; a floating dtype is required')
    return value

def merge(base, overlay):
    result=copy.deepcopy(base)
    for key,value in overlay.items():
        result[key]=merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else copy.deepcopy(value)
    return result

def expand_source_path(value, field):
    def replace(match):
        name=match.group(1);content=os.environ.get(name)
        if content is None or not content.strip():raise ValueError(f'{field}: environment variable {name} is not set or is empty')
        return content
    return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}',replace,value)

def validate(config):
    exact_keys(config,set(SECTIONS)|{'target','dependencies','work_dir','_root'},'config',('model','source','target'))
    for section,keys in SECTIONS.items():exact_keys(config[section],keys,section)
    exact_keys(config['model'],SECTIONS['model'],'model',('name','adapter_file','params'))
    safe_name(config['model']['name'])
    if not isinstance(config['model']['params'],dict) or not isinstance(config['data']['params'],dict):raise ValueError('model.params/data.params must be mappings')
    if 'directory' in config['data'] and (not isinstance(config['data']['directory'],str) or not config['data']['directory'].strip()):raise ValueError('data.directory must be a nonempty path')
    kind=config['source'].get('kind')
    required={'transformers':('ref','loader'),'torch':('factory',),'onnx':('ref',)}
    if kind not in required:raise ValueError('source.kind must be transformers, torch or onnx')
    for name in required[kind]:
        value=config['source'].get(name)
        if not isinstance(value,str) or not value.strip():raise ValueError(f'source.{name} must be a nonempty string')
    validate_target(config['target'])
    if config['build']['backend'] not in ('ge_session','atc_om'):raise ValueError('unknown build.backend')
    exact_keys(config['build']['options'],('aicore_num','precision_mode','graph_run_mode'),'build.options')
    if config['build']['backend']=='atc_om' and 'graph_run_mode' in config['build']['options']:raise ValueError('graph_run_mode is a GE Session option')
    for section in ('build','reference'):
        validate_dtype(config[section]['dtype'],f'{section}.dtype',floating_only=kind!='onnx')
    if config['reference']['device'] not in ('cpu','npu'):raise ValueError('reference.device must be cpu or npu')
    validate_case_ids(config['verification']['case_ids'])
    if 'script' in config['verification'] and (not isinstance(config['verification']['script'],str) or not config['verification']['script'].strip()):raise ValueError('verification.script must be a nonempty path')
    if not isinstance(config['verification'].get('params',{}),dict):raise ValueError('verification.params must be a mapping')
    for name,policy in config['verification']['outputs'].items():
        exact_keys(policy,('policy','scope','cosine_min','relative_l2_max','rtol','atol','max_abs_error','top_k'),f'verification.outputs.{name}')
        if policy.get('policy','numeric') not in ('numeric','exact','topk'):raise ValueError('unsupported accuracy policy')
        if policy.get('scope','per_sample') not in ('per_sample','tensor'):raise ValueError('unsupported accuracy scope')
    b=config['benchmark']
    if any(type(b[k]) is not int or b[k]<1 for k in ('instances','iterations')) or type(b['warmup']) is not int or b['warmup']<0:raise ValueError('invalid benchmark counts')
    if type(b['seed']) is not int or not 0<=b['seed']<=2**64-1:raise ValueError('benchmark.seed must be uint64')
    exact_keys(config['dependencies'],('custom_ops','ge_passes','parser_plugins'),'dependencies')
    ids=[]
    for entries in config['dependencies'].values():
        if not isinstance(entries,list):raise ValueError('dependency entries must be lists')
        for entry in entries:
            exact_keys(entry,('id','source','revision','build_script','python_imports'),'dependency',('id','source','build_script'))
            safe_name(entry['id']);ids.append(entry['id'])
    if len(ids)!=len(set(ids)):raise ValueError('duplicate dependency IDs')
    return config

def load_authoring(paths, *, target=None, device=None, catalog=CATALOG, source_paths=True):
    if isinstance(paths,(str,Path)):paths=[paths]
    layers=[];common=copy.deepcopy(DEFAULTS)
    for path in paths:
        if Path(path).suffix.lower() not in ('.yaml','.yml'):raise ValueError('commands require YAML --config')
        layer=yaml.safe_load(Path(path).read_text())
        exact_keys(layer,set(SECTIONS)|{'target','dependencies','work_dir','platforms'},str(path))
        base={k:v for k,v in layer.items() if k != 'platforms'}
        common=merge(common,base);layers.append((base,layer.get('platforms',{})))
    if not layers:raise ValueError('at least one YAML is required')
    chosen=select_target(target if target is not None else common.get('target','auto'),device=device,catalog=catalog)
    config=copy.deepcopy(DEFAULTS)
    for base,platforms in layers:
        if not isinstance(platforms,dict):raise ValueError('platforms must be a mapping')
        for family,branch in platforms.items():
            exact_keys(branch,(set(SECTIONS)-{'target'})|{'dependencies','work_dir'},f'platforms.{family}')
            probe=merge(common,branch);probe['target']=chosen
            if probe['source'].get('kind')=='onnx':probe['model'].setdefault('adapter_file','core/native_onnx.py')
            validate(probe)
        if platforms and chosen['family'] not in platforms:raise ValueError(f"no platform configuration for {chosen['family']}")
        for layer in (base,platforms.get(chosen['family'],{})):
            config=merge(config,layer)
    config['target']=chosen
    if config['source']['kind']=='onnx':
        config['model'].setdefault('adapter_file','core/native_onnx.py')
    validate(config)
    config['_root']=str(ROOT)
    config['work_dir']=resolve(ROOT,config.get('work_dir',f"work/{config['model']['name']}"))
    if not Path(config['work_dir']).is_relative_to(ROOT) or Path(config['work_dir'])==ROOT:raise ValueError('work_dir must be a subdirectory of the project')
    config['model']['adapter_file']=resolve(ROOT,config['model']['adapter_file'])
    if config['data'].get('script'):config['data']['script']=resolve(ROOT,config['data']['script'])
    if config['verification'].get('script'):config['verification']['script']=resolve(ROOT,config['verification']['script'])
    config['data'].setdefault('directory',str(Path(config['work_dir'])/'inputs'))
    if 'directory' in config['data']:
        config['data']['directory']=resolve(ROOT,config['data']['directory'])
        if not Path(config['data']['directory']).is_relative_to(ROOT) or Path(config['data']['directory'])==ROOT:raise ValueError('data.directory must be a subdirectory of the project')
    if source_paths and 'ref' in config['source']:
        config['source']['ref']=resolve(ROOT,expand_source_path(config['source']['ref'],'source.ref'))
    if source_paths and 'files' in config['source']:config['source']['files']=[resolve(ROOT,expand_source_path(p,'source.files')) for p in config['source']['files']]
    for entries in config['dependencies'].values():
        for entry in entries:
            for key in ('source','build_script'):entry[key]=resolve(ROOT,entry[key])
    return config
