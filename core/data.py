"""Model-owned case generation and reading; no dataset manifests."""
import importlib.util
import re
from pathlib import Path
from urllib.parse import quote, unquote
from core.contracts import safe_name, tensor_name, digest


def validate_case_ids(case_ids):
    if case_ids=='all':return
    if not isinstance(case_ids,list) or not case_ids or len(set(case_ids))!=len(case_ids):raise ValueError('case_ids must be a nonempty unique list or all')
    for name in case_ids:
        safe_name(name)
        if name=='all':raise ValueError('all must be used alone')


def select_cases(cases, case_ids='all'):
    validate_case_ids(case_ids)
    if case_ids=='all':return list(cases)
    by_id={c['id']:c for c in cases}
    if set(case_ids)-set(by_id):raise ValueError(f'unknown case IDs: {sorted(set(case_ids)-set(by_id))}')
    return [by_id[name] for name in case_ids]


def folders(directory):
    directory=Path(directory)
    def key(path):return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)',path.name)]
    paths=sorted((p for p in directory.iterdir() if p.is_dir()),key=key)
    if not paths:raise ValueError(f'no cases in {directory}; run data first')
    return paths


def provider(config):
    if config['data'].get('script'):
        spec=importlib.util.spec_from_file_location('_ge_case_generator',config['data']['script'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    from core.model_api import load_adapter
    return load_adapter(config['model']['adapter_file'])


def validate_cases(cases):
    cases=list(cases);ids=[]
    for case in cases:
        if set(case)!={'id','payload','samples'} or type(case['samples']) is not int or case['samples']<1:raise ValueError('invalid case')
        safe_name(case['id']);ids.append(case['id'])
    if not ids or len(ids)!=len(set(ids)):raise ValueError('empty pool or duplicate case IDs')
    digest(cases)
    return cases


def generate(config):
    import numpy as np
    module=provider(config)
    if config['data'].get('script'):return validate_cases(module.generate(config))
    directory=Path(config['data']['directory']);directory.mkdir(parents=True,exist_ok=True)
    cases=validate_cases(module.generate_cases(config['data']['params']))
    for case in cases:
        folder=directory/case['id'];folder.mkdir(exist_ok=True)
        arrays=module.input_arrays(case,config)
        filenames={quote(tensor_name(name),safe='')+'.npy':value for name,value in arrays.items()}
        for name,value in filenames.items():
            np.save(folder/name,value,allow_pickle=False)
        for path in folder.glob('*.npy'):
            if path.name not in filenames:path.unlink()
    expected={case['id'] for case in cases}
    for folder in folders(directory):
        if folder.name not in expected:
            for path in folder.glob('*.npy'):path.unlink()
            folder.rmdir()
    return load(config)


def load(config):
    import numpy as np
    module=provider(config)
    if config['data'].get('script'):
        return validate_cases(module.load(config))
    cases=[]
    for folder in folders(config['data']['directory']):
        arrays={}
        for path in folder.glob('*.npy'):
            name=tensor_name(unquote(path.stem,errors='strict'))
            if name in arrays:raise ValueError(f'duplicate tensor file: {name!r}')
            arrays[name]=np.load(path,allow_pickle=False)
        cases.append(module.case_from_inputs(folder.name,arrays,config))
    return validate_cases(cases)
