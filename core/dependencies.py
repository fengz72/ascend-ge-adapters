"""Build selected dependencies and activate their standard install directories."""
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

def entries(config):
    return [entry for group in config['dependencies'].values() for entry in group]

def directory(config, entry):
    return Path(config['work_dir'])/'dependencies'/entry['id']

def prepare(config):
    outputs=[]
    for entry in entries(config):
        output=directory(config,entry);output.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,GE_SRC_DIR=entry['source'],GE_BUILD_DIR=str(output/'build'),
                 GE_INSTALL_DIR=str(output/'install'),GE_PYTHON=sys.executable,
                 TORCH_EXTENSIONS_DIR=str(Path(config['work_dir'])/'torch_extensions'))
        try:
            with (output/'build.log').open('w') as log:
                subprocess.run(['bash',entry['build_script']],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        except Exception as error:
            shutil.rmtree(output/'install',ignore_errors=True)
            log=output/'build.log'
            detail=log.read_text(errors='replace')[-12000:] if log.is_file() else ''
            raise RuntimeError(f"dependency {entry['id']}: {error}\n{detail}\nlog: {log}") from error
        outputs.append(str(output/'install'))
    return outputs

def environment(config):
    env=os.environ.copy();paths={k:[] for k in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH','PYTHONPATH')}
    for entry in entries(config):
        install=directory(config,entry)/'install'
        if not install.is_dir():raise FileNotFoundError(f"dependency {entry['id']} is not built; run prepare")
        for folder,key in (('opp','ASCEND_CUSTOM_OPP_PATH'),('lib','LD_LIBRARY_PATH'),('python','PYTHONPATH')):
            if (install/folder).exists():paths[key].append(str(install/folder))
        paths['LD_LIBRARY_PATH'].extend(str(p.parent) for p in install.rglob('*.so'))
    for key,values in paths.items():
        values.extend(p for p in env.get(key,'').split(os.pathsep) if p)
        if values:env[key]=os.pathsep.join(dict.fromkeys(values))
        else:env.pop(key,None)
    env['TORCH_EXTENSIONS_DIR']=str(Path(config['work_dir'])/'torch_extensions')
    return env

def activate(config):
    os.environ.update(environment(config))
    sys.path[:0]=[p for p in os.environ.get('PYTHONPATH','').split(os.pathsep) if p and p not in sys.path]
    for entry in entries(config):
        for name in entry.get('python_imports',[]):importlib.import_module(name)
