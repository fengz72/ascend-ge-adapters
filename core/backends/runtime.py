"""Native execution and self-contained graph IO; no build or data generation."""
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from core.config import ROOT
from core.contracts import validate_io
from core.targets import require_device

RUNTIME=ROOT/'runtime/build/ge_runtime'
RESULT_PREFIX='GE_RESULT '


def _tail(text,limit=4000):
    return text[-limit:] if len(text)>limit else text


def _kill_group(process):
    """Kill exactly the child's own process group (CANN compiler workers can
    inherit stdout and outlive a killed runtime); then reap the child."""
    try:
        group=os.getpgid(process.pid)
    except OSError:
        group=None
    if group is not None:
        try:
            os.killpg(group,signal.SIGKILL)
        except OSError:
            pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def execute(runtime_bin,plan,*,log_path=None,timeout=None,should_abort=None,
            poll_interval=0.2,error=RuntimeError,aborted_error=RuntimeError):
    """One ge_runtime subprocess under ``plan``; returns (result, log_text, wall_seconds).

    The child runs in its own process group (start_new_session) so a timeout or an
    abort kills exactly that group -- never any other PID. The run is bounded by
    ``timeout`` seconds (None = unbounded) and is polled through ``should_abort()``
    every ``poll_interval`` seconds. Stdout+stderr go to ``log_path`` (a regular
    file, parents created, kept as evidence) or to a temporary file. Exactly one
    GE_RESULT line must come back; a nonzero exit whose parsed status is 'failed'
    returns the result with an 'error' detail attached, every other deviation
    raises ``error`` with the log tail.
    """
    runtime_bin=Path(runtime_bin)
    if not runtime_bin.is_file():
        raise error(f"runtime binary missing: {runtime_bin}; build it: bash runtime/build.sh")
    # A regular file (not an inherited pipe) lets us report the parent's failure
    # without waiting for a surviving compiler worker's stdout to close.
    if log_path is None:
        log_file=tempfile.TemporaryFile(mode='w+',encoding='utf-8',errors='replace')
        log_name=None
    else:
        log_path=Path(log_path);log_path.parent.mkdir(parents=True,exist_ok=True)
        log_file=log_path.open('w+',encoding='utf-8',errors='replace')
        log_name=str(log_path)
    started=time.perf_counter()
    payload=json.dumps(plan)
    with log_file as log:
        try:
            process=subprocess.Popen([str(runtime_bin)],stdin=subprocess.PIPE,stdout=log,
                                      stderr=subprocess.STDOUT,text=True,start_new_session=True)
        except OSError as failure:
            raise error(f"cannot execute runtime {runtime_bin}: {failure}") from failure
        try:
            process.stdin.write(payload)
            process.stdin.close()
        except (BrokenPipeError,OSError):
            pass
        while process.poll() is None:
            elapsed=time.perf_counter()-started
            if should_abort is not None and should_abort():
                _kill_group(process)
                raise aborted_error(f"ge_runtime aborted before completion (plan mode {plan.get('mode')})")
            if timeout is not None and elapsed>=timeout:
                log.seek(0)
                raise error(f"ge_runtime did not finish within {timeout}s (plan mode {plan.get('mode')}); "
                            f"log tail:\n{_tail(log.read())}")
            time.sleep(poll_interval if timeout is None else min(poll_interval,max(0.0,timeout-elapsed)))
        log.seek(0)
        text=log.read()
    wall=time.perf_counter()-started
    messages=[line[len(RESULT_PREFIX):] for line in text.splitlines() if line.startswith(RESULT_PREFIX)]
    if len(messages)!=1:
        detail=(f"ge_runtime (exit {process.returncode}) produced {len(messages)} GE_RESULT lines "
                f"instead of exactly one; log tail:\n{_tail(text)}")
        raise error(detail+(f"\nlog: {log_name}" if log_name else ''))
    try:
        result=json.loads(messages[0])
    except ValueError as failure:
        raise error(f"ge_runtime GE_RESULT is not JSON: {messages[0][:500]}") from failure
    if process.returncode and result.get('status')!='failed':
        detail=f"ge_runtime exited {process.returncode}; log tail:\n{_tail(text)}"
        raise error(detail+(f"\nlog: {log_name}" if log_name else ''))
    if process.returncode:
        detail=f"ge_runtime exited {process.returncode}; log tail:\n{_tail(text)}"
        result['error']=detail+(f"\nlog: {log_name}" if log_name else '')
    return result,text,wall


def call(plan,path=None):
    if not RUNTIME.is_file():raise FileNotFoundError(f'{RUNTIME}; build runtime first: bash runtime/build.sh')
    result,_,_=execute(RUNTIME,plan,log_path=path)
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
