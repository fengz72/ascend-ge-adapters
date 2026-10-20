"""Format measured results; no artifact or optimization policy."""
import csv
from pathlib import Path
from core.contracts import read_json

def report(paths):
    lines=['| model | status | instances | AIC/AIV | accuracy | profiling | calls/s | samples/s | p50 ms | p99 ms | errors |',
           '|---|---|---:|---|---|---|---:|---:|---:|---:|---:|']
    for path in paths:
        r=read_json(path);c=r['config'];steps=r.get('steps',{});p=r.get('performance',steps.get('bench',{}).get('performance',{}));lat=p.get('e2e_ms',{})
        instances=len(p['instances']) if p.get('instances') else (1 if r['stage']=='infer' else '—')
        values=[c['model']['name'],r['status'],instances,c['build']['options'].get('aicore_num','default'),
                r.get('accuracy',steps.get('compare',{}).get('accuracy',{})).get('status','not_run'),p.get('profiling_enabled','—'),p.get('qps','—'),p.get('samples_per_second','—'),lat.get('p50','—'),lat.get('p99','—'),p.get('errors','—')]
        lines.append('| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v).replace("|",r"\|") for v in values)+' |')
    lines+=['','profiling_enabled=true marks timings collected with profiling overhead.','Latency includes H2D, descriptors, execution and synchronization; compile/load/warmup are excluded.']
    return '\n'.join(lines)


BENCHMARK_COLUMNS=(
    'run_id','started_at','status','phase','repeat','model','backend','soc','device','dtype',
    'aic_cores','aiv_cores','kernel_aic_cores','batch_size','instances','seed',
    'requested_calls','completed_calls','pool_size','pool_sha256',
    'call_qps','business_qps','p50_ms','p99_ms','max_ms','errors','profiling_enabled',
    'latency_scope','measurement_ms','warmup_ms','hbm_base_mb','hbm_after_setup_mb',
    'duration_seconds','kernel_library_sha256','result_path','error',
)


def benchmark_row(path):
    """One measurement row with explicit units; preserve failures and raw results."""
    path=Path(path);r=read_json(path);c=r['config'];x=r.get('experiment',{})
    p=r.get('performance',r.get('steps',{}).get('bench',{}).get('performance',{}))
    lat=p.get('e2e_ms',{});samples=p.get('samples',{});hbm=p.get('hbm_mb',{})
    budget=str(c['build']['options'].get('aicore_num','')).split('|')
    batch=samples.get('min') if samples.get('min')==samples.get('max') else None
    values={
        'run_id':r.get('run_id',path.parent.name),'started_at':r.get('started_at',''),
        'status':r['status'],'phase':x.get('phase',''),'repeat':x.get('repeat',''),
        'model':c['model']['name'],'backend':c['build']['backend'],'soc':c['target']['soc'],
        'device':r.get('device'),'dtype':c['build']['dtype'],
        'aic_cores':budget[0] if len(budget)==2 else '',
        'aiv_cores':budget[1] if len(budget)==2 else '',
        'kernel_aic_cores':x.get('kernel_aic_cores',''),
        'batch_size':int(batch) if batch is not None else x.get('batch_size',''),
        'instances':len(p['instances']) if p.get('instances') else c['benchmark']['instances'],
        'seed':c['benchmark']['seed'],'requested_calls':c['benchmark']['iterations'],
        'completed_calls':p.get('requests'),'pool_size':p.get('sampling',{}).get('pool_size'),
        'pool_sha256':x.get('pool_sha256',''),'call_qps':p.get('qps'),
        'business_qps':p.get('samples_per_second'),'p50_ms':lat.get('p50'),
        'p99_ms':lat.get('p99'),'max_ms':lat.get('max'),'errors':p.get('errors'),
        'profiling_enabled':p.get('profiling_enabled'),'latency_scope':p.get('latency_scope'),
        'measurement_ms':p.get('wall_ms'),'warmup_ms':p.get('warmup',{}).get('ms'),
        'hbm_base_mb':hbm.get('base'),'hbm_after_setup_mb':hbm.get('after_setup'),
        'duration_seconds':r.get('duration_seconds'),
        'kernel_library_sha256':x.get('kernel_library_sha256',''),
        'result_path':str(path.resolve()),'error':r.get('error',''),
    }
    return {name:values.get(name) for name in BENCHMARK_COLUMNS}


def export_benchmarks(paths,output):
    """Regenerate the trend table from existing result.json files."""
    with Path(output).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=BENCHMARK_COLUMNS)
        writer.writeheader()
        for path in paths:writer.writerow(benchmark_row(path))
