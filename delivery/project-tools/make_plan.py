from pathlib import Path
import argparse,subprocess,json,sys
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--instances',type=int);fixed=p.add_mutually_exclusive_group();fixed.add_argument('--fixed',action='store_true');fixed.add_argument('--case-id',choices=[f'case-{i:04d}' for i in range(5)]);p.add_argument('--capture',action='store_true');g=p.add_mutually_exclusive_group();g.add_argument('--fusion',action='store_true');g.add_argument('--attention-v3',action='store_true');p.add_argument('--cores');p.add_argument('--precision-mode');p.add_argument('--seconds',type=float,default=30);p.add_argument('--max-seconds',type=float,default=180);p.add_argument('--requests',type=int,default=2000);p.add_argument('--seed',type=int,default=42);p.add_argument('--profile',action='store_true');p.add_argument('--monitor',action='store_true');a=p.parse_args()
cmd=[sys.executable,str(r/'runner/models/bge/replay.py'),'--prepare-only','--device',str(json.loads((r/'tools/identity.json').read_text())['logical_device']),'--config',str(r/'runner/models/bge/config.yaml'),'--graph',str(r/'materials/exported/model.onnx'),'--output-dir',str(r/'work/prepared'/a.name),'--min-seconds',str(a.seconds),'--max-seconds',str(a.max_seconds),'--min-requests',str(a.requests),'--seed',str(a.seed),'--run-id',a.name]
if a.instances is not None:cmd+=['--instances',str(a.instances)]
if a.fixed or a.case_id:cmd+=['--case-id',a.case_id or 'case-0002']
if a.capture:cmd+=['--capture']
if a.fusion:cmd+=['--preload',str(r/'candidates/libbge_add_layernorm_pass_v2.so')]
if a.attention_v3:cmd+=['--preload',str(r/'candidates-v3/libbge_add_layernorm_attention_v3.so')]
if a.cores:cmd+=['--aicore-num',a.cores]
if a.precision_mode is not None:cmd+=['--precision-mode',a.precision_mode]
subprocess.run(cmd,check=True,cwd=r/'runner')
plan=json.loads((r/'work/prepared'/a.name/'plan.json').read_text());plan['output_dir']=str(r/'work'/a.name)
plan['monitor_resources']=True
if a.profile:plan['observability']['profiling']=True
out=r/'work/plans'/(a.name+'.json');out.write_text(json.dumps(plan,indent=2));print(out)
