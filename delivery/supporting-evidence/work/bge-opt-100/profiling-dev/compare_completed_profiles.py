"""Local-only analysis of the two completed, explicitly archived BGE profiles."""
from pathlib import Path
import collections,csv,hashlib,json,math,statistics

BASE=Path(__file__).resolve().parents[1]/'profile-analysis'
OUT=BASE/'comparison';OUT.mkdir(parents=True,exist_ok=True)
REMOTE='/export/home/ext.hejun23/workspace/bge-opt-100'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(name):
 root=BASE/'work'/name;session=next(root.glob('bench-profiling/PROF_*'))
 csvpath=next(session.glob('mindstudio_profiler_output/op_summary*.csv'));ops=rows(csvpath)
 stepfile=next(session.glob('mindstudio_profiler_output/step_trace*.csv'));steps=rows(stepfile)
 events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
 result=json.loads((root/'result.json').read_text());plan=json.loads((BASE/Path(result['plan']).relative_to(REMOTE)).read_text())
 status=json.loads((BASE/'work/jobs'/name/'status.json').read_text());resource=json.loads((root/'resources.jsonl').read_text().splitlines()[0])
 assert status['state']=='finished' and status['exit_code']==0 and status['process_group_stopped']
 assert result['returncode']==0 and result['runtime']['status']=='success'
 assert len(steps)==len(events)==30 and set(x['Model ID'] for x in steps)=={'1'}
 assert all(x['case_id']=='case-0002' and x['instance']==0 for x in events)
 assert [x['phase'] for x in events]==['warmup']*10+['measure']*20
 assert [x['request_id'] for x in events[10:]]==list(range(20))
 assert [int(x['Iteration ID']) for x in steps]==list(range(1,31))
 assert str(resource['runtime_pid']) in session.name, 'PROF session PID does not match native monitor'
 bounds=[(float(s['Iteration End(us)'])-float(s['Iteration Time(us)']),float(s['Iteration End(us)']),int(s['Iteration ID'])) for s in steps]
 perstep=collections.Counter();byname=collections.defaultdict(dict);unmapped=[]
 for line,op in enumerate(ops,2):
  start=float(op['Task Start Time(us)']);end=start+float(op['Task Duration(us)'])
  ids=[i for a,z,i in bounds if a-.5<=start and end<=z+.5]
  if len(ids)!=1:unmapped.append({'line':line,'matches':ids,'op':op['Op Name']});continue
  it=ids[0];perstep[it]+=1
  if op['OP Type'] in ['Add','LayerNorm','AddLayerNorm']:
   assert it not in byname[op['Op Name']],('duplicate target name in iteration',line,op)
   byname[op['Op Name']][it]={'csv_line':line,**op}
 assert not unmapped,unmapped[:3]
 offsets=[float(s['Iteration End(us)'])-e['completed_ns']/1000 for s,e in zip(steps,events)]
 clock_spread=max(offsets)-min(offsets)
 assert clock_spread<100,'iteration/event ordering clock offset is not stable'
 return {'name':name,'root':root,'session':session,'csv_path':csvpath,'step_path':stepfile,'ops':ops,'steps':steps,'events':events,'byname':byname,'plan':plan,'status':status,'result':result,'native_pid':resource['runtime_pid'],'clock_offset_spread_us':clock_spread,'perstep':dict(perstep)}

# Minimal protobuf wire reader only extracts Model.graph/Graph.node/Node names and
# input/output strings, not attributes, weights or inference. Validate the known
# 3137-node/400-initializer artifact and use it solely for Add -> LN connectivity.
def varint(b,i):
 value=shift=0
 while True:
  c=b[i];i+=1;value|=(c&127)<<shift
  if c<128:return value,i
  shift+=7
  assert shift<=63

def fields(b):
 i=0
 while i<len(b):
  key,i=varint(b,i);n,t=key>>3,key&7
  if t==2:size,i=varint(b,i);v=b[i:i+size];i+=size
  elif t==0:v,i=varint(b,i)
  elif t in (1,5):size=8 if t==1 else 4;v=b[i:i+size];i+=size
  else:raise ValueError('unsupported wire type '+str(t))
  assert i<=len(b)
  yield n,t,v

def source_nodes(path):
 graph=next(v for n,t,v in fields(Path(path).read_bytes()) if n==7)
 nodes={};initializers=0
 for n,t,v in fields(graph):
  if n==5:initializers+=1
  if n!=1:continue
  node={'inputs':[],'outputs':[]}
  for a,wire,value in fields(v):
   if a in (1,2):node['inputs' if a==1 else 'outputs'].append(value.decode())
   elif a in (3,4):node['name' if a==3 else 'type']=value.decode()
  assert node['name'] not in nodes
  nodes[node['name']]=node
 assert len(nodes)==3137 and initializers==400
 return nodes

def stat(items):
 duration=[float(x['Task Duration(us)']) for x in items]
 return {'count':len(items),'task_total_us':sum(duration),'task_mean_us':statistics.mean(duration),'task_min_us':min(duration),'task_max_us':max(duration),'block_num_counts':dict(collections.Counter(x['Block Num'] for x in items)),'task_types':dict(collections.Counter(x['Task Type'] for x in items)),'input_dtypes':sorted(set(x['Input Data Types'] for x in items)),'input_shapes':sorted(set(x['Input Shapes'] for x in items)),'aiv_time_mean_us':statistics.mean(float(x['aiv_time(us)']) for x in items),'wait_mean_us':statistics.mean(float(x['Task Wait Time(us)']) for x in items)}

base=load('base-fixed-profile');fused=load('fusion-fixed-profile')
identity=json.loads((BASE/'tools/identity.json').read_text())
graph=BASE/'materials/exported/model.onnx';assert sha(graph)==identity['graph']['sha256'];nodes=source_nodes(graph)
assert base['plan']['inputs']['cases']==fused['plan']['inputs']['cases']
# Independent completed-run/input checks were printed by the managed controller.
# Bind those exact records to the copied result and plan hashes.
controller_log=Path(__file__).resolve().parents[1]/'stage2.log'
controller_reports=[]
for line in controller_log.read_text().splitlines():
 try:entry=json.loads(line)
 except ValueError:continue
 if isinstance(entry,dict) and entry.get('schema')=='bge-run-artifact-identity/2':controller_reports.append(entry)
for r in [base,fused]:
 entry=next(q['runs'][r['name']] for q in controller_reports if r['name'] in q['runs'])
 assert entry['passed'] and entry['result_sha256']==sha(r['root']/'result.json')
 assert entry['plan_sha256']==sha(BASE/Path(r['result']['plan']).relative_to(REMOTE))
 assert entry['binary_sha256']==identity['binary']['sha256'] and entry['packed_graph_sha256']==identity['graph']['sha256']
 assert entry['input_binary_sha256']=='27ce13c73cb35762ee7d0396d4425829d7e25a3b8ae2a12b37ed7f96b8690aa5'
 r['controller_identity']=entry
for r in [base,fused]:
 assert r['plan']['device']==identity['logical_device'] and r['plan']['instances']==1
 assert r['status']['devices']==identity['physical_devices']
 assert r['plan']['options']==base['plan']['options']=={'graph_run_mode':0,'precision_mode':'must_keep_origin_dtype'}
 assert r['plan']['observability']=={'profiling':True,'dump':False}
 assert r['result']['runtime']['performance']['profiling_enabled'] is True
 assert all(x['Device_id']==str(identity['logical_device']) and x['Model ID']=='1' for x in r['ops'])

names=sorted(n for n in fused['byname'] if n.endswith('/BgeAddLayerNorm'))
assert len(names)==48
pair_rows=[];all_a=[];all_l=[];all_f=[]
selected=list(range(11,31));csv_out=[]
for new in names:
 old_ln=new.removesuffix('/BgeAddLayerNorm');prefix=old_ln.removesuffix('/LayerNorm/Add_1/LayerNorm');assert prefix!=old_ln
 old_add=prefix+'/Add';reduce=prefix+'/LayerNorm/ReduceMean';final=prefix+'/LayerNorm/Add_1'
 assert nodes[old_add]['type']=='Add' and nodes[reduce]['type']=='ReduceMean' and nodes[final]['type']=='Add'
 assert nodes[reduce]['inputs'][0]==nodes[old_add]['outputs'][0]
 assert set(base['byname'][old_ln])==set(base['byname'][old_add])==set(fused['byname'][new])==set(range(1,31))
 adds=[base['byname'][old_add][i] for i in selected];lns=[base['byname'][old_ln][i] for i in selected];fs=[fused['byname'][new][i] for i in selected]
 a=stat(adds);l=stat(lns);f=stat(fs);before=a['task_mean_us']+l['task_mean_us'];after=f['task_mean_us']
 row={'residual_add':old_add,'layernorm':old_ln,'fused':new,'source_add_to_layernorm_reduce_edge':[nodes[old_add]['outputs'][0],nodes[reduce]['inputs'][0]],'selected_iteration_ids':selected,'add':a,'layernorm_before':l,'fused_after':f,'before_pair_mean_us':before,'after_mean_us':after,'change_pct':(after/before-1)*100}
 pair_rows.append(row);all_a+=adds;all_l+=lns;all_f+=fs
 for i in selected:
  ar=base['byname'][old_add][i];lr=base['byname'][old_ln][i];fr=fused['byname'][new][i]
  csv_out.append({'iteration':i,'phase':'measure','request_id':i-11,'residual_add':old_add,'layernorm':old_ln,'fused':new,'add_duration_us':ar['Task Duration(us)'],'ln_duration_us':lr['Task Duration(us)'],'fused_duration_us':fr['Task Duration(us)'],'add_block_num':ar['Block Num'],'ln_block_num':lr['Block Num'],'fused_block_num':fr['Block Num'],'add_csv_line':ar['csv_line'],'ln_csv_line':lr['csv_line'],'fused_csv_line':fr['csv_line']})

aggregate={'pairs':48,'verified_measure_iterations':20,'base_add':stat(all_a),'base_layernorm':stat(all_l),'fused_add_layernorm':stat(all_f)}
aggregate['before_pair_task_total_us']=aggregate['base_add']['task_total_us']+aggregate['base_layernorm']['task_total_us']
aggregate['after_task_total_us']=aggregate['fused_add_layernorm']['task_total_us']
aggregate['target_task_sum_per_verified_measure_before_us']=aggregate['before_pair_task_total_us']/20
aggregate['target_task_sum_per_verified_measure_after_us']=aggregate['after_task_total_us']/20
aggregate['change_pct']=(aggregate['after_task_total_us']/aggregate['before_pair_task_total_us']-1)*100
aggregate['pairs_faster']=sum(x['change_pct']<0 for x in pair_rows)
aggregate['pairs_slower']=sum(x['change_pct']>0 for x in pair_rows)
aggregate['pair_change_pct_range']=[min(x['change_pct'] for x in pair_rows),max(x['change_pct'] for x in pair_rows)]
aggregate['by_encoder_location']={}
for location in ['attention_output','ffn_output']:
 group=[r for r in pair_rows if ('/attention/output/' in r['layernorm'])==(location=='attention_output')]
 before=sum(r['before_pair_mean_us'] for r in group)/len(group);after=sum(r['after_mean_us'] for r in group)/len(group)
 aggregate['by_encoder_location'][location]={'pairs':len(group),'add_mean_us':statistics.mean(r['add']['task_mean_us'] for r in group),'ln_before_mean_us':statistics.mean(r['layernorm_before']['task_mean_us'] for r in group),'pair_before_mean_us':before,'fused_after_mean_us':after,'change_pct':(after/before-1)*100,'pairs_faster':sum(r['change_pct']<0 for r in group),'pairs_slower':sum(r['change_pct']>0 for r in group)}

raws={}
for r in [base,fused]:
 raws[r['name']]={'session':str(r['session'].relative_to(BASE)),'csv_sha256':sha(r['csv_path']),'step_sha256':sha(r['step_path']),'native_pid':r['native_pid'],'controller_identity':r['controller_identity'],'logical_device':r['plan']['device'],'physical_locks':r['status']['devices'],'model_ids':['1'],'infer_id_present':False,'csv_rows':len(r['ops']),'step_iterations':30,'warmup_events':10,'measure_events':20,'per_iteration_rows':r['perstep'],'clock_offset_spread_us':r['clock_offset_spread_us'],'unmapped_kernel_rows':0,'target_nodes_once_per_iteration':True,'selected_measured_iterations':selected,'phase_mapping_basis':'single instance; 30 sequential events = 30 sequential step intervals; no unmapped kernels; 48 target names exactly once per interval; profiler-end minus event-complete offset stable within reported spread; first 10 warmup, next 20 measured'}
report={'kind':'bge100-matched-kernel-profile-diagnostic','passed':True,'device_tasks_started_by_analysis':False,'source_graph_sha256':sha(graph),'controller_log_sha256':sha(controller_log),'runs':raws,'aggregate':aggregate,'pairs':pair_rows,'limitations':['One short profiled A/B pair; kernel diagnostics are not independent formal QPS/p99 acceptance.','Task durations can overlap and their sums are not wall-clock latency; Task Wait Time is not summed as kernel work.','No output-port memory-write volume conclusion follows from four fused output descriptors.','No proof that core count is optimal or that fusion helps all shapes/concurrency configurations.','Profiler timestamps use their own clock; ordinal phase mapping is supported by interval/task counts and near-constant completion offset, not equality of absolute clocks.','No matched final GE dump was captured; residual-Add edge uses the unchanged packed ONNX and names from actual profiling plus the loaded pass/replacement log.']}
(OUT/'add-layernorm-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'matched-kernel-rows.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(csv_out[0]));writer.writeheader();writer.writerows(csv_out)
print(json.dumps({'runs':raws,'aggregate':aggregate},indent=2))
