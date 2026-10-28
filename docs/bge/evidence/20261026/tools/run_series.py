import argparse,pathlib,json,subprocess,sys,datetime
ROOT=pathlib.Path('/export/home/ext.hejun23/workspace/bge/work/bge-opt-20261026')
p=argparse.ArgumentParser();p.add_argument('names',nargs='+');p.add_argument('--env',action='append',default=[]);a=p.parse_args()
for name in a.names:
 cmd=[sys.executable,str(ROOT/'tools/run_checked.py'),str(ROOT/'plans'/(name+'.json')),'--job',name,'--timeout','650']
 for e in a.env:cmd.extend(['--env',e])
 print('BEGIN_ROUND',name,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
 q=subprocess.run(cmd)
 if q.returncode:raise SystemExit(q.returncode)
 d=json.loads((ROOT/name/'result.json').read_text());perf=d.get('runtime',{}).get('performance',{})
 (ROOT/name/'performance.json').write_text(json.dumps(perf,indent=2))
 print('ROUND_RESULT',name,json.dumps(perf),flush=True)
 if d['returncode'] or d.get('runtime',{}).get('status')!='success' or perf.get('errors')!=0:raise SystemExit('round failed')
 if perf.get('capture_enabled') or perf.get('profiling_enabled') or perf.get('latency_scope')!='h2d_desc_execute_sync_d2h':raise SystemExit('invalid formal scope')
 if not perf.get('statistics_sufficient') or perf.get('wall_ms',0)<60000 or perf.get('window_successes',0)<10000:raise SystemExit('formal sample insufficient')
 print('END_ROUND',name,flush=True)
