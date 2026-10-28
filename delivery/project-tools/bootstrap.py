from pathlib import Path
import subprocess,json,shutil,yaml,hashlib
r=Path('/export/home/ext.hejun23/workspace/bge-opt-100');repo=r/'runner'
for d in ['work/jobs','work/plans','work/prepared','work/tmp','tools/checks']:(r/d).mkdir(parents=True,exist_ok=True)
def run(cmd,cwd=repo):
 print('CPU_COMMAND',json.dumps(cmd),flush=True);subprocess.run(cmd,cwd=cwd,check=True)
run(['git','apply','--check',str(r/'tools/dual-workload-replay.patch')]);run(['git','apply',str(r/'tools/dual-workload-replay.patch')])
p=repo/'models/bge/config.yaml';cfg=yaml.safe_load(p.read_text());cfg['source']['ref']=str(r/'materials/model.onnx');cfg['build']['options'].pop('aicore_num',None);cfg['data']['params']['samples_file']=str(r/'materials/sample_inputs.json');cfg['verification']['params']['expected_outputs']=str(r/'materials/expected_outputs.json');p.write_text(yaml.safe_dump(cfg,sort_keys=False))
(r/'tools/config-registry.json').write_text(json.dumps({'historical_configs':9,'known':['base','fusion-v2','multi8','multi4','multi2','fp16','mixed','fusion-v1','multi16'],'new':{}},indent=2))
run(['python3','-m','unittest','discover','-s','tests','-p','test_bge_replay_cli.py','-v'])
run(['cmake','-S','runtime','-B','runtime/build','-DCMAKE_BUILD_TYPE=Release','-DASCEND_HOME=/usr/local/Ascend/cann-9.0.0'])
run(['cmake','--build','runtime/build','-j8'])
run(['ctest','--test-dir','runtime/build','--output-on-failure'])
run(['python3',str(r/'candidates/build_add_layernorm_v2.py'),'--cann','/usr/local/Ascend/cann-9.0.0','--directory',str(r/'candidates')])
print('CPU_BUILD_DONE',flush=True)
