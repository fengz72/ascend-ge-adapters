"""Regressions for comparison completeness, opaque tensor names and failed measurements."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from core.config import ROOT
from core.worker import save_rows, compare_outputs


class OutputContracts(unittest.TestCase):
    def test_onnx_export_preserves_slice_nodes_opsets_and_runtime_parameters(self):
        import onnx
        import onnxruntime as ort
        from onnx import helper,TensorProto
        from core import native_onnx
        from core.exporters.onnx import export
        for opset in (11,12,13,14):
            with self.subTest(opset=opset),tempfile.TemporaryDirectory() as directory:
                directory=Path(directory);source=directory/'source.onnx'
                graph=helper.make_graph([
                    helper.make_node('Slice',['x','starts','ends','axes','steps'],['y'])
                ],'slice', [helper.make_tensor_value_info('x',TensorProto.FLOAT,['batch','length'])]+
                   [helper.make_tensor_value_info(n,TensorProto.INT64,[1]) for n in ('ends','axes','steps')],
                [helper.make_tensor_value_info('y',TensorProto.FLOAT,['batch','out'])],initializer=[
                    helper.make_tensor('starts',TensorProto.INT64,[1],[0])])
                model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',opset)]);model.ir_version=7
                onnx.checker.check_model(model);onnx.save(model,source)
                handle=native_onnx.prepare_candidate(str(source),{'build':{'dtype':'float32'}})
                artifact,_=export(handle.export_spec(),{},directory)
                exported=onnx.load(artifact['path'])
                self.assertEqual(list(model.opset_import),list(exported.opset_import))
                for field in ('node','initializer','input','output'):
                    self.assertEqual([x.SerializeToString() for x in getattr(model.graph,field)],
                                     [x.SerializeToString() for x in getattr(exported.graph,field)])
                original=ort.InferenceSession(str(source),providers=['CPUExecutionProvider'])
                candidate=ort.InferenceSession(artifact['path'],providers=['CPUExecutionProvider'])
                for axis,step,end in ((1,1,2),(1,2,4),(0,1,1),(1,-1,-5)):
                    values={'x':np.arange(8,dtype=np.float32).reshape(2,4),
                            'ends':np.array([end],dtype=np.int64),
                            'axes':np.array([axis],dtype=np.int64),
                            'steps':np.array([step],dtype=np.int64)}
                    np.testing.assert_array_equal(original.run(None,values)[0],candidate.run(None,values)[0])

    def test_all_archived_outputs_need_policies_and_matching_names(self):
        with tempfile.TemporaryDirectory() as directory:
            config={'work_dir':directory,'verification':{'case_ids':'all','outputs':{}}}
            good={'outputs':{'scores':np.array([1]),'encoder/metadata/alignment':np.array([0])},'alignment':{'rows':[0]}}
            wrong=copy.deepcopy(good);wrong['outputs']['encoder/metadata/alignment'][0]=999
            save_rows(Path(directory,'golden.npz'),{'case-0':good})
            save_rows(Path(directory,'outputs.npz'),{'case-0':wrong})
            with self.assertRaisesRegex(ValueError,'every business output'):compare_outputs(config)
            for names in (['scores'],['scores','extra'],['scores','encoder/metadata/alignment','extra']):
                config['verification']['outputs']={name:{'policy':'exact'} for name in names}
                with self.assertRaisesRegex(ValueError,'unconfigured output'):compare_outputs(config)
            config['verification']['outputs']={name:{'policy':'exact'} for name in good['outputs']}
            result=compare_outputs(config)
            self.assertEqual(result['accuracy']['case_ids'],['case-0'])
            self.assertEqual(result['status'],'failed')
            save_rows(Path(directory,'outputs.npz'),{'case-0':good})
            self.assertEqual(compare_outputs(config)['status'],'success')
            wrong['outputs'].pop('encoder/metadata/alignment')
            save_rows(Path(directory,'outputs.npz'),{'case-0':wrong})
            with self.assertRaisesRegex(ValueError,'unconfigured output'):compare_outputs(config)

    def test_onnx_names_roundtrip_npy_export_and_archive_without_alias_manifest(self):
        import onnx
        from onnx import helper,TensorProto
        from core import data,native_onnx
        from core.exporters.onnx import export
        names=['input:0','encoder/input','literal%2F','输入','../escape']
        outputs=['encoder/output:0','result/metadata/alignment','literal%out','输出','../result']
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as directory:
            directory=Path(directory);path=directory/'source.onnx'
            graph=helper.make_graph([helper.make_node('Identity',[a],[b]) for a,b in zip(names,outputs)],'names',
                [helper.make_tensor_value_info(n,TensorProto.FLOAT,[1,2]) for n in names],
                [helper.make_tensor_value_info(n,TensorProto.FLOAT,[1,2]) for n in outputs])
            model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)]);model.ir_version=8
            onnx.checker.check_model(model);onnx.save(model,path)
            config={'model':{'adapter_file':str(ROOT/'core/native_onnx.py'),'params':{}},
                    'data':{'directory':str(directory/'inputs'),'params':{'samples':1,'count':2,'inputs':{n:{'dtype':'float32','shape':[1,2]} for n in names}}},
                    'build':{'dtype':'float32'},'reference':{'dtype':'float32'}}
            cases=data.generate(config)
            self.assertEqual(cases,data.load(config))
            paths=list((directory/'inputs').rglob('*.npy'))
            self.assertEqual(len(paths),10)
            self.assertTrue(all(p.parent.name in ('case-0','case-1') for p in paths))
            self.assertTrue(any(p.name=='input%3A0.npy' for p in paths))
            handle=native_onnx.prepare_reference(str(path),config)
            rows={c['id']:handle.decode(handle.run(handle.encode(c)),c) for c in cases}
            for name in ('golden.npz','outputs.npz'):save_rows(directory/name,rows)
            config.update(work_dir=str(directory),verification={'case_ids':'all','outputs':{n:{'policy':'exact'} for n in outputs}})
            self.assertEqual(compare_outputs(config)['status'],'success')
            graph,io=export(handle.export_spec(),handle.encode(cases[0]),directory)
            self.assertEqual([n['node'] for n in io['inputs']],names)
            self.assertEqual([n.name for n in onnx.load(graph['path']).graph.output],outputs)
            self.assertFalse(list((directory/'inputs').rglob('*.json')))


class FailedMeasurements(unittest.TestCase):
    def payload(self):
        return {'status':'failed','error':'execute failed','performance':{'requests':3,'errors':1,'qps':2.0}}

    def test_native_nonzero_preserves_structured_partial_measurements_and_log(self):
        from core.backends.runtime import call
        payload=self.payload()
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory,'runtime');binary.touch();log=Path(directory,'bench.log')
            process=subprocess.CompletedProcess([],1,'ACL execute returned 500002\nGE_RESULT '+json.dumps(payload)+'\n')
            def run(*args,**kwargs):
                kwargs['stdout'].write(process.stdout)
                return process
            with patch('core.backends.runtime.RUNTIME',binary),patch('core.backends.runtime.subprocess.run',side_effect=run):
                result=call({},log)
            self.assertEqual(result['performance'],payload['performance'])
            self.assertIn('500002',result['error']);self.assertIn(str(log),result['error'])
            self.assertEqual(log.read_text(),process.stdout)
            process.stdout='load graph failed\n'
            with patch('core.backends.runtime.RUNTIME',binary),patch('core.backends.runtime.subprocess.run',side_effect=run):
                with self.assertRaisesRegex(RuntimeError,'load graph failed'):call({},log)

    def test_crashed_runtime_does_not_wait_for_inherited_stdout_pipe(self):
        import os
        import signal
        import sys
        import time
        from core.backends.runtime import call
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory,'runtime');child_file=Path(directory,'child.pid')
            binary.write_text(
                '#!'+sys.executable+'\nimport os,time\n'
                'child=os.fork()\n'
                'if child == 0:\n    time.sleep(8)\n    os._exit(0)\n'
                f'open({str(child_file)!r},"w").write(str(child))\n'
                'print("compiler crashed",flush=True)\nos._exit(17)\n'
            )
            binary.chmod(0o700)
            began=time.monotonic()
            try:
                with patch('core.backends.runtime.RUNTIME',binary):
                    with self.assertRaisesRegex(RuntimeError,'exit 17.*compiler crashed'):
                        call({})
                self.assertLess(time.monotonic()-began,4)
            finally:
                if child_file.exists():
                    try:os.kill(int(child_file.read_text()),signal.SIGTERM)
                    except ProcessLookupError:pass

    def test_worker_preserves_partial_result_and_cleans_only_packed_buffers(self):
        from core.worker import infer_or_bench
        recipe=SimpleNamespace(encode=lambda c:{'a':np.array([1],dtype=np.int64)},output_capacities=lambda c:{'scores':4})
        adapter=SimpleNamespace(prepare_runtime=lambda config,io:recipe)
        spec={'inputs':[{'name':'a','dtype':'int64','shape':[-1]}],'outputs':[{'name':'scores'}]}
        with tempfile.TemporaryDirectory() as directory:
            csv=Path(directory,'requests.csv');csv.write_text('raw data')
            with patch('core.worker.load_adapter',return_value=adapter), \
                 patch('core.backends.runtime.graph_files',return_value={}), \
                 patch('core.backends.runtime.read_io',return_value=spec), \
                 patch('core.backends.runtime.run',return_value=self.payload()):
                result=infer_or_bench('bench',{'model':{'adapter_file':'adapter.py'}},[{'id':'case-0','samples':1,'payload':{}}],directory,0)
            self.assertEqual(result['status'],'failed');self.assertEqual(result['performance']['requests'],3)
            self.assertEqual(list(Path(directory).iterdir()),[csv])

    def test_execute_keeps_partial_bench_in_independent_and_run_results(self):
        from core.pipeline import execute,STEPS
        hw={'device':0,'soc':'Ascend950PR_958b','capacity':{'aic':32,'aiv':64}}
        for stage in ('bench','run'):
            with tempfile.TemporaryDirectory(dir=ROOT/'work') as directory:
                path=Path(directory,'model.yaml');output=Path(directory,'runs')
                path.write_text((ROOT/'models/tiny/config.yaml').read_text()+f'\nwork_dir: {output}\n')
                with patch('core.targets.inspect_device_isolated',return_value=hw), \
                     patch('core.pipeline.perform',side_effect=lambda step,*args:self.payload() if step=='bench' else {'status':'success'}) as perform:
                    with self.assertRaisesRegex(ValueError,'execute failed'):execute(stage,path,device=0)
                record=json.loads(next(output.rglob('result.json')).read_text())
                self.assertEqual(record['status'],'failed')
                partial=record if stage=='bench' else record['steps']['bench']
                self.assertEqual(partial['performance']['requests'],3)
                self.assertEqual(partial['performance']['errors'],1)
                self.assertEqual([c.args[0] for c in perform.call_args_list],['bench'] if stage=='bench' else list(STEPS))

if __name__=='__main__':unittest.main()
