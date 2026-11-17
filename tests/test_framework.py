import contextlib
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from core.config import ROOT,load_authoring
from core.data import select_cases
from core.verify import numeric,compare

SOC='Ascend950PR_958b'
HW={'device':6,'soc':SOC,'capacity':{'aic':32,'aiv':64}}

class Framework(unittest.TestCase):
    @contextlib.contextmanager
    def pipeline_config(self):
        work=ROOT/'work';work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as directory:
            output=Path(directory,'run')
            path=Path(directory,'model.yaml')
            path.write_text((ROOT/'models/tiny/config.yaml').read_text()+f'\nwork_dir: {output}\n')
            yield path,output

    def test_python_entry_reloads_yaml_and_validates_overrides(self):
        from core.pipeline import execute
        with self.pipeline_config() as (path,directory), patch('core.pipeline.perform',return_value={'status':'success'}) as perform:
            execute('prepare',path)
            self.assertEqual(perform.call_args.args[1]['benchmark']['instances'],2)
            path.write_text(path.read_text().replace('instances: 2','instances: 3'))
            execute('prepare',path,profiling=False)
            self.assertEqual(perform.call_args.args[1]['benchmark']['instances'],3)
            for options in ({'instances':0},{'case_ids':[]},{'case_ids':['case-0','case-0']}):
                with self.assertRaises(ValueError):execute('prepare',path,**options)
            self.assertEqual(perform.call_count,2)
            self.assertEqual(len(list(directory.rglob('result.json'))),2)

    def test_bench_does_not_run_prerequisites_and_keeps_full_pool(self):
        from core.pipeline import execute
        cases=[{'id':f'case-{i}','samples':2,'payload':{}} for i in range(100)]
        with self.pipeline_config() as (path,directory):
            with patch('core.targets.inspect_device_isolated',return_value=HW), \
                 patch('core.pipeline.worker',return_value={'performance':{'requests':1000}}) as worker, \
                 patch('core.data.load',return_value=cases), \
                 patch('core.data.generate',side_effect=AssertionError('generated data')), \
                 patch('core.dependencies.prepare',side_effect=AssertionError('built dependencies')):
                result=execute('bench',path,device=6,case_ids=['case-0'])
            worker.assert_called_once()
            self.assertEqual(worker.call_args.args[0],'bench')
            self.assertEqual(worker.call_args.args[2],cases)
            self.assertFalse(list(directory.rglob('*.yaml')))
            self.assertFalse(list(directory.rglob('build.json')))
            self.assertNotIn('baseline',json.loads(Path(result['result']).read_text()))

    def test_run_only_sequences_the_seven_independent_steps(self):
        from core.pipeline import execute,STEPS
        with self.pipeline_config() as (path,directory), \
             patch('core.targets.inspect_device_isolated',return_value=HW), \
             patch('core.pipeline.perform',return_value={'status':'success'}) as perform:
            execute('run',path,device=6,case_ids=['all'],instances=3)
            self.assertEqual([c.args[0] for c in perform.call_args_list],list(STEPS))
            config=perform.call_args.args[1]
            self.assertTrue(all(call.args[1] is config for call in perform.call_args_list))
            self.assertEqual(config['verification']['case_ids'],'all')
            self.assertEqual(config['benchmark']['instances'],3)
            self.assertEqual(STEPS,('data','baseline','prepare','build','infer','compare','bench'))

    def test_original_and_ge_use_same_precision_subset_bench_keeps_full_pool(self):
        from core.pipeline import perform
        config=load_authoring(ROOT/'models/tiny/config.yaml')
        cases=[{'id':f'case-{i}','samples':2,'payload':{}} for i in range(100)]
        with tempfile.TemporaryDirectory() as directory, patch('core.data.load',return_value=cases), patch('core.pipeline.worker',return_value={}) as worker:
            for selection,expected in ((['case-0'],cases[:1]),(['case-7','case-99'],[cases[7],cases[99]]),('all',cases)):
                config['verification']['case_ids']=selection
                for stage in ('baseline','infer'):
                    perform(stage,config,Path(directory),6)
                    self.assertEqual(worker.call_args.args[2],expected)
                perform('bench',config,Path(directory),6)
                self.assertEqual(worker.call_args.args[2],cases)
            with patch('core.data.load',side_effect=AssertionError('compare read inputs')):
                perform('compare',config,Path(directory),None)
                self.assertEqual(worker.call_args.args[2],[])

    def test_run_stops_after_failed_comparison(self):
        from core.pipeline import execute
        with self.pipeline_config() as (path,directory),patch('core.targets.inspect_device_isolated',return_value=HW):
            with patch('core.pipeline.perform',side_effect=lambda stage,*args:{'status':'failed' if stage=='compare' else 'success'}) as perform:
                with self.assertRaisesRegex(ValueError,'accuracy failed'):execute('run',path,device=6)
            self.assertNotIn('bench',[call.args[0] for call in perform.call_args_list])
            result=json.loads(next(directory.rglob('result.json')).read_text())
            self.assertEqual(result['steps']['compare']['status'],'failed')

    def test_accuracy_selection_preserves_pool(self):
        cases=[{'id':f'case-{i}','samples':2,'payload':{}} for i in range(3)];old=copy.deepcopy(cases)
        self.assertEqual(select_cases(cases,[cases[-1]['id']]),[cases[-1]])
        self.assertEqual(select_cases(cases,'all'),cases);self.assertEqual(cases,old)
        for ids in ([],['missing'],['all'],[cases[0]['id']]*2):
            with self.assertRaises(ValueError):select_cases(cases,ids)

    def test_catalog_duplicate_chip_and_bad_capacity_are_rejected(self):
        from core.targets import read_catalog
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,'targets.yaml')
            path.write_text('a3:\n  SameChip: {}\na5:\n  SameChip: {}\n')
            with self.assertRaisesRegex(ValueError,'duplicate chip'):read_catalog(path)
            path.write_text('a5:\n  PrivateChip:\n    expected_capacity:\n      aic: 0\n')
            with self.assertRaisesRegex(ValueError,'omit unknown'):read_catalog(path)

    def test_parent_hardware_probe_is_isolated(self):
        from core.targets import inspect_device_isolated
        completed=subprocess.CompletedProcess([],0,json.dumps(HW),'')
        with patch('core.targets.inspect_device',side_effect=AssertionError('parent touched NPU')),patch('core.targets.subprocess.run',return_value=completed):
            self.assertEqual(inspect_device_isolated(6),HW)

    def test_no_empty_custom_opp_environment_and_multiple_paths_are_preserved(self):
        from core.dependencies import environment
        config=load_authoring(ROOT/'models/tiny/config.yaml')
        with patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':''}):
            self.assertNotIn('ASCEND_CUSTOM_OPP_PATH',environment(config))
        with patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':'/one:/two'}):
            self.assertEqual(environment(config)['ASCEND_CUSTOM_OPP_PATH'],'/one:/two')

    def test_custom_opp_symlink_preserves_vendor_directory_identity(self):
        from core.dependencies import environment
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);vendor=root/'vendors/custom_prefix_attn'
            vendor.mkdir(parents=True)
            install=root/'dependencies/prefix/install';install.mkdir(parents=True)
            (install/'opp').symlink_to(vendor,target_is_directory=True)
            config={'work_dir':str(root),'dependencies':{'custom_ops':[{'id':'prefix'}]}}
            with patch.dict(os.environ,{'ASCEND_CUSTOM_OPP_PATH':'/existing/vendor'}):
                paths=environment(config)['ASCEND_CUSTOM_OPP_PATH'].split(os.pathsep)
            self.assertEqual(paths,[str(vendor),'/existing/vendor'])

    def test_packed_input_is_one_binary_and_named_shapes_are_checked(self):
        from core.worker import encode
        from types import SimpleNamespace
        handle=SimpleNamespace(encode=lambda c:{'a':np.asarray(c['payload'],dtype=np.int32)},output_capacities=lambda c:{'scores':8})
        io_spec={'inputs':[{'name':'a','dtype':'int32','shape':[-1]}],'outputs':[{'name':'scores'}]}
        cases=[{'id':'one','samples':1,'payload':[1,2]},{'id':'two','samples':1,'payload':[3,4,5]}]
        with tempfile.TemporaryDirectory() as directory:
            _,rows=encode(handle,cases,io_spec,directory)
            self.assertEqual([p.name for p in Path(directory).iterdir()],['inputs.bin'])
            self.assertEqual([r['tensors'][0]['offset'] for r in rows],[0,8])
            np.testing.assert_array_equal(np.fromfile(Path(directory,'inputs.bin'),np.int32),[1,2,3,4,5])
            io_spec['inputs'][0]['shape']=[2]
            with self.assertRaisesRegex(ValueError,'contract'):encode(handle,cases,io_spec,directory)

    def test_numeric_zero_nonfinite_and_single_bad_sample(self):
        self.assertTrue(numeric(np.zeros((2,4)),np.zeros((2,4)),{})['passed'])
        a=np.ones((10000,4));b=a.copy();b[0]=0
        self.assertFalse(numeric(a,b,{})['passed'])
        self.assertFalse(numeric(np.ones(2),np.array([np.nan,1]),{})['passed'])

    def test_exact_names_and_alignment(self):
        value={'outputs':{'x':np.array([2**60],np.int64)},'alignment':{'rows':['a']}}
        self.assertTrue(compare(value,value,{'x':{'policy':'exact'}})['passed'])
        wrong=copy.deepcopy(value);wrong['outputs']['x'][0]+=1
        self.assertFalse(compare(value,wrong,{'x':{'policy':'exact'}})['passed'])
        wrong=copy.deepcopy(value);wrong['alignment']['rows']=['b']
        with self.assertRaises(ValueError):compare(value,wrong,{'x':{}})

    def test_io_marker_parser_skips_large_constants(self):
        from core.exporters.torchair import parse_bindings
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,'graph.pbtxt')
            path.write_text('node {\n name: "constant"\n op: "Const"\n attr { s: "'+('x}'*100000)+'" }\n}\n'+'''node {
 name: "arg4"
 op: "Data"
 attr { key: "index" value { s: 'i: 0\\n' } }
 attr { key: "_ge_adapter_name" value { s: 's: "features"\\n' } }
}
''')
            self.assertEqual(parse_bindings(path),[(0,'features','arg4')])

class IndependentRuntime(unittest.TestCase):
    def test_bench_worker_does_not_load_source_or_export(self):
        from core.worker import execute
        recipe=SimpleNamespace(encode=lambda c:{'a':np.array([1],dtype=np.int64)},output_capacities=lambda c:{'scores':4})
        adapter=SimpleNamespace(prepare_runtime=lambda config,io:recipe)
        io_spec={'inputs':[{'name':'a','dtype':'int64','shape':[-1]}],'outputs':[{'name':'scores'}]}
        with tempfile.TemporaryDirectory() as directory:
            config={'model':{'adapter_file':'recipe.py'}}
            job={'stage':'bench','config':config,'cases':[{'id':'case-0','samples':1,'payload':{}}],'output':directory,'device':6}
            with patch('core.worker.load_source',side_effect=AssertionError('loaded source')), \
                 patch('core.worker.load_adapter',return_value=adapter),patch('core.dependencies.activate',side_effect=AssertionError('activated Python operators')), \
                 patch('core.backends.runtime.graph_files',return_value={'path':'model.air','format':'air'}), \
                 patch('core.backends.runtime.read_io',return_value=io_spec), \
                 patch('core.backends.runtime.run',return_value={'status':'success','performance':{'requests':1000}}):
                result=execute(job)
            self.assertEqual(result['performance']['requests'],1000)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_missing_graph_fails_without_building_or_loading_model(self):
        from core.worker import execute
        with tempfile.TemporaryDirectory() as directory:
            job={'stage':'bench','config':{},'cases':[],'output':directory,'device':6}
            with patch('core.worker.load_source',side_effect=AssertionError('loaded source')), \
                 patch('core.dependencies.prepare',side_effect=AssertionError('installed dependencies')), \
                 patch('core.backends.runtime.graph_files',side_effect=FileNotFoundError('model.air; run build first')):
                with self.assertRaisesRegex(FileNotFoundError,'run build first'):execute(job)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_output_named_alignment_does_not_collide_with_business_metadata(self):
        from core.worker import save_rows,compare_outputs
        value={'outputs':{'alignment':np.ones((1,2),np.float32)},'alignment':{'rows':['r0']}}
        with tempfile.TemporaryDirectory() as directory:
            config={'work_dir':directory,'verification':{'case_ids':'all','outputs':{'alignment':{}}}}
            for name in ('golden.npz','outputs.npz'):save_rows(Path(directory,name),{'case-0':value})
            self.assertEqual(compare_outputs(config)['status'],'success')

    def test_comparison_needs_only_golden_outputs_and_policy(self):
        from core.worker import save_rows,compare_outputs
        good={'outputs':{'scores':np.ones((1,2),np.float32)},'alignment':{'rows':['r0']}}
        bad={'outputs':{'scores':np.zeros((1,2),np.float32)},'alignment':{'rows':['r0']}}
        with tempfile.TemporaryDirectory() as directory:
            config={'work_dir':directory,'verification':{'case_ids':'all','outputs':{'scores':{}}}}
            save_rows(Path(directory,'golden.npz'),{'case-0':good,'case-1':good})
            save_rows(Path(directory,'outputs.npz'),{'case-0':bad,'case-1':bad})
            result=compare_outputs(config)
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['accuracy']['failed_cases'],['case-0','case-1'])
            save_rows(Path(directory,'outputs.npz'),{'case-0':good})
            with self.assertRaisesRegex(ValueError,'case IDs differ'):compare_outputs(config)
            Path(directory,'outputs.npz').unlink()
            with self.assertRaises(FileNotFoundError):compare_outputs(config)

if __name__=='__main__':unittest.main()
