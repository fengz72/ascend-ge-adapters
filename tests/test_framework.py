import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np
import yaml
from core.config import ROOT,load_authoring,validate
from core.contracts import digest
from core.data import generate,select_cases
from core.verify import numeric,compare

QWEN=ROOT/'models/qwen2.5-0.5b/config/model.yaml'
SOC='Ascend950PR_958b'
HW={'device':6,'soc':SOC,'capacity':{'aic':32,'aiv':64}}

class Framework(unittest.TestCase):
    def setUp(self):self.enterContext(patch.dict(os.environ,{'QWEN_MODEL_DIR':'/weights/Qwen'}))
    def qwen(self,**kwargs):return load_authoring(QWEN,target=SOC,**kwargs)

    def test_qwen_platform_defaults(self):
        for soc,attention,cores,instances in (
            ('Ascend910_9382','prefix_910','8|16',5),
            ('Ascend950PR_958b','prefix_a5','16|32',3),
        ):
            with self.subTest(soc=soc):
                config=load_authoring(QWEN,target=soc,source_paths=False)
                self.assertEqual(config['model']['params']['attention'],attention)
                self.assertEqual(config['build']['options']['aicore_num'],cores)
                self.assertEqual(config['data']['params']['batch_size'],16)
                self.assertEqual(config['benchmark']['instances'],instances)

    def test_yaml_changes_and_environment_are_read_again_without_output_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,'model.yaml')
            authoring=yaml.safe_load(QWEN.read_text())
            authoring['platforms']['a5']['benchmark']={'instances':1}
            path.write_text(yaml.safe_dump(authoring,sort_keys=False))
            first=load_authoring(path,target=SOC)
            path.write_text(path.read_text().replace('instances: 1','instances: 8'))
            os.environ['QWEN_MODEL_DIR']='/new/weights'
            second=load_authoring(path,target=SOC)
            self.assertEqual(first['benchmark']['instances'],1)
            self.assertEqual(second['benchmark']['instances'],8)
            self.assertEqual(first['source']['ref'],'/weights/Qwen')
            self.assertEqual(second['source']['ref'],'/new/weights')
            self.assertEqual([f.name for f in Path(directory).iterdir()],['model.yaml'])

    @contextlib.contextmanager
    def pipeline_config(self):
        work=ROOT/'work';work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as directory:
            output=Path(directory,'run')
            path=Path(directory,'model.yaml')
            path.write_text((ROOT/'models/tiny/config.yaml').read_text()+f'\nwork_dir: {output}\n')
            yield path,output

    def test_main_only_passes_arguments_to_execute(self):
        from core.pipeline import main
        with patch('core.pipeline.execute',return_value={'status':'success'}) as execute, \
             patch('core.pipeline.load_authoring',side_effect=AssertionError('main loaded YAML')), \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            main(['bench','--config',str(QWEN),'--device','6','--instances','8','--profiling'])
        execute.assert_called_once_with('bench',config=str(QWEN),device=6,target=None,targets=None,instances=8,profiling=True)
        self.assertEqual(json.loads(stdout.getvalue())['status'],'success')

    def test_python_bench_reads_yaml_and_overrides_without_weight_environment(self):
        from core.pipeline import execute
        with self.pipeline_config() as (path,directory), patch.dict(os.environ):
            path.write_text(QWEN.read_text()+f'\nwork_dir: {directory}\n')
            original=path.read_text();os.environ.pop('QWEN_MODEL_DIR',None)
            with patch('core.targets.inspect_device_isolated',return_value=HW), \
                 patch('core.pipeline.perform',return_value={'status':'success'}) as perform:
                result=execute('bench',path,device=6,instances=8,profiling=True)
            config=perform.call_args.args[1]
            self.assertEqual(config['benchmark']['instances'],8)
            self.assertTrue(config['observability']['profiling'])
            self.assertEqual(config['source']['ref'],'${QWEN_MODEL_DIR}')
            self.assertEqual(config['build']['options']['aicore_num'],
                             yaml.safe_load(QWEN.read_text())['platforms']['a5']['build']['options']['aicore_num'])
            self.assertEqual(path.read_text(),original)
            self.assertEqual(json.loads(Path(result['result']).read_text())['config'],config)
            self.assertEqual(list(path.parent.rglob('*.yaml')),[path])

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

    def test_removed_module_file_flags_and_commands(self):
        from core.pipeline import main
        self.assertIsNone(importlib.util.find_spec('core.cli'))
        for extra in (['--cases','cases.json'],['--build','build.json'],['--lock','lock.json']):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):main(['bench','--config',str(QWEN),'--device','6',*extra])
        for command in ('resolve','check','verify','profile'):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):main([command])

    def test_device_is_required_for_both_entries(self):
        from core.pipeline import execute,main
        for stage in ('build','infer','bench','run'):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):main([stage,'--config',str(QWEN)])
            with self.assertRaisesRegex(ValueError,'explicit nonnegative NPU index'):execute(stage,QWEN)

    def test_python_entry_accepts_one_yaml_path_and_has_no_dict_compatibility(self):
        from core.pipeline import execute
        for value in ({'model':{}},[QWEN]):
            with self.assertRaises(TypeError):execute('prepare',value)
        with self.assertRaisesRegex(ValueError,'unknown step'):execute('check',QWEN)

    def test_main_displays_original_error(self):
        from core.pipeline import main
        with patch('core.pipeline.execute',side_effect=FileNotFoundError('model.air; run build first')), \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                main(['bench','--config',str(QWEN),'--device','6'])
        self.assertEqual(error.exception.code,1)
        self.assertIn('bench: FileNotFoundError: model.air; run build first',stderr.getvalue())

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

    def test_generator_without_directory_does_not_import_adapter_or_create_files(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'config.json').write_text('{"vocab_size":64}')
            config=self.qwen();config['source']['ref']=directory;config['model']['adapter_file']='/must/not/import.py'
            config['data']['params']['batch_size']=10
            config['data'].pop('directory')
            cases=generate(config)
            self.assertEqual(len(cases),100)
            self.assertEqual([p.name for p in Path(directory).iterdir()],['config.json'])
            self.assertEqual(digest(cases),'66b937d4b7dee78a2a98605f55f70aee34c59a699dd6502787e30f8e0c717b95')

    def test_case_directory_roundtrip_reuses_100_npy_folders_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'config.json').write_text('{"vocab_size":64}')
            inputs=Path(directory,'inputs')
            config=self.qwen();config['source']['ref']=directory;config['data']['directory']=str(inputs)
            config['data']['params']['batch_size']=10
            config['model']['adapter_file']='/must/not/import.py'
            first=generate(config)
            paths=sorted(inputs.iterdir())
            self.assertEqual([p.name for p in paths],[f'case-{i:04d}' for i in range(100)])
            for folder in paths:
                self.assertEqual(sorted(p.name for p in folder.iterdir()),
                                 ['actual_seq_lengths.npy','input_ids.npy','position_ids.npy'])
                for path in folder.iterdir():
                    value=np.load(path,allow_pickle=False)
                    self.assertEqual(value.dtype,np.int64)
                    self.assertEqual(value.ndim,1)
            spec=importlib.util.spec_from_file_location('generator',QWEN.parent.parent/'generate_cases.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.assertEqual(first,[module.load_case(p) for p in paths])
            self.assertEqual(digest(first),'66b937d4b7dee78a2a98605f55f70aee34c59a699dd6502787e30f8e0c717b95')
            with patch('numpy.save',side_effect=AssertionError('unchanged data was rewritten')):
                self.assertEqual(first,generate(config))
            self.assertFalse(list(inputs.rglob('*.json')))

    def test_case_directory_repairs_partial_pool_and_follows_changed_yaml_params(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'config.json').write_text('{"vocab_size":64}')
            inputs=Path(directory,'inputs')
            config=self.qwen();config['source']['ref']=directory;config['data']['directory']=str(inputs)
            original=generate(config)
            (inputs/'case-0000/actual_seq_lengths.npy').unlink()
            (inputs/'case-0001/position_ids.npy').write_bytes(b'interrupted')
            path=inputs/'case-0002/input_ids.npy'
            np.save(path,np.load(path).astype(np.float64))
            self.assertEqual(generate(config),original)
            config['data']['params'].update(count=2,seed=1)
            changed=generate(config)
            self.assertEqual(len(changed),2)
            self.assertNotEqual(changed,original[:2])
            self.assertEqual([p.name for p in sorted(inputs.iterdir())],['case-0000','case-0001'])
            saved={p:p.read_bytes() for p in inputs.rglob('*.npy')}
            config['data']['params']['count']=0
            with self.assertRaises(ValueError):generate(config)
            self.assertEqual(saved,{p:p.read_bytes() for p in inputs.rglob('*.npy')})

    def test_qwen_bench_rejects_stale_batch_in_any_case_before_worker(self):
        from core.data import load
        from core.pipeline import perform
        spec=importlib.util.spec_from_file_location('generator',QWEN.parent.parent/'generate_cases.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=load_authoring(QWEN,target='Ascend910_9382')
        self.assertEqual(config['data']['params']['batch_size'],16)
        self.assertEqual(config['benchmark']['instances'],5)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'config.json').write_text('{"vocab_size":64}')
            config['source']['ref']=directory
            config['data']['directory']=str(Path(directory,'inputs'))
            config['data']['params']['count']=2
            expected=generate(config)
            self.assertEqual(load(config),expected)
            stale=list(module.generate_cases(64,count=2,batch_size=14))[-1]
            folder=Path(config['data']['directory'],stale['id'])
            for name,value in module.packed_inputs(stale).items():np.save(folder/f'{name}.npy',value)
            saved={p:p.read_bytes() for p in Path(config['data']['directory']).rglob('*.npy')}
            config['source']['ref']='/must/not/load/weights'
            with patch('core.pipeline.worker') as worker:
                with self.assertRaisesRegex(ValueError,'case-0001: actual batch 14 != configured batch_size 16; run data first'):
                    perform('bench',config,Path(directory),1)
                worker.assert_not_called()
            self.assertEqual(saved,{p:p.read_bytes() for p in saved})
            config['source']['ref']=directory
            self.assertEqual(generate(config),expected)
            self.assertEqual(load(config),expected)

    def test_npy_case_loader_rejects_invalid_bounds_and_positions(self):
        spec=importlib.util.spec_from_file_location('generator',QWEN.parent.parent/'generate_cases.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        case=next(module.generate_cases(64,count=1))
        values=module.packed_inputs(case)
        with tempfile.TemporaryDirectory() as directory:
            for name,value in values.items():np.save(Path(directory,f'{name}.npy'),value)
            bounds=values['actual_seq_lengths'].copy();bounds[-1]+=1
            np.save(Path(directory,'actual_seq_lengths.npy'),bounds)
            with self.assertRaisesRegex(ValueError,'bounds'):module.load_case(directory)
            np.save(Path(directory,'actual_seq_lengths.npy'),values['actual_seq_lengths'])
            positions=values['position_ids'].copy();positions[0]+=1
            np.save(Path(directory,'position_ids.npy'),positions)
            with self.assertRaisesRegex(ValueError,'position_ids'):module.load_case(directory)

    def test_data_directory_resolves_inside_project_without_creating_it(self):
        authored=yaml.safe_load(QWEN.read_text())['data']['directory']
        self.assertEqual(self.qwen()['data']['directory'],str(ROOT/authored))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,'overlay.yaml')
            for value in ('/tmp/inputs','../inputs','.'):
                path.write_text(f'data:\n  directory: {value}\n')
                with self.assertRaisesRegex(ValueError,'subdirectory of the project'):
                    load_authoring([QWEN,path],target=SOC)

    def test_historical_length_distribution_and_token_sequence(self):
        spec=importlib.util.spec_from_file_location('generator',QWEN.parent.parent/'generate_cases.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cases=list(module.generate_cases(64));stats=module.summarize(cases)
        self.assertEqual(digest(cases),'66b937d4b7dee78a2a98605f55f70aee34c59a699dd6502787e30f8e0c717b95')
        self.assertEqual(stats['business_samples'],1000)
        self.assertAlmostEqual(stats['total_seq_mean'],149.345)
        self.assertAlmostEqual(stats['total_seq_p99'],212.02)
        self.assertEqual(stats['total_seq_max'],218)
        self.assertEqual(digest(cases),digest(list(module.generate_cases(64))))
        self.assertNotEqual(digest(cases),digest(list(module.generate_cases(64,seed=1))))

    def test_batch_changes_preserve_historical_prefixes_and_base_requests(self):
        spec=importlib.util.spec_from_file_location('generator',QWEN.parent.parent/'generate_cases.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        base=list(module.generate_cases(64,count=4,batch_size=10))
        larger=list(module.generate_cases(64,count=4,batch_size=32))
        largest=list(module.generate_cases(64,count=4,batch_size=64))
        smaller=list(module.generate_cases(64,count=4,batch_size=4))
        for index,(a,b,c,d) in enumerate(zip(base,larger,largest,smaller)):
            self.assertEqual(a['payload']['prefix'],b['payload']['prefix'])
            self.assertEqual(a['payload']['prefix'],d['payload']['prefix'])
            self.assertEqual(a['payload']['requests'],b['payload']['requests'][:10])
            self.assertEqual(b['payload']['requests'],c['payload']['requests'][:32])
            self.assertEqual([r['tokens'] for r in d['payload']['requests']],
                             [a['payload']['requests'][(index+j)%10]['tokens'] for j in range(4)])
            self.assertEqual([r['id'] for r in d['payload']['requests']],['r0','r1','r2','r3'])

    def test_accuracy_selection_preserves_pool(self):
        cases=[{'id':f'case-{i}','samples':2,'payload':{}} for i in range(3)];old=copy.deepcopy(cases)
        self.assertEqual(select_cases(cases,[cases[-1]['id']]),[cases[-1]])
        self.assertEqual(select_cases(cases,'all'),cases);self.assertEqual(cases,old)
        for ids in ([],['missing'],['all'],[cases[0]['id']]*2):
            with self.assertRaises(ValueError):select_cases(cases,ids)

    def test_platforms_budgets_and_no_device_in_configuration(self):
        a5=self.qwen();a3=load_authoring(QWEN,target='Ascend910_9382')
        self.assertEqual(a5['model']['params']['attention'],'prefix_a5')
        self.assertEqual(a5['build']['options']['aicore_num'],
                         yaml.safe_load(QWEN.read_text())['platforms']['a5']['build']['options']['aicore_num'])
        self.assertEqual(a3['model']['params']['attention'],'prefix_910')
        self.assertEqual(a3['build']['options']['aicore_num'],'8|16')
        self.assertNotIn('device',a5['target']);self.assertNotIn('artifacts',a5)

    def test_explicit_and_detected_chip_select_same_config(self):
        for soc,device,capacity in [('Ascend910_9382',1,{'aic':24,'aiv':48}),
                                    (SOC,6,HW['capacity'])]:
            hardware={'device':device,'soc':soc,'capacity':capacity}
            with self.subTest(soc=soc),patch('core.targets.inspect_device_isolated',return_value=hardware):
                self.assertEqual(load_authoring(QWEN,target=soc),load_authoring(QWEN,device=device))

    def test_chip_override_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,'old.yaml')
            path.write_text(QWEN.read_text()+'\nchips: {}\n')
            with self.assertRaisesRegex(ValueError,'chips'):load_authoring(path,target=SOC)

    def test_platform_and_device_mismatch_fails(self):
        with patch('core.targets.inspect_device_isolated',return_value=HW):
            with self.assertRaises(ValueError):load_authoring(QWEN,target='a3',device=6)

    def test_catalog_duplicate_chip_and_bad_capacity_are_rejected(self):
        from core.targets import read_catalog
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,'targets.yaml')
            path.write_text('a3:\n  SameChip: {}\na5:\n  SameChip: {}\n')
            with self.assertRaisesRegex(ValueError,'duplicate chip'):read_catalog(path)
            path.write_text('a5:\n  PrivateChip:\n    expected_capacity:\n      aic: 0\n')
            with self.assertRaisesRegex(ValueError,'omit unknown'):read_catalog(path)

    def test_inactive_branch_typos_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory,'bad.yaml');path.write_text('platforms:\n  a3:\n    benchmark:\n      instnaces: 2\n')
            with self.assertRaisesRegex(ValueError,'instnaces'):load_authoring([QWEN,path],target=SOC)

    def test_invalid_configuration_and_removed_metadata_options(self):
        config=self.qwen()
        for key,value in [('execution',{}),('artifacts',{}),('schema','ge-run/1')]:
            bad=copy.deepcopy(config);bad[key]=value
            with self.assertRaises(ValueError):validate(bad)
        for count in (0,-1,True,1.5):
            bad=copy.deepcopy(config);bad['benchmark']['instances']=count
            with self.assertRaises(ValueError):validate(bad)
        with patch.dict(os.environ):
            os.environ.pop('QWEN_MODEL_DIR',None)
            with self.assertRaisesRegex(ValueError,'QWEN_MODEL_DIR'):self.qwen()

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
