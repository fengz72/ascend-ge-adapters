import copy
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2 import modeling_qwen2 as hf

PATH = Path('models/qwen2.5-0.5b/qwen_npu_adapter.py').resolve()


class QwenPatches(unittest.TestCase):
    def setUp(self):
        # Tests restore HF methods; production uses separate workers/containers.
        for owner, name in [(hf.Qwen2RMSNorm, 'forward'), (hf, 'apply_rotary_pos_emb'),
                            (hf.Qwen2Attention, 'forward'), (hf.Qwen2Model, 'forward'),
                            (hf.Qwen2ForCausalLM, 'forward')]:
            self.addCleanup(setattr, owner, name, getattr(owner, name))
        spec = importlib.util.spec_from_file_location('customer_qwen', PATH)
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        with torch.random.fork_rng():
            torch.manual_seed(4)
            self.raw = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32,
                intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=64)).eval()
        self.config = {'model': {'params': {'score_token_ids_file': 'config/target_tokens.json',
                                   'operators': 'torch', 'attention': 'fia',
                                   'prune_output': True, 'max_seq_len': 64}},
                       'target': {'family': 'a3'}}
        generator_spec = importlib.util.spec_from_file_location('qwen_case_generator', PATH.parent/'generate_cases.py')
        generator = importlib.util.module_from_spec(generator_spec)
        generator_spec.loader.exec_module(generator)
        self.generator = generator
        self.cases = list(generator.generate_cases(64, count=2, batch_size=2,
            dist='fixed', seq_len=10, prefix_min=3, prefix_max=3, seed=9))
        self.cases[1]['payload']['requests'][0]['tokens'] += [7, 8]
        self.cases[1]['payload']['requests'][1]['tokens'] = self.cases[1]['payload']['requests'][1]['tokens'][:3]
        third = copy.deepcopy(self.cases[1]['payload']['requests'][1])
        third['id'] = 'additional-request'
        self.cases[1]['payload']['requests'].append(third)
        self.cases[1]['samples'] = 3
        # Generator defaults must not restrict external/actual case batches.
        self.config['data'] = {'params': {'batch_size': 10}}

    def test_patches_reuse_hf_modules_and_match_original_golden(self):
        reference = self.adapter.prepare_reference(self.raw, self.config)
        expected = {c['id']: reference.decode(reference.run(reference.encode(c)), c) for c in self.cases}
        original_attention = hf.Qwen2Attention.forward
        candidates = []
        for kind, prune in [('fia', True), ('prefix_910', True), ('prefix_a5', True), ('fia', False)]:
            config = copy.deepcopy(self.config)
            config['model']['params'].update(attention=kind, prune_output=prune)
            source = copy.deepcopy(self.raw)
            decoder, mlp, projection = (source.model.layers[0], source.model.layers[0].mlp,
                                        source.model.layers[0].self_attn.q_proj)
            candidate = self.adapter.prepare_candidate(source, config)
            self.assertIs(candidate.model, source)
            self.assertIs(candidate.model.model.layers[0], decoder)
            self.assertIs(candidate.model.model.layers[0].mlp, mlp)
            self.assertIs(candidate.model.model.layers[0].self_attn.q_proj, projection)
            self.assertTrue(candidate.report['global_patch'])
            candidates.append(candidate)
        self.assertIsNot(hf.Qwen2Attention.forward, original_attention)
        for case in reversed(self.cases):
            for candidate in candidates:
                actual = candidate.decode(candidate.run(candidate.encode(case)), case)
                self.assertEqual(actual['alignment'], expected[case['id']]['alignment'])
                np.testing.assert_allclose(actual['outputs']['scores'], expected[case['id']]['outputs']['scores'],
                                           atol=2e-6, rtol=2e-5)

    def test_reference_selected_positions_match_full_hf_logits(self):
        reference = self.adapter.prepare_reference(self.raw, self.config)
        for case in self.cases:
            inputs = reference.encode(case)
            with torch.inference_mode():
                full = self.raw(**inputs, use_cache=False).logits.numpy()
            actual = reference.decode(reference.run(inputs), case)['outputs']['scores']
            for row, request in enumerate(case['payload']['requests']):
                position = len(case['payload']['prefix']) + len(request['tokens']) - 1
                np.testing.assert_allclose(actual[row],full[row,position,reference.ids],rtol=1e-6,atol=1e-7)

    def test_prefix_encoding_and_export_allow_varying_batch_tokens_prefix_and_bounds(self):
        config = copy.deepcopy(self.config)
        config['model']['params'].update(attention='prefix_a5', operators='npu')
        handle = self.adapter.Handle(self.raw, config, True)
        cases = copy.deepcopy(self.cases)
        cases[1]['payload']['prefix'].append(5)
        first, second, repeated = [handle.encode(c) for c in (cases[0], cases[1], cases[0])]
        self.assertNotEqual(first['input_ids'].numel(), second['input_ids'].numel())
        self.assertNotEqual(first['actual_seq_lengths'].tolist(), second['actual_seq_lengths'].tolist())
        self.assertTrue(torch.equal(first['input_ids'], repeated['input_ids']))
        inputs = handle.export_spec()['io']['inputs']
        self.assertEqual(inputs[0]['shape'], [-1])
        self.assertEqual(inputs[2]['shape'], [-1])
        self.assertTrue(all(node['dynamic_dims'] == [0] for node in inputs))
        self.assertEqual(first['actual_seq_lengths'].numel(), 3)
        self.assertEqual(second['actual_seq_lengths'].numel(), 4)
        spec = handle.export_spec()['io']
        self.assertEqual(spec['outputs'][0]['shape'], [-1, 64])
        self.assertEqual(spec['outputs'][0]['dynamic_dims'], [0])
        for case in cases:
            self.assertEqual(handle.output_capacities(case)['scores'], case['samples'] * 64 * 4)
        # The worker must accept both batches with the same declared IOSpec.
        from core.worker import encode
        with tempfile.TemporaryDirectory() as directory:
            _, records = encode(handle, cases, spec, Path(directory))
        self.assertEqual([record['samples'] for record in records], [2, 3])

    def test_prefix_keeps_nonempty_prefix_and_request_count_checks(self):
        config = copy.deepcopy(self.config)
        config['model']['params']['attention'] = 'prefix_a5'
        handle = self.adapter.Handle(self.raw, config, True)
        empty = copy.deepcopy(self.cases[0]); empty['payload']['prefix'] = []
        with self.assertRaisesRegex(ValueError, 'nonempty prefix'):
            handle.encode(empty)
        wrong = copy.deepcopy(self.cases[0]); wrong['samples'] = 1
        with self.assertRaisesRegex(ValueError, 'sample count'):
            handle.encode(wrong)

    def test_saved_npy_tensors_match_prefix_model_inputs(self):
        config = copy.deepcopy(self.config)
        config['model']['params']['attention'] = 'prefix_a5'
        handle = self.adapter.Handle(self.raw, config, True)
        for case in self.cases:
            actual = handle.encode(case)
            with tempfile.TemporaryDirectory() as directory:
                for name, value in self.generator.packed_inputs(case).items():
                    path = Path(directory, f'{name}.npy')
                    np.save(path, value, allow_pickle=False)
                    np.testing.assert_array_equal(np.load(path, allow_pickle=False), actual[name].numpy())

    def test_reference_rejects_process_contaminated_by_candidate_patch(self):
        reference = self.adapter.prepare_reference(self.raw, self.config)
        inputs = reference.encode(self.cases[0])
        self.adapter.adapt(copy.deepcopy(self.raw), self.config)
        with self.assertRaisesRegex(RuntimeError, 'clean process'):
            reference.run(inputs)
        with self.assertRaisesRegex(RuntimeError, 'clean process'):
            self.adapter.prepare_reference(copy.deepcopy(self.raw), self.config)

    def test_method_patch_is_idempotent_and_head_is_not_pruned_twice(self):
        model, _ = self.adapter.adapt(self.raw, self.config)
        forward = hf.Qwen2Attention.forward
        self.adapter.apply_patches()
        self.adapter.apply_patches()
        self.assertIs(hf.Qwen2Attention.forward, forward)
        with self.assertRaisesRegex(ValueError, 'already adapted'):
            self.adapter.adapt(model, self.config)

    def test_runtime_recipe_needs_no_weights_and_matches_candidate_inputs(self):
        for kind in ('fia', 'prefix_a5', 'prefix_910'):
            config = copy.deepcopy(self.config)
            config['model']['params']['attention'] = kind
            model_handle = self.adapter.Handle(self.raw, config, True)
            runtime = self.adapter.prepare_runtime(config, {'outputs':[{'name':'scores','shape':[-1,8],'dtype':'float16'}]})
            for case in self.cases:
                for name, value in model_handle.encode(case).items():
                    np.testing.assert_array_equal(runtime.encode(case)[name], value.numpy())
                self.assertEqual(runtime.output_capacities(case)['scores'], case['samples']*8*2)

    def test_standalone_patch_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'adapter.py')
            shutil.copy(PATH, path)
            Path(directory, 'config').mkdir()
            shutil.copy(PATH.parent/'config/target_tokens.json', Path(directory, 'config/target_tokens.json'))
            spec = importlib.util.spec_from_file_location('delivered_adapter', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model, report = module.adapt(self.raw, self.config)
            self.assertIs(model, self.raw)
            self.assertTrue(report['global_patch'])
            self.assertEqual(report['patches'], list(module.PATCH_TARGETS))
            self.assertNotIn('from core', path.read_text())


if __name__ == '__main__':
    unittest.main()
