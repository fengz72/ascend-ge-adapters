"""CPU tests for the unified-route production adapter (unified_adapter.py).

No NPU, no real 0.8B weights.  The handle/encode/decode/export contracts run
on a fake folder config (the REAL builder + the SHARED official rope provider);
the independent reference handle runs on a tiny multimodal
Qwen3_5ForConditionalGeneration against the official recipe (and is
cross-checked bit-exact against the validated P2b text reference of the base
adapter); the candidate composition proves the mixed-dtype placement contract
(vision FP32 / language BF16 / head26 FP32, every parameter and buffer on one
device after the explicit device move) on the real production composition
path, plus the dynamic export trace example validated by the core exporter.
"""
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "startlux-decision-v1.1-0.8b-instant"
ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(MODEL_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load("startlux_npu_adapter_for_unified_adapter_test", "startlux_npu_adapter.py")
multimodal = load("startlux_multimodal_for_adapter_test", "multimodal.py")
unified_preprocess = load("startlux_unified_preprocess_adapter_test",
                          "unified_preprocess.py")
adapter = load("startlux_unified_adapter_test", "unified_adapter.py")


def fake_folder_config(image_token=100, vision_start=102, vision_end=103, vocab=128):
    return types.SimpleNamespace(
        image_token_id=image_token, vision_start_token_id=vision_start,
        vision_end_token_id=vision_end,
        text_config=types.SimpleNamespace(vocab_size=vocab),
        vision_config=types.SimpleNamespace(spatial_merge_size=2))


class StubPreprocessor(unified_preprocess.OfficialPreprocessor):
    """A real OfficialPreprocessor whose folder config is a fake namespace."""

    def __init__(self, config=None):
        self.model_dir = MODEL_DIR
        self._config = config or fake_folder_config()
        self._tokenizer = None
        self._processor = None
        self._rope_index = None
        self.min_width = unified_preprocess.DEFAULT_MIN_PHYSICAL_WIDTH


def decision_config(**params):
    return {"model": {"params": {"max_seq_len": 2048, "compiled_seq_len": 256,
                                 "chunk_size": 64, "max_batch": 4,
                                 "fixtures_file": "fixtures/p2b-readouts.json", **params}},
            "source": {"ref": "unused-folder"}}


def tiny_multimodal_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    text = dict(hidden_size=64, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                intermediate_size=128, num_hidden_layers=4, linear_num_key_heads=2,
                linear_num_value_heads=2, linear_key_head_dim=16,
                linear_value_head_dim=16, linear_conv_kernel_dim=4, vocab_size=128,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                 "partial_rotary_factor": 0.5,
                                 "mrope_section": [3, 3, 2]})
    vision = dict(depth=1, hidden_size=64, num_heads=4, patch_size=16,
                  spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
                  num_position_embeddings=256, out_hidden_size=64,
                  intermediate_size=128)
    cfg = Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=True,
                        image_token_id=100, video_token_id=101,
                        vision_start_token_id=102, vision_end_token_id=103)
    cfg.text_config.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    return cfg


def tiny_multimodal_folder():
    """A saved tiny multimodal model folder that looks like a StartLux folder."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    base_dir = Path(os.environ.get("TMPDIR") or (ROOT / "work" / "tmp"))
    base_dir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="unified-tiny-model-", dir=str(base_dir)))
    torch.manual_seed(11)
    model = Qwen3_5ForConditionalGeneration(tiny_multimodal_config()).eval()
    model.to(torch.bfloat16)          # the pinned checkpoint stores BF16: the restore
    for layer in model.model.language_model.layers:          # checks are exact this way;
        if layer.layer_type == "linear_attention":            # A_log is stored F32 there,
            layer.linear_attn.A_log.data = layer.linear_attn.A_log.data.float()
    model.save_pretrained(folder, safe_serialization=True)
    (folder / "decision_config.json").write_text(json.dumps({
        "letter_token_ids": list(range(26, 52)),
        "temperature_by_type": {"choice": 1.4364, "noul": 1.0411, "score": 1.4021}}))
    return folder


def tiny_source(folder, dtype):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    return Qwen3_5ForConditionalGeneration.from_pretrained(
        str(folder), dtype=dtype, local_files_only=True).eval()


def image_row(seed):
    """One official [1, 8, 8]-grid image row of the tiny folder (16 merged tokens)."""
    cfg = tiny_multimodal_config()
    generator = torch.Generator().manual_seed(seed)
    ids = [1, cfg.vision_start_token_id] + [cfg.image_token_id] * 16 \
        + [cfg.vision_end_token_id, 2]
    return {"input_ids": torch.tensor([ids], dtype=torch.int64),
            "pixel_values": torch.randn(64, 1536, generator=generator),
            "image_grid_thw": torch.tensor([[1, 8, 8]], dtype=torch.int64),
            "mm_token_type_ids": torch.tensor([[0, 0] + [1] * 16 + [0, 0]],
                                              dtype=torch.int64)}


def text_row(seed, length=20):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 100, (1, length), generator=generator).long()


class AdapterTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        holder = tempfile.TemporaryDirectory(
            prefix="unified-decision-", dir=os.environ.get("TMPDIR")
            or str(ROOT / "work" / "tmp"))
        cls.addClassCleanup(holder.cleanup)
        cls.decision_folder = Path(holder.name)
        (cls.decision_folder / "decision_config.json").write_text(json.dumps({
            "letter_token_ids": list(range(32, 58)),
            "temperature_by_type": {"choice": 1.4364, "noul": 1.0411, "score": 1.4021}}))
        (cls.decision_folder / "config.json").write_text(
            json.dumps({"text_config": {"vocab_size": 300}, "image_token_id": 100,
                        "vision_start_token_id": 102, "vision_end_token_id": 103}))

    def config(self, **params):
        cfg = decision_config(**params)
        cfg["source"]["ref"] = str(self.decision_folder)
        return cfg


class TestAdapterContract(AdapterTestCase):
    def test_describe_is_the_production_multimodal_describe(self):
        config = self.config()
        report = adapter.describe(config)
        production = multimodal.describe(config)
        self.assertEqual(report["io"], production["io"])
        self.assertEqual([node["name"] for node in report["io"]["inputs"]],
                         ["input_ids", "position_ids", "actual_seq_lengths",
                          "pixel_values", "image_feature_index"])
        self.assertEqual([node["name"] for node in report["io"]["outputs"]], ["scores"])
        self.assertEqual(report["bounds"]["image_grid_thw"], [1, 16, 16])

    def test_prepare_runtime_encodes_the_production_payload(self):
        handle = adapter.prepare_runtime(self.config(), None)
        self.assertIsInstance(handle, adapter.UnifiedRuntimeHandle)
        handle._preprocessor = StubPreprocessor()        # fake folder config, real builder
        case = {"id": "u-b2", "samples": 2, "payload": {"rows": [
            {"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6, 7]}]}}
        arrays = handle.encode(case)
        # the accepted minimum-physical-width floor comes from compiled_seq_len (256):
        # short rows pad to 256, actual_seq_lengths keep the TRUE lengths
        self.assertEqual(arrays["input_ids"].shape, (2, 256))
        self.assertEqual(arrays["actual_seq_lengths"].tolist(), [3, 4])
        self.assertTrue((arrays["input_ids"][:, 4:] == 0).all())
        self.assertTrue((arrays["image_feature_index"][:, 4:] == -1).all())
        self.assertTrue((arrays["image_feature_index"] == -1).all())
        decoded = handle.decode({"scores": np.zeros((2, 26), dtype=np.float32)}, case)
        self.assertEqual(decoded["alignment"]["rows"], ["u-b2", "u-b2"])
        self.assertEqual(decoded["alignment"]["columns"], list(range(32, 58)))
        self.assertEqual(handle.output_capacities(case), {"scores": 2 * 26 * 4})

    def test_describe_reports_the_encode_policy(self):
        report = adapter.describe(self.config())
        policy = report["encode_policy"]
        self.assertEqual(policy["min_physical_width"], 256)       # compiled_seq_len
        self.assertIn("max(min_physical_width", policy["rule"])
        self.assertIn("NOT fixed", policy["dynamic_s_note"])
        self.assertFalse(policy["real_ge_validated_here"])

    def test_encode_rejects_bad_samples_and_cap_overruns(self):
        handle = adapter.prepare_runtime(self.config(max_batch=2), None)
        handle._preprocessor = StubPreprocessor()
        for case in (
                {"id": "a", "samples": 2.0, "payload": {"rows": [{"input_ids": [1]},
                                                                 {"input_ids": [2]}]}},
                {"id": "b", "samples": True, "payload": {"rows": [{"input_ids": [1]},
                                                                  {"input_ids": [2]}]}},
                {"id": "c", "samples": 2, "payload": {"rows": [{"input_ids": [1]}]}},
                {"id": "d", "samples": 3, "payload": {"rows": [{"input_ids": [1]},
                                                               {"input_ids": [2]},
                                                               {"input_ids": [3]}]}},
        ):
            with self.assertRaises(ValueError):
                handle.encode(case)


class TestReferenceHandle(AdapterTestCase):
    """The INDEPENDENT official CPU FP32 baseline on a tiny multimodal model."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.folder = tiny_multimodal_folder()
        cls.addClassCleanup(lambda: __import__("shutil").rmtree(cls.folder,
                                                                ignore_errors=True))

    def config(self, **params):
        cfg = decision_config(**params)
        cfg["source"]["ref"] = str(self.folder)
        return cfg

    def handle(self):
        source = tiny_source(self.folder, torch.float32)
        handle = adapter.prepare_reference(source, self.config())
        handle.preprocessor    # the lazy preprocessor resolves the same folder config
        return handle

    def test_letters_and_vocab_come_from_the_folder(self):
        handle = self.handle()
        self.assertEqual(handle.letters26, list(range(26, 52)))
        self.assertEqual(handle.vocab_size, 128)

    def test_text_rows_match_the_validated_base_reference_bitexact(self):
        handle = self.handle()
        rows = [text_row(21), text_row(22, 24)]
        mine = handle.run({"rows": [{"input_ids": row} for row in rows]})["scores"]
        text_handle = base.prepare_reference(tiny_source(self.folder, torch.float32),
                                             self.config())
        case = {"id": "t", "samples": 2, "payload": {
            "input_ids": [row[0].tolist() for row in rows],
            "position_ids": [list(range(row.shape[1])) for row in rows],
            "actual_seq_lengths": [row.shape[1] for row in rows]}}
        official = text_handle.run(text_handle.encode(case))["scores"]
        self.assertTrue(torch.equal(mine, official))

    def test_batch_rows_are_independent_and_unpadded(self):
        handle = self.handle()
        row_a = {"input_ids": text_row(31, 17)}
        row_b = {"input_ids": text_row(32, 40)}
        batched = handle.run({"rows": [row_a, row_b]})["scores"]
        alone_a = handle.run({"rows": [row_a]})["scores"]
        alone_b = handle.run({"rows": [row_b]})["scores"]
        self.assertTrue(torch.equal(batched[0], alone_a[0]))
        self.assertTrue(torch.equal(batched[1], alone_b[0]))

    def test_image_row_official_recipe(self):
        handle = self.handle()
        row = image_row(41)
        scores = handle.run({"rows": [row]})["scores"]
        self.assertEqual(tuple(scores.shape), (1, 26))
        self.assertTrue(torch.isfinite(scores).all())
        # the recipe is the official one: recompute it here from the same modules
        source = tiny_source(self.folder, torch.float32)
        inner, body = source.model, source.model.language_model
        handle.bidirectional.enable(body)          # the official patch on this instance
        head_rows = source.get_output_embeddings().weight.index_select(
            0, torch.tensor(handle.letters26)).float()
        with torch.inference_mode():
            embeds = inner.get_input_embeddings()(row["input_ids"])
            feats = inner.get_image_features(row["pixel_values"], row["image_grid_thw"],
                                             return_dict=True).pooler_output
            feats = torch.cat(list(feats), 0).to(embeds.dtype)
            mask, _ = inner.get_placeholder_mask(row["input_ids"], inputs_embeds=embeds,
                                                  image_features=feats)
            embeds = embeds.masked_scatter(mask, feats)
            positions, _ = inner.get_rope_index(
                row["input_ids"], image_grid_thw=row["image_grid_thw"], attention_mask=None,
                mm_token_type_ids=row["mm_token_type_ids"])
            hidden = handle.bidirectional.text_forward(body, inputs_embeds=embeds,
                                                       position_ids=positions)
            reference = hidden[0, -1].float() @ head_rows.transpose(0, 1)
        self.assertTrue(torch.equal(scores[0], reference))

    def test_image_pixels_change_the_scores(self):
        handle = self.handle()
        row_a = image_row(51)
        row_b = image_row(52)
        row_b["input_ids"] = row_a["input_ids"]
        row_b["mm_token_type_ids"] = row_a["mm_token_type_ids"]
        row_b["image_grid_thw"] = row_a["image_grid_thw"]
        scores_a = handle.run({"rows": [row_a]})["scores"]
        scores_b = handle.run({"rows": [row_b]})["scores"]
        self.assertFalse(torch.equal(scores_a, scores_b))

    def test_decode_and_capacities(self):
        handle = self.handle()
        case = {"id": "img", "samples": 1, "payload": {"rows": [{"input_ids": [1, 2]}]}}
        scores = torch.randn(1, 26)
        decoded = handle.decode({"scores": scores}, case)
        self.assertEqual(decoded["outputs"]["scores"].shape, (1, 26))
        self.assertEqual(decoded["alignment"]["rows"], ["img"])
        self.assertEqual(handle.output_capacities(case), {"scores": 26 * 4})


class TestCandidateComposition(AdapterTestCase):
    """The production composition path: mixed dtypes, explicit device move, export spec."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.folder = tiny_multimodal_folder()
        cls.addClassCleanup(lambda: __import__("shutil").rmtree(cls.folder,
                                                                ignore_errors=True))

    def config(self, **params):
        cfg = decision_config(**params)
        cfg["source"]["ref"] = str(self.folder)
        return cfg

    def candidate(self):
        source = tiny_source(self.folder, torch.bfloat16)
        return adapter.build_unified_candidate(source, self.config())

    def test_placement_report_and_mixed_dtypes(self):
        unified, report = self.candidate()
        placement = report["placement"]
        self.assertEqual(placement["vision"], "float32")
        self.assertEqual(placement["language_trunk"], "bfloat16")
        self.assertEqual(placement["head26"], "float32")
        self.assertIn("pos_embeds", placement["buffers"])
        for name, param in unified.named_parameters():
            self.assertEqual(param.device.type, "cpu", name)
        for name, param in unified.vision.named_parameters():
            self.assertEqual(param.dtype, torch.float32, name)
        for name, param in unified.text.body.named_parameters():
            expected = torch.float32 if name.endswith("linear_attn.A_log") \
                else torch.bfloat16
            self.assertEqual(param.dtype, expected, name)
        self.assertEqual(unified.text.head_rows.dtype, torch.float32)
        self.assertIn("restore", report)
        # the restored A_log parameters carry REAL hierarchical names and are F32
        a_log = [n for n, p in unified.named_parameters() if n.endswith(".linear_attn.A_log")]
        self.assertTrue(a_log)
        self.assertIn("text.body.layers.0.linear_attn.A_log", a_log)
        for name in a_log:
            self.assertEqual(dict(unified.named_parameters())[name].dtype, torch.float32)
        self.assertEqual(placement["a_log_float32"], sorted(a_log))

    def test_stray_tensor_is_caught_by_the_placement_verifier(self):
        unified, _report = self.candidate()
        unified.vision.pos_embeds = unified.vision.pos_embeds.data.to("meta")
        with self.assertRaises(RuntimeError):
            adapter._verify_candidate_placement(unified, torch.device("cpu"))

    def _legal_mixed_wrapper(self):
        """A minimal wrapper with the REAL hierarchical names of the contract:
        F32 vision + BF16 body weights + F32 linear_attn.A_log + F32 head26,
        all on one device -- exactly the production shape, and it must PASS."""
        import torch.nn as nn

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.in_proj_qkv = nn.Linear(64, 64, bias=False).to(torch.bfloat16)
                self.A_log = nn.Parameter(torch.randn(4).float())

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_attn = Attn()

        class Body(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(128, 64).to(torch.bfloat16)
                self.layers = nn.ModuleList([Layer() for _ in range(3)])

        class Vision(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList(
                    [nn.Sequential(nn.Linear(64, 64).float())])
                self.register_buffer("pos_embeds", torch.zeros(2, 64))

        class Text(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = Body()
                self.head_rows = nn.Parameter(torch.randn(26, 64).float())

        class Unified(nn.Module):
            def __init__(self):
                super().__init__()
                self.text = Text()
                self.vision = Vision()

        return Unified().eval()

    def test_verifier_accepts_the_legal_mixed_contract(self):
        unified = self._legal_mixed_wrapper()
        report = adapter._verify_candidate_placement(unified, torch.device("cpu"))
        self.assertEqual(report["a_log_float32"],
                         [f"text.body.layers.{i}.linear_attn.A_log" for i in range(3)])

    def test_verifier_rejects_a_log_read_back_as_bfloat16(self):
        import torch.nn as nn

        unified = self._legal_mixed_wrapper()
        unified.text.body.layers[0].linear_attn.A_log = nn.Parameter(
            unified.text.body.layers[0].linear_attn.A_log.data.to(torch.bfloat16))
        with self.assertRaises(RuntimeError) as ctx:
            adapter._verify_candidate_placement(unified, torch.device("cpu"))
        self.assertIn("layers.0.linear_attn.A_log", str(ctx.exception))

    def test_verifier_rejects_a_regular_body_weight_in_float32(self):
        unified = self._legal_mixed_wrapper()
        unified.text.body.embed_tokens = unified.text.body.embed_tokens.float()
        with self.assertRaises(RuntimeError) as ctx:
            adapter._verify_candidate_placement(unified, torch.device("cpu"))
        self.assertIn("embed_tokens.weight", str(ctx.exception))

    def test_verifier_rejects_unknown_parameters(self):
        unified = self._legal_mixed_wrapper()
        unified.something_else = torch.nn.Parameter(torch.zeros(2))
        with self.assertRaises(RuntimeError) as ctx:
            adapter._verify_candidate_placement(unified, torch.device("cpu"))
        self.assertIn("unknown parameter", str(ctx.exception))

    def test_export_spec_carries_the_dynamic_trace_example(self):
        from core.contracts import validate_io
        from core.exporters.torchair import _example_values

        unified, report = self.candidate()
        handle = adapter.UnifiedCandidateHandle(unified, self.config(), report)
        spec = handle.export_spec()
        validate_io(spec["io"])
        example = spec["example_inputs"]
        self.assertEqual(set(example), {"input_ids", "position_ids",
                                        "actual_seq_lengths", "pixel_values",
                                        "image_feature_index"})
        values = _example_values(torch, spec["io"], example)   # raises on any violation
        self.assertEqual(values[0].shape[0], 2)                 # B >= 2 on both dynamic axes
        self.assertEqual(values[0].shape[1], 65)                # S >= 2
        self.assertIs(spec["model"], unified)

    def test_candidate_handle_encode_and_caps(self):
        unified, report = self.candidate()
        handle = adapter.UnifiedCandidateHandle(unified, self.config(), report)
        handle._preprocessor = StubPreprocessor()
        case = {"id": "u-b1", "samples": 1, "payload": {"rows": [{"input_ids": [7, 8, 9]}]}}
        arrays = handle.encode(case)
        for name in ("input_ids", "position_ids", "actual_seq_lengths", "pixel_values",
                     "image_feature_index"):
            self.assertIn(name, arrays)
            self.assertTrue(torch.is_tensor(arrays[name]))
        self.assertEqual(arrays["input_ids"].shape, (1, 256))    # compiled_seq_len floor
        self.assertEqual(int(arrays["actual_seq_lengths"][0]), 3)
        self.assertEqual(handle.output_capacities(case), {"scores": 26 * 4})

    def test_non_bfloat16_source_is_rejected(self):
        source = tiny_source(self.folder, torch.float32)
        with self.assertRaises(ValueError):
            adapter.build_unified_candidate(source, self.config())


if __name__ == "__main__":
    unittest.main()
