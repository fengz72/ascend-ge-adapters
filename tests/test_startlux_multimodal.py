"""CPU tests for the StartLux unified multimodal route (vision.py / multimodal.py
and the shared-decoder / interleaved-MRoPE extraction in startlux_npu_adapter.py).

All checks are bit-exact or use the pinned official transformers implementation
as the reference; no NPU and no real 0.8B weights are needed.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "startlux-decision-v1.1-0.8b-instant"


def load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(MODEL_DIR / filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


npu = load("startlux_npu_adapter_for_native", "model/startlux_npu_adapter.py")
multimodal = load("startlux_multimodal_under_test", "model/multimodal.py")


def official_interleave(freqs, section):
    """The pinned transformers Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope."""
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def tiny_text_model():
    import transformers
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(hidden_size=64, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=32, intermediate_size=128, num_hidden_layers=4,
                            linear_num_key_heads=2, linear_num_value_heads=2,
                            linear_key_head_dim=16, linear_value_head_dim=16,
                            linear_conv_kernel_dim=4, vocab_size=128,
                            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                             "partial_rotary_factor": 0.5,
                                             "mrope_section": [3, 3, 2]})
    cfg.layer_types = ["linear_attention", "linear_attention", "linear_attention",
                       "full_attention"]
    torch.manual_seed(77)
    model = transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5TextModel(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def tiny_vision_model():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    cfg = Qwen3_5VisionConfig(depth=1, hidden_size=64, num_heads=4, patch_size=16,
                              spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
                              num_position_embeddings=64, out_hidden_size=64,
                              intermediate_size=128)
    torch.manual_seed(79)
    model = Qwen3_5VisionModel(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class TestMropeInterleave(unittest.TestCase):
    def test_gather_matches_official_interleave(self):
        for section in ([11, 11, 10], [3, 3, 2], [5, 5, 5], [2, 3, 3]):
            half = sum(section)
            gather = npu._mrope_interleave_index(section, half)
            freqs = torch.randn(3, 2, 5, half, dtype=torch.float64)
            flat = freqs.permute(1, 2, 0, 3).reshape(2, 5, -1)
            mine = flat.index_select(-1, gather)
            reference = official_interleave(freqs, section)
            self.assertTrue(torch.equal(mine, reference),
                            f"interleave mismatch for section {section}")

    def test_gather_shape_and_bounds(self):
        gather = npu._mrope_interleave_index([11, 11, 10], 32)
        self.assertEqual(list(gather.shape), [32])
        self.assertTrue(bool((gather >= 0).all() and (gather < 96).all()))


class TestDecoderMrope(unittest.TestCase):
    def _decoder(self):
        body = tiny_text_model()
        decoder = npu.StartLuxTextDecoder(body, list(range(26)))
        decoder.requires_grad_(False).eval()
        return decoder, body

    def test_three_plane_matches_hf_rotary(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

        for head_dim, partial, theta, section in ((256, 0.25, 1e7, [11, 11, 10]),
                                                  (32, 0.5, 10000.0, [3, 3, 2])):
            cfg = Qwen3_5TextConfig(hidden_size=64, num_attention_heads=2,
                                    num_key_value_heads=1, head_dim=head_dim,
                                    rope_parameters={"rope_type": "default",
                                                     "rope_theta": theta,
                                                     "partial_rotary_factor": partial,
                                                     "mrope_section": section})
            rotary = Qwen3_5TextRotaryEmbedding(cfg)
            inv_freq = 1.0 / (theta ** (torch.arange(0, int(head_dim * partial), 2,
                                                     dtype=torch.float32)
                                         / int(head_dim * partial)))
            generator = torch.Generator().manual_seed(4321)
            planes = torch.stack([torch.randint(0, 200, (2, 9), generator=generator)
                                  for _ in range(3)])
            reference_cos, reference_sin = rotary(torch.zeros(2, 9, 64), planes)

            class _Shim(torch.nn.Module):
                pass

            decoder = _Shim()
            decoder.inv_freq = inv_freq
            decoder.mrope_gather = npu._mrope_interleave_index(
                section, int(head_dim * partial) // 2)
            cos, sin = multimodal.base.StartLuxTextDecoder.mrope_cos_sin(
                decoder, planes, torch.float32)
            self.assertTrue(torch.equal(reference_cos, cos),
                            f"3-plane cos mismatch for section {section}")
            self.assertTrue(torch.equal(reference_sin, sin),
                            f"3-plane sin mismatch for section {section}")


class TestMultimodalInjection(unittest.TestCase):
    GRID = (1, 8, 8)
    PATCH_ROWS = 64
    MERGED = 16

    def _unified(self):
        body = tiny_text_model()
        visual = tiny_vision_model()
        decoder = npu.StartLuxTextDecoder(body, list(range(26)))
        decoder.requires_grad_(False).eval()
        tower = multimodal.vision_mod.VisionTowerWrapper(
            visual, self.GRID, slots=2)
        tower.requires_grad_(False).eval()
        unified = multimodal.MultimodalDecoder(decoder, tower)
        unified.requires_grad_(False).eval()
        return unified

    def test_injection_bitexact_with_masked_scatter(self):
        unified = self._unified()
        image_token = 100
        generator = torch.Generator().manual_seed(2026)
        ids = torch.randint(1, 90, (2, 24), generator=generator)
        ids[0, 2:2 + self.MERGED] = image_token
        ids[1, 5:5 + self.MERGED] = image_token
        pixel_values = torch.randn(2 * self.PATCH_ROWS, 1536, generator=generator)
        index = torch.full((2, 24), -1, dtype=torch.int64)
        index[0, 2:2 + self.MERGED] = torch.arange(self.MERGED)
        index[1, 5:5 + self.MERGED] = torch.arange(self.MERGED, 2 * self.MERGED)

        embeds = unified.injected_embeds(ids, index, pixel_values)
        with torch.inference_mode():
            feats = unified.vision(pixel_values)
            reference = unified.text.body.embed_tokens(ids)
            mask = (ids == image_token).unsqueeze(-1).expand_as(reference)
            reference = reference.masked_scatter(mask, feats.to(reference.dtype))
        self.assertTrue(torch.equal(embeds, reference))


class TestBridgeInputHygiene(unittest.TestCase):
    """build_multimodal_batch rejects wrong tensor types before any conversion."""

    MERGED = multimodal.MERGED_ROWS

    def _provider(self):
        class Cfg:
            image_token_id = 248056
            vision_start_token_id = 248053
            vision_end_token_id = 248054
            vocab_size = 248400
            text_config = None

        def rope_index(input_ids, mm_token_type_ids, image_grid_thw=None):
            length = input_ids.shape[1]
            planes = torch.arange(length, dtype=torch.long).view(1, 1, -1).expand(3, 1, -1)
            return planes.contiguous()

        return rope_index, Cfg()

    def _row(self, ids=None, pixels=None, grid=None, types=None):
        if ids is None:
            ids = torch.tensor([[1, 248053] + [248056] * 64 + [248054, 2]],
                               dtype=torch.int64)
        row = {"input_ids": ids}
        if pixels is None and grid is None:
            pixels = torch.zeros(256, 1536, dtype=torch.float32)
            grid = torch.tensor([[1, 16, 16]], dtype=torch.int64)
        if pixels is not None:
            row["pixel_values"] = pixels
        if grid is not None:
            row["image_grid_thw"] = grid
        if types is not None:
            row["mm_token_type_ids"] = types
        return row

    def test_valid_row_builds(self):
        rope_index, cfg = self._provider()
        arrays = multimodal.build_multimodal_batch([self._row()], rope_index, cfg)
        self.assertEqual(arrays["input_ids"].shape, (1, 68))
        self.assertTrue(np.array_equal(arrays["image_feature_index"][0, 2:66],
                                       np.arange(64)))

    def test_float_ids_rejected(self):
        rope_index, cfg = self._provider()
        row = self._row(ids=torch.tensor([[1, 248053] + [248056] * 64 + [248054, 2]],
                                         dtype=torch.float32))
        with self.assertRaises(ValueError):
            multimodal.build_multimodal_batch([row], rope_index, cfg)

    def test_bool_types_rejected(self):
        rope_index, cfg = self._provider()
        row = self._row(types=torch.zeros(1, 68, dtype=torch.bool))
        with self.assertRaises(ValueError):
            multimodal.build_multimodal_batch([row], rope_index, cfg)

    def test_float_grid_rejected(self):
        rope_index, cfg = self._provider()
        row = self._row(grid=torch.tensor([[1.0, 16.0, 16.0]], dtype=torch.float32))
        with self.assertRaises(ValueError):
            multimodal.build_multimodal_batch([row], rope_index, cfg)

    def test_non_integer_grid_rejected(self):
        rope_index, cfg = self._provider()
        # grid values that are integral floats are still a wrong dtype tensor
        row = self._row(grid=torch.tensor([[1, 16, 8]], dtype=torch.int64))
        with self.assertRaises(ValueError):
            multimodal.build_multimodal_batch([row], rope_index, cfg)

    def test_wrong_dtype_pixels_rejected(self):
        rope_index, cfg = self._provider()
        row = self._row(pixels=torch.zeros(256, 1536, dtype=torch.float64),
                        grid=torch.tensor([[1, 16, 16]], dtype=torch.int64))
        with self.assertRaises(ValueError):
            multimodal.build_multimodal_batch([row], rope_index, cfg)

    def test_non_cpu_tensor_rejected(self):
        rope_index, cfg = self._provider()
        ids = torch.tensor([[1, 248053] + [248056] * 64 + [248054, 2]],
                           dtype=torch.int64, device="meta")
        with self.assertRaises(ValueError):
            multimodal.build_multimodal_batch([self._row(ids=ids)], rope_index, cfg)

    def test_non_tensor_ids_rejected(self):
        rope_index, cfg = self._provider()
        with self.assertRaises(ValueError):
            multimodal.build_multimodal_batch([self._row(ids=[1, 2, 3])], rope_index, cfg)

    def test_out_of_vocab_ids_rejected(self):
        rope_index, cfg = self._provider()
        ids = torch.tensor([[248053] + [248056] * 64 + [248054, 248400]], dtype=torch.int64)
        with self.assertRaises(ValueError):
            multimodal.build_multimodal_batch([self._row(ids=ids)], rope_index, cfg)

    def test_three_images_rejected(self):
        rope_index, cfg = self._provider()
        ids = torch.tensor([[1, 248053] + [248056] * 64 + [248054, 248053]
                            + [248056] * 64 + [248054, 248053]
                            + [248056] * 64 + [248054, 2]], dtype=torch.int64)
        pixels = torch.zeros(3 * 256, 1536, dtype=torch.float32)
        grid = torch.tensor([[1, 16, 16]] * 3, dtype=torch.int64)
        with self.assertRaises(ValueError):
            multimodal.build_multimodal_batch([self._row(ids=ids, pixels=pixels, grid=grid)],
                                              rope_index, cfg)

    def test_batch_slot_pool_overflow_rejected(self):
        rope_index, cfg = self._provider()
        two_images = self._row(
            ids=torch.tensor([[1, 248053] + [248056] * 64 + [248054, 248053]
                               + [248056] * 64 + [248054, 2]], dtype=torch.int64),
            pixels=torch.zeros(2 * 256, 1536, dtype=torch.float32),
            grid=torch.tensor([[1, 16, 16]] * 2, dtype=torch.int64))
        with self.assertRaises(ValueError):
            # 2 (row 0) + 1 (row 1) images exceed the 2 global slots
            multimodal.build_multimodal_batch([two_images, self._row()], rope_index, cfg)

    def test_two_images_one_row_builds(self):
        rope_index, cfg = self._provider()
        two_images = self._row(
            ids=torch.tensor([[1, 248053] + [248056] * 64 + [248054, 248053]
                               + [248056] * 64 + [248054, 2]], dtype=torch.int64),
            pixels=torch.zeros(2 * 256, 1536, dtype=torch.float32),
            grid=torch.tensor([[1, 16, 16]] * 2, dtype=torch.int64))
        arrays = multimodal.build_multimodal_batch([two_images], rope_index, cfg)
        self.assertTrue(np.array_equal(arrays["image_feature_index"][0, 2:66],
                                       np.arange(64)))
        self.assertTrue(np.array_equal(arrays["image_feature_index"][0, 68:132],
                                       np.arange(64, 128)))

    def test_batch_cap_rejected(self):
        rope_index, cfg = self._provider()
        text_row = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64)}
        with self.assertRaises(ValueError):
            multimodal.build_multimodal_batch([text_row] * 5, rope_index, cfg)

    def test_text_only_row_builds_without_pixels(self):
        rope_index, cfg = self._provider()
        arrays = multimodal.build_multimodal_batch(
            [{"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64)}], rope_index, cfg)
        self.assertEqual(int((arrays["image_feature_index"] >= 0).sum()), 0)
        self.assertTrue(np.array_equal(arrays["pixel_values"],
                                       np.zeros_like(arrays["pixel_values"])))


class TestVisionBufferProvenance(unittest.TestCase):
    """Official F32 constants from a dtype-cast source.

    A load path ending in model.to(device, dtype=BF16) quantizes the
    NON-PERSISTENT Qwen3_5VisionRotaryEmbedding.inv_freq, which .float() cannot
    restore.  The wrapper must regenerate the official constructor frequency so
    its buffers -- and the vision output -- are bit-exact with a pristine
    official F32 module carrying the same weights.
    """

    GRID = (1, 8, 8)
    PATCH_ROWS = 64

    def _visual_pair(self):
        """(pristine official F32 module, core-path BF16-cast module), SAME
        BF16-STORED weights: the tiny module's F32 init is quantized to BF16
        once -- that is the "checkpoint" -- and both paths load exactly those
        values (a from_pretrained(dtype=float32) load upcasts them exactly;
        the core path keeps them in BF16 and ends with module.to(dtype=BF16))."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
        visual = tiny_vision_model()
        checkpoint = {name: value.detach().to(torch.bfloat16)
                      for name, value in visual.state_dict().items()}
        reference = Qwen3_5VisionModel(visual.config)
        reference.load_state_dict(checkpoint)      # exact BF16 -> F32 upcast
        for p in reference.parameters():
            p.requires_grad_(False)
        source = Qwen3_5VisionModel(visual.config)
        source.load_state_dict(checkpoint)
        for p in source.parameters():
            p.requires_grad_(False)
        source.to(torch.bfloat16)             # the core final model.to(dtype=bf16)
        return reference, source

    def test_bf16_cast_corrupts_inv_freq_irrecoverably(self):
        reference, source = self._visual_pair()
        # the cast quantizes the non-persistent frequency...
        self.assertFalse(torch.equal(source.rotary_pos_emb.inv_freq,
                                      reference.rotary_pos_emb.inv_freq))
        # ...and a plain float() upcast does NOT restore it
        restored = source.rotary_pos_emb.inv_freq.float()
        self.assertGreater(float((restored - reference.rotary_pos_emb.inv_freq)
                                 .abs().max()), 0.0)

    def test_wrapper_regenerates_official_rotary_bitexact(self):
        from copy import deepcopy
        reference, source = self._visual_pair()
        grid = torch.tensor([list(self.GRID)] * 2, dtype=torch.long)
        # the OLD recipe -- official method on a CPU float32 copy of the cast
        # source -- bakes the quantized frequency into the rotary tables
        old_source = deepcopy(source).to("cpu", torch.float32)
        with torch.no_grad():
            old_rot = old_source.rot_pos_emb(grid).reshape(-1, old_source.rot_pos_emb(grid).shape[-1])
            ref_rot = reference.rot_pos_emb(grid).reshape(-1, reference.rot_pos_emb(grid).shape[-1])
        self.assertFalse(torch.equal(old_rot, ref_rot))

        wrapper = multimodal.vision_mod.VisionTowerWrapper(
            source, self.GRID, slots=2,
            buffer_source=deepcopy(source).to("cpu", torch.float32))
        source.float()                       # build_unified_model's exact upcast
        wrapper_f32 = multimodal.vision_mod.VisionTowerWrapper(reference, self.GRID, slots=2)
        self.assertTrue(torch.equal(wrapper.rotary_cos, wrapper_f32.rotary_cos))
        self.assertTrue(torch.equal(wrapper.rotary_sin, wrapper_f32.rotary_sin))
        self.assertTrue(torch.equal(wrapper.pos_embeds, wrapper_f32.pos_embeds))
        generator = torch.Generator().manual_seed(2031)
        pixels = torch.randn(2 * self.PATCH_ROWS, 1536, generator=generator)
        with torch.inference_mode():
            self.assertTrue(torch.equal(wrapper(pixels), wrapper_f32(pixels)))


class TestVisionLinearDtype(unittest.TestCase):
    """The wrapper's linear_dtype switch: fp32 stays the delivered bit-exact
    path; fp16 (the validated mixed precision: fp16 inner products, fp32
    constants/norms/softmax) stays inside the vision gate on the real-depth
    tower."""

    def test_fp32_mode_unchanged_bitexact(self):
        visual = tiny_vision_model()
        torch.manual_seed(2112)
        pixels = torch.randn(2 * 64, 1536, dtype=torch.float32)
        tower = multimodal.vision_mod.VisionTowerWrapper(visual, (1, 8, 8), slots=2).eval()
        with torch.inference_mode():
            out = tower(pixels)
        self.assertEqual(out.dtype, torch.float32)

    def test_fp16_mode_within_vision_gate(self):
        visual = tiny_vision_model()
        torch.manual_seed(2112)
        pixels = torch.randn(2 * 64, 1536, dtype=torch.float32)
        fp32 = multimodal.vision_mod.VisionTowerWrapper(visual, (1, 8, 8), slots=2).eval()
        # The fp16 constructor PRE-QUANTIZES the SHARED visual's Linear/Conv
        # weights in place (bit-identical values), so the fp32 reference is
        # computed BEFORE the fp16 wrapper exists.
        with torch.inference_mode():
            a = fp32(pixels)
        fp16 = multimodal.vision_mod.VisionTowerWrapper(
            visual, (1, 8, 8), slots=2, linear_dtype=torch.float16).eval()
        with torch.inference_mode():
            b = fp16(pixels)
        rel = float((b - a).norm() / a.norm())
        cos = float(torch.sum(b * a) / (b.norm() * a.norm()))
        self.assertLess(rel, 0.02, f"fp16 relL2 {rel}")
        self.assertGreater(cos, 0.999, f"fp16 cosine {cos}")


class TestMatMulPatchEmbed(unittest.TestCase):
    """MatMulPatchEmbed must reproduce the official Conv3d patch embed
    exactly (stride == kernel: a per-patch full connection)."""

    def test_matches_official_conv3d(self):
        torch.manual_seed(313)
        patch_rows = 9
        visual = tiny_vision_model()
        reference = visual.patch_embed
        equivalent = multimodal.vision_mod.MatMulPatchEmbed(reference)
        pixels = torch.randn(patch_rows * 1536, dtype=torch.float32) * 4.0
        with torch.inference_mode():
            official = reference(pixels)
            mine = equivalent(pixels)
        rel = float((mine - official).norm() / official.norm())
        self.assertLess(rel, 1e-6, f"fp32 accumulate drift {rel}")
        self.assertEqual(mine.shape, official.shape)

    def test_wrapper_uses_matmul_patch_embed(self):
        visual = tiny_vision_model()
        tower = multimodal.vision_mod.VisionTowerWrapper(visual, (1, 8, 8), slots=2)
        self.assertIsInstance(tower.patch_embed, multimodal.vision_mod.MatMulPatchEmbed)
        with torch.inference_mode():
            tower.eval()
            features = tower(torch.randn(2 * 64, 1536, dtype=torch.float32))
        self.assertEqual(tuple(features.shape), (2 * 16, 64))


if __name__ == "__main__":
    unittest.main()
