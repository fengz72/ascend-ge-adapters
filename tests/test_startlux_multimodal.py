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
        decoder = npu.StartLuxTextDecoder(body, list(range(26)), chunk_size=4)
        decoder.requires_grad_(False).eval()
        return decoder, body

    def test_text_positions_bitexact_with_previous_formula(self):
        decoder, _body = self._decoder()
        generator = torch.Generator().manual_seed(1234)
        positions = torch.randint(0, 300, (2, 7), generator=generator)
        for dtype in (torch.float32, torch.bfloat16):
            old = npu.rope_cos_sin(positions, decoder.inv_freq, dtype)
            new = decoder.mrope_cos_sin(positions, dtype)
            self.assertTrue(torch.equal(old[0], new[0]) and torch.equal(old[1], new[1]),
                            f"2D text positions drifted for {dtype}")

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
        decoder = npu.StartLuxTextDecoder(body, list(range(26)), chunk_size=4)
        decoder.requires_grad_(False).eval()
        tower = multimodal.vision_mod.VisionTowerWrapper(
            visual, self.GRID, slots=2)
        tower.requires_grad_(False).eval()
        unified = multimodal.MultimodalDecoder(decoder, tower, slots=2)
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


class TestBatchValidation(unittest.TestCase):
    MERGED = multimodal.MERGED_ROWS

    def _arrays(self, **overrides):
        arrays = {
            "input_ids": np.zeros((1, 8), dtype=np.int64),
            "position_ids": np.zeros((3, 1, 8), dtype=np.int64),
            "actual_seq_lengths": np.ones((1,), dtype=np.int32),
            "pixel_values": np.zeros((2 * multimodal.PATCH_ROWS, 1536), dtype=np.float32),
            "image_feature_index": np.full((1, 8), -1, dtype=np.int64),
        }
        arrays.update(overrides)
        return arrays

    def test_minimal_batch_valid(self):
        self.assertTrue(multimodal.validate_batch_arrays(self._arrays()))

    def test_batch_cap(self):
        arrays = self._arrays(
            input_ids=np.zeros((5, 8), dtype=np.int64),
            position_ids=np.zeros((3, 5, 8), dtype=np.int64),
            actual_seq_lengths=np.ones((5,), dtype=np.int32),
            image_feature_index=np.full((5, 8), -1, dtype=np.int64))
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)

    def test_image_tokens_in_padding_rejected(self):
        arrays = self._image_arrays()
        arrays["actual_seq_lengths"] = np.array([8], dtype=np.int32)
        arrays["input_ids"][0, 8] = 248056
        arrays["image_feature_index"][0, 8] = 0
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(
                arrays, image_token=248056, vision_start=248053, vision_end=248054)

    def _image_arrays(self):
        width = 72
        arrays = self._arrays(
            input_ids=np.zeros((1, width), dtype=np.int64),
            position_ids=np.zeros((3, 1, width), dtype=np.int64),
            actual_seq_lengths=np.array([width], dtype=np.int32),
            image_feature_index=np.full((1, width), -1, dtype=np.int64))
        arrays["input_ids"][0, 1] = 248053           # vision_start
        arrays["input_ids"][0, 2:66] = 248056        # 64 image pads
        arrays["input_ids"][0, 66] = 248054          # vision_end
        arrays["image_feature_index"][0, 2:66] = np.arange(64)
        return arrays

    def test_valid_image_batch_accepted(self):
        arrays = self._image_arrays()
        self.assertTrue(multimodal.validate_batch_arrays(
            arrays, vocab_size=248400, image_token=248056,
            vision_start=248053, vision_end=248054))

    def test_non_contiguous_feature_run_rejected(self):
        arrays = self._image_arrays()
        arrays["image_feature_index"][0, 2] = 63
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(
                arrays, image_token=248056, vision_start=248053, vision_end=248054)

    def test_slot_pool_overflow_rejected(self):
        width = 198
        arrays = self._arrays(
            input_ids=np.zeros((1, width), dtype=np.int64),
            position_ids=np.zeros((3, 1, width), dtype=np.int64),
            actual_seq_lengths=np.array([width], dtype=np.int32),
            image_feature_index=np.full((1, width), -1, dtype=np.int64))
        arrays["input_ids"][0, :192] = 248056
        arrays["image_feature_index"][0, :192] = np.concatenate(
            [np.arange(64), np.arange(64, 128), np.arange(64)])
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(
                arrays, image_token=248056, vision_start=248053, vision_end=248054)

    def test_index_token_consistency_rejected(self):
        arrays = self._image_arrays()
        arrays["image_feature_index"][0, 2:66] = -1
        with self.assertRaises(ValueError):
            # image tokens present but no index entries
            multimodal.validate_batch_arrays(
                arrays, image_token=248056, vision_start=248053, vision_end=248054)

    def test_index_out_of_pool_rejected(self):
        # 128..191 sit inside a 64-token group starting at a slot boundary but
        # the vision tower only has 2*64 feature rows
        arrays = self._image_arrays()
        arrays["image_feature_index"][0, 2:66] = np.arange(128, 192)
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(
                arrays, image_token=248056, vision_start=248053, vision_end=248054)

    def test_index_below_minus_one_rejected(self):
        arrays = self._image_arrays()
        arrays["image_feature_index"][0, 0] = -2
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)

    def test_two_images_sharing_one_slot_rejected(self):
        # both groups mapped to 0..63: not the unique official row-major order
        width = 134
        arrays = self._arrays(
            input_ids=np.zeros((1, width), dtype=np.int64),
            position_ids=np.zeros((3, 1, width), dtype=np.int64),
            actual_seq_lengths=np.array([width], dtype=np.int32),
            image_feature_index=np.full((1, width), -1, dtype=np.int64))
        arrays["input_ids"][0, 2:66] = 248056
        arrays["input_ids"][0, 70:134] = 248056
        arrays["image_feature_index"][0, 2:66] = np.arange(64)
        arrays["image_feature_index"][0, 70:134] = np.arange(64)
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(
                arrays, image_token=248056, vision_start=248053, vision_end=248054)

    def test_swapped_slot_order_across_rows_rejected(self):
        # row 0 gets the SECOND slot (64..127) and row 1 the first: wrong order
        arrays = self._arrays(
            input_ids=np.zeros((2, 70), dtype=np.int64),
            position_ids=np.zeros((3, 2, 70), dtype=np.int64),
            actual_seq_lengths=np.array([70, 70], dtype=np.int32),
            image_feature_index=np.full((2, 70), -1, dtype=np.int64))
        arrays["input_ids"][0, 2:66] = 248056
        arrays["input_ids"][1, 2:66] = 248056
        arrays["image_feature_index"][0, 2:66] = np.arange(64, 128)
        arrays["image_feature_index"][1, 2:66] = np.arange(64)
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(
                arrays, image_token=248056, vision_start=248053, vision_end=248054)

    def test_hole_in_feature_order_rejected(self):
        arrays = self._image_arrays()
        arrays["image_feature_index"][0, 2:66] = np.arange(1, 65)
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(
                arrays, image_token=248056, vision_start=248053, vision_end=248054)

    def test_float_and_bool_arrays_rejected(self):
        arrays = self._image_arrays()
        arrays["input_ids"] = arrays["input_ids"].astype(np.float32)
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)
        arrays = self._image_arrays()
        arrays["position_ids"] = arrays["position_ids"].astype(np.int32)
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)
        arrays = self._image_arrays()
        arrays["image_feature_index"] = arrays["image_feature_index"].astype(np.bool_)
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)
        arrays = self._image_arrays()
        arrays["actual_seq_lengths"] = arrays["actual_seq_lengths"].astype(np.int64)
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)
        arrays = self._image_arrays()
        arrays["pixel_values"] = arrays["pixel_values"].astype(np.float64)
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)

    def test_vocab_bounds_enforced(self):
        arrays = self._image_arrays()
        arrays["input_ids"][0, 66] = 300
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays, vocab_size=300)
        arrays["input_ids"][0, 66] = -1
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays, vocab_size=300)

    def test_rank_and_shape_strictness(self):
        arrays = self._image_arrays()
        arrays["input_ids"] = arrays["input_ids"][0]           # 1-D
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)
        arrays = self._image_arrays()
        arrays["position_ids"] = arrays["position_ids"][0]      # 2-D
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)
        arrays = self._image_arrays()
        arrays["pixel_values"] = np.zeros((256, 1536), dtype=np.float32)  # one slot only
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)

    def test_row_structure_rejections(self):
        class Cfg:
            vision_start_token_id = 90
            vision_end_token_id = 91
            image_token_id = 92

        # 60 pads is not the 64 the grid requires
        row = [90] + [92] * 60 + [91] + [1, 2]
        with self.assertRaises(ValueError):
            multimodal._check_row_structure(row, Cfg, 1, 64)
        # unbracketed group
        row = [1] + [92] * 64 + [2]
        with self.assertRaises(ValueError):
            multimodal._check_row_structure(row, Cfg, 1, 64)
        # start marker mismatched (an extra vision_start, counts disagree)
        row = [90, 90] + [92] * 64 + [91, 2]
        with self.assertRaises(ValueError):
            multimodal._check_row_structure(row, Cfg, 1, 64)
        # end marker replaced by a normal token
        row = [1, 90] + [92] * 64 + [5, 2]
        with self.assertRaises(ValueError):
            multimodal._check_row_structure(row, Cfg, 1, 64)
        # correct structure passes
        row = [1, 90] + [92] * 64 + [91, 2]
        groups = multimodal._check_row_structure(row, Cfg, 1, 64)
        self.assertEqual(groups, [(2, 66)])
        # two images, correct structure
        row = [1, 90] + [92] * 64 + [91, 90] + [92] * 64 + [91, 2]
        groups = multimodal._check_row_structure(row, Cfg, 2, 64)
        self.assertEqual(groups, [(2, 66), (68, 132)])


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


class TestPaddingIndexStrictness(unittest.TestCase):
    """image_feature_index must never mark padding positions (unconditional)."""

    MERGED = multimodal.MERGED_ROWS

    def _arrays(self, length=1, width=67):
        return {
            "input_ids": np.zeros((1, width), dtype=np.int64),
            "position_ids": np.zeros((3, 1, width), dtype=np.int64),
            "actual_seq_lengths": np.full((1,), length, dtype=np.int32),
            "pixel_values": np.zeros((2 * multimodal.PATCH_ROWS, 1536), dtype=np.float32),
            "image_feature_index": np.full((1, width), -1, dtype=np.int64),
        }

    def test_padding_index_run_rejected_unconditionally(self):
        # the exact regression counter-example: B1, S67, L1, no valid image,
        # and the 64 padding positions carry arange(64) -- passes token checks
        # and the row-major order check, so only the explicit padding-index
        # rule can catch it. Rejected with AND without marker ids supplied.
        arrays = self._arrays()
        arrays["image_feature_index"][0, 1:65] = np.arange(64)
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays, vocab_size=248400,
                                              image_token=248056,
                                              vision_start=248053, vision_end=248054)

    def test_padding_positions_nonzero_rejected(self):
        arrays = self._arrays()
        arrays["position_ids"][:, 0, 1:] = 7
        with self.assertRaises(ValueError):
            multimodal.validate_batch_arrays(arrays)

    def test_valid_prefix_with_clean_padding_accepted(self):
        arrays = self._arrays()
        self.assertTrue(multimodal.validate_batch_arrays(arrays))


class TestRuntimeHandleEncode(unittest.TestCase):
    """MultimodalRuntimeHandle.encode: raw payload typing + marker validation.

    The handle is the array-only pipeline entry (prepare_runtime); it must not
    be a second, laxer door around build_multimodal_batch's strict validation:
    nested float/bool lists may never be silently truncated into integer
    arrays, and the placeholder marker structure is checked with the real
    config token ids.
    """

    VOCAB = 248400
    TOKENS = {"image_token_id": 248056, "vision_start_token_id": 248053,
              "vision_end_token_id": 248054}

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmp = tempfile.TemporaryDirectory()
        folder = Path(cls._tmp.name) / "model"
        folder.mkdir()
        (folder / "decision_config.json").write_text(__import__("json").dumps({
            "letter_token_ids": list(range(26)),
            "temperature_by_type": {"choice": 1.0, "score": 1.0, "noul": 1.0}}))
        (folder / "config.json").write_text(__import__("json").dumps({
            "text_config": {"vocab_size": cls.VOCAB}, **cls.TOKENS}))
        cls.config = {"model": {"params": {"max_seq_len": 2048, "compiled_seq_len": 256,
                                           "chunk_size": 64, "max_batch": 4,
                                           "fixtures_file": "decider/fixtures/p2b-readouts.json"}},
                      "source": {"ref": str(folder)}}
        cls.handle = multimodal.MultimodalRuntimeHandle(cls.config, {"schema": "ge-io/1"})
        cls.zeros_pixels = [[0.0] * 1536 for _ in range(2 * multimodal.PATCH_ROWS)]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _case(self, **payload):
        base = {
            "input_ids": [[1, 2, 3]],
            "position_ids": [[[0, 0, 0]], [[1, 1, 1]], [[2, 2, 2]]],
            "actual_seq_lengths": [3],
            "pixel_values": self.zeros_pixels,
            "image_feature_index": [[-1, -1, -1]],
        }
        base.update(payload)
        return {"id": "case-1", "samples": 1, "payload": base}

    def test_valid_text_payload_round_trips(self):
        arrays = self.handle.encode(self._case())
        self.assertEqual(arrays["input_ids"].dtype, np.int64)
        self.assertEqual(arrays["pixel_values"].dtype, np.float32)
        self.assertEqual(arrays["pixel_values"].shape,
                         (2 * multimodal.PATCH_ROWS, 1536))
        self.assertTrue(np.array_equal(arrays["image_feature_index"],
                                       np.full((1, 3), -1, dtype=np.int64)))

    def test_float_ids_rejected_not_truncated(self):
        with self.assertRaises(ValueError):
            self.handle.encode(self._case(input_ids=[[1.5, 2.5, 3.5]]))

    def test_bool_ids_rejected_not_truncated(self):
        with self.assertRaises(ValueError):
            self.handle.encode(self._case(input_ids=[[True, False, True]]))

    def test_int_pixels_rejected(self):
        with self.assertRaises(ValueError):
            self.handle.encode(self._case(pixel_values=[[0] * 1536
                                                        for _ in range(2 * multimodal.PATCH_ROWS)]))

    def test_ragged_payload_rejected(self):
        with self.assertRaises(ValueError):
            self.handle.encode(self._case(input_ids=[[1, 2], [3]]))

    def test_out_of_int32_range_lengths_rejected(self):
        with self.assertRaises(ValueError):
            self.handle.encode(self._case(actual_seq_lengths=[2 ** 40]))

    def test_marker_structure_enforced_through_handle(self):
        # 64 image pads WITHOUT the vision_start/vision_end brackets: without
        # the marker ids the handle would accept the mapping as text tokens;
        # with them (the fix) the official placeholder structure is required.
        pads = [self.TOKENS["image_token_id"]] * 64
        payload = {
            "input_ids": [[1] + pads + [2, 3]],
            "position_ids": [[[0] * 66]] * 3,
            "actual_seq_lengths": [66],
            "pixel_values": self.zeros_pixels,
            "image_feature_index": [list(range(64)) + [-1, -1]],
        }
        with self.assertRaises(ValueError):
            self.handle.encode({"id": "case-2", "samples": 1, "payload": payload})

    def test_valid_bracketed_payload_accepted(self):
        start, pad, end = (self.TOKENS["vision_start_token_id"],
                           self.TOKENS["image_token_id"],
                           self.TOKENS["vision_end_token_id"])
        row = [1, start] + [pad] * 64 + [end, 2, 3]      # 69 tokens, pads at 2..65
        payload = {
            "input_ids": [row],
            "position_ids": [[[i] * 69] for i in range(3)],
            "actual_seq_lengths": [69],
            "pixel_values": self.zeros_pixels,
            "image_feature_index": [[-1, -1] + list(range(64)) + [-1] * 3],
        }
        arrays = self.handle.encode({"id": "case-3", "samples": 1, "payload": payload})
        self.assertTrue(np.array_equal(arrays["image_feature_index"][0, 2:66],
                                       np.arange(64)))


class TestVisionBufferProvenance(unittest.TestCase):
    """Official F32 constants from a core-path (dtype-cast) source.

    core.sources.load_source finishes with model.to(device, dtype=BF16); that
    quantizes the NON-PERSISTENT Qwen3_5VisionRotaryEmbedding.inv_freq, which
    .float() cannot restore (work/startlux-cannbot/review-vision-rotary-buffer-
    cast.json). The wrapper must regenerate the official constructor frequency
    so its buffers -- and the vision output -- are bit-exact with a pristine
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
        self.assertTrue(wrapper.rotary_inv_freq_restored)
        source.float()                       # build_unified_model's exact upcast
        wrapper_f32 = multimodal.vision_mod.VisionTowerWrapper(reference, self.GRID, slots=2)
        self.assertFalse(wrapper_f32.rotary_inv_freq_restored)
        self.assertTrue(torch.equal(wrapper.rotary_cos, wrapper_f32.rotary_cos))
        self.assertTrue(torch.equal(wrapper.rotary_sin, wrapper_f32.rotary_sin))
        self.assertTrue(torch.equal(wrapper.pos_embeds, wrapper_f32.pos_embeds))
        generator = torch.Generator().manual_seed(2031)
        pixels = torch.randn(2 * self.PATCH_ROWS, 1536, generator=generator)
        with torch.inference_mode():
            self.assertTrue(torch.equal(wrapper(pixels), wrapper_f32(pixels)))

    def test_buffers_follow_source_device(self):
        reference, _source = self._visual_pair()
        wrapper = multimodal.vision_mod.VisionTowerWrapper(reference, self.GRID, slots=2)
        device = next(reference.parameters()).device
        for name in ("pos_embeds", "rotary_cos", "rotary_sin"):
            self.assertEqual(getattr(wrapper, name).device, device)
        # the real non-CPU source-device coverage is the NPU eager stage of
        # work/startlux-validation/p6-unified (device audit, no masking .to)


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
