#!/usr/bin/env python3

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from pose_point_depth_mv.evaluate_proobjaverse_official_ss_support_vs_reconviagen import (
    _coordinate_contract,
    _metric_improvement,
    _sample_strict_stock_support,
    support_quality,
)
from trellis.pipelines.trellis_image_to_3d import TrellisVGGTTo3DPipeline


class SupportComparisonTest(unittest.TestCase):
    def test_exact_support_is_perfect(self) -> None:
        coords = np.asarray(
            [[0, 1, 1, 1], [0, 2, 2, 2], [0, 2, 2, 3]], dtype=np.int32
        )
        contract = _coordinate_contract(coords, label="exact")
        quality = support_quality(coords, coords)
        self.assertTrue(contract["passed"])
        self.assertEqual(quality["iou"], 1.0)
        self.assertEqual(quality["precision"], 1.0)
        self.assertEqual(quality["recall"], 1.0)
        self.assertEqual(quality["f1"], 1.0)
        self.assertEqual(quality["count_abs_log_error"], 0.0)
        self.assertEqual(quality["component_count_abs_error"], 0.0)
        self.assertEqual(quality["largest_component_ratio_abs_error"], 0.0)

    def test_false_positive_and_negative_are_explicit(self) -> None:
        target = np.asarray([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=np.int32)
        predicted = np.asarray([[0, 1, 1, 1], [0, 3, 3, 3]], dtype=np.int32)
        quality = support_quality(predicted, target)
        self.assertEqual(quality["intersection_count"], 1)
        self.assertEqual(quality["false_positive_count"], 1)
        self.assertEqual(quality["false_negative_count"], 1)
        self.assertAlmostEqual(quality["iou"], 1.0 / 3.0)
        self.assertEqual(quality["precision"], 0.5)
        self.assertEqual(quality["recall"], 0.5)

    def test_duplicate_coordinate_contract_fails(self) -> None:
        duplicate = np.asarray([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=np.int32)
        with self.assertRaises(RuntimeError):
            _coordinate_contract(duplicate, label="duplicate")

    def test_error_metrics_use_lower_is_better_sign(self) -> None:
        self.assertGreater(_metric_improvement("iou", 0.8, 0.6), 0.0)
        self.assertGreater(
            _metric_improvement("count_abs_log_error", 0.1, 0.3), 0.0
        )
        self.assertGreater(
            _metric_improvement("component_count_abs_error", 1.0, 4.0), 0.0
        )

    def test_evaluator_calls_original_reconviagen_ss_only_branch(self) -> None:
        class FakePipeline:
            def __init__(self) -> None:
                self.kwargs = None

            def run(self, **kwargs):
                self.kwargs = kwargs
                coords = torch.tensor([[0, 2, 3, 4]], dtype=torch.int32)
                noise = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)
                return None, coords, noise

        pipeline = FakePipeline()
        coords, audit = _sample_strict_stock_support(
            pipeline,
            [object()] * 8,
            seed=42,
            sampler_params={"steps": 30},
        )
        self.assertEqual(coords.tolist(), [[0, 2, 3, 4]])
        self.assertTrue(audit["reconviagen_original_run_ss_only"])
        self.assertIsNotNone(pipeline.kwargs)
        self.assertTrue(pipeline.kwargs["return_ss_support_only"])
        self.assertFalse(pipeline.kwargs["preprocess_image"])
        self.assertEqual(pipeline.kwargs["seed"], 42)
        self.assertEqual(
            pipeline.kwargs["sparse_structure_sampler_params"], {"steps": 30}
        )

    def test_original_reconviagen_ss_only_branch_stops_before_slat(self) -> None:
        class DummySampler:
            def sample(self, _model, noise, **_kwargs):
                return SimpleNamespace(samples=torch.zeros_like(noise))

        class DummyDecoder:
            def __call__(self, _latent):
                occupancy = torch.zeros((1, 1, 2, 2, 2), dtype=torch.float32)
                occupancy[0, 0, 1, 0, 1] = 1.0
                return occupancy

        def forbidden(*_args, **_kwargs):
            raise AssertionError("SLat path must not execute in SS-only mode")

        fake = SimpleNamespace(
            VGGT_dtype=torch.float16,
            low_vram=False,
            device=torch.device("cpu"),
            models={
                "sparse_structure_flow_model": SimpleNamespace(
                    resolution=2, in_channels=1
                ),
                "sparse_structure_decoder": DummyDecoder(),
            },
            sparse_structure_sampler_params={},
            sparse_structure_sampler=DummySampler(),
            vggt_feat=lambda _images: ([torch.zeros((1, 1, 6, 1))], None),
            encode_image=lambda _images: torch.zeros((1, 1, 6, 1024)),
            get_ss_cond=lambda *_args: {"cond": None, "neg_cond": None},
            get_slat_cond=forbidden,
            sample_slat=forbidden,
            decode_slat=forbidden,
        )
        torch.manual_seed(123)
        expected_noise = torch.randn((1, 1, 2, 2, 2))
        outputs, coords, noise = TrellisVGGTTo3DPipeline.run(
            fake,
            image=[object()],
            seed=123,
            return_ss_support_only=True,
        )
        self.assertIsNone(outputs)
        self.assertTrue(torch.equal(noise, expected_noise))
        self.assertEqual(coords.cpu().tolist(), [[0, 1, 0, 1]])


if __name__ == "__main__":
    unittest.main()
