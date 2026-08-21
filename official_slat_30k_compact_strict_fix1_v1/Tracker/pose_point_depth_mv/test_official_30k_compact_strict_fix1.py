#!/usr/bin/env python3
"""CPU-only contract tests for compact-v2 plus strict-fix1 integration.

These tests deliberately avoid model construction, CUDA, and real cache data.
They cover only the runtime boundary introduced by the integration: a compact
lifting payload remains on CPU through DDP and supplies image geometry as
metadata while selected DINO/K/T views retain the legacy values and order.
"""

from __future__ import annotations

import unittest

import torch

from pose_point_depth_mv import train_native_slat_genrecon as training
from pose_point_depth_mv.native_slat_genrecon import (
    select_sparse_frustum_inputs_cpu,
    validate_strict_cpu_lifting_sample,
)
from pose_point_depth_mv.native_ss_genrecon import select_dino_features


def _compact_lifting_sample(views: int = 8) -> dict[str, object]:
    visual = torch.arange(
        views * 4 * 1024, dtype=torch.float32
    ).reshape(views, 4, 1024).to(torch.float16)
    return {
        "format": "pose_point_depth_mv.proobjaverse_official_slat_compact_object.v2",
        "uid": "fixture",
        "object_uid": "fixture",
        "visual_patch_features": visual,
        "intrinsics": torch.arange(
            views * 9, dtype=torch.float32
        ).reshape(views, 3, 3),
        "extrinsics": torch.arange(
            views * 16, dtype=torch.float32
        ).reshape(views, 4, 4),
        "image_size": (518, 518),
        "view_ids": torch.arange(views, dtype=torch.int64),
        "stock_condition": torch.zeros(1, 4, 1024, dtype=torch.float16),
        "grid_transform": "identity",
        "extrinsics_type": "world_to_camera",
        "camera_forward_sign": 1.0,
        "compact_projection_only": True,
    }


class Official30KCompactStrictFix1Tests(unittest.TestCase):
    def test_ddp_does_not_own_input_device_migration(self) -> None:
        kwargs = training.strict_perf_ddp_kwargs()
        self.assertIsNone(kwargs["device_ids"])
        self.assertNotIn("output_device", kwargs)
        self.assertFalse(kwargs["broadcast_buffers"])
        self.assertFalse(kwargs["find_unused_parameters"])
        self.assertTrue(kwargs["gradient_as_bucket_view"])

    def test_compact_image_size_replaces_only_zero_depth_shape_metadata(self) -> None:
        sample = _compact_lifting_sample()
        self.assertNotIn("predicted_depth", sample)
        inventory = validate_strict_cpu_lifting_sample(sample)
        self.assertGreater(inventory["tensor_count"], 0)
        self.assertGreater(inventory["tensor_bytes"], 0)
        self.assertEqual(sample["image_size"], (518, 518))

    def test_compact_cpu_selection_is_exact_for_2_4_8_views(self) -> None:
        sample = _compact_lifting_sample()
        order = torch.tensor([7, 0, 5, 2, 6, 1, 4, 3], dtype=torch.long)
        source_visual = select_dino_features(sample["visual_patch_features"])
        for count in (2, 4, 8):
            with self.subTest(count=count):
                indices = order[:count]
                visual, intrinsics, extrinsics, image_shape, _ = (
                    select_sparse_frustum_inputs_cpu(sample, indices)
                )
                self.assertEqual(visual.device.type, "cpu")
                self.assertEqual(intrinsics.device.type, "cpu")
                self.assertEqual(extrinsics.device.type, "cpu")
                self.assertTrue(
                    torch.equal(visual, source_visual.index_select(0, indices))
                )
                self.assertTrue(
                    torch.equal(
                        intrinsics,
                        sample["intrinsics"].index_select(0, indices),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        extrinsics,
                        sample["extrinsics"].index_select(0, indices),
                    )
                )
                self.assertEqual(image_shape, (518, 518))

    def test_legacy_depth_shape_fallback_is_unchanged(self) -> None:
        sample = _compact_lifting_sample()
        sample.pop("image_size")
        sample["predicted_depth"] = torch.zeros(8, 37, 41)
        _, _, _, image_shape, _ = select_sparse_frustum_inputs_cpu(sample, None)
        self.assertEqual(image_shape, (37, 41))

    def test_invalid_compact_image_size_fails_closed(self) -> None:
        sample = _compact_lifting_sample()
        sample["image_size"] = (518, 0)
        with self.assertRaisesRegex(ValueError, "image_size"):
            validate_strict_cpu_lifting_sample(sample)

    def test_performance_flags_stay_out_of_checkpoint_identity(self) -> None:
        args = training.make_parser().parse_args(
            [
                "--cache_manifest", "compact_slat.json",
                "--lifting_cache_manifest", "compact_lifting.json",
                "--target_decoder_audit", "decoder.json",
                "--native_ss_report", "ss.json",
                "--stock_slat_freeze", "freeze.json",
                "--output_dir", "output",
                "--grad_accum", "1",
                "--num_workers", "2",
                "--prefetch_factor", "2",
                "--persistent_workers",
                "--pin_memory",
                "--torch_num_threads", "2",
                "--torch_num_interop_threads", "1",
            ]
        )
        training.validate_args(args)
        identity = training.checkpoint_args(args)
        for name in (
            "prefetch_factor",
            "persistent_workers",
            "pin_memory",
            "torch_num_threads",
            "torch_num_interop_threads",
        ):
            self.assertNotIn(name, identity)
        self.assertEqual(args.grad_accum, 1)


if __name__ == "__main__":
    unittest.main()
