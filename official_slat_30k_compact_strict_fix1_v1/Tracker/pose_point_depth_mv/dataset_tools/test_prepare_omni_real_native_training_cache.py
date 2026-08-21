from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ar_ss_flow.shared_object_preprocessing import (
    canonical_json_sha256,
    shared_preprocessing_contract,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_native_training_cache import (
    TargetQualityError,
    build_lifting_config,
    fuse_precomputed_dino,
    mesh_target_coords,
    source_geometry,
    training_cache_admission,
    voxelize_object_points,
)


class OmniRealNativeTrainingCacheTest(unittest.TestCase):
    def test_mesh_outside_runtime_grid_is_typed_quality_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            mesh_path = Path(temporary) / "outside.obj"
            mesh_path.write_text(
                "v 1 1 1\n"
                "v 2 1 1\n"
                "v 1 2 1\n"
                "f 1 2 3\n",
                encoding="utf-8",
            )
            with self.assertRaises(TargetQualityError) as raised:
                mesh_target_coords(
                    mesh_path,
                    object_uid="outside",
                    surface_points=128,
                    seed=42,
                    max_surface_outside_ratio=0.05,
                )
        self.assertEqual(raised.exception.code, "mesh_o_outside_runtime_grid")
        self.assertEqual(
            raised.exception.details["surface_outside_ratio"], 1.0
        )

    def test_training_admission_allows_only_typed_quality_rejections(self) -> None:
        admitted = training_cache_admission(
            selected_object_count=470,
            completed_object_count=378,
            hard_failure_count=0,
            quality_rejection_count=92,
            allow_target_quality_rejections=True,
            min_completed_objects=350,
        )
        self.assertTrue(admitted["passed"])
        hard_failure = training_cache_admission(
            selected_object_count=470,
            completed_object_count=377,
            hard_failure_count=1,
            quality_rejection_count=92,
            allow_target_quality_rejections=True,
            min_completed_objects=350,
        )
        self.assertFalse(hard_failure["passed"])

    def test_lifting_contract_is_split_invariant(self) -> None:
        runtime = {
            "selected_view_count": 8,
            "feature_resolution": 518,
            "foreground_margin": 1.1,
            "alpha_threshold": 0.8,
        }
        train = build_lifting_config({**runtime, "source_manifest": "/train.json"})
        dev = build_lifting_config({**runtime, "source_manifest": "/dev.json"})
        self.assertEqual(train, dev)
        self.assertEqual(canonical_json_sha256(train), canonical_json_sha256(dev))
        self.assertEqual(train["selected_view_count"], 8)
        self.assertEqual(train["coordinate_frame"], "input-derived runtime-O")
        self.assertIn("scale removed", train["lifting_extrinsics"])

    def test_voxelize_runtime_points_ignores_outside_tail(self) -> None:
        points = np.asarray(
            [[-0.49, -0.49, -0.49], [0.0, 0.0, 0.0], [0.8, 0.0, 0.0]],
            dtype=np.float32,
        )
        coords = voxelize_object_points(points)
        self.assertEqual(coords.shape, (2, 3))
        self.assertTrue(np.array_equal(coords[0], [0, 0, 0]))
        self.assertTrue(np.array_equal(coords[1], [32, 32, 32]))

    def test_source_geometry_binds_condition_record(self) -> None:
        contract = shared_preprocessing_contract(
            resolution=518, foreground_margin=1.1, alpha_threshold=0.8
        )
        geometry = {
            "contract": contract,
            "contract_hash": canonical_json_sha256(contract),
            "source_sizes_wh": [[640, 480], [640, 480]],
            "source_to_feature_affines": [np.eye(3).tolist()] * 2,
            "crop_boxes_xyxy": [[0, 0, 518, 518]] * 2,
            "foreground_retained_fractions": [1.0, 1.0],
        }
        geometry["geometry_hash"] = canonical_json_sha256(geometry)
        condition = {
            "format": "pose_point_depth_mv.real_input_frontend.v2",
            "shared_image_geometry": geometry,
            "undistortion": [
                {"output_K": np.eye(3).tolist()},
                {"output_K": np.eye(3).tolist()},
            ],
        }
        condition_hash = canonical_json_sha256(condition)
        condition["condition_sha256"] = condition_hash
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "condition.json"
            path.write_text(json.dumps(condition), encoding="utf-8")
            runtime = {
                "category": "box",
                "object_id": "box_001",
                "condition_record": str(path),
                "condition_sha256": condition_hash,
            }
            arrays = {
                "source_to_feature_affine": np.stack([np.eye(3)] * 2).astype(
                    np.float32
                ),
                "K_feature": np.stack([np.eye(3)] * 2).astype(np.float32),
                "selected_source_view_index": np.asarray([2, 7]),
            }
            preprocessing, source_k, affines, view_ids = source_geometry(
                runtime, arrays
            )
        self.assertEqual(preprocessing["shared_geometry"], contract)
        self.assertEqual(preprocessing["shared_geometry_hash"], geometry["geometry_hash"])
        self.assertEqual(tuple(source_k.shape), (2, 3, 3))
        self.assertEqual(tuple(affines.shape), (2, 3, 3))
        self.assertEqual(view_ids.tolist(), [2, 7])

    def test_common_sim3_scale_preserves_precomputed_projection(self) -> None:
        torch.manual_seed(7)
        visual = torch.randn((2, 37 * 37, 3072), dtype=torch.float32)
        coords = np.asarray(
            [[30, 30, 30], [31, 32, 33], [34, 31, 32]], dtype=np.int32
        )
        intrinsics = torch.tensor(
            [[[500.0, 0.0, 259.0], [0.0, 500.0, 259.0], [0.0, 0.0, 1.0]]]
            * 2
        )
        extrinsics = torch.eye(4).repeat(2, 1, 1)
        extrinsics[:, 2, 3] = 2.0
        masks = torch.ones((2, 518, 518), dtype=torch.float32)
        reference, _ = fuse_precomputed_dino(
            visual_patch_features=visual,
            coords=coords,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            masks=masks,
            device=torch.device("cpu"),
        )
        scaled = extrinsics.clone()
        scaled[:, :3, :] *= 7.0
        candidate, _ = fuse_precomputed_dino(
            visual_patch_features=visual,
            coords=coords,
            intrinsics=intrinsics,
            extrinsics=scaled,
            masks=masks,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.allclose(reference, candidate, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
