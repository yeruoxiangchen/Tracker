#!/usr/bin/env python3

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pose_point_depth_mv.build_local_lh_slats import (
    NATIVE_OBJAVERSE_PRIMARY_TARGET_POLICY,
    NATIVE_OBJAVERSE_TARGET_CONTRACT,
    camera_corruption_partner_indices,
    grid_sample_with_validity,
    index_render_samples,
    object_inputs,
    project_points_to_grid,
    projection_gate_failures,
    sha256_file,
    sparse_coordinate_iou,
    voxel_centers_render_space,
)


class LocalLhSLatGeometryTest(unittest.TestCase):
    def _object_row(
        self, root: Path, uid: str, *, repaired: bool = True
    ) -> dict:
        coords = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        z = np.zeros((8, 16, 16, 16), dtype=np.float16)
        latent = root / f"{uid}.npz"
        payload = {
            "target_coords": coords,
            "z": z,
            "pixal3d_rotation": np.eye(3, dtype=np.float32),
            "source_glb": np.array(str(root / "mesh.glb")),
        }
        if repaired:
            payload.update(
                {
                    "repair_format": np.array("object_level_ss_repair.v1"),
                    "repair_target_mode": np.array("decoder_projected"),
                }
            )
        np.savez_compressed(latent, **payload)
        import hashlib

        return {
            "uid": uid,
            "_latent_root": str(root),
            "ss_latent": str(latent),
            "ss_repair_target_sha256": hashlib.sha256(coords.tobytes()).hexdigest(),
            "ss_repair_z_sha256": hashlib.sha256(z.tobytes()).hexdigest(),
            "_image_root": str(root),
            "_mask_root": str(root),
            "_image_size": 512,
            "_camera_forward_sign": 1.0,
            "frames": [],
        }

    def test_object_inputs_requires_object_level_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mesh.glb").write_bytes(b"mesh")
            row = self._object_row(root, "sample_seq000", repaired=False)
            with self.assertRaisesRegex(RuntimeError, "lacks object_level_ss_repair"):
                object_inputs("sample", [row], max_views=16)

    def test_object_inputs_accepts_hash_bound_repaired_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mesh.glb").write_bytes(b"mesh")
            rows = [
                self._object_row(root, "sample_seq000"),
                self._object_row(root, "sample_seq001"),
            ]
            resolved = object_inputs("sample", rows, max_views=16)
            self.assertEqual(resolved["coords"].shape, (2, 3))
            self.assertEqual(len(resolved["source_latents"]), 2)

    def test_object_inputs_rejects_stale_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mesh.glb").write_bytes(b"mesh")
            row = self._object_row(root, "sample_seq000")
            row["ss_repair_target_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "target hash mismatch"):
                object_inputs("sample", [row], max_views=16)

    def _native_object_row(
        self,
        root: Path,
        uid: str,
        *,
        coords: np.ndarray | None = None,
        z_value: float = 0.0,
    ) -> dict:
        object_uid = uid.rsplit("_seq", 1)[0]
        coords = np.asarray(
            coords if coords is not None else [[1, 2, 3], [4, 5, 6]],
            dtype=np.int32,
        )
        z = np.full((8, 16, 16, 16), z_value, dtype=np.float16)
        source_glb = root / "mesh.glb"
        latent = root / f"{uid}.npz"
        np.savez_compressed(
            latent,
            target_coords=coords,
            z=z,
            pixal3d_rotation=np.eye(3, dtype=np.float32),
            source_glb=np.array(str(source_glb)),
            uid=np.array(uid),
            object_uid=np.array(object_uid),
            renderer=np.array("blender"),
        )
        return {
            "uid": uid,
            "object_uid": object_uid,
            "_manifest_format": "pixal3d_multiview.objaverse_sparse.v1",
            "_manifest_renderer": "blender",
            "_latent_root": str(root),
            "ss_latent": str(latent),
            "source_glb": str(source_glb),
            "num_voxels": len(coords),
            "quality_flags": {"renderer": "blender"},
            "_image_root": str(root),
            "_mask_root": str(root),
            "_image_size": 512,
            "_camera_forward_sign": 1.0,
            "frames": [],
        }

    def test_object_inputs_accepts_native_objaverse_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mesh.glb").write_bytes(b"mesh")
            rows = [
                self._native_object_row(root, "sample_seq000"),
                self._native_object_row(root, "sample_seq001"),
            ]
            resolved = object_inputs(
                "sample",
                rows,
                max_views=16,
                target_contract=NATIVE_OBJAVERSE_TARGET_CONTRACT,
            )
            self.assertEqual(resolved["coords"].shape, (2, 3))
            self.assertEqual(
                {row["target_contract"] for row in resolved["source_latents"]},
                {NATIVE_OBJAVERSE_TARGET_CONTRACT},
            )
            self.assertTrue(
                all(not row["repair_format"] for row in resolved["source_latents"])
            )
            self.assertEqual(
                resolved["target_selection"]["policy"],
                NATIVE_OBJAVERSE_PRIMARY_TARGET_POLICY,
            )
            self.assertEqual(
                resolved["target_selection"]["primary_uid"], "sample_seq000"
            )

    def test_native_contract_uses_uid_primary_and_allows_audited_sequence_targets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mesh.glb").write_bytes(b"mesh")
            primary = np.asarray(
                [[1, 2, 3], [2, 2, 3], [3, 2, 3], [4, 2, 3]], dtype=np.int32
            )
            secondary = np.asarray(
                [[1, 2, 3], [2, 2, 3], [3, 2, 3], [5, 2, 3]], dtype=np.int32
            )
            rows = [
                self._native_object_row(
                    root, "sample_seq001", coords=secondary, z_value=1.0
                ),
                self._native_object_row(root, "sample_seq000", coords=primary),
            ]
            resolved = object_inputs(
                "sample",
                rows,
                max_views=16,
                target_contract=NATIVE_OBJAVERSE_TARGET_CONTRACT,
                min_native_sequence_target_iou=0.50,
            )
            np.testing.assert_array_equal(resolved["coords"], primary)
            self.assertEqual(resolved["target_selection"]["primary_uid"], "sample_seq000")
            self.assertEqual(
                resolved["target_selection"]["z_policy"],
                "validated_per_sequence_not_required_to_match",
            )
            self.assertAlmostEqual(
                resolved["target_selection"]["observed_min_target_iou"], 0.6
            )

    def test_native_contract_rejects_sequence_target_below_iou_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mesh.glb").write_bytes(b"mesh")
            rows = [
                self._native_object_row(
                    root,
                    "sample_seq000",
                    coords=np.asarray([[1, 1, 1], [2, 2, 2]], dtype=np.int32),
                ),
                self._native_object_row(
                    root,
                    "sample_seq001",
                    coords=np.asarray([[30, 30, 30], [31, 31, 31]], dtype=np.int32),
                ),
            ]
            with self.assertRaisesRegex(RuntimeError, "native sequence target IoU"):
                object_inputs(
                    "sample",
                    rows,
                    max_views=16,
                    target_contract=NATIVE_OBJAVERSE_TARGET_CONTRACT,
                    min_native_sequence_target_iou=0.75,
                )

    def test_sparse_coordinate_iou_uses_unique_voxel_sets(self) -> None:
        left = np.asarray([[1, 2, 3], [1, 2, 3], [4, 5, 6]], dtype=np.int32)
        right = np.asarray([[1, 2, 3], [7, 8, 9]], dtype=np.int32)
        self.assertAlmostEqual(sparse_coordinate_iou(left, right), 1.0 / 3.0)

    def test_native_objaverse_contract_rejects_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mesh.glb").write_bytes(b"mesh")
            row = self._native_object_row(root, "sample_seq000")
            row["num_voxels"] += 1
            with self.assertRaisesRegex(RuntimeError, "native identity differs"):
                object_inputs(
                    "sample",
                    [row],
                    max_views=16,
                    target_contract=NATIVE_OBJAVERSE_TARGET_CONTRACT,
                )

    def test_frozen_manifest_recovers_protocol_from_bound_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(
                json.dumps({"voxel_resolution": 64, "image_size": 512}),
                encoding="utf-8",
            )
            frozen = root / "frozen.json"
            frozen.write_text(
                json.dumps(
                    {
                        "format": "pixal3d_multiview.objaverse_sparse.v1",
                        "extrinsics_type": "c2w",
                        "image_root": "/",
                        "mask_root": "/",
                        "latent_root": "/",
                        "samples": [
                            {
                                "uid": "sample_seq000",
                                "object_uid": "sample",
                                "source_manifest": str(source),
                                "source_manifest_sha256": sha256_file(source),
                                "frames": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            indexed, bindings = index_render_samples([str(frozen)])
            self.assertEqual(indexed["sample"][0]["_image_size"], 512)
            self.assertEqual(
                bindings[0]["protocol"],
                {"voxel_resolution": 64, "image_size": 512},
            )
            self.assertEqual(
                bindings[0]["protocol_audit"]["metadata_sources"],
                {
                    "voxel_resolution": "sample_source_manifests",
                    "image_size": "sample_source_manifests",
                },
            )

    def test_frozen_manifest_rejects_changed_bound_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(
                json.dumps({"voxel_resolution": 64, "image_size": 512}),
                encoding="utf-8",
            )
            expected_sha256 = sha256_file(source)
            source.write_text(
                json.dumps({"voxel_resolution": 32, "image_size": 512}),
                encoding="utf-8",
            )
            frozen = root / "frozen.json"
            frozen.write_text(
                json.dumps(
                    {
                        "format": "pixal3d_multiview.objaverse_sparse.v1",
                        "extrinsics_type": "c2w",
                        "image_root": "/",
                        "mask_root": "/",
                        "latent_root": "/",
                        "samples": [
                            {
                                "uid": "sample_seq000",
                                "object_uid": "sample",
                                "source_manifest": str(source),
                                "source_manifest_sha256": expected_sha256,
                                "frames": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "source manifest changed"):
                index_render_samples([str(frozen)])

    def test_voxel_centers_and_rotation(self) -> None:
        coords = np.asarray([[0, 31, 63], [63, 32, 0]], dtype=np.int32)
        rotation = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        points = voxel_centers_render_space(coords, rotation)
        base = (coords.astype(np.float32) + 0.5) / 64.0 - 0.5
        np.testing.assert_array_equal(points, base @ rotation.T)

    def test_center_projection_align_corners_false(self) -> None:
        points = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        intrinsics = torch.tensor(
            [[[100.0, 0.0, 255.5], [0.0, 100.0, 255.5], [0.0, 0.0, 1.0]]]
        )
        c2w = torch.eye(4)[None]
        grid, valid = project_points_to_grid(
            points,
            intrinsics,
            c2w,
            source_height=512,
            source_width=512,
            target_height=518,
            target_width=518,
        )
        self.assertTrue(bool(valid.item()))
        self.assertTrue(torch.equal(grid, torch.zeros_like(grid)))

    def test_behind_camera_projection_is_forced_to_zero(self) -> None:
        points = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=torch.float32
        )
        intrinsics = torch.tensor(
            [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]
        )
        grid, valid = project_points_to_grid(
            points,
            intrinsics,
            torch.eye(4)[None],
            source_height=2,
            source_width=2,
            target_height=2,
            target_width=2,
        )
        # Both numerical grids land at the image centre.  Only the explicit
        # depth validity mask can prevent the behind-camera sample leaking.
        self.assertTrue(torch.equal(grid[0, 0], grid[0, 1]))
        self.assertEqual(valid.tolist(), [[True, False]])
        feature_map = torch.full((1, 1, 2, 2), 7.0)
        sampled = grid_sample_with_validity(feature_map, grid, valid)
        self.assertEqual(sampled[:, :, 0].tolist(), [[7.0, 0.0]])

    def test_left_right_camera_sentinel_is_asymmetric(self) -> None:
        points = torch.tensor([[0.25, 0.0, 1.0]], dtype=torch.float32)
        intrinsics = torch.tensor(
            [
                [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]],
                [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]],
            ]
        )
        c2w = torch.eye(4).repeat(2, 1, 1)
        c2w[0, 0, 3] = -0.25
        c2w[1, 0, 3] = 0.25
        grid, valid = project_points_to_grid(
            points,
            intrinsics,
            c2w,
            source_height=4,
            source_width=4,
            target_height=4,
            target_width=4,
        )
        self.assertTrue(bool(valid.all()))
        self.assertGreater(float(grid[0, 0, 0]), float(grid[1, 0, 0]))

    def test_projection_gate_includes_camera_corruption(self) -> None:
        stats = {
            "visible_view_fraction_mean": 0.8,
            "mask_support_view_fraction_mean": 0.6,
            "mask_support_ge2_ratio": 0.9,
            "mask_support_ge4_ratio": 0.7,
            "zero_mask_support_ratio": 0.0,
            "camera_corruption_mask_support_drop": 0.0,
        }
        failures = projection_gate_failures(
            stats,
            min_visible_view_fraction_mean=0.25,
            min_mask_support_view_fraction_mean=0.10,
            min_mask_support_ge2_ratio=0.50,
            min_mask_support_ge4_ratio=0.20,
            max_zero_mask_support_ratio=0.25,
            min_camera_corruption_mask_support_drop=0.02,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("camera_corruption_mask_support_drop", failures[0])

    def test_camera_corruption_stays_within_sequence(self) -> None:
        frames = [
            {"uid": "seq0", "frame_index": index} for index in range(8)
        ] + [{"uid": "seq1", "frame_index": index} for index in range(8)]
        partners = camera_corruption_partner_indices(frames)
        self.assertEqual(partners[:8], [4, 5, 6, 7, 0, 1, 2, 3])
        self.assertEqual(partners[8:], [12, 13, 14, 15, 8, 9, 10, 11])


if __name__ == "__main__":
    unittest.main()
