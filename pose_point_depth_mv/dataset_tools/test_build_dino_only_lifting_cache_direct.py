from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from reconvggt_ar_adapter_a.pointpose_ss_condition import (
    PHYSICAL_FEATURE_NAMES,
    feature_schema_hash,
)
from pose_point_depth_mv.dataset_tools.build_dino_only_lifting_cache_direct import (
    DirectPointPoseDataset,
    build_sample,
)


class FakeDino:
    default_image_resolution = 518

    def encode_image(self, images):
        views = len(images)
        # Five prefix tokens plus a square 2x2 patch grid.
        values = torch.arange(views * 9 * 1024, dtype=torch.float32)
        return values.reshape(views, 9, 1024) / float(values.numel())


class DirectDinoCacheTest(unittest.TestCase):
    def test_pointpose_to_direct_dino_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            mask_root = root / "masks"
            latent_root = root / "latents"
            physical_root = root / "pointpose" / "physical"
            for directory in (image_root, mask_root, latent_root, physical_root):
                directory.mkdir(parents=True)
            frames = []
            for index in range(2):
                rgb = np.zeros((64, 64, 3), dtype=np.uint8)
                rgb[16:48, 18:46] = (80 + index * 20, 140, 200)
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[16:48, 18:46] = 255
                Image.fromarray(rgb).save(image_root / f"v{index}.png")
                Image.fromarray(mask).save(mask_root / f"v{index}.png")
                frames.append(
                    {
                        "image": f"v{index}.png",
                        "mask": f"v{index}.png",
                        "intrinsic": [
                            [80.0, 0.0, 31.5],
                            [0.0, 80.0, 31.5],
                            [0.0, 0.0, 1.0],
                        ],
                        "extrinsic": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, -2.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                )
            latent = latent_root / "sample.npz"
            np.savez_compressed(
                latent,
                z=np.zeros((8, 16, 16, 16), dtype=np.float16),
                target_coords=np.asarray([[32, 32, 32]], dtype=np.int32),
            )
            physical = physical_root / "sample.npz"
            np.savez_compressed(
                physical,
                physical_grid=np.zeros(
                    (len(PHYSICAL_FEATURE_NAMES), 16, 16, 16), dtype=np.float16
                ),
                prior_coords=np.asarray([[32, 32, 32]], dtype=np.int32),
                prior_conf=np.asarray([1.0], dtype=np.float16),
                view_ids=np.asarray([0, 1], dtype=np.int32),
            )
            source_manifest = root / "source.json"
            source_manifest.write_text(
                json.dumps(
                    {
                        "format": "pixal3d_multiview.objaverse_sparse.v1",
                        "image_root": str(image_root),
                        "mask_root": str(mask_root),
                        "latent_root": str(latent_root),
                        "extrinsics_type": "c2w",
                        "camera_forward_sign": 1.0,
                        "samples": [
                            {
                                "uid": "sample",
                                "object_uid": "object",
                                "ss_latent": "sample.npz",
                                "frames": frames,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            prior_manifest = root / "prior.json"
            prior_manifest.write_text(
                json.dumps(
                    {
                        "format": "trellis_point_prior_mv_v1",
                        "grid_transform": "pixal3d_rotation",
                        "samples": [{"uid": "sample", "grid_transform": "pixal3d_rotation"}],
                    }
                ),
                encoding="utf-8",
            )
            pointpose_root = root / "pointpose"
            pointpose_manifest = pointpose_root / "manifest.json"
            pointpose_manifest.write_text(
                json.dumps(
                    {
                        "format": "reconvggt.pointpose_ss_cache.v1",
                        "output_dir": str(pointpose_root),
                        "source_manifest": str(source_manifest),
                        "prior_manifest": str(prior_manifest),
                        "feature_names": list(PHYSICAL_FEATURE_NAMES),
                        "feature_schema_hash": feature_schema_hash(),
                        "samples": [
                            {
                                "uid": "sample",
                                "object_uid": "object",
                                "physical_grid": "physical/sample.npz",
                                "ss_latent": str(latent),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            dataset = DirectPointPoseDataset(pointpose_manifest)
            batch = dataset[0]
            payload = build_sample(
                FakeDino(),
                batch,
                source_manifest=pointpose_manifest,
                source_manifest_sha256="source-sha",
                input_binding={"binding_sha256": "input-sha"},
                config_hash="config-sha",
                image_resolution=518,
                foreground_margin=1.10,
                alpha_threshold=0.80,
                ss_context_tokens=8,
                save_correct_geometry=True,
            )
            self.assertEqual(tuple(payload["visual_patch_features"].shape), (2, 4, 1024))
            self.assertEqual(tuple(payload["stock_condition"].shape), (1, 8, 1024))
            self.assertEqual(len(payload["slat_condition"]["cond"]), 2)
            self.assertFalse(payload["dino_only_direct_build"]["vggt_model_loaded"])
            self.assertFalse(payload["dino_only_direct_build"]["vggt_model_executed"])
            self.assertEqual(
                tuple(payload["correct_geometry"]["patch_grid"].shape),
                (2, 4096, 2),
            )
            expected_k = torch.matmul(
                payload["source_to_feature_affines"].float(),
                payload["source_intrinsics"].float(),
            )
            self.assertTrue(torch.allclose(payload["intrinsics"].float(), expected_k))


if __name__ == "__main__":
    unittest.main()
