from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pose_point_depth_mv.dataset_tools.build_existing_mixed1k_dataset import (
    allocation_by_largest_remainder,
    file_sha256,
    load_objaverse,
    load_omni,
    select_preview,
    validate_complete_marker,
    write_dataset,
    write_preview,
)


def write_manifest(root: Path, source: str, object_ids: list[str]) -> Path:
    images = root / "images"
    masks = root / "masks"
    latents = root / "latents"
    meshes = root / "meshes"
    for directory in (images, masks, latents, meshes):
        directory.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, object_uid in enumerate(object_ids):
        uid = f"{object_uid}_seq000"
        image = images / f"{uid}.png"
        mask = masks / f"{uid}.png"
        latent = latents / f"{uid}.npz"
        mesh = meshes / f"{object_uid}.obj"
        Image.new("RGB", (8, 8), (index, 10, 20)).save(image)
        Image.new("L", (8, 8), 255).save(mask)
        latent.write_bytes(b"npz")
        mesh.write_text("v 0 0 0\n", encoding="utf-8")
        samples.append(
            {
                "uid": uid,
                "object_uid": object_uid,
                "sequence_idx": 0,
                "ss_latent": latent.name,
                "source_glb": str(mesh),
                "frames": [
                    {
                        "image": image.name,
                        "mask": mask.name,
                        "intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    }
                ],
            }
        )
    path = root / f"{source}.json"
    path.write_text(
        json.dumps(
            {
                "samples": samples,
                "image_root": str(images),
                "mask_root": str(masks),
                "latent_root": str(latents),
            }
        ),
        encoding="utf-8",
    )
    return path


def audit_row(root: Path, object_uid: str, tier: str) -> dict:
    return {
        "object_uid": object_uid,
        "source_glb": str(root / "meshes" / f"{object_uid}.obj"),
        "final_tier": tier,
        "mesh_audit": {"mesh_valid": True},
    }


def reviewed_audit_row(root: Path, object_uid: str, tier: str) -> dict:
    return {
        **audit_row(root, object_uid, tier),
        "human_reviewed": True,
        "semantic_subject_label": "keep_single_subject",
    }


class ExistingMixed1kTests(unittest.TestCase):
    def test_largest_remainder_is_exact(self) -> None:
        allocation = allocation_by_largest_remainder(
            {"objaverse_A": 183, "objaverse_B": 531, "objaverse_C": 183, "omni": 123},
            100,
        )
        self.assertEqual(sum(allocation.values()), 100)
        self.assertEqual(
            allocation,
            {"objaverse_A": 18, "objaverse_B": 52, "objaverse_C": 18, "omni": 12},
        )

    def test_mixed_paths_and_object_unique_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj_path = write_manifest(root / "obj", "obj", ["oa", "ob", "oc"])
            audit_path = root / "objects.json"
            audit_path.write_text(
                json.dumps(
                    [
                        audit_row(root / "obj", "oa", "A"),
                        audit_row(root / "obj", "ob", "B"),
                        audit_row(root / "obj", "oc", "C"),
                    ]
                ),
                encoding="utf-8",
            )
            omni_path = write_manifest(root / "omni", "omni", ["omni_x", "omni_y"])
            obj_samples, obj_objects = load_objaverse(obj_path, audit_path)
            omni_samples, omni_objects = load_omni([omni_path])
            samples = obj_samples + omni_samples
            objects = obj_objects + omni_objects
            self.assertTrue(all(Path(row["ss_latent"]).is_absolute() for row in samples))
            self.assertTrue(
                all(Path(frame["image"]).is_absolute() for row in samples for frame in row["frames"])
            )
            selected, allocation = select_preview(objects, samples, count=5, seed=7)
            self.assertEqual(len(selected), 5)
            self.assertEqual(len({row["object_uid"] for row in selected}), 5)
            self.assertEqual(sum(allocation.values()), 5)

    def test_initial_finetune_rejects_unreviewed_objaverse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj_path = write_manifest(root / "obj", "obj", ["oa"])
            audit_path = root / "objects.json"
            audit_path.write_text(
                json.dumps([audit_row(root / "obj", "oa", "B")]),
                encoding="utf-8",
            )
            obj_samples, obj_objects = load_objaverse(obj_path, audit_path)
            with self.assertRaisesRegex(ValueError, "human-reviewed"):
                write_dataset(
                    root / "dataset",
                    samples=obj_samples,
                    objects=obj_objects,
                    preview_manifest={
                        "html": str(root / "preview" / "index.html"),
                        "preview_sha256": "unused",
                        "review_status": "pending",
                    },
                    source_bindings=[],
                    seed=7,
                    training_mode="initial_finetune",
                )

    def test_reviewed_initial_finetune_is_training_ready_but_nonformal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obj_path = write_manifest(root / "obj", "obj", ["oa"])
            omni_path = write_manifest(root / "omni", "omni", ["omni_x"])
            audit_path = root / "objects.json"
            audit_path.write_text(
                json.dumps([reviewed_audit_row(root / "obj", "oa", "C")]),
                encoding="utf-8",
            )
            obj_samples, obj_objects = load_objaverse(obj_path, audit_path)
            omni_samples, omni_objects = load_omni([omni_path])
            samples = [*obj_samples, *omni_samples]
            objects = [*obj_objects, *omni_objects]
            preview_rows, allocation = select_preview(
                objects, samples, count=2, seed=7
            )
            preview = write_preview(
                root / "preview",
                preview_rows,
                seed=7,
                allocation=allocation,
                review_status="objaverse_reviewed",
            )
            report = write_dataset(
                root / "dataset",
                samples=samples,
                objects=objects,
                preview_manifest=preview,
                source_bindings=[],
                seed=7,
                training_mode="initial_finetune",
            )
            manifest = json.loads(
                (root / "dataset" / "train.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(report["training_ready"])
            self.assertTrue(manifest["training_ready"])
            self.assertFalse(manifest["formal"])
            self.assertEqual(
                manifest["review_status"], "objaverse_reviewed"
            )
            self.assertEqual(
                manifest["admission_policy"]["fresh_objaverse"], "not included"
            )

    def test_complete_marker_binds_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_manifest(root, "omni", ["omni_x"])
            marker = root / "_WORKER_COMPLETE.json"
            marker.write_text(
                json.dumps(
                    {
                        "schema": (
                            "tracker.mixed_multiview_render_shard_complete.v1"
                        ),
                        "render_manifest_sha256": file_sha256(manifest),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(validate_complete_marker(manifest), marker)
            marker.write_text(
                json.dumps(
                    {
                        "schema": (
                            "tracker.mixed_multiview_render_shard_complete.v1"
                        ),
                        "render_manifest_sha256": "wrong",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_complete_marker(manifest)


if __name__ == "__main__":
    unittest.main()
