from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pose_point_depth_mv.dataset_tools.freeze_reviewed_mixed1k_dataset import (
    MARKER_FORMAT,
    file_sha256,
    freeze,
)


def write_manifest(
    root: Path,
    source: str,
    object_ids: list[str],
    *,
    source_meshes: dict[str, Path] | None = None,
) -> Path:
    images = root / "images"
    masks = root / "masks"
    latents = root / "latents"
    meshes = root / "meshes"
    for directory in (images, masks, latents, meshes):
        directory.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, uid in enumerate(object_ids):
        image = images / f"{uid}.png"
        mask = masks / f"{uid}.png"
        latent = latents / f"{uid}.npz"
        mesh = (source_meshes or {}).get(uid, meshes / f"{uid}.obj")
        mesh.parent.mkdir(parents=True, exist_ok=True)
        if not mesh.exists():
            mesh.write_text("v 0 0 0\n", encoding="utf-8")
        Image.new("RGB", (4, 4), (index, 20, 30)).save(image)
        Image.new("L", (4, 4), 255).save(mask)
        latent.write_bytes(b"latent")
        samples.append(
            {
                "uid": f"{uid}_seq000",
                "object_uid": uid,
                "source_glb": str(mesh.resolve()),
                "ss_latent": latent.name,
                "frames": [{"image": image.name, "mask": mask.name}],
            }
        )
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "format": "test.render.v1",
                "extrinsics_type": "c2w",
                "camera_forward_sign": 1.0,
                "coordinate_frame": "normalized_object",
                "canonical_latent_frame": "TRELLIS_64",
                "num_views": 1,
                "images_are_masked": True,
                "image_size": 4,
                "voxel_resolution": 64,
                "image_root": str(images),
                "mask_root": str(masks),
                "latent_root": str(latents),
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return path


def write_render_root(root: Path, source: str, object_ids: list[str]) -> Path:
    shard = root / source / "shard_000"
    manifest = write_manifest(shard, source, object_ids)
    (shard / "_WORKER_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": MARKER_FORMAT,
                "source": source,
                "shard_index": 0,
                "render_manifest_sha256": file_sha256(manifest),
            }
        ),
        encoding="utf-8",
    )
    return root


def write_review(path: Path, rows: list[tuple[str, Path]], *, reviewed: bool = True) -> Path:
    path.write_text(
        json.dumps(
            [
                {
                    "review_id": f"R{index:03d}",
                    "object_uid": uid,
                    "source_glb": str(mesh.resolve()),
                    "source_group": "test_objaverse",
                    "human_reviewed": reviewed,
                    "semantic_subject_label": "keep_single_subject",
                    "semantic_reviewer": "unit_test",
                }
                for index, (uid, mesh) in enumerate(rows)
            ]
        ),
        encoding="utf-8",
    )
    return path


class ReviewedMixed1kFreezeTests(unittest.TestCase):
    def make_case(self, root: Path) -> argparse.Namespace:
        legacy_ids = [f"legacy_{index}" for index in range(4)]
        legacy = write_manifest(root / "legacy", "objaverse", legacy_ids)
        fresh_root = write_render_root(
            root / "fresh_render", "objaverse", ["fresh_0", "fresh_1"]
        )
        omni_root = write_render_root(
            root / "omni_render", "omni", ["omni_0", "omni_1"]
        )
        payloads = [
            json.loads(legacy.read_text(encoding="utf-8")),
            json.loads(
                (fresh_root / "objaverse/shard_000/manifest.json").read_text(
                    encoding="utf-8"
                )
            ),
        ]
        reviewed_rows = []
        for payload in payloads:
            reviewed_rows.extend(
                (sample["object_uid"], Path(sample["source_glb"]))
                for sample in payload["samples"]
            )
        review = write_review(root / "review.json", reviewed_rows)
        excluded = write_manifest(root / "excluded", "objaverse", ["old_holdout"])
        return argparse.Namespace(
            legacy_objaverse_manifest=str(legacy),
            objaverse_review=[str(review)],
            objaverse_render_root=[str(fresh_root)],
            omni_render_root=[str(omni_root)],
            exclude_manifest=[str(excluded)],
            expected_objaverse_shards_per_root=1,
            expected_omni_shards_per_root=1,
            val_objects=2,
            test_objects=2,
            min_objects=8,
            max_objects=8,
            seed=7,
            output_dir=str(root / "frozen"),
        )

    def test_freeze_writes_object_and_source_disjoint_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_case(root)
            report = freeze(args)
            self.assertTrue(report["passed"])
            self.assertTrue(all(report["hard_guards"].values()))
            self.assertEqual(
                report["summary"]["split_object_counts"],
                {"train": 4, "val": 2, "test": 2},
            )
            split_objects = {}
            for split in ("train", "val", "test"):
                payload = json.loads(
                    (root / "frozen" / f"{split}.json").read_text(encoding="utf-8")
                )
                self.assertEqual(payload["image_size"], 4)
                self.assertEqual(payload["voxel_resolution"], 64)
                split_objects[split] = {
                    row["object_uid"] for row in payload["object_records"]
                }
            self.assertFalse(split_objects["train"] & split_objects["val"])
            self.assertFalse(split_objects["train"] & split_objects["test"])

    def test_complete_output_reuse_rebinds_current_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_case(root)
            first = freeze(args)
            second = freeze(args)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            review_path = Path(args.objaverse_review[0])
            rows = json.loads(review_path.read_text(encoding="utf-8"))
            rows[0]["semantic_reviewer"] = "changed_after_freeze"
            review_path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source bindings differ"):
                freeze(args)

    def test_unreviewed_objaverse_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_case(root)
            review_path = Path(args.objaverse_review[0])
            rows = json.loads(review_path.read_text(encoding="utf-8"))
            rows[0]["human_reviewed"] = False
            review_path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not human-reviewed"):
                freeze(args)

    def test_missing_completion_marker_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_case(root)
            marker = (
                Path(args.objaverse_render_root[0])
                / "objaverse/shard_000/_WORKER_COMPLETE.json"
            )
            marker.unlink()
            with self.assertRaisesRegex(ValueError, "completed shard mismatch"):
                freeze(args)

    def test_old_holdout_source_cannot_reenter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_case(root)
            excluded = json.loads(Path(args.exclude_manifest[0]).read_text(encoding="utf-8"))
            review = json.loads(Path(args.objaverse_review[0]).read_text(encoding="utf-8"))
            excluded["samples"][0]["source_glb"] = review[0]["source_glb"]
            Path(args.exclude_manifest[0]).write_text(json.dumps(excluded), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "old val/holdout"):
                freeze(args)

    def test_duplicate_reviewed_source_mesh_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_case(root)
            review_path = Path(args.objaverse_review[0])
            rows = json.loads(review_path.read_text(encoding="utf-8"))
            rows[1]["source_glb"] = rows[0]["source_glb"]
            review_path.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate reviewed source mesh"):
                freeze(args)

    def test_objaverse_omni_protocol_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_case(root)
            manifest = (
                Path(args.omni_render_root[0]) / "omni/shard_000/manifest.json"
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["image_size"] = 8
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            marker_path = manifest.parent / "_WORKER_COMPLETE.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["render_manifest_sha256"] = file_sha256(manifest)
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "Objaverse/Omni render metadata mismatch"
            ):
                freeze(args)


if __name__ == "__main__":
    unittest.main()
