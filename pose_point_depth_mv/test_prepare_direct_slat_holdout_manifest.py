from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.prepare_direct_slat_holdout_manifest import (
    collect_seen_identities,
    render_rows,
    select_rendered_holdout,
    select_unseen_candidates,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DirectSLatHoldoutManifestTests(unittest.TestCase):
    def make_assets(self, root: Path, count: int) -> list[Path]:
        output = []
        for index in range(count):
            path = root / f"asset_{index}.glb"
            path.write_bytes(f"asset-{index}".encode("ascii"))
            output.append(path)
        return output

    def test_candidate_selection_excludes_seen_identity_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = self.make_assets(root, 5)
            seen_path = root / "seen.json"
            seen_path.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "object_uid": "seen",
                                "source_glb": str(assets[0]),
                                "source_glb_sha256": file_sha256(assets[0]),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            seen = collect_seen_identities([seen_path])
            rows = [
                {"object_uid": "seen", "source_glb": str(assets[1])},
                {"object_uid": "alias", "source_glb": str(assets[0])},
            ] + [
                {
                    "object_uid": f"new-{index}",
                    "source_glb": str(path),
                }
                for index, path in enumerate(assets[2:])
            ]
            selected = select_unseen_candidates(
                rows, seen=seen, count=2, seed=17
            )
            self.assertEqual(len(selected), 2)
            self.assertNotIn("seen", {row["object_uid"] for row in selected})
            self.assertNotIn(
                file_sha256(assets[0]),
                {row["source_glb_sha256"] for row in selected},
            )

    def test_render_selection_is_exact_and_one_sequence_per_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = self.make_assets(root, 4)
            image_root = root / "images"
            mask_root = root / "masks"
            latent_root = root / "latents"
            for path in (image_root, mask_root, latent_root):
                path.mkdir()
            samples = []
            for object_index in range(4):
                for sequence_index in range(2):
                    uid = f"obj-{object_index}_seq{sequence_index:03d}"
                    image = image_root / f"{uid}.png"
                    mask = mask_root / f"{uid}.png"
                    latent = latent_root / f"{uid}.npz"
                    image.write_bytes(b"image")
                    mask.write_bytes(b"mask")
                    latent.write_bytes(b"latent")
                    samples.append(
                        {
                            "uid": uid,
                            "object_uid": f"obj-{object_index}",
                            "source_glb": str(assets[object_index]),
                            "ss_latent": latent.name,
                            "frames": [
                                {
                                    "image": image.name,
                                    "mask": mask.name,
                                }
                            ],
                        }
                    )
            manifest_path = root / "render.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "format": "pixal3d_multiview.objaverse_sparse.v1",
                        "image_root": str(image_root),
                        "mask_root": str(mask_root),
                        "latent_root": str(latent_root),
                        "extrinsics_type": "c2w",
                        "camera_forward_sign": 1.0,
                        "coordinate_frame": "frame",
                        "canonical_latent_frame": "latent",
                        "image_size": 512,
                        "samples": samples,
                    }
                ),
                encoding="utf-8",
            )
            metadata, rows, _ = render_rows([manifest_path])
            self.assertEqual(metadata["image_size"], 512)
            seen = {
                "object_uids": {"obj-0"},
                "source_glb_paths": {str(assets[0].resolve())},
                "source_glb_sha256": {file_sha256(assets[0])},
            }
            selected = select_rendered_holdout(
                rows,
                seen=seen,
                count=2,
                seed=23,
                eligible_object_uids={"obj-2", "obj-3"},
            )
            self.assertEqual(len(selected), 2)
            self.assertEqual(
                len({row["object_uid"] for row in selected}), 2
            )
            self.assertTrue(
                all(row["uid"].endswith("seq000") for row in selected)
            )
            self.assertEqual(
                {row["object_uid"] for row in selected},
                {"obj-2", "obj-3"},
            )
            self.assertTrue(
                all("_manifest_path" not in row for row in selected)
            )


if __name__ == "__main__":
    unittest.main()
