from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
from PIL import Image

from ReconViaGen.prepare_proobjaverse_30k import (
    AUDIT_FORMAT,
    COMBINED_FORMAT,
    SELECTION_FORMAT,
    audit_combined,
    canonical_sha256,
    link_base,
    read_json,
)


class PrepareProObjaverse30KTest(unittest.TestCase):
    def _payload(self, root: Path, uid: str, shard: str) -> dict:
        render = root / "renders_random_env" / shard / f"{uid}.tar"
        slat = root / "lh-slats" / shard / f"{uid}.npz"
        render.parent.mkdir(parents=True, exist_ok=True)
        slat.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            slat,
            coords=np.asarray([[0, 1, 2], [63, 62, 61]], dtype=np.uint8),
            feats=np.ones((2, 8), dtype=np.float32),
        )
        meta = json.dumps(
            {"intrinsic": np.eye(3).tolist(), "extrinsic": np.eye(4).tolist()}
        ).encode("utf-8")
        image_buffer = io.BytesIO()
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(
            image_buffer, format="WEBP"
        )
        with tarfile.open(render, "w") as archive:
            for name, value in (
                (f"{uid}/000.json", meta),
                (f"{uid}/000.rgba.webp", image_buffer.getvalue()),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
        return {
            "uid": uid,
            "shard": shard,
            "render": {
                "path": f"renders_random_env/{shard}/{uid}.tar",
                "size": render.stat().st_size,
            },
            "slat": {
                "path": f"lh-slats/{shard}/{uid}.npz",
                "size": slat.stat().st_size,
            },
        }

    def _selection(
        self,
        path: Path,
        rows: list[dict],
        *,
        excluded_sha: str | None = None,
    ) -> dict:
        body = {
            "format": SELECTION_FORMAT,
            "repo_id": "Stable-X/ProObjaverse-300K",
            "repo_type": "dataset",
            "revision": "test-revision",
            "selected_pair_count": len(rows),
            "selected": rows,
        }
        if excluded_sha is not None:
            body.update(
                {
                    "excluded_uid_count": 2,
                    "excluded_selection_bindings": [
                        {"selection_sha256": excluded_sha}
                    ],
                }
            )
        payload = {**body, "selection_sha256": canonical_sha256(body)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def test_hardlink_and_combined_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_root = root / "base"
            target_root = root / "combined"
            state = root / "state"
            base_rows = [
                self._payload(base_root, "base-a", "shard-0001"),
                self._payload(base_root, "base-b", "shard-0002"),
            ]
            base_path = root / "base_selection.json"
            base = self._selection(base_path, base_rows)
            link_base(
                argparse.Namespace(
                    source_root=str(base_root),
                    target_root=str(target_root),
                    selection=str(base_path),
                    expected_count=2,
                    report=str(state / "link.json"),
                )
            )
            for row in base_rows:
                source = base_root / row["render"]["path"]
                target = target_root / row["render"]["path"]
                self.assertEqual(source.stat().st_ino, target.stat().st_ino)

            extension_rows = [
                self._payload(target_root, "extra-a", "shard-0007"),
                self._payload(target_root, "extra-b", "shard-0008"),
            ]
            extension_path = root / "extension_selection.json"
            self._selection(
                extension_path,
                extension_rows,
                excluded_sha=base["selection_sha256"],
            )
            output = root / "audit"
            audit_combined(
                argparse.Namespace(
                    data_root=str(target_root),
                    selection=[str(base_path), str(extension_path)],
                    expected_counts="2,2",
                    expected_total=4,
                    schema_samples=4,
                    seed=42,
                    output_dir=str(output),
                )
            )
            combined = read_json(output / "combined_selection_30k.json")
            report = read_json(output / "audit_report.json")
            self.assertEqual(combined["format"], COMBINED_FORMAT)
            self.assertEqual(combined["pair_count"], 4)
            self.assertFalse(
                combined["target_contract"]["trellis2_shape_slat_compatible"]
            )
            self.assertEqual(report["format"], AUDIT_FORMAT)
            self.assertTrue(report["passed"])
            self.assertEqual(report["schema_sample_count"], 4)


if __name__ == "__main__":
    unittest.main()
