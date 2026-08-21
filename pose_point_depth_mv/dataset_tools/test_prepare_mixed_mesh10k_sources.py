from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from pose_point_depth_mv.dataset_tools.prepare_mixed_mesh10k_sources import (
    EXTRACT_REPORT,
    audit_omni_category,
    extract_one_omni_archive,
)


def write_scan_object(
    category: Path,
    name: str,
    *,
    obj: bytes = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
    mtl: bytes = b"newmtl material\n",
    texture: bytes = b"jpeg",
) -> None:
    scan = category / name / "Scan"
    scan.mkdir(parents=True)
    (scan / "Scan.obj").write_bytes(obj)
    (scan / "Scan.mtl").write_bytes(mtl)
    (scan / "Scan.jpg").write_bytes(texture)


class OmniObjectAuditTest(unittest.TestCase):
    def test_individual_invalid_object_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            category = Path(raw) / "watermelon"
            write_scan_object(category, "watermelon_001")
            write_scan_object(category, "watermelon_015", obj=b"")

            accepted, rejected = audit_omni_category(
                category,
                min_obj_bytes=16,
            )

            self.assertEqual(
                [row["object_name"] for row in accepted],
                ["watermelon_001"],
            )
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["object_name"], "watermelon_015")
            self.assertEqual(
                rejected[0]["reason"],
                "missing_or_empty_scan_assets",
            )
            self.assertEqual(
                rejected[0]["assets"],
                ["watermelon_015/Scan/Scan.obj"],
            )

    def test_category_without_valid_objects_remains_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            category = Path(raw) / "broken"
            write_scan_object(category, "broken_001", obj=b"")
            with self.assertRaisesRegex(
                RuntimeError,
                "contains no valid objects",
            ):
                audit_omni_category(category, min_obj_bytes=16)

    def test_archive_extract_records_rejections_and_is_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            category = source / "watermelon"
            write_scan_object(category, "watermelon_001")
            write_scan_object(category, "watermelon_015", obj=b"")
            archive = root / "watermelon.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for path in sorted(category.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(category)
                    payload = path.read_bytes()
                    info = tarfile.TarInfo(str(relative))
                    info.size = len(payload)
                    handle.addfile(info, io.BytesIO(payload))

            task = {
                "archive": str(archive),
                "extract_root": str(root / "extracted"),
                "category": "watermelon",
                "min_obj_bytes": 16,
            }
            first = extract_one_omni_archive(task)
            second = extract_one_omni_archive(task)

            self.assertEqual(first["object_count"], 1)
            self.assertEqual(first["rejected_object_count"], 1)
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(
                second["rejected_objects_sha256"],
                first["rejected_objects_sha256"],
            )
            self.assertTrue(
                (
                    root
                    / "extracted"
                    / "categories"
                    / "watermelon"
                    / EXTRACT_REPORT
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
