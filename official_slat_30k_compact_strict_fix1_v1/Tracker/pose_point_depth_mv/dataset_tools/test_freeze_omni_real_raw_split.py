from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pose_point_depth_mv.dataset_tools.freeze_omni_real_raw_split import (
    SOURCE_INVENTORY_FORMAT,
    build_dev_benchmark,
    build_eligibility_inventory,
    build_raw_split,
    build_split_extraction_inventory,
    verify_dev_benchmark,
    verify_eligibility_inventory,
    verify_raw_split,
    verify_split_extraction_inventory,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def object_row(root: Path, category: str, index: int, passed: bool = True) -> dict:
    object_id = f"{category}_{index:03d}"
    checks = {
        "images_present": True,
        "image_mask_pair_ratio": True,
        "colmap_text_complete": True,
        "poses_bounds_present": passed,
        "scan_obj_present": True,
    }
    return {
        "category": category,
        "object_id": object_id,
        "archive": str(root / f"{category}.tar.gz"),
        "scan_obj": str(root / "scans" / category / object_id / "Scan" / "Scan.obj"),
        "image_count": 100,
        "mask_count": 100,
        "paired_count": 100,
        "colmap_files": ["cameras.txt", "images.txt", "points3D.txt"],
        "checks": checks,
        "passed": passed,
    }


def source_inventory(root: Path) -> Path:
    for category in ("cup", "bowl"):
        (root / f"{category}.tar.gz").write_bytes(b"test archive")
    objects = [
        object_row(root, "cup", 1),
        object_row(root, "cup", 2),
        object_row(root, "cup", 3, passed=False),
        object_row(root, "bowl", 1),
        object_row(root, "bowl", 2),
    ]
    categories = []
    for category in ("cup", "bowl"):
        rows = [row for row in objects if row["category"] == category]
        categories.append(
            {
                "category": category,
                "archive": str(root / f"{category}.tar.gz"),
                "archive_bytes": 1000,
                "archive_member_count": 10,
                "video_object_count": len(rows),
                "passed_object_count": sum(row["passed"] for row in rows),
                "objects": rows,
                "passed": all(row["passed"] for row in rows),
            }
        )
    path = root / "source_inventory.json"
    write_json(
        path,
        {
            "format": SOURCE_INVENTORY_FORMAT,
            "category_count": len(categories),
            "video_object_count": len(objects),
            "passed_object_count": sum(row["passed"] for row in objects),
            "categories": categories,
            "objects": objects,
            "passed": False,
        },
    )
    return path


class FreezeOmniRealRawSplitTest(unittest.TestCase):
    def test_eligibility_preserves_rejections_without_failing_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_inventory(root)
            output = root / "eligibility"

            report = build_eligibility_inventory(source, output, required_eligible_count=3)

            self.assertTrue(report["passed"])
            self.assertFalse(report["source_inventory_passed"])
            self.assertEqual(report["eligible_object_count"], 4)
            self.assertEqual(report["rejected_object_count"], 1)
            self.assertEqual(report["rejected_objects"][0]["object_id"], "cup_003")
            self.assertEqual(
                report["rejected_objects"][0]["rejection_reasons"],
                ["poses_bounds_present"],
            )
            verified = verify_eligibility_inventory(output / "inventory.json", source)
            self.assertEqual(verified["eligible_object_id_hash"], report["eligible_object_id_hash"])

    def test_split_excludes_reviewed_and_rejected_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_inventory(root)
            eligibility_dir = root / "eligibility"
            eligibility = build_eligibility_inventory(
                source, eligibility_dir, required_eligible_count=3
            )
            by_id = {
                row["object_id"]: row
                for row in eligibility["eligible_objects"] + eligibility["rejected_objects"]
            }
            reviewed = root / "reviewed.json"
            write_json(
                reviewed,
                {
                    "passed": True,
                    "excluded_object_count": 1,
                    "objects": [
                        {
                            "real_object_id": "cup_001",
                            "scan_obj": by_id["cup_001"]["scan_obj"],
                        }
                    ],
                },
            )
            pilot = root / "pilot.json"
            write_json(
                pilot,
                {
                    "passed": True,
                    "pilot_object_count": 2,
                    "exclusion_applied": False,
                    "formal_split_allowed": True,
                    "objects": [
                        {
                            "object_id": "cup_001",
                            "scan_obj": by_id["cup_001"]["scan_obj"],
                        },
                        {
                            "object_id": "bowl_001",
                            "scan_obj": by_id["bowl_001"]["scan_obj"],
                        },
                    ],
                },
            )
            split_dir = root / "split"

            report = build_raw_split(
                eligibility_dir / "inventory.json",
                reviewed,
                pilot,
                split_dir,
                train_count=1,
                dev_count=1,
                holdout_count=1,
                expected_reviewed_count=1,
                expected_pilot_count=2,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(
                report["split_counts"],
                {"train": 1, "dev": 1, "holdout": 1, "reserve": 0},
            )
            self.assertEqual(report["rejected_object_ids"], ["cup_003"])
            self.assertEqual(report["excluded_old_omni_count"], 1)
            self.assertEqual(report["pilot29_eligible_count"], 1)
            selected = set()
            for name in ("train", "dev", "holdout", "reserve"):
                body = json.loads((split_dir / f"{name}.json").read_text())
                selected.update(row["object_id"] for row in body["objects"])
            self.assertNotIn("cup_001", selected)
            self.assertNotIn("cup_003", selected)
            verify_raw_split(
                split_dir / "split_report.json",
                eligibility_dir / "inventory.json",
                reviewed,
                pilot,
            )

            train_inventory_path = root / "train_inventory.json"
            train_inventory = build_split_extraction_inventory(
                split_dir / "train.json", train_inventory_path
            )
            self.assertEqual(train_inventory["format"], SOURCE_INVENTORY_FORMAT)
            self.assertEqual(train_inventory["video_object_count"], 1)
            self.assertEqual(train_inventory["split"], "train")
            self.assertFalse(train_inventory["training_ready"])
            verify_split_extraction_inventory(
                split_dir / "train.json", train_inventory_path
            )

            train_path = split_dir / "train.json"
            train_path.write_text(train_path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split file changed: train"):
                verify_raw_split(
                    split_dir / "split_report.json",
                    eligibility_dir / "inventory.json",
                    reviewed,
                    pilot,
                )

    def test_split_shortfall_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_inventory(root)
            eligibility_dir = root / "eligibility"
            build_eligibility_inventory(source, eligibility_dir, required_eligible_count=3)
            source_payload = json.loads(source.read_text())
            by_id = {row["object_id"]: row for row in source_payload["objects"]}
            reviewed = root / "reviewed.json"
            write_json(
                reviewed,
                {
                    "passed": True,
                    "excluded_object_count": 1,
                    "objects": [
                        {
                            "real_object_id": "cup_001",
                            "scan_obj": by_id["cup_001"]["scan_obj"],
                        }
                    ],
                },
            )
            pilot = root / "pilot.json"
            write_json(
                pilot,
                {
                    "passed": True,
                    "pilot_object_count": 1,
                    "exclusion_applied": False,
                    "formal_split_allowed": True,
                    "objects": [
                        {
                            "object_id": "cup_001",
                            "scan_obj": by_id["cup_001"]["scan_obj"],
                        }
                    ],
                },
            )

            report = build_raw_split(
                eligibility_dir / "inventory.json",
                reviewed,
                pilot,
                root / "shortfall",
                train_count=2,
                dev_count=1,
                holdout_count=1,
                expected_reviewed_count=1,
                expected_pilot_count=1,
            )

            self.assertFalse(report["passed"])
            self.assertEqual(report["shortfall"], 1)
            self.assertIsNone(report["split_counts"])

    def test_pilot_free_eval_and_frozen_dev_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = source_inventory(root)
            eligibility_dir = root / "eligibility"
            eligibility = build_eligibility_inventory(
                source, eligibility_dir, required_eligible_count=3
            )
            by_id = {
                row["object_id"]: row
                for row in eligibility["eligible_objects"]
                + eligibility["rejected_objects"]
            }
            reviewed = root / "reviewed.json"
            write_json(
                reviewed,
                {
                    "passed": True,
                    "excluded_object_count": 1,
                    "objects": [
                        {
                            "real_object_id": "cup_001",
                            "scan_obj": by_id["cup_001"]["scan_obj"],
                        }
                    ],
                },
            )
            pilot = root / "pilot.json"
            write_json(
                pilot,
                {
                    "passed": True,
                    "pilot_object_count": 2,
                    "exclusion_applied": False,
                    "formal_split_allowed": True,
                    "objects": [
                        {
                            "object_id": "cup_001",
                            "scan_obj": by_id["cup_001"]["scan_obj"],
                        },
                        {
                            "object_id": "bowl_001",
                            "scan_obj": by_id["bowl_001"]["scan_obj"],
                        },
                    ],
                },
            )
            split_dir = root / "pilot_free_split"
            report = build_raw_split(
                eligibility_dir / "inventory.json",
                reviewed,
                pilot,
                split_dir,
                train_count=1,
                dev_count=1,
                holdout_count=1,
                expected_reviewed_count=1,
                expected_pilot_count=2,
                exclude_pilot_from_eval=True,
            )

            self.assertTrue(report["passed"])
            self.assertTrue(report["exclude_pilot_from_eval"])
            self.assertEqual(
                report["pilot29_split_counts"],
                {"train": 1, "dev": 0, "holdout": 0, "reserve": 0},
            )
            verify_raw_split(
                split_dir / "split_report.json",
                eligibility_dir / "inventory.json",
                reviewed,
                pilot,
            )

            benchmark_dir = root / "benchmark"
            benchmark = build_dev_benchmark(
                split_dir / "dev.json",
                pilot,
                benchmark_dir,
                benchmark_count=1,
                expected_pilot_count=2,
            )
            self.assertTrue(benchmark["passed"])
            self.assertEqual(benchmark["benchmark_object_count"], 1)
            self.assertEqual(benchmark["remainder_object_count"], 0)
            inventory = json.loads(
                (benchmark_dir / "benchmark1_inventory.json").read_text()
            )
            self.assertEqual(inventory["video_object_count"], 1)
            self.assertFalse(inventory["training_ready"])
            verify_dev_benchmark(
                benchmark_dir / "benchmark_report.json",
                split_dir / "dev.json",
                pilot,
            )


if __name__ == "__main__":
    unittest.main()
