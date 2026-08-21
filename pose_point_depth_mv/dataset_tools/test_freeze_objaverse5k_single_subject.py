from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pose_point_depth_mv.dataset_tools.freeze_objaverse5k_single_subject import (
    freeze,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FreezeObjaverse5KTest(unittest.TestCase):
    def make_fixture(self, root: Path, object_count: int = 96) -> tuple[Path, list[dict]]:
        audit = root / "audit"
        meshes = root / "meshes"
        objects = []
        for index in range(object_count):
            token = f"{index:032x}"
            uid = f"objaverse_{token}"
            source = meshes / f"{token}.glb"
            source.parent.mkdir(parents=True, exist_ok=True)
            payload = f"mesh-{index}".encode("ascii")
            source.write_bytes(payload)
            stat = source.stat()
            hard = index % 3 == 0
            objects.append(
                {
                    "object_uid": uid,
                    "source_glb": str(source),
                    "auto_tier": "B",
                    "final_tier": "B",
                    "human_reviewed": False,
                    "render_evidence": False,
                    "accepted_sequences": 0,
                    "quality_flags": {
                        "low_texture": index % 17 == 0,
                        "flat_gray_blob": False,
                        "low_projection_support": False,
                    },
                    "mesh_audit": {
                        "source_exists": True,
                        "source_size": stat.st_size,
                        "source_mtime_ns": stat.st_mtime_ns,
                        "source_sha256": hashlib.sha256(payload).hexdigest(),
                        "mesh_valid": True,
                        "geometry_count": 8 if hard else 1,
                        "scene_instance_count": 8 if hard else 1,
                        "vertex_count": 100,
                        "face_count": 200,
                        "aspect_ratio": 3.0,
                        "dominant_area_ratio": 0.7,
                    },
                }
            )
        write_json(audit / "objects.json", objects)
        write_json(
            audit / "report.json",
            {
                "passed": True,
                "summary": {"object_count": len(objects)},
            },
        )
        return audit, objects

    def args(
        self, audit: Path, exclusion: Path, output: Path, **overrides: object
    ) -> argparse.Namespace:
        values = {
            "audit_root": str(audit),
            "exclude_manifest": [str(exclusion)],
            "output_dir": str(output),
            "target_objects": 20,
            "hard_fraction": 0.25,
            "reserve_objects": 4,
            "seed": 20260810,
            "shard_count": 4,
            "audit_sample_count": 8,
            "clean_max_geometry_count": 4,
            "clean_max_aspect_ratio": 6.0,
            "clean_min_dominant_area_ratio": 0.55,
            "complex_face_count": 450000,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_freezes_strata_exclusions_and_reuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, objects = self.make_fixture(root)
            exclusion = root / "exclude.json"
            write_json(
                exclusion,
                {
                    "object_records": [
                        {"object_uid": objects[0]["object_uid"]},
                        {"source_glb": objects[1]["source_glb"]},
                        {
                            "source_glb_sha256": objects[2]["mesh_audit"][
                                "source_sha256"
                            ]
                        },
                    ]
                },
            )
            output = root / "freeze"
            report = freeze(self.args(audit, exclusion, output))
            self.assertTrue(report["passed"])
            self.assertFalse(report["training_ready"])
            self.assertEqual(report["summary"]["selected_object_count"], 20)
            self.assertEqual(report["summary"]["reserve_object_count"], 4)
            self.assertEqual(report["summary"]["excluded_overlap_object_count"], 3)

            selection = json.loads((output / "selection.json").read_text())
            counts = {}
            selected_uids = set()
            for row in selection["objects"]:
                selected_uids.add(row["object_uid"])
                counts[row["source_stage_class"]] = (
                    counts.get(row["source_stage_class"], 0) + 1
                )
            self.assertEqual(counts["source_clean_candidate"], 15)
            self.assertEqual(counts["source_hard_candidate"], 5)
            self.assertFalse(
                selected_uids & {objects[index]["object_uid"] for index in range(3)}
            )

            plan = json.loads((output / "build_plan.json").read_text())
            self.assertEqual(plan["schema"], "tracker.mixed_mesh10k_sources.v1")
            self.assertEqual(plan["objaverse"]["object_count"], 20)
            self.assertEqual(len(plan["objaverse"]["shards"]), 4)
            self.assertEqual(
                sum(row["object_count"] for row in plan["objaverse"]["shards"]),
                20,
            )

            reused = freeze(self.args(audit, exclusion, output))
            self.assertEqual(reused["artifact_sha256"], report["artifact_sha256"])

    def test_rejects_insufficient_clean_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit, _ = self.make_fixture(root, object_count=12)
            exclusion = root / "exclude.json"
            write_json(exclusion, {"samples": []})
            with self.assertRaisesRegex(RuntimeError, "pool has"):
                freeze(
                    self.args(
                        audit,
                        exclusion,
                        root / "freeze",
                        target_objects=20,
                        reserve_objects=0,
                    )
                )


if __name__ == "__main__":
    unittest.main()
