from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import trimesh

from pose_point_depth_mv.dataset_tools.restructure_single_object_dataset import (
    audit_mesh_task,
    classify,
    deduplicate_source_assets,
    distribute_shards,
    manifest_object_uids,
    manifest_source_asset_identities,
    parse_source_plan_limits,
    select_source_plan_candidates,
)
from pose_point_depth_mv.dataset_tools.prepare_mixed_mesh10k_sources import (
    shard_index,
)


def policy_args() -> argparse.Namespace:
    return argparse.Namespace(
        a_max_geometry_count=4,
        a_max_aspect_ratio=6.0,
        a_min_dominant_area_ratio=0.55,
        c_min_geometry_count=101,
        c_min_aspect_ratio=12.0,
        c_secondary_geometry_count=21,
        c_max_dominant_area_ratio=0.25,
    )


def row(audit: dict, *, rendered: bool = True) -> dict:
    return {
        "mesh_audit": audit,
        "render_evidence": rendered,
        "accepted_sequences": 1 if rendered else 0,
        "failure_counts": {},
        "quality_flags": {
            "low_texture": False,
            "flat_gray_blob": False,
            "low_projection_support": False,
        },
    }


class SingleObjectRestructureTests(unittest.TestCase):
    def test_manifest_object_uids_cover_samples_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(
                json.dumps(
                    {
                        "samples": [
                            {"object_uid": "object_a"},
                            {"uid": "object_b_seq001"},
                        ],
                        "failures": [{"uid": "object_c_seq000"}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                manifest_object_uids(path),
                {"object_a", "object_b", "object_c"},
            )

    def test_manifest_source_assets_cover_uid_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.glb"
            mesh.touch()
            path = root / "split.json"
            path.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "object_uid": "raw_uid",
                                "source_glb": str(mesh),
                            }
                        ],
                        "failures": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                manifest_source_asset_identities(path),
                {str(mesh.resolve())},
            )

    def test_source_asset_dedup_prefers_explicit_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory) / "mesh.glb"
            mesh.touch()
            objects = {
                "raw_uid": {
                    "source_glb": str(mesh),
                    "explicit_manifest": True,
                    "render_evidence": True,
                },
                "objaverse_raw_uid": {
                    "source_glb": str(mesh),
                    "explicit_manifest": False,
                    "render_evidence": True,
                },
            }
            selected, report = deduplicate_source_assets(objects)
            self.assertEqual(set(selected), {"raw_uid"})
            self.assertEqual(report["removed_alias_count"], 1)
            self.assertEqual(
                report["aliases"][0]["removed_uids"],
                ["objaverse_raw_uid"],
            )

    def test_source_plan_limits_parse_and_reject_duplicates(self) -> None:
        self.assertEqual(
            parse_source_plan_limits(["objaverse=2000", "omni=0"]),
            {"objaverse": 2000, "omni": 0},
        )
        with self.assertRaises(ValueError):
            parse_source_plan_limits(["objaverse=2", "objaverse=3"])

    def test_source_plan_limit_preserves_manifest_and_override_objects(self) -> None:
        objects = {
            "obj_a": {
                "source": "objaverse",
                "source_plan_candidate": True,
                "manifest_indices": [],
            },
            "obj_b": {
                "source": "objaverse",
                "source_plan_candidate": True,
                "manifest_indices": [],
            },
            "obj_rendered": {
                "source": "objaverse",
                "source_plan_candidate": True,
                "manifest_indices": [2],
            },
            "omni_new": {
                "source": "omni",
                "source_plan_candidate": True,
                "manifest_indices": [],
            },
            "omni_rendered": {
                "source": "omni",
                "source_plan_candidate": True,
                "manifest_indices": [3],
            },
            "legacy": {
                "source": "legacy",
                "source_plan_candidate": False,
                "manifest_indices": [1],
            },
        }
        selected, report = select_source_plan_candidates(
            objects,
            {"objaverse": 1, "omni": 0},
            seed=7,
            forced_uids={"obj_b"},
        )
        self.assertIn("legacy", selected)
        self.assertIn("obj_rendered", selected)
        self.assertIn("obj_b", selected)
        self.assertIn("omni_rendered", selected)
        self.assertNotIn("omni_new", selected)
        self.assertEqual(
            report["sources"]["omni"]["source_plan_only_selected"], 0
        )

    def test_sharding_matches_frozen_source_tool(self) -> None:
        for uid in ("a", "objaverse_x", "omni_y"):
            shards = distribute_shards({uid: "/mesh"}, 16)
            actual = next(index for index, shard in enumerate(shards) if uid in shard)
            self.assertEqual(actual, shard_index(uid, 16))

    def test_valid_box_is_high_confidence_only_with_render_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mesh_path = Path(directory) / "box.glb"
            trimesh.creation.box().export(mesh_path)
            _uid, audit = audit_mesh_task(("box", str(mesh_path), True))
            self.assertTrue(audit["mesh_valid"])
            self.assertIsNotNone(audit["source_sha256"])
            self.assertEqual(classify(row(audit), policy_args())[0], "A")
            self.assertEqual(
                classify(row(audit, rendered=False), policy_args())[0], "B"
            )

    def test_missing_mesh_is_hard_reject(self) -> None:
        _uid, audit = audit_mesh_task(("missing", "/definitely/missing.glb", False))
        tier, reasons = classify(row(audit, rendered=False), policy_args())
        self.assertEqual(tier, "R")
        self.assertIn("hard_invalid_mesh", reasons[0])

    def test_multipart_is_not_automatically_rejected(self) -> None:
        audit = {
            "mesh_valid": True,
            "geometry_count": 12,
            "aspect_ratio": 2.0,
            "dominant_area_ratio": 0.10,
        }
        tier, reasons = classify(row(audit), policy_args())
        self.assertEqual(tier, "B")
        self.assertTrue(any("not an automatic rejection" in reason for reason in reasons))

    def test_extreme_scene_like_asset_is_review_candidate(self) -> None:
        audit = {
            "mesh_valid": True,
            "geometry_count": 150,
            "aspect_ratio": 2.0,
            "dominant_area_ratio": 0.10,
        }
        tier, reasons = classify(row(audit), policy_args())
        self.assertEqual(tier, "C")
        self.assertTrue(any("human review" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
