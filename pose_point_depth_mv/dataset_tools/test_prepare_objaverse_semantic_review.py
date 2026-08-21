from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from pose_point_depth_mv.dataset_tools.prepare_objaverse_semantic_review import (
    DECISIONS,
    merge_review_csv,
    normalize_decision,
    sample_object_uid,
    source_identities,
)


class ObjaverseSemanticReviewTests(unittest.TestCase):
    def test_sample_object_uid_falls_back_to_sequence_uid(self) -> None:
        self.assertEqual(sample_object_uid({"uid": "abc_seq001"}), "abc")
        self.assertEqual(
            sample_object_uid({"object_uid": "obj", "uid": "ignored_seq000"}),
            "obj",
        )

    def test_source_identity_uses_resolved_mesh_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = root / "mesh.glb"
            mesh.touch()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [{"source_glb": "mesh.glb"}],
                        "failures": [],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(source_identities([manifest]), {str(mesh.resolve())})

    def test_review_labels_are_explicit(self) -> None:
        self.assertEqual(
            DECISIONS,
            {
                "keep_single_subject",
                "reject_scene_or_fragment",
                "uncertain_review",
            },
        )

    def test_binary_review_labels_are_normalized(self) -> None:
        self.assertEqual(normalize_decision("1"), "keep_single_subject")
        self.assertEqual(normalize_decision("0"), "reject_scene_or_fragment")
        self.assertEqual(
            normalize_decision("uncertain_review"), "uncertain_review"
        )
        with self.assertRaisesRegex(ValueError, "invalid semantic decision"):
            normalize_decision("")

    def test_merge_review_csv_requires_exact_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.csv"
            decisions = root / "decisions.csv"
            output = root / "merged.csv"
            base.write_text(
                "review_id,semantic_subject_label,rejection_reason,reviewer,source_glb\n"
                "L0001,,,,a.glb\n"
                "L0002,,,,b.glb\n",
                encoding="utf-8",
            )
            decisions.write_text(
                "review_id,semantic_subject_label,rejection_reason,reviewer\n"
                "L0001,keep_single_subject,,test\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                merge_review_csv(base, decisions, output)

    def test_merge_review_csv_preserves_base_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.csv"
            decisions = root / "decisions.csv"
            output = root / "merged.csv"
            base.write_text(
                "review_id,semantic_subject_label,rejection_reason,reviewer,source_glb\n"
                "L0001,,,,a.glb\n",
                encoding="utf-8",
            )
            decisions.write_text(
                "review_id,semantic_subject_label,rejection_reason,reviewer\n"
                "L0001,reject_scene_or_fragment,room,test\n",
                encoding="utf-8",
            )
            counts = merge_review_csv(base, decisions, output)
            self.assertEqual(counts, {"reject_scene_or_fragment": 1})
            with output.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["source_glb"], "a.glb")
            self.assertEqual(row["rejection_reason"], "room")


if __name__ == "__main__":
    unittest.main()
