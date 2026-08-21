from __future__ import annotations

from pathlib import Path
import unittest

from pose_point_depth_mv.select_objaverse_training_overlap_subset import (
    _is_objaverse_path,
    _require_objaverse_path,
    stable_rank,
)
from pose_point_depth_mv.training_overlap_objaverse import (
    OBJAVERSE2K_DEV64_PROTOCOL_FORMAT,
    OBJAVERSE2K_DEV64_SCOPE,
    TRAINING_OVERLAP_PROTOCOL_FORMAT,
    TRAINING_OVERLAP_SCOPE,
    expected_view_count,
    validate_selection,
)


def overlap_selection(source_scope: str = "mixed_objaverse_train") -> dict:
    samples = [
        {
            "uid": "obj-a_seq001",
            "object_uid": "obj-a",
            "dataset_source": "objaverse",
            "training_overlap_selection": {"expected_view_count": 4},
        },
        {
            "uid": "obj-b_seq000",
            "object_uid": "obj-b",
            "dataset_source": "objaverse",
            "training_overlap_selection": {"expected_view_count": 8},
        },
    ]
    return {
        "samples": samples,
        "training_overlap_protocol": {
            "format": TRAINING_OVERLAP_PROTOCOL_FORMAT,
            "scope": TRAINING_OVERLAP_SCOPE,
            "source_scope": source_scope,
            "formal": False,
            "training_overlap": True,
            "training_object_disjoint": False,
            "source_mesh_disjoint": False,
            "object_count": 2,
            "selected_uids": [row["uid"] for row in samples],
            "selected_object_uids": [row["object_uid"] for row in samples],
            "passed": True,
        },
    }


def dev64_selection() -> dict:
    samples = [
        {
            "uid": f"obj-{index:02d}_seq000",
            "object_uid": f"obj-{index:02d}",
            "dataset_source": "objaverse",
            "objaverse2k_dev64_selection": {"expected_view_count": 4},
        }
        for index in range(64)
    ]
    return {
        "samples": samples,
        "objaverse2k_dev64_protocol": {
            "format": OBJAVERSE2K_DEV64_PROTOCOL_FORMAT,
            "scope": OBJAVERSE2K_DEV64_SCOPE,
            "formal": False,
            "training_overlap": False,
            "training_object_disjoint": True,
            "source_mesh_disjoint": True,
            "object_count": 64,
            "selected_uids": [row["uid"] for row in samples],
            "selected_object_uids": [row["object_uid"] for row in samples],
            "passed": True,
        },
    }


class TrainingOverlapObjaverseTest(unittest.TestCase):
    def test_protocol_is_explicitly_nonformal_and_overlapping(self) -> None:
        contract = validate_selection(overlap_selection())
        self.assertEqual(contract.scope, TRAINING_OVERLAP_SCOPE)
        self.assertTrue(contract.training_overlap)
        self.assertFalse(contract.training_object_disjoint)
        self.assertFalse(contract.source_mesh_disjoint)
        self.assertEqual(expected_view_count(overlap_selection()["samples"][0]), 4)

    def test_protocol_rejects_omni_row_even_if_scope_says_mixed_objaverse(self) -> None:
        payload = overlap_selection()
        payload["samples"][0]["dataset_source"] = "omni"
        with self.assertRaisesRegex(RuntimeError, "non-Objaverse"):
            validate_selection(payload)

    def test_canonical_path_filter_rejects_omni_and_accepts_objaverse(self) -> None:
        objaverse = Path(
            "/data/zjr/Objaverse/.objaverse/hf-objaverse-v1/glbs/000-001/a.glb"
        )
        omni = Path("/data/zjr/OmniObject3D/categories/apple/Scan.obj")
        self.assertTrue(_is_objaverse_path(objaverse))
        self.assertFalse(_is_objaverse_path(omni))
        with self.assertRaisesRegex(RuntimeError, "not a canonical Objaverse GLB"):
            _require_objaverse_path(omni, label="fixture")

    def test_stable_rank_is_repeatable_and_seed_sensitive(self) -> None:
        first = stable_rank(42, "object", "mixed_objaverse_train", "abc")
        self.assertEqual(first, stable_rank(42, "object", "mixed_objaverse_train", "abc"))
        self.assertNotEqual(first, stable_rank(43, "object", "mixed_objaverse_train", "abc"))

    def test_frozen_test16_cannot_be_relabelled_as_overlap(self) -> None:
        payload = overlap_selection()
        payload["training_overlap_protocol"]["training_object_disjoint"] = True
        with self.assertRaisesRegex(RuntimeError, "protocol differs"):
            validate_selection(payload)

    def test_dev64_is_disjoint_but_remains_nonformal(self) -> None:
        payload = dev64_selection()
        contract = validate_selection(payload)
        self.assertEqual(contract.scope, OBJAVERSE2K_DEV64_SCOPE)
        self.assertFalse(contract.training_overlap)
        self.assertTrue(contract.training_object_disjoint)
        self.assertTrue(contract.source_mesh_disjoint)
        self.assertEqual(expected_view_count(payload["samples"][0]), 4)
        payload["objaverse2k_dev64_protocol"]["formal"] = True
        with self.assertRaisesRegex(RuntimeError, "dev64 contract differs"):
            validate_selection(payload)


if __name__ == "__main__":
    unittest.main()
