#!/usr/bin/env python3

import tempfile
from pathlib import Path
import unittest

import numpy as np

from pose_point_depth_mv.dataset_tools.repair_reviewed_frozen_object_ss import (
    is_reusable_object_repair,
    validate_object_metadata,
)


def pack(coords: np.ndarray, *, repaired: bool = True) -> dict[str, np.ndarray]:
    payload = {
        "z": np.zeros((8, 16, 16, 16), dtype=np.float16),
        "target_coords": np.asarray(coords, dtype=np.int32),
        "normalize_center": np.zeros(3, dtype=np.float32),
        "normalize_scale": np.array(1.0, dtype=np.float32),
        "source_glb": np.array("/tmp/mesh.glb"),
        "pixal3d_rotation": np.eye(3, dtype=np.float32),
    }
    if repaired:
        payload.update(
            {
                "mesh_target_coords": np.asarray(coords, dtype=np.int32),
                "repair_format": np.array("object_level_ss_repair.v1"),
                "repair_target_mode": np.array("decoder_projected"),
                "surface_seed": np.array(42, dtype=np.uint32),
            }
        )
    return payload


class ReviewedObjectSsRepairTest(unittest.TestCase):
    def test_reuses_equal_object_level_repair(self) -> None:
        coords = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        self.assertTrue(
            is_reusable_object_repair(
                [pack(coords), pack(coords)], target_mode="decoder_projected"
            )
        )

    def test_rebuilds_unrepaired_or_different_sequences(self) -> None:
        first = np.asarray([[1, 2, 3]], dtype=np.int32)
        second = np.asarray([[1, 2, 4]], dtype=np.int32)
        self.assertFalse(
            is_reusable_object_repair(
                [pack(first, repaired=False), pack(second, repaired=False)],
                target_mode="decoder_projected",
            )
        )
        self.assertFalse(
            is_reusable_object_repair(
                [pack(first), pack(second)], target_mode="decoder_projected"
            )
        )

    def test_metadata_mismatch_is_rejected(self) -> None:
        coords = np.asarray([[1, 2, 3]], dtype=np.int32)
        first = pack(coords)
        second = pack(coords)
        second["normalize_scale"] = np.array(2.0, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "normalize_scale differs"):
            validate_object_metadata("sample", [first, second])


if __name__ == "__main__":
    unittest.main()
