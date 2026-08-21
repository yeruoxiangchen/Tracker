from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.dataset_tools.freeze_omni_real_raw_cache_eligibility import (
    ELIGIBILITY_FORMAT,
    freeze_eligible_raw_cache,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import (
    RAW_CACHE_FORMAT,
)


class FreezeOmniRealRawCacheEligibilityTest(unittest.TestCase):
    def test_freezes_sparse_and_view_eligible_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            output = root / "eligible" / "raw_cache_report.json"
            rows = [
                {
                    "category": "fruit",
                    "object_id": "good",
                    "registered_pair_count": 8,
                    "sparse_point_count": 100,
                },
                {
                    "category": "fruit",
                    "object_id": "few_points",
                    "registered_pair_count": 12,
                    "sparse_point_count": 81,
                },
                {
                    "category": "tool",
                    "object_id": "few_views",
                    "registered_pair_count": 7,
                    "sparse_point_count": 500,
                },
            ]
            source.write_text(
                json.dumps(
                    {
                        "format": RAW_CACHE_FORMAT,
                        "object_count": len(rows),
                        "objects": rows,
                        "training_ready": False,
                        "passed": True,
                    }
                ),
                encoding="utf-8",
            )

            report = freeze_eligible_raw_cache(
                source,
                output,
                expected_source_objects=3,
                min_eligible_objects=1,
                min_registered_pairs=8,
                min_sparse_points=100,
            )
            reused = freeze_eligible_raw_cache(
                source,
                output,
                expected_source_objects=3,
                min_eligible_objects=1,
                min_registered_pairs=8,
                min_sparse_points=100,
            )

            self.assertTrue(report["passed"])
            self.assertEqual(report, reused)
            self.assertEqual(report["object_count"], 1)
            self.assertEqual(report["objects"][0]["object_id"], "good")
            self.assertEqual(report["eligibility"]["format"], ELIGIBILITY_FORMAT)
            self.assertEqual(report["eligibility"]["excluded_object_count"], 2)
            self.assertEqual(
                [row["object_key"] for row in report["eligibility"]["excluded"]],
                ["fruit:few_points", "tool:few_views"],
            )
            self.assertFalse(report["training_ready"])


if __name__ == "__main__":
    unittest.main()
