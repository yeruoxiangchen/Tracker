from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.dataset_tools.adjudicate_omni_real_mesh_alignment import (
    MANIFEST_FORMAT as ALIGNMENT_FORMAT,
)
from pose_point_depth_mv.dataset_tools.freeze_omni_real_native_training_subset import (
    freeze_subset,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_runtime_inputs import (
    MANIFEST_FORMAT as RUNTIME_FORMAT,
)


class FreezeOmniRealNativeTrainingSubsetTest(unittest.TestCase):
    def test_freezes_matching_passed_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_path = root / "runtime.json"
            alignment_path = root / "alignment.json"
            rows = [
                {"category": "box", "object_id": f"box_{index:03d}", "passed": True}
                for index in range(3)
            ]
            runtime_path.write_text(
                json.dumps(
                    {
                        "format": RUNTIME_FORMAT,
                        "raw_cache_report_sha256": "raw",
                        "build_config": {"feature_resolution": 518},
                        "build_config_sha256": "build",
                        "objects": rows,
                        "failures": [],
                        "passed": True,
                    }
                ),
                encoding="utf-8",
            )
            aligned = [
                {
                    **row,
                    "automatic_passed": index != 1,
                    "source_automatic_passed": index == 0,
                }
                for index, row in enumerate(rows)
            ]
            alignment_path.write_text(
                json.dumps(
                    {
                        "format": ALIGNMENT_FORMAT,
                        "raw_cache_report_sha256": "raw",
                        "alignment_quality_policy": {"name": "frozen"},
                        "objects": aligned,
                        "failures": [],
                        "passed": False,
                    }
                ),
                encoding="utf-8",
            )
            report = freeze_subset(
                runtime_path,
                alignment_path,
                root / "out",
                expected_source_objects=3,
                min_objects=2,
            )
            runtime_subset = json.loads(
                (root / "out/runtime_input_manifest.json").read_text()
            )
            alignment_subset = json.loads(
                (root / "out/alignment_adjudicated.json").read_text()
            )
            self.assertEqual(report["qualified_object_count"], 2)
            self.assertEqual(report["excluded_object_keys"], ["box:box_001"])
            self.assertEqual(
                [row["object_id"] for row in runtime_subset["objects"]],
                ["box_000", "box_002"],
            )
            self.assertEqual(
                [row["object_id"] for row in alignment_subset["objects"]],
                ["box_000", "box_002"],
            )
            self.assertTrue(runtime_subset["passed"])
            self.assertTrue(alignment_subset["passed"])


if __name__ == "__main__":
    unittest.main()
