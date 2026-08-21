from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.native_ss_genrecon_no_vggt import (
    NATIVE_SS_NO_VGGT_CALIBRATION,
    select_manifest_order_object_indices,
)
from pose_point_depth_mv.no_vggt_ss_evidence import (
    NO_VGGT_EXPLORATORY_SS_DEPLOYMENT,
    freeze_exploratory_ss_deployment,
    load_no_vggt_ss_evidence,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


class NoVggtSSEvidenceTest(unittest.TestCase):
    def test_manifest_order_selection_does_not_uid_sort(self) -> None:
        rows = [
            {"uid": "b0", "object_uid": "b"},
            {"uid": "a0", "object_uid": "a"},
            {"uid": "a1", "object_uid": "a"},
            {"uid": "c0", "object_uid": "c"},
        ]
        self.assertEqual(
            select_manifest_order_object_indices(rows, start=0, end=2), [0, 1]
        )
        self.assertEqual(
            select_manifest_order_object_indices(rows, start=2, end=3), [3]
        )

    def test_exploratory_report_preserves_failed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            protocol = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_step": 2000,
                "weights": "ema",
                "steps": 25,
                "cfg_interval": [0.5, 1.0],
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
                "amp_dtype": "bf16",
                "condition_scale_policy": "learned_projection_only",
                "post_cfg_cap": False,
            }
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "format": NATIVE_SS_NO_VGGT_CALIBRATION,
                        "passed": False,
                        "protocol": protocol,
                        "candidates": [
                            {
                                "cfg_strength": 5.0,
                                "object_count": 16,
                                "checks": {
                                    "iou_gain_mean": True,
                                    "latent_mse_gain_mean": False,
                                },
                                "summary": {},
                                "count_summary": {},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "deployment.json"
            payload = freeze_exploratory_ss_deployment(
                calibration,
                output,
                cfg_strength=5.0,
                expected_objects=16,
                diagnostic_scope="unit_test_nonformal",
            )
            self.assertEqual(payload["format"], NO_VGGT_EXPLORATORY_SS_DEPLOYMENT)
            self.assertIs(payload["formal"], False)
            self.assertIs(payload["passed"], False)
            self.assertEqual(payload["failed_quality_checks"], ["latent_mse_gain_mean"])
            loaded, binding = load_no_vggt_ss_evidence(output)
            self.assertEqual(loaded, payload)
            self.assertIs(binding["formal"], False)
            self.assertIs(binding["exploratory"], True)
            self.assertEqual(binding["cfg_strength"], 5.0)


if __name__ == "__main__":
    unittest.main()
