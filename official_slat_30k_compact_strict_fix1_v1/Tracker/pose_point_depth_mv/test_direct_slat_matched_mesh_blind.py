from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.direct_slat_matched_mesh_blind import (
    PROTOCOL_FORMAT,
    bind_file,
    canonical_sha256,
    formal_decision,
    load_protocol,
    select_fresh_rows,
    summarize_seed_directions,
)


class DirectSLatMatchedMeshBlindTests(unittest.TestCase):
    def test_selection_is_fresh_object_unique_and_deterministic(self) -> None:
        rows = []
        for object_index in range(6):
            for seed in (42, 43, 44):
                rows.append(
                    {
                        "object_uid": f"object_{object_index}",
                        "uid": f"object_{object_index}_seq000",
                        "support_seed": seed,
                    }
                )
        kwargs = {
            "seeds": (42, 43, 44),
            "excluded_object_uids": {"object_1"},
            "expected_objects": 4,
            "selection_seed": 20260729,
        }
        first = select_fresh_rows(rows, **kwargs)
        second = select_fresh_rows(rows, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertNotIn("object_1", {row["object_uid"] for row in first})
        for row in first:
            self.assertEqual(set(row["cache_indices"]), {"42", "43", "44"})

    def test_protocol_hash_and_file_bindings_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bound = root / "manifest.json"
            bound.write_text("{}\n", encoding="utf-8")
            protocol = {
                "format": PROTOCOL_FORMAT,
                "formal": True,
                "bindings": {"cache_manifest": bind_file(bound)},
                "selection": {
                    "expected_objects": 1,
                    "excluded_object_uids": ["old_object"],
                },
                "selected": [
                    {
                        "object_uid": "new_object",
                        "uid": "new_object_seq000",
                        "cache_indices": {"42": 0},
                    }
                ],
            }
            protocol["protocol_sha256"] = canonical_sha256(protocol)
            path = root / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            self.assertEqual(
                load_protocol(path)["protocol_sha256"],
                protocol["protocol_sha256"],
            )
            bound.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "binding changed"):
                load_protocol(path)

    def test_formal_decision_uses_object_bootstrap_and_seed_direction(self) -> None:
        summary = {
            "chamfer_l1_improvement": {
                "mean": 0.002,
                "median": 0.001,
                "positive_rate": 0.625,
                "bootstrap_mean_95_ci": [0.0001, 0.003],
            },
            "fscore_0p02_delta": {"mean": 0.001},
            "normal_consistency_delta": {"mean": 0.002},
            "largest_component_ratio_delta": {"mean": 0.0003},
        }
        records = [
            {
                "joint_seed": seed,
                "chamfer_l1_improvement": value,
                "fscore_0p02_delta": 0.0,
                "normal_consistency_delta": 0.0,
                "largest_component_ratio_delta": 0.0,
            }
            for seed, value in ((42, 0.1), (43, 0.2), (44, -0.01))
        ]
        by_seed = summarize_seed_directions(
            records,
            (
                "chamfer_l1_improvement",
                "fscore_0p02_delta",
                "normal_consistency_delta",
                "largest_component_ratio_delta",
            ),
        )
        thresholds = {
            "min_chamfer_bootstrap_lower": 0.0,
            "min_chamfer_median": 0.0,
            "min_chamfer_object_win_rate": 0.5,
            "min_positive_seed_fraction": 2.0 / 3.0,
            "secondary_mean_floors": {
                "fscore_0p02_delta": 0.0,
                "normal_consistency_delta": 0.0,
                "largest_component_ratio_delta": 0.0,
            },
        }
        decision = formal_decision(summary, by_seed, thresholds)
        self.assertTrue(decision["formal_pass"])
        self.assertAlmostEqual(decision["positive_seed_fraction"], 2.0 / 3.0)
        summary["normal_consistency_delta"]["mean"] = -1.0e-6
        self.assertFalse(
            formal_decision(summary, by_seed, thresholds)["formal_pass"]
        )


if __name__ == "__main__":
    unittest.main()
