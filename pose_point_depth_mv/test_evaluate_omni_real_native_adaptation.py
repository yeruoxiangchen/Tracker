#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.evaluate_omni_real_native_adaptation import (
    RAW_CACHE_FORMAT,
    RUNTIME_MANIFEST_FORMAT,
    SOURCE_INVENTORY_FORMAT,
    SPLIT_ROWS_FORMAT,
    SURFACE_FIELDS,
    _formal_holdout_binding,
    _paired_delta,
    adaptation_decision,
)
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


def row(method: str, chamfer: float, score: float) -> dict:
    output = {"method": method, "object_key": "category:object", "seed": 42}
    for field in SURFACE_FIELDS:
        output[field] = chamfer if field.startswith("chamfer") else score
    return output


class OmniRealNativeAdaptationTest(unittest.TestCase):
    def test_better_adapted_model_passes_primary_decision(self) -> None:
        comparison = _paired_delta(
            [row("adapted", 0.1, 0.9), row("parent", 0.2, 0.8)],
            left="adapted",
            right="parent",
        )
        decision = adaptation_decision(comparison)
        self.assertTrue(decision["primary_passed"])
        self.assertTrue(decision["secondary_all_nonnegative"])

    def test_chamfer_regression_fails_primary_decision(self) -> None:
        comparison = _paired_delta(
            [row("adapted", 0.3, 0.9), row("parent", 0.2, 0.8)],
            left="adapted",
            right="parent",
        )
        self.assertFalse(adaptation_decision(comparison)["primary_passed"])

    def test_formal_binding_requires_exact_holdout_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            object_row = {"category": "category", "object_id": "object"}
            split_path = root / "holdout.json"
            split_path.write_text(
                json.dumps(
                    {
                        "format": SPLIT_ROWS_FORMAT,
                        "split": "holdout",
                        "object_count": 1,
                        "objects": [object_row],
                        "training_ready": False,
                    }
                ),
                encoding="utf-8",
            )
            inventory_path = root / "inventory.json"
            inventory_path.write_text(
                json.dumps(
                    {
                        "format": SOURCE_INVENTORY_FORMAT,
                        "passed": True,
                        "video_object_count": 1,
                        "split": "holdout",
                        "source_split": str(split_path),
                        "source_split_sha256": sha256_file(split_path),
                        "objects": [object_row],
                    }
                ),
                encoding="utf-8",
            )
            raw_path = root / "raw.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "format": RAW_CACHE_FORMAT,
                        "passed": True,
                        "object_count": 1,
                        "inventory": str(inventory_path),
                        "inventory_sha256": sha256_file(inventory_path),
                    }
                ),
                encoding="utf-8",
            )
            runtime_path = root / "runtime.json"
            runtime = {
                "format": RUNTIME_MANIFEST_FORMAT,
                "passed": True,
                "raw_cache_report": str(raw_path),
                "raw_cache_report_sha256": sha256_file(raw_path),
            }
            runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
            binding = _formal_holdout_binding(
                split_path=split_path,
                label_keys={"category:object"},
                runtime_path=runtime_path,
                runtime=runtime,
                expected_objects=1,
            )
            self.assertTrue(binding["passed"])

            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            inventory["split"] = "dev"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            runtime["raw_cache_report_sha256"] = sha256_file(raw_path)
            with self.assertRaisesRegex(RuntimeError, "holdout inventory"):
                _formal_holdout_binding(
                    split_path=split_path,
                    label_keys={"category:object"},
                    runtime_path=runtime_path,
                    runtime=runtime,
                    expected_objects=1,
                )


if __name__ == "__main__":
    unittest.main()
