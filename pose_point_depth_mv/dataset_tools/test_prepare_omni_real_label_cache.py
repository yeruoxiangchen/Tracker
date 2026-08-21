from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from pose_point_depth_mv.dataset_tools.prepare_omni_real_label_cache import (
    build_object_label,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import sha256_file
from pose_point_depth_mv.real_object_canonicalization import canonical_json_sha256


class PrepareOmniRealLabelCacheTest(unittest.TestCase):
    def _inputs(self, root: Path, *, automatic_passed: bool = True):
        raw_cache = root / "raw.npz"
        np.savez_compressed(raw_cache, sentinel=np.asarray([1]))
        raw_hash = sha256_file(raw_cache)

        runtime_dir = root / "runtime"
        runtime_dir.mkdir()
        T_O2W = np.eye(4)
        T_O2W[:3, :3] *= 2.0
        T_O2W[:3, 3] = [1.0, -2.0, 3.0]
        T_W2O = np.linalg.inv(T_O2W)
        runtime_cache = runtime_dir / "runtime_input_cache.npz"
        np.savez_compressed(runtime_cache, T_O2W=T_O2W, T_W2O=T_W2O)
        condition = {
            "format": "test.observable.condition.v1",
            "condition_scope": "observable inputs only",
        }
        condition_hash = canonical_json_sha256(condition)
        condition["condition_sha256"] = condition_hash
        condition_path = runtime_dir / "condition.json"
        condition_path.write_text(json.dumps(condition) + "\n", encoding="utf-8")
        (runtime_dir / "report.json").write_text("{}\n", encoding="utf-8")
        runtime = {
            "category": "cup",
            "object_id": "cup_001",
            "passed": True,
            "cache_npz": str(runtime_cache),
            "condition_record": str(condition_path),
            "condition_sha256": condition_hash,
            "source_raw_cache_sha256": raw_hash,
        }

        alignment_dir = root / "alignment"
        alignment_dir.mkdir()
        T_Scan2W = np.eye(4)
        T_Scan2W[:3, 3] = [3.0, 4.0, 5.0]
        alignment_cache = alignment_dir / "coarse_alignment.npz"
        np.savez_compressed(alignment_cache, T_Scan_to_COLMAP_W=T_Scan2W)
        scan = root / "Scan.obj"
        scan.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
        )
        (alignment_dir / "report.json").write_text("{}\n", encoding="utf-8")
        alignment = {
            "category": "cup",
            "object_id": "cup_001",
            "automatic_passed": automatic_passed,
            "alignment_quality_policy": {"name": "frozen-test-policy"},
            "alignment_quality_checks": {"inlier_rate_3pct": automatic_passed},
            "median_normalized": 0.02,
            "inlier_rate_3pct": 0.55 if not automatic_passed else 0.75,
            "p90_normalized_diagnostic": 0.15,
            "cache_npz": str(alignment_cache),
            "raw_cache_sha256": raw_hash,
            "scan_obj": str(scan),
        }
        return runtime, alignment, T_W2O, T_Scan2W, condition_hash

    def test_builds_label_in_runtime_o_without_changing_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, alignment, T_W2O, T_Scan2W, condition_hash = self._inputs(root)

            output = root / "labels"
            report, reused = build_object_label(runtime, alignment, output_dir=output)
            second, second_reused = build_object_label(
                runtime, alignment, output_dir=output
            )

            self.assertFalse(reused)
            self.assertTrue(second_reused)
            self.assertEqual(report["condition_sha256"], condition_hash)
            self.assertEqual(second["condition_sha256"], condition_hash)
            self.assertFalse(report["gt_fields_exported_to_model_condition"])
            self.assertFalse(report["training_ready"])
            with np.load(report["label_cache"], allow_pickle=False) as payload:
                expected = T_W2O @ T_Scan2W
                np.testing.assert_allclose(payload["T_Scan2O"], expected)
            first_vertex = next(
                line for line in Path(report["mesh_o"]).read_text().splitlines()
                if line.startswith("v ")
            )
            values = np.asarray([float(value) for value in first_vertex.split()[1:4]])
            np.testing.assert_allclose(values, (T_W2O @ T_Scan2W)[:3, 3])

    def test_low_confidence_alignment_requires_explicit_warning_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, alignment, _, _, condition_hash = self._inputs(
                root, automatic_passed=False
            )
            with self.assertRaisesRegex(RuntimeError, "GT alignment did not pass"):
                build_object_label(runtime, alignment, output_dir=root / "blocked")

            report, reused = build_object_label(
                runtime,
                alignment,
                output_dir=root / "included",
                include_alignment_quality_warning=True,
            )
            self.assertFalse(reused)
            self.assertFalse(report["alignment_quality_passed"])
            self.assertTrue(report["alignment_quality_warning_included"])
            self.assertEqual(report["condition_sha256"], condition_hash)
            self.assertFalse(report["gt_fields_exported_to_model_condition"])


if __name__ == "__main__":
    unittest.main()
