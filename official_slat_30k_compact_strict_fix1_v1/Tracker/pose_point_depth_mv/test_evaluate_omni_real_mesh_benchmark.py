#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import trimesh

from pose_point_depth_mv.evaluate_omni_real_mesh_benchmark import (
    METHOD_FORMATS,
    RECORD_METHODS,
    SURFACE_FIELDS,
    _paired_delta,
    _records_by_pair,
    require_identical_pair_coverage,
    validate_method_runtime_binding,
)
from pose_point_depth_mv.mesh_benchmark_metrics import surface_metrics
from pose_point_depth_mv.omni_real_benchmark_common import sha256_file


def inference_manifest(method: str) -> dict:
    rows = [
        {
            "method": RECORD_METHODS[method],
            "object_key": object_key,
            "seed": seed,
            "passed": True,
        }
        for object_key in ("category:a", "category:b")
        for seed in (42, 43)
    ]
    return {
        "format": METHOD_FORMATS[method],
        "object_count": 2,
        "record_count": 4,
        "seeds": [42, 43],
        "objects": rows,
    }


def metric_row(method: str, *, chamfer: float, score: float) -> dict:
    row = {
        "method": method,
        "object_key": "category:a",
        "seed": 42,
    }
    for field in SURFACE_FIELDS:
        row[field] = chamfer if field.startswith("chamfer") else score
    return row


def write_runtime_manifest(
    root: Path,
    *,
    version: int,
    image_value: int = 64,
    selected_indices: tuple[int, int] = (0, 1),
    reference_view_index: int = 0,
) -> tuple[Path, dict]:
    root.mkdir(parents=True)
    raw_cache = root.parent / "raw_cache.npz"
    raw_report = root.parent / "raw_cache_report.json"
    if not raw_cache.exists():
        raw_cache.write_bytes(b"same raw cache")
        raw_report.write_text('{"passed":true}\n', encoding="utf-8")
    rgb_paths = []
    mask_paths = []
    for index in range(2):
        rgb = root / f"view_{index:02d}_rgb.png"
        mask = root / f"view_{index:02d}_mask.png"
        Image.new("RGB", (4, 4), color=(image_value + index, 2, 3)).save(rgb)
        Image.new("L", (4, 4), color=255 - index).save(mask)
        rgb_paths.append(str(rgb))
        mask_paths.append(str(mask))
    row = {
        "format": f"pose_point_depth_mv.omni_real_runtime_input_object.v{version}",
        "category": "category",
        "object_id": "a",
        "object_key": "category:a",
        "source_raw_cache": str(raw_cache),
        "source_raw_cache_sha256": sha256_file(raw_cache),
        "selected_view_count": 2,
        "selected_source_view_indices": list(selected_indices),
        "selected_frame_names": ["000.jpg", "001.jpg"],
        "reference_view_index": reference_view_index,
        "prepared_rgb_paths": rgb_paths,
        "prepared_mask_paths": mask_paths,
        "forbidden_gt_fields_absent": True,
        "training_ready": False,
        "passed": True,
    }
    build_config = {
        "input_frontend_format": f"frontend.v{version}",
        "selected_view_count": 2,
        "view_selection": "even",
        "reference_view": "largest_mask",
        "feature_resolution": 518,
        "foreground_margin": 1.1,
        "alpha_threshold": 0.8,
    }
    manifest = {
        "format": f"pose_point_depth_mv.omni_real_runtime_input_manifest.v{version}",
        "raw_cache_report": str(raw_report),
        "raw_cache_report_sha256": sha256_file(raw_report),
        "build_config": build_config,
        "selected_object_count": 1,
        "completed_object_count": 1,
        "objects": [row],
        "failures": [],
        "passed": True,
    }
    path = root / "runtime_input_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


class OmniRealMeshBenchmarkTest(unittest.TestCase):
    def test_identical_mesh_has_exact_identity_metrics(self) -> None:
        mesh = trimesh.creation.box(extents=(0.7, 0.8, 0.9))
        metrics = surface_metrics(
            mesh, mesh.copy(), count=2048, seed=42, thresholds=(0.01, 0.02)
        )
        self.assertEqual(metrics["chamfer_l1"], 0.0)
        self.assertEqual(metrics["chamfer_l2"], 0.0)
        self.assertEqual(metrics["fscore_0p01"], 1.0)
        self.assertEqual(metrics["fscore_0p02"], 1.0)
        self.assertEqual(metrics["normal_consistency"], 1.0)

    def test_paired_improvement_is_positive_when_left_is_better(self) -> None:
        records = [
            metric_row("left", chamfer=0.1, score=0.9),
            metric_row("right", chamfer=0.2, score=0.8),
        ]
        comparison = _paired_delta(records, left="left", right="right")
        for values in comparison["metrics"].values():
            self.assertGreater(values["mean"], 0.0)
            self.assertEqual(values["positive_rate"], 1.0)

    def test_manifest_rejects_incomplete_object_seed_product(self) -> None:
        manifest = inference_manifest("native_v2_full")
        manifest["objects"].pop()
        manifest["record_count"] = 3
        with self.assertRaisesRegex(RuntimeError, "complete object/seed product"):
            _records_by_pair(manifest, method="native_v2_full")

    def test_three_methods_must_have_identical_coverage(self) -> None:
        rows = {
            method: _records_by_pair(inference_manifest(method), method=method)
            for method in METHOD_FORMATS
        }
        expected = require_identical_pair_coverage(
            rows, label_keys={"category:a", "category:b"}
        )
        self.assertEqual(len(expected), 4)

        rows["pixal3d_official"].pop(("category:b", 43))
        with self.assertRaisesRegex(RuntimeError, "identical object/seed pairs"):
            require_identical_pair_coverage(
                rows, label_keys={"category:a", "category:b"}
            )

    def test_external_runtime_v1_v2_visible_inputs_may_be_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_path, _ = write_runtime_manifest(root / "v1", version=1)
            reference_path, reference = write_runtime_manifest(root / "v2", version=2)
            binding = validate_method_runtime_binding(
                "reconviagen_original",
                {
                    "runtime_input_manifest": str(legacy_path),
                    "runtime_input_manifest_sha256": sha256_file(legacy_path),
                },
                reference_runtime_path=reference_path,
                reference_runtime=reference,
                reference_runtime_sha256=sha256_file(reference_path),
            )
            self.assertTrue(binding["passed"])
            self.assertEqual(
                binding["binding_mode"],
                "audited_external_visible_v1_v2_equivalence",
            )
            self.assertEqual(
                binding["external_visible_input_signature_sha256"],
                binding["reference_visible_input_signature_sha256"],
            )

    def test_external_runtime_rejects_changed_visible_inputs(self) -> None:
        mutations = {
            "image": {"image_value": 99},
            "selected view": {"selected_indices": (0, 2)},
            "reference view": {"reference_view_index": 1},
        }
        for label, kwargs in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                legacy_path, _ = write_runtime_manifest(
                    root / "v1", version=1, **kwargs
                )
                reference_path, reference = write_runtime_manifest(
                    root / "v2", version=2
                )
                with self.assertRaisesRegex(RuntimeError, "not external-input-equivalent"):
                    validate_method_runtime_binding(
                        "pixal3d_official",
                        {
                            "runtime_input_manifest": str(legacy_path),
                            "runtime_input_manifest_sha256": sha256_file(legacy_path),
                        },
                        reference_runtime_path=reference_path,
                        reference_runtime=reference,
                        reference_runtime_sha256=sha256_file(reference_path),
                    )

    def test_native_runtime_mismatch_remains_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_path, _ = write_runtime_manifest(root / "v1", version=1)
            reference_path, reference = write_runtime_manifest(root / "v2", version=2)
            with self.assertRaisesRegex(RuntimeError, "exact runtime-v2"):
                validate_method_runtime_binding(
                    "native_v2_full",
                    {
                        "runtime_input_manifest": str(legacy_path),
                        "runtime_input_manifest_sha256": sha256_file(legacy_path),
                    },
                    reference_runtime_path=reference_path,
                    reference_runtime=reference,
                    reference_runtime_sha256=sha256_file(reference_path),
                )


if __name__ == "__main__":
    unittest.main()
