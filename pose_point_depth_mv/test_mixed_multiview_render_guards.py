import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from pose_point_depth_mv.dataset_tools.build_objaverse_multiview_sparse_data import (
    SourceAssetIncompatibleError,
    circular_azimuth_stats,
    classify_failure,
    look_at_c2w,
    make_ar_random_trajectory,
    make_camera_ring,
    object_azimuth_balanced_indices,
    select_multiview_frames,
    validate_blender_bounds,
)
from pose_point_depth_mv.dataset_tools.render_failure_taxonomy import (
    canonical_failure_class,
)
from pose_point_depth_mv.dataset_tools.run_mixed_multiview_render_worker import (
    builder_command,
    validate_render_inventory,
)


class MixedMultiviewRenderGuardTest(unittest.TestCase):
    @staticmethod
    def _selection_args(**overrides):
        values = {
            "selected_views": 8,
            "frame_selection_policy": "object_azimuth_balanced",
            "min_good_candidate_views": 8,
            "selection_min_fg_pixels_per_view": 1,
            "selection_min_fg_area_ratio": 0.001,
            "selection_min_bbox_margin_px": 1.0,
            "selection_max_bbox_area_ratio": 0.95,
            "selection_target_fg_area_ratio": 0.06,
            "selection_margin_score_px": 64.0,
            "selection_diversity_weight": 0.75,
            "min_fg_pixels_per_view": 1,
            "min_fg_area_ratio": 0.001,
            "min_bbox_margin_px": 1.0,
            "max_bbox_area_ratio": 0.95,
            "min_selected_azimuth_coverage": 240.0,
            "max_selected_azimuth_gap": 120.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _opaque_candidates(count: int):
        image = np.full((64, 64, 3), 127, dtype=np.uint8)
        alpha = np.zeros((64, 64), dtype=np.uint8)
        alpha[8:56, 8:56] = 255
        return [(image.copy(), alpha.copy()) for _ in range(count)]

    def test_object_azimuth_selection_is_uniform_on_full_ring(self) -> None:
        c2w = make_camera_ring(24, radius=2.0, elev_deg=20.0)
        selected, stats = object_azimuth_balanced_indices(
            range(24), c2w, np.ones(24, dtype=np.float32), 8
        )
        self.assertEqual(len(selected), 8)
        self.assertAlmostEqual(stats["maximum_azimuth_gap_degrees"], 45.0)
        self.assertAlmostEqual(stats["azimuth_coverage_degrees"], 315.0)

        selected_again, stats_again = select_multiview_frames(
            self._opaque_candidates(24), c2w, self._selection_args()
        )
        self.assertEqual(selected_again, selected)
        self.assertGreaterEqual(stats_again["azimuth_coverage_degrees"], 240.0)
        self.assertLessEqual(stats_again["maximum_azimuth_gap_degrees"], 120.0)

    def test_narrow_arc_is_rejected_by_wide_coverage_contract(self) -> None:
        cameras = []
        for angle in np.linspace(-90.0, 90.0, 24):
            radians = np.radians(angle)
            eye = np.asarray(
                [2.0 * np.sin(radians), 0.4, 2.0 * np.cos(radians)],
                dtype=np.float32,
            )
            cameras.append(look_at_c2w(eye))
        c2w = np.stack(cameras)
        with self.assertRaisesRegex(ValueError, "object-azimuth coverage"):
            select_multiview_frames(
                self._opaque_candidates(24), c2w, self._selection_args()
            )
        angles = np.degrees(
            np.arctan2(c2w[:, 0, 3], c2w[:, 2, 3])
        )
        self.assertLess(circular_azimuth_stats(angles)["azimuth_coverage_degrees"], 240.0)

    def test_wide_ar_profile_satisfies_coverage_across_random_seeds(self) -> None:
        trajectory_args = SimpleNamespace(
            radius_min=1.35,
            radius_max=3.0,
            azimuth_span_min=285.0,
            azimuth_span_max=330.0,
            elevation_min=-10.0,
            elevation_max=45.0,
            elevation_drift=20.0,
            radius_drift=0.20,
            target_jitter=0.12,
            azimuth_jitter=2.0,
            elevation_jitter=3.0,
            radius_jitter=0.08,
            camera_lateral_jitter=0.06,
            lookat_jitter=0.08,
            roll_jitter=8.0,
        )
        for seed in range(100):
            c2w = make_ar_random_trajectory(
                24, trajectory_args, np.random.default_rng(seed)
            )
            _selected, stats = object_azimuth_balanced_indices(
                range(24), c2w, np.ones(24, dtype=np.float32), 8
            )
            self.assertGreaterEqual(stats["azimuth_coverage_degrees"], 240.0)
            self.assertLessEqual(stats["maximum_azimuth_gap_degrees"], 120.0)

    def test_wide_worker_command_binds_trajectory_contract(self) -> None:
        args = SimpleNamespace(
            python="/env/python",
            max_objects_per_shard=0,
            seed=20260811,
            sequences_per_object=2,
            vis_count_per_shard=8,
            blender_path="/tmp/blender",
            blender_engine="CYCLES",
            blender_samples=16,
            blender_bounds_tolerance=0.001,
            blender_cycles_device="CUDA",
            candidate_views=24,
            selected_views=8,
            trajectory_profile="wide_ar",
            xvfb_run_path=None,
        )
        command = builder_command(
            args,
            Path("/tmp/source.json"),
            Path("/tmp/render"),
            Path("/tmp/preview"),
        )
        pairs = dict(zip(command, command[1:]))
        self.assertEqual(pairs["--frame_selection_policy"], "object_azimuth_balanced")
        self.assertEqual(pairs["--azimuth_span_min"], "285.0")
        self.assertEqual(pairs["--azimuth_span_max"], "330.0")
        self.assertEqual(pairs["--min_selected_azimuth_coverage"], "240.0")
        self.assertEqual(pairs["--max_selected_azimuth_gap"], "120.0")

    def test_blender_bounds_audit_accepts_matching_import_frame(self) -> None:
        vertices = np.asarray(
            [
                [-0.2, -0.4, -0.1],
                [0.3, 0.4, 0.1],
            ],
            dtype=np.float32,
        )
        metadata = {
            "normalized_scene_bounds": {
                "minimum": [-0.2, -0.4, -0.1],
                "maximum": [0.3, 0.4, 0.1],
            },
            "source_suffix": ".obj",
            "source_import_policy": "obj_forward_neg_z_up_y_to_blender_v1",
            "normalization_policy": "imported_frame_center_scale_v2",
        }
        audit = validate_blender_bounds(metadata, vertices, 1.0e-3)
        self.assertLessEqual(audit["normalized_scene_bounds_max_abs"], 1.0e-6)
        self.assertEqual(
            audit["source_import_policy"],
            "obj_forward_neg_z_up_y_to_blender_v1",
        )

    def test_blender_bounds_audit_rejects_axis_swap(self) -> None:
        vertices = np.asarray(
            [
                [-0.2, -0.4, -0.1],
                [0.2, 0.4, 0.1],
            ],
            dtype=np.float32,
        )
        metadata = {
            "normalized_scene_bounds": {
                "minimum": [-0.2, -0.1, -0.4],
                "maximum": [0.2, 0.1, 0.4],
            }
        }
        with self.assertRaises(SourceAssetIncompatibleError):
            validate_blender_bounds(metadata, vertices, 1.0e-3)

    def test_source_bounds_mismatch_is_asset_rejection(self) -> None:
        error = (
            "SourceAssetIncompatibleError: source asset bounds incompatible "
            "between Blender and trimesh: max_abs=56.0"
        )
        self.assertEqual(classify_failure(error), "source_asset_rejected")

    def test_real_blender_failure_remains_infrastructure_failure(self) -> None:
        error = "RuntimeError: Blender render failed (exit_code=134)"
        self.assertEqual(classify_failure(error), "renderer_failed")

    def test_legacy_empty_scene_is_canonical_asset_rejection(self) -> None:
        failure = {
            "failure_class": "other_failed",
            "error": "ValueError: scene has no mesh geometry",
        }
        self.assertEqual(
            canonical_failure_class(failure),
            "source_asset_rejected",
        )

    def _validate(
        self,
        failure_class: str,
        error: str = "",
    ) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a = root / "a.glb"
            source_b = root / "b.glb"
            source_a.write_bytes(b"a")
            source_b.write_bytes(b"b")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "uid": "a_seq000",
                                "object_uid": "a",
                                "source_glb": str(source_a),
                            }
                        ],
                        "failures": [
                            {
                                "uid": "a",
                                "sequence_idx": 1,
                                "source_glb": str(source_a),
                                "failure_class": "frame_selection_rejected",
                            },
                            {
                                "uid": "b",
                                "sequence_idx": 0,
                                "source_glb": str(source_b),
                                "failure_class": failure_class,
                                "error": error,
                            },
                            {
                                "uid": "b",
                                "sequence_idx": 1,
                                "source_glb": str(source_b),
                                "failure_class": failure_class,
                                "error": error,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return validate_render_inventory(
                manifest,
                {"a": str(source_a), "b": str(source_b)},
                sequences_per_object=2,
            )

    def test_asset_rejections_do_not_fail_worker_inventory(self) -> None:
        passed, reason = self._validate("source_asset_rejected")
        self.assertTrue(passed, reason)

    def test_renderer_failures_still_fail_worker_inventory(self) -> None:
        passed, reason = self._validate("renderer_failed")
        self.assertFalse(passed)
        self.assertIn("infrastructure failures", reason)

    def test_legacy_empty_scene_does_not_fail_worker_inventory(self) -> None:
        passed, reason = self._validate(
            "other_failed",
            "ValueError: scene has no mesh geometry",
        )
        self.assertTrue(passed, reason)

    def test_unknown_other_failure_still_fails_worker_inventory(self) -> None:
        passed, reason = self._validate(
            "other_failed",
            "RuntimeError: unexpected failure",
        )
        self.assertFalse(passed)
        self.assertIn("first_raw_class=other_failed", reason)


if __name__ == "__main__":
    unittest.main()
