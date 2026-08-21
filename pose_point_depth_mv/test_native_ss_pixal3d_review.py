from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from pose_point_depth_mv.prepare_native_ss_pixal3d_review import (
    DEFAULT_SOURCE_VIEW_TARGETS,
    parse_source_view_targets,
    selected_rows,
    selection_rank,
)
from pose_point_depth_mv.render_native_ss_pixal3d_fourway import parse_review_modes


class NativeSSPixal3DReviewTest(unittest.TestCase):
    def test_default_source_view_targets_are_stable(self) -> None:
        text = ",".join(
            f"{source}:{views}" for source, views in DEFAULT_SOURCE_VIEW_TARGETS
        )
        self.assertEqual(parse_source_view_targets(text), list(DEFAULT_SOURCE_VIEW_TARGETS))
        self.assertEqual(
            selection_rank(20260802, "omni", 4, "omni_banana_008_seq001"),
            "5f5afc06ea3faa321ef6c3db8c3527d51dd4592401a66bceed0ef7d12ad1ae10",
        )

    def test_duplicate_source_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "each source"):
            parse_source_view_targets("omni:2,omni:4")

    def test_gt_size_guard_precedes_random_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            large_mesh = root / "large.obj"
            small_mesh = root / "small.obj"
            large_mesh.write_bytes(b"x" * 101)
            small_mesh.write_bytes(b"x" * 20)
            large_latent = root / "large.npz"
            small_latent = root / "small.npz"
            np.savez(large_latent, source_glb=str(large_mesh))
            np.savez(small_latent, source_glb=str(small_mesh))
            split = {
                "phases": {
                    "final": [
                        {
                            "uid": "large",
                            "source": "omni",
                            "view_count": 4,
                        },
                        {
                            "uid": "small",
                            "source": "omni",
                            "view_count": 4,
                        },
                    ]
                }
            }
            selected = selected_rows(
                split=split,
                manifest_by_uid={
                    "large": {"ss_latent": str(large_latent)},
                    "small": {"ss_latent": str(small_latent)},
                },
                targets=[("omni", 4)],
                selection_seed=20260802,
                max_gt_source_bytes=100,
            )
            self.assertEqual([row["uid"] for row in selected], ["small"])
            self.assertEqual(selected[0]["gt_source_mesh_size_bytes"], 20)

    def test_review_mode_order_is_canonical_then_shape(self) -> None:
        self.assertEqual(
            parse_review_modes("shape_aligned,canonical_pose"),
            ["canonical_pose", "shape_aligned"],
        )


if __name__ == "__main__":
    unittest.main()
