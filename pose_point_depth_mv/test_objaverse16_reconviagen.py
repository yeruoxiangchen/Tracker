from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from pose_point_depth_mv.evaluate_objaverse16_reconviagen import (
    decoder_to_source_mesh,
    paired_comparison,
)
from pose_point_depth_mv.render_direct_slat_fourway import (
    LATENT_DECODER_TO_REFERENCE,
)


def _record(chamfer: float, fscore: float) -> dict:
    return {
        "surface": {
            "chamfer_l1": chamfer,
            "fscore_0p02": fscore,
        }
    }


class Objaverse16ReconViaGenTest(unittest.TestCase):
    def test_decoder_axis_transform_matches_transform_pose_convention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decoder.obj"
            source = trimesh.creation.box(extents=(0.2, 0.4, 0.8))
            source.apply_translation((0.1, -0.3, 0.2))
            source.export(path)
            loaded = decoder_to_source_mesh(path)
            expected = source.copy()
            expected.apply_transform(LATENT_DECODER_TO_REFERENCE)
            np.testing.assert_allclose(loaded.bounds, expected.bounds, atol=1.0e-8)
            np.testing.assert_allclose(loaded.extents, [0.2, 0.8, 0.4], atol=1.0e-8)

    def test_paired_delta_is_positive_when_current_is_better(self) -> None:
        current = {
            "a": _record(0.10, 0.60),
            "b": _record(0.30, 0.20),
        }
        recon = {
            "a": _record(0.20, 0.50),
            "b": _record(0.20, 0.40),
        }
        result = paired_comparison(current, recon)
        self.assertEqual(
            result["chamfer_l1_wins"],
            {"current_no_vggt": 1, "reconviagen_original": 1, "ties": 0},
        )
        chamfer = result["metric_deltas"]["chamfer_l1_current_improvement"]
        fscore = result["metric_deltas"]["fscore_0p02_current_improvement"]
        self.assertAlmostEqual(chamfer["mean"], 0.0)
        self.assertAlmostEqual(fscore["mean"], -0.05)
        self.assertEqual(chamfer["positive_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
