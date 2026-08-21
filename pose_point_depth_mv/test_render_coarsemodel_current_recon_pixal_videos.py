from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import imageio.v2 as imageio
import numpy as np

from pose_point_depth_mv.render_coarsemodel_current_recon_pixal_videos import (
    autoframe_video,
    fit_square,
    largest_chroma_component_bbox,
    resample_video,
)


class CoarseModelThreewayVideoTests(unittest.TestCase):
    def test_fit_and_temporal_resample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            frames = [np.full((32, 48, 3), index * 30, dtype=np.uint8) for index in range(4)]
            imageio.mimsave(source, frames, fps=4)
            output = resample_video(source, count=7, resolution=32)
            self.assertEqual(len(output), 7)
            self.assertTrue(all(frame.shape == (32, 32, 3) for frame in output))
            fitted = fit_square(np.zeros((10, 20, 3), dtype=np.uint8), 40)
            self.assertEqual(fitted.shape, (40, 40, 3))

    def test_autoframe_equalizes_foreground_without_stretching(self) -> None:
        source_shapes = [(20, 40), (52, 26), (12, 18)]
        resulting_bboxes = []
        for height, width in source_shapes:
            frame = np.full((96, 96, 3), 128, dtype=np.uint8)
            top = (96 - height) // 2
            left = (96 - width) // 2
            frame[top : top + height, left : left + width] = (220, 30, 80)
            adjusted, report = autoframe_video(
                [frame, frame],
                target_fill=0.72,
                chroma_threshold=18,
            )
            self.assertTrue(report["fixed_crop_across_frames"])
            bbox = largest_chroma_component_bbox(
                adjusted[0],
                chroma_threshold=18,
            )
            self.assertIsNotNone(bbox)
            assert bbox is not None
            resulting_bboxes.append(bbox)
            output_height = bbox[3] - bbox[1]
            output_width = bbox[2] - bbox[0]
            self.assertAlmostEqual(output_width / output_height, width / height, delta=0.12)
        fills = [max(x1 - x0, y1 - y0) / 96.0 for x0, y0, x1, y1 in resulting_bboxes]
        self.assertLess(max(fills) - min(fills), 0.04)
        for fill in fills:
            self.assertAlmostEqual(fill, 0.72, delta=0.04)

    def test_autoframe_uses_fixed_union_crop_and_right_roi(self) -> None:
        frames = []
        for offset in (0, 8):
            frame = np.full((80, 80, 3), 128, dtype=np.uint8)
            frame[25:40, 5:20] = (220, 30, 80)
            frame[30:50, 52 + offset : 62 + offset] = (30, 220, 80)
            frames.append(frame)
        adjusted, report = autoframe_video(
            frames,
            target_fill=0.7,
            chroma_threshold=18,
            roi_x_fraction=(0.5, 1.0),
        )
        self.assertEqual(report["foreground_bbox_xyxy_exclusive"], [52, 30, 70, 50])
        self.assertEqual(len({frame.shape for frame in adjusted}), 1)

    def test_autoframe_rejects_background_only_video(self) -> None:
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "no colorful foreground"):
            autoframe_video([frame], target_fill=0.72, chroma_threshold=18)


if __name__ == "__main__":
    unittest.main()
