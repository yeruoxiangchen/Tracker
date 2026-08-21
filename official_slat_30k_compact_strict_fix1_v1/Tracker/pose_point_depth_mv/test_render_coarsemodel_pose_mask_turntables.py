from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from pose_point_depth_mv.bunny_review.common import binding
from pose_point_depth_mv.render_coarsemodel_pose_mask_turntables import (
    FORMAT,
    reusable_turntable,
)


class CoarseModelPoseMaskTurntableTest(unittest.TestCase):
    def test_reusable_turntable_binds_source_config_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mesh.obj"
            video = root / "video.mp4"
            sheet = root / "sheet.png"
            source.write_text("mesh", encoding="utf-8")
            video.write_bytes(b"video")
            sheet.write_bytes(b"sheet")
            config = {"render_frames": 48}
            report = {
                "format": FORMAT,
                "source_mesh": binding(source),
                "video": binding(video),
                "contact_sheet": binding(sheet),
                "render_config": config,
                "passed": True,
            }
            self.assertTrue(
                reusable_turntable(
                    report,
                    source_mesh=binding(source),
                    render_config=config,
                )
            )
            video.write_bytes(b"changed")
            self.assertFalse(
                reusable_turntable(
                    report,
                    source_mesh=binding(source),
                    render_config=config,
                )
            )
