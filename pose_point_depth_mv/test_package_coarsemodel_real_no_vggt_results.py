from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh

from pose_point_depth_mv.package_coarsemodel_real_no_vggt_results import (
    export_world_mesh,
    input_sheet,
)


class CoarseModelResultPackageTests(unittest.TestCase):
    def test_world_export_and_masked_input_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "mesh_o.obj"
            world = root / "mesh_w.obj"
            trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(source)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = [4.0, 5.0, 6.0]
            binding = export_world_mesh(source, world, transform)
            self.assertEqual(Path(binding["path"]), world.resolve())
            loaded = trimesh.load(world, force="mesh", process=False)
            np.testing.assert_allclose(loaded.centroid, [4.0, 5.0, 6.0], atol=1.0e-6)

            images = []
            masks = []
            for index in range(2):
                image_path = root / f"image_{index}.png"
                mask_path = root / f"mask_{index}.png"
                Image.new("RGB", (24, 20), (100, 120, 140)).save(image_path)
                mask = np.zeros((20, 24), dtype=np.uint8)
                mask[4:16, 5:19] = 255
                Image.fromarray(mask).save(mask_path)
                images.append(image_path)
                masks.append(mask_path)
            sheet_path = root / "sheet.png"
            sheet_binding = input_sheet(images, masks, sheet_path)
            self.assertEqual(Path(sheet_binding["path"]), sheet_path.resolve())
            with Image.open(sheet_path) as sheet:
                self.assertEqual(sheet.size, (1280, 346))


if __name__ == "__main__":
    unittest.main()
