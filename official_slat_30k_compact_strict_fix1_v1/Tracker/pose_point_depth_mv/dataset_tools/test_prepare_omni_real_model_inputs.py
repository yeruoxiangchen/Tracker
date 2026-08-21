from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from pose_point_depth_mv.dataset_tools.prepare_omni_real_model_inputs import (
    load_runtime_lifting_geometry,
)
from pose_point_depth_mv.real_object_canonicalization import (
    normalize_similarity_extrinsics,
)


class PrepareOmniRealModelInputsTest(unittest.TestCase):
    def _write_cache(self, path: Path, *, include_lifting: bool = True) -> np.ndarray:
        physical = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
        physical[:, :3, :3] *= 2.5
        physical[:, :3, 3] = np.asarray(
            [[0.2, -0.1, 4.0], [0.0, 0.3, 3.5], [-0.4, 0.1, 4.2]]
        )
        values = {
            "K_feature": np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
            "T_O2C": physical,
            "P_O": np.zeros((12, 3), dtype=np.float32),
        }
        if include_lifting:
            values["T_O2C_lifting"] = normalize_similarity_extrinsics(physical)
        np.savez_compressed(path, **values)
        return physical

    def test_loader_consumes_normalized_lifting_extrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.npz"
            physical = self._write_cache(path)
            _, extrinsics, _ = load_runtime_lifting_geometry(path)
            np.testing.assert_allclose(
                extrinsics,
                normalize_similarity_extrinsics(physical).astype(np.float32),
            )
            self.assertFalse(np.allclose(extrinsics, physical.astype(np.float32)))

    def test_loader_rejects_v1_cache_without_lifting_extrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_v1.npz"
            self._write_cache(path, include_lifting=False)
            with self.assertRaisesRegex(RuntimeError, "lacks v2 lifting geometry"):
                load_runtime_lifting_geometry(path)

    def test_loader_rejects_tampered_lifting_extrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_bad.npz"
            physical = self._write_cache(path)
            lifting = normalize_similarity_extrinsics(physical)
            lifting[0, 0, 3] += 0.1
            np.savez_compressed(
                path,
                K_feature=np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
                T_O2C=physical,
                T_O2C_lifting=lifting,
                P_O=np.zeros((12, 3), dtype=np.float32),
            )
            with self.assertRaisesRegex(RuntimeError, "contract changed"):
                load_runtime_lifting_geometry(path)

if __name__ == "__main__":
    unittest.main()
