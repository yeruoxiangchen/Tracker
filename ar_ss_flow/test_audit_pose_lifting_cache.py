from __future__ import annotations

import unittest

import torch

from ar_ss_flow.audit_pose_lifting_cache import compare_cached_geometry


def example_geometry() -> dict[str, torch.Tensor]:
    return {
        "image_grid": torch.zeros((2, 8, 2), dtype=torch.float32),
        "patch_grid": torch.ones((2, 8, 2), dtype=torch.float32),
        "camera_depth": torch.full((2, 8), 2.0, dtype=torch.float32),
        "valid": torch.ones((2, 8), dtype=torch.bool),
    }


class CachedGeometryAuditTests(unittest.TestCase):
    def test_exact_geometry_passes(self) -> None:
        fresh = example_geometry()
        cached = {name: value.clone() for name, value in fresh.items()}
        report = compare_cached_geometry(cached, fresh, max_abs_diff=1.0e-5)
        self.assertTrue(report["passed"])
        self.assertEqual(report["max_abs_diff"], 0.0)
        self.assertEqual(report["valid_mismatch_count"], 0)

    def test_numeric_corruption_fails(self) -> None:
        fresh = example_geometry()
        cached = {name: value.clone() for name, value in fresh.items()}
        cached["patch_grid"][0, 0, 0] += 1.0e-3
        report = compare_cached_geometry(cached, fresh, max_abs_diff=1.0e-5)
        self.assertFalse(report["passed"])
        self.assertAlmostEqual(report["max_abs_diff"], 1.0e-3, places=6)
        self.assertFalse(report["fields"]["patch_grid"]["passed"])

    def test_valid_bit_flip_fails(self) -> None:
        fresh = example_geometry()
        cached = {name: value.clone() for name, value in fresh.items()}
        cached["valid"][0, 0] = False
        report = compare_cached_geometry(cached, fresh, max_abs_diff=1.0e-5)
        self.assertFalse(report["passed"])
        self.assertEqual(report["valid_mismatch_count"], 1)

    def test_missing_geometry_or_key_fails(self) -> None:
        fresh = example_geometry()
        missing_geometry = compare_cached_geometry(
            None, fresh, max_abs_diff=1.0e-5
        )
        self.assertFalse(missing_geometry["passed"])
        cached = {name: value.clone() for name, value in fresh.items()}
        del cached["camera_depth"]
        missing_key = compare_cached_geometry(
            cached, fresh, max_abs_diff=1.0e-5
        )
        self.assertFalse(missing_key["passed"])
        self.assertEqual(missing_key["missing_keys"], ["camera_depth"])

    def test_wrong_dtype_shape_and_nonfinite_fail(self) -> None:
        fresh = example_geometry()
        wrong_dtype = {name: value.clone() for name, value in fresh.items()}
        wrong_dtype["image_grid"] = wrong_dtype["image_grid"].double()
        self.assertFalse(
            compare_cached_geometry(
                wrong_dtype, fresh, max_abs_diff=1.0e-5
            )["passed"]
        )

        wrong_shape = {name: value.clone() for name, value in fresh.items()}
        wrong_shape["camera_depth"] = wrong_shape["camera_depth"][:, :-1]
        self.assertFalse(
            compare_cached_geometry(
                wrong_shape, fresh, max_abs_diff=1.0e-5
            )["passed"]
        )

        nonfinite = {name: value.clone() for name, value in fresh.items()}
        nonfinite["patch_grid"][0, 0, 0] = float("nan")
        self.assertFalse(
            compare_cached_geometry(
                nonfinite, fresh, max_abs_diff=1.0e-5
            )["passed"]
        )


if __name__ == "__main__":
    unittest.main()
