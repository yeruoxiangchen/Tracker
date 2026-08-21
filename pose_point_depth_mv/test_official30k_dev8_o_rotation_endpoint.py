#!/usr/bin/env python3

import numpy as np
import torch

from pose_point_depth_mv.evaluate_official30k_dev8_o_rotation_endpoint import (
    ARMS,
    arm_to_official,
    rotate_lifting_sample,
)


def _sample(kind: str) -> dict:
    w2c = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
    centers = np.asarray(
        ((2.0, 0.0, 0.2), (0.0, 2.0, 0.1), (-2.0, 0.0, -0.1), (0.0, -2.0, 0.0)),
        dtype=np.float32,
    )
    # Only centers are used by the phone-axis estimator in this test.
    w2c[:, :3, 3] = -centers
    extrinsics = w2c if kind == "w2c" else np.linalg.inv(w2c)
    return {
        "extrinsics": torch.from_numpy(extrinsics),
        "extrinsics_type": kind,
    }


def test_all_arms_are_proper() -> None:
    sample = _sample("w2c")
    for arm in ARMS:
        transform, _ = arm_to_official(arm, sample)
        rotation = transform[:3, :3]
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6)


def test_w2c_reexpression_preserves_camera_points() -> None:
    sample = _sample("w2c")
    transform, _ = arm_to_official("official_o_rx90", sample)
    rotated = rotate_lifting_sample(sample, transform)
    point_arm = np.asarray((0.1, -0.2, 0.3, 1.0), dtype=np.float32)
    point_official = transform.astype(np.float32) @ point_arm
    expected = sample["extrinsics"].numpy() @ point_official
    actual = rotated["extrinsics"].numpy() @ point_arm
    assert np.allclose(actual, expected, atol=1.0e-6)


def test_c2w_reexpression_preserves_camera_pose() -> None:
    sample = _sample("c2w")
    transform, _ = arm_to_official("official_o_ry90", sample)
    rotated = rotate_lifting_sample(sample, transform)
    expected = np.linalg.inv(transform).astype(np.float32) @ sample["extrinsics"].numpy()
    assert np.allclose(rotated["extrinsics"].numpy(), expected, atol=1.0e-6)


if __name__ == "__main__":
    test_all_arms_are_proper()
    test_w2c_reexpression_preserves_camera_points()
    test_c2w_reexpression_preserves_camera_pose()
    print("PASS")
