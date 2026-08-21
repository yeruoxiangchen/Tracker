#!/usr/bin/env python3
"""Unit checks for the official-compatible model-O coordinate conversion."""

from __future__ import annotations

import numpy as np

from pose_point_depth_mv.coarsemodel_snoopy_official_compatible_model_o import (
    _transform_runtime_arrays,
    model_to_source_runtime_o,
)


def main() -> None:
    rng = np.random.default_rng(20260819)
    scale = 3.25
    old_o2w = np.eye(4)
    old_o2w[:3, :3] *= scale
    old_o2w[:3, 3] = (1.0, -2.0, 0.5)
    w2c = np.repeat(np.eye(4)[None], 3, axis=0)
    w2c[:, :3, 3] = rng.normal(size=(3, 3))
    old_o2c = w2c @ old_o2w[None]
    old_lifting = old_o2c.copy()
    old_lifting[:, :3] /= scale
    source = {
        "T_O2W": old_o2w,
        "T_W2O": np.linalg.inv(old_o2w),
        "T_O2C": old_o2c,
        "T_O2C_lifting": old_lifting,
        "T_C2O": np.linalg.inv(old_o2c),
        "P_O": rng.normal(size=(11, 3)).astype(np.float32),
        "K_feature": np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0),
    }
    transformed, audit = _transform_runtime_arrays(source)
    q = model_to_source_runtime_o()
    assert np.allclose(transformed["T_O2W"], old_o2w @ q)
    assert np.allclose(transformed["T_O2C"], w2c @ transformed["T_O2W"])
    assert np.allclose(transformed["P_O"] @ q[:3, :3].T, source["P_O"])
    assert all(audit["checks"].values())
    assert np.isclose(np.linalg.det(q[:3, :3]), 1.0)
    print("PASS")


if __name__ == "__main__":
    main()
