from __future__ import annotations

import torch

from pose_point_depth_mv.freeze_objaverse16_test import (
    expected_prior_views,
    select_samples,
)
from pose_point_depth_mv.prepare_objaverse16_no_vggt_model_inputs import (
    FORBIDDEN_MODEL_FIELDS,
    OBJECT_FORMAT,
    build_payload,
)


def _fake_manifest() -> dict:
    samples = []
    records = []
    groups = {"legacy897": 12, "gap_objaverse288": 5, "pilot_objaverse217": 5}
    for group, count in groups.items():
        for index in range(count):
            object_uid = f"{group}_{index:02d}"
            records.append(
                {
                    "object_uid": object_uid,
                    "dataset_source": "objaverse",
                    "source_group": group,
                }
            )
            for sequence in range(2):
                samples.append(
                    {
                        "uid": f"{object_uid}_seq{sequence:03d}",
                        "object_uid": object_uid,
                        "dataset_source": "objaverse",
                        "source_group": group,
                    }
                )
    return {"samples": samples, "object_records": records}


def test_objaverse16_selection_is_deterministic_and_quota_bound() -> None:
    quotas = {"legacy897": 10, "gap_objaverse288": 3, "pilot_objaverse217": 3}
    left, left_records = select_samples(_fake_manifest(), seed=20260810, quotas=quotas)
    right, right_records = select_samples(_fake_manifest(), seed=20260810, quotas=quotas)
    assert [row["uid"] for row in left] == [row["uid"] for row in right]
    assert left_records == right_records
    assert len(left) == 16
    assert len({row["object_uid"] for row in left}) == 16
    assert {group: sum(row["source_group"] == group for row in left) for group in quotas} == quotas


def test_expected_prior_views_replays_builder_rng() -> None:
    assert expected_prior_views(
        count=16, seed=20260810, choices=[2, 4, 8]
    ) == [8, 4, 8, 8, 4, 4, 2, 4, 2, 8, 8, 4, 2, 2, 8, 8]


def test_model_payload_excludes_targets_points_and_vggt() -> None:
    views = 1
    patches = torch.zeros((views, 1369, 1024), dtype=torch.float16)
    context = torch.zeros((1, 1369, 1024), dtype=torch.float16)
    source = {
        "format": "ar_ss_flow.pose_lifting_cache.v1",
        "uid": "object_seq000",
        "object_uid": "object",
        "visual_patch_features": patches,
        "stock_condition": context,
        "slat_condition": {"cond": [context], "neg_cond": [context.clone()]},
        "intrinsics": torch.eye(3).unsqueeze(0),
        "extrinsics": torch.eye(4).unsqueeze(0),
        "grid_transform": "pixal3d_rotation",
        "extrinsics_type": "c2w",
        "camera_forward_sign": 1.0,
        "feature_image_size": [518, 518],
        "dino_only_context_contract": {"vggt_model_executed": False},
        "ss_latent": "/forbidden/target.npz",
        "prior_coords": torch.zeros((3, 3), dtype=torch.int32),
    }
    payload = build_payload(
        source,
        uid="object_seq000",
        object_uid="object",
        source_group="legacy897",
    )
    assert payload["format"] == OBJECT_FORMAT
    assert not (FORBIDDEN_MODEL_FIELDS & set(payload))
    assert payload["point_cloud_tensor_present"] is False
    assert payload["target_or_mesh_consumed"] is False
    assert payload["vggt_model_executed"] is False
