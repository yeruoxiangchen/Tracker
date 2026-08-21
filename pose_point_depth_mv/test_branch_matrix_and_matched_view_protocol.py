from __future__ import annotations

from pathlib import Path

import pytest

from pose_point_depth_mv.analyze_pixal3d_multiview_branch_matrix import (
    resolve_stock_mesh,
    summarize_records,
)
from pose_point_depth_mv.freeze_matched_view_protocol import (
    VIEW_POSITIONS,
    stable_identity_seed,
)


def _record(case_id: str, stock: float, full: float, pixal: float):
    def method(chamfer: float):
        return {
            "surface": {
                "chamfer_l1": chamfer,
                "fscore_0p02": 1.0 - chamfer,
                "normal_consistency": 0.5,
                "precision_0p02": 0.6,
                "recall_0p02": 0.4,
            }
        }

    return {
        "case_id": case_id,
        "methods": {
            "corrected_ss_native_slat": method(stock),
            "current_full": method(full),
            "pixal3d_singleview": method(pixal),
        },
    }


def test_summary_positive_means_lhs_better_and_detects_dominant_case():
    summary = summarize_records(
        [
            _record("stable", stock=0.10, full=0.09, pixal=0.08),
            _record("collapse", stock=0.02, full=0.20, pixal=0.07),
        ]
    )
    comparison = summary["comparisons"]["full_minus_stock"]
    assert comparison["chamfer_improvement"]["mean"] < 0.0
    assert comparison["fscore_0p02_delta"]["mean"] < 0.0
    assert summary["full_stock_sensitivity"]["excluded_case"]["case_id"] == "collapse"


def test_resolve_stock_mesh_requires_expected_pair_layout(tmp_path: Path):
    full = tmp_path / "pair" / "full" / "mesh_canonical.obj"
    stock = tmp_path / "pair" / "stock" / "mesh_canonical.obj"
    full.parent.mkdir(parents=True)
    stock.parent.mkdir(parents=True)
    full.write_text("full", encoding="utf-8")
    stock.write_text("stock", encoding="utf-8")
    assert resolve_stock_mesh(full) == stock.resolve()
    with pytest.raises(ValueError):
        resolve_stock_mesh(tmp_path / "other.obj")


def test_noise_seed_is_object_stable_and_stage_separated():
    first = stable_identity_seed(object_uid="abc", joint_seed=42, stage="ss")
    assert first == stable_identity_seed(
        object_uid="abc", joint_seed=42, stage="ss"
    )
    assert first != stable_identity_seed(
        object_uid="abc", joint_seed=42, stage="slat"
    )
    assert first != stable_identity_seed(
        object_uid="def", joint_seed=42, stage="ss"
    )


def test_view_policy_is_strictly_nested():
    assert set(VIEW_POSITIONS[2]) < set(VIEW_POSITIONS[4])
    assert set(VIEW_POSITIONS[4]) < set(VIEW_POSITIONS[8])
