from __future__ import annotations

from pathlib import Path

import numpy as np

from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (
    INFERENCE_RESULT_FORMAT,
    OFFICIAL_GEOMETRY_EXPORT,
    OFFICIAL_POSTPROCESS,
    pixal3d_mesh_path,
    pixal3d_result_path,
    select_protocol_cases,
    sha256_file,
    validate_official_inference_result,
)
from pose_point_depth_mv.evaluate_direct_slat_pixal3d_utility import (
    METRICS,
    OFFICIAL_FINAL_EXPORT_FROM_DECODED,
    aggregate_utility,
    case_canonical_transform,
    fixed_float_value,
    utility_decision,
)


def _surface(chamfer: float, fscore: float) -> dict[str, float]:
    values = {
        "chamfer_l1": chamfer,
        "pred_to_gt_mean": chamfer * 0.8,
        "gt_to_pred_mean": chamfer * 1.2,
        "fscore_0p02": fscore,
        "normal_consistency": fscore,
        "precision_0p02": fscore,
        "recall_0p02": fscore,
    }
    assert set(values) == set(METRICS)
    return values


def _case(
    object_uid: str,
    seed: int,
    *,
    stock: tuple[float, float],
    full: tuple[float, float],
    pixal: tuple[float, float],
    pixal_success: bool = True,
):
    methods = {
        "stock": {"success": True, "surface": _surface(*stock)},
        "full": {"success": True, "surface": _surface(*full)},
        "pixal3d_native": (
            {"success": True, "surface": _surface(*pixal)}
            if pixal_success
            else {"success": False, "error": "missing"}
        ),
    }
    return {"object_uid": object_uid, "joint_seed": seed, "methods": methods}


def test_object_aggregation_uses_positive_lhs_better_signs():
    rows = [
        _case(
            "a",
            seed,
            stock=(0.10, 0.50),
            full=(0.08, 0.60),
            pixal=(0.09, 0.55),
        )
        for seed in (42, 43)
    ]
    summary = aggregate_utility(
        rows,
        expected_seeds=[42, 43],
        bootstrap_samples=100,
        seed=1,
    )
    u1 = summary["comparisons"]["u1_full_minus_stock"]["metrics"]
    u2 = summary["comparisons"]["u2_full_minus_pixal3d_native"]["metrics"]
    assert u1["chamfer_l1"]["mean"] > 0.0
    assert u1["fscore_0p02"]["mean"] > 0.0
    assert u2["chamfer_l1"]["mean"] > 0.0
    assert u2["fscore_0p02"]["mean"] > 0.0
    assert summary["object_count"] == 1


def test_pixal_failure_is_counted_and_blocks_u2():
    rows = [
        _case(
            f"object-{index}",
            42,
            stock=(0.10, 0.50),
            full=(0.08, 0.60),
            pixal=(0.09, 0.55),
            pixal_success=index != 0,
        )
        for index in range(32)
    ]
    summary = aggregate_utility(
        rows,
        expected_seeds=[42],
        bootstrap_samples=100,
        seed=2,
    )
    u2 = summary["comparisons"]["u2_full_minus_pixal3d_native"]
    assert u2["complete_object_count"] == 31
    assert u2["failed_object_count"] == 1
    decision = utility_decision(summary, profile="exploratory")
    assert decision["u1_full_gt_stock"]["passed"] is True
    assert decision["u2_full_gt_pixal3d_native"]["passed"] is False
    assert decision["passed"] is False


def test_smoke_only_uses_u1_to_allow_exploratory():
    rows = [
        _case(
            f"object-{index}",
            42,
            stock=(0.10, 0.50),
            full=(0.08, 0.60),
            pixal=(0.07, 0.70),
        )
        for index in range(6)
    ]
    summary = aggregate_utility(
        rows,
        expected_seeds=[42],
        bootstrap_samples=100,
        seed=3,
    )
    decision = utility_decision(summary, profile="smoke")
    assert decision["u1_full_gt_stock"]["passed"] is True
    assert decision["u2_full_gt_pixal3d_native"]["passed"] is False
    assert decision["passed"] is True
    assert decision["continue_to_next_stage"] is True
    assert decision["long_training_unlocked"] is False


def test_case_transform_is_metadata_only_finite_and_invertible():
    case = {
        "case_id": "example",
        "selected_frame": {
            "extrinsics_type": "c2w",
            "extrinsic": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    }
    matrix = case_canonical_transform(
        case,
        {"derivation": {"uniform_scale": 0.9}},
    )
    assert matrix.shape == (4, 4)
    assert abs(float(np.linalg.det(matrix[:3, :3]))) > 0.0
    assert np.allclose(matrix[:3, 3], np.zeros(3), rtol=0.0, atol=1.0e-12)


def test_official_final_export_composes_both_axis_conversions():
    expected = np.diag([-1.0, 1.0, -1.0, 1.0])
    assert np.allclose(OFFICIAL_FINAL_EXPORT_FROM_DECODED, expected)


def test_official_output_paths_do_not_alias_legacy_raw_mesh(tmp_path: Path):
    protocol_path = tmp_path / "protocol.json"
    mesh_path = pixal3d_mesh_path(protocol_path, "case")
    result_path = pixal3d_result_path(protocol_path, "case")
    assert mesh_path.name == "mesh_official_postprocessed.glb"
    assert result_path.name == "result_official_postprocessed.json"
    assert mesh_path != protocol_path.parent / "pixal3d" / "case" / "mesh.obj"


def test_exact_case_selector_preserves_requested_protocol_order():
    protocol = {
        "cases": [
            {"case_id": "a", "value": 1},
            {"case_id": "b", "value": 2},
            {"case_id": "c", "value": 3},
        ]
    }
    selected = select_protocol_cases(protocol, "c,a")
    assert [row["case_id"] for row in selected] == ["c", "a"]
    assert select_protocol_cases(protocol, "") == protocol["cases"]


def test_exact_case_selector_rejects_missing_and_duplicate_ids():
    protocol = {"cases": [{"case_id": "a"}]}
    for value, message in (
        ("missing", "absent from the frozen protocol"),
        ("a,a", "non-empty and unique"),
    ):
        try:
            select_protocol_cases(protocol, value)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"selector must reject {value!r}")


def test_official_result_validator_rejects_legacy_geometry(tmp_path: Path):
    protocol = {"protocol_sha256": "protocol"}
    case = {"case_id": "case"}
    mesh_path = tmp_path / "mesh_official_postprocessed.glb"
    mesh_path.write_bytes(b"glb")
    valid = {
        "format": INFERENCE_RESULT_FORMAT,
        "protocol_sha256": "protocol",
        "mesh": str(mesh_path),
        "mesh_sha256": sha256_file(mesh_path),
        "geometry_export": OFFICIAL_GEOMETRY_EXPORT,
        "postprocess": OFFICIAL_POSTPROCESS,
    }
    validate_official_inference_result(
        valid,
        protocol=protocol,
        case=case,
        mesh_path=mesh_path,
    )
    legacy = dict(valid)
    legacy["geometry_export"] = "decoded pre-remesh OBJ"
    try:
        validate_official_inference_result(
            legacy,
            protocol=protocol,
            case=case,
            mesh_path=mesh_path,
        )
    except RuntimeError as error:
        assert "not the official final export" in str(error)
    else:
        raise AssertionError("legacy decoded geometry must be rejected")


def test_float32_canonical_margin_matches_registered_decimal():
    margin = fixed_float_value(
        [0.8999999761581421] * 6,
        label="test canonical margin",
    )
    assert np.isclose(margin, 0.9, rtol=0.0, atol=1.0e-6)
