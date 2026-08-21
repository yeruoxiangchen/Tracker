from __future__ import annotations

import numpy as np
import trimesh

from pose_point_depth_mv.evaluate_reconviagen_stock_full_mesh import (
    LATENT_DECODER_TO_REFERENCE,
    aggregate_results,
    extended_surface_metrics,
    metric_delta,
)


def inverse_axis_mesh(reference: trimesh.Trimesh) -> trimesh.Trimesh:
    value = reference.copy()
    value.apply_transform(np.linalg.inv(LATENT_DECODER_TO_REFERENCE))
    return value


def test_fixed_decoder_axis_conversion_recovers_reference_frame() -> None:
    reference = trimesh.creation.box(extents=(0.8, 0.5, 0.3))
    decoder = inverse_axis_mesh(reference)
    decoder.apply_transform(LATENT_DECODER_TO_REFERENCE)
    assert np.allclose(decoder.vertices, reference.vertices)
    metrics = extended_surface_metrics(
        decoder,
        reference,
        count=2000,
        seed=7,
        thresholds=(0.01, 0.02, 0.05),
    )
    assert metrics["chamfer_l1"] < 1.0e-12
    assert metrics["fscore_0p02"] == 1.0


def test_directed_quantiles_expose_spurious_geometry() -> None:
    reference = trimesh.creation.icosphere(subdivisions=2, radius=0.35)
    appendage = trimesh.creation.box(extents=(0.45, 0.08, 0.08))
    appendage.apply_translation((0.50, 0.0, 0.0))
    prediction = trimesh.util.concatenate((reference, appendage))
    metrics = extended_surface_metrics(
        prediction,
        reference,
        count=12000,
        seed=11,
        thresholds=(0.01, 0.02, 0.05),
    )
    assert metrics["pred_to_gt_p95"] > metrics["gt_to_pred_p95"]
    assert metrics["pred_to_gt_outlier_ratio_0p05"] > 0.01
    assert metrics["precision_0p02"] < metrics["recall_0p02"]


def test_metric_delta_is_positive_only_when_lhs_is_better() -> None:
    assert metric_delta(0.10, 0.20, "chamfer_l1") > 0.0
    assert metric_delta(0.20, 0.10, "pred_to_gt_p95") < 0.0
    assert metric_delta(0.80, 0.60, "fscore_0p02") > 0.0
    assert metric_delta(0.60, 0.80, "normal_consistency") < 0.0


def fake_method(chamfer: float, fscore: float) -> dict:
    surface = {
        "chamfer_l1": chamfer,
        "fscore_0p02": fscore,
    }
    structure = {
        "largest_component_ratio": 1.0,
        "component_count": 1,
        "small_component_vertex_ratio_lt100": 0.0,
        "boundary_edge_count": 0,
        "boundary_total_length": 0.0,
        "nonmanifold_edge_count": 0,
        "is_watertight": True,
        "is_winding_consistent": True,
    }
    return {"surface": surface, "structure": structure}


def fake_record(object_uid: str, seed: int, offset: float) -> dict:
    methods = {
        "reconviagen_original": fake_method(0.30 + offset, 0.50),
        "direct_stock": fake_method(0.20 + offset, 0.60),
        "direct_full": fake_method(0.10 + offset, 0.70),
    }
    comparisons = {}
    pairs = (
        ("full_minus_reconviagen", "direct_full", "reconviagen_original"),
        ("stock_minus_reconviagen", "direct_stock", "reconviagen_original"),
        ("full_minus_stock", "direct_full", "direct_stock"),
    )
    for comparison_id, lhs, rhs in pairs:
        comparisons[comparison_id] = {
            "metrics": {
                "chamfer_l1": metric_delta(
                    methods[lhs]["surface"]["chamfer_l1"],
                    methods[rhs]["surface"]["chamfer_l1"],
                    "chamfer_l1",
                ),
                "fscore_0p02": metric_delta(
                    methods[lhs]["surface"]["fscore_0p02"],
                    methods[rhs]["surface"]["fscore_0p02"],
                    "fscore_0p02",
                ),
                "largest_component_ratio": 0.0,
            }
        }
    return {
        "object_uid": object_uid,
        "joint_seed": seed,
        "methods": methods,
        "comparisons": comparisons,
    }


def test_aggregation_averages_seeds_before_objects() -> None:
    records = [
        fake_record("a", 42, 0.00),
        fake_record("a", 43, 0.02),
        fake_record("b", 42, 0.04),
        fake_record("b", 43, 0.06),
    ]
    object_rows, method_summary, comparison_summary = aggregate_results(
        records,
        surface_keys=["chamfer_l1", "fscore_0p02"],
        bootstrap_samples=100,
    )
    assert len(object_rows) == 2
    assert all(row["seed_count"] == 2 for row in object_rows)
    assert method_summary["direct_full"]["chamfer_l1"]["count"] == 2
    full_vs_stock = comparison_summary["full_minus_stock"]["metrics"]
    assert np.isclose(full_vs_stock["chamfer_l1"]["mean"], 0.10)
    assert np.isclose(full_vs_stock["fscore_0p02"]["mean"], 0.10)
    assert full_vs_stock["chamfer_l1"]["positive_rate"] == 1.0


if __name__ == "__main__":
    test_fixed_decoder_axis_conversion_recovers_reference_frame()
    test_directed_quantiles_expose_spurious_geometry()
    test_metric_delta_is_positive_only_when_lhs_is_better()
    test_aggregation_averages_seeds_before_objects()
    print("evaluate_reconviagen_stock_full_mesh tests passed")
