#!/usr/bin/env python3
"""Pure helpers for Direct-SLAT runtime repeatability diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch


PROCESS_REPORT_FORMAT = "pose_point_depth_mv.direct_slat_runtime_process.v1"
AGGREGATE_REPORT_FORMAT = "pose_point_depth_mv.direct_slat_runtime_aggregate.v1"

REPEAT_METRICS = (
    "latent_feature_rms",
    "latent_feature_max_abs",
    "chamfer_l1_abs",
    "fscore_0p02_abs",
    "largest_component_ratio_abs",
    "boundary_edge_count_abs",
    "boundary_total_length_abs",
    "nonmanifold_edge_count_abs",
    "component_count_abs",
)
TOPOLOGY_CATEGORIES = (
    "is_watertight",
    "zero_boundary",
    "nonmanifold_free",
)
SLAT_SCOPES = ("slat_same_process", "slat_independent_process")


def higher_quantile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("quantile values must be a non-empty finite vector")
    try:
        return float(np.quantile(array, float(quantile), method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(array, float(quantile), interpolation="higher"))


def value_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95_higher": higher_quantile(array, 0.95),
        "max": float(array.max()),
    }


def sparse_payload_diff(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> dict[str, Any]:
    left_coords = left["coords"]
    right_coords = right["coords"]
    left_feats = left["feats"].float()
    right_feats = right["feats"].float()
    coords_exact = torch.equal(left_coords, right_coords)
    compatible = coords_exact and left_feats.shape == right_feats.shape
    if not compatible:
        return {
            "coords_exact": bool(coords_exact),
            "features_exact": False,
            "changed_fraction": 1.0,
            "latent_feature_max_abs": 1.0e30,
            "latent_feature_rms": 1.0e30,
        }
    difference = left_feats - right_feats
    features_exact = torch.equal(left["feats"], right["feats"])
    return {
        "coords_exact": True,
        "features_exact": bool(features_exact),
        "changed_fraction": (
            float(torch.count_nonzero(difference).item() / difference.numel())
            if difference.numel()
            else 0.0
        ),
        "latent_feature_max_abs": (
            float(difference.abs().max().item()) if difference.numel() else 0.0
        ),
        "latent_feature_rms": (
            float(torch.sqrt(torch.mean(difference.square())).item())
            if difference.numel()
            else 0.0
        ),
    }


def record_metric_diff(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[dict[str, float], dict[str, bool]]:
    left_structure = left["structure"]
    right_structure = right["structure"]
    metrics = {
        "chamfer_l1_abs": abs(
            float(left["surface"]["chamfer_l1"])
            - float(right["surface"]["chamfer_l1"])
        ),
        "fscore_0p02_abs": abs(
            float(left["surface"]["fscore_0p02"])
            - float(right["surface"]["fscore_0p02"])
        ),
        "largest_component_ratio_abs": abs(
            float(left_structure["largest_component_ratio"])
            - float(right_structure["largest_component_ratio"])
        ),
        "boundary_edge_count_abs": abs(
            float(left_structure["boundary_edge_count"])
            - float(right_structure["boundary_edge_count"])
        ),
        "boundary_total_length_abs": abs(
            float(left_structure["boundary_total_length"])
            - float(right_structure["boundary_total_length"])
        ),
        "nonmanifold_edge_count_abs": abs(
            float(left_structure["nonmanifold_edge_count"])
            - float(right_structure["nonmanifold_edge_count"])
        ),
        "component_count_abs": abs(
            float(left_structure["component_count"])
            - float(right_structure["component_count"])
        ),
    }
    changed = {
        "is_watertight": bool(left_structure["is_watertight"])
        != bool(right_structure["is_watertight"]),
        "zero_boundary": (int(left_structure["boundary_edge_count"]) == 0)
        != (int(right_structure["boundary_edge_count"]) == 0),
        "nonmanifold_free": (
            int(left_structure["nonmanifold_edge_count"]) == 0
        )
        != (int(right_structure["nonmanifold_edge_count"]) == 0),
    }
    return metrics, changed


def summarize_comparisons(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("runtime comparison contains no rows")

    def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for name in REPEAT_METRICS:
            values = [
                float(row["metric_abs_diff"][name])
                for row in group
                if name in row["metric_abs_diff"]
            ]
            if values:
                metrics[name] = value_summary(values)
        return {
            "comparison_count": len(group),
            "metric_abs_diff": metrics,
            "topology_flip": {
                name: {
                    "count": int(
                        sum(bool(row["topology_changed"][name]) for row in group)
                    ),
                    "rate": float(
                        np.mean(
                            [bool(row["topology_changed"][name]) for row in group]
                        )
                    ),
                }
                for name in TOPOLOGY_CATEGORIES
            },
            "hard_integrity_failure_count": int(
                sum(not bool(row["hard_integrity_passed"]) for row in group)
            ),
            "coordinate_mismatch_count": int(
                sum(row.get("coords_exact") is False for row in group)
            ),
        }

    scopes: dict[str, Any] = {}
    by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_scope[str(row["scope"])].append(row)
    for scope, scoped in sorted(by_scope.items()):
        by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_branch_object: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
            list
        )
        for row in scoped:
            branch = str(row["branch"])
            object_uid = str(row["object_uid"])
            by_branch[branch].append(row)
            by_object[object_uid].append(row)
            by_branch_object[(branch, object_uid)].append(row)
        scopes[scope] = {
            "overall": summarize_group(scoped),
            "by_branch": {
                key: summarize_group(value) for key, value in sorted(by_branch.items())
            },
            "by_object": {
                key: summarize_group(value) for key, value in sorted(by_object.items())
            },
            "by_branch_object": {
                f"{branch}|{object_uid}": summarize_group(value)
                for (branch, object_uid), value in sorted(by_branch_object.items())
            },
        }
    return {"comparison_count": len(records), "scopes": scopes}


def evaluate_runtime(
    rows: Iterable[dict[str, Any]],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    records = [row for row in rows if str(row["scope"]) in SLAT_SCOPES]
    if not records:
        raise ValueError("runtime decision has no SLAT comparisons")
    same_process_by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scope_branch: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_scope_branch[(str(row["scope"]), str(row["branch"]))].append(row)
        if str(row["scope"]) == "slat_same_process":
            same_process_by_branch[str(row["branch"])].append(row)
    if not same_process_by_branch:
        raise ValueError("runtime decision has no same-process SLAT comparisons")

    worst_p95: dict[str, float] = {}
    global_max: dict[str, float] = {}
    for name, limit in criteria["regular_p95_max"].items():
        branch_values = []
        for group_rows in same_process_by_branch.values():
            values = [float(row["metric_abs_diff"][name]) for row in group_rows]
            branch_values.append(higher_quantile(values, 0.95))
        all_values = [float(row["metric_abs_diff"][name]) for row in records]
        worst_p95[name] = float(max(branch_values))
        global_max[name] = float(max(all_values))
        if name not in criteria["catastrophic_max"]:
            raise ValueError(f"catastrophic limit missing for metric={name}")

    worst_flip_rate = {
        name: float(
            max(
                np.mean([bool(row["topology_changed"][name]) for row in values])
                for values in by_scope_branch.values()
            )
        )
        for name in criteria["topology_flip_rate_max"]
    }
    checks = {
        **{
            f"p95_{name}": worst_p95[name] <= float(limit)
            for name, limit in criteria["regular_p95_max"].items()
        },
        **{
            f"max_{name}": global_max[name] <= float(limit)
            for name, limit in criteria["catastrophic_max"].items()
        },
        **{
            f"flip_rate_{name}": worst_flip_rate[name] <= float(limit)
            for name, limit in criteria["topology_flip_rate_max"].items()
        },
        "hard_integrity": all(
            bool(row["hard_integrity_passed"]) for row in records
        ),
        "coordinates_exact": all(row.get("coords_exact") is not False for row in records),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "worst_branch_same_process_p95": worst_p95,
        "global_max": global_max,
        "worst_scope_branch_topology_flip_rate": worst_flip_rate,
        "comparison_count": len(records),
    }


def repeat_policy_from_criteria(criteria: dict[str, Any]) -> dict[str, Any]:
    exporter_metrics = (
        "chamfer_l1_abs",
        "fscore_0p02_abs",
        "largest_component_ratio_abs",
        "boundary_edge_count_abs",
        "boundary_total_length_abs",
        "nonmanifold_edge_count_abs",
        "component_count_abs",
    )
    return {
        "mode": "multi_repeat_p95",
        "quantile": 0.95,
        "quantile_method": "higher",
        "regular_p95_max": {
            name: float(criteria["regular_p95_max"][name])
            for name in exporter_metrics
        },
        "catastrophic_max": {
            name: float(criteria["catastrophic_max"][name])
            for name in exporter_metrics
        },
        "topology_flip_rate_max": {
            name: float(criteria["topology_flip_rate_max"][name])
            for name in TOPOLOGY_CATEGORIES
        },
        "hard_integrity_is_absolute": True,
        "interpretation": (
            "p95 is the regular within-runtime repeat floor; max is a catastrophic "
            "diagnostic and topology categories use preregistered flip-rate limits"
        ),
    }
