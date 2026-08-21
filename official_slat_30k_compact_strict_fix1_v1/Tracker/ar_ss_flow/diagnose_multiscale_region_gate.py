#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from ar_ss_flow.pairwise_multiscale_region_gate import Candidate
from ar_ss_flow.summarize_pairwise_multiscale_region_gate import (
    MODES,
    evaluate_probe_volume,
    load_json,
    load_npz,
    mode_index_map,
)


DIAGNOSTIC_METRICS = (
    "wrong_gain_vs_object",
    "shuffle_gain_vs_object",
    "wrong_gain_vs_wrong_gate",
    "shuffle_gain_vs_shuffle_gate",
    "wrong_spatial_rank_correlation",
    "shuffle_spatial_rank_correlation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose C1.6 selected multi-scale region gate on fresh objects."
    )
    parser.add_argument("--calibration_report", required=True)
    parser.add_argument("--fresh_report", required=True)
    parser.add_argument("--fresh_volumes", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=20000)
    parser.add_argument("--permutation_samples", type=int, default=20000)
    parser.add_argument("--correlation_gate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def average_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot average empty rows")
    return {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in rows[0]
    }


def summarize_metric(
    values: list[float],
    *,
    rng: np.random.Generator,
    bootstrap_samples: int,
    permutation_samples: int,
    gate: float | None = None,
) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "positive_rate": None,
            "gate_pass_rate": None,
            "bootstrap_ci95_mean": [None, None],
            "sign_flip_p_one_sided": None,
        }

    if bootstrap_samples > 0:
        indices = rng.integers(
            0, x.size, size=(int(bootstrap_samples), x.size)
        )
        bootstrap_means = x[indices].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    else:
        ci_low = ci_high = float(np.mean(x))

    observed = float(np.mean(x))
    if permutation_samples > 0:
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(int(permutation_samples), x.size),
        )
        null_means = (signs * x[None, :]).mean(axis=1)
        p_value = float(
            (1 + np.count_nonzero(null_means >= observed))
            / (int(permutation_samples) + 1)
        )
    else:
        p_value = None

    return {
        "count": int(x.size),
        "mean": observed,
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "positive_rate": float(np.mean(x > 0.0)),
        "gate_pass_rate": (
            None if gate is None else float(np.mean(x >= float(gate)))
        ),
        "bootstrap_ci95_mean": [float(ci_low), float(ci_high)],
        "sign_flip_p_one_sided": p_value,
    }


def classify_correlation(
    stats: dict[str, Any],
    *,
    threshold: float,
) -> str:
    mean = stats["mean"]
    ci_low = stats["bootstrap_ci95_mean"][0]
    p_value = stats["sign_flip_p_one_sided"]

    if mean is None:
        return "no_data"
    if mean >= threshold and ci_low > 0.0 and p_value < 0.05:
        return "strict_threshold_supported"
    if mean > 0.0 and ci_low > 0.0 and p_value < 0.05:
        return "stable_positive_but_below_threshold"
    if mean > 0.0:
        return "positive_mean_but_uncertain"
    return "no_positive_spatial_evidence"


def main() -> None:
    args = parse_args()

    calibration_report = load_json(args.calibration_report)
    fresh_report = load_json(args.fresh_report)
    arrays = load_npz(args.fresh_volumes)

    selected = calibration_report["selected_candidate"]
    candidate = Candidate(
        name=str(selected["name"]),
        divisions=int(selected["divisions"]),
        shrinkage_kappa=(
            None
            if selected.get("shrinkage_kappa") is None
            else float(selected["shrinkage_kappa"])
        ),
    )

    config_args = calibration_report.get("args", {})
    trim_fraction = float(config_args.get("trim_fraction", 0.10))
    min_region_voxels = int(config_args.get("min_region_voxels", 8))
    side = int(fresh_report["protocol"]["volume_side"])
    mapping = mode_index_map(fresh_report)

    report: dict[str, Any] = {
        "stage": "C1.6 selected region gate diagnostic",
        "candidate": {
            "name": candidate.name,
            "divisions": candidate.divisions,
            "region_count": candidate.divisions ** 3,
            "shrinkage_kappa": candidate.shrinkage_kappa,
        },
        "volume_side": side,
        "trim_fraction": trim_fraction,
        "min_region_voxels": min_region_voxels,
        "correlation_gate": float(args.correlation_gate),
        "bootstrap_samples": int(args.bootstrap_samples),
        "permutation_samples": int(args.permutation_samples),
        "per_mode": {},
    }

    print("=" * 100)
    print("C1.6 selected region gate diagnostic")
    print("candidate:", report["candidate"])

    for mode_number, mode in enumerate(MODES):
        grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
        row_ids = np.flatnonzero(
            arrays["probe_mode_index"] == int(mapping[mode])
        )

        for row_id in row_ids:
            row = evaluate_probe_volume(
                arrays,
                row=int(row_id),
                side=side,
                candidate=candidate,
                trim_fraction=trim_fraction,
                min_region_voxels=min_region_voxels,
            )
            if row is None:
                continue

            object_index = int(arrays["probe_object_index"][row_id])
            grouped[object_index].append(row)

        object_rows: list[dict[str, Any]] = []
        for object_index, rows in sorted(grouped.items()):
            averaged = average_rows(rows)
            object_rows.append(
                {
                    "object_index": int(object_index),
                    "volume_count": len(rows),
                    **averaged,
                }
            )

        mode_result: dict[str, Any] = {
            "object_count": len(object_rows),
            "object_rows": object_rows,
            "metrics": {},
            "valid_region_count": {},
        }

        valid_counts = np.asarray(
            [row["valid_region_count"] for row in object_rows],
            dtype=np.float64,
        )
        mode_result["valid_region_count"] = {
            "mean": float(np.mean(valid_counts)) if valid_counts.size else None,
            "median": (
                float(np.median(valid_counts)) if valid_counts.size else None
            ),
            "min": float(np.min(valid_counts)) if valid_counts.size else None,
            "max": float(np.max(valid_counts)) if valid_counts.size else None,
        }

        for metric_number, metric in enumerate(DIAGNOSTIC_METRICS):
            values = [float(row[metric]) for row in object_rows]
            rng = np.random.default_rng(
                int(args.seed) + mode_number * 1009 + metric_number * 9176
            )
            gate = (
                float(args.correlation_gate)
                if metric.endswith("spatial_rank_correlation")
                else 0.0
            )
            mode_result["metrics"][metric] = summarize_metric(
                values,
                rng=rng,
                bootstrap_samples=int(args.bootstrap_samples),
                permutation_samples=int(args.permutation_samples),
                gate=gate,
            )

        shuffle_stats = mode_result["metrics"][
            "shuffle_spatial_rank_correlation"
        ]
        wrong_stats = mode_result["metrics"][
            "wrong_spatial_rank_correlation"
        ]
        mode_result["wrong_correlation_classification"] = (
            classify_correlation(
                wrong_stats,
                threshold=float(args.correlation_gate),
            )
        )
        mode_result["shuffle_correlation_classification"] = (
            classify_correlation(
                shuffle_stats,
                threshold=float(args.correlation_gate),
            )
        )

        worst_shuffle = sorted(
            object_rows,
            key=lambda row: row["shuffle_spatial_rank_correlation"],
        )[:5]
        mode_result["worst_shuffle_correlation_objects"] = [
            {
                "object_index": row["object_index"],
                "shuffle_spatial_rank_correlation":
                    row["shuffle_spatial_rank_correlation"],
                "shuffle_gain_vs_object": row["shuffle_gain_vs_object"],
                "shuffle_gain_vs_shuffle_gate":
                    row["shuffle_gain_vs_shuffle_gate"],
                "valid_region_count": row["valid_region_count"],
            }
            for row in worst_shuffle
        ]

        report["per_mode"][mode] = mode_result

        print("-" * 100)
        print(
            f"{mode}: objects={len(object_rows)} "
            f"valid_regions_mean="
            f"{mode_result['valid_region_count']['mean']}"
        )

        for metric in DIAGNOSTIC_METRICS:
            stats = mode_result["metrics"][metric]
            print(
                f"  {metric}: "
                f"mean={stats['mean']:+.6f} "
                f"median={stats['median']:+.6f} "
                f"ci95=[{stats['bootstrap_ci95_mean'][0]:+.6f},"
                f"{stats['bootstrap_ci95_mean'][1]:+.6f}] "
                f"positive={stats['positive_rate']:.4f} "
                f"gate_pass={stats['gate_pass_rate']:.4f} "
                f"p_one_sided={stats['sign_flip_p_one_sided']:.6f}"
            )

        print(
            "  wrong_corr_classification:",
            mode_result["wrong_correlation_classification"],
        )
        print(
            "  shuffle_corr_classification:",
            mode_result["shuffle_correlation_classification"],
        )
        print("  worst shuffle-correlation objects:")
        for row in mode_result["worst_shuffle_correlation_objects"]:
            print(
                "   ",
                f"object={row['object_index']} "
                f"corr={row['shuffle_spatial_rank_correlation']:+.6f} "
                f"vs_object={row['shuffle_gain_vs_object']:+.6f} "
                f"vs_shuffle_gate="
                f"{row['shuffle_gain_vs_shuffle_gate']:+.6f} "
                f"regions={row['valid_region_count']:.1f}",
            )

    shuffle_classes = {
        mode: report["per_mode"][mode][
            "shuffle_correlation_classification"
        ]
        for mode in MODES
    }
    report["diagnostic_summary"] = {
        "shuffle_classification_by_mode": shuffle_classes,
        "all_shuffle_strict_supported": all(
            value == "strict_threshold_supported"
            for value in shuffle_classes.values()
        ),
        "all_shuffle_stably_positive": all(
            value in {
                "strict_threshold_supported",
                "stable_positive_but_below_threshold",
            }
            for value in shuffle_classes.values()
        ),
        "interpretation_rule": {
            "all_shuffle_strict_supported":
                "The original 0.05 shuffle spatial gate has statistical support.",
            "all_shuffle_stably_positive":
                "Shuffle spatial signal is reproducibly positive but may be below the preregistered effect-size threshold.",
            "otherwise":
                "Do not treat grid64 as a general pose/shuffle-specific region gate.",
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 100)
    print("diagnostic_summary:", report["diagnostic_summary"])
    print("output:", output_path)


if __name__ == "__main__":
    main()
