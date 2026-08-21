#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from reconvggt_ar_adapter_a.pointpose_ss_condition import PHYSICAL_FEATURE_NAMES
from reconvggt_ar_adapter_a.sparse_anchor_flow import (
    FEATURE_INDEX,
    build_sparse_anchor_masks,
    dropout_sparse_prior,
    make_mask_only_physical_grid,
    make_point_only_physical_grid,
    shift_sparse_prior,
)


def parse_shifts(text: str) -> list[tuple[int, int, int]]:
    output: list[tuple[int, int, int]] = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        values = tuple(int(value.strip()) for value in item.split(","))
        if len(values) != 3 or values == (0, 0, 0):
            raise ValueError(f"invalid nonzero 3D shift: {item!r}")
        output.append(values)
    if not output:
        raise ValueError("at least one shift is required")
    return output


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_rate": float((array > 0).mean()),
    }


def resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def load_sample(manifest_path: Path, row: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    physical_path = resolve_path(manifest_path, str(row["physical_grid"]))
    latent_path = resolve_path(manifest_path, str(row["ss_latent"]))
    with np.load(physical_path) as data:
        physical = np.asarray(data["physical_grid"], dtype=np.float32)
    with np.load(latent_path) as data:
        target_coords = np.asarray(data["target_coords"], dtype=np.int64)[:, -3:]
    return torch.from_numpy(physical).unsqueeze(0), torch.from_numpy(target_coords)


def prior_consistency(physical: torch.Tensor) -> dict[str, float | int]:
    occupancy = physical[:, FEATURE_INDEX["prior_occupancy"]] > 0.5
    count = int(occupancy.sum().item())
    if not count:
        return {
            "prior_count": 0,
            "mask_support_mean": 0.0,
            "mask_hit_mean": 0.0,
            "outside_mean": 1.0,
            "hull_inside_mean": 0.0,
            "score": -1.0,
        }

    def selected(name: str) -> float:
        values = physical[:, FEATURE_INDEX[name]][occupancy]
        return float(values.float().mean().item())

    support = selected("mask_support_fraction")
    hit = selected("mask_hit_ratio")
    outside = selected("outside_visible_ratio")
    hull = selected("visual_hull_inside")
    return {
        "prior_count": count,
        "mask_support_mean": support,
        "mask_hit_mean": hit,
        "outside_mean": outside,
        "hull_inside_mean": hull,
        "score": support + hit + hull - outside,
    }


def target_overlap(physical: torch.Tensor, masks: dict[str, torch.Tensor]) -> float:
    prior = physical[:, FEATURE_INDEX["prior_occupancy"] : FEATURE_INDEX["prior_occupancy"] + 1] > 0.5
    denominator = int(prior.sum().item())
    if denominator == 0:
        return 0.0
    return float((prior & masks["target16"]).sum().item() / denominator)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    checks = report["checks"]
    lines = [
        "# Sparse-anchor Input Audit",
        "",
        f"- decision: `{'PASS' if report['passed'] else 'FAIL'}`",
        f"- samples / objects: `{summary['sample_count']} / {summary['object_count']}`",
        f"- shifts: `{report['args']['shifts']}`",
        "",
        "## Label Coverage",
        "",
        "| metric | mean | median | min | max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in (
        "prior_cells",
        "positive16_cells",
        "negative16_cells",
        "positive64_voxels",
        "negative64_voxels",
        "correct_target_overlap",
    ):
        row = summary[key]
        lines.append(
            f"| {key} | {row['mean']:.6f} | {row['median']:.6f} | "
            f"{row['min']:.6f} | {row['max']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Controlled Corruption",
            "",
            "| metric | mean | median | positive rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in (
        "correct_minus_shift_consistency",
        "correct_minus_shift_target_overlap",
        "correct_shift_feature_rms",
        "correct_minus_dropout_consistency",
    ):
        row = summary[key]
        lines.append(
            f"| {key} | {row['mean']:.8f} | {row['median']:.8f} | "
            f"{row.get('positive_rate', 0.0):.2%} |"
        )
    lines.extend(["", "## Checks", ""])
    for key, passed in checks.items():
        lines.append(f"- {key}: `{'PASS' if passed else 'FAIL'}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit sparse-anchor coverage and controlled-corruption separability."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--shifts", default="1,0,0;-1,0,0;0,1,0;0,-1,0;0,0,1;0,0,-1;2,0,0")
    parser.add_argument("--dropout_keep_ratio", type=float, default=0.5)
    parser.add_argument("--prior_confidence_min", type=float, default=0.25)
    parser.add_argument("--prior_mask_support_min", type=float, default=0.0)
    parser.add_argument("--anchor_radius_16", type=int, default=1)
    parser.add_argument("--outside_visible_min", type=float, default=0.5)
    parser.add_argument("--outside_ratio_min", type=float, default=0.9)
    parser.add_argument("--negative_surface_margin_64", type=int, default=1)
    parser.add_argument("--min_consistency_win_rate", type=float, default=0.55)
    args = parser.parse_args()

    manifest_path = Path(args.cache_manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(payload.get("feature_names", [])) != PHYSICAL_FEATURE_NAMES:
        raise RuntimeError("cache feature schema does not match PHYSICAL_FEATURE_NAMES")
    samples = list(payload.get("samples", []))
    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]
    if not samples:
        raise RuntimeError("cache manifest has no selected samples")
    shifts = parse_shifts(args.shifts)
    generator = torch.Generator(device="cpu").manual_seed(20260713)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in samples:
        uid = str(row["uid"])
        try:
            physical, target_coords = load_sample(manifest_path, row)
            masks = build_sparse_anchor_masks(
                physical,
                target_coords,
                prior_confidence_min=float(args.prior_confidence_min),
                prior_mask_support_min=float(args.prior_mask_support_min),
                anchor_radius_16=int(args.anchor_radius_16),
                outside_visible_min=float(args.outside_visible_min),
                outside_ratio_min=float(args.outside_ratio_min),
                negative_surface_margin_64=int(args.negative_surface_margin_64),
            )
            correct_consistency = prior_consistency(physical)
            correct_overlap = target_overlap(physical, masks)
            shifted_rows: list[dict[str, float | list[int]]] = []
            for shift in shifts:
                corrupted = shift_sparse_prior(physical, shift)
                corrupted_masks = build_sparse_anchor_masks(
                    corrupted,
                    target_coords,
                    prior_confidence_min=float(args.prior_confidence_min),
                    prior_mask_support_min=float(args.prior_mask_support_min),
                    anchor_radius_16=int(args.anchor_radius_16),
                    outside_visible_min=float(args.outside_visible_min),
                    outside_ratio_min=float(args.outside_ratio_min),
                    negative_surface_margin_64=int(args.negative_surface_margin_64),
                )
                corrupted_consistency = prior_consistency(corrupted)
                feature_rms = float(
                    (physical[:, :4] - corrupted[:, :4]).float().square().mean().sqrt().item()
                )
                shifted_rows.append(
                    {
                        "shift": list(shift),
                        "feature_rms": feature_rms,
                        "correct_minus_consistency": float(
                            correct_consistency["score"] - corrupted_consistency["score"]
                        ),
                        "correct_minus_target_overlap": float(
                            correct_overlap - target_overlap(corrupted, corrupted_masks)
                        ),
                    }
                )
            dropped = dropout_sparse_prior(
                physical,
                float(args.dropout_keep_ratio),
                generator=generator,
            )
            dropped_consistency = prior_consistency(dropped)
            mask_only = make_mask_only_physical_grid(physical)
            point_only = make_point_only_physical_grid(physical)
            overlap_count = int((masks["positive64"] & masks["negative64"]).sum().item())
            rows.append(
                {
                    "uid": uid,
                    "object_uid": str(row.get("object_uid", uid)),
                    "prior_cells": int((physical[:, FEATURE_INDEX["prior_occupancy"]] > 0.5).sum()),
                    "positive16_cells": int(masks["positive16"].sum()),
                    "negative16_cells": int(masks["negative16"].sum()),
                    "positive64_voxels": int(masks["positive64"].sum()),
                    "negative64_voxels": int(masks["negative64"].sum()),
                    "label_overlap": overlap_count,
                    "correct_consistency": correct_consistency,
                    "correct_target_overlap": correct_overlap,
                    "shifted": shifted_rows,
                    "correct_minus_dropout_consistency": float(
                        correct_consistency["score"] - dropped_consistency["score"]
                    ),
                    "mask_only_nonzero": int(torch.count_nonzero(mask_only[:, :11]).item()),
                    "point_only_nonzero": int(torch.count_nonzero(point_only[:, :11]).item()),
                }
            )
        except Exception as exc:
            failures.append({"uid": uid, "error": f"{type(exc).__name__}: {exc}"})

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    shifted = [item for row in rows for item in row["shifted"]]
    summary = {
        "sample_count": len(rows),
        "object_count": len({row["object_uid"] for row in rows}),
        "failed_sample_count": len(failures),
        "prior_cells": distribution(values("prior_cells")),
        "positive16_cells": distribution(values("positive16_cells")),
        "negative16_cells": distribution(values("negative16_cells")),
        "positive64_voxels": distribution(values("positive64_voxels")),
        "negative64_voxels": distribution(values("negative64_voxels")),
        "correct_target_overlap": distribution(values("correct_target_overlap")),
        "correct_minus_shift_consistency": distribution(
            [float(item["correct_minus_consistency"]) for item in shifted]
        ),
        "correct_minus_shift_target_overlap": distribution(
            [float(item["correct_minus_target_overlap"]) for item in shifted]
        ),
        "correct_shift_feature_rms": distribution(
            [float(item["feature_rms"]) for item in shifted]
        ),
        "correct_minus_dropout_consistency": distribution(
            values("correct_minus_dropout_consistency")
        ),
    }
    checks = {
        "all_samples_loaded": len(failures) == 0 and len(rows) == len(samples),
        "nonempty_positive16_all": all(row["positive16_cells"] > 0 for row in rows),
        "nonempty_negative16_all": all(row["negative16_cells"] > 0 for row in rows),
        "nonempty_positive64_all": all(row["positive64_voxels"] > 0 for row in rows),
        "nonempty_negative64_all": all(row["negative64_voxels"] > 0 for row in rows),
        "exclusive_labels": all(row["label_overlap"] == 0 for row in rows),
        "controlled_corruption_changes_input": (
            float(summary["correct_shift_feature_rms"]["min"] or 0.0) > 0.0
        ),
        "correct_consistency_win_rate": (
            float(summary["correct_minus_shift_consistency"].get("positive_rate", 0.0))
            >= float(args.min_consistency_win_rate)
        ),
        "correct_target_overlap_win_rate": (
            float(summary["correct_minus_shift_target_overlap"].get("positive_rate", 0.0))
            >= float(args.min_consistency_win_rate)
        ),
    }
    report = {
        "format": "reconvggt.sparse_anchor_input_audit.v1",
        "args": vars(args),
        "feature_names": list(PHYSICAL_FEATURE_NAMES),
        "summary": summary,
        "checks": checks,
        "passed": all(checks.values()),
        "rows": rows,
        "failed_samples": failures,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(output_dir / "report.md", report)
    (output_dir / "failed_samples.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
