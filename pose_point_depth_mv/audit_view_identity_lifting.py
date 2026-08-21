#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.view_identity_lifting import (
    VIEW_IDENTITY_CONTROL_NAMES,
    VIEW_IDENTITY_GEOMETRY_NAMES,
    build_view_identity_evidence,
    view_identity_schema_hash,
)


DEPTH_RESIDUAL_INDEX = VIEW_IDENTITY_GEOMETRY_NAMES.index(
    "depth_residual_normalized"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit view-preserving pose/depth lifting before training."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_multiview_support_ratio", type=float, default=0.01)
    parser.add_argument("--min_nonidentity_ratio", type=float, default=0.80)
    parser.add_argument("--min_direction_ratio", type=float, default=0.60)
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean().item()),
        "median": float(tensor.median().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def weighted_normalized_mse(
    left: torch.Tensor,
    right: torch.Tensor,
    weight: torch.Tensor,
) -> float:
    expanded = weight[..., None].float()
    denominator = (left.float().square() * expanded).sum().clamp_min(1.0e-6)
    numerator = ((left.float() - right.float()).square() * expanded).sum()
    return float((numerator / denominator).item())


def cross_view_cosine(
    visual: torch.Tensor,
    weight: torch.Tensor,
) -> float:
    normalized = F.normalize(visual.float(), dim=-1, eps=1.0e-6)
    numerator = visual.new_zeros((), dtype=torch.float32)
    denominator = visual.new_zeros((), dtype=torch.float32)
    views = int(visual.shape[0])
    for first in range(views):
        for second in range(first + 1, views):
            pair_weight = torch.minimum(weight[first], weight[second]).float()
            cosine = (normalized[first] * normalized[second]).sum(dim=-1)
            numerator = numerator + (cosine * pair_weight).sum()
            denominator = denominator + pair_weight.sum()
    return float((numerator / denominator.clamp_min(1.0e-6)).item())


def weighted_depth_residual(
    geometry: torch.Tensor,
    weight: torch.Tensor,
) -> float:
    value = geometry[..., DEPTH_RESIDUAL_INDEX].float()
    return float((value * weight).sum().div(weight.sum().clamp_min(1.0e-6)).item())


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# View-identity Pose-guided Lifting Audit",
        "",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Samples: `{report['sample_count']}`",
        f"- Schema: `{report['schema_hash']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- `{name}`: `{value}`" for name, value in report["checks"].items()
    )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(report["summary"], indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    count = len(dataset) if args.max_samples <= 0 else min(
        len(dataset), int(args.max_samples)
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index in range(count):
        sample = dataset[index]
        uid = str(sample["uid"])
        try:
            correct = build_view_identity_evidence(
                sample, device=device, mode="correct"
            )
            fixed_weight = correct["view_weight"].float()
            support_count = fixed_weight.gt(1.0e-6).sum(dim=0)
            row: dict[str, Any] = {
                "uid": uid,
                "object_uid": str(sample.get("object_uid", uid)),
                "views": int(correct["views"]),
                "depth_enabled": bool(correct["depth_enabled"]),
                "multiview_support_ratio": float(
                    support_count.ge(2).float().mean().item()
                ),
                "correct_cross_view_cosine": cross_view_cosine(
                    correct["sampled_visual"], fixed_weight
                ),
                "correct_depth_residual": weighted_depth_residual(
                    correct["geometry"], fixed_weight
                ),
            }
            for mode in VIEW_IDENTITY_CONTROL_NAMES:
                control = build_view_identity_evidence(
                    sample, device=device, mode=mode
                )
                row[f"{mode}_visual_normalized_mse"] = weighted_normalized_mse(
                    correct["sampled_visual"],
                    control["sampled_visual"],
                    fixed_weight,
                )
                row[f"{mode}_geometry_normalized_mse"] = weighted_normalized_mse(
                    correct["geometry"], control["geometry"], fixed_weight
                )
                control_cosine = cross_view_cosine(
                    control["sampled_visual"], fixed_weight
                )
                control_depth = weighted_depth_residual(
                    control["geometry"], fixed_weight
                )
                row[f"{mode}_cross_view_cosine"] = control_cosine
                row[f"{mode}_cosine_advantage"] = (
                    row["correct_cross_view_cosine"] - control_cosine
                )
                row[f"{mode}_depth_residual"] = control_depth
                row[f"{mode}_depth_advantage"] = (
                    control_depth - row["correct_depth_residual"]
                )
            records.append(row)
            print(f"[view_identity_audit] {index + 1}/{count} uid={uid}", flush=True)
        except Exception as error:  # noqa: BLE001 - audit records sample failures
            failures.append({"uid": uid, "error": repr(error)})
            print(f"[view_identity_audit] FAIL uid={uid}: {error}", flush=True)

    summary: dict[str, Any] = {
        "multiview_support_ratio": summarize(
            [float(row["multiview_support_ratio"]) for row in records]
        ),
        "depth_enabled_ratio": (
            sum(bool(row["depth_enabled"]) for row in records) / max(len(records), 1)
        ),
        "controls": {},
    }
    direction_checks: dict[str, bool] = {}
    nonidentity_checks: dict[str, bool] = {}
    for mode in VIEW_IDENTITY_CONTROL_NAMES:
        visual_mse = [
            float(row[f"{mode}_visual_normalized_mse"]) for row in records
        ]
        geometry_mse = [
            float(row[f"{mode}_geometry_normalized_mse"]) for row in records
        ]
        cosine_advantage = [
            float(row[f"{mode}_cosine_advantage"]) for row in records
        ]
        depth_advantage = [
            float(row[f"{mode}_depth_advantage"]) for row in records
        ]
        nonidentity = [
            max(visual, geometry) > 1.0e-6
            for visual, geometry in zip(visual_mse, geometry_mse)
        ]
        direction_values = (
            depth_advantage
            if mode.startswith("depth_")
            else cosine_advantage
        )
        nonidentity_ratio = sum(nonidentity) / max(len(nonidentity), 1)
        direction_ratio = sum(value > 0.0 for value in direction_values) / max(
            len(direction_values), 1
        )
        nonidentity_checks[mode] = nonidentity_ratio >= float(
            args.min_nonidentity_ratio
        )
        direction_checks[mode] = direction_ratio >= float(args.min_direction_ratio)
        summary["controls"][mode] = {
            "visual_normalized_mse": summarize(visual_mse),
            "geometry_normalized_mse": summarize(geometry_mse),
            "cosine_advantage": summarize(cosine_advantage),
            "depth_advantage": summarize(depth_advantage),
            "nonidentity_ratio": nonidentity_ratio,
            "direction_ratio": direction_ratio,
            "direction_metric": (
                "depth_residual" if mode.startswith("depth_") else "cross_view_cosine"
            ),
        }

    checks = {
        "no_sample_failures": not failures and len(records) == count,
        "multiview_support_nonempty": float(
            summary["multiview_support_ratio"]["mean"]
        )
        >= float(args.min_multiview_support_ratio),
        "depth_enabled_for_all_samples": float(summary["depth_enabled_ratio"]) == 1.0,
        "every_control_nonidentity": all(nonidentity_checks.values()),
        "every_control_directional": all(direction_checks.values()),
    }
    report = {
        "stage": "view-identity pose/depth lifting pre-training audit",
        "passed": all(checks.values()),
        "args": vars(args),
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "schema_hash": view_identity_schema_hash(),
        "sample_count": count,
        "audited_count": len(records),
        "checks": checks,
        "summary": summary,
        "failures": failures,
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "report.md")
    print(json.dumps({"passed": report["passed"], "checks": checks}, indent=2))
    if args.fail_on_decision and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
