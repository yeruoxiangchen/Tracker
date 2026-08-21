#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch


AUDIT_VERSION = "pose_point_depth_mv.pre_c0_2c_audit.v1"
STEPS = (100, 200)
SEMANTIC_NAMES = {1: "surface", 2: "free_space", 3: "occluded", 4: "boundary"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit voxel flips, depth strata, and symmetric 3D tolerance."
    )
    parser.add_argument("--step100_dir", required=True)
    parser.add_argument("--step200_dir", required=True)
    parser.add_argument("--gaussian_step100_dir", required=True)
    parser.add_argument("--gaussian_step200_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_voxel_positive_ratio", type=float, default=0.60)
    parser.add_argument("--min_object_local_pass_rate", type=float, default=0.65)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def load_report(directory: Path) -> dict[str, Any]:
    path = directory / "report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_maps(directory: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report = load_report(directory)
    maps_dir = directory / "voxel_maps"
    paths = sorted(maps_dir.glob("*.pt"))
    if len(paths) != int(report["sample_count"]):
        raise RuntimeError(
            f"{directory}: map count={len(paths)} != sample_count={report['sample_count']}"
        )
    maps: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = torch.load(path, map_location="cpu")
        uid = str(payload["uid"])
        if uid in maps:
            raise RuntimeError(f"duplicate map uid={uid}")
        required = {
            "active_mask",
            "hard_margin",
            "hardest_control_index",
            "training_control_margins",
            "audit_maps",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise RuntimeError(f"uid={uid} missing audit map fields: {missing}")
        maps[uid] = payload
    return report, maps


def safe_selected_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.float().mean().item()) if selected.numel() else 0.0


def flip_audit(
    step100_maps: dict[str, dict[str, Any]],
    step200_maps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(step100_maps) != set(step200_maps):
        raise RuntimeError("step100/step200 map UID sets differ")
    records: list[dict[str, Any]] = []
    for uid in sorted(step100_maps):
        first = step100_maps[uid]
        second = step200_maps[uid]
        active = first["active_mask"].bool()
        if not torch.equal(active, second["active_mask"].bool()):
            raise RuntimeError(f"uid={uid} active support changed between checkpoints")
        before = first["hard_margin"].float()
        after = second["hard_margin"].float()
        delta = after - before
        positive_before = active & before.gt(0)
        negative_before = active & ~before.gt(0)
        positive_after = active & after.gt(0)
        negative_after = active & ~after.gt(0)
        active_count = int(active.sum().item())
        old_positive_count = int(positive_before.sum().item())
        old_negative_count = int(negative_before.sum().item())
        negative_to_positive = negative_before & positive_after
        positive_to_negative = positive_before & negative_after
        positive_larger = positive_before & positive_after & delta.gt(0)
        negative_more_negative = negative_before & negative_after & delta.lt(0)

        def ratios(mask: torch.Tensor, source_count: int) -> dict[str, float | int]:
            count = int(mask.sum().item())
            return {
                "count": count,
                "fraction_active": float(count / max(active_count, 1)),
                "fraction_source": float(count / max(source_count, 1)),
            }

        records.append(
            {
                "uid": uid,
                "views": int(first["views"]),
                "active_count": active_count,
                "negative_to_positive": ratios(negative_to_positive, old_negative_count),
                "positive_to_negative": ratios(positive_to_negative, old_positive_count),
                "positive_to_larger_positive": ratios(
                    positive_larger, old_positive_count
                ),
                "negative_to_more_negative": ratios(
                    negative_more_negative, old_negative_count
                ),
                "original_positive_margin_delta_mean": safe_selected_mean(
                    delta, positive_before
                ),
                "original_negative_margin_delta_mean": safe_selected_mean(
                    delta, negative_before
                ),
                "positive_ratio_step100": float(
                    positive_before.float().sum().div(max(active_count, 1)).item()
                ),
                "positive_ratio_step200": float(
                    positive_after.float().sum().div(max(active_count, 1)).item()
                ),
            }
        )

    aggregate: dict[str, Any] = {}
    for transition in (
        "negative_to_positive",
        "positive_to_negative",
        "positive_to_larger_positive",
        "negative_to_more_negative",
    ):
        aggregate[transition] = {
            key: summarize([float(row[transition][key]) for row in records])
            for key in ("fraction_active", "fraction_source")
        }
    for key in (
        "original_positive_margin_delta_mean",
        "original_negative_margin_delta_mean",
        "positive_ratio_step100",
        "positive_ratio_step200",
    ):
        aggregate[key] = summarize([float(row[key]) for row in records])
    aggregate["net_positive_coverage_change"] = float(
        aggregate["positive_ratio_step200"]["mean"]
        - aggregate["positive_ratio_step100"]["mean"]
    )
    return {"aggregate": aggregate, "records": records}


def rank_quartiles(values_by_uid: dict[str, float]) -> tuple[dict[str, str], dict[str, Any]]:
    ordered = sorted(values_by_uid.items(), key=lambda item: (item[1], item[0]))
    assignments: dict[str, str] = {}
    groups: dict[str, list[float]] = {f"Q{index}": [] for index in range(1, 5)}
    for rank, (uid, value) in enumerate(ordered):
        quartile = min(4, int(rank * 4 / max(len(ordered), 1)) + 1)
        name = f"Q{quartile}"
        assignments[uid] = name
        groups[name].append(float(value))
    definitions = {
        name: {"object_count": len(group), "source_value": summarize(group)}
        for name, group in groups.items()
    }
    return assignments, definitions


def within_object_quartile(values: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    selected = values[active].float()
    labels = torch.zeros_like(values, dtype=torch.int8)
    if not selected.numel():
        return labels
    boundaries = torch.quantile(selected, torch.tensor((0.25, 0.50, 0.75)))
    labels[active] = torch.bucketize(selected, boundaries, right=False).to(torch.int8) + 1
    return labels


def metric_for_mask(
    payload: dict[str, Any], mask: torch.Tensor
) -> dict[str, Any] | None:
    active = payload["active_mask"].bool()
    selected = active & mask.bool()
    if not bool(selected.any().item()):
        return None
    depth_margin = payload["training_control_margins"]["depth_view_cyclic1"].float()
    hard_index = payload["hardest_control_index"].long()
    controls = list(payload["training_controls"])
    result: dict[str, Any] = {
        "voxel_count": int(selected.sum().item()),
        "depth_margin_mean": safe_selected_mean(depth_margin, selected),
        "depth_positive_ratio": safe_selected_mean(depth_margin.gt(0).float(), selected),
        "hardest_control_identity": {},
    }
    for index, name in enumerate(controls):
        result["hardest_control_identity"][name] = safe_selected_mean(
            hard_index.eq(index).float(), selected
        )
    return result


def summarize_group_rows(rows: list[dict[str, Any]], controls: list[str]) -> dict[str, Any]:
    return {
        "object_count": len(rows),
        "voxel_count": int(sum(int(row["voxel_count"]) for row in rows)),
        "depth_margin": summarize([float(row["depth_margin_mean"]) for row in rows]),
        "depth_positive_ratio": summarize(
            [float(row["depth_positive_ratio"]) for row in rows]
        ),
        "hardest_control_identity": {
            name: summarize(
                [float(row["hardest_control_identity"][name]) for row in rows]
            )
            for name in controls
        },
    }


def stratified_audit(maps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first = next(iter(maps.values()))
    controls = list(first["training_controls"])
    median_values = {
        uid: float(payload["depth_calibration_median_abs_residual"])
        for uid, payload in maps.items()
    }
    p90_values = {
        uid: float(payload["depth_calibration_p90_abs_residual"])
        for uid, payload in maps.items()
    }
    median_assignment, median_definition = rank_quartiles(median_values)
    p90_assignment, p90_definition = rank_quartiles(p90_values)
    rows_by_dimension: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def add(dimension: str, group: str, row: dict[str, Any] | None) -> None:
        if row is not None:
            rows_by_dimension.setdefault(dimension, {}).setdefault(group, []).append(row)

    for uid, payload in maps.items():
        active = payload["active_mask"].bool()
        audit = payload["audit_maps"]
        all_active = torch.ones_like(active)
        add("view_count", str(int(payload["views"])), metric_for_mask(payload, all_active))
        add(
            "calibration_median_residual_quartile",
            median_assignment[uid],
            metric_for_mask(payload, all_active),
        )
        add(
            "calibration_p90_residual_quartile",
            p90_assignment[uid],
            metric_for_mask(payload, all_active),
        )
        for dimension, map_name in (
            ("reliability_quartile", "raw_reliability"),
            ("depth_confidence_quartile", "depth_confidence"),
        ):
            labels = within_object_quartile(audit[map_name].float(), active)
            for quartile in range(1, 5):
                add(
                    dimension,
                    f"Q{quartile}",
                    metric_for_mask(payload, labels.eq(quartile)),
                )
        semantic = audit["depth_semantic_label"].to(torch.int8)
        for label, name in SEMANTIC_NAMES.items():
            add("depth_semantic", name, metric_for_mask(payload, semantic.eq(label)))

    dimensions = {
        dimension: {
            group: summarize_group_rows(rows, controls)
            for group, rows in sorted(groups.items())
        }
        for dimension, groups in rows_by_dimension.items()
    }
    reliability = dimensions.get("reliability_quartile", {})
    confidence = dimensions.get("depth_confidence_quartile", {})

    def q4_q1_delta(groups: dict[str, Any], metric: str) -> float | None:
        if "Q1" not in groups or "Q4" not in groups:
            return None
        return float(groups["Q4"][metric]["mean"] - groups["Q1"][metric]["mean"])

    return {
        "quartile_definitions": {
            "calibration_median_abs_residual": median_definition,
            "calibration_p90_abs_residual": p90_definition,
            "voxel_quartiles": "within-object active-voxel quartiles",
        },
        "dimensions": dimensions,
        "reliability_q4_minus_q1": {
            "depth_margin_mean": q4_q1_delta(reliability, "depth_margin"),
            "depth_positive_ratio": q4_q1_delta(
                reliability, "depth_positive_ratio"
            ),
        },
        "depth_confidence_q4_minus_q1": {
            "depth_margin_mean": q4_q1_delta(confidence, "depth_margin"),
            "depth_positive_ratio": q4_q1_delta(
                confidence, "depth_positive_ratio"
            ),
        },
    }


def primary_metrics(report: dict[str, Any]) -> dict[str, Any]:
    primary = report["primary"]
    return {
        "passed": bool(report["passed"]),
        "hard_margin_mean": float(primary["hard_margin_mean"]["object"]["mean"]),
        "hard_margin_ci_low": float(
            primary["hard_margin_mean"]["object_bootstrap_95_ci"][0]
        ),
        "voxel_positive_ratio": float(
            primary["voxel_positive_ratio"]["object"]["mean"]
        ),
        "local_object_pass_rate": float(primary["local_object_pass_rate"]),
        "failed_checks": [name for name, value in report["checks"].items() if not value],
    }


def neighborhood_comparison(
    exact_reports: dict[int, dict[str, Any]],
    gaussian_reports: dict[int, dict[str, Any]],
    *,
    coverage_gate: float,
    local_gate: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for step in STEPS:
        exact = exact_reports[step]
        gaussian = gaussian_reports[step]
        protocol = gaussian.get("gate_protocol", {})
        if protocol.get("spatial_tolerance") != "gaussian3":
            raise RuntimeError(f"step={step} is not a gaussian3 report")
        if not bool(protocol.get("spatial_tolerance_symmetric_across_branches")):
            raise RuntimeError(f"step={step} gaussian3 protocol is not symmetric")
        before = primary_metrics(exact)
        after = primary_metrics(gaussian)
        output[str(step)] = {
            "exact": before,
            "gaussian3": after,
            "delta": {
                key: float(after[key] - before[key])
                for key in (
                    "hard_margin_mean",
                    "hard_margin_ci_low",
                    "voxel_positive_ratio",
                    "local_object_pass_rate",
                )
            },
            "crosses_coverage_gate": bool(
                before["voxel_positive_ratio"] < coverage_gate
                and after["voxel_positive_ratio"] >= coverage_gate
            ),
            "crosses_local_gate": bool(
                before["local_object_pass_rate"] < local_gate
                and after["local_object_pass_rate"] >= local_gate
            ),
        }
    return output


def markdown_table_for_strata(groups: dict[str, Any]) -> list[str]:
    lines = [
        "| Group | Objects | Depth margin | Positive ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in groups.items():
        lines.append(
            f"| {name} | {row['object_count']} | "
            f"{row['depth_margin']['mean']:+.6f} | "
            f"{100.0 * row['depth_positive_ratio']['mean']:.2f}% |"
        )
    return lines


def write_markdown(report: dict[str, Any], path: Path) -> None:
    flip = report["voxel_flip"]["aggregate"]
    lines = [
        "# C0.2c前置三项审计",
        "",
        "## Step100 -> Step200 voxel翻转",
        "",
        f"- 新增正voxel（active占比）: `{100.0 * flip['negative_to_positive']['fraction_active']['mean']:.2f}%`",
        f"- 丢失正voxel（active占比）: `{100.0 * flip['positive_to_negative']['fraction_active']['mean']:.2f}%`",
        f"- 原正voxel平均margin增量: `{flip['original_positive_margin_delta_mean']['mean']:+.6f}`",
        f"- 原负voxel平均margin增量: `{flip['original_negative_margin_delta_mean']['mean']:+.6f}`",
        f"- 净positive coverage变化: `{100.0 * flip['net_positive_coverage_change']:+.2f}pp`",
        "",
    ]
    for step in ("100", "200"):
        strata = report["depth_strata"][step]
        lines.extend([f"## Step{step} reliability分层", ""])
        lines.extend(
            markdown_table_for_strata(
                strata["dimensions"].get("reliability_quartile", {})
            )
        )
        lines.extend(["", f"## Step{step} depth confidence分层", ""])
        lines.extend(
            markdown_table_for_strata(
                strata["dimensions"].get("depth_confidence_quartile", {})
            )
        )
        lines.extend(["", f"## Step{step} depth语义分层", ""])
        lines.extend(
            markdown_table_for_strata(
                strata["dimensions"].get("depth_semantic", {})
            )
        )
        lines.append("")
    lines.extend(
        [
            "## 对称gaussian3邻域审计",
            "",
            "| Step | Protocol | Hard mean | CI low | Voxel positive | Local pass |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for step, row in report["neighborhood_comparison"].items():
        for protocol in ("exact", "gaussian3"):
            value = row[protocol]
            lines.append(
                f"| {step} | {protocol} | {value['hard_margin_mean']:+.6f} | "
                f"{value['hard_margin_ci_low']:+.6f} | "
                f"{100.0 * value['voxel_positive_ratio']:.2f}% | "
                f"{100.0 * value['local_object_pass_rate']:.2f}% |"
            )
    lines.extend(
        [
            "",
            "完整对象记录、view-count/calibration quartile分层和hardest-control分布见`report.json`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    exact_dirs = {100: Path(args.step100_dir), 200: Path(args.step200_dir)}
    gaussian_dirs = {
        100: Path(args.gaussian_step100_dir),
        200: Path(args.gaussian_step200_dir),
    }
    exact_reports: dict[int, dict[str, Any]] = {}
    exact_maps: dict[int, dict[str, dict[str, Any]]] = {}
    gaussian_reports: dict[int, dict[str, Any]] = {}
    for step in STEPS:
        exact_reports[step], exact_maps[step] = load_maps(exact_dirs[step])
        gaussian_reports[step] = load_report(gaussian_dirs[step])
        if int(exact_reports[step]["checkpoint_step"]) != step:
            raise RuntimeError(f"exact report checkpoint mismatch at step={step}")
        if int(gaussian_reports[step]["checkpoint_step"]) != step:
            raise RuntimeError(f"gaussian report checkpoint mismatch at step={step}")
        if exact_reports[step]["cache_config_hash"] != gaussian_reports[step][
            "cache_config_hash"
        ]:
            raise RuntimeError(f"cache protocol mismatch at step={step}")

    report = {
        "format": AUDIT_VERSION,
        "stage": "pre-C0.2c voxel dynamics, depth strata, and tolerance audit",
        "inputs": {
            "exact": {str(step): str(exact_dirs[step].resolve()) for step in STEPS},
            "gaussian3": {
                str(step): str(gaussian_dirs[step].resolve()) for step in STEPS
            },
        },
        "protocol": {
            "object_balanced": True,
            "voxel_quartiles": "within-object active support",
            "sample_quartiles": "rank-balanced across objects",
            "depth_semantic_labels": {
                "surface": "abs(signed residual) <= fitted tolerance",
                "free_space": "voxel lies in front of observed surface",
                "occluded": "voxel lies behind observed surface",
                "boundary": "no semantic has >=60% weighted-view agreement",
            },
            "gaussian3": "fixed [1,2,1]^3 support-normalized evidence aggregation",
            "gaussian3_symmetric_across_correct_and_controls": True,
        },
        "voxel_flip": flip_audit(exact_maps[100], exact_maps[200]),
        "depth_strata": {
            str(step): stratified_audit(exact_maps[step]) for step in STEPS
        },
        "neighborhood_comparison": neighborhood_comparison(
            exact_reports,
            gaussian_reports,
            coverage_gate=float(args.min_voxel_positive_ratio),
            local_gate=float(args.min_object_local_pass_rate),
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "report.md")
    print(json.dumps({
        "format": report["format"],
        "voxel_flip": report["voxel_flip"]["aggregate"],
        "reliability_q4_minus_q1": {
            step: report["depth_strata"][step]["reliability_q4_minus_q1"]
            for step in ("100", "200")
        },
        "neighborhood_comparison": report["neighborhood_comparison"],
    }, indent=2))


if __name__ == "__main__":
    main()
