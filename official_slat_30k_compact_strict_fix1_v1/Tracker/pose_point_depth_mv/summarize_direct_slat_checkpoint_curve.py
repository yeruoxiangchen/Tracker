"""Summarize matched Direct-SLAT checkpoint evaluations.

This tool is intentionally read-only with respect to checkpoints and evaluation
artifacts.  It verifies that teacher-forced and same-noise Mesh reports use the
same protocols across checkpoints, then writes one immutable aggregate report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORMAT = "pose_point_depth_mv.direct_slat_checkpoint_curve.v1"
MESH_METRICS = (
    "chamfer_l1_improvement",
    "fscore_0p02_delta",
    "normal_consistency_delta",
    "largest_component_ratio_delta",
)
TRAIN_MEAN_KEYS = (
    "gain_vs_stock",
    "raw_delta_ratio_max",
    "raw_delta_excess_loss",
    "support_dropout_loss",
    "wrong_support_stock_loss",
)
ROLLOUT_MEAN_KEYS = (
    "rollout_gain_vs_stock",
    "rollout_loss",
    "rollout_stock_loss",
    "endpoint_x0_loss",
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def parse_step_paths(values: Sequence[str], label: str) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use STEP=PATH syntax: {value!r}")
        raw_step, raw_path = value.split("=", 1)
        step = int(raw_step)
        if step <= 0:
            raise ValueError(f"{label} step must be positive: {step}")
        if step in parsed:
            raise ValueError(f"duplicate {label} step: {step}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing for step {step}: {path}")
        parsed[step] = path
    return parsed


def finite_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {label}: {value!r}")
    return result


def mean_or_none(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        finite_float(row[key], key)
        for row in rows
        if key in row and row[key] is not None
    ]
    return statistics.fmean(values) if values else None


def rate(rows: Sequence[Mapping[str, Any]], predicate: Any) -> float | None:
    if not rows:
        return None
    return sum(bool(predicate(row)) for row in rows) / len(rows)


def bind_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def checkpoint_step_from_name(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("step_"):
        raise RuntimeError(f"checkpoint filename is not step_NNNNNN.pt: {path}")
    return int(stem.removeprefix("step_"))


def training_window(
    history: Sequence[Mapping[str, Any]],
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    rows = [
        row
        for row in history
        if start_step <= int(row["step"]) <= end_step
    ]
    expected_count = end_step - start_step + 1
    if len(rows) != expected_count:
        raise RuntimeError(
            f"training history window {start_step}-{end_step} has "
            f"{len(rows)} rows, expected {expected_count}"
        )
    rollout_rows = [row for row in rows if bool(row.get("rollout_evaluated"))]
    support_dropout_rows = [
        row for row in rows if bool(row.get("support_dropout_evaluated"))
    ]
    wrong_support_rows = [
        row for row in rows if bool(row.get("wrong_support_evaluated"))
    ]
    return {
        "start_step": start_step,
        "end_step": end_step,
        "micro_steps": len(rows),
        "rollout_events": len(rollout_rows),
        "support_dropout_events": len(support_dropout_rows),
        "wrong_support_events": len(wrong_support_rows),
        "means": {
            key: mean_or_none(rows, key)
            for key in TRAIN_MEAN_KEYS
        }
        | {
            key: mean_or_none(rollout_rows, key)
            for key in ROLLOUT_MEAN_KEYS
        },
        "rates": {
            "training_bound_flag_rate": rate(
                rows, lambda row: bool(row.get("delta_clip_activated"))
            ),
            "training_scale_reduced_rate": rate(
                rows,
                lambda row: finite_float(
                    row.get("delta_clip_scale", 1.0), "delta_clip_scale"
                )
                < 1.0 - 1e-6,
            ),
        },
    }


def object_ids(report: Mapping[str, Any]) -> list[str]:
    rows = report.get("object_rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("evaluation report has no object_rows")
    result = []
    for row in rows:
        if not isinstance(row, dict) or "object_uid" not in row:
            raise RuntimeError("invalid object_rows entry")
        result.append(str(row["object_uid"]))
    return result


def protocol_signature_teacher(report: Mapping[str, Any]) -> dict[str, Any]:
    cache = report.get("cache_identity", {})
    return {
        "evaluation": report.get("evaluation"),
        "teacher_prediction_policy": report.get("teacher_prediction_policy"),
        "manifest_sha256": cache.get("manifest_sha256"),
        "object_uid_hash": cache.get("object_uid_hash"),
        "object_ids": object_ids(report),
        "noise_seeds": report.get("noise_seeds"),
        "t_values": report.get("t_values"),
        "support_scale": report.get("support_scale"),
        "slat_delta_policy": report.get("slat_delta_policy"),
    }


def protocol_signature_mesh(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "object_ids": object_ids(report),
        "joint_seeds": report.get("joint_seeds"),
        "same_coordinates": report.get("same_coordinates"),
        "same_noise": report.get("same_noise"),
        "slat_delta_policy": report.get("slat_delta_policy"),
    }


def ensure_same_protocol(
    reports: Mapping[int, Mapping[str, Any]],
    signature_fn: Any,
    label: str,
) -> dict[str, Any]:
    reference_step = min(reports)
    reference = signature_fn(reports[reference_step])
    for step, report in sorted(reports.items()):
        candidate = signature_fn(report)
        if candidate != reference:
            raise RuntimeError(
                f"{label} protocol mismatch between step {reference_step} "
                f"and step {step}"
            )
    return reference


def summarize_teacher(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    records = report.get("records")
    if not isinstance(summary, dict) or not isinstance(records, list):
        raise RuntimeError("invalid teacher report")
    wrong_values = [
        finite_float(
            row["wrong_support_stock_reversion_advantage"],
            "wrong_support_stock_reversion_advantage",
        )
        for row in records
        if row.get("wrong_support_stock_reversion_advantage") is not None
    ]
    return {
        "object_count": int(report["object_count"]),
        "record_count": len(records),
        "full_gain_vs_stock": summary["full_gain_vs_stock"],
        "lora_only_gain_vs_stock": summary.get("lora_only_gain_vs_stock"),
        "adapter_only_gain_vs_stock": summary.get("adapter_only_gain_vs_stock"),
        "full_gain_vs_lora_only": summary.get("full_gain_vs_lora_only"),
        "full_gain_vs_adapter_only": summary.get("full_gain_vs_adapter_only"),
        "wrong_support_stock_reversion": {
            "count": len(wrong_values),
            "mean": statistics.fmean(wrong_values) if wrong_values else None,
            "median": statistics.median(wrong_values) if wrong_values else None,
            "positive_rate": (
                sum(value > 0.0 for value in wrong_values) / len(wrong_values)
                if wrong_values
                else None
            ),
        },
        "core_pass": bool(report.get("core_pass")),
        "strong_pass": bool(report.get("strong_pass")),
        "scope_guard": report.get("scope_guard"),
    }


def summarize_mesh(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("invalid Mesh report")
    for metric in MESH_METRICS:
        if metric not in summary:
            raise RuntimeError(f"Mesh report is missing {metric}")
    return {
        "formal": bool(report.get("formal")),
        "object_count": int(report["object_count"]),
        "joint_seeds": report["joint_seeds"],
        "summary": {metric: summary[metric] for metric in MESH_METRICS},
        "scope_guard": report.get("scope_guard"),
    }


def discover_flow_stats(mesh_report_path: Path) -> list[Path]:
    mesh_root = mesh_report_path.parent
    paths = sorted(mesh_root.glob("slat/*/full_flow_stats.json"))
    if not paths:
        raise RuntimeError(f"no full_flow_stats.json below {mesh_root}")
    return paths


def summarize_flow_stats(paths: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    policy: dict[str, Any] | None = None
    for path in paths:
        payload = load_json(path)
        current_policy = {
            "policy_version": payload.get("policy_version"),
            "slat_delta_scale": payload.get("slat_delta_scale"),
            "slat_delta_rms_ratio_cap": payload.get("slat_delta_rms_ratio_cap"),
            "slat_delta_bound_mode": payload.get("slat_delta_bound_mode"),
            "support_interval_policy": payload.get("support_interval_policy"),
            "cfg_strength": payload.get("cfg_strength"),
            "cfg_interval": payload.get("cfg_interval"),
        }
        if policy is None:
            policy = current_policy
        elif current_policy != policy:
            raise RuntimeError(f"flow policy mismatch: {path}")
        current_rows = payload.get("by_timestep")
        if not isinstance(current_rows, list):
            raise RuntimeError(f"invalid by_timestep: {path}")
        rows.extend(current_rows)
    active = [
        row
        for row in rows
        if finite_float(row.get("support_active", 0.0), "support_active") > 0.5
    ]
    inactive = [
        row
        for row in rows
        if finite_float(row.get("support_active", 0.0), "support_active") <= 0.5
    ]
    if not active:
        raise RuntimeError("no support-active deployed flow rows")

    def ratios(
        selected: Iterable[Mapping[str, Any]], numerator: str
    ) -> list[float]:
        values: list[float] = []
        for row in selected:
            denominator = finite_float(
                row["stock_guided_velocity_rms"], "stock_guided_velocity_rms"
            )
            if denominator <= 0:
                raise RuntimeError("stock_guided_velocity_rms must be positive")
            values.append(finite_float(row[numerator], numerator) / denominator)
        return values

    raw_ratios = ratios(active, "raw_guided_delta_rms")
    effective_ratios = ratios(active, "effective_guided_delta_rms")
    scales = [
        finite_float(row["guided_delta_clip_scale"], "guided_delta_clip_scale")
        for row in active
    ]
    cap = finite_float(
        (policy or {})["slat_delta_rms_ratio_cap"],
        "slat_delta_rms_ratio_cap",
    )
    if cap <= 0:
        raise RuntimeError("slat_delta_rms_ratio_cap must be positive")
    official_clip_flags = [
        finite_float(
            row.get("guided_delta_clip_activated", 0.0),
            "guided_delta_clip_activated",
        )
        > 0.5
        for row in active
    ]
    raw_over_cap = [value > cap for value in raw_ratios]
    low_t_deltas = [
        finite_float(
            row["effective_guided_delta_rms"], "effective_guided_delta_rms"
        )
        for row in inactive
    ]
    return {
        "files": len(paths),
        "timestep_calls": len(rows),
        "support_active_calls": len(active),
        "support_inactive_calls": len(inactive),
        "policy": policy,
        "active_interval": {
            "raw_ratio_over_cap_calls": sum(raw_over_cap),
            "raw_ratio_over_cap_rate": sum(raw_over_cap) / len(raw_over_cap),
            "official_clip_flag_calls": sum(official_clip_flags),
            "official_clip_flag_rate": sum(official_clip_flags)
            / len(official_clip_flags),
            "smooth_scale_reduced_calls": sum(
                scale < 1.0 - 1e-6 for scale in scales
            ),
            "smooth_scale_reduced_rate": sum(
                scale < 1.0 - 1e-6 for scale in scales
            )
            / len(scales),
            "clip_scale_mean": statistics.fmean(scales),
            "clip_scale_min": min(scales),
            "raw_delta_to_stock_rms_mean": statistics.fmean(raw_ratios),
            "raw_delta_to_stock_rms_max": max(raw_ratios),
            "effective_delta_to_stock_rms_mean": statistics.fmean(
                effective_ratios
            ),
            "effective_delta_to_stock_rms_max": max(effective_ratios),
        },
        "inactive_interval": {
            "max_effective_guided_delta_rms": max(low_t_deltas, default=0.0),
            "exact_stock_pass": all(value == 0.0 for value in low_t_deltas),
        },
    }


def metric_value(
    row: Mapping[str, Any], section: str, metric: str, statistic: str
) -> Any:
    if section == "training":
        return row["training_window"]["means"].get(metric)
    if section == "teacher":
        return row["teacher"][metric].get(statistic)
    if section == "mesh":
        return row["mesh"]["summary"][metric].get(statistic)
    if section == "flow":
        return row["deployed_flow"]["active_interval"].get(metric)
    raise KeyError(section)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "step",
        "train_gain_vs_stock_mean",
        "train_rollout_gain_vs_stock_mean",
        "train_endpoint_x0_loss_mean",
        "train_raw_delta_excess_loss_mean",
        "teacher_gain_mean",
        "teacher_gain_median",
        "teacher_win_rate",
        "mesh_chamfer_mean",
        "mesh_chamfer_median",
        "mesh_chamfer_win_rate",
        "mesh_fscore_mean",
        "mesh_fscore_median",
        "mesh_fscore_win_rate",
        "mesh_normal_mean",
        "mesh_lcr_mean",
        "deployed_raw_ratio_over_cap_rate",
        "deployed_official_clip_flag_rate",
        "deployed_raw_delta_ratio_mean",
        "deployed_raw_delta_ratio_max",
        "deployed_effective_delta_ratio_mean",
        "deployed_effective_delta_ratio_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "step": row["step"],
                    "train_gain_vs_stock_mean": metric_value(
                        row, "training", "gain_vs_stock", "mean"
                    ),
                    "train_rollout_gain_vs_stock_mean": metric_value(
                        row, "training", "rollout_gain_vs_stock", "mean"
                    ),
                    "train_endpoint_x0_loss_mean": metric_value(
                        row, "training", "endpoint_x0_loss", "mean"
                    ),
                    "train_raw_delta_excess_loss_mean": metric_value(
                        row, "training", "raw_delta_excess_loss", "mean"
                    ),
                    "teacher_gain_mean": metric_value(
                        row, "teacher", "full_gain_vs_stock", "mean"
                    ),
                    "teacher_gain_median": metric_value(
                        row, "teacher", "full_gain_vs_stock", "median"
                    ),
                    "teacher_win_rate": metric_value(
                        row, "teacher", "full_gain_vs_stock", "positive_rate"
                    ),
                    "mesh_chamfer_mean": metric_value(
                        row, "mesh", "chamfer_l1_improvement", "mean"
                    ),
                    "mesh_chamfer_median": metric_value(
                        row, "mesh", "chamfer_l1_improvement", "median"
                    ),
                    "mesh_chamfer_win_rate": metric_value(
                        row,
                        "mesh",
                        "chamfer_l1_improvement",
                        "positive_rate",
                    ),
                    "mesh_fscore_mean": metric_value(
                        row, "mesh", "fscore_0p02_delta", "mean"
                    ),
                    "mesh_fscore_median": metric_value(
                        row, "mesh", "fscore_0p02_delta", "median"
                    ),
                    "mesh_fscore_win_rate": metric_value(
                        row, "mesh", "fscore_0p02_delta", "positive_rate"
                    ),
                    "mesh_normal_mean": metric_value(
                        row, "mesh", "normal_consistency_delta", "mean"
                    ),
                    "mesh_lcr_mean": metric_value(
                        row,
                        "mesh",
                        "largest_component_ratio_delta",
                        "mean",
                    ),
                    "deployed_raw_ratio_over_cap_rate": metric_value(
                        row, "flow", "raw_ratio_over_cap_rate", "mean"
                    ),
                    "deployed_official_clip_flag_rate": metric_value(
                        row, "flow", "official_clip_flag_rate", "mean"
                    ),
                    "deployed_raw_delta_ratio_mean": metric_value(
                        row, "flow", "raw_delta_to_stock_rms_mean", "mean"
                    ),
                    "deployed_raw_delta_ratio_max": metric_value(
                        row, "flow", "raw_delta_to_stock_rms_max", "mean"
                    ),
                    "deployed_effective_delta_ratio_mean": metric_value(
                        row,
                        "flow",
                        "effective_delta_to_stock_rms_mean",
                        "mean",
                    ),
                    "deployed_effective_delta_ratio_max": metric_value(
                        row,
                        "flow",
                        "effective_delta_to_stock_rms_max",
                        "mean",
                    ),
                }
            )


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "Direct SLAT matched checkpoint curve",
        "====================================",
        "DESCRIPTIVE / EXPLORATORY: teacher evidence is diagnostic; "
        "the 6-object Mesh protocol cannot establish a science claim.",
        "",
        "step | teacher gain(mean/median/win) | "
        "Mesh Chamfer(mean/median/win) | F-score mean | "
        "raw>cap | raw/effective delta ratio",
        "-----|-------------------------------|"
        "--------------------------------|--------------|"
        "---------------|--------------------------",
    ]
    for row in rows:
        teacher = row["teacher"]["full_gain_vs_stock"]
        mesh = row["mesh"]["summary"]
        flow = row["deployed_flow"]["active_interval"]
        lines.append(
            f"{row['step']:4d} | "
            f"{fmt(teacher['mean'])}/{fmt(teacher['median'])}/"
            f"{float(teacher['positive_rate']):.3f} | "
            f"{fmt(mesh['chamfer_l1_improvement']['mean'])}/"
            f"{fmt(mesh['chamfer_l1_improvement']['median'])}/"
            f"{float(mesh['chamfer_l1_improvement']['positive_rate']):.3f} | "
            f"{fmt(mesh['fscore_0p02_delta']['mean'])} | "
            f"{float(flow['raw_ratio_over_cap_rate']):.3f} | "
            f"{float(flow['raw_delta_to_stock_rms_mean']):.4f}/"
            f"{float(flow['effective_delta_to_stock_rms_mean']):.4f}"
        )
    lines.extend(
        [
            "",
            "Interpretation rules:",
            "- Prefer Mesh utility over teacher-forced or training proxy gains.",
            "- A checkpoint is not confirmatory when Mesh bootstrap intervals cross zero.",
            "- A rising teacher/rollout gain with flat Mesh utility and rising "
            "raw>cap rate indicates proxy/rollout/decoder mismatch.",
            "- raw>cap is the fraction of support-active deployed calls whose "
            "raw residual RMS ratio exceeds the configured cap. Smooth scaling "
            "itself is continuously active and is not treated as saturation.",
            "- Any selected checkpoint must be re-tested on a fresh blind, larger, "
            "multi-seed protocol.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate matched Direct-SLAT checkpoint evaluations"
    )
    parser.add_argument("--training_report", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="STEP=PATH",
        help="repeat once per checkpoint",
    )
    parser.add_argument(
        "--teacher_report",
        action="append",
        default=[],
        metavar="STEP=PATH",
        help="repeat once per checkpoint",
    )
    parser.add_argument(
        "--mesh_report",
        action="append",
        default=[],
        metavar="STEP=PATH",
        help="repeat once per checkpoint",
    )
    parser.add_argument(
        "--expected_steps",
        default="100,200,300,400",
        help="comma-separated checkpoint steps",
    )
    parser.add_argument(
        "--window_steps",
        type=int,
        default=100,
        help="training-history window ending at each checkpoint",
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    expected_steps = [int(value) for value in args.expected_steps.split(",")]
    if not expected_steps or expected_steps != sorted(set(expected_steps)):
        raise ValueError("--expected_steps must be sorted and unique")
    if args.window_steps <= 0:
        raise ValueError("--window_steps must be positive")

    training_report_path = args.training_report.expanduser().resolve()
    if not training_report_path.is_file():
        raise FileNotFoundError(training_report_path)
    checkpoints = parse_step_paths(args.checkpoint, "checkpoint")
    teacher_paths = parse_step_paths(args.teacher_report, "teacher_report")
    mesh_paths = parse_step_paths(args.mesh_report, "mesh_report")
    expected_set = set(expected_steps)
    for label, mapping in (
        ("checkpoint", checkpoints),
        ("teacher_report", teacher_paths),
        ("mesh_report", mesh_paths),
    ):
        if set(mapping) != expected_set:
            raise RuntimeError(
                f"{label} steps {sorted(mapping)} != expected {expected_steps}"
            )

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable output already exists: {output_dir}")

    training_report = load_json(training_report_path)
    history = training_report.get("history")
    if not isinstance(history, list):
        raise RuntimeError("training report has no history")
    if int(training_report.get("step", -1)) < max(expected_steps):
        raise RuntimeError("training report does not reach the final expected step")

    teacher_reports = {step: load_json(path) for step, path in teacher_paths.items()}
    mesh_reports = {step: load_json(path) for step, path in mesh_paths.items()}
    teacher_protocol = ensure_same_protocol(
        teacher_reports, protocol_signature_teacher, "teacher"
    )
    mesh_protocol = ensure_same_protocol(
        mesh_reports, protocol_signature_mesh, "Mesh"
    )

    previous_step = 0
    rows: list[dict[str, Any]] = []
    for step in expected_steps:
        checkpoint = checkpoints[step]
        filename_step = checkpoint_step_from_name(checkpoint)
        if filename_step != step:
            raise RuntimeError(
                f"checkpoint filename step {filename_step} != declared step {step}"
            )
        teacher = teacher_reports[step]
        mesh = mesh_reports[step]
        if int(teacher.get("checkpoint_step", -1)) != step:
            raise RuntimeError(f"teacher checkpoint_step mismatch at {step}")
        if int(mesh.get("checkpoint_step", -1)) != step:
            raise RuntimeError(f"Mesh checkpoint_step mismatch at {step}")
        start_step = max(previous_step + 1, step - args.window_steps + 1)
        rows.append(
            {
                "step": step,
                "checkpoint": bind_file(checkpoint),
                "training_window": training_window(history, start_step, step),
                "teacher_report": bind_file(teacher_paths[step]),
                "teacher": summarize_teacher(teacher),
                "mesh_report": bind_file(mesh_paths[step]),
                "mesh": summarize_mesh(mesh),
                "deployed_flow": summarize_flow_stats(
                    discover_flow_stats(mesh_paths[step])
                ),
            }
        )
        previous_step = step

    report = {
        "format": FORMAT,
        "formal": False,
        "scope": "matched exploratory checkpoint selection",
        "expected_steps": expected_steps,
        "window_steps": args.window_steps,
        "training_report": bind_file(training_report_path),
        "protocols": {
            "teacher": teacher_protocol,
            "mesh": mesh_protocol,
        },
        "rows": rows,
        "decision": {
            "automatic_checkpoint_selection": False,
            "reason": (
                "The matched Mesh protocol has six objects and one joint seed; "
                "it diagnoses trends but cannot establish Full > Stock."
            ),
            "required_next_gate": (
                "fresh blind, larger-object, multi-seed same-coordinate "
                "same-noise Mesh evaluation"
            ),
        },
        "code_binding": bind_file(Path(__file__).resolve()),
    }

    output_dir.mkdir(parents=True)
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(output_dir / "curve.csv", rows)
    write_summary(output_dir / "summary.txt", rows)
    print((output_dir / "summary.txt").read_text(encoding="utf-8"), end="")
    print(f"\nreport: {output_dir / 'report.json'}")
    print(f"curve: {output_dir / 'curve.csv'}")


if __name__ == "__main__":
    main()
