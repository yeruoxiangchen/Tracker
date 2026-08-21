#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_OUTPUTS_ROOT = Path(
    "/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs"
)
RUN_NAME_TEMPLATE = (
    "pointpose_j1a2a_content_visual8_"
    "overfit16_s50_seed{seed}_2gpu_bf16"
)

EXPECTED_GRADIENT_GROUPS = (
    "physical_encoder",
    "visual_query_projection",
    "physical_cross_attention",
    "physical_output_projection",
    "alignment_gate",
)

TRAJECTORY_METRICS = (
    "flow_loss",
    "stock_flow_loss",
    "shuffled_flow_loss",
    "stock_minus_correct_flow_loss",
    "shuffled_minus_correct_flow_loss",
    "relative_correct_gain",
    "shuffled_stock_loss_relative",
    "alignment_loss",
    "condition_delta_to_stock_ratio",
    "regularization_delta_ratio",
    "positive_alignment_probability",
    "negative_alignment_probability",
    "clip_total_norm",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    output: list[float] = []
    for row in rows:
        value = row.get(key)
        if is_finite_number(value):
            output.append(float(value))
    return output


def tail_mean(
    rows: list[dict[str, Any]],
    key: str,
    count: int = 3,
) -> float | None:
    values = numeric_values(rows, key)
    if not values:
        return None
    return float(statistics.fmean(values[-count:]))


def final_value(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:
    for row in reversed(rows):
        value = row.get(key)
        if is_finite_number(value):
            return float(value)
    return None


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if is_finite_number(value):
        number = float(value)
        if number == 0:
            return "0"
        if abs(number) < 1.0e-4 or abs(number) >= 1.0e4:
            return f"{number:.3e}"
        return f"{number:.{digits}f}"
    return str(value)


def parse_seeds(text: str) -> list[int]:
    seeds = [
        int(item.strip())
        for item in str(text).split(",")
        if item.strip()
    ]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"expected unique seeds, got {text!r}")
    return seeds


def summarize_gradient_groups(
    history: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    group_names: set[str] = set()
    for row in history:
        gradients = row.get("gradient_norms", {})
        if isinstance(gradients, dict):
            group_names.update(str(key) for key in gradients)

    summary: dict[str, dict[str, Any]] = {}
    for group in sorted(group_names):
        values: list[float] = []
        for row in history:
            gradients = row.get("gradient_norms", {})
            if not isinstance(gradients, dict):
                continue
            value = gradients.get(group)
            if is_finite_number(value):
                values.append(float(value))

        summary[group] = {
            "count": len(values),
            "finite": bool(values) and all(
                math.isfinite(value) for value in values
            ),
            "nonzero_count": sum(value > 0 for value in values),
            "nonzero_rate": (
                sum(value > 0 for value in values) / len(values)
                if values
                else 0.0
            ),
            "first": values[0] if values else None,
            "last": values[-1] if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def summarize_stages(
    history: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    final_stage_rows: dict[str, dict[str, Any]] = {}

    for row in reversed(history):
        per_stage = row.get("per_stage", {})
        if not isinstance(per_stage, dict):
            continue
        for stage_name, stage_values in per_stage.items():
            if (
                stage_name not in final_stage_rows
                and isinstance(stage_values, dict)
            ):
                final_stage_rows[str(stage_name)] = stage_values

    output: dict[str, dict[str, Any]] = {}
    for stage_name, row in sorted(final_stage_rows.items()):
        correct_gate = row.get("correct_alignment_probability")
        shuffled_gate = row.get("shuffled_alignment_probability")

        gate_gap = None
        if (
            is_finite_number(correct_gate)
            and is_finite_number(shuffled_gate)
        ):
            gate_gap = float(correct_gate) - float(shuffled_gate)

        output[stage_name] = {
            "correct_alignment_probability": correct_gate,
            "shuffled_alignment_probability": shuffled_gate,
            "alignment_probability_gap": gate_gap,
            "correct_delta_to_hidden_ratio": row.get(
                "correct_delta_to_hidden_ratio"
            ),
            "shuffled_delta_to_hidden_ratio": row.get(
                "shuffled_delta_to_hidden_ratio"
            ),
            "correct_attended_rms": row.get("correct_attended_rms"),
            "shuffled_attended_rms": row.get("shuffled_attended_rms"),
            "correct_shuffled_attended_rms": row.get(
                "correct_shuffled_attended_rms"
            ),
            "correct_shuffled_attended_cosine": row.get(
                "correct_shuffled_attended_cosine"
            ),
            "correct_shuffled_context_delta_rms": row.get(
                "correct_shuffled_context_delta_rms"
            ),
            "correct_shuffled_context_delta_cosine": row.get(
                "correct_shuffled_context_delta_cosine"
            ),
        }

    return output


def summarize_seed(
    *,
    outputs_root: Path,
    seed: int,
    expected_updates: int,
) -> dict[str, Any]:
    run_dir = outputs_root / RUN_NAME_TEMPLATE.format(seed=seed)
    train_path = run_dir / "train_report.json"
    audit_path = run_dir / "finite_run_audit.json"

    train = load_json(train_path)
    audit = load_json(audit_path)

    history_raw = train.get("history", [])
    history = [
        row for row in history_raw
        if isinstance(row, dict)
    ]

    model = train.get("model", {})
    if not isinstance(model, dict):
        model = {}

    trainable_audit = model.get("trainable_parameter_audit", {})
    if not isinstance(trainable_audit, dict):
        trainable_audit = {}

    architecture_audit = model.get("architecture_audit", {})
    if not isinstance(architecture_audit, dict):
        architecture_audit = {}

    gradient_groups = summarize_gradient_groups(history)
    stage_summary = summarize_stages(history)

    all_rows_finite = bool(history) and all(
        bool(row.get("forward_finite", False))
        and bool(row.get("gradient_finite", False))
        and bool(row.get("update_finite", False))
        for row in history
    )
    all_logged_updates_applied = bool(history) and all(
        bool(row.get("optimizer_step_applied", False))
        for row in history
    )

    expected_groups_present = all(
        group in gradient_groups
        for group in EXPECTED_GRADIENT_GROUPS
    )
    expected_groups_nonzero = all(
        gradient_groups.get(group, {}).get("nonzero_count", 0) > 0
        for group in EXPECTED_GRADIENT_GROUPS
    )
    expected_groups_finite = all(
        gradient_groups.get(group, {}).get("finite", False)
        for group in EXPECTED_GRADIENT_GROUPS
    )

    frozen_modules_ok = (
        int(trainable_audit.get("vggt_trainable", -1)) == 0
        and int(trainable_audit.get("image_encoder_trainable", -1)) == 0
        and int(trainable_audit.get("stock_bridge_trainable", -1)) == 0
        and int(trainable_audit.get("stock_flow_trainable", -1)) == 0
        and int(trainable_audit.get("adapter_trainable", 0)) > 0
    )

    architecture_ok = bool(
        architecture_audit.get("passed", True)
    )

    checks = {
        "finite_audit_passed": bool(audit.get("passed", False)),
        "expected_updates": (
            int(audit.get("applied_optimizer_updates", -1))
            == expected_updates
        ),
        "completed_step": (
            int(audit.get("completed_global_step", -1))
            == expected_updates
        ),
        "checkpoint_step": (
            int(audit.get("checkpoint_step", -1))
            == expected_updates
        ),
        "nonfinite_attempts_zero": (
            int(audit.get("nonfinite_attempts", -1)) == 0
            and int(train.get("nonfinite_attempts", -1)) == 0
        ),
        "all_logged_rows_finite": all_rows_finite,
        "all_logged_updates_applied": all_logged_updates_applied,
        "expected_gradient_groups_present": expected_groups_present,
        "expected_gradient_groups_finite": expected_groups_finite,
        "expected_gradient_groups_nonzero": expected_groups_nonzero,
        "frozen_modules_ok": frozen_modules_ok,
        "architecture_audit_passed": architecture_ok,
        "checkpoint_trainable_state_finite": bool(
            audit.get(
                "checks", {}
            ).get("checkpoint_trainable_state_finite", False)
        ),
        "checkpoint_optimizer_state_finite": bool(
            audit.get(
                "checks", {}
            ).get("checkpoint_optimizer_state_finite", False)
        ),
        "checkpoint_scaler_state_finite": bool(
            audit.get(
                "checks", {}
            ).get("checkpoint_scaler_state_finite", False)
        ),
    }

    trajectory: dict[str, Any] = {}
    for key in TRAJECTORY_METRICS:
        trajectory[key] = {
            "first": (
                numeric_values(history, key)[0]
                if numeric_values(history, key)
                else None
            ),
            "last": final_value(history, key),
            "last3_mean": tail_mean(history, key, 3),
        }

    positive_gate = final_value(
        history, "positive_alignment_probability"
    )
    negative_gate = final_value(
        history, "negative_alignment_probability"
    )
    gate_gap = None
    if positive_gate is not None and negative_gate is not None:
        gate_gap = positive_gate - negative_gate

    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "train_report": str(train_path),
        "finite_audit": str(audit_path),
        "stage": train.get("stage"),
        "amp_dtype": train.get("args", {}).get("amp_dtype"),
        "world_size": train.get("world_size"),
        "effective_batch_size": train.get("effective_batch_size"),
        "dataset_size": train.get("dataset_size"),
        "unique_object_count": train.get("unique_object_count"),
        "history_row_count": len(history),
        "first_logged_step": (
            history[0].get("step") if history else None
        ),
        "last_logged_step": (
            history[-1].get("step") if history else None
        ),
        "applied_optimizer_updates": audit.get(
            "applied_optimizer_updates"
        ),
        "completed_global_step": audit.get("completed_global_step"),
        "checkpoint_step": audit.get("checkpoint_step"),
        "nonfinite_attempts": audit.get("nonfinite_attempts"),
        "trainable_parameter_audit": trainable_audit,
        "architecture_audit": architecture_audit,
        "checks": checks,
        "gradient_groups": gradient_groups,
        "trajectory": trajectory,
        "final_alignment_probability_gap": gate_gap,
        "per_stage": stage_summary,
        "passed": all(checks.values()),
    }


def markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in rows
    )
    return lines


def build_markdown(report: dict[str, Any]) -> str:
    seeds = report["seeds"]
    lines = [
        "# J1a.2-A 36.4 Training Health Summary",
        "",
        f"- overall: `{'PASS' if report['passed'] else 'FAIL'}`",
        f"- expected updates per seed: `{report['expected_updates']}`",
        "",
        "## Engineering health",
        "",
    ]

    health_rows: list[list[str]] = []
    for row in seeds:
        health_rows.append(
            [
                str(row["seed"]),
                fmt(row["passed"]),
                str(row["applied_optimizer_updates"]),
                str(row["completed_global_step"]),
                str(row["nonfinite_attempts"]),
                fmt(row["checks"]["all_logged_rows_finite"]),
                fmt(row["checks"]["expected_gradient_groups_nonzero"]),
                fmt(row["checks"]["frozen_modules_ok"]),
                fmt(row["checks"]["architecture_audit_passed"]),
            ]
        )

    lines.extend(
        markdown_table(
            [
                "seed",
                "health",
                "updates",
                "step",
                "nonfinite",
                "rows finite",
                "all grad groups",
                "frozen",
                "arch",
            ],
            health_rows,
        )
    )

    lines.extend(
        [
            "",
            "## Training trajectory",
            "",
            "These are noisy training-log diagnostics, not the final "
            "correct-vs-shuffled functionality decision.",
            "",
        ]
    )

    trajectory_rows: list[list[str]] = []
    for row in seeds:
        trajectory = row["trajectory"]
        trajectory_rows.append(
            [
                str(row["seed"]),
                fmt(
                    trajectory[
                        "stock_minus_correct_flow_loss"
                    ]["last3_mean"]
                ),
                fmt(
                    trajectory[
                        "shuffled_minus_correct_flow_loss"
                    ]["last3_mean"]
                ),
                fmt(
                    trajectory["relative_correct_gain"]["last3_mean"]
                ),
                fmt(
                    trajectory[
                        "shuffled_stock_loss_relative"
                    ]["last3_mean"]
                ),
                fmt(
                    trajectory[
                        "condition_delta_to_stock_ratio"
                    ]["last"]
                ),
                fmt(row["final_alignment_probability_gap"]),
                fmt(
                    trajectory["clip_total_norm"]["last"]
                ),
            ]
        )

    lines.extend(
        markdown_table(
            [
                "seed",
                "stock-correct last3",
                "shuffled-correct last3",
                "relative gain last3",
                "shuffled-stock rel last3",
                "final cond delta/stock",
                "final gate gap",
                "final grad norm",
            ],
            trajectory_rows,
        )
    )

    lines.extend(["", "## Gradient groups", ""])
    gradient_rows: list[list[str]] = []
    for row in seeds:
        for group in EXPECTED_GRADIENT_GROUPS:
            values = row["gradient_groups"].get(group, {})
            gradient_rows.append(
                [
                    str(row["seed"]),
                    group,
                    fmt(values.get("first")),
                    fmt(values.get("last")),
                    fmt(values.get("max")),
                    fmt(values.get("nonzero_rate")),
                ]
            )

    lines.extend(
        markdown_table(
            [
                "seed",
                "group",
                "first",
                "last",
                "max",
                "nonzero rate",
            ],
            gradient_rows,
        )
    )

    stage_rows: list[list[str]] = []
    for row in seeds:
        for stage_name, stage in row["per_stage"].items():
            stage_rows.append(
                [
                    str(row["seed"]),
                    stage_name,
                    fmt(stage.get(
                        "correct_alignment_probability"
                    )),
                    fmt(stage.get(
                        "shuffled_alignment_probability"
                    )),
                    fmt(stage.get(
                        "alignment_probability_gap"
                    )),
                    fmt(stage.get(
                        "correct_delta_to_hidden_ratio"
                    )),
                    fmt(stage.get(
                        "shuffled_delta_to_hidden_ratio"
                    )),
                    fmt(stage.get(
                        "correct_shuffled_attended_cosine"
                    )),
                    fmt(stage.get(
                        "correct_shuffled_context_delta_cosine"
                    )),
                ]
            )

    if stage_rows:
        lines.extend(["", "## Final per-stage diagnostics", ""])
        lines.extend(
            markdown_table(
                [
                    "seed",
                    "stage",
                    "correct gate",
                    "shuffled gate",
                    "gate gap",
                    "correct delta ratio",
                    "shuffled delta ratio",
                    "attended cosine",
                    "delta cosine",
                ],
                stage_rows,
            )
        )

    lines.extend(["", "## Decision", ""])
    if report["passed"]:
        lines.append(
            "36.4 engineering health is `PASS`. The three runs are "
            "eligible for the fixed-noise, multi-t 36.5 evaluation."
        )
    else:
        lines.append(
            "36.4 engineering health is `FAIL`. Do not use these "
            "checkpoints for a functionality conclusion before fixing "
            "the failed checks."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize J1a.2-A section 36.4 overfit16 training health "
            "for seeds 42/43/44."
        )
    )
    parser.add_argument(
        "--outputs_root",
        default=str(DEFAULT_OUTPUTS_ROOT),
    )
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--expected_updates", type=int, default=50)
    parser.add_argument(
        "--output_dir",
        default=(
            "/home/zjr/Tracker/reconvggt_ar_adapter_a/outputs/"
            "pointpose_j1a2a_content_visual8_"
            "overfit16_s50_training_health_summary"
        ),
    )
    parser.add_argument("--fail_on_health", action="store_true")
    args = parser.parse_args()

    if args.expected_updates <= 0:
        raise ValueError("expected_updates must be positive")

    seeds = parse_seeds(args.seeds)
    outputs_root = Path(args.outputs_root)

    summaries = [
        summarize_seed(
            outputs_root=outputs_root,
            seed=seed,
            expected_updates=int(args.expected_updates),
        )
        for seed in seeds
    ]

    report = {
        "format": "reconvggt.j1a2a.section36_4.training_health.v1",
        "outputs_root": str(outputs_root),
        "expected_updates": int(args.expected_updates),
        "training_seeds": seeds,
        "seeds": summaries,
        "passed": all(row["passed"] for row in summaries),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    markdown = build_markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8")

    print(markdown, end="")
    print(f"\nJSON: {json_path}")
    print(f"Markdown: {markdown_path}")

    if args.fail_on_health and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
