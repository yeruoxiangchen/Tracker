#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


FORMAT = "ar_ss_flow.pair_feature_local_velocity.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a C3 pair-feature SS training run and checkpoint."
    )
    parser.add_argument("--train_report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected_steps", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    args = parse_args()
    report_path = Path(args.train_report)
    checkpoint_path = Path(args.checkpoint)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state = checkpoint.get("model_trainable_state", {})
    optimizer = checkpoint.get("optimizer", {})
    scaler = checkpoint.get("scaler", {})
    summary = report.get("model_summary", {})
    equivalence = summary.get("stock_equivalence", {})
    completed = int(report.get("completed_steps", -1))
    checkpoint_step = int(checkpoint.get("step", -1))
    expected = int(args.expected_steps)

    checks = {
        "report_format": report.get("format") == FORMAT,
        "checkpoint_format": checkpoint.get("format") == FORMAT,
        "expected_steps": completed == expected,
        "checkpoint_step": checkpoint_step == expected,
        "report_finite": bool(report.get("finite", False)),
        "trainable_state_nonempty": bool(state),
        "trainable_state_finite": finite_tree(state),
        "optimizer_state_nonempty": bool(optimizer),
        "optimizer_state_finite": finite_tree(optimizer),
        "scaler_state_finite": finite_tree(scaler),
        "stock_flow_frozen": int(summary.get("stock_flow_trainable_parameters", -1))
        == 0,
        "flow_lora_disabled": summary.get("flow_lora_enabled") is False,
        "correspondence_frozen": summary.get("correspondence_model_frozen") is True,
        "slat_disabled": summary.get("slat_enabled") is False,
        "physical_off_exact_stock": float(
            equivalence.get("physical_off_max_abs_diff", float("inf"))
        )
        == 0.0,
        "null_exact_stock": float(equivalence.get("null_max_abs_diff", float("inf")))
        == 0.0,
        "zero_init_exact_stock": float(
            equivalence.get("zero_init_enabled_max_abs_diff", float("inf"))
        )
        == 0.0,
    }
    passed = all(checks.values())
    output = {
        "stage": "C3 pair-feature SS finite training audit",
        "passed": passed,
        "checks": checks,
        "train_report": str(report_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "expected_steps": expected,
        "completed_steps": completed,
        "checkpoint_step": checkpoint_step,
        "history_rows": len(report.get("history", [])),
        "trainable_tensor_count": len(state),
        "stock_equivalence": equivalence,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if args.fail_on_error and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
