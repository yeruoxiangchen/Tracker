#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def _finite_tree(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a PointPose SS LoRA training run.")
    parser.add_argument("--train_report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--expected_updates", type=int, required=True)
    parser.add_argument("--max_nonfinite_attempts", type=int, default=0)
    args = parser.parse_args()

    report_path = Path(args.train_report)
    checkpoint_path = Path(args.checkpoint)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_trainable_state", {})
    optimizer_state = checkpoint.get("optimizer", {})
    scaler_state = checkpoint.get("scaler", {})
    bad_state_keys = sorted(key for key, value in state.items() if not _finite_tree(value))
    applied = int(report.get("applied_optimizer_updates", -1))
    nonfinite = int(report.get("nonfinite_attempts", -1))
    start_step = int(report.get("start_global_step", 0))
    completed_step = int(report.get("completed_global_step", checkpoint.get("step", -1)))
    checkpoint_step = int(checkpoint.get("step", -1))
    history = report.get("history", [])
    finite_update_rows = sum(
        bool(row.get("update_finite")) and bool(row.get("optimizer_step_applied"))
        for row in history
    )
    skipped_rows = sum(not bool(row.get("optimizer_step_applied")) for row in history)
    history_consistent = all(
        bool(row.get("update_finite")) == bool(row.get("optimizer_step_applied"))
        for row in history
    )
    optimizer_finite = _finite_tree(optimizer_state)
    scaler_finite = _finite_tree(scaler_state)

    checks = {
        "expected_updates": applied == int(args.expected_updates),
        "step_accounting": start_step + applied == completed_step,
        "completed_step_matches_checkpoint": completed_step == checkpoint_step,
        "nonfinite_within_limit": 0 <= nonfinite <= int(args.max_nonfinite_attempts),
        "history_skips_match_nonfinite_attempts": skipped_rows == nonfinite,
        "history_update_flags_consistent": history_consistent,
        "checkpoint_trainable_state_nonempty": bool(state),
        "checkpoint_trainable_state_finite": not bad_state_keys,
        "checkpoint_optimizer_state_nonempty": bool(optimizer_state),
        "checkpoint_optimizer_state_finite": optimizer_finite,
        "checkpoint_scaler_state_finite": scaler_finite,
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "train_report": str(report_path),
        "checkpoint": str(checkpoint_path),
        "applied_optimizer_updates": applied,
        "start_global_step": start_step,
        "completed_global_step": completed_step,
        "checkpoint_step": checkpoint_step,
        "nonfinite_attempts": nonfinite,
        "history_finite_update_rows": finite_update_rows,
        "history_skipped_rows": skipped_rows,
        "trainable_tensor_count": len(state),
        "nonfinite_trainable_tensor_count": len(bad_state_keys),
        "nonfinite_trainable_tensor_keys": bad_state_keys,
        "optimizer_state_finite": optimizer_finite,
        "scaler_state_finite": scaler_finite,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
