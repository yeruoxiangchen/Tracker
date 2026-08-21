#!/usr/bin/env python3
"""Summarize condition-only versus condition+LoRA on official GT support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pose_point_depth_mv.eval_direct_slat_flow import summarize
from pose_point_depth_mv.evaluate_proobjaverse_official_slat_gt_support import (
    REPORT_FORMAT as EVAL_FORMAT,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import canonical_sha256


REPORT_FORMAT = "pose_point_depth_mv.proobjaverse_official_slat_arm_decision.v1"


def load_report(path: str | Path, arm: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("format") != EVAL_FORMAT or value.get("passed") is not True:
        raise RuntimeError(f"invalid official GT-support report: {path}")
    if value.get("run_config", {}).get("arm") != arm:
        raise RuntimeError(f"expected {arm} report: {path}")
    return value


def positive_gate(summary: dict[str, Any]) -> bool:
    chamfer = summary["chamfer_l1_improvement"]
    return bool(
        float(chamfer["mean"]) > 0.0
        and float(chamfer["median"]) > 0.0
        and float(chamfer["positive_rate"]) >= 0.625
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition_only_report", required=True)
    parser.add_argument("--condition_lora_report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected_split",
        choices=("train", "dev"),
        default="train",
        help="train is the first-stage fitting gate; dev is only a later gate",
    )
    parser.add_argument("--expected_objects", type=int, default=16)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    args = parser.parse_args()
    condition = load_report(args.condition_only_report, "condition_only")
    lora = load_report(args.condition_lora_report, "condition_lora")
    condition_split = str(condition["run_config"].get("official_split", ""))
    lora_split = str(lora["run_config"].get("official_split", ""))
    if condition_split != str(args.expected_split) or lora_split != condition_split:
        raise RuntimeError(
            "official arm split mismatch: "
            f"condition={condition_split!r} lora={lora_split!r} "
            f"expected={args.expected_split!r}"
        )
    left = {row["object_uid"]: row for row in condition["object_rows"]}
    right = {row["object_uid"]: row for row in lora["object_rows"]}
    if set(left) != set(right) or len(left) != int(args.expected_objects):
        raise RuntimeError(
            "official arms require exactly the same frozen objects: "
            f"condition={len(left)} lora={len(right)} "
            f"expected={args.expected_objects}"
        )
    condition_protocol = str(
        condition["run_config"].get("official_protocol_sha256", "")
    )
    lora_protocol = str(lora["run_config"].get("official_protocol_sha256", ""))
    if not condition_protocol or lora_protocol != condition_protocol:
        raise RuntimeError("official arm protocol binding mismatch")
    metrics = (
        "chamfer_l1_improvement",
        "fscore_0p02_delta",
        "normal_consistency_delta",
        "largest_component_ratio_delta",
    )
    incremental_rows = [
        {
            "object_uid": uid,
            **{name: float(right[uid][name]) - float(left[uid][name]) for name in metrics},
        }
        for uid in sorted(left)
    ]
    incremental = {
        name: summarize(
            [row[name] for row in incremental_rows],
            bootstrap_samples=int(args.bootstrap_samples),
            seed=20260829 + index,
        )
        for index, name in enumerate(metrics)
    }
    condition_gate = positive_gate(condition["summary"])
    lora_gate = positive_gate(lora["summary"])
    lora_increment_gate = positive_gate(incremental)
    if condition_gate and lora_increment_gate:
        diagnosis = "posed-DINO condition works; small SLat attention LoRA adds value"
        selected_arm = "condition_lora"
    elif condition_gate:
        diagnosis = "posed-DINO condition works; SLat attention LoRA is not justified"
        selected_arm = "condition_only"
    elif lora_gate:
        diagnosis = "condition-only capacity is insufficient; joint small LoRA is needed"
        selected_arm = "condition_lora"
    else:
        diagnosis = (
            "neither arm fits decoded official targets better than Stock; do not blame "
            "Native-SS predicted support and do not start rollout training"
        )
        selected_arm = None
    report = {
        "format": REPORT_FORMAT,
        "passed": True,
        "formal": False,
        "object_count": len(left),
        "official_protocol_sha256": condition_protocol,
        "evaluation_split": condition_split,
        "training_overlap": condition_split == "train",
        "condition_only_stock_gate": condition_gate,
        "condition_lora_stock_gate": lora_gate,
        "condition_lora_increment_gate": lora_increment_gate,
        "incremental_condition_lora_minus_condition_only": incremental,
        "incremental_object_rows": incremental_rows,
        "diagnosis": diagnosis,
        "selected_arm": selected_arm,
        "fit_advantage_established": selected_arm is not None,
        "proceed_to_scale_training": (
            selected_arm is not None and condition_split == "train"
        ),
        "proceed_to_predicted_support_bridge": (
            selected_arm is not None and condition_split == "dev"
        ),
        "scope_guard": (
            "official-target Train16 training-overlap fitting diagnosis only; a pass "
            "permits scaling the selected arm but does not establish generalization "
            "and does not permit consuming the predicted-support bridge"
            if condition_split == "train"
            else "official-target development generalization diagnosis only; the "
            "predicted-support bridge remains disjoint until this later gate passes"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if selected_arm is not None else 3)


if __name__ == "__main__":
    main()
