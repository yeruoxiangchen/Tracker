#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv.correspondence_head import (
    CONTINUOUS_SOFT_WEIGHT_VERSION,
    HARD_ADMITTED_SOFT_WEIGHT_VERSION,
    VOXEL_SELFCAL_VERSION,
)
from pose_point_depth_mv.summarize_neighborhood_multiseed import (
    NEIGHBORHOOD_MULTISEED_VERSION,
)
from pose_point_depth_mv.summarize_voxel_selfcal_multiseed import (
    protocol_signature,
)


C1_GATE_MANIFEST_VERSION = "pose_point_depth_mv.c1_gate_manifest.v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a passed N3 result and export only bounded neighborhood "
            "gate inputs for a future local C1 supervision stage."
        )
    )
    parser.add_argument("--multiseed_report", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--map_subdir", default="c0_3_train16")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_seed", type=int, required=True)
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def valid_map(payload: dict[str, Any], threshold: float) -> tuple[bool, dict[str, Any]]:
    expected_shape = (16, 16, 16)
    active = payload.get("active_mask")
    gate = payload.get("gate_mask")
    hard = payload.get("hard_margin")
    admitted = payload.get("hard_admitted_soft_weight")
    continuous = payload.get("continuous_soft_weight")
    tensors = (active, gate, hard, admitted, continuous)
    shape_ok = all(
        torch.is_tensor(value) and tuple(value.shape) == expected_shape
        for value in tensors
    )
    if not shape_ok:
        return False, {"shape_ok": False}
    active = active.bool()
    gate = gate.bool()
    hard = hard.float()
    admitted = admitted.float()
    continuous = continuous.float()
    finite = bool(
        torch.isfinite(hard).all().item()
        and torch.isfinite(admitted).all().item()
        and torch.isfinite(continuous).all().item()
    )
    continuous_protocol = payload.get("continuous_soft_weight_protocol", {})
    continuous_max_scale = float(continuous_protocol.get("max_scale", -1.0))
    admitted_bounded = bool(((admitted >= 0.0) & (admitted <= 1.0)).all().item())
    continuous_bounded = bool(
        continuous_max_scale > 0.0
        and ((continuous >= 0.0) & (continuous <= continuous_max_scale)).all().item()
    )
    gate_expected = active & hard.gt(float(threshold))
    inactive_hard = hard[~active]
    inactive_admitted = admitted[~active]
    inactive_continuous = continuous[~active]
    inactive_hard_zero = not inactive_hard.numel() or float(
        inactive_hard.abs().max().item()
    ) == 0.0
    inactive_admitted_zero = not inactive_admitted.numel() or float(
        inactive_admitted.abs().max().item()
    ) == 0.0
    inactive_continuous_zero = not inactive_continuous.numel() or float(
        inactive_continuous.abs().max().item()
    ) == 0.0
    nonpositive_admitted = admitted[hard <= 0.0]
    nonpositive_admitted_zero = not nonpositive_admitted.numel() or float(
        nonpositive_admitted.abs().max().item()
    ) == 0.0
    outside_gate_admitted = admitted[~gate]
    outside_gate_admitted_zero = not outside_gate_admitted.numel() or float(
        outside_gate_admitted.abs().max().item()
    ) == 0.0
    checks = {
        "shape_ok": shape_ok,
        "finite": finite,
        "hard_admitted_bounded": admitted_bounded,
        "continuous_bounded": continuous_bounded,
        "hard_gate_matches_report_threshold": bool(torch.equal(gate, gate_expected)),
        "gate_inside_active": not bool(gate[~active].any().item()),
        "inactive_hard_zero": inactive_hard_zero,
        "inactive_hard_admitted_zero": inactive_admitted_zero,
        "inactive_continuous_zero": inactive_continuous_zero,
        "nonpositive_margin_hard_admitted_zero": nonpositive_admitted_zero,
        "outside_hard_gate_hard_admitted_zero": outside_gate_admitted_zero,
        "hard_admitted_version": payload.get(
            "hard_admitted_soft_weight_protocol", {}
        ).get("version")
        == HARD_ADMITTED_SOFT_WEIGHT_VERSION,
        "continuous_version": continuous_protocol.get("version")
        == CONTINUOUS_SOFT_WEIGHT_VERSION,
    }
    checks["continuous_positive_outside_hard_gate"] = bool(
        continuous[active & ~gate].gt(0.0).any().item()
    )
    # The final property is diagnostic: an object can legitimately have every
    # active voxel admitted, so N4 does not require it per object or globally.
    map_passed = all(
        value
        for name, value in checks.items()
        if name != "continuous_positive_outside_hard_gate"
    )
    return map_passed, checks


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    multiseed_path = Path(args.multiseed_report)
    multiseed = json.loads(multiseed_path.read_text(encoding="utf-8"))
    if multiseed.get("format") != NEIGHBORHOOD_MULTISEED_VERSION:
        raise ValueError("unexpected N3 multi-seed report format")

    run_dir = Path(args.run_dir).resolve()
    report_path = run_dir / args.map_subdir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != VOXEL_SELFCAL_VERSION:
        raise ValueError("unexpected C0.3 report format")
    if int(report.get("training_seed", -1)) != int(args.expected_seed):
        raise ValueError("C0.3 report seed does not match --expected_seed")
    seed_matches = [
        row
        for row in multiseed.get("per_seed", [])
        if int(row.get("seed", -1)) == int(args.expected_seed)
    ]
    unique_seed_match = len(seed_matches) == 1
    matched_seed = seed_matches[0] if unique_seed_match else {}
    n3_run_dir = Path(matched_seed.get("run_dir", "/__missing_n3_run__")).resolve()
    run_matches_n3 = unique_seed_match and n3_run_dir == run_dir
    signature_matches_n3 = (
        protocol_signature(report) == multiseed.get("protocol_signature")
    )
    checkpoint_matches_n3 = (
        str(report.get("checkpoint", "")) == str(matched_seed.get("checkpoint", ""))
    )
    checkpoint_hash_matches_n3 = (
        bool(report.get("checkpoint_sha256"))
        and str(report.get("checkpoint_sha256"))
        == str(matched_seed.get("checkpoint_sha256", ""))
    )
    maps_dir = report_path.parent / "voxel_maps"

    rows: list[dict[str, Any]] = []
    map_failures: list[dict[str, Any]] = []
    continuous_outside_hard_gate = False
    threshold = float(report["threshold"])
    for record in report["records"]:
        uid = str(record["uid"])
        path = maps_dir / f"{uid}.pt"
        payload = torch.load(path, map_location="cpu")
        passed, checks = valid_map(payload, threshold)
        if not passed:
            map_failures.append({"uid": uid, "checks": checks})
            continue
        continuous_outside_hard_gate = (
            continuous_outside_hard_gate
            or bool(checks["continuous_positive_outside_hard_gate"])
        )
        rows.append(
            {
                "uid": uid,
                "object_uid": str(record["object_uid"]),
                "map_file": str(path.resolve()),
                "active_mask_key": "active_mask",
                "hard_gate_key": "gate_mask",
                "formal_c1_weight_key": "hard_admitted_soft_weight",
                "ablation_c1_weight_key": "continuous_soft_weight",
                "views": int(record["views"]),
            }
        )

    checks = {
        "n3_multiseed_passed": multiseed.get("passed") is True,
        "n3_flow_lora_disabled": multiseed.get("flow_lora_enabled") is False,
        "seed_has_unique_n3_entry": unique_seed_match,
        "source_run_is_exact_n3_run": run_matches_n3,
        "source_protocol_signature_matches_n3": signature_matches_n3,
        "source_checkpoint_matches_n3": checkpoint_matches_n3,
        "source_checkpoint_sha256_matches_n3": checkpoint_hash_matches_n3,
        "source_split_is_train16": report.get("split_name") == "train16",
        "source_report_passed": report.get("passed") is True,
        "source_training_is_gaussian3": report.get("training_spatial_tolerance")
        == "gaussian3",
        "source_evaluation_is_gaussian3": report.get(
            "evaluation_spatial_tolerance"
        )
        == "gaussian3",
        "source_protocol_matches_training": report.get(
            "spatial_tolerance_matches_training"
        )
        is True,
        "source_formal_weight_is_hard_admitted": report.get(
            "hard_admitted_soft_weight_protocol", {}
        ).get("formal_n3_gate")
        is True,
        "source_formal_weight_is_not_probability": report.get(
            "hard_admitted_soft_weight_protocol", {}
        ).get("is_calibrated_probability")
        is False,
        "continuous_weight_is_ablation_only": report.get(
            "continuous_soft_weight_protocol", {}
        ).get("c1_ablation_only")
        is True,
        "all_maps_present_and_valid": not map_failures
        and len(rows) == len(report["records"]),
        "one_map_per_object": len({row["object_uid"] for row in rows}) == len(rows),
    }
    passed = all(checks.values())
    manifest = {
        "format": C1_GATE_MANIFEST_VERSION,
        "stage": "N4 C1 neighborhood gate admission and input export",
        "passed": passed,
        "supervision_status": "inputs_only_not_trained",
        "allowed_next_supervision": "local_occupancy_or_decoder_logit",
        "flow_lora_enabled": False,
        "decoder_trainable": False,
        "raw_hard_margin_allowed_as_scale": False,
        "formal_c1_weight_key": "hard_admitted_soft_weight",
        "ablation_c1_weight_key": "continuous_soft_weight",
        "formal_n3_protocol_remains_hard_admitted": True,
        "continuous_positive_outside_hard_gate_observed": continuous_outside_hard_gate,
        "source_multiseed_report": str(multiseed_path.resolve()),
        "source_c0_report": str(report_path.resolve()),
        "source_run_dir": str(run_dir),
        "source_checkpoint": report["checkpoint"],
        "source_checkpoint_sha256": report["checkpoint_sha256"],
        "source_cache_manifest": report["cache_manifest"],
        "training_seed": int(args.expected_seed),
        "sample_count": len(rows),
        "checks": checks,
        "map_failures": map_failures,
        "samples": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "admission.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "checks": checks,
                "sample_count": len(rows),
                "flow_lora_enabled": False,
                "decoder_trainable": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    if args.fail_on_decision and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
