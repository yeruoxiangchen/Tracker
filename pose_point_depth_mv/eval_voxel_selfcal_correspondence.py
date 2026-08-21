#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_point_depth_mv.correspondence_head import (
    CORRESPONDENCE_HEAD_VERSION,
    CORRESPONDENCE_CHECKPOINT_VERSION,
    DEFAULT_HELDOUT_CONTROLS,
    CONTINUOUS_SOFT_WEIGHT_VERSION,
    HARD_ADMITTED_SOFT_WEIGHT_VERSION,
    VOXEL_SELFCAL_VERSION,
    ViewCorrespondenceHead,
    correspondence_architecture_hash,
    continuous_voxel_gate_weight,
    correct_voxel_reliability_weight,
    hard_admitted_soft_weight,
    load_correspondence_head_state,
    voxel_self_calibration,
)
from pose_point_depth_mv.eval_local_target_probe import object_balanced, summarize
from pose_point_depth_mv.eval_correspondence_head import permute_evidence_views
from pose_point_depth_mv.view_identity_lifting import (
    SPATIAL_VIEW_MISALIGNED_CONTROL,
    SPATIAL_TOLERANCE_DEFINITION,
    SPATIAL_TOLERANCE_MODES,
    SPATIAL_TOLERANCE_VERSION,
    VIEW_IDENTITY_CONTROL_NAMES,
    VIEW_IDENTITY_EVIDENCE_VERSION,
    apply_symmetric_spatial_tolerance,
    build_view_identity_evidence,
    spatially_misalign_view_evidence,
    view_identity_schema_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate voxel-level self-calibration before C1 local gating."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="16-63")
    parser.add_argument(
        "--split_name",
        choices=("train16", "fresh48", "holdout"),
        required=True,
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_voxel_positive_ratio", type=float, default=0.60)
    parser.add_argument("--min_per_object_positive_ratio", type=float, default=0.50)
    parser.add_argument("--min_object_local_pass_rate", type=float, default=0.65)
    parser.add_argument("--min_heldout_gate_positive_ratio", type=float, default=0.65)
    parser.add_argument("--min_spatial_control_object_win_rate", type=float, default=0.65)
    parser.add_argument("--min_spatial_control_gate_positive_ratio", type=float, default=0.65)
    parser.add_argument("--min_spatial_std", type=float, default=1.0e-4)
    parser.add_argument("--max_permutation_diff", type=float, default=1.0e-5)
    parser.add_argument("--save_maps", action="store_true")
    parser.add_argument(
        "--spatial_tolerance",
        choices=("checkpoint", *SPATIAL_TOLERANCE_MODES),
        default="checkpoint",
        help="Use the checkpoint training protocol unless an ablation overrides it.",
    )
    parser.add_argument(
        "--allow_spatial_tolerance_mismatch",
        action="store_true",
        help="Required for an intentional exact-vs-neighborhood ablation.",
    )
    parser.add_argument("--soft_gate_temperature", type=float, default=0.25)
    parser.add_argument("--soft_gate_reliability_power", type=float, default=1.0)
    parser.add_argument("--continuous_gate_max_scale", type=float, default=0.10)
    parser.add_argument("--fail_on_decision", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "p25": float(torch.quantile(values, 0.25).item()),
        "p75": float(torch.quantile(values, 0.75).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def depth_semantic_maps(
    signed_normalized_residual: torch.Tensor,
    fixed_weight: torch.Tensor,
    active_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Create mutually exclusive audit labels without changing model input.

    Residuals are normalized so 0.25 equals one fitted depth tolerance. A voxel
    is a boundary/ambiguous voxel when no semantic receives 60% weighted view
    agreement. Otherwise the dominant class is surface, free space, or occluded.
    """

    signed = signed_normalized_residual.detach().float()
    weight = fixed_weight.detach().float().clamp_min(0.0)
    active = active_mask.detach().bool()
    if signed.shape != weight.shape or active.shape != (int(weight.shape[1]),):
        raise ValueError("depth semantic audit shape mismatch")
    mass = weight.sum(dim=0).clamp_min(1.0e-6)
    surface = ((signed.abs() <= 0.25).float() * weight).sum(dim=0) / mass
    free_space = ((signed > 0.25).float() * weight).sum(dim=0) / mass
    occluded = ((signed < -0.25).float() * weight).sum(dim=0) / mass
    fractions = torch.stack((surface, free_space, occluded), dim=0)
    dominant_fraction, dominant_index = fractions.max(dim=0)
    # 0=inactive, 1=surface, 2=free-space, 3=occluded, 4=boundary/ambiguous.
    labels = dominant_index.to(torch.int8) + 1
    labels = torch.where(
        dominant_fraction >= 0.60,
        labels,
        torch.full_like(labels, 4),
    )
    labels = labels.masked_fill(~active, 0)
    return {
        "labels": labels,
        "surface_fraction": surface.masked_fill(~active, 0.0),
        "free_space_fraction": free_space.masked_fill(~active, 0.0),
        "occluded_fraction": occluded.masked_fill(~active, 0.0),
        "dominant_fraction": dominant_fraction.masked_fill(~active, 0.0),
    }


def gate_topology(mask: torch.Tensor) -> dict[str, float | int]:
    """Report topology only; it is not a C0.1 decision gate yet."""

    if mask.shape != (16, 16, 16):
        raise ValueError(f"expected 16^3 gate mask, got {tuple(mask.shape)}")
    values = mask.detach().cpu().bool()
    total = int(values.sum().item())
    if total == 0:
        return {
            "component_count": 0,
            "largest_component_fraction": 0.0,
            "boundary_fraction": 0.0,
        }
    visited = torch.zeros_like(values)
    component_sizes: list[int] = []
    for seed in torch.nonzero(values, as_tuple=False).tolist():
        seed_tuple = tuple(int(value) for value in seed)
        if bool(visited[seed_tuple].item()):
            continue
        stack = [seed_tuple]
        visited[seed_tuple] = True
        size = 0
        while stack:
            x, y, z = stack.pop()
            size += 1
            for dx, dy, dz in (
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ):
                nx, ny, nz = x + dx, y + dy, z + dz
                if (
                    0 <= nx < 16
                    and 0 <= ny < 16
                    and 0 <= nz < 16
                    and bool(values[nx, ny, nz].item())
                    and not bool(visited[nx, ny, nz].item())
                ):
                    visited[nx, ny, nz] = True
                    stack.append((nx, ny, nz))
        component_sizes.append(size)
    boundary = 0
    for x, y, z in torch.nonzero(values, as_tuple=False).tolist():
        for dx, dy, dz in (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ):
            nx, ny, nz = x + dx, y + dy, z + dz
            if (
                nx < 0
                or nx >= 16
                or ny < 0
                or ny >= 16
                or nz < 0
                or nz >= 16
                or not bool(values[nx, ny, nz].item())
            ):
                boundary += 1
                break
    return {
        "component_count": len(component_sizes),
        "largest_component_fraction": float(max(component_sizes) / total),
        "boundary_fraction": float(boundary / total),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# C0.1 Voxel-level Self-calibrated Correspondence",
        "",
        f"- Split: `{report['split_name']}`",
        f"- Decision: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Checkpoint step: `{report['checkpoint_step']}`",
        f"- Objects: `{report['object_count']}`",
        f"- Threshold: `{report['threshold']}`",
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
            "## Primary Voxel Metrics",
            "",
            "```json",
            json.dumps(report["primary"], indent=2),
            "```",
            "",
            "## Held-out Local Controls",
            "",
            "```json",
            json.dumps(report["heldout_controls"], indent=2),
            "```",
            "",
            "## Spatial Misalignment Control",
            "",
            "```json",
            json.dumps(report["spatial_control"], indent=2),
            "```",
            "",
            "## Hardest Training Control Distribution",
            "",
            "```json",
            json.dumps(report["hardest_training_control"], indent=2),
            "```",
            "",
            "## View-count Groups",
            "",
            "```json",
            json.dumps(report["view_groups"], indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    maps_dir = output_dir / "voxel_maps"
    if args.save_maps:
        maps_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha256 = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != CORRESPONDENCE_CHECKPOINT_VERSION:
        raise ValueError("unexpected C0 correspondence checkpoint format")
    saved_args = checkpoint.get("args", {})
    model_summary = checkpoint.get("model_summary", {})
    training_spatial_tolerance = str(
        model_summary.get("protocol", {}).get(
            "training_spatial_tolerance",
            saved_args.get("spatial_tolerance", "exact"),
        )
    )
    if training_spatial_tolerance not in SPATIAL_TOLERANCE_MODES:
        raise RuntimeError(
            "checkpoint has invalid training spatial tolerance: "
            f"{training_spatial_tolerance!r}"
        )
    spatial_tolerance = (
        training_spatial_tolerance
        if args.spatial_tolerance == "checkpoint"
        else str(args.spatial_tolerance)
    )
    tolerance_matches_training = spatial_tolerance == training_spatial_tolerance
    if not tolerance_matches_training and not args.allow_spatial_tolerance_mismatch:
        raise RuntimeError(
            "evaluation spatial tolerance differs from checkpoint training "
            "protocol; pass --allow_spatial_tolerance_mismatch only for an "
            "explicit ablation"
        )
    if float(args.soft_gate_temperature) <= 0.0:
        raise ValueError("soft gate temperature must be positive")
    if float(args.soft_gate_reliability_power) < 0.0:
        raise ValueError("soft gate reliability power must be non-negative")
    if not 0.0 < float(args.continuous_gate_max_scale) <= 1.0:
        raise ValueError("continuous gate max scale must be in (0,1]")
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    if str(model_summary.get("cache_config_hash")) != dataset.config_hash:
        raise RuntimeError("C0.1 checkpoint/cache hash mismatch")
    train_objects = set(str(value) for value in model_summary.get("train_object_uids", ()))
    eval_objects = {str(row.get("object_uid", row["uid"])) for row in dataset.rows}
    overlap = sorted(train_objects & eval_objects)
    if args.split_name in {"fresh48", "holdout"} and overlap:
        raise RuntimeError(
            f"{args.split_name} C0 evaluation leaks train objects: {overlap}"
        )
    if args.split_name == "train16" and eval_objects != train_objects:
        raise RuntimeError("train16 object set differs from C0.1 checkpoint")

    head = ViewCorrespondenceHead(
        visual_channels=dataset.visual_feature_dim,
        hidden_dim=int(saved_args["hidden_dim"]),
        pair_hidden_dim=int(saved_args["pair_hidden_dim"]),
        min_views=int(saved_args["min_views"]),
    ).to(device).eval()
    load_correspondence_head_state(head, checkpoint["model_trainable_state"])
    head_metadata = head.metadata()
    saved_head_metadata = model_summary.get("head")
    if saved_head_metadata != head_metadata:
        raise RuntimeError("C0.1 checkpoint head metadata differs from runtime head")
    train_controls = tuple(model_summary["protocol"]["train_controls"])
    heldout_controls = tuple(
        mode for mode in DEFAULT_HELDOUT_CONTROLS if mode not in train_controls
    )
    if not heldout_controls:
        raise RuntimeError("C0.1 requires controls not seen during training")
    amp_name = str(saved_args.get("amp_dtype", "bf16"))
    use_amp = amp_name != "none"
    amp_dtype = torch.float16 if amp_name == "fp16" else torch.bfloat16
    count = len(dataset) if args.max_samples <= 0 else min(
        len(dataset), int(args.max_samples)
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    permutation_max_abs_diff = 0.0
    hard_admitted_inactive_max_abs = 0.0
    continuous_inactive_max_abs = 0.0
    hard_admitted_finite_bounded = True
    continuous_finite_bounded = True

    for index in range(count):
        sample = dataset[index]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        try:
            correct_evidence = build_view_identity_evidence(
                sample, device=device, mode="correct"
            )
            all_evidence = {
                mode: build_view_identity_evidence(sample, device=device, mode=mode)
                for mode in VIEW_IDENTITY_CONTROL_NAMES
            }
            spatial_evidence = spatially_misalign_view_evidence(correct_evidence)
            exact_fixed_weight = correct_evidence["view_weight"].float()
            if spatial_tolerance != "exact":
                correct_evidence, fixed_weight = apply_symmetric_spatial_tolerance(
                    correct_evidence,
                    fixed_correct_weight=exact_fixed_weight,
                    mode=spatial_tolerance,
                )
                all_evidence = {
                    mode: apply_symmetric_spatial_tolerance(
                        evidence,
                        fixed_correct_weight=exact_fixed_weight,
                        mode=spatial_tolerance,
                    )[0]
                    for mode, evidence in all_evidence.items()
                }
                spatial_evidence = apply_symmetric_spatial_tolerance(
                    spatial_evidence,
                    fixed_correct_weight=exact_fixed_weight,
                    mode=spatial_tolerance,
                )[0]
            else:
                fixed_weight = exact_fixed_weight
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                correct = head(correct_evidence, view_weight_override=fixed_weight)
                all_results = {
                    mode: head(evidence, view_weight_override=fixed_weight)
                    for mode, evidence in all_evidence.items()
                }
                spatial_result = head(
                    spatial_evidence, view_weight_override=fixed_weight
                )
                permutation = torch.arange(
                    int(correct_evidence["views"]) - 1, -1, -1, device=device
                )
                permuted_evidence = permute_evidence_views(
                    correct_evidence, permutation
                )
                permuted = head(
                    permuted_evidence,
                    view_weight_override=fixed_weight.index_select(0, permutation),
                )
            permutation_max_abs_diff = max(
                permutation_max_abs_diff,
                float(
                    (
                        correct["voxel_score"].float()
                        - permuted["voxel_score"].float()
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
            calibration = voxel_self_calibration(
                correct,
                {mode: all_results[mode] for mode in train_controls},
                threshold=float(args.threshold),
            )
            active = calibration["active_mask"]
            gate = calibration["gate_mask"]
            if bool(gate[~active].any().item()):
                raise RuntimeError(f"uid={uid} gate escaped fixed active support")
            inactive_hard = calibration["hard_margin"][~active]
            if inactive_hard.numel() and float(inactive_hard.abs().max().item()) != 0.0:
                raise RuntimeError(f"uid={uid} inactive hard margin is not zero")
            if bool(calibration["hard_control_index"][~active].ne(-1).any().item()):
                raise RuntimeError(f"uid={uid} inactive hard-control index is not -1")
            hard_values = calibration["hard_margin"][active]
            mean_values = calibration["mean_margin"][active]
            if hard_values.numel() == 0:
                raise RuntimeError(f"uid={uid} has no active self-calibration voxels")
            hard_summary = tensor_summary(hard_values)
            mean_summary = tensor_summary(mean_values)
            hard_mean_abs = hard_values.abs().mean().clamp_min(1.0e-8)
            topology = gate_topology(gate.reshape(16, 16, 16))
            reliability = correct_voxel_reliability_weight(
                correct_evidence,
                active,
                min_views=int(saved_args["min_views"]),
                floor=float(saved_args.get("voxel_reliability_floor", 0.10)),
                power=float(saved_args.get("voxel_reliability_power", 1.0)),
            )
            admitted_weight = hard_admitted_soft_weight(
                calibration["hard_margin"],
                reliability["raw_weight"],
                active,
                temperature=float(args.soft_gate_temperature),
                reliability_power=float(args.soft_gate_reliability_power),
            )
            continuous_weight = continuous_voxel_gate_weight(
                calibration["hard_margin"],
                reliability["raw_weight"],
                active,
                temperature=float(args.soft_gate_temperature),
                reliability_power=float(args.soft_gate_reliability_power),
                max_scale=float(args.continuous_gate_max_scale),
            )
            admitted_selected = admitted_weight[active]
            continuous_selected = continuous_weight[active]
            inactive_admitted = admitted_weight[~active]
            inactive_continuous = continuous_weight[~active]
            hard_admitted_inactive_max_abs = max(
                hard_admitted_inactive_max_abs,
                (
                    float(inactive_admitted.abs().max().item())
                    if inactive_admitted.numel()
                    else 0.0
                ),
            )
            continuous_inactive_max_abs = max(
                continuous_inactive_max_abs,
                (
                    float(inactive_continuous.abs().max().item())
                    if inactive_continuous.numel()
                    else 0.0
                ),
            )
            hard_admitted_finite_bounded = hard_admitted_finite_bounded and bool(
                torch.isfinite(admitted_weight).all().item()
                and (admitted_weight >= 0.0).all().item()
                and (admitted_weight <= 1.0).all().item()
            )
            continuous_finite_bounded = continuous_finite_bounded and bool(
                torch.isfinite(continuous_weight).all().item()
                and (continuous_weight >= 0.0).all().item()
                and (continuous_weight <= float(args.continuous_gate_max_scale)).all().item()
            )
            row: dict[str, Any] = {
                "uid": uid,
                "object_uid": object_uid,
                "views": int(correct_evidence["views"]),
                "depth_calibration_median_abs_residual": float(
                    sample["depth_calibration"].get("median_abs_residual") or 0.0
                ),
                "depth_calibration_p90_abs_residual": float(
                    sample["depth_calibration"].get("p90_abs_residual") or 0.0
                ),
                "active_ratio": float(calibration["active_ratio"].item()),
                "gate_ratio": float(calibration["gate_ratio"].item()),
                "gate_fraction_active": float(
                    calibration["gate_fraction_of_active"].item()
                ),
                "hard_margin_mean": float(hard_summary["mean"]),
                "hard_margin_median": float(hard_summary["median"]),
                "hard_margin_std": float(hard_values.std(unbiased=False).item()),
                "hard_margin_iqr": float(
                    hard_summary["p75"] - hard_summary["p25"]
                ),
                "hard_margin_normalized_std": float(
                    hard_values.std(unbiased=False).div(hard_mean_abs).item()
                ),
                "hard_voxel_positive_ratio": float(hard_values.gt(0).float().mean().item()),
                "mean_margin_mean": float(mean_summary["mean"]),
                "sample_hard_margin": float(
                    correct["sample_score"].float().item()
                    - max(
                        all_results[mode]["sample_score"].float().item()
                        for mode in train_controls
                    )
                ),
                "gate_component_count": int(topology["component_count"]),
                "gate_largest_component_fraction": float(
                    topology["largest_component_fraction"]
                ),
                "gate_boundary_fraction": float(topology["boundary_fraction"]),
                "hard_admitted_soft_weight_mean": float(admitted_selected.mean().item()),
                "hard_admitted_soft_weight_median": float(admitted_selected.median().item()),
                "hard_admitted_soft_weight_p90": float(
                    torch.quantile(admitted_selected, 0.90).item()
                ),
                "hard_admitted_soft_weight_nonzero_ratio": float(
                    admitted_selected.gt(0.0).float().mean().item()
                ),
                "continuous_soft_weight_mean": float(continuous_selected.mean().item()),
                "continuous_soft_weight_median": float(continuous_selected.median().item()),
                "continuous_soft_weight_p90": float(
                    torch.quantile(continuous_selected, 0.90).item()
                ),
                "continuous_soft_weight_nonzero_ratio": float(
                    continuous_selected.gt(0.0).float().mean().item()
                ),
            }
            hard_index = calibration["hard_control_index"][active]
            for control_index, mode in enumerate(train_controls):
                row[f"hardest_{mode}_ratio"] = float(
                    hard_index.eq(control_index).float().mean().item()
                )
            control_summaries: dict[str, Any] = {}
            for mode, result in all_results.items():
                margin = (
                    correct["voxel_score"].float()
                    - result["voxel_score"].float()
                )
                active_values = margin[active]
                gated_values = margin[gate]
                summary = tensor_summary(active_values)
                gated_summary = tensor_summary(gated_values)
                row[f"{mode}_margin_mean"] = float(summary["mean"])
                row[f"{mode}_margin_median"] = float(summary["median"])
                row[f"{mode}_voxel_positive_ratio"] = float(
                    active_values.gt(0).float().mean().item()
                )
                row[f"{mode}_gate_positive_ratio"] = (
                    float(gated_values.gt(0).float().mean().item())
                    if gated_values.numel()
                    else 0.0
                )
                control_summaries[mode] = {
                    "active_margin": summary,
                    "gated_margin": gated_summary,
                    "active_positive_ratio": row[f"{mode}_voxel_positive_ratio"],
                    "gated_positive_ratio": row[f"{mode}_gate_positive_ratio"],
                }
            spatial_margin = (
                correct["voxel_score"].float() - spatial_result["voxel_score"].float()
            )
            spatial_active_values = spatial_margin[active]
            spatial_gated_values = spatial_margin[gate]
            spatial_summary = tensor_summary(spatial_active_values)
            row.update(
                {
                    "spatial_margin_mean": float(spatial_summary["mean"]),
                    "spatial_margin_median": float(spatial_summary["median"]),
                    "spatial_voxel_positive_ratio": float(
                        spatial_active_values.gt(0).float().mean().item()
                    ),
                    "spatial_gate_positive_ratio": (
                        float(spatial_gated_values.gt(0).float().mean().item())
                        if spatial_gated_values.numel()
                        else 0.0
                    ),
                }
            )
            records.append(row)
            if args.save_maps:
                semantic = depth_semantic_maps(
                    correct_evidence["signed_normalized_depth_residual"],
                    fixed_weight,
                    active,
                )
                torch.save(
                    {
                        "format": VOXEL_SELFCAL_VERSION,
                        "uid": uid,
                        "object_uid": object_uid,
                        "checkpoint_step": int(checkpoint["step"]),
                        "cache_config_hash": dataset.config_hash,
                        "training_controls": list(train_controls),
                        "heldout_controls": list(heldout_controls),
                        "threshold": float(args.threshold),
                        "spatial_tolerance": spatial_tolerance,
                        "spatial_tolerance_version": (
                            None
                            if spatial_tolerance == "exact"
                            else SPATIAL_TOLERANCE_VERSION
                        ),
                        "views": int(correct_evidence["views"]),
                        "depth_calibration_median_abs_residual": row[
                            "depth_calibration_median_abs_residual"
                        ],
                        "depth_calibration_p90_abs_residual": row[
                            "depth_calibration_p90_abs_residual"
                        ],
                        "hardest_control_index": calibration["hard_control_index"]
                        .reshape(16, 16, 16)
                        .cpu(),
                        "active_mask": active.reshape(16, 16, 16).cpu(),
                        "gate_mask": gate.reshape(16, 16, 16).cpu(),
                        "correct_score": correct["voxel_score"].float()
                        .reshape(16, 16, 16)
                        .cpu(),
                        "hard_margin": calibration["hard_margin"]
                        .reshape(16, 16, 16)
                        .cpu(),
                        "raw_hard_margin": calibration["raw_hard_margin"]
                        .reshape(16, 16, 16)
                        .cpu(),
                        "mean_margin": calibration["mean_margin"]
                        .reshape(16, 16, 16)
                        .cpu(),
                        "hard_admitted_soft_weight": admitted_weight
                        .reshape(16, 16, 16)
                        .cpu(),
                        "continuous_soft_weight": continuous_weight
                        .reshape(16, 16, 16)
                        .cpu(),
                        "hard_admitted_soft_weight_protocol": {
                            "version": HARD_ADMITTED_SOFT_WEIGHT_VERSION,
                            "temperature": float(args.soft_gate_temperature),
                            "reliability_power": float(
                                args.soft_gate_reliability_power
                            ),
                            "uses_raw_correct_reliability": True,
                            "inactive_zero": True,
                        },
                        "continuous_soft_weight_protocol": {
                            "version": CONTINUOUS_SOFT_WEIGHT_VERSION,
                            "temperature": float(args.soft_gate_temperature),
                            "reliability_power": float(args.soft_gate_reliability_power),
                            "max_scale": float(args.continuous_gate_max_scale),
                            "inactive_zero": True,
                            "margin_sign_truncated": False,
                            "c1_ablation_only": True,
                        },
                        "audit_maps": {
                            "reliability_weight": reliability["weight"]
                            .reshape(16, 16, 16)
                            .cpu(),
                            "raw_reliability": reliability["raw_weight"]
                            .reshape(16, 16, 16)
                            .cpu(),
                            "view_support_fraction": reliability[
                                "view_support_fraction"
                            ]
                            .reshape(16, 16, 16)
                            .cpu(),
                            "pair_support_quality": reliability[
                                "pair_support_quality"
                            ]
                            .reshape(16, 16, 16)
                            .cpu(),
                            "depth_reliability": reliability["depth_reliability"]
                            .reshape(16, 16, 16)
                            .cpu(),
                            "depth_confidence": reliability[
                                "depth_confidence_mean"
                            ]
                            .reshape(16, 16, 16)
                            .cpu(),
                            "depth_consistency": reliability[
                                "depth_consistency_mean"
                            ]
                            .reshape(16, 16, 16)
                            .cpu(),
                            "visibility_agreement": reliability[
                                "visibility_agreement"
                            ]
                            .reshape(16, 16, 16)
                            .cpu(),
                            "depth_semantic_label": semantic["labels"]
                            .reshape(16, 16, 16)
                            .cpu(),
                            "surface_fraction": semantic["surface_fraction"]
                            .reshape(16, 16, 16)
                            .cpu(),
                            "free_space_fraction": semantic["free_space_fraction"]
                            .reshape(16, 16, 16)
                            .cpu(),
                            "occluded_fraction": semantic["occluded_fraction"]
                            .reshape(16, 16, 16)
                            .cpu(),
                        },
                        "training_control_margins": {
                            mode: (
                                correct["voxel_score"].float()
                                - all_results[mode]["voxel_score"].float()
                            )
                            .masked_fill(~active, 0.0)
                            .reshape(16, 16, 16)
                            .cpu()
                            for mode in train_controls
                        },
                        "heldout_margins": {
                            mode: (
                                correct["voxel_score"].float()
                                - all_results[mode]["voxel_score"].float()
                            )
                            .reshape(16, 16, 16)
                            .cpu()
                            for mode in heldout_controls
                        },
                        "spatial_view_misaligned_margin": spatial_margin
                        .masked_fill(~active, 0.0)
                        .reshape(16, 16, 16)
                        .cpu(),
                        "spatial_misalignment_shifts": spatial_evidence[
                            "spatial_misalignment_shifts"
                        ],
                        "summary": control_summaries,
                    },
                    maps_dir / f"{uid}.pt",
                )
            print(f"[voxel_selfcal] {index + 1}/{count} uid={uid}", flush=True)
        except Exception as error:  # noqa: BLE001 - preserve audit failures
            failures.append({"uid": uid, "error": repr(error)})
            print(f"[voxel_selfcal] FAIL uid={uid}: {error}", flush=True)

    one_sequence_per_object = len({row["object_uid"] for row in records}) == len(
        records
    )
    hard_mean = object_balanced(
        records, "hard_margin_mean", bootstrap_samples=int(args.bootstrap_samples)
    )
    hard_median = object_balanced(
        records, "hard_margin_median", bootstrap_samples=int(args.bootstrap_samples)
    )
    positive_ratio = object_balanced(
        records,
        "hard_voxel_positive_ratio",
        bootstrap_samples=int(args.bootstrap_samples),
    )
    spatial_std = object_balanced(
        records, "hard_margin_std", bootstrap_samples=int(args.bootstrap_samples)
    )
    local_pass_rate = mean(
        float(row["hard_voxel_positive_ratio"])
        >= float(args.min_per_object_positive_ratio)
        for row in records
    ) if records else 0.0
    heldout_report: dict[str, Any] = {}
    heldout_checks: dict[str, bool] = {}
    for mode in heldout_controls:
        margin = object_balanced(
            records,
            f"{mode}_margin_mean",
            bootstrap_samples=int(args.bootstrap_samples),
        )
        voxel_ratio = object_balanced(
            records,
            f"{mode}_voxel_positive_ratio",
            bootstrap_samples=int(args.bootstrap_samples),
        )
        gate_ratio = object_balanced(
            records,
            f"{mode}_gate_positive_ratio",
            bootstrap_samples=int(args.bootstrap_samples),
        )
        heldout_report[mode] = {
            "margin": margin,
            "voxel_positive_ratio": voxel_ratio,
            "gate_positive_ratio": gate_ratio,
        }
        heldout_checks[f"{mode}_margin_mean_positive"] = (
            float(margin["object"]["mean"]) > 0.0
        )
        heldout_checks[f"{mode}_margin_median_positive"] = (
            float(margin["object"]["median"]) > 0.0
        )
        heldout_checks[f"{mode}_margin_ci_positive"] = (
            float(margin["object_bootstrap_95_ci"][0]) > 0.0
        )
        heldout_checks[f"{mode}_voxel_positive_ratio"] = (
            float(voxel_ratio["object"]["mean"])
            >= float(args.min_voxel_positive_ratio)
        )
        heldout_checks[f"{mode}_gate_positive_ratio"] = (
            float(gate_ratio["object"]["mean"])
            >= float(args.min_heldout_gate_positive_ratio)
        )

    spatial_margin = object_balanced(
        records,
        "spatial_margin_mean",
        bootstrap_samples=int(args.bootstrap_samples),
    )
    spatial_voxel_ratio = object_balanced(
        records,
        "spatial_voxel_positive_ratio",
        bootstrap_samples=int(args.bootstrap_samples),
    )
    spatial_gate_ratio = object_balanced(
        records,
        "spatial_gate_positive_ratio",
        bootstrap_samples=int(args.bootstrap_samples),
    )
    spatial_control_report = {
        "name": SPATIAL_VIEW_MISALIGNED_CONTROL,
        "definition": (
            "per-view distinct 3d visual/geometry rolls with fixed correct "
            "view_weight"
        ),
        "margin": spatial_margin,
        "voxel_positive_ratio": spatial_voxel_ratio,
        "gate_positive_ratio": spatial_gate_ratio,
    }
    spatial_control_checks = {
        "spatial_control_margin_mean_positive": float(
            spatial_margin["object"]["mean"]
        ) > 0.0,
        "spatial_control_margin_median_positive": float(
            spatial_margin["object"]["median"]
        ) > 0.0,
        "spatial_control_margin_ci_positive": float(
            spatial_margin["object_bootstrap_95_ci"][0]
        ) > 0.0,
        "spatial_control_object_win_rate": float(
            spatial_margin["object_win_rate"]
        ) >= float(args.min_spatial_control_object_win_rate),
        "spatial_control_voxel_positive_ratio": float(
            spatial_voxel_ratio["object"]["mean"]
        ) >= float(args.min_voxel_positive_ratio),
        "spatial_control_gate_positive_ratio": float(
            spatial_gate_ratio["object"]["mean"]
        ) >= float(args.min_spatial_control_gate_positive_ratio),
    }

    view_groups: dict[str, Any] = {}
    for views in sorted({int(row["views"]) for row in records}):
        group = [row for row in records if int(row["views"]) == views]
        view_groups[str(views)] = {
            "object_count": len(group),
            "hard_margin_mean": summarize(
                [float(row["hard_margin_mean"]) for row in group]
            ),
            "voxel_positive_ratio": summarize(
                [float(row["hard_voxel_positive_ratio"]) for row in group]
            ),
            "local_object_pass_rate": mean(
                float(row["hard_voxel_positive_ratio"])
                >= float(args.min_per_object_positive_ratio)
                for row in group
            ),
            "hard_admitted_soft_weight": summarize(
                [float(row["hard_admitted_soft_weight_mean"]) for row in group]
            ),
        }

    checks = {
        "no_sample_failures": not failures and len(records) == count,
        "one_sequence_per_object": one_sequence_per_object,
        "voxel_permutation_invariant": permutation_max_abs_diff
        <= float(args.max_permutation_diff),
        "hard_margin_object_mean_positive": float(hard_mean["object"]["mean"]) > 0.0,
        "hard_margin_object_median_positive": float(hard_median["object"]["median"])
        > 0.0,
        "hard_margin_bootstrap_ci_positive": float(
            hard_mean["object_bootstrap_95_ci"][0]
        )
        > 0.0,
        "voxel_positive_ratio": float(positive_ratio["object"]["mean"])
        >= float(args.min_voxel_positive_ratio),
        "local_object_pass_rate": local_pass_rate
        >= float(args.min_object_local_pass_rate),
        "spatial_margin_nonconstant": float(spatial_std["object"]["mean"])
        >= float(args.min_spatial_std),
        "hard_admitted_soft_weight_finite_bounded": hard_admitted_finite_bounded,
        "hard_admitted_soft_weight_inactive_zero": hard_admitted_inactive_max_abs == 0.0,
        "continuous_soft_weight_finite_bounded": continuous_finite_bounded,
        "continuous_soft_weight_inactive_zero": continuous_inactive_max_abs == 0.0,
        "spatial_control_passes": all(spatial_control_checks.values()),
        "every_heldout_local_control_passes": all(heldout_checks.values()),
        **heldout_checks,
        **spatial_control_checks,
    }
    report = {
        "format": VOXEL_SELFCAL_VERSION,
        "stage": (
            (
                "C0.1 voxel-level self-calibrated correspondence"
                if spatial_tolerance == "exact"
                else "C0.3 neighborhood-aware voxel correspondence"
            )
            if tolerance_matches_training
            else "C0 symmetric spatial-tolerance ablation"
        ),
        "gate_protocol": {
            "scope": "per_voxel",
            "definition": "correct_voxel_score-max_training_control_voxel_score",
            "uses_object_score_as_gate": False,
            "fixed_correct_support": True,
            "threshold_source": "command_line_preregistered",
            "inactive_margin_policy": "masked_to_zero",
            "spatial_tolerance": spatial_tolerance,
            "spatial_tolerance_version": (
                None
                if spatial_tolerance == "exact"
                else SPATIAL_TOLERANCE_VERSION
            ),
            "spatial_tolerance_definition": (
                None
                if spatial_tolerance == "exact"
                else SPATIAL_TOLERANCE_DEFINITION
            ),
            "training_spatial_tolerance": training_spatial_tolerance,
            "evaluation_matches_training": tolerance_matches_training,
            "spatial_tolerance_symmetric_across_branches": True,
            "spatial_tolerance_fixed_correct_support": True,
        },
        "passed": all(checks.values()),
        "split_name": args.split_name,
        "training_seed": int(saved_args["seed"]),
        "args": vars(args),
        "training_spatial_tolerance": training_spatial_tolerance,
        "evaluation_spatial_tolerance": spatial_tolerance,
        "spatial_tolerance_matches_training": tolerance_matches_training,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_format": checkpoint["format"],
        "checkpoint_step": int(checkpoint["step"]),
        "cache_manifest": str(dataset.manifest_path.resolve()),
        "cache_config_hash": dataset.config_hash,
        "protocol_hash": model_summary.get("protocol_hash"),
        "head_metadata": head_metadata,
        "head_version": CORRESPONDENCE_HEAD_VERSION,
        "evidence_version": VIEW_IDENTITY_EVIDENCE_VERSION,
        "evidence_schema_hash": view_identity_schema_hash(),
        "model_architecture_hash": correspondence_architecture_hash(head_metadata),
        "training_controls": list(train_controls),
        "heldout_control_names": list(heldout_controls),
        "threshold": float(args.threshold),
        "object_count": len({row["object_uid"] for row in records}),
        "sample_count": len(records),
        "voxel_permutation_max_abs_diff": permutation_max_abs_diff,
        "primary": {
            "hard_margin_mean": hard_mean,
            "hard_margin_median": hard_median,
            "voxel_positive_ratio": positive_ratio,
            "spatial_std": spatial_std,
            "spatial_iqr": object_balanced(
                records,
                "hard_margin_iqr",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "normalized_spatial_std": object_balanced(
                records,
                "hard_margin_normalized_std",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "local_object_pass_rate": local_pass_rate,
            "gate_fraction_active": object_balanced(
                records,
                "gate_fraction_active",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "gate_component_count": object_balanced(
                records,
                "gate_component_count",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "gate_largest_component_fraction": object_balanced(
                records,
                "gate_largest_component_fraction",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "gate_boundary_fraction": object_balanced(
                records,
                "gate_boundary_fraction",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "hard_admitted_soft_weight": object_balanced(
                records,
                "hard_admitted_soft_weight_mean",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "hard_admitted_soft_weight_nonzero_ratio": object_balanced(
                records,
                "hard_admitted_soft_weight_nonzero_ratio",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "continuous_soft_weight": object_balanced(
                records,
                "continuous_soft_weight_mean",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
            "continuous_soft_weight_nonzero_ratio": object_balanced(
                records,
                "continuous_soft_weight_nonzero_ratio",
                bootstrap_samples=int(args.bootstrap_samples),
            ),
        },
        "hard_admitted_soft_weight_protocol": {
            "version": HARD_ADMITTED_SOFT_WEIGHT_VERSION,
            "definition": (
                "tanh(relu(hard_margin)/temperature) * "
                "raw_correct_reliability**power"
            ),
            "temperature": float(args.soft_gate_temperature),
            "reliability_power": float(args.soft_gate_reliability_power),
            "inactive_zero": True,
            "margin_sign_truncated": True,
            "is_calibrated_probability": False,
            "formal_n3_gate": True,
            "flow_lora_enabled": False,
        },
        "continuous_soft_weight_protocol": {
            "version": CONTINUOUS_SOFT_WEIGHT_VERSION,
            "definition": (
                "max_scale * sigmoid(hard_margin/temperature) * "
                "raw_correct_reliability**power"
            ),
            "temperature": float(args.soft_gate_temperature),
            "reliability_power": float(args.soft_gate_reliability_power),
            "max_scale": float(args.continuous_gate_max_scale),
            "inactive_zero": True,
            "margin_sign_truncated": False,
            "is_calibrated_probability": False,
            "formal_n3_gate": False,
            "c1_ablation_only": True,
            "flow_lora_enabled": False,
        },
        "heldout_controls": heldout_report,
        "spatial_control": spatial_control_report,
        "hardest_training_control": {
            mode: object_balanced(
                records,
                f"hardest_{mode}_ratio",
                bootstrap_samples=int(args.bootstrap_samples),
            )
            for mode in train_controls
        },
        "view_groups": view_groups,
        "checks": checks,
        "failures": failures,
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "report.md")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "hard_margin_mean": hard_mean["object"],
                "voxel_positive_ratio": positive_ratio["object"],
                "local_object_pass_rate": local_pass_rate,
                "voxel_permutation_max_abs_diff": permutation_max_abs_diff,
                "heldout": {
                    mode: {
                        "margin": value["margin"]["object"],
                        "voxel_positive_ratio": value["voxel_positive_ratio"][
                            "object"
                        ],
                        "gate_positive_ratio": value["gate_positive_ratio"][
                            "object"
                        ],
                    }
                    for mode, value in heldout_report.items()
                },
                "spatial_control": {
                    "margin": spatial_control_report["margin"]["object"],
                    "voxel_positive_ratio": spatial_control_report[
                        "voxel_positive_ratio"
                    ]["object"],
                    "gate_positive_ratio": spatial_control_report[
                        "gate_positive_ratio"
                    ]["object"],
                },
            },
            indent=2,
        )
    )
    if args.fail_on_decision and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
