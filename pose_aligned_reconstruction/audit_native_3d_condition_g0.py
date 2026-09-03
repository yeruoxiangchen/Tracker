#!/usr/bin/env python3
"""G0 audit for native SS/SLAT image lifting and exact Stock controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
from pose_aligned_reconstruction.native_3d_condition import (
    NATIVE_3D_CONDITION_VERSION,
    NativeConditionSLatDataset,
    NativeViewAggregator,
    build_native_slat_components,
    build_native_ss_components,
    canonical_json_sha256,
    dense_native_coords,
    project_native_features,
    sha256_file,
    validate_lifting_feature_metadata,
)
from pose_aligned_reconstruction.train_native_slat_flow import (
    stock_equivalence_audit as slat_stock_equivalence_audit,
)
from pose_aligned_reconstruction.train_native_ss_flow import (
    stock_equivalence_audit as ss_stock_equivalence_audit,
)


G0_REPORT_VERSION = "pose_point_depth_mv.native_3d_condition_g0.v1"


def cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"expected CUDA device, got {device}")
    return 0 if device.index is None else int(device.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--slat_cache_manifest", default="")
    parser.add_argument("--target_decoder_audit", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--indices", default="0")
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--max_sparse_points", type=int, default=2048)
    parser.add_argument("--feature_source", choices=("dino", "vggt", "all"), default="dino")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_roundtrip_error", type=float, default=1.0e-5)
    parser.add_argument("--max_permutation_error", type=float, default=1.0e-5)
    parser.add_argument("--min_control_difference", type=float, default=1.0e-6)
    parser.add_argument("--min_supported_fraction", type=float, default=1.0e-4)
    parser.add_argument("--max_slat_points", type=int, default=40960)
    parser.add_argument("--check_models", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def permute_views(sample: dict[str, Any], permutation: torch.Tensor) -> dict[str, Any]:
    result = dict(sample)
    for key in (
        "visual_patch_features",
        "predicted_depth",
        "depth_confidence",
        "masks",
        "intrinsics",
        "extrinsics",
    ):
        result[key] = sample[key].index_select(0, permutation.cpu())
    result.pop("correct_geometry", None)
    return result


def sparse32_coords(sample: dict[str, Any], max_points: int) -> torch.Tensor:
    prior = sample.get("prior_coords")
    if torch.is_tensor(prior) and prior.ndim == 2 and int(prior.shape[1]) in (3, 4):
        xyz64 = prior[:, -3:].to(torch.int64)
        xyz32 = torch.div(xyz64, 2, rounding_mode="floor").clamp(0, 31)
        xyz32 = torch.unique(xyz32, dim=0, sorted=True)
    else:
        xyz32 = dense_native_coords(32, with_batch=False)
    if len(xyz32) > int(max_points):
        positions = torch.linspace(0, len(xyz32) - 1, int(max_points)).round().long()
        xyz32 = xyz32.index_select(0, positions)
    return torch.cat(
        (torch.zeros((len(xyz32), 1), dtype=torch.int64), xyz32), dim=1
    ).to(torch.int32)


def roundtrip_error(
    evidence: dict[str, Any], sample: dict[str, Any], *, device: torch.device
) -> float:
    camera = evidence["camera"].float()
    camera_h = torch.cat(
        (
            camera,
            torch.ones((*camera.shape[:-1], 1), device=device),
        ),
        dim=-1,
    )
    extrinsics = sample["extrinsics"].to(device=device, dtype=torch.float32)
    c2w = (
        extrinsics
        if str(sample["extrinsics_type"]) == "c2w"
        else torch.linalg.inv(extrinsics)
    )
    recovered = torch.einsum("vij,vnj->vni", c2w, camera_h)[..., :3]
    error = torch.linalg.vector_norm(recovered - evidence["points"][None], dim=-1)
    return float(error.max().item()) if error.numel() else float("inf")


@torch.no_grad()
def audit_projection_sample(
    sample: dict[str, Any],
    *,
    device: torch.device,
    feature_source: str,
    hidden_dim: int,
    max_sparse_points: int,
) -> dict[str, Any]:
    coords16 = dense_native_coords(16, with_batch=True)
    coords32 = sparse32_coords(sample, max_sparse_points)
    correct16 = project_native_features(
        sample,
        coords16,
        resolution=16,
        device=device,
        feature_source=feature_source,
    )
    corrupt16 = project_native_features(
        sample,
        coords16,
        resolution=16,
        device=device,
        feature_source=feature_source,
        mode="pose_cyclic1",
    )
    correct32 = project_native_features(
        sample,
        coords32,
        resolution=32,
        device=device,
        feature_source=feature_source,
    )
    feature_channels = int(correct16["projected_visual"].shape[-1])
    torch.manual_seed(1701)
    aggregator = NativeViewAggregator(
        visual_channels=feature_channels, hidden_dim=int(hidden_dim)
    ).to(device).eval()
    aggregate16, _ = aggregator(
        correct16["projected_visual"],
        correct16["per_view_geometry"],
        correct16["base_weight"],
    )
    views = int(correct16["view_count"])
    permutation = torch.arange(views - 1, -1, -1, dtype=torch.long)
    permuted_sample = permute_views(sample, permutation)
    permuted16 = project_native_features(
        permuted_sample,
        coords16,
        resolution=16,
        device=device,
        feature_source=feature_source,
    )
    permuted_aggregate, _ = aggregator(
        permuted16["projected_visual"],
        permuted16["per_view_geometry"],
        permuted16["base_weight"],
    )
    supported16 = correct16["base_weight"].sum(dim=0) > 0
    supported32 = correct32["base_weight"].sum(dim=0) > 0
    visual_difference = (
        correct16["projected_visual"] - corrupt16["projected_visual"]
    ).abs()
    return {
        "uid": str(sample["uid"]),
        "view_count": views,
        "feature_channels": feature_channels,
        "roundtrip_max_error_16": roundtrip_error(correct16, sample, device=device),
        "roundtrip_max_error_32": roundtrip_error(correct32, sample, device=device),
        "permutation_max_abs": float(
            (aggregate16 - permuted_aggregate).abs().max().item()
        ),
        "correct_vs_pose_corrupt_visual_max_abs": float(visual_difference.max().item()),
        "correct_vs_pose_corrupt_visual_mean_abs": float(visual_difference.mean().item()),
        "supported_fraction_16": float(supported16.float().mean().item()),
        "supported_fraction_32": float(supported32.float().mean().item()),
        "sparse32_point_count": int(coords32.shape[0]),
        "coords32_preserved": bool(
            torch.equal(correct32["coords"].cpu(), coords32.to(torch.int32))
        ),
        "finite": bool(
            torch.isfinite(correct16["projected_visual"]).all().item()
            and torch.isfinite(correct32["projected_visual"]).all().item()
            and torch.isfinite(aggregate16).all().item()
        ),
    }


def main() -> None:
    args = parse_args()
    if int(args.max_samples) <= 0 or int(args.max_sparse_points) <= 0:
        raise ValueError("max_samples/max_sparse_points must be positive")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(cuda_device_index(device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = PoseLiftingCacheDataset(
        args.lifting_cache_manifest, indices=args.indices
    )
    feature_identity = validate_lifting_feature_metadata(
        visual_feature_dim=dataset.visual_feature_dim,
        feature_metadata=dataset.feature_metadata,
        feature_source=str(args.feature_source),
    )
    rows = []
    for index in range(min(len(dataset), int(args.max_samples))):
        row = audit_projection_sample(
            dataset[index],
            device=device,
            feature_source=str(args.feature_source),
            hidden_dim=int(args.hidden_dim),
            max_sparse_points=int(args.max_sparse_points),
        )
        row["checks"] = {
            "roundtrip_16": row["roundtrip_max_error_16"]
            <= float(args.max_roundtrip_error),
            "roundtrip_32": row["roundtrip_max_error_32"]
            <= float(args.max_roundtrip_error),
            "view_permutation": row["permutation_max_abs"]
            <= float(args.max_permutation_error),
            "pose_control_changes_evidence": row[
                "correct_vs_pose_corrupt_visual_max_abs"
            ]
            >= float(args.min_control_difference),
            "support_16": row["supported_fraction_16"]
            >= float(args.min_supported_fraction),
            "support_32": row["supported_fraction_32"]
            >= float(args.min_supported_fraction),
            "coords32_preserved": row["coords32_preserved"],
            "finite": row["finite"],
        }
        row["passed"] = all(row["checks"].values())
        rows.append(row)
        print(f"[native_g0:projection] {json.dumps(row)}", flush=True)

    model_audits: dict[str, Any] = {"checked": False}
    if args.check_models:
        if device.type != "cuda":
            raise ValueError("--check_models requires a CUDA device")
        ss_sampler, ss_model, _, ss_summary, _ = build_native_ss_components(
            pretrained=args.pretrained,
            hidden_dim=int(args.hidden_dim),
            feature_source=str(args.feature_source),
            gradient_checkpointing=False,
            need_decoder=False,
            device=device,
        )
        ss_model.eval()
        ss_audit = ss_stock_equivalence_audit(
            model=ss_model,
            sampler=ss_sampler,
            sample=dataset[0],
            device=device,
            fresh=True,
        )
        ss_model.cpu()
        del ss_model, ss_sampler
        torch.cuda.empty_cache()
        if not args.slat_cache_manifest:
            raise ValueError("--check_models requires --slat_cache_manifest")
        slat_dataset = NativeConditionSLatDataset(
            args.slat_cache_manifest,
            args.lifting_cache_manifest,
            indices="0",
        )
        slat_sampler, slat_model, _, normalization, slat_summary = (
            build_native_slat_components(
                pretrained=args.pretrained,
                hidden_dim=int(args.hidden_dim),
                feature_source=str(args.feature_source),
                gradient_checkpointing=False,
                device=device,
            )
        )
        mean = torch.tensor(normalization["mean"], device=device)[None]
        std = torch.tensor(normalization["std"], device=device)[None]
        slat_model.eval()
        slat_audit = slat_stock_equivalence_audit(
            model=slat_model,
            sampler=slat_sampler,
            sample=slat_dataset[0],
            mean=mean,
            std=std,
            device=device,
            max_points=int(args.max_slat_points),
            fresh=True,
        )
        model_audits = {
            "checked": True,
            "ss": ss_audit,
            "slat": slat_audit,
            "ss_block_count": ss_summary["block_count"],
            "slat_block_count": slat_summary["block_count"],
        }
        slat_model.cpu()
        del slat_model, slat_sampler
        torch.cuda.empty_cache()

    thresholds = {
        "max_roundtrip_error": float(args.max_roundtrip_error),
        "max_permutation_error": float(args.max_permutation_error),
        "min_control_difference": float(args.min_control_difference),
        "min_supported_fraction": float(args.min_supported_fraction),
    }
    report = {
        "format": G0_REPORT_VERSION,
        "condition_version": NATIVE_3D_CONDITION_VERSION,
        "lifting_cache_manifest": str(Path(args.lifting_cache_manifest).resolve()),
        "lifting_cache_manifest_sha256": sha256_file(args.lifting_cache_manifest),
        "slat_cache_manifest": (
            str(Path(args.slat_cache_manifest).resolve())
            if args.slat_cache_manifest
            else ""
        ),
        "slat_cache_manifest_sha256": (
            sha256_file(args.slat_cache_manifest) if args.slat_cache_manifest else ""
        ),
        "feature_source": str(args.feature_source),
        "feature_identity": feature_identity,
        "thresholds": thresholds,
        "rows": rows,
        "model_audits": model_audits,
    }
    report["passed"] = bool(
        rows
        and all(row["passed"] for row in rows)
        and (
            not args.check_models
            or (
                model_audits.get("checked") is True
                and model_audits["ss"].get("passed") is True
                and model_audits["slat"].get("passed") is True
            )
        )
    )
    report["report_sha256"] = canonical_json_sha256(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "output": str(output_dir)}), flush=True)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
