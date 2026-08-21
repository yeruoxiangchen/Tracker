#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reconvggt_ar_adapter_a.pointpose_patch_features import (
    PROJECTED_PATCH_EVIDENCE_COUNT,
    PROJECTED_PATCH_FEATURE_NAMES,
    make_null_projected_patch_features,
)
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import (
    PointPoseCacheDataset,
    encode_frozen_features,
    rgba_images,
)
from reconvggt_ar_adapter_a.train_stock_preserving_pointpose_bridge import (
    HardNegativeMiner,
    architecture_audit,
    build_bridge_condition_inputs,
    build_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit J1a.2-B1 K/T projected view-patch PointPose correspondence."
    )
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physical_hidden_dim", type=int, default=128)
    parser.add_argument("--content_fusion_dim", type=int, default=128)
    parser.add_argument("--fusion_stages", default="0,1")
    args = parser.parse_args()
    args.bridge_fusion_mode = "pose_guided_patch"
    args.bridge_last_blocks = 1
    args.physical_heads = 8
    args.local_fusion_hidden_dim = 128
    args.content_fusion_heads = 4
    args.gradient_checkpointing = False

    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    dataset = PointPoseCacheDataset(args.cache_manifest, indices="all")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index {args.index} outside dataset size {len(dataset)}")
    negative_miner = HardNegativeMiner(dataset)
    sample = dataset[int(args.index)]
    negative = dataset[negative_miner[int(args.index)]]
    pipeline, model, model_summary = build_models(args, device)
    model.eval()

    images = rgba_images(sample["image_paths"], sample["mask_paths"], pipeline)
    aggregated, _ = encode_frozen_features(pipeline, images)
    correct, shuffled, projection = build_bridge_condition_inputs(
        model, sample, negative, aggregated, device
    )
    null = make_null_projected_patch_features(correct)
    evidence_difference = correct[..., :PROJECTED_PATCH_EVIDENCE_COUNT] - shuffled[
        ..., :PROJECTED_PATCH_EVIDENCE_COUNT
    ]
    geometry_correct = correct[..., PROJECTED_PATCH_EVIDENCE_COUNT:]
    geometry_shuffled = shuffled[..., PROJECTED_PATCH_EVIDENCE_COUNT:]
    expected_length = int(aggregated[0].shape[1]) * (int(aggregated[0].shape[2]) - 5)
    projection_checks = {
        "feature_shape_matches_visual_context": bool(
            correct.shape
            == (
                1,
                expected_length,
                len(PROJECTED_PATCH_FEATURE_NAMES),
            )
        ),
        "correct_shuffled_shapes_match": tuple(correct.shape) == tuple(shuffled.shape),
        "correct_shuffled_geometry_exact": bool(
            torch.equal(geometry_correct, geometry_shuffled)
        ),
        "correct_shuffled_evidence_differs": bool(
            evidence_difference.abs().amax().item() > 0.0
        ),
        "null_evidence_zero": bool(
            torch.equal(
                null[..., :PROJECTED_PATCH_EVIDENCE_COUNT],
                torch.zeros_like(null[..., :PROJECTED_PATCH_EVIDENCE_COUNT]),
            )
        ),
        "null_pose_ray_uv_exact": bool(
            torch.equal(
                null[..., PROJECTED_PATCH_EVIDENCE_COUNT:], geometry_correct
            )
        ),
        "all_views_have_projected_points": bool(
            all(
                int(value) > 0
                for value in projection["correct"][
                    "projected_point_count_per_view"
                ]
            )
        ),
        "all_views_have_occupied_patches": bool(
            all(
                int(value) > 0
                for value in projection["correct"][
                    "occupied_patch_count_per_view"
                ]
            )
        ),
    }
    architecture = architecture_audit(
        pipeline,
        model,
        sample,
        device,
        negative_sample=negative,
        raise_on_failure=False,
    )
    passed = bool(all(projection_checks.values()) and architecture["passed"])
    report = {
        "format": "reconvggt.j1a2b1.projected_patch_audit.v1",
        "args": vars(args),
        "correct_uid": sample["uid"],
        "shuffled_uid": negative["uid"],
        "model": model_summary,
        "projection": projection,
        "projection_checks": projection_checks,
        "architecture_audit": architecture,
        "passed": passed,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# J1a.2-B1 Projected Patch Audit",
        "",
        f"- decision: `{'PASS' if passed else 'FAIL'}`",
        f"- correct uid: `{sample['uid']}`",
        f"- shuffled uid: `{negative['uid']}`",
        f"- feature shape: `{list(correct.shape)}`",
        "",
        "## Projection checks",
        "",
    ]
    lines.extend(
        f"- {key}: `{'PASS' if value else 'FAIL'}`"
        for key, value in projection_checks.items()
    )
    lines.extend(
        (
            "",
            "## Stock/null checks",
            "",
            f"- architecture audit: `{'PASS' if architecture['passed'] else 'FAIL'}`",
            f"- hard stock exact: `{architecture['hard_stock_route_exact']}`",
            f"- null condition exact: `{architecture['null_present_condition_exact']}`",
            f"- null token max abs: `{architecture['null_centered_token_max_abs']}`",
        )
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
