from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    MultiviewSparseManifestDataset,
    POSE_MODES,
    apply_pose_mode,
    build_image_cond_model,
    build_geometry_adapter,
    build_view_aggregator,
    collate_single,
    diffuse,
    load_sparse_flow_model,
    make_multiview_condition,
    velocity_target,
)
from pixal3d_multiview.sparse_condition import SparseMultiviewConditionBuilder  # noqa: E402


def load_checkpoint_weights(
    model: torch.nn.Module,
    checkpoint: str,
    view_aggregator: torch.nn.Module | None = None,
    geometry_adapter: torch.nn.Module | None = None,
) -> dict:
    state = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    view_missing = view_unexpected = []
    if view_aggregator is not None:
        if "view_aggregator" not in state:
            raise ValueError(
                f"Checkpoint has no view_aggregator weights: {checkpoint}. "
                "Use --view_aggregator none or pass the matching aggregator checkpoint."
            )
        view_missing, view_unexpected = view_aggregator.load_state_dict(state["view_aggregator"], strict=False)
    geom_missing = geom_unexpected = []
    if geometry_adapter is not None:
        if "geometry_adapter" not in state:
            raise ValueError(
                f"Checkpoint has no geometry_adapter weights: {checkpoint}. "
                "Use --geometry_adapter none or pass the matching adapter checkpoint."
            )
        geom_missing, geom_unexpected = geometry_adapter.load_state_dict(state["geometry_adapter"], strict=False)
    return {
        "checkpoint": checkpoint,
        "checkpoint_step": int(state.get("step", -1)),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "view_aggregator_missing_keys": len(view_missing),
        "view_aggregator_unexpected_keys": len(view_unexpected),
        "has_view_aggregator": "view_aggregator" in state,
        "geometry_adapter_missing_keys": len(geom_missing),
        "geometry_adapter_unexpected_keys": len(geom_unexpected),
        "has_geometry_adapter": "geometry_adapter" in state,
    }


def fixed_t_value(index: int, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.fixed_t >= 0:
        value = float(args.fixed_t)
    else:
        rng = np.random.default_rng(args.seed + index * 1009)
        value = float(rng.uniform(args.t_min, args.t_max))
    return torch.tensor([value], dtype=torch.float32, device=device)


def evaluate(model, image_cond_model, condition_builder, loader, args, device: torch.device) -> dict:
    model.eval()
    losses = []
    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.max_samples:
                break

            x_0 = batch["x_0"].unsqueeze(0).to(device=device, dtype=torch.float32)
            expected = (model.in_channels, model.resolution, model.resolution, model.resolution)
            if tuple(x_0.shape[1:]) != expected:
                raise ValueError(f"bad z shape for {batch['uid']}: got {tuple(x_0.shape[1:])}, expected {expected}")

            cond_batch = apply_pose_mode(batch, args.pose_mode, args.seed + index * 104729)
            cond = make_multiview_condition(condition_builder, image_cond_model, cond_batch, args, device)
            gen = torch.Generator(device=device)
            gen.manual_seed(args.seed + index * 9176 + 17)
            noise = torch.randn(x_0.shape, device=device, dtype=x_0.dtype, generator=gen)
            t = fixed_t_value(index, args, device)
            x_t = diffuse(x_0, t, noise, args.sigma_min)
            target = velocity_target(x_0, noise, args.sigma_min)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp_dtype == "bf16"):
                pred = model(x_t, t * 1000.0, cond)
                loss = F.mse_loss(pred.float(), target.float())
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            rows.append(
                {
                    "index": index,
                    "uid": batch["uid"],
                    "loss": loss_value,
                    "t": float(t.detach().cpu().item()),
                    "latent_path": batch["latent_path"],
                    "pose_mode": args.pose_mode,
                    "pose_permutation": cond_batch.get("pose_permutation"),
                }
            )

    arr = np.asarray(losses, dtype=np.float64)
    return {
        "num_samples": int(arr.size),
        "loss_mean": float(arr.mean()) if arr.size else None,
        "loss_median": float(np.median(arr)) if arr.size else None,
        "loss_min": float(arr.min()) if arr.size else None,
        "loss_max": float(arr.max()) if arr.size else None,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed train loss for pixal3d_multiview sparse checkpoints.")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint_only", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument("--sparse_flow_model", default="TencentARC/Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16")
    parser.add_argument("--image_cond_model", default="/home/zjr/Tracker/models/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fixed_t", type=float, default=0.5)
    parser.add_argument("--t_min", type=float, default=0.05)
    parser.add_argument("--t_max", type=float, default=0.95)
    parser.add_argument("--sigma_min", type=float, default=1e-5)
    parser.add_argument("--amp_dtype", choices=["none", "bf16"], default="bf16")
    parser.add_argument("--ss_grid_resolution", type=int, default=16)
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--pose_mode", choices=list(POSE_MODES), default="correct")
    parser.add_argument("--no_apply_mask", action="store_true")
    parser.add_argument("--no_auto_volume", action="store_true")
    parser.add_argument("--vh_min_visible_views", type=int, default=1)
    parser.add_argument("--vh_min_support_views", type=int, default=2)
    parser.add_argument("--vh_min_support_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_resolution", type=int, default=48)
    parser.add_argument("--vh_volume_initial_extent_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_padding", type=float, default=1.25)
    parser.add_argument("--vh_volume_min_extent", type=float, default=0.05)
    parser.add_argument("--vh_volume_refine_steps", type=int, default=2)
    parser.add_argument("--no_visibility_depth", action="store_true")
    parser.add_argument("--vh_visibility_resolution", type=int, default=48)
    parser.add_argument("--vh_visibility_dilation", type=int, default=3)
    parser.add_argument("--visibility_depth_tolerance", type=float, default=0.0)
    parser.add_argument("--visibility_depth_tolerance_ratio", type=float, default=0.15)
    parser.add_argument("--visibility_weight_min", type=float, default=0.05)
    parser.add_argument("--empty_policy", choices=["zero", "visible", "border", "soft"], default="zero")
    parser.add_argument("--fallback_weight", type=float, default=1.0)
    parser.add_argument("--support_confidence_power", type=float, default=1.0)
    parser.add_argument("--global_fusion", choices=["concat", "mean", "first"], default="concat")
    parser.add_argument("--geometry_feature_mode", choices=["none", "add", "replace"], default="none")
    parser.add_argument("--geometry_feature_scale", type=float, default=1.0)
    parser.add_argument("--view_aggregator", choices=["none", "gated"], default="none")
    parser.add_argument("--view_aggregator_geom_dim", type=int, default=11)
    parser.add_argument("--view_aggregator_reduced_dim", type=int, default=128)
    parser.add_argument("--view_aggregator_hidden_dim", type=int, default=256)
    parser.add_argument("--view_aggregator_dropout", type=float, default=0.0)
    parser.add_argument("--view_aggregator_residual_scale", type=float, default=1.0)
    parser.add_argument("--geometry_adapter", choices=["none", "mlp"], default="none")
    parser.add_argument("--geometry_adapter_dim", type=int, default=0)
    parser.add_argument("--geometry_adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--geometry_adapter_dropout", type=float, default=0.0)
    parser.add_argument("--geometry_adapter_residual_scale", type=float, default=1.0)
    parser.add_argument("--ablation_name", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = MultiviewSparseManifestDataset(
        args.train_manifest,
        max_frames=args.max_frames,
        apply_mask=not args.no_apply_mask,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_single)

    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ablation_name": args.ablation_name,
        "train_manifest": args.train_manifest,
        "max_samples": args.max_samples,
        "max_frames": args.max_frames,
        "fixed_t": args.fixed_t,
        "seed": args.seed,
        "pose_mode": args.pose_mode,
        "no_apply_mask": args.no_apply_mask,
        "no_auto_volume": args.no_auto_volume,
        "no_visibility_depth": args.no_visibility_depth,
        "empty_policy": args.empty_policy,
        "fallback_weight": args.fallback_weight,
        "support_confidence_power": args.support_confidence_power,
        "global_fusion": args.global_fusion,
        "geometry_feature_mode": args.geometry_feature_mode,
        "geometry_feature_scale": args.geometry_feature_scale,
        "view_aggregator": args.view_aggregator,
        "geometry_adapter": args.geometry_adapter,
    }
    if args.checkpoint:
        model = load_sparse_flow_model(args.model_path, args.sparse_flow_model, device)
        view_aggregator = build_view_aggregator(args, image_cond_model, device)
        if view_aggregator is not None:
            view_aggregator.eval()
        geometry_adapter = build_geometry_adapter(args, image_cond_model, device)
        if geometry_adapter is not None:
            geometry_adapter.eval()
        checkpoint_info = load_checkpoint_weights(model, args.checkpoint, view_aggregator, geometry_adapter)
        condition_builder = SparseMultiviewConditionBuilder(device=device, low_vram=False)
        condition_builder.view_aggregator = view_aggregator
        condition_builder.geometry_adapter = geometry_adapter
        result["checkpoint_info"] = checkpoint_info
        result["checkpoint"] = evaluate(model, image_cond_model, condition_builder, loader, args, device)
        if not args.checkpoint_only:
            model = load_sparse_flow_model(args.model_path, args.sparse_flow_model, device)
            base_condition_builder = SparseMultiviewConditionBuilder(device=device, low_vram=False)
            result["base"] = evaluate(model, image_cond_model, base_condition_builder, loader, args, device)
    else:
        model = load_sparse_flow_model(args.model_path, args.sparse_flow_model, device)
        condition_builder = SparseMultiviewConditionBuilder(device=device, low_vram=False)
        result["base"] = evaluate(model, image_cond_model, condition_builder, loader, args, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.quiet:
        compact = {
            "output": str(output_path),
            "pose_mode": args.pose_mode,
            "checkpoint_loss_mean": result.get("checkpoint", {}).get("loss_mean"),
            "base_loss_mean": result.get("base", {}).get("loss_mean"),
        }
        print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
