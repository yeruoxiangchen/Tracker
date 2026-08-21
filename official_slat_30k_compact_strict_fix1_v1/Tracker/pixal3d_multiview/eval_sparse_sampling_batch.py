from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d import models as pixal3d_models  # noqa: E402
from pixal3d.pipelines.samplers import FlowEulerSampler  # noqa: E402
from pixal3d_multiview.eval_fixed_train_loss import load_checkpoint_weights  # noqa: E402
from pixal3d_multiview.sample_sparse_checkpoint import (  # noqa: E402
    load_target_coords,
    make_preview,
    sparse_overlap_metrics,
    write_ply,
)
from pixal3d_multiview.sparse_condition import SparseMultiviewConditionBuilder  # noqa: E402
from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    MultiviewSparseManifestDataset,
    POSE_MODES,
    apply_pose_mode,
    build_geometry_adapter,
    build_image_cond_model,
    build_view_aggregator,
    load_sparse_flow_model,
    make_multiview_condition,
)


def parse_indices(spec: str, total: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            out.extend(range(int(start_s), int(end_s) + 1))
        else:
            out.append(int(part))
    if not out:
        out = [0]
    bad = [idx for idx in out if idx < 0 or idx >= total]
    if bad:
        raise IndexError(f"indices out of range for dataset size={total}: {bad}")
    return out


@torch.no_grad()
def sample_coords(
    denoiser,
    decoder,
    cond: dict,
    *,
    seed: int,
    steps: int,
    rescale_t: float,
    device: torch.device,
) -> np.ndarray:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    noise = torch.randn(
        1,
        denoiser.in_channels,
        denoiser.resolution,
        denoiser.resolution,
        denoiser.resolution,
        device=device,
        generator=generator,
    )
    sampler = FlowEulerSampler(sigma_min=1e-5)
    z_s = sampler.sample(
        denoiser,
        noise,
        cond=cond,
        steps=int(steps),
        rescale_t=float(rescale_t),
        verbose=False,
    ).samples
    decoded = decoder(z_s) > 0
    return torch.argwhere(decoded)[:, [0, 2, 3, 4]].int().detach().cpu().numpy()


def summarize(rows: list[dict], model_name: str) -> dict:
    selected = [row for row in rows if row["model"] == model_name]
    keys = ["iou", "target_recall", "pred_precision", "pred_unique", "target_unique"]
    out = {"count": len(selected)}
    for key in keys:
        values = np.asarray([row[key] for row in selected], dtype=np.float64)
        if values.size:
            out[key] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "model",
        "sample_index",
        "uid",
        "pred_unique",
        "target_unique",
        "intersection",
        "iou",
        "target_recall",
        "pred_precision",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch sparse sampling comparison for base vs pixal3d_multiview checkpoint.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="0,1,5,10,20,30,50,80,100")
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument("--sparse_flow_model", default="TencentARC/Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16")
    parser.add_argument("--sparse_decoder", default="microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16")
    parser.add_argument("--image_cond_model", default="/home/zjr/Tracker/models/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rescale_t", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
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
    parser.add_argument(
        "--view_aggregator_geom_mode",
        choices=["full", "no_xyz", "uv_depth_only", "support_only"],
        default="full",
    )
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MultiviewSparseManifestDataset(
        args.manifest,
        max_frames=args.max_frames,
        apply_mask=not args.no_apply_mask,
    )
    indices = parse_indices(args.indices, len(dataset))

    denoiser = load_sparse_flow_model(args.model_path, args.sparse_flow_model, device).eval()
    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    decoder = pixal3d_models.from_pretrained(args.sparse_decoder).to(device).eval()

    condition_builder = SparseMultiviewConditionBuilder(device=device, low_vram=False)
    rows: list[dict] = []

    def run_model(model_name: str) -> None:
        for sample_index in tqdm(indices, desc=f"Sampling {model_name}", unit="sample", dynamic_ncols=True):
            batch = dataset[sample_index]
            cond_batch = apply_pose_mode(batch, args.pose_mode, args.seed + sample_index * 104729)
            cond = make_multiview_condition(condition_builder, image_cond_model, cond_batch, args, device)
            coords = sample_coords(
                denoiser,
                decoder,
                cond,
                seed=args.seed + sample_index * 1009,
                steps=args.steps,
                rescale_t=args.rescale_t,
                device=device,
            )
            target_coords = load_target_coords(batch["latent_path"])
            metrics = sparse_overlap_metrics(coords, target_coords)
            sample_dir = output_dir / f"{model_name}_idx{sample_index:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sample_dir / "sparse_sample.npz", pred_coords=coords, target_coords=target_coords)
            write_ply(sample_dir / "pred_sparse_coords.ply", coords, resolution=64, color=(240, 220, 80))
            write_ply(sample_dir / "target_sparse_coords.ply", target_coords, resolution=64, color=(80, 200, 255))
            make_preview(coords, target_coords, resolution=64, path=sample_dir / "sparse_preview.png")
            row = {
                "model": model_name,
                "sample_index": int(sample_index),
                "uid": batch["uid"],
                "output_dir": str(sample_dir),
                "pose_mode": args.pose_mode,
                "pose_permutation": cond_batch.get("pose_permutation"),
                **metrics,
            }
            rows.append(row)
            (sample_dir / "summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    run_model("base")
    view_aggregator = build_view_aggregator(args, image_cond_model, device)
    if view_aggregator is not None:
        view_aggregator.eval()
    geometry_adapter = build_geometry_adapter(args, image_cond_model, device)
    if geometry_adapter is not None:
        geometry_adapter.eval()
    checkpoint_info = load_checkpoint_weights(denoiser, args.checkpoint, view_aggregator, geometry_adapter)
    denoiser.eval()
    condition_builder.last_multiview_stats = {}
    condition_builder.view_aggregator = view_aggregator
    condition_builder.geometry_adapter = geometry_adapter
    run_model("checkpoint")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ablation_name": args.ablation_name,
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "checkpoint_info": checkpoint_info,
        "indices": indices,
        "steps": args.steps,
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
        "view_aggregator_geom_mode": args.view_aggregator_geom_mode,
        "geometry_adapter": args.geometry_adapter,
        "base": summarize(rows, "base"),
        "checkpoint": summarize(rows, "checkpoint"),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "metrics.csv", rows)
    if args.quiet:
        compact = {
            "output_dir": str(output_dir),
            "pose_mode": args.pose_mode,
            "base": result["base"],
            "checkpoint": result["checkpoint"],
        }
        print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
