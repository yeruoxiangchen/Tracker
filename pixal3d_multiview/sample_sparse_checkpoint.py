from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d import models as pixal3d_models  # noqa: E402
from pixal3d.pipelines.samplers import FlowEulerSampler  # noqa: E402
from pixal3d_multiview.eval_fixed_train_loss import load_checkpoint_weights  # noqa: E402
from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    MultiviewSparseManifestDataset,
    build_geometry_adapter,
    build_image_cond_model,
    build_view_aggregator,
    collate_single,
    load_sparse_flow_model,
    make_multiview_condition,
)
from pixal3d_multiview.sparse_condition import SparseMultiviewConditionBuilder  # noqa: E402


def write_ply(path: Path, coords: np.ndarray, resolution: int, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if coords.size == 0:
        points = np.zeros((0, 3), dtype=np.float32)
    else:
        xyz = coords[:, -3:].astype(np.float32)
        points = (xyz + 0.5) / float(resolution) - 0.5
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {color[0]} {color[1]} {color[2]}\n")


def draw_projection(coords: np.ndarray, resolution: int, axis_pair: tuple[int, int], size: int = 512) -> Image.Image:
    image = Image.new("RGB", (size, size), (8, 8, 8))
    draw = ImageDraw.Draw(image)
    if coords.size == 0:
        return image
    xyz = coords[:, -3:].astype(np.float32)
    a, b = axis_pair
    pts = xyz[:, [a, b]] / max(resolution - 1, 1)
    pts[:, 1] = 1.0 - pts[:, 1]
    pix = np.clip(np.round(pts * (size - 1)).astype(np.int32), 0, size - 1)
    for x, y in pix:
        draw.rectangle((int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1), fill=(240, 220, 80))
    return image


def make_preview(pred_coords: np.ndarray, target_coords: np.ndarray, resolution: int, path: Path) -> None:
    panels = []
    for coords, title_color in ((pred_coords, (240, 220, 80)), (target_coords, (80, 200, 255))):
        for axes in ((0, 1), (0, 2), (1, 2)):
            panel = draw_projection(coords, resolution, axes)
            if title_color != (240, 220, 80):
                arr = np.asarray(panel).copy()
                mask = arr[..., 0] > 100
                arr[mask] = np.asarray(title_color, dtype=np.uint8)
                panel = Image.fromarray(arr)
            panels.append(panel)
    w, h = panels[0].size
    sheet = Image.new("RGB", (w * 3, h * 2), (20, 20, 20))
    for idx, panel in enumerate(panels):
        sheet.paste(panel, ((idx % 3) * w, (idx // 3) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def load_target_coords(latent_path: str) -> np.ndarray:
    with np.load(latent_path) as data:
        if "target_coords" not in data:
            return np.zeros((0, 3), dtype=np.int32)
        return data["target_coords"].astype(np.int32)


def sparse_overlap_metrics(pred_coords: np.ndarray, target_coords: np.ndarray) -> dict:
    pred_xyz = pred_coords[:, -3:].astype(np.int32) if pred_coords.size else np.zeros((0, 3), dtype=np.int32)
    target_xyz = target_coords[:, -3:].astype(np.int32) if target_coords.size else np.zeros((0, 3), dtype=np.int32)
    pred_set = set(map(tuple, pred_xyz.tolist()))
    target_set = set(map(tuple, target_xyz.tolist()))
    intersection = len(pred_set & target_set)
    union = len(pred_set | target_set)
    metrics = {
        "pred_unique": len(pred_set),
        "target_unique": len(target_set),
        "intersection": intersection,
        "iou": float(intersection / union) if union else 0.0,
        "target_recall": float(intersection / len(target_set)) if target_set else 0.0,
        "pred_precision": float(intersection / len(pred_set)) if pred_set else 0.0,
    }
    if pred_xyz.size:
        metrics["pred_bbox_min"] = pred_xyz.min(axis=0).tolist()
        metrics["pred_bbox_max"] = pred_xyz.max(axis=0).tolist()
    if target_xyz.size:
        metrics["target_bbox_min"] = target_xyz.min(axis=0).tolist()
        metrics["target_bbox_max"] = target_xyz.max(axis=0).tolist()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample sparse coords from a pixal3d_multiview sparse checkpoint.")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument("--sparse_flow_model", default="TencentARC/Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16")
    parser.add_argument("--sparse_decoder", default="microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16")
    parser.add_argument("--image_cond_model", default="/home/zjr/Tracker/models/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rescale_t", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ss_grid_resolution", type=int, default=16)
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
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
        args.train_manifest,
        max_frames=args.max_frames,
        apply_mask=not args.no_apply_mask,
    )
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"--sample_index {args.sample_index} out of range for dataset size {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_single)
    batch = next(item for idx, item in enumerate(loader) if idx == args.sample_index)

    denoiser = load_sparse_flow_model(args.model_path, args.sparse_flow_model, device)
    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    view_aggregator = build_view_aggregator(args, image_cond_model, device)
    if view_aggregator is not None:
        view_aggregator.eval()
    geometry_adapter = build_geometry_adapter(args, image_cond_model, device)
    if geometry_adapter is not None:
        geometry_adapter.eval()
    checkpoint_info = load_checkpoint_weights(denoiser, args.checkpoint, view_aggregator, geometry_adapter)
    denoiser.eval()
    condition_builder = SparseMultiviewConditionBuilder(device=device, low_vram=False)
    condition_builder.view_aggregator = view_aggregator
    condition_builder.geometry_adapter = geometry_adapter

    with torch.no_grad():
        cond = make_multiview_condition(condition_builder, image_cond_model, batch, args, device)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
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
            steps=args.steps,
            rescale_t=args.rescale_t,
            verbose=True,
            tqdm_desc="Sampling sparse latent",
        ).samples

        decoder = pixal3d_models.from_pretrained(args.sparse_decoder).to(device).eval()
        decoded = decoder(z_s) > 0
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int().detach().cpu().numpy()

    target_coords = load_target_coords(batch["latent_path"])
    np.savez_compressed(
        output_dir / "sparse_sample.npz",
        pred_coords=coords,
        target_coords=target_coords,
        uid=np.array(batch["uid"]),
        checkpoint=np.array(args.checkpoint),
    )
    write_ply(output_dir / "pred_sparse_coords.ply", coords, resolution=64, color=(240, 220, 80))
    write_ply(output_dir / "target_sparse_coords.ply", target_coords, resolution=64, color=(80, 200, 255))
    make_preview(coords, target_coords, resolution=64, path=output_dir / "sparse_preview.png")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ablation_name": args.ablation_name,
        "uid": batch["uid"],
        "sample_index": args.sample_index,
        "empty_policy": args.empty_policy,
        "fallback_weight": args.fallback_weight,
        "support_confidence_power": args.support_confidence_power,
        "global_fusion": args.global_fusion,
        "geometry_feature_mode": args.geometry_feature_mode,
        "geometry_feature_scale": args.geometry_feature_scale,
        "view_aggregator": args.view_aggregator,
        "geometry_adapter": args.geometry_adapter,
        "checkpoint_info": checkpoint_info,
        "pred_coords": int(coords.shape[0]),
        "target_coords": int(target_coords.shape[0]),
        "sparse_overlap": sparse_overlap_metrics(coords, target_coords),
        "output_dir": str(output_dir),
        "files": {
            "npz": str(output_dir / "sparse_sample.npz"),
            "pred_ply": str(output_dir / "pred_sparse_coords.ply"),
            "target_ply": str(output_dir / "target_sparse_coords.ply"),
            "preview_png": str(output_dir / "sparse_preview.png"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
