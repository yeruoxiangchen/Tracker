from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d_multiview.train_sparse_multiview import (  # noqa: E402
    MultiviewSparseManifestDataset,
    POSE_MODES,
    apply_pose_mode,
    build_image_cond_model,
    make_multiview_condition,
)
from pixal3d_multiview.sparse_condition import SparseMultiviewConditionBuilder  # noqa: E402


PIXAL3D_DEFAULT_CAMERA_ANGLE_X = 0.8575560450553894


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


def image_to_tensor(image_cond_model, image, device: torch.device) -> torch.Tensor:
    image = image.resize((image_cond_model.image_size, image_cond_model.image_size))
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0).to(device)


def tensor_stats(tensor: torch.Tensor) -> dict:
    data = tensor.detach().float()
    finite = torch.isfinite(data)
    finite_data = data[finite]
    if finite_data.numel() == 0:
        return {
            "shape": list(data.shape),
            "finite_ratio": 0.0,
            "mean": None,
            "std": None,
            "abs_mean": None,
            "min": None,
            "max": None,
            "zero_ratio": None,
        }
    return {
        "shape": list(data.shape),
        "finite_ratio": float(finite.float().mean().item()),
        "mean": float(finite_data.mean().item()),
        "std": float(finite_data.std(unbiased=False).item()),
        "abs_mean": float(finite_data.abs().mean().item()),
        "min": float(finite_data.min().item()),
        "max": float(finite_data.max().item()),
        "zero_ratio": float((finite_data == 0).float().mean().item()),
    }


def cosine_stats(left: torch.Tensor, right: torch.Tensor) -> dict:
    if left.shape != right.shape:
        return {"same_shape": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    left_f = left.detach().float().reshape(-1, left.shape[-1])
    right_f = right.detach().float().reshape(-1, right.shape[-1])
    cos = F.cosine_similarity(left_f, right_f, dim=-1)
    return {
        "same_shape": True,
        "mean": float(cos.mean().item()),
        "median": float(cos.median().item()),
        "min": float(cos.min().item()),
        "max": float(cos.max().item()),
    }


def numeric_mean(rows: list[dict], path: list[str]) -> float | None:
    values = []
    for row in rows:
        cur = row
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and isinstance(cur, (int, float)):
            values.append(float(cur))
    return float(np.mean(values)) if values else None


def write_markdown(path: Path, result: dict) -> None:
    summary = result["summary"]
    lines = [
        "# Adapter Condition 分布补充测试",
        "",
        f"测试时间：`{result.get('timestamp_utc', '')}`",
        f"消融名称：`{result.get('ablation_name', '')}`",
        f"manifest：`{result['manifest']}`",
        f"indices：`{result['indices']}`",
        f"pose_mode：`{result['pose_mode']}`",
        f"empty_policy：`{result.get('empty_policy', 'zero')}`",
        f"fallback_weight：`{result.get('fallback_weight', 1.0)}`",
        f"support_confidence_power：`{result.get('support_confidence_power', 1.0)}`",
        f"global_fusion：`{result.get('global_fusion', 'concat')}`",
        f"geometry_feature_mode：`{result.get('geometry_feature_mode', 'none')}`",
        f"geometry_feature_scale：`{result.get('geometry_feature_scale', 1.0)}`",
        "",
        "## 聚合结果",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## 如何阅读",
            "",
            "- `native_first_global_vs_multiview_first_global_cos_mean` 应接近 1；它验证同一张图的 DINO global tokens 没被 adapter 破坏。",
            "- `native_first_proj_vs_multiview_proj_cos_mean` 不要求接近 1；它衡量 Pixal3D 单视图默认投影和当前多视图投影在 sparse grid 上的特征差异。",
            "- `multiview_proj_zero_ratio_mean` 越高，说明投影聚合后空特征越多，adapter 对 sparse flow 越难用。",
            "- `support_mean` 和 `zero_support` 来自 mask + pose 的投影采样统计，用来判断多视图几何是否给 grid 点提供足够图像支持。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Pixal3D native single-view condition with pixal3d_multiview adapter condition.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="0,1,5,10,20,30,50,80")
    parser.add_argument("--image_cond_model", default="/home/zjr/Tracker/models/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--ss_grid_resolution", type=int, default=16)
    parser.add_argument("--camera_angle_x", type=float, default=PIXAL3D_DEFAULT_CAMERA_ANGLE_X)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--mesh_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pose_mode", choices=list(POSE_MODES), default="correct")
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
    parser.add_argument("--ablation_name", default="")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MultiviewSparseManifestDataset(
        args.manifest,
        max_frames=args.max_frames,
        apply_mask=not args.no_apply_mask,
    )
    indices = parse_indices(args.indices, len(dataset))
    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    condition_builder = SparseMultiviewConditionBuilder(device=device, low_vram=False)

    rows = []
    camera_angle_x = torch.tensor([args.camera_angle_x], dtype=torch.float32, device=device)
    distance = torch.tensor([args.distance], dtype=torch.float32, device=device)
    mesh_scale = torch.tensor([args.mesh_scale], dtype=torch.float32, device=device)

    for sample_index in tqdm(indices, desc="Adapter stats", unit="sample", dynamic_ncols=True):
        batch = dataset[sample_index]
        first_image = batch["images"][0]
        first_tensor = image_to_tensor(image_cond_model, first_image, device)
        native_global, native_proj = image_cond_model(
            first_tensor,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )

        cond_batch = apply_pose_mode(batch, args.pose_mode, args.seed + sample_index * 104729)
        condition_builder.last_multiview_stats = {}
        multiview = make_multiview_condition(condition_builder, image_cond_model, cond_batch, args, device)
        multiview_global = multiview["global"]
        multiview_proj = multiview["proj"]
        multiview_first_global = multiview_global[:, : native_global.shape[1], :]

        row = {
            "sample_index": int(sample_index),
            "uid": batch["uid"],
            "pose_mode": args.pose_mode,
            "pose_permutation": cond_batch.get("pose_permutation"),
            "native_global": tensor_stats(native_global),
            "native_proj": tensor_stats(native_proj),
            "multiview_global": tensor_stats(multiview_global),
            "multiview_proj": tensor_stats(multiview_proj),
            "native_first_global_vs_multiview_first_global_cos": cosine_stats(native_global, multiview_first_global),
            "native_first_proj_vs_multiview_proj_cos": cosine_stats(native_proj, multiview_proj),
            "adapter_stats": condition_builder.last_multiview_stats,
        }
        rows.append(row)

    summary = {
        "count": len(rows),
        "native_first_global_vs_multiview_first_global_cos_mean": numeric_mean(
            rows, ["native_first_global_vs_multiview_first_global_cos", "mean"]
        ),
        "native_first_proj_vs_multiview_proj_cos_mean": numeric_mean(
            rows, ["native_first_proj_vs_multiview_proj_cos", "mean"]
        ),
        "native_proj_abs_mean": numeric_mean(rows, ["native_proj", "abs_mean"]),
        "multiview_proj_abs_mean": numeric_mean(rows, ["multiview_proj", "abs_mean"]),
        "native_proj_zero_ratio_mean": numeric_mean(rows, ["native_proj", "zero_ratio"]),
        "multiview_proj_zero_ratio_mean": numeric_mean(rows, ["multiview_proj", "zero_ratio"]),
        "support_mean": numeric_mean(rows, ["adapter_stats", "ss_condition", "lr_projection", "support_mean"]),
        "zero_support_mean": numeric_mean(rows, ["adapter_stats", "ss_condition", "lr_projection", "zero_support"]),
        "raw_zero_support_mean": numeric_mean(rows, ["adapter_stats", "ss_condition", "lr_projection", "raw_zero_support"]),
        "fallback_points_mean": numeric_mean(rows, ["adapter_stats", "ss_condition", "lr_projection", "fallback_points"]),
        "support_confidence_mean": numeric_mean(rows, ["adapter_stats", "ss_condition", "lr_projection", "support_confidence_mean"]),
        "global_token_count_mean": numeric_mean(rows, ["adapter_stats", "ss_condition", "global", "global_token_count"]),
        "visibility_nonzero_ratio_mean": numeric_mean(
            rows,
            [
                "adapter_stats",
                "ss_condition",
                "lr_projection",
                "visibility",
                "visible_weight_nonzero_ratio",
            ],
        ),
    }
    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ablation_name": args.ablation_name,
        "manifest": args.manifest,
        "indices": indices,
        "image_cond_model": args.image_cond_model,
        "pose_mode": args.pose_mode,
        "empty_policy": args.empty_policy,
        "fallback_weight": args.fallback_weight,
        "support_confidence_power": args.support_confidence_power,
        "global_fusion": args.global_fusion,
        "geometry_feature_mode": args.geometry_feature_mode,
        "geometry_feature_scale": args.geometry_feature_scale,
        "camera_angle_x": args.camera_angle_x,
        "distance": args.distance,
        "mesh_scale": args.mesh_scale,
        "summary": summary,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "summary.md", result)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
