from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d import models as pixal3d_models  # noqa: E402
from pixal3d.pipelines.samplers import FlowEulerSampler  # noqa: E402
from pixal3d_multiview.eval_sparse_sampling_batch import sample_coords  # noqa: E402
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
    build_image_cond_model,
    load_sparse_flow_model,
    make_multiview_condition,
)


NATIVE_DEFAULT_CAMERA_ANGLE_X = 0.8575560450553894
NATIVE_DEFAULT_DISTANCE = 2.0
NATIVE_DEFAULT_MESH_SCALE = 1.0


@dataclass(frozen=True)
class NativeCameraParams:
    camera_angle_x: float
    distance: float
    mesh_scale: float


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


def parse_int_list(spec: str) -> list[int]:
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or [0]


def parse_str_list(spec: str) -> list[str]:
    return [part.strip() for part in str(spec).split(",") if part.strip()]


def native_camera_params(mode: str) -> NativeCameraParams:
    mode = str(mode).strip()
    base = NativeCameraParams(
        camera_angle_x=NATIVE_DEFAULT_CAMERA_ANGLE_X,
        distance=NATIVE_DEFAULT_DISTANCE,
        mesh_scale=NATIVE_DEFAULT_MESH_SCALE,
    )
    if mode == "default":
        return base
    if mode == "fov_narrow":
        return NativeCameraParams(base.camera_angle_x * 0.75, base.distance, base.mesh_scale)
    if mode == "fov_wide":
        return NativeCameraParams(base.camera_angle_x * 1.25, base.distance, base.mesh_scale)
    if mode == "distance_near":
        return NativeCameraParams(base.camera_angle_x, base.distance * 0.75, base.mesh_scale)
    if mode == "distance_far":
        return NativeCameraParams(base.camera_angle_x, base.distance * 1.25, base.mesh_scale)
    if mode == "scale_small":
        return NativeCameraParams(base.camera_angle_x, base.distance, base.mesh_scale * 0.80)
    if mode == "scale_large":
        return NativeCameraParams(base.camera_angle_x, base.distance, base.mesh_scale * 1.20)
    raise ValueError(f"unknown native camera param mode: {mode}")


def crop_by_mask(image: Image.Image, mask: torch.Tensor, padding: float = 1.10, threshold: float = 0.5) -> Image.Image:
    mask_np = mask.detach().cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    ys, xs = np.where(mask_np > float(threshold))
    if xs.size == 0 or ys.size == 0:
        return image
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    size = max(x1 - x0 + 1, y1 - y0 + 1)
    size = max(1, int(round(size * float(padding))))
    left = int(round(cx - size * 0.5))
    top = int(round(cy - size * 0.5))
    right = left + size
    bottom = top + size

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    crop_left = max(left, 0)
    crop_top = max(top, 0)
    crop_right = min(right, image.width)
    crop_bottom = min(bottom, image.height)
    if crop_right <= crop_left or crop_bottom <= crop_top:
        return image
    patch = image.crop((crop_left, crop_top, crop_right, crop_bottom))
    canvas.paste(patch, (crop_left - left, crop_top - top))
    return canvas


def native_image(batch: dict, frame_index: int, policy: str, mask_threshold: float) -> Image.Image:
    images = batch["images"]
    masks = batch["masks"]
    if frame_index < 0 or frame_index >= len(images):
        raise IndexError(f"native frame index {frame_index} out of range for sample {batch['uid']} with {len(images)} frames")
    image = images[frame_index]
    if policy == "full":
        return image
    if policy == "crop_mask":
        return crop_by_mask(image, masks[frame_index], threshold=mask_threshold)
    raise ValueError(f"unknown native image policy: {policy}")


@torch.no_grad()
def make_native_condition(
    image_cond_model,
    image: Image.Image,
    params: NativeCameraParams,
    device: torch.device,
) -> dict:
    camera_angle_x = torch.tensor([params.camera_angle_x], device=device, dtype=torch.float32)
    distance = torch.tensor([params.distance], device=device, dtype=torch.float32)
    mesh_scale = torch.tensor([params.mesh_scale], device=device, dtype=torch.float32)
    z_global, z_proj = image_cond_model(
        [image],
        camera_angle_x=camera_angle_x,
        distance=distance,
        mesh_scale=mesh_scale,
    )
    return {"global": z_global, "proj": z_proj}


def coords_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    xyz = coords[:, -3:].astype(np.int32) if coords.size else np.zeros((0, 3), dtype=np.int32)
    return set(map(tuple, xyz.tolist()))


def coord_set_iou(a: np.ndarray, b: np.ndarray) -> float:
    aset = coords_set(a)
    bset = coords_set(b)
    union = len(aset | bset)
    if union == 0:
        return 0.0
    return float(len(aset & bset) / union)


def summarize(rows: list[dict], group_keys: Iterable[str]) -> list[dict]:
    metric_keys = ("iou", "target_recall", "pred_precision", "pred_unique", "target_unique", "intersection")
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    group_keys = tuple(group_keys)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    summary = []
    for key, selected in sorted(grouped.items()):
        out = {name: value for name, value in zip(group_keys, key)}
        out["count"] = len(selected)
        for metric in metric_keys:
            values = np.asarray([float(row.get(metric, 0.0)) for row in selected], dtype=np.float64)
            out[f"{metric}_mean"] = float(values.mean()) if values.size else None
            out[f"{metric}_median"] = float(np.median(values)) if values.size else None
            out[f"{metric}_min"] = float(values.min()) if values.size else None
            out[f"{metric}_max"] = float(values.max()) if values.size else None
        summary.append(out)
    return summary


def summarize_pairwise(rows: list[dict], group_keys: Iterable[str]) -> list[dict]:
    metric_keys = ("iou_delta", "target_recall_delta", "pred_precision_delta", "pred_unique_delta", "pred_overlap_iou")
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    group_keys = tuple(group_keys)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)
    summary = []
    for key, selected in sorted(grouped.items()):
        out = {name: value for name, value in zip(group_keys, key)}
        out["count"] = len(selected)
        for metric in metric_keys:
            values = np.asarray([float(row.get(metric, 0.0)) for row in selected], dtype=np.float64)
            out[f"{metric}_mean"] = float(values.mean()) if values.size else None
            out[f"{metric}_median"] = float(np.median(values)) if values.size else None
            out[f"{metric}_min"] = float(values.min()) if values.size else None
            out[f"{metric}_max"] = float(values.max()) if values.size else None
        summary.append(out)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def write_report(path: Path, result: dict) -> None:
    lines = [
        "# Native Pixal3D Sparse Flow vs Multiview Base",
        "",
        f"时间：`{result['timestamp_utc']}`",
        f"manifest：`{result['manifest']}`",
        f"indices：`{result['indices']}`",
        f"steps：`{result['steps']}`",
        "",
        "## 这个测试在比较什么",
        "",
        "1. `native_pixal3d`：原版 Pixal3D sparse flow 的 native projection condition。它只使用单张图像和 `camera_angle_x / distance / mesh_scale`，不接 AR 多视角外参。",
        "2. `multiview_base`：同一个 Pixal3D sparse flow 权重，但输入当前 `pixal3d_multiview` 的多视角 adapter condition，不加载任何训练 checkpoint。",
        "3. 二者使用同一个 sparse decoder、同一个 sampling seed、同一个 `target_coords` voxel overlap 指标。",
        "",
        "注意：`native_pixal3d` 不是 AR-pose 模型，因此它没有 `shuffle/reverse/cyclic` 这种多视角 pose mode。这里的 native pose sensitivity 只能看原生 camera 参数扰动，或者换不同 frame 输入。",
        "",
        "## Sparse IoU 汇总",
        "",
        "| family | condition | IoU | recall | precision | pred unique | target unique |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(
            "| {family} | {condition} | {iou_mean:.6f} | {target_recall_mean:.6f} | "
            "{pred_precision_mean:.6f} | {pred_unique_mean:.1f} | {target_unique_mean:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Sensitivity vs baseline",
            "",
            "| family | condition | baseline | IoU delta | recall delta | precision delta | pred overlap IoU |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["pairwise_summary"]:
        lines.append(
            "| {family} | {condition} | {baseline_condition} | {iou_delta_mean:.6f} | "
            "{target_recall_delta_mean:.6f} | {pred_precision_delta_mean:.6f} | {pred_overlap_iou_mean:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 如何阅读",
            "",
            "- 如果 `native_pixal3d/default` 的 IoU 也很低，说明当前 Objaverse AR-like dataset + exact `target_coords` overlap 对原版 single-view sparse condition 也很苛刻。",
            "- 如果 `native_pixal3d` 明显高于 `multiview_base/correct`，说明当前 multiview adapter 是主要瓶颈。",
            "- 如果 `multiview_base/correct` 和 `reverse/cyclic/noise` 接近，说明当前 base adapter 对 AR pose 不敏感。",
            "- `pred_overlap_iou` 衡量同一 sample 下不同 condition 输出 coords 自身有多像。它不是和 target 比，而是看 condition 扰动是否真的改变 sparse 输出。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare native Pixal3D sparse flow with pixal3d_multiview base sparse condition.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="0,1,5,10,20,30,50,80,100")
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument("--sparse_flow_model", default="TencentARC/Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16")
    parser.add_argument("--sparse_decoder", default="microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16")
    parser.add_argument("--image_cond_model", default="/home/zjr/Tracker/models/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--rescale_t", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ss_grid_resolution", type=int, default=16)
    parser.add_argument("--run_native", type=int, default=1)
    parser.add_argument("--run_multiview_base", type=int, default=1)
    parser.add_argument("--native_frame_indices", default="0")
    parser.add_argument("--native_image_policies", default="crop_mask")
    parser.add_argument("--native_param_modes", default="default,fov_narrow,fov_wide,distance_near,distance_far,scale_small,scale_large")
    parser.add_argument("--multiview_pose_modes", default="correct,reverse,cyclic_shift1,cyclic_shift2,noise,large_noise,identity")
    parser.add_argument("--save_previews", action="store_true")
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
    native_frame_indices = parse_int_list(args.native_frame_indices)
    native_image_policies = parse_str_list(args.native_image_policies)
    native_param_modes = parse_str_list(args.native_param_modes)
    multiview_pose_modes = parse_str_list(args.multiview_pose_modes)
    invalid_pose_modes = [mode for mode in multiview_pose_modes if mode not in POSE_MODES]
    if invalid_pose_modes:
        raise ValueError(f"invalid multiview pose modes: {invalid_pose_modes}")

    denoiser = load_sparse_flow_model(args.model_path, args.sparse_flow_model, device).eval()
    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    decoder = pixal3d_models.from_pretrained(args.sparse_decoder).to(device).eval()
    condition_builder = SparseMultiviewConditionBuilder(device=device, low_vram=False)

    rows: list[dict] = []
    coords_by_key: dict[tuple[int, str], np.ndarray] = {}
    metrics_by_key: dict[tuple[int, str], dict] = {}

    def record(
        *,
        sample_index: int,
        batch: dict,
        family: str,
        condition: str,
        coords: np.ndarray,
        target_coords: np.ndarray,
        metadata: dict,
    ) -> None:
        metrics = sparse_overlap_metrics(coords, target_coords)
        row = {
            "family": family,
            "condition": condition,
            "sample_index": int(sample_index),
            "uid": batch["uid"],
            **metadata,
            **metrics,
        }
        rows.append(row)
        coords_by_key[(sample_index, condition)] = coords
        metrics_by_key[(sample_index, condition)] = metrics
        if args.save_previews:
            sample_dir = output_dir / "samples" / f"{condition}_idx{sample_index:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sample_dir / "sparse_sample.npz", pred_coords=coords, target_coords=target_coords)
            write_ply(sample_dir / "pred_sparse_coords.ply", coords, resolution=64, color=(240, 220, 80))
            write_ply(sample_dir / "target_sparse_coords.ply", target_coords, resolution=64, color=(80, 200, 255))
            make_preview(coords, target_coords, resolution=64, path=sample_dir / "sparse_preview.png")
            (sample_dir / "summary.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    if bool(args.run_native):
        native_tasks = [
            (sample_index, policy, frame_index, param_mode)
            for sample_index in indices
            for policy in native_image_policies
            for frame_index in native_frame_indices
            for param_mode in native_param_modes
        ]
        for sample_index, policy, frame_index, param_mode in tqdm(native_tasks, desc="Native Pixal3D", unit="case", dynamic_ncols=True):
            batch = dataset[sample_index]
            target_coords = load_target_coords(batch["latent_path"])
            image = native_image(batch, frame_index, policy, args.mask_threshold)
            params = native_camera_params(param_mode)
            cond = make_native_condition(image_cond_model, image, params, device)
            seed = args.seed + sample_index * 1009
            coords = sample_coords(
                denoiser,
                decoder,
                cond,
                seed=seed,
                steps=args.steps,
                rescale_t=args.rescale_t,
                device=device,
            )
            condition = f"native_{policy}_f{frame_index}_{param_mode}"
            record(
                sample_index=sample_index,
                batch=batch,
                family="native_pixal3d",
                condition=condition,
                coords=coords,
                target_coords=target_coords,
                metadata={
                    "image_policy": policy,
                    "frame_index": int(frame_index),
                    "native_param_mode": param_mode,
                    **asdict(params),
                },
            )

    if bool(args.run_multiview_base):
        mv_tasks = [(sample_index, pose_mode) for sample_index in indices for pose_mode in multiview_pose_modes]
        for sample_index, pose_mode in tqdm(mv_tasks, desc="Multiview base", unit="case", dynamic_ncols=True):
            batch = dataset[sample_index]
            target_coords = load_target_coords(batch["latent_path"])
            cond_batch = apply_pose_mode(batch, pose_mode, args.seed + sample_index * 104729)
            cond = make_multiview_condition(condition_builder, image_cond_model, cond_batch, args, device)
            seed = args.seed + sample_index * 1009
            coords = sample_coords(
                denoiser,
                decoder,
                cond,
                seed=seed,
                steps=args.steps,
                rescale_t=args.rescale_t,
                device=device,
            )
            condition = f"multiview_base_{pose_mode}"
            record(
                sample_index=sample_index,
                batch=batch,
                family="multiview_base",
                condition=condition,
                coords=coords,
                target_coords=target_coords,
                metadata={
                    "pose_mode": pose_mode,
                    "pose_permutation": cond_batch.get("pose_permutation"),
                },
            )

    pairwise_rows: list[dict] = []
    native_baseline = None
    if native_image_policies and native_frame_indices:
        native_baseline = f"native_{native_image_policies[0]}_f{native_frame_indices[0]}_default"
    multiview_baseline = "multiview_base_correct"
    for row in rows:
        sample_index = int(row["sample_index"])
        baseline_condition = native_baseline if row["family"] == "native_pixal3d" else multiview_baseline
        if not baseline_condition or (sample_index, baseline_condition) not in coords_by_key:
            continue
        base_metrics = metrics_by_key[(sample_index, baseline_condition)]
        curr_coords = coords_by_key[(sample_index, row["condition"])]
        base_coords = coords_by_key[(sample_index, baseline_condition)]
        pairwise_rows.append(
            {
                "family": row["family"],
                "condition": row["condition"],
                "baseline_condition": baseline_condition,
                "sample_index": sample_index,
                "uid": row["uid"],
                "iou_delta": float(row["iou"] - base_metrics["iou"]),
                "target_recall_delta": float(row["target_recall"] - base_metrics["target_recall"]),
                "pred_precision_delta": float(row["pred_precision"] - base_metrics["pred_precision"]),
                "pred_unique_delta": int(row["pred_unique"] - base_metrics["pred_unique"]),
                "pred_overlap_iou": coord_set_iou(curr_coords, base_coords),
            }
        )

    summary_rows = summarize(rows, group_keys=("family", "condition"))
    pairwise_summary = summarize_pairwise(pairwise_rows, group_keys=("family", "condition", "baseline_condition"))
    result = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "manifest": args.manifest,
        "indices": indices,
        "steps": int(args.steps),
        "rescale_t": float(args.rescale_t),
        "native_frame_indices": native_frame_indices,
        "native_image_policies": native_image_policies,
        "native_param_modes": native_param_modes,
        "multiview_pose_modes": multiview_pose_modes,
        "summary": summary_rows,
        "pairwise_summary": pairwise_summary,
        "rows": rows,
        "pairwise_rows": pairwise_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(output_dir / "metrics.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "pairwise_vs_baseline.csv", pairwise_rows)
    write_csv(output_dir / "pairwise_summary.csv", pairwise_summary)
    write_report(output_dir / "report.md", result)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary_rows[:5]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
