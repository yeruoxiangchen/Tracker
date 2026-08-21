#!/usr/bin/env python3
"""Development Gaussian-appearance audit for the final no-VGGT Native SLat.

The evaluator replays Stock and Full from the exact Native-SS coordinates and
master noise bound by an existing inference manifest.  It decodes both SLat
outputs with the frozen TRELLIS Gaussian decoder and renders them in the eight
registered runtime-O input cameras.  These are conditioning-view diagnostics,
not held-out novel-view scores.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


os.environ.setdefault("SPCONV_ALGO", "native")
TRACKER_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRACKER_ROOT, TRACKER_ROOT / "ReconViaGen"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs import (  # noqa: E402
    MANIFEST_FORMAT as MODEL_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_OBJECT_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now  # noqa: E402
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (  # noqa: E402
    canonical_coords,
    sparse_noise_from_master,
)
from pose_point_depth_mv.native_slat_genrecon import (  # noqa: E402
    NativeSLatCalibratedCFGFlow,
    NativeSLatStockFlow,
)
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (  # noqa: E402
    build_native_slat_no_vggt_components,
    load_trainable_state_dict,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import load_stock_slat_freeze  # noqa: E402
from pose_point_depth_mv.omni_real_benchmark_common import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    load_json,
    object_key,
    select_rows,
    sha256_file,
    to_device_tree,
    validate_bound_file,
)
from pose_point_depth_mv.real_object_canonicalization import (  # noqa: E402
    normalize_similarity_extrinsics,
)
from trellis.modules import sparse as sp  # noqa: E402


REPORT_FORMAT = "pose_point_depth_mv.native_no_vggt_gaussian_appearance.v1"
OBJECT_REPORT_FORMAT = (
    "pose_point_depth_mv.native_no_vggt_gaussian_appearance_object.v1"
)
INFERENCE_MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_native_no_vggt_mixed_inference_manifest.v1"
)
BRANCHES = ("stock", "full")
HIGHER_IS_BETTER = ("masked_psnr", "crop_ssim", "alpha_iou")
LOWER_IS_BETTER = ("masked_l1", "crop_lpips")


def parse_csv_int(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("CSV integers must be non-empty and unique")
    return values


def normalize_render_cameras(
    K_pixels: np.ndarray,
    T_O2C: np.ndarray,
    T_O2C_lifting: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized OpenCV intrinsics and rigid O-to-camera matrices."""

    intrinsics = np.asarray(K_pixels, dtype=np.float64).copy()
    physical = np.asarray(T_O2C, dtype=np.float64)
    cached_rigid = np.asarray(T_O2C_lifting, dtype=np.float64)
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError("K_pixels must be [V,3,3]")
    if width <= 0 or height <= 0 or len(intrinsics) != len(physical):
        raise ValueError("invalid render image size or camera count")
    rigid = normalize_similarity_extrinsics(physical)
    if rigid.shape != cached_rigid.shape or not np.allclose(
        rigid, cached_rigid, rtol=1.0e-8, atol=1.0e-9
    ):
        raise RuntimeError("physical T_O2C and cached rigid lifting pose differ")
    intrinsics[:, 0, :] /= float(width)
    intrinsics[:, 1, :] /= float(height)
    if not np.isfinite(intrinsics).all():
        raise ValueError("normalized intrinsics contain non-finite values")
    return intrinsics.astype(np.float32), rigid.astype(np.float32)


def render_registered_center_splat(
    gaussian: Any,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    *,
    resolution: int,
    surface_tolerance: float = 0.02,
    opacity_density: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render decoded Gaussians in an exact registered OpenCV camera.

    TRELLIS' bundled CUDA rasterizer silently culls every primitive for some
    real narrow-FoV cameras whose normalized object-to-camera distance is much
    larger than its training-time orbit cameras.  Pulling those cameras closer
    makes an image, but changes the registered projection.  This audit instead
    projects the decoded Gaussian centers with the frozen K/T matrices, keeps
    the front surface per pixel, and performs deterministic bilinear alpha
    splatting.  It is deliberately a conditioning-view appearance diagnostic,
    not a replacement for the official free-view renderer.
    """

    if resolution <= 0 or surface_tolerance < 0 or opacity_density <= 0:
        raise ValueError("invalid registered center-splat configuration")
    xyz = gaussian.get_xyz.float()
    color = gaussian.get_color.float().clamp(0, 1)
    opacity = gaussian.get_opacity.float().reshape(-1).clamp(0, 1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or color.shape != xyz.shape:
        raise ValueError("invalid decoded Gaussian tensors")
    camera = xyz @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    depth = camera[:, 2]
    safe_depth = depth.clamp_min(1.0e-8)
    u = (intrinsic[0, 0] * camera[:, 0] / safe_depth + intrinsic[0, 2]) * resolution
    v = (intrinsic[1, 1] * camera[:, 1] / safe_depth + intrinsic[1, 2]) * resolution

    x0 = torch.floor(u).to(torch.long)
    y0 = torch.floor(v).to(torch.long)
    du = u - x0
    dv = v - y0
    neighbor_x = torch.stack((x0, x0 + 1, x0, x0 + 1), dim=1).reshape(-1)
    neighbor_y = torch.stack((y0, y0, y0 + 1, y0 + 1), dim=1).reshape(-1)
    bilinear = torch.stack(
        ((1 - du) * (1 - dv), du * (1 - dv), (1 - du) * dv, du * dv),
        dim=1,
    ).reshape(-1)
    repeated_depth = depth[:, None].expand(-1, 4).reshape(-1)
    repeated_opacity = opacity[:, None].expand(-1, 4).reshape(-1)
    repeated_color = color[:, None, :].expand(-1, 4, -1).reshape(-1, 3)
    valid = (
        (repeated_depth > 0)
        & (neighbor_x >= 0)
        & (neighbor_x < resolution)
        & (neighbor_y >= 0)
        & (neighbor_y < resolution)
        & (bilinear > 0)
    )
    if not bool(valid.any()):
        zeros = torch.zeros((resolution, resolution), device=xyz.device, dtype=torch.float32)
        return zeros[..., None].expand(-1, -1, 3).contiguous(), zeros
    pixel = (neighbor_y[valid] * resolution + neighbor_x[valid]).to(torch.long)
    z = repeated_depth[valid]
    weight = bilinear[valid] * repeated_opacity[valid]
    rgb = repeated_color[valid]
    pixel_count = resolution * resolution
    z_buffer = torch.full((pixel_count,), torch.inf, device=xyz.device, dtype=torch.float32)
    z_buffer.scatter_reduce_(0, pixel, z, reduce="amin", include_self=True)
    front = z <= z_buffer[pixel] + float(surface_tolerance)
    pixel = pixel[front]
    weight = weight[front]
    rgb = rgb[front]
    weight_sum = torch.zeros((pixel_count,), device=xyz.device, dtype=torch.float32)
    color_sum = torch.zeros((pixel_count, 3), device=xyz.device, dtype=torch.float32)
    weight_sum.scatter_add_(0, pixel, weight)
    color_sum.scatter_add_(0, pixel[:, None].expand(-1, 3), rgb * weight[:, None])
    alpha = 1.0 - torch.exp(-float(opacity_density) * weight_sum)
    rendered = color_sum / weight_sum.clamp_min(1.0e-8)[:, None]
    rendered = rendered * alpha[:, None]
    return rendered.reshape(resolution, resolution, 3), alpha.reshape(resolution, resolution)


def mask_quality(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be HxW")
    area = int(binary.sum())
    area_ratio = float(area / max(binary.size, 1))
    if area == 0:
        return {
            "area_ratio": area_ratio,
            "largest_component_ratio": 0.0,
            "passed": False,
            "reasons": ["empty_mask"],
        }
    from scipy import ndimage

    labels, count = ndimage.label(binary)
    sizes = np.bincount(labels.reshape(-1))[1:]
    largest_ratio = float(sizes.max() / area) if count else 0.0
    reasons = []
    if not 0.005 <= area_ratio <= 0.95:
        reasons.append("implausible_area_ratio")
    if largest_ratio < 0.80:
        reasons.append("fragmented_foreground")
    return {
        "area_ratio": area_ratio,
        "largest_component_ratio": largest_ratio,
        "passed": not reasons,
        "reasons": reasons,
    }


def _bbox(mask: np.ndarray, margin: int = 12) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot crop an empty mask")
    height, width = mask.shape
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + margin + 1),
        min(height, int(ys.max()) + margin + 1),
    )


def appearance_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    predicted_alpha: np.ndarray,
    *,
    alpha_threshold: float,
    use_lpips: bool,
) -> dict[str, float | None]:
    pred = np.asarray(prediction, dtype=np.float32)
    gt = np.asarray(target, dtype=np.float32)
    mask = np.asarray(target_mask, dtype=bool)
    alpha = np.asarray(predicted_alpha, dtype=np.float32)
    if pred.shape != gt.shape or pred.shape[:2] != mask.shape or pred.shape[2] != 3:
        raise ValueError("appearance arrays have incompatible shapes")
    if alpha.shape != mask.shape or not mask.any():
        raise ValueError("alpha/mask has incompatible shape or empty target")
    pred = np.clip(pred, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)
    diff = pred[mask] - gt[mask]
    mse = float(np.mean(np.square(diff)))
    masked_psnr = float(-10.0 * math.log10(max(mse, 1.0e-12)))
    predicted_mask = alpha >= float(alpha_threshold)
    union = int(np.logical_or(predicted_mask, mask).sum())
    alpha_iou = float(np.logical_and(predicted_mask, mask).sum() / max(union, 1))

    left, top, right, bottom = _bbox(mask)
    crop_pred = torch.from_numpy(pred[top:bottom, left:right]).permute(2, 0, 1)[None]
    crop_gt = torch.from_numpy(gt[top:bottom, left:right]).permute(2, 0, 1)[None]
    crop_pred = F.interpolate(
        crop_pred, size=(256, 256), mode="bilinear", align_corners=False
    )
    crop_gt = F.interpolate(
        crop_gt, size=(256, 256), mode="bilinear", align_corners=False
    )
    from trellis.utils.loss_utils import ssim

    crop_ssim = float(ssim(crop_pred, crop_gt).item())
    crop_lpips: float | None = None
    if use_lpips:
        from trellis.utils.loss_utils import lpips

        crop_lpips = float(lpips(crop_pred.cuda(), crop_gt.cuda()).item())
    return {
        "masked_l1": float(np.mean(np.abs(diff))),
        "masked_psnr": masked_psnr,
        "crop_ssim": crop_ssim,
        "crop_lpips": crop_lpips,
        "alpha_iou": alpha_iou,
    }


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def _contact_sheet(
    targets: list[Path], stock: list[Path], full: list[Path], destination: Path
) -> None:
    rows = list(zip(targets, stock, full))
    if not rows:
        return
    size = Image.open(rows[0][0]).size
    header = 32
    canvas = Image.new("RGB", (size[0] * 3, (size[1] + header) * len(rows)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = _font(15)
    for row_index, paths in enumerate(rows):
        top = row_index * (size[1] + header)
        for column, (label, path) in enumerate(zip(("Input", "Stock GS", "Full GS"), paths)):
            image = Image.open(path).convert("RGB")
            canvas.paste(image, (column * size[0], top + header))
            draw.text((column * size[0] + 8, top + 8), label, fill=(245, 245, 245), font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def summarize_objects(objects: list[dict[str, Any]], *, reliable_only: bool) -> dict[str, Any]:
    selected = [row for row in objects if not reliable_only or row["mask_quality_pass"]]
    output: dict[str, Any] = {"object_count": len(selected), "branches": {}, "full_improvement": {}}
    if not selected:
        return output
    for branch in BRANCHES:
        output["branches"][branch] = {
            metric: _summary([float(row["branch_means"][branch][metric]) for row in selected])
            for metric in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER)
            if all(row["branch_means"][branch].get(metric) is not None for row in selected)
        }
    for metric in HIGHER_IS_BETTER:
        values = [
            float(row["branch_means"]["full"][metric])
            - float(row["branch_means"]["stock"][metric])
            for row in selected
        ]
        output["full_improvement"][metric] = {**_summary(values), "positive_rate": float(np.mean(np.asarray(values) > 0))}
    for metric in LOWER_IS_BETTER:
        if not all(row["branch_means"]["full"].get(metric) is not None for row in selected):
            continue
        values = [
            float(row["branch_means"]["stock"][metric])
            - float(row["branch_means"]["full"][metric])
            for row in selected
        ]
        output["full_improvement"][metric] = {**_summary(values), "positive_rate": float(np.mean(np.asarray(values) > 0))}
    return output


def _load_model_payload(row: dict[str, Any]) -> dict[str, Any]:
    path = validate_bound_file(row["model_input"], row["model_input_sha256"], label="DINO-only model input")
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != MODEL_OBJECT_FORMAT or payload.get("object_key") != object_key(row):
        raise RuntimeError(f"DINO-only payload identity differs: {path}")
    return payload


def _to_textured_glb_with_grad(
    postprocessing_utils: Any,
    gs: Any,
    decoded_mesh: Any,
    *,
    texture_size: int,
) -> Any:
    # TRELLIS optimizes a texture Parameter inside to_glb(). The evaluator's
    # outer no_grad scope is correct for inference but must not cover baking.
    with torch.enable_grad():
        return postprocessing_utils.to_glb(
            gs,
            decoded_mesh,
            simplify=0.95,
            texture_size=int(texture_size),
            verbose=True,
        )


def _atomic_export_glb_mesh(glb: Any, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    try:
        glb.export(temporary, file_type="glb")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _export_glb(gs: Any, decoded_mesh: Any, destination: Path, *, texture_size: int) -> None:
    from trellis.utils import postprocessing_utils

    glb = _to_textured_glb_with_grad(
        postprocessing_utils,
        gs,
        decoded_mesh,
        texture_size=texture_size,
    )
    _atomic_export_glb_mesh(glb, destination)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_objects", type=int, default=6)
    parser.add_argument("--object_offset", type=int, default=0)
    parser.add_argument(
        "--selection_manifest",
        help=(
            "Optional qualitative-review selection manifest. When provided, "
            "the exact ordered object keys in its selected rows replace the "
            "offset/max_objects slice. This does not select a checkpoint or "
            "change inference parameters."
        ),
    )
    parser.add_argument("--views", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--alpha_threshold", type=float, default=0.05)
    parser.add_argument("--surface_tolerance", type=float, default=0.02)
    parser.add_argument("--opacity_density", type=float, default=2.0)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--with_lpips", action="store_true")
    parser.add_argument("--export_glb", action="store_true")
    parser.add_argument("--texture_size", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    return parser


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    if (
        args.max_objects <= 0
        or args.object_offset < 0
        or args.resolution <= 0
        or args.surface_tolerance < 0
        or args.opacity_density <= 0
    ):
        raise ValueError("invalid object range or resolution")
    if int(args.resolution) != 518:
        raise ValueError("v1 appearance audit is frozen to the 518x518 runtime inputs")
    view_indices = parse_csv_int(args.views)
    inference_path = Path(args.inference_manifest).expanduser().resolve()
    inference = load_json(inference_path)
    if (
        inference.get("format") != INFERENCE_MANIFEST_FORMAT
        or inference.get("method") != "native_no_vggt_mixed"
        or inference.get("passed") is not True
        or inference.get("vggt_model_executed") is not False
    ):
        raise RuntimeError("the source is not a passed final no-VGGT inference manifest")
    seed = int(args.seed)
    if seed not in [int(value) for value in inference.get("seeds", [])]:
        raise ValueError(f"seed {seed} was not frozen by the inference manifest")
    model_manifest_path = validate_bound_file(
        inference["model_input_manifest"],
        inference["model_input_manifest_sha256"],
        label="DINO-only model manifest",
    )
    model_manifest = load_json(model_manifest_path)
    if model_manifest.get("format") != MODEL_MANIFEST_FORMAT or model_manifest.get("passed") is not True:
        raise RuntimeError("DINO-only model manifest did not pass")
    ordered_rows = select_rows(model_manifest["objects"], None)
    inference_keys = {str(row["object_key"]) for row in inference["objects"] if int(row["seed"]) == seed}
    ordered_rows = [row for row in ordered_rows if object_key(row) in inference_keys]
    selection_path = None
    selection_sha256 = None
    if args.selection_manifest:
        selection_path = Path(args.selection_manifest).expanduser().resolve()
        selection = load_json(selection_path)
        selected_entries = selection.get("selected")
        if not isinstance(selected_entries, list) or not selected_entries:
            raise RuntimeError("selection manifest has no selected object rows")
        selected_keys = [str(row["object_key"]) for row in selected_entries]
        if len(selected_keys) != len(set(selected_keys)):
            raise RuntimeError("selection manifest contains duplicate object keys")
        by_key = {object_key(row): row for row in ordered_rows}
        missing = [key for key in selected_keys if key not in by_key]
        if missing:
            raise RuntimeError(f"selected objects are absent from inference: {missing}")
        selected_rows = [by_key[key] for key in selected_keys]
        selection_sha256 = sha256_file(selection_path)
    else:
        selected_rows = ordered_rows[
            args.object_offset : args.object_offset + args.max_objects
        ]
        if (
            len(selected_rows)
            != min(args.max_objects, max(0, len(ordered_rows) - args.object_offset))
            or not selected_rows
        ):
            raise RuntimeError("object selection is incomplete")
    global_positions = {object_key(row): index for index, row in enumerate(ordered_rows)}

    slat_path = validate_bound_file(
        inference["native_slat_checkpoint"],
        inference["native_slat_checkpoint_sha256"],
        label="Native-SLat checkpoint",
    )
    stock_path = validate_bound_file(
        inference["stock_slat_freeze"],
        inference["stock_slat_freeze_sha256"],
        label="Stock SLat freeze",
    )
    checkpoint = torch.load(slat_path, map_location="cpu")
    stock_freeze = load_stock_slat_freeze(stock_path)
    upstream = dict(checkpoint["model_summary"]["upstream_native_ss"])
    validate_native_slat_no_vggt_checkpoint(
        checkpoint,
        pretrained=args.pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=upstream,
        allow_v2_parent=False,
    )
    decoder_kind = "mesh_and_gaussian" if args.export_glb else "gaussian"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    sampler, model, decoder, _, defaults, normalization = build_native_slat_no_vggt_components(
        pretrained=args.pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=upstream,
        lora_rank=int(checkpoint["args"]["lora_rank"]),
        lora_alpha=int(checkpoint["args"]["lora_alpha"]),
        condition_channels=int(checkpoint["args"]["condition_channels"]),
        gradient_checkpointing=False,
        need_decoder=True,
        decoder_kind=decoder_kind,
        decoder_to_device=False,
        device=device,
    )
    if decoder is None:
        raise RuntimeError("frozen Gaussian decoder was not loaded")
    gs_decoder = decoder["gaussian"] if isinstance(decoder, dict) else decoder
    mesh_decoder = decoder.get("mesh") if isinstance(decoder, dict) else None
    state_key = "ema_trainable_state" if args.weights == "ema" else "model_trainable_state"
    load_trainable_state_dict(model, checkpoint[state_key])
    model.eval()
    gs_decoder.eval()
    if mesh_decoder is not None:
        mesh_decoder.eval()
    params = dict(defaults)
    params.setdefault("guidance_rescale", 0.0)
    source_sampling = dict(inference["objects"][0]["sampling"])
    if canonical_sha256(params) != canonical_sha256(source_sampling):
        raise RuntimeError("runtime SLat sampling differs from source inference")
    mean = torch.tensor(normalization["mean"], device=device)[None]
    std = torch.tensor(normalization["std"], device=device)[None]
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume only for the same run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "format": REPORT_FORMAT,
        "source_inference_manifest": str(inference_path),
        "source_inference_manifest_sha256": sha256_file(inference_path),
        "checkpoint": str(slat_path),
        "checkpoint_sha256": sha256_file(slat_path),
        "weights": args.weights,
        "stock_slat_freeze": str(stock_path),
        "stock_slat_freeze_sha256": sha256_file(stock_path),
        "seed": seed,
        "object_keys": [object_key(row) for row in selected_rows],
        "selection_manifest": str(selection_path) if selection_path else None,
        "selection_manifest_sha256": selection_sha256,
        "view_indices": view_indices,
        "resolution": int(args.resolution),
        "alpha_threshold": float(args.alpha_threshold),
        "renderer": "registered_center_splat_v1",
        "surface_tolerance": float(args.surface_tolerance),
        "opacity_density": float(args.opacity_density),
        "with_lpips": bool(args.with_lpips),
        "export_glb": bool(args.export_glb),
        "texture_size": int(args.texture_size),
        "sampling": params,
        "same_native_ss_coordinates": True,
        "same_initial_noise": True,
        "camera_policy": "exact normalized OpenCV K plus verified rigid runtime-O T_O2C; no camera pull-in",
        "metric_scope": "registered conditioning views only",
    }
    run_config_sha = canonical_sha256(run_config)
    run_config_path = output_dir / "run_config.json"
    if run_config_path.is_file():
        frozen_run = load_json(run_config_path)
        if frozen_run.get("run_config_sha256") != run_config_sha:
            raise RuntimeError("resumed output binds a different run configuration")
    else:
        atomic_json(
            run_config_path,
            {**run_config, "run_config_sha256": run_config_sha},
        )
    object_reports: list[dict[str, Any]] = []
    inference_root = inference_path.parent
    for selected_index, row in enumerate(selected_rows):
        key = object_key(row)
        final_dir = output_dir / "objects" / row["category"] / row["object_id"]
        final_report = final_dir / "report.json"
        if final_report.is_file():
            existing = load_json(final_report)
            if existing.get("format") != OBJECT_REPORT_FORMAT or existing.get("run_config_sha256") != run_config_sha:
                raise RuntimeError(f"stale resumed object={key}")
            object_reports.append(existing)
            print(f"[gaussian_appearance] reuse {selected_index + 1}/{len(selected_rows)} {key}", flush=True)
            continue
        staging = final_dir.parent / f".{row['object_id']}.gaussian-building"
        if staging.exists():
            if not args.resume:
                raise RuntimeError(f"partial object exists: {staging}")
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        model.to(device)
        payload = _load_model_payload(row)
        condition = to_device_tree(payload["slat_condition"], device)
        coord_path = inference_root / "ss_coords" / row["category"] / row["object_id"] / f"seed_{seed}.npz"
        with np.load(coord_path, allow_pickle=False) as values:
            coords_np = canonical_coords(values["coords"], resolution=64)
        object_position = global_positions[key]
        master_seed = seed * 2000003 + object_position * 2017 + 7919
        generator = torch.Generator(device=device).manual_seed(master_seed)
        master = torch.randn((64, 64, 64, 8), generator=generator, device=device)
        initial = sparse_noise_from_master(coords_np, master, device=device)
        latents: dict[str, Any] = {}
        wrapper: dict[str, Any] = {}
        for branch in BRANCHES:
            noise = sp.SparseTensor(feats=initial.feats.clone(), coords=initial.coords.clone())
            flow: Any
            if branch == "stock":
                flow = NativeSLatStockFlow(model)
            else:
                flow = NativeSLatCalibratedCFGFlow(model, condition["cond"], payload, enabled=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                latents[branch] = sampler.sample(flow, noise, **condition, **params, verbose=False).samples
            if branch == "full":
                wrapper = flow.summary()
                if flow.positive_calls <= 0 or flow.negative_calls <= 0:
                    raise RuntimeError("Full SLat CFG missed a branch")
        if not torch.equal(latents["stock"].coords, latents["full"].coords):
            raise RuntimeError("Stock/Full changed sparse coordinates")
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        with np.load(row["runtime_cache"], allow_pickle=False) as camera_values:
            K_norm, T_rigid = normalize_render_cameras(
                camera_values["K_feature"],
                camera_values["T_O2C"],
                camera_values["T_O2C_lifting"],
                width=518,
                height=518,
            )
        if any(index < 0 or index >= len(K_norm) for index in view_indices):
            raise ValueError(f"view index exceeds runtime camera count for {key}")
        quality_rows = []
        targets = []
        masks = []
        for index in view_indices:
            target = np.asarray(Image.open(row["prepared_rgb_paths"][index]).convert("RGB"), dtype=np.float32) / 255.0
            mask = np.asarray(Image.open(row["prepared_mask_paths"][index]).convert("L"), dtype=np.float32) / 255.0 >= 0.5
            targets.append(target)
            masks.append(mask)
            quality_rows.append({"view_index": index, **mask_quality(mask)})
        branch_records: dict[str, list[dict[str, Any]]] = {}
        branch_artifacts: dict[str, dict[str, Any]] = {}
        render_paths: dict[str, list[Path]] = {}
        for branch in BRANCHES:
            denormalized = latents[branch] * std + mean
            gs_decoder.to(device)
            gs = gs_decoder(denormalized)[0]
            if args.export_glb:
                gs_decoder.cpu()
                gc.collect()
                torch.cuda.empty_cache()
            branch_dir = staging / branch
            branch_dir.mkdir()
            ply_path = branch_dir / "gaussian.ply"
            gs.save_ply(ply_path)
            artifacts = {
                "gaussian_ply": str(final_dir / branch / ply_path.name),
                "gaussian_ply_sha256": sha256_file(ply_path),
            }
            if args.export_glb:
                assert mesh_decoder is not None
                mesh_decoder.to(device)
                decoded_mesh = mesh_decoder(denormalized)[0]
                mesh_decoder.cpu()
                gc.collect()
                torch.cuda.empty_cache()
                glb_path = branch_dir / "mesh_textured.glb"
                _export_glb(gs, decoded_mesh, glb_path, texture_size=args.texture_size)
                artifacts.update(
                    {
                        "textured_glb": str(final_dir / branch / glb_path.name),
                        "textured_glb_sha256": sha256_file(glb_path),
                    }
                )
                del decoded_mesh
            records = []
            paths = []
            for local_index, view_index in enumerate(view_indices):
                extrinsic = torch.from_numpy(T_rigid[view_index]).to(device)
                intrinsic = torch.from_numpy(K_norm[view_index]).to(device)
                color, alpha = render_registered_center_splat(
                    gs,
                    extrinsic,
                    intrinsic,
                    resolution=int(args.resolution),
                    surface_tolerance=float(args.surface_tolerance),
                    opacity_density=float(args.opacity_density),
                )
                color_np = color.float().cpu().numpy()
                alpha_np = alpha.float().cpu().numpy()
                if not np.isfinite(color_np).all() or not np.isfinite(alpha_np).all():
                    raise RuntimeError(f"non-finite Gaussian render branch={branch} view={view_index}")
                if int((alpha_np >= float(args.alpha_threshold)).sum()) == 0:
                    raise RuntimeError(
                        f"empty registered Gaussian render branch={branch} view={view_index}"
                    )
                color_path = branch_dir / f"view_{view_index:02d}_rgb.png"
                alpha_path = branch_dir / f"view_{view_index:02d}_alpha.png"
                Image.fromarray(np.clip(np.rint(color_np * 255), 0, 255).astype(np.uint8)).save(color_path)
                Image.fromarray(np.clip(np.rint(alpha_np * 255), 0, 255).astype(np.uint8)).save(alpha_path)
                metrics = appearance_metrics(
                    color_np,
                    targets[local_index],
                    masks[local_index],
                    alpha_np,
                    alpha_threshold=float(args.alpha_threshold),
                    use_lpips=bool(args.with_lpips),
                )
                records.append(
                    {
                        "view_index": view_index,
                        "rgb": str(final_dir / branch / color_path.name),
                        "alpha": str(final_dir / branch / alpha_path.name),
                        **metrics,
                    }
                )
                paths.append(color_path)
            branch_records[branch] = records
            branch_artifacts[branch] = artifacts
            render_paths[branch] = paths
            del gs, denormalized
            torch.cuda.empty_cache()
        gs_decoder.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        target_paths = [Path(row["prepared_rgb_paths"][index]) for index in view_indices]
        _contact_sheet(target_paths, render_paths["stock"], render_paths["full"], staging / "input_stock_full_contact_sheet.png")
        branch_means = {
            branch: {
                metric: (
                    float(np.mean([float(record[metric]) for record in branch_records[branch]]))
                    if branch_records[branch][0][metric] is not None
                    else None
                )
                for metric in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER)
            }
            for branch in BRANCHES
        }
        object_report = {
            "format": OBJECT_REPORT_FORMAT,
            "created_at_utc": utc_now(),
            "run_config_sha256": run_config_sha,
            "object_key": key,
            "category": row["category"],
            "object_id": row["object_id"],
            "seed": seed,
            "coord_count": int(len(coords_np)),
            "same_native_ss_coordinates": True,
            "same_initial_noise": True,
            "mask_quality_pass": all(value["passed"] for value in quality_rows),
            "mask_quality": quality_rows,
            "branch_means": branch_means,
            "branches": branch_records,
            "artifacts": branch_artifacts,
            "contact_sheet": str(final_dir / "input_stock_full_contact_sheet.png"),
            "wrapper": wrapper,
            "passed": True,
        }
        atomic_json(staging / "report.json", object_report)
        if final_dir.exists():
            raise RuntimeError(f"final object directory unexpectedly exists: {final_dir}")
        staging.replace(final_dir)
        object_report = load_json(final_dir / "report.json")
        object_reports.append(object_report)
        print(f"[gaussian_appearance] {selected_index + 1}/{len(selected_rows)} {key}", flush=True)
        del payload, condition, master, initial, latents
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "formal": False,
        "run_config": run_config,
        "run_config_sha256": run_config_sha,
        "object_count": len(object_reports),
        "view_count_per_object": len(view_indices),
        "record_count": len(object_reports) * len(view_indices) * len(BRANCHES),
        "all_objects": summarize_objects(object_reports, reliable_only=False),
        "reliable_mask_subset": summarize_objects(object_reports, reliable_only=True),
        "low_quality_mask_objects": [row["object_key"] for row in object_reports if not row["mask_quality_pass"]],
        "objects": object_reports,
        "scope_guard": (
            "development registered-conditioning-view Gaussian appearance audit; "
            "not a held-out novel-view, textured-GLB, or formal Holdout64 claim"
        ),
        "passed": len(object_reports) == len(selected_rows),
    }
    atomic_json(output_dir / "report.json", report)
    print(json.dumps({
        "passed": report["passed"],
        "objects": report["object_count"],
        "views_per_object": report["view_count_per_object"],
        "low_quality_mask_objects": report["low_quality_mask_objects"],
        "all_objects": report["all_objects"],
        "reliable_mask_subset": report["reliable_mask_subset"],
        "report": str(output_dir / "report.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
