#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from ar_pose_trellis.camera import crop_resize_with_intrinsics, ensure_resized_with_intrinsics
from ar_pose_trellis.projected_condition import PIXAL3D_ROTATION


def parse_indices(text: str, count: int) -> list[int]:
    if text.strip().lower() in {"", "all"}:
        return list(range(count))
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [idx for idx in out if 0 <= idx < count]


def load_manifest(data_root: Path, split: str) -> tuple[dict, list[dict]]:
    manifest_path = data_root / f"{split}.json"
    if not manifest_path.exists():
        manifest_path = data_root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload["samples"] if isinstance(payload, dict) else payload
    return payload if isinstance(payload, dict) else {}, samples


def resolve_path(root: str | None, path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute() or root is None:
        return path_obj
    return Path(root) / path_obj


def resolve_npz(data_root: Path, sample: dict) -> Path:
    path = Path(sample["npz"])
    return path if path.is_absolute() else data_root / path


def load_case(data_root: Path, payload: dict, sample: dict, args: argparse.Namespace):
    if "npz" in sample:
        npz_path = resolve_npz(data_root, sample)
        with np.load(npz_path) as data:
            images = torch.from_numpy(data["images"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            alpha = torch.from_numpy(data["alpha"].astype(np.float32) / 255.0)[:, None]
            intrinsics = torch.from_numpy(data["intrinsics"].astype(np.float32))
            extrinsics = torch.from_numpy(data["extrinsics"].astype(np.float32))
            target_coords = data["target_coords"]
        return images, alpha, intrinsics, extrinsics, target_coords, str(npz_path)

    frames = sample.get("frames")
    if not frames:
        raise ValueError(f"sample has neither npz nor frames: {sample.keys()}")
    image_root = sample.get("image_root", payload.get("image_root"))
    mask_root = sample.get("mask_root", payload.get("mask_root"))
    latent_root = sample.get("latent_root", payload.get("latent_root"))
    top_intrinsic = sample.get("intrinsic", payload.get("intrinsic"))
    latent_rel = sample.get("ss_latent", sample.get("ss_latent_path", sample.get("latent")))
    if latent_rel is None:
        raise ValueError(f"sample {sample.get('uid')} has no ss_latent")
    latent_path = resolve_path(latent_root, latent_rel)
    with np.load(latent_path) as latent:
        target_coords = latent["target_coords"] if "target_coords" in latent else np.zeros((0, 3), dtype=np.int32)

    images = []
    alpha = []
    intrinsics = []
    extrinsics = []
    frames = frames[: args.num_views] if args.num_views > 0 else frames
    for frame in frames:
        image = np.asarray(Image.open(resolve_path(image_root, frame["image"])).convert("RGB")).astype(np.float32) / 255.0
        mask_path = frame.get("mask", sample.get("mask"))
        if mask_path is not None:
            mask = Image.open(resolve_path(mask_root, mask_path)).convert("L")
            if mask.size != (image.shape[1], image.shape[0]):
                mask = mask.resize((image.shape[1], image.shape[0]), Image.NEAREST)
            mask_arr = np.asarray(mask).astype(np.float32) / 255.0
        else:
            mask_arr = np.ones(image.shape[:2], dtype=np.float32)
        intrinsic = frame.get("intrinsic", top_intrinsic)
        if intrinsic is None:
            raise ValueError(f"frame has no intrinsic: {frame.get('image')}")
        images.append(torch.from_numpy(image).permute(2, 0, 1))
        alpha.append(torch.from_numpy(mask_arr[None]))
        intrinsics.append(torch.tensor(intrinsic, dtype=torch.float32))
        extrinsics.append(torch.tensor(frame["extrinsic"], dtype=torch.float32))
    return (
        torch.stack(images, dim=0),
        torch.stack(alpha, dim=0),
        torch.stack(intrinsics, dim=0),
        torch.stack(extrinsics, dim=0),
        target_coords,
        str(latent_path),
    )


def coords_to_points(coords: np.ndarray, resolution: int = 64) -> torch.Tensor:
    xyz = coords[:, -3:].astype(np.float32, copy=False)
    return torch.from_numpy((xyz + 0.5) / float(resolution) - 0.5)


def random_grid_points(count: int, seed: int, resolution: int = 64) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    coords = rng.integers(0, resolution, size=(count, 3), dtype=np.int32)
    return coords_to_points(coords, resolution=resolution)


def apply_grid_transform(points: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "identity":
        return points
    if mode == "pixal3d_rotation":
        rot = torch.tensor(PIXAL3D_ROTATION, dtype=points.dtype, device=points.device)
        return points @ rot.T
    raise ValueError(f"Unsupported grid_transform={mode!r}")


def project_support(
    points: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    *,
    extrinsics_are_c2w: bool,
    camera_forward_sign: float,
    min_support: float,
    min_support_ratio: float,
) -> dict:
    device = masks.device
    dtype = torch.float32
    points = points.to(device=device, dtype=dtype)
    masks = masks.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    extrinsics = extrinsics.to(device=device, dtype=dtype)

    if masks.ndim == 4 and masks.shape[1] == 1:
        pass
    else:
        raise ValueError(f"masks should be [V,1,H,W], got {tuple(masks.shape)}")

    w2c = torch.linalg.inv(extrinsics) if extrinsics_are_c2w else extrinsics
    ones = torch.ones((points.shape[0], 1), device=device, dtype=dtype)
    pts_h = torch.cat([points, ones], dim=1)
    cam = torch.einsum("vij,nj->vni", w2c, pts_h)[..., :3]
    signed_depth = cam[..., 2] * float(camera_forward_sign)
    valid_depth = signed_depth > 1e-6
    z = signed_depth.clamp_min(1e-6)
    u = intrinsics[:, 0, 0, None] * (cam[..., 0] / z) + intrinsics[:, 0, 2, None]
    vv = intrinsics[:, 1, 1, None] * (cam[..., 1] / z) + intrinsics[:, 1, 2, None]

    height, width = masks.shape[-2:]
    in_image = valid_depth & (u >= 0.0) & (u <= width - 1.0) & (vv >= 0.0) & (vv <= height - 1.0)
    grid_x = 2.0 * (u + 0.5) / float(width) - 1.0
    grid_y = 2.0 * (vv + 0.5) / float(height) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(masks.shape[0], points.shape[0], 1, 2)
    mask_values = F.grid_sample(
        masks,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(masks.shape[0], points.shape[0]).clamp(0.0, 1.0)
    mask_values = mask_values * in_image.to(dtype)

    visible = in_image.to(dtype).sum(dim=0)
    support = mask_values.sum(dim=0)
    ratio = support / visible.clamp_min(1.0)
    supported = (support >= float(min_support)) & (ratio >= float(min_support_ratio))
    return {
        "count": int(points.shape[0]),
        "visible_mean": float(visible.mean().item()) if points.numel() else 0.0,
        "visible_nonzero_ratio": float((visible > 0).float().mean().item()) if points.numel() else 0.0,
        "support_mean": float(support.mean().item()) if points.numel() else 0.0,
        "support_median": float(support.median().item()) if points.numel() else 0.0,
        "support_ratio_mean": float(ratio.mean().item()) if points.numel() else 0.0,
        "support_ratio_median": float(ratio.median().item()) if points.numel() else 0.0,
        "supported_ratio": float(supported.float().mean().item()) if points.numel() else 0.0,
    }


def summarize(rows: list[dict]) -> dict:
    out = {"num_samples": len(rows)}
    for prefix in ("target", "random"):
        keys = rows[0][prefix].keys() if rows else []
        out[prefix] = {}
        for key in keys:
            values = [row[prefix][key] for row in rows if isinstance(row[prefix].get(key), (int, float))]
            if values:
                out[prefix][key] = float(np.mean(values))
    if rows:
        out["target_minus_random_supported_ratio"] = (
            out["target"].get("supported_ratio", 0.0) - out["random"].get("supported_ratio", 0.0)
        )
        out["target_minus_random_support_ratio_mean"] = (
            out["target"].get("support_ratio_mean", 0.0) - out["random"].get("support_ratio_mean", 0.0)
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--extrinsics_type", choices=["c2w", "w2c"], default="c2w")
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument("--min_support", type=float, default=0.5)
    parser.add_argument("--min_support_ratio", type=float, default=0.15)
    parser.add_argument("--grid_transform", choices=["identity", "pixal3d_rotation"], default="identity")
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    payload, samples = load_manifest(data_root, args.split)
    indices = parse_indices(args.indices, len(samples))
    rows = []
    for item_idx, sample_idx in enumerate(indices):
        images, alpha, intrinsics, extrinsics, target_coords, source_path = load_case(
            data_root,
            payload,
            samples[sample_idx],
            args,
        )
        if args.no_crop:
            _, alpha_pre, intr_pre = ensure_resized_with_intrinsics(images, alpha, intrinsics, resolution=518)
        else:
            _, alpha_pre, intr_pre = crop_resize_with_intrinsics(images, alpha, intrinsics, resolution=518, no_background=True)

        target_points = coords_to_points(target_coords)
        if args.max_points > 0 and target_points.shape[0] > args.max_points:
            gen = torch.Generator()
            gen.manual_seed(args.seed + sample_idx)
            keep = torch.randperm(target_points.shape[0], generator=gen)[: args.max_points]
            target_points = target_points[keep]
        random_points = random_grid_points(target_points.shape[0], args.seed + 100000 + sample_idx)
        target_points_proj = apply_grid_transform(target_points, args.grid_transform)
        random_points_proj = apply_grid_transform(random_points, args.grid_transform)
        target_stats = project_support(
            target_points_proj,
            alpha_pre,
            intr_pre,
            extrinsics,
            extrinsics_are_c2w=args.extrinsics_type == "c2w",
            camera_forward_sign=args.camera_forward_sign,
            min_support=args.min_support,
            min_support_ratio=args.min_support_ratio,
        )
        random_stats = project_support(
            random_points_proj,
            alpha_pre,
            intr_pre,
            extrinsics,
            extrinsics_are_c2w=args.extrinsics_type == "c2w",
            camera_forward_sign=args.camera_forward_sign,
            min_support=args.min_support,
            min_support_ratio=args.min_support_ratio,
        )
        row = {
            "index": int(sample_idx),
            "source": source_path,
            "target": target_stats,
            "random": random_stats,
        }
        rows.append(row)
        print(
            "[projected_alignment] "
            f"idx={sample_idx} target_supported={target_stats['supported_ratio']:.3f} "
            f"random_supported={random_stats['supported_ratio']:.3f} "
            f"target_support_ratio={target_stats['support_ratio_mean']:.3f}",
            flush=True,
        )

    payload = {
        "args": vars(args),
        "summary": summarize(rows),
        "rows": rows,
    }
    print(json.dumps(payload["summary"], indent=2))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
