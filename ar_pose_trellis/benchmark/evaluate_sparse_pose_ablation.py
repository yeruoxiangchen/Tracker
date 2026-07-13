#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


TRACKER_ROOT = Path(__file__).resolve().parents[2]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
DEFAULT_PYTHON = "/home/zjr/anaconda3/envs/reconviagen/bin/python"
VALID_MODES = ("correct", "identity", "shuffle", "noise")


def load_testsets(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    datasets = payload.get("datasets", payload if isinstance(payload, list) else None)
    if not datasets:
        raise ValueError(f"No datasets found in {path}")
    return datasets


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stable_int_seed(*parts: str, base_seed: int = 0) -> int:
    joined = "|".join(parts).encode("utf-8")
    digest = hashlib.sha1(joined).hexdigest()
    return int((base_seed + int(digest[:8], 16)) % (2**32 - 1))


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis.astype(np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def perturb_pose_matrix(
    matrix: list[list[float]],
    rng: np.random.Generator,
    rot_deg: float,
    trans_std: float,
) -> list[list[float]]:
    pose = np.asarray(matrix, dtype=np.float64).copy()
    delta_r = rotation_matrix(rng.normal(size=3), math.radians(float(rot_deg)))
    pose[:3, :3] = delta_r @ pose[:3, :3]
    pose[:3, 3] += rng.normal(0.0, float(trans_std), size=3)
    return pose.astype(float).tolist()


def make_manifest_variant(
    source_manifest: Path,
    output_path: Path,
    mode: str,
    max_frames: int,
    seed: int,
    noise_rot_deg: float,
    noise_trans_std: float,
) -> Path:
    data = load_json(source_manifest)
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Manifest has no frames: {source_manifest}")
    if max_frames > 0:
        frames = frames[:max_frames]
    frames = [dict(frame) for frame in frames]

    rng = np.random.default_rng(seed)
    if mode == "correct":
        pass
    elif mode == "identity":
        identity = np.eye(4, dtype=float).tolist()
        for frame in frames:
            frame["extrinsic"] = identity
    elif mode == "shuffle":
        extrinsics = [frame["extrinsic"] for frame in frames]
        if len(extrinsics) > 1:
            order = rng.permutation(len(extrinsics))
            while np.all(order == np.arange(len(extrinsics))):
                order = rng.permutation(len(extrinsics))
            for frame, idx in zip(frames, order):
                frame["extrinsic"] = extrinsics[int(idx)]
    elif mode == "noise":
        for frame in frames:
            frame["extrinsic"] = perturb_pose_matrix(frame["extrinsic"], rng, noise_rot_deg, noise_trans_std)
    else:
        raise ValueError(f"Unsupported mode {mode}; valid modes={VALID_MODES}")

    data["frames"] = frames
    data["pose_ablation_mode"] = mode
    data["pose_ablation_seed"] = int(seed)
    if mode == "noise":
        data["pose_ablation_noise"] = {
            "rot_deg": float(noise_rot_deg),
            "trans_std": float(noise_trans_std),
        }
    write_json(output_path, data)
    return output_path


def run_command(cmd: list[str], dry_run: bool, env: dict[str, str]) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(TRACKER_ROOT), env=env, check=True)


def load_coords(path: Path) -> np.ndarray:
    coords = torch.load(path, map_location="cpu")
    if torch.is_tensor(coords):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords)
    coords_np = np.asarray(coords_np)
    if coords_np.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if coords_np.ndim != 2 or coords_np.shape[1] < 3:
        raise ValueError(f"Expected coords [N,3+] in {path}, got {coords_np.shape}")
    coords_np = coords_np[:, -3:].astype(np.int64, copy=False)
    good = np.isfinite(coords_np).all(axis=1)
    coords_np = coords_np[good]
    coords_np = coords_np[(coords_np >= 0).all(axis=1) & (coords_np < 64).all(axis=1)]
    if coords_np.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.unique(coords_np, axis=0)


def load_reference_coords(path: Path) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            coords = loaded["target_coords"]
        finally:
            loaded.close()
    else:
        coords = loaded
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(f"Expected target coords [N,3+] in {path}, got {coords.shape}")
    coords = coords[:, -3:].astype(np.int64, copy=False)
    coords = coords[(coords >= 0).all(axis=1) & (coords < 64).all(axis=1)]
    return np.unique(coords, axis=0)


def coords_to_set(coords: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(v) for v in row) for row in coords.tolist()}


def nearest_distances(src: np.ndarray, dst: np.ndarray, chunk: int = 512) -> np.ndarray:
    if src.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if dst.shape[0] == 0:
        return np.full((src.shape[0],), np.inf, dtype=np.float32)
    out = np.empty((src.shape[0],), dtype=np.float32)
    dst_f = dst.astype(np.float32)
    for start in range(0, src.shape[0], chunk):
        sub = src[start : start + chunk].astype(np.float32)
        diff = sub[:, None, :] - dst_f[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        out[start : start + chunk] = np.sqrt(np.min(d2, axis=1))
    return out


def sample_coords(coords: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if coords.shape[0] <= count:
        return coords
    ids = rng.choice(coords.shape[0], size=count, replace=False)
    return coords[ids]


def compare_sparse_coords(
    pred: np.ndarray,
    ref: np.ndarray,
    sample_points: int,
    seed: int,
    fscore_thresholds: list[float],
) -> dict[str, Any]:
    pred_set = coords_to_set(pred)
    ref_set = coords_to_set(ref)
    intersection = len(pred_set & ref_set)
    union = len(pred_set | ref_set)
    precision_voxel = intersection / max(len(pred_set), 1)
    recall_voxel = intersection / max(len(ref_set), 1)

    rng = np.random.default_rng(seed)
    pred_s = sample_coords(pred, sample_points, rng)
    ref_s = sample_coords(ref, sample_points, rng)
    pred_to_ref = nearest_distances(pred_s, ref_s)
    ref_to_pred = nearest_distances(ref_s, pred_s)

    out: dict[str, Any] = {
        "pred_count": int(pred.shape[0]),
        "ref_count": int(ref.shape[0]),
        "empty": bool(pred.shape[0] == 0),
        "voxel_intersection": int(intersection),
        "voxel_union": int(union),
        "voxel_iou": float(intersection / union) if union > 0 else 0.0,
        "voxel_precision": float(precision_voxel),
        "voxel_recall": float(recall_voxel),
        "chamfer_vox": float(0.5 * (np.mean(pred_to_ref) + np.mean(ref_to_pred)))
        if pred.shape[0] > 0 and ref.shape[0] > 0
        else None,
        "pred_to_ref_mean_vox": float(np.mean(pred_to_ref)) if pred.shape[0] > 0 and ref.shape[0] > 0 else None,
        "ref_to_pred_mean_vox": float(np.mean(ref_to_pred)) if pred.shape[0] > 0 and ref.shape[0] > 0 else None,
    }
    for threshold in fscore_thresholds:
        if pred.shape[0] == 0 or ref.shape[0] == 0:
            precision = recall = fscore = 0.0
        else:
            precision = float((pred_to_ref <= threshold).mean())
            recall = float((ref_to_pred <= threshold).mean())
            denom = precision + recall
            fscore = float(2.0 * precision * recall / denom) if denom > 0 else 0.0
        key = f"fscore_{threshold:g}vox"
        out[key] = fscore
        out[f"precision_{threshold:g}vox"] = precision
        out[f"recall_{threshold:g}vox"] = recall
    return out


def run_sparse_generation(
    item: dict,
    mode: str,
    args: argparse.Namespace,
    env: dict[str, str],
) -> tuple[Path, Path]:
    name = item["name"]
    source_manifest = Path(item["manifest"]).resolve()
    case_root = Path(args.output_root) / name
    manifest_path = case_root / "manifests" / f"{mode}.json"
    output_dir = case_root / mode
    make_manifest_variant(
        source_manifest,
        manifest_path,
        mode=mode,
        max_frames=args.max_frames,
        seed=stable_int_seed(name, mode, base_seed=args.seed),
        noise_rot_deg=args.noise_rot_deg,
        noise_trans_std=args.noise_trans_std,
    )

    coords_path = output_dir / "coords.pt"
    stats_path = output_dir / "sparse_stats.json"
    if args.skip_existing and coords_path.exists() and stats_path.exists():
        print(f"[sparse_ablation] skip existing {name}/{mode}: {output_dir}", flush=True)
        return coords_path, stats_path

    image_root = item.get("image_root")
    mask_root = item.get("mask_root")
    if image_root is None or mask_root is None:
        manifest = load_json(source_manifest)
        image_root = manifest.get("image_root")
        mask_root = manifest.get("mask_root")
    if image_root is None or mask_root is None:
        raise ValueError(f"{name} needs image_root and mask_root")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "ar_pose_trellis/generate_ar_pose_mesh.py",
        "--weights",
        args.weights,
        "--checkpoint",
        args.checkpoint,
        "--manifest",
        str(manifest_path),
        "--image_root",
        str(image_root),
        "--mask_root",
        str(mask_root),
        "--output_dir",
        str(output_dir),
        "--max_frames",
        "-1",
        "--ss_steps",
        str(args.ss_steps),
        "--ss_guidance_strength",
        str(args.ss_guidance_strength),
        "--ss_min_coords",
        str(args.ss_min_coords),
        "--only_sparse",
    ]
    if args.cond_fp16:
        cmd.append("--cond_fp16")
    if args.no_crop:
        cmd.append("--no_crop")
    if args.pose_only:
        cmd.append("--pose_only")
    if args.image_only:
        cmd.append("--image_only")
    if args.condition_mode != "ray":
        cmd.extend(["--condition_mode", args.condition_mode])
    if args.projected_grid_resolution != 16:
        cmd.extend(["--projected_grid_resolution", str(args.projected_grid_resolution)])
    if args.projected_min_support != 0.5:
        cmd.extend(["--projected_min_support", str(args.projected_min_support)])
    if args.projected_min_support_ratio != 0.15:
        cmd.extend(["--projected_min_support_ratio", str(args.projected_min_support_ratio)])
    if args.projected_grid_transform != "identity":
        cmd.extend(["--projected_grid_transform", args.projected_grid_transform])
    if args.absolute_pose_condition:
        cmd.append("--absolute_pose_condition")
    if float(args.visual_hull_prior_weight) != 0.0:
        cmd.extend(
            [
                "--visual_hull_prior_weight",
                str(args.visual_hull_prior_weight),
                "--visual_hull_mask_threshold",
                str(args.visual_hull_mask_threshold),
                "--visual_hull_min_visible_views",
                str(args.visual_hull_min_visible_views),
            ]
        )
    run_command(cmd, args.dry_run, env)
    return coords_path, stats_path


def ensure_import_paths() -> None:
    for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def build_inprocess_pipeline(args: argparse.Namespace):
    ensure_import_paths()
    from ar_pose_trellis.pipeline import TrellisARPoseTo3DPipeline

    if args.pose_only and args.image_only:
        raise ValueError("--pose_only and --image_only are mutually exclusive.")
    pipeline = TrellisARPoseTo3DPipeline.from_pretrained(
        args.weights,
        checkpoint_path=args.checkpoint,
        device="cuda",
        use_image_features=not args.pose_only,
        use_pose_features=not args.image_only,
        cond_fp16=args.cond_fp16,
        condition_mode=args.condition_mode,
        projected_grid_resolution=args.projected_grid_resolution,
        projected_min_support=args.projected_min_support,
        projected_min_support_ratio=args.projected_min_support_ratio,
        projected_grid_transform=args.projected_grid_transform,
        apply_lora=True,
    )
    return pipeline


def run_sparse_generation_inprocess(
    item: dict,
    mode: str,
    args: argparse.Namespace,
    pipeline,
) -> tuple[Path, Path]:
    ensure_import_paths()
    from ar_pose_trellis.generate_ar_pose_mesh import load_manifest
    from ar_pose_trellis.visual_hull import visual_hull_logit_bias

    name = item["name"]
    source_manifest = Path(item["manifest"]).resolve()
    case_root = Path(args.output_root) / name
    manifest_path = case_root / "manifests" / f"{mode}.json"
    output_dir = case_root / mode
    make_manifest_variant(
        source_manifest,
        manifest_path,
        mode=mode,
        max_frames=args.max_frames,
        seed=stable_int_seed(name, mode, base_seed=args.seed),
        noise_rot_deg=args.noise_rot_deg,
        noise_trans_std=args.noise_trans_std,
    )

    coords_path = output_dir / "coords.pt"
    stats_path = output_dir / "sparse_stats.json"
    if args.skip_existing and coords_path.exists() and stats_path.exists():
        print(f"[sparse_ablation] skip existing {name}/{mode}: {output_dir}", flush=True)
        return coords_path, stats_path

    image_root = item.get("image_root")
    mask_root = item.get("mask_root")
    if image_root is None or mask_root is None:
        manifest = load_json(source_manifest)
        image_root = manifest.get("image_root")
        mask_root = manifest.get("mask_root")
    if image_root is None or mask_root is None:
        raise ValueError(f"{name} needs image_root and mask_root")

    output_dir.mkdir(parents=True, exist_ok=True)
    images, masks, intrinsics, extrinsics, extr_type = load_manifest(
        str(manifest_path),
        str(image_root),
        str(mask_root),
        max_frames=-1,
    )
    images_pre, masks_pre, intrinsics_pre = pipeline.prepare_inputs(
        images,
        intrinsics=intrinsics,
        masks=masks,
        crop_foreground=not args.no_crop,
        no_background=True,
    )
    extrinsics = extrinsics.to(pipeline.device).float()

    torch.manual_seed(args.seed)
    ss_cond = pipeline.encode_ss_condition(
        images_pre,
        intrinsics_pre,
        extrinsics,
        masks=masks_pre,
        extrinsics_are_c2w=extr_type == "c2w",
        camera_forward_sign=1.0,
        reference_relative_pose=not args.absolute_pose_condition,
    )
    logit_prior = None
    logit_prior_stats = None
    if float(args.visual_hull_prior_weight) != 0.0:
        logit_prior, logit_prior_stats = visual_hull_logit_bias(
            masks_pre
            if masks_pre is not None
            else torch.ones(
                (images_pre.shape[0], 1, images_pre.shape[-2], images_pre.shape[-1]),
                device=images_pre.device,
                dtype=images_pre.dtype,
            ),
            intrinsics_pre,
            extrinsics,
            extrinsics_are_c2w=extr_type == "c2w",
            resolution=pipeline.sparse_logit_resolution,
            mask_threshold=args.visual_hull_mask_threshold,
            min_visible_views=args.visual_hull_min_visible_views,
            weight=args.visual_hull_prior_weight,
        )
    coords = pipeline.sample_sparse_structure(
        ss_cond,
        num_samples=1,
        sampler_params={"steps": args.ss_steps, "cfg_strength": args.ss_guidance_strength},
        threshold=0.0,
        min_coords=args.ss_min_coords,
        logit_prior=logit_prior,
        logit_prior_stats=logit_prior_stats,
    )
    torch.save(coords.detach().cpu(), coords_path)
    write_json(stats_path, getattr(pipeline, "last_sparse_stats", {}))
    del images, masks, intrinsics, extrinsics, images_pre, masks_pre, intrinsics_pre, ss_cond, coords
    torch.cuda.empty_cache()
    return coords_path, stats_path


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                out[f"{key}.{sub_key}"] = sub_value
        else:
            out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whether AR camera poses improve sparse structure by pose ablations."
    )
    parser.add_argument("--testsets", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--modes", default="correct,identity,shuffle,noise")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--ss_guidance_strength", type=float, default=1.0)
    parser.add_argument(
        "--ss_min_coords",
        type=int,
        default=0,
        help="Keep 0 for raw sparse-pose evaluation. Use 4096 only for fallback visualization.",
    )
    parser.add_argument("--noise_rot_deg", type=float, default=15.0)
    parser.add_argument("--noise_trans_std", type=float, default=0.10)
    parser.add_argument("--sample_points", type=int, default=8000)
    parser.add_argument("--fscore_thresholds", default="1,2,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cond_fp16", action="store_true")
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--pose_only", action="store_true")
    parser.add_argument("--image_only", action="store_true")
    parser.add_argument(
        "--condition_mode",
        choices=["ray", "projected"],
        default="ray",
        help="Sparse condition type for the AR-pose pipeline.",
    )
    parser.add_argument("--projected_grid_resolution", type=int, default=16)
    parser.add_argument("--projected_min_support", type=float, default=0.5)
    parser.add_argument("--projected_min_support_ratio", type=float, default=0.15)
    parser.add_argument("--projected_grid_transform", choices=["identity", "pixal3d_rotation"], default="identity")
    parser.add_argument("--absolute_pose_condition", action="store_true")
    parser.add_argument("--visual_hull_prior_weight", type=float, default=0.0)
    parser.add_argument("--visual_hull_mask_threshold", type=float, default=0.5)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--inprocess",
        action="store_true",
        help="Load the AR-pose TRELLIS pipeline once and run all sparse ablations in the same process.",
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    bad_modes = [mode for mode in modes if mode not in VALID_MODES]
    if bad_modes:
        raise ValueError(f"Unsupported modes {bad_modes}; valid modes={VALID_MODES}")
    thresholds = [float(x) for x in args.fscore_thresholds.split(",") if x.strip()]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("ATTN_BACKEND", "flash_attn")
    env.setdefault("SPCONV_ALGO", "native")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

    reports = []
    failures = []
    pipeline = None
    if args.inprocess and not args.dry_run:
        pipeline = build_inprocess_pipeline(args)
    for item in load_testsets(args.testsets):
        name = item["name"]
        ref_path = Path(item["reference_coords"]).resolve() if item.get("reference_coords") else None
        if ref_path is None or not ref_path.exists():
            raise ValueError(f"{name} needs reference_coords for sparse evaluation")
        ref_coords = load_reference_coords(ref_path)

        for mode in modes:
            try:
                if pipeline is not None:
                    coords_path, stats_path = run_sparse_generation_inprocess(item, mode, args, pipeline)
                else:
                    coords_path, stats_path = run_sparse_generation(item, mode, args, env)
                if args.dry_run:
                    continue
                pred_coords = load_coords(coords_path)
                sparse_stats = load_json(stats_path) if stats_path.exists() else {}
                metrics = compare_sparse_coords(
                    pred_coords,
                    ref_coords,
                    sample_points=args.sample_points,
                    seed=args.seed,
                    fscore_thresholds=thresholds,
                )
                report = {
                    "name": name,
                    "mode": mode,
                    "coords_path": str(coords_path),
                    "stats_path": str(stats_path),
                    "reference_coords": str(ref_path),
                    "sparse_stats": sparse_stats,
                    "metrics": metrics,
                }
                reports.append(report)
                print(
                    "[sparse_ablation] "
                    f"{name}/{mode}: pred={metrics['pred_count']} "
                    f"fallback={sparse_stats.get('used_topk_fallback')} "
                    f"iou={metrics['voxel_iou']:.4f} "
                    f"chamfer={metrics['chamfer_vox']}",
                    flush=True,
                )
            except Exception as exc:
                failure = {"name": name, "mode": mode, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print(f"[sparse_ablation] FAILED {name}/{mode}: {failure['error']}", flush=True)
                if not args.continue_on_error:
                    raise

    if args.dry_run:
        return

    summary = {
        "testsets": args.testsets,
        "checkpoint": args.checkpoint,
        "modes": modes,
        "ss_min_coords": args.ss_min_coords,
        "reports": reports,
        "failures": failures,
    }
    report_path = output_root / "sparse_pose_ablation_report.json"
    write_json(report_path, summary)

    rows = [flatten_row(report) for report in reports]
    csv_path = output_root / "sparse_pose_ablation_report.csv"
    if rows:
        keys = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    if failures:
        write_json(output_root / "sparse_pose_ablation_failures.json", {"failures": failures})
    print(f"[sparse_ablation] wrote {report_path}")
    if rows:
        print(f"[sparse_ablation] wrote {csv_path}")


if __name__ == "__main__":
    main()
