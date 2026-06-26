#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

from trellis_point_prior_mv.common import parse_indices, resolve_path, write_json  # noqa: E402
from trellis_point_prior_mv.eval_mesh_frozen_downstream import (  # noqa: E402
    apply_base_guidance_to_logits,
    apply_mask_and_crop,
    coords_np_to_torch,
    expand_modes_with_stage2_specs,
    is_stage2_union_mode,
    load_stage2_bundle,
    mesh_artifact_metrics_from_obj,
    mesh_basic_metrics,
    parse_stage2_topk_specs,
    prepare_cond,
    resolve_stage2_topk,
    sample_slat_mesh,
    sample_stage2_logits,
    sample_stock_sparse,
    stage2_topk_label,
    torch_coords_to_np,
    union_stage2_with_stock_coords,
)
from trellis_point_prior_mv.eval_sparse_inpaint import topk_coords_from_logits  # noqa: E402
from trellis_point_prior_mv.sparse_coord_tools import (  # noqa: E402
    filter_sparse_coords,
    sparse_diagnostic_metrics,
)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_condition_images(sample: dict, *, max_frames: int, resolution: int) -> list:
    frames = sample.get("frames") or []
    if max_frames > 0:
        frames = frames[: int(max_frames)]
    if not frames:
        raise ValueError(f"sample {sample.get('uid')} has no frames")
    images = []
    for frame in frames:
        images.append(apply_mask_and_crop(Path(frame["image"]), Path(frame["mask"]), resolution))
    return images


def coords_to_points(coords: np.ndarray, resolution: int = 64) -> np.ndarray:
    xyz = coords[:, -3:].astype(np.float32, copy=False) if coords.size else np.zeros((0, 3), dtype=np.float32)
    return (xyz + 0.5) / float(resolution) - 0.5


def sample_reference_points(model_path: str | None, sample_count: int, seed: int) -> np.ndarray:
    if not model_path:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        import trimesh
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    path = Path(model_path)
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    if mesh.vertices is None or len(mesh.vertices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, int(sample_count))
    except Exception:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        replace = verts.shape[0] < sample_count
        ids = rng.choice(verts.shape[0], size=int(sample_count), replace=replace)
        points = verts[ids]
    return np.asarray(points, dtype=np.float32)


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return points.reshape(0, 3)
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    center = (pmin + pmax) * 0.5
    scale = max(float((pmax - pmin).max()), 1e-6)
    return (points - center[None, :]) / scale


def mesh_to_points(mesh, sample_count: int, seed: int) -> np.ndarray:
    try:
        import trimesh
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    tri = mesh.to_trimesh(transform_pose=False)
    if tri.vertices is None or len(tri.vertices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)
    try:
        points, _ = trimesh.sample.sample_surface(tri, int(sample_count))
    except Exception:
        verts = np.asarray(tri.vertices, dtype=np.float32)
        replace = verts.shape[0] < sample_count
        ids = rng.choice(verts.shape[0], size=int(sample_count), replace=replace)
        points = verts[ids]
    return np.asarray(points, dtype=np.float32)


def bidirectional_distance_metrics(prefix: str, a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return {f"{prefix}_enabled": 0}
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return {f"{prefix}_enabled": 0}
    a_tree = cKDTree(a)
    b_tree = cKDTree(b)
    a_to_b = b_tree.query(a, k=1)[0]
    b_to_a = a_tree.query(b, k=1)[0]
    return {
        f"{prefix}_enabled": 1,
        f"{prefix}_a_to_b_mean": float(np.mean(a_to_b)),
        f"{prefix}_a_to_b_median": float(np.median(a_to_b)),
        f"{prefix}_b_to_a_mean": float(np.mean(b_to_a)),
        f"{prefix}_b_to_a_median": float(np.median(b_to_a)),
        f"{prefix}_chamfer_l2_mean": float(np.mean(a_to_b**2) + np.mean(b_to_a**2)),
        f"{prefix}_a_count": int(a.shape[0]),
        f"{prefix}_b_count": int(b.shape[0]),
    }


def real_reference_metrics(mesh, sample: dict, prior_coords: np.ndarray, args: argparse.Namespace, seed: int) -> dict[str, float | int]:
    mesh_points = mesh_to_points(mesh, int(args.mesh_eval_samples), seed)
    mesh_norm = normalize_points(mesh_points)
    ref_points = sample_reference_points(sample.get("reference_model"), int(args.mesh_eval_samples), seed + 17)
    ref_norm = normalize_points(ref_points)
    prior_points = coords_to_points(prior_coords)
    out: dict[str, float | int] = {}
    out.update(bidirectional_distance_metrics("ref_norm", mesh_norm, ref_norm))
    out.update(bidirectional_distance_metrics("prior", mesh_points, prior_points))
    out.update(bidirectional_distance_metrics("prior_norm", mesh_norm, normalize_points(prior_points)))
    return out


def coords_from_stage2_logits_abs(logits: torch.Tensor, topk: int, args: argparse.Namespace, device: torch.device, seed: int) -> torch.Tensor:
    pred = topk_coords_from_logits(logits, topk)
    return coords_np_to_torch(pred, device, max_coords=args.max_coords, seed=seed)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_json(args.manifest)
    samples = payload["samples"]
    indices = parse_indices(args.indices, len(samples))
    raw_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    stage2_specs = parse_stage2_topk_specs(args)
    modes, stage2_mode_specs = expand_modes_with_stage2_specs(raw_modes, stage2_specs)

    print(f"[real_slam_mesh] loading pipeline weights={args.weights}", flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)
    pipeline.low_vram = False
    pipeline.models["sparse_structure_decoder"].to(device)
    stage2_bundle = load_stage2_bundle(args, pipeline, device) if stage2_mode_specs else None

    rows: list[dict] = []
    for order, sample_idx in enumerate(indices):
        sample = samples[sample_idx]
        uid = str(sample.get("uid", sample_idx))
        sample_dir = output_dir / f"{sample_idx:04d}_{uid}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        with np.load(resolve_path(payload.get("prior_root"), sample["prior_npz"])) as data:
            prior_coords = np.asarray(data["prior_coords"], dtype=np.int32)
            prior_conf = np.asarray(data["prior_conf"], dtype=np.float32) if "prior_conf" in data else np.ones((prior_coords.shape[0],), dtype=np.float32)
        if prior_coords.shape[0] == 0:
            raise ValueError(f"empty prior for {uid}")
        images = load_condition_images(sample, max_frames=args.max_frames, resolution=args.resolution)
        cond, cond_count = prepare_cond(pipeline, images, args.cond_mode)
        target_unique_estimate = int(args.target_unique_estimate or args.absolute_topk_reference or max(prior_coords.shape[0] * 8, 12000))

        coord_bank: dict[str, torch.Tensor] = {}
        stage2_info: dict[str, dict] = {}
        stage2_logits = None
        base_guidance_coords_np: np.ndarray | None = None
        for mode_idx, mode in enumerate(modes):
            mode_seed = int(args.seed + sample_idx * 1009 + mode_idx * 17)
            if mode == "prior_sparse":
                coords = coords_np_to_torch(prior_coords, device, max_coords=args.max_coords, seed=mode_seed)
            elif mode == "stock_sparse":
                coords = sample_stock_sparse(pipeline, cond, cond_count, args, mode_seed)
                if args.max_coords > 0 and coords.shape[0] > args.max_coords:
                    coords = coords_np_to_torch(torch_coords_to_np(coords), device, max_coords=args.max_coords, seed=mode_seed)
            elif mode in stage2_mode_specs:
                if stage2_bundle is None:
                    raise ValueError("stage2 modes require --stage2_checkpoint")
                if stage2_logits is None:
                    stage2_logits = sample_stage2_logits(stage2_bundle, pipeline, prior_coords, prior_conf, args, device, mode_seed)
                topk, info = resolve_stage2_topk(stage2_mode_specs[mode], target_unique_estimate)
                guided_logits = stage2_logits
                effective_topk = int(topk)
                if str(args.stage2_base_guidance).strip().lower() not in {"", "none"}:
                    if str(args.stage2_base_guidance).strip().lower() not in {"stock", "stock_sparse", "base", "base_sparse"}:
                        raise ValueError(f"unsupported --stage2_base_guidance={args.stage2_base_guidance!r}")
                    if base_guidance_coords_np is None:
                        if "stock_sparse" in coord_bank:
                            base_guidance_coords_np = torch_coords_to_np(coord_bank["stock_sparse"])
                        else:
                            base_coords = sample_stock_sparse(pipeline, cond, cond_count, args, mode_seed + 39031)
                            base_guidance_coords_np = torch_coords_to_np(base_coords)
                    guided_logits, candidate_count, base_info = apply_base_guidance_to_logits(
                        stage2_logits,
                        base_guidance_coords_np,
                        radius=float(args.stage2_base_radius),
                        min_candidates=int(args.stage2_base_min_candidates),
                    )
                    if not int(base_info.get("stage2_base_guidance_fallback_unmasked", 0)):
                        effective_topk = min(effective_topk, int(candidate_count))
                    info = {
                        **info,
                        **base_info,
                        "stage2_topk_requested": int(topk),
                        "stage2_topk_effective": int(effective_topk),
                    }
                raw_coords = topk_coords_from_logits(guided_logits, effective_topk)
                raw_metrics = sparse_diagnostic_metrics(
                    "stage2_raw_sparse",
                    raw_coords,
                    prior_coords,
                    sample,
                    prior_radius=float(args.filter_prior_radius),
                    min_support_views=int(args.filter_min_support_views),
                    min_support_ratio=float(args.filter_min_support_ratio),
                    visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
                    visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
                    grid_resolution=int(args.filter_grid_resolution),
                    mask_threshold=int(args.filter_mask_threshold),
                )
                filter_metrics: dict[str, Any] = {}
                coords_np = raw_coords
                if str(args.stage2_sparse_filter).strip().lower() not in {"", "none", "raw"}:
                    coords_np, filter_metrics = filter_sparse_coords(
                        raw_coords,
                        prior_coords,
                        sample,
                        filter_spec=str(args.stage2_sparse_filter),
                        prior_radius=float(args.filter_prior_radius),
                        min_component_size=int(args.filter_min_component_size),
                        min_support_views=int(args.filter_min_support_views),
                        min_support_ratio=float(args.filter_min_support_ratio),
                        visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
                        visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
                        grid_resolution=int(args.filter_grid_resolution),
                        mask_threshold=int(args.filter_mask_threshold),
                    )
                    if coords_np.shape[0] < int(args.filter_min_coords) and bool(args.filter_fallback_unfiltered):
                        filter_metrics["sparse_filter_fallback_unfiltered"] = 1
                        filter_metrics["sparse_filter_fallback_reason"] = (
                            f"filtered_count {coords_np.shape[0]} < filter_min_coords {int(args.filter_min_coords)}"
                        )
                        coords_np = raw_coords
                    else:
                        filter_metrics["sparse_filter_fallback_unfiltered"] = 0
                        filter_metrics["sparse_filter_fallback_reason"] = ""
                pre_union_metrics: dict[str, Any] = {}
                union_metrics: dict[str, Any] = {"stage2_stock_union_enabled": 0}
                if is_stage2_union_mode(mode, args):
                    stage2_before_union_np = coords_np
                    if base_guidance_coords_np is None:
                        if "stock_sparse" in coord_bank:
                            base_guidance_coords_np = torch_coords_to_np(coord_bank["stock_sparse"])
                        else:
                            base_coords = sample_stock_sparse(pipeline, cond, cond_count, args, mode_seed + 39031)
                            base_guidance_coords_np = torch_coords_to_np(base_coords)
                    coords_np, union_metrics = union_stage2_with_stock_coords(
                        base_guidance_coords_np,
                        stage2_before_union_np,
                        resolution=int(args.filter_grid_resolution),
                    )
                    pre_union_metrics = sparse_diagnostic_metrics(
                        "stage2_pre_union_sparse",
                        stage2_before_union_np,
                        prior_coords,
                        sample,
                        prior_radius=float(args.filter_prior_radius),
                        min_support_views=int(args.filter_min_support_views),
                        min_support_ratio=float(args.filter_min_support_ratio),
                        visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
                        visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
                        grid_resolution=int(args.filter_grid_resolution),
                        mask_threshold=int(args.filter_mask_threshold),
                    )
                filtered_metrics = sparse_diagnostic_metrics(
                    "stage2_final_sparse",
                    coords_np,
                    prior_coords,
                    sample,
                    prior_radius=float(args.filter_prior_radius),
                    min_support_views=int(args.filter_min_support_views),
                    min_support_ratio=float(args.filter_min_support_ratio),
                    visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
                    visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
                    grid_resolution=int(args.filter_grid_resolution),
                    mask_threshold=int(args.filter_mask_threshold),
                )
                coords = coords_np_to_torch(coords_np, device, max_coords=args.max_coords, seed=mode_seed)
                stage2_info[mode] = {
                    **info,
                    "stage2_topk_reference": int(target_unique_estimate),
                    "stage2_topk_absolute": int(effective_topk),
                    "stage2_sparse_filter": str(args.stage2_sparse_filter),
                    **raw_metrics,
                    **filter_metrics,
                    **pre_union_metrics,
                    **union_metrics,
                    **filtered_metrics,
                }
            else:
                raise ValueError(f"unsupported mode={mode!r}")
            coord_bank[mode] = coords

        for mode, coords in coord_bank.items():
            mode_dir = sample_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            coords_np = torch_coords_to_np(coords)
            np.savez_compressed(mode_dir / "sparse_coords.npz", coords=coords_np)
            print(f"[real_slam_mesh] sample={sample_idx} uid={uid} mode={mode} coords={coords_np.shape[0]}", flush=True)
            mesh = sample_slat_mesh(pipeline, cond, cond_count, coords, args, int(args.seed + sample_idx * 3571))
            tri = mesh.to_trimesh(transform_pose=bool(args.transform_mesh_pose))
            obj_path = mode_dir / "mesh.obj"
            tri.export(obj_path)
            normalization = sample.get("normalization") or {}
            projection_diag = sample.get("projection_diagnostics") or {}
            projection_metrics = {
                f"prior_build_{k}": v
                for k, v in projection_diag.items()
                if isinstance(v, (int, float))
            }
            metrics = {
                "sample_order": int(order),
                "sample_index": int(sample_idx),
                "uid": uid,
                "dataset_root": sample.get("dataset_root"),
                "mode": mode,
                "prior_source": sample.get("prior_source"),
                "normalization_source": normalization.get("source"),
                "fallback_used": int(bool(sample.get("fallback_used", False))),
                "fallback_reason": sample.get("fallback_reason", ""),
                "image_count": len(images),
                "prior_point_count": int(prior_coords.shape[0]),
                "coord_count": int(coords_np.shape[0]),
                "reference_model": sample.get("reference_model"),
                "mesh_obj": str(obj_path),
                **projection_metrics,
                **mesh_basic_metrics(mesh),
                **mesh_artifact_metrics_from_obj(obj_path),
                **real_reference_metrics(mesh, sample, prior_coords, args, int(args.seed + sample_idx * 7919)),
            }
            if mode in stage2_info:
                metrics.update(stage2_info[mode])
            rows.append(metrics)
            write_json(mode_dir / "metrics.json", metrics)
            torch.cuda.empty_cache()

    report = {"args": vars(args), "rows": rows, "summary": summarize(rows)}
    write_json(output_dir / "report.json", report)
    write_csv(output_dir / "report.csv", rows)
    print(f"[real_slam_mesh] wrote {output_dir / 'report.json'}", flush=True)


def summarize(rows: list[dict]) -> dict:
    out: dict[str, Any] = {"count": len(rows), "by_mode": {}}
    numeric_keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float))})
    for mode in sorted({row["mode"] for row in rows}):
        rr = [row for row in rows if row["mode"] == mode]
        out["by_mode"][mode] = {"count": len(rr)}
        for key in numeric_keys:
            vals = [float(r[key]) for r in rr if key in r and isinstance(r[key], (int, float))]
            if vals:
                out["by_mode"][mode][f"{key}_mean"] = float(np.mean(vals))
                out["by_mode"][mode][f"{key}_median"] = float(np.median(vals))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRELLIS Stage2 point-prior mesh inference on real CoarseModel/SLAM prior manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--modes", default="stage2_correct")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--cond_mode", choices=["first", "mean", "multi_stochastic"], default="multi_stochastic")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--ss_guidance_strength", type=float, default=7.5)
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--slat_guidance_strength", type=float, default=7.5)
    parser.add_argument("--slat_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)
    parser.add_argument("--max_coords", type=int, default=0)
    parser.add_argument("--mesh_eval_samples", type=int, default=6000)
    parser.add_argument("--transform_mesh_pose", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--stage2_topk", default="12000")
    parser.add_argument("--stage2_topk_specs", default=None)
    parser.add_argument("--target_unique_estimate", type=int, default=16000)
    parser.add_argument("--absolute_topk_reference", type=int, default=16000)
    parser.add_argument("--known_latent_clamp_strength", type=float, default=1.0)
    parser.add_argument("--known_clamp_start_t", type=float, default=0.5)
    parser.add_argument("--known_logit_boost", type=float, default=0.0)
    parser.add_argument("--known_conf_power", type=float, default=1.0)
    parser.add_argument("--known_use_confidence", action="store_true")
    parser.add_argument("--clamp_initial_noise", dest="clamp_initial_noise", action="store_true", default=True)
    parser.add_argument("--no_clamp_initial_noise", dest="clamp_initial_noise", action="store_false")
    parser.add_argument("--guidance_strength", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--latent_channels", type=int, default=8)
    parser.add_argument("--latent_grid_resolution", type=int, default=16)
    parser.add_argument("--cond_channels", type=int, default=1024)
    parser.add_argument(
        "--stage2_base_guidance",
        default="none",
        help="Optional topology prior for Stage2 top-k selection. Use stock_sparse to restrict Stage2 ranking near stock/base sparse coords.",
    )
    parser.add_argument("--stage2_base_radius", type=float, default=3.0)
    parser.add_argument("--stage2_base_min_candidates", type=int, default=512)
    parser.add_argument(
        "--stage2_union_stock",
        action="store_true",
        help="Union every Stage2 sparse output with stock/base sparse before frozen slat. Equivalent to using mode stage2_union_stock.",
    )
    parser.add_argument(
        "--stage2_sparse_filter",
        default="none",
        help=(
            "Comma-separated eval-time filter steps for Stage2 sparse coords before slat. "
            "Supported: none, largest_component, min_component_size, prior_radius, projection_support."
        ),
    )
    parser.add_argument("--filter_prior_radius", type=float, default=4.0)
    parser.add_argument("--filter_min_component_size", type=int, default=64)
    parser.add_argument("--filter_min_support_views", type=int, default=1)
    parser.add_argument("--filter_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--visual_hull_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--filter_grid_resolution", type=int, default=64)
    parser.add_argument("--filter_mask_threshold", type=int, default=127)
    parser.add_argument("--filter_min_coords", type=int, default=128)
    parser.add_argument("--filter_fallback_unfiltered", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
