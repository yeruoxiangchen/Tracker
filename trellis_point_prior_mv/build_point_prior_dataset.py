#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from trellis_point_prior_mv.common import (  # noqa: E402
    load_manifest,
    load_sample_frames,
    load_target_latent,
    make_slam_like_prior,
    parse_indices,
    parse_int_list,
    resolve_path,
    write_json,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> None:
    source_manifest = Path(args.source_manifest)
    payload, samples = load_manifest(source_manifest)
    indices = parse_indices(args.indices, len(samples))
    output_dir = Path(args.output_dir)
    prior_dir = output_dir / "priors"
    prior_dir.mkdir(parents=True, exist_ok=True)

    latent_root = payload.get("latent_root")
    extrinsics_type = args.extrinsics_type or payload.get("extrinsics_type", "c2w")
    camera_forward_sign = float(args.camera_forward_sign if args.camera_forward_sign is not None else payload.get("camera_forward_sign", 1.0))
    point_count_choices = parse_int_list(args.point_count_choices)
    view_count_choices = parse_int_list(args.num_prior_views_choices)

    rows = []
    out_samples = []
    for out_idx, sample_idx in enumerate(indices):
        sample = samples[sample_idx]
        uid = str(sample.get("uid", sample.get("id", sample_idx)))
        latent_rel = sample.get("ss_latent", sample.get("ss_latent_path", sample.get("latent")))
        if latent_rel is None:
            raise ValueError(f"sample {uid} has no ss_latent")
        latent_path = resolve_path(latent_root, latent_rel)
        _z, target_coords = load_target_latent(latent_path)
        intrinsics, extrinsics, mask_paths = load_sample_frames(payload, sample, max_frames=args.max_frames)
        rng = np.random.default_rng(int(args.seed) + int(sample_idx) * 1009)
        prior = make_slam_like_prior(
            target_coords,
            intrinsics,
            extrinsics,
            mask_paths,
            rng=rng,
            grid_transform=args.grid_transform,
            extrinsics_type=extrinsics_type,
            camera_forward_sign=camera_forward_sign,
            num_prior_views_choices=view_count_choices,
            point_count_choices=point_count_choices,
            min_support=args.min_support,
            min_support_ratio=args.min_support_ratio,
            dropout_min=args.dropout_min,
            dropout_max=args.dropout_max,
            coord_jitter=args.coord_jitter,
            outlier_ratio=args.outlier_ratio,
            front_depth=not args.no_front_depth,
            front_depth_epsilon=args.front_depth_epsilon,
            allow_support_fallback=args.allow_support_fallback,
        )
        shard = uid[:2] if len(uid) >= 2 else "00"
        prior_path = prior_dir / shard / f"{uid}.npz"
        prior_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prior_path,
            prior_coords=prior["prior_coords"],
            prior_conf=prior["prior_conf"],
            target_coords=target_coords.astype(np.int32),
            view_ids=prior["view_ids"],
        )
        rel_prior = str(prior_path.relative_to(output_dir))
        rel_latent = str(latent_path)
        out_samples.append(
            {
                "uid": uid,
                "source_index": int(sample_idx),
                "source_manifest": str(source_manifest),
                "ss_latent": rel_latent,
                "prior_npz": rel_prior,
                "num_voxels": int(target_coords.shape[0]),
                "prior_point_count": int(prior["actual_point_count"]),
                "view_count": int(prior["view_count"]),
                "grid_transform": args.grid_transform,
            }
        )
        row = {
            "out_index": out_idx,
            "source_index": sample_idx,
            "uid": uid,
            "target_count": int(target_coords.shape[0]),
            "surface_count": int(prior["surface_count"]),
            "supported_surface_count": int(prior["supported_surface_count"]),
            "sampling_pool_count": int(prior["sampling_pool_count"]),
            "prior_point_count": int(prior["actual_point_count"]),
            "view_count": int(prior["view_count"]),
            "dropout": float(prior["dropout"]),
            "support_failed": int(bool(prior["support_failed"])),
            "fallback_used": int(bool(prior["fallback_used"])),
            "support_visible_mean": float(prior["support_visible_mean"]),
            "support_supported_ratio": float(prior["support_supported_ratio"]),
        }
        rows.append(row)
        if (out_idx + 1) % max(1, args.log_every) == 0:
            print(
                f"[build_point_prior] {out_idx + 1}/{len(indices)} uid={uid} "
                f"target={row['target_count']} prior={row['prior_point_count']} "
                f"views={row['view_count']}",
                flush=True,
            )

    manifest = {
        "format": "trellis_point_prior_mv_v1",
        "source_manifest": str(source_manifest),
        "output_dir": str(output_dir),
        "latent_root": None,
        "prior_root": str(output_dir),
        "extrinsics_type": extrinsics_type,
        "camera_forward_sign": camera_forward_sign,
        "grid_transform": args.grid_transform,
        "samples": out_samples,
        "build_args": vars(args),
        "summary": {
            "num_samples": len(out_samples),
            "prior_point_mean": float(np.mean([r["prior_point_count"] for r in rows])) if rows else 0.0,
            "target_count_mean": float(np.mean([r["target_count"] for r in rows])) if rows else 0.0,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_csv(output_dir / "build_report.csv", rows)
    write_json(output_dir / "build_report.json", {"rows": rows, "summary": manifest["summary"]})
    print(f"[build_point_prior] wrote {output_dir / 'manifest.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid_transform", choices=["identity", "pixal3d_rotation"], default="pixal3d_rotation")
    parser.add_argument("--extrinsics_type", choices=["c2w", "w2c"], default=None)
    parser.add_argument("--camera_forward_sign", type=float, default=None)
    parser.add_argument("--num_prior_views_choices", default="1,2,4,8")
    parser.add_argument("--point_count_choices", default="50,100,300,800,1500")
    parser.add_argument("--min_support", type=float, default=1.0)
    parser.add_argument("--min_support_ratio", type=float, default=0.45)
    parser.add_argument("--dropout_min", type=float, default=0.0)
    parser.add_argument("--dropout_max", type=float, default=0.65)
    parser.add_argument("--coord_jitter", type=int, default=1)
    parser.add_argument("--outlier_ratio", type=float, default=0.03)
    parser.add_argument("--no_front_depth", action="store_true", help="Disable z-buffer front-surface visibility filtering.")
    parser.add_argument("--front_depth_epsilon", type=float, default=0.02)
    parser.add_argument(
        "--allow_support_fallback",
        action="store_true",
        help="If support filtering finds no visible surface points, fall back to all surface coords. Default is disabled to expose failures.",
    )
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
