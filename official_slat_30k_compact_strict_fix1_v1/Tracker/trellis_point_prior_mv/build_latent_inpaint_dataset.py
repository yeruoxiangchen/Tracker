#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from trellis import models as trellis_models  # noqa: E402

from trellis_point_prior_mv.common import (  # noqa: E402
    coords_to_batched_occ,
    load_manifest,
    load_target_latent,
    parse_indices,
    resolve_path,
    write_json,
)
from trellis_point_prior_mv.eval_sparse_vae_sanity import coords4  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_prior(sample: dict[str, Any], payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    prior_path = resolve_path(payload.get("prior_root"), sample["prior_npz"])
    with np.load(prior_path) as data:
        coords = np.asarray(data["prior_coords"], dtype=np.int32)
        conf = np.asarray(data["prior_conf"], dtype=np.float32) if "prior_conf" in data else np.ones((coords.shape[0],), dtype=np.float32)
    return coords[:, -3:].astype(np.int32, copy=False), conf.reshape(-1)


def latent_mask_from_coords(coords_np: np.ndarray, *, latent_resolution: int = 16, source_resolution: int = 64) -> np.ndarray:
    xyz = coords4(coords_np)[:, -3:]
    mask = np.zeros((1, latent_resolution, latent_resolution, latent_resolution), dtype=np.float32)
    if xyz.size == 0:
        return mask
    scale = max(1, int(source_resolution) // int(latent_resolution))
    latent = np.floor_divide(xyz, scale).clip(0, latent_resolution - 1)
    mask[0, latent[:, 0], latent[:, 1], latent[:, 2]] = 1.0
    return mask


def array_stats(prefix: str, arr: np.ndarray) -> dict[str, float | int | str]:
    arr = np.asarray(arr, dtype=np.float32)
    return {
        f"{prefix}_shape": "x".join(str(x) for x in arr.shape),
        f"{prefix}_mean": float(arr.mean()) if arr.size else 0.0,
        f"{prefix}_std": float(arr.std()) if arr.size else 0.0,
        f"{prefix}_abs_mean": float(np.abs(arr).mean()) if arr.size else 0.0,
        f"{prefix}_abs_max": float(np.abs(arr).max()) if arr.size else 0.0,
    }


def load_encoder(args: argparse.Namespace, device: torch.device):
    print(f"[build_latent_inpaint] loading ss encoder weights={args.weights}", flush=True)
    encoder = trellis_models.from_pretrained(
        f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
        if os.path.isdir(args.weights)
        else f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
    ).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def encode_prior_latent(encoder, prior_coords: np.ndarray, device: torch.device) -> np.ndarray:
    coords = torch.from_numpy(coords4(prior_coords)).to(device=device, dtype=torch.long)
    occ = coords_to_batched_occ(
        coords,
        1,
        resolution=64,
        device=device,
        dtype=next(encoder.parameters()).dtype,
    )
    latent = encoder(occ, sample_posterior=False).to(torch.float32)
    return latent[0].detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    payload, samples = load_manifest(args.manifest)
    indices = parse_indices(args.indices, len(samples))
    output_dir = Path(args.output_dir)
    latent_dir = output_dir / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(args, device)

    out_samples: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for out_idx, sample_idx in enumerate(indices):
        sample = samples[sample_idx]
        uid = str(sample.get("uid", sample_idx))
        q_gt, target_coords = load_target_latent(sample["ss_latent"])
        prior_coords, prior_conf = load_prior(sample, payload)
        q_vis = encode_prior_latent(encoder, prior_coords, device)
        m_s = latent_mask_from_coords(
            prior_coords,
            latent_resolution=int(args.latent_grid_resolution),
            source_resolution=int(args.source_grid_resolution),
        )

        shard = uid[:2] if len(uid) >= 2 else "00"
        latent_path = latent_dir / shard / f"{uid}.npz"
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            latent_path,
            q_gt=q_gt.astype(np.float32),
            q_vis=q_vis.astype(np.float32),
            m_s=m_s.astype(np.float32),
            prior_coords=prior_coords.astype(np.int32),
            prior_conf=prior_conf.astype(np.float32),
            target_coords=target_coords.astype(np.int32),
        )

        row = {
            "out_index": int(out_idx),
            "source_index": int(sample_idx),
            "uid": uid,
            "latent_npz": str(latent_path.relative_to(output_dir)),
            "prior_point_count": int(coords4(prior_coords).shape[0]),
            "target_point_count": int(coords4(target_coords).shape[0]),
            "mask_cell_count": int(m_s.sum()),
            "mask_cell_ratio": float(m_s.mean()),
            **array_stats("q_gt", q_gt),
            **array_stats("q_vis", q_vis),
        }
        rows.append(row)
        out_samples.append(
            {
                "uid": uid,
                "source_index": int(sample_idx),
                "source_manifest": sample.get("source_manifest", payload.get("source_manifest", str(args.manifest))),
                "source_ss_latent": sample["ss_latent"],
                "source_prior_npz": sample["prior_npz"],
                "latent_npz": str(latent_path.relative_to(output_dir)),
                "prior_point_count": row["prior_point_count"],
                "target_point_count": row["target_point_count"],
                "mask_cell_count": row["mask_cell_count"],
            }
        )
        if (out_idx + 1) % max(1, int(args.log_every)) == 0:
            print(
                f"[build_latent_inpaint] {out_idx + 1}/{len(indices)} uid={uid} "
                f"prior={row['prior_point_count']} target={row['target_point_count']} mask={row['mask_cell_count']}",
                flush=True,
            )

    manifest = {
        "format": "trellis_point_prior_latent_inpaint_v1",
        "source_manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "latent_root": str(output_dir),
        "weights": args.weights,
        "source_grid_resolution": int(args.source_grid_resolution),
        "latent_grid_resolution": int(args.latent_grid_resolution),
        "samples": out_samples,
        "build_args": vars(args),
        "summary": {
            "num_samples": len(out_samples),
            "prior_point_mean": float(np.mean([r["prior_point_count"] for r in rows])) if rows else 0.0,
            "target_point_mean": float(np.mean([r["target_point_count"] for r in rows])) if rows else 0.0,
            "mask_cell_mean": float(np.mean([r["mask_cell_count"] for r in rows])) if rows else 0.0,
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_csv(output_dir / "build_report.csv", rows)
    write_json(output_dir / "build_report.json", {"rows": rows, "summary": manifest["summary"]})
    print(f"[build_latent_inpaint] wrote {output_dir / 'manifest.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build q_vis/q_gt latent dataset for point-prior sparse latent inpainting.")
    parser.add_argument("--manifest", required=True, help="Existing point-prior manifest.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--source_grid_resolution", type=int, default=64)
    parser.add_argument("--latent_grid_resolution", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
