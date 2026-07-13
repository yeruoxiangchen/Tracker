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

from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

from trellis_point_prior_mv.common import load_manifest, parse_indices, resolve_path, sparse_overlap_metrics, write_json  # noqa: E402
from trellis_point_prior_mv.eval_sparse_inpaint import topk_coords_from_logits  # noqa: E402
from trellis_point_prior_mv.eval_sparse_vae_sanity import coords4, threshold_coords_from_logits  # noqa: E402
from trellis_point_prior_mv.sparse_coord_tools import component_metrics, nearest_prior_metrics  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_int_choices(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError(f"empty integer choices: {text!r}")
    return out


def parse_decode_specs(text: str) -> list[str]:
    specs: list[str] = []
    seen: set[str] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        lowered = part.lower()
        if lowered in {"target", "target_unique", "gt", "gt_unique"}:
            spec = "target_unique"
        else:
            try:
                value = int(part)
            except ValueError as exc:
                raise ValueError(
                    f"invalid decode top-k spec {part!r}; use positive integers or target_unique"
                ) from exc
            if value <= 0:
                raise ValueError(f"decode top-k must be positive, got {value}")
            spec = str(value)
        if spec not in seen:
            specs.append(spec)
            seen.add(spec)
    if not specs:
        raise ValueError(f"empty decode specs: {text!r}")
    return specs


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    mask_bool = np.asarray(mask, dtype=bool)
    if radius <= 0 or not mask_bool.any():
        return mask_bool
    try:
        from scipy.ndimage import binary_dilation

        structure = np.ones((3, 3, 3), dtype=bool)
        return binary_dilation(mask_bool, structure=structure, iterations=radius)
    except Exception:
        out = mask_bool.copy()
        for _ in range(radius):
            padded = np.pad(out, 1, mode="constant", constant_values=False)
            grown = np.zeros_like(out)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        grown |= padded[1 + dx : 1 + dx + out.shape[0], 1 + dy : 1 + dy + out.shape[1], 1 + dz : 1 + dz + out.shape[2]]
            out = grown
        return out


def latent_mask_from_prior(
    prior_coords: np.ndarray,
    *,
    mask_dilate64: int,
    mask_dilate16: int,
    source_resolution: int = 64,
    latent_resolution: int = 16,
) -> np.ndarray:
    xyz = coords4(prior_coords)[:, -3:]
    mask64 = np.zeros((source_resolution, source_resolution, source_resolution), dtype=bool)
    if xyz.size:
        mask64[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = True
    mask64 = dilate_mask(mask64, int(mask_dilate64))
    mask16 = np.zeros((latent_resolution, latent_resolution, latent_resolution), dtype=bool)
    coords64 = np.argwhere(mask64)
    if coords64.size:
        scale = max(1, int(source_resolution) // int(latent_resolution))
        latent = np.floor_divide(coords64, scale).clip(0, latent_resolution - 1)
        mask16[latent[:, 0], latent[:, 1], latent[:, 2]] = True
    mask16 = dilate_mask(mask16, int(mask_dilate16))
    return mask16.astype(np.float32)[None, :, :, :]


def normalize_latent_mask(mask: np.ndarray, *, latent_resolution: int) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[0] == 1:
        pass
    elif arr.ndim == 3:
        arr = arr[None, :, :, :]
    else:
        raise ValueError(f"expected latent mask shape (1,D,H,W), (D,H,W), or (1,1,D,H,W), got {arr.shape}")
    expected = (1, int(latent_resolution), int(latent_resolution), int(latent_resolution))
    if tuple(arr.shape) != expected:
        raise ValueError(f"latent mask shape mismatch: got {arr.shape}, expected {expected}")
    return arr.astype(np.float32, copy=False)


def load_decoder(args: argparse.Namespace, device: torch.device):
    print(f"[latent_splice_sanity] loading pipeline weights={args.weights}", flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for p in decoder.parameters():
        p.requires_grad = False
    return decoder


@torch.no_grad()
def decode_latent(decoder, latent_np: np.ndarray, device: torch.device) -> torch.Tensor:
    latent = torch.from_numpy(np.asarray(latent_np, dtype=np.float32)).to(device=device, dtype=next(decoder.parameters()).dtype)
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    logits = decoder(latent).float()
    if logits.shape[1] != 1:
        logits = logits.max(dim=1, keepdim=True).values
    return logits


def prefixed_overlap(prefix: str, pred: np.ndarray, ref: np.ndarray) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in sparse_overlap_metrics(pred, ref).items()}


def splice_stats(q_splice: np.ndarray, q_gt: np.ndarray, q_vis: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    known = mask.astype(bool)
    unknown = ~known
    known_b = np.broadcast_to(known, q_gt.shape)
    unknown_b = np.broadcast_to(unknown, q_gt.shape)
    out: dict[str, float | int] = {
        "mask_cell_count": int(mask.sum()),
        "mask_cell_ratio": float(mask.mean()),
        "q_splice_mean": float(q_splice.mean()),
        "q_splice_std": float(q_splice.std()),
        "q_splice_abs_mean": float(np.abs(q_splice).mean()),
        "q_splice_vs_q_gt_l1": float(np.abs(q_splice - q_gt).mean()),
        "q_vis_vs_q_gt_l1": float(np.abs(q_vis - q_gt).mean()),
    }
    out["q_splice_known_vs_q_vis_l1"] = float(np.abs(q_splice[known_b] - q_vis[known_b]).mean()) if known_b.any() else 0.0
    out["q_splice_unknown_vs_q_gt_l1"] = float(np.abs(q_splice[unknown_b] - q_gt[unknown_b]).mean()) if unknown_b.any() else 0.0
    return out


def add_decode_rows(
    *,
    rows: list[dict[str, Any]],
    base_row: dict[str, Any],
    logits: torch.Tensor,
    source_dir: Path,
    target_coords: np.ndarray,
    prior_coords: np.ndarray,
    input_coords: np.ndarray,
    topk_specs: list[str],
    threshold: float,
    target_unique: int,
    prior_radius: float,
) -> None:
    thresh_coords = threshold_coords_from_logits(logits, threshold)
    np.savez_compressed(source_dir / f"decode_threshold_{threshold:g}.npz", coords=thresh_coords)
    rows.append(
        {
            **base_row,
            "decode": f"threshold_{threshold:g}",
            "decoded_unique": int(coords4(thresh_coords).shape[0]),
            **component_metrics("decoded", thresh_coords),
            **nearest_prior_metrics("decoded_to_prior", thresh_coords, prior_coords, radius=float(prior_radius)),
            **prefixed_overlap("decoded_vs_input", thresh_coords, input_coords),
            **prefixed_overlap("decoded_vs_target", thresh_coords, target_coords),
        }
    )
    for spec in topk_specs:
        topk = int(target_unique if spec == "target_unique" else int(spec))
        pred = topk_coords_from_logits(logits, topk)
        np.savez_compressed(source_dir / f"decode_topk_{spec}.npz", coords=pred)
        rows.append(
            {
                **base_row,
                "decode": f"topk_{spec}",
                "topk": int(topk),
                "decoded_unique": int(coords4(pred).shape[0]),
                **component_metrics("decoded", pred),
                **nearest_prior_metrics("decoded_to_prior", pred, prior_coords, radius=float(prior_radius)),
                **prefixed_overlap("decoded_vs_input", pred, input_coords),
                **prefixed_overlap("decoded_vs_target", pred, target_coords),
            }
        )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows), "by_source_decode": {}}
    numeric = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    keys = sorted({(row.get("source"), row.get("decode")) for row in rows})
    for source, decode in keys:
        rr = [row for row in rows if row.get("source") == source and row.get("decode") == decode]
        name = f"{source}/{decode}"
        out["by_source_decode"][name] = {"count": len(rr)}
        for key in numeric:
            vals = [float(r[key]) for r in rr if isinstance(r.get(key), (int, float))]
            if vals:
                out["by_source_decode"][name][f"{key}_mean"] = float(np.mean(vals))
                out["by_source_decode"][name][f"{key}_median"] = float(np.median(vals))
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    payload, samples = load_manifest(args.manifest)
    indices = parse_indices(args.indices, len(samples))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder = load_decoder(args, device)
    topk_specs = parse_decode_specs(args.topk)
    dilate64_choices = parse_int_choices(args.mask_dilate64)
    dilate16_choices = parse_int_choices(args.mask_dilate16)

    rows: list[dict[str, Any]] = []
    for order, sample_idx in enumerate(indices):
        sample = samples[sample_idx]
        uid = str(sample.get("uid", sample_idx))
        latent_path = resolve_path(payload.get("latent_root"), sample["latent_npz"])
        with np.load(latent_path) as data:
            q_gt = np.asarray(data["q_gt"], dtype=np.float32)
            q_vis = np.asarray(data["q_vis"], dtype=np.float32)
            saved_m_s = normalize_latent_mask(
                np.asarray(data["m_s"], dtype=np.float32),
                latent_resolution=int(args.latent_grid_resolution),
            )
            prior_coords = np.asarray(data["prior_coords"], dtype=np.int32)
            target_coords = np.asarray(data["target_coords"], dtype=np.int32)
        target_unique = int(coords4(target_coords).shape[0])
        sample_dir = output_dir / f"{sample_idx:04d}_{uid[:12]}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[latent_splice_sanity] sample={sample_idx} uid={uid} "
            f"prior={coords4(prior_coords).shape[0]} target={target_unique}",
            flush=True,
        )

        baselines = {
            "q_gt": (q_gt, target_coords),
            "q_vis": (q_vis, prior_coords),
        }
        for source, (latent, input_coords) in baselines.items():
            source_dir = sample_dir / source
            source_dir.mkdir(parents=True, exist_ok=True)
            logits = decode_latent(decoder, latent, device)
            base_row = {
                "sample_order": int(order),
                "sample_index": int(sample_idx),
                "uid": uid,
                "source": source,
                "mask_dilate64": -1,
                "mask_dilate16": -1,
                "prior_unique": int(coords4(prior_coords).shape[0]),
                "target_unique": target_unique,
                "input_unique": int(coords4(input_coords).shape[0]),
                **component_metrics("input", input_coords),
                **prefixed_overlap("input_vs_target", input_coords, target_coords),
            }
            add_decode_rows(
                rows=rows,
                base_row=base_row,
                logits=logits,
                source_dir=source_dir,
                target_coords=target_coords,
                prior_coords=prior_coords,
                input_coords=input_coords,
                topk_specs=topk_specs,
                threshold=float(args.threshold),
                target_unique=target_unique,
                prior_radius=float(args.prior_radius),
            )

        for d64 in dilate64_choices:
            for d16 in dilate16_choices:
                mask = latent_mask_from_prior(
                    prior_coords,
                    mask_dilate64=int(d64),
                    mask_dilate16=int(d16),
                    source_resolution=int(args.source_grid_resolution),
                    latent_resolution=int(args.latent_grid_resolution),
                )
                saved_mask_l1 = float(np.abs(saved_m_s - mask).mean())
                q_splice = mask * q_vis + (1.0 - mask) * q_gt
                source = f"q_splice_d64_{d64}_d16_{d16}"
                source_dir = sample_dir / source
                source_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(source_dir / "q_splice.npz", q_splice=q_splice.astype(np.float32), mask=mask.astype(np.float32))
                logits = decode_latent(decoder, q_splice, device)
                base_row = {
                    "sample_order": int(order),
                    "sample_index": int(sample_idx),
                    "uid": uid,
                    "source": source,
                    "mask_dilate64": int(d64),
                    "mask_dilate16": int(d16),
                    "prior_unique": int(coords4(prior_coords).shape[0]),
                    "target_unique": target_unique,
                    "input_unique": target_unique,
                    "saved_m_s_vs_recomputed_mask_l1": saved_mask_l1,
                    "saved_m_s_cell_count": int(saved_m_s.sum()),
                    **splice_stats(q_splice, q_gt, q_vis, mask),
                }
                add_decode_rows(
                    rows=rows,
                    base_row=base_row,
                    logits=logits,
                    source_dir=source_dir,
                    target_coords=target_coords,
                    prior_coords=prior_coords,
                    input_coords=target_coords,
                    topk_specs=topk_specs,
                    threshold=float(args.threshold),
                    target_unique=target_unique,
                    prior_radius=float(args.prior_radius),
                )

    report = {"args": vars(args), "rows": rows, "summary": summarize(rows)}
    write_json(output_dir / "report.json", report)
    write_csv(output_dir / "report.csv", rows)
    print(f"[latent_splice_sanity] wrote {output_dir / 'report.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent splice sanity: decode m_s*q_vis + (1-m_s)*q_gt before training.")
    parser.add_argument("--manifest", required=True, help="Latent inpaint manifest from build_latent_inpaint_dataset.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="0")
    parser.add_argument("--mask_dilate64", default="0,1,2")
    parser.add_argument("--mask_dilate16", default="0,1")
    parser.add_argument("--topk", default="4096,8192,target_unique")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--source_grid_resolution", type=int, default=64)
    parser.add_argument("--latent_grid_resolution", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
