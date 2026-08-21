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
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

from trellis_point_prior_mv.common import (  # noqa: E402
    coords_to_batched_occ,
    load_manifest,
    load_target_latent,
    parse_indices,
    resolve_path,
    sparse_overlap_metrics,
    write_json,
)
from trellis_point_prior_mv.eval_sparse_inpaint import parse_topk_specs, topk_coords_from_logits  # noqa: E402
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


def coords4(coords: np.ndarray) -> np.ndarray:
    xyz = coords[:, -3:].astype(np.int32, copy=False) if coords.size else np.zeros((0, 3), dtype=np.int32)
    if xyz.size:
        valid = ((xyz >= 0) & (xyz < 64)).all(axis=1)
        xyz = xyz[valid]
        if xyz.shape[0] > 1:
            xyz = np.unique(xyz, axis=0)
    return np.concatenate([np.zeros((xyz.shape[0], 1), dtype=np.int32), xyz], axis=1)


def threshold_coords_from_logits(logits: torch.Tensor, threshold: float) -> np.ndarray:
    if logits.ndim != 5 or logits.shape[0] != 1:
        raise ValueError(f"expected logits [1,C,D,H,W], got {tuple(logits.shape)}")
    if logits.shape[1] != 1:
        logits = logits.max(dim=1, keepdim=True).values
    idx = torch.argwhere(logits > float(threshold))
    if idx.numel() == 0:
        return np.zeros((0, 4), dtype=np.int32)
    return idx[:, [0, 2, 3, 4]].detach().cpu().numpy().astype(np.int32)


def load_prior(sample: dict[str, Any], payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    prior_path = resolve_path(payload.get("prior_root"), sample["prior_npz"])
    with np.load(prior_path) as data:
        coords = np.asarray(data["prior_coords"], dtype=np.int32)
        conf = np.asarray(data["prior_conf"], dtype=np.float32) if "prior_conf" in data else np.ones((coords.shape[0],), dtype=np.float32)
    return coords[:, -3:].astype(np.int32, copy=False), conf.reshape(-1)


def load_models(args: argparse.Namespace, device: torch.device):
    print(f"[sparse_vae_sanity] loading pipeline weights={args.weights}", flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for p in decoder.parameters():
        p.requires_grad = False
    encoder = trellis_models.from_pretrained(
        f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
        if os.path.isdir(args.weights)
        else f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
    ).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, decoder


@torch.no_grad()
def encode_decode_coords(
    encoder,
    decoder,
    coords_np: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.from_numpy(coords4(coords_np)).to(device=device, dtype=torch.long)
    occ = coords_to_batched_occ(
        coords,
        1,
        resolution=64,
        device=device,
        dtype=next(encoder.parameters()).dtype,
    )
    latent = encoder(occ, sample_posterior=False).to(torch.float32)
    logits = decoder(latent.to(dtype=next(decoder.parameters()).dtype)).float()
    if logits.shape[1] != 1:
        logits = logits.max(dim=1, keepdim=True).values
    return latent, logits


@torch.no_grad()
def decode_latent(decoder, latent_np: np.ndarray, device: torch.device) -> torch.Tensor:
    latent = torch.from_numpy(latent_np).to(device=device, dtype=next(decoder.parameters()).dtype)
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    if latent.ndim != 5:
        raise ValueError(f"expected latent [C,D,H,W] or [1,C,D,H,W], got {tuple(latent.shape)}")
    logits = decoder(latent).float()
    if logits.shape[1] != 1:
        logits = logits.max(dim=1, keepdim=True).values
    return logits


def latent_stats(prefix: str, latent: torch.Tensor | np.ndarray) -> dict[str, float | int]:
    arr = latent.detach().float().cpu().numpy() if torch.is_tensor(latent) else np.asarray(latent, dtype=np.float32)
    return {
        f"{prefix}_latent_shape": "x".join(str(x) for x in arr.shape),
        f"{prefix}_latent_mean": float(arr.mean()) if arr.size else 0.0,
        f"{prefix}_latent_std": float(arr.std()) if arr.size else 0.0,
        f"{prefix}_latent_abs_mean": float(np.abs(arr).mean()) if arr.size else 0.0,
        f"{prefix}_latent_abs_max": float(np.abs(arr).max()) if arr.size else 0.0,
    }


def logits_stats(prefix: str, logits: torch.Tensor) -> dict[str, float | int]:
    arr = logits.detach().float().cpu().numpy()
    return {
        f"{prefix}_logits_min": float(arr.min()),
        f"{prefix}_logits_max": float(arr.max()),
        f"{prefix}_logits_mean": float(arr.mean()),
        f"{prefix}_logits_std": float(arr.std()),
        f"{prefix}_logits_pos_count": int((arr > 0).sum()),
    }


def prefixed_overlap(prefix: str, pred: np.ndarray, ref: np.ndarray) -> dict[str, float | int]:
    return {f"{prefix}_{k}": v for k, v in sparse_overlap_metrics(pred, ref).items()}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows), "by_source_decode": {}}
    numeric = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float))})
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload, samples = load_manifest(args.manifest)
    indices = parse_indices(args.indices, len(samples))
    topk_specs = parse_topk_specs(args.topk)
    encoder, decoder = load_models(args, device)

    rows: list[dict[str, Any]] = []
    for order, sample_idx in enumerate(indices):
        sample = samples[sample_idx]
        uid = str(sample.get("uid", sample_idx))
        sample_dir = output_dir / f"{sample_idx:04d}_{uid[:12]}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        target_latent, target_coords = load_target_latent(sample["ss_latent"])
        prior_coords, prior_conf = load_prior(sample, payload)
        target_unique = len({tuple(x) for x in coords4(target_coords)[:, -3:].tolist()})

        sources: list[tuple[str, np.ndarray | None, np.ndarray | None, torch.Tensor]] = []
        prior_latent, prior_logits = encode_decode_coords(encoder, decoder, prior_coords, device)
        sources.append(("prior_coords_vae", prior_coords, prior_latent.detach().cpu().numpy(), prior_logits))

        target_occ_latent, target_occ_logits = encode_decode_coords(encoder, decoder, target_coords, device)
        sources.append(("target_coords_vae", target_coords, target_occ_latent.detach().cpu().numpy(), target_occ_logits))

        target_latent_logits = decode_latent(decoder, target_latent, device)
        sources.append(("target_latent_npz", target_coords, target_latent, target_latent_logits))

        print(
            f"[sparse_vae_sanity] sample={sample_idx} uid={uid} "
            f"prior={prior_coords.shape[0]} target={target_unique}",
            flush=True,
        )

        for source_name, input_coords, latent_np, logits in sources:
            source_dir = sample_dir / source_name
            source_dir.mkdir(parents=True, exist_ok=True)
            base_row: dict[str, Any] = {
                "sample_order": int(order),
                "sample_index": int(sample_idx),
                "uid": uid,
                "source": source_name,
                "prior_input_unique": int(len({tuple(x) for x in coords4(prior_coords)[:, -3:].tolist()})),
                "target_unique": int(target_unique),
            }
            if input_coords is not None:
                base_row["input_unique"] = int(len({tuple(x) for x in coords4(input_coords)[:, -3:].tolist()}))
                base_row.update(component_metrics("input", coords4(input_coords)))
                base_row.update(prefixed_overlap("input_vs_target", coords4(input_coords), target_coords))
            if latent_np is not None:
                np.savez_compressed(source_dir / "latent.npz", latent=np.asarray(latent_np, dtype=np.float32))
                base_row.update(latent_stats(source_name, latent_np))
            base_row.update(logits_stats(source_name, logits))

            thresh_coords = threshold_coords_from_logits(logits, args.threshold)
            np.savez_compressed(source_dir / f"decode_threshold_{args.threshold:g}.npz", coords=thresh_coords)
            row = {
                **base_row,
                "decode": f"threshold_{args.threshold:g}",
                "decoded_unique": int(len({tuple(x) for x in thresh_coords[:, -3:].tolist()})),
                **component_metrics("decoded", thresh_coords),
                **nearest_prior_metrics("decoded_to_prior", thresh_coords, prior_coords, radius=float(args.prior_radius)),
                **prefixed_overlap("decoded_vs_input", thresh_coords, input_coords if input_coords is not None else target_coords),
                **prefixed_overlap("decoded_vs_target", thresh_coords, target_coords),
            }
            rows.append(row)

            for spec in topk_specs:
                topk = target_unique if spec == "target_unique" else int(spec)
                pred = topk_coords_from_logits(logits, topk)
                label = f"topk_{spec}"
                np.savez_compressed(source_dir / f"decode_{label}.npz", coords=pred)
                row = {
                    **base_row,
                    "decode": label,
                    "topk": int(topk),
                    "decoded_unique": int(len({tuple(x) for x in pred[:, -3:].tolist()})),
                    **component_metrics("decoded", pred),
                    **nearest_prior_metrics("decoded_to_prior", pred, prior_coords, radius=float(args.prior_radius)),
                    **prefixed_overlap("decoded_vs_input", pred, input_coords if input_coords is not None else target_coords),
                    **prefixed_overlap("decoded_vs_target", pred, target_coords),
                }
                rows.append(row)

    report = {
        "args": vars(args),
        "rows": rows,
        "summary": summarize(rows),
    }
    write_json(output_dir / "report.json", report)
    write_csv(output_dir / "report.csv", rows)
    print(f"[sparse_vae_sanity] wrote {output_dir / 'report.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode/decode sanity check for TRELLIS sparse-structure VAE on point priors.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="0")
    parser.add_argument("--topk", default="4096,8192,target_unique")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
