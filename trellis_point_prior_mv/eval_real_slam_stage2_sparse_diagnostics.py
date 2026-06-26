#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import re
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
    load_stage2_bundle,
    parse_stage2_topk_specs,
    resolve_stage2_topk,
    sample_stage2_logits,
    stage2_topk_label,
)
from trellis_point_prior_mv.eval_sparse_inpaint import topk_coords_from_logits  # noqa: E402
from trellis_point_prior_mv.sparse_coord_tools import (  # noqa: E402
    coords_xyz,
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


def filter_label(spec: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(spec).strip()).strip("_")
    return text or "none"


def summarize(rows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows), "by_filter_mode": {}}
    numeric_keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float))})
    for key in sorted({str(row["filter_spec"]) for row in rows}):
        rr = [row for row in rows if str(row["filter_spec"]) == key]
        out["by_filter_mode"][key] = {"count": len(rr)}
        for nk in numeric_keys:
            vals = [float(r[nk]) for r in rr if nk in r and isinstance(r[nk], (int, float))]
            if vals:
                out["by_filter_mode"][key][f"{nk}_mean"] = float(np.mean(vals))
                out["by_filter_mode"][key][f"{nk}_median"] = float(np.median(vals))
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(args.manifest)
    samples = payload["samples"]
    indices = parse_indices(args.indices, len(samples))
    topk_specs = parse_stage2_topk_specs(args)
    filter_specs = [x.strip() for x in str(args.filter_specs).split(";") if x.strip()]
    if not filter_specs:
        filter_specs = ["none"]

    print(f"[stage2_sparse_diag] loading pipeline weights={args.weights}", flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)
    pipeline.low_vram = False
    pipeline.models["sparse_structure_decoder"].to(device)
    stage2_bundle = load_stage2_bundle(args, pipeline, device)
    if stage2_bundle is None:
        raise ValueError("--stage2_checkpoint is required")

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
        seed = int(args.seed + sample_idx * 1009)
        logits = sample_stage2_logits(stage2_bundle, pipeline, prior_coords, prior_conf, args, device, seed)
        target_unique_estimate = int(args.target_unique_estimate or args.absolute_topk_reference or max(prior_coords.shape[0] * 8, 12000))

        for spec in topk_specs:
            topk, topk_info = resolve_stage2_topk(spec, target_unique_estimate)
            raw_coords = topk_coords_from_logits(logits, topk)
            raw_metrics = sparse_diagnostic_metrics(
                "raw",
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
            for filter_spec in filter_specs:
                if filter_spec.strip().lower() in {"none", "raw", ""}:
                    final_coords = raw_coords
                    filter_metrics: dict[str, Any] = {
                        "sparse_filter_spec": "none",
                        "sparse_filter_input_count": int(coords_xyz(raw_coords).shape[0]),
                        "sparse_filter_output_count": int(coords_xyz(raw_coords).shape[0]),
                        "sparse_filter_total_keep_ratio": 1.0,
                    }
                else:
                    final_coords, filter_metrics = filter_sparse_coords(
                        raw_coords,
                        prior_coords,
                        sample,
                        filter_spec=filter_spec,
                        prior_radius=float(args.filter_prior_radius),
                        min_component_size=int(args.filter_min_component_size),
                        min_support_views=int(args.filter_min_support_views),
                        min_support_ratio=float(args.filter_min_support_ratio),
                        visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
                        visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
                        grid_resolution=int(args.filter_grid_resolution),
                        mask_threshold=int(args.filter_mask_threshold),
                    )
                final_metrics = sparse_diagnostic_metrics(
                    "final",
                    final_coords,
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
                label = f"stage2_{stage2_topk_label(spec)}__{filter_label(filter_spec)}"
                mode_dir = sample_dir / label
                mode_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(mode_dir / "sparse_coords.npz", coords=coords_xyz(final_coords).astype(np.int32))
                row = {
                    "sample_order": int(order),
                    "sample_index": int(sample_idx),
                    "uid": uid,
                    "dataset_root": sample.get("dataset_root"),
                    "prior_point_count": int(prior_coords.shape[0]),
                    "topk_spec": str(spec),
                    "topk_label": stage2_topk_label(spec),
                    "topk_absolute": int(topk),
                    "filter_spec": str(filter_spec),
                    "filter_label": filter_label(filter_spec),
                    **topk_info,
                    **raw_metrics,
                    **filter_metrics,
                    **final_metrics,
                }
                rows.append(row)
                write_json(mode_dir / "metrics.json", row)
                print(
                    f"[stage2_sparse_diag] sample={sample_idx} uid={uid} topk={topk} "
                    f"filter={filter_spec} final={row['final_coord_count']} "
                    f"comp={row['final_component_count']} largest={row['final_largest_component_ratio']:.3f} "
                    f"support_keep={row['final_projection_keep_ratio']:.3f}",
                    flush=True,
                )
        torch.cuda.empty_cache()

    report = {"args": vars(args), "rows": rows, "summary": summarize(rows)}
    write_json(output_dir / "report.json", report)
    write_csv(output_dir / "report.csv", rows)
    print(f"[stage2_sparse_diag] wrote {output_dir / 'report.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse-level diagnostics for real/AR Stage2 point-prior outputs before slat.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--stage2_topk", default="4096")
    parser.add_argument("--stage2_topk_specs", default=None)
    parser.add_argument("--target_unique_estimate", type=int, default=16000)
    parser.add_argument("--absolute_topk_reference", type=int, default=16000)
    parser.add_argument("--filter_specs", default="none;largest_component;largest_component,prior_radius;largest_component,prior_radius,projection_support")
    parser.add_argument("--filter_prior_radius", type=float, default=4.0)
    parser.add_argument("--filter_min_component_size", type=int, default=64)
    parser.add_argument("--filter_min_support_views", type=int, default=1)
    parser.add_argument("--filter_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--visual_hull_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--filter_grid_resolution", type=int, default=64)
    parser.add_argument("--filter_mask_threshold", type=int, default=127)
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
    parser.add_argument("--max_coords", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
