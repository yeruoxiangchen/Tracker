#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import trellis.pipelines.trellis_image_to_3d as trellis_image_to_3d  # noqa: E402
from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from reconvggt_ar_adapter_a.eval_b4_delta_prior_alignment import _load_prior_manifest_sample, _set_compare, _xyz_set  # noqa: E402
from reconvggt_ar_adapter_a.inspect_and_sanity import DreamSimStub, force_eval, load_images, normalize_image_cond  # noqa: E402
from reconvggt_ar_adapter_a.projection_token_features import parse_ar_pose_file, select_pose_records, summarize_pose_features  # noqa: E402
from reconvggt_ar_adapter_a.run_b3_adapter_injection_smoke import _component_stats, _load_images_with_masks  # noqa: E402
from reconvggt_ar_adapter_a.run_b5_ss_cond_residual_smoke import build_ar_cond_residual, coords_np, delta_report, summarize_tensor  # noqa: E402
from reconvggt_ar_adapter_a.train_b5_ss_cond_residual_adapter import SSCondResidualAdapter, adapter_stats  # noqa: E402
from reconvggt_ar_adapter_a.train_b_projection_adapter import build_projection_features  # noqa: E402
from trellis_point_prior_mv.sparse_coord_tools import sparse_diagnostic_metrics  # noqa: E402


def install_dreamsim_stub() -> None:
    def _stub_dreamsim(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        return DreamSimStub().to(device), None

    trellis_image_to_3d.dreamsim = _stub_dreamsim


def parse_checkpoint_spec(spec: str) -> list[Path]:
    paths: list[Path] = []
    for part in str(spec).split(","):
        item = part.strip()
        if not item:
            continue
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted((p / "checkpoints").glob("adapter_step_*.ckpt")))
            last = p / "checkpoints" / "last.ckpt"
            if last.exists():
                paths.append(last)
        else:
            paths.append(p)
    deduped: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    if not deduped:
        raise ValueError(f"No checkpoints resolved from: {spec}")
    missing = [str(p) for p in deduped if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    return deduped


def checkpoint_label(path: Path) -> str:
    parent = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
    return f"{parent}__{path.stem}"


def load_adapter(adapter: SSCondResidualAdapter, checkpoint: Path) -> dict[str, Any]:
    state = torch.load(str(checkpoint), map_location="cpu")
    state_dict = state.get("state_dict", state)
    missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Adapter checkpoint mismatch for {checkpoint}: missing={missing}, unexpected={unexpected}")
    return {
        "path": str(checkpoint),
        "metadata": state.get("metadata"),
        "rows_tail": state.get("rows", [])[-3:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate B5 SS cond residual adapter checkpoints with one pipeline load.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint files or run dirs.")
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--max_views", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--mask_dir", default="")
    parser.add_argument("--mask_mode", choices=["none", "apply"], default="none")
    parser.add_argument("--mask_background", choices=["black", "white"], default="black")
    parser.add_argument("--points3d_txt", default="")
    parser.add_argument("--point_prior_npz", default="")
    parser.add_argument("--colmap_sparse_dir", default="")
    parser.add_argument("--mask_projection_mode", choices=["none", "filter_points", "token_mask", "filter_points_token_mask"], default="none")
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--token_mask_min_ratio", type=float, default=0.05)
    parser.add_argument("--point_mask_support_min_views", type=int, default=0)
    parser.add_argument("--point_mask_support_min_ratio", type=float, default=0.0)
    parser.add_argument("--patch_start_idx", type=int, default=5)
    parser.add_argument("--image_resolution", type=int, default=518)
    parser.add_argument("--token_grid_side", type=int, default=37)
    parser.add_argument("--point_projection_rotation_mode", choices=["c2w", "w2c"], default="c2w")
    parser.add_argument("--point_projection_min_depth", type=float, default=1.0e-4)
    parser.add_argument("--default_fx", type=float, default=485.845947)
    parser.add_argument("--default_fy", type=float, default=485.744232)
    parser.add_argument("--default_cx", type=float, default=322.973236)
    parser.add_argument("--default_cy", type=float, default=237.599487)
    parser.add_argument("--default_image_width", type=int, default=640)
    parser.add_argument("--default_image_height", type=int, default=480)
    parser.add_argument("--load_dreamsim", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--ss_cfg_strength", type=float, default=7.5)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--ss_rescale_t", type=float, default=3.0)
    parser.add_argument("--target_scale", type=float, default=-0.02)
    parser.add_argument("--residual_seed", type=int, default=20260707)
    parser.add_argument("--residual_token_mode", choices=["constant", "random", "ramp"], default="random")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--prior_manifest", default="")
    parser.add_argument("--prior_uid", default="")
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--projection_min_support_views", type=int, default=1)
    parser.add_argument("--projection_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--visual_hull_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--pass_adapter_target_iou", type=float, default=0.93)
    parser.add_argument("--pass_mask_delta", type=float, default=0.30)
    parser.add_argument("--pass_outside_delta", type=float, default=-0.15)
    parser.add_argument("--pass_max_component_count", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = parse_checkpoint_spec(args.checkpoints)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if not args.load_dreamsim:
        install_dreamsim_stub()

    print(f"[B5 sweep] loading pipeline pretrained={args.pretrained} device={device}", flush=True)
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    pipeline._device = device
    pipeline.low_vram = bool(args.low_vram)
    force_eval(pipeline)
    if hasattr(pipeline, "birefnet_model") and not pipeline.low_vram:
        pipeline.birefnet_model.to(device)
    if not pipeline.low_vram:
        for model in pipeline.models.values():
            model.to(device)
        pipeline.VGGT_model.to(device)
    force_eval(pipeline)

    if args.mask_mode == "apply":
        if not args.mask_dir:
            raise ValueError("--mask_dir is required when --mask_mode=apply")
        images, image_names, mask_summaries = _load_images_with_masks(
            Path(args.image_dir),
            mask_dir=Path(args.mask_dir),
            max_views=int(args.max_views),
            mask_background=args.mask_background,
        )
    else:
        images, image_names = load_images(Path(args.image_dir), max_views=int(args.max_views), preprocess=args.preprocess, pipeline=pipeline)
        mask_summaries = None

    pose_records_all = parse_ar_pose_file(
        args.pose_file,
        default_intrinsics=(args.default_fx, args.default_fy, args.default_cx, args.default_cy),
        default_image_size=(args.default_image_width, args.default_image_height),
    )
    pose_records = select_pose_records(image_names, pose_records_all)
    print(f"[B5 sweep] loaded {len(images)} images, {len(pose_records)} poses, {len(checkpoints)} checkpoints", flush=True)

    torch.manual_seed(int(args.seed))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=getattr(pipeline, "VGGT_dtype", torch.float16)):
        aggregated_tokens_list, _ = pipeline.vggt_feat(images)
        raw_image_cond = pipeline.encode_image(images)
    b, n, _, _ = aggregated_tokens_list[0].shape
    image_cond = normalize_image_cond(raw_image_cond, batch=b, views=n)
    ss_image_cond = image_cond[:, :, int(args.patch_start_idx) :]

    projection_features, _point_features, point_projection_summary, point_projection_source = build_projection_features(
        args=args,
        image_names=image_names,
        pose_records=pose_records,
        token_device=aggregated_tokens_list[0].device,
        token_dtype=torch.float32,
    )
    projection_features = projection_features.float()

    with torch.no_grad():
        ss_cond_base = pipeline.get_ss_cond(ss_image_cond, aggregated_tokens_list, int(args.num_samples))
    cond_base = ss_cond_base["cond"].detach()
    neg_cond = ss_cond_base["neg_cond"].detach()
    ar_cond_encoding, residual_build = build_ar_cond_residual(
        projection_features,
        cond_base,
        seed=int(args.residual_seed),
        token_mode=str(args.residual_token_mode),
    )
    target_delta = ar_cond_encoding.detach() * float(args.target_scale)
    target_cond = (cond_base + target_delta).detach()

    prior_sample = prior_coords = prior_summary = None
    if args.prior_manifest:
        prior_sample, prior_coords, prior_summary = _load_prior_manifest_sample(Path(args.prior_manifest), args.prior_uid)

    ss_sampler_params = {
        "steps": int(args.ss_steps),
        "cfg_strength": float(args.ss_cfg_strength),
        "cfg_interval": [0.6, 1.0],
        "guidance_rescale": float(args.ss_guidance_rescale),
        "rescale_t": float(args.ss_rescale_t),
    }
    flow = pipeline.models["sparse_structure_flow_model"].to(cond_base.device).eval()
    torch.manual_seed(int(args.seed))
    ss_noise = torch.randn(
        int(args.num_samples),
        flow.in_channels,
        int(flow.resolution),
        int(flow.resolution),
        int(flow.resolution),
        device=cond_base.device,
    )

    def sample_and_report(name: str, cond: torch.Tensor) -> tuple[np.ndarray, dict[str, Any]]:
        ss_cond = {"cond": cond.to(dtype=cond_base.dtype), "neg_cond": neg_cond}
        print(f"[B5 sweep] sampling {name}", flush=True)
        coords = pipeline.sample_sparse_structure(ss_cond, int(args.num_samples), ss_sampler_params, noise=ss_noise.clone())
        coords_array = coords_np(coords)
        prior_alignment = None
        if prior_sample is not None and prior_coords is not None:
            prior_alignment = sparse_diagnostic_metrics(
                "b5_sweep_sparse",
                coords_array,
                prior_coords,
                prior_sample,
                prior_radius=float(args.prior_radius),
                min_support_views=int(args.projection_min_support_views),
                min_support_ratio=float(args.projection_min_support_ratio),
                visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
                visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
                grid_resolution=64,
                mask_threshold=int(args.mask_threshold),
            )
        return coords_array, {"sparse": _component_stats(coords_array), "prior_alignment": prior_alignment}

    baseline_coords, baseline_report = sample_and_report("baseline", cond_base)
    target_coords, target_report = sample_and_report("target_residual", target_cond)
    target_report["delta_vs_baseline"] = delta_report(
        baseline_coords=baseline_coords,
        candidate_coords=target_coords,
        prior_sample=prior_sample,
        prior_coords=prior_coords,
        prior_radius=float(args.prior_radius),
        projection_min_support_views=int(args.projection_min_support_views),
        projection_min_support_ratio=float(args.projection_min_support_ratio),
        visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
        visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
        mask_threshold=int(args.mask_threshold),
    )

    adapter = SSCondResidualAdapter(channels=int(cond_base.shape[-1]), hidden_dim=int(args.hidden_dim)).to(cond_base.device).eval()
    checkpoint_reports: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        label = checkpoint_label(ckpt)
        loaded = load_adapter(adapter, ckpt)
        with torch.no_grad():
            adapter_delta = adapter(cond_base, ar_cond_encoding)
        cond = cond_base + adapter_delta.detach()
        coords, row_report = sample_and_report(label, cond)
        row_report["delta_vs_baseline"] = delta_report(
            baseline_coords=baseline_coords,
            candidate_coords=coords,
            prior_sample=prior_sample,
            prior_coords=prior_coords,
            prior_radius=float(args.prior_radius),
            projection_min_support_views=int(args.projection_min_support_views),
            projection_min_support_ratio=float(args.projection_min_support_ratio),
            visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
            visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
            mask_threshold=int(args.mask_threshold),
        )
        row_report["adapter_vs_target_set_compare"] = _set_compare(_xyz_set(target_coords), _xyz_set(coords))
        stats = adapter_stats(adapter_delta, target_delta)
        direction = (row_report["delta_vs_baseline"].get("direction_summary") or {})
        set_cmp = row_report["adapter_vs_target_set_compare"]
        sparse = row_report["sparse"]
        passed = (
            float(set_cmp.get("iou", 0.0)) >= float(args.pass_adapter_target_iou)
            and float(direction.get("added_minus_removed_projection_any_mask_hit_ratio", -1.0)) > float(args.pass_mask_delta)
            and float(direction.get("added_minus_removed_visible_outside_mask_event_ratio", 1.0)) <= float(args.pass_outside_delta)
            and int(sparse.get("component_count", 10**9)) <= int(args.pass_max_component_count)
        )
        row = {
            "label": label,
            "checkpoint": str(ckpt),
            "loaded": loaded,
            "adapter_stats": stats,
            "report": row_report,
            "passed": passed,
        }
        checkpoint_reports.append(row)
        print(
            "[B5 sweep] "
            f"{label} pass={passed} "
            f"target_iou={set_cmp.get('iou')} "
            f"norm={stats.get('norm_ratio_mean')} "
            f"mask={direction.get('added_minus_removed_projection_any_mask_hit_ratio')} "
            f"outside={direction.get('added_minus_removed_visible_outside_mask_event_ratio')} "
            f"comp={sparse.get('component_count')}",
            flush=True,
        )

    pass_rows = [row for row in checkpoint_reports if row["passed"]]
    best_by_target_iou = max(checkpoint_reports, key=lambda r: r["report"]["adapter_vs_target_set_compare"].get("iou", -1.0))
    best_by_outside = min(
        checkpoint_reports,
        key=lambda r: (r["report"]["delta_vs_baseline"].get("direction_summary") or {}).get(
            "added_minus_removed_visible_outside_mask_event_ratio",
            1.0,
        ),
    )
    report = {
        "args": vars(args),
        "scope": "B5 SS cond residual adapter checkpoint sweep; baseline/target sampled once; adapter checkpoints sampled with shared noise",
        "image_names": image_names,
        "mask_summaries": mask_summaries,
        "cond_base": summarize_tensor(cond_base),
        "ar_cond_encoding": summarize_tensor(ar_cond_encoding),
        "target_delta": summarize_tensor(target_delta),
        "residual_build": residual_build,
        "point_projection_source": point_projection_source,
        "point_projection_summary": point_projection_summary,
        "projection_features": summarize_pose_features(projection_features),
        "prior_summary": prior_summary,
        "baseline": baseline_report,
        "target_residual": target_report,
        "checkpoints": checkpoint_reports,
        "pass_count": len(pass_rows),
        "pass_labels": [row["label"] for row in pass_rows],
        "best_by_target_iou": best_by_target_iou["label"],
        "best_by_outside": best_by_outside["label"],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# B5 SS Cond Residual Checkpoint Sweep",
        "",
        f"- pass_count: {len(pass_rows)} / {len(checkpoint_reports)}",
        f"- pass_labels: {[row['label'] for row in pass_rows]}",
        f"- best_by_target_iou: {best_by_target_iou['label']}",
        f"- best_by_outside: {best_by_outside['label']}",
        "",
        "```text",
    ]
    for row in checkpoint_reports:
        direction = row["report"]["delta_vs_baseline"].get("direction_summary") or {}
        set_cmp = row["report"]["adapter_vs_target_set_compare"]
        stats = row["adapter_stats"]
        sparse = row["report"]["sparse"]
        lines.append(
            f"{row['label']} pass={row['passed']} "
            f"target_iou={set_cmp.get('iou')} "
            f"norm={stats.get('norm_ratio_mean')} "
            f"mask={direction.get('added_minus_removed_projection_any_mask_hit_ratio')} "
            f"outside={direction.get('added_minus_removed_visible_outside_mask_event_ratio')} "
            f"comp={sparse.get('component_count')}"
        )
    lines.extend(["```", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[B5 sweep] wrote {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
