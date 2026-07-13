#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from reconvggt_ar_adapter_a.eval_b4_delta_prior_alignment import (  # noqa: E402
    _coords_from_set,
    _direction_summary,
    _load_prior_manifest_sample,
    _set_compare,
    _xyz_set,
)
from reconvggt_ar_adapter_a.inspect_and_sanity import (  # noqa: E402
    DreamSimStub,
    force_eval,
    load_images,
    normalize_image_cond,
    tensor_summary,
)
from reconvggt_ar_adapter_a.projection_token_features import (  # noqa: E402
    parse_ar_pose_file,
    select_pose_records,
    summarize_pose_features,
)
from reconvggt_ar_adapter_a.run_b3_adapter_injection_smoke import (  # noqa: E402
    _component_stats,
    _load_images_with_masks,
)
from reconvggt_ar_adapter_a.train_b_projection_adapter import build_projection_features  # noqa: E402
from trellis_point_prior_mv.sparse_coord_tools import sparse_diagnostic_metrics  # noqa: E402


def install_dreamsim_stub() -> None:
    def _stub_dreamsim(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        return DreamSimStub().to(device), None

    trellis_image_to_3d.dreamsim = _stub_dreamsim


def parse_float_list(spec: str) -> list[float]:
    vals = [float(x.strip()) for x in str(spec).split(",") if x.strip()]
    if not vals:
        raise ValueError("scale list is empty")
    return vals


def scale_name(scale: float) -> str:
    prefix = "p" if scale >= 0 else "m"
    body = f"{abs(float(scale)):.6f}".rstrip("0").rstrip(".")
    return f"{prefix}{body.replace('.', 'p')}"


def coords_np(coords: torch.Tensor) -> np.ndarray:
    return coords.detach().cpu().numpy().astype(np.int32, copy=False)


def summarize_tensor(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach().float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "mean": float(y.mean().item()) if y.numel() else 0.0,
        "std": float(y.std().item()) if y.numel() > 1 else 0.0,
        "rms": float(torch.sqrt((y * y).mean()).item()) if y.numel() else 0.0,
        "abs_max": float(y.abs().max().item()) if y.numel() else 0.0,
    }


def build_ar_cond_residual(
    projection_features: torch.Tensor,
    cond_base: torch.Tensor,
    *,
    seed: int,
    token_mode: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a deterministic AR/prior cond-shaped direction for eval-time smoke.

    This is intentionally not trainable.  It verifies whether an AR-shaped
    residual at the SS condition level can move sparse sampling in a useful
    direction before introducing any adapter training.
    """

    if cond_base.ndim != 3:
        raise ValueError(f"Expected cond_base shape [B,T,C], got {tuple(cond_base.shape)}")
    if projection_features.ndim != 4:
        raise ValueError(f"Expected projection_features shape [B,V,S,F], got {tuple(projection_features.shape)}")
    b, t, c = cond_base.shape
    pf = projection_features.float()
    if pf.shape[0] != b:
        raise ValueError(f"Batch mismatch: projection_features={tuple(pf.shape)}, cond={tuple(cond_base.shape)}")

    mean = pf.mean(dim=(1, 2))
    std = pf.std(dim=(1, 2))
    maxv = pf.amax(dim=(1, 2))
    summary = torch.cat((mean, std, maxv), dim=-1)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    w = torch.randn((summary.shape[-1], c), generator=gen, dtype=torch.float32)
    w = w / math.sqrt(float(max(1, summary.shape[-1])))
    channel = summary.cpu() @ w
    channel = channel.to(device=cond_base.device, dtype=torch.float32)
    channel = channel - channel.mean(dim=-1, keepdim=True)
    channel = channel / channel.std(dim=-1, keepdim=True).clamp_min(1.0e-6)

    if token_mode == "constant":
        token = torch.ones((1, t, 1), device=cond_base.device, dtype=torch.float32)
    elif token_mode == "random":
        token = torch.randn((t, 1), generator=gen, dtype=torch.float32).to(cond_base.device)
        token = token - token.mean()
        token = token / token.std().clamp_min(1.0e-6)
        token = token.reshape(1, t, 1)
    elif token_mode == "ramp":
        token = torch.linspace(-1.0, 1.0, t, device=cond_base.device, dtype=torch.float32).reshape(1, t, 1)
        token = token / token.std().clamp_min(1.0e-6)
    else:
        raise ValueError(f"Unsupported token_mode={token_mode!r}")

    residual = token * channel[:, None, :]
    residual = residual - residual.mean(dim=(1, 2), keepdim=True)
    residual = residual / torch.sqrt((residual * residual).mean(dim=(1, 2), keepdim=True)).clamp_min(1.0e-6)
    cond_rms = torch.sqrt((cond_base.detach().float() * cond_base.detach().float()).mean(dim=(1, 2), keepdim=True)).clamp_min(1.0e-6)
    residual = residual * cond_rms
    return residual.to(device=cond_base.device, dtype=cond_base.dtype), {
        "seed": int(seed),
        "token_mode": token_mode,
        "summary_shape": list(summary.shape),
        "cond_rms": [float(v) for v in cond_rms.detach().cpu().reshape(-1)],
        "residual": summarize_tensor(residual),
    }


def delta_report(
    *,
    baseline_coords: np.ndarray,
    candidate_coords: np.ndarray,
    prior_sample: dict[str, Any] | None,
    prior_coords: np.ndarray | None,
    prior_radius: float,
    projection_min_support_views: int,
    projection_min_support_ratio: float,
    visual_hull_min_visible_views: int,
    visual_hull_min_support_ratio: float,
    mask_threshold: int,
) -> dict[str, Any] | None:
    if prior_sample is None or prior_coords is None:
        return None
    baseline_set = _xyz_set(baseline_coords)
    candidate_set = _xyz_set(candidate_coords)
    subsets = {
        "baseline": _coords_from_set(baseline_set),
        "candidate": _coords_from_set(candidate_set),
        "added": _coords_from_set(candidate_set - baseline_set),
        "removed": _coords_from_set(baseline_set - candidate_set),
        "kept": _coords_from_set(candidate_set & baseline_set),
        "union": _coords_from_set(candidate_set | baseline_set),
    }
    subset_metrics: dict[str, dict[str, Any]] = {}
    for name, coords in subsets.items():
        metric_name = "adapter" if name == "candidate" else name
        subset_metrics[name] = sparse_diagnostic_metrics(
            metric_name,
            coords,
            prior_coords,
            prior_sample,
            prior_radius=float(prior_radius),
            min_support_views=int(projection_min_support_views),
            min_support_ratio=float(projection_min_support_ratio),
            visual_hull_min_visible_views=int(visual_hull_min_visible_views),
            visual_hull_min_support_ratio=float(visual_hull_min_support_ratio),
            grid_resolution=64,
            mask_threshold=int(mask_threshold),
        )
    report = {
        "set_compare": _set_compare(baseline_set, candidate_set),
        "subset_metrics": subset_metrics,
    }
    # Reuse B4 field names by aliasing candidate -> adapter.
    report["subset_metrics"]["adapter"] = subset_metrics["candidate"]
    report["direction_summary"] = _direction_summary(report)
    return report


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# B5 SS Condition Residual Smoke",
        "",
        "## Scope",
        "",
        "```text",
        "SS-only eval-time residual smoke.",
        "VGGT / sparse_structure_vggt_cond / sparse flow are frozen.",
        "SLAT is not executed.",
        "No decoded sparse coord loss is used for training in this stage.",
        "```",
        "",
        "## B5.0 Interface",
        "",
        "```text",
        f"cond_base: {report['cond_base']}",
        f"projection_features: {report['projection_features']}",
        f"residual: {report['residual_build']}",
        "```",
        "",
        "## B5.1 Candidates",
        "",
        "```text",
    ]
    for cand in report["candidates"]:
        delta = cand.get("delta_vs_baseline") or {}
        direction = delta.get("direction_summary") or {}
        lines.extend(
            [
                f"{cand['name']}:",
                f"  scale = {cand['scale']}",
                f"  coord_count = {cand['sparse'].get('coord_count')}",
                f"  component_count = {cand['sparse'].get('component_count')}",
                f"  largest_component_ratio = {cand['sparse'].get('largest_component_ratio')}",
                f"  set_iou_vs_baseline = {(delta.get('set_compare') or {}).get('iou')}",
                f"  added_minus_removed_within_prior_radius_ratio = {direction.get('added_minus_removed_within_prior_radius_ratio')}",
                f"  added_minus_removed_projection_any_mask_hit_ratio = {direction.get('added_minus_removed_projection_any_mask_hit_ratio')}",
                f"  added_minus_removed_visible_outside_mask_event_ratio = {direction.get('added_minus_removed_visible_outside_mask_event_ratio')}",
                f"  adapter_minus_baseline_visible_outside_mask_event_ratio = {direction.get('adapter_minus_baseline_visible_outside_mask_event_ratio')}",
                "",
            ]
        )
    lines.extend(["```", "", "## Judgment", "", report.get("judgment", "")])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="B5 SS-condition residual eval-time smoke / candidate search.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--output_dir", required=True)
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
    parser.add_argument("--residual_scales", default="0,0.02,-0.02,0.05,-0.05")
    parser.add_argument("--residual_seed", type=int, default=20260707)
    parser.add_argument("--residual_token_mode", choices=["constant", "random", "ramp"], default="random")
    parser.add_argument("--prior_manifest", default="")
    parser.add_argument("--prior_uid", default="")
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--projection_min_support_views", type=int, default=1)
    parser.add_argument("--projection_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--visual_hull_min_support_ratio", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if not args.load_dreamsim:
        install_dreamsim_stub()

    print(f"[B5] loading pipeline pretrained={args.pretrained} device={device}", flush=True)
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

    mask_summaries = None
    if args.mask_mode == "apply":
        if not args.mask_dir:
            raise ValueError("--mask_dir is required when --mask_mode=apply")
        if args.preprocess:
            raise ValueError("--preprocess is not supported with --mask_mode=apply")
        images, image_names, mask_summaries = _load_images_with_masks(
            Path(args.image_dir),
            mask_dir=Path(args.mask_dir),
            max_views=int(args.max_views),
            mask_background=args.mask_background,
        )
    else:
        images, image_names = load_images(Path(args.image_dir), max_views=int(args.max_views), preprocess=args.preprocess, pipeline=pipeline)

    pose_records_all = parse_ar_pose_file(
        args.pose_file,
        default_intrinsics=(args.default_fx, args.default_fy, args.default_cx, args.default_cy),
        default_image_size=(args.default_image_width, args.default_image_height),
    )
    pose_records = select_pose_records(image_names, pose_records_all)
    print(f"[B5] loaded {len(images)} images and {len(pose_records)} poses", flush=True)

    torch.manual_seed(int(args.seed))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=getattr(pipeline, "VGGT_dtype", torch.float16)):
        aggregated_tokens_list, _ = pipeline.vggt_feat(images)
        raw_image_cond = pipeline.encode_image(images)
    b, n, _, _ = aggregated_tokens_list[0].shape
    image_cond = normalize_image_cond(raw_image_cond, batch=b, views=n)
    ss_image_cond = image_cond[:, :, int(args.patch_start_idx) :]

    projection_features, point_features, point_projection_summary, point_projection_source = build_projection_features(
        args=args,
        image_names=image_names,
        pose_records=pose_records,
        token_device=aggregated_tokens_list[0].device,
        token_dtype=torch.float32,
    )
    projection_features = projection_features.float()
    point_features = point_features.float()

    with torch.no_grad():
        ss_cond_base = pipeline.get_ss_cond(ss_image_cond, aggregated_tokens_list, int(args.num_samples))
    cond_base = ss_cond_base["cond"].detach()
    neg_cond = ss_cond_base["neg_cond"].detach()
    residual_unit, residual_build = build_ar_cond_residual(
        projection_features,
        cond_base,
        seed=int(args.residual_seed),
        token_mode=str(args.residual_token_mode),
    )

    prior_sample = prior_coords = prior_summary = None
    if args.prior_manifest:
        prior_sample, prior_coords, prior_summary = _load_prior_manifest_sample(Path(args.prior_manifest), args.prior_uid)

    ss_flow_model = pipeline.models["sparse_structure_flow_model"]
    ss_sampler_params = {
        "steps": int(args.ss_steps),
        "cfg_strength": float(args.ss_cfg_strength),
        "cfg_interval": [0.6, 1.0],
        "guidance_rescale": float(args.ss_guidance_rescale),
        "rescale_t": float(args.ss_rescale_t),
    }
    reso = int(ss_flow_model.resolution)
    sample_device = aggregated_tokens_list[0].device
    torch.manual_seed(int(args.seed))
    ss_noise = torch.randn(
        int(args.num_samples),
        ss_flow_model.in_channels,
        reso,
        reso,
        reso,
        device=sample_device,
    )

    candidates: list[dict[str, Any]] = []
    baseline_coords: np.ndarray | None = None
    for scale in parse_float_list(args.residual_scales):
        name = f"scale_{scale_name(scale)}"
        cand_dir = output_dir / name
        cand_dir.mkdir(parents=True, exist_ok=True)
        delta = residual_unit * float(scale)
        ss_cond = {
            "cond": (cond_base + delta).to(dtype=cond_base.dtype),
            "neg_cond": neg_cond,
        }
        print(f"[B5] sampling {name}", flush=True)
        coords = pipeline.sample_sparse_structure(ss_cond, int(args.num_samples), ss_sampler_params, noise=ss_noise.clone())
        coords_array = coords_np(coords)
        np.savez_compressed(cand_dir / "coords.npz", coords=coords_array)
        if baseline_coords is None and abs(float(scale)) < 1.0e-12:
            baseline_coords = coords_array
        sparse = _component_stats(coords_array)
        prior_alignment = None
        if prior_sample is not None and prior_coords is not None:
            prior_alignment = sparse_diagnostic_metrics(
                "b5_sparse",
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
        cand = {
            "name": name,
            "scale": float(scale),
            "output_dir": str(cand_dir),
            "cond_delta": summarize_tensor(delta),
            "sparse": sparse,
            "prior_alignment": prior_alignment,
        }
        candidates.append(cand)

    if baseline_coords is None:
        raise ValueError("--residual_scales must include 0 for baseline comparison")
    for cand in candidates:
        cand_coords = np.load(Path(cand["output_dir"]) / "coords.npz")["coords"]
        cand["delta_vs_baseline"] = delta_report(
            baseline_coords=baseline_coords,
            candidate_coords=cand_coords,
            prior_sample=prior_sample,
            prior_coords=prior_coords,
            prior_radius=float(args.prior_radius),
            projection_min_support_views=int(args.projection_min_support_views),
            projection_min_support_ratio=float(args.projection_min_support_ratio),
            visual_hull_min_visible_views=int(args.visual_hull_min_visible_views),
            visual_hull_min_support_ratio=float(args.visual_hull_min_support_ratio),
            mask_threshold=int(args.mask_threshold),
        )
        (Path(cand["output_dir"]) / "report.json").write_text(json.dumps(cand, indent=2, ensure_ascii=False), encoding="utf-8")

    judgment = (
        "B5.0 confirms the SS-condition residual interface is shape-compatible and SS-only. "
        "B5.1 is eval-time candidate search; decoded coord metrics are diagnostics only, not a differentiable training loss. "
        "A useful candidate should improve added-minus-removed prior/mask direction without increasing visible outside-mask events."
    )
    report = {
        "args": vars(args),
        "image_names": image_names,
        "mask_summaries": mask_summaries,
        "point_projection_source": point_projection_source,
        "point_projection_summary": point_projection_summary,
        "projection_features": summarize_pose_features(projection_features),
        "point_features": summarize_pose_features(point_features),
        "cond_base": summarize_tensor(cond_base),
        "neg_cond": summarize_tensor(neg_cond),
        "residual_build": residual_build,
        "prior_summary": prior_summary,
        "candidates": candidates,
        "differentiability_note": {
            "teacher_forced_flow_losses": "potentially differentiable if computed before sampling/decode",
            "decoded_coords_iou_component_visual_hull": "diagnostic/non-differentiable in this smoke because sampling + threshold/argwhere are discrete",
            "b5_scope": "eval-time residual candidate search only; no training",
        },
        "judgment": judgment,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "report.md", report)
    print(f"[B5] wrote {output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
