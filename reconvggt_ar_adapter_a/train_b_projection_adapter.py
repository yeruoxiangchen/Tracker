#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import trellis.pipelines.trellis_image_to_3d as trellis_image_to_3d  # noqa: E402
from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402

from reconvggt_ar_adapter_a.inspect_and_sanity import (  # noqa: E402
    DreamSimStub,
    force_eval,
    load_images,
    normalize_image_cond,
)
from reconvggt_ar_adapter_a.projection_token_features import (  # noqa: E402
    build_colmap_point_projection_features,
    build_point_projection_features,
    build_pose_token_features,
    build_token_mask_features,
    load_points3d_txt,
    load_prior_npz_points,
    parse_ar_pose_file,
    parse_colmap_cameras,
    parse_colmap_images,
    select_pose_records,
    summarize_pose_features,
)
from reconvggt_ar_adapter_a.token_adapter import (  # noqa: E402
    ProjectionAwareSpatialTokenAdapter,
    max_abs_tree_diff,
    parse_layer_indices,
)


def install_dreamsim_stub() -> None:
    def _stub_dreamsim(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        return DreamSimStub().to(device), None

    trellis_image_to_3d.dreamsim = _stub_dreamsim


def build_projection_features(
    *,
    args: argparse.Namespace,
    image_names: list[str],
    pose_records,
    token_device: torch.device,
    token_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None, str | None]:
    mask_projection_mode = str(getattr(args, "mask_projection_mode", "none") or "none")
    if mask_projection_mode not in {"none", "filter_points", "token_mask", "filter_points_token_mask"}:
        raise ValueError(f"Unsupported mask_projection_mode={mask_projection_mode!r}")
    mask_dir = str(getattr(args, "mask_dir", "") or "")
    point_support_min_views = int(getattr(args, "point_mask_support_min_views", 0) or 0)
    point_support_min_ratio = float(getattr(args, "point_mask_support_min_ratio", 0.0) or 0.0)
    use_point_support_filter = bool(mask_dir) and (point_support_min_views > 0 or point_support_min_ratio > 0.0)
    if not mask_dir and (point_support_min_views > 0 or point_support_min_ratio > 0.0):
        raise ValueError("--mask_dir is required when point mask support filtering is enabled")
    use_mask_filter = bool(mask_dir) and mask_projection_mode in {"filter_points", "filter_points_token_mask"}
    use_token_mask = bool(mask_dir) and mask_projection_mode in {"token_mask", "filter_points_token_mask"}
    point_mask_dir = mask_dir if (use_mask_filter or use_point_support_filter) else None
    pose_features = build_pose_token_features(
        pose_records,
        token_grid_side=args.token_grid_side,
        image_resolution=args.image_resolution,
        device=token_device,
        dtype=token_dtype,
    )
    parts = [pose_features]
    point_summary = None
    point_source = None
    source_count = int(bool(args.points3d_txt)) + int(bool(args.point_prior_npz)) + int(bool(args.colmap_sparse_dir))
    if source_count != 1:
        raise ValueError("B2 training requires exactly one point source: --colmap_sparse_dir, --points3d_txt, or --point_prior_npz")
    if args.colmap_sparse_dir:
        sparse_dir = Path(args.colmap_sparse_dir)
        points, conf = load_points3d_txt(sparse_dir / "points3D.txt")
        point_features, point_summary = build_colmap_point_projection_features(
            points,
            conf,
            image_names,
            colmap_cameras=parse_colmap_cameras(sparse_dir / "cameras.txt"),
            colmap_images=parse_colmap_images(sparse_dir / "images.txt"),
            token_grid_side=args.token_grid_side,
            image_resolution=args.image_resolution,
            device=token_device,
            dtype=token_dtype,
            min_depth=args.point_projection_min_depth,
            mask_dir=point_mask_dir,
            mask_threshold=int(getattr(args, "mask_threshold", 127)),
            point_mask_support_min_views=point_support_min_views,
            point_mask_support_min_ratio=point_support_min_ratio,
        )
        point_source = str(sparse_dir)
    elif args.points3d_txt:
        points, conf = load_points3d_txt(args.points3d_txt)
        point_features, point_summary = build_point_projection_features(
            points,
            conf,
            pose_records,
            token_grid_side=args.token_grid_side,
            image_resolution=args.image_resolution,
            device=token_device,
            dtype=token_dtype,
            rotation_mode=args.point_projection_rotation_mode,
            min_depth=args.point_projection_min_depth,
            mask_dir=point_mask_dir,
            image_names=image_names if point_mask_dir else None,
            mask_threshold=int(getattr(args, "mask_threshold", 127)),
            point_mask_support_min_views=point_support_min_views,
            point_mask_support_min_ratio=point_support_min_ratio,
        )
        point_source = str(args.points3d_txt)
    else:
        points, conf = load_prior_npz_points(args.point_prior_npz)
        point_features, point_summary = build_point_projection_features(
            points,
            conf,
            pose_records,
            token_grid_side=args.token_grid_side,
            image_resolution=args.image_resolution,
            device=token_device,
            dtype=token_dtype,
            rotation_mode=args.point_projection_rotation_mode,
            min_depth=args.point_projection_min_depth,
            mask_dir=point_mask_dir,
            image_names=image_names if point_mask_dir else None,
            mask_threshold=int(getattr(args, "mask_threshold", 127)),
            point_mask_support_min_views=point_support_min_views,
            point_mask_support_min_ratio=point_support_min_ratio,
        )
        point_source = str(args.point_prior_npz)
    parts.append(point_features)
    token_mask_summary = None
    if use_token_mask:
        token_mask_features, token_mask_summary = build_token_mask_features(
            image_names,
            mask_dir=mask_dir,
            token_grid_side=args.token_grid_side,
            image_resolution=args.image_resolution,
            device=token_device,
            dtype=token_dtype,
            mask_threshold=int(getattr(args, "mask_threshold", 127)),
            token_min_ratio=float(getattr(args, "token_mask_min_ratio", 0.05)),
        )
        parts.append(token_mask_features)
    if point_summary is not None:
        point_summary = dict(point_summary)
        point_summary["mask_projection_mode"] = mask_projection_mode
        point_summary["token_mask_summary"] = token_mask_summary
        point_summary["point_mask_support_min_views"] = point_support_min_views
        point_summary["point_mask_support_min_ratio"] = point_support_min_ratio
    return torch.cat(parts, dim=-1), point_features, point_summary, point_source


def adapter_biases(
    adapter: ProjectionAwareSpatialTokenAdapter,
    base_tokens: list[torch.Tensor],
    projection_features: torch.Tensor,
    *,
    prefix_tokens: int,
) -> dict[int, torch.Tensor]:
    adapted = adapter(base_tokens, projection_features)
    biases: dict[int, torch.Tensor] = {}
    for layer_idx in adapter.layer_indices:
        biases[int(layer_idx)] = adapted[layer_idx][:, :, prefix_tokens:] - base_tokens[layer_idx][:, :, prefix_tokens:]
    return biases


def raw_adapter_biases(
    adapter: ProjectionAwareSpatialTokenAdapter,
    projection_features: torch.Tensor,
) -> dict[int, torch.Tensor]:
    biases: dict[int, torch.Tensor] = {}
    for layer_idx in adapter.layer_indices:
        block = adapter.blocks[str(layer_idx)]
        biases[int(layer_idx)] = adapter.apply_feature_gate(block(projection_features.float()), projection_features)
    return biases


def bias_energy(bias: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(bias.float() * bias.float(), dim=-1).clamp_min(eps))


def summarize_energy(energy: torch.Tensor, hit: torch.Tensor) -> dict[str, float]:
    hit_mask = hit > 0.5
    miss_mask = ~hit_mask
    out = {
        "mean": float(energy.mean().detach().cpu().item()),
        "max": float(energy.max().detach().cpu().item()),
        "hit_ratio": float(hit_mask.float().mean().detach().cpu().item()),
    }
    out["hit_mean"] = float(energy[hit_mask].mean().detach().cpu().item()) if bool(hit_mask.any()) else 0.0
    out["miss_mean"] = float(energy[miss_mask].mean().detach().cpu().item()) if bool(miss_mask.any()) else 0.0
    out["separation"] = out["hit_mean"] - out["miss_mean"]
    return out


def make_target_dirs(adapter: ProjectionAwareSpatialTokenAdapter, *, seed: int, device: torch.device) -> dict[int, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    out: dict[int, torch.Tensor] = {}
    for layer_idx in adapter.layer_indices:
        dim = int(adapter.token_dims[int(layer_idx)])
        direction = torch.randn((dim,), generator=gen, dtype=torch.float32)
        direction = direction / direction.norm().clamp_min(1.0e-8)
        out[int(layer_idx)] = direction.to(device=device)
    return out


def directional_score(bias: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return (bias.float() * direction.view(1, 1, 1, -1)).sum(dim=-1)


def summarize_score(score: torch.Tensor, hit: torch.Tensor) -> dict[str, float]:
    hit_mask = hit > 0.5
    miss_mask = ~hit_mask
    out = {
        "mean": float(score.mean().detach().cpu().item()),
        "abs_max": float(score.abs().max().detach().cpu().item()),
    }
    out["hit_mean"] = float(score[hit_mask].mean().detach().cpu().item()) if bool(hit_mask.any()) else 0.0
    out["miss_mean"] = float(score[miss_mask].mean().detach().cpu().item()) if bool(miss_mask.any()) else 0.0
    out["separation"] = out["hit_mean"] - out["miss_mean"]
    return out


def summarize_gate_feature(projection_features: torch.Tensor, gate_feature_index: int | None) -> dict[str, Any] | None:
    if gate_feature_index is None:
        return None
    idx = int(gate_feature_index)
    if idx < 0:
        idx = int(projection_features.shape[-1]) + idx
    if idx < 0 or idx >= int(projection_features.shape[-1]):
        raise IndexError(
            f"gate_feature_index={gate_feature_index} resolves to {idx}, "
            f"but projection feature dim is {projection_features.shape[-1]}"
        )
    gate = projection_features[..., idx].detach().float()
    return {
        "gate_feature_index": int(gate_feature_index),
        "resolved_index": int(idx),
        "shape": list(gate.shape),
        "mean": float(gate.mean().cpu().item()) if gate.numel() else 0.0,
        "max": float(gate.max().cpu().item()) if gate.numel() else 0.0,
        "min": float(gate.min().cpu().item()) if gate.numel() else 0.0,
        "nonzero_ratio": float((gate > 0).float().mean().cpu().item()) if gate.numel() else 0.0,
        "ge_0p05_ratio": float((gate >= 0.05).float().mean().cpu().item()) if gate.numel() else 0.0,
        "ge_0p5_ratio": float((gate >= 0.5).float().mean().cpu().item()) if gate.numel() else 0.0,
    }


def condition_diff_metrics(pipeline, image_cond: torch.Tensor, base_tokens: list[torch.Tensor], adapted_tokens: list[torch.Tensor], args: argparse.Namespace) -> dict[str, Any]:
    ss_image_cond = image_cond[:, :, args.patch_start_idx :] if args.ss_image_cond_mode == "skip_prefix" else image_cond
    ss_base = pipeline.get_ss_cond(ss_image_cond, base_tokens, num_samples=1)
    ss_adapted = pipeline.get_ss_cond(ss_image_cond, adapted_tokens, num_samples=1)
    out = {"ss_cond_max_abs_diff": max_abs_tree_diff(ss_base["cond"], ss_adapted["cond"])}
    if args.check_slat:
        slat_base = pipeline.get_slat_cond(image_cond, base_tokens, num_samples=1)
        slat_adapted = pipeline.get_slat_cond(image_cond, adapted_tokens, num_samples=1)
        out["slat_cond_max_abs_diff"] = max_abs_tree_diff(slat_base["cond"], slat_adapted["cond"])
    else:
        out["slat_cond_max_abs_diff"] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="B2 adapter-only point-prior signal-injection smoke training.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--max_views", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--adapter_layers", default="4,11,17,23")
    parser.add_argument("--adapter_mode", choices=["bias"], default="bias")
    parser.add_argument("--patch_start_idx", type=int, default=5)
    parser.add_argument("--image_resolution", type=int, default=518)
    parser.add_argument("--token_grid_side", type=int, default=37)
    parser.add_argument("--points3d_txt", default="")
    parser.add_argument("--point_prior_npz", default="")
    parser.add_argument("--colmap_sparse_dir", default="")
    parser.add_argument("--mask_dir", default="", help="Optional mask dir for B4.1 mask-aware projection features.")
    parser.add_argument(
        "--mask_projection_mode",
        choices=["none", "filter_points", "token_mask", "filter_points_token_mask"],
        default="none",
    )
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--token_mask_min_ratio", type=float, default=0.05)
    parser.add_argument("--point_mask_support_min_views", type=int, default=0)
    parser.add_argument("--point_mask_support_min_ratio", type=float, default=0.0)
    parser.add_argument("--adapter_gate_feature_index", type=int, default=None)
    parser.add_argument("--adapter_gate_power", type=float, default=1.0)
    parser.add_argument("--point_projection_rotation_mode", choices=["c2w", "w2c"], default="c2w")
    parser.add_argument("--point_projection_min_depth", type=float, default=1.0e-4)
    parser.add_argument("--default_fx", type=float, default=485.845947)
    parser.add_argument("--default_fy", type=float, default=485.744232)
    parser.add_argument("--default_cx", type=float, default=322.973236)
    parser.add_argument("--default_cy", type=float, default=237.599487)
    parser.add_argument("--default_image_width", type=int, default=640)
    parser.add_argument("--default_image_height", type=int, default=480)
    parser.add_argument("--load_dreamsim", action="store_true")
    parser.add_argument("--check_slat", action="store_true")
    parser.add_argument("--ss_image_cond_mode", choices=["skip_prefix", "full"], default="skip_prefix")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--bias_target_scale", type=float, default=0.02)
    parser.add_argument("--miss_weight", type=float, default=0.25)
    parser.add_argument("--bias_l2_weight", type=float, default=0.01)
    parser.add_argument("--direction_seed", type=int, default=1234)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[B2-train] loading pipeline pretrained={args.pretrained} device={device}", flush=True)
    if not args.load_dreamsim:
        install_dreamsim_stub()
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

    images, image_names = load_images(Path(args.image_dir), max_views=args.max_views, preprocess=args.preprocess, pipeline=pipeline)
    pose_records_all = parse_ar_pose_file(
        args.pose_file,
        default_intrinsics=(args.default_fx, args.default_fy, args.default_cx, args.default_cy),
        default_image_size=(args.default_image_width, args.default_image_height),
    )
    pose_records = select_pose_records(image_names, pose_records_all)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=getattr(pipeline, "VGGT_dtype", torch.float16)):
        aggregated_tokens_list, _ = pipeline.vggt_feat(images)
        raw_image_cond = pipeline.encode_image(images)
    b, n, _, _ = aggregated_tokens_list[0].shape
    image_cond = normalize_image_cond(raw_image_cond, batch=b, views=n)
    base_tokens = [x.detach() for x in aggregated_tokens_list]
    selected_layers = parse_layer_indices(args.adapter_layers)
    projection_features, point_features, point_summary, point_source = build_projection_features(
        args=args,
        image_names=image_names,
        pose_records=pose_records,
        token_device=base_tokens[0].device,
        token_dtype=torch.float32,
    )
    if point_summary is None or int(point_summary.get("inside_count_total", 0)) <= 0:
        raise ValueError(f"point projection has no token support: {point_summary}")

    adapter = ProjectionAwareSpatialTokenAdapter.from_tokens(
        base_tokens,
        feature_dim=int(projection_features.shape[-1]),
        hidden_dim=args.adapter_hidden_dim,
        layer_indices=selected_layers,
        prefix_tokens=args.patch_start_idx,
        mode=args.adapter_mode,
        gate_feature_index=args.adapter_gate_feature_index,
        gate_power=float(args.adapter_gate_power),
    ).to(device=base_tokens[0].device)
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(args.lr), weight_decay=0.0)

    projection_features = projection_features.float()
    point_features = point_features.float()
    gate_feature_stats = summarize_gate_feature(projection_features, args.adapter_gate_feature_index)
    hit = point_features[..., 0]
    log_count = point_features[..., 1]
    target = hit * float(args.bias_target_scale) * (0.25 + log_count)
    miss_weight = torch.where(hit > 0.5, torch.ones_like(hit), torch.full_like(hit, float(args.miss_weight)))
    target_dirs = make_target_dirs(adapter, seed=int(args.direction_seed), device=base_tokens[0].device)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        initial_diff = condition_diff_metrics(pipeline, image_cond, base_tokens, adapter(base_tokens, projection_features), args)
    for step in range(1, int(args.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        biases = raw_adapter_biases(adapter, projection_features)
        losses = []
        score_tensors = []
        energies = []
        for layer_idx, bias in biases.items():
            score = directional_score(bias, target_dirs[int(layer_idx)])
            score_tensors.append(score.detach())
            e = bias_energy(bias)
            energies.append(e.detach())
            losses.append(((score - target) ** 2 * miss_weight).mean())
        directional_loss = torch.stack(losses).mean()
        l2_loss = torch.stack([b.float().pow(2).mean() for b in biases.values()]).mean()
        loss = directional_loss + float(args.bias_l2_weight) * l2_loss
        loss.backward()
        optimizer.step()

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.max_steps):
            merged_energy = torch.stack(energies).mean(dim=0)
            merged_score = torch.stack(score_tensors).mean(dim=0)
            energy_stats = summarize_energy(merged_energy, hit)
            score_stats = summarize_score(merged_score, hit)
            row = {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "directional_loss": float(directional_loss.detach().cpu().item()),
                "bias_l2_loss": float(l2_loss.detach().cpu().item()),
                **{f"energy_{k}": v for k, v in energy_stats.items()},
                **{f"score_{k}": v for k, v in score_stats.items()},
            }
            rows.append(row)
            print(
                "[B2-train] "
                f"step={step} loss={row['loss']:.6g} directional={row['directional_loss']:.6g} "
                f"score_hit={row['score_hit_mean']:.6g} score_miss={row['score_miss_mean']:.6g} "
                f"hit={row['energy_hit_mean']:.6g} miss={row['energy_miss_mean']:.6g} "
                f"sep={row['energy_separation']:.6g}",
                flush=True,
            )
        if step % int(args.save_every) == 0 or step == int(args.max_steps):
            torch.save(
                {
                    "state_dict": adapter.state_dict(),
                    "metadata": adapter.metadata(),
                    "args": vars(args),
                    "point_projection_summary": point_summary,
                    "point_projection_source": point_source,
                    "projection_features": summarize_pose_features(projection_features),
                    "gate_feature_stats": gate_feature_stats,
                    "target_dirs": {str(k): v.detach().cpu() for k, v in target_dirs.items()},
                    "rows": rows,
                },
                ckpt_dir / f"adapter_step_{step:06d}.ckpt",
            )

    adapter.eval()
    with torch.no_grad():
        adapted_tokens = adapter(base_tokens, projection_features)
        final_diff = condition_diff_metrics(pipeline, image_cond, base_tokens, adapted_tokens, args)
        final_token_diffs = {
            str(idx): max_abs_tree_diff(base_tokens[idx], adapted_tokens[idx])
            for idx in selected_layers
        }
        final_biases = raw_adapter_biases(adapter, projection_features)
        final_energy = torch.stack([bias_energy(b) for b in final_biases.values()]).mean(dim=0)
        final_score = torch.stack([directional_score(b, target_dirs[int(k)]) for k, b in final_biases.items()]).mean(dim=0)
        final_energy_stats = summarize_energy(final_energy, hit)
        final_score_stats = summarize_score(final_score, hit)

    final_ckpt = ckpt_dir / "last.ckpt"
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "metadata": adapter.metadata(),
            "args": vars(args),
            "point_projection_summary": point_summary,
            "point_projection_source": point_source,
            "projection_features": summarize_pose_features(projection_features),
            "gate_feature_stats": gate_feature_stats,
            "target_dirs": {str(k): v.detach().cpu() for k, v in target_dirs.items()},
            "rows": rows,
            "initial_condition_diff": initial_diff,
            "final_condition_diff": final_diff,
            "final_token_diffs": final_token_diffs,
            "final_energy_stats": final_energy_stats,
            "final_score_stats": final_score_stats,
        },
        final_ckpt,
    )

    report = {
        "args": vars(args),
        "image_names": image_names,
        "point_projection_source": point_source,
        "point_projection_summary": point_summary,
        "projection_features": summarize_pose_features(projection_features),
        "gate_feature_stats": gate_feature_stats,
        "adapter": adapter.metadata(),
        "initial_condition_diff": initial_diff,
        "final_condition_diff": final_diff,
        "final_token_diffs": final_token_diffs,
        "final_energy_stats": final_energy_stats,
        "final_score_stats": final_score_stats,
        "rows": rows,
        "checkpoint": str(final_ckpt),
        "b2_scope": "adapter-only signal-injection smoke; not a mesh-quality training objective",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# ReconVGGT AR Adapter B2 Training Report",
        "",
        f"- point source: `{point_source}`",
        f"- point projection summary: `{point_summary}`",
        f"- projection feature shape: `{report['projection_features']['shape']}`",
        f"- gate feature stats: `{gate_feature_stats}`",
        f"- checkpoint: `{final_ckpt}`",
        f"- initial condition diff: `{initial_diff}`",
        f"- final condition diff: `{final_diff}`",
        f"- final energy stats: `{final_energy_stats}`",
        f"- final score stats: `{final_score_stats}`",
        "",
        "B2 只是 adapter-only signal-injection smoke：用 directional score loss 验证 point-prior projection feature 能否产生受控 spatial-token bias，并监控 ss/slat condition drift。",
        "它不是最终 mesh 质量训练目标。",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[B2-train] wrote {output_dir / 'report.json'}", flush=True)
    print(f"[B2-train] wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
