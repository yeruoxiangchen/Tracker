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
    tensor_summary,
)
from reconvggt_ar_adapter_a.projection_token_features import (  # noqa: E402
    build_colmap_point_projection_features,
    build_point_projection_features,
    build_pose_token_features,
    load_points3d_txt,
    load_prior_npz_points,
    parse_colmap_cameras,
    parse_colmap_images,
    parse_ar_pose_file,
    select_pose_records,
    summarize_pose_features,
)
from reconvggt_ar_adapter_a.token_adapter import (  # noqa: E402
    ProjectionAwareSpatialTokenAdapter,
    infer_token_layout,
    max_abs_tree_diff,
    parse_layer_indices,
)


def install_dreamsim_stub() -> None:
    def _stub_dreamsim(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        return DreamSimStub().to(device), None

    trellis_image_to_3d.dreamsim = _stub_dreamsim


def _layer_table(selected_layers: list[int], selected_layouts: dict[str, dict[str, Any]]) -> list[str]:
    rows = [
        "| layer | shape | prefix | spatial tokens | side | pixel/token | square |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for idx in selected_layers:
        row = selected_layouts[str(idx)]
        rows.append(
            "| {layer} | {shape} | {prefix} | {spatial} | {side} | {ppt} | {square} |".format(
                layer=idx,
                shape=row["shape"],
                prefix=row["prefix_tokens"],
                spatial=row["spatial_tokens"],
                side=row["spatial_side"],
                ppt=row["pixel_per_token"],
                square=row["square_spatial_grid"],
            )
        )
    return rows


def _bias_energy(bias: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(bias.float() * bias.float(), dim=-1).clamp_min(eps))


def _summarize_energy(energy: torch.Tensor, hit: torch.Tensor) -> dict[str, float]:
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


def _summarize_score(score: torch.Tensor, hit: torch.Tensor) -> dict[str, float]:
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


def _raw_adapter_biases(adapter: ProjectionAwareSpatialTokenAdapter, projection_features: torch.Tensor) -> dict[int, torch.Tensor]:
    out: dict[int, torch.Tensor] = {}
    for layer_idx in adapter.layer_indices:
        out[int(layer_idx)] = adapter.blocks[str(layer_idx)](projection_features.float())
    return out


def _quantile_input(values: torch.Tensor, *, max_values: int = 1_000_000) -> tuple[torch.Tensor, bool]:
    if values.numel() <= max_values:
        return values, False
    stride = max(1, (int(values.numel()) + int(max_values) - 1) // int(max_values))
    sampled = values[::stride]
    if sampled.numel() > max_values:
        sampled = sampled[:max_values]
    return sampled.contiguous(), True


def _summarize_abs_values(values: list[torch.Tensor]) -> dict[str, float | int | bool]:
    nonempty = [v.detach().float().reshape(-1).cpu() for v in values if v.numel()]
    if not nonempty:
        return {
            "numel": 0,
            "quantile_numel": 0,
            "quantile_approx": False,
            "max": 0.0,
            "mean": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "rmse": 0.0,
        }
    diff = torch.cat(nonempty, dim=0)
    q_diff, quantile_approx = _quantile_input(diff)
    return {
        "numel": int(diff.numel()),
        "quantile_numel": int(q_diff.numel()),
        "quantile_approx": bool(quantile_approx),
        "max": float(diff.max().item()),
        "mean": float(diff.mean().item()),
        "p95": float(torch.quantile(q_diff, 0.95).item()),
        "p99": float(torch.quantile(q_diff, 0.99).item()),
        "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
    }


def _abs_diff_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return (a.detach().float() - b.detach().float()).abs()


def _diff_tensor_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    return _summarize_abs_values([_abs_diff_tensor(a, b)])


def _collect_tree_abs_diffs(a, b, out: list[torch.Tensor]) -> None:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        out.append(_abs_diff_tensor(a, b))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError(f"Dict key mismatch: {sorted(a)} vs {sorted(b)}")
        for key in sorted(a):
            _collect_tree_abs_diffs(a[key], b[key], out)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f"Sequence length mismatch: {len(a)} vs {len(b)}")
        for item_a, item_b in zip(a, b):
            _collect_tree_abs_diffs(item_a, item_b, out)
        return
    raise TypeError(f"Unsupported diff types: {type(a)} vs {type(b)}")


def _tree_diff_stats(a, b) -> dict[str, float | int]:
    diffs: list[torch.Tensor] = []
    _collect_tree_abs_diffs(a, b, diffs)
    return _summarize_abs_values(diffs)


def _load_target_dirs(state: object, *, device: torch.device) -> dict[int, torch.Tensor] | None:
    if not isinstance(state, dict) or "target_dirs" not in state:
        return None
    target_dirs_raw = state.get("target_dirs")
    if not isinstance(target_dirs_raw, dict):
        return None
    out: dict[int, torch.Tensor] = {}
    for key, value in target_dirs_raw.items():
        try:
            layer_idx = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, torch.Tensor):
            out[layer_idx] = value.detach().float().to(device=device)
    return out or None


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="B-stage AR pose projection-aware VGGT token adapter sanity check.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--max_views", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--adapter_layers", default="4,11,17,23")
    parser.add_argument("--adapter_mode", choices=["bias"], default="bias")
    parser.add_argument("--patch_start_idx", type=int, default=5)
    parser.add_argument("--image_resolution", type=int, default=518)
    parser.add_argument("--token_grid_side", type=int, default=37)
    parser.add_argument("--points3d_txt", default="", help="Optional COLMAP-style points3D.txt to project into VGGT token grid.")
    parser.add_argument("--point_prior_npz", default="", help="Optional point-prior npz with source_points/source_conf.")
    parser.add_argument("--colmap_sparse_dir", default="", help="Optional COLMAP sparse dir with cameras.txt/images.txt/points3D.txt for calibrated projection.")
    parser.add_argument(
        "--point_projection_rotation_mode",
        choices=["c2w", "w2c"],
        default="c2w",
        help="Pose convention for projecting point prior into token grid.",
    )
    parser.add_argument("--point_projection_min_depth", type=float, default=1.0e-4)
    parser.add_argument("--default_fx", type=float, default=485.845947)
    parser.add_argument("--default_fy", type=float, default=485.744232)
    parser.add_argument("--default_cx", type=float, default=322.973236)
    parser.add_argument("--default_cy", type=float, default=237.599487)
    parser.add_argument("--default_image_width", type=int, default=640)
    parser.add_argument("--default_image_height", type=int, default=480)
    parser.add_argument("--check_slat", action="store_true")
    parser.add_argument("--load_dreamsim", action="store_true")
    parser.add_argument(
        "--ss_image_cond_mode",
        choices=["skip_prefix", "full"],
        default="skip_prefix",
    )
    parser.add_argument("--save_adapter", default="")
    parser.add_argument("--load_adapter", default="", help="Optional adapter checkpoint to evaluate non-zero trained adapter drift.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"[B-sanity] loading pipeline pretrained={args.pretrained} device={device}", flush=True)
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

    images, image_names = load_images(
        Path(args.image_dir),
        max_views=args.max_views,
        preprocess=args.preprocess,
        pipeline=pipeline,
    )
    pose_records_all = parse_ar_pose_file(
        args.pose_file,
        default_intrinsics=(args.default_fx, args.default_fy, args.default_cx, args.default_cy),
        default_image_size=(args.default_image_width, args.default_image_height),
    )
    pose_records = select_pose_records(image_names, pose_records_all)
    print(f"[B-sanity] loaded {len(images)} images and {len(pose_records)} matched poses", flush=True)

    with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=getattr(pipeline, "VGGT_dtype", torch.float16)):
        aggregated_tokens_list, input_tensor = pipeline.vggt_feat(images)
    b, n, _, _ = aggregated_tokens_list[0].shape
    raw_image_cond = pipeline.encode_image(images)
    image_cond = normalize_image_cond(raw_image_cond, batch=b, views=n)

    selected_layers = parse_layer_indices(args.adapter_layers)
    layouts = [
        infer_token_layout(
            idx,
            aggregated_tokens_list[idx],
            prefix_tokens=args.patch_start_idx,
            image_resolution=args.image_resolution,
        ).__dict__
        for idx in range(len(aggregated_tokens_list))
    ]
    selected_layouts = {str(idx): layouts[idx] for idx in selected_layers}

    for idx in selected_layers:
        row = selected_layouts[str(idx)]
        if not row["square_spatial_grid"]:
            raise ValueError(f"Layer {idx} spatial tokens are not a square grid: {row}")
        if int(row["spatial_side"]) != int(args.token_grid_side):
            raise ValueError(
                f"Layer {idx} spatial side {row['spatial_side']} != token_grid_side {args.token_grid_side}"
            )

    pose_projection_features = build_pose_token_features(
        pose_records,
        token_grid_side=args.token_grid_side,
        image_resolution=args.image_resolution,
        device=aggregated_tokens_list[0].device,
        dtype=torch.float32,
    )
    projection_parts = [pose_projection_features]
    point_projection_summary = None
    point_projection_source = None
    point_projection_features = None
    if args.points3d_txt or args.point_prior_npz or args.colmap_sparse_dir:
        source_count = int(bool(args.points3d_txt)) + int(bool(args.point_prior_npz)) + int(bool(args.colmap_sparse_dir))
        if source_count != 1:
            raise ValueError("Use exactly one of --points3d_txt, --point_prior_npz, or --colmap_sparse_dir")
        if args.colmap_sparse_dir:
            sparse_dir = Path(args.colmap_sparse_dir)
            points, point_conf = load_points3d_txt(sparse_dir / "points3D.txt")
            point_projection_source = str(sparse_dir)
            point_projection_features, point_projection_summary = build_colmap_point_projection_features(
                points,
                point_conf,
                image_names,
                colmap_cameras=parse_colmap_cameras(sparse_dir / "cameras.txt"),
                colmap_images=parse_colmap_images(sparse_dir / "images.txt"),
                token_grid_side=args.token_grid_side,
                image_resolution=args.image_resolution,
                device=aggregated_tokens_list[0].device,
                dtype=torch.float32,
                min_depth=args.point_projection_min_depth,
            )
        else:
            if args.points3d_txt:
                points, point_conf = load_points3d_txt(args.points3d_txt)
                point_projection_source = str(args.points3d_txt)
            else:
                points, point_conf = load_prior_npz_points(args.point_prior_npz)
                point_projection_source = str(args.point_prior_npz)
            point_projection_features, point_projection_summary = build_point_projection_features(
                points,
                point_conf,
                pose_records,
                token_grid_side=args.token_grid_side,
                image_resolution=args.image_resolution,
                device=aggregated_tokens_list[0].device,
                dtype=torch.float32,
                rotation_mode=args.point_projection_rotation_mode,
                min_depth=args.point_projection_min_depth,
            )
        projection_parts.append(point_projection_features)
    projection_features = torch.cat(projection_parts, dim=-1)
    feature_dim = int(projection_features.shape[-1])
    adapter = ProjectionAwareSpatialTokenAdapter.from_tokens(
        aggregated_tokens_list,
        feature_dim=feature_dim,
        hidden_dim=args.adapter_hidden_dim,
        layer_indices=selected_layers,
        prefix_tokens=args.patch_start_idx,
        mode=args.adapter_mode,
    ).to(device=aggregated_tokens_list[0].device)
    adapter_checkpoint_loaded = None
    target_dirs = None
    if args.load_adapter:
        state = torch.load(args.load_adapter, map_location=aggregated_tokens_list[0].device)
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
        target_dirs = _load_target_dirs(state, device=aggregated_tokens_list[0].device)
        adapter_checkpoint_loaded = {
            "path": str(args.load_adapter),
            "missing": list(missing),
            "unexpected": list(unexpected),
            "target_dirs_loaded": target_dirs is not None,
        }
    adapted_tokens_list = adapter(aggregated_tokens_list, projection_features)

    token_diffs = {
        str(idx): max_abs_tree_diff(aggregated_tokens_list[idx], adapted_tokens_list[idx])
        for idx in selected_layers
    }
    token_diff_stats = {
        str(idx): _diff_tensor_stats(aggregated_tokens_list[idx], adapted_tokens_list[idx])
        for idx in selected_layers
    }
    prefix_diffs = {
        str(idx): max_abs_tree_diff(
            aggregated_tokens_list[idx][:, :, : args.patch_start_idx],
            adapted_tokens_list[idx][:, :, : args.patch_start_idx],
        )
        for idx in selected_layers
    }
    prefix_diff_stats = {
        str(idx): _diff_tensor_stats(
            aggregated_tokens_list[idx][:, :, : args.patch_start_idx],
            adapted_tokens_list[idx][:, :, : args.patch_start_idx],
        )
        for idx in selected_layers
    }

    ss_image_cond = image_cond[:, :, args.patch_start_idx :] if args.ss_image_cond_mode == "skip_prefix" else image_cond
    ss_cond_base = pipeline.get_ss_cond(ss_image_cond, aggregated_tokens_list, num_samples=1)
    ss_cond_adapted = pipeline.get_ss_cond(ss_image_cond, adapted_tokens_list, num_samples=1)
    ss_cond_max_abs_diff = max_abs_tree_diff(ss_cond_base["cond"], ss_cond_adapted["cond"])
    ss_cond_diff_stats = _tree_diff_stats(ss_cond_base["cond"], ss_cond_adapted["cond"])

    slat_cond_max_abs_diff = None
    slat_cond_diff_stats = None
    if args.check_slat:
        slat_cond_base = pipeline.get_slat_cond(image_cond, aggregated_tokens_list, num_samples=1)
        slat_cond_adapted = pipeline.get_slat_cond(image_cond, adapted_tokens_list, num_samples=1)
        slat_cond_max_abs_diff = max_abs_tree_diff(slat_cond_base["cond"], slat_cond_adapted["cond"])
        slat_cond_diff_stats = _tree_diff_stats(slat_cond_base["cond"], slat_cond_adapted["cond"])

    token_passed = all(v == 0.0 for v in token_diffs.values())
    prefix_passed = all(v == 0.0 for v in prefix_diffs.values())
    ss_passed = ss_cond_max_abs_diff == 0.0
    slat_passed = True if slat_cond_max_abs_diff is None else slat_cond_max_abs_diff == 0.0
    eval_mode = "loaded_adapter_drift" if adapter_checkpoint_loaded is not None else "zero_init_sanity"
    adapter_energy_stats = None
    adapter_score_stats = None
    if point_projection_features is not None:
        raw_biases = _raw_adapter_biases(adapter, projection_features)
        merged_energy = torch.stack([_bias_energy(b) for b in raw_biases.values()]).mean(dim=0)
        adapter_energy_stats = _summarize_energy(merged_energy, point_projection_features[..., 0].float())
        if target_dirs is not None:
            score_terms = []
            for layer_idx, bias in raw_biases.items():
                direction = target_dirs.get(int(layer_idx))
                if direction is not None:
                    score_terms.append((bias.float() * direction.view(1, 1, 1, -1)).sum(dim=-1))
            if score_terms:
                merged_score = torch.stack(score_terms).mean(dim=0)
                adapter_score_stats = _summarize_score(merged_score, point_projection_features[..., 0].float())

    report: dict[str, Any] = {
        "args": vars(args),
        "eval_mode": eval_mode,
        "image_names": image_names,
        "pose_file": str(args.pose_file),
        "matched_pose_names": [r.image_name for r in pose_records],
        "input_tensor": tensor_summary(input_tensor),
        "image_cond_layout": {
            "shape": list(image_cond.shape),
            "raw_shape": list(raw_image_cond.shape),
            "prefix_tokens": int(args.patch_start_idx),
            "spatial_tokens_after_prefix": int(image_cond.shape[2] - args.patch_start_idx),
        },
        "selected_layers": selected_layers,
        "selected_layouts": selected_layouts,
        "pose_projection_features": summarize_pose_features(pose_projection_features),
        "point_projection_source": point_projection_source,
        "point_projection_summary": point_projection_summary,
        "projection_features": summarize_pose_features(projection_features),
        "adapter": adapter.metadata(),
        "adapter_checkpoint_loaded": adapter_checkpoint_loaded,
        "adapter_energy_stats": adapter_energy_stats,
        "adapter_score_stats": adapter_score_stats,
        "drift_eval": {
            "selected_token_max_abs_diff": token_diffs,
            "selected_token_diff_stats": token_diff_stats,
            "prefix_token_max_abs_diff": prefix_diffs,
            "prefix_token_diff_stats": prefix_diff_stats,
            "ss_cond_max_abs_diff": ss_cond_max_abs_diff,
            "ss_cond_diff_stats": ss_cond_diff_stats,
            "slat_cond_max_abs_diff": slat_cond_max_abs_diff,
            "slat_cond_diff_stats": slat_cond_diff_stats,
            "prefix_passed": prefix_passed,
        },
        "zero_init_sanity": {
            "selected_token_max_abs_diff": token_diffs,
            "selected_token_diff_stats": token_diff_stats,
            "prefix_token_max_abs_diff": prefix_diffs,
            "prefix_token_diff_stats": prefix_diff_stats,
            "ss_cond_max_abs_diff": ss_cond_max_abs_diff,
            "ss_cond_diff_stats": ss_cond_diff_stats,
            "slat_cond_max_abs_diff": slat_cond_max_abs_diff,
            "slat_cond_diff_stats": slat_cond_diff_stats,
            "token_passed": token_passed,
            "prefix_passed": prefix_passed,
            "ss_passed": ss_passed,
            "slat_passed": slat_passed,
            "passed": token_passed and prefix_passed and ss_passed and slat_passed,
        },
        "b_stage_structure": {
            "name": "projection_aware_spatial_token_bias",
            "input": "AR poses.txt + optional point prior -> [B,V,37*37,F] pose/ray/point-projection token features",
            "injection": "VGGT aggregated_tokens_list selected layers, spatial tokens only after prefix=5",
            "prefix_policy": "first 5 tokens unchanged",
            "zero_init_policy": "final projection layer is zero-initialized; stock ReconViaGen behavior is preserved",
            "training_next": "train adapter weights only after zero-init ss/slat diff remains exactly zero",
        },
    }

    if args.save_adapter:
        save_path = Path(args.save_adapter)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": adapter.state_dict(),
                "metadata": adapter.metadata(),
                "feature_summary": summarize_pose_features(projection_features),
                "report": report,
            },
            save_path,
        )
        report["adapter_checkpoint"] = str(save_path)

    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# ReconVGGT AR Adapter B Projection Sanity Report",
        "",
        f"- images: {len(images)}",
        f"- matched poses: {len(pose_records)}",
        f"- selected layers: {selected_layers}",
        f"- projection feature shape: `{list(projection_features.shape)}`",
        f"- projection feature dim: `{feature_dim}`",
        f"- pose feature dim: `{pose_projection_features.shape[-1]}`",
        f"- point feature source: `{point_projection_source}`",
        f"- point projection summary: `{point_projection_summary}`",
        f"- adapter checkpoint loaded: `{adapter_checkpoint_loaded}`",
        f"- eval mode: `{eval_mode}`",
        f"- adapter energy stats: `{adapter_energy_stats}`",
        f"- adapter score stats: `{adapter_score_stats}`",
        f"- ss_cond_max_abs_diff: `{ss_cond_max_abs_diff}`",
        f"- ss_cond_diff_stats: `{ss_cond_diff_stats}`",
        f"- slat_cond_max_abs_diff: `{slat_cond_max_abs_diff}`",
        f"- slat_cond_diff_stats: `{slat_cond_diff_stats}`",
        f"- zero_init_passed: `{report['zero_init_sanity']['passed']}`",
        f"- prefix_passed: `{prefix_passed}`",
        f"- token_passed: `{token_passed}`",
        f"- ss_passed: `{ss_passed}`",
        f"- slat_passed: `{slat_passed}`",
        "",
        "## Selected Token Layouts",
        "",
        *_layer_table(selected_layers, selected_layouts),
        "",
        "## B-Stage Structure",
        "",
        "```text",
        "AR poses.txt + optional points3D/prior_npz",
        "  -> per-view 37x37 token-grid pose/ray/intrinsics features",
        "  -> optional per-view 37x37 point-prior projection occupancy/support features",
        "  -> zero-init MLP",
        "  -> spatial-token bias on VGGT layers [4,11,17,23]",
        "  -> ReconViaGen get_ss_cond / get_slat_cond",
        "```",
        "",
        "Prefix/global/register tokens `[0:5]` are copied unchanged. Spatial token count must be `37*37=1369`.",
        "Only after this report remains exactly zero-diff should B-stage training be enabled.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[B-sanity] wrote {output_dir / 'report.json'}", flush=True)
    print(f"[B-sanity] wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
