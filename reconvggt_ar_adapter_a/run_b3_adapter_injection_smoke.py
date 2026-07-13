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
from PIL import Image

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
from reconvggt_ar_adapter_a.token_adapter import (  # noqa: E402
    ProjectionAwareSpatialTokenAdapter,
    parse_layer_indices,
)
from reconvggt_ar_adapter_a.train_b_projection_adapter import build_projection_features, summarize_gate_feature  # noqa: E402
from trellis_point_prior_mv.sparse_coord_tools import sparse_diagnostic_metrics  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def install_dreamsim_stub() -> None:
    def _stub_dreamsim(*args, **kwargs):
        device = kwargs.get("device", "cpu")
        return DreamSimStub().to(device), None

    trellis_image_to_3d.dreamsim = _stub_dreamsim


def _load_images_with_masks(
    image_dir: Path,
    *,
    mask_dir: Path,
    max_views: int,
    mask_background: str,
) -> tuple[list[Image.Image], list[str], list[dict[str, Any]]]:
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if max_views > 0:
        paths = paths[:max_views]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if mask_background == "black":
        bg = np.array([0, 0, 0], dtype=np.uint8)
    elif mask_background == "white":
        bg = np.array([255, 255, 255], dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported mask_background={mask_background!r}")

    images: list[Image.Image] = []
    names: list[str] = []
    summaries: list[dict[str, Any]] = []
    for path in paths:
        mask_path = mask_dir / f"{path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {path.name}: {mask_path}")
        image = Image.open(path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        rgb = np.asarray(image).copy()
        alpha = np.asarray(mask) > 127
        rgb[~alpha] = bg
        images.append(Image.fromarray(rgb, mode="RGB"))
        names.append(str(path))
        summaries.append(
            {
                "image": str(path),
                "mask": str(mask_path),
                "foreground_ratio": float(alpha.mean()),
                "foreground_pixels": int(alpha.sum()),
                "total_pixels": int(alpha.size),
            }
        )
    return images, names, summaries


def _coords_np(coords: torch.Tensor) -> np.ndarray:
    return coords.detach().cpu().numpy().astype(np.int32, copy=False)


def _coord_set(coords: np.ndarray) -> set[tuple[int, int, int, int]]:
    if coords.size == 0:
        return set()
    return {tuple(int(v) for v in row) for row in coords.reshape(-1, coords.shape[-1])}


def _component_stats(coords: np.ndarray) -> dict[str, float | int]:
    if coords.size == 0:
        return {
            "coord_count": 0,
            "component_count": 0,
            "largest_component_count": 0,
            "largest_component_ratio": 0.0,
        }
    spatial = coords[:, -3:].astype(np.int32, copy=False)
    points = {tuple(int(v) for v in row) for row in spatial}
    visited: set[tuple[int, int, int]] = set()
    component_count = 0
    largest = 0
    for point in points:
        if point in visited:
            continue
        component_count += 1
        stack = [point]
        visited.add(point)
        size = 0
        while stack:
            x, y, z = stack.pop()
            size += 1
            for nb in (
                (x - 1, y, z),
                (x + 1, y, z),
                (x, y - 1, z),
                (x, y + 1, z),
                (x, y, z - 1),
                (x, y, z + 1),
            ):
                if nb in points and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        largest = max(largest, size)
    count = int(len(points))
    return {
        "coord_count": int(coords.shape[0]),
        "unique_spatial_count": count,
        "component_count": int(component_count),
        "largest_component_count": int(largest),
        "largest_component_ratio": float(largest / max(1, count)),
    }


def _compare_coords(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    set_a = _coord_set(a)
    set_b = _coord_set(b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    only_a = len(set_a - set_b)
    only_b = len(set_b - set_a)
    return {
        "a_count": int(len(set_a)),
        "b_count": int(len(set_b)),
        "intersection": int(inter),
        "union": int(union),
        "only_a": int(only_a),
        "only_b": int(only_b),
        "iou": float(inter / max(1, union)),
        "a_keep_ratio": float(inter / max(1, len(set_a))),
        "b_keep_ratio": float(inter / max(1, len(set_b))),
    }


def _resolve_manifest_relative(path: str, roots: list[Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    for root in roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return roots[0] / candidate


def _load_prior_manifest_sample(manifest_path: Path, uid: str = "") -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") or []
    if not samples:
        raise ValueError(f"No samples found in prior manifest: {manifest_path}")
    if uid:
        matches = [sample for sample in samples if str(sample.get("uid", "")) == str(uid)]
        if not matches:
            raise ValueError(f"uid={uid!r} not found in prior manifest {manifest_path}")
        sample = matches[0]
    else:
        sample = samples[0]

    prior_root = Path(payload.get("prior_root") or manifest_path.parent)
    prior_npz = _resolve_manifest_relative(
        str(sample.get("prior_npz", "")),
        [prior_root, manifest_path.parent, Path(str(sample.get("dataset_root", ".")))],
    )
    if not prior_npz.exists():
        raise FileNotFoundError(f"Prior npz not found: {prior_npz}")
    prior_payload = np.load(prior_npz)
    if "prior_coords" in prior_payload:
        prior_coords = prior_payload["prior_coords"]
    elif "coords" in prior_payload:
        prior_coords = prior_payload["coords"]
    else:
        raise KeyError(f"No prior_coords/coords key in {prior_npz}")

    summary = {
        "manifest": str(manifest_path),
        "uid": str(sample.get("uid", "")),
        "prior_npz": str(prior_npz),
        "prior_coord_count": int(np.asarray(prior_coords).reshape(-1, np.asarray(prior_coords).shape[-1]).shape[0])
        if np.asarray(prior_coords).size
        else 0,
        "sample_dataset_root": str(sample.get("dataset_root", "")),
        "sample_sparse_subdir": str(sample.get("sparse_subdir", "")),
        "sample_frame_count": int(len(sample.get("frames") or [])),
    }
    return sample, np.asarray(prior_coords, dtype=np.int32), summary


def _load_adapter(
    *,
    ckpt: str,
    tokens: list[torch.Tensor],
    feature_dim: int,
    hidden_dim: int,
    layers: list[int],
    prefix_tokens: int,
    allow_partial: bool = False,
    gate_feature_index: int | None = None,
    gate_power: float = 1.0,
) -> tuple[ProjectionAwareSpatialTokenAdapter, dict[str, Any]]:
    adapter = ProjectionAwareSpatialTokenAdapter.from_tokens(
        tokens,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        layer_indices=layers,
        prefix_tokens=prefix_tokens,
        mode="bias",
        gate_feature_index=gate_feature_index,
        gate_power=gate_power,
    ).to(device=tokens[0].device)
    state = torch.load(ckpt, map_location=tokens[0].device)
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = adapter.load_state_dict(state_dict, strict=False)
    if (missing or unexpected) and not bool(allow_partial):
        raise RuntimeError(f"Adapter checkpoint mismatch: missing={list(missing)}, unexpected={list(unexpected)}")
    return adapter, {
        "path": str(ckpt),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "strict_load_passed": not (missing or unexpected),
        "allow_partial": bool(allow_partial),
        "metadata": state.get("metadata") if isinstance(state, dict) else None,
    }


def _scaled_tokens(base: list[torch.Tensor], adapted: list[torch.Tensor], scale: float) -> list[torch.Tensor]:
    if float(scale) == 1.0:
        return adapted
    out: list[torch.Tensor] = []
    for b, a in zip(base, adapted):
        out.append(b + (a - b) * float(scale))
    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="B3 eval-time ReconVGGT AR adapter injection smoke.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--pose_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--max_views", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--mask_dir", default="", help="Optional object mask dir. Use with --mask_mode apply.")
    parser.add_argument(
        "--mask_mode",
        choices=["none", "apply"],
        default="none",
        help="apply keeps full-frame geometry and blacks/whites out background; it does not crop.",
    )
    parser.add_argument("--mask_background", choices=["black", "white"], default="black")
    parser.add_argument("--adapter_ckpt", default="")
    parser.add_argument("--adapter_runtime_scale", type=float, default=1.0)
    parser.add_argument("--allow_partial_adapter_load", action="store_true")
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--adapter_layers", default="4,11,17,23")
    parser.add_argument("--adapter_gate_feature_index", type=int, default=None)
    parser.add_argument("--adapter_gate_power", type=float, default=1.0)
    parser.add_argument("--patch_start_idx", type=int, default=5)
    parser.add_argument("--image_resolution", type=int, default=518)
    parser.add_argument("--token_grid_side", type=int, default=37)
    parser.add_argument("--points3d_txt", default="")
    parser.add_argument("--point_prior_npz", default="")
    parser.add_argument("--colmap_sparse_dir", default="")
    parser.add_argument(
        "--mask_projection_mode",
        choices=["none", "filter_points", "token_mask", "filter_points_token_mask"],
        default="none",
    )
    parser.add_argument("--token_mask_min_ratio", type=float, default=0.05)
    parser.add_argument("--point_mask_support_min_views", type=int, default=0)
    parser.add_argument("--point_mask_support_min_ratio", type=float, default=0.0)
    parser.add_argument("--prior_manifest", default="", help="Optional real-SLAM manifest for AR prior/projection diagnostics.")
    parser.add_argument("--prior_uid", default="", help="Optional sample uid inside --prior_manifest.")
    parser.add_argument("--prior_radius", type=float, default=4.0)
    parser.add_argument("--projection_min_support_views", type=int, default=1)
    parser.add_argument("--projection_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--visual_hull_min_support_ratio", type=float, default=0.0)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--point_projection_rotation_mode", choices=["c2w", "w2c"], default="c2w")
    parser.add_argument("--point_projection_min_depth", type=float, default=1.0e-4)
    parser.add_argument("--default_fx", type=float, default=485.845947)
    parser.add_argument("--default_fy", type=float, default=485.744232)
    parser.add_argument("--default_cx", type=float, default=322.973236)
    parser.add_argument("--default_cy", type=float, default=237.599487)
    parser.add_argument("--default_image_width", type=int, default=640)
    parser.add_argument("--default_image_height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--ss_cfg_strength", type=float, default=7.5)
    parser.add_argument("--ss_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--ss_rescale_t", type=float, default=3.0)
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--slat_cfg_strength", type=float, default=7.5)
    parser.add_argument("--slat_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)
    parser.add_argument("--run_slat", action="store_true")
    parser.add_argument("--formats", default="mesh", help="Comma-separated decode formats used only with --run_slat.")
    parser.add_argument("--load_dreamsim", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    if not args.load_dreamsim:
        install_dreamsim_stub()
    print(f"[B3] loading pipeline pretrained={args.pretrained} device={device}", flush=True)
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
            raise ValueError("--preprocess is not supported with --mask_mode=apply; keep full-frame mask geometry.")
        images, image_names, mask_summaries = _load_images_with_masks(
            Path(args.image_dir),
            mask_dir=Path(args.mask_dir),
            max_views=args.max_views,
            mask_background=args.mask_background,
        )
    else:
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
    print(f"[B3] loaded {len(images)} images and {len(pose_records)} matched poses", flush=True)

    torch.manual_seed(int(args.seed))
    with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=getattr(pipeline, "VGGT_dtype", torch.float16)):
        aggregated_tokens_list, input_tensor = pipeline.vggt_feat(images)
    b, n, _, _ = aggregated_tokens_list[0].shape
    image_cond = normalize_image_cond(pipeline.encode_image(images), batch=b, views=n)

    projection_features = None
    point_projection_summary = None
    point_projection_source = None
    gate_feature_stats = None
    adapter_loaded = None
    tokens_for_cond = aggregated_tokens_list
    if args.adapter_ckpt:
        projection_features, _, point_projection_summary, point_projection_source = build_projection_features(
            args=args,
            image_names=image_names,
            pose_records=pose_records,
            token_device=aggregated_tokens_list[0].device,
            token_dtype=torch.float32,
        )
        gate_feature_stats = summarize_gate_feature(projection_features, args.adapter_gate_feature_index)
        adapter, adapter_loaded = _load_adapter(
            ckpt=args.adapter_ckpt,
            tokens=aggregated_tokens_list,
            feature_dim=int(projection_features.shape[-1]),
            hidden_dim=int(args.adapter_hidden_dim),
            layers=parse_layer_indices(args.adapter_layers),
            prefix_tokens=int(args.patch_start_idx),
            allow_partial=bool(args.allow_partial_adapter_load),
            gate_feature_index=args.adapter_gate_feature_index,
            gate_power=float(args.adapter_gate_power),
        )
        adapted_tokens = adapter(aggregated_tokens_list, projection_features)
        tokens_for_cond = _scaled_tokens(aggregated_tokens_list, adapted_tokens, float(args.adapter_runtime_scale))

    ss_image_cond = image_cond[:, :, int(args.patch_start_idx) :]
    ss_cond = pipeline.get_ss_cond(ss_image_cond, tokens_for_cond, int(args.num_samples))
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
    print("[B3] sampling sparse structure", flush=True)
    coords = pipeline.sample_sparse_structure(ss_cond, int(args.num_samples), ss_sampler_params, noise=ss_noise)
    coords_array = _coords_np(coords)
    np.savez_compressed(output_dir / "coords.npz", coords=coords_array)

    prior_summary = None
    prior_alignment = None
    if args.prior_manifest:
        print(f"[B3] computing prior/projection diagnostics from {args.prior_manifest}", flush=True)
        try:
            prior_sample, prior_coords, prior_summary = _load_prior_manifest_sample(Path(args.prior_manifest), args.prior_uid)
            prior_alignment = sparse_diagnostic_metrics(
                "b3_sparse",
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
        except Exception as exc:  # noqa: BLE001
            prior_alignment = {
                "b3_sparse_prior_alignment_error": repr(exc),
            }

    outputs_summary: dict[str, Any] = {}
    if args.run_slat:
        formats = [x.strip() for x in args.formats.split(",") if x.strip()]
        slat_sampler_params = {
            "steps": int(args.slat_steps),
            "cfg_strength": float(args.slat_cfg_strength),
            "cfg_interval": [0.6, 1.0],
            "guidance_rescale": float(args.slat_guidance_rescale),
            "rescale_t": float(args.slat_rescale_t),
        }
        print(f"[B3] sampling SLAT and decoding formats={formats}", flush=True)
        torch.manual_seed(int(args.seed))
        slat_cond = pipeline.get_slat_cond(image_cond, tokens_for_cond, int(args.num_samples))
        slat = pipeline.sample_slat(slat_cond, coords, slat_sampler_params)
        decoded = pipeline.decode_slat(slat, formats)
        if "mesh" in decoded:
            mesh = decoded["mesh"][0]
            outputs_summary["mesh"] = {
                "success": bool(getattr(mesh, "success", True)),
                "vertex_count": int(mesh.vertices.shape[0]),
                "face_count": int(mesh.faces.shape[0]),
            }
            try:
                mesh.to_trimesh().export(output_dir / "mesh.obj")
                outputs_summary["mesh"]["obj_path"] = str(output_dir / "mesh.obj")
            except Exception as exc:  # noqa: BLE001
                outputs_summary["mesh"]["obj_export_error"] = repr(exc)
        if "gaussian" in decoded:
            gs = decoded["gaussian"][0]
            outputs_summary["gaussian"] = {
                "point_count": int(getattr(gs, "_xyz").shape[0]) if hasattr(gs, "_xyz") else None,
            }
            try:
                gs.save_ply(str(output_dir / "gaussian.ply"))
                outputs_summary["gaussian"]["ply_path"] = str(output_dir / "gaussian.ply")
            except Exception as exc:  # noqa: BLE001
                outputs_summary["gaussian"]["ply_export_error"] = repr(exc)

    report = {
        "args": vars(args),
        "mode": "adapter" if args.adapter_ckpt else "baseline",
        "adapter_loaded": adapter_loaded,
        "adapter_runtime_scale": float(args.adapter_runtime_scale),
        "image_names": image_names,
        "mask_mode": str(args.mask_mode),
        "mask_dir": str(args.mask_dir) if args.mask_dir else None,
        "mask_background": str(args.mask_background),
        "mask_summaries": mask_summaries,
        "matched_pose_names": [r.image_name for r in pose_records],
        "input_tensor": tensor_summary(input_tensor),
        "image_cond_shape": list(image_cond.shape),
        "projection_features": summarize_pose_features(projection_features) if projection_features is not None else None,
        "gate_feature_stats": gate_feature_stats,
        "point_projection_source": point_projection_source,
        "point_projection_summary": point_projection_summary,
        "sparse": _component_stats(coords_array),
        "prior_summary": prior_summary,
        "prior_alignment": prior_alignment,
        "outputs": outputs_summary,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# ReconVGGT AR Adapter B3 Injection Smoke",
        "",
        f"- mode: `{report['mode']}`",
        f"- adapter: `{adapter_loaded}`",
        f"- adapter_runtime_scale: `{args.adapter_runtime_scale}`",
        f"- gate_feature_stats: `{gate_feature_stats}`",
        f"- images: `{len(image_names)}`",
        f"- sparse: `{report['sparse']}`",
        f"- prior_summary: `{prior_summary}`",
        f"- prior_alignment: `{prior_alignment}`",
        f"- outputs: `{outputs_summary}`",
        "",
        "This is an eval-time smoke test. It does not train ReconViaGen.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[B3] wrote {output_dir / 'report.json'}", flush=True)
    print(f"[B3] wrote {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
