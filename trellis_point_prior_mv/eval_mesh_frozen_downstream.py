#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
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
from PIL import Image

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline  # noqa: E402

from trellis_point_prior_mv.common import (  # noqa: E402
    load_target_latent,
    parse_indices,
    resolve_path,
    sparse_overlap_metrics,
    write_json,
)
from trellis_point_prior_mv.eval_sparse_inpaint import topk_coords_from_logits  # noqa: E402


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


def apply_mask_and_crop(image_path: Path, mask_path: Path, resolution: int) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    rgba = np.asarray(image).copy()
    alpha = np.asarray(mask)
    rgba[:, :, 3] = np.where(alpha > 127, 255, 0).astype(np.uint8)
    rgba[:, :, :3] = rgba[:, :, :3] * (rgba[:, :, 3:4] > 0)
    ys, xs = np.nonzero(rgba[:, :, 3] > 0)
    if len(xs) == 0:
        side = max(image.size)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(Image.fromarray(rgba), ((side - image.width) // 2, (side - image.height) // 2))
        return canvas.resize((resolution, resolution), Image.Resampling.BILINEAR)

    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    size = max(1, int(max(right - left + 1, bottom - top + 1) * 1.1))
    crop = (
        max(0, int(center_x - size // 2)),
        max(0, int(center_y - size // 2)),
        min(image.width, int(center_x + size // 2)),
        min(image.height, int(center_y + size // 2)),
    )
    cropped = Image.fromarray(rgba).crop(crop)
    side = max(cropped.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2))
    return canvas.resize((resolution, resolution), Image.Resampling.BILINEAR)


def load_source_sample(point_payload: dict, point_sample: dict, cache: dict[str, tuple[dict, list[dict]]]) -> tuple[dict, dict]:
    source_manifest = str(point_sample.get("source_manifest") or point_payload.get("source_manifest"))
    if not source_manifest:
        raise ValueError(f"missing source_manifest for sample {point_sample.get('uid')}")
    if source_manifest not in cache:
        payload = load_json(source_manifest)
        samples = payload.get("samples", payload if isinstance(payload, list) else None)
        if samples is None:
            raise ValueError(f"source manifest has no samples: {source_manifest}")
        cache[source_manifest] = (payload if isinstance(payload, dict) else {}, samples)
    source_payload, source_samples = cache[source_manifest]
    source_index = int(point_sample.get("source_index", 0))
    return source_payload, source_samples[source_index]


def load_condition_images(
    point_payload: dict,
    point_sample: dict,
    source_cache: dict[str, tuple[dict, list[dict]]],
    *,
    max_frames: int,
    resolution: int,
) -> list[Image.Image]:
    source_payload, source_sample = load_source_sample(point_payload, point_sample, source_cache)
    image_root = source_sample.get("image_root", source_payload.get("image_root"))
    mask_root = source_sample.get("mask_root", source_payload.get("mask_root"))
    frames = source_sample.get("frames") or []
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"source sample {source_sample.get('uid')} has no frames")
    images = []
    for frame in frames:
        image_path = resolve_path(image_root, frame["image"])
        mask_rel = frame.get("mask", source_sample.get("mask"))
        if mask_rel is None:
            raise ValueError(f"frame {frame.get('image')} has no mask")
        mask_path = resolve_path(mask_root, mask_rel)
        images.append(apply_mask_and_crop(image_path, mask_path, resolution))
    return images


def coords_np_to_torch(coords: np.ndarray, device: torch.device, *, max_coords: int = 0, seed: int = 0) -> torch.Tensor:
    xyz = coords[:, -3:].astype(np.int32, copy=False) if coords.size else np.zeros((0, 3), dtype=np.int32)
    if xyz.size:
        valid = ((xyz >= 0) & (xyz < 64)).all(axis=1)
        xyz = xyz[valid]
        if xyz.shape[0] > 1:
            xyz = np.unique(xyz, axis=0)
        if max_coords > 0 and xyz.shape[0] > max_coords:
            rng = np.random.default_rng(seed)
            keep = rng.choice(xyz.shape[0], size=int(max_coords), replace=False)
            xyz = xyz[np.sort(keep)]
    batch = np.zeros((xyz.shape[0], 1), dtype=np.int32)
    coords4 = np.concatenate([batch, xyz.astype(np.int32)], axis=1)
    return torch.from_numpy(coords4).to(device=device, dtype=torch.int32)


def torch_coords_to_np(coords: torch.Tensor) -> np.ndarray:
    if coords.numel() == 0:
        return np.zeros((0, 4), dtype=np.int32)
    return coords.detach().cpu().numpy().astype(np.int32)


def prepare_cond(pipeline, images: list[Image.Image], mode: str) -> tuple[dict, int]:
    if mode == "first":
        return pipeline.get_cond([images[0]]), 1
    cond = pipeline.get_cond(images)
    if mode == "mean":
        return {
            "cond": cond["cond"].mean(dim=0, keepdim=True),
            "neg_cond": cond["neg_cond"][:1],
        }, 1
    if mode == "multi_stochastic":
        # Match TrellisImageTo3DPipeline.run_multi_image(): the positive
        # condition keeps all image tokens for sampler-side stochastic view
        # selection, while CFG negative condition must remain batch size 1.
        cond = {
            "cond": cond["cond"],
            "neg_cond": cond["neg_cond"][:1],
        }
        return cond, len(images)
    raise ValueError(f"unsupported cond_mode={mode!r}")


def validate_cond_shapes(cond: dict, cond_count: int, *, context: str) -> None:
    pos = cond.get("cond")
    neg = cond.get("neg_cond")
    if not isinstance(pos, torch.Tensor) or not isinstance(neg, torch.Tensor):
        raise TypeError(f"{context}: cond/neg_cond must be torch tensors")
    if pos.ndim != 3 or neg.ndim != 3:
        raise ValueError(f"{context}: expected [B,T,C] cond tensors, got cond={tuple(pos.shape)} neg={tuple(neg.shape)}")
    if cond_count > 1 and pos.shape[0] != cond_count:
        raise ValueError(f"{context}: cond_count={cond_count} but cond batch={pos.shape[0]}")
    if neg.shape[0] != 1:
        raise ValueError(f"{context}: neg_cond batch must be 1 for CFG, got {neg.shape[0]}")


def sample_stock_sparse(pipeline, cond: dict, cond_count: int, args: argparse.Namespace, seed: int) -> torch.Tensor:
    validate_cond_shapes(cond, cond_count, context="sample_stock_sparse")
    torch.manual_seed(seed)
    params = {
        "steps": int(args.ss_steps),
        "cfg_strength": float(args.ss_guidance_strength),
    }
    if args.cond_mode == "multi_stochastic" and cond_count > 1:
        with pipeline.inject_sampler_multi_image("sparse_structure_sampler", cond_count, int(args.ss_steps), mode="stochastic"):
            return pipeline.sample_sparse_structure(cond, 1, params)
    return pipeline.sample_sparse_structure(cond, 1, params)


def sample_slat_mesh(pipeline, cond: dict, cond_count: int, coords: torch.Tensor, args: argparse.Namespace, seed: int):
    validate_cond_shapes(cond, cond_count, context="sample_slat_mesh")
    torch.manual_seed(seed)
    params = {
        "steps": int(args.slat_steps),
        "cfg_strength": float(args.slat_guidance_strength),
    }
    if args.slat_guidance_rescale is not None:
        params["guidance_rescale"] = float(args.slat_guidance_rescale)
    if args.slat_rescale_t is not None:
        params["rescale_t"] = float(args.slat_rescale_t)
    if args.cond_mode == "multi_stochastic" and cond_count > 1:
        with pipeline.inject_sampler_multi_image("slat_sampler", cond_count, int(args.slat_steps), mode="stochastic"):
            slat = pipeline.sample_slat(cond, coords, params)
    else:
        slat = pipeline.sample_slat(cond, coords, params)
    return pipeline.decode_slat(slat, ["mesh"])["mesh"][0]


def mesh_basic_metrics(mesh) -> dict[str, Any]:
    verts = mesh.vertices.detach().float().cpu().numpy() if hasattr(mesh.vertices, "detach") else np.asarray(mesh.vertices)
    faces = mesh.faces.detach().cpu().numpy() if hasattr(mesh.faces, "detach") else np.asarray(mesh.faces)
    out: dict[str, Any] = {
        "mesh_success": int(bool(getattr(mesh, "success", verts.shape[0] > 0 and faces.shape[0] > 0))),
        "vertex_count": int(verts.shape[0]),
        "face_count": int(faces.shape[0]),
    }
    if verts.size:
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        extent = vmax - vmin
        out.update(
            {
                "bbox_min_x": float(vmin[0]),
                "bbox_min_y": float(vmin[1]),
                "bbox_min_z": float(vmin[2]),
                "bbox_max_x": float(vmax[0]),
                "bbox_max_y": float(vmax[1]),
                "bbox_max_z": float(vmax[2]),
                "extent_x": float(extent[0]),
                "extent_y": float(extent[1]),
                "extent_z": float(extent[2]),
                "extent_max": float(extent.max()),
                "extent_min": float(extent.min()),
                "extent_ratio": float(extent.min() / max(extent.max(), 1e-8)),
            }
        )
    return out


def coords_to_points(coords: np.ndarray, resolution: int = 64) -> np.ndarray:
    xyz = coords[:, -3:].astype(np.float32, copy=False) if coords.size else np.zeros((0, 3), dtype=np.float32)
    return (xyz + 0.5) / float(resolution) - 0.5


def mesh_target_distance_metrics(mesh, target_coords: np.ndarray, sample_count: int, seed: int) -> dict[str, float | int]:
    try:
        import trimesh
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover - dependency diagnostics
        return {"mesh_eval_enabled": 0, "mesh_eval_error": f"dependency_missing: {exc}"}

    tri = mesh.to_trimesh(transform_pose=False)
    if tri.vertices.shape[0] == 0 or tri.faces is None or len(tri.faces) == 0:
        return {"mesh_eval_enabled": 0, "mesh_eval_error": "empty_mesh"}
    rng = np.random.default_rng(seed)
    count = int(max(1, sample_count))
    try:
        pts, _ = trimesh.sample.sample_surface(tri, count)
    except Exception:
        verts = np.asarray(tri.vertices)
        ids = rng.choice(verts.shape[0], size=min(count, verts.shape[0]), replace=False)
        pts = verts[ids]
    target_pts = coords_to_points(target_coords)
    if target_pts.shape[0] == 0 or pts.shape[0] == 0:
        return {"mesh_eval_enabled": 0, "mesh_eval_error": "empty_points"}
    if target_pts.shape[0] > count:
        ids = rng.choice(target_pts.shape[0], size=count, replace=False)
        target_eval = target_pts[ids]
    else:
        target_eval = target_pts
    mesh_tree = cKDTree(pts)
    target_tree = cKDTree(target_eval)
    target_to_mesh = mesh_tree.query(target_eval, k=1)[0]
    mesh_to_target = target_tree.query(pts, k=1)[0]
    return {
        "mesh_eval_enabled": 1,
        "target_to_mesh_mean": float(np.mean(target_to_mesh)),
        "target_to_mesh_median": float(np.median(target_to_mesh)),
        "mesh_to_target_mean": float(np.mean(mesh_to_target)),
        "mesh_to_target_median": float(np.median(mesh_to_target)),
        "chamfer_l2_mean": float(np.mean(target_to_mesh**2) + np.mean(mesh_to_target**2)),
        "eval_target_points": int(target_eval.shape[0]),
        "eval_mesh_points": int(pts.shape[0]),
    }


def load_stage2_bundle(args: argparse.Namespace, pipeline, device: torch.device):
    if not args.stage2_checkpoint:
        return None
    from ar_pose_trellis.pipeline import apply_lora_to_ss_flow
    from trellis import models as trellis_models
    from trellis_point_prior_mv.common import SparsePointPriorCond
    from trellis_point_prior_mv.eval_sparse_inpaint_stage2 import (
        build_inpaint_condition,
        inject_known_logits,
        sample_latent_with_known_reinjection,
    )

    base_flow = copy.deepcopy(pipeline.models["sparse_structure_flow_model"]).to(device).eval()
    flow = apply_lora_to_ss_flow(base_flow, r=args.lora_rank, alpha=args.lora_alpha).to(device).eval()
    encoder = trellis_models.from_pretrained(
        f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
        if os.path.isdir(args.weights)
        else f"{args.weights}/ckpts/ss_enc_conv3d_16l8_fp16"
    ).to(device).eval()
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    ss_cond = SparsePointPriorCond(
        latent_channels=args.latent_channels,
        cond_channels=args.cond_channels,
        grid_resolution=args.latent_grid_resolution,
    ).to(device).eval()
    state = torch.load(args.stage2_checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    flow_state = {k.replace("ss_flow_model.", "", 1): v for k, v in state.items() if k.startswith("ss_flow_model.")}
    cond_state = {k.replace("ss_cond.", "", 1): v for k, v in state.items() if k.startswith("ss_cond.")}
    if flow_state:
        missing, unexpected = flow.load_state_dict(flow_state, strict=False)
        print(f"[mesh_frozen] stage2 flow load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    missing, unexpected = ss_cond.load_state_dict(cond_state, strict=False)
    print(f"[mesh_frozen] stage2 cond load missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return {
        "flow": flow,
        "encoder": encoder,
        "decoder": decoder,
        "ss_cond": ss_cond,
        "build_inpaint_condition": build_inpaint_condition,
        "sample_latent_with_known_reinjection": sample_latent_with_known_reinjection,
        "inject_known_logits": inject_known_logits,
    }


def parse_stage2_topk_specs(args: argparse.Namespace) -> list[str]:
    raw = args.stage2_topk_specs or args.stage2_topk
    specs = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not specs:
        raise ValueError("stage2 top-k spec list is empty")
    return specs


def stage2_topk_label(spec: str) -> str:
    label = re.sub(r"[^0-9A-Za-z]+", "_", spec.strip()).strip("_").lower()
    return label or "topk"


def _parse_int_suffix(text: str, prefix: str) -> int:
    value = text[len(prefix) :]
    if not value:
        raise ValueError(f"missing value in top-k option {text!r}")
    return int(value)


def resolve_stage2_topk(spec: str, target_unique: int) -> tuple[int, dict[str, Any]]:
    raw = spec.strip()
    if not raw:
        raise ValueError("empty stage2 top-k spec")
    normalized = raw.lower().replace("target_unique", "targetunique")
    parts = [p for p in re.split(r"[@_+]", normalized) if p]
    base = parts[0]
    cap = None
    min_k = None
    for part in parts[1:]:
        if part.startswith("cap"):
            cap = _parse_int_suffix(part, "cap")
        elif part.startswith("min"):
            min_k = _parse_int_suffix(part, "min")
        else:
            raise ValueError(f"unsupported stage2 top-k option {part!r} in spec {raw!r}")

    ratio = None
    base_kind = "absolute"
    if base in {"target", "targetunique", "target_unique", "tu"}:
        topk = int(target_unique)
        base_kind = "target_unique"
        ratio = 1.0
    elif base.startswith("ratio"):
        ratio = float(base[len("ratio") :])
        topk = int(round(float(target_unique) * ratio))
        base_kind = "ratio"
    elif base.startswith("r") and len(base) > 1:
        ratio = float(base[1:])
        topk = int(round(float(target_unique) * ratio))
        base_kind = "ratio"
    elif base.startswith("x") and len(base) > 1:
        ratio = float(base[1:])
        topk = int(round(float(target_unique) * ratio))
        base_kind = "ratio"
    elif base.endswith("x") and len(base) > 1:
        ratio = float(base[:-1])
        topk = int(round(float(target_unique) * ratio))
        base_kind = "ratio"
    elif base.startswith("abs"):
        topk = int(base[len("abs") :])
        ratio = float(topk) / max(float(target_unique), 1.0)
    else:
        topk = int(base)
        ratio = float(topk) / max(float(target_unique), 1.0)

    if min_k is not None:
        topk = max(topk, int(min_k))
    if cap is not None:
        topk = min(topk, int(cap))
    topk = max(1, min(int(topk), 64**3))
    info = {
        "stage2_topk_spec": raw,
        "stage2_topk_label": stage2_topk_label(raw),
        "stage2_topk": int(topk),
        "stage2_topk_base_kind": base_kind,
        "stage2_topk_base_ratio": None if ratio is None else float(ratio),
        "stage2_topk_cap": cap,
        "stage2_topk_min": min_k,
        "stage2_topk_ratio_to_target": float(topk) / max(float(target_unique), 1.0),
    }
    return topk, info


def expand_modes_with_stage2_specs(raw_modes: list[str], stage2_specs: list[str]) -> tuple[list[str], dict[str, str]]:
    expanded: list[str] = []
    stage2_mode_specs: dict[str, str] = {}
    multi_specs = len(stage2_specs) > 1
    for mode in raw_modes:
        if mode == "stage2_correct" and multi_specs:
            used: set[str] = set()
            for spec in stage2_specs:
                label = stage2_topk_label(spec)
                name = f"stage2_correct_{label}"
                if name in used:
                    raise ValueError(f"duplicate stage2 top-k label {label!r}; use distinct specs")
                used.add(name)
                expanded.append(name)
                stage2_mode_specs[name] = spec
        else:
            expanded.append(mode)
            if mode == "stage2_correct":
                stage2_mode_specs[mode] = stage2_specs[0]
    return expanded, stage2_mode_specs


def sample_stage2_logits(bundle: dict, pipeline, prior_coords: np.ndarray, prior_conf: np.ndarray, args: argparse.Namespace, device: torch.device, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    flow = bundle["flow"]
    cond, partial_latent, latent_mask, confidence = bundle["build_inpaint_condition"](
        bundle["encoder"],
        bundle["ss_cond"],
        prior_coords,
        prior_conf,
        device,
        known_conf_power=args.known_conf_power,
    )
    noise = torch.randn(1, flow.in_channels, int(flow.resolution), int(flow.resolution), int(flow.resolution), device=device)
    latent = bundle["sample_latent_with_known_reinjection"](
        pipeline,
        flow,
        cond,
        noise,
        partial_latent,
        latent_mask,
        confidence,
        args,
    )
    logits = bundle["decoder"](latent)
    if logits.shape[1] != 1:
        logits = logits.max(dim=1, keepdim=True).values
    return bundle["inject_known_logits"](logits.float(), prior_coords, args.known_logit_boost)


def coords_from_stage2_logits(logits: torch.Tensor, topk: int, args: argparse.Namespace, device: torch.device, seed: int) -> torch.Tensor:
    pred = topk_coords_from_logits(logits, topk)
    return coords_np_to_torch(pred, device, max_coords=args.max_coords, seed=seed)


def summarize(rows: list[dict]) -> dict:
    out: dict[str, Any] = {"count": len(rows), "by_mode": {}}
    numeric_keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float))})
    for mode in sorted({row["mode"] for row in rows}):
        rr = [row for row in rows if row["mode"] == mode]
        out["by_mode"][mode] = {"count": len(rr)}
        for key in numeric_keys:
            vals = [float(r[key]) for r in rr if key in r and isinstance(r[key], (int, float))]
            if vals:
                out["by_mode"][mode][f"{key}_mean"] = float(np.mean(vals))
                out["by_mode"][mode][f"{key}_median"] = float(np.median(vals))
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    point_payload = load_json(args.manifest)
    samples = point_payload["samples"]
    indices = parse_indices(args.indices, len(samples))
    raw_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    stage2_specs = parse_stage2_topk_specs(args)
    modes, stage2_mode_specs = expand_modes_with_stage2_specs(raw_modes, stage2_specs)
    torch.manual_seed(args.seed)

    print(f"[mesh_frozen] loading pipeline weights={args.weights}", flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.weights)
    pipeline.to(device)
    pipeline.low_vram = False
    pipeline.models["sparse_structure_decoder"].to(device)

    source_cache: dict[str, tuple[dict, list[dict]]] = {}
    stage2_bundle = load_stage2_bundle(args, pipeline, device) if stage2_mode_specs else None
    rows: list[dict] = []
    for order, sample_idx in enumerate(indices):
        point_sample = samples[sample_idx]
        uid = str(point_sample.get("uid", sample_idx))
        sample_dir = output_dir / f"{sample_idx:04d}_{uid[:12]}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        images = load_condition_images(
            point_payload,
            point_sample,
            source_cache,
            max_frames=args.max_frames,
            resolution=args.resolution,
        )
        cond, cond_count = prepare_cond(pipeline, images, args.cond_mode)
        _z, target_coords = load_target_latent(point_sample["ss_latent"])
        target_unique = len({tuple(x) for x in target_coords[:, -3:].astype(np.int32).tolist()})
        with np.load(resolve_path(point_payload.get("prior_root"), point_sample["prior_npz"])) as data:
            prior_coords = np.asarray(data["prior_coords"], dtype=np.int64)
            prior_conf = np.asarray(data["prior_conf"], dtype=np.float32) if "prior_conf" in data else np.ones((prior_coords.shape[0],), dtype=np.float32)

        coord_bank: dict[str, torch.Tensor] = {}
        stage2_topk_info: dict[str, dict[str, Any]] = {}
        stage2_logits = None
        stage2_seed = int(args.seed + sample_idx * 1009 + 991)
        for mode in modes:
            mode_seed = int(args.seed + sample_idx * 1009 + len(coord_bank) * 17)
            if mode == "target_sparse":
                coords = coords_np_to_torch(target_coords, device, max_coords=args.max_coords, seed=mode_seed)
            elif mode == "prior_sparse":
                coords = coords_np_to_torch(prior_coords, device, max_coords=args.max_coords, seed=mode_seed)
            elif mode == "stock_sparse":
                coords = sample_stock_sparse(pipeline, cond, cond_count, args, mode_seed)
                if args.max_coords > 0 and coords.shape[0] > args.max_coords:
                    coords = coords_np_to_torch(torch_coords_to_np(coords), device, max_coords=args.max_coords, seed=mode_seed)
            elif mode in stage2_mode_specs:
                if stage2_bundle is None:
                    raise ValueError("stage2_correct requires --stage2_checkpoint")
                if stage2_logits is None:
                    stage2_logits = sample_stage2_logits(stage2_bundle, pipeline, prior_coords, prior_conf, args, device, stage2_seed)
                topk, info = resolve_stage2_topk(stage2_mode_specs[mode], target_unique)
                coords = coords_from_stage2_logits(stage2_logits, topk, args, device, mode_seed)
                stage2_topk_info[mode] = info
            else:
                raise ValueError(f"unsupported mode={mode!r}")
            if coords.shape[0] == 0:
                print(f"[mesh_frozen] skip empty coords sample={sample_idx} mode={mode}", flush=True)
                continue
            coord_bank[mode] = coords

        for mode, coords in coord_bank.items():
            mode_dir = sample_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            coords_np = torch_coords_to_np(coords)
            np.savez_compressed(mode_dir / "sparse_coords.npz", coords=coords_np)
            sparse_metrics = sparse_overlap_metrics(coords_np, target_coords)
            print(
                f"[mesh_frozen] sample={sample_idx} uid={uid} mode={mode} "
                f"coords={coords_np.shape[0]} sparse_iou={sparse_metrics['iou']:.4f}",
                flush=True,
            )
            mesh = sample_slat_mesh(pipeline, cond, cond_count, coords, args, int(args.seed + sample_idx * 3571))
            tri = mesh.to_trimesh(transform_pose=bool(args.transform_mesh_pose))
            obj_path = mode_dir / "mesh.obj"
            tri.export(obj_path)
            metrics = {
                "sample_order": order,
                "sample_index": sample_idx,
                "uid": uid,
                "mode": mode,
                "image_count": len(images),
                "coord_count": int(coords_np.shape[0]),
                "sparse_iou": sparse_metrics["iou"],
                "sparse_target_recall": sparse_metrics["target_recall"],
                "sparse_pred_precision": sparse_metrics["pred_precision"],
                "sparse_intersection": sparse_metrics["intersection"],
                "target_unique": sparse_metrics["target_unique"],
                "mesh_obj": str(obj_path),
                **mesh_basic_metrics(mesh),
                **mesh_target_distance_metrics(mesh, target_coords, args.mesh_eval_samples, int(args.seed + sample_idx * 7919)),
            }
            if mode in stage2_topk_info:
                metrics.update(stage2_topk_info[mode])
            rows.append(metrics)
            write_json(mode_dir / "metrics.json", metrics)
            torch.cuda.empty_cache()

    report = {
        "args": vars(args),
        "rows": rows,
        "summary": summarize(rows),
    }
    write_json(output_dir / "report.json", report)
    write_csv(output_dir / "report.csv", rows)
    print(f"[mesh_frozen] wrote {output_dir / 'report.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen stock slat/mesh downstream eval for alternate sparse coords.")
    parser.add_argument("--manifest", required=True, help="Point-prior manifest with source_manifest/source_index fields.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--indices", default="0")
    parser.add_argument("--modes", default="target_sparse,prior_sparse,stock_sparse")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--cond_mode", choices=["first", "mean", "multi_stochastic"], default="multi_stochastic")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--ss_guidance_strength", type=float, default=7.5)
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--slat_guidance_strength", type=float, default=7.5)
    parser.add_argument("--slat_guidance_rescale", type=float, default=0.5)
    parser.add_argument("--slat_rescale_t", type=float, default=3.0)
    parser.add_argument("--max_coords", type=int, default=0, help="Optional random coord cap to avoid OOM. 0 keeps all coords.")
    parser.add_argument("--mesh_eval_samples", type=int, default=8000)
    parser.add_argument("--transform_mesh_pose", action="store_true", help="Export mesh with TRELLIS z-up to y-up transform.")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--stage2_checkpoint", default=None)
    parser.add_argument(
        "--stage2_topk",
        default="target_unique",
        help="Backward-compatible single stage2 top-k spec. Supports target_unique, int, r0.75, r1.0_cap12000.",
    )
    parser.add_argument(
        "--stage2_topk_specs",
        default=None,
        help="Comma-separated stage2 top-k specs. If multiple specs are given, stage2_correct is expanded into one mode per spec.",
    )
    parser.add_argument("--known_latent_clamp_strength", type=float, default=1.0)
    parser.add_argument("--known_clamp_start_t", type=float, default=1.0)
    parser.add_argument("--known_logit_boost", type=float, default=0.0)
    parser.add_argument("--known_conf_power", type=float, default=1.0)
    parser.add_argument("--known_use_confidence", action="store_true")
    parser.add_argument("--clamp_initial_noise", dest="clamp_initial_noise", action="store_true", default=True)
    parser.add_argument("--no_clamp_initial_noise", dest="clamp_initial_noise", action="store_false")
    parser.add_argument("--guidance_strength", type=float, default=1.0, help="Stage2 sparse guidance strength.")
    parser.add_argument("--steps", type=int, default=12, help="Stage2 sparse sampling steps.")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--latent_channels", type=int, default=8)
    parser.add_argument("--latent_grid_resolution", type=int, default=16)
    parser.add_argument("--cond_channels", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    main()
