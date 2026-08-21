#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime
import json
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_WHEEL_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_WHEEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_pose_trellis.pipeline import TrellisARPoseTo3DPipeline
from ar_pose_trellis.visual_hull import visual_hull_logit_bias


POSE_MODES = ("correct", "identity", "shuffle", "reverse", "cyclic_shift1", "cyclic_shift2", "noise", "large_noise")


def parse_indices(spec: str, size: int) -> list[int]:
    if str(spec).strip().lower() in {"", "all"}:
        return list(range(size))
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    bad = [idx for idx in out if idx < 0 or idx >= size]
    if bad:
        raise IndexError(f"indices out of range for dataset size {size}: {bad}")
    return out


def resolve_path(root: str | None, path: str) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute() or root is None:
        return path_obj
    return Path(root) / path_obj


def load_manifest(path: str) -> tuple[dict, list[dict]]:
    with open(path, "r") as f:
        payload = json.load(f)
    samples = payload.get("samples", payload if isinstance(payload, list) else None)
    if samples is None:
        raise ValueError(f"manifest has no samples list: {path}")
    return payload if isinstance(payload, dict) else {}, samples


def load_sample(payload: dict, sample: dict, max_frames: int):
    image_root = sample.get("image_root", payload.get("image_root"))
    mask_root = sample.get("mask_root", payload.get("mask_root"))
    latent_root = sample.get("latent_root", payload.get("latent_root"))
    top_intrinsic = sample.get("intrinsic", payload.get("intrinsic"))
    frames = sample["frames"][:max_frames] if max_frames > 0 else sample["frames"]

    images, masks, intrinsics, extrinsics = [], [], [], []
    for frame in frames:
        image = np.asarray(Image.open(resolve_path(image_root, frame["image"])).convert("RGB")).astype(np.float32) / 255.0
        mask_path = frame.get("mask", sample.get("mask"))
        if mask_path is not None:
            mask = Image.open(resolve_path(mask_root, mask_path)).convert("L")
            if mask.size != (image.shape[1], image.shape[0]):
                mask = mask.resize((image.shape[1], image.shape[0]), Image.NEAREST)
            mask_arr = np.asarray(mask).astype(np.float32) / 255.0
        else:
            mask_arr = np.ones(image.shape[:2], dtype=np.float32)
        intrinsic = frame.get("intrinsic", top_intrinsic)
        if intrinsic is None:
            raise ValueError(f"missing intrinsic for frame {frame.get('image')}")
        images.append(torch.from_numpy(image).permute(2, 0, 1))
        masks.append(torch.from_numpy(mask_arr[None]))
        intrinsics.append(torch.tensor(intrinsic, dtype=torch.float32))
        extrinsics.append(torch.tensor(frame["extrinsic"], dtype=torch.float32))

    latent_rel = sample.get("ss_latent", sample.get("ss_latent_path", sample.get("latent")))
    if latent_rel is None:
        raise ValueError(f"sample {sample.get('uid')} has no ss_latent")
    latent_path = resolve_path(latent_root, latent_rel)
    with np.load(latent_path) as latent:
        target_coords = latent["target_coords"].astype(np.int32)
    return {
        "uid": str(sample.get("uid", sample.get("id", ""))),
        "images": torch.stack(images, dim=0),
        "masks": torch.stack(masks, dim=0),
        "intrinsics": torch.stack(intrinsics, dim=0),
        "extrinsics": torch.stack(extrinsics, dim=0),
        "target_coords": target_coords,
        "latent_path": str(latent_path),
        "extrinsics_type": str(sample.get("extrinsics_type", payload.get("extrinsics_type", "c2w"))),
    }


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = torch.nn.functional.normalize(axis.float(), dim=0)
    x, y, z = axis
    c = torch.cos(angle.float())
    s = torch.sin(angle.float())
    one = torch.ones_like(c)
    zero = torch.zeros_like(c)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, device=axis.device, dtype=torch.float32)
    return c * eye + (one - c) * (axis[:, None] * axis[None, :]) + s * k


def perturb_c2w(c2w: torch.Tensor, seed: int, max_rot_deg: float, trans_scale: float) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    out = c2w.clone()
    for view_idx in range(c2w.shape[0]):
        axis = torch.randn(3, generator=gen, dtype=torch.float32, device=c2w.device)
        angle = (torch.rand((), generator=gen, device=c2w.device) * 2.0 - 1.0) * max_rot_deg * torch.pi / 180.0
        rot = axis_angle_to_matrix(axis, angle).to(dtype=c2w.dtype, device=c2w.device)
        trans = (torch.rand(3, generator=gen, device=c2w.device, dtype=c2w.dtype) * 2.0 - 1.0) * float(trans_scale)
        out[view_idx, :3, :3] = rot @ out[view_idx, :3, :3]
        out[view_idx, :3, 3] = out[view_idx, :3, 3] + trans
    return out


def apply_pose_mode(extrinsics: torch.Tensor, mode: str, extrinsics_type: str, seed: int) -> torch.Tensor:
    mode = mode.lower()
    if mode == "correct":
        return extrinsics
    if mode == "identity":
        return torch.eye(4, dtype=extrinsics.dtype).repeat(extrinsics.shape[0], 1, 1)
    if mode == "shuffle":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        perm = torch.randperm(extrinsics.shape[0], generator=gen)
        if torch.equal(perm, torch.arange(extrinsics.shape[0])):
            perm = torch.roll(perm, shifts=1)
        return extrinsics[perm]
    if mode == "reverse":
        return torch.flip(extrinsics, dims=[0])
    if mode in {"cyclic_shift1", "cyclic_shift2"}:
        shift = 1 if mode == "cyclic_shift1" else 2
        return torch.roll(extrinsics, shifts=shift, dims=0)
    if mode in {"noise", "large_noise"}:
        c2w = extrinsics if extrinsics_type == "c2w" else torch.linalg.inv(extrinsics)
        c2w = perturb_c2w(
            c2w,
            seed=seed,
            max_rot_deg=35.0 if mode == "noise" else 90.0,
            trans_scale=0.25 if mode == "noise" else 0.75,
        )
        return c2w if extrinsics_type == "c2w" else torch.linalg.inv(c2w)
    raise ValueError(f"unknown pose mode: {mode}")


def sparse_overlap_metrics(pred_coords: np.ndarray, target_coords: np.ndarray) -> dict:
    pred_xyz = pred_coords[:, -3:].astype(np.int32) if pred_coords.size else np.zeros((0, 3), dtype=np.int32)
    target_xyz = target_coords[:, -3:].astype(np.int32) if target_coords.size else np.zeros((0, 3), dtype=np.int32)
    pred_set = set(map(tuple, pred_xyz.tolist()))
    target_set = set(map(tuple, target_xyz.tolist()))
    inter = len(pred_set & target_set)
    union = len(pred_set | target_set)
    return {
        "pred_unique": len(pred_set),
        "target_unique": len(target_set),
        "intersection": inter,
        "iou": float(inter / union) if union else 0.0,
        "target_recall": float(inter / len(target_set)) if target_set else 0.0,
        "pred_precision": float(inter / len(pred_set)) if pred_set else 0.0,
    }


def parse_fixed_topk_specs(spec: str) -> list[str]:
    spec = str(spec or "").strip()
    if not spec or spec.lower() in {"none", "off", "false", "0"}:
        return []
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        low = part.lower()
        if low in {"target", "target_unique", "gt", "gt_unique"}:
            out.append("target_unique")
        else:
            value = int(part)
            if value <= 0:
                raise ValueError(f"fixed top-k must be > 0, got {part!r}")
            out.append(str(value))
    return out


def resolve_fixed_topk(spec: str, target_unique: int) -> int:
    if spec == "target_unique":
        return int(target_unique)
    return int(spec)


def selection_label(spec: str | None) -> str:
    if spec is None:
        return "threshold"
    return f"topk_{spec}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    out = {}
    for mode in sorted({row["pose_mode"] for row in rows}):
        mode_rows = [row for row in rows if row["pose_mode"] == mode]
        summary = {"count": len(mode_rows)}
        for key in ("iou", "target_recall", "pred_precision", "pred_unique", "target_unique", "intersection"):
            values = np.asarray([row[key] for row in mode_rows], dtype=np.float64)
            summary[f"{key}_mean"] = float(values.mean()) if values.size else 0.0
            summary[f"{key}_median"] = float(np.median(values)) if values.size else 0.0
        out[mode] = summary
    return out


def summarize_by_selection(rows: list[dict]) -> dict:
    out = {}
    for selection in sorted({row.get("selection_name", "threshold") for row in rows}):
        out[selection] = summarize([row for row in rows if row.get("selection_name", "threshold") == selection])
    return out


def paired_pose_analysis(rows: list[dict], metrics: tuple[str, ...] = ("iou", "target_recall")) -> dict:
    analysis = {}
    selections = sorted({row.get("selection_name", "threshold") for row in rows})
    for selection in selections:
        selection_rows = [row for row in rows if row.get("selection_name", "threshold") == selection]
        by_index: dict[int, dict[str, dict]] = defaultdict(dict)
        for row in selection_rows:
            by_index[int(row["index"])][str(row["pose_mode"])] = row

        selection_analysis = {}
        for metric in metrics:
            valid_items = {idx: item for idx, item in by_index.items() if "correct" in item}
            ranks = []
            top1_count = 0
            top1_modes = Counter()
            paired = {}
            for idx, item in valid_items.items():
                ordered = sorted(item.keys(), key=lambda mode: float(item[mode][metric]), reverse=True)
                if ordered:
                    top1_modes[ordered[0]] += 1
                if ordered and ordered[0] == "correct":
                    top1_count += 1
                if "correct" in ordered:
                    ranks.append(ordered.index("correct") + 1)

            wrong_modes = sorted({mode for item in valid_items.values() for mode in item.keys() if mode != "correct"})
            for wrong_mode in wrong_modes:
                diffs = []
                wins = 0
                ties = 0
                count = 0
                for item in valid_items.values():
                    if wrong_mode not in item:
                        continue
                    correct_value = float(item["correct"][metric])
                    wrong_value = float(item[wrong_mode][metric])
                    diff = correct_value - wrong_value
                    diffs.append(diff)
                    wins += int(correct_value > wrong_value)
                    ties += int(correct_value == wrong_value)
                    count += 1
                diffs_arr = np.asarray(diffs, dtype=np.float64)
                paired[wrong_mode] = {
                    "count": int(count),
                    "correct_wins": int(wins),
                    "ties": int(ties),
                    "win_rate": float(wins / count) if count else 0.0,
                    "mean_delta": float(diffs_arr.mean()) if diffs_arr.size else 0.0,
                    "median_delta": float(np.median(diffs_arr)) if diffs_arr.size else 0.0,
                }

            ranks_arr = np.asarray(ranks, dtype=np.float64)
            selection_analysis[metric] = {
                "count": int(len(valid_items)),
                "correct_top1": int(top1_count),
                "correct_top1_rate": float(top1_count / len(valid_items)) if valid_items else 0.0,
                "correct_rank_mean": float(ranks_arr.mean()) if ranks_arr.size else 0.0,
                "correct_rank_median": float(np.median(ranks_arr)) if ranks_arr.size else 0.0,
                "top1_modes": dict(top1_modes),
                "paired": paired,
            }
        analysis[selection] = selection_analysis
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--indices", default="0-15")
    parser.add_argument("--pose_modes", default="correct,identity,shuffle,noise")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--ss_guidance_strength", type=float, default=1.0)
    parser.add_argument("--ss_threshold", type=float, default=0.0)
    parser.add_argument("--ss_min_coords", type=int, default=0)
    parser.add_argument(
        "--fixed_topk",
        default="",
        help=(
            "Comma-separated exact top-k selections from decoder logits, e.g. "
            "'4096,8192,target_unique'. When set, threshold/min_coords are used only for metadata."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--condition_mode", choices=["ray", "projected"], default="projected")
    parser.add_argument("--projected_grid_resolution", type=int, default=16)
    parser.add_argument("--projected_min_support", type=float, default=0.5)
    parser.add_argument("--projected_min_support_ratio", type=float, default=0.15)
    parser.add_argument("--projected_grid_transform", choices=["identity", "pixal3d_rotation"], default="identity")
    parser.add_argument("--cond_fp16", action="store_true")
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument("--visual_hull_prior_weight", type=float, default=0.0)
    parser.add_argument("--visual_hull_mask_threshold", type=float, default=0.5)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this process. Check CUDA_VISIBLE_DEVICES, nvidia-smi, "
            "and that the command was not launched in a CPU-only shell."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, samples = load_manifest(args.manifest)
    indices = parse_indices(args.indices, len(samples))
    pose_modes = [mode.strip() for mode in args.pose_modes.split(",") if mode.strip()]
    bad_modes = [mode for mode in pose_modes if mode not in POSE_MODES]
    if bad_modes:
        raise ValueError(f"unsupported pose modes: {bad_modes}")
    fixed_topk_specs = parse_fixed_topk_specs(args.fixed_topk)

    pipeline = TrellisARPoseTo3DPipeline.from_pretrained(
        args.weights,
        checkpoint_path=args.checkpoint,
        device=args.device,
        cond_fp16=args.cond_fp16,
        condition_mode=args.condition_mode,
        projected_grid_resolution=args.projected_grid_resolution,
        projected_min_support=args.projected_min_support,
        projected_min_support_ratio=args.projected_min_support_ratio,
        projected_grid_transform=args.projected_grid_transform,
        apply_lora=True,
    )

    rows = []
    for sample_idx in indices:
        sample = load_sample(payload, samples[sample_idx], args.max_frames)
        images_pre, masks_pre, intr_pre = pipeline.prepare_inputs(
            sample["images"],
            intrinsics=sample["intrinsics"],
            masks=sample["masks"],
            crop_foreground=not args.no_crop,
            no_background=True,
        )
        extr_type = sample["extrinsics_type"].lower()
        for pose_order, pose_mode in enumerate(pose_modes):
            extrinsics = apply_pose_mode(
                sample["extrinsics"],
                pose_mode,
                extrinsics_type=extr_type,
                seed=args.seed + sample_idx * 101 + pose_order,
            ).to(pipeline.device).float()
            torch.manual_seed(args.seed)
            ss_cond = pipeline.encode_ss_condition(
                images_pre,
                intr_pre,
                extrinsics,
                masks=masks_pre,
                extrinsics_are_c2w=extr_type == "c2w",
                camera_forward_sign=args.camera_forward_sign,
                reference_relative_pose=False,
            )
            logit_prior = None
            logit_prior_stats = None
            if float(args.visual_hull_prior_weight) != 0.0:
                logit_prior, logit_prior_stats = visual_hull_logit_bias(
                    masks_pre,
                    intr_pre,
                    extrinsics,
                    extrinsics_are_c2w=extr_type == "c2w",
                    resolution=pipeline.sparse_logit_resolution,
                    mask_threshold=args.visual_hull_mask_threshold,
                    min_visible_views=args.visual_hull_min_visible_views,
                    weight=args.visual_hull_prior_weight,
                )
            logits, logit_stats = pipeline.sample_sparse_logits(
                ss_cond,
                num_samples=1,
                sampler_params={"steps": args.ss_steps, "cfg_strength": args.ss_guidance_strength},
                logit_prior=logit_prior,
                logit_prior_stats=logit_prior_stats,
            )
            target_metrics = sparse_overlap_metrics(
                np.zeros((0, 4), dtype=np.int32),
                sample["target_coords"],
            )
            select_specs: list[str | None] = fixed_topk_specs if fixed_topk_specs else [None]
            for topk_spec in select_specs:
                fixed_topk = None
                if topk_spec is not None:
                    fixed_topk = resolve_fixed_topk(topk_spec, target_metrics["target_unique"])
                coords, select_stats = pipeline.sparse_coords_from_logits(
                    logits,
                    threshold=args.ss_threshold,
                    min_coords=args.ss_min_coords,
                    fixed_topk=fixed_topk,
                )
                sparse_stats = {**select_stats, **logit_stats}
                pred = coords.detach().cpu().numpy().astype(np.int32)
                metrics = sparse_overlap_metrics(pred, sample["target_coords"])
                row = {
                    "index": sample_idx,
                    "uid": sample["uid"],
                    "pose_mode": pose_mode,
                    "selection_name": selection_label(topk_spec),
                    "topk_spec": topk_spec or "",
                    "latent_path": sample["latent_path"],
                    **metrics,
                    **sparse_stats,
                }
                rows.append(row)
                print(
                    "[ar_projected_pixal_eval] "
                    f"idx={sample_idx} pose={pose_mode} selection={row['selection_name']} "
                    f"iou={metrics['iou']:.4f} recall={metrics['target_recall']:.4f} "
                    f"precision={metrics['pred_precision']:.4f}",
                    flush=True,
                )

    summary = {
        "args": vars(args),
        "summary": summarize(rows),
        "summary_by_selection": summarize_by_selection(rows),
        "paired_analysis": paired_pose_analysis(rows),
        "rows": rows,
    }
    (output_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "report.csv", rows)
    print(json.dumps(summary["summary"], indent=2), flush=True)
    print(f"[ar_projected_pixal_eval] wrote {output_dir / 'report.json'}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        output_dir = None
        if "--output_dir" in sys.argv:
            idx = sys.argv.index("--output_dir")
            if idx + 1 < len(sys.argv):
                output_dir = Path(sys.argv[idx + 1])
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "argv": sys.argv,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            (output_dir / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            print(f"[ar_projected_pixal_eval] FAILED; wrote {output_dir / 'failure.json'}", flush=True)
        raise
