#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


TRACKER_ROOT = Path(__file__).resolve().parents[2]


def load_manifest(data_root: Path, split: str) -> list[dict]:
    path = data_root / f"{split}.json"
    if not path.exists():
        path = data_root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", payload if isinstance(payload, list) else None)
    if not samples:
        raise ValueError(f"No samples found in {path}")
    return samples


def write_case(sample: dict, data_root: Path, output_root: Path, max_views: int) -> dict:
    npz_path = Path(sample["npz"])
    if not npz_path.is_absolute():
        npz_path = data_root / npz_path
    with np.load(npz_path) as data:
        images = data["images"]
        alpha = data["alpha"]
        intrinsics = data["intrinsics"].astype(np.float32)
        extrinsics = data["extrinsics"].astype(np.float32)
        source_glb = str(data["source_glb"]) if "source_glb" in data else sample.get("source_glb")
        uid = str(data["uid"]) if "uid" in data else sample["uid"]
        target_coords = data["target_coords"].astype(np.int32)
        normalize_center = data["normalize_center"].astype(float).tolist() if "normalize_center" in data else None
        normalize_scale = float(data["normalize_scale"]) if "normalize_scale" in data else None
        render_mode = str(data["render_mode"]) if "render_mode" in data else sample.get("render_mode", "unknown")

    if max_views > 0:
        images = images[:max_views]
        alpha = alpha[:max_views]
        intrinsics = intrinsics[:max_views]
        extrinsics = extrinsics[:max_views]

    safe_uid = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in uid)
    case_dir = output_root / safe_uid
    image_dir = case_dir / "images"
    mask_dir = case_dir / "masks"
    rgb_dir = case_dir / "rgb"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, (image, mask, k, c2w) in enumerate(zip(images, alpha, intrinsics, extrinsics)):
        image_name = f"frame_{i:04d}.jpg"
        mask_name = f"frame_{i:04d}.png"
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(image_dir / image_name, quality=95)
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(rgb_dir / image_name, quality=95)
        Image.fromarray(mask.astype(np.uint8), mode="L").save(mask_dir / mask_name)
        frames.append(
            {
                "image": image_name,
                "mask": mask_name,
                "intrinsic": k.tolist(),
                "extrinsic": c2w.tolist(),
            }
        )

    gt_dir = case_dir / "gt"
    gt_dir.mkdir(exist_ok=True)
    if source_glb and Path(source_glb).exists():
        target = gt_dir / Path(source_glb).name
        if not target.exists():
            shutil.copy2(source_glb, target)
        source_mesh_for_eval = str(target)
    else:
        source_mesh_for_eval = source_glb

    target_coords_path = gt_dir / "target_coords.npy"
    np.save(target_coords_path, target_coords)

    ar_manifest = {
        "dataset_dir": str(case_dir),
        "image_root": str(image_dir),
        "mask_root": str(mask_dir),
        "extrinsics_type": "c2w",
        "camera_convention": "synthetic OpenCV c2w, x-right y-down z-forward",
        "frames": frames,
    }
    ar_manifest_path = case_dir / "arpose_manifest.json"
    ar_manifest_path.write_text(json.dumps(ar_manifest, indent=2), encoding="utf-8")

    info = {
        "uid": uid,
        "dataset_dir": str(case_dir),
        "manifest": str(ar_manifest_path),
        "image_root": str(image_dir),
        "mask_root": str(mask_dir),
        "source_npz": str(npz_path),
        "source_glb": source_glb,
        "source_mesh_for_eval": source_mesh_for_eval,
        "target_coords": str(target_coords_path),
        "normalize_center": normalize_center,
        "normalize_scale": normalize_scale,
        "num_views": int(len(frames)),
        "render_mode": render_mode,
        "note": f"Objaverse held-out synthetic case exported from render_mode={render_mode}.",
    }
    (case_dir / "case_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return {
        "name": safe_uid,
        "dataset_dir": str(case_dir),
        "manifest": str(ar_manifest_path),
        "image_root": str(image_dir),
        "mask_root": str(mask_dir),
        "reference_coords": str(target_coords_path),
        "reference_mesh": source_mesh_for_eval,
        "note": info["note"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Generated objaverse_pose data root containing manifest/train/val JSON and samples/*.npz.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output_root", default=str(TRACKER_ROOT / "ar_pose_trellis" / "outputs" / "benchmarks" / "objaverse_holdout_cases"))
    parser.add_argument("--testsets_out", default=str(TRACKER_ROOT / "ar_pose_trellis" / "outputs" / "benchmarks" / "objaverse_holdout_testsets.json"))
    parser.add_argument("--max_cases", type=int, default=8)
    parser.add_argument("--max_views", type=int, default=8)
    parser.add_argument("--start_index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = load_manifest(data_root, args.split)
    selected = samples[args.start_index :]
    if args.max_cases > 0:
        selected = selected[: args.max_cases]
    datasets = [write_case(sample, data_root, output_root, args.max_views) for sample in selected]

    payload = {
        "description": "Objaverse held-out synthetic cases exported for AR-pose TRELLIS and ReconViaGen comparison.",
        "source_data_root": str(data_root),
        "split": args.split,
        "datasets": datasets,
    }
    out_path = Path(args.testsets_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[export_holdout] wrote {out_path}")
    print(f"[export_holdout] cases={len(datasets)} output_root={output_root}")


if __name__ == "__main__":
    main()
