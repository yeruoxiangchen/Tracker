#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.astype(np.float64)
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qz * qx + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qz * qx - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_cameras(path: Path) -> dict[int, dict]:
    cameras = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(x) for x in parts[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            f, cx, cy = params[:3]
            fx = fy = f
        else:
            raise ValueError(f"Unsupported camera model {model} in {path}")
        cameras[camera_id] = {
            "model": model,
            "width": width,
            "height": height,
            "intrinsic": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        }
    return cameras


def read_images(path: Path, cameras: dict[int, dict]) -> list[dict]:
    frames = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = parts[9]

        r_w2c = qvec_to_rotmat(qvec)
        t_w2c = tvec.reshape(3, 1)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = r_w2c
        w2c[:3, 3:4] = t_w2c
        c2w = np.linalg.inv(w2c)

        cam = cameras[camera_id]
        frames.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "image": name,
                "mask": Path(name).with_suffix(".png").name,
                "intrinsic": cam["intrinsic"],
                "extrinsic": c2w.tolist(),
                "width": cam["width"],
                "height": cam["height"],
            }
        )
        # Skip POINTS2D line if present.
        if i < len(lines) and not lines[i].strip().startswith("#"):
            i += 1
    return sorted(frames, key=lambda f: f["image"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/home/zjr/Tracker/CoarseModel/datasets/GOOD_MESH_TEST")
    parser.add_argument("--sparse_dir", default="")
    parser.add_argument("--output", default="/home/zjr/Tracker/ar_pose_trellis/test_manifests/GOOD_MESH_TEST_arpose.json")
    parser.add_argument("--image_dir_name", default="images")
    parser.add_argument("--mask_dir_name", default="masks")
    parser.add_argument("--max_frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    sparse_dir = Path(args.sparse_dir).resolve() if args.sparse_dir else dataset_dir / "sparse" / "0"
    cameras = read_cameras(sparse_dir / "cameras.txt")
    frames = read_images(sparse_dir / "images.txt", cameras)

    image_dir = dataset_dir / args.image_dir_name
    mask_dir = dataset_dir / args.mask_dir_name
    filtered = []
    missing = []
    for frame in frames:
        image_path = image_dir / frame["image"]
        mask_path = mask_dir / frame["mask"]
        if not image_path.exists() or not mask_path.exists():
            missing.append(frame["image"])
            continue
        filtered.append(frame)
    if args.max_frames > 0:
        filtered = filtered[: args.max_frames]
    if not filtered:
        raise ValueError(f"No valid image/mask frames found in {dataset_dir}")

    payload = {
        "dataset_dir": str(dataset_dir),
        "image_root": str(image_dir),
        "mask_root": str(mask_dir),
        "extrinsics_type": "c2w",
        "camera_convention": "COLMAP converted to c2w, x-right y-down z-forward",
        "frames": filtered,
        "missing": missing,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[coarse2ar] wrote {output}")
    print(f"[coarse2ar] frames={len(filtered)} missing={len(missing)}")


if __name__ == "__main__":
    main()
