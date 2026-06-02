#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


TRACKER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = "/home/zjr/anaconda3/envs/reconviagen/bin/python"


def load_testsets(path: str) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    datasets = payload.get("datasets", payload if isinstance(payload, list) else None)
    if not datasets:
        raise ValueError(f"No datasets found in {path}")
    return datasets


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(TRACKER_ROOT), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testsets", default=str(TRACKER_ROOT / "ar_pose_trellis" / "benchmark" / "testsets.json"))
    parser.add_argument("--output_root", default=str(TRACKER_ROOT / "ar_pose_trellis" / "outputs" / "benchmarks" / "reconviagen"))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--manifest_root", default=str(TRACKER_ROOT / "ar_pose_trellis" / "outputs" / "benchmarks" / "manifests"))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--mesh_simplify", type=float, default=None)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    manifest_root = Path(args.manifest_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    for item in load_testsets(args.testsets):
        name = item["name"]
        dataset_dir = Path(item["dataset_dir"]).resolve() if item.get("dataset_dir") else None
        manifest_path = Path(item["manifest"]).resolve() if item.get("manifest") else manifest_root / f"{name}.json"
        if not item.get("manifest"):
            if dataset_dir is None:
                raise ValueError(f"{name} has neither dataset_dir nor manifest")
            manifest_cmd = [
                args.python,
                "ar_pose_trellis/coarse_dataset_to_ar_manifest.py",
                "--dataset_dir",
                str(dataset_dir),
                "--output",
                str(manifest_path),
            ]
            if args.max_frames > 0:
                manifest_cmd += ["--max_frames", str(args.max_frames)]
            run_command(manifest_cmd, args.dry_run)

        image_root = item.get("image_root") or (str(dataset_dir / "images") if dataset_dir is not None else None)
        mask_root = item.get("mask_root") or (str(dataset_dir / "masks") if dataset_dir is not None else None)
        if image_root is None or mask_root is None:
            raise ValueError(f"{name} needs image_root and mask_root")
        dataset_output_root = output_root / name
        dataset_output_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python,
            "ar_pose_trellis/benchmark/run_reconviagen_multiview.py",
            "--manifest",
            str(manifest_path),
            "--image_root",
            str(image_root),
            "--mask_root",
            str(mask_root),
            "--output_dir",
            str(dataset_output_root),
            "--resolution",
            str(args.resolution),
            "--seeds",
            args.seeds,
            "--max_frames",
            str(args.max_frames),
        ]
        if args.mesh_simplify is not None:
            cmd += ["--mesh_simplify", str(args.mesh_simplify)]
        if args.skip_video:
            cmd.append("--skip_video")
        run_command(cmd, args.dry_run)


if __name__ == "__main__":
    main()
