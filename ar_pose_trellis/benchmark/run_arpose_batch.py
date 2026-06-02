#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    parser.add_argument("--output_root", default=str(TRACKER_ROOT / "ar_pose_trellis" / "benchmark_outputs" / "arpose"))
    parser.add_argument("--manifest_root", default=str(TRACKER_ROOT / "ar_pose_trellis" / "benchmark_outputs" / "manifests"))
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--ss_steps", type=int, default=12)
    parser.add_argument("--slat_steps", type=int, default=12)
    parser.add_argument("--ss_guidance_strength", type=float, default=1.0)
    parser.add_argument("--slat_guidance_strength", type=float, default=3.0)
    parser.add_argument(
        "--ss_min_coords",
        type=int,
        default=4096,
        help="Top-k sparse fallback count. Use 0 to measure raw sparse failures without fallback.",
    )
    parser.add_argument("--preview_frames", type=int, default=72)
    parser.add_argument("--preview_resolution", type=int, default=320)
    parser.add_argument("--preview_fps", type=int, default=15)
    parser.add_argument("--cond_fp16", action="store_true")
    parser.add_argument("--pose_only", action="store_true")
    parser.add_argument("--image_only", action="store_true")
    parser.add_argument("--visual_hull_prior_weight", type=float, default=0.0)
    parser.add_argument("--visual_hull_mask_threshold", type=float, default=0.5)
    parser.add_argument("--visual_hull_min_visible_views", type=int, default=1)
    parser.add_argument("--skip_glb", action="store_true")
    parser.add_argument("--skip_preview", action="store_true")
    parser.add_argument("--only_sparse", action="store_true")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with later datasets if one ARPose generation command fails.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = load_testsets(args.testsets)
    output_root = Path(args.output_root)
    manifest_root = Path(args.manifest_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    failures = []

    for item in datasets:
        name = item["name"]
        dataset_dir = Path(item["dataset_dir"]).resolve() if item.get("dataset_dir") else None
        manifest_path = Path(item["manifest"]).resolve() if item.get("manifest") else manifest_root / f"{name}.json"
        output_dir = output_root / name

        if item.get("manifest"):
            print(f"[run_arpose_batch] using existing manifest for {name}: {manifest_path}", flush=True)
        else:
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

        gen_cmd = [
            args.python,
            "ar_pose_trellis/generate_ar_pose_mesh.py",
            "--weights",
            args.weights,
            "--checkpoint",
            args.checkpoint,
            "--manifest",
            str(manifest_path),
            "--image_root",
            str(image_root),
            "--mask_root",
            str(mask_root),
            "--output_dir",
            str(output_dir),
            "--max_frames",
            str(args.max_frames),
            "--ss_steps",
            str(args.ss_steps),
            "--slat_steps",
            str(args.slat_steps),
            "--ss_guidance_strength",
            str(args.ss_guidance_strength),
            "--slat_guidance_strength",
            str(args.slat_guidance_strength),
            "--ss_min_coords",
            str(args.ss_min_coords),
            "--preview_frames",
            str(args.preview_frames),
            "--preview_resolution",
            str(args.preview_resolution),
            "--preview_fps",
            str(args.preview_fps),
        ]
        if args.cond_fp16:
            gen_cmd.append("--cond_fp16")
        if args.pose_only:
            gen_cmd.append("--pose_only")
        if args.image_only:
            gen_cmd.append("--image_only")
        if float(args.visual_hull_prior_weight) != 0.0:
            gen_cmd.extend(
                [
                    "--visual_hull_prior_weight",
                    str(args.visual_hull_prior_weight),
                    "--visual_hull_mask_threshold",
                    str(args.visual_hull_mask_threshold),
                    "--visual_hull_min_visible_views",
                    str(args.visual_hull_min_visible_views),
                ]
            )
        if args.skip_glb:
            gen_cmd.append("--skip_glb")
        if args.skip_preview:
            gen_cmd.append("--skip_preview")
        if args.only_sparse:
            gen_cmd.append("--only_sparse")
        try:
            run_command(gen_cmd, args.dry_run)
        except subprocess.CalledProcessError as exc:
            failure = {
                "name": name,
                "returncode": exc.returncode,
                "command": exc.cmd,
                "output_dir": str(output_dir),
            }
            failures.append(failure)
            print(f"[run_arpose_batch] FAILED {name}: returncode={exc.returncode}", flush=True)
            if not args.continue_on_error:
                raise

    if failures:
        failure_path = output_root / "arpose_batch_failures.json"
        failure_path.write_text(json.dumps({"failures": failures}, indent=2), encoding="utf-8")
        print(f"[run_arpose_batch] failures written: {failure_path}", flush=True)
        if not args.continue_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
