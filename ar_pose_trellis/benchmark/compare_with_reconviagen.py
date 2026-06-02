#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path


TRACKER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = "/home/zjr/anaconda3/envs/reconviagen/bin/python"


def run_command(cmd: list[str], dry_run: bool, env: dict[str, str], continue_on_error: bool = False) -> bool:
    print("\n[compare] " + " ".join(cmd), flush=True)
    if dry_run:
        return True
    try:
        subprocess.run(cmd, cwd=str(TRACKER_ROOT), env=env, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[compare] FAILED returncode={exc.returncode}: {' '.join(cmd)}", flush=True)
        if not continue_on_error:
            raise
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AR-pose TRELLIS and direct ReconViaGen on the same testsets, then evaluate meshes."
    )
    parser.add_argument("--testsets", required=True, help="Benchmark testsets JSON.")
    parser.add_argument("--checkpoint", default=None, help="AR-pose TRELLIS checkpoint. Required unless --skip_arpose.")
    parser.add_argument("--output_root", default="", help="Comparison output root. Defaults to benchmark_outputs/compare_<timestamp>.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--weights", default="microsoft/TRELLIS-image-large")

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
    parser.add_argument("--cond_fp16", action="store_true")
    parser.add_argument("--skip_glb", action="store_true")
    parser.add_argument("--skip_preview", action="store_true")

    parser.add_argument("--recon_resolution", type=int, default=518)
    parser.add_argument("--recon_seeds", default="0")
    parser.add_argument("--recon_mesh_simplify", type=float, default=None)
    parser.add_argument("--recon_skip_video", action="store_true", help="Do not render ReconViaGen preview videos.")
    parser.add_argument("--recon_render_video", action="store_true", help="Render ReconViaGen preview videos.")

    parser.add_argument("--sample_points", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--skip_arpose", action="store_true")
    parser.add_argument("--skip_reconviagen", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue later stages if a generation stage fails. Missing methods are skipped by evaluation.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_arpose and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --skip_arpose is set")

    if args.output_root:
        output_root = Path(args.output_root)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = TRACKER_ROOT / "ar_pose_trellis" / "benchmark_outputs" / f"compare_{stamp}"
    arpose_root = output_root / "arpose"
    recon_root = output_root / "reconviagen"
    eval_root = output_root / "eval"
    manifest_root = output_root / "manifests"

    env = os.environ.copy()
    env.setdefault("ATTN_BACKEND", "flash_attn")
    env.setdefault("SPCONV_ALGO", "native")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    print(f"[compare] output_root={output_root}", flush=True)

    if not args.skip_arpose:
        cmd = [
            args.python,
            "ar_pose_trellis/benchmark/run_arpose_batch.py",
            "--testsets",
            args.testsets,
            "--output_root",
            str(arpose_root),
            "--manifest_root",
            str(manifest_root),
            "--weights",
            args.weights,
            "--checkpoint",
            args.checkpoint,
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
        ]
        if args.cond_fp16:
            cmd.append("--cond_fp16")
        if args.skip_glb:
            cmd.append("--skip_glb")
        if args.skip_preview:
            cmd.append("--skip_preview")
        if args.continue_on_error:
            cmd.append("--continue_on_error")
        run_command(cmd, args.dry_run, env, args.continue_on_error)

    if not args.skip_reconviagen:
        cmd = [
            args.python,
            "ar_pose_trellis/benchmark/run_reconviagen_batch.py",
            "--testsets",
            args.testsets,
            "--output_root",
            str(recon_root),
            "--manifest_root",
            str(manifest_root),
            "--resolution",
            str(args.recon_resolution),
            "--seeds",
            args.recon_seeds,
            "--max_frames",
            str(args.max_frames),
        ]
        if args.recon_mesh_simplify is not None:
            cmd += ["--mesh_simplify", str(args.recon_mesh_simplify)]
        if args.recon_skip_video and args.recon_render_video:
            raise ValueError("Use only one of --recon_skip_video and --recon_render_video")
        if args.recon_skip_video or not args.recon_render_video:
            cmd.append("--skip_video")
        run_command(cmd, args.dry_run, env, args.continue_on_error)

    if not args.skip_eval:
        cmd = [
            args.python,
            "ar_pose_trellis/benchmark/evaluate_meshes.py",
            "--testsets",
            args.testsets,
            "--arpose_root",
            str(arpose_root),
            "--reconviagen_root",
            str(recon_root),
            "--output_dir",
            str(eval_root),
            "--sample_points",
            str(args.sample_points),
            "--seed",
            str(args.seed),
        ]
        run_command(cmd, args.dry_run, env, args.continue_on_error)

    print(f"\n[compare] done. outputs: {output_root}", flush=True)
    print(f"[compare] eval report: {eval_root / 'benchmark_report.json'}", flush=True)


if __name__ == "__main__":
    main()
