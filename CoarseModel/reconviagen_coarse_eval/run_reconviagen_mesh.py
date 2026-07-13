#!/usr/bin/env python3

"""Run ReconViaGen on a prepared CoarseModel-style dataset and install its mesh."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from common import DEFAULT_OUTPUT_ROOT, RECON_ROOT, ensure_dir, export_mesh_as_obj, write_json


def latest_child_dir(path: Path, not_before: float = 0.0) -> Optional[Path]:
    if not path.exists():
        return None
    children = [p for p in path.iterdir() if p.is_dir() and p.stat().st_mtime >= not_before]
    if not children:
        children = [p for p in path.iterdir() if p.is_dir()]
    if not children:
        return None
    return max(children, key=lambda p: p.stat().st_mtime)


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    return data


def summarize_recon_report(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    candidates = data.get("candidates") or []
    input_manifest = data.get("input_manifest") or {}
    return {
        "selected_seed": data.get("selected_seed"),
        "selected_candidate": data.get("selected_candidate"),
        "candidate_count": len(candidates) if isinstance(candidates, list) else None,
        "candidates": candidates,
        "candidate_seeds": data.get("candidate_seeds"),
        "selected_indices": data.get("selected_indices"),
        "selected_names": data.get("selected_names"),
        "mesh_simplify": data.get("mesh_simplify"),
        "input_manifest_selected_indices": input_manifest.get("selected_indices") if isinstance(input_manifest, dict) else None,
        "input_manifest_selected_names": input_manifest.get("selected_names") if isinstance(input_manifest, dict) else None,
        "input_manifest_frame_filter": input_manifest.get("frame_filter") if isinstance(input_manifest, dict) else None,
    }


def install_mesh_for_coarsemodel(dataset_dir: Path, mesh_path: Path) -> Dict[str, str]:
    dataset_name = dataset_dir.name
    models_dir = ensure_dir(dataset_dir / "models")
    raw_obj = models_dir / f"{dataset_name}.obj"
    norm_obj = models_dir / f"{dataset_name}_norm.obj"
    export_mesh_as_obj(mesh_path, raw_obj)
    # gen_template_auto will overwrite *_norm.obj after scale normalization. Creating
    # an initial copy makes the dataset immediately inspectable before templates run.
    export_mesh_as_obj(mesh_path, norm_obj)
    return {"raw_obj": str(raw_obj), "initial_norm_obj": str(norm_obj)}


def run_recon_generation(
    dataset_dir: Path,
    output_root: Path,
    python_bin: str,
    seeds: Optional[str],
    num_candidates: Optional[int],
    mesh_simplify: Optional[float],
    resolution: int,
    run_refine: bool,
    refine_steps: int,
    existing_mesh: Optional[Path] = None,
) -> Dict[str, object]:
    dataset_dir = dataset_dir.resolve()
    dataset_name = dataset_dir.name
    recon_root = ensure_dir(output_root / "reconviagen" / dataset_name)
    run_root = ensure_dir(output_root / "runs" / dataset_name)

    if existing_mesh is not None:
        mesh_path = existing_mesh.resolve()
        installed = install_mesh_for_coarsemodel(dataset_dir, mesh_path)
        source_candidate_report = read_json_if_exists(mesh_path.parent / "candidate_report.json")
        source_rebuild_report = read_json_if_exists(mesh_path.parent / "rebuild_report.json")
        report = {
            "mode": "existing_mesh",
            "dataset_dir": str(dataset_dir),
            "mesh_path": str(mesh_path),
            "installed_mesh": installed,
            "recon_output_dir": None,
            "source_candidate_report": str(mesh_path.parent / "candidate_report.json")
            if (mesh_path.parent / "candidate_report.json").exists()
            else None,
            "source_candidate_summary": summarize_recon_report(source_candidate_report),
            "source_rebuild_report": str(mesh_path.parent / "rebuild_report.json")
            if (mesh_path.parent / "rebuild_report.json").exists()
            else None,
            "source_rebuild_summary": summarize_recon_report(source_rebuild_report),
        }
        write_json(run_root / "recon_generation_report.json", report)
        return report

    script = RECON_ROOT / "rebuild_mesh_from_coarse_dataset.py"
    if not script.exists():
        raise FileNotFoundError(script)

    cmd = [
        python_bin,
        "-u",
        str(script),
        "--dataset_dir",
        str(dataset_dir),
        "--source",
        "dataset_masks",
        "--output_root",
        str(recon_root),
        "--resolution",
        str(resolution),
    ]
    if seeds:
        cmd += ["--seeds", seeds]
    if num_candidates is not None:
        cmd += ["--num_candidates", str(num_candidates)]
    if mesh_simplify is not None:
        cmd += ["--mesh_simplify", str(mesh_simplify)]
    if run_refine:
        cmd += ["--run_refine", "--refine_steps", str(refine_steps)]

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("SPCONV_ALGO", "native")
    env.setdefault("ATTN_BACKEND", "flash_attn")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start = time.time()
    cmd_txt = " ".join(cmd)
    (run_root / "recon_generation_command.txt").write_text(cmd_txt + "\n", encoding="utf-8")
    print(f"[recon_eval] running: {cmd_txt}", flush=True)
    proc = subprocess.run(cmd, cwd=str(RECON_ROOT), env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ReconViaGen generation failed with returncode={proc.returncode}")

    recon_output_dir = latest_child_dir(recon_root, not_before=start - 1.0)
    if recon_output_dir is None:
        raise RuntimeError(f"No ReconViaGen output directory found under {recon_root}")
    glb_path = recon_output_dir / "reconstructed_object.glb"
    if not glb_path.exists():
        raise FileNotFoundError(f"Expected generated mesh: {glb_path}")

    installed = install_mesh_for_coarsemodel(dataset_dir, glb_path)
    rebuild_report = recon_output_dir / "rebuild_report.json"
    rebuild_data = read_json_if_exists(rebuild_report)
    report = {
        "mode": "generated",
        "dataset_dir": str(dataset_dir),
        "recon_output_dir": str(recon_output_dir),
        "mesh_path": str(glb_path),
        "point_cloud_path": str(recon_output_dir / "reconstructed_object.ply"),
        "preview_path": str(recon_output_dir / "reconstructed_object.mp4"),
        "rebuild_report": str(rebuild_report) if rebuild_report.exists() else None,
        "rebuild_summary": summarize_recon_report(rebuild_data),
        "installed_mesh": installed,
        "command": cmd,
    }
    write_json(run_root / "recon_generation_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--num_candidates", type=int, default=None)
    parser.add_argument("--mesh_simplify", type=float, default=0.75)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--run_refine", action="store_true")
    parser.add_argument("--refine_steps", type=int, default=20)
    parser.add_argument("--existing_mesh", default=None)
    args = parser.parse_args()

    run_recon_generation(
        dataset_dir=Path(args.dataset_dir),
        output_root=Path(args.output_root),
        python_bin=args.python_bin,
        seeds=args.seeds,
        num_candidates=args.num_candidates,
        mesh_simplify=args.mesh_simplify,
        resolution=args.resolution,
        run_refine=args.run_refine,
        refine_steps=args.refine_steps,
        existing_mesh=Path(args.existing_mesh) if args.existing_mesh else None,
    )


if __name__ == "__main__":
    main()
