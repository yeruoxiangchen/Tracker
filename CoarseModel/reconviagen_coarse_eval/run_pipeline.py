#!/usr/bin/env python3

"""End-to-end synthetic evaluation for ReconViaGen mesh + CoarseModel 4-stage refine."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from common import (
    DEFAULT_OUTPUT_ROOT,
    evaluate_mesh_basic,
    evaluate_mesh_quality,
    evaluate_mesh_quality_against_points,
    load_sparse_target_points,
    load_manifest,
    prepare_existing_coarse_dataset,
    prepare_coarse_dataset_from_pixal_sample,
    sample_case_name,
    write_json,
)
from run_coarsemodel_4stage_eval import run_coarsemodel_eval
from run_reconviagen_mesh import run_recon_generation


def parse_stages(text: str) -> List[str]:
    stages = [s.strip() for s in text.split(",") if s.strip()]
    valid = {"prepare", "recon", "mesh_eval", "coarse"}
    unknown = [s for s in stages if s not in valid]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}; valid={sorted(valid)}")
    return stages


def infer_case_name(manifest: Path, sample_index: int, case_prefix: str) -> str:
    samples, _ = load_manifest(manifest)
    return sample_case_name(samples[sample_index], sample_index, prefix=case_prefix)


def load_prepared(output_root: Path, case_name: str) -> Dict[str, object]:
    path = output_root / "runs" / case_name / "prepared_sample.json"
    if not path.exists():
        raise FileNotFoundError(f"Prepared sample report not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dataset_dir", default=None, help="Existing CoarseModel dataset, e.g. CoarseModel/datasets/GOOD_MESH_TEST")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case_name", default=None)
    parser.add_argument("--case_prefix", default="pixalv9")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--stages", default="prepare,recon,mesh_eval,coarse")
    parser.add_argument("--python_bin", default=sys.executable)

    parser.add_argument("--recon_seeds", default="0")
    parser.add_argument("--recon_num_candidates", type=int, default=None)
    parser.add_argument("--mesh_simplify", type=float, default=0.75)
    parser.add_argument("--recon_resolution", type=int, default=518)
    parser.add_argument("--recon_run_refine", action="store_true")
    parser.add_argument("--recon_refine_steps", type=int, default=20)
    parser.add_argument("--existing_recon_mesh", default=None)
    parser.add_argument("--force_recon_generate", action="store_true")

    parser.add_argument("--mesh_eval_samples", type=int, default=20000)
    parser.add_argument("--mesh_eval_icp_iters", type=int, default=12)
    parser.add_argument("--mesh_eval_reference", choices=["auto", "sparse", "glb", "dataset_model", "basic"], default="auto")

    parser.add_argument("--template_version", default="v1")
    parser.add_argument("--repre_version", default="v1")
    parser.add_argument("--template_views", type=int, default=24)
    parser.add_argument("--template_inplane", type=int, default=4)
    parser.add_argument("--cluster_num", type=int, default=1024)
    parser.add_argument("--pca_components", type=int, default=256)
    parser.add_argument("--force_repre", action="store_true")
    parser.add_argument("--skip_templates", action="store_true")
    parser.add_argument("--skip_repre", action="store_true")
    parser.add_argument("--skip_deformation", action="store_true")
    parser.add_argument("--reuse_existing_coarse_assets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--asset_link_mode", choices=["symlink", "copy"], default="symlink")
    args = parser.parse_args()

    if not args.manifest and not args.dataset_dir:
        raise ValueError("Either --manifest or --dataset_dir is required")
    manifest = Path(args.manifest) if args.manifest else None
    source_dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    output_root = Path(args.output_root)
    stages = parse_stages(args.stages)
    max_frames: Optional[int] = None if args.max_frames is not None and args.max_frames <= 0 else args.max_frames
    if args.case_name:
        case_name = args.case_name
    elif source_dataset_dir is not None:
        case_name = source_dataset_dir.name
    else:
        case_name = infer_case_name(manifest, args.sample_index, args.case_prefix)

    report: Dict[str, object] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(manifest) if manifest else None,
        "source_dataset_dir": str(source_dataset_dir) if source_dataset_dir else None,
        "sample_index": int(args.sample_index),
        "case_name": case_name,
        "stages": stages,
        "max_frames": max_frames,
        "output_root": str(output_root),
    }

    if "prepare" in stages:
        if source_dataset_dir is not None:
            prepared = prepare_existing_coarse_dataset(
                dataset_dir=source_dataset_dir,
                output_root=output_root,
                case_name=case_name,
                max_frames=max_frames,
                link_mode=args.asset_link_mode,
            )
        else:
            prepared = prepare_coarse_dataset_from_pixal_sample(
                manifest_path=manifest,
                sample_index=args.sample_index,
                output_root=output_root,
                case_name=case_name,
                max_frames=max_frames,
                case_prefix=args.case_prefix,
            )
    else:
        prepared = load_prepared(output_root, case_name)
    report["prepared"] = prepared

    dataset_dir = Path(str(prepared["dataset_dir"]))
    recon_report: Optional[Dict[str, object]] = None
    if "recon" in stages:
        existing_recon_mesh = Path(args.existing_recon_mesh) if args.existing_recon_mesh else None
        if existing_recon_mesh is None and not args.force_recon_generate:
            default_recon_mesh = prepared.get("default_recon_mesh")
            if default_recon_mesh:
                existing_recon_mesh = Path(str(default_recon_mesh))
        recon_report = run_recon_generation(
            dataset_dir=dataset_dir,
            output_root=output_root,
            python_bin=args.python_bin,
            seeds=args.recon_seeds,
            num_candidates=args.recon_num_candidates,
            mesh_simplify=args.mesh_simplify,
            resolution=args.recon_resolution,
            run_refine=args.recon_run_refine,
            refine_steps=args.recon_refine_steps,
            existing_mesh=existing_recon_mesh,
        )
    elif args.existing_recon_mesh:
        recon_report = run_recon_generation(
            dataset_dir=dataset_dir,
            output_root=output_root,
            python_bin=args.python_bin,
            seeds=None,
            num_candidates=None,
            mesh_simplify=None,
            resolution=args.recon_resolution,
            run_refine=False,
            refine_steps=0,
            existing_mesh=Path(args.existing_recon_mesh),
        )
    else:
        recon_path = output_root / "runs" / case_name / "recon_generation_report.json"
        if recon_path.exists():
            with recon_path.open("r", encoding="utf-8") as f:
                recon_report = json.load(f)
    report["recon"] = recon_report

    if "mesh_eval" in stages:
        if recon_report is None:
            raise RuntimeError("mesh_eval requires recon report or --existing_recon_mesh")
        pred_mesh = recon_report["mesh_path"]
        mesh_eval_path = output_root / "mesh_quality" / case_name / "mesh_quality_report.json"
        mesh_eval_reference = args.mesh_eval_reference
        if mesh_eval_reference == "auto":
            if prepared.get("ss_latent_path"):
                mesh_eval_reference = "sparse"
            elif prepared.get("source_glb"):
                mesh_eval_reference = "glb"
            elif prepared.get("source_model_mesh"):
                mesh_eval_reference = "dataset_model"
            else:
                mesh_eval_reference = "basic"
        if mesh_eval_reference == "sparse":
            ss_latent_path = prepared.get("ss_latent_path")
            if not ss_latent_path:
                raise RuntimeError("Prepared sample has no ss_latent_path for sparse mesh evaluation")
            mesh_eval = evaluate_mesh_quality_against_points(
                pred_mesh_path=pred_mesh,
                gt_points=load_sparse_target_points(str(ss_latent_path)),
                output_json=mesh_eval_path,
                samples=args.mesh_eval_samples,
                seed=args.sample_index,
                icp_iters=args.mesh_eval_icp_iters,
            )
        elif mesh_eval_reference == "glb":
            gt_mesh = prepared.get("source_glb")
            if not gt_mesh:
                raise RuntimeError("Prepared sample has no source_glb for mesh evaluation")
            mesh_eval = evaluate_mesh_quality(
                pred_mesh_path=pred_mesh,
                gt_mesh_path=str(gt_mesh),
                output_json=mesh_eval_path,
                samples=args.mesh_eval_samples,
                seed=args.sample_index,
                icp_iters=args.mesh_eval_icp_iters,
            )
        elif mesh_eval_reference == "dataset_model":
            gt_mesh = prepared.get("source_model_mesh")
            if not gt_mesh:
                raise RuntimeError("Prepared sample has no source_model_mesh for dataset_model mesh evaluation")
            mesh_eval = evaluate_mesh_quality(
                pred_mesh_path=pred_mesh,
                gt_mesh_path=str(gt_mesh),
                output_json=mesh_eval_path,
                samples=args.mesh_eval_samples,
                seed=args.sample_index,
                icp_iters=args.mesh_eval_icp_iters,
            )
            mesh_eval["gt_reference"] = "existing_dataset_model_obj_not_strict_gt"
            write_json(mesh_eval_path, mesh_eval)
        else:
            mesh_eval = evaluate_mesh_basic(
                pred_mesh_path=pred_mesh,
                output_json=mesh_eval_path,
            )
        report["mesh_eval"] = mesh_eval

    if "coarse" in stages:
        coarse = run_coarsemodel_eval(
            dataset_dir=dataset_dir,
            output_root=output_root,
            template_version=args.template_version,
            repre_version=args.repre_version,
            template_views=args.template_views,
            template_inplane=args.template_inplane,
            cluster_num=args.cluster_num,
            pca_components=args.pca_components,
            force_repre=args.force_repre,
            skip_templates=args.skip_templates,
            skip_repre=args.skip_repre,
            skip_deformation=args.skip_deformation,
            reuse_existing_assets=args.reuse_existing_coarse_assets,
            asset_link_mode=args.asset_link_mode,
        )
        report["coarse"] = coarse

    top_report = output_root / "runs" / case_name / "pipeline_report.json"
    write_json(top_report, report)
    print(json.dumps({"pipeline_report": str(top_report), "case_name": case_name}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
