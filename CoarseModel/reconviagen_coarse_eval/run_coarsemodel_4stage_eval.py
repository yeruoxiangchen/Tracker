#!/usr/bin/env python3

"""Evaluate CoarseModel 4-stage pose/deformation on a ReconViaGen mesh."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import cv2
import numpy as np
import trimesh

from common import (
    COARSE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    add_coarse_import_paths,
    contour_chamfer_px,
    ensure_dir,
    link_or_copy,
    mask_metrics,
    patch_coarse_app_config,
    summarize_rows,
    write_json,
)


def patch_module_app_configs(workspace: Path, modules: List[Any]) -> None:
    datasets = ensure_dir(workspace / "datasets")
    results = ensure_dir(workspace / "results")
    for module in modules:
        app_config = getattr(module, "AppConfig", None)
        if app_config is None:
            continue
        app_config.PROJECT_ROOT = workspace
        app_config.DATASETS_PATH = datasets
        app_config.OUTPUT_ROOT = results
        app_config.TEMPLATE_ROOT = results / "templates"
        app_config.REPRE_ROOT = results / "object_repre"
        app_config.REFINE_ROOT = results / "refine_model"


def write_template_config(config_path: Path, dataset_name: str, version: str, min_views: int, inplane: int) -> Path:
    from core import gen_template_auto

    payload = gen_template_auto.default_template_config(dataset_name)
    opts = payload["gen_template_auto_opts"]
    opts["version"] = version
    opts["object_dataset"] = dataset_name
    opts["min_num_viewpoints"] = int(min_views)
    opts["num_inplane_rotations"] = int(inplane)
    opts["overwrite"] = True
    write_json(config_path, payload)
    return config_path


def write_repre_config(
    config_path: Path,
    dataset_name: str,
    version: str,
    templates_version: str,
    cluster_num: int,
    pca_components: int,
) -> Path:
    from core import gen_repre_auto

    payload = gen_repre_auto.default_repre_config(dataset_name)
    opts = payload["gen_repre_auto_opts"]
    opts["version"] = version
    opts["templates_version"] = templates_version
    opts["object_dataset"] = dataset_name
    opts["cluster_num"] = int(cluster_num)
    opts["pca_components"] = int(pca_components)
    opts["overwrite"] = True
    write_json(config_path, payload)
    return config_path


def write_infer_config(config_path: Path, dataset_name: str, version: str, repre_version: str) -> Path:
    from core import gen_repre_auto

    payload = gen_repre_auto.default_infer_config(dataset_name)
    opts = payload["infer_opts"]
    opts["version"] = version
    opts["repre_version"] = repre_version
    opts["object_dataset"] = dataset_name
    opts["use_optimization_cache"] = False
    opts["vis_results"] = True
    opts["debug"] = True
    write_json(config_path, payload)
    return config_path


def generate_templates_if_needed(
    dataset_name: str,
    workspace: Path,
    version: str,
    min_views: int,
    inplane: int,
    force: bool,
) -> Path:
    from core import gen_template_auto

    config_path = workspace / "configs" / "gen_templates" / f"{dataset_name}.json"
    metadata_path = workspace / "results" / "templates" / version / dataset_name / "1" / "metadata.json"
    if metadata_path.exists() and not force:
        print(f"[coarse_eval] templates exist, skip: {metadata_path}", flush=True)
        return config_path

    write_template_config(config_path, dataset_name, version, min_views, inplane)
    opts = gen_template_auto.config_util.load_opts_from_json(
        path=str(config_path),
        opts_types={"gen_template_auto_opts": gen_template_auto.GenTemplateAutoOpts},
    )["gen_template_auto_opts"]
    gen_template_auto.synthesize_templates(opts)
    return config_path


def generate_repre_if_needed(
    dataset_name: str,
    workspace: Path,
    version: str,
    templates_version: str,
    cluster_num: int,
    pca_components: int,
    force: bool,
) -> Path:
    from core import gen_repre_auto
    from utils import feature_util

    config_path = workspace / "configs" / "gen_repre" / f"{dataset_name}.json"
    repre_path = workspace / "results" / "object_repre" / dataset_name / version / "1" / "repre.pth"
    if repre_path.exists() and not force:
        print(f"[coarse_eval] repre exists, skip: {repre_path}", flush=True)
        return config_path

    write_repre_config(config_path, dataset_name, version, templates_version, cluster_num, pca_components)
    opts = gen_repre_auto.config_util.load_opts_from_json(
        path=str(config_path),
        opts_types={"gen_repre_auto_opts": gen_repre_auto.GenRepreAutoOpts},
    )["gen_repre_auto_opts"]

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"[coarse_eval] loading feature extractor on {device}: {opts.extractor_name}", flush=True)
    extractor = feature_util.make_feature_extractor(opts.extractor_name)
    extractor.to(device)
    extractor.eval()
    for object_lid in opts.object_lids or [1]:
        gen_repre_auto.generate_repre(opts, opts.object_dataset, object_lid, device, extractor)
    return config_path


def reuse_existing_assets_if_available(
    dataset_name: str,
    workspace: Path,
    template_version: str,
    repre_version: str,
    link_mode: str = "symlink",
) -> Dict[str, Any]:
    """Expose existing CoarseModel templates/repre under the eval workspace.

    This avoids regenerating templates when the environment lacks pyrender.
    Paths under the eval workspace may be symlinks, but all CoarseModel outputs
    still go to the eval workspace.
    """
    copied = {}
    src_template = COARSE_ROOT / "results" / "templates" / template_version / dataset_name
    dst_template = workspace / "results" / "templates" / template_version / dataset_name
    if src_template.exists():
        link_or_copy(src_template, dst_template, mode=link_mode)
        copied["templates"] = {"source": str(src_template), "target": str(dst_template), "mode": link_mode}

    src_repre = COARSE_ROOT / "results" / "object_repre" / dataset_name / repre_version
    dst_repre = workspace / "results" / "object_repre" / dataset_name / repre_version
    if src_repre.exists():
        link_or_copy(src_repre, dst_repre, mode=link_mode)
        copied["repre"] = {"source": str(src_repre), "target": str(dst_repre), "mode": link_mode}

    src_infer = COARSE_ROOT / "configs" / "infer" / dataset_name
    dst_infer = workspace / "configs" / "infer" / dataset_name
    if src_infer.exists():
        link_or_copy(src_infer, dst_infer, mode=link_mode)
        copied["infer_config"] = {"source": str(src_infer), "target": str(dst_infer), "mode": link_mode}

    src_template_config = COARSE_ROOT / "configs" / "gen_templates" / f"{dataset_name}.json"
    dst_template_config = workspace / "configs" / "gen_templates" / f"{dataset_name}.json"
    if src_template_config.exists():
        link_or_copy(src_template_config, dst_template_config, mode=link_mode)
        copied["template_config"] = {"source": str(src_template_config), "target": str(dst_template_config), "mode": link_mode}

    src_repre_config = COARSE_ROOT / "configs" / "gen_repre" / f"{dataset_name}.json"
    dst_repre_config = workspace / "configs" / "gen_repre" / f"{dataset_name}.json"
    if src_repre_config.exists():
        link_or_copy(src_repre_config, dst_repre_config, mode=link_mode)
        copied["repre_config"] = {"source": str(src_repre_config), "target": str(dst_repre_config), "mode": link_mode}

    return copied


def load_model_vertices_faces(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, Any]:
    from core.util import load_obj_with_visual

    dataset_name = dataset_dir.name
    model_path = dataset_dir / "models" / f"{dataset_name}_norm.obj"
    if not model_path.exists():
        model_path = dataset_dir / "models" / f"{dataset_name}.obj"
    vertices, faces, visual = load_obj_with_visual(str(model_path))
    if vertices is None or len(vertices) == 0:
        raise RuntimeError(f"Empty model vertices: {model_path}")
    return np.asarray(vertices), np.asarray(faces), visual


def evaluate_projection_masks(
    all_optimization_data: List[Dict[str, Any]],
    vertices: np.ndarray,
    faces: np.ndarray,
    T_M2W: np.ndarray,
    output_dir: Path,
    label: str,
) -> Dict[str, Any]:
    from script.vis_util import draw_mask_contour, render_mesh_mask

    vis_dir = ensure_dir(output_dir / "mask_eval" / label)
    rows = []
    for data in all_optimization_data:
        image = cv2.imread(data["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        mask_path = data["image_path"].replace("/rgb/", "/masks/")
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        pred = render_mesh_mask(
            image.shape,
            data["K"],
            data["T_w2c"] @ T_M2W,
            vertices,
            faces,
        )
        row = {
            "frame_name": data.get("frame_name", Path(data["image_path"]).name),
            **mask_metrics(pred, gt),
            **contour_chamfer_px(pred, gt),
        }
        rows.append(row)

        overlay = image.copy()
        overlay = draw_mask_contour(overlay, gt, color=(0, 255, 0), thickness=2)
        overlay = draw_mask_contour(overlay, pred, color=(0, 0, 255), thickness=2)
        cv2.imwrite(str(vis_dir / f"{row['frame_name']}"), overlay)
        cv2.imwrite(str(vis_dir / f"{Path(row['frame_name']).stem}_pred_mask.png"), pred)

    summary = summarize_rows(rows)
    report = {"label": label, "rows": rows, "summary": summary, "vis_dir": str(vis_dir)}
    write_json(output_dir / "mask_eval" / f"{label}.json", report)
    return report


def export_refined_mesh(vertices: np.ndarray, faces: np.ndarray, visual: Any, output_dir: Path) -> str:
    refine_dir = ensure_dir(output_dir / "refine")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual.copy() if visual is not None else None, process=False)
    path = refine_dir / "refined_model.obj"
    mesh.export(path)
    return str(path)


def run_coarsemodel_eval(
    dataset_dir: Path,
    output_root: Path,
    template_version: str,
    repre_version: str,
    template_views: int,
    template_inplane: int,
    cluster_num: int,
    pca_components: int,
    force_repre: bool,
    skip_templates: bool,
    skip_repre: bool,
    skip_deformation: bool,
    reuse_existing_assets: bool = True,
    asset_link_mode: str = "symlink",
) -> Dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    dataset_name = dataset_dir.name
    workspace = output_root / "workspace"
    coarse_out = ensure_dir(output_root / "coarsemodel" / dataset_name)

    add_coarse_import_paths()
    patch_coarse_app_config(workspace)

    from core import gen_repre_auto, gen_template_auto
    from core.config import InferOpts
    from core.global_optimize_2stage_defor_newinit_test import optimize_global_pose, refine_model_with_deformation_graph
    from core.preprocess_sim_test import build_optimization_data
    from utils import config_util

    patch_module_app_configs(workspace, [gen_template_auto, gen_repre_auto])

    reused_assets = {}
    if reuse_existing_assets:
        reused_assets = reuse_existing_assets_if_available(
            dataset_name=dataset_name,
            workspace=workspace,
            template_version=template_version,
            repre_version=repre_version,
            link_mode=asset_link_mode,
        )
        if reused_assets:
            print(f"[coarse_eval] reused existing assets: {json.dumps(reused_assets, ensure_ascii=False)}", flush=True)

    if not skip_templates:
        try:
            generate_templates_if_needed(
                dataset_name=dataset_name,
                workspace=workspace,
                version=template_version,
                min_views=template_views,
                inplane=template_inplane,
                force=force_repre,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "pyrender" and reused_assets.get("templates"):
                print("[coarse_eval] pyrender missing, but reused templates are available; continue.", flush=True)
            else:
                raise
    if not skip_repre:
        generate_repre_if_needed(
            dataset_name=dataset_name,
            workspace=workspace,
            version=repre_version,
            templates_version=template_version,
            cluster_num=cluster_num,
            pca_components=pca_components,
            force=force_repre,
        )

    infer_path = workspace / "configs" / "infer" / dataset_name / f"{dataset_name}.json"
    write_infer_config(infer_path, dataset_name, version=template_version, repre_version=repre_version)
    opts = config_util.load_opts_from_json(path=str(infer_path), opts_types={"infer_opts": InferOpts})["infer_opts"]

    rgb_dir = dataset_dir / "rgb"
    mask_dir = dataset_dir / "masks"
    colmap_dir = dataset_dir / "sparse" / "0"
    vertices, faces, visual = load_model_vertices_faces(dataset_dir)

    print("[coarse_eval] building optimization data", flush=True)
    all_optimization_data, all_corresps_data = build_optimization_data(
        rgb_dir=str(rgb_dir),
        mask_dir=str(mask_dir),
        colmap_dir=str(colmap_dir),
        opts=opts,
    )
    if len(all_optimization_data) == 0:
        raise RuntimeError("CoarseModel produced no valid optimization frames")

    print("[coarse_eval] optimizing global pose/scale", flush=True)
    T_M2W_final, result, final_scale = optimize_global_pose(
        all_optimization_data=all_optimization_data,
        all_corresps_data=all_corresps_data,
        model_vertices=vertices,
        model_faces=faces,
        output_dir=str(coarse_out),
        colmap_dir=str(colmap_dir),
        mask_dir=str(mask_dir),
        init_frame_id=0,
    )

    pose_mask_report = evaluate_projection_masks(
        all_optimization_data=all_optimization_data,
        vertices=vertices,
        faces=faces,
        T_M2W=T_M2W_final,
        output_dir=coarse_out,
        label="after_pose",
    )

    refine_report = {"enabled": False}
    if not skip_deformation:
        print("[coarse_eval] refining mesh with deformation graph", flush=True)
        V_world_h = np.hstack([vertices, np.ones((len(vertices), 1))])
        V_world = (T_M2W_final @ V_world_h.T).T[:, :3]
        refined_vertices = refine_model_with_deformation_graph(
            model_vertices=V_world,
            model_faces=faces,
            output_dir=str(coarse_out),
            T_M2W=np.eye(4),
            all_optimization_data=all_optimization_data,
        )
        refined_mesh_path = export_refined_mesh(refined_vertices, faces, visual, coarse_out)
        refined_mask_report = evaluate_projection_masks(
            all_optimization_data=all_optimization_data,
            vertices=refined_vertices,
            faces=faces,
            T_M2W=np.eye(4),
            output_dir=coarse_out,
            label="after_deformation",
        )
        refine_report = {
            "enabled": True,
            "refined_mesh": refined_mesh_path,
            "mask_eval": refined_mask_report,
        }

    report = {
        "dataset_dir": str(dataset_dir),
        "dataset_name": dataset_name,
        "coarsemodel_entry_equivalent": "CoarseModel/estimation_4stage_defo_fin.py",
        "workspace": str(workspace),
        "output_dir": str(coarse_out),
        "num_optimization_frames": len(all_optimization_data),
        "T_M2W_final": np.asarray(T_M2W_final).tolist(),
        "final_scale": np.asarray(final_scale).tolist(),
        "optimization_cost": float(result.cost),
        "optimization_status": int(result.status),
        "optimization_message": str(result.message),
        "mask_eval_after_pose": pose_mask_report,
        "deformation": refine_report,
        "template_version": template_version,
        "repre_version": repre_version,
        "reused_existing_assets": reused_assets,
    }
    write_json(coarse_out / "coarsemodel_4stage_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
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

    run_coarsemodel_eval(
        dataset_dir=Path(args.dataset_dir),
        output_root=Path(args.output_root),
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


if __name__ == "__main__":
    main()
