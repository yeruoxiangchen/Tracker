#!/usr/bin/env python3

"""Standalone feature-representation generation for ReconViaGen meshes.

This file is intentionally independent from ``gen_repre.py``.  It reuses the
shared feature/representation utilities, but it does not import or call the
old representation-generation script.
"""

import argparse
import json
import logging as py_logging
import os
import sys
import types
from pathlib import Path
from typing import List, NamedTuple, Optional, Union


CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parent
EXTRACTOR_NAME = "dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1"
PathLike = Union[str, os.PathLike]


def _install_png_fallback() -> None:
    try:
        import png  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class _Writer:
        def __init__(self, width, height, greyscale=True, bitdepth=16):
            self.width = width
            self.height = height
            self.bitdepth = bitdepth

        def write(self, file_obj, rows):
            import numpy as np
            from PIL import Image

            dtype = np.uint16 if self.bitdepth == 16 else np.uint8
            arr = np.asarray(list(rows), dtype=dtype).reshape(self.height, self.width)
            Image.fromarray(arr).save(file_obj, format="PNG")

    png_module = types.ModuleType("png")
    png_module.Writer = _Writer
    sys.modules["png"] = png_module


def _install_faiss_fallback() -> None:
    try:
        import faiss  # noqa: F401
        import faiss.contrib.torch_utils  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class _Index:
        def __init__(self, centroids):
            self.centroids = centroids

        def search(self, samples, k):
            import torch

            samples_t = torch.as_tensor(samples, dtype=torch.float32)
            centroids_t = torch.as_tensor(self.centroids, dtype=torch.float32)
            dist_chunks = []
            id_chunks = []
            for start in range(0, samples_t.shape[0], 8192):
                chunk = samples_t[start:start + 8192]
                dist = torch.cdist(chunk, centroids_t)
                vals, ids = torch.topk(dist, k=min(k, centroids_t.shape[0]), largest=False)
                dist_chunks.append(vals)
                id_chunks.append(ids)
            return torch.cat(dist_chunks, dim=0), torch.cat(id_chunks, dim=0)

    class _Kmeans:
        def __init__(self, num_dims, num_centroids, niter=50, gpu=False, verbose=True, seed=0, spherical=False):
            self.num_centroids = num_centroids
            self.niter = niter
            self.seed = seed
            self.centroids = None
            self.index = None

        def train(self, samples):
            import torch

            samples_t = torch.as_tensor(samples, dtype=torch.float32)
            torch.manual_seed(self.seed)
            k = min(self.num_centroids, samples_t.shape[0])
            centroids = samples_t[torch.randperm(samples_t.shape[0])[:k]].clone()
            for _ in range(self.niter):
                ids = torch.cdist(samples_t, centroids).argmin(dim=1)
                new_centroids = centroids.clone()
                for cid in range(k):
                    mask = ids == cid
                    if torch.any(mask):
                        new_centroids[cid] = samples_t[mask].mean(dim=0)
                if torch.allclose(new_centroids, centroids, atol=1e-5):
                    centroids = new_centroids
                    break
                centroids = new_centroids
            self.centroids = centroids.cpu().numpy()
            self.index = _Index(self.centroids)

    faiss_module = types.ModuleType("faiss")
    faiss_module.Kmeans = _Kmeans
    contrib_module = types.ModuleType("faiss.contrib")
    torch_utils_module = types.ModuleType("faiss.contrib.torch_utils")
    faiss_module.contrib = contrib_module
    contrib_module.torch_utils = torch_utils_module
    sys.modules["faiss"] = faiss_module
    sys.modules["faiss.contrib"] = contrib_module
    sys.modules["faiss.contrib.torch_utils"] = torch_utils_module


def _prepare_import_path() -> None:
    for path in (str(CORE_DIR), str(PROJECT_ROOT), str(PROJECT_ROOT / "external" / "dinov2")):
        if path not in sys.path:
            sys.path.insert(0, path)
    _install_png_fallback()
    _install_faiss_fallback()


_prepare_import_path()

import numpy as np
import torch

import inout
from config import AppConfig
from utils import (
    cluster_util,
    config_util,
    feature_util,
    json_util,
    logging,
    misc,
    projector_util,
    repre_util2 as repre_util,
    template_util,
)
from utils.misc import array_to_tensor
from utils.structs import PinholePlaneCameraModel


class GenRepreAutoOpts(NamedTuple):
    version: str
    templates_version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None
    extractor_name: str = EXTRACTOR_NAME
    grid_cell_size: float = 14.0
    apply_pca: bool = True
    pca_components: int = 256
    pca_whiten: bool = False
    pca_max_samples_for_fitting: int = 100000
    cluster_features: bool = True
    cluster_num: int = 2048
    template_desc_opts: Optional[repre_util.TemplateDescOpts] = None
    overwrite: bool = True
    debug: bool = True


def default_repre_config(dataset_name: str) -> dict:
    return {
        "gen_repre_auto_opts": {
            "version": "v1",
            "templates_version": "v1",
            "object_dataset": dataset_name,
            "object_lids": [1],
            "extractor_name": EXTRACTOR_NAME,
            "grid_cell_size": 14.0,
            "apply_pca": True,
            "pca_components": 256,
            "pca_whiten": False,
            "pca_max_samples_for_fitting": 100000,
            "cluster_features": True,
            "cluster_num": 2048,
            "template_desc_opts": {
                "desc_type": "tfidf",
                "tfidf_knn_metric": "l2",
                "tfidf_knn_k": 3,
                "tfidf_soft_assign": False,
                "tfidf_soft_sigma_squared": 10.0,
            },
            "overwrite": True,
            "debug": True,
        }
    }


def default_infer_config(dataset_name: str) -> dict:
    return {
        "infer_opts": {
            "version": "v1",
            "repre_version": "v1",
            "object_dataset": dataset_name,
            "object_lids": [1],
            "crop": True,
            "crop_rel_pad": 0.2,
            "crop_size": [420, 420],
            "extractor_name": EXTRACTOR_NAME,
            "grid_cell_size": 14.0,
            "match_template_type": "tfidf",
            "match_top_n_templates": 5,
            "match_feat_matching_type": "cyclic_buddies",
            "match_top_k_buddies": 300,
            "pnp_type": "opencv",
            "pnp_ransac_iter": 400,
            "pnp_required_ransac_conf": 0.99,
            "pnp_inlier_thresh": 10.0,
            "pnp_refine_lm": True,
            "final_pose_type": "best_coarse",
            "use_detections": True,
            "num_preds_factor": 1.0,
            "vis_results": True,
            "debug": True,
        }
    }


def write_repre_config(dataset_name: str, config_path: Optional[PathLike] = None) -> Path:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "gen_repre" / f"{dataset_name}.json"
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(default_repre_config(dataset_name), f, indent=4)
    return config_path


def write_infer_config(dataset_name: str) -> Path:
    config_dir = PROJECT_ROOT / "configs" / "infer" / dataset_name
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{dataset_name}.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(default_infer_config(dataset_name), f, indent=4)
    return config_path


def _load_template_camera(camera_sample: dict) -> PinholePlaneCameraModel:
    return PinholePlaneCameraModel(
        width=camera_sample["ImageSizeX"],
        height=camera_sample["ImageSizeY"],
        f=(camera_sample["fx"], camera_sample["fy"]),
        c=(camera_sample["cx"], camera_sample["cy"]),
        T_world_from_eye=np.array(camera_sample["T_WorldFromCamera"]),
    )


def generate_raw_repre(
    opts: GenRepreAutoOpts,
    object_dataset: str,
    object_lid: int,
    extractor: torch.nn.Module,
    device: str,
    ) -> repre_util.FeatureBasedObjectRepre:
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    logger.setLevel(py_logging.WARNING)
    metadata_path = (
        AppConfig.OUTPUT_ROOT
        / "templates"
        / opts.templates_version
        / opts.object_dataset
        / str(object_lid)
        / "metadata.json"
    )
    metadata = json_util.load_json(metadata_path)
    print(f"[gen_repre_auto] Loading {len(metadata)} templates from {metadata_path}", flush=True)

    feat_vectors_list = []
    feat_to_vertex_ids_list = []
    vertices_in_model_list = []
    feat_to_template_ids_list = []
    templates_list = []
    template_cameras_list = []

    for template_id, sample in enumerate(metadata):
        if template_id == 0 or (template_id + 1) % 25 == 0 or template_id + 1 == len(metadata):
            print(f"\r[gen_repre_auto] Registering features {template_id + 1}/{len(metadata)}", flush=True)
        camera = _load_template_camera(sample["cameras"])
        image_arr = inout.load_im(sample["rgb_image_path"])
        depth_scale_meta_path = sample["depth_map_path"] + ".json"
        if not os.path.exists(depth_scale_meta_path):
            raise RuntimeError(
                "Template depth map has no depth-scale metadata. "
                "It was likely generated by the old depth writer that rounded "
                f"sub-unit depths to integers: {sample['depth_map_path']}. "
                "Regenerate templates before regenerating the object representation."
            )
        depth_arr = inout.load_depth(sample["depth_map_path"])
        mask_arr = inout.load_im(sample["binary_mask_path"])

        image_chw = array_to_tensor(image_arr).to(torch.float32).permute(2, 0, 1).to(device) / 255.0
        depth_hw = array_to_tensor(depth_arr).to(torch.float32).to(device)
        mask_hw = array_to_tensor(mask_arr).to(torch.float32).to(device)

        pose = sample["pose"]
        T_world_from_model_np = np.eye(4)
        T_world_from_model_np[:3, :3] = pose["R"]
        T_world_from_model_np[:3, 3:] = pose["t"]
        T_world_from_model = array_to_tensor(T_world_from_model_np).to(torch.float32).to(device)
        T_model_from_world = torch.linalg.inv(T_world_from_model)
        T_world_from_camera = array_to_tensor(camera.T_world_from_eye).to(torch.float32).to(device)
        T_model_from_camera = torch.matmul(T_model_from_world, T_world_from_camera)

        feat_vectors, feat_to_vertex_ids, vertices_in_model = feature_util.get_visual_features_registered_in_3d(
            image_chw=image_chw,
            depth_image_hw=depth_hw,
            object_mask=mask_hw,
            camera=camera,
            T_model_from_camera=T_model_from_camera,
            extractor=extractor,
            grid_cell_size=opts.grid_cell_size,
            debug=False,
        )
        if feat_vectors.numel() == 0:
            continue

        feat_vectors_list.append(feat_vectors)
        feat_to_vertex_ids_list.append(feat_to_vertex_ids)
        vertices_in_model_list.append(vertices_in_model)
        feat_to_template_ids_list.append(template_id * torch.ones(feat_vectors.shape[0], dtype=torch.int32, device=device))
        templates_list.append((image_chw * 255).to(torch.uint8))
        camera_copy = camera.copy()
        camera_copy.extrinsics = torch.linalg.inv(T_model_from_camera)
        template_cameras_list.append(camera_copy)

    if not feat_vectors_list:
        raise RuntimeError(f"No registered features generated for {object_dataset}")

    return repre_util.FeatureBasedObjectRepre(
        vertices=torch.cat(vertices_in_model_list),
        feat_vectors=torch.cat(feat_vectors_list),
        feat_opts=repre_util.FeatureOpts(extractor_name=opts.extractor_name),
        feat_to_vertex_ids=torch.cat(feat_to_vertex_ids_list),
        feat_to_template_ids=torch.cat(feat_to_template_ids_list),
        templates=torch.stack(templates_list),
        template_cameras_cam_from_model=template_cameras_list,
    )


def generate_repre(opts: GenRepreAutoOpts, dataset: str, lid: int, device: str, extractor: torch.nn.Module) -> None:
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    logger.setLevel(py_logging.WARNING)
    base_repre_dir = AppConfig.OUTPUT_ROOT / "object_repre"
    output_dir = repre_util.get_object_repre_dir_path(str(base_repre_dir), opts.version, dataset, lid)
    if os.path.exists(output_dir) and not opts.overwrite:
        raise ValueError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    json_util.save_json(os.path.join(output_dir, "config.json"), opts)

    print(f"[gen_repre_auto] Generating raw representation for dataset={dataset}, object={lid}", flush=True)
    repre = generate_raw_repre(opts, dataset, lid, extractor, device)
    feat_vectors = repre.feat_vectors
    print(f"[gen_repre_auto] Raw feature count: {feat_vectors.shape[0]}", flush=True)

    if opts.apply_pca:
        print("[gen_repre_auto] Applying PCA", flush=True)
        logger.info("Preparing PCA...")
        pca = projector_util.PCAProjector(n_components=opts.pca_components, whiten=opts.pca_whiten)
        pca.fit(feat_vectors, max_samples=opts.pca_max_samples_for_fitting)
        repre.feat_raw_projectors.append(pca)
        feat_vectors = pca.transform(feat_vectors)

    if opts.cluster_features:
        cluster_num = min(opts.cluster_num, feat_vectors.shape[0])
        print(f"[gen_repre_auto] Clustering features into {cluster_num} words", flush=True)
        logger.info(f"Clustering {feat_vectors.shape[0]} features into {cluster_num} visual words...")
        centroids, cluster_ids, _ = cluster_util.kmeans(feat_vectors, num_centroids=cluster_num, verbose=True)
        repre.feat_cluster_centroids = centroids
        repre.feat_to_cluster_ids = cluster_ids

    if opts.template_desc_opts is not None:
        print(f"[gen_repre_auto] Computing {opts.template_desc_opts.desc_type} template descriptors", flush=True)
        repre.template_desc_opts = opts.template_desc_opts
        if opts.template_desc_opts.desc_type != "tfidf":
            raise ValueError(f"Unknown descriptor type: {opts.template_desc_opts.desc_type}")
        repre.template_descs, repre.feat_cluster_idfs = template_util.calc_tfidf_descriptors(
            feat_vectors=feat_vectors,
            feat_words=repre.feat_cluster_centroids,
            feat_to_word_ids=repre.feat_to_cluster_ids,
            feat_to_template_ids=repre.feat_to_template_ids,
            num_templates=len(repre.templates),
            tfidf_knn_k=opts.template_desc_opts.tfidf_knn_k,
            tfidf_soft_assign=opts.template_desc_opts.tfidf_soft_assign,
            tfidf_soft_sigma_squared=opts.template_desc_opts.tfidf_soft_sigma_squared,
        )

    if repre.feat_raw_projectors and isinstance(repre.feat_raw_projectors[0], projector_util.PCAProjector):
        repre.feat_vis_projectors = [repre.feat_raw_projectors[0]]
    else:
        pca_vis = projector_util.PCAProjector(n_components=3, whiten=False)
        pca_vis.fit(feat_vectors, max_samples=opts.pca_max_samples_for_fitting)
        repre.feat_vis_projectors = [pca_vis]

    repre.feat_vectors = feat_vectors
    repre_util.save_object_repre(repre, output_dir)
    print(f"[gen_repre_auto] Saved representation: {os.path.join(output_dir, 'repre.pth')}", flush=True)


def generate_repre_for_dataset(dataset_name: str, config_path: Optional[PathLike] = None) -> Path:
    config_path = write_repre_config(dataset_name, config_path)
    write_infer_config(dataset_name)
    opts = config_util.load_opts_from_json(
        path=str(config_path),
        opts_types={"gen_repre_auto_opts": GenRepreAutoOpts},
    )["gen_repre_auto_opts"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gen_repre_auto] Loading feature extractor on {device}: {opts.extractor_name}", flush=True)
    extractor = feature_util.make_feature_extractor(opts.extractor_name)
    extractor.to(device)
    extractor.eval()
    for object_lid in opts.object_lids or [1]:
        generate_repre(opts, opts.object_dataset, object_lid, device, extractor)
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config-path", default=None)
    args = parser.parse_args()
    generate_repre_for_dataset(args.dataset_name, args.config_path)


if __name__ == "__main__":
    main()
