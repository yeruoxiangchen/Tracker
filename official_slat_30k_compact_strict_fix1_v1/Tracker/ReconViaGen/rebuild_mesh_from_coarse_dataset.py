import argparse
import json
import math
import os
import re
import sys
import time

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")
os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
for _cache_dir in [
    os.environ["NUMBA_CACHE_DIR"],
    os.environ["MPLCONFIGDIR"],
    os.environ["XDG_CACHE_HOME"],
    os.environ["TORCH_HOME"],
]:
    os.makedirs(_cache_dir, exist_ok=True)

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from run_local import (  # noqa: E402
    get_candidate_seeds,
    get_mesh_simplify_ratio,
    run_limited_candidate_generation,
    save_generation_inputs,
)
import ar_pose_quality as arq  # noqa: E402
from trellis.utils import postprocessing_utils, render_utils  # noqa: E402

import imageio  # noqa: E402
import torch  # noqa: E402


DEFAULT_DATASET_DIR = "/home/zjr/Tracker/CoarseModel/datasets/reconviagen_20260514_071732"
DEFAULT_TRELLIS_REPO = "Stable-X/trellis-vggt-v0-2"
DEFAULT_VGGT_REPO = "Stable-X/vggt-object-v0-1"
DEFAULT_BIREFNET_REPO = "ZhengPeng7/BiRefNet"


def hf_snapshot_path(repo_id):
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = os.path.join(hf_home, "hub", f"models--{repo_id.replace('/', '--')}")
    refs_main = os.path.join(model_dir, "refs", "main")
    if os.path.exists(refs_main):
        with open(refs_main, "r") as f:
            revision = f.read().strip()
        snapshot = os.path.join(model_dir, "snapshots", revision)
        if os.path.isdir(snapshot):
            return snapshot

    snapshots_dir = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        snapshots = [
            os.path.join(snapshots_dir, name)
            for name in sorted(os.listdir(snapshots_dir))
            if os.path.isdir(os.path.join(snapshots_dir, name))
        ]
        if snapshots:
            return snapshots[-1]
    raise FileNotFoundError(f"Local Hugging Face snapshot not found for {repo_id}: {model_dir}")


def init_pipeline_from_local_cache(trellis_model_path=None, vggt_model_path=None, birefnet_model_path=None):
    from trellis.pipelines.trellis_image_to_3d import (  # noqa: WPS433
        AutoModelForImageSegmentation,
        TrellisImageTo3DPipeline,
        TrellisVGGTTo3DPipeline,
        VGGT,
    )

    torch_hub_dir = os.path.join(os.environ["TORCH_HOME"], "hub")
    torch.hub.set_dir(torch_hub_dir)
    dinov2_hubconf = os.path.join(torch_hub_dir, "facebookresearch_dinov2_main", "hubconf.py")
    if not os.path.exists(dinov2_hubconf):
        raise FileNotFoundError(
            "DINOv2 torch hub cache is missing. Expected local file: "
            f"{dinov2_hubconf}. Do not let torch.hub fall back to network here."
        )

    trellis_model_path = trellis_model_path or hf_snapshot_path(DEFAULT_TRELLIS_REPO)
    vggt_model_path = vggt_model_path or hf_snapshot_path(DEFAULT_VGGT_REPO)
    birefnet_model_path = birefnet_model_path or hf_snapshot_path(DEFAULT_BIREFNET_REPO)

    print("Initializing Pipeline into VRAM from local cache...")
    print(f"[Rebuild] Trellis:  {trellis_model_path}")
    print(f"[Rebuild] VGGT:     {vggt_model_path}")
    print(f"[Rebuild] BiRefNet: {birefnet_model_path}")

    base_pipeline = TrellisImageTo3DPipeline.from_pretrained(trellis_model_path)
    pipeline = TrellisVGGTTo3DPipeline()
    pipeline.__dict__ = base_pipeline.__dict__
    pipeline.VGGT_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    vggt_model = VGGT.from_pretrained(vggt_model_path)
    pipeline.VGGT_model = vggt_model.to(pipeline.device)
    del pipeline.VGGT_model.depth_head
    del pipeline.VGGT_model.track_head
    pipeline.VGGT_model.eval()

    pipeline.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
        birefnet_model_path,
        trust_remote_code=True,
    ).to(pipeline.device)
    pipeline.birefnet_model.eval()

    pipeline._device = torch.device("cuda")
    pipeline.low_vram = True
    pipeline.birefnet_model.cuda()
    return pipeline


def load_dreamsim_from_local_cache(pipeline, cache_dir=None):
    from trellis.pipelines.trellis_image_to_3d import dreamsim  # noqa: WPS433

    cache_dir = os.path.abspath(cache_dir or os.path.join(BASE_DIR, "weights", "dreamsim"))
    dino_repo_dir = os.path.join(cache_dir, "facebookresearch_dino_main")
    dino_hubconf = os.path.join(dino_repo_dir, "hubconf.py")
    if not os.path.exists(dino_hubconf):
        raise FileNotFoundError(
            "DreamSim DINO local torch hub cache is missing. Expected: "
            f"{dino_hubconf}"
        )

    original_torch_hub_load = torch.hub.load

    def _load_local_dino(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dino:main":
            kwargs = dict(kwargs)
            kwargs["source"] = "local"
            return original_torch_hub_load(dino_repo_dir, model, *args, **kwargs)
        return original_torch_hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = _load_local_dino
    try:
        model, _ = dreamsim(
            pretrained=True,
            device=pipeline.device,
            dreamsim_type="dino_vitb16",
            cache_dir=cache_dir,
        )
    finally:
        torch.hub.load = original_torch_hub_load

    pipeline.dreamsim_model = model
    pipeline.dreamsim_model.eval()
    return cache_dir


def make_unique_output_dir(output_root, dataset_name):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, f"{dataset_name}_{timestamp}")
    suffix = 1
    while os.path.exists(output_dir):
        output_dir = os.path.join(output_root, f"{dataset_name}_{timestamp}_{suffix:02d}")
        suffix += 1
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_dataset_meta(dataset_dir):
    meta_path = os.path.join(dataset_dir, "reconviagen_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def frame_id_from_name(name, fallback):
    match = re.search(r"frame_(\d+)", os.path.basename(name), re.IGNORECASE)
    return int(match.group(1)) if match else int(fallback)


def apply_mask_and_crop_with_meta(image_path, mask_path, resolution=518):
    image = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    rgba = np.array(image)
    alpha = np.array(mask)
    rgba[:, :, 3] = np.where(alpha > 127, 255, 0).astype(np.uint8)
    output = Image.fromarray(rgba, mode="RGBA")

    ys, xs = np.nonzero(rgba[:, :, 3] > 204)
    if len(xs) == 0:
        width, height = output.size
        side = max(width, height)
        resized = output.resize((resolution, resolution), Image.Resampling.BILINEAR)
        return resized, {
            "original_size": [int(width), int(height)],
            "crop_box": [0, 0, int(width), int(height)],
            "pad": [0, 0],
            "side": int(side),
            "scale": float(resolution / float(side)),
            "empty_mask": True,
        }

    left, right = xs.min(), xs.max()
    top, bottom = ys.min(), ys.max()
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    size = max(1, int(max(right - left, bottom - top) * 1.1))

    crop_box = (
        max(0, int(center_x - size // 2)),
        max(0, int(center_y - size // 2)),
        min(output.width, int(center_x + size // 2)),
        min(output.height, int(center_y + size // 2)),
    )
    output = output.crop(crop_box)

    width, height = output.size
    side = max(width, height)
    pad_x = (side - width) // 2
    pad_y = (side - height) // 2
    padded = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    padded.paste(output, (pad_x, pad_y))
    padded = padded.resize((resolution, resolution), Image.Resampling.BILINEAR)

    arr = np.array(padded)
    fg = arr[:, :, 3] > 204
    arr[:, :, :3] = arr[:, :, :3] * fg[:, :, None]
    meta = {
        "original_size": [int(image.width), int(image.height)],
        "crop_box": [int(v) for v in crop_box],
        "pad": [int(pad_x), int(pad_y)],
        "side": int(side),
        "scale": float(resolution / float(side)),
        "empty_mask": False,
    }
    return Image.fromarray(arr, mode="RGBA"), meta


def apply_mask_and_crop(image_path, mask_path, resolution=518):
    image, _ = apply_mask_and_crop_with_meta(image_path, mask_path, resolution=resolution)
    return image


def load_from_dataset_masks(dataset_dir, resolution):
    meta = load_dataset_meta(dataset_dir)
    image_dir = os.path.join(dataset_dir, "rgb")
    if not os.path.isdir(image_dir):
        image_dir = os.path.join(dataset_dir, "images")
    mask_dir = os.path.join(dataset_dir, "masks")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found under {dataset_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    selected_names = meta.get("selected_frames") or sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    selected_images = []
    selected_indices = []
    loaded_names = []
    crop_metas_by_name = {}
    missing = []

    for order, name in enumerate(selected_names):
        image_path = os.path.join(image_dir, name)
        stem = os.path.splitext(name)[0]
        mask_path = os.path.join(mask_dir, f"{stem}.png")
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            missing.append({"image": image_path, "mask": mask_path})
            continue
        image, crop_meta = apply_mask_and_crop_with_meta(image_path, mask_path, resolution=resolution)
        selected_images.append(image)
        selected_indices.append(frame_id_from_name(name, order))
        loaded_names.append(name)
        crop_metas_by_name[name] = crop_meta

    if not selected_images:
        raise RuntimeError(f"No usable image/mask pairs found. Missing examples: {missing[:3]}")
    if missing:
        print(f"[Rebuild] 跳过 {len(missing)} 个缺失 image/mask 的帧")
    return selected_images, loaded_names, selected_indices, {
        "source": "dataset_masks",
        "dataset_dir": dataset_dir,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "resolution": resolution,
        "missing_count": len(missing),
        "crop_metas_by_name": crop_metas_by_name,
    }


def load_from_session_previews(dataset_dir):
    meta = load_dataset_meta(dataset_dir)
    preview_dir = meta.get("session_preview_dir")
    if not preview_dir:
        raise FileNotFoundError("reconviagen_meta.json has no session_preview_dir")
    selected_indices = meta.get("selected_indices") or []
    selected_names = meta.get("selected_frames") or [f"frame_{idx:04d}.jpg" for idx in selected_indices]
    if not selected_indices:
        selected_indices = list(range(len(selected_names)))

    selected_images = []
    loaded_names = []
    loaded_indices = []
    missing = []
    for order, idx in enumerate(selected_indices):
        preview_path = os.path.join(preview_dir, f"{idx}.png")
        if not os.path.exists(preview_path):
            missing.append(preview_path)
            continue
        selected_images.append(Image.open(preview_path).convert("RGBA"))
        loaded_names.append(selected_names[order] if order < len(selected_names) else f"frame_{idx:04d}.jpg")
        loaded_indices.append(int(idx))

    if not selected_images:
        raise RuntimeError(f"No session previews found. Missing examples: {missing[:3]}")
    if missing:
        print(f"[Rebuild] 跳过 {len(missing)} 个缺失 preview 的帧")
    return selected_images, loaded_names, loaded_indices, {
        "source": "session_previews",
        "dataset_dir": dataset_dir,
        "preview_dir": preview_dir,
        "missing_count": len(missing),
    }


def load_inputs(dataset_dir, source, resolution):
    if source == "dataset_masks":
        return load_from_dataset_masks(dataset_dir, resolution)
    if source == "session_previews":
        return load_from_session_previews(dataset_dir)
    raise ValueError(f"Unsupported source: {source}")


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = [float(v) for v in qvec]
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )


def read_colmap_cameras_txt(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(v) for v in parts[4:]]
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_colmap_images_txt(path):
    images = {}
    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        image_id = int(parts[0])
        qvec = [float(v) for v in parts[1:5]]
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float32)
        cam_id = int(parts[8])
        name = parts[9]
        images[name] = {
            "image_id": image_id,
            "cam_id": cam_id,
            "R": qvec_to_rotmat(qvec),
            "t": tvec,
        }
        i += 2
    return images


def colmap_camera_to_pixel_K(camera):
    model = camera["model"]
    params = camera["params"]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_RADIAL":
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        raise ValueError(f"Unsupported COLMAP camera model for refine: {model}")
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def crop_adjust_normalized_K(K_pixel, crop_meta, resolution):
    K = np.array(K_pixel, dtype=np.float32, copy=True)
    left, top, _, _ = crop_meta["crop_box"]
    pad_x, pad_y = crop_meta["pad"]
    scale = float(crop_meta["scale"])
    K[0, 2] = (K[0, 2] - float(left) + float(pad_x)) * scale
    K[1, 2] = (K[1, 2] - float(top) + float(pad_y)) * scale
    K[0, 0] *= scale
    K[1, 1] *= scale

    K[0, 0] /= float(resolution)
    K[1, 1] /= float(resolution)
    K[0, 2] /= float(resolution)
    K[1, 2] /= float(resolution)
    return K


def build_refine_cameras_from_sparse(dataset_dir, selected_names, input_info, resolution, device, radius=1.5):
    sparse_dir = os.path.join(dataset_dir, "sparse", "0")
    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    images_path = os.path.join(sparse_dir, "images.txt")
    if not os.path.exists(cameras_path) or not os.path.exists(images_path):
        raise FileNotFoundError(f"Missing sparse/0 cameras.txt or images.txt under {dataset_dir}")

    cameras = read_colmap_cameras_txt(cameras_path)
    images = read_colmap_images_txt(images_path)
    crop_metas = input_info.get("crop_metas_by_name") or {}

    centers = []
    frame_records = []
    for name in selected_names:
        if name not in images:
            raise KeyError(f"Selected frame is not in sparse/0 images.txt: {name}")
        record = images[name]
        camera = cameras[record["cam_id"]]
        R_w2c = record["R"].astype(np.float32)
        t_w2c = record["t"].astype(np.float32)
        center = -R_w2c.T @ t_w2c
        centers.append(center)
        frame_records.append((name, record, camera, R_w2c, center))

    centers = np.stack(centers, axis=0)
    centroid = centers.mean(axis=0)
    centered = centers - centroid[None]
    avg_radius = float(np.mean(np.linalg.norm(centered, axis=1)))
    scale = float(radius / (avg_radius + 1e-6))

    extrinsics = []
    intrinsics = []
    for name, record, camera, R_w2c, center in frame_records:
        center_norm = (center - centroid) * scale
        t_norm = -R_w2c @ center_norm
        T_w2c = np.eye(4, dtype=np.float32)
        T_w2c[:3, :3] = R_w2c
        T_w2c[:3, 3] = t_norm
        extrinsics.append(T_w2c)

        K_pixel = colmap_camera_to_pixel_K(camera)
        crop_meta = crop_metas.get(name)
        if crop_meta is None:
            crop_meta = {
                "crop_box": [0, 0, camera["width"], camera["height"]],
                "pad": [0, 0],
                "scale": float(resolution / float(max(camera["width"], camera["height"]))),
            }
        intrinsics.append(crop_adjust_normalized_K(K_pixel, crop_meta, resolution))

    print(
        f"[Rebuild] refine cameras: {len(extrinsics)} frames, "
        f"center={centroid.tolist()}, scale={scale:.6f}",
        flush=True,
    )
    return (
        torch.tensor(np.stack(extrinsics), dtype=torch.float32, device=device),
        torch.tensor(np.stack(intrinsics), dtype=torch.float32, device=device),
        {
            "source": "sparse/0",
            "frame_count": len(extrinsics),
            "center": [float(v) for v in centroid.tolist()],
            "scale": scale,
            "target_radius": float(radius),
            "intrinsics": "crop-adjusted normalized OpenCV intrinsics for 518x518 inputs",
        },
    )


def parse_seed_list(seed_text):
    if not seed_text:
        return get_candidate_seeds()
    seeds = []
    for item in seed_text.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    unique = []
    for seed in seeds:
        if seed not in unique:
            unique.append(seed)
        if len(unique) >= 3:
            break
    return unique or get_candidate_seeds()


def refine_best_candidate(
    pipeline,
    best_candidate,
    selected_images,
    extrinsics,
    intrinsics,
    appearance_lr,
    appearance_start_t,
    refine_steps,
    seed,
):
    print(
        "[Rebuild] run_refine: "
        f"appearance_lr={appearance_lr}, appearance_start_t={appearance_start_t}, steps={refine_steps}",
        flush=True,
    )
    device = pipeline.device
    slat_sampler_params = {}
    if refine_steps is not None and refine_steps > 0:
        slat_sampler_params["steps"] = int(refine_steps)

    dummy_input_points = torch.zeros((len(selected_images), 3), dtype=torch.long, device=device)
    refined_outputs = pipeline.run_refine(
        image=selected_images,
        ss_learning_rate=1e-3,
        ss_start_t=0.6,
        apperance_learning_rate=float(appearance_lr),
        apperance_start_t=float(appearance_start_t),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        ss_noise=best_candidate["ss_noise"],
        input_points=dummy_input_points,
        ss_refine_type="No",
        coords=best_candidate["coords"],
        seed=int(seed),
        sparse_structure_sampler_params={},
        slat_sampler_params=slat_sampler_params,
        formats=["gaussian", "mesh"],
        mode="multidiffusion",
    )
    best_candidate["outputs"] = refined_outputs
    torch.cuda.empty_cache()
    return {
        "enabled": True,
        "appearance_lr": float(appearance_lr),
        "appearance_start_t": float(appearance_start_t),
        "steps": int(refine_steps) if refine_steps else None,
        "seed": int(seed),
        "type": "appearance_only_coords_fixed",
    }


def export_best_candidate(best_candidate, output_dir, mesh_simplify):
    base_filename = os.path.join(output_dir, "reconstructed_object")
    outputs = best_candidate["outputs"]
    gs, mesh = outputs["gaussian"][0], outputs["mesh"][0]

    print("[Rebuild] 保存 Gaussian PLY...")
    gs.save_ply(f"{base_filename}.ply")

    print(f"[Rebuild] 保存 GLB mesh，simplify={mesh_simplify:.2f}...")
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=1024, verbose=False)
    glb.export(f"{base_filename}.glb")
    del glb

    print("[Rebuild] 渲染 MP4 预览...")
    video_color = render_utils.render_video(gs, num_frames=120)["color"]
    video_geo = render_utils.render_video(mesh, num_frames=120)["normal"]
    video = [np.concatenate([video_color[i], video_geo[i]], axis=1) for i in range(len(video_color))]
    imageio.mimsave(f"{base_filename}.mp4", video, fps=15)
    del video_color, video_geo, video
    torch.cuda.empty_cache()
    return {
        "ply": f"{base_filename}.ply",
        "glb": f"{base_filename}.glb",
        "mp4": f"{base_filename}.mp4",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild only ReconViaGen mesh from a prepared CoarseModel dataset."
    )
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--source", choices=["dataset_masks", "session_previews"], default="dataset_masks")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--resolution", type=int, default=518)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 0 or 0,1. Max 3.")
    parser.add_argument("--num_candidates", type=int, default=None, help="Used when --seeds is omitted. Max 3.")
    parser.add_argument("--mesh_simplify", type=float, default=None)
    parser.add_argument("--prepare_only", action="store_true", help="Only write input previews/manifests, skip model inference.")
    parser.add_argument("--run_refine", action="store_true", help="Run Trellis run_refine after the best base candidate is selected.")
    parser.add_argument("--refine_steps", type=int, default=20, help="SLAT sampler steps used by run_refine. Original default is 50.")
    parser.add_argument("--refine_appearance_lr", type=float, default=1e-3)
    parser.add_argument("--refine_appearance_start_t", type=float, default=0.6)
    parser.add_argument("--refine_camera_radius", type=float, default=1.5)
    parser.add_argument("--dreamsim_cache_dir", default=os.path.join(BASE_DIR, "weights", "dreamsim"))
    parser.add_argument("--trellis_model_path", default=None, help="Optional local Stable-X/trellis-vggt-v0-2 snapshot path.")
    parser.add_argument("--vggt_model_path", default=None, help="Optional local Stable-X/vggt-object-v0-1 snapshot path.")
    parser.add_argument("--birefnet_model_path", default=None, help="Optional local ZhengPeng7/BiRefNet snapshot path.")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    dataset_name = os.path.basename(dataset_dir.rstrip(os.sep))
    output_root = args.output_root or os.path.join(dataset_dir, "reconviagen_rebuild")
    os.makedirs(output_root, exist_ok=True)
    output_dir = make_unique_output_dir(output_root, dataset_name)

    if args.num_candidates is not None and args.seeds is None:
        os.environ["RECON_NUM_CANDIDATES"] = str(max(1, min(int(args.num_candidates), 3)))
    candidate_seeds = parse_seed_list(args.seeds)
    mesh_simplify = get_mesh_simplify_ratio() if args.mesh_simplify is None else float(np.clip(args.mesh_simplify, 0.0, 0.95))

    print(f"[Rebuild] dataset: {dataset_dir}")
    print(f"[Rebuild] output:  {output_dir}")
    print(f"[Rebuild] source:  {args.source}")
    print(f"[Rebuild] seeds:   {candidate_seeds}")

    selected_images, selected_names, selected_indices, input_info = load_inputs(
        dataset_dir,
        args.source,
        args.resolution,
    )
    frame_filter = {
        "enabled": False,
        "reason": "rebuild_from_coarse_dataset",
        "source": args.source,
    }
    input_manifest = save_generation_inputs(
        output_dir,
        selected_images,
        selected_names,
        selected_indices,
        selected_indices,
        frame_filter,
        candidate_seeds,
    )

    if args.prepare_only:
        report = {
            "dataset_dir": dataset_dir,
            "dataset_name": dataset_name,
            "source": args.source,
            "input_info": input_info,
            "selected_indices": selected_indices,
            "selected_names": selected_names,
            "candidate_seeds": candidate_seeds,
            "mesh_simplify": mesh_simplify,
            "input_manifest": input_manifest,
            "output_files": None,
            "prepare_only": True,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(output_dir, "rebuild_report.json"), "w") as f:
            json.dump(report, f, indent=4)
        print("[Rebuild] prepare_only 完成")
        print(f"[Rebuild] inputs: {os.path.join(output_dir, 'inputs')}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this shell. ReconViaGen/Trellis mesh generation requires a CUDA GPU. "
            "Run this script on the same GPU-enabled environment used by server.py/run_local.py."
        )

    pipeline = init_pipeline_from_local_cache(
        trellis_model_path=args.trellis_model_path,
        vggt_model_path=args.vggt_model_path,
        birefnet_model_path=args.birefnet_model_path,
    )
    best_candidate, candidate_report = run_limited_candidate_generation(
        pipeline,
        selected_images,
        selected_names,
        dataset_dir,
        candidate_seeds,
        pose_rerank_enabled=arq.pose_rerank_enabled_from_env(),
        pose_rerank_weight=arq.pose_rerank_weight_from_env(),
    )
    refine_report = {"enabled": False}
    refine_camera_report = None
    if args.run_refine:
        dreamsim_cache_dir = load_dreamsim_from_local_cache(pipeline, args.dreamsim_cache_dir)
        extrinsics, intrinsics, refine_camera_report = build_refine_cameras_from_sparse(
            dataset_dir,
            selected_names,
            input_info,
            args.resolution,
            pipeline.device,
            radius=args.refine_camera_radius,
        )
        refine_report = refine_best_candidate(
            pipeline,
            best_candidate,
            selected_images,
            extrinsics,
            intrinsics,
            appearance_lr=args.refine_appearance_lr,
            appearance_start_t=args.refine_appearance_start_t,
            refine_steps=args.refine_steps,
            seed=best_candidate["seed"],
        )
        refine_report["dreamsim_cache_dir"] = dreamsim_cache_dir
    output_files = export_best_candidate(best_candidate, output_dir, mesh_simplify)

    report = {
        "dataset_dir": dataset_dir,
        "dataset_name": dataset_name,
        "source": args.source,
        "input_info": input_info,
        "selected_indices": selected_indices,
        "selected_names": selected_names,
        "candidate_seeds": candidate_seeds,
        "selected_seed": int(best_candidate["seed"]),
        "selected_candidate": best_candidate["metrics"],
        "candidates": candidate_report,
        "pose_rerank_enabled": arq.pose_rerank_enabled_from_env(),
        "pose_rerank_weight": arq.pose_rerank_weight_from_env(),
        "pose_rerank_note": "Enabled at the candidate API level; for COLMAP-only datasets without poses.txt it is skipped with reason=no_pose_file.",
        "refine": refine_report,
        "refine_camera": refine_camera_report,
        "mesh_simplify": mesh_simplify,
        "input_manifest": input_manifest,
        "output_files": output_files,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(output_dir, "rebuild_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print("[Rebuild] 完成")
    print(json.dumps(output_files, indent=4))


if __name__ == "__main__":
    main()
