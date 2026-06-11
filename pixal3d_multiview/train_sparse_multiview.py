from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")

TRACKER_ROOT = Path(__file__).resolve().parents[1]
PIXAL3D_ROOT = TRACKER_ROOT / "Pixal3D"
sys.path.insert(0, str(TRACKER_ROOT))
sys.path.insert(0, str(PIXAL3D_ROOT))

from pixal3d_multiview.multiview_projection import (  # noqa: E402
    estimate_object_volume_from_visual_hull,
)


IMAGE_COND_CONFIG = {
    "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    "image_size": 512,
    "grid_resolution": 16,
}

POSE_MODES = ("correct", "shuffle", "reverse", "noise", "large_noise", "identity")


def _resolve(root: Optional[str], path: str) -> str:
    if os.path.isabs(path) or not root:
        return path
    return os.path.join(root, path)


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_sparse_flow_model(model_path: str, sparse_flow_model: str, device: torch.device):
    from pixal3d import models as pixal3d_models

    if sparse_flow_model:
        model = pixal3d_models.from_pretrained(sparse_flow_model)
    else:
        from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline

        class SparseOnlyPixal3DPipeline(Pixal3DImageTo3DPipeline):
            model_names_to_load = ["sparse_structure_flow_model"]

        pipe = SparseOnlyPixal3DPipeline.from_pretrained(model_path)
        model = pipe.models["sparse_structure_flow_model"]
    return model.to(device)


def build_image_cond_model(device: torch.device, grid_resolution: int, model_name: str):
    try:
        from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    except ImportError as exc:
        raise ImportError(
            "Failed to import Pixal3D DinoV3ProjFeatureExtractor. The active environment likely has a "
            "transformers build without DINOv3ViTModel. Install Pixal3D's required transformers version "
            "or run this script in the Pixal3D-compatible environment."
        ) from exc

    cfg = dict(IMAGE_COND_CONFIG)
    cfg["grid_resolution"] = int(grid_resolution)
    cfg["model_name"] = model_name
    model = DinoV3ProjFeatureExtractor(**cfg)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_view_aggregator(args: argparse.Namespace, image_cond_model, device: torch.device):
    mode = getattr(args, "view_aggregator", "none")
    if mode == "none":
        return None
    if mode != "gated":
        raise ValueError(f"Unknown view_aggregator: {mode}")
    if bool(getattr(image_cond_model, "use_naf_upsample", False)):
        raise ValueError("The gated view aggregator currently supports the LR DINO projection path only")
    from pixal3d_multiview.view_aggregator import ViewGatedAggregator

    feature_dim = int(getattr(image_cond_model, "embed_dim", image_cond_model.model.config.hidden_size))
    return ViewGatedAggregator(
        feature_dim=feature_dim,
        geom_dim=int(args.view_aggregator_geom_dim),
        reduced_dim=int(args.view_aggregator_reduced_dim),
        hidden_dim=int(args.view_aggregator_hidden_dim),
        dropout=float(args.view_aggregator_dropout),
        residual_scale=float(args.view_aggregator_residual_scale),
    ).to(device)


def build_geometry_adapter(args: argparse.Namespace, image_cond_model, device: torch.device):
    mode = getattr(args, "geometry_adapter", "none")
    if mode == "none":
        return None
    if mode != "mlp":
        raise ValueError(f"Unknown geometry_adapter: {mode}")
    from pixal3d_multiview.geometry_adapter import GEOMETRY_FEATURE_DIM, GeometryConsistencyAdapter

    feature_dim = int(getattr(image_cond_model, "embed_dim", image_cond_model.model.config.hidden_size))
    if bool(getattr(image_cond_model, "use_naf_upsample", False)):
        feature_dim *= 2
    geometry_dim = int(args.geometry_adapter_dim or GEOMETRY_FEATURE_DIM)
    return GeometryConsistencyAdapter(
        feature_dim=feature_dim,
        geometry_dim=geometry_dim,
        hidden_dim=int(args.geometry_adapter_hidden_dim),
        dropout=float(args.geometry_adapter_dropout),
        residual_scale=float(args.geometry_adapter_residual_scale),
    ).to(device)


class MultiviewSparseManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        *,
        image_root: Optional[str] = None,
        mask_root: Optional[str] = None,
        latent_root: Optional[str] = None,
        max_frames: int = 8,
        apply_mask: bool = True,
    ):
        self.manifest_path = manifest
        self.image_root = image_root
        self.mask_root = mask_root
        self.latent_root = latent_root
        self.max_frames = int(max_frames)
        self.apply_mask = bool(apply_mask)
        data = _load_json(manifest)
        samples = data.get("samples", data.get("instances", data.get("framesets", None)))
        if samples is None and isinstance(data, list):
            samples = data
        if samples is None:
            raise ValueError("Training manifest should contain samples/instances/framesets list")
        self.samples = samples
        self.top_intrinsic = data.get("intrinsic")
        self.extrinsics_type = data.get("extrinsics_type", "c2w")
        self.default_image_root = data.get("image_root", image_root)
        self.default_mask_root = data.get("mask_root", mask_root)
        self.default_latent_root = data.get("latent_root", latent_root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        uid = str(sample.get("uid", sample.get("id", index)))
        frames = sample.get("frames")
        if frames is None:
            raise ValueError(f"sample {uid} has no frames")
        frames = frames[: self.max_frames] if self.max_frames > 0 else frames
        if len(frames) == 0:
            raise ValueError(f"sample {uid} has no usable frames")

        image_root = sample.get("image_root", self.default_image_root)
        mask_root = sample.get("mask_root", self.default_mask_root)
        latent_root = sample.get("latent_root", self.default_latent_root)
        top_intrinsic = sample.get("intrinsic", self.top_intrinsic)
        extrinsics_type = sample.get("extrinsics_type", self.extrinsics_type)

        images = []
        masks = []
        intrinsics = []
        extrinsics = []
        source_sizes = []
        for frame in frames:
            image_path = _resolve(image_root, frame["image"])
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            source_sizes.append((width, height))

            mask_path = frame.get("mask", sample.get("mask"))
            if mask_path is not None:
                mask = Image.open(_resolve(mask_root, mask_path)).convert("L")
                if mask.size != image.size:
                    mask = mask.resize(image.size, Image.NEAREST)
                mask_arr = np.asarray(mask).astype(np.float32) / 255.0
            else:
                mask_arr = np.ones((height, width), dtype=np.float32)

            if self.apply_mask:
                rgb = np.asarray(image).astype(np.float32) / 255.0
                rgb = rgb * mask_arr[..., None]
                image = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))

            intrinsic = frame.get("intrinsic", top_intrinsic)
            if intrinsic is None:
                raise ValueError(f"missing intrinsic for sample={uid} frame={frame.get('image')}")
            images.append(image)
            masks.append(torch.from_numpy(mask_arr[None]))
            intrinsics.append(torch.tensor(intrinsic, dtype=torch.float32))
            extrinsics.append(torch.tensor(frame["extrinsic"], dtype=torch.float32))

        latent_path = sample.get("ss_latent", sample.get("ss_latent_path", sample.get("latent")))
        if latent_path is None:
            raise ValueError(f"sample {uid} has no ss_latent/ss_latent_path")
        latent_npz = np.load(_resolve(latent_root, latent_path))
        if "z" not in latent_npz:
            raise ValueError(f"sparse latent npz should contain key 'z': {latent_path}")
        x_0 = torch.from_numpy(latent_npz["z"]).float()
        if x_0.ndim == 5 and x_0.shape[0] == 1:
            x_0 = x_0[0]
        if x_0.ndim != 4:
            raise ValueError(f"expected sparse latent z [C,D,H,W], got {tuple(x_0.shape)} for {latent_path}")

        return {
            "uid": uid,
            "images": images,
            "masks": torch.stack(masks, dim=0),
            "intrinsics": torch.stack(intrinsics, dim=0),
            "extrinsics": torch.stack(extrinsics, dim=0),
            "source_sizes": source_sizes,
            "extrinsics_type": extrinsics_type,
            "x_0": x_0,
            "latent_path": _resolve(latent_root, latent_path),
        }


def collate_single(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("pixal3d_multiview sparse training currently supports batch_size=1")
    return batch[0]


def apply_pose_mode(batch: dict, pose_mode: str, seed: int) -> dict:
    """Return a shallow batch copy with intentionally perturbed camera poses.

    This is for evaluation ablations only. Images, masks and intrinsics stay in
    their original order so `shuffle` breaks the view-pose correspondence.
    """
    pose_mode = str(pose_mode).lower()
    if pose_mode == "correct":
        return batch
    if "extrinsics" not in batch:
        raise ValueError("batch has no extrinsics for pose ablation")

    out = dict(batch)
    extrinsics = batch["extrinsics"].clone()
    extrinsics_type = str(batch.get("extrinsics_type", "c2w")).lower()
    if pose_mode == "shuffle":
        num_views = int(extrinsics.shape[0])
        if num_views > 1:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            perm = torch.randperm(num_views, generator=gen)
            if torch.equal(perm, torch.arange(num_views)):
                perm = torch.roll(perm, shifts=1)
            extrinsics = extrinsics[perm]
            out["pose_permutation"] = perm.tolist()
        else:
            out["pose_permutation"] = [0]
    elif pose_mode == "reverse":
        num_views = int(extrinsics.shape[0])
        perm = torch.arange(num_views - 1, -1, -1)
        extrinsics = extrinsics[perm]
        out["pose_permutation"] = perm.tolist()
    elif pose_mode in {"noise", "large_noise"}:
        c2w = extrinsics if extrinsics_type == "c2w" else torch.linalg.inv(extrinsics)
        rot_deg = 35.0 if pose_mode == "noise" else 90.0
        trans_scale = 0.25 if pose_mode == "noise" else 0.75
        c2w = perturb_c2w_poses(c2w, seed=seed, max_rot_deg=rot_deg, trans_scale=trans_scale)
        extrinsics = c2w if extrinsics_type == "c2w" else torch.linalg.inv(c2w)
        out["pose_permutation"] = None
        out["pose_noise"] = {"max_rot_deg": rot_deg, "trans_scale": trans_scale}
    elif pose_mode == "identity":
        eye = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device)
        extrinsics = eye.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
        out["pose_permutation"] = None
    else:
        raise ValueError(f"Unknown pose_mode: {pose_mode}")
    out["extrinsics"] = extrinsics
    out["pose_mode"] = pose_mode
    return out


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = torch.nn.functional.normalize(axis.float(), dim=0)
    x, y, z = axis[0], axis[1], axis[2]
    c = torch.cos(angle.float())
    s = torch.sin(angle.float())
    one = torch.ones_like(c)
    zero = torch.zeros_like(c)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, device=axis.device, dtype=torch.float32)
    outer = axis[:, None] * axis[None, :]
    return c * eye + (one - c) * outer + s * k


def perturb_c2w_poses(c2w: torch.Tensor, *, seed: int, max_rot_deg: float, trans_scale: float) -> torch.Tensor:
    device = c2w.device
    dtype = c2w.dtype
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    out = c2w.clone()
    for view_idx in range(int(c2w.shape[0])):
        axis = torch.randn(3, generator=gen).to(device=device, dtype=torch.float32)
        angle = (torch.rand((), generator=gen).to(device=device) * 2.0 - 1.0) * float(max_rot_deg) * torch.pi / 180.0
        rot_delta = axis_angle_to_matrix(axis, angle).to(device=device, dtype=dtype)
        trans_delta = (torch.rand(3, generator=gen).to(device=device, dtype=dtype) * 2.0 - 1.0) * float(trans_scale)
        out[view_idx, :3, :3] = rot_delta @ out[view_idx, :3, :3]
        out[view_idx, :3, 3] = out[view_idx, :3, 3] + trans_delta
    return out


def parse_sample_indices(spec: str, dataset_size: int) -> list[int]:
    if not spec:
        return []
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid descending sample index range: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    bad = [idx for idx in indices if idx < 0 or idx >= dataset_size]
    if bad:
        raise IndexError(f"sample_indices out of range for dataset size {dataset_size}: {bad}")
    return indices


def configure_trainable(model: torch.nn.Module, mode: str) -> int:
    mode = mode.lower()
    trainable = 0
    for name, param in model.named_parameters():
        if mode == "none":
            flag = False
        elif mode == "all":
            flag = True
        elif mode == "proj_only":
            flag = "cross_attn" in name or "proj" in name
        elif mode == "proj_linear":
            flag = "proj_linear" in name
        else:
            raise ValueError(f"Unknown trainable mode: {mode}")
        param.requires_grad_(flag)
        if flag:
            trainable += param.numel()
    return trainable


def sample_t(batch_size: int, device: torch.device, schedule: str) -> torch.Tensor:
    if schedule == "uniform":
        return torch.rand(batch_size, device=device)
    if schedule == "logitNormal":
        return torch.sigmoid(torch.randn(batch_size, device=device) + 1.0)
    raise ValueError(f"Unknown t_schedule: {schedule}")


def diffuse(x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, sigma_min: float) -> torch.Tensor:
    t_view = t.view(-1, *[1 for _ in range(x_0.ndim - 1)])
    return (1.0 - t_view) * x_0 + (sigma_min + (1.0 - sigma_min) * t_view) * noise


def velocity_target(x_0: torch.Tensor, noise: torch.Tensor, sigma_min: float) -> torch.Tensor:
    return (1.0 - sigma_min) * noise - x_0


def make_multiview_condition(
    condition_builder,
    image_cond_model,
    batch: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    masks = batch["masks"].to(device=device, dtype=torch.float32)
    intrinsics = batch["intrinsics"].to(device=device, dtype=torch.float32)
    extrinsics = batch["extrinsics"].to(device=device, dtype=torch.float32)
    extrinsics_are_c2w = str(batch["extrinsics_type"]).lower() == "c2w"

    object_to_world = None
    volume_extent = None
    if not args.no_auto_volume:
        estimate = estimate_object_volume_from_visual_hull(
            masks,
            intrinsics,
            extrinsics,
            extrinsics_are_c2w=extrinsics_are_c2w,
            camera_forward_sign=args.camera_forward_sign,
            mask_threshold=args.mask_threshold,
            resolution=args.vh_volume_resolution,
            min_visible_views=args.vh_min_visible_views,
            min_support_views=args.vh_min_support_views,
            min_support_ratio=args.vh_min_support_ratio,
            initial_extent_ratio=args.vh_volume_initial_extent_ratio,
            padding=args.vh_volume_padding,
            min_extent=args.vh_volume_min_extent,
            refine_steps=args.vh_volume_refine_steps,
        )
        object_to_world = estimate.object_to_world
        volume_extent = float(estimate.extent_world)
        condition_builder.last_multiview_stats["object_volume_estimate"] = estimate.to_dict()

    if args.visibility_depth_tolerance > 0:
        depth_tolerance = float(args.visibility_depth_tolerance)
    elif volume_extent is not None:
        depth_tolerance = max(volume_extent * float(args.visibility_depth_tolerance_ratio), 1e-4)
    else:
        depth_tolerance = 0.03

    condition_builder._front_depth_cache = {}
    condition_builder._visibility_enabled = bool((not args.no_visibility_depth) and object_to_world is not None)
    condition_builder._visibility_depth_tolerance = depth_tolerance
    condition_builder._visibility_weight_min = float(args.visibility_weight_min)
    condition_builder._vh_visibility_resolution = int(args.vh_visibility_resolution)
    condition_builder._vh_visibility_dilation = int(args.vh_visibility_dilation)
    condition_builder._vh_min_visible_views = int(args.vh_min_visible_views)
    condition_builder._vh_min_support_views = int(args.vh_min_support_views)
    condition_builder._vh_min_support_ratio = float(args.vh_min_support_ratio)
    condition_builder.image_cond_model_ss = image_cond_model

    cond_pack = condition_builder.get_multiview_proj_cond_ss(
        batch["images"],
        intrinsics,
        extrinsics,
        batch["source_sizes"],
        masks=masks,
        extrinsics_are_c2w=extrinsics_are_c2w,
        camera_forward_sign=args.camera_forward_sign,
        object_to_world=object_to_world,
        mask_threshold=args.mask_threshold,
        empty_policy=getattr(args, "empty_policy", "zero"),
        fallback_weight=getattr(args, "fallback_weight", 1.0),
        support_confidence_power=getattr(args, "support_confidence_power", 1.0),
        global_fusion=getattr(args, "global_fusion", "concat"),
        geometry_feature_mode=getattr(args, "geometry_feature_mode", "none"),
        geometry_feature_scale=getattr(args, "geometry_feature_scale", 1.0),
    )
    return cond_pack["cond"]


def save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    args: argparse.Namespace,
    view_aggregator: Optional[torch.nn.Module] = None,
    geometry_adapter: Optional[torch.nn.Module] = None,
    name: str = "last.pt",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    if view_aggregator is not None:
        payload["view_aggregator"] = view_aggregator.state_dict()
        payload["view_aggregator_config"] = {
            "type": getattr(args, "view_aggregator", "none"),
            "geom_dim": getattr(args, "view_aggregator_geom_dim", None),
            "reduced_dim": getattr(args, "view_aggregator_reduced_dim", None),
            "hidden_dim": getattr(args, "view_aggregator_hidden_dim", None),
            "dropout": getattr(args, "view_aggregator_dropout", None),
            "residual_scale": getattr(args, "view_aggregator_residual_scale", None),
        }
    if geometry_adapter is not None:
        payload["geometry_adapter"] = geometry_adapter.state_dict()
        payload["geometry_adapter_config"] = {
            "type": getattr(args, "geometry_adapter", "none"),
            "geometry_dim": getattr(args, "geometry_adapter_dim", None),
            "hidden_dim": getattr(args, "geometry_adapter_hidden_dim", None),
            "dropout": getattr(args, "geometry_adapter_dropout", None),
            "residual_scale": getattr(args, "geometry_adapter_residual_scale", None),
        }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    path = output_dir / name
    torch.save(payload, path)
    torch.save(payload, output_dir / "last.pt")


def load_resume(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler=None,
    view_aggregator: Optional[torch.nn.Module] = None,
    geometry_adapter: Optional[torch.nn.Module] = None,
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    if view_aggregator is not None:
        if "view_aggregator" not in state:
            raise ValueError(
                f"--resume checkpoint has no view_aggregator state, "
                f"but current --view_aggregator is not none: {path}"
            )
        view_aggregator.load_state_dict(state["view_aggregator"], strict=False)
    if geometry_adapter is not None:
        if "geometry_adapter" not in state:
            raise ValueError(
                f"--resume checkpoint has no geometry_adapter state, "
                f"but current --geometry_adapter is not none: {path}"
            )
        geometry_adapter.load_state_dict(state["geometry_adapter"], strict=False)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("step", 0)), int(state.get("epoch", 0))


def load_init_weights(
    path: str,
    model: torch.nn.Module,
    view_aggregator: Optional[torch.nn.Module] = None,
    geometry_adapter: Optional[torch.nn.Module] = None,
) -> dict:
    state = torch.load(path, map_location="cpu")
    model_missing, model_unexpected = model.load_state_dict(state["model"], strict=False)
    info = {
        "path": path,
        "source_step": int(state.get("step", 0)),
        "source_epoch": int(state.get("epoch", 0)),
        "model_missing": len(model_missing),
        "model_unexpected": len(model_unexpected),
    }
    if view_aggregator is not None:
        if "view_aggregator" not in state:
            raise ValueError(
                f"--init_weights checkpoint has no view_aggregator state, "
                f"but current --view_aggregator is not none: {path}"
            )
        view_missing, view_unexpected = view_aggregator.load_state_dict(state["view_aggregator"], strict=False)
        info["view_aggregator_missing"] = len(view_missing)
        info["view_aggregator_unexpected"] = len(view_unexpected)
    if geometry_adapter is not None:
        if "geometry_adapter" in state:
            geom_missing, geom_unexpected = geometry_adapter.load_state_dict(state["geometry_adapter"], strict=False)
            info["geometry_adapter_missing"] = len(geom_missing)
            info["geometry_adapter_unexpected"] = len(geom_unexpected)
            info["geometry_adapter_initialized"] = "checkpoint"
        else:
            info["geometry_adapter_missing"] = None
            info["geometry_adapter_unexpected"] = None
            info["geometry_adapter_initialized"] = "fresh_zero_init"
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pixal3D sparse flow with independent multi-view visual-hull projection condition.")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--mask_root", default=None)
    parser.add_argument("--latent_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="TencentARC/Pixal3D")
    parser.add_argument(
        "--sparse_flow_model",
        default="TencentARC/Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16",
        help="Direct Pixal3D sparse flow model path without .json suffix. Empty string falls back to model_path pipeline parsing.",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--init_weights",
        default="",
        help=(
            "Load model/view-aggregator weights only, without optimizer/scaler/step/epoch. "
            "Use this when changing trainable parameters from a previous checkpoint."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_frames", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--sample_indices",
        default="",
        help="Optional comma/range subset of manifest sample indices, e.g. '0' or '0,3,5-7'. Useful for overfit checks.",
    )
    parser.add_argument("--no_shuffle", action="store_true", help="Disable DataLoader shuffling.")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--trainable", choices=["none", "proj_only", "proj_linear", "all"], default="proj_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--amp_dtype", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--t_schedule", choices=["uniform", "logitNormal"], default="logitNormal")
    parser.add_argument("--sigma_min", type=float, default=1e-5)
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1)
    parser.add_argument("--image_cond_model", default=IMAGE_COND_CONFIG["model_name"])
    parser.add_argument("--ss_grid_resolution", type=int, default=16)
    parser.add_argument("--camera_forward_sign", type=float, default=1.0)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--no_apply_mask", action="store_true")
    parser.add_argument("--no_auto_volume", action="store_true")
    parser.add_argument("--vh_min_visible_views", type=int, default=1)
    parser.add_argument("--vh_min_support_views", type=int, default=2)
    parser.add_argument("--vh_min_support_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_resolution", type=int, default=48)
    parser.add_argument("--vh_volume_initial_extent_ratio", type=float, default=0.6)
    parser.add_argument("--vh_volume_padding", type=float, default=1.25)
    parser.add_argument("--vh_volume_min_extent", type=float, default=0.05)
    parser.add_argument("--vh_volume_refine_steps", type=int, default=2)
    parser.add_argument("--no_visibility_depth", action="store_true")
    parser.add_argument("--vh_visibility_resolution", type=int, default=48)
    parser.add_argument("--vh_visibility_dilation", type=int, default=3)
    parser.add_argument("--visibility_depth_tolerance", type=float, default=0.0)
    parser.add_argument("--visibility_depth_tolerance_ratio", type=float, default=0.15)
    parser.add_argument("--visibility_weight_min", type=float, default=0.05)
    parser.add_argument("--empty_policy", choices=["zero", "visible", "border", "soft"], default="zero")
    parser.add_argument("--fallback_weight", type=float, default=1.0)
    parser.add_argument("--support_confidence_power", type=float, default=1.0)
    parser.add_argument("--global_fusion", choices=["concat", "mean", "first"], default="concat")
    parser.add_argument(
        "--geometry_feature_mode",
        choices=["none", "add", "replace"],
        default="none",
        help="Inject explicit geometry support features into cond['proj'] without changing its shape.",
    )
    parser.add_argument("--geometry_feature_scale", type=float, default=1.0)
    parser.add_argument("--view_aggregator", choices=["none", "gated"], default="none")
    parser.add_argument("--view_aggregator_geom_dim", type=int, default=11)
    parser.add_argument("--view_aggregator_reduced_dim", type=int, default=128)
    parser.add_argument("--view_aggregator_hidden_dim", type=int, default=256)
    parser.add_argument("--view_aggregator_dropout", type=float, default=0.0)
    parser.add_argument("--view_aggregator_residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--freeze_view_aggregator",
        action="store_true",
        help="Use the loaded view aggregator as a frozen conditioner while training sparse flow parameters.",
    )
    parser.add_argument("--geometry_adapter", choices=["none", "mlp"], default="none")
    parser.add_argument("--geometry_adapter_dim", type=int, default=0, help="0 uses the default explicit geometry feature dimension.")
    parser.add_argument("--geometry_adapter_hidden_dim", type=int, default=256)
    parser.add_argument("--geometry_adapter_dropout", type=float, default=0.0)
    parser.add_argument("--geometry_adapter_residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--freeze_geometry_adapter",
        action="store_true",
        help="Load but do not train the explicit geometry adapter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("This first sparse trainer supports --batch_size 1 only.")
    if args.resume and args.init_weights:
        raise ValueError("Use either --resume for exact continuation or --init_weights for weights-only initialization, not both.")
    if args.freeze_view_aggregator and args.view_aggregator == "none":
        raise ValueError("--freeze_view_aggregator requires --view_aggregator gated.")
    if args.freeze_view_aggregator and not (args.init_weights or args.resume):
        raise ValueError("--freeze_view_aggregator requires --init_weights or --resume to avoid freezing a fresh random aggregator.")
    if args.freeze_geometry_adapter and args.geometry_adapter == "none":
        raise ValueError("--freeze_geometry_adapter requires --geometry_adapter mlp.")
    if args.freeze_geometry_adapter and not (args.init_weights or args.resume):
        raise ValueError("--freeze_geometry_adapter requires --init_weights or --resume to avoid freezing a fresh random adapter.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    base_dataset = MultiviewSparseManifestDataset(
        args.train_manifest,
        image_root=args.image_root,
        mask_root=args.mask_root,
        latent_root=args.latent_root,
        max_frames=args.max_frames,
        apply_mask=not args.no_apply_mask,
    )
    selected_indices = parse_sample_indices(args.sample_indices, len(base_dataset))
    dataset = Subset(base_dataset, selected_indices) if selected_indices else base_dataset
    if selected_indices:
        selected_rows = [
            {
                "index": idx,
                "uid": str(base_dataset.samples[idx].get("uid", base_dataset.samples[idx].get("id", idx))),
            }
            for idx in selected_indices
        ]
        with open(output_dir / "selected_samples.json", "w") as f:
            json.dump(selected_rows, f, indent=2)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=bool((not args.no_shuffle) and len(dataset) > 1),
        num_workers=args.num_workers,
        collate_fn=collate_single,
    )

    print(
        f"[train_sparse_multiview] samples={len(dataset)} "
        f"base_samples={len(base_dataset)} sample_indices={selected_indices or 'all'} output={output_dir}"
    )
    image_cond_model = build_image_cond_model(device, args.ss_grid_resolution, args.image_cond_model)
    view_aggregator = build_view_aggregator(args, image_cond_model, device)
    if view_aggregator is not None:
        if args.freeze_view_aggregator:
            view_aggregator.eval()
            for param in view_aggregator.parameters():
                param.requires_grad_(False)
        else:
            view_aggregator.train()
    view_aggregator_count = sum(p.numel() for p in view_aggregator.parameters() if p.requires_grad) if view_aggregator is not None else 0
    geometry_adapter = build_geometry_adapter(args, image_cond_model, device)
    if geometry_adapter is not None:
        if args.freeze_geometry_adapter:
            geometry_adapter.eval()
            for param in geometry_adapter.parameters():
                param.requires_grad_(False)
        else:
            geometry_adapter.train()
    geometry_adapter_count = sum(p.numel() for p in geometry_adapter.parameters() if p.requires_grad) if geometry_adapter is not None else 0

    denoiser = load_sparse_flow_model(args.model_path, args.sparse_flow_model, device)
    trainable_count = configure_trainable(denoiser, args.trainable)
    if trainable_count + view_aggregator_count + geometry_adapter_count <= 0:
        raise ValueError(
            f"No trainable parameters selected: --trainable {args.trainable}, "
            f"--view_aggregator {args.view_aggregator}, --geometry_adapter {args.geometry_adapter}"
        )
    if trainable_count > 0:
        denoiser.train()
    else:
        denoiser.eval()
    if args.init_weights:
        init_info = load_init_weights(args.init_weights, denoiser, view_aggregator, geometry_adapter)
        print(f"[train_sparse_multiview] initialized weights only: {json.dumps(init_info, ensure_ascii=False)}")
    print(
        f"[train_sparse_multiview] sparse_flow_trainable={trainable_count:,} mode={args.trainable} "
        f"view_aggregator_trainable={view_aggregator_count:,} type={args.view_aggregator} "
        f"freeze_view_aggregator={args.freeze_view_aggregator} "
        f"geometry_adapter_trainable={geometry_adapter_count:,} type={args.geometry_adapter} "
        f"freeze_geometry_adapter={args.freeze_geometry_adapter}"
    )
    from pixal3d_multiview.sparse_condition import SparseMultiviewConditionBuilder

    condition_builder = SparseMultiviewConditionBuilder(device=device, low_vram=False)
    condition_builder.view_aggregator = view_aggregator
    condition_builder.geometry_adapter = geometry_adapter

    optimizer_params = [p for p in denoiser.parameters() if p.requires_grad]
    if view_aggregator is not None:
        optimizer_params.extend([p for p in view_aggregator.parameters() if p.requires_grad])
    if geometry_adapter is not None:
        optimizer_params.extend([p for p in geometry_adapter.parameters() if p.requires_grad])
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp_dtype == "fp16")

    global_step = 0
    start_epoch = 0
    if args.resume:
        global_step, start_epoch = load_resume(args.resume, denoiser, optimizer, scaler, view_aggregator, geometry_adapter)
        print(f"[train_sparse_multiview] resumed step={global_step} epoch={start_epoch}")

    start_time = time.time()
    planned_total_steps = min(
        int(args.max_steps),
        int(global_step + max(0, args.max_epochs - start_epoch) * len(loader)),
    )
    pbar = tqdm(
        total=planned_total_steps,
        initial=min(global_step, planned_total_steps),
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )
    for epoch in range(start_epoch, args.max_epochs):
        for batch in loader:
            if global_step >= args.max_steps:
                break

            x_0 = batch["x_0"].unsqueeze(0).to(device=device, dtype=torch.float32)
            expected = (denoiser.in_channels, denoiser.resolution, denoiser.resolution, denoiser.resolution)
            if tuple(x_0.shape[1:]) != expected:
                raise ValueError(
                    f"sparse latent shape mismatch for {batch['uid']}: got {tuple(x_0.shape[1:])}, expected {expected}. "
                    "Use Pixal3D sparse latent z for the same sparse flow model."
                )

            cond = make_multiview_condition(condition_builder, image_cond_model, batch, args, device)
            if random.random() < args.cfg_drop_prob:
                cond = {
                    "global": torch.zeros_like(cond["global"]),
                    "proj": torch.zeros_like(cond["proj"]),
                }

            noise = torch.randn_like(x_0)
            t = sample_t(x_0.shape[0], device, args.t_schedule)
            x_t = diffuse(x_0, t, noise, args.sigma_min)
            target = velocity_target(x_0, noise, args.sigma_min)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = denoiser(x_t, t * 1000.0, cond)
                loss = F.mse_loss(pred.float(), target.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % args.log_every == 0 or global_step == 1:
                elapsed = max(time.time() - start_time, 1e-6)
                pbar.set_postfix(
                    {
                        "epoch": epoch,
                        "loss": f"{float(loss.detach().cpu().item()):.4g}",
                        "lr": f"{float(optimizer.param_groups[0]['lr']):.2e}",
                        "uid": str(batch["uid"])[:10],
                        "steps/s": f"{global_step / elapsed:.2f}",
                    },
                    refresh=False,
                )
            pbar.update(1)
            if global_step % args.save_every == 0:
                save_checkpoint(
                    output_dir,
                    global_step,
                    epoch,
                    denoiser,
                    optimizer,
                    scaler,
                    args,
                    view_aggregator=view_aggregator,
                    geometry_adapter=geometry_adapter,
                    name=f"step_{global_step}.pt",
                )

        if global_step >= args.max_steps:
            break

    pbar.close()
    save_checkpoint(
        output_dir,
        global_step,
        args.max_epochs,
        denoiser,
        optimizer,
        scaler,
        args,
        view_aggregator=view_aggregator,
        geometry_adapter=geometry_adapter,
        name="final.pt",
    )
    print(f"[train_sparse_multiview] done step={global_step} final={output_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
