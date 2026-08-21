#!/usr/bin/env python3
"""Build an immutable pose-lifting training cache with DINO only.

Unlike ``ar_ss_flow/build_pose_lifting_cache.py``, this builder never imports,
loads, or executes VGGT.  It consumes the existing PointPose cache so masks,
known cameras, sparse point priors, and SS targets keep the established
training contract, while visual tokens come directly from the frozen TRELLIS
DINO image encoder.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


TRACKER_ROOT = Path(__file__).resolve().parents[2]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
for dependency in (TRACKER_ROOT, RECONVIAGEN_ROOT):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from ar_ss_flow.pose_lifting import (  # noqa: E402
    LIFTING_CACHE_VERSION,
    LIFTING_METADATA_NAMES,
    build_projection_geometry,
    schema_hash,
)
from ar_ss_flow.shared_object_preprocessing import (  # noqa: E402
    canonical_json_sha256,
    prepare_shared_object_views,
    shared_preprocessing_contract,
    transform_intrinsics,
)
from pose_point_depth_mv.dino_only_condition import (  # noqa: E402
    DEFAULT_SS_CONTEXT_TOKENS,
    DINO_ONLY_LIFTING_VERSION,
    build_dino_only_contexts,
    dino_only_feature_metadata,
    tensor_tree_sha256,
    validate_dino_only_lifting_contract,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import (  # noqa: E402
    PHYSICAL_FEATURE_NAMES,
    feature_schema_hash,
)


BUILDER_VERSION = "pose_point_depth_mv.direct_dino_only_lifting_builder.v1"
MARKER_FORMAT = "pose_point_depth_mv.direct_dino_only_lifting_marker.v1"
COMPLETE_MARKER = "_DINO_ONLY_LIFTING_COMPLETE.json"
SAFE_UID = re.compile(r"^[A-Za-z0-9._-]+$")


def parse_indices(spec: str, size: int) -> list[int]:
    text = str(spec).strip().lower()
    if text in {"", "all"}:
        return list(range(size))
    output: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            output.extend(range(int(start), int(end) + 1))
        else:
            output.append(int(item))
    bad = [index for index in output if index < 0 or index >= size]
    if bad:
        raise IndexError(f"indices outside size={size}: {bad}")
    return output


class DirectPointPoseDataset:
    """Read only the PointPose fields needed by the DINO-only builder.

    This intentionally does not import the historical PointPose trainer,
    because that module imports a VGGT pipeline at module initialization.
    """

    def __init__(self, manifest: str | Path, *, indices: str = "all") -> None:
        self.manifest_path = Path(manifest).expanduser().resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != "reconvggt.pointpose_ss_cache.v1":
            raise ValueError(f"unsupported PointPose format={payload.get('format')!r}")
        if tuple(payload.get("feature_names", ())) != PHYSICAL_FEATURE_NAMES:
            raise ValueError("PointPose physical feature names differ")
        if payload.get("feature_schema_hash") != feature_schema_hash():
            raise ValueError("PointPose physical feature schema hash differs")
        all_rows = payload.get("samples")
        if not isinstance(all_rows, list) or not all_rows:
            raise ValueError("PointPose manifest has no samples")
        selected = parse_indices(indices, len(all_rows))
        self.samples = [dict(all_rows[index]) for index in selected]
        self.root = Path(payload.get("output_dir", self.manifest_path.parent)).resolve()
        source_path = Path(str(payload.get("source_manifest", ""))).expanduser()
        prior_path = Path(str(payload.get("prior_manifest", ""))).expanduser()
        if not source_path.is_absolute():
            source_path = (self.manifest_path.parent / source_path).resolve()
        if not prior_path.is_absolute():
            prior_path = (self.manifest_path.parent / prior_path).resolve()
        self.source_path = source_path
        self.prior_path = prior_path
        self.source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        self.prior_payload = json.loads(prior_path.read_text(encoding="utf-8"))
        self.source_by_uid = self._unique_rows(
            self.source_payload.get("samples"), label="source"
        )
        self.prior_by_uid = self._unique_rows(
            self.prior_payload.get("samples"), label="prior"
        )

    @staticmethod
    def _unique_rows(value: Any, *, label: str) -> dict[str, dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"{label} manifest has no samples")
        output: dict[str, dict[str, Any]] = {}
        for row in value:
            uid = str(row.get("uid", ""))
            if not uid or uid in output:
                raise ValueError(f"{label} manifest has empty/duplicate uid={uid!r}")
            output[uid] = row
        return output

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        uid = str(row["uid"])
        source = self.source_by_uid.get(uid)
        prior = self.prior_by_uid.get(uid)
        if source is None or prior is None:
            raise KeyError(f"uid={uid} missing from source/prior manifest")
        physical_path = _resolve_row_file(self.root, row["physical_grid"])
        with np.load(physical_path) as payload:
            prior_coords = np.asarray(payload["prior_coords"], dtype=np.int64)
            prior_conf = np.asarray(payload["prior_conf"], dtype=np.float32)
            view_ids = np.asarray(payload["view_ids"], dtype=np.int64)
        frames = source.get("frames") or []
        if (
            view_ids.ndim != 1
            or not len(view_ids)
            or (view_ids < 0).any()
            or (view_ids >= len(frames)).any()
        ):
            raise ValueError(f"uid={uid} invalid PointPose view IDs")
        selected_frames = [frames[int(view_id)] for view_id in view_ids]
        image_root = _resolve_bound_root(
            str(source.get("image_root", self.source_payload.get("image_root", ""))),
            self.source_path,
        )
        mask_root = _resolve_bound_root(
            str(source.get("mask_root", self.source_payload.get("mask_root", ""))),
            self.source_path,
        )
        latent_root = _resolve_bound_root(
            str(source.get("latent_root", self.source_payload.get("latent_root", ""))),
            self.source_path,
        )
        image_paths = [
            str(_resolve_row_file(image_root, frame["image"]))
            for frame in selected_frames
        ]
        mask_paths = [
            str(_resolve_row_file(mask_root, frame["mask"]))
            for frame in selected_frames
        ]
        ss_latent = _resolve_row_file(latent_root, source["ss_latent"])
        intrinsics = np.asarray(
            [frame["intrinsic"] for frame in selected_frames], dtype=np.float32
        )
        extrinsics = np.asarray(
            [frame["extrinsic"] for frame in selected_frames], dtype=np.float32
        )
        if intrinsics.shape != (len(view_ids), 3, 3) or extrinsics.shape != (
            len(view_ids),
            4,
            4,
        ):
            raise ValueError(f"uid={uid} invalid selected K/T")
        return {
            "uid": uid,
            "object_uid": str(row.get("object_uid", source.get("object_uid", uid))),
            "prior_coords": torch.from_numpy(prior_coords),
            "prior_conf": torch.from_numpy(prior_conf),
            "view_ids": torch.from_numpy(view_ids),
            "image_paths": image_paths,
            "mask_paths": mask_paths,
            "ss_latent": str(ss_latent),
            "intrinsics": torch.from_numpy(intrinsics),
            "extrinsics": torch.from_numpy(extrinsics),
            "grid_transform": str(
                prior.get(
                    "grid_transform",
                    self.prior_payload.get("grid_transform", "pixal3d_rotation"),
                )
            ),
            "extrinsics_type": str(
                self.source_payload.get("extrinsics_type", "c2w")
            ),
            "camera_forward_sign": float(
                self.source_payload.get("camera_forward_sign", 1.0)
            ),
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _resolve_row_file(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _resolve_bound_root(value: str | Path, owner: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else owner.parent / path).resolve()


class FrozenDinoEncoder:
    """Minimal TRELLIS-compatible DINO encoder with no pipeline/VGGT import."""

    default_image_resolution = 518

    def __init__(self, model_name: str, device: torch.device) -> None:
        repository = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if not repository.is_dir():
            raise FileNotFoundError(
                "pinned local DINOv2 torch-hub repository is missing: "
                f"{repository}; direct cache building never falls back to the network"
            )
        self.device = device
        self.model = torch.hub.load(
            str(repository), str(model_name), source="local", pretrained=True
        )
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.tensor(
            (0.485, 0.456, 0.406), device=device, dtype=torch.float32
        )[None, :, None, None]
        self.std = torch.tensor(
            (0.229, 0.224, 0.225), device=device, dtype=torch.float32
        )[None, :, None, None]

    @torch.no_grad()
    def encode_image(self, images: list[Image.Image]) -> torch.Tensor:
        if not images or not all(isinstance(image, Image.Image) for image in images):
            raise ValueError("DINO input must be a non-empty PIL image list")
        tensors = []
        for image in images:
            resized = image.convert("RGB").resize(
                (self.default_image_resolution, self.default_image_resolution),
                Image.Resampling.LANCZOS,
            )
            array = np.asarray(resized, dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(array).permute(2, 0, 1))
        batch = torch.stack(tensors).to(self.device)
        batch = (batch - self.mean) / self.std
        features = self.model(batch, is_training=True)["x_prenorm"]
        return F.layer_norm(features, features.shape[-1:])


def _build_dino_pipeline(dino_model: str, device: torch.device) -> FrozenDinoEncoder:
    return FrozenDinoEncoder(str(dino_model), device)


@torch.no_grad()
def encode_dino_views(pipeline: Any, images: list[Any]) -> torch.Tensor:
    raw = pipeline.encode_image(images)
    if raw.ndim == 3:
        image_cond = raw.unsqueeze(0)
    elif raw.ndim == 4:
        image_cond = raw
    else:
        raise ValueError(f"unexpected DINO condition shape={tuple(raw.shape)}")
    if image_cond.shape[:2] != (1, len(images)) or image_cond.shape[-1] != 1024:
        raise ValueError(
            "DINO condition view/channel layout differs: "
            f"{tuple(image_cond.shape)} vs expected (1,{len(images)},tokens,1024)"
        )
    if int(image_cond.shape[2]) <= 5:
        raise RuntimeError(f"DINO condition has no patch tokens: {tuple(image_cond.shape)}")
    patches = image_cond[0, :, 5:].detach()
    if not bool(torch.isfinite(patches.float()).all().item()):
        raise RuntimeError("DINO patch tensor contains non-finite values")
    return patches


def _sample_input_binding(
    *,
    dataset: DirectPointPoseDataset,
    source_index: int,
    batch: dict[str, Any],
) -> dict[str, Any]:
    row = dataset.samples[source_index]
    files = [
        _resolve_row_file(dataset.root, row["physical_grid"]),
        _resolve_row_file(dataset.root, row["ss_latent"]),
        *[Path(value).expanduser().resolve() for value in batch["image_paths"]],
        *[Path(value).expanduser().resolve() for value in batch["mask_paths"]],
    ]
    records = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"path": str(path), "sha256": sha256_file(path)})
    return {
        "uid": str(batch["uid"]),
        "files": records,
        "binding_sha256": canonical_json_sha256(records),
    }


def build_sample(
    pipeline: Any,
    batch: dict[str, Any],
    *,
    source_manifest: Path,
    source_manifest_sha256: str,
    input_binding: dict[str, Any],
    config_hash: str,
    image_resolution: int,
    foreground_margin: float,
    alpha_threshold: float,
    ss_context_tokens: int,
    save_correct_geometry: bool,
) -> dict[str, Any]:
    uid = str(batch["uid"])
    prepared = prepare_shared_object_views(
        list(batch["image_paths"]),
        list(batch["mask_paths"]),
        resolution=int(image_resolution),
        foreground_margin=float(foreground_margin),
        alpha_threshold=float(alpha_threshold),
    )
    source_intrinsics = batch["intrinsics"].numpy().astype(np.float32)
    feature_intrinsics = transform_intrinsics(
        source_intrinsics,
        prepared.source_to_feature_affines,
    )
    extrinsics = batch["extrinsics"].numpy().astype(np.float32)
    patches = encode_dino_views(pipeline, prepared.images)
    contexts = build_dino_only_contexts(
        patches,
        ss_context_tokens=int(ss_context_tokens),
    )
    visual = contexts["visual_patch_features"].to(torch.float16).cpu()
    stock = contexts["stock_condition"].to(torch.float16).cpu()
    slat_condition = {
        key: [value.to(torch.float16).cpu() for value in values]
        for key, values in contexts["slat_condition"].items()
    }
    condition_hash = tensor_tree_sha256(slat_condition)
    geometry_record = prepared.geometry_record()
    sample_geometry_identity = {
        "shared_geometry_hash": geometry_record["geometry_hash"],
        "view_ids": batch["view_ids"].to(torch.int64).tolist(),
        "source_intrinsics": source_intrinsics.tolist(),
        "feature_intrinsics": feature_intrinsics.tolist(),
    }
    views = int(visual.shape[0])
    if views != len(batch["image_paths"]):
        raise RuntimeError(f"uid={uid} DINO/source view count mismatch")
    payload: dict[str, Any] = {
        "format": LIFTING_CACHE_VERSION,
        "uid": uid,
        "object_uid": str(batch["object_uid"]),
        "visual_patch_features": visual,
        "predicted_depth": torch.zeros(
            (views, int(image_resolution), int(image_resolution)),
            dtype=torch.float16,
        ),
        "depth_confidence": torch.ones(
            (views, int(image_resolution), int(image_resolution)),
            dtype=torch.float16,
        ),
        "masks": torch.from_numpy(prepared.masks).to(torch.float16),
        "intrinsics": torch.from_numpy(feature_intrinsics),
        "source_intrinsics": torch.from_numpy(source_intrinsics),
        "source_to_feature_affines": torch.from_numpy(
            prepared.source_to_feature_affines
        ),
        "extrinsics": torch.from_numpy(extrinsics),
        "view_ids": batch["view_ids"].to(torch.int32),
        "prior_coords": batch["prior_coords"].to(torch.int32),
        "prior_confidence": batch["prior_conf"].to(torch.float16),
        "stock_condition": stock,
        "slat_condition": slat_condition,
        "runtime_condition_sha256": condition_hash,
        "ss_latent": str(batch["ss_latent"]),
        "image_paths": list(batch["image_paths"]),
        "mask_paths": list(batch["mask_paths"]),
        "grid_transform": str(batch["grid_transform"]),
        "extrinsics_type": str(batch["extrinsics_type"]),
        "camera_forward_sign": float(batch["camera_forward_sign"]),
        "source_image_sizes_wh": [list(size) for size in prepared.source_sizes],
        "feature_image_size": [int(image_resolution)] * 2,
        "preprocessing": {
            "shared_geometry": prepared.contract,
            "shared_geometry_hash": geometry_record["geometry_hash"],
            "stock_condition": "deterministic raw DINO token context",
            "lifting_features": "DINO-only patches on known camera geometry",
            "depth_policy": "zero_placeholder_not_consumed",
            "source_to_feature_affines": geometry_record[
                "source_to_feature_affines"
            ],
            "crop_boxes_xyxy": geometry_record["crop_boxes_xyxy"],
            "foreground_retained_fractions": geometry_record[
                "foreground_retained_fractions"
            ],
            "intrinsics_rule": "K_feature=A@K_source",
            "sample_geometry_identity_hash": canonical_json_sha256(
                sample_geometry_identity
            ),
        },
        "depth_calibration": {
            "enabled": False,
            "fallback": "DINO-only cache has no monocular depth channel",
            "reason": "DINO-only frustum projection consumes image shape only",
        },
        "dino_only_context_contract": contexts["context_contract"],
        "dino_only_direct_build": {
            "version": BUILDER_VERSION,
            "lifting_version": DINO_ONLY_LIFTING_VERSION,
            "source_cache_manifest": str(source_manifest),
            "source_cache_manifest_sha256": source_manifest_sha256,
            "sample_input_binding_sha256": input_binding["binding_sha256"],
            "output_config_hash": config_hash,
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
        },
        "slat_condition_provenance": {
            "source": BUILDER_VERSION,
            "condition_tree_sha256": condition_hash,
            "sample_input_binding_sha256": input_binding["binding_sha256"],
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
        },
    }
    if save_correct_geometry:
        patch_count = int(visual.shape[1])
        patch_side = int(round(patch_count**0.5))
        geometry = build_projection_geometry(
            intrinsics=payload["intrinsics"],
            extrinsics=payload["extrinsics"],
            grid_transform=payload["grid_transform"],
            extrinsics_type=payload["extrinsics_type"],
            camera_forward_sign=payload["camera_forward_sign"],
            image_height=int(image_resolution),
            image_width=int(image_resolution),
            patch_grid_side=patch_side,
            volume_side=16,
        )
        payload["correct_geometry"] = {
            "image_grid": geometry["image_grid"].to(torch.float32).cpu(),
            "patch_grid": geometry["patch_grid"].to(torch.float32).cpu(),
            "camera_depth": geometry["camera_depth"].to(torch.float32).cpu(),
            "valid": geometry["valid"].to(torch.bool).cpu(),
        }
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dino_model", default="dinov2_vitl14_reg")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_resolution", type=int, default=518)
    parser.add_argument("--foreground_margin", type=float, default=1.10)
    parser.add_argument("--alpha_threshold", type=float, default=0.80)
    parser.add_argument(
        "--ss_context_tokens", type=int, default=DEFAULT_SS_CONTEXT_TOKENS
    )
    parser.add_argument("--save_correct_geometry", action="store_true")
    parser.add_argument("--allow_failures", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    source_path = Path(args.source_cache_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if int(args.image_resolution) != 518:
        raise ValueError("released DINO patch contract requires image_resolution=518")
    if int(args.ss_context_tokens) <= 0:
        raise ValueError("ss_context_tokens must be positive")
    dataset = DirectPointPoseDataset(source_path, indices=args.indices)
    if int(args.max_samples) > 0:
        dataset.samples = dataset.samples[: int(args.max_samples)]
    if not dataset.samples:
        raise ValueError("selected PointPose cache is empty")
    source_sha = sha256_file(source_path)
    preprocessing = shared_preprocessing_contract(
        resolution=int(args.image_resolution),
        foreground_margin=float(args.foreground_margin),
        alpha_threshold=float(args.alpha_threshold),
    )
    no_vggt = {
        "version": DINO_ONLY_LIFTING_VERSION,
        "stock_condition_source": "deterministic_dino_token_context",
        "slat_condition_source": "per_view_raw_dino_token_context",
        "ss_context_token_cap": int(args.ss_context_tokens),
        "depth_policy": "zero_placeholder_not_consumed",
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
    }
    config = {
        "builder_version": BUILDER_VERSION,
        "dino_model": str(args.dino_model),
        "image_resolution": int(args.image_resolution),
        "geometric_preprocessing": preprocessing,
        "geometric_preprocessing_hash": canonical_json_sha256(preprocessing),
        "save_correct_geometry": bool(args.save_correct_geometry),
        "no_vggt": no_vggt,
    }
    config_hash = canonical_json_sha256(config)
    selected_uids = [str(row["uid"]) for row in dataset.samples]
    if len(selected_uids) != len(set(selected_uids)):
        raise ValueError("selected PointPose rows contain duplicate UIDs")
    run_binding = {
        "format": BUILDER_VERSION,
        "source_cache_manifest": str(source_path),
        "source_cache_manifest_sha256": source_sha,
        "selected_uids": selected_uids,
        "config": config,
        "config_hash": config_hash,
    }
    binding_path = output_dir / "run_config.json"
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if binding_path.is_file():
        existing = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing != run_binding:
            raise RuntimeError("resume source/config binding changed")
    else:
        atomic_json(binding_path, run_binding)

    complete_path = output_dir / COMPLETE_MARKER
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        manifest_path = output_dir / "lifting_manifest.json"
        same_source_config = (
            complete.get("format") == MARKER_FORMAT
            and complete.get("source_cache_manifest_sha256") == source_sha
            and complete.get("config_hash") == config_hash
        )
        if complete.get("passed") is True:
            if (
                not same_source_config
                or int(complete.get("sample_count", -1)) != len(selected_uids)
                or not manifest_path.is_file()
                or complete.get("manifest_sha256") != sha256_file(manifest_path)
            ):
                raise RuntimeError(f"completed DINO cache binding changed: {output_dir}")
            print(
                json.dumps(
                    {
                        "reused": True,
                        "manifest": str(manifest_path),
                        "sample_count": len(selected_uids),
                        "config_hash": config_hash,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return
        if not same_source_config:
            raise RuntimeError(f"failed DINO cache binding changed: {output_dir}")

    device = torch.device(args.device)
    pipeline = _build_dino_pipeline(str(args.dino_model), device)
    if int(getattr(pipeline, "default_image_resolution", 0)) != int(
        args.image_resolution
    ):
        raise RuntimeError("DINO pipeline/shared preprocessing resolution differs")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    patch_counts: set[int] = set()
    audits: list[dict[str, Any]] = []
    try:
        for source_index in range(len(dataset)):
            batch = dataset[source_index]
            uid = str(batch["uid"])
            if not SAFE_UID.fullmatch(uid):
                raise ValueError(f"unsafe cache UID={uid!r}")
            relative = Path("samples") / uid[:2] / f"{uid}.pt"
            destination = output_dir / relative
            try:
                input_binding = _sample_input_binding(
                    dataset=dataset,
                    source_index=source_index,
                    batch=batch,
                )
                if destination.is_file():
                    payload = torch.load(destination, map_location="cpu")
                    direct = dict(payload.get("dino_only_direct_build", {}))
                    if (
                        payload.get("format") != LIFTING_CACHE_VERSION
                        or str(payload.get("uid", "")) != uid
                        or direct.get("output_config_hash") != config_hash
                        or direct.get("sample_input_binding_sha256")
                        != input_binding["binding_sha256"]
                    ):
                        raise RuntimeError(f"stale direct DINO cache: {destination}")
                else:
                    payload = build_sample(
                        pipeline,
                        batch,
                        source_manifest=source_path,
                        source_manifest_sha256=source_sha,
                        input_binding=input_binding,
                        config_hash=config_hash,
                        image_resolution=int(args.image_resolution),
                        foreground_margin=float(args.foreground_margin),
                        alpha_threshold=float(args.alpha_threshold),
                        ss_context_tokens=int(args.ss_context_tokens),
                        save_correct_geometry=bool(args.save_correct_geometry),
                    )
                    atomic_torch_save(destination, payload)
                condition_hash = tensor_tree_sha256(payload["slat_condition"])
                if condition_hash != payload["slat_condition_provenance"].get(
                    "condition_tree_sha256"
                ):
                    raise RuntimeError(f"uid={uid} SLat condition hash mismatch")
                patch_counts.add(int(payload["visual_patch_features"].shape[1]))
                rows.append(
                    {
                        "uid": uid,
                        "object_uid": str(batch["object_uid"]),
                        "cache_file": str(relative),
                        "cache_file_sha256": sha256_file(destination),
                        "ss_latent": str(batch["ss_latent"]),
                        "view_count": int(payload["visual_patch_features"].shape[0]),
                        "prior_point_count": int(len(batch["prior_coords"])),
                        "depth_calibration_enabled": False,
                        "depth_match_count": 0,
                        "source_input_binding_sha256": input_binding[
                            "binding_sha256"
                        ],
                    }
                )
                audits.append(
                    {
                        "uid": uid,
                        "visual_shape": list(payload["visual_patch_features"].shape),
                        "stock_condition_shape": list(
                            payload["stock_condition"].shape
                        ),
                        "mask_nonzero_ratio": float(
                            (payload["masks"] > 0.5).float().mean().item()
                        ),
                        "shared_geometry_hash": payload["preprocessing"][
                            "shared_geometry_hash"
                        ],
                    }
                )
                if (source_index + 1) % max(1, int(args.log_every)) == 0 or (
                    source_index + 1 == len(dataset)
                ):
                    print(
                        f"[direct_dino_only] {source_index + 1}/{len(dataset)} "
                        f"uid={uid} views={rows[-1]['view_count']}",
                        flush=True,
                    )
            except Exception as error:
                failures.append({"uid": uid, "error": repr(error)})
                print(f"[direct_dino_only] FAILED uid={uid}: {error!r}", flush=True)
                if not args.allow_failures:
                    raise
    finally:
        del pipeline
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows or len(patch_counts) != 1:
        raise RuntimeError(
            f"direct DINO cache invalid rows={len(rows)} patch_counts={sorted(patch_counts)}"
        )
    patch_count = next(iter(patch_counts))
    passed = not failures
    manifest = {
        "format": LIFTING_CACHE_VERSION,
        "created_at_utc": utc_now(),
        "output_dir": str(output_dir),
        "source_cache_manifest": str(source_path),
        "source_cache_manifest_sha256": source_sha,
        "stock_condition_source": "deterministic DINO-only token context",
        "lifting_feature_source": "direct frozen DINO patches; no VGGT",
        "sample_count": len(rows),
        "object_count": len(
            {str(row.get("object_uid", row["uid"])) for row in rows}
        ),
        "failure_count": len(failures),
        "feature_metadata": dino_only_feature_metadata(patch_count=patch_count),
        "visual_feature_dim": 1024,
        "metadata_names": list(LIFTING_METADATA_NAMES),
        "metadata_schema_hash": schema_hash(),
        "config": config,
        "config_hash": config_hash,
        "samples": rows,
        "passed": passed,
        "training_ready": passed,
    }
    proxy = SimpleNamespace(
        visual_feature_dim=manifest["visual_feature_dim"],
        feature_metadata=manifest["feature_metadata"],
        config=manifest["config"],
        config_hash=manifest["config_hash"],
    )
    manifest["no_vggt_contract"] = validate_dino_only_lifting_contract(proxy)
    manifest_path = output_dir / "lifting_manifest.json"
    atomic_json(manifest_path, manifest)
    audit = {
        "format": "pose_point_depth_mv.direct_dino_only_lifting_audit.v1",
        "passed": passed,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_cache_manifest_sha256": source_sha,
        "config_hash": config_hash,
        "sample_count": len(rows),
        "failure_count": len(failures),
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "samples": audits,
        "failures": failures,
    }
    atomic_json(output_dir / "cache_audit.json", audit)
    atomic_json(output_dir / "failed_samples.json", failures)
    marker = {
        "format": MARKER_FORMAT,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_cache_manifest_sha256": source_sha,
        "config_hash": config_hash,
        "sample_count": len(rows),
        "failure_count": len(failures),
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "passed": passed,
    }
    atomic_json(output_dir / COMPLETE_MARKER, marker)
    print(
        json.dumps(
            {
                "passed": passed,
                "manifest": str(manifest_path),
                "sample_count": len(rows),
                "object_count": manifest["object_count"],
                "failure_count": len(failures),
                "config_hash": config_hash,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
