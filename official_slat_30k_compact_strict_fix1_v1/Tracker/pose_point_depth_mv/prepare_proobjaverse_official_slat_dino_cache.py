#!/usr/bin/env python3
"""Prepare GT-support, posed-DINO caches from official ProObjaverse pairs."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
from PIL import Image
import torch

from ar_ss_flow.pose_lifting import (
    LIFTING_CACHE_VERSION,
    LIFTING_METADATA_NAMES,
    schema_hash,
)
from ar_ss_flow.shared_object_preprocessing import (
    canonical_json_sha256 as preprocessing_sha256,
    prepare_shared_object_arrays,
    transform_intrinsics,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs import (
    FEATURE_CONTRACT,
    _build_dino_pipeline,
)
from pose_point_depth_mv.dino_only_condition import (
    DINO_ONLY_LIFTING_VERSION,
    build_dino_only_contexts,
    dino_only_feature_metadata,
)
from pose_point_depth_mv.direct_slat_flow import DIRECT_SLAT_CACHE_VERSION
from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    SPLIT_FORMAT,
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from reconvggt_ar_adapter_a.inspect_and_sanity import normalize_image_cond


CACHE_FORMAT = "pose_point_depth_mv.proobjaverse_official_slat_dino_cache.v1"
OBJECT_RECORD_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_slat_dino_cache_object.v1"
)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _load_views_with_audit(
    path: Path, uid: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load every complete RGB/camera pair, including from a trailing-truncated tar.

    A small number of files in the pinned ProObjaverse revision have a valid Hub
    content hash but end during the final member.  ``TarFile.getnames`` rejects
    the whole archive in that case even though the preceding views are intact.
    Only complete JSON/RGBA pairs before the read error are admitted here; the
    recovery is surfaced in the cache report and still has to provide enough
    views for the requested deterministic selection.
    """
    views: list[dict[str, Any]] = []
    with tarfile.open(path, "r") as archive:
        members: dict[str, tarfile.TarInfo] = {}
        read_error = ""
        while True:
            try:
                member = archive.next()
            except tarfile.ReadError as error:
                read_error = str(error)
                break
            if member is None:
                break
            if member.name in members:
                raise RuntimeError(f"duplicate tar member {member.name}: {path}")
            members[member.name] = member
        json_names = sorted(name for name in members if name.endswith(".json"))
        for meta_name in json_names:
            stem = Path(meta_name).stem
            image_name = f"{uid}/{stem}.rgba.webp"
            if image_name not in members:
                continue
            try:
                meta_handle = archive.extractfile(members[meta_name])
                image_handle = archive.extractfile(members[image_name])
                if meta_handle is None or image_handle is None:
                    continue
                meta_bytes = meta_handle.read()
                image_bytes = image_handle.read()
            except tarfile.ReadError:
                # A header may be readable even though its payload is the
                # truncated final member.  Never admit that partial view.
                continue
            if meta_handle is None or image_handle is None:
                raise RuntimeError(f"incomplete view {stem} in {path}")
            meta = json.loads(meta_bytes)
            with Image.open(io.BytesIO(image_bytes)) as handle:
                rgba = np.asarray(handle.convert("RGBA"), dtype=np.uint8)
            # Stable-X/ProObjaverse stores a camera-to-world pose even though
            # the metadata field is named ``extrinsic``.  Every local
            # projection/lifting utility in this repository consumes W2C, so
            # bind and perform the conversion here exactly once.
            camera_to_world = np.asarray(meta["extrinsic"], dtype=np.float32)
            intrinsic = np.asarray(meta["intrinsic"], dtype=np.float32)
            rotation = camera_to_world[:3, :3]
            center = camera_to_world[:3, 3]
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = rotation.T
            extrinsic[:3, 3] = -(rotation.T @ center)
            views.append(
                {
                    "id": int(meta.get("image_index", int(stem))),
                    "rgba": rgba,
                    "intrinsic": intrinsic,
                    "extrinsic": extrinsic,
                    "camera_to_world": camera_to_world,
                    "center": center.astype(np.float64),
                }
            )
    if not views:
        raise ValueError(f"no official views: {path}")
    audit = {
        "archive_complete": not bool(read_error),
        "archive_recovered": bool(read_error),
        "archive_read_error": read_error,
        "complete_view_count": len(views),
        "member_count_before_error": len(members),
    }
    return views, audit


def _load_views(path: Path, uid: str) -> list[dict[str, Any]]:
    views, _ = _load_views_with_audit(path, uid)
    return views


def _pose_diverse_indices(views: list[dict[str, Any]], count: int, uid: str) -> list[int]:
    count = min(int(count), len(views))
    centers = np.stack([row["center"] for row in views])
    radii = np.linalg.norm(centers, axis=1, keepdims=True)
    directions = centers / np.maximum(radii, 1.0e-12)
    seed_hashes = [
        canonical_sha256({"uid": uid, "view": int(row["id"])}) for row in views
    ]
    selected = [min(range(len(views)), key=lambda index: seed_hashes[index])]
    while len(selected) < count:
        candidates = [index for index in range(len(views)) if index not in selected]
        # Maximise the minimum angular separation from already selected views.
        def score(index: int) -> tuple[float, str]:
            cosine = directions[selected] @ directions[index]
            minimum_angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)).min())
            return minimum_angle, "".join(chr(255 - ord(c)) for c in seed_hashes[index])

        selected.append(max(candidates, key=score))
    return selected


def _front_sign(extrinsics: np.ndarray) -> float:
    origin_depth = extrinsics[:, 2, 3]
    positive = int(np.sum(origin_depth > 0.0))
    negative = int(np.sum(origin_depth < 0.0))
    if positive == negative:
        raise RuntimeError("official cameras have ambiguous forward-axis sign")
    sign = 1.0 if positive > negative else -1.0
    if float(np.mean(origin_depth * sign > 0.0)) < 0.95:
        raise RuntimeError("official cameras do not consistently face the origin")
    return sign


def _make_config(
    *, args: argparse.Namespace, split: dict[str, Any], ss_binding: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    from ar_ss_flow.shared_object_preprocessing import shared_preprocessing_contract

    preprocessing = shared_preprocessing_contract(
        resolution=518, foreground_margin=1.10, alpha_threshold=0.80
    )
    slat_config = {
        "pretrained": str(args.pretrained),
        "condition_arch": "native_ss_genrecon_v2",
        "native_ss_deployment": ss_binding,
        "target_source": {
            "kind": "official Stable-X/ProObjaverse-300K lh-slats",
            "protocol_sha256": split["protocol_sha256"],
            "split": split["name"],
            "coordinate_resolution": 64,
            "support_policy": "official_gt_slat_coordinates",
        },
        "training_coordinate_policy": (
            "official GT SLat coordinates; Native SS is bound only as the later "
            "deployment bridge and is not executed in this cache"
        ),
        "official_camera_contract": {
            "source_field": "extrinsic",
            "source_semantics": "camera_to_world",
            "runtime_semantics": "world_to_camera",
            "conversion": "R_w2c=R_c2w.T;t_w2c=-R_c2w.T@C_world",
        },
    }
    lifting_config = {
        "pretrained": str(args.pretrained),
        "image_resolution": 518,
        "geometric_preprocessing": preprocessing,
        "geometric_preprocessing_hash": preprocessing_sha256(preprocessing),
        "no_vggt": {
            "version": DINO_ONLY_LIFTING_VERSION,
            "stock_condition_source": "deterministic_dino_token_context",
            "slat_condition_source": "per_view_raw_dino_token_context",
            "depth_policy": "zero_placeholder_not_consumed",
        },
        "official_slat_protocol_sha256": split["protocol_sha256"],
        "official_slat_split": split["name"],
    }
    return slat_config, lifting_config


def _object_paths(output: Path, uid: str) -> dict[str, Path]:
    root = output / "objects" / uid[:2] / uid
    return {
        "root": root,
        "support": root / "support.pt",
        "physical": root / "physical.pt",
        "condition": root / "condition.pt",
        "lifting": root / "lifting.pt",
        "ss_latent": root / "unused_ss_placeholder.npz",
        "record": root / "cache_record.json",
    }


def _materialized_rows(
    *,
    uid: str,
    target_path: Path,
    paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_hash = sha256_file(target_path)
    slat_row = {
        "uid": uid,
        "object_uid": uid,
        "support_seed": 42,
        "target_file": str(target_path),
        "target_file_sha256": target_hash,
        "support_file": str(paths["support"].resolve()),
        "support_file_sha256": sha256_file(paths["support"]),
        "physical_file": str(paths["physical"].resolve()),
        "physical_file_sha256": sha256_file(paths["physical"]),
        "condition_file": str(paths["condition"].resolve()),
        "condition_file_sha256": sha256_file(paths["condition"]),
        "source_lh_slat": str(target_path),
        "source_lh_slat_sha256": target_hash,
        "source_glb": "",
        "source_glb_sha256": "",
        "ss_latent": str(paths["ss_latent"].resolve()),
        "ss_latent_sha256": sha256_file(paths["ss_latent"]),
    }
    lifting_row = {
        "uid": uid,
        "object_uid": uid,
        "cache_file": str(paths["lifting"].resolve()),
        "cache_file_sha256": sha256_file(paths["lifting"]),
    }
    return slat_row, lifting_row


def _try_reuse_cached_object(
    *,
    output: Path,
    uid: str,
    target_path: Path,
    coords3: np.ndarray,
    selected_views: int,
    slat_config_hash: str,
    lifting_config_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Validate and reconstruct one object from an interrupted cache build."""
    paths = _object_paths(output, uid)
    required = tuple(
        paths[key]
        for key in ("support", "physical", "condition", "lifting", "ss_latent")
    )
    present = [path.is_file() for path in required]
    if not any(present):
        return None
    if not all(present):
        missing = [str(path) for path, exists in zip(required, present) if not exists]
        raise RuntimeError(
            f"incomplete cached object must be inspected, uid={uid} missing={missing}"
        )

    if paths["record"].is_file():
        sidecar = load_json(paths["record"])
        if (
            sidecar.get("format") != OBJECT_RECORD_FORMAT
            or sidecar.get("uid") != uid
            or sidecar.get("slat_config_hash") != slat_config_hash
            or sidecar.get("lifting_config_hash") != lifting_config_hash
        ):
            raise RuntimeError(f"cached object sidecar identity mismatch: {paths['record']}")
        slat_row = dict(sidecar["slat_row"])
        lifting_row = dict(sidecar["lifting_row"])
        record = dict(sidecar["record"])
    else:
        support = torch.load(paths["support"], map_location="cpu", weights_only=False)
        if (
            support.get("format") != DIRECT_SLAT_CACHE_VERSION
            or support.get("uid") != uid
            or support.get("object_uid") != uid
            or support.get("config_hash") != slat_config_hash
        ):
            raise RuntimeError(f"cached support identity mismatch: {paths['support']}")
        cached_coords = support.get("corrected_coords64")
        if not isinstance(cached_coords, torch.Tensor) or not np.array_equal(
            cached_coords.cpu().numpy().astype(np.int32, copy=False), coords3
        ):
            raise RuntimeError(f"cached official coordinates mismatch: {paths['support']}")
        lifting = torch.load(paths["lifting"], map_location="cpu", weights_only=False)
        view_ids = [int(value) for value in lifting.get("view_ids", []).tolist()]
        visual = lifting.get("visual_patch_features")
        if (
            lifting.get("format") != LIFTING_CACHE_VERSION
            or lifting.get("uid") != uid
            or lifting.get("object_uid") != uid
            or lifting.get("official_gt_support_only") is not True
            or len(view_ids) != int(selected_views)
            or not isinstance(visual, torch.Tensor)
        ):
            raise RuntimeError(f"cached lifting identity mismatch: {paths['lifting']}")
        with np.load(paths["ss_latent"], allow_pickle=False) as placeholder:
            if not bool(np.asarray(placeholder["official_gt_support_only"]).item()):
                raise RuntimeError(
                    f"cached placeholder identity mismatch: {paths['ss_latent']}"
                )
        slat_row, lifting_row = _materialized_rows(
            uid=uid, target_path=target_path, paths=paths
        )
        record = {
            "uid": uid,
            "candidate_view_count": None,
            "selected_view_ids": view_ids,
            "camera_forward_sign": float(lifting["camera_forward_sign"]),
            "official_camera_conversion": "camera_to_world_to_world_to_camera",
            "coord_count": len(coords3),
            "visual_shape": list(visual.shape),
            "render_archive_complete": None,
            "render_archive_recovered": None,
            "render_archive_read_error": "not re-read during interrupted-cache recovery",
            "cache_reused_after_interruption": True,
        }
        atomic_json(
            paths["record"],
            {
                "format": OBJECT_RECORD_FORMAT,
                "uid": uid,
                "slat_config_hash": slat_config_hash,
                "lifting_config_hash": lifting_config_hash,
                "slat_row": slat_row,
                "lifting_row": lifting_row,
                "record": record,
            },
        )

    for row, file_keys in (
        (slat_row, ("target_file", "support_file", "physical_file", "condition_file", "ss_latent")),
        (lifting_row, ("cache_file",)),
    ):
        for file_key in file_keys:
            path = Path(row[file_key])
            hash_key = f"{file_key}_sha256"
            if not path.is_file() or row.get(hash_key) != sha256_file(path):
                raise RuntimeError(f"cached file hash mismatch: {path}")
    return slat_row, lifting_row, record


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--native_ss_report", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--dino_model", default="dinov2_vitl14_reg")
    parser.add_argument("--selected_views", type=int, default=8)
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--ss_context_tokens", type=int, default=4096)
    return parser


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    if int(args.selected_views) <= 0 or int(args.ss_context_tokens) <= 0:
        raise ValueError("selected_views/ss_context_tokens must be positive")
    split_path = Path(args.split_manifest).expanduser().resolve()
    split = load_json(split_path)
    if split.get("format") != SPLIT_FORMAT:
        raise ValueError(f"unexpected split format={split.get('format')!r}")
    rows = list(split["rows"])
    if int(args.max_objects) > 0:
        rows = rows[: int(args.max_objects)]
    if not rows:
        raise ValueError("official cache selection is empty")
    _, ss_binding = load_no_vggt_ss_evidence(args.native_ss_report)
    stock_freeze = load_json(args.stock_slat_freeze)
    normalization = {
        key: [float(value) for value in values]
        for key, values in stock_freeze["slat_normalization"].items()
    }
    slat_config, lifting_config = _make_config(
        args=args, split=split, ss_binding=ss_binding
    )
    slat_config_hash = canonical_sha256(slat_config)
    lifting_config_hash = canonical_sha256(lifting_config)
    normalization_hash = canonical_sha256(normalization)
    output = Path(args.output_dir).expanduser().resolve()
    report_path = output / "report.json"
    if output.exists():
        if report_path.is_file():
            report = load_json(report_path)
            expected = {
                "protocol_sha256": split["protocol_sha256"],
                "split": split["name"],
                "object_count": len(rows),
                "selected_views": int(args.selected_views),
            }
            actual = {key: report.get(key) for key in expected}
            if report.get("passed") is True and actual == expected:
                print(json.dumps({"reused": True, **actual}, indent=2))
                return
        if not output.is_dir():
            raise RuntimeError(f"official cache output is not a directory: {output}")
        print(
            f"[official_slat_dino_cache] resume interrupted output: {output}",
            flush=True,
        )
    else:
        output.mkdir(parents=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pipeline = _build_dino_pipeline(args.dino_model, device)
    slat_rows: list[dict[str, Any]] = []
    lifting_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for position, source in enumerate(rows, start=1):
        uid = str(source["uid"])
        target_path = Path(source["slat_npz"]).resolve()
        with np.load(target_path, allow_pickle=False) as target:
            coords3 = np.asarray(target["coords"], dtype=np.int32)
        reused = _try_reuse_cached_object(
            output=output,
            uid=uid,
            target_path=target_path,
            coords3=coords3,
            selected_views=int(args.selected_views),
            slat_config_hash=slat_config_hash,
            lifting_config_hash=lifting_config_hash,
        )
        if reused is not None:
            slat_row, lifting_row, record = reused
            slat_rows.append(slat_row)
            lifting_rows.append(lifting_row)
            records.append(record)
            print(
                f"[official_slat_dino_cache:reuse] {position}/{len(rows)} {uid}",
                flush=True,
            )
            continue

        views, archive_audit = _load_views_with_audit(
            Path(source["render_tar"]), uid
        )
        if len(views) < int(args.selected_views):
            raise RuntimeError(
                f"official render archive has only {len(views)} complete views; "
                f"need {args.selected_views}: {source['render_tar']}"
            )
        if archive_audit["archive_recovered"]:
            print(
                f"[official_slat_dino_cache:recover_tar] uid={uid} "
                f"complete_views={len(views)} error={archive_audit['archive_read_error']!r}",
                flush=True,
            )
        selected_indices = _pose_diverse_indices(
            views, int(args.selected_views), uid
        )
        selected = [views[index] for index in selected_indices]
        rgba = [row["rgba"] for row in selected]
        shared = prepare_shared_object_arrays(
            [value[..., :3] for value in rgba],
            [value[..., 3] for value in rgba],
            resolution=518,
            foreground_margin=1.10,
            alpha_threshold=0.80,
        )
        source_intrinsics = np.stack([row["intrinsic"] for row in selected])
        intrinsics = transform_intrinsics(
            source_intrinsics, shared.source_to_feature_affines
        )
        extrinsics = np.stack([row["extrinsic"] for row in selected])
        forward_sign = _front_sign(extrinsics)
        raw = pipeline.encode_image(shared.images)
        encoded = normalize_image_cond(raw, batch=1, views=len(selected))
        dino_patch = encoded[0, :, int(FEATURE_CONTRACT["patch_start_idx"]) :]
        contexts = build_dino_only_contexts(
            dino_patch, ss_context_tokens=int(args.ss_context_tokens)
        )
        visual = contexts["visual_patch_features"].to(torch.float16).cpu()
        condition = {
            key: [value.to(torch.float16).cpu() for value in values]
            for key, values in contexts["slat_condition"].items()
        }
        paths = _object_paths(output, uid)
        support_path = paths["support"]
        physical_path = paths["physical"]
        condition_path = paths["condition"]
        lifting_path = paths["lifting"]
        ss_latent_path = paths["ss_latent"]
        common = {
            "format": DIRECT_SLAT_CACHE_VERSION,
            "uid": uid,
            "object_uid": uid,
            "config_hash": slat_config_hash,
        }
        atomic_torch_save(
            support_path,
            {
                **common,
                "seed": 42,
                "corrected_ss": torch.zeros((1, 8, 16, 16, 16)),
                "occupancy_logits64": torch.zeros((1, 1, 64, 64, 64)),
                "corrected_coords64": torch.from_numpy(coords3),
                "unused_native_ss_placeholder": True,
            },
        )
        atomic_torch_save(
            physical_path,
            {**common, "unused_native_placeholder": True},
        )
        atomic_torch_save(
            condition_path,
            {**common, "condition": condition},
        )
        atomic_npz(
            ss_latent_path,
            z=np.zeros((8, 16, 16, 16), dtype=np.float32),
            target_coords=np.zeros((1, 3), dtype=np.int32),
            official_gt_support_only=np.asarray(True),
        )
        shared_geometry = shared.contract
        geometry_identity = {
            "shared_geometry_hash": preprocessing_sha256(shared_geometry),
            "view_ids": [int(row["id"]) for row in selected],
            "source_intrinsics": source_intrinsics.astype(np.float32).tolist(),
            "feature_intrinsics": intrinsics.astype(np.float32).tolist(),
        }
        lifting_payload = {
            "format": LIFTING_CACHE_VERSION,
            "uid": uid,
            "object_uid": uid,
            "visual_patch_features": visual,
            "predicted_depth": torch.zeros((len(selected), 518, 518), dtype=torch.float16),
            "depth_confidence": torch.zeros((len(selected), 518, 518), dtype=torch.float16),
            "masks": torch.from_numpy(shared.masks).to(torch.float16),
            "intrinsics": torch.from_numpy(intrinsics),
            "extrinsics": torch.from_numpy(extrinsics),
            "prior_coords": torch.zeros((1, 3), dtype=torch.int32),
            "prior_confidence": torch.zeros((1,), dtype=torch.float32),
            "stock_condition": contexts["stock_condition"].to(torch.float16).cpu(),
            "ss_latent": str(ss_latent_path.resolve()),
            "grid_transform": "identity",
            "extrinsics_type": "w2c",
            "camera_forward_sign": forward_sign,
            "depth_calibration": {"enabled": False},
            "view_ids": torch.tensor(
                [int(row["id"]) for row in selected], dtype=torch.int64
            ),
            "source_intrinsics": torch.from_numpy(source_intrinsics.astype(np.float32)),
            "source_to_feature_affines": torch.from_numpy(
                shared.source_to_feature_affines.astype(np.float32)
            ),
            "preprocessing": {
                "shared_geometry": shared_geometry,
                "shared_geometry_hash": preprocessing_sha256(shared_geometry),
                "sample_geometry_identity_hash": preprocessing_sha256(
                    geometry_identity
                ),
            },
            "official_gt_support_only": True,
        }
        atomic_torch_save(lifting_path, lifting_payload)
        slat_row, lifting_row = _materialized_rows(
            uid=uid, target_path=target_path, paths=paths
        )
        record = {
            "uid": uid,
            "candidate_view_count": len(views),
            "selected_view_ids": [int(row["id"]) for row in selected],
            "camera_forward_sign": forward_sign,
            "official_camera_conversion": "camera_to_world_to_world_to_camera",
            "coord_count": len(coords3),
            "visual_shape": list(visual.shape),
            "render_archive_complete": archive_audit["archive_complete"],
            "render_archive_recovered": archive_audit["archive_recovered"],
            "render_archive_read_error": archive_audit["archive_read_error"],
            "cache_reused_after_interruption": False,
        }
        atomic_json(
            paths["record"],
            {
                "format": OBJECT_RECORD_FORMAT,
                "uid": uid,
                "slat_config_hash": slat_config_hash,
                "lifting_config_hash": lifting_config_hash,
                "slat_row": slat_row,
                "lifting_row": lifting_row,
                "record": record,
            },
        )
        slat_rows.append(slat_row)
        lifting_rows.append(lifting_row)
        records.append(record)
        print(f"[official_slat_dino_cache] {position}/{len(rows)} {uid}", flush=True)

    slat_manifest = {
        "format": DIRECT_SLAT_CACHE_VERSION,
        "materialized": True,
        "output_dir": str(output),
        "config": slat_config,
        "config_hash": slat_config_hash,
        "slat_normalization": normalization,
        "slat_normalization_hash": normalization_hash,
        "samples": slat_rows,
        "sample_count": len(slat_rows),
        "object_count": len(slat_rows),
        "official_gt_support_only": True,
    }
    lifting_manifest = {
        "format": LIFTING_CACHE_VERSION,
        "output_dir": str(output),
        "source_cache_manifest": str(split_path),
        "stock_condition_source": "deterministic_dino_token_context",
        "lifting_feature_source": "official_pose_raw_dino",
        "selection": {
            "mode": "frozen_official_slat_split_prefix",
            "split": split["name"],
            "count": len(rows),
        },
        "samples": lifting_rows,
        "sample_count": len(lifting_rows),
        "object_count": len(lifting_rows),
        "failure_count": 0,
        "feature_metadata": {
            **dino_only_feature_metadata(
                patch_count=int(records[0]["visual_shape"][1])
            ),
            "vggt_model_executed": False,
        },
        "visual_feature_dim": 1024,
        "metadata_names": list(LIFTING_METADATA_NAMES),
        "metadata_schema_hash": schema_hash(),
        "config": lifting_config,
        "config_hash": lifting_config_hash,
        "depth_calibration_enabled_count": 0,
        "split_identity": f"official_slat.{split['protocol_sha256']}.{split['name']}",
    }
    slat_manifest_path = output / "slat_manifest.json"
    lifting_manifest_path = output / "lifting_manifest.json"
    atomic_json(slat_manifest_path, slat_manifest)
    atomic_json(lifting_manifest_path, lifting_manifest)
    report = {
        "format": CACHE_FORMAT,
        "passed": True,
        "formal": False,
        "protocol_sha256": split["protocol_sha256"],
        "split": split["name"],
        "object_count": len(rows),
        "selected_views": int(args.selected_views),
        "slat_manifest": str(slat_manifest_path.resolve()),
        "slat_manifest_sha256": sha256_file(slat_manifest_path),
        "lifting_manifest": str(lifting_manifest_path.resolve()),
        "lifting_manifest_sha256": sha256_file(lifting_manifest_path),
        "native_ss_binding_only": ss_binding,
        "native_ss_executed": False,
        "vggt_executed": False,
        "interrupted_cache_reused_object_count": sum(
            row.get("cache_reused_after_interruption") is True for row in records
        ),
        "truncated_render_archive_recovery_count": sum(
            row.get("render_archive_recovered") is True for row in records
        ),
        "truncated_render_archive_recovery_uids": [
            row["uid"]
            for row in records
            if row.get("render_archive_recovered") is True
        ],
        "records": records,
        "scope_guard": (
            "official GT SLat support diagnosis only; predicted Native-SS support "
            "is intentionally deferred to the disjoint bridge split"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(report_path, report)
    print(json.dumps({key: report[key] for key in ("passed", "split", "object_count", "selected_views", "slat_manifest", "lifting_manifest")}, indent=2))


if __name__ == "__main__":
    main()
