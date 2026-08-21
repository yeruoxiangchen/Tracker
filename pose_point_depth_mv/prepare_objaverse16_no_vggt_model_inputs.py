#!/usr/bin/env python3
"""Package frozen Objaverse DINO/pose tensors without targets or point tensors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from ar_ss_flow.pose_lifting import LIFTING_CACHE_VERSION
from pose_point_depth_mv.dino_only_condition import (
    tensor_tree_sha256,
    validate_dino_only_lifting_contract,
)
from pose_point_depth_mv.training_overlap_objaverse import validate_selection


OBJECT_FORMAT = "pose_point_depth_mv.objaverse16_dino_pose_model_input.v1"
MANIFEST_FORMAT = "pose_point_depth_mv.objaverse16_dino_pose_model_input_manifest.v1"
FORBIDDEN_MODEL_FIELDS = {
    "ss_latent",
    "target",
    "target_coords",
    "source_glb",
    "prior_coords",
    "prior_confidence",
    "points_o",
    "predicted_depth_source",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resolve_cache_file(row: dict[str, Any], root: Path) -> Path:
    path = Path(str(row["cache_file"]))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def object_key(object_uid: str) -> str:
    return f"objaverse:{object_uid}"


def validate_source_contract(lifting: dict[str, Any]) -> dict[str, Any]:
    if lifting.get("format") != LIFTING_CACHE_VERSION or lifting.get("passed") is not True:
        raise RuntimeError("DINO-only lifting manifest did not pass")
    proxy = SimpleNamespace(
        visual_feature_dim=int(lifting["visual_feature_dim"]),
        feature_metadata=dict(lifting["feature_metadata"]),
        config=dict(lifting["config"]),
        config_hash=str(lifting["config_hash"]),
    )
    return validate_dino_only_lifting_contract(proxy)


def build_payload(
    source: dict[str, Any], *, uid: str, object_uid: str, source_group: str
) -> dict[str, Any]:
    if source.get("format") != LIFTING_CACHE_VERSION:
        raise ValueError(f"uid={uid} source lifting sample format differs")
    if str(source.get("uid")) != uid or str(source.get("object_uid")) != object_uid:
        raise RuntimeError(f"uid={uid} lifting sample identity differs")
    required_tensors = (
        "visual_patch_features",
        "stock_condition",
        "intrinsics",
        "extrinsics",
    )
    if not all(torch.is_tensor(source.get(name)) for name in required_tensors):
        raise ValueError(f"uid={uid} source lacks required DINO/pose tensors")
    visual = source["visual_patch_features"]
    views = int(visual.shape[0])
    if visual.ndim != 3 or tuple(visual.shape[1:]) != (1369, 1024):
        raise ValueError(f"uid={uid} invalid DINO tensor shape={tuple(visual.shape)}")
    if source["intrinsics"].shape != (views, 3, 3) or source["extrinsics"].shape != (
        views,
        4,
        4,
    ):
        raise ValueError(f"uid={uid} camera tensor shapes differ from DINO views")
    slat = source.get("slat_condition")
    if not isinstance(slat, dict) or sorted(slat) != ["cond", "neg_cond"]:
        raise ValueError(f"uid={uid} invalid SLat condition")
    if len(slat["cond"]) != views or len(slat["neg_cond"]) != views:
        raise ValueError(f"uid={uid} SLat condition/view count differs")
    tensors = [source[name] for name in required_tensors]
    tensors.extend(slat["cond"])
    tensors.extend(slat["neg_cond"])
    if not all(bool(value.isfinite().all().item()) for value in tensors):
        raise ValueError(f"uid={uid} model input contains non-finite tensors")

    image_size = source.get("feature_image_size", (518, 518))
    height, width = map(int, image_size)
    payload = {
        "format": OBJECT_FORMAT,
        "uid": uid,
        "object_uid": object_uid,
        "object_key": object_key(object_uid),
        "dataset_source": "objaverse",
        "source_group": source_group,
        "visual_patch_features": visual.detach().cpu(),
        "stock_condition": source["stock_condition"].detach().cpu(),
        "slat_condition": {
            key: [value.detach().cpu() for value in values]
            for key, values in slat.items()
        },
        "intrinsics": source["intrinsics"].detach().cpu(),
        "extrinsics": source["extrinsics"].detach().cpu(),
        "grid_transform": str(source["grid_transform"]),
        "extrinsics_type": str(source["extrinsics_type"]),
        "camera_forward_sign": float(source["camera_forward_sign"]),
        "projection_image_size": (height, width),
        "dino_only_context_contract": dict(source["dino_only_context_contract"]),
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "point_cloud_tensor_present": False,
        "point_cloud_consumed": False,
        "target_or_mesh_consumed": False,
        "scope_guard": (
            "inference payload contains DINO and posed cameras only; target, source "
            "Mesh, SS latent, point coordinates, VGGT features and VGGT depth are absent"
        ),
    }
    payload["condition_sha256"] = tensor_tree_sha256(
        {
            "visual": payload["visual_patch_features"],
            "stock": payload["stock_condition"],
            "slat": payload["slat_condition"],
            "intrinsics": payload["intrinsics"],
            "extrinsics": payload["extrinsics"],
        }
    )
    leaked = sorted(FORBIDDEN_MODEL_FIELDS & set(payload))
    if leaked:
        raise RuntimeError(f"uid={uid} forbidden model fields leaked: {leaked}")
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--lifting_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    selection_path = Path(args.selection_manifest).expanduser().resolve()
    lifting_path = Path(args.lifting_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    selection = load_json(selection_path)
    selection_contract = validate_selection(selection)
    lifting = load_json(lifting_path)
    no_vggt_contract = validate_source_contract(lifting)
    lifting_root = Path(lifting.get("output_dir", lifting_path.parent)).resolve()
    lifting_rows = {str(row["uid"]): row for row in lifting["samples"]}
    selected = list(selection["samples"])
    if len(lifting_rows) != len(lifting["samples"]):
        raise RuntimeError("lifting manifest contains duplicate uids")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for position, selected_row in enumerate(selected, start=1):
        uid = str(selected_row["uid"])
        object_uid = str(selected_row["object_uid"])
        lifting_row = lifting_rows.get(uid)
        if lifting_row is None:
            raise KeyError(f"selected uid is absent from lifting cache: {uid}")
        if str(lifting_row.get("object_uid")) != object_uid:
            raise RuntimeError(f"uid={uid} lifting object identity differs")
        source_file = resolve_cache_file(lifting_row, lifting_root)
        source_sha = sha256_file(source_file)
        if source_sha != str(lifting_row.get("cache_file_sha256")):
            raise RuntimeError(f"uid={uid} lifting sample SHA differs")
        destination = output_dir / "objects" / object_uid / "model_input.pt"
        if destination.is_file():
            if not args.resume:
                raise FileExistsError(destination)
            payload = torch.load(destination, map_location="cpu")
            if (
                payload.get("format") != OBJECT_FORMAT
                or payload.get("uid") != uid
                or payload.get("object_uid") != object_uid
                or payload.get("source_lifting_sample_sha256") != source_sha
            ):
                raise RuntimeError(f"stale Objaverse model input: {destination}")
        else:
            source = torch.load(source_file, map_location="cpu")
            payload = build_payload(
                source,
                uid=uid,
                object_uid=object_uid,
                source_group=str(selected_row["source_group"]),
            )
            payload["source_lifting_sample_sha256"] = source_sha
            atomic_torch_save(destination, payload)
        model_sha = sha256_file(destination)
        report = {
            "format": OBJECT_FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "category": "objaverse",
            "object_id": object_uid,
            "object_key": object_key(object_uid),
            "uid": uid,
            "object_uid": object_uid,
            "source_group": str(selected_row["source_group"]),
            "condition_sha256": str(payload["condition_sha256"]),
            "model_input": str(destination),
            "model_input_sha256": model_sha,
            "source_lifting_sample": str(source_file),
            "source_lifting_sample_sha256": source_sha,
            "view_count": int(payload["visual_patch_features"].shape[0]),
            "vggt_model_loaded": False,
            "vggt_model_executed": False,
            "point_cloud_tensor_present": False,
            "point_cloud_consumed": False,
            "target_or_mesh_consumed": False,
            "passed": True,
        }
        atomic_json(destination.with_name("report.json"), report)
        reports.append(report)
        print(
            f"[objaverse16_model_input] {position}/{len(selected)} uid={uid} "
            f"views={report['view_count']}",
            flush=True,
        )

    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_scope": selection_contract.scope,
        "formal": False,
        "training_overlap": selection_contract.training_overlap,
        "training_object_disjoint": selection_contract.training_object_disjoint,
        "source_mesh_disjoint": selection_contract.source_mesh_disjoint,
        "source_scope": selection_contract.source_scope,
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        # infer_omni_real_native_v2 consumes these generic binding fields.
        "runtime_input_manifest": str(selection_path),
        "runtime_input_manifest_sha256": sha256_file(selection_path),
        "lifting_manifest": str(lifting_path),
        "lifting_manifest_sha256": sha256_file(lifting_path),
        "lifting_config_hash": str(lifting["config_hash"]),
        "no_vggt_contract": no_vggt_contract,
        "selected_object_count": len(selected),
        "completed_object_count": len(reports),
        "objects": reports,
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "point_cloud_tensor_present": False,
        "point_cloud_consumed": False,
        "target_or_mesh_consumed": False,
        "passed": len(reports) == selection_contract.object_count,
    }
    manifest_path = output_dir / "model_input_manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "objects": len(reports),
                "point_cloud_consumed": False,
                "target_or_mesh_consumed": False,
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
