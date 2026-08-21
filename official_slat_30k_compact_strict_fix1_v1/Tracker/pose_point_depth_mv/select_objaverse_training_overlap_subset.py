#!/usr/bin/env python3
"""Select a deterministic Objaverse-only subset from a model training cache."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ar_ss_flow.pose_lifting import LIFTING_CACHE_VERSION
from pose_point_depth_mv.direct_slat_flow import DIRECT_SLAT_CACHE_VERSION
from pose_point_depth_mv.freeze_objaverse16_test import (
    canonical_json_sha256,
    sha256_file,
)
from pose_point_depth_mv.training_overlap_objaverse import (
    SOURCE_SCOPES,
    TRAINING_OVERLAP_PROTOCOL_FORMAT,
    TRAINING_OVERLAP_SCOPE,
)


SELECTION_FORMAT = "pose_point_depth_mv.objaverse_training_overlap_selection.v1"
LIFTING_SUBSET_FORMAT = "pose_point_depth_mv.objaverse_training_overlap_lifting_subset.v1"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def resolve_from(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def stable_rank(seed: int, namespace: str, *values: str) -> str:
    encoded = "\0".join((str(int(seed)), namespace, *map(str, values))).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_objaverse_path(path: Path) -> bool:
    lowered = str(path).lower()
    parts = {part.lower() for part in path.parts}
    return (
        "objaverse" in lowered
        and ".objaverse" in parts
        and "hf-objaverse-v1" in parts
        and "glbs" in parts
        and "omni" not in lowered
        and path.suffix.lower() == ".glb"
    )


def _require_objaverse_path(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not _is_objaverse_path(path):
        raise RuntimeError(f"{label} is not a canonical Objaverse GLB: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _latent_source_glb(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        if "source_glb" not in payload.files:
            raise RuntimeError(f"{label} latent has no source_glb: {path}")
        return _require_objaverse_path(str(payload["source_glb"]), label=label)


def _validate_mixed_domain_binding(
    *, meta_path: Path, selected_path: Path, expected_format: str
) -> dict[str, Any]:
    meta = load_json(meta_path)
    if meta.get("format") != expected_format or meta.get("passed") is not True:
        raise RuntimeError(f"mixed parent manifest did not pass: {meta_path}")
    domains = list(meta.get("domains", []))
    synthetic = [row for row in domains if str(row.get("name")) == "synthetic"]
    real = [row for row in domains if str(row.get("name")) == "real"]
    if len(synthetic) != 1 or len(real) != 1:
        raise RuntimeError("mixed parent must bind exactly synthetic and real domains")
    binding = synthetic[0]
    bound_path = Path(str(binding.get("manifest", ""))).expanduser().resolve()
    if bound_path != selected_path or sha256_file(bound_path) != binding.get("manifest_sha256"):
        raise RuntimeError("requested source is not the hash-bound Mixed synthetic domain")
    real_path = Path(str(real[0].get("manifest", ""))).expanduser().resolve()
    if selected_path == real_path:
        raise RuntimeError("Mixed real/Omni domain cannot be selected")
    return {
        "meta_manifest": str(meta_path),
        "meta_manifest_sha256": sha256_file(meta_path),
        "selected_domain": "synthetic",
        "rejected_domain": "real",
        "selected_manifest": str(selected_path),
        "selected_manifest_sha256": sha256_file(selected_path),
        "real_manifest": str(real_path),
        "real_manifest_sha256": str(real[0].get("manifest_sha256", "")),
        "passed": True,
    }


def _load_visible_inputs(
    lifting_path: Path, lifting: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    root = Path(str(lifting.get("output_dir", lifting_path.parent))).resolve()
    cache_path = resolve_from(root, str(row["cache_file"]))
    if sha256_file(cache_path) != str(row.get("cache_file_sha256", "")):
        raise RuntimeError(f"lifting sample hash changed: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    uid = str(row["uid"])
    object_uid = str(row["object_uid"])
    if cache.get("uid") != uid or cache.get("object_uid") != object_uid:
        raise RuntimeError(f"lifting cache identity differs: {uid}")
    view_ids_value = cache.get("view_ids", [])
    if torch.is_tensor(view_ids_value):
        view_ids_value = view_ids_value.detach().cpu().tolist()
    view_ids = [int(value) for value in view_ids_value]
    images = [Path(str(value)).resolve() for value in cache.get("image_paths", [])]
    masks = [Path(str(value)).resolve() for value in cache.get("mask_paths", [])]
    if (
        not view_ids
        or len(view_ids) != len(set(view_ids))
        or len(images) != len(view_ids)
        or len(masks) != len(view_ids)
        or int(row.get("view_count", -1)) != len(view_ids)
    ):
        raise RuntimeError(f"lifting visible-view contract differs: {uid}")
    for path in [*images, *masks]:
        lowered = str(path).lower()
        if not path.is_file() or "objaverse" not in lowered or "omni" in lowered:
            raise RuntimeError(f"non-Objaverse visible input rejected: {path}")
    return {
        "view_count": len(view_ids),
        "view_ids": view_ids,
        "source_view_indices": list(view_ids),
        "image_paths": [str(path) for path in images],
        "mask_paths": [str(path) for path in masks],
        "source_cache": str(cache_path),
        "source_cache_sha256": sha256_file(cache_path),
    }


def build_selection(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    lifting_path = Path(args.lifting_manifest).expanduser().resolve()
    slat_path = Path(args.slat_manifest).expanduser().resolve()
    lifting = load_json(lifting_path)
    slat = load_json(slat_path)
    if (
        lifting.get("format") != LIFTING_CACHE_VERSION
        or lifting.get("passed") is not True
        or lifting.get("training_ready") is not True
    ):
        raise RuntimeError("source lifting manifest is not training-ready")
    if (
        slat.get("format") != DIRECT_SLAT_CACHE_VERSION
        or slat.get("materialized") is not True
        or not slat.get("samples")
        or not slat.get("objects")
    ):
        raise RuntimeError("source SLat cache manifest is not materialized")
    bound_lifting = resolve_from(slat_path.parent, str(slat["source_lifting_manifest"]))
    if (
        bound_lifting != lifting_path
        or sha256_file(lifting_path) != slat.get("source_lifting_manifest_sha256")
    ):
        raise RuntimeError("SLat and lifting training manifests are not hash-bound")

    source_scope = str(args.source_scope)
    source_audit: dict[str, Any]
    if source_scope == "mixed_objaverse_train":
        if not args.mixed_lifting_meta_manifest or not args.mixed_slat_meta_manifest:
            raise ValueError("Mixed scope requires both parent meta manifests")
        lifting_meta_path = Path(args.mixed_lifting_meta_manifest).expanduser().resolve()
        slat_meta_path = Path(args.mixed_slat_meta_manifest).expanduser().resolve()
        source_audit = {
            "lifting": _validate_mixed_domain_binding(
                meta_path=lifting_meta_path,
                selected_path=lifting_path,
                expected_format="pose_point_depth_mv.mixed_no_vggt_lifting_manifest.v1",
            ),
            "slat": _validate_mixed_domain_binding(
                meta_path=slat_meta_path,
                selected_path=slat_path,
                expected_format="pose_point_depth_mv.mixed_no_vggt_slat_manifest.v1",
            ),
        }
    else:
        split = dict(lifting.get("objaverse2k_split", {}))
        if split.get("name") != "train" or int(lifting.get("object_count", -1)) != 2135:
            raise RuntimeError("Objaverse2K source is not the frozen train2135 split")
        source_audit = {
            "split_name": "train",
            "object_count": 2135,
            "passed": True,
        }

    lifting_rows = {str(row["uid"]): row for row in lifting.get("samples", [])}
    if len(lifting_rows) != len(lifting.get("samples", [])):
        raise RuntimeError("source lifting manifest has duplicate UIDs")
    slat_samples = {str(row["uid"]): row for row in slat["samples"]}
    slat_objects = {str(row["object_uid"]): row for row in slat["objects"]}
    if len(slat_samples) != len(slat["samples"]) or len(slat_objects) != len(slat["objects"]):
        raise RuntimeError("source SLat manifest has duplicate identities")

    eligible_objects: set[str] = set()
    rejected_objects: set[str] = set()
    for object_uid, row in slat_objects.items():
        path = Path(str(row.get("source_glb", ""))).expanduser().resolve()
        if _is_objaverse_path(path):
            eligible_objects.add(object_uid)
        else:
            rejected_objects.add(object_uid)
    if source_scope == "objaverse2k_train" and rejected_objects:
        raise RuntimeError(
            f"Objaverse2K train cache contains {len(rejected_objects)} non-Objaverse objects"
        )

    by_object: dict[str, list[str]] = defaultdict(list)
    rejected_sequence_count = 0
    for uid in sorted(set(lifting_rows).intersection(slat_samples)):
        left = lifting_rows[uid]
        right = slat_samples[uid]
        object_uid = str(left.get("object_uid", ""))
        if object_uid in rejected_objects:
            rejected_sequence_count += 1
        elif object_uid in eligible_objects and str(right.get("object_uid", "")) == object_uid:
            by_object[object_uid].append(uid)
    source_filter_audit = {
        "source_object_count": len(slat_objects),
        "eligible_hf_objaverse_glb_object_count": len(eligible_objects),
        "rejected_non_objaverse_object_count": len(rejected_objects),
        "rejected_non_objaverse_sequence_count": rejected_sequence_count,
        "selection_candidate_object_count": len(by_object),
        "selected_non_objaverse_object_count": 0,
        "policy": (
            "accept only canonical .objaverse/hf-objaverse-v1/glbs/*.glb paths; "
            "reject OmniObject3D and every other source"
        ),
        "passed": True,
    }
    source_audit["object_source_filter"] = source_filter_audit
    ranked_objects = sorted(
        by_object,
        key=lambda value: stable_rank(int(args.seed), "object", source_scope, value),
    )
    if len(ranked_objects) < int(args.count):
        raise RuntimeError(f"only {len(ranked_objects)} jointly bound objects are available")

    selected: list[dict[str, Any]] = []
    selected_lifting: list[dict[str, Any]] = []
    for position, object_uid in enumerate(ranked_objects[: int(args.count)]):
        uids = sorted(
            by_object[object_uid],
            key=lambda value: stable_rank(int(args.seed), "sequence", object_uid, value),
        )
        uid = uids[0]
        lifting_row = dict(lifting_rows[uid])
        lifting_root = Path(str(lifting.get("output_dir", lifting_path.parent))).resolve()
        lifting_row["cache_file"] = str(
            resolve_from(lifting_root, str(lifting_row["cache_file"]))
        )
        sample_row = dict(slat_samples[uid])
        object_row = dict(slat_objects[object_uid])
        source_glb = _require_objaverse_path(
            object_row.get("source_glb", sample_row.get("source_glb", "")),
            label=f"uid={uid} source_glb",
        )
        sample_glb = _require_objaverse_path(
            sample_row.get("source_glb", source_glb), label=f"uid={uid} sample source_glb"
        )
        if sample_glb != source_glb:
            raise RuntimeError(f"uid={uid} SLat sample/object GLBs differ")
        expected_glb_sha = str(object_row.get("source_glb_sha256", ""))
        if sha256_file(source_glb) != expected_glb_sha:
            raise RuntimeError(f"uid={uid} Objaverse GLB hash changed")
        lifting_latent_glb = _latent_source_glb(
            lifting_row["ss_latent"], label=f"uid={uid} lifting"
        )
        slat_latent_glb = _latent_source_glb(
            sample_row["ss_latent"], label=f"uid={uid} SLat"
        )
        if lifting_latent_glb != source_glb or slat_latent_glb != source_glb:
            raise RuntimeError(f"uid={uid} latent/source object identities differ")
        visible = _load_visible_inputs(lifting_path, lifting, lifting_row)
        selected.append(
            {
                **sample_row,
                "uid": uid,
                "object_uid": object_uid,
                "dataset_source": "objaverse",
                "source_group": source_scope,
                "source_glb": str(source_glb),
                "source_glb_sha256": expected_glb_sha,
                "visible_inputs": visible,
                "training_overlap_selection": {
                    "position": position,
                    "selection_seed": int(args.seed),
                    "expected_view_count": int(visible["view_count"]),
                    "sequence_candidate_count": len(uids),
                },
            }
        )
        selected_lifting.append(lifting_row)

    now = datetime.now(timezone.utc).isoformat()
    protocol = {
        "format": TRAINING_OVERLAP_PROTOCOL_FORMAT,
        "scope": TRAINING_OVERLAP_SCOPE,
        "source_scope": source_scope,
        "formal": False,
        "training_overlap": True,
        "training_object_disjoint": False,
        "source_mesh_disjoint": False,
        "selection_seed": int(args.seed),
        "object_count": len(selected),
        "sequence_count": len(selected),
        "selected_uids": [str(row["uid"]) for row in selected],
        "selected_object_uids": [str(row["object_uid"]) for row in selected],
        "source_lifting_manifest": str(lifting_path),
        "source_lifting_manifest_sha256": sha256_file(lifting_path),
        "source_slat_manifest": str(slat_path),
        "source_slat_manifest_sha256": sha256_file(slat_path),
        "source_audit": source_audit,
        "objaverse_only_audit": {
            "object_count": len(selected),
            **source_filter_audit,
            "all_source_glbs_under_hf_objaverse": True,
            "all_visible_inputs_contain_objaverse": True,
            "omni_or_real_rows_selected": 0,
            "passed": True,
        },
        "selection_rule": (
            "SHA256(seed, source_scope, object_uid), then one jointly bound sequence "
            "by SHA256(seed, object_uid, uid); model outputs were not read"
        ),
        "limitations": (
            "training-set overlap diagnostic only; cannot be reported as generalization"
        ),
        "passed": True,
    }
    protocol["protocol_sha256"] = canonical_json_sha256(protocol)
    selection = {
        "format": SELECTION_FORMAT,
        "created_at_utc": now,
        "status": "training_overlap_diagnostic",
        "formal": False,
        "training_overlap": True,
        "training_ready": False,
        "samples": selected,
        "training_overlap_protocol": protocol,
        "passed": True,
    }
    lifting_subset = dict(lifting)
    lifting_subset.update(
        {
            "format": LIFTING_CACHE_VERSION,
            "created_at_utc": now,
            "samples": selected_lifting,
            "sample_count": len(selected_lifting),
            "sequence_count": len(selected_lifting),
            "object_count": len(selected_lifting),
            "source_training_lifting_manifest": str(lifting_path),
            "source_training_lifting_manifest_sha256": sha256_file(lifting_path),
            "training_overlap_protocol_sha256": protocol["protocol_sha256"],
            "training_overlap_subset_format": LIFTING_SUBSET_FORMAT,
            "training_overlap": True,
            "passed": True,
            "training_ready": False,
        }
    )
    return selection, lifting_subset


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_scope", choices=SOURCE_SCOPES, required=True)
    parser.add_argument("--lifting_manifest", required=True)
    parser.add_argument("--slat_manifest", required=True)
    parser.add_argument("--output_selection", required=True)
    parser.add_argument("--output_lifting_subset", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_lifting_meta_manifest")
    parser.add_argument("--mixed_slat_meta_manifest")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if int(args.count) <= 0:
        raise ValueError("count must be positive")
    selection_path = Path(args.output_selection).expanduser().resolve()
    lifting_subset_path = Path(args.output_lifting_subset).expanduser().resolve()
    selection, lifting_subset = build_selection(args)
    if selection_path.exists() or lifting_subset_path.exists():
        if not args.resume or not selection_path.is_file() or not lifting_subset_path.is_file():
            raise FileExistsError("selection outputs already exist or are partial")
        existing = load_json(selection_path)
        if (
            existing.get("training_overlap_protocol", {}).get("protocol_sha256")
            != selection["training_overlap_protocol"]["protocol_sha256"]
        ):
            raise RuntimeError("existing training-overlap selection differs")
    else:
        atomic_json(selection_path, selection)
        atomic_json(lifting_subset_path, lifting_subset)
    print(
        json.dumps(
            {
                "passed": True,
                "formal": False,
                "training_overlap": True,
                "source_scope": args.source_scope,
                "objects": int(args.count),
                "selection": str(selection_path),
                "lifting_subset": str(lifting_subset_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
