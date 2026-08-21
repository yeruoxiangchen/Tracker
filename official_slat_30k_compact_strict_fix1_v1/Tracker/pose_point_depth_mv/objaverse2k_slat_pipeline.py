#!/usr/bin/env python3
"""Freeze and assemble the object-disjoint Objaverse2K SLat pipeline."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from ar_ss_flow.pose_lifting import LIFTING_CACHE_VERSION
from pose_point_depth_mv.build_local_lh_slats import (
    LOCAL_LH_SLAT_VERSION,
    NATIVE_OBJAVERSE_MIN_SEQUENCE_TARGET_IOU,
    NATIVE_OBJAVERSE_PRIMARY_TARGET_POLICY,
    NATIVE_OBJAVERSE_TARGET_CONTRACT,
    index_render_samples,
    object_inputs,
)
from pose_point_depth_mv.direct_slat_flow import (
    DIRECT_SLAT_CACHE_VERSION,
    canonical_json_sha256,
)


SPLIT_FORMAT = "pose_point_depth_mv.objaverse2k_slat_split.v1"
SPLIT_MARKER_FORMAT = "pose_point_depth_mv.objaverse2k_slat_split_complete.v1"
TARGET_MARKER_FORMAT = "pose_point_depth_mv.objaverse2k_lh_slats_complete.v1"
TARGET_PREFLIGHT_FORMAT = "pose_point_depth_mv.objaverse2k_native_target_preflight.v1"
CACHE_MERGE_FORMAT = "pose_point_depth_mv.objaverse2k_slat_cache_merge.v1"
SPLIT_MARKER = "_OBJAVERSE2K_SLAT_SPLIT_COMPLETE.json"
TARGET_MARKER = "_OBJAVERSE2K_LOCAL_LH_SLATS_COMPLETE.json"
TARGET_PREFLIGHT = "native_target_preflight.json"
CACHE_MARKER = "_OBJAVERSE2K_SLAT_CACHE_MERGE_COMPLETE.json"
NATIVE_NORMALIZATION_BINDING_FORMAT = (
    "pose_point_depth_mv.native_objaverse_canonical_normalization.v1"
)
NATIVE_RENDER_FORMAT = "pixal3d_multiview.objaverse_sparse.v1"
NATIVE_CANONICAL_FRAME = "pixal3d_sparse_structure"
NATIVE_NORMALIZATION_POLICY = "imported_frame_center_scale_v2"
PATH_FIELDS = (
    "target_file",
    "support_file",
    "physical_file",
    "condition_file",
    "source_lh_slat",
    "source_glb",
    "ss_latent",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_csv(value: str) -> list[str]:
    result = [item.strip() for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


def parse_int_csv(value: str) -> list[int]:
    result = [int(item) for item in parse_csv(value)]
    if len(result) != len(set(result)):
        raise ValueError("integer CSV values must be unique")
    return result


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def stable_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(
        f"{int(seed)}\0{namespace}\0{value}".encode("utf-8")
    ).hexdigest()


def validate_lifting_manifest(
    path: str | Path,
    *,
    expected_objects: int | None = None,
    expected_samples: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).expanduser().resolve()
    payload = load_json(manifest_path)
    rows = payload.get("samples")
    if (
        payload.get("format") != LIFTING_CACHE_VERSION
        or payload.get("passed") is not True
        or payload.get("training_ready") is not True
        or not isinstance(rows, list)
        or not rows
    ):
        raise RuntimeError(f"not a passed lifting manifest: {manifest_path}")
    uids = [str(row.get("uid", "")) for row in rows]
    objects = [str(row.get("object_uid", row.get("uid", ""))) for row in rows]
    if not all(uids) or len(uids) != len(set(uids)) or not all(objects):
        raise RuntimeError("lifting manifest has empty or duplicate sequence identity")
    if int(payload.get("sample_count", -1)) != len(rows):
        raise RuntimeError("lifting sample_count differs from rows")
    if int(payload.get("object_count", -1)) != len(set(objects)):
        raise RuntimeError("lifting object_count differs from rows")
    if expected_objects is not None and len(set(objects)) != int(expected_objects):
        raise RuntimeError(
            f"lifting object count={len(set(objects))} expected={expected_objects}"
        )
    if expected_samples is not None and len(rows) != int(expected_samples):
        raise RuntimeError(
            f"lifting sample count={len(rows)} expected={expected_samples}"
        )
    return manifest_path, payload


def resolve_native_objaverse_normalization_bindings(
    cache_manifest_path: str | Path,
    cache_manifest: dict[str, Any],
    objects: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Resolve legacy Objaverse margins through the frozen manifest hash chain."""

    cache_path = Path(cache_manifest_path).expanduser().resolve()
    source_path = resolve_path(
        cache_path.parent, str(cache_manifest.get("source_lifting_manifest", ""))
    )
    expected_source_sha = str(cache_manifest.get("source_lifting_manifest_sha256", ""))
    if not source_path.is_file() or sha256_file(source_path) != expected_source_sha:
        raise RuntimeError("direct-SLAT cache source lifting manifest changed")
    source = load_json(source_path)
    split = source.get("objaverse2k_split")
    if split is None:
        return {}
    if split.get("format") != SPLIT_FORMAT:
        raise RuntimeError("direct-SLAT cache has an invalid Objaverse2K split contract")

    inventory_path = Path(str(split.get("render_inventory", ""))).resolve()
    expected_inventory_sha = str(split.get("render_inventory_sha256", ""))
    if not inventory_path.is_file() or sha256_file(inventory_path) != expected_inventory_sha:
        raise RuntimeError("Objaverse2K render inventory changed")
    inventory = load_json(inventory_path)
    if inventory.get("format") != "pose_point_depth_mv.objaverse2k_render_inventory.v1":
        raise RuntimeError("unsupported Objaverse2K render inventory")
    inventory_payload = dict(inventory)
    claimed_inventory_hash = str(inventory_payload.pop("inventory_sha256", ""))
    if canonical_json_sha256(inventory_payload) != claimed_inventory_hash:
        raise RuntimeError("Objaverse2K render inventory canonical hash mismatch")
    split_source_path = Path(str(split.get("source_lifting_manifest", ""))).resolve()
    split_source_sha = str(split.get("source_lifting_manifest_sha256", ""))
    if (
        Path(str(inventory.get("source_lifting_manifest", ""))).resolve()
        != split_source_path
        or inventory.get("source_lifting_manifest_sha256") != split_source_sha
        or not split_source_path.is_file()
        or sha256_file(split_source_path) != split_source_sha
    ):
        raise RuntimeError("Objaverse2K render inventory source binding changed")

    by_latent: dict[str, dict[str, Any]] = {}
    total_samples = 0
    manifest_bindings = inventory.get("manifests")
    if not isinstance(manifest_bindings, list) or not manifest_bindings:
        raise RuntimeError("Objaverse2K render inventory has no manifests")
    for manifest_binding in manifest_bindings:
        manifest_path = Path(str(manifest_binding.get("path", ""))).resolve()
        manifest_sha = str(manifest_binding.get("sha256", ""))
        if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_sha:
            raise RuntimeError(f"Objaverse render manifest changed: {manifest_path}")
        render = load_json(manifest_path)
        rows = render.get("samples")
        if render.get("format") != NATIVE_RENDER_FORMAT or not isinstance(rows, list):
            raise RuntimeError(f"unsupported Objaverse render manifest: {manifest_path}")
        if int(manifest_binding.get("sample_count", -1)) != len(rows):
            raise RuntimeError(f"Objaverse render sample count changed: {manifest_path}")
        object_count = len(
            {str(row.get("object_uid", row.get("uid", ""))) for row in rows}
        )
        if int(manifest_binding.get("object_count", -1)) != object_count:
            raise RuntimeError(f"Objaverse render object count changed: {manifest_path}")
        if render.get("canonical_latent_frame") != NATIVE_CANONICAL_FRAME:
            raise RuntimeError(f"Objaverse canonical frame changed: {manifest_path}")
        margin = float(render.get("build_config", {}).get("canonical_margin", float("nan")))
        if not math.isfinite(margin) or margin <= 0.0:
            raise RuntimeError(f"Objaverse canonical margin is invalid: {manifest_path}")
        builder_binding = dict(render.get("code_bindings", {}).get("dataset_builder", {}))
        builder_sha = str(builder_binding.get("sha256", ""))
        if len(builder_sha) != 64:
            raise RuntimeError(f"Objaverse dataset-builder binding is absent: {manifest_path}")
        latent_root = Path(str(render.get("latent_root", ""))).resolve()
        for row in rows:
            uid = str(row.get("uid", ""))
            object_uid = str(row.get("object_uid", uid))
            latent_path = resolve_path(latent_root, str(row.get("ss_latent", "")))
            source_glb = Path(str(row.get("source_glb", ""))).resolve()
            normalization_policy = str(
                row.get("renderer_audit", {}).get("normalization_policy", "")
            )
            if not uid or not object_uid or not latent_path.is_file() or not source_glb.is_file():
                raise RuntimeError(f"Objaverse render identity is incomplete: {uid!r}")
            if normalization_policy != NATIVE_NORMALIZATION_POLICY:
                raise RuntimeError(
                    f"Objaverse normalization policy changed uid={uid}: "
                    f"{normalization_policy!r}"
                )
            binding = {
                "format": NATIVE_NORMALIZATION_BINDING_FORMAT,
                "canonical_margin": margin,
                "canonical_latent_frame": NATIVE_CANONICAL_FRAME,
                "normalization_policy": normalization_policy,
                "uid": uid,
                "object_uid": object_uid,
                "ss_latent": str(latent_path),
                "source_glb": str(source_glb),
                "render_manifest": str(manifest_path),
                "render_manifest_sha256": manifest_sha,
                "render_inventory": str(inventory_path),
                "render_inventory_sha256": expected_inventory_sha,
                "dataset_builder_sha256": builder_sha,
            }
            previous = by_latent.setdefault(str(latent_path), binding)
            if previous != binding:
                raise RuntimeError(f"Objaverse latent has conflicting bindings: {latent_path}")
        total_samples += len(rows)
    if int(inventory.get("sample_count", -1)) != total_samples:
        raise RuntimeError("Objaverse render inventory sample union changed")

    selected: dict[str, dict[str, Any]] = {}
    for row in objects:
        latent_path = Path(str(row.get("ss_latent", ""))).resolve()
        binding = by_latent.get(str(latent_path))
        if binding is None:
            raise RuntimeError(f"Objaverse cache latent is absent from inventory: {latent_path}")
        object_uid = str(row.get("object_uid", ""))
        source_glb = Path(str(row.get("source_glb", ""))).resolve()
        if binding["object_uid"] != object_uid or Path(binding["source_glb"]) != source_glb:
            raise RuntimeError(f"Objaverse cache/render identity differs: {object_uid}")
        expected_glb_sha = str(row.get("source_glb_sha256", ""))
        if not expected_glb_sha or sha256_file(source_glb) != expected_glb_sha:
            raise RuntimeError(f"Objaverse source GLB changed: {source_glb}")
        selected[str(latent_path)] = binding
    return selected


def assign_object_workers(
    rows: list[dict[str, Any]], num_workers: int
) -> list[dict[str, Any]]:
    if int(num_workers) <= 0:
        raise ValueError("num_workers must be positive")
    by_object: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_object[str(row.get("object_uid", row["uid"]))].append(index)
    bins: list[list[tuple[str, list[int]]]] = [[] for _ in range(int(num_workers))]
    loads = [0] * int(num_workers)
    for object_uid, indices in sorted(
        by_object.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        worker = min(range(int(num_workers)), key=lambda value: (loads[value], value))
        bins[worker].append((object_uid, indices))
        loads[worker] += len(indices)
    assignments = []
    for worker, values in enumerate(bins):
        objects = sorted(object_uid for object_uid, _ in values)
        indices = sorted(index for _, object_indices in values for index in object_indices)
        assignments.append(
            {
                "worker_index": worker,
                "object_count": len(objects),
                "sample_count": len(indices),
                "object_uids": objects,
                "object_uid_hash": canonical_json_sha256(objects),
                "indices": indices,
                "uid_hash": canonical_json_sha256(
                    [str(rows[index]["uid"]) for index in indices]
                ),
            }
        )
    if sorted(index for row in assignments for index in row["indices"]) != list(
        range(len(rows))
    ):
        raise AssertionError("worker row partition is incomplete")
    object_sets = [set(row["object_uids"]) for row in assignments]
    if any(
        object_sets[left].intersection(object_sets[right])
        for left in range(len(object_sets))
        for right in range(left + 1, len(object_sets))
    ):
        raise AssertionError("worker partition split an object")
    return assignments


def _absolute_lifting_rows(
    rows: Iterable[dict[str, Any]], source: dict[str, Any], source_path: Path
) -> list[dict[str, Any]]:
    source_root = Path(source.get("output_dir", source_path.parent)).resolve()
    output = []
    for row in rows:
        current = dict(row)
        for field in ("cache_file", "ss_latent"):
            if current.get(field):
                current[field] = str(resolve_path(source_root, current[field]))
        output.append(current)
    return output


def _split_manifest(
    *,
    name: str,
    rows: list[dict[str, Any]],
    source_path: Path,
    source: dict[str, Any],
    output_path: Path,
    split_contract: dict[str, Any],
    workers: list[dict[str, Any]],
) -> dict[str, Any]:
    object_uids = sorted(
        {str(row.get("object_uid", row["uid"])) for row in rows}
    )
    payload = {
        key: value
        for key, value in source.items()
        if key
        not in {
            "created_at_utc",
            "output_dir",
            "source_cache_manifest",
            "source_cache_manifest_sha256",
            "source_lifting_manifest",
            "source_lifting_manifest_sha256",
            "sample_count",
            "object_count",
            "samples",
            "shard_merge",
        }
    }
    payload.update(
        {
            "format": LIFTING_CACHE_VERSION,
            "created_at_utc": utc_now(),
            "output_dir": str(output_path.parent),
            "source_cache_manifest": str(source_path),
            "source_cache_manifest_sha256": sha256_file(source_path),
            "source_lifting_manifest": str(source_path),
            "source_lifting_manifest_sha256": sha256_file(source_path),
            "sample_count": len(rows),
            "object_count": len(object_uids),
            "failure_count": 0,
            "samples": rows,
            "passed": True,
            "training_ready": True,
            "objaverse2k_split": {
                **split_contract,
                "name": name,
                "object_count": len(object_uids),
                "sample_count": len(rows),
                "object_uid_hash": canonical_json_sha256(object_uids),
                "workers": workers,
            },
        }
    )
    return payload


def command_prepare(args: argparse.Namespace) -> None:
    source_path, source = validate_lifting_manifest(
        args.source_lifting_manifest,
        expected_objects=int(args.expected_source_objects),
        expected_samples=int(args.expected_source_samples),
    )
    audit_path, audit = validate_lifting_manifest(args.audit_lifting_manifest)
    audit_objects = {
        str(row.get("object_uid", row["uid"])) for row in audit["samples"]
    }
    if len(audit_objects) != int(args.expected_audit_objects):
        raise RuntimeError("Audit64 object count differs from the frozen expectation")
    source_objects = sorted(
        {str(row.get("object_uid", row["uid"])) for row in source["samples"]}
    )
    if not audit_objects.issubset(source_objects):
        raise RuntimeError("Audit64 is not a subset of the source 2K snapshot")

    render_paths = [Path(value).expanduser().resolve() for value in parse_csv(args.render_manifests)]
    render_inventory = []
    render_by_uid: dict[str, tuple[str, Path]] = {}
    for path in render_paths:
        payload = load_json(path)
        if payload.get("format") != "pixal3d_multiview.objaverse_sparse.v1":
            raise RuntimeError(f"unsupported render manifest: {path}")
        root = Path(payload["latent_root"]).resolve()
        rows = payload.get("samples")
        if not isinstance(rows, list):
            raise RuntimeError(f"render manifest has no samples: {path}")
        for row in rows:
            uid = str(row["uid"])
            identity = (
                str(row.get("object_uid", uid)),
                resolve_path(root, row["ss_latent"]),
            )
            previous = render_by_uid.setdefault(uid, identity)
            if previous != identity:
                raise RuntimeError(f"render UID has conflicting identities: {uid}")
        render_inventory.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "sample_count": len(rows),
                "object_count": len(
                    {str(row.get("object_uid", row["uid"])) for row in rows}
                ),
                "trajectory_mode": payload.get("trajectory_mode"),
                "selected_views": payload.get("build_config", {}).get(
                    "selected_views", payload.get("num_views")
                ),
            }
        )
    source_root = Path(source.get("output_dir", source_path.parent)).resolve()
    missing_render = []
    for row in source["samples"]:
        uid = str(row["uid"])
        expected = (
            str(row.get("object_uid", uid)),
            resolve_path(source_root, row["ss_latent"]),
        )
        if render_by_uid.get(uid) != expected:
            missing_render.append(uid)
    if missing_render:
        raise RuntimeError(
            f"source/render identities differ for {len(missing_render)} rows; "
            f"first={missing_render[:8]}"
        )

    eligible_dev = sorted(
        set(source_objects) - audit_objects,
        key=lambda uid: (stable_rank(args.seed, "dev64", uid), uid),
    )
    if len(eligible_dev) < int(args.dev_objects):
        raise RuntimeError("not enough non-Audit64 objects for dev split")
    dev_objects = set(eligible_dev[: int(args.dev_objects)])
    train_objects = set(source_objects) - dev_objects
    if audit_objects - train_objects:
        raise AssertionError("Audit64 unexpectedly left the training split")
    source_rows = sorted(
        _absolute_lifting_rows(source["samples"], source, source_path),
        key=lambda row: (
            str(row.get("object_uid", row["uid"])),
            str(row["uid"]),
        ),
    )
    train_rows = [
        row
        for row in source_rows
        if str(row.get("object_uid", row["uid"])) in train_objects
    ]
    dev_rows = [
        row
        for row in source_rows
        if str(row.get("object_uid", row["uid"])) in dev_objects
    ]
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    render_inventory_path = output_dir / "render_manifest_inventory.json"
    render_payload = {
        "format": "pose_point_depth_mv.objaverse2k_render_inventory.v1",
        "source_lifting_manifest": str(source_path),
        "source_lifting_manifest_sha256": sha256_file(source_path),
        "manifests": render_inventory,
        "sample_count": len(render_by_uid),
    }
    render_payload["inventory_sha256"] = canonical_json_sha256(render_payload)
    atomic_json(render_inventory_path, render_payload)
    split_contract = {
        "format": SPLIT_FORMAT,
        "formal": False,
        "purpose": "Objaverse2K SLat development checkpoint selection",
        "source_lifting_manifest": str(source_path),
        "source_lifting_manifest_sha256": sha256_file(source_path),
        "audit_lifting_manifest": str(audit_path),
        "audit_lifting_manifest_sha256": sha256_file(audit_path),
        "audit_object_count": len(audit_objects),
        "audit_objects_remain_in_train": True,
        "dev_object_disjoint_from_audit64": True,
        "train_dev_object_disjoint": True,
        "seed": int(args.seed),
        "dev_selection": "stable_sha256_rank_excluding_audit64",
        "render_inventory": str(render_inventory_path),
        "render_inventory_sha256": sha256_file(render_inventory_path),
        "sequence_order": "object_uid_then_uid_ascending",
        "object_primary_sequence": NATIVE_OBJAVERSE_PRIMARY_TARGET_POLICY,
    }
    manifests = {}
    for name, rows in (("train", train_rows), ("dev", dev_rows)):
        split_dir = output_dir / name
        split_dir.mkdir()
        workers = assign_object_workers(rows, int(args.num_workers))
        for worker in workers:
            atomic_text(
                split_dir / f"worker_{worker['worker_index']:03d}_indices.txt",
                ",".join(str(index) for index in worker["indices"]) + "\n",
            )
        manifest_path = split_dir / "lifting_manifest.json"
        payload = _split_manifest(
            name=name,
            rows=rows,
            source_path=source_path,
            source=source,
            output_path=manifest_path,
            split_contract=split_contract,
            workers=workers,
        )
        atomic_json(manifest_path, payload)
        manifests[name] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "sample_count": len(rows),
            "object_count": len(
                {str(row.get("object_uid", row["uid"])) for row in rows}
            ),
            "object_uid_hash": payload["objaverse2k_split"]["object_uid_hash"],
        }
    marker = {
        "format": SPLIT_MARKER_FORMAT,
        "created_at_utc": utc_now(),
        "source_lifting_manifest": str(source_path),
        "source_lifting_manifest_sha256": sha256_file(source_path),
        "render_inventory": str(render_inventory_path),
        "render_inventory_sha256": sha256_file(render_inventory_path),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "source_object_count": len(source_objects),
        "source_sample_count": len(source_rows),
        "audit_object_count": len(audit_objects),
        "audit_object_uid_hash": canonical_json_sha256(sorted(audit_objects)),
        "manifests": manifests,
        "passed": True,
    }
    atomic_json(output_dir / SPLIT_MARKER, marker)
    print(json.dumps(marker, indent=2, ensure_ascii=False))


def load_split_bundle(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    value = Path(path).expanduser().resolve()
    marker_path = value / SPLIT_MARKER if value.is_dir() else value
    marker = load_json(marker_path)
    if marker.get("format") != SPLIT_MARKER_FORMAT or marker.get("passed") is not True:
        raise RuntimeError("split bundle is not complete")
    manifests = {}
    for name in ("train", "dev"):
        binding = marker.get("manifests", {}).get(name, {})
        manifest_path = Path(str(binding.get("path", ""))).resolve()
        if not manifest_path.is_file() or sha256_file(manifest_path) != binding.get("sha256"):
            raise RuntimeError(f"{name} split manifest changed")
        payload = load_json(manifest_path)
        split = payload.get("objaverse2k_split", {})
        if split.get("format") != SPLIT_FORMAT or split.get("name") != name:
            raise RuntimeError(f"invalid {name} split contract")
        manifests[name] = payload
    train_objects = {
        str(row.get("object_uid", row["uid"])) for row in manifests["train"]["samples"]
    }
    dev_objects = {
        str(row.get("object_uid", row["uid"])) for row in manifests["dev"]["samples"]
    }
    if train_objects.intersection(dev_objects):
        raise RuntimeError("train/dev object leakage")
    return marker_path, marker, manifests


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty list")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def command_preflight_targets(args: argparse.Namespace) -> None:
    split_path, split_marker, splits = load_split_bundle(args.split_bundle)
    output_path = split_path.parent / TARGET_PREFLIGHT
    if output_path.exists():
        raise FileExistsError(output_path)
    minimum_iou = float(args.min_native_sequence_target_iou)
    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("min_native_sequence_target_iou must lie in [0,1]")
    render_paths = [
        Path(value).expanduser().resolve() for value in parse_csv(args.render_manifests)
    ]
    render_index, render_bindings = index_render_samples(
        [str(path) for path in render_paths]
    )
    inventory = load_json(split_marker["render_inventory"])
    if sha256_file(split_marker["render_inventory"]) != split_marker.get(
        "render_inventory_sha256"
    ):
        raise RuntimeError("split render inventory changed")
    expected_render_bindings = {
        (str(Path(row["path"]).resolve()), str(row["sha256"]))
        for row in inventory.get("manifests", [])
    }
    actual_render_bindings = {
        (str(Path(row["path"]).resolve()), str(row["sha256"]))
        for row in render_bindings
    }
    if actual_render_bindings != expected_render_bindings:
        raise RuntimeError("preflight render manifests differ from the frozen split")
    expected_rows = [
        row for name in ("train", "dev") for row in splits[name]["samples"]
    ]
    expected_objects = sorted(
        {str(row.get("object_uid", row["uid"])) for row in expected_rows}
    )
    expected_uids = {str(row["uid"]) for row in expected_rows}
    actual_uids = {
        str(row["uid"])
        for object_uid in expected_objects
        for row in render_index.get(object_uid, [])
    }
    if set(expected_objects) - set(render_index):
        raise RuntimeError("preflight render manifests miss split objects")
    if expected_uids != actual_uids:
        raise RuntimeError(
            "preflight render sequences differ from the frozen train/dev split"
        )

    records = []
    comparison_ious: list[float] = []
    comparison_count_ratios: list[float] = []
    for ordinal, object_uid in enumerate(expected_objects, start=1):
        resolved = object_inputs(
            object_uid,
            render_index[object_uid],
            max_views=0,
            target_contract=NATIVE_OBJAVERSE_TARGET_CONTRACT,
            min_native_sequence_target_iou=minimum_iou,
        )
        selection = dict(resolved["target_selection"])
        primary_uid = str(selection["primary_uid"])
        object_uids = sorted(
            str(row["uid"])
            for row in expected_rows
            if str(row.get("object_uid", row["uid"])) == object_uid
        )
        if not object_uids or primary_uid != object_uids[0]:
            raise RuntimeError(
                f"object={object_uid} primary target differs from split first sequence"
            )
        for row in resolved["source_latents"]:
            if not row["is_primary_target"]:
                comparison_ious.append(float(row["target_iou_to_primary"]))
                comparison_count_ratios.append(
                    float(row["target_count_ratio_to_primary"])
                )
        records.append(
            {
                "object_uid": object_uid,
                "primary_uid": primary_uid,
                "sequence_count": int(selection["sequence_count"]),
                "observed_min_target_iou": float(
                    selection["observed_min_target_iou"]
                ),
                "observed_min_target_count_ratio": float(
                    selection["observed_min_target_count_ratio"]
                ),
            }
        )
        if ordinal % 100 == 0 or ordinal == len(expected_objects):
            print(
                f"[objaverse2k_target_preflight] {ordinal}/{len(expected_objects)}",
                flush=True,
            )

    iou_summary = {
        "comparison_count": len(comparison_ious),
        "minimum": min(comparison_ious) if comparison_ious else 1.0,
        "p01": _quantile(comparison_ious, 0.01) if comparison_ious else 1.0,
        "p05": _quantile(comparison_ious, 0.05) if comparison_ious else 1.0,
        "median": _quantile(comparison_ious, 0.50) if comparison_ious else 1.0,
        "maximum": max(comparison_ious) if comparison_ious else 1.0,
        "below_minimum_count": sum(
            value < minimum_iou for value in comparison_ious
        ),
    }
    count_ratio_summary = {
        "minimum": (
            min(comparison_count_ratios) if comparison_count_ratios else 1.0
        ),
        "p01": (
            _quantile(comparison_count_ratios, 0.01)
            if comparison_count_ratios
            else 1.0
        ),
        "median": (
            _quantile(comparison_count_ratios, 0.50)
            if comparison_count_ratios
            else 1.0
        ),
        "maximum": (
            max(comparison_count_ratios) if comparison_count_ratios else 1.0
        ),
    }
    report = {
        "format": TARGET_PREFLIGHT_FORMAT,
        "created_at_utc": utc_now(),
        "split_bundle": str(split_path),
        "split_bundle_sha256": sha256_file(split_path),
        "render_manifests": render_bindings,
        "target_contract": NATIVE_OBJAVERSE_TARGET_CONTRACT,
        "primary_target_policy": NATIVE_OBJAVERSE_PRIMARY_TARGET_POLICY,
        "minimum_sequence_target_iou": minimum_iou,
        "object_count": len(expected_objects),
        "sequence_count": len(expected_rows),
        "object_uid_hash": canonical_json_sha256(expected_objects),
        "sequence_uid_hash": canonical_json_sha256(sorted(expected_uids)),
        "sequence_target_iou": iou_summary,
        "sequence_target_count_ratio": count_ratio_summary,
        "records": records,
        "passed": iou_summary["below_minimum_count"] == 0,
    }
    if report["passed"] is not True:
        raise RuntimeError("native target preflight did not pass")
    atomic_json(output_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


def command_finalize_targets(args: argparse.Namespace) -> None:
    split_path, split_marker, splits = load_split_bundle(args.split_bundle)
    preflight_path = split_path.parent / TARGET_PREFLIGHT
    preflight = load_json(preflight_path)
    if (
        preflight.get("format") != TARGET_PREFLIGHT_FORMAT
        or preflight.get("passed") is not True
        or preflight.get("split_bundle") != str(split_path)
        or preflight.get("split_bundle_sha256") != sha256_file(split_path)
    ):
        raise RuntimeError("native target preflight is absent, stale, or failed")
    target_root = Path(args.target_root).expanduser().resolve()
    run_config_path = target_root / "run_config.json"
    run_config = load_json(run_config_path)
    config = run_config.get("config", {})
    if (
        config.get("format") != LOCAL_LH_SLAT_VERSION
        or config.get("target_contract") != NATIVE_OBJAVERSE_TARGET_CONTRACT
        or config.get("native_primary_target_policy")
        != NATIVE_OBJAVERSE_PRIMARY_TARGET_POLICY
        or float(config.get("min_native_sequence_target_iou", -1.0))
        != float(preflight["minimum_sequence_target_iou"])
        or canonical_json_sha256(config) != run_config.get("config_hash")
    ):
        raise RuntimeError("target root does not use the native Objaverse contract")
    target_render_bindings = {
        (str(Path(row["path"]).resolve()), str(row["sha256"]))
        for row in config.get("render_manifests", [])
    }
    preflight_render_bindings = {
        (str(Path(row["path"]).resolve()), str(row["sha256"]))
        for row in preflight.get("render_manifests", [])
    }
    if target_render_bindings != preflight_render_bindings:
        raise RuntimeError("target builder render inputs differ from target preflight")
    expected_bindings = {
        (str(Path(split_marker["manifests"][name]["path"]).resolve()), split_marker["manifests"][name]["sha256"])
        for name in ("train", "dev")
    }
    actual_bindings = {
        (str(Path(row["path"]).resolve()), str(row["sha256"]))
        for row in config.get("lifting_manifests", [])
    }
    if actual_bindings != expected_bindings:
        raise RuntimeError("target builder did not bind the frozen train/dev manifests")
    expected_objects = {
        str(row.get("object_uid", row["uid"]))
        for split in splits.values()
        for row in split["samples"]
    }
    seen: set[str] = set()
    report_bindings = []
    for rank in range(int(args.world_size)):
        report_path = target_root / f"rank_{rank:03d}_report.json"
        report = load_json(report_path)
        if (
            report.get("format") != LOCAL_LH_SLAT_VERSION
            or report.get("config_hash") != run_config["config_hash"]
            or int(report.get("rank", -1)) != rank
            or int(report.get("world_size", -1)) != int(args.world_size)
            or report.get("passed") is not True
            or report.get("failures")
        ):
            raise RuntimeError(f"target rank {rank} did not complete cleanly")
        records = report.get("records", [])
        objects = {str(row["object_uid"]) for row in records}
        if len(objects) != len(records) or seen.intersection(objects):
            raise RuntimeError("target rank reports overlap or duplicate objects")
        for row in records:
            output = Path(row["output"]).resolve()
            if sha256_file(output) != row.get("output_sha256"):
                raise RuntimeError(f"target artifact changed: {output}")
            contracts = {
                source.get("target_contract")
                for source in row.get("source_latents", [])
            }
            if contracts != {NATIVE_OBJAVERSE_TARGET_CONTRACT}:
                raise RuntimeError("target record contains a non-native source contract")
            selection = row.get("target_selection", {})
            if (
                selection.get("policy") != NATIVE_OBJAVERSE_PRIMARY_TARGET_POLICY
                or float(selection.get("minimum_sequence_target_iou", -1.0))
                != float(preflight["minimum_sequence_target_iou"])
                or float(selection.get("observed_min_target_iou", -1.0))
                < float(preflight["minimum_sequence_target_iou"])
            ):
                raise RuntimeError("target record primary-sequence audit differs")
            primary_sources = [
                source
                for source in row.get("source_latents", [])
                if source.get("is_primary_target") is True
            ]
            if len(primary_sources) != 1:
                raise RuntimeError("target record does not bind exactly one primary latent")
        seen.update(objects)
        report_bindings.append(
            {"path": str(report_path), "sha256": sha256_file(report_path), "rank": rank}
        )
    if seen != expected_objects:
        raise RuntimeError(
            f"target object union differs: got={len(seen)} expected={len(expected_objects)}"
        )
    marker = {
        "format": TARGET_MARKER_FORMAT,
        "created_at_utc": utc_now(),
        "split_bundle": str(split_path),
        "split_bundle_sha256": sha256_file(split_path),
        "run_config": str(run_config_path),
        "run_config_sha256": sha256_file(run_config_path),
        "config_hash": run_config["config_hash"],
        "target_contract": NATIVE_OBJAVERSE_TARGET_CONTRACT,
        "target_preflight": str(preflight_path),
        "target_preflight_sha256": sha256_file(preflight_path),
        "world_size": int(args.world_size),
        "object_count": len(seen),
        "object_uid_hash": canonical_json_sha256(sorted(seen)),
        "rank_reports": report_bindings,
        "passed": True,
    }
    atomic_json(target_root / TARGET_MARKER, marker)
    print(json.dumps(marker, indent=2, ensure_ascii=False))


def _absolute_cache_row(row: dict[str, Any], root: Path) -> dict[str, Any]:
    output = dict(row)
    for field in PATH_FIELDS:
        if output.get(field):
            output[field] = str(resolve_path(root, output[field]))
    return output


def command_merge_cache(args: argparse.Namespace) -> None:
    split_path, _, splits = load_split_bundle(args.split_bundle)
    split_name = str(args.split)
    split = splits[split_name]
    expected_workers = list(split["objaverse2k_split"]["workers"])
    input_dirs = [Path(value).expanduser().resolve() for value in parse_csv(args.input_dirs)]
    if len(input_dirs) != len(expected_workers):
        raise RuntimeError("cache input directory count differs from split workers")
    seeds = parse_int_csv(args.ss_seeds)
    config: dict[str, Any] | None = None
    config_hash = ""
    normalization: dict[str, Any] | None = None
    normalization_hash = ""
    frozen_ss: dict[str, Any] | None = None
    all_rows: dict[str, dict[str, Any]] = {}
    all_objects: dict[str, dict[str, Any]] = {}
    input_bindings = []
    split_manifest_expected_path = Path(args.split_bundle).resolve()
    if split_manifest_expected_path.is_dir():
        split_manifest_expected_path = split_manifest_expected_path / split_name / "lifting_manifest.json"
    else:
        split_manifest_expected_path = Path(split["objaverse2k_split"]["source_lifting_manifest"])
        split_manifest_expected_path = Path(
            load_json(Path(args.split_bundle).resolve())["manifests"][split_name]["path"]
        ).resolve()
    split_manifest_expected_sha = sha256_file(split_manifest_expected_path)
    for worker, input_dir in zip(expected_workers, input_dirs):
        manifest_path = input_dir / "manifest.json"
        payload = load_json(manifest_path)
        if (
            payload.get("format") != DIRECT_SLAT_CACHE_VERSION
            or payload.get("materialized") is not True
        ):
            raise RuntimeError(f"worker cache is not materialized: {manifest_path}")
        current_config = dict(payload.get("config", {}))
        if canonical_json_sha256(current_config) != payload.get("config_hash"):
            raise RuntimeError("worker cache config hash mismatch")
        if (
            Path(current_config.get("source_lifting_manifest", "")).resolve()
            != split_manifest_expected_path
            or current_config.get("source_lifting_manifest_sha256")
            != split_manifest_expected_sha
        ):
            raise RuntimeError("worker cache did not bind the frozen split manifest")
        if [int(value) for value in current_config.get("ss_seeds", [])] != seeds:
            raise RuntimeError("worker cache seed set differs")
        if config is None:
            config = current_config
            config_hash = str(payload["config_hash"])
            normalization = dict(payload["slat_normalization"])
            normalization_hash = str(payload["slat_normalization_hash"])
            frozen_ss = dict(payload["frozen_ss"])
        elif (
            current_config != config
            or payload.get("config_hash") != config_hash
            or payload.get("slat_normalization") != normalization
            or payload.get("slat_normalization_hash") != normalization_hash
            or payload.get("frozen_ss") != frozen_ss
        ):
            raise RuntimeError("worker cache protocols differ")
        root = Path(payload.get("output_dir", input_dir)).resolve()
        rows = [_absolute_cache_row(row, root) for row in payload.get("samples", [])]
        objects = [_absolute_cache_row(row, root) for row in payload.get("objects", [])]
        actual_objects = {str(row["object_uid"]) for row in rows}
        if actual_objects != set(worker["object_uids"]):
            raise RuntimeError(
                f"cache worker={worker['worker_index']} object partition differs"
            )
        for row in rows:
            identity = f"{row['uid']}@{int(row['support_seed'])}"
            if identity in all_rows:
                raise RuntimeError(f"duplicate merged cache row: {identity}")
            all_rows[identity] = row
        for row in objects:
            object_uid = str(row["object_uid"])
            if object_uid in all_objects:
                raise RuntimeError(f"duplicate merged target object: {object_uid}")
            all_objects[object_uid] = row
        input_bindings.append(
            {
                "worker_index": int(worker["worker_index"]),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "sample_count": len(rows),
                "object_count": len(actual_objects),
            }
        )
    expected_ids = [
        f"{row['uid']}@{seed}"
        for row in split["samples"]
        for seed in seeds
    ]
    if set(all_rows) != set(expected_ids) or len(all_rows) != len(expected_ids):
        raise RuntimeError(
            f"merged UID/seed union differs: got={len(all_rows)} expected={len(expected_ids)}"
        )
    expected_objects = sorted(
        {str(row.get("object_uid", row["uid"])) for row in split["samples"]}
    )
    if set(all_objects) != set(expected_objects):
        raise RuntimeError("merged target object union differs")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    if config is None or normalization is None or frozen_ss is None:
        raise AssertionError("no worker cache was loaded")
    manifest = {
        "format": DIRECT_SLAT_CACHE_VERSION,
        "materialized": True,
        "output_dir": str(output_dir),
        "source_lifting_manifest": str(split_manifest_expected_path),
        "source_lifting_manifest_sha256": split_manifest_expected_sha,
        "slat_root": config["slat_root"],
        "config": config,
        "config_hash": config_hash,
        "slat_normalization": normalization,
        "slat_normalization_hash": normalization_hash,
        "sample_count": len(expected_ids),
        "sequence_count": len(split["samples"]),
        "object_count": len(expected_objects),
        "uid_hash": canonical_json_sha256(expected_ids),
        "object_uid_hash": canonical_json_sha256(expected_objects),
        "objects": [all_objects[uid] for uid in expected_objects],
        "samples": [all_rows[identity] for identity in expected_ids],
        "target_coverage": {
            "candidate_object_count": len(expected_objects),
            "matched_object_count": len(expected_objects),
            "missing_object_count": 0,
            "coverage": 1.0,
            "missing_objects": [],
            "passed": True,
            "hard_guard": (
                "targets contain external ground-truth feats[N,8]/coords[N,3]; "
                "stock SLAT samples are forbidden as supervision"
            ),
        },
        "frozen_ss": frozen_ss,
        "shard_merge": {
            "format": CACHE_MERGE_FORMAT,
            "split_bundle": str(split_path),
            "split_bundle_sha256": sha256_file(split_path),
            "split": split_name,
            "source_worker_count": len(input_dirs),
            "copy_policy": "absolute artifact references; tensor files not copied",
            "inputs": input_bindings,
        },
    }
    manifest_path = output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    marker = {
        "format": CACHE_MERGE_FORMAT,
        "created_at_utc": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_bundle": str(split_path),
        "split_bundle_sha256": sha256_file(split_path),
        "split": split_name,
        "config_hash": config_hash,
        "sample_count": len(expected_ids),
        "sequence_count": len(split["samples"]),
        "object_count": len(expected_objects),
        "worker_count": len(input_dirs),
        "passed": True,
        "training_ready": True,
    }
    atomic_json(output_dir / CACHE_MARKER, marker)
    print(json.dumps(marker, indent=2, ensure_ascii=False))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source_lifting_manifest", required=True)
    prepare.add_argument("--audit_lifting_manifest", required=True)
    prepare.add_argument("--render_manifests", required=True)
    prepare.add_argument("--output_dir", required=True)
    prepare.add_argument("--dev_objects", type=int, default=64)
    prepare.add_argument("--seed", type=int, default=20260811)
    prepare.add_argument("--num_workers", type=int, default=4)
    prepare.add_argument("--expected_source_objects", type=int, default=2199)
    prepare.add_argument("--expected_source_samples", type=int, default=4137)
    prepare.add_argument("--expected_audit_objects", type=int, default=64)
    prepare.set_defaults(handler=command_prepare)

    preflight = subparsers.add_parser("preflight-targets")
    preflight.add_argument("--split_bundle", required=True)
    preflight.add_argument("--render_manifests", required=True)
    preflight.add_argument(
        "--min_native_sequence_target_iou",
        type=float,
        default=NATIVE_OBJAVERSE_MIN_SEQUENCE_TARGET_IOU,
    )
    preflight.set_defaults(handler=command_preflight_targets)

    finalize = subparsers.add_parser("finalize-targets")
    finalize.add_argument("--split_bundle", required=True)
    finalize.add_argument("--target_root", required=True)
    finalize.add_argument("--world_size", type=int, default=4)
    finalize.set_defaults(handler=command_finalize_targets)

    merge = subparsers.add_parser("merge-cache")
    merge.add_argument("--split_bundle", required=True)
    merge.add_argument("--split", choices=("train", "dev"), required=True)
    merge.add_argument("--input_dirs", required=True)
    merge.add_argument("--output_dir", required=True)
    merge.add_argument("--ss_seeds", required=True)
    merge.set_defaults(handler=command_merge_cache)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
