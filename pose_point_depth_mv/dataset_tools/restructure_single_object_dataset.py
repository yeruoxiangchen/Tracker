#!/usr/bin/env python3
"""Build an immutable, incremental A/B/C/R single-object data audit.

This tool deliberately separates hard asset validity from the softer question
"is this a single object?".  Geometry-node count is useful triage evidence, but
it is not treated as semantic ground truth: high-count assets become C
scene-like *candidates* for review rather than automatic hard rejects.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw


FORMAT = "pose_point_depth_mv.single_object_restructure.v1"
SOURCE_PLAN_FORMAT = "tracker.mixed_mesh10k_sources.v1"
COMPLETE_MARKER_FORMAT = "tracker.mixed_multiview_render_shard_complete.v1"
TIERS = ("A", "B", "C", "R")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally audit source meshes and completed render shards for "
            "the single-object reconstruction main task."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_plan", action="append", default=[])
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument(
        "--exclude_manifest",
        action="append",
        default=[],
        help=(
            "Exclude every object UID present in this manifest before candidate "
            "selection. Intended for historical val/holdout leakage guards."
        ),
    )
    parser.add_argument("--render_root", action="append", default=[])
    parser.add_argument("--overrides_csv")
    parser.add_argument(
        "--reuse_mesh_audit",
        help=(
            "Previous objects.json (or its output directory). Reuse a mesh "
            "audit only when source path, byte size, and mtime still match."
        ),
    )
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument(
        "--source_plan_limit",
        action="append",
        default=[],
        metavar="SOURCE=COUNT",
        help=(
            "Deterministically retain at most COUNT source-plan-only candidates "
            "for SOURCE. Objects present in explicit manifests, completed render "
            "shards, or manual overrides are always retained. Repeat per source; "
            "COUNT=0 excludes unobserved candidates for that source."
        ),
    )
    parser.add_argument("--source_selection_seed", type=int, default=20260730)
    parser.add_argument("--hash_source_files", action="store_true")
    parser.add_argument("--write_previews", action="store_true")
    parser.add_argument("--preview_frames", type=int, default=4)
    parser.add_argument("--max_previews_per_tier", type=int, default=64)
    parser.add_argument("--preview_size", type=int, default=160)
    parser.add_argument("--require_review_for_training", action="store_true")
    parser.add_argument("--a_max_geometry_count", type=int, default=4)
    parser.add_argument("--a_max_aspect_ratio", type=float, default=6.0)
    parser.add_argument("--a_min_dominant_area_ratio", type=float, default=0.55)
    parser.add_argument("--c_min_geometry_count", type=int, default=101)
    parser.add_argument("--c_min_aspect_ratio", type=float, default=12.0)
    parser.add_argument("--c_secondary_geometry_count", type=int, default=21)
    parser.add_argument("--c_max_dominant_area_ratio", type=float, default=0.25)
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_bound_path(value: str | Path, parent: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = parent / path
    return path.resolve()


def infer_source(uid: str) -> str:
    if uid.startswith("objaverse_"):
        return "objaverse"
    if uid.startswith("omni_"):
        return "omni"
    return "legacy"


def empty_object(uid: str, source_glb: str, source: str) -> dict[str, Any]:
    return {
        "object_uid": uid,
        "source": source,
        "source_glb": str(Path(source_glb).expanduser()),
        "source_plan_candidate": False,
        "explicit_manifest": False,
        "sample_uids": [],
        "accepted_sequences": 0,
        "failure_counts": {},
        "quality_flags": {
            "low_texture": False,
            "flat_gray_blob": False,
            "low_projection_support": False,
        },
        "preview_frames": [],
        "manifest_indices": [],
        "render_evidence": False,
    }


def merge_object(
    objects: dict[str, dict[str, Any]],
    uid: str,
    source_glb: str,
    source: str,
) -> dict[str, Any]:
    if uid not in objects:
        objects[uid] = empty_object(uid, source_glb, source)
    row = objects[uid]
    old = resolve_bound_path(row["source_glb"], Path.cwd())
    new = resolve_bound_path(source_glb, Path.cwd())
    if old != new:
        # /data migrations commonly retain symlinks.  Resolve before declaring a
        # scientific identity conflict.
        if old.resolve() != new.resolve():
            raise RuntimeError(
                f"object {uid} has conflicting source meshes: {old} != {new}"
            )
    if row["source"] == "legacy" and source != "legacy":
        row["source"] = source
    return row


def sample_object_uid(sample: dict[str, Any]) -> str:
    value = str(sample.get("object_uid", ""))
    if value:
        return value
    uid = str(sample.get("uid", ""))
    if "_seq" in uid:
        return uid.rsplit("_seq", 1)[0]
    return uid


def manifest_object_uids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    failures = payload.get("failures", [])
    if not isinstance(samples, list) or not isinstance(failures, list):
        raise ValueError(f"{path}: samples/failures must be lists")
    return {
        uid
        for row in [*samples, *failures]
        if (uid := sample_object_uid(row))
    }


def source_asset_identity(source_glb: str | Path) -> str:
    return str(Path(source_glb).expanduser().resolve())


def manifest_source_asset_identities(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    failures = payload.get("failures", [])
    if not isinstance(samples, list) or not isinstance(failures, list):
        raise ValueError(f"{path}: samples/failures must be lists")
    return {
        source_asset_identity(source_glb)
        for row in [*samples, *failures]
        if (source_glb := str(row.get("source_glb", "")))
    }


def collect_manifest(
    path: Path,
    objects: dict[str, dict[str, Any]],
    bindings: list[dict[str, Any]],
    manifest_payloads: list[tuple[Path, dict[str, Any], int]],
    *,
    origin: str,
    completed_marker: Path | None = None,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    failures = payload.get("failures", [])
    if not isinstance(samples, list) or not isinstance(failures, list):
        raise ValueError(f"{path}: samples/failures must be lists")
    binding_index = len(bindings)
    binding = {
        "kind": "render_manifest" if completed_marker else "manifest",
        "origin": origin,
        "path": str(path),
        "sha256": sha256_file(path),
        "accepted_sequences": len(samples),
        "failed_sequences": len(failures),
    }
    if completed_marker is not None:
        marker = json.loads(completed_marker.read_text(encoding="utf-8"))
        if marker.get("schema") != COMPLETE_MARKER_FORMAT:
            raise ValueError(
                f"{completed_marker}: unsupported marker schema "
                f"{marker.get('schema')}"
            )
        if marker.get("render_manifest_sha256") != binding["sha256"]:
            raise RuntimeError(f"completed render manifest changed: {path}")
        binding["complete_marker"] = str(completed_marker)
        binding["complete_marker_sha256"] = sha256_file(completed_marker)
    bindings.append(binding)
    manifest_payloads.append((path, payload, binding_index))

    image_root = resolve_bound_path(payload.get("image_root", path.parent), path.parent)
    for sample in samples:
        uid = sample_object_uid(sample)
        source_glb = str(sample.get("source_glb", ""))
        if not uid or not source_glb:
            raise ValueError(f"{path}: accepted sample lacks object_uid/source_glb")
        row = merge_object(objects, uid, source_glb, infer_source(uid))
        row["explicit_manifest"] = bool(
            row["explicit_manifest"] or completed_marker is None
        )
        row["accepted_sequences"] += 1
        row["render_evidence"] = True
        row["manifest_indices"].append(binding_index)
        sample_uid = str(sample.get("uid", uid))
        row["sample_uids"].append(sample_uid)
        flags = sample.get("quality_flags", {})
        for key in row["quality_flags"]:
            row["quality_flags"][key] = bool(
                row["quality_flags"][key] or flags.get(key, False)
            )
        if not row["preview_frames"]:
            for frame in sample.get("frames", []):
                image = frame.get("image")
                if image:
                    row["preview_frames"].append(
                        str(resolve_bound_path(str(image), image_root))
                    )

    for failure in failures:
        uid = sample_object_uid(failure)
        source_glb = str(failure.get("source_glb", ""))
        if not uid or not source_glb:
            continue
        row = merge_object(objects, uid, source_glb, infer_source(uid))
        row["explicit_manifest"] = bool(
            row["explicit_manifest"] or completed_marker is None
        )
        row["render_evidence"] = True
        row["manifest_indices"].append(binding_index)
        failure_class = str(failure.get("failure_class", "unknown_failure"))
        counts = Counter(row["failure_counts"])
        counts[failure_class] += 1
        row["failure_counts"] = dict(sorted(counts.items()))


def collect_source_plan(
    path: Path,
    objects: dict[str, dict[str, Any]],
    bindings: list[dict[str, Any]],
    source_plan_meta: list[dict[str, Any]],
) -> None:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != SOURCE_PLAN_FORMAT:
        raise ValueError(f"{path}: unsupported source plan {plan.get('schema')}")
    root = path.parent
    plan_binding = {
        "kind": "source_plan",
        "path": str(path),
        "sha256": sha256_file(path),
        "sources": list(plan.get("sources", [])),
    }
    bindings.append(plan_binding)
    meta = {
        "path": str(path),
        "shard_counts": {},
        "sources": [],
    }
    for source in plan.get("sources", []):
        source_spec = plan[source]
        manifest_path = resolve_bound_path(source_spec["manifest"], root)
        if sha256_file(manifest_path) != source_spec["manifest_sha256"]:
            raise RuntimeError(f"frozen source manifest changed: {manifest_path}")
        full = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(full, dict):
            raise ValueError(f"{manifest_path}: expected UID-to-mesh mapping")
        if len(full) != int(source_spec["object_count"]):
            raise RuntimeError(f"source object count changed: {manifest_path}")
        for shard in source_spec["shards"]:
            shard_path = resolve_bound_path(shard["path"], root)
            if sha256_file(shard_path) != shard["manifest_sha256"]:
                raise RuntimeError(f"frozen source shard changed: {shard_path}")
        for uid, source_glb in full.items():
            row = merge_object(objects, str(uid), str(source_glb), str(source))
            row["source_plan_candidate"] = True
        meta["sources"].append(str(source))
        meta["shard_counts"][str(source)] = int(source_spec["shard_count"])
    source_plan_meta.append(meta)


def parse_source_plan_limits(values: list[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        source, separator, count_text = str(value).partition("=")
        source = source.strip()
        count_text = count_text.strip()
        if not separator or not source or not count_text:
            raise ValueError(
                "--source_plan_limit must use SOURCE=COUNT, "
                f"got {value!r}"
            )
        try:
            count = int(count_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid source-plan limit count in {value!r}"
            ) from exc
        if count < 0:
            raise ValueError("source-plan limits must be nonnegative")
        if source in limits:
            raise ValueError(f"duplicate source-plan limit for {source!r}")
        limits[source] = count
    return limits


def source_candidate_key(uid: str, source: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{int(seed)}\0{source}\0{uid}".encode("utf-8")
    ).hexdigest()
    return digest, uid


def select_source_plan_candidates(
    objects: dict[str, dict[str, Any]],
    limits: dict[str, int],
    *,
    seed: int,
    forced_uids: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    forced = forced_uids or set()
    selected = set(objects)
    source_rows: dict[str, dict[str, int]] = {}
    for source, limit in sorted(limits.items()):
        candidates = [
            uid
            for uid, row in objects.items()
            if row.get("source") == source
            and row.get("source_plan_candidate") is True
            and not row.get("manifest_indices")
            and uid not in forced
        ]
        candidates.sort(key=lambda uid: source_candidate_key(uid, source, seed))
        retained = set(candidates[:limit])
        selected.difference_update(set(candidates) - retained)
        source_rows[source] = {
            "source_plan_only_before_limit": len(candidates),
            "source_plan_only_selected": len(retained),
            "source_plan_only_excluded": len(candidates) - len(retained),
            "evidence_or_override_preserved": sum(
                1
                for uid, row in objects.items()
                if row.get("source") == source
                and row.get("source_plan_candidate") is True
                and (bool(row.get("manifest_indices")) or uid in forced)
            ),
        }
    filtered = {uid: objects[uid] for uid in sorted(selected)}
    return filtered, {
        "policy": (
            "limits apply only to source-plan-only candidates; explicit "
            "manifest, completed-render, and override objects are preserved"
        ),
        "seed": int(seed),
        "limits": dict(sorted(limits.items())),
        "object_count_before": len(objects),
        "object_count_after": len(filtered),
        "sources": source_rows,
    }


def deduplicate_source_assets(
    objects: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_asset: dict[str, list[str]] = {}
    for uid, row in objects.items():
        identity = source_asset_identity(row["source_glb"])
        by_asset.setdefault(identity, []).append(uid)

    aliases = []
    removed: set[str] = set()
    for identity, uids in sorted(by_asset.items()):
        if len(uids) < 2:
            continue
        ordered = sorted(
            uids,
            key=lambda uid: (
                not bool(objects[uid].get("explicit_manifest")),
                not bool(objects[uid].get("render_evidence")),
                uid,
            ),
        )
        winner = ordered[0]
        losers = ordered[1:]
        removed.update(losers)
        aliases.append(
            {
                "source_asset_identity": identity,
                "retained_uid": winner,
                "removed_uids": losers,
                "retained_explicit_manifest": bool(
                    objects[winner].get("explicit_manifest")
                ),
            }
        )

    filtered = {
        uid: row for uid, row in sorted(objects.items()) if uid not in removed
    }
    return filtered, {
        "policy": (
            "resolved source-mesh identity; explicit manifests take precedence, "
            "then completed render evidence, then lexical UID"
        ),
        "object_count_before": len(objects),
        "object_count_after": len(filtered),
        "duplicate_source_asset_count": len(aliases),
        "removed_alias_count": len(removed),
        "aliases": aliases,
    }


def completed_render_manifests(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for marker in sorted(root.glob("*/shard_*/_WORKER_COMPLETE.json")):
        manifest = marker.parent / "manifest.json"
        if not manifest.is_file():
            raise RuntimeError(f"completion marker lacks manifest: {marker}")
        pairs.append((manifest, marker))
    return pairs


def load_overrides(path: Path | None) -> tuple[dict[str, dict[str, str]], dict | None]:
    if path is None:
        return {}, None
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"object_uid", "final_tier", "reason", "reviewer"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns are {sorted(required)}")
        for record in reader:
            uid = record["object_uid"].strip()
            tier = record["final_tier"].strip().upper()
            if not uid:
                continue
            if not tier:
                # The generated template intentionally contains blank review
                # rows.  A partially completed CSV is a valid incremental
                # override set.
                continue
            if tier not in TIERS:
                raise ValueError(f"{path}: invalid tier {tier!r} for {uid}")
            if uid in rows:
                raise ValueError(f"{path}: duplicate override for {uid}")
            reason = record["reason"].strip()
            reviewer = record["reviewer"].strip()
            if not reason or not reviewer:
                raise ValueError(
                    f"{path}: reviewed row {uid} requires reason and reviewer"
                )
            rows[uid] = {
                "final_tier": tier,
                "reason": reason,
                "reviewer": reviewer,
            }
    return rows, {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(rows),
    }


def audit_mesh_task(task: tuple[str, str, bool]) -> tuple[str, dict[str, Any]]:
    uid, source_glb, hash_source = task
    path = Path(source_glb).expanduser()
    base: dict[str, Any] = {
        "source_exists": path.is_file(),
        "source_size": path.stat().st_size if path.is_file() else None,
        "source_mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
        "source_sha256": None,
        "mesh_valid": False,
        "mesh_error": None,
    }
    if not path.is_file():
        base["mesh_error"] = "source file is missing"
        return uid, base
    try:
        if hash_source:
            base["source_sha256"] = sha256_file(path)
        loaded = trimesh.load(path, force="scene", process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = [
                mesh
                for mesh in loaded.geometry.values()
                if isinstance(mesh, trimesh.Trimesh)
                and len(mesh.vertices) > 0
                and len(mesh.faces) > 0
            ]
            scene_bounds = np.asarray(loaded.bounds, dtype=np.float64)
            scene_instance_count = len(list(loaded.graph.nodes_geometry))
        elif isinstance(loaded, trimesh.Trimesh):
            meshes = [loaded] if len(loaded.vertices) and len(loaded.faces) else []
            scene_bounds = np.asarray(loaded.bounds, dtype=np.float64)
            scene_instance_count = 1
        else:
            meshes = []
            scene_bounds = np.empty((0, 3), dtype=np.float64)
            scene_instance_count = 0
        if not meshes:
            raise ValueError("scene has no non-empty triangular mesh geometry")
        if scene_bounds.shape != (2, 3):
            raise ValueError("scene has invalid bounds")
        bounds_min = scene_bounds[0]
        bounds_max = scene_bounds[1]
        extent = bounds_max - bounds_min
        if not np.isfinite(extent).all() or float(extent.max()) <= 1.0e-10:
            raise ValueError("mesh has invalid or degenerate bounds")
        positive = extent[extent > max(float(extent.max()) * 1.0e-6, 1.0e-12)]
        aspect = float(extent.max() / positive.min()) if len(positive) else math.inf
        faces = [int(len(mesh.faces)) for mesh in meshes]
        vertices = [int(len(mesh.vertices)) for mesh in meshes]
        areas = []
        for mesh in meshes:
            try:
                area = float(mesh.area)
            except BaseException:
                area = 0.0
            areas.append(area if math.isfinite(area) and area > 0 else 0.0)
        total_area = float(sum(areas))
        total_faces = int(sum(faces))
        base.update(
            {
                "mesh_valid": True,
                "geometry_count": len(meshes),
                "scene_instance_count": scene_instance_count,
                "vertex_count": int(sum(vertices)),
                "face_count": total_faces,
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
                "extent": extent.tolist(),
                "aspect_ratio": aspect,
                "surface_area": total_area,
                "dominant_area_ratio": (
                    float(max(areas) / total_area) if total_area > 0 else 0.0
                ),
                "dominant_face_ratio": (
                    float(max(faces) / total_faces) if total_faces > 0 else 0.0
                ),
            }
        )
    except BaseException as exc:
        base["mesh_error"] = f"{type(exc).__name__}: {exc}"
    return uid, base


def classify(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    reasons: list[str] = []
    audit = row["mesh_audit"]
    if not audit["mesh_valid"]:
        return "R", [f"hard_invalid_mesh: {audit['mesh_error']}"]
    source_asset_failures = int(row["failure_counts"].get("source_asset_rejected", 0))
    if (
        row["render_evidence"]
        and row["accepted_sequences"] == 0
        and source_asset_failures > 0
    ):
        return "R", ["all observed render attempts rejected the source asset"]

    geom = int(audit["geometry_count"])
    aspect = float(audit["aspect_ratio"])
    dominant = float(audit["dominant_area_ratio"])
    if (
        geom >= args.c_min_geometry_count
        or aspect >= args.c_min_aspect_ratio
        or (
            geom >= args.c_secondary_geometry_count
            and dominant <= args.c_max_dominant_area_ratio
        )
    ):
        reasons.append("scene-like candidate; requires semantic human review")
        if geom >= args.c_min_geometry_count:
            reasons.append(f"geometry_count={geom}>={args.c_min_geometry_count}")
        if aspect >= args.c_min_aspect_ratio:
            reasons.append(
                f"aspect_ratio={aspect:.3f}>={args.c_min_aspect_ratio:.3f}"
            )
        if (
            geom >= args.c_secondary_geometry_count
            and dominant <= args.c_max_dominant_area_ratio
        ):
            reasons.append(
                f"geometry_count={geom}, dominant_area_ratio={dominant:.3f}"
            )
        return "C", reasons

    flags = row["quality_flags"]
    high_confidence = (
        row["render_evidence"]
        and row["accepted_sequences"] > 0
        and geom <= args.a_max_geometry_count
        and aspect <= args.a_max_aspect_ratio
        and dominant >= args.a_min_dominant_area_ratio
        and not flags["low_texture"]
        and not flags["flat_gray_blob"]
        and not flags["low_projection_support"]
    )
    if high_confidence:
        return "A", [
            "high-confidence single-object candidate with accepted render evidence"
        ]

    reasons.append("valid but ambiguous/soft-quality candidate")
    if not row["render_evidence"]:
        reasons.append("render evidence not available yet")
    if geom > args.a_max_geometry_count:
        reasons.append(f"multipart geometry_count={geom}; not an automatic rejection")
    if flags["low_texture"]:
        reasons.append("low_texture")
    if row["failure_counts"]:
        reasons.append(f"render_failures={row['failure_counts']}")
    return "B", reasons


def distribute_shards(mapping: dict[str, str], shard_count: int) -> list[dict[str, str]]:
    shards: list[dict[str, str]] = [dict() for _ in range(shard_count)]
    for uid, source_glb in sorted(mapping.items()):
        digest = hashlib.sha256(uid.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], byteorder="big", signed=False) % shard_count
        shards[index][uid] = source_glb
    return shards


def write_filtered_source_plan(
    storage_root: Path,
    identity_root: Path,
    objects: list[dict[str, Any]],
    source_plan_meta: list[dict[str, Any]],
    *,
    require_review: bool,
) -> dict[str, Any] | None:
    if not source_plan_meta:
        return None
    # Multiple plans are supported only when they do not disagree on shard count.
    shard_counts: dict[str, int] = {}
    for meta in source_plan_meta:
        for source, count in meta["shard_counts"].items():
            if source in shard_counts and shard_counts[source] != count:
                raise RuntimeError(f"source {source} has conflicting shard counts")
            shard_counts[source] = count
    eligible = [
        row
        for row in objects
        if row["final_tier"] in ("A", "B")
        and (row["human_reviewed"] or not require_review)
        and row["mesh_audit"]["mesh_valid"]
    ]
    plan_root = storage_root / "training_candidate_source_plan"
    identity_plan_root = identity_root / "training_candidate_source_plan"
    plan: dict[str, Any] = {
        "schema": SOURCE_PLAN_FORMAT,
        "sources": [],
        "output_dir": str(identity_plan_root),
        "shard_policy": "sha256(uid) mod shard_count",
        "single_object_restructure": {
            "format": FORMAT,
            "draft": not require_review,
            "tiers": ["A", "B"],
        },
    }
    for source in sorted(shard_counts):
        mapping = {
            row["object_uid"]: row["source_glb"]
            for row in eligible
            if row["source"] == source
        }
        if not mapping:
            continue
        plan["sources"].append(source)
        manifest_rel = Path(f"{source}_meshes.json")
        write_json(plan_root / manifest_rel, mapping)
        shard_rows = []
        for index, shard in enumerate(distribute_shards(mapping, shard_counts[source])):
            rel = Path(f"{source}_shards") / f"shard_{index:03d}.json"
            write_json(plan_root / rel, shard)
            shard_rows.append(
                {
                    "index": index,
                    "path": str(rel),
                    "object_count": len(shard),
                    "uid_sha256": canonical_json_sha256(sorted(shard)),
                    "manifest_sha256": sha256_file(plan_root / rel),
                }
            )
        plan[source] = {
            "object_count": len(mapping),
            "manifest": str(manifest_rel),
            "manifest_sha256": sha256_file(plan_root / manifest_rel),
            "shard_count": shard_counts[source],
            "shards": shard_rows,
        }
    write_json(plan_root / "build_plan.json", plan)
    return {
        "path": str(identity_plan_root / "build_plan.json"),
        "sha256": sha256_file(plan_root / "build_plan.json"),
        "sources": plan["sources"],
        "object_count": sum(plan[s]["object_count"] for s in plan["sources"]),
        "draft": not require_review,
    }


def write_filtered_manifests(
    storage_root: Path,
    identity_root: Path,
    manifest_payloads: list[tuple[Path, dict[str, Any], int]],
    tier_by_uid: dict[str, str],
    reviewed_by_uid: dict[str, bool],
    *,
    require_review: bool,
) -> list[dict[str, Any]]:
    reports = []
    for index, (source_path, payload, _binding_index) in enumerate(manifest_payloads):
        for label, tiers in (
            ("a", {"A"}),
            ("b", {"B"}),
            ("ab", {"A", "B"}),
            ("c_review", {"C"}),
            ("r_rejected", {"R"}),
        ):
            samples = []
            for sample in payload.get("samples", []):
                uid = sample_object_uid(sample)
                if tier_by_uid.get(uid) not in tiers:
                    continue
                if label in {"a", "b", "ab"} and require_review and not reviewed_by_uid.get(uid):
                    continue
                samples.append(sample)
            if not samples and label not in {"ab", "c_review", "r_rejected"}:
                continue
            filtered = dict(payload)
            filtered["samples"] = samples
            filtered["failures"] = [
                failure
                for failure in payload.get("failures", [])
                if tier_by_uid.get(sample_object_uid(failure)) in tiers
            ]
            filtered["single_object_restructure"] = {
                "format": FORMAT,
                "source_manifest": str(source_path),
                "tier_selection": sorted(tiers),
                "require_review_for_training": require_review,
                "draft": not require_review,
            }
            name = f"{index:03d}_{source_path.stem}_{label}.json"
            out = storage_root / "manifests" / name
            write_json(out, filtered)
            reports.append(
                {
                    "path": str(identity_root / "manifests" / name),
                    "sha256": sha256_file(out),
                    "sample_count": len(samples),
                    "tiers": sorted(tiers),
                    "draft": not require_review,
                }
            )
    return reports


def choose_evenly(values: list[str], count: int) -> list[str]:
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[int(index)] for index in indices]


def preview_tile(
    paths: list[str], title: str, *, frame_count: int, cell_size: int
) -> Image.Image | None:
    paths = choose_evenly(paths, frame_count)
    images = []
    for value in paths:
        try:
            with Image.open(value) as image:
                images.append(image.convert("RGB"))
        except OSError:
            continue
    if not images:
        return None
    header = 24
    canvas = Image.new("RGB", (cell_size * len(images), cell_size + header), "black")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title[:90], fill="white")
    for index, image in enumerate(images):
        image.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
        x = index * cell_size + (cell_size - image.width) // 2
        y = header + (cell_size - image.height) // 2
        canvas.paste(image, (x, y))
    return canvas


def write_review_previews(
    storage_root: Path,
    identity_root: Path,
    objects: list[dict[str, Any]],
    *,
    frame_count: int,
    cell_size: int,
    max_per_tier: int,
) -> dict[str, Any]:
    review = storage_root / "review"
    identity_review = identity_root / "review"
    rows = []
    counts = Counter()
    for row in objects:
        tier = row["final_tier"]
        if counts[tier] >= max_per_tier or not row["preview_frames"]:
            continue
        title = (
            f"{tier} {row['object_uid']} geom={row['mesh_audit'].get('geometry_count')}"
        )
        tile = preview_tile(
            row["preview_frames"],
            title,
            frame_count=frame_count,
            cell_size=cell_size,
        )
        if tile is None:
            continue
        rel = Path("images") / tier / f"{row['object_uid']}.jpg"
        (review / rel).parent.mkdir(parents=True, exist_ok=True)
        tile.save(review / rel, quality=90)
        rows.append((row, rel))
        counts[tier] += 1
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Single-object A/B/C/R review</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee}"
        "article{margin:18px 0;padding:10px;background:#222}"
        "img{max-width:100%}code{color:#9ee}</style>",
        "<h1>Single-object A/B/C/R provisional review</h1>",
        "<p>C is only a scene-like candidate; geometry count is not semantic ground truth.</p>",
    ]
    for row, rel in rows:
        audit = row["mesh_audit"]
        parts.append(
            "<article>"
            f"<h2>{html.escape(row['final_tier'])} "
            f"<code>{html.escape(row['object_uid'])}</code></h2>"
            f"<p>geometry={audit.get('geometry_count')}, "
            f"aspect={audit.get('aspect_ratio')}, "
            f"dominant_area={audit.get('dominant_area_ratio')}</p>"
            f"<p>{html.escape('; '.join(row['tier_reasons']))}</p>"
            f"<img src='{html.escape(str(rel))}'>"
            "</article>"
        )
    (review / "index.html").parent.mkdir(parents=True, exist_ok=True)
    (review / "index.html").write_text("\n".join(parts), encoding="utf-8")
    return {
        "index": str(identity_review / "index.html"),
        "preview_counts": dict(sorted(counts.items())),
    }


def make_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_plan_limits": parse_source_plan_limits(args.source_plan_limit),
        "source_selection_seed": int(args.source_selection_seed),
        "hash_source_files": bool(args.hash_source_files),
        "write_previews": bool(args.write_previews),
        "preview_frames": int(args.preview_frames),
        "max_previews_per_tier": int(args.max_previews_per_tier),
        "preview_size": int(args.preview_size),
        "require_review_for_training": bool(args.require_review_for_training),
        "a_max_geometry_count": args.a_max_geometry_count,
        "a_max_aspect_ratio": args.a_max_aspect_ratio,
        "a_min_dominant_area_ratio": args.a_min_dominant_area_ratio,
        "c_min_geometry_count": args.c_min_geometry_count,
        "c_min_aspect_ratio": args.c_min_aspect_ratio,
        "c_secondary_geometry_count": args.c_secondary_geometry_count,
        "c_max_dominant_area_ratio": args.c_max_dominant_area_ratio,
    }


def load_reusable_mesh_audits(
    value: str | None,
    objects: dict[str, dict[str, Any]],
    *,
    require_source_hash: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if not value:
        return {}, None
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "objects.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected objects.json list")
    reusable: dict[str, dict[str, Any]] = {}
    for old in payload:
        uid = str(old.get("object_uid", ""))
        if uid not in objects:
            continue
        current_path = Path(objects[uid]["source_glb"]).expanduser()
        audit = old.get("mesh_audit")
        if not isinstance(audit, dict) or not current_path.is_file():
            continue
        if require_source_hash and not audit.get("source_sha256"):
            continue
        if resolve_bound_path(old["source_glb"], Path.cwd()) != current_path.resolve():
            continue
        stat = current_path.stat()
        if (
            audit.get("source_size") != stat.st_size
            or audit.get("source_mtime_ns") != stat.st_mtime_ns
        ):
            continue
        if audit.get("mesh_valid") is not True:
            # Recheck invalid assets: downloads or extractors may repair them
            # without preserving every filesystem timestamp.
            continue
        reusable[uid] = audit
    return reusable, {
        "path": str(path),
        "sha256": sha256_file(path),
        "available_count": len(payload),
        "reusable_count": len(reusable),
        "policy": "source path + byte size + mtime exact; invalid rows are re-audited",
    }


def main() -> None:
    args = parse_args()
    if not args.source_plan and not args.manifest and not args.render_root:
        raise SystemExit("at least one --source_plan/--manifest/--render_root is required")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    source_plan_limits = parse_source_plan_limits(args.source_plan_limit)
    if source_plan_limits and args.max_objects > 0:
        raise ValueError(
            "--max_objects cannot be combined with --source_plan_limit; "
            "the global prefix limit could remove preserved evidence objects"
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    objects: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    manifest_payloads: list[tuple[Path, dict[str, Any], int]] = []
    source_plan_meta: list[dict[str, Any]] = []

    for value in args.source_plan:
        path = Path(value).expanduser().resolve()
        collect_source_plan(path, objects, bindings, source_plan_meta)
    for value in args.manifest:
        path = Path(value).expanduser().resolve()
        collect_manifest(
            path,
            objects,
            bindings,
            manifest_payloads,
            origin="explicit_manifest",
        )
    for value in args.render_root:
        root = Path(value).expanduser().resolve()
        completed = completed_render_manifests(root)
        bindings.append(
            {
                "kind": "render_root_inventory",
                "path": str(root),
                "completed_shard_count": len(completed),
                "complete_markers_sha256": canonical_json_sha256(
                    [sha256_file(marker) for _manifest, marker in completed]
                ),
            }
        )
        for manifest, marker in completed:
            collect_manifest(
                manifest,
                objects,
                bindings,
                manifest_payloads,
                origin=str(root),
                completed_marker=marker,
            )

    excluded_uids: set[str] = set()
    excluded_source_assets: set[str] = set()
    exclusion_bindings = []
    for value in args.exclude_manifest:
        path = Path(value).expanduser().resolve()
        uids = manifest_object_uids(path)
        source_assets = manifest_source_asset_identities(path)
        excluded_uids.update(uids)
        excluded_source_assets.update(source_assets)
        binding = {
            "kind": "excluded_object_manifest",
            "path": str(path),
            "sha256": sha256_file(path),
            "object_count": len(uids),
            "source_asset_count": len(source_assets),
        }
        bindings.append(binding)
        exclusion_bindings.append(binding)
    excluded_present = sorted(excluded_uids & set(objects))
    excluded_present_source_assets = sorted(
        {
            source_asset_identity(row["source_glb"])
            for row in objects.values()
        }
        & excluded_source_assets
    )
    if excluded_present or excluded_present_source_assets:
        objects = {
            uid: row
            for uid, row in objects.items()
            if uid not in excluded_uids
            and source_asset_identity(row["source_glb"])
            not in excluded_source_assets
        }

    overrides_path = (
        Path(args.overrides_csv).expanduser().resolve() if args.overrides_csv else None
    )
    overrides, override_binding = load_overrides(overrides_path)
    if override_binding:
        bindings.append({"kind": "manual_overrides", **override_binding})
    excluded_overrides = sorted(set(overrides) & excluded_uids)
    if excluded_overrides:
        raise ValueError(
            "manual overrides contain UIDs excluded by --exclude_manifest: "
            f"{excluded_overrides[:10]}"
        )

    objects, source_alias_report = deduplicate_source_assets(objects)

    selection_report = None
    if source_plan_limits:
        objects, selection_report = select_source_plan_candidates(
            objects,
            source_plan_limits,
            seed=int(args.source_selection_seed),
            forced_uids=set(overrides),
        )
    sorted_uids = sorted(objects)
    if args.max_objects > 0:
        sorted_uids = sorted_uids[: args.max_objects]
        objects = {uid: objects[uid] for uid in sorted_uids}
    unknown_overrides = sorted(set(overrides) - set(objects))
    if unknown_overrides:
        raise ValueError(f"overrides contain unknown UIDs: {unknown_overrides[:10]}")
    reusable_audits, reuse_binding = load_reusable_mesh_audits(
        args.reuse_mesh_audit,
        objects,
        require_source_hash=bool(args.hash_source_files),
    )
    if reuse_binding:
        bindings.append({"kind": "reused_mesh_audit", **reuse_binding})
    config = make_config(args)
    code_sha256 = sha256_file(Path(__file__).resolve())
    input_signature = canonical_json_sha256(
        {
            "format": FORMAT,
            "bindings": bindings,
            "config": config,
            "uids": sorted_uids,
            "code_sha256": code_sha256,
        }
    )
    existing_report = output_dir / "report.json"
    if output_dir.exists():
        if existing_report.is_file():
            report = json.loads(existing_report.read_text(encoding="utf-8"))
            if (
                report.get("format") == FORMAT
                and report.get("passed") is True
                and report.get("input_signature") == input_signature
            ):
                print(
                    json.dumps(
                        {
                            "reused": True,
                            "output_dir": str(output_dir),
                            "report": str(existing_report),
                            "summary": report["summary"],
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                return
        raise RuntimeError(
            f"immutable output exists with different/incomplete identity: {output_dir}"
        )

    pending_uids = [uid for uid in sorted_uids if uid not in reusable_audits]
    print(
        f"[single_object_restructure] objects={len(objects)} "
        f"reused_mesh_audits={len(reusable_audits)} "
        f"pending_mesh_audits={len(pending_uids)} workers={args.workers}",
        flush=True,
    )
    tasks = [
        (uid, objects[uid]["source_glb"], bool(args.hash_source_files))
        for uid in pending_uids
    ]
    audit_results: dict[str, dict[str, Any]] = dict(reusable_audits)
    if tasks:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for done, (uid, result) in enumerate(executor.map(audit_mesh_task, tasks), 1):
                audit_results[uid] = result
                if done == 1 or done % 100 == 0 or done == len(tasks):
                    print(
                        f"[single_object_restructure] mesh {done}/{len(tasks)}",
                        flush=True,
                    )

    rows = []
    for uid in sorted_uids:
        row = objects[uid]
        row["sample_uids"] = sorted(set(row["sample_uids"]))
        row["manifest_indices"] = sorted(set(row["manifest_indices"]))
        row["mesh_audit"] = audit_results[uid]
        auto_tier, reasons = classify(row, args)
        row["auto_tier"] = auto_tier
        override = overrides.get(uid)
        if override:
            row["final_tier"] = override["final_tier"]
            row["human_reviewed"] = True
            row["reviewer"] = override["reviewer"]
            row["tier_reasons"] = [
                f"manual override from auto tier {auto_tier}: {override['reason']}"
            ]
        else:
            row["final_tier"] = auto_tier
            row["human_reviewed"] = False
            row["reviewer"] = None
            row["tier_reasons"] = reasons
        rows.append(row)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent)
    )
    try:
        write_json(staging / "objects.json", rows)
        for tier in TIERS:
            selected = [row["object_uid"] for row in rows if row["final_tier"] == tier]
            (staging / f"tier_{tier.lower()}_uids.txt").write_text(
                "".join(f"{uid}\n" for uid in selected), encoding="utf-8"
            )
        with (staging / "manual_overrides_template.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["object_uid", "final_tier", "reason", "reviewer"])
            for row in rows:
                if row["final_tier"] != "R":
                    writer.writerow([row["object_uid"], "", "", ""])

        tier_by_uid = {row["object_uid"]: row["final_tier"] for row in rows}
        reviewed_by_uid = {row["object_uid"]: row["human_reviewed"] for row in rows}
        filtered_manifests = write_filtered_manifests(
            staging,
            output_dir,
            manifest_payloads,
            tier_by_uid,
            reviewed_by_uid,
            require_review=args.require_review_for_training,
        )
        source_plan_report = write_filtered_source_plan(
            staging,
            output_dir,
            rows,
            source_plan_meta,
            require_review=args.require_review_for_training,
        )
        preview_report = None
        if args.write_previews:
            preview_report = write_review_previews(
                staging,
                output_dir,
                rows,
                frame_count=args.preview_frames,
                cell_size=args.preview_size,
                max_per_tier=args.max_previews_per_tier,
            )

        tier_counts = Counter(row["final_tier"] for row in rows)
        auto_tier_counts = Counter(row["auto_tier"] for row in rows)
        source_counts = Counter(row["source"] for row in rows)
        eligible = [
            row
            for row in rows
            if row["final_tier"] in ("A", "B")
            and (
                row["human_reviewed"]
                or not args.require_review_for_training
            )
        ]
        final_ab = [row for row in rows if row["final_tier"] in ("A", "B")]
        report = {
            "format": FORMAT,
            "passed": True,
            "created_at_utc": utc_now(),
            "output_dir": str(output_dir),
            "input_signature": input_signature,
            "bindings": bindings,
            "config": config,
            "code_binding": {
                "path": str(Path(__file__).resolve()),
                "sha256": code_sha256,
            },
            "summary": {
                "object_count": len(rows),
                "source_counts": dict(sorted(source_counts.items())),
                "auto_tier_counts": {
                    tier: auto_tier_counts.get(tier, 0) for tier in TIERS
                },
                "final_tier_counts": {
                    tier: tier_counts.get(tier, 0) for tier in TIERS
                },
                "human_reviewed_count": sum(row["human_reviewed"] for row in rows),
                "training_candidate_object_count": len(eligible),
                "mesh_invalid_count": sum(
                    not row["mesh_audit"]["mesh_valid"] for row in rows
                ),
                "completed_render_shard_count": sum(
                    binding.get("completed_shard_count", 0)
                    for binding in bindings
                    if binding["kind"] == "render_root_inventory"
                ),
            },
            "source_plan_selection": selection_report,
            "object_exclusions": {
                "bindings": exclusion_bindings,
                "excluded_uid_count": len(excluded_uids),
                "excluded_present_count": len(excluded_present),
                "excluded_source_asset_count": len(excluded_source_assets),
                "excluded_present_source_asset_count": len(
                    excluded_present_source_assets
                ),
            },
            "source_asset_deduplication": source_alias_report,
            "policy": {
                "A": "high-confidence rendered single-object candidate",
                "B": "valid but ambiguous/multipart/soft-quality candidate",
                "C": "scene-like candidate; human semantic review required",
                "R": "hard invalid or all observed renders rejected source asset",
                "geometry_count_guard": (
                    "geometry-node count is triage evidence, not semantic ground truth"
                ),
                "training_ready": bool(args.require_review_for_training)
                and bool(final_ab)
                and all(row["human_reviewed"] for row in final_ab),
                "draft": not bool(args.require_review_for_training),
            },
            "filtered_manifests": filtered_manifests,
            "training_candidate_source_plan": source_plan_report,
            "review": preview_report,
        }
        write_json(staging / "report.json", report)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "passed": True,
                "output_dir": str(output_dir),
                "report": str(output_dir / "report.json"),
                "summary": report["summary"],
                "training_candidate_source_plan": report[
                    "training_candidate_source_plan"
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
