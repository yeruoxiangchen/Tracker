#!/usr/bin/env python3
"""Audit frozen mesh source frames in Blender without rendering image frames."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh

try:
    from .build_objaverse_multiview_sparse_data import (
        BLENDER_RENDER_SCRIPT,
        build_render_buffers,
        load_meshes,
        normalize_vertices,
        sha256_file,
        validate_blender_bounds,
    )
except ImportError:
    from build_objaverse_multiview_sparse_data import (
        BLENDER_RENDER_SCRIPT,
        build_render_buffers,
        load_meshes,
        normalize_vertices,
        sha256_file,
        validate_blender_bounds,
    )


SOURCE_SCHEMA = "tracker.mixed_mesh10k_sources.v1"
REPORT_SCHEMA = "tracker.mixed_mesh_blender_frame_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare trimesh/Pixal3D canonical bounds with Blender-imported "
            "bounds before any expensive frame rendering."
        )
    )
    parser.add_argument("--source_plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blender_path", required=True)
    parser.add_argument("--sources", default="objaverse,omni")
    parser.add_argument(
        "--uids",
        default="",
        help="Optional comma-separated frozen UIDs; otherwise take the first N per source.",
    )
    parser.add_argument("--objects_per_source", type=int, default=4)
    parser.add_argument("--canonical_margin", type=float, default=0.9)
    parser.add_argument("--bounds_tolerance", type=float, default=1.0e-3)
    return parser.parse_args()


def load_frozen_objects(plan_path: Path, plan: dict, source: str) -> dict[str, str]:
    source_plan = plan.get(source)
    if not isinstance(source_plan, dict):
        raise ValueError(f"source is absent from frozen plan: {source}")
    objects: dict[str, str] = {}
    for shard in source_plan["shards"]:
        manifest_path = plan_path.parent / str(shard["path"])
        expected_sha256 = str(shard["manifest_sha256"])
        actual_sha256 = sha256_file(manifest_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"frozen source shard changed: {manifest_path}; "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{manifest_path}: expected UID-to-mesh object")
        objects.update({str(uid): str(path) for uid, path in payload.items()})
    if len(objects) != int(source_plan["object_count"]):
        raise RuntimeError(
            f"{source} frozen object count changed: "
            f"expected={source_plan['object_count']} actual={len(objects)}"
        )
    return objects


def select_objects(
    objects_by_source: dict[str, dict[str, str]],
    requested_uids: list[str],
    objects_per_source: int,
) -> list[tuple[str, str, str]]:
    selected: list[tuple[str, str, str]] = []
    if requested_uids:
        for uid in requested_uids:
            matches = [
                (source, objects[uid])
                for source, objects in objects_by_source.items()
                if uid in objects
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"requested UID must occur exactly once in selected sources: "
                    f"{uid} matches={len(matches)}"
                )
            source, path = matches[0]
            selected.append((source, uid, path))
        return selected

    for source, objects in objects_by_source.items():
        for uid, path in list(objects.items())[:objects_per_source]:
            selected.append((source, uid, path))
    return selected


def audit_one(
    blender_path: Path,
    source: str,
    uid: str,
    mesh_path: Path,
    canonical_margin: float,
    bounds_tolerance: float,
) -> dict:
    meshes = load_meshes(str(mesh_path))
    mesh = trimesh.util.concatenate(meshes)
    _, center, scale = normalize_vertices(
        np.asarray(mesh.vertices, dtype=np.float32),
        canonical_margin,
    )
    vertices_render, _, _, _ = build_render_buffers(
        meshes,
        center,
        scale,
        canonical_margin,
    )
    expected_minimum = vertices_render.min(axis=0).astype(float).tolist()
    expected_maximum = vertices_render.max(axis=0).astype(float).tolist()

    with tempfile.TemporaryDirectory(
        prefix="mixed_mesh_blender_frame_audit_"
    ) as temporary:
        root = Path(temporary)
        metadata_path = root / "metadata.json"
        request_path = root / "request.json"
        request = {
            "glb_path": str(mesh_path),
            "metadata_path": str(metadata_path),
            "center": center.astype(float).tolist(),
            "scale": float(scale),
            "margin": float(canonical_margin),
            "bounds_only": True,
            "expected_normalized_scene_bounds": {
                "minimum": expected_minimum,
                "maximum": expected_maximum,
            },
            "bounds_tolerance": float(bounds_tolerance),
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")
        command = [
            str(blender_path),
            "-b",
            "--python",
            str(BLENDER_RENDER_SCRIPT),
            "--",
            str(request_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else None
        )
        if metadata is None:
            tail = "\n".join((completed.stdout or "").splitlines()[-40:])
            raise RuntimeError(
                f"Blender wrote no source-frame metadata; exit={completed.returncode}\n{tail}"
            )
        bounds_audit = validate_blender_bounds(
            metadata,
            vertices_render,
            bounds_tolerance,
        )
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout or "").splitlines()[-40:])
            raise RuntimeError(
                f"Blender source-frame audit failed; exit={completed.returncode}\n{tail}"
            )
    return {
        "source": source,
        "uid": uid,
        "mesh_path": str(mesh_path),
        "mesh_sha256": sha256_file(mesh_path),
        "bounds": bounds_audit,
    }


def main() -> None:
    args = parse_args()
    if args.objects_per_source <= 0:
        raise ValueError("--objects_per_source must be positive")
    if args.bounds_tolerance <= 0:
        raise ValueError("--bounds_tolerance must be positive")
    plan_path = Path(args.source_plan).expanduser().resolve()
    blender_path = Path(args.blender_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not blender_path.is_file():
        raise FileNotFoundError(f"Blender binary is missing: {blender_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"unsupported source plan schema: {plan.get('schema')}")
    sources = [value.strip() for value in args.sources.split(",") if value.strip()]
    if not sources:
        raise ValueError("--sources cannot be empty")
    objects_by_source = {
        source: load_frozen_objects(plan_path, plan, source)
        for source in sources
    }
    requested_uids = [
        value.strip() for value in args.uids.split(",") if value.strip()
    ]
    selected = select_objects(
        objects_by_source,
        requested_uids,
        args.objects_per_source,
    )

    records = []
    failures = []
    for ordinal, (source, uid, mesh) in enumerate(selected, start=1):
        print(
            f"[source_frame_audit] {ordinal}/{len(selected)} "
            f"source={source} uid={uid}",
            flush=True,
        )
        try:
            records.append(
                audit_one(
                    blender_path,
                    source,
                    uid,
                    Path(mesh).expanduser().resolve(),
                    args.canonical_margin,
                    args.bounds_tolerance,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "source": source,
                    "uid": uid,
                    "mesh_path": str(Path(mesh).expanduser().resolve()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": bool(len(records) == len(selected) and not failures),
        "source_plan": str(plan_path),
        "source_plan_sha256": sha256_file(plan_path),
        "blender_path": str(blender_path),
        "blender_sha256": sha256_file(blender_path),
        "blender_script": str(BLENDER_RENDER_SCRIPT),
        "blender_script_sha256": sha256_file(BLENDER_RENDER_SCRIPT),
        "canonical_margin": float(args.canonical_margin),
        "bounds_tolerance": float(args.bounds_tolerance),
        "selected_count": len(selected),
        "passed_count": len(records),
        "failure_count": len(failures),
        "records": records,
        "failures": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "selected_count": report["selected_count"],
                "passed_count": report["passed_count"],
                "failure_count": report["failure_count"],
                "output": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if report["passed"] else 3)


if __name__ == "__main__":
    main()
