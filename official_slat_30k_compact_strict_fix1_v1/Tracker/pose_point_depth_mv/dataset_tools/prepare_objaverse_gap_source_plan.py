#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SOURCE_SCHEMA = "tracker.mixed_mesh10k_sources.v1"
REPORT_FORMAT = "pose_point_depth_mv.objaverse_gap_source_plan.v1"
COMPLETE_MARKER_FORMAT = "tracker.mixed_multiview_render_shard_complete.v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_path(root: Path, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def shard_index(uid: str, count: int) -> int:
    digest = hashlib.sha256(uid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % count


def selection_key(seed: int, uid: str) -> str:
    return hashlib.sha256(
        f"{seed}\0objaverse_gap_candidate\0{uid}".encode("utf-8")
    ).hexdigest()


def manifest_source_paths(paths: Iterable[Path]) -> set[str]:
    output: set[str] = set()
    for path in paths:
        payload = load_json(path)
        for row in [*payload.get("samples", []), *payload.get("failures", [])]:
            value = str(row.get("source_glb", ""))
            if value:
                output.add(source_path(path.parent, value))
    return output


def completed_render_manifests(roots: Iterable[Path]) -> list[Path]:
    manifests = []
    for root in roots:
        for marker_path in sorted(root.glob("*/shard_*/_WORKER_COMPLETE.json")):
            marker = load_json(marker_path)
            if marker.get("schema") != COMPLETE_MARKER_FORMAT:
                raise ValueError(f"unsupported complete marker: {marker_path}")
            manifest = marker_path.parent / "manifest.json"
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            if marker.get("render_manifest_sha256") != sha256_file(manifest):
                raise ValueError(f"completed render manifest changed: {manifest}")
            manifests.append(manifest.resolve())
    return manifests


def select_gap_sources(
    sources: dict[str, str],
    audit_by_source: dict[str, dict[str, Any]],
    excluded_sources: set[str],
    *,
    allowed_tiers: set[str],
    seed: int,
    shard_count: int,
    objects_per_shard: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    pools: list[list[tuple[str, str]]] = [[] for _ in range(shard_count)]
    excluded = Counter()
    for uid, raw_mesh in sources.items():
        mesh = str(Path(raw_mesh).expanduser().resolve())
        if mesh in excluded_sources:
            excluded["historical_or_rendered_source_mesh"] += 1
            continue
        audit = audit_by_source.get(mesh)
        if audit is None:
            excluded["missing_existing_mesh_audit"] += 1
            continue
        if audit.get("mesh_audit", {}).get("mesh_valid") is not True:
            excluded["hard_invalid_mesh"] += 1
            continue
        if str(audit.get("auto_tier", "")) not in allowed_tiers:
            excluded["scene_risk_auto_tier"] += 1
            continue
        if not Path(mesh).is_file() or Path(mesh).stat().st_size <= 0:
            excluded["missing_source_mesh"] += 1
            continue
        pools[shard_index(uid, shard_count)].append((uid, mesh))

    selected: dict[str, str] = {}
    pool_counts = []
    for index, pool in enumerate(pools):
        ranked = sorted(pool, key=lambda row: selection_key(seed, row[0]))
        if len(ranked) < objects_per_shard:
            raise RuntimeError(
                f"shard {index} has only {len(ranked)} eligible candidates, "
                f"below requested {objects_per_shard}"
            )
        chosen = ranked[:objects_per_shard]
        selected.update(chosen)
        pool_counts.append(len(ranked))
    expected = shard_count * objects_per_shard
    if len(selected) != expected:
        raise RuntimeError(f"selected {len(selected)} objects, expected {expected}")
    return selected, {
        "eligible_pool_count": sum(pool_counts),
        "eligible_pool_count_by_shard": pool_counts,
        "excluded_counts": dict(sorted(excluded.items())),
        "selected_count": len(selected),
        "selected_count_by_shard": [objects_per_shard] * shard_count,
    }


def write_source_plan(
    output_dir: Path,
    selected: dict[str, str],
    *,
    shard_count: int,
    report: dict[str, Any],
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"immutable output exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_dir.parent)
    )
    try:
        manifest = staging / "objaverse_meshes.json"
        write_json(manifest, dict(sorted(selected.items())))
        shard_reports = []
        for index in range(shard_count):
            shard = {
                uid: mesh
                for uid, mesh in sorted(selected.items())
                if shard_index(uid, shard_count) == index
            }
            relative = Path("objaverse_shards") / f"shard_{index:03d}.json"
            path = staging / relative
            write_json(path, shard)
            shard_reports.append(
                {
                    "index": index,
                    "path": str(relative),
                    "object_count": len(shard),
                    "uid_sha256": sha256_json(sorted(shard)),
                    "manifest_sha256": sha256_file(path),
                }
            )
        plan = {
            "schema": SOURCE_SCHEMA,
            "sources": ["objaverse"],
            "output_dir": str(output_dir),
            "shard_policy": "sha256(uid) mod shard_count",
            "objaverse": {
                "object_count": len(selected),
                "manifest": manifest.name,
                "manifest_sha256": sha256_file(manifest),
                "shard_count": shard_count,
                "shards": shard_reports,
            },
        }
        write_json(staging / "build_plan.json", plan)
        write_json(
            staging / "report.json",
            {
                **report,
                "format": REPORT_FORMAT,
                "status": "complete",
                "output_dir": str(output_dir),
                "build_plan_sha256": sha256_file(staging / "build_plan.json"),
            },
        )
        staging.rename(output_dir)
    except Exception:
        # Keep failed staging data for inspection; never replace an output silently.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_plan", required=True)
    parser.add_argument("--mesh_audit_objects", required=True)
    parser.add_argument("--exclude_manifest", action="append", default=[])
    parser.add_argument("--exclude_render_root", action="append", default=[])
    parser.add_argument("--allowed_auto_tier", action="append", default=[])
    parser.add_argument("--selection_seed", type=int, default=20260730)
    parser.add_argument("--shard_count", type=int, default=16)
    parser.add_argument("--objects_per_shard", type=int, default=18)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    source_plan_path = Path(args.source_plan).expanduser().resolve()
    source_plan = load_json(source_plan_path)
    if source_plan.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"unsupported source plan: {source_plan_path}")
    objaverse = source_plan["objaverse"]
    source_manifest = source_plan_path.parent / str(objaverse["manifest"])
    if sha256_file(source_manifest) != objaverse["manifest_sha256"]:
        raise ValueError(f"source manifest changed: {source_manifest}")
    sources = load_json(source_manifest)

    audit_path = Path(args.mesh_audit_objects).expanduser().resolve()
    audit_rows = load_json(audit_path)
    audit_by_source = {
        str(Path(row["source_glb"]).expanduser().resolve()): row for row in audit_rows
    }
    if len(audit_by_source) != len(audit_rows):
        raise ValueError("mesh audit contains duplicate source paths")

    exclusion_manifests = [
        Path(value).expanduser().resolve() for value in args.exclude_manifest
    ]
    render_roots = [
        Path(value).expanduser().resolve() for value in args.exclude_render_root
    ]
    render_manifests = completed_render_manifests(render_roots)
    excluded_sources = manifest_source_paths([*exclusion_manifests, *render_manifests])
    allowed_tiers = set(args.allowed_auto_tier or ["A", "B"])
    selected, selection = select_gap_sources(
        sources,
        audit_by_source,
        excluded_sources,
        allowed_tiers=allowed_tiers,
        seed=int(args.selection_seed),
        shard_count=int(args.shard_count),
        objects_per_shard=int(args.objects_per_shard),
    )
    write_source_plan(
        Path(args.output_dir).expanduser().resolve(),
        selected,
        shard_count=int(args.shard_count),
        report={
            "selection": selection,
            "selection_seed": int(args.selection_seed),
            "allowed_auto_tiers": sorted(allowed_tiers),
            "source_plan": {
                "path": str(source_plan_path),
                "sha256": sha256_file(source_plan_path),
            },
            "source_manifest": {
                "path": str(source_manifest.resolve()),
                "sha256": sha256_file(source_manifest),
                "object_count": len(sources),
            },
            "mesh_audit": {
                "path": str(audit_path),
                "sha256": sha256_file(audit_path),
                "object_count": len(audit_rows),
            },
            "exclusion_manifest_count": len(exclusion_manifests),
            "completed_render_manifest_count": len(render_manifests),
            "excluded_source_mesh_count": len(excluded_sources),
        },
    )
    print(json.dumps(selection, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
