#!/usr/bin/env python3
"""Freeze a leakage-safe, stratified Objaverse source plan near 5K objects.

The 29,999-object audit is a source-mesh inventory, not a rendered training
manifest.  This tool selects immutable source GLBs for the expensive render
stage.  It deliberately calls the two strata source-stage candidates: render,
mask, camera, projection, texture, and semantic checks still have to run.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FORMAT = "pose_point_depth_mv.objaverse5k_source_freeze.v1"
SOURCE_PLAN_FORMAT = "tracker.mixed_mesh10k_sources.v1"
SEQ_SUFFIX = re.compile(r"_seq\d+$")
UID_KEYS = {"object_uid", "uid"}
SOURCE_KEYS = {"source_glb", "source_mesh", "mesh_path"}
SOURCE_HASH_KEYS = {
    "source_glb_sha256",
    "source_mesh_sha256",
    "mesh_sha256",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def resolve_path(value: str | Path, parent: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = parent / path
    return path.resolve()


def canonical_uid(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "/" in text or "\\" in text or text.endswith(".glb"):
        text = Path(text).stem
    text = SEQ_SUFFIX.sub("", text)
    if text.startswith("objaverse_"):
        text = text[len("objaverse_") :]
    return text.lower()


def walk_mappings(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_mappings(child)


def collect_exclusion_identities(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    uids: set[str] = set()
    sources: set[str] = set()
    hashes: set[str] = set()
    for row in walk_mappings(payload):
        for key in UID_KEYS:
            if key in row:
                value = canonical_uid(row.get(key))
                if value:
                    uids.add(value)
        for key in SOURCE_KEYS:
            raw = row.get(key)
            if isinstance(raw, str) and raw.strip():
                source = resolve_path(raw, path.parent)
                sources.add(str(source))
                uid = canonical_uid(source)
                if uid:
                    uids.add(uid)
        for key in SOURCE_HASH_KEYS:
            value = str(row.get(key, "")).strip().lower()
            if len(value) == 64:
                hashes.add(value)
    return {
        "uids": uids,
        "sources": sources,
        "hashes": hashes,
        "binding": {
            **bind_file(path),
            "uid_identity_count": len(uids),
            "source_identity_count": len(sources),
            "source_hash_count": len(hashes),
        },
    }


def stable_digest(seed: int, label: str, uid: str) -> str:
    return hashlib.sha256(
        f"{int(seed)}\0{label}\0{uid}".encode("utf-8")
    ).hexdigest()


def source_shard(uid: str, shard_count: int) -> int:
    digest = hashlib.sha256(uid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % shard_count


def split_quota(total: int, shard_count: int, seed: int, label: str) -> dict[int, int]:
    base, remainder = divmod(total, shard_count)
    order = sorted(
        range(shard_count),
        key=lambda index: stable_digest(seed, f"{label}:quota", str(index)),
    )
    result = {index: base for index in range(shard_count)}
    for index in order[:remainder]:
        result[index] += 1
    return result


def hard_reasons(row: dict[str, Any], args: argparse.Namespace) -> list[str]:
    audit = row["mesh_audit"]
    flags = row.get("quality_flags", {})
    reasons = []
    if int(audit["geometry_count"]) > int(args.clean_max_geometry_count):
        reasons.append("multipart_or_attachment")
    if float(audit["aspect_ratio"]) > float(args.clean_max_aspect_ratio):
        reasons.append("thin_or_elongated")
    if float(audit["dominant_area_ratio"]) < float(
        args.clean_min_dominant_area_ratio
    ):
        reasons.append("low_dominant_area_ratio")
    if int(audit["face_count"]) >= int(args.complex_face_count):
        reasons.append("complex_high_face_mesh")
    if bool(flags.get("low_texture", False)):
        reasons.append("observed_low_texture")
    return reasons


def compact_object(
    row: dict[str, Any],
    *,
    stage_class: str,
    reasons: list[str],
    shard_count: int,
    seed: int,
) -> dict[str, Any]:
    audit = row["mesh_audit"]
    uid = str(row["object_uid"])
    return {
        "object_uid": uid,
        "canonical_objaverse_uid": canonical_uid(uid),
        "source_glb": str(Path(row["source_glb"]).expanduser().resolve()),
        "source_stage_class": stage_class,
        "hard_reasons": reasons,
        "source_shard": source_shard(uid, shard_count),
        "selection_key": stable_digest(seed, stage_class, uid),
        "source_audit": {
            "auto_tier": row.get("auto_tier"),
            "final_tier": row.get("final_tier"),
            "human_reviewed": bool(row.get("human_reviewed", False)),
            "render_evidence": bool(row.get("render_evidence", False)),
            "accepted_sequences": int(row.get("accepted_sequences", 0)),
            "quality_flags": copy.deepcopy(row.get("quality_flags", {})),
            "geometry_count": int(audit["geometry_count"]),
            "scene_instance_count": int(audit["scene_instance_count"]),
            "vertex_count": int(audit["vertex_count"]),
            "face_count": int(audit["face_count"]),
            "aspect_ratio": float(audit["aspect_ratio"]),
            "dominant_area_ratio": float(audit["dominant_area_ratio"]),
            "source_size": int(audit["source_size"]),
            "source_mtime_ns": int(audit["source_mtime_ns"]),
            "source_sha256": audit.get("source_sha256"),
        },
    }


def round_robin_hard(
    rows: list[dict[str, Any]], count: int, seed: int, label: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        signature = "+".join(row["hard_reasons"])
        groups[signature].append(row)
    for signature, values in groups.items():
        values.sort(
            key=lambda row: stable_digest(
                seed, f"{label}:{signature}", str(row["object_uid"])
            )
        )
    signatures = sorted(
        groups,
        key=lambda value: stable_digest(seed, f"{label}:stratum", value),
    )
    selected: list[dict[str, Any]] = []
    offsets = Counter()
    while signatures and len(selected) < count:
        active = []
        for signature in signatures:
            index = offsets[signature]
            values = groups[signature]
            if index < len(values):
                selected.append(values[index])
                offsets[signature] += 1
                if len(selected) == count:
                    break
            if offsets[signature] < len(values):
                active.append(signature)
        signatures = active
    return selected


def select_by_shard(
    rows: list[dict[str, Any]],
    count: int,
    *,
    shard_count: int,
    seed: int,
    label: str,
    hard: bool,
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("selection count must be nonnegative")
    if len(rows) < count:
        raise RuntimeError(f"{label} pool has {len(rows)} objects, requires {count}")
    by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[int(row["source_shard"])].append(row)
    quotas = split_quota(count, shard_count, seed, label)
    selected: list[dict[str, Any]] = []
    shortage = 0
    for shard in range(shard_count):
        values = by_shard.get(shard, [])
        quota = quotas[shard]
        if hard:
            chosen = round_robin_hard(values, min(quota, len(values)), seed, label)
        else:
            chosen = sorted(
                values,
                key=lambda row: stable_digest(
                    seed, f"{label}:shard:{shard}", str(row["object_uid"])
                ),
            )[:quota]
        selected.extend(chosen)
        shortage += quota - len(chosen)
    if shortage:
        selected_uids = {str(row["object_uid"]) for row in selected}
        remainder = [
            row for row in rows if str(row["object_uid"]) not in selected_uids
        ]
        if hard:
            selected.extend(round_robin_hard(remainder, shortage, seed, f"{label}:fill"))
        else:
            selected.extend(
                sorted(
                    remainder,
                    key=lambda row: stable_digest(
                        seed, f"{label}:fill", str(row["object_uid"])
                    ),
                )[:shortage]
            )
    if len(selected) != count:
        raise RuntimeError(f"{label} selected {len(selected)} objects, expected {count}")
    return selected


def make_source_plan(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    shard_count: int,
    prefix: str,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    mapping = {
        str(row["object_uid"]): str(row["source_glb"])
        for row in sorted(rows, key=lambda value: str(value["object_uid"]))
    }
    manifest_name = f"{prefix}objaverse_meshes.json"
    shard_dir = f"{prefix}objaverse_shards"
    write_json(root / manifest_name, mapping)
    shard_rows = []
    for index in range(shard_count):
        shard = {
            uid: source
            for uid, source in mapping.items()
            if source_shard(uid, shard_count) == index
        }
        relative = Path(shard_dir) / f"shard_{index:03d}.json"
        write_json(root / relative, shard)
        shard_rows.append(
            {
                "index": index,
                "path": str(relative),
                "object_count": len(shard),
                "uid_sha256": canonical_json_sha256(sorted(shard)),
                "manifest_sha256": file_sha256(root / relative),
            }
        )
    plan = {
        "schema": SOURCE_PLAN_FORMAT,
        "sources": ["objaverse"],
        "output_dir": str(root.resolve()),
        "shard_policy": "sha256(uid) mod shard_count",
        "objaverse5k_source_freeze": metadata,
        "objaverse": {
            "object_count": len(mapping),
            "manifest": manifest_name,
            "manifest_sha256": file_sha256(root / manifest_name),
            "shard_count": shard_count,
            "shards": shard_rows,
        },
    }
    return plan, mapping


def validate_source_binding(row: dict[str, Any]) -> str | None:
    source = Path(row["source_glb"]).expanduser()
    audit = row["mesh_audit"]
    if not source.is_file():
        return "source_missing"
    stat = source.stat()
    if stat.st_size <= 0:
        return "source_empty"
    if stat.st_size != int(audit["source_size"]):
        return "source_size_changed"
    if stat.st_mtime_ns != int(audit["source_mtime_ns"]):
        return "source_mtime_changed"
    return None


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    audit_root = Path(args.audit_root).expanduser().resolve()
    objects_path = audit_root / "objects.json"
    audit_report_path = audit_root / "report.json"
    if not objects_path.is_file() or not audit_report_path.is_file():
        raise FileNotFoundError(f"audit root lacks objects.json/report.json: {audit_root}")
    audit_report = load_json(audit_report_path)
    objects = load_json(objects_path)
    if audit_report.get("passed") is not True or not isinstance(objects, list):
        raise ValueError("source audit is not a passed object inventory")
    expected_count = int(audit_report.get("summary", {}).get("object_count", -1))
    if expected_count != len(objects):
        raise RuntimeError(
            f"source audit object count changed: report={expected_count} actual={len(objects)}"
        )

    exclusion_paths = [Path(value).expanduser().resolve() for value in args.exclude_manifest]
    if not exclusion_paths:
        raise ValueError("at least one --exclude_manifest is required")
    exclusion_sets = [collect_exclusion_identities(path) for path in exclusion_paths]
    excluded_uids = set().union(*(row["uids"] for row in exclusion_sets))
    excluded_sources = set().union(*(row["sources"] for row in exclusion_sets))
    excluded_hashes = set().union(*(row["hashes"] for row in exclusion_sets))

    hard_count = int(round(int(args.target_objects) * float(args.hard_fraction)))
    clean_count = int(args.target_objects) - hard_count
    reserve_hard = int(round(int(args.reserve_objects) * float(args.hard_fraction)))
    reserve_clean = int(args.reserve_objects) - reserve_hard
    config = {
        "target_objects": int(args.target_objects),
        "clean_objects": clean_count,
        "hard_objects": hard_count,
        "hard_fraction": float(args.hard_fraction),
        "reserve_objects": int(args.reserve_objects),
        "reserve_clean_objects": reserve_clean,
        "reserve_hard_objects": reserve_hard,
        "seed": int(args.seed),
        "shard_count": int(args.shard_count),
        "clean_max_geometry_count": int(args.clean_max_geometry_count),
        "clean_max_aspect_ratio": float(args.clean_max_aspect_ratio),
        "clean_min_dominant_area_ratio": float(
            args.clean_min_dominant_area_ratio
        ),
        "complex_face_count": int(args.complex_face_count),
        "semantic_policy": (
            "A/B valid source-mesh candidates only; C scene-like and R invalid "
            "objects are rejected; source-stage labels require render/QC follow-up"
        ),
    }
    if config["target_objects"] <= 0 or config["reserve_objects"] < 0:
        raise ValueError("target must be positive and reserve nonnegative")
    if not 0.0 <= config["hard_fraction"] <= 1.0:
        raise ValueError("--hard_fraction must be within [0, 1]")
    if config["shard_count"] <= 0:
        raise ValueError("--shard_count must be positive")

    source_bindings = {
        "audit_report": bind_file(audit_report_path),
        "audit_objects": bind_file(objects_path),
        "exclusion_manifests": [row["binding"] for row in exclusion_sets],
    }
    code_binding = bind_file(Path(__file__))
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        report_path = output_dir / "report.json"
        if not report_path.is_file():
            raise FileExistsError(f"partial output exists: {output_dir}")
        existing = load_json(report_path)
        if (
            existing.get("passed") is not True
            or existing.get("config") != config
            or existing.get("source_bindings") != source_bindings
            or existing.get("code_binding") != code_binding
        ):
            raise ValueError("existing immutable freeze differs from this invocation")
        for relative, expected in existing.get("artifact_sha256", {}).items():
            path = output_dir / relative
            if not path.is_file() or file_sha256(path) != expected:
                raise ValueError(f"frozen artifact changed: {path}")
        print(json.dumps({"reused": True, **existing["summary"]}, indent=2))
        return existing

    pools: dict[str, list[dict[str, Any]]] = {"clean": [], "hard": []}
    rejection_counts = Counter()
    leakage_rows = []
    seen_uids: set[str] = set()
    seen_sources: set[str] = set()
    for raw in objects:
        uid = str(raw.get("object_uid", ""))
        canonical = canonical_uid(uid)
        source = str(Path(str(raw.get("source_glb", ""))).expanduser().resolve())
        audit = raw.get("mesh_audit", {})
        if not uid or not canonical or not source:
            rejection_counts["missing_identity"] += 1
            continue
        if canonical in seen_uids or source in seen_sources:
            rejection_counts["duplicate_identity"] += 1
            continue
        seen_uids.add(canonical)
        seen_sources.add(source)
        if raw.get("final_tier") not in ("A", "B"):
            rejection_counts[f"tier_{raw.get('final_tier', 'missing')}"] += 1
            continue
        if audit.get("mesh_valid") is not True:
            rejection_counts["mesh_invalid"] += 1
            continue
        flags = raw.get("quality_flags", {})
        if bool(flags.get("flat_gray_blob", False)):
            rejection_counts["flat_gray_blob"] += 1
            continue
        if bool(flags.get("low_projection_support", False)):
            rejection_counts["low_projection_support"] += 1
            continue
        matched = []
        source_hash = str(audit.get("source_sha256") or "").lower()
        if canonical in excluded_uids:
            matched.append("object_uid")
        if source in excluded_sources:
            matched.append("source_glb")
        if source_hash and source_hash in excluded_hashes:
            matched.append("source_sha256")
        if matched:
            rejection_counts["historical_or_holdout_overlap"] += 1
            leakage_rows.append(
                {
                    "object_uid": uid,
                    "source_glb": source,
                    "matched_identities": matched,
                }
            )
            continue
        binding_error = validate_source_binding(raw)
        if binding_error:
            rejection_counts[binding_error] += 1
            continue
        reasons = hard_reasons(raw, args)
        stage_class = "source_hard_candidate" if reasons else "source_clean_candidate"
        compact = compact_object(
            raw,
            stage_class=stage_class,
            reasons=reasons,
            shard_count=config["shard_count"],
            seed=config["seed"],
        )
        pools["hard" if reasons else "clean"].append(compact)

    clean_selected = select_by_shard(
        pools["clean"],
        clean_count,
        shard_count=config["shard_count"],
        seed=config["seed"],
        label="primary_clean",
        hard=False,
    )
    hard_selected = select_by_shard(
        pools["hard"],
        hard_count,
        shard_count=config["shard_count"],
        seed=config["seed"],
        label="primary_hard",
        hard=True,
    )
    selected = clean_selected + hard_selected
    selected_uids = {str(row["object_uid"]) for row in selected}
    clean_remaining = [
        row for row in pools["clean"] if str(row["object_uid"]) not in selected_uids
    ]
    hard_remaining = [
        row for row in pools["hard"] if str(row["object_uid"]) not in selected_uids
    ]
    clean_reserve = select_by_shard(
        clean_remaining,
        reserve_clean,
        shard_count=config["shard_count"],
        seed=config["seed"],
        label="reserve_clean",
        hard=False,
    )
    hard_reserve = select_by_shard(
        hard_remaining,
        reserve_hard,
        shard_count=config["shard_count"],
        seed=config["seed"],
        label="reserve_hard",
        hard=True,
    )
    reserve = clean_reserve + hard_reserve

    staging = Path(f"{output_dir}.staging")
    if staging.exists():
        raise FileExistsError(f"stale staging output exists: {staging}")
    staging.mkdir(parents=True)
    try:
        selection_metadata = {
            "format": FORMAT,
            "seed": config["seed"],
            "clean_count": clean_count,
            "hard_count": hard_count,
            "requires_post_render_qc": True,
            "requires_semantic_audit": True,
        }
        plan, mapping = make_source_plan(
            staging,
            selected,
            shard_count=config["shard_count"],
            prefix="",
            metadata=selection_metadata,
        )
        reserve_plan, reserve_mapping = make_source_plan(
            staging,
            reserve,
            shard_count=config["shard_count"],
            prefix="reserve_",
            metadata={**selection_metadata, "reserve_only": True},
        )
        plan["output_dir"] = str(output_dir)
        reserve_plan["output_dir"] = str(output_dir)
        write_json(staging / "build_plan.json", plan)
        write_json(staging / "reserve_build_plan.json", reserve_plan)
        selection = {
            "format": FORMAT,
            "created_at_utc": utc_now(),
            "status": "source_frozen_pending_render_qc",
            "training_ready": False,
            "config": config,
            "objects": sorted(selected, key=lambda row: str(row["object_uid"])),
        }
        write_json(staging / "selection.json", selection)
        write_json(
            staging / "reserves.json",
            {
                "format": FORMAT,
                "status": "reserve_not_scheduled_for_render",
                "objects": sorted(reserve, key=lambda row: str(row["object_uid"])),
            },
        )
        write_json(
            staging / "excluded_holdout.json",
            {
                "format": FORMAT,
                "objects": sorted(leakage_rows, key=lambda row: row["object_uid"]),
                "exclusion_bindings": source_bindings["exclusion_manifests"],
            },
        )
        audit_count = min(int(args.audit_sample_count), len(selected))
        audit_rows = sorted(
            selected,
            key=lambda row: stable_digest(
                config["seed"], "audit_sample", str(row["object_uid"])
            ),
        )[:audit_count]
        write_json(
            staging / "audit_samples.json",
            {
                "format": FORMAT,
                "policy": "stable SHA256 sample across both source-stage strata",
                "objects": audit_rows,
            },
        )

        shard_counts = Counter(int(row["source_shard"]) for row in selected)
        class_counts = Counter(str(row["source_stage_class"]) for row in selected)
        hard_reason_counts = Counter(
            reason for row in hard_selected for reason in row["hard_reasons"]
        )
        summary = {
            "source_audit_object_count": len(objects),
            "eligible_clean_pool": len(pools["clean"]),
            "eligible_hard_pool": len(pools["hard"]),
            "selected_object_count": len(mapping),
            "selected_class_counts": dict(sorted(class_counts.items())),
            "reserve_object_count": len(reserve_mapping),
            "excluded_overlap_object_count": len(leakage_rows),
            "selected_shard_counts": {
                str(index): shard_counts[index]
                for index in range(config["shard_count"])
            },
            "hard_reason_counts": dict(sorted(hard_reason_counts.items())),
            "other_rejection_counts": dict(sorted(rejection_counts.items())),
        }
        hard_guards = {
            "source_audit_passed": True,
            "exact_primary_count": len(mapping) == config["target_objects"],
            "exact_clean_count": len(clean_selected) == clean_count,
            "exact_hard_count": len(hard_selected) == hard_count,
            "exact_reserve_count": len(reserve_mapping) == config["reserve_objects"],
            "selected_uids_unique": len(mapping) == len(selected_uids),
            "selected_sources_unique": len({row["source_glb"] for row in selected})
            == len(selected),
            "primary_reserve_disjoint": not (
                selected_uids & {str(row["object_uid"]) for row in reserve}
            ),
            "historical_and_holdout_overlap_excluded": all(
                canonical_uid(uid) not in excluded_uids for uid in mapping
            ),
            "only_a_b_mesh_valid_sources": True,
            "source_assets_bound_by_size_and_mtime": True,
            "render_qc_still_required": True,
            "semantic_audit_still_required": True,
        }
        if not all(hard_guards.values()):
            raise RuntimeError(f"source freeze hard guard failed: {hard_guards}")

        artifact_names = [
            "selection.json",
            "reserves.json",
            "excluded_holdout.json",
            "audit_samples.json",
            "objaverse_meshes.json",
            "reserve_objaverse_meshes.json",
            "build_plan.json",
            "reserve_build_plan.json",
            *[
                f"objaverse_shards/shard_{index:03d}.json"
                for index in range(config["shard_count"])
            ],
            *[
                f"reserve_objaverse_shards/shard_{index:03d}.json"
                for index in range(config["shard_count"])
            ],
        ]
        artifact_sha256 = {
            name: file_sha256(staging / name) for name in artifact_names
        }
        report = {
            "format": FORMAT,
            "passed": True,
            "status": "source_frozen_pending_render_qc",
            "training_ready": False,
            "output_dir": str(output_dir),
            "summary": summary,
            "hard_guards": hard_guards,
            "config": config,
            "source_bindings": source_bindings,
            "code_binding": code_binding,
            "artifact_sha256": artifact_sha256,
            "next": (
                "render build_plan.json, then enforce per-view render/mask/camera/"
                "projection/texture QC and semantic review before training"
            ),
        }
        write_json(staging / "report.json", report)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(json.dumps({"reused": False, **summary}, indent=2, ensure_ascii=False))
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_root", required=True)
    parser.add_argument("--exclude_manifest", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_objects", type=int, default=5000)
    parser.add_argument("--hard_fraction", type=float, default=0.25)
    parser.add_argument("--reserve_objects", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--shard_count", type=int, default=16)
    parser.add_argument("--audit_sample_count", type=int, default=128)
    parser.add_argument("--clean_max_geometry_count", type=int, default=4)
    parser.add_argument("--clean_max_aspect_ratio", type=float, default=6.0)
    parser.add_argument("--clean_min_dominant_area_ratio", type=float, default=0.55)
    parser.add_argument("--complex_face_count", type=int, default=450000)
    return parser


def main() -> None:
    freeze(make_parser().parse_args())


if __name__ == "__main__":
    main()
