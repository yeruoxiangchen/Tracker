#!/usr/bin/env python3
"""Aggregate a completed mixed-render pilot without selecting model results."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

try:
    from .render_failure_taxonomy import (
        FAILURE_TAXONOMY_SCHEMA,
        INFRASTRUCTURE_FAILURES,
        canonical_failure_class,
    )
except ImportError:
    from render_failure_taxonomy import (
        FAILURE_TAXONOMY_SCHEMA,
        INFRASTRUCTURE_FAILURES,
        canonical_failure_class,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render_root", required=True)
    parser.add_argument("--source", choices=("objaverse", "omni"), required=True)
    parser.add_argument("--expected_shards", type=int, required=True)
    parser.add_argument("--sequences_per_object", type=int, default=2)
    parser.add_argument("--max_low_texture_ratio", type=float, default=0.22)
    parser.add_argument("--target_objects", type=int, default=6000)
    parser.add_argument("--reserve_fraction", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def max_low_texture_objects(
    normal_objects: int,
    low_only_objects: int,
    ratio: float,
) -> int:
    if ratio >= 1.0:
        return low_only_objects
    if ratio <= 0.0 or normal_objects <= 0:
        return 0
    allowed = math.floor(ratio * normal_objects / (1.0 - ratio))
    return min(low_only_objects, max(0, allowed))


def main() -> None:
    args = parse_args()
    if args.expected_shards <= 0 or args.sequences_per_object <= 0:
        raise ValueError("expected shards and sequences per object must be positive")
    if not 0.0 <= args.max_low_texture_ratio <= 1.0:
        raise ValueError("--max_low_texture_ratio must be in [0, 1]")
    if args.target_objects <= 0 or args.reserve_fraction < 0:
        raise ValueError("target objects must be positive and reserve nonnegative")

    source_root = Path(args.render_root).expanduser().resolve() / args.source
    missing_markers = []
    manifest_bindings = []
    metadata_reference = None
    failure_counts: Counter[str] = Counter()
    raw_failure_counts: Counter[str] = Counter()
    canonicalized_failure_count = 0
    by_object: dict[str, list[dict]] = defaultdict(list)
    attempts_by_object: Counter[str] = Counter()
    bounds_values: list[float] = []
    accepted_sequences = 0
    attempted_sequences = 0

    for shard_index in range(args.expected_shards):
        shard_root = source_root / f"shard_{shard_index:03d}"
        marker_path = shard_root / "_WORKER_COMPLETE.json"
        manifest_path = shard_root / "manifest.json"
        if not marker_path.is_file() or not manifest_path.is_file():
            missing_markers.append(shard_index)
            continue
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest_sha256 = sha256_file(manifest_path)
        if marker.get("render_manifest_sha256") != manifest_sha256:
            raise RuntimeError(f"shard {shard_index}: marker/manifest SHA mismatch")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = {
            "build_config": payload.get("build_config"),
            "code_bindings": payload.get("code_bindings"),
            "quality_policy": payload.get("quality_policy"),
        }
        if metadata_reference is None:
            metadata_reference = metadata
        elif metadata != metadata_reference:
            raise RuntimeError(f"shard {shard_index}: incompatible build metadata")

        samples = payload.get("samples", [])
        failures = payload.get("failures", [])
        accepted_sequences += len(samples)
        attempted_sequences += len(samples) + len(failures)
        for sample in samples:
            object_uid = str(sample["object_uid"])
            attempts_by_object[object_uid] += 1
            by_object[object_uid].append(sample)
            value = sample.get("renderer_audit", {}).get(
                "normalized_scene_bounds_max_abs"
            )
            if value is not None:
                bounds_values.append(float(value))
        for failure in failures:
            object_uid = str(failure.get("object_uid", failure.get("uid", "")))
            attempts_by_object[object_uid] += 1
            raw_class = str(failure.get("failure_class", "missing"))
            canonical_class = canonical_failure_class(failure)
            raw_failure_counts[raw_class] += 1
            failure_counts[canonical_class] += 1
            canonicalized_failure_count += int(canonical_class != raw_class)
        manifest_bindings.append(
            {
                "shard_index": shard_index,
                "path": str(manifest_path),
                "sha256": manifest_sha256,
            }
        )

    if missing_markers:
        report = {
            "passed": False,
            "reason": "pilot shards are incomplete",
            "missing_shards": missing_markers,
            "completed_shards": args.expected_shards - len(missing_markers),
            "expected_shards": args.expected_shards,
        }
    else:
        wrong_attempt_counts = {
            uid: count
            for uid, count in attempts_by_object.items()
            if count != args.sequences_per_object
        }
        if wrong_attempt_counts:
            raise RuntimeError(
                "per-object sequence count differs: "
                f"{list(wrong_attempt_counts.items())[:5]}"
            )
        attempted_objects = len(attempts_by_object)
        successful_objects = len(by_object)
        normal_objects = sum(
            any(
                not bool(sample.get("quality_flags", {}).get("low_texture", False))
                for sample in samples
            )
            for samples in by_object.values()
        )
        low_only_objects = successful_objects - normal_objects
        admitted_low_objects = max_low_texture_objects(
            normal_objects,
            low_only_objects,
            args.max_low_texture_ratio,
        )
        admitted_objects = normal_objects + admitted_low_objects
        admitted_yield = admitted_objects / max(attempted_objects, 1)
        suggested_candidates = (
            math.ceil(
                args.target_objects
                * (1.0 + args.reserve_fraction)
                / admitted_yield
            )
            if admitted_yield > 0
            else None
        )
        infrastructure_count = sum(
            failure_counts[name] for name in INFRASTRUCTURE_FAILURES
        )
        report = {
            "passed": infrastructure_count == 0,
            "source": args.source,
            "completed_shards": args.expected_shards,
            "expected_shards": args.expected_shards,
            "attempted_objects": attempted_objects,
            "attempted_sequences": attempted_sequences,
            "accepted_sequences": accepted_sequences,
            "sequence_acceptance_rate": (
                accepted_sequences / max(attempted_sequences, 1)
            ),
            "objects_with_any_accepted_sequence": successful_objects,
            "object_acceptance_rate": (
                successful_objects / max(attempted_objects, 1)
            ),
            "normal_texture_objects": normal_objects,
            "low_texture_only_objects": low_only_objects,
            "max_low_texture_ratio": args.max_low_texture_ratio,
            "admitted_low_texture_objects": admitted_low_objects,
            "estimated_admitted_objects": admitted_objects,
            "estimated_admitted_object_yield": admitted_yield,
            "target_objects": args.target_objects,
            "reserve_fraction": args.reserve_fraction,
            "suggested_candidate_objects": suggested_candidates,
            "failure_counts": dict(sorted(failure_counts.items())),
            "raw_failure_counts": dict(sorted(raw_failure_counts.items())),
            "canonicalized_failure_count": canonicalized_failure_count,
            "failure_taxonomy": FAILURE_TAXONOMY_SCHEMA,
            "infrastructure_failure_count": infrastructure_count,
            "renderer_bounds_max_abs": quantiles(bounds_values),
            "build_metadata": metadata_reference,
            "manifest_bindings": manifest_bindings,
        }

    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report.get("passed") else 3)


if __name__ == "__main__":
    main()
