"""Safely seed a new endpoint worker from a failed worker's SS coordinates.

Only per-object/per-seed Native-SS coordinate artifacts are checkpoint
independent.  A failed worker report, run identity, target Mesh, SLat output or
surface metric is never reused by this utility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "official_ss_with_vggt_perf_v1.partial_ss_coord_reuse.v1"
SOURCE_FORMAT = "official_ss_with_vggt_perf_v1.predicted_ss_slat_endpoint.v1.worker"
TARGET_FORMAT = "official_ss_with_vggt_perf_v1.predicted_ss_slat_endpoint.v2.worker"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _pair_id(object_uid: str, seed: int) -> str:
    return hashlib.sha256(f"{object_uid}|{int(seed)}".encode("utf-8")).hexdigest()[:24]


def _validate_identity(source: Path, target: Path) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    source_identity = _load_json(source / "run_identity.json")
    target_identity = _load_json(target / "run_identity.json")
    if source_identity.get("format") != SOURCE_FORMAT:
        raise RuntimeError("partial reuse source is not the failed endpoint-v1 worker")
    if target_identity.get("format") != TARGET_FORMAT:
        raise RuntimeError("partial reuse target is not the endpoint-v2 worker")
    source_stable = dict(source_identity)
    target_stable = dict(target_identity)
    source_stable.pop("format", None)
    target_stable.pop("format", None)
    if source_stable != target_stable:
        raise RuntimeError("partial reuse source/target run identities differ")
    object_uids = target_identity.get("object_uids")
    seeds = target_identity.get("joint_seeds")
    if (
        not isinstance(object_uids, list)
        or not object_uids
        or len(object_uids) != len(set(str(value) for value in object_uids))
        or not isinstance(seeds, list)
        or not seeds
        or len(seeds) != len(set(int(value) for value in seeds))
    ):
        raise RuntimeError("target worker object/seed identity is invalid")
    expected = {
        (str(object_uid), int(seed))
        for object_uid in object_uids
        for seed in seeds
    }
    return target_identity, expected


def _validate_pair(
    audit_path: Path,
    npz_path: Path,
    *,
    expected: set[tuple[str, int]],
) -> dict[str, Any]:
    if (
        not audit_path.is_file()
        or audit_path.is_symlink()
        or not npz_path.is_file()
        or npz_path.is_symlink()
    ):
        raise RuntimeError(f"coordinate pair is not two regular files: {audit_path}")
    audit = _load_json(audit_path)
    key = (str(audit.get("object_uid", "")), int(audit.get("seed", -1)))
    if key not in expected or audit_path.stem != _pair_id(*key):
        raise RuntimeError(f"coordinate pair object/seed identity differs: {audit_path}")
    digest = sha256_file(npz_path)
    if str(audit.get("coords_npz_sha256", "")) != digest:
        raise RuntimeError(f"coordinate pair NPZ SHA256 differs: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != {"stock", "native"}:
            raise RuntimeError(f"coordinate pair NPZ schema differs: {npz_path}")
        stock = np.asarray(payload["stock"])
        native = np.asarray(payload["native"])
    if (
        stock.ndim != 2
        or native.ndim != 2
        or stock.shape[1] not in (3, 4)
        or native.shape[1] not in (3, 4)
        or len(stock) != int(audit.get("stock_count", -1))
        or len(native) != int(audit.get("native_count", -1))
        or audit.get("same_initial_noise") is not True
        or audit.get("passed") is not True
    ):
        raise RuntimeError(f"coordinate pair numerical contract differs: {npz_path}")
    for label, coords in (("stock", stock), ("native", native)):
        xyz = coords[:, -3:]
        if (
            (coords.shape[1] == 4 and np.any(coords[:, 0] != 0))
            or np.any(xyz < 0)
            or np.any(xyz >= 64)
            or len(np.unique(xyz, axis=0)) != len(xyz)
        ):
            raise RuntimeError(
                f"coordinate pair {label} sparse-coordinate contract differs: {npz_path}"
            )
    return {
        "stem": audit_path.stem,
        "object_uid": key[0],
        "seed": key[1],
        "npz_sha256": digest,
        "json_sha256": sha256_file(audit_path),
    }


def _hardlink(source: Path, target: Path) -> None:
    if target.exists():
        raise RuntimeError(f"partial reuse target unexpectedly exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as error:
        raise RuntimeError(
            f"partial reuse requires source/target on one filesystem: {error}"
        ) from error
    if not target.is_file() or sha256_file(target) != sha256_file(source):
        raise RuntimeError(f"partial reuse hard-link verification failed: {target}")


def reuse_partial_ss_coords(source_worker: Path, target_worker: Path) -> dict[str, Any]:
    source_worker = source_worker.expanduser().resolve(strict=True)
    target_worker = target_worker.expanduser().resolve(strict=True)
    if source_worker == target_worker:
        raise ValueError("partial reuse source and target workers must differ")
    target_identity, expected = _validate_identity(source_worker, target_worker)
    source_root = source_worker / "ss_coords"
    target_root = target_worker / "ss_coords"
    target_root.mkdir(parents=True, exist_ok=True)
    manifest = target_worker / "partial_ss_coord_reuse.json"
    if manifest.exists():
        raise RuntimeError(f"partial reuse manifest already exists: {manifest}")

    source_audits = sorted(source_root.glob("*.json"))
    source_npzs = sorted(source_root.glob("*.npz"))
    if not source_audits or {path.stem for path in source_audits} != {
        path.stem for path in source_npzs
    }:
        raise RuntimeError("partial source coordinate JSON/NPZ coverage differs")

    linked: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    skipped_incomplete: list[str] = []
    seen: set[tuple[str, int]] = set()
    for source_audit in source_audits:
        source_npz = source_audit.with_suffix(".npz")
        source_record = _validate_pair(source_audit, source_npz, expected=expected)
        key = (source_record["object_uid"], int(source_record["seed"]))
        if key in seen:
            raise RuntimeError(f"duplicate partial source coordinate pair: {key}")
        seen.add(key)
        target_audit = target_root / source_audit.name
        target_npz = target_root / source_npz.name
        target_presence = (target_audit.exists(), target_npz.exists())
        if target_presence == (True, True):
            target_record = _validate_pair(target_audit, target_npz, expected=expected)
            kept.append(target_record)
            continue
        if target_presence != (False, False):
            # A worker interrupted between its NPZ and JSON writes will safely
            # recompute this pair on resume.  Never combine halves from runs.
            skipped_incomplete.append(source_audit.stem)
            continue
        _hardlink(source_npz, target_npz)
        _hardlink(source_audit, target_audit)
        linked.append(source_record)

    stable = {
        "format": FORMAT,
        "source_worker": str(source_worker),
        "target_worker": str(target_worker),
        "source_format": SOURCE_FORMAT,
        "target_format": TARGET_FORMAT,
        "run_identity_equal_except_format": True,
        "object_start": int(target_identity["object_start"]),
        "object_end": int(target_identity["object_end"]),
        "joint_seeds": [int(value) for value in target_identity["joint_seeds"]],
        "source_valid_pair_count": len(source_audits),
        "linked_pair_count": len(linked),
        "kept_complete_target_pair_count": len(kept),
        "skipped_incomplete_target_pair_count": len(skipped_incomplete),
        "skipped_incomplete_target_stems": skipped_incomplete,
        "linked_pairs": linked,
        "kept_target_pairs": kept,
        "reuse_scope": "native_ss_coordinate_pairs_only",
        "worker_report_reused": False,
        "run_identity_reused": False,
        "target_mesh_reused": False,
        "slat_or_surface_result_reused": False,
    }
    payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **stable}
    temporary = manifest.with_name(f".{manifest.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest)
    return {**stable, "manifest": str(manifest), "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_worker", required=True)
    parser.add_argument("--target_worker", required=True)
    args = parser.parse_args()
    result = reuse_partial_ss_coords(
        Path(args.source_worker),
        Path(args.target_worker),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
