"""Reuse checkpoint-independent endpoint artifacts with fail-closed identity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FORMAT = "official_ss_with_vggt_perf_v1.endpoint_independent_reuse.v1"

_ENDPOINT_IDENTITY_EXTENSION_KEYS = frozenset(
    {
        "ss_cache_manifest",
        "ss_cache_manifest_sha256",
        "endpoint_version",
        "branch_semantics",
        "slat_support_input",
        "gt_support_used_as_slat_input",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload is not an object: {path}")
    return payload


def _validate_source_identity_binding(
    source_report: dict[str, Any], source_identity: dict[str, Any]
) -> dict[str, Any]:
    """Validate the base-worker identity and its endpoint-only annotation.

    The base evaluator writes ``run_identity.json`` before inference and uses it
    for byte-for-byte resume validation.  The with-VGGT endpoint wrapper then
    annotates only ``report.json["run_identity"]`` with six endpoint fields.
    Requiring both dictionaries to be identical therefore rejects valid,
    completed endpoint-v2 workers.  Keep the base identity strict and admit
    only the audited endpoint extension.
    """

    report_identity = source_report.get("run_identity")
    if not isinstance(report_identity, dict):
        raise RuntimeError("source worker report lacks run identity")
    if report_identity == source_identity:
        return report_identity

    base_projection = {
        key: value
        for key, value in report_identity.items()
        if key not in _ENDPOINT_IDENTITY_EXTENSION_KEYS
    }
    if base_projection != source_identity:
        raise RuntimeError("source worker report/base run_identity binding differs")
    extension_keys = set(report_identity) - set(source_identity)
    if extension_keys != _ENDPOINT_IDENTITY_EXTENSION_KEYS:
        raise RuntimeError(
            "source worker endpoint run_identity extension differs: "
            f"actual={sorted(extension_keys)} "
            f"expected={sorted(_ENDPOINT_IDENTITY_EXTENSION_KEYS)}"
        )

    # Import lazily so the lightweight legacy/equality path does not initialize
    # TRELLIS.  The completed endpoint source must carry the exact frozen
    # contract, not merely a matching set of extra identity keys.
    from .ss_slat_endpoint import endpoint_contract

    contract = endpoint_contract()
    if source_report.get("with_vggt_endpoint_contract") != contract:
        raise RuntimeError("source worker endpoint contract differs")
    if report_identity["endpoint_version"] != contract["version"]:
        raise RuntimeError("source worker endpoint identity version differs")
    if report_identity["branch_semantics"] != contract["branches"]:
        raise RuntimeError("source worker endpoint branch semantics differ")
    if report_identity["slat_support_input"] != "predicted_only":
        raise RuntimeError("source worker endpoint support policy differs")
    if report_identity["gt_support_used_as_slat_input"] is not False:
        raise RuntimeError("source worker endpoint GT-support policy differs")

    ss_cache = Path(str(report_identity["ss_cache_manifest"])).resolve(strict=True)
    if report_identity["ss_cache_manifest_sha256"] != sha256_file(ss_cache):
        raise RuntimeError("source worker SS cache manifest SHA256 differs")
    return report_identity


def _link_verified(source: Path, target: Path) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"reuse source must be a regular non-symlink file: {source}")
    source_hash = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise RuntimeError(f"reuse target is not a regular non-symlink file: {target}")
        if sha256_file(target) != source_hash:
            raise RuntimeError(f"existing reuse target content differs: {target}")
        reused_existing = True
    else:
        try:
            os.link(source, target)
        except OSError as error:
            raise RuntimeError(
                f"hard-link reuse failed; source and target must share a filesystem: {error}"
            ) from error
        reused_existing = False
    if sha256_file(target) != source_hash:
        raise RuntimeError(f"linked reuse target hash differs: {target}")
    return {
        "relative_path": str(source),
        "bytes": source.stat().st_size,
        "sha256": source_hash,
    }


def prepare_reuse(
    *,
    source_worker: Path,
    target_worker: Path,
    source_step: int,
    target_step: int,
    source_checkpoint: Path,
    target_checkpoint: Path,
    reuse_target_meshes: bool = True,
) -> dict[str, Any]:
    source_worker = source_worker.resolve(strict=True)
    target_worker = target_worker.resolve()
    if source_worker == target_worker:
        raise ValueError("source and target worker roots must differ")
    source_report_path = source_worker / "report.json"
    source_identity_path = source_worker / "run_identity.json"
    source_report = _load_json(source_report_path)
    source_identity = _load_json(source_identity_path)
    if source_report.get("complete") is not True:
        raise RuntimeError("reuse requires a complete source worker report")
    source_identity = _validate_source_identity_binding(
        source_report, source_identity
    )
    if int(source_identity.get("expected_trained_slat_step", -1)) != int(source_step):
        raise RuntimeError("source worker checkpoint step differs")
    if Path(str(source_identity.get("trained_slat_checkpoint", ""))).resolve() != source_checkpoint.resolve(strict=True):
        raise RuntimeError("source worker checkpoint path differs")
    if str(source_identity.get("trained_slat_checkpoint_sha256", "")) != sha256_file(source_checkpoint):
        raise RuntimeError("source worker checkpoint SHA256 differs")
    object_uids = source_identity.get("object_uids")
    seeds = source_identity.get("joint_seeds")
    if not isinstance(object_uids, list) or not object_uids:
        raise RuntimeError("source worker lacks object UID identity")
    if not isinstance(seeds, list) or not seeds:
        raise RuntimeError("source worker lacks joint seed identity")
    expected_pairs = len(object_uids) * len(seeds)

    coord_root = source_worker / "ss_coords"
    coord_jsons = sorted(coord_root.glob("*.json"))
    coord_npzs = sorted(coord_root.glob("*.npz"))
    if len(coord_jsons) != expected_pairs or len(coord_npzs) != expected_pairs:
        raise RuntimeError(
            "source SS coordinate cache coverage differs: "
            f"json={len(coord_jsons)} npz={len(coord_npzs)} expected={expected_pairs}"
        )
    expected_keys = {(str(uid), int(seed)) for uid in object_uids for seed in seeds}
    observed_keys: set[tuple[str, int]] = set()
    for audit_path in coord_jsons:
        audit = _load_json(audit_path)
        key = (str(audit.get("object_uid", "")), int(audit.get("seed", -1)))
        if key not in expected_keys or key in observed_keys:
            raise RuntimeError(f"source SS coordinate identity differs: {audit_path}")
        observed_keys.add(key)
        npz_path = audit_path.with_suffix(".npz")
        if str(audit.get("coords_npz_sha256", "")) != sha256_file(npz_path):
            raise RuntimeError(f"source SS coordinate SHA256 differs: {npz_path}")
    if observed_keys != expected_keys:
        raise RuntimeError("source SS coordinate object/seed coverage differs")

    target_meshes: list[Path] = []
    if reuse_target_meshes:
        target_mesh_root = source_worker / "target_mesh_cache"
        target_meshes = sorted(target_mesh_root.glob("*.npz"))
        if len(target_meshes) != len(object_uids):
            raise RuntimeError(
                "source target Mesh cache coverage differs: "
                f"actual={len(target_meshes)} expected={len(object_uids)}"
            )
        if {path.stem for path in target_meshes} != {str(uid) for uid in object_uids}:
            raise RuntimeError("source target Mesh UID coverage differs")

    target_worker.mkdir(parents=True, exist_ok=True)
    linked: list[dict[str, Any]] = []
    for source in (*coord_jsons, *coord_npzs, *target_meshes):
        relative = source.relative_to(source_worker)
        record = _link_verified(source, target_worker / relative)
        record["relative_path"] = str(relative)
        linked.append(record)

    stable = {
        "format": FORMAT,
        "source_worker": str(source_worker),
        "source_report": str(source_report_path),
        "source_report_sha256": sha256_file(source_report_path),
        "source_checkpoint": str(source_checkpoint.resolve(strict=True)),
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_step": int(source_step),
        "target_checkpoint": str(target_checkpoint.resolve(strict=True)),
        "target_checkpoint_sha256": sha256_file(target_checkpoint),
        "target_step": int(target_step),
        "reuse_scope": (
            "ss_coords_and_target_meshes"
            if reuse_target_meshes
            else "ss_coords_only"
        ),
        "object_count": len(object_uids),
        "joint_seeds": [int(seed) for seed in seeds],
        "ss_pair_count": expected_pairs,
        "target_mesh_count": len(target_meshes),
        "linked_file_count": len(linked),
        "linked_files": linked,
    }
    manifest_path = target_worker / "checkpoint_independent_reuse.json"
    if manifest_path.is_file():
        previous = _load_json(manifest_path)
        previous.pop("created_at_utc", None)
        # Backward-compatible interpretation of manifests written before the
        # explicit reuse-scope field was introduced.  Their historical behavior
        # always included both SS coords and target Meshes.
        if "reuse_scope" not in previous and reuse_target_meshes:
            previous["reuse_scope"] = "ss_coords_and_target_meshes"
        if previous != stable:
            raise RuntimeError("existing checkpoint-independent reuse manifest differs")
    else:
        payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **stable}
        temporary = manifest_path.with_name(f".{manifest_path.name}.tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, manifest_path)
    return {**stable, "manifest": str(manifest_path), "passed": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_worker", required=True)
    parser.add_argument("--target_worker", required=True)
    parser.add_argument("--source_step", required=True, type=int)
    parser.add_argument("--target_step", required=True, type=int)
    parser.add_argument("--source_checkpoint", required=True)
    parser.add_argument("--target_checkpoint", required=True)
    parser.add_argument(
        "--ss_coords_only",
        action="store_true",
        help="reuse only checkpoint-independent Native-SS coordinate artifacts",
    )
    args = parser.parse_args()
    result = prepare_reuse(
        source_worker=Path(args.source_worker),
        target_worker=Path(args.target_worker),
        source_step=args.source_step,
        target_step=args.target_step,
        source_checkpoint=Path(args.source_checkpoint),
        target_checkpoint=Path(args.target_checkpoint),
        reuse_target_meshes=not args.ss_coords_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
