#!/usr/bin/env python3
"""Freeze unseen Objaverse candidates and an exact quality-passed holdout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


CANDIDATE_AUDIT_FORMAT = (
    "pose_point_depth_mv.direct_slat_holdout_candidates.v1"
)
HOLDOUT_AUDIT_FORMAT = "pose_point_depth_mv.direct_slat_holdout_manifest.v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_csv(value: str) -> list[str]:
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("manifest CSV must be non-empty and unique")
    return values


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_source_path(
    manifest_path: Path,
    payload: dict[str, Any],
    value: str | Path,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root = Path(str(payload.get("output_dir", manifest_path.parent)))
    if not root.is_absolute():
        root = manifest_path.parent / root
    return (root / path).resolve()


def payload_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = []
    for key in ("samples", "objects"):
        value = payload.get(key, [])
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    return rows


def collect_seen_identities(
    manifests: Iterable[str | Path],
) -> dict[str, set[str]]:
    object_uids: set[str] = set()
    source_paths: set[str] = set()
    source_hashes: set[str] = set()
    for value in manifests:
        manifest_path = Path(value).resolve()
        payload = load_json(manifest_path)
        if not isinstance(payload, dict):
            raise ValueError(f"seen manifest must be a JSON object: {manifest_path}")
        for row in payload_rows(payload):
            object_uid = str(row.get("object_uid", row.get("uid", "")))
            if object_uid:
                object_uids.add(object_uid)
            source = row.get("source_glb")
            if source:
                source_paths.add(
                    str(resolve_source_path(manifest_path, payload, source))
                )
            source_hash = str(row.get("source_glb_sha256", ""))
            if source_hash:
                source_hashes.add(source_hash)
    if not object_uids or not source_paths:
        raise ValueError("seen manifests must expose object UIDs and source GLBs")
    return {
        "object_uids": object_uids,
        "source_glb_paths": source_paths,
        "source_glb_sha256": source_hashes,
    }


def deterministic_order(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    def rank(row: dict[str, Any]) -> tuple[str, str, str]:
        uid = str(row["object_uid"])
        source = str(row["source_glb"])
        digest = hashlib.sha256(
            f"{int(seed)}|{uid}|{source}".encode("utf-8")
        ).hexdigest()
        return digest, uid, source

    return sorted(rows, key=rank)


def raw_source_rows(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path).resolve()
    payload = load_json(manifest_path)
    rows: list[dict[str, str]] = []
    if isinstance(payload, dict) and "samples" not in payload:
        items = payload.items()
        for uid, source in items:
            source_path = Path(str(source))
            if not source_path.is_absolute():
                source_path = manifest_path.parent / source_path
            rows.append(
                {
                    "object_uid": str(uid),
                    "source_glb": str(source_path.resolve()),
                }
            )
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        for index, row in enumerate(payload["samples"]):
            uid = str(row.get("object_uid", row.get("uid", index)))
            source = row.get("source_glb", row.get("glb", row.get("path")))
            if source is None:
                raise ValueError(f"raw source row has no GLB: index={index}")
            rows.append(
                {
                    "object_uid": uid,
                    "source_glb": str(
                        resolve_source_path(manifest_path, payload, source)
                    ),
                }
            )
    elif isinstance(payload, list):
        for index, row in enumerate(payload):
            if isinstance(row, str):
                source = row
                uid = Path(row).stem
            else:
                uid = str(row.get("object_uid", row.get("uid", index)))
                source = row.get("source_glb", row.get("glb", row.get("path")))
                if source is None:
                    raise ValueError(f"raw source row has no GLB: index={index}")
            source_path = Path(str(source))
            if not source_path.is_absolute():
                source_path = manifest_path.parent / source_path
            rows.append(
                {
                    "object_uid": uid,
                    "source_glb": str(source_path.resolve()),
                }
            )
    else:
        raise ValueError(f"unsupported raw source manifest: {manifest_path}")
    identities = [(row["object_uid"], row["source_glb"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("raw source manifest contains duplicate object/path rows")
    return rows


def select_unseen_candidates(
    source_rows: Iterable[dict[str, Any]],
    *,
    seen: dict[str, set[str]],
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    eligible = [
        {
            "object_uid": str(row["object_uid"]),
            "source_glb": str(Path(str(row["source_glb"])).resolve()),
        }
        for row in source_rows
        if str(row["object_uid"]) not in seen["object_uids"]
        and str(Path(str(row["source_glb"])).resolve())
        not in seen["source_glb_paths"]
    ]
    selected = []
    selected_paths: set[str] = set()
    selected_hashes: set[str] = set()
    for row in deterministic_order(eligible, seed=seed):
        source = Path(row["source_glb"])
        if not source.is_file():
            continue
        source_hash = sha256_file(source)
        if (
            source_hash in seen["source_glb_sha256"]
            or row["source_glb"] in selected_paths
            or source_hash in selected_hashes
        ):
            continue
        selected_paths.add(row["source_glb"])
        selected_hashes.add(source_hash)
        selected.append({**row, "source_glb_sha256": source_hash})
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise RuntimeError(
            f"only {len(selected)} unseen source assets available; requested {count}"
        )
    return selected


def render_rows(
    manifest_paths: Iterable[str | Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    metadata: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    bindings = []
    seen_uids: set[str] = set()
    for value in manifest_paths:
        path = Path(value).resolve()
        payload = load_json(path)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("samples"), list
        ):
            raise ValueError(f"render manifest has no samples: {path}")
        current = copy.deepcopy(payload)
        current.pop("samples", None)
        current.pop("failures", None)
        current.pop("manifest_hash", None)
        if metadata is None:
            metadata = current
        else:
            for key in (
                "format",
                "image_root",
                "mask_root",
                "latent_root",
                "extrinsics_type",
                "camera_forward_sign",
                "coordinate_frame",
                "canonical_latent_frame",
                "image_size",
            ):
                if metadata.get(key) != current.get(key):
                    raise ValueError(f"render manifests disagree on {key}")
        for row in payload["samples"]:
            uid = str(row.get("uid", ""))
            if not uid or uid in seen_uids:
                raise ValueError(f"missing or duplicate render uid={uid!r}")
            seen_uids.add(uid)
            copied = copy.deepcopy(row)
            copied["_manifest_path"] = str(path)
            copied["_manifest_roots"] = {
                key: payload[key]
                for key in ("image_root", "mask_root", "latent_root")
            }
            rows.append(copied)
        bindings.append({"path": str(path), "sha256": sha256_file(path)})
    if metadata is None or not rows:
        raise ValueError("render selection has no accepted samples")
    return metadata, rows, bindings


def validate_render_artifacts(row: dict[str, Any]) -> None:
    manifest_path = Path(row["_manifest_path"])
    roots = row["_manifest_roots"]
    latent_root = Path(str(roots["latent_root"]))
    image_root = Path(str(roots["image_root"]))
    mask_root = Path(str(roots["mask_root"]))
    if not latent_root.is_absolute():
        latent_root = manifest_path.parent / latent_root
    if not image_root.is_absolute():
        image_root = manifest_path.parent / image_root
    if not mask_root.is_absolute():
        mask_root = manifest_path.parent / mask_root
    latent = Path(str(row["ss_latent"]))
    if not latent.is_absolute():
        latent = latent_root / latent
    required = [latent]
    for frame in row.get("frames", []):
        image = Path(str(frame["image"]))
        mask = Path(str(frame["mask"]))
        required.append(image if image.is_absolute() else image_root / image)
        required.append(mask if mask.is_absolute() else mask_root / mask)
    if not row.get("frames") or any(not path.is_file() for path in required):
        raise RuntimeError(f"render sample artifacts are incomplete: {row.get('uid')}")


def select_rendered_holdout(
    rows: Iterable[dict[str, Any]],
    *,
    seen: dict[str, set[str]],
    count: int,
    seed: int,
    eligible_object_uids: set[str] | None = None,
) -> list[dict[str, Any]]:
    by_object: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        object_uid = str(row.get("object_uid", ""))
        source_value = row.get("source_glb")
        if not object_uid or not source_value:
            raise ValueError("render sample lacks object_uid/source_glb")
        source = str(Path(str(source_value)).resolve())
        if (
            object_uid in seen["object_uids"]
            or source in seen["source_glb_paths"]
            or (
                eligible_object_uids is not None
                and object_uid not in eligible_object_uids
            )
        ):
            continue
        copied = copy.deepcopy(row)
        copied["source_glb"] = source
        by_object.setdefault(object_uid, []).append(copied)
    representatives = []
    for object_uid, candidates in by_object.items():
        candidates.sort(key=lambda row: str(row["uid"]))
        representatives.append(candidates[0])
    selected = []
    selected_paths: set[str] = set()
    selected_hashes: set[str] = set()
    for row in deterministic_order(representatives, seed=seed):
        validate_render_artifacts(row)
        source_hash = sha256_file(row["source_glb"])
        if (
            source_hash in seen["source_glb_sha256"]
            or row["source_glb"] in selected_paths
            or source_hash in selected_hashes
        ):
            continue
        selected_paths.add(row["source_glb"])
        selected_hashes.add(source_hash)
        cleaned = {
            key: value
            for key, value in row.items()
            if not key.startswith("_manifest_")
        }
        cleaned["source_glb_sha256"] = source_hash
        selected.append(cleaned)
        if len(selected) == int(count):
            break
    if len(selected) != int(count):
        raise RuntimeError(
            f"only {len(selected)} unseen quality-passed objects available; "
            f"requested exactly {count}"
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("candidates", "freeze_rendered"), required=True
    )
    parser.add_argument("--seen_manifests", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--audit_output", required=True)
    parser.add_argument("--selection_seed", type=int, default=20260725)
    parser.add_argument("--source_manifest", default="")
    parser.add_argument("--candidate_objects", type=int, default=512)
    parser.add_argument("--render_manifests", default="")
    parser.add_argument("--candidate_audit", default="")
    parser.add_argument(
        "--eligibility_report",
        default="",
        help=(
            "Optional local-lh-slats rank report. Only successful data-quality "
            "records are eligible; model A/C outputs are never accepted."
        ),
    )
    parser.add_argument("--holdout_objects", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_manifest).resolve()
    audit_path = Path(args.audit_output).resolve()
    if output_path.exists() or audit_path.exists():
        raise FileExistsError("holdout manifest outputs are immutable")
    seen_paths = parse_csv(args.seen_manifests)
    seen = collect_seen_identities(seen_paths)
    if args.mode == "candidates":
        if not args.source_manifest:
            raise ValueError("--source_manifest is required in candidates mode")
        count = int(args.candidate_objects)
        if count <= 0:
            raise ValueError("candidate_objects must be positive")
        source_path = Path(args.source_manifest).resolve()
        selected = select_unseen_candidates(
            raw_source_rows(source_path),
            seen=seen,
            count=count,
            seed=int(args.selection_seed),
        )
        output = {
            row["object_uid"]: row["source_glb"] for row in selected
        }
        atomic_json(output_path, output)
        audit = {
            "format": CANDIDATE_AUDIT_FORMAT,
            "passed": True,
            "selection_seed": int(args.selection_seed),
            "selection_policy": (
                "SHA256(seed|object_uid|resolved_source_glb), after exact "
                "object/path/SHA disjointness from all seen manifests"
            ),
            "source_manifest": {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
            },
            "seen_manifests": [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": sha256_file(path),
                }
                for path in seen_paths
            ],
            "candidate_count": len(selected),
            "candidates": selected,
            "output_manifest": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
            },
        }
    else:
        if not args.render_manifests:
            raise ValueError(
                "--render_manifests is required in freeze_rendered mode"
            )
        count = int(args.holdout_objects)
        if count <= 0:
            raise ValueError("holdout_objects must be positive")
        metadata, rows, input_bindings = render_rows(
            parse_csv(args.render_manifests)
        )
        if not args.candidate_audit:
            raise ValueError(
                "freeze_rendered mode requires the immutable candidate audit"
            )
        candidate_audit_path = Path(args.candidate_audit).resolve()
        candidate_audit = load_json(candidate_audit_path)
        candidate_body = dict(candidate_audit)
        candidate_saved_hash = str(candidate_body.pop("audit_sha256", ""))
        if (
            candidate_audit.get("format") != CANDIDATE_AUDIT_FORMAT
            or candidate_audit.get("passed") is not True
            or canonical_sha256(candidate_body) != candidate_saved_hash
        ):
            raise RuntimeError("candidate freeze audit is invalid")
        candidate_output = dict(candidate_audit.get("output_manifest", {}))
        candidate_manifest_path = Path(
            str(candidate_output.get("path", ""))
        ).resolve()
        if (
            not candidate_manifest_path.is_file()
            or sha256_file(candidate_manifest_path)
            != str(candidate_output.get("sha256", ""))
        ):
            raise RuntimeError("candidate source manifest changed")
        candidate_manifest = load_json(candidate_manifest_path)
        if not isinstance(candidate_manifest, dict):
            raise RuntimeError("candidate source manifest is not a UID/path mapping")
        candidate_identities = {
            (str(uid), str(Path(str(source)).resolve()))
            for uid, source in candidate_manifest.items()
        }
        if any(
            not str(row.get("object_uid", ""))
            or not str(row.get("source_glb", ""))
            for row in rows
        ):
            raise RuntimeError("accepted render rows lack object/source identity")
        render_identities = {
            (
                str(row.get("object_uid", "")),
                str(Path(str(row.get("source_glb", ""))).resolve()),
            )
            for row in rows
        }
        if (
            not render_identities
            or not render_identities.issubset(candidate_identities)
        ):
            raise RuntimeError(
                "accepted render rows are not a subset of the frozen candidates"
            )
        candidate_freeze_binding = {
            "audit": {
                "path": str(candidate_audit_path),
                "sha256": sha256_file(candidate_audit_path),
            },
            "candidate_manifest": {
                "path": str(candidate_manifest_path),
                "sha256": sha256_file(candidate_manifest_path),
            },
            "source_manifest": dict(candidate_audit["source_manifest"]),
            "candidate_count": int(candidate_audit["candidate_count"]),
        }
        eligibility_binding = None
        eligible_object_uids = None
        if args.eligibility_report:
            eligibility_path = Path(args.eligibility_report).resolve()
            eligibility = load_json(eligibility_path)
            if (
                not isinstance(eligibility, dict)
                or eligibility.get("format")
                != "pose_point_depth_mv.local_lh_slats.v2"
                or eligibility.get(
                    "projection_camera_corruption_gate_failures"
                )
                != []
            ):
                raise RuntimeError(
                    "eligibility report failed local-SLAT family/corruption gates"
                )
            eligible_rows = list(eligibility.get("records", []))
            eligible_object_uids = {
                str(row.get("object_uid", "")) for row in eligible_rows
            }
            eligible_object_uids.discard("")
            if len(eligible_object_uids) != len(eligible_rows):
                raise RuntimeError(
                    "eligibility report has missing or duplicate object records"
                )
            for row in eligible_rows:
                output = Path(str(row.get("output", "")))
                if (
                    not output.is_file()
                    or sha256_file(output)
                    != str(row.get("output_sha256", ""))
                ):
                    raise RuntimeError(
                        "eligible local-SLAT artifact changed: "
                        f"{row.get('object_uid')}"
                    )
            eligibility_binding = {
                "path": str(eligibility_path),
                "sha256": sha256_file(eligibility_path),
                "successful_object_count": len(eligible_object_uids),
                "model_outputs_read": False,
            }
        selected = select_rendered_holdout(
            rows,
            seen=seen,
            count=count,
            seed=int(args.selection_seed),
            eligible_object_uids=eligible_object_uids,
        )
        output = metadata
        output["samples"] = selected
        output["failures"] = []
        output["blind_holdout_selection"] = {
            "policy": (
                "quality-passed rows only, then "
                "SHA256(seed|object_uid|resolved_source_glb)"
            ),
            "selection_seed": int(args.selection_seed),
            "sample_count": len(selected),
            "object_count": len(selected),
            "one_sequence_per_object": True,
            "model_outputs_read": False,
        }
        output["manifest_hash"] = canonical_sha256(
            {key: value for key, value in output.items() if key != "manifest_hash"}
        )
        atomic_json(output_path, output)
        audit = {
            "format": HOLDOUT_AUDIT_FORMAT,
            "passed": True,
            "selection_seed": int(args.selection_seed),
            "input_render_manifests": input_bindings,
            "candidate_freeze": candidate_freeze_binding,
            "eligibility_report": eligibility_binding,
            "seen_manifests": [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": sha256_file(path),
                }
                for path in seen_paths
            ],
            "accepted_render_sample_count": len(rows),
            "selected_object_count": len(selected),
            "selected": [
                {
                    "uid": row["uid"],
                    "object_uid": row["object_uid"],
                    "source_glb": row["source_glb"],
                    "source_glb_sha256": row["source_glb_sha256"],
                }
                for row in selected
            ],
            "overlap_counts": {
                "object_uid": 0,
                "source_glb_path": 0,
                "source_glb_sha256": 0,
            },
            "model_outputs_read": False,
            "output_manifest": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "manifest_hash": output["manifest_hash"],
            },
        }
    audit["audit_sha256"] = canonical_sha256(audit)
    atomic_json(audit_path, audit)
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
