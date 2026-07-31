#!/usr/bin/env python3
"""Freeze or invalidate the post-selection Direct-SLAT holdout data protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pose_point_depth_mv.direct_slat_blind import (
    HOLDOUT_INTEGRITY_FORMAT,
    atomic_json,
    bind_file,
    canonical_sha256,
    sha256_file,
)
from pose_point_depth_mv.direct_slat_flow import DIRECT_SLAT_CACHE_VERSION


INVALID_FORMAT = "pose_point_depth_mv.direct_slat_holdout_invalid.v2"
SELECTION_FORMAT = "pose_point_depth_mv.direct_slat_holdout_manifest.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("freeze", "invalidate"), required=True)
    parser.add_argument("--invalid_marker", required=True)
    parser.add_argument("--stage", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--selection_audit", default="")
    parser.add_argument("--lifting_manifest", default="")
    parser.add_argument("--lifting_audit", default="")
    parser.add_argument("--stock_replay_audit", default="")
    parser.add_argument("--cache_manifest", default="")
    parser.add_argument("--target_decoder_audit", default="")
    parser.add_argument("--expected_objects", type=int, default=32)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def object_uids(payload: dict[str, Any]) -> set[str]:
    rows = [
        row
        for key in ("samples", "objects", "records", "selected")
        for row in payload.get(key, [])
        if isinstance(row, dict)
    ]
    result = {
        str(row.get("object_uid", row.get("uid", ""))) for row in rows
    }
    result.discard("")
    return result


def write_exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def invalidate(args: argparse.Namespace, marker: Path) -> None:
    if not args.stage.strip() or not args.reason.strip():
        raise ValueError("invalidate mode requires --stage and --reason")
    payload = {
        "format": INVALID_FORMAT,
        "invalid": True,
        "stage": args.stage.strip(),
        "reason": args.reason.strip(),
        "policy": (
            "This frozen object set is not executable. Do not replace failed "
            "objects or select from the remaining candidate pool; create a new "
            "versioned holdout protocol."
        ),
    }
    payload["marker_sha256"] = canonical_sha256(payload)
    write_exclusive_json(marker, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


def freeze(args: argparse.Namespace, marker: Path) -> None:
    if marker.exists():
        raise RuntimeError(f"holdout version is invalid and cannot be frozen: {marker}")
    required = {
        "selection_audit": args.selection_audit,
        "lifting_manifest": args.lifting_manifest,
        "lifting_audit": args.lifting_audit,
        "stock_replay_audit": args.stock_replay_audit,
        "cache_manifest": args.cache_manifest,
        "target_decoder_audit": args.target_decoder_audit,
    }
    if any(not value for value in required.values()) or not args.output:
        raise ValueError("freeze mode requires every audit input and --output")
    paths = {name: Path(value).resolve() for name, value in required.items()}
    payloads = {name: load_json(path) for name, path in paths.items()}
    selection = payloads["selection_audit"]
    selection_body = dict(selection)
    saved_audit_sha = str(selection_body.pop("audit_sha256", ""))
    if (
        selection.get("format") != SELECTION_FORMAT
        or selection.get("passed") is not True
        or selection.get("model_outputs_read") is not False
        or canonical_sha256(selection_body) != saved_audit_sha
    ):
        raise RuntimeError("frozen holdout selection audit is invalid")

    expected_count = int(args.expected_objects)
    expected_uids = object_uids({"selected": selection.get("selected", [])})
    lifting = payloads["lifting_manifest"]
    lifting_uids = object_uids(lifting)
    lifting_audit = payloads["lifting_audit"]
    stock_audit = payloads["stock_replay_audit"]
    cache = payloads["cache_manifest"]
    target_audit = payloads["target_decoder_audit"]
    cache_uids = object_uids(cache)
    target_uids = object_uids(target_audit)
    lifting_hash = sha256_file(paths["lifting_manifest"])
    cache_hash = sha256_file(paths["cache_manifest"])
    checks = {
        "selection_exact_object_count": len(expected_uids) == expected_count,
        "lifting_exact_same_objects": lifting_uids == expected_uids,
        "lifting_manifest_exact_count": int(
            lifting.get("object_count", lifting.get("sample_count", -1))
        )
        == expected_count,
        "lifting_audit_passed": lifting_audit.get("passed") is True,
        "lifting_audit_binds_manifest": str(
            lifting_audit.get(
                "cache_manifest_sha256", lifting_audit.get("manifest_sha256", "")
            )
        )
        == lifting_hash,
        "stock_replay_audit_passed": stock_audit.get("passed") is True,
        "stock_replay_audit_binds_manifest": str(
            stock_audit.get("cache_manifest_sha256", "")
        )
        == lifting_hash,
        "direct_slat_cache_materialized": (
            cache.get("format") == DIRECT_SLAT_CACHE_VERSION
            and cache.get("materialized") is True
        ),
        "direct_slat_cache_exact_same_objects": cache_uids == expected_uids,
        "direct_slat_cache_exact_count": (
            int(cache.get("object_count", -1)) == expected_count
            and int(cache.get("sequence_count", -1)) == expected_count
            and int(cache.get("sample_count", -1)) == 3 * expected_count
        ),
        "target_decoder_audit_passed": target_audit.get("passed") is True,
        "target_decoder_audit_binds_cache": str(
            target_audit.get("cache_manifest_sha256", "")
        )
        == cache_hash,
        "target_decoder_audit_exact_same_objects": target_uids == expected_uids,
    }
    if not all(checks.values()):
        raise RuntimeError(f"post-selection holdout integrity failed: {checks}")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = {
        "format": HOLDOUT_INTEGRITY_FORMAT,
        "passed": True,
        "object_count": expected_count,
        "object_uids_sha256": canonical_sha256(sorted(expected_uids)),
        "checks": checks,
        "bindings": {
            name: bind_file(path) for name, path in paths.items()
        },
        "freeze_policy": (
            "All post-selection quality checks passed for exactly the B8 object "
            "set. Any later data-quality failure invalidates this version; no "
            "failed object may be replaced."
        ),
    }
    report["integrity_sha256"] = canonical_sha256(report)
    atomic_json(output, report)
    output.chmod(0o444)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    marker = Path(args.invalid_marker).resolve()
    if args.mode == "invalidate":
        invalidate(args, marker)
    else:
        freeze(args, marker)


if __name__ == "__main__":
    main()
