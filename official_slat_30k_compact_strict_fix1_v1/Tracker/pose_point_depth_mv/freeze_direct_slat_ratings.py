#!/usr/bin/env python3
"""Freeze complete Direct-SLAT rating CSVs without reading the blind key."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
from typing import Any

from pose_point_depth_mv.direct_slat_blind import (
    PUBLIC_ARCHIVE_FORMAT,
    PUBLIC_BUNDLE_FORMAT,
    RATINGS_FREEZE_FORMAT,
    atomic_json,
    canonical_sha256,
    parse_csv,
    read_and_validate_rater_csv,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public_bundle_manifest", required=True)
    parser.add_argument("--public_archive_manifest", required=True)
    parser.add_argument("--rater_scores", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe public bundle path: {value!r}")
    return path.as_posix()


def validate_public_bundle(path: Path) -> tuple[dict[str, Any], list[str]]:
    root = path.parent
    manifest = load_json(path)
    body = dict(manifest)
    saved = str(body.pop("public_bundle_sha256", ""))
    if (
        manifest.get("format") != PUBLIC_BUNDLE_FORMAT
        or manifest.get("complete") is not True
        or manifest.get("portable_paths") is not True
        or canonical_sha256(body) != saved
    ):
        raise RuntimeError("public bundle manifest is invalid")
    listed: set[str] = set()
    for row in manifest.get("files", []):
        relative = safe_relative_path(str(row.get("path", "")))
        if relative in listed:
            raise RuntimeError(f"duplicate public bundle path: {relative}")
        artifact = root / relative
        if (
            not artifact.is_file()
            or sha256_file(artifact) != str(row.get("sha256", ""))
            or int(artifact.stat().st_size) != int(row.get("size", -1))
        ):
            raise RuntimeError(f"public bundle artifact changed: {artifact}")
        listed.add(relative)
    actual = {
        artifact.relative_to(root).as_posix()
        for artifact in root.rglob("*")
        if artifact.is_file() and artifact != path
    }
    if listed != actual or len(listed) != int(manifest.get("file_count", -1)):
        raise RuntimeError("public bundle manifest does not cover exact payload")
    blind_manifest = root / "blind_manifest.csv"
    with blind_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pair_ids = [str(row.get("pair_id", "")).strip() for row in rows]
    if (
        not pair_ids
        or "" in pair_ids
        or len(pair_ids) != len(set(pair_ids))
        or len(pair_ids) != int(manifest.get("pair_count", -1))
    ):
        raise RuntimeError("public blind manifest pair coverage is invalid")
    return manifest, pair_ids


def validate_archive(
    path: Path,
    *,
    public_manifest: dict[str, Any],
    public_manifest_path: Path,
) -> tuple[dict[str, Any], Path]:
    record = load_json(path)
    body = dict(record)
    saved = str(body.pop("archive_manifest_sha256", ""))
    archive_name = safe_relative_path(str(record.get("archive_file", "")))
    if "/" in archive_name:
        raise RuntimeError("public archive must be adjacent to its manifest")
    archive = path.parent / archive_name
    if (
        record.get("format") != PUBLIC_ARCHIVE_FORMAT
        or record.get("complete") is not True
        or canonical_sha256(body) != saved
        or record.get("public_bundle_sha256")
        != public_manifest["public_bundle_sha256"]
        or record.get("protocol_sha256") != public_manifest["protocol_sha256"]
        or record.get("formal_completion_sha256")
        != public_manifest["formal_completion_sha256"]
        or record.get("public_bundle_manifest_sha256")
        != sha256_file(public_manifest_path)
        or not archive.is_file()
        or sha256_file(archive) != str(record.get("archive_sha256", ""))
        or int(archive.stat().st_size) != int(record.get("archive_size", -1))
    ):
        raise RuntimeError("public archive binding is invalid")
    return record, archive


def main() -> None:
    args = parse_args()
    public_manifest_path = Path(args.public_bundle_manifest).resolve()
    archive_manifest_path = Path(args.public_archive_manifest).resolve()
    public_manifest, pair_ids = validate_public_bundle(public_manifest_path)
    archive_manifest, archive = validate_archive(
        archive_manifest_path,
        public_manifest=public_manifest,
        public_manifest_path=public_manifest_path,
    )
    score_paths = [Path(value).resolve() for value in parse_csv(args.rater_scores, str)]
    ratings = [
        read_and_validate_rater_csv(path, expected_pair_ids=pair_ids)
        for path in score_paths
    ]
    rater_ids = [str(row["rater_id"]) for row in ratings]
    if len(ratings) < 3 or len(rater_ids) != len(set(rater_ids)):
        raise RuntimeError("ratings freeze requires at least three unique raters")

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = {
        "format": RATINGS_FREEZE_FORMAT,
        "complete": True,
        "blind_key_read": False,
        "protocol_sha256": public_manifest["protocol_sha256"],
        "formal_completion_sha256": public_manifest[
            "formal_completion_sha256"
        ],
        "public_bundle": {
            "manifest_path": str(public_manifest_path),
            "manifest_sha256": sha256_file(public_manifest_path),
            "public_bundle_sha256": public_manifest["public_bundle_sha256"],
        },
        "public_archive": {
            "manifest_path": str(archive_manifest_path),
            "manifest_sha256": sha256_file(archive_manifest_path),
            "archive_path": str(archive.resolve()),
            "archive_sha256": archive_manifest["archive_sha256"],
        },
        "expected_pair_count": len(pair_ids),
        "expected_pair_ids_sha256": canonical_sha256(sorted(pair_ids)),
        "ratings": [
            {
                "path": row["path"],
                "sha256": row["sha256"],
                "rater_id": row["rater_id"],
            }
            for row in ratings
        ],
        "rater_count": len(ratings),
        "freeze_policy": (
            "CSV schema, coverage, rater identity, public bundle SHA, and archive "
            "SHA were validated before any process reads the blind key."
        ),
    }
    report["ratings_freeze_sha256"] = canonical_sha256(report)
    atomic_json(output, report)
    output.chmod(0o444)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
