#!/usr/bin/env python3
"""Create a portable, hashed, read-only Direct-SLAT public rating bundle."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from typing import Any

from pose_point_depth_mv.direct_slat_blind import (
    PROTOCOL_FORMAT,
    PUBLIC_ARCHIVE_FORMAT,
    PUBLIC_BUNDLE_FORMAT,
    atomic_json,
    canonical_sha256,
    sha256_file,
)


PUBLIC_SOURCES = ("blind_pairs", "score_templates", "blind_manifest.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--blind_output_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--archive_manifest", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    body = dict(protocol)
    saved = str(body.pop("protocol_sha256", ""))
    if protocol.get("format") != PROTOCOL_FORMAT or canonical_sha256(body) != saved:
        raise RuntimeError("formal protocol is not a valid v2 frozen protocol")
    if protocol.get("mode") != "confirmatory" or protocol.get("formal") is not True:
        raise RuntimeError("public bundle requires a formal confirmatory protocol")
    return protocol


def validate_completion(root: Path, protocol_sha256: str) -> tuple[Path, dict[str, Any]]:
    path = root / "completion_manifest.json"
    completion = load_json(path)
    if (
        completion.get("complete") is not True
        or completion.get("formal") is not True
        or completion.get("mode") != "confirmatory"
        or completion.get("all_records_passed") is not True
        or int(completion.get("runtime_exit_code", -1)) != 0
        or completion.get("science_decision_emitted") is not False
        or completion.get("protocol_sha256") != protocol_sha256
    ):
        raise RuntimeError("formal blind export is incomplete or failed")
    listed = {}
    for row in completion.get("files", []):
        relative = safe_relative_path(str(row.get("path", "")))
        if relative in listed:
            raise RuntimeError(f"duplicate completion path: {relative}")
        artifact = root / relative
        if not artifact.is_file() or sha256_file(artifact) != str(row.get("sha256", "")):
            raise RuntimeError(f"formal export artifact changed: {artifact}")
        listed[relative] = str(row["sha256"])
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "completion_manifest.json"
    }
    if set(listed) != actual:
        raise RuntimeError("formal completion manifest does not cover exact export")
    return path, completion


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"path is not a safe relative POSIX path: {value!r}")
    return path.as_posix()


def validate_blind_manifest(root: Path) -> int:
    path = root / "blind_manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("public blind manifest has no rows")
    pair_ids = [str(row.get("pair_id", "")) for row in rows]
    if "" in pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise RuntimeError("public blind manifest pair IDs are missing or duplicated")
    for row in rows:
        for column in ("A_mesh", "B_mesh", "A_preview", "B_preview"):
            value = str(row.get(column, ""))
            if not value:
                if column.endswith("_preview"):
                    continue
                raise RuntimeError(f"public blind manifest lacks {column}")
            relative = safe_relative_path(value)
            if not relative.startswith("blind_pairs/"):
                raise RuntimeError(f"public artifact escapes blind_pairs/: {relative}")
            if not (root / relative).is_file():
                raise RuntimeError(f"public artifact is missing: {relative}")
    return len(rows)


def payload_files(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size": int(path.stat().st_size),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "public_bundle_manifest.json"
    ]


def deterministic_tar(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
        paths = [source, *sorted(source.rglob("*"))]
        for path in paths:
            relative = path.relative_to(source)
            arcname = PurePosixPath("public_blind_bundle", *relative.parts).as_posix()
            info = tarfile.TarInfo(arcname)
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o555
                archive.addfile(info)
            elif path.is_file():
                info.size = int(path.stat().st_size)
                info.mode = 0o444
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                raise RuntimeError(f"public bundle contains unsupported entry: {path}")
    os.replace(temporary, destination)


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    blind_root = Path(args.blind_output_dir).resolve()
    output = Path(args.output_dir).resolve()
    archive = Path(args.archive).resolve()
    archive_manifest = Path(args.archive_manifest).resolve()
    for destination in (output, archive, archive_manifest):
        if destination.exists():
            raise FileExistsError(destination)
    protocol = validate_protocol(protocol_path)
    completion_path, completion = validate_completion(
        blind_root, str(protocol["protocol_sha256"])
    )

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for name in PUBLIC_SOURCES:
            source = blind_root / name
            destination = temporary / name
            if source.is_dir():
                shutil.copytree(source, destination, copy_function=shutil.copy2)
            elif source.is_file():
                shutil.copy2(source, destination)
            else:
                raise FileNotFoundError(source)
        pair_count = validate_blind_manifest(temporary)
        files = payload_files(temporary)
        manifest = {
            "format": PUBLIC_BUNDLE_FORMAT,
            "complete": True,
            "portable_paths": True,
            "protocol_sha256": protocol["protocol_sha256"],
            "formal_completion_sha256": sha256_file(completion_path),
            "formal_sealed_report_sha256": completion["sealed_report_sha256"],
            "pair_count": pair_count,
            "file_count": len(files),
            "files": files,
            "manifest_scope": (
                "Every public payload file is listed. This manifest excludes "
                "itself to avoid a recursive self-hash."
            ),
        }
        manifest["public_bundle_sha256"] = canonical_sha256(manifest)
        atomic_json(temporary / "public_bundle_manifest.json", manifest)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    archive.parent.mkdir(parents=True, exist_ok=True)
    deterministic_tar(output, archive)
    archive_record = {
        "format": PUBLIC_ARCHIVE_FORMAT,
        "complete": True,
        "protocol_sha256": protocol["protocol_sha256"],
        "formal_completion_sha256": sha256_file(completion_path),
        "public_bundle_sha256": manifest["public_bundle_sha256"],
        "public_bundle_manifest_sha256": sha256_file(
            output / "public_bundle_manifest.json"
        ),
        "archive_file": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_size": int(archive.stat().st_size),
        "archive_properties": "deterministic uncompressed tar, uid/gid/mtime fixed",
    }
    archive_record["archive_manifest_sha256"] = canonical_sha256(archive_record)
    atomic_json(archive_manifest, archive_record)
    make_read_only(output)
    archive.chmod(0o444)
    archive_manifest.chmod(0o444)
    print(
        json.dumps(
            {
                "public_bundle": str(output),
                "public_bundle_sha256": manifest["public_bundle_sha256"],
                "archive": str(archive),
                "archive_sha256": archive_record["archive_sha256"],
                "archive_manifest": str(archive_manifest),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
