#!/usr/bin/env python3
"""Prepare and audit the frozen ProObjaverse 30K raw OSS transfer.

The payload upload itself is deliberately left to ossutil.  This helper owns
the immutable local inventory and the post-upload comparison.  All payloads
are below ossutil's multipart threshold, so an OSS ETag is expected to be the
MD5 of the uploaded file and can be compared without downloading it again.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


FORMAT_INVENTORY = "reconviagen.proobjaverse_30k_oss_local_inventory.v1"
FORMAT_REMOTE_AUDIT = "reconviagen.proobjaverse_30k_oss_remote_audit.v1"
FORMAT_COMPLETION = "reconviagen.proobjaverse_30k_oss_completion.v1"

EXPECTED_SELECTION_FILE_SHA256 = (
    "8125521278a97c6120a6246fe47eb0872fb905e0fb631a0c0189116a19b53f48"
)
EXPECTED_SELECTION_SHA256 = (
    "c9ef45c9687fdb7e68f44eb70fd5249f24def9bcc3bf80a9c6d57572a3e6c6de"
)
EXPECTED_UID_SHA256 = (
    "46557538eb88ffc9aea0514a390bf7bcf866d3caf76c8f5b9db26d8357010089"
)
EXPECTED_SOURCE_AUDIT_FILE_SHA256 = (
    "ad1947dd37fc89059fe57019dc035d78c39c75f5bcafddae42ffa700224dc364"
)
EXPECTED_PAIR_COUNT = 30_000
EXPECTED_FILE_COUNT = 60_000
EXPECTED_PAYLOAD_BYTES = 125_437_607_123
OSSUTIL_MULTIPART_THRESHOLD = 100 * 1024 * 1024

_UID_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_RE = re.compile(r"^shard-[0-9]{4}$")
_ETAG_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_EMPTY_OBJECT_ETAG = "d41d8cd98f00b204e9800998ecf8427e"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["absolute_path"])
    before = path.stat()
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"payload changed while hashing: {path}")
    if before.st_size != int(record["size"]):
        raise RuntimeError(
            f"payload size changed: {path}: expected={record['size']} actual={before.st_size}"
        )
    return {
        "kind": record["kind"],
        "uid": record["uid"],
        "shard": record["shard"],
        "relative_path": record["relative_path"],
        "size": before.st_size,
        "allocated_bytes": before.st_blocks * 512,
        "device": before.st_dev,
        "inode": before.st_ino,
        "hardlink_count": before.st_nlink,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
    }


def _validate_frozen_source(
    data_root: Path,
    selection_path: Path,
    source_audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if _sha256_file(selection_path) != EXPECTED_SELECTION_FILE_SHA256:
        raise RuntimeError(f"frozen selection file SHA256 mismatch: {selection_path}")
    if _sha256_file(source_audit_path) != EXPECTED_SOURCE_AUDIT_FILE_SHA256:
        raise RuntimeError(f"frozen source audit file SHA256 mismatch: {source_audit_path}")

    selection = _read_json(selection_path)
    source_audit = _read_json(source_audit_path)
    if selection.get("format") != "reconviagen.proobjaverse_paired_combined.v1":
        raise RuntimeError(f"unexpected frozen selection format={selection.get('format')!r}")
    if selection.get("pair_count") != EXPECTED_PAIR_COUNT:
        raise RuntimeError("frozen selection pair count is not 30000")
    if selection.get("expected_payload_bytes") != EXPECTED_PAYLOAD_BYTES:
        raise RuntimeError("frozen selection payload byte count changed")
    if selection.get("combined_selection_sha256") != EXPECTED_SELECTION_SHA256:
        raise RuntimeError("frozen selection identity changed")
    if selection.get("uid_sha256") != EXPECTED_UID_SHA256:
        raise RuntimeError("frozen UID identity changed")
    if source_audit.get("passed") is not True:
        raise RuntimeError("frozen source audit did not pass")
    for key, expected in (
        ("pair_count", EXPECTED_PAIR_COUNT),
        ("render_count", EXPECTED_PAIR_COUNT),
        ("slat_count", EXPECTED_PAIR_COUNT),
        ("verified_payload_bytes", EXPECTED_PAYLOAD_BYTES),
        ("component_overlap_count", 0),
        ("unselected_payload_count", 0),
    ):
        if source_audit.get(key) != expected:
            raise RuntimeError(
                f"frozen source audit field changed: {key}: "
                f"expected={expected!r} actual={source_audit.get(key)!r}"
            )

    selected = selection.get("selected")
    if not isinstance(selected, list) or len(selected) != EXPECTED_PAIR_COUNT:
        raise RuntimeError("frozen selection does not contain exactly 30000 rows")

    seen_uids: set[str] = set()
    expected_paths: set[str] = set()
    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        if not isinstance(row, dict):
            raise TypeError(f"selection row {index} is not an object")
        uid = row.get("uid")
        shard = row.get("shard")
        if not isinstance(uid, str) or _UID_RE.fullmatch(uid) is None:
            raise RuntimeError(f"invalid UID at selection row {index}: {uid!r}")
        if uid in seen_uids:
            raise RuntimeError(f"duplicate UID in frozen selection: {uid}")
        seen_uids.add(uid)
        if not isinstance(shard, str) or _SHARD_RE.fullmatch(shard) is None:
            raise RuntimeError(f"invalid shard at selection row {index}: {shard!r}")

        for kind, prefix, suffix in (
            ("render", "renders_random_env", ".tar"),
            ("slat", "lh-slats", ".npz"),
        ):
            payload = row.get(kind)
            if not isinstance(payload, dict):
                raise RuntimeError(f"missing {kind} payload at selection row {index}")
            relative = payload.get("path")
            expected_size = payload.get("size")
            if not isinstance(relative, str) or not isinstance(expected_size, int):
                raise RuntimeError(f"invalid {kind} descriptor at selection row {index}")
            relative_posix = PurePosixPath(relative)
            if relative_posix.is_absolute() or ".." in relative_posix.parts:
                raise RuntimeError(f"unsafe payload path: {relative}")
            required = PurePosixPath(prefix) / shard / f"{uid}{suffix}"
            if relative_posix != required:
                raise RuntimeError(
                    f"payload path does not match frozen UID/shard: {relative} != {required}"
                )
            if relative in expected_paths:
                raise RuntimeError(f"duplicate payload path: {relative}")
            expected_paths.add(relative)
            absolute = data_root / Path(*relative_posix.parts)
            if absolute.is_symlink():
                raise RuntimeError(f"symlink payload is forbidden: {absolute}")
            if not absolute.is_file():
                raise FileNotFoundError(f"missing frozen payload: {absolute}")
            actual_size = absolute.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(
                    f"payload size mismatch: {absolute}: "
                    f"expected={expected_size} actual={actual_size}"
                )
            records.append(
                {
                    "kind": kind,
                    "uid": uid,
                    "shard": shard,
                    "relative_path": relative,
                    "absolute_path": str(absolute),
                    "size": expected_size,
                }
            )

    if len(records) != EXPECTED_FILE_COUNT:
        raise RuntimeError(f"expected 60000 payload records, found {len(records)}")
    if sum(int(record["size"]) for record in records) != EXPECTED_PAYLOAD_BYTES:
        raise RuntimeError("selection payload sizes no longer sum to the frozen byte count")

    actual_paths: set[str] = set()
    for top_level in ("renders_random_env", "lh-slats"):
        root = data_root / top_level
        if not root.is_dir():
            raise FileNotFoundError(f"missing payload directory: {root}")
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise RuntimeError(f"symlink inside frozen payload tree is forbidden: {candidate}")
            if candidate.is_file():
                actual_paths.add(candidate.relative_to(data_root).as_posix())
    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    if missing or extra:
        raise RuntimeError(
            "frozen payload tree differs from the selection: "
            f"missing_count={len(missing)} extra_count={len(extra)} "
            f"missing_examples={missing[:5]} extra_examples={extra[:5]}"
        )
    records.sort(key=lambda value: value["relative_path"])
    return selection, source_audit, records


def _load_inventory(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"inventory line {line_number} is not an object")
            records.append(value)
    if len(records) != EXPECTED_FILE_COUNT:
        raise RuntimeError(f"inventory does not contain 60000 records: {path}")
    return records


def prepare(args: argparse.Namespace) -> int:
    data_root = args.data_root.resolve(strict=True)
    selection_path = args.selection.resolve(strict=True)
    source_audit_path = args.source_audit.resolve(strict=True)
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = state_dir / "local_inventory.jsonl"
    report_path = state_dir / "local_inventory_report.json"

    selection, source_audit, expected_records = _validate_frozen_source(
        data_root, selection_path, source_audit_path
    )

    if inventory_path.exists() and report_path.exists() and not args.rehash:
        report = _read_json(report_path)
        inventory_sha256 = _sha256_file(inventory_path)
        reuse_checks = {
            "format": report.get("format") == FORMAT_INVENTORY,
            "passed": report.get("passed") is True,
            "file_count": report.get("file_count") == EXPECTED_FILE_COUNT,
            "pair_count": report.get("pair_count") == EXPECTED_PAIR_COUNT,
            "logical_bytes": report.get("logical_bytes") == EXPECTED_PAYLOAD_BYTES,
            "selection_file_sha256": report.get("selection_file_sha256")
            == EXPECTED_SELECTION_FILE_SHA256,
            "source_audit_file_sha256": report.get("source_audit_file_sha256")
            == EXPECTED_SOURCE_AUDIT_FILE_SHA256,
            "inventory_jsonl_sha256": report.get("inventory_jsonl_sha256")
            == inventory_sha256,
        }
        if not all(reuse_checks.values()):
            raise RuntimeError(
                "existing inventory cannot be reused; pass --rehash only after auditing: "
                + json.dumps(reuse_checks, sort_keys=True)
            )
        _load_inventory(inventory_path)
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        print(f"REUSED_INVENTORY={inventory_path}")
        return 0

    temporary = inventory_path.with_name(inventory_path.name + ".tmp")
    content_digest = hashlib.sha256()
    counts_by_kind: Counter[str] = Counter()
    counts_by_shard: Counter[str] = Counter()
    logical_bytes = 0
    allocated_bytes_by_unique_inode: dict[tuple[int, int], int] = {}
    hardlinked_file_count = 0
    maximum_file_size = 0

    with temporary.open("w", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.hash_workers) as pool:
            for index, record in enumerate(pool.map(_hash_payload, expected_records), start=1):
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                content_digest.update(record["relative_path"].encode("utf-8"))
                content_digest.update(b"\0")
                content_digest.update(str(record["size"]).encode("ascii"))
                content_digest.update(b"\0")
                content_digest.update(record["sha256"].encode("ascii"))
                content_digest.update(b"\n")
                counts_by_kind[record["kind"]] += 1
                counts_by_shard[record["shard"]] += 1
                logical_bytes += int(record["size"])
                inode_key = (int(record["device"]), int(record["inode"]))
                allocated_bytes_by_unique_inode.setdefault(
                    inode_key, int(record["allocated_bytes"])
                )
                hardlinked_file_count += int(record["hardlink_count"] > 1)
                maximum_file_size = max(maximum_file_size, int(record["size"]))
                if index % 1000 == 0 or index == EXPECTED_FILE_COUNT:
                    print(
                        f"[local_inventory] {index}/{EXPECTED_FILE_COUNT} "
                        f"bytes={logical_bytes}",
                        flush=True,
                    )
    os.replace(temporary, inventory_path)

    if logical_bytes != EXPECTED_PAYLOAD_BYTES:
        raise RuntimeError("hashed logical byte count differs from frozen source identity")
    if maximum_file_size >= OSSUTIL_MULTIPART_THRESHOLD:
        raise RuntimeError(
            "at least one payload reaches the multipart threshold; OSS ETag cannot be "
            "used as a simple MD5 contract"
        )

    report = {
        "format": FORMAT_INVENTORY,
        "passed": True,
        "created_at_utc": _utc_now(),
        "data_root": str(data_root),
        "selection": str(selection_path),
        "selection_file_sha256": EXPECTED_SELECTION_FILE_SHA256,
        "selection_sha256": selection["combined_selection_sha256"],
        "uid_sha256": selection["uid_sha256"],
        "source_audit": str(source_audit_path),
        "source_audit_file_sha256": EXPECTED_SOURCE_AUDIT_FILE_SHA256,
        "source_audit_internal_report_sha256": source_audit.get("report_sha256"),
        "pair_count": EXPECTED_PAIR_COUNT,
        "file_count": EXPECTED_FILE_COUNT,
        "render_count": counts_by_kind["render"],
        "slat_count": counts_by_kind["slat"],
        "logical_bytes": logical_bytes,
        "allocated_bytes_unique_inodes": sum(allocated_bytes_by_unique_inode.values()),
        "unique_inode_count": len(allocated_bytes_by_unique_inode),
        "hardlinked_file_count": hardlinked_file_count,
        "maximum_file_size": maximum_file_size,
        "ossutil_multipart_threshold": OSSUTIL_MULTIPART_THRESHOLD,
        "all_payloads_single_part": True,
        "shard_count": len(counts_by_shard),
        "counts_by_shard": dict(sorted(counts_by_shard.items())),
        "dataset_content_sha256": content_digest.hexdigest(),
        "inventory_jsonl": str(inventory_path),
        "inventory_jsonl_sha256": _sha256_file(inventory_path),
    }
    if report["render_count"] != EXPECTED_PAIR_COUNT:
        raise RuntimeError("local inventory render count mismatch")
    if report["slat_count"] != EXPECTED_PAIR_COUNT:
        raise RuntimeError("local inventory SLat count mismatch")
    _write_json_atomic(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"LOCAL_INVENTORY_PASS={report_path}")
    return 0


def _parse_oss_listing(
    listing_path: Path, oss_root: str
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    prefix = oss_root.rstrip("/") + "/payload/"
    objects: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    directory_markers: list[str] = []
    seen_directory_markers: set[str] = set()
    with listing_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            parts = raw_line.strip().split()
            if not parts or not parts[-1].startswith("oss://"):
                continue
            url = parts[-1]
            if not url.startswith(prefix):
                continue
            if len(parts) < 4:
                raise RuntimeError(f"cannot parse ossutil ls line: {raw_line.rstrip()}")
            try:
                size = int(parts[-4])
            except ValueError as exc:
                raise RuntimeError(
                    f"cannot parse object size from ossutil ls line: {raw_line.rstrip()}"
                ) from exc
            etag = parts[-2].strip('"').lower()
            relative_path = url[len(prefix) :]
            # Recursive ossutil uploads may materialize POSIX directories as
            # zero-byte OSS objects.  They are transport metadata, not dataset
            # payload.  Only accept the exact, auditable directory-marker
            # representation; a non-empty or non-empty-MD5 key ending in '/'
            # remains an ordinary object and will fail as an unexpected extra.
            if (
                relative_path.endswith("/")
                and size == 0
                and etag == _EMPTY_OBJECT_ETAG
            ):
                if relative_path in seen_directory_markers:
                    duplicates.append(relative_path)
                else:
                    seen_directory_markers.add(relative_path)
                    directory_markers.append(relative_path)
                continue
            if relative_path in objects:
                duplicates.append(relative_path)
            objects[relative_path] = {"size": size, "etag": etag, "url": url}
    return objects, duplicates, directory_markers


def _limited(values: Iterable[Any], count: int = 20) -> list[Any]:
    result: list[Any] = []
    for value in values:
        result.append(value)
        if len(result) >= count:
            break
    return result


def audit_remote(args: argparse.Namespace) -> int:
    inventory_records = _load_inventory(args.inventory.resolve(strict=True))
    local: dict[str, dict[str, Any]] = {}
    for record in inventory_records:
        relative = record.get("relative_path")
        if not isinstance(relative, str):
            raise RuntimeError("local inventory record has no relative_path")
        if relative in local:
            raise RuntimeError(f"duplicate path in local inventory: {relative}")
        local[relative] = record
    remote, duplicates, directory_markers = _parse_oss_listing(
        args.remote_listing.resolve(strict=True), args.oss_root
    )

    local_paths = set(local)
    remote_paths = set(remote)
    missing = sorted(local_paths - remote_paths)
    extra = sorted(remote_paths - local_paths)
    size_mismatches: list[dict[str, Any]] = []
    hash_mismatches: list[dict[str, Any]] = []
    non_md5_etags: list[dict[str, Any]] = []
    remote_bytes = 0
    for path in sorted(local_paths & remote_paths):
        expected = local[path]
        actual = remote[path]
        remote_bytes += int(actual["size"])
        if int(expected["size"]) != int(actual["size"]):
            size_mismatches.append(
                {"path": path, "expected": expected["size"], "actual": actual["size"]}
            )
        etag = str(actual["etag"])
        if _ETAG_RE.fullmatch(etag) is None:
            non_md5_etags.append({"path": path, "etag": etag})
        elif str(expected["md5"]).lower() != etag:
            hash_mismatches.append(
                {"path": path, "expected_md5": expected["md5"], "remote_etag": etag}
            )

    passed = not (
        missing
        or extra
        or duplicates
        or size_mismatches
        or hash_mismatches
        or non_md5_etags
    )
    report = {
        "format": FORMAT_REMOTE_AUDIT,
        "passed": passed,
        "audited_at_utc": _utc_now(),
        "oss_root": args.oss_root.rstrip("/"),
        "local_inventory": str(args.inventory.resolve(strict=True)),
        "local_inventory_sha256": _sha256_file(args.inventory.resolve(strict=True)),
        "expected_object_count": EXPECTED_FILE_COUNT,
        "remote_object_count": len(remote),
        "remote_listing_entry_count": len(remote) + len(directory_markers),
        "validated_directory_marker_count": len(directory_markers),
        "validated_directory_marker_examples": directory_markers[:20],
        "expected_payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "remote_payload_bytes": remote_bytes,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "duplicate_count": len(duplicates),
        "size_mismatch_count": len(size_mismatches),
        "hash_mismatch_count": len(hash_mismatches),
        "non_md5_etag_count": len(non_md5_etags),
        "missing_examples": missing[:20],
        "extra_examples": extra[:20],
        "duplicate_examples": duplicates[:20],
        "size_mismatch_examples": size_mismatches[:20],
        "hash_mismatch_examples": hash_mismatches[:20],
        "non_md5_etag_examples": non_md5_etags[:20],
        "verification_contract": (
            "all payload files are below 100 MiB and are uploaded as single-part OSS "
            "objects; each remote ETag must equal the frozen local MD5; transport-only "
            "directory markers are accepted only when the key ends in '/', size is zero, "
            "and ETag is the MD5 of empty content"
        ),
    }
    _write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if not passed:
        print("REMOTE_INVENTORY_AUDIT_FAILED", file=sys.stderr)
        return 3
    print(f"REMOTE_INVENTORY_AUDIT_PASS={args.output.resolve()}")
    return 0


def complete(args: argparse.Namespace) -> int:
    local_report_path = args.local_report.resolve(strict=True)
    remote_report_path = args.remote_report.resolve(strict=True)
    local_report = _read_json(local_report_path)
    remote_report = _read_json(remote_report_path)
    if local_report.get("passed") is not True:
        raise RuntimeError("local inventory did not pass")
    if remote_report.get("passed") is not True:
        raise RuntimeError("remote inventory audit did not pass")
    completion = {
        "format": FORMAT_COMPLETION,
        "completed": True,
        "completed_at_utc": _utc_now(),
        "oss_root": args.oss_root.rstrip("/"),
        "pair_count": EXPECTED_PAIR_COUNT,
        "payload_object_count": EXPECTED_FILE_COUNT,
        "payload_bytes": EXPECTED_PAYLOAD_BYTES,
        "selection_file_sha256": EXPECTED_SELECTION_FILE_SHA256,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "uid_sha256": EXPECTED_UID_SHA256,
        "source_audit_file_sha256": EXPECTED_SOURCE_AUDIT_FILE_SHA256,
        "dataset_content_sha256": local_report["dataset_content_sha256"],
        "local_inventory_jsonl_sha256": local_report["inventory_jsonl_sha256"],
        "local_inventory_report_sha256": _sha256_file(local_report_path),
        "remote_inventory_audit_sha256": _sha256_file(remote_report_path),
        "remote_integrity": {
            "missing_count": remote_report["missing_count"],
            "extra_count": remote_report["extra_count"],
            "duplicate_count": remote_report["duplicate_count"],
            "size_mismatch_count": remote_report["size_mismatch_count"],
            "hash_mismatch_count": remote_report["hash_mismatch_count"],
            "non_md5_etag_count": remote_report["non_md5_etag_count"],
        },
    }
    _write_json_atomic(args.output.resolve(), completion)
    print(json.dumps(completion, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"COMPLETION_READY={args.output.resolve()}")
    return 0


def _path(value: str) -> Path:
    return Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="freeze local file hashes")
    prepare_parser.add_argument("--data_root", type=_path, required=True)
    prepare_parser.add_argument("--selection", type=_path, required=True)
    prepare_parser.add_argument("--source_audit", type=_path, required=True)
    prepare_parser.add_argument("--state_dir", type=_path, required=True)
    prepare_parser.add_argument("--hash_workers", type=int, default=4)
    prepare_parser.add_argument("--rehash", action="store_true")
    prepare_parser.set_defaults(func=prepare)

    audit_parser = subparsers.add_parser(
        "audit-remote", help="compare an ossutil ls listing with the frozen inventory"
    )
    audit_parser.add_argument("--inventory", type=_path, required=True)
    audit_parser.add_argument("--remote_listing", type=_path, required=True)
    audit_parser.add_argument("--oss_root", required=True)
    audit_parser.add_argument("--output", type=_path, required=True)
    audit_parser.set_defaults(func=audit_remote)

    complete_parser = subparsers.add_parser(
        "complete", help="write the local completion marker after all audits pass"
    )
    complete_parser.add_argument("--local_report", type=_path, required=True)
    complete_parser.add_argument("--remote_report", type=_path, required=True)
    complete_parser.add_argument("--oss_root", required=True)
    complete_parser.add_argument("--output", type=_path, required=True)
    complete_parser.set_defaults(func=complete)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "hash_workers", 1) < 1:
        parser.error("--hash_workers must be >= 1")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
