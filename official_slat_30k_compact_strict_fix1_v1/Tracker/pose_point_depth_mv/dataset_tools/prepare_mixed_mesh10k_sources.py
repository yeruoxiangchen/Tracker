#!/usr/bin/env python3
"""Audit, extract, freeze, and shard Objaverse + OmniObject3D mesh sources.

This stage deliberately stops before rendering.  It turns mutable downloader
outputs into immutable, deterministic mesh manifests that can be consumed by
``build_objaverse_multiview_sparse_data.py``.  OmniObject3D archives are
extracted without trusting archive paths or special files.  An Objaverse-only
plan may be emitted early; the final mixed plan still requires both sources.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


SCHEMA = "tracker.mixed_mesh10k_sources.v1"
EXTRACT_REPORT = "_EXTRACT_REPORT.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create immutable Objaverse/OmniObject3D mesh source manifests "
            "after the selected source downloads have completed."
        )
    )
    parser.add_argument(
        "--sources",
        choices=("both", "objaverse"),
        default="both",
        help=(
            "both creates the formal mixed source plan; objaverse creates an "
            "early subset plan that can be rendered while Omni downloads."
        ),
    )
    parser.add_argument("--objaverse_uids_file", required=True)
    parser.add_argument("--objaverse_limit", type=int, default=30000)
    parser.add_argument(
        "--objaverse_manifest",
        action="append",
        required=True,
        help="Repeat for every downloaded UID-to-GLB manifest.",
    )
    parser.add_argument("--min_objaverse_coverage", type=float, default=0.98)
    parser.add_argument("--objaverse_shards", type=int, default=16)

    parser.add_argument("--omni_remote_paths_file", required=True)
    parser.add_argument("--omni_archive_root", required=True)
    parser.add_argument("--omni_extract_root", required=True)
    parser.add_argument("--expected_omni_archives", type=int, default=216)
    parser.add_argument("--min_omni_objects", type=int, default=4000)
    parser.add_argument("--min_omni_obj_bytes", type=int, default=1024)
    parser.add_argument(
        "--max_omni_rejected_objects",
        type=int,
        default=64,
        help=(
            "Maximum number of individual Omni objects with missing, empty, "
            "or implausibly small Scan assets. Rejected objects are audited "
            "and excluded; unsafe archives and empty categories still fail."
        ),
    )
    parser.add_argument(
        "--max_omni_rejected_fraction",
        type=float,
        default=0.02,
        help=(
            "Maximum rejected Omni object fraction across all extracted "
            "categories. Both the absolute and fractional guards must pass."
        ),
    )
    parser.add_argument("--omni_extract_workers", type=int, default=2)
    parser.add_argument("--omni_shards", type=int, default=8)

    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_nonempty_lines(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [value for value in values if value]


def ensure_unique(values: Iterable[str], label: str) -> list[str]:
    result = list(values)
    if len(result) != len(set(result)):
        counts: dict[str, int] = {}
        for value in result:
            counts[value] = counts.get(value, 0) + 1
        repeated = sorted(value for value, count in counts.items() if count > 1)
        raise ValueError(f"{label} contains duplicate values: {repeated[:10]}")
    return result


def normalize_objaverse_uid(value: str) -> str:
    value = str(value).strip()
    return value[:-4] if value.endswith(".glb") else value


def parse_objaverse_manifest(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "samples" in payload:
        raise ValueError(f"{path}: expected a downloader UID-to-path JSON object")
    result: dict[str, str] = {}
    for raw_uid, raw_path in payload.items():
        uid = normalize_objaverse_uid(str(raw_uid))
        mesh_path = str(raw_path)
        if uid in result and result[uid] != mesh_path:
            raise ValueError(f"{path}: UID {uid} maps to multiple paths")
        result[uid] = mesh_path
    return result


def load_objaverse_sources(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    uid_file = Path(args.objaverse_uids_file).expanduser().resolve()
    all_uids = ensure_unique(
        (normalize_objaverse_uid(value) for value in read_nonempty_lines(uid_file)),
        "Objaverse UID file",
    )
    if args.objaverse_limit <= 0:
        raise ValueError("--objaverse_limit must be positive")
    if len(all_uids) < args.objaverse_limit:
        raise ValueError(
            f"Objaverse UID file has only {len(all_uids)} entries, "
            f"below requested limit {args.objaverse_limit}"
        )
    expected_uids = all_uids[: args.objaverse_limit]
    expected_set = set(expected_uids)

    combined: dict[str, str] = {}
    manifest_bindings = []
    for raw_path in args.objaverse_manifest:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing Objaverse manifest: {path}")
        rows = parse_objaverse_manifest(path)
        for uid, mesh_path in rows.items():
            previous = combined.get(uid)
            if previous is not None and Path(previous).resolve() != Path(mesh_path).resolve():
                raise ValueError(
                    f"Objaverse UID {uid} has conflicting paths: {previous} vs {mesh_path}"
                )
            combined[uid] = mesh_path
        manifest_bindings.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "entry_count": len(rows),
            }
        )

    frozen: dict[str, str] = {}
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    for uid in expected_uids:
        raw_path = combined.get(uid)
        if raw_path is None:
            missing.append(uid)
            continue
        mesh_path = Path(raw_path).expanduser().resolve()
        if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
            invalid.append({"uid": uid, "path": str(mesh_path)})
            continue
        frozen[f"objaverse_{uid}"] = str(mesh_path)

    coverage = len(frozen) / max(len(expected_uids), 1)
    if coverage < args.min_objaverse_coverage:
        raise RuntimeError(
            "Objaverse download coverage is below the hard gate: "
            f"{len(frozen)}/{len(expected_uids)}={coverage:.6f} "
            f"< {args.min_objaverse_coverage:.6f}; "
            f"missing={len(missing)} invalid={len(invalid)}"
        )

    extras = sorted(set(combined) - expected_set)
    report = {
        "uids_file": {
            "path": str(uid_file),
            "sha256": sha256_file(uid_file),
            "total_uid_count": len(all_uids),
        },
        "requested_count": len(expected_uids),
        "available_count": len(frozen),
        "coverage": coverage,
        "minimum_coverage": float(args.min_objaverse_coverage),
        "missing_count": len(missing),
        "missing_uids": missing,
        "invalid_count": len(invalid),
        "invalid": invalid,
        "ignored_extra_manifest_uid_count": len(extras),
        "manifests": manifest_bindings,
    }
    return frozen, report


class HashingReader:
    """Minimal binary reader that hashes every compressed byte tarfile reads."""

    def __init__(self, handle: BinaryIO):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.handle.read(size)
        if data:
            self.digest.update(data)
        return data

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def safe_member_path(name: str) -> Path | None:
    pure = PurePosixPath(name)
    if pure.is_absolute():
        raise ValueError(f"archive contains absolute path: {name}")
    parts = [part for part in pure.parts if part not in ("", ".")]
    if not parts:
        return None
    if any(part == ".." for part in parts):
        raise ValueError(f"archive path escapes extraction root: {name}")
    return Path(*parts)


def audit_omni_category(
    category_dir: Path,
    *,
    min_obj_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not category_dir.is_dir():
        raise RuntimeError(f"Omni category directory is missing: {category_dir}")
    objects: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for object_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
        scan_dir = object_dir / "Scan"
        obj_path = scan_dir / "Scan.obj"
        mtl_path = scan_dir / "Scan.mtl"
        texture_path = scan_dir / "Scan.jpg"
        missing_or_empty = [
            str(path.relative_to(category_dir))
            for path in (obj_path, mtl_path, texture_path)
            if not path.is_file() or path.stat().st_size <= 0
        ]
        if missing_or_empty:
            rejected.append(
                {
                    "object_name": object_dir.name,
                    "reason": "missing_or_empty_scan_assets",
                    "assets": missing_or_empty,
                }
            )
            continue
        if obj_path.stat().st_size < min_obj_bytes:
            rejected.append(
                {
                    "object_name": object_dir.name,
                    "reason": "implausibly_small_obj",
                    "asset": str(obj_path.relative_to(category_dir)),
                    "obj_bytes": obj_path.stat().st_size,
                    "minimum_obj_bytes": min_obj_bytes,
                }
            )
            continue
        objects.append(
            {
                "object_name": object_dir.name,
                "obj_path": str(obj_path.resolve()),
                "obj_bytes": obj_path.stat().st_size,
                "mtl_bytes": mtl_path.stat().st_size,
                "texture_bytes": texture_path.stat().st_size,
            }
        )
    if not objects:
        raise RuntimeError(
            f"Omni category contains no valid objects: {category_dir}; "
            f"rejected={len(rejected)} first={rejected[:3]}"
        )
    return objects, rejected


def rejection_sha256(rejected: list[dict[str, Any]]) -> str:
    return sha256_json(
        sorted(
            rejected,
            key=lambda row: (
                str(row.get("object_name", "")),
                str(row.get("reason", "")),
            ),
        )
    )


def extract_one_omni_archive(task: dict[str, Any]) -> dict[str, Any]:
    archive = Path(task["archive"])
    extract_root = Path(task["extract_root"])
    category = str(task["category"])
    min_obj_bytes = int(task["min_obj_bytes"])
    categories_root = extract_root / "categories"
    final_dir = categories_root / category

    if final_dir.exists():
        report_path = final_dir / EXTRACT_REPORT
        if not report_path.is_file():
            raise RuntimeError(
                f"partial Omni extraction exists without {EXTRACT_REPORT}: {final_dir}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        stat = archive.stat()
        if (
            report.get("archive_bytes") != stat.st_size
            or report.get("archive_mtime_ns") != stat.st_mtime_ns
        ):
            raise RuntimeError(f"archive changed after extraction: {archive}")
        objects, rejected = audit_omni_category(
            final_dir,
            min_obj_bytes=min_obj_bytes,
        )
        if len(objects) != report.get("object_count"):
            raise RuntimeError(f"extracted Omni category inventory changed: {final_dir}")
        expected_rejected_count = int(report.get("rejected_object_count", 0))
        if len(rejected) != expected_rejected_count:
            raise RuntimeError(
                f"extracted Omni category rejection inventory changed: {final_dir}; "
                f"actual={len(rejected)} expected={expected_rejected_count}"
            )
        expected_rejected_sha256 = report.get("rejected_objects_sha256")
        if (
            expected_rejected_sha256 is not None
            and rejection_sha256(rejected) != expected_rejected_sha256
        ):
            raise RuntimeError(
                f"extracted Omni category rejected objects changed: {final_dir}"
            )
        return {**report, "reused": True}

    categories_root.mkdir(parents=True, exist_ok=True)
    staging_root = extract_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{category}.", dir=staging_root))

    try:
        with archive.open("rb") as raw:
            hashing_reader = HashingReader(raw)
            with tarfile.open(fileobj=hashing_reader, mode="r|gz") as tar:
                for member in tar:
                    relative = safe_member_path(member.name)
                    if relative is None:
                        continue
                    destination = staging / relative
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        raise ValueError(
                            f"archive contains forbidden non-regular member: "
                            f"{member.name} type={member.type!r}"
                        )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"cannot read archive member: {member.name}")
                    with source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)

        objects, rejected = audit_omni_category(
            staging,
            min_obj_bytes=min_obj_bytes,
        )
        stat = archive.stat()
        report = {
            "category": category,
            "archive": str(archive.resolve()),
            "archive_bytes": stat.st_size,
            "archive_mtime_ns": stat.st_mtime_ns,
            "archive_sha256": hashing_reader.hexdigest(),
            "discovered_object_count": len(objects) + len(rejected),
            "object_count": len(objects),
            "rejected_object_count": len(rejected),
            "rejected_objects": rejected,
            "rejected_objects_sha256": rejection_sha256(rejected),
            "objects_sha256": sha256_json(
                [
                    {
                        "object_name": row["object_name"],
                        "obj_bytes": row["obj_bytes"],
                        "mtl_bytes": row["mtl_bytes"],
                        "texture_bytes": row["texture_bytes"],
                    }
                    for row in objects
                ]
            ),
            "reused": False,
        }
        write_json(staging / EXTRACT_REPORT, report)
        os.replace(staging, final_dir)
        return report
    except BaseException:
        # Preserve the staging directory for diagnosis; a subsequent run uses a
        # new uniquely named staging path and never trusts this partial output.
        raise


def expected_omni_archives(args: argparse.Namespace) -> list[dict[str, Any]]:
    remote_path_file = Path(args.omni_remote_paths_file).expanduser().resolve()
    remote_paths = ensure_unique(
        read_nonempty_lines(remote_path_file),
        "Omni remote path list",
    )
    if (
        args.expected_omni_archives > 0
        and len(remote_paths) != args.expected_omni_archives
    ):
        raise RuntimeError(
            f"Omni remote path list has {len(remote_paths)} entries, "
            f"expected exactly {args.expected_omni_archives}"
        )
    archive_root = Path(args.omni_archive_root).expanduser().resolve()
    rows = []
    seen_names: set[str] = set()
    missing = []
    for remote_path in remote_paths:
        name = PurePosixPath(remote_path).name
        if not name.endswith(".tar.gz"):
            raise ValueError(f"unexpected Omni remote archive name: {remote_path}")
        if name in seen_names:
            raise ValueError(f"duplicate Omni archive basename: {name}")
        seen_names.add(name)
        archive = archive_root / name
        if not archive.is_file() or archive.stat().st_size <= 0:
            missing.append(str(archive))
            continue
        rows.append(
            {
                "category": name[: -len(".tar.gz")],
                "remote_path": remote_path,
                "archive": str(archive),
                "archive_bytes": archive.stat().st_size,
            }
        )
    if missing:
        raise RuntimeError(
            f"Omni download is incomplete: {len(missing)}/{len(remote_paths)} "
            f"archives missing or empty; first={missing[:5]}"
        )
    return rows


def load_omni_sources(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    archive_rows = expected_omni_archives(args)
    extract_root = Path(args.omni_extract_root).expanduser().resolve()
    tasks = [
        {
            "archive": row["archive"],
            "extract_root": str(extract_root),
            "category": row["category"],
            "min_obj_bytes": args.min_omni_obj_bytes,
        }
        for row in archive_rows
    ]

    workers = max(1, int(args.omni_extract_workers))
    if args.max_omni_rejected_objects < 0:
        raise ValueError("--max_omni_rejected_objects must be non-negative")
    if not 0.0 <= args.max_omni_rejected_fraction <= 1.0:
        raise ValueError("--max_omni_rejected_fraction must be in [0, 1]")
    reports: list[dict[str, Any]] = []
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            report = extract_one_omni_archive(task)
            reports.append(report)
            print(
                f"[omni_extract] {index}/{len(tasks)} "
                f"{report['category']} objects={report['object_count']} "
                f"reused={report['reused']}",
                flush=True,
            )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(extract_one_omni_archive, task): task
                for task in tasks
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_task):
                report = future.result()
                reports.append(report)
                completed += 1
                print(
                    f"[omni_extract] {completed}/{len(tasks)} "
                    f"{report['category']} objects={report['object_count']} "
                    f"reused={report['reused']}",
                    flush=True,
                )

    reports.sort(key=lambda row: row["category"])
    categories_root = extract_root / "categories"
    frozen: dict[str, str] = {}
    object_rows: list[dict[str, Any]] = []
    for report in reports:
        category_dir = categories_root / str(report["category"])
        rows, rejected = audit_omni_category(
            category_dir,
            min_obj_bytes=args.min_omni_obj_bytes,
        )
        expected_rejected_count = int(report.get("rejected_object_count", 0))
        if len(rejected) != expected_rejected_count:
            raise RuntimeError(
                f"Omni rejection inventory changed after extraction: "
                f"{category_dir}; actual={len(rejected)} "
                f"expected={expected_rejected_count}"
            )
        for row in rows:
            object_name = str(row["object_name"])
            uid = f"omni_{object_name}"
            if uid in frozen:
                raise RuntimeError(f"duplicate Omni object UID after extraction: {uid}")
            frozen[uid] = str(row["obj_path"])
            object_rows.append(
                {
                    "uid": uid,
                    "category": report["category"],
                    **row,
                }
            )

    rejected_rows = [
        {
            "category": str(report["category"]),
            **row,
        }
        for report in reports
        for row in report.get("rejected_objects", [])
    ]
    discovered_count = len(frozen) + len(rejected_rows)
    rejected_fraction = len(rejected_rows) / max(discovered_count, 1)
    if len(rejected_rows) > args.max_omni_rejected_objects:
        raise RuntimeError(
            "too many individual Omni objects were rejected: "
            f"{len(rejected_rows)} > {args.max_omni_rejected_objects}; "
            f"first={rejected_rows[:5]}"
        )
    if rejected_fraction > args.max_omni_rejected_fraction:
        raise RuntimeError(
            "Omni rejected object fraction is above the hard gate: "
            f"{len(rejected_rows)}/{discovered_count}={rejected_fraction:.6f} "
            f"> {args.max_omni_rejected_fraction:.6f}; "
            f"first={rejected_rows[:5]}"
        )
    if len(frozen) < args.min_omni_objects:
        raise RuntimeError(
            f"only {len(frozen)} valid Omni objects were extracted, "
            f"below hard minimum {args.min_omni_objects}"
        )
    remote_file = Path(args.omni_remote_paths_file).expanduser().resolve()
    report = {
        "remote_paths_file": {
            "path": str(remote_file),
            "sha256": sha256_file(remote_file),
            "entry_count": len(archive_rows),
        },
        "archive_root": str(Path(args.omni_archive_root).expanduser().resolve()),
        "extract_root": str(extract_root),
        "archive_count": len(reports),
        "discovered_object_count": discovered_count,
        "object_count": len(frozen),
        "minimum_object_count": int(args.min_omni_objects),
        "rejected_object_count": len(rejected_rows),
        "rejected_object_fraction": rejected_fraction,
        "maximum_rejected_object_count": int(args.max_omni_rejected_objects),
        "maximum_rejected_object_fraction": float(
            args.max_omni_rejected_fraction
        ),
        "rejected_objects": rejected_rows,
        "rejected_objects_sha256": rejection_sha256(rejected_rows),
        "archive_reports": reports,
        "objects_sha256": sha256_json(
            [
                {
                    "uid": row["uid"],
                    "category": row["category"],
                    "obj_path": row["obj_path"],
                    "obj_bytes": row["obj_bytes"],
                }
                for row in sorted(object_rows, key=lambda value: value["uid"])
            ]
        ),
    }
    return frozen, report


def shard_index(uid: str, count: int) -> int:
    digest = hashlib.sha256(uid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % count


def write_shards(
    staging: Path,
    source_name: str,
    values: dict[str, str],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError(f"{source_name} shard count must be positive")
    shards: list[dict[str, str]] = [dict() for _ in range(count)]
    for uid, mesh_path in sorted(values.items()):
        shards[shard_index(uid, count)][uid] = mesh_path

    reports = []
    for index, shard in enumerate(shards):
        relative = Path(f"{source_name}_shards") / f"shard_{index:03d}.json"
        path = staging / relative
        write_json(path, shard)
        reports.append(
            {
                "index": index,
                "path": str(relative),
                "object_count": len(shard),
                "uid_sha256": sha256_json(sorted(shard)),
                "manifest_sha256": sha256_file(path),
            }
        )
    # SHA-256 modulo assignment is deterministic but not exactly balanced.
    # Each shard count follows a binomial marginal with
    # sigma=sqrt(N * p * (1-p)), p=1/count.  The old guard compared the
    # max-min range against 3% of the mean; for 30k objects / 16 shards that
    # allowed only 57 objects even though the expected random range is already
    # of that order.  Guard the maximum absolute deviation with a conservative
    # 5-sigma bound instead.  This still catches a broken/non-uniform sharder
    # while avoiding false failures from ordinary hash variance.
    object_counts = [int(row["object_count"]) for row in reports]
    expected = len(values) / count
    probability = 1.0 / count
    sigma = math.sqrt(len(values) * probability * (1.0 - probability))
    max_abs_deviation = max(abs(value - expected) for value in object_counts)
    tolerance = max(8, int(math.ceil(5.0 * sigma)))
    if max_abs_deviation > tolerance:
        raise RuntimeError(
            f"{source_name} hash shards are unexpectedly imbalanced: "
            f"counts={object_counts}, expected={expected:.3f}, "
            f"max_abs_deviation={max_abs_deviation:.3f}, "
            f"five_sigma_tolerance={tolerance}"
        )
    return reports


def main() -> None:
    args = parse_args()
    selected_sources = (
        ["objaverse", "omni"] if args.sources == "both" else ["objaverse"]
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        report_path = output_dir / "source_report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("schema") != SCHEMA or report.get("passed") is not True:
                raise RuntimeError(f"existing source freeze is not reusable: {output_dir}")
            report_sources = report.get("sources", ["objaverse", "omni"])
            if report_sources != selected_sources:
                raise RuntimeError(
                    f"existing source freeze has sources={report_sources}, "
                    f"requested={selected_sources}: {output_dir}"
                )
            summary = {
                "reused": True,
                "output_dir": str(output_dir),
                "sources": selected_sources,
                "objaverse_objects": report["objaverse"]["available_count"],
            }
            if "omni" in report:
                summary["omni_objects"] = report["omni"]["object_count"]
            print(
                json.dumps(summary, indent=2),
                flush=True,
            )
            return
        raise RuntimeError(
            f"output directory already exists without a complete report: {output_dir}"
        )

    objaverse, objaverse_report = load_objaverse_sources(args)
    print(
        f"[objaverse_audit] available={len(objaverse)} "
        f"coverage={objaverse_report['coverage']:.6f}",
        flush=True,
    )
    omni: dict[str, str] | None = None
    omni_report: dict[str, Any] | None = None
    if args.sources == "both":
        omni, omni_report = load_omni_sources(args)
        print(f"[omni_audit] available={len(omni)}", flush=True)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging.",
            dir=output_dir.parent,
        )
    )
    try:
        objaverse_manifest = staging / "objaverse_meshes.json"
        write_json(objaverse_manifest, objaverse)
        objaverse_shards = write_shards(
            staging,
            "objaverse",
            objaverse,
            args.objaverse_shards,
        )

        build_plan = {
            "schema": SCHEMA,
            "sources": selected_sources,
            "output_dir": str(output_dir),
            "shard_policy": "sha256(uid) mod shard_count",
            "objaverse": {
                "object_count": len(objaverse),
                "manifest": "objaverse_meshes.json",
                "manifest_sha256": sha256_file(objaverse_manifest),
                "shard_count": args.objaverse_shards,
                "shards": objaverse_shards,
            },
        }
        if omni is not None:
            omni_manifest = staging / "omni_meshes.json"
            write_json(omni_manifest, omni)
            omni_shards = write_shards(
                staging,
                "omni",
                omni,
                args.omni_shards,
            )
            build_plan["omni"] = {
                "object_count": len(omni),
                "manifest": "omni_meshes.json",
                "manifest_sha256": sha256_file(omni_manifest),
                "shard_count": args.omni_shards,
                "shards": omni_shards,
            }
        write_json(staging / "build_plan.json", build_plan)
        report = {
            "schema": SCHEMA,
            "passed": True,
            "sources": selected_sources,
            "formal_mixed_source_plan": args.sources == "both",
            "output_dir": str(output_dir),
            "code_binding": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "objaverse": objaverse_report,
            "build_plan_sha256": sha256_file(staging / "build_plan.json"),
            "hard_guards": {
                "objaverse_coverage": True,
            },
        }
        if omni is not None and omni_report is not None:
            report["omni"] = omni_report
            report["hard_guards"].update(
                {
                    "all_expected_omni_archives_present": True,
                    "omni_archives_safely_extracted": True,
                    "minimum_omni_object_count": True,
                    "omni_rejected_objects_within_budget": (
                        omni_report["rejected_object_count"]
                        <= omni_report["maximum_rejected_object_count"]
                        and omni_report["rejected_object_fraction"]
                        <= omni_report["maximum_rejected_object_fraction"]
                    ),
                    "unique_prefixed_object_uids": not bool(
                        set(objaverse) & set(omni)
                    ),
                }
            )
        if not all(report["hard_guards"].values()):
            raise RuntimeError(f"source hard guard failed: {report['hard_guards']}")
        write_json(staging / "source_report.json", report)
        os.replace(staging, output_dir)
    except BaseException:
        # Preserve staging for diagnosis.  The immutable destination is never
        # exposed until every manifest and report has been written.
        raise

    print(
        json.dumps(
            {
                "reused": False,
                "passed": True,
                "sources": selected_sources,
                "output_dir": str(output_dir),
                "objaverse_objects": len(objaverse),
                "objaverse_shards": args.objaverse_shards,
                **(
                    {
                        "omni_objects": len(omni),
                        "omni_shards": args.omni_shards,
                    }
                    if omni is not None
                    else {}
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
