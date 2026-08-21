#!/usr/bin/env python3
"""Materialize and audit a disjoint 5K + 25K ProObjaverse training root.

``link-base`` maps a completed frozen subset into a new data root with hard
links.  It never copies the large render/latent payloads and never mutates the
source subset.

``audit`` binds multiple frozen selections, verifies exact file sizes and UID
disjointness, rejects files outside the selected 30K union, samples tar/NPZ
schemas, and freezes a combined selection for later training provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any

import numpy as np
from PIL import Image


SELECTION_FORMAT = "reconviagen.proobjaverse_paired_subset.v1"
LINK_REPORT_FORMAT = "reconviagen.proobjaverse_hardlink_materialization.v1"
COMBINED_FORMAT = "reconviagen.proobjaverse_paired_combined.v1"
AUDIT_FORMAT = "reconviagen.proobjaverse_combined_audit.v1"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_selection(path: Path) -> dict[str, Any]:
    selection = read_json(path)
    body = {
        key: value
        for key, value in selection.items()
        if key not in {"created_at_utc", "selection_sha256"}
    }
    if selection.get("format") != SELECTION_FORMAT:
        raise RuntimeError(f"unsupported selection format: {path}")
    if selection.get("selection_sha256") != canonical_sha256(body):
        raise RuntimeError(f"selection identity hash is invalid: {path}")
    rows = list(selection.get("selected", []))
    uids = [str(row.get("uid", "")) for row in rows]
    if (
        len(rows) != int(selection.get("selected_pair_count", -1))
        or not all(uids)
        or len(uids) != len(set(uids))
    ):
        raise RuntimeError(f"selection UID accounting is invalid: {path}")
    return selection


def safe_local_path(root: Path, remote_path: str) -> Path:
    relative = PurePosixPath(remote_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe repository path: {remote_path!r}")
    return root.joinpath(*relative.parts)


def validate_row(row: dict[str, Any]) -> dict[str, Any]:
    uid = str(row.get("uid", ""))
    shard = str(row.get("shard", ""))
    render = dict(row.get("render", {}))
    slat = dict(row.get("slat", {}))
    expected_render = f"renders_random_env/{shard}/{uid}.tar"
    expected_slat = f"lh-slats/{shard}/{uid}.npz"
    if not uid or not shard.startswith("shard-"):
        raise ValueError(f"invalid UID/shard row: {row}")
    if render.get("path") != expected_render or slat.get("path") != expected_slat:
        raise ValueError(f"UID/path binding differs for uid={uid}")
    if int(render.get("size", 0)) <= 0 or int(slat.get("size", 0)) <= 0:
        raise ValueError(f"non-positive official file size for uid={uid}")
    return {
        "uid": uid,
        "shard": shard,
        "render": {"path": expected_render, "size": int(render["size"])},
        "slat": {"path": expected_slat, "size": int(slat["size"])},
    }


def _freeze_json(path: Path, payload: dict[str, Any], label: str) -> None:
    if path.exists():
        existing = read_json(path)
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite changed {label}: {path}")
        return
    atomic_json(path, payload)


def link_base(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    selection_path = Path(args.selection).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if source_root == target_root:
        raise ValueError("source-root and target-root must differ")
    selection = load_selection(selection_path)
    rows = [validate_row(row) for row in selection["selected"]]
    if args.expected_count and len(rows) != args.expected_count:
        raise RuntimeError(
            f"base selection count differs: got={len(rows)} expected={args.expected_count}"
        )
    target_root.mkdir(parents=True, exist_ok=True)
    linked_files = 0
    reused_files = 0
    linked_bytes = 0
    for index, row in enumerate(rows, start=1):
        for kind in ("render", "slat"):
            file_row = row[kind]
            source = safe_local_path(source_root, file_row["path"])
            target = safe_local_path(target_root, file_row["path"])
            expected_size = int(file_row["size"])
            if not source.is_file() or source.stat().st_size != expected_size:
                raise RuntimeError(
                    f"base subset is incomplete or changed: {source} "
                    f"expected_size={expected_size}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not target.is_file() or target.stat().st_size != expected_size:
                    raise RuntimeError(f"partial/conflicting target file: {target}")
                source_stat = source.stat()
                target_stat = target.stat()
                if (source_stat.st_dev, source_stat.st_ino) != (
                    target_stat.st_dev,
                    target_stat.st_ino,
                ):
                    raise RuntimeError(
                        f"existing base target is not the expected hard link: {target}"
                    )
                reused_files += 1
            else:
                try:
                    os.link(source, target)
                except OSError as error:
                    raise RuntimeError(
                        f"cannot hard-link {source} to {target}; both roots must be "
                        "on the same filesystem"
                    ) from error
                linked_files += 1
            linked_bytes += expected_size
        if index % 500 == 0 or index == len(rows):
            print(
                f"[link-base] pairs={index}/{len(rows)} "
                f"new_files={linked_files} reused_files={reused_files}",
                flush=True,
            )
    report = {
        "format": LINK_REPORT_FORMAT,
        "passed": True,
        "source_root": str(source_root),
        "target_root": str(target_root),
        "selection": str(selection_path),
        "selection_file_sha256": sha256_file(selection_path),
        "selection_sha256": str(selection["selection_sha256"]),
        "pair_count": len(rows),
        "file_count": len(rows) * 2,
        "new_hardlink_count": linked_files,
        "reused_hardlink_count": reused_files,
        "logical_bytes": linked_bytes,
        "additional_payload_bytes": 0,
        "contract": "target payloads share device/inode with the frozen source 5K",
    }
    body = dict(report)
    # Counts of newly-created versus reused links depend on resume history and
    # are therefore excluded from the durable identity.
    identity = {
        key: value
        for key, value in body.items()
        if key not in {"new_hardlink_count", "reused_hardlink_count"}
    }
    report["report_sha256"] = canonical_sha256(identity)
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def _schema_audit(row: dict[str, Any], data_root: Path) -> dict[str, Any]:
    uid = row["uid"]
    slat_path = safe_local_path(data_root, row["slat"]["path"])
    render_path = safe_local_path(data_root, row["render"]["path"])
    with np.load(slat_path, allow_pickle=False) as payload:
        if set(payload.files) != {"coords", "feats"}:
            raise ValueError(f"unexpected SLat fields={payload.files}: {slat_path}")
        coords = np.asarray(payload["coords"])
        feats = np.asarray(payload["feats"])
    if (
        coords.ndim != 2
        or coords.shape[1] != 3
        or coords.dtype != np.uint8
        or feats.shape != (len(coords), 8)
        or feats.dtype != np.float32
        or not len(coords)
        or not np.isfinite(feats).all()
        or int(coords.min()) < 0
        or int(coords.max()) >= 64
    ):
        raise ValueError(
            f"invalid TRELLIS.1 SLat schema for uid={uid}: "
            f"coords={coords.shape}/{coords.dtype} feats={feats.shape}/{feats.dtype}"
        )
    with tarfile.open(render_path, "r") as archive:
        json_members = sorted(
            (member for member in archive.getmembers() if member.name.endswith(".json")),
            key=lambda member: member.name,
        )
        image_members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.name.endswith(".rgba.webp")
            ),
            key=lambda member: member.name,
        )
        json_ids = {Path(member.name).stem for member in json_members}
        image_ids = {
            Path(member.name).name.removesuffix(".rgba.webp")
            for member in image_members
        }
        if not json_members or json_ids != image_ids:
            raise ValueError(f"render tar has incomplete view pairs: {render_path}")
        if any(
            not Path(member.name).parts or Path(member.name).parts[0] != uid
            for member in [*json_members, *image_members]
        ):
            raise ValueError(f"render tar UID layout differs: {render_path}")
        meta_handle = archive.extractfile(json_members[0])
        image_handle = archive.extractfile(image_members[0])
        if meta_handle is None or image_handle is None:
            raise RuntimeError(f"cannot read sampled render tar: {render_path}")
        meta = json.load(meta_handle)
        intrinsic = np.asarray(meta.get("intrinsic"), dtype=np.float64)
        extrinsic = np.asarray(meta.get("extrinsic"), dtype=np.float64)
        if (
            intrinsic.shape != (3, 3)
            or extrinsic.shape != (4, 4)
            or not np.isfinite(intrinsic).all()
            or not np.isfinite(extrinsic).all()
        ):
            raise ValueError(f"invalid sampled camera metadata: {render_path}")
        with Image.open(io.BytesIO(image_handle.read())) as image:
            image.load()
            image_size = list(image.size)
            image_mode = image.mode
    return {
        "uid": uid,
        "shard": row["shard"],
        "slat_token_count": int(len(coords)),
        "view_count": len(json_members),
        "sample_image_size": image_size,
        "sample_image_mode": image_mode,
    }


def _filesystem_uid_map(root: Path, category: str, suffix: str) -> dict[str, str]:
    category_root = root / category
    if not category_root.is_dir():
        raise FileNotFoundError(category_root)
    result: dict[str, str] = {}
    for path in sorted(category_root.glob(f"shard-*/*{suffix}")):
        uid = path.name[: -len(suffix)]
        if uid in result:
            raise RuntimeError(
                f"duplicate UID in {category}: uid={uid} paths={result[uid]},{path}"
            )
        result[uid] = str(path.resolve())
    return result


def audit_combined(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    selection_paths = [Path(value).expanduser().resolve() for value in args.selection]
    expected_counts = [int(value) for value in args.expected_counts.split(",") if value]
    if len(selection_paths) < 2 or len(selection_paths) != len(expected_counts):
        raise ValueError("audit requires matching --selection entries and --expected-counts")
    selections = [load_selection(path) for path in selection_paths]
    repo_ids = {str(selection["repo_id"]) for selection in selections}
    revisions = {str(selection["revision"]) for selection in selections}
    if len(repo_ids) != 1 or len(revisions) != 1:
        raise RuntimeError("component selections use different repositories or revisions")

    union: dict[str, dict[str, Any]] = {}
    component_bindings = []
    previous_selection_shas: set[str] = set()
    expected_bytes = 0
    for component_index, (path, selection, expected_count) in enumerate(
        zip(selection_paths, selections, expected_counts)
    ):
        rows = [validate_row(row) for row in selection["selected"]]
        if len(rows) != expected_count:
            raise RuntimeError(
                f"component count differs for {path}: got={len(rows)} expected={expected_count}"
            )
        current_uids = {row["uid"] for row in rows}
        overlap = set(union) & current_uids
        if overlap:
            raise RuntimeError(
                f"component selections overlap by {len(overlap)} UIDs; first={min(overlap)}"
            )
        if component_index > 0:
            excluded_shas = {
                str(binding.get("selection_sha256"))
                for binding in selection.get("excluded_selection_bindings", [])
            }
            missing_exclusion = previous_selection_shas - excluded_shas
            if missing_exclusion:
                raise RuntimeError(
                    f"extension selection is not hard-bound to all previous selections: "
                    f"missing={sorted(missing_exclusion)}"
                )
        for selection_rank, row in enumerate(rows):
            row["component_index"] = component_index
            row["component_selection_rank"] = selection_rank
            row["component_selection_sha256"] = str(selection["selection_sha256"])
            union[row["uid"]] = row
            expected_bytes += int(row["render"]["size"]) + int(row["slat"]["size"])
        previous_selection_shas.add(str(selection["selection_sha256"]))
        component_bindings.append(
            {
                "path": str(path),
                "file_sha256": sha256_file(path),
                "selection_sha256": str(selection["selection_sha256"]),
                "pair_count": len(rows),
                "uid_sha256": canonical_sha256(sorted(current_uids)),
            }
        )
    if len(union) != int(args.expected_total):
        raise RuntimeError(
            f"combined pair count differs: got={len(union)} expected={args.expected_total}"
        )

    verified_bytes = 0
    for index, row in enumerate(union.values(), start=1):
        for kind in ("render", "slat"):
            path = safe_local_path(data_root, row[kind]["path"])
            expected_size = int(row[kind]["size"])
            if not path.is_file() or path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"selected payload is missing or changed: {path} expected={expected_size}"
                )
            verified_bytes += expected_size
        if index % 2500 == 0 or index == len(union):
            print(f"[audit:size] pairs={index}/{len(union)}", flush=True)
    if verified_bytes != expected_bytes:
        raise AssertionError("verified byte accounting differs")

    render_map = _filesystem_uid_map(data_root, "renders_random_env", ".tar")
    slat_map = _filesystem_uid_map(data_root, "lh-slats", ".npz")
    selected_uids = set(union)
    if set(render_map) != selected_uids or set(slat_map) != selected_uids:
        raise RuntimeError(
            "30K data root contains missing or unselected payloads: "
            f"render_missing={len(selected_uids - set(render_map))} "
            f"render_extra={len(set(render_map) - selected_uids)} "
            f"slat_missing={len(selected_uids - set(slat_map))} "
            f"slat_extra={len(set(slat_map) - selected_uids)}"
        )

    schema_count = min(int(args.schema_samples), len(union))
    if schema_count <= 0:
        raise ValueError("schema-samples must be positive")
    schema_rows = sorted(
        union.values(),
        key=lambda row: hashlib.sha256(
            f"{int(args.seed)}:{row['uid']}".encode("ascii")
        ).hexdigest(),
    )[:schema_count]
    schema_records = []
    for index, row in enumerate(schema_rows, start=1):
        schema_records.append(_schema_audit(row, data_root))
        print(
            f"[audit:schema] {index}/{schema_count} uid={row['uid']}",
            flush=True,
        )

    combined_rows = [union[uid] for uid in sorted(union)]
    combined: dict[str, Any] = {
        "format": COMBINED_FORMAT,
        "formal": False,
        "data_root": str(data_root),
        "repository": {
            "repo_id": next(iter(repo_ids)),
            "revision": next(iter(revisions)),
        },
        "component_selections": component_bindings,
        "pair_count": len(combined_rows),
        "expected_payload_bytes": expected_bytes,
        "uid_sha256": canonical_sha256([row["uid"] for row in combined_rows]),
        "training_layout": {
            "renders": "renders_random_env/shard-*/{uid}.tar",
            "slats": "lh-slats/shard-*/{uid}.npz",
            "reconviagen_tar_dataset_directly_readable": True,
        },
        "target_contract": {
            "family": "TRELLIS.1 / ReconViaGen original",
            "slat_channels": 8,
            "slat_grid_resolution": 64,
            "trellis2_shape_slat_compatible": False,
            "guard": (
                "These official lh-slats directly supervise ReconViaGen/TRELLIS.1. "
                "They are not 32-channel TRELLIS.2 Shape-SLat targets."
            ),
        },
        "selected": combined_rows,
    }
    combined["combined_selection_sha256"] = canonical_sha256(combined)
    combined_path = output_dir / "combined_selection_30k.json"
    _freeze_json(combined_path, combined, "combined 30K selection")

    token_counts = np.asarray(
        [row["slat_token_count"] for row in schema_records], dtype=np.int64
    )
    view_counts = np.asarray(
        [row["view_count"] for row in schema_records], dtype=np.int64
    )
    report: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "passed": True,
        "formal": False,
        "data_root": str(data_root),
        "combined_selection": str(combined_path),
        "combined_selection_file_sha256": sha256_file(combined_path),
        "combined_selection_sha256": combined["combined_selection_sha256"],
        "pair_count": len(union),
        "render_count": len(render_map),
        "slat_count": len(slat_map),
        "verified_payload_bytes": verified_bytes,
        "component_pair_counts": expected_counts,
        "component_overlap_count": 0,
        "unselected_payload_count": 0,
        "schema_sample_count": schema_count,
        "schema_summary": {
            "slat_token_min": int(token_counts.min()),
            "slat_token_median": float(np.median(token_counts)),
            "slat_token_max": int(token_counts.max()),
            "view_count_min": int(view_counts.min()),
            "view_count_median": float(np.median(view_counts)),
            "view_count_max": int(view_counts.max()),
        },
        "schema_records": schema_records,
        "training_guard": combined["target_contract"]["guard"],
    }
    report["report_sha256"] = canonical_sha256(report)
    report_path = output_dir / "audit_report.json"
    _freeze_json(report_path, report, "combined 30K audit")
    print(
        json.dumps(
            {
                "passed": True,
                "pairs": len(union),
                "render_count": len(render_map),
                "slat_count": len(slat_map),
                "logical_gib": verified_bytes / (1024**3),
                "combined_selection": str(combined_path),
                "audit_report": str(report_path),
                "report_sha256": report["report_sha256"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    link = commands.add_parser("link-base", help="Hard-link a completed base subset.")
    link.add_argument("--source-root", required=True)
    link.add_argument("--target-root", required=True)
    link.add_argument("--selection", required=True)
    link.add_argument("--expected-count", type=int, default=5000)
    link.add_argument("--report", required=True)
    link.set_defaults(handler=link_base)

    audit = commands.add_parser("audit", help="Freeze and audit an exact combined root.")
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--selection", action="append", required=True)
    audit.add_argument("--expected-counts", default="5000,25000")
    audit.add_argument("--expected-total", type=int, default=30000)
    audit.add_argument("--schema-samples", type=int, default=64)
    audit.add_argument("--seed", type=int, default=20260813)
    audit.add_argument("--output-dir", required=True)
    audit.set_defaults(handler=audit_combined)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
