#!/usr/bin/env python3
"""Freeze and audit a downloaded official ProObjaverse SLat target pool.

The protocol deliberately separates target-domain diagnosis from Native-SS
support prediction.  Selection follows the already frozen 5K download list;
filesystem enumeration order is never used as an experimental identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
from typing import Any

import numpy as np


SELECTION_FORMAT = "reconviagen.proobjaverse_paired_subset.v1"
PROTOCOL_FORMAT = "pose_point_depth_mv.proobjaverse_official_slat_protocol.v1"
SPLIT_FORMAT = "pose_point_depth_mv.proobjaverse_official_slat_split.v1"
AUDIT_FORMAT = "pose_point_depth_mv.proobjaverse_official_slat_cpu_audit.v1"
RESERVED_SPLITS = {
    "decoder_audit": 32,
    "predicted_support_bridge": 32,
    "dev": 64,
}
DEFAULT_TRAIN_COUNT = 1872
DEFAULT_SPLITS = {**RESERVED_SPLITS, "train": DEFAULT_TRAIN_COUNT}


def split_counts_for_train(train_count: int) -> dict[str, int]:
    if int(train_count) <= 0:
        raise ValueError("train_count must be positive")
    return {**RESERVED_SPLITS, "train": int(train_count)}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _complete_rows(root: Path, selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, source in enumerate(selection["selected"]):
        render = root / str(source["render"]["path"])
        slat = root / str(source["slat"]["path"])
        if not render.is_file() or not slat.is_file():
            continue
        if render.stat().st_size != int(source["render"]["size"]):
            continue
        if slat.stat().st_size != int(source["slat"]["size"]):
            continue
        rows.append(
            {
                "uid": str(source["uid"]),
                "selection_rank": rank,
                "shard": str(source["shard"]),
                "render_tar": str(render.resolve()),
                "render_size": render.stat().st_size,
                "slat_npz": str(slat.resolve()),
                "slat_size": slat.stat().st_size,
            }
        )
    return rows


def freeze_protocol(args: argparse.Namespace) -> None:
    root = Path(args.data_root).expanduser().resolve()
    selection_path = Path(args.selection).expanduser().resolve()
    selection = load_json(selection_path)
    if selection.get("format") != SELECTION_FORMAT:
        raise ValueError(f"unexpected selection format={selection.get('format')!r}")
    requested_splits = split_counts_for_train(int(args.train_count))
    if int(args.pair_count) != sum(requested_splits.values()):
        raise ValueError(
            "pair_count must exactly cover the disjoint split contract: "
            f"expected={sum(requested_splits.values())} "
            f"for train_count={args.train_count}, got={args.pair_count}"
        )
    complete = _complete_rows(root, selection)
    if len(complete) < int(args.pair_count):
        print(
            json.dumps(
                {
                    "passed": False,
                    "complete_pairs": len(complete),
                    "required_pairs": int(args.pair_count),
                    "remaining": int(args.pair_count) - len(complete),
                },
                indent=2,
            )
        )
        raise SystemExit(3)

    pool = complete[: int(args.pair_count)]
    # Hash ordering makes split membership independent of shard/download order,
    # while pool membership remains bound to the frozen 5K selection order.
    ordered = sorted(
        pool,
        key=lambda row: hashlib.sha256(
            f"{int(args.seed)}:{row['uid']}".encode("ascii")
        ).hexdigest(),
    )
    splits: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for name in ("decoder_audit", "predicted_support_bridge", "dev", "train"):
        count = requested_splits[name]
        splits[name] = ordered[cursor : cursor + count]
        cursor += count
    assert cursor == len(pool)

    pool_identity = [
        {
            key: row[key]
            for key in (
                "uid",
                "selection_rank",
                "shard",
                "render_tar",
                "render_size",
                "slat_npz",
                "slat_size",
            )
        }
        for row in pool
    ]
    protocol_body: dict[str, Any] = {
        "format": PROTOCOL_FORMAT,
        "formal": False,
        "data_root": str(root),
        "selection": str(selection_path),
        "selection_file_sha256": sha256_file(selection_path),
        "selection_sha256": str(selection["selection_sha256"]),
        "repository": {
            "repo_id": selection["repo_id"],
            "revision": selection["revision"],
        },
        "pair_count": len(pool),
        "seed": int(args.seed),
        "pool_sha256": canonical_sha256(pool_identity),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_uid_sha256": {
            name: canonical_sha256([row["uid"] for row in rows])
            for name, rows in splits.items()
        },
        "scope_guard": (
            "development target-domain diagnosis only; decoder_audit, dev and "
            "predicted_support_bridge are disjoint from training and are not a "
            "final untouched benchmark"
        ),
    }
    protocol_body["protocol_sha256"] = canonical_sha256(protocol_body)
    output = Path(args.output_dir).expanduser().resolve()
    protocol_path = output / "protocol.json"
    if output.exists():
        if not protocol_path.is_file() or load_json(protocol_path) != protocol_body:
            raise RuntimeError(f"refusing to overwrite changed protocol: {output}")
        print(json.dumps({"reused": True, **protocol_body}, indent=2))
        return
    output.mkdir(parents=True)
    atomic_json(protocol_path, protocol_body)
    for name, rows in splits.items():
        body: dict[str, Any] = {
            "format": SPLIT_FORMAT,
            "formal": False,
            "protocol": str(protocol_path),
            "protocol_sha256": protocol_body["protocol_sha256"],
            "name": name,
            "count": len(rows),
            "rows": rows,
        }
        body["manifest_sha256"] = canonical_sha256(body)
        atomic_json(output / f"{name}.json", body)
    print(json.dumps({"reused": False, **protocol_body}, indent=2))


def _audit_slat(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as value:
        if set(value.files) != {"coords", "feats"}:
            raise ValueError(f"unexpected SLat fields={value.files}: {path}")
        coords = np.asarray(value["coords"])
        feats = np.asarray(value["feats"])
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.dtype != np.uint8:
        raise ValueError(f"invalid official coords {coords.shape}/{coords.dtype}: {path}")
    if feats.shape != (len(coords), 8) or feats.dtype != np.float32:
        raise ValueError(f"invalid official feats {feats.shape}/{feats.dtype}: {path}")
    if not len(coords) or len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError(f"empty/duplicate official coordinates: {path}")
    if int(coords.min()) < 0 or int(coords.max()) >= 64:
        raise ValueError(f"official coordinates outside [0,63]: {path}")
    if not np.isfinite(feats).all():
        raise ValueError(f"non-finite official features: {path}")
    return {
        "point_count": len(coords),
        "coord_min": int(coords.min()),
        "coord_max": int(coords.max()),
        "feature_abs_max": float(np.abs(feats).max()),
        "feature_rms": float(np.sqrt(np.mean(np.square(feats, dtype=np.float64)))),
    }


def _audit_tar(path: Path, uid: str) -> dict[str, Any]:
    with tarfile.open(path, "r") as archive:
        names = archive.getnames()
        json_names = sorted(name for name in names if name.endswith(".json"))
        rgba_names = sorted(name for name in names if name.endswith(".rgba.webp"))
        json_stems = {Path(name).stem for name in json_names}
        rgba_stems = {Path(name).name.removesuffix(".rgba.webp") for name in rgba_names}
        if not json_names or json_stems != rgba_stems:
            raise ValueError(f"incomplete official views: {path}")
        if any(Path(name).parts[0] != uid for name in json_names + rgba_names):
            raise ValueError(f"tar UID/layout mismatch: {path}")
        centers = []
        for name in json_names:
            handle = archive.extractfile(name)
            if handle is None:
                raise RuntimeError(f"cannot read {name} from {path}")
            meta = json.load(handle)
            intrinsic = np.asarray(meta["intrinsic"], dtype=np.float64)
            extrinsic = np.asarray(meta["extrinsic"], dtype=np.float64)
            if intrinsic.shape != (3, 3) or extrinsic.shape != (4, 4):
                raise ValueError(f"invalid camera shape: {path}/{name}")
            if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
                raise ValueError(f"non-finite camera: {path}/{name}")
            # The official field is named ``extrinsic`` but stores C2W.  Audit
            # that source convention directly and also verify the derived W2C
            # camera consistently faces the canonical origin.
            rotation = extrinsic[:3, :3]
            if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2.0e-3):
                raise ValueError(f"non-rigid official C2W camera: {path}/{name}")
            center = extrinsic[:3, 3]
            centers.append(center)
            origin_camera = rotation.T @ (-center)
            if abs(float(origin_camera[2])) < 1.0e-6:
                raise ValueError(f"official camera is tangent to origin: {path}/{name}")
        centers_array = np.stack(centers)
    return {
        "view_count": len(json_names),
        "camera_radius_min": float(np.linalg.norm(centers_array, axis=1).min()),
        "camera_radius_max": float(np.linalg.norm(centers_array, axis=1).max()),
        "source_extrinsics_type": "camera_to_world",
        "runtime_extrinsics_type": "world_to_camera",
    }


def audit_protocol(args: argparse.Namespace) -> None:
    split_path = Path(args.split_manifest).expanduser().resolve()
    split = load_json(split_path)
    if split.get("format") != SPLIT_FORMAT:
        raise ValueError(f"unexpected split format={split.get('format')!r}")
    rows = list(split["rows"])
    if int(args.max_objects) > 0:
        rows = rows[: int(args.max_objects)]
    if not rows:
        raise ValueError("audit selection is empty")
    records = []
    for index, row in enumerate(rows, start=1):
        slat_path = Path(row["slat_npz"])
        tar_path = Path(row["render_tar"])
        if slat_path.stat().st_size != int(row["slat_size"]):
            raise RuntimeError(f"SLat size changed: {slat_path}")
        if tar_path.stat().st_size != int(row["render_size"]):
            raise RuntimeError(f"render tar size changed: {tar_path}")
        records.append(
            {
                "uid": row["uid"],
                "slat": _audit_slat(slat_path),
                "render": _audit_tar(tar_path, str(row["uid"])),
            }
        )
        print(f"[official_slat_cpu_audit] {index}/{len(rows)} {row['uid']}", flush=True)
    point_counts = np.asarray([row["slat"]["point_count"] for row in records])
    view_counts = np.asarray([row["render"]["view_count"] for row in records])
    report: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "passed": True,
        "formal": False,
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "protocol_sha256": split["protocol_sha256"],
        "split": split["name"],
        "object_count": len(records),
        "summary": {
            "point_count_min": int(point_counts.min()),
            "point_count_median": float(np.median(point_counts)),
            "point_count_max": int(point_counts.max()),
            "view_count_min": int(view_counts.min()),
            "view_count_max": int(view_counts.max()),
        },
        "records": records,
        "scope_guard": "CPU schema/camera audit only; this does not measure Mesh quality",
    }
    report["report_sha256"] = canonical_sha256(report)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and load_json(output) != report:
        raise RuntimeError(f"refusing to overwrite changed audit: {output}")
    atomic_json(output, report)
    print(json.dumps({key: report[key] for key in ("passed", "split", "object_count", "summary", "report_sha256")}, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--data_root", required=True)
    freeze.add_argument("--selection", required=True)
    freeze.add_argument("--output_dir", required=True)
    freeze.add_argument("--pair_count", type=int, default=2000)
    freeze.add_argument(
        "--train_count",
        type=int,
        default=DEFAULT_TRAIN_COUNT,
        help=(
            "training objects; 128 additional objects are always reserved as "
            "decoder_audit32, predicted_support_bridge32 and dev64"
        ),
    )
    freeze.add_argument("--seed", type=int, default=20260813)
    freeze.set_defaults(handler=freeze_protocol)
    audit = commands.add_parser("audit")
    audit.add_argument("--split_manifest", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--max_objects", type=int, default=0)
    audit.set_defaults(handler=audit_protocol)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
