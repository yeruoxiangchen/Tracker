#!/usr/bin/env python3
"""Freeze multiple completed DINO-only cache shards as one training manifest.

Sample tensors are not copied.  The merged manifest stores absolute cache-file
paths and cryptographically binds every source shard manifest and completion
marker.  Use a new output directory whenever the admitted shard set changes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ar_ss_flow.pose_lifting import (
    LIFTING_CACHE_VERSION,
    LIFTING_METADATA_NAMES,
    schema_hash,
)
from pose_point_depth_mv.dino_only_condition import (
    validate_dino_only_lifting_contract,
)


SOURCE_MARKER_FORMAT = "pose_point_depth_mv.direct_dino_only_lifting_marker.v1"
MERGE_FORMAT = "pose_point_depth_mv.dino_only_lifting_shard_merge.v1"
MERGE_MARKER_FORMAT = "pose_point_depth_mv.dino_only_lifting_shard_merge_marker.v1"
SOURCE_MARKER = "_DINO_ONLY_LIFTING_COMPLETE.json"
MERGE_MARKER = "_DINO_ONLY_LIFTING_MERGE_COMPLETE.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_shards(spec: str) -> set[int] | None:
    text = str(spec).strip().lower()
    if text in {"", "all"}:
        return None
    output: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            output.update(range(int(start), int(end) + 1))
        else:
            output.add(int(item))
    if not output or min(output) < 0:
        raise ValueError("--shards must select nonnegative indices")
    return output


def shard_index(path: Path) -> int | None:
    for parent in path.parents:
        if parent.name.startswith("shard_"):
            try:
                return int(parent.name.removeprefix("shard_"))
            except ValueError:
                return None
    return None


def discover_manifests(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value).expanduser().resolve() for value in args.shard_manifest]
    if args.input_root:
        root = Path(args.input_root).expanduser().resolve()
        paths.extend(
            sorted(
                path.resolve()
                for path in root.glob("shards/shard_*/dino_only/lifting_manifest.json")
            )
        )
    selected = parse_shards(args.shards)
    unique = sorted(
        {
            path
            for path in paths
            if selected is None or shard_index(path) in selected
        }
    )
    if not unique:
        raise ValueError("no DINO-only shard manifests discovered")
    return unique


def validate_source_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    marker_path = path.parent / SOURCE_MARKER
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker = load_json(marker_path)
    if (
        marker.get("format") != SOURCE_MARKER_FORMAT
        or marker.get("passed") is not True
        or marker.get("vggt_model_loaded") is not False
        or marker.get("vggt_model_executed") is not False
        or marker.get("manifest_sha256") != sha256_file(path)
    ):
        raise RuntimeError(f"invalid direct DINO completion marker: {marker_path}")
    payload = load_json(path)
    if (
        payload.get("format") != LIFTING_CACHE_VERSION
        or payload.get("passed") is not True
        or payload.get("training_ready") is not True
        or int(payload.get("failure_count", -1)) != 0
        or not payload.get("samples")
    ):
        raise RuntimeError(f"DINO shard is not training-ready: {path}")
    proxy = SimpleNamespace(
        visual_feature_dim=payload.get("visual_feature_dim"),
        feature_metadata=payload.get("feature_metadata"),
        config=payload.get("config"),
        config_hash=payload.get("config_hash"),
    )
    validate_dino_only_lifting_contract(proxy)
    return payload, marker


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root")
    parser.add_argument("--shard_manifest", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shards", default="all")
    parser.add_argument("--expected_shards", type=int, default=0)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_paths = discover_manifests(args)
    if int(args.expected_shards) > 0 and len(source_paths) != int(args.expected_shards):
        raise RuntimeError(
            f"expected {args.expected_shards} DINO shards, found {len(source_paths)}"
        )
    source_records: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    object_uids: set[str] = set()
    reference: dict[str, Any] | None = None
    config_hash: str | None = None

    for shard_position, path in enumerate(source_paths):
        payload, marker = validate_source_manifest(path)
        current_contract = {
            "config": payload["config"],
            "config_hash": payload["config_hash"],
            "feature_metadata": payload["feature_metadata"],
            "visual_feature_dim": payload["visual_feature_dim"],
            "metadata_names": payload["metadata_names"],
            "metadata_schema_hash": payload["metadata_schema_hash"],
        }
        if reference is None:
            reference = current_contract
            config_hash = str(payload["config_hash"])
        elif current_contract != reference:
            raise RuntimeError(f"DINO shard cache contracts differ: {path}")
        source_root = Path(payload.get("output_dir", path.parent)).resolve()
        source_rows = payload["samples"]
        for row in source_rows:
            uid = str(row.get("uid", ""))
            object_uid = str(row.get("object_uid", uid))
            if not uid or uid in seen_uids:
                raise ValueError(f"empty/duplicate cache UID={uid!r} from {path}")
            cache_file = Path(str(row["cache_file"])).expanduser()
            if not cache_file.is_absolute():
                cache_file = (source_root / cache_file).resolve()
            if not cache_file.is_file():
                raise FileNotFoundError(cache_file)
            expected_sha = str(row.get("cache_file_sha256", ""))
            actual_sha = sha256_file(cache_file)
            if expected_sha and expected_sha != actual_sha:
                raise RuntimeError(f"source sample hash changed: {cache_file}")
            seen_uids.add(uid)
            object_uids.add(object_uid)
            output_rows.append(
                {
                    **row,
                    "cache_file": str(cache_file),
                    "cache_file_sha256": actual_sha,
                    "source_shard_position": shard_position,
                    "source_shard_manifest": str(path),
                }
            )
        source_records.append(
            {
                "position": shard_position,
                "manifest": str(path),
                "manifest_sha256": sha256_file(path),
                "completion_marker": str(path.parent / SOURCE_MARKER),
                "completion_marker_sha256": sha256_file(path.parent / SOURCE_MARKER),
                "config_hash": payload["config_hash"],
                "sample_count": len(source_rows),
                "object_count": int(payload["object_count"]),
                "source_cache_manifest": payload.get("source_cache_manifest"),
                "source_cache_manifest_sha256": payload.get(
                    "source_cache_manifest_sha256"
                ),
                "marker": marker,
            }
        )

    assert reference is not None and config_hash is not None
    inventory = {
        "format": MERGE_FORMAT,
        "created_at_utc": utc_now(),
        "source_shard_count": len(source_records),
        "sample_count": len(output_rows),
        "object_count": len(object_uids),
        "config_hash": config_hash,
        "source_shards": source_records,
    }
    input_identity = hashlib.sha256(
        json.dumps(
            [
                (row["manifest_sha256"], row["completion_marker_sha256"])
                for row in source_records
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    inventory["input_identity_sha256"] = input_identity

    if output_dir.exists():
        marker_path = output_dir / MERGE_MARKER
        manifest_path = output_dir / "lifting_manifest.json"
        if marker_path.is_file() and manifest_path.is_file():
            old = load_json(marker_path)
            if (
                old.get("format") == MERGE_MARKER_FORMAT
                and old.get("passed") is True
                and old.get("input_identity_sha256") == input_identity
                and old.get("manifest_sha256") == sha256_file(manifest_path)
            ):
                print(
                    json.dumps(
                        {
                            "reused": True,
                            "manifest": str(manifest_path),
                            "source_shards": len(source_records),
                            "samples": len(output_rows),
                            "objects": len(object_uids),
                        },
                        indent=2,
                    )
                )
                return
        raise RuntimeError(f"immutable merge output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    inventory_path = output_dir / "source_inventory.json"
    atomic_json(inventory_path, inventory)
    manifest = {
        "format": LIFTING_CACHE_VERSION,
        "created_at_utc": utc_now(),
        "output_dir": str(output_dir),
        "source_cache_manifest": str(inventory_path),
        "source_cache_manifests": [str(path) for path in source_paths],
        "source_inventory_sha256": sha256_file(inventory_path),
        "stock_condition_source": "deterministic DINO-only token context",
        "lifting_feature_source": "direct frozen DINO patches; sharded no-copy merge",
        "sample_count": len(output_rows),
        "object_count": len(object_uids),
        "failure_count": 0,
        "feature_metadata": reference["feature_metadata"],
        "visual_feature_dim": reference["visual_feature_dim"],
        "metadata_names": list(LIFTING_METADATA_NAMES),
        "metadata_schema_hash": schema_hash(),
        "config": reference["config"],
        "config_hash": config_hash,
        "samples": output_rows,
        "passed": True,
        "training_ready": True,
        "shard_merge": {
            "format": MERGE_FORMAT,
            "source_shard_count": len(source_records),
            "input_identity_sha256": input_identity,
            "copy_policy": "absolute sample references; tensor files not copied",
        },
    }
    proxy = SimpleNamespace(
        visual_feature_dim=manifest["visual_feature_dim"],
        feature_metadata=manifest["feature_metadata"],
        config=manifest["config"],
        config_hash=manifest["config_hash"],
    )
    manifest["no_vggt_contract"] = validate_dino_only_lifting_contract(proxy)
    manifest_path = output_dir / "lifting_manifest.json"
    atomic_json(manifest_path, manifest)
    marker = {
        "format": MERGE_MARKER_FORMAT,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_inventory": str(inventory_path),
        "source_inventory_sha256": sha256_file(inventory_path),
        "input_identity_sha256": input_identity,
        "source_shard_count": len(source_records),
        "sample_count": len(output_rows),
        "object_count": len(object_uids),
        "config_hash": config_hash,
        "passed": True,
    }
    atomic_json(output_dir / MERGE_MARKER, marker)
    print(json.dumps(marker, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
