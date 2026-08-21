#!/usr/bin/env python3
"""Freeze 16 unseen Objaverse objects from the reviewed mixed1k test split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL_FORMAT = "pose_point_depth_mv.frozen_objaverse_test16.v1"
DEFAULT_TEST = Path(
    "/data/zjr/reviewed_mixed1k_semantic_object_ss_repaired_v1_20260730/test.json"
)
DEFAULT_TRAINING_LIFTING = (
    "/data/zjr/native_ss_no_vggt_mixed1k_20260807_v1/"
    "lifting_train868_dino_only_v1/lifting_manifest.json"
)
DEFAULT_REAL_TRAINING_LIFTING = (
    "/data/zjr/native_no_vggt_mixed_real376_synth868_20260808_v1/"
    "lifting_real376_dino_only_v1/lifting_manifest.json"
)
DEFAULT_QUOTAS = {
    "legacy897": 10,
    "gap_objaverse288": 3,
    "pilot_objaverse217": 3,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, *parts: object) -> str:
    text = "\0".join((str(int(seed)), *(str(part) for part in parts)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_quotas(value: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for item in str(value).split(","):
        name, separator, count = item.strip().partition("=")
        if not separator or not name or int(count) <= 0 or name in output:
            raise argparse.ArgumentTypeError(f"invalid source-group quota: {item!r}")
        output[name] = int(count)
    if not output:
        raise argparse.ArgumentTypeError("source-group quotas cannot be empty")
    return output


def parse_int_csv(value: str) -> list[int]:
    result = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or min(result) <= 0:
        raise argparse.ArgumentTypeError("view choices must be unique positive integers")
    return result


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("samples"), list):
        raise ValueError(f"manifest has no samples list: {path}")
    return payload


def object_uids(payload: dict[str, Any]) -> set[str]:
    records = payload.get("object_records")
    if isinstance(records, list) and records:
        return {str(row["object_uid"]) for row in records}
    return {
        str(row.get("object_uid", row.get("uid", "")))
        for row in payload.get("samples", [])
        if row.get("object_uid", row.get("uid"))
    }


def mesh_identities(payload: dict[str, Any]) -> set[str]:
    rows: Iterable[dict[str, Any]] = payload.get("object_records") or payload["samples"]
    identities: set[str] = set()
    for row in rows:
        value = row.get("source_glb")
        if value:
            path = Path(str(value)).expanduser()
            identities.add(f"{path.name}:{path.resolve(strict=False)}")
    return identities


def select_samples(
    payload: dict[str, Any], *, seed: int, quotas: dict[str, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload["samples"]:
        if str(row.get("dataset_source", "")) == "objaverse":
            candidates[str(row["object_uid"])].append(row)
    records = {
        str(row["object_uid"]): row
        for row in payload.get("object_records", [])
        if str(row.get("dataset_source", "")) == "objaverse"
    }
    selected_objects: list[str] = []
    for group, quota in quotas.items():
        available = [
            object_uid
            for object_uid, record in records.items()
            if str(record.get("source_group")) == group and object_uid in candidates
        ]
        if len(available) < int(quota):
            raise RuntimeError(
                f"source group {group!r} has {len(available)} objects, needs {quota}"
            )
        available.sort(key=lambda uid: stable_rank(seed, "object", group, uid))
        selected_objects.extend(available[: int(quota)])

    selected_samples: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    for object_uid in selected_objects:
        rows = sorted(
            candidates[object_uid],
            key=lambda row: stable_rank(seed, "sequence", object_uid, row["uid"]),
        )
        selected_samples.append(dict(rows[0]))
        record = dict(records[object_uid])
        record["sample_count"] = 1
        selected_records.append(record)
    return selected_samples, selected_records


def expected_prior_views(
    *, count: int, seed: int, choices: list[int], available_views: int = 8
) -> list[int]:
    valid = [value for value in choices if value <= int(available_views)]
    if not valid:
        valid = [int(available_views)]
    return [
        int(np.random.default_rng(int(seed) + index * 1009).choice(valid))
        for index in range(int(count))
    ]


def verify_selected_files(
    samples: list[dict[str, Any]], *, latent_root: str | None
) -> None:
    root = Path(latent_root).resolve() if latent_root else None
    missing: list[str] = []
    for row in samples:
        latent = Path(str(row["ss_latent"]))
        if not latent.is_absolute() and root is not None:
            latent = root / latent
        paths = [latent, Path(str(row["source_glb"]))]
        for frame in row.get("frames", []):
            paths.extend((Path(str(frame["image"])), Path(str(frame["mask"]))))
        missing.extend(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"selected inputs are missing; first={missing[:8]}")


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    test_path = Path(args.test_manifest).expanduser().resolve()
    train_path = Path(args.train_manifest).expanduser().resolve()
    val_path = Path(args.val_manifest).expanduser().resolve()
    test = load_manifest(test_path)
    train = load_manifest(train_path)
    val = load_manifest(val_path)
    if test.get("format") != "pixal3d_multiview.objaverse_sparse.v1":
        raise ValueError(f"unsupported test manifest format={test.get('format')!r}")
    if str(test.get("split")) != "test":
        raise ValueError("Objaverse16 must be frozen from the untouched test split")

    selected, selected_records = select_samples(
        test, seed=int(args.seed), quotas=dict(args.group_quotas)
    )
    expected_count = sum(args.group_quotas.values())
    if len(selected) != expected_count or len({row["object_uid"] for row in selected}) != expected_count:
        raise RuntimeError("selection did not produce one sequence per requested object")
    if any(len(row.get("frames", [])) != 8 for row in selected):
        raise RuntimeError("frozen Objaverse test samples must each expose 8 source views")
    if not args.skip_file_checks:
        verify_selected_files(selected, latent_root=test.get("latent_root"))

    selected_uids = {str(row["object_uid"]) for row in selected}
    selected_meshes = mesh_identities({"samples": selected})
    train_uids, val_uids, test_uids = map(object_uids, (train, val, test))
    train_meshes, val_meshes = map(mesh_identities, (train, val))
    split_audit = {
        "train_val_object_overlap": sorted(train_uids & val_uids),
        "train_test_object_overlap": sorted(train_uids & test_uids),
        "val_test_object_overlap": sorted(val_uids & test_uids),
        "selected_train_object_overlap": sorted(selected_uids & train_uids),
        "selected_val_object_overlap": sorted(selected_uids & val_uids),
        "selected_train_source_mesh_overlap": sorted(selected_meshes & train_meshes),
        "selected_val_source_mesh_overlap": sorted(selected_meshes & val_meshes),
    }
    if any(split_audit.values()):
        raise RuntimeError(f"reviewed split disjoint audit failed: {split_audit}")

    training_audits: list[dict[str, Any]] = []
    for value in args.training_lifting_manifest:
        path = Path(value).expanduser().resolve()
        training = load_manifest(path)
        training_uids = object_uids(training)
        overlap = sorted(selected_uids & training_uids)
        training_audits.append(
            {
                "manifest": str(path),
                "manifest_sha256": sha256_file(path),
                "object_count": len(training_uids),
                "selected_object_overlap": overlap,
                "passed": not overlap,
            }
        )
    if not training_audits or not all(row["passed"] for row in training_audits):
        raise RuntimeError(f"deployment training-object audit failed: {training_audits}")

    prior_views = expected_prior_views(
        count=len(selected),
        seed=int(args.point_prior_seed),
        choices=list(args.prior_view_choices),
    )
    for position, (row, view_count) in enumerate(zip(selected, prior_views)):
        row["objaverse16_selection"] = {
            "position": position,
            "selection_seed": int(args.seed),
            "expected_point_prior_view_count": view_count,
            "point_prior_seed": int(args.point_prior_seed),
        }

    source_group_counts = Counter(str(row["source_group"]) for row in selected)
    protocol = {
        "format": PROTOCOL_FORMAT,
        "scope": "frozen_objaverse_test16",
        "formal": False,
        "selection_seed": int(args.seed),
        "point_prior_seed": int(args.point_prior_seed),
        "point_prior_view_choices": list(args.prior_view_choices),
        "expected_point_prior_view_histogram": {
            str(key): value for key, value in sorted(Counter(prior_views).items())
        },
        "source_group_quotas": dict(args.group_quotas),
        "source_group_counts": dict(sorted(source_group_counts.items())),
        "object_count": len(selected),
        "sequence_count": len(selected),
        "selected_uids": [str(row["uid"]) for row in selected],
        "selected_object_uids": [str(row["object_uid"]) for row in selected],
        "source_test_manifest": str(test_path),
        "source_test_manifest_sha256": sha256_file(test_path),
        "parent_split_manifests": {
            "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "val": {"path": str(val_path), "sha256": sha256_file(val_path)},
            "test": {"path": str(test_path), "sha256": sha256_file(test_path)},
        },
        "split_disjoint_audit": split_audit,
        "deployment_training_audits": training_audits,
        "training_object_disjoint": True,
        "source_mesh_disjoint": True,
        "selection_rule": (
            "SHA256(seed, source_group, object_uid), then one sequence by "
            "SHA256(seed, object_uid, uid); no model result was read"
        ),
        "passed": True,
    }
    protocol["protocol_sha256"] = canonical_json_sha256(protocol)

    output = dict(test)
    output.update(
        {
            "status": "frozen_objaverse_test16",
            "training_ready": False,
            "formal_confirmatory": False,
            "selection_policy": protocol["selection_rule"],
            "split_stats": {
                "test": {
                    "object_count": len(selected),
                    "sample_count": len(selected),
                    "object_counts_by_group": {
                        f"objaverse:{key}": value
                        for key, value in sorted(source_group_counts.items())
                    },
                }
            },
            "samples": selected,
            "object_records": selected_records,
            "objaverse16_protocol": protocol,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
        }
    )
    return output


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_manifest", default=str(DEFAULT_TEST))
    parser.add_argument("--train_manifest", default=str(DEFAULT_TEST.with_name("train.json")))
    parser.add_argument("--val_manifest", default=str(DEFAULT_TEST.with_name("val.json")))
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--point_prior_seed", type=int, default=20260810)
    parser.add_argument(
        "--group_quotas",
        type=parse_quotas,
        default=DEFAULT_QUOTAS,
        help="comma-separated source_group=count values",
    )
    parser.add_argument(
        "--prior_view_choices", type=parse_int_csv, default=[2, 4, 8]
    )
    parser.add_argument(
        "--training_lifting_manifest",
        action="append",
        default=None,
        help="repeat for every deployment training lifting manifest",
    )
    parser.add_argument("--skip_file_checks", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.training_lifting_manifest is None:
        args.training_lifting_manifest = [
            DEFAULT_TRAINING_LIFTING,
            DEFAULT_REAL_TRAINING_LIFTING,
        ]
    output_path = Path(args.output_manifest).expanduser().resolve()
    payload = build_protocol(args)
    if output_path.is_file():
        if not args.resume:
            raise FileExistsError(output_path)
        existing = load_manifest(output_path)
        expected = payload["objaverse16_protocol"]
        observed = existing.get("objaverse16_protocol", {})
        stable_keys = (
            "format",
            "scope",
            "selection_seed",
            "point_prior_seed",
            "source_group_quotas",
            "source_test_manifest_sha256",
            "protocol_sha256",
        )
        if any(observed.get(key) != expected.get(key) for key in stable_keys):
            raise RuntimeError("existing Objaverse16 freeze differs from requested protocol")
        payload = existing
    else:
        atomic_json(output_path, payload)
    protocol = payload["objaverse16_protocol"]
    print(
        json.dumps(
            {
                "passed": protocol["passed"],
                "formal": protocol["formal"],
                "scope": protocol["scope"],
                "objects": protocol["object_count"],
                "source_groups": protocol["source_group_counts"],
                "expected_prior_views": protocol[
                    "expected_point_prior_view_histogram"
                ],
                "manifest": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
