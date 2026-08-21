#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"manifest must contain a samples list: {path}")
    return payload


def compatible_metadata(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    base = copy.deepcopy(payloads[0])
    base.pop("samples", None)
    base.pop("failures", None)
    for key in ("image_root", "mask_root", "latent_root", "extrinsics_type", "camera_forward_sign"):
        values = {json.dumps(payload.get(key), sort_keys=True) for payload in payloads}
        if len(values) != 1:
            raise ValueError(f"input manifests disagree on {key}: {values}")
    return base


def manifest_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_split(
    path: Path,
    metadata: dict[str, Any],
    samples: list[dict[str, Any]],
    split: str,
    *,
    group_key: str,
) -> str:
    payload = copy.deepcopy(metadata)
    payload["samples"] = samples
    payload["failures"] = []
    payload["object_disjoint_split"] = split
    payload["split_stats"] = {
        "policy": f"{group_key}_disjoint",
        "split": split,
        "sample_count": len(samples),
        "object_count": len({str(sample["object_uid"]) for sample in samples}),
    }
    payload["manifest_hash"] = manifest_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(payload["manifest_hash"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-split Pixal/Objaverse manifests by object_uid.")
    parser.add_argument("--input_manifests", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--val_objects", type=int, default=128)
    parser.add_argument("--holdout_objects", type=int, default=128)
    parser.add_argument("--max_sequences_per_object", type=int, default=0)
    parser.add_argument("--group_key", choices=["object_uid", "source_glb"], default="object_uid")
    parser.add_argument("--write_one_sequence_per_object", action="store_true")
    args = parser.parse_args()

    payloads = [load_manifest(Path(path)) for path in args.input_manifests]
    metadata = compatible_metadata(payloads)
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_uids: dict[str, str] = {}
    for payload in payloads:
        for sample in payload["samples"]:
            uid = str(sample.get("uid", ""))
            object_uid = str(sample.get("object_uid", ""))
            source_glb = str(sample.get("source_glb", ""))
            if not uid or not object_uid or not source_glb:
                raise ValueError(f"sample lacks uid/object_uid/source_glb: {sample}")
            canonical = json.dumps(sample, sort_keys=True, ensure_ascii=False)
            if uid in seen_uids and seen_uids[uid] != canonical:
                raise ValueError(f"duplicate uid has inconsistent content: {uid}")
            if uid in seen_uids:
                continue
            seen_uids[uid] = canonical
            by_object[str(sample[args.group_key])].append(sample)

    object_ids = sorted(by_object)
    rng = random.Random(int(args.seed))
    rng.shuffle(object_ids)
    val_n = int(args.val_objects)
    holdout_n = int(args.holdout_objects)
    if val_n <= 0 or holdout_n <= 0 or val_n + holdout_n >= len(object_ids):
        raise ValueError(
            f"invalid split sizes: objects={len(object_ids)} val={val_n} holdout={holdout_n}"
        )
    val_ids = set(object_ids[:val_n])
    holdout_ids = set(object_ids[val_n : val_n + holdout_n])
    train_ids = set(object_ids[val_n + holdout_n :])

    max_sequences = int(args.max_sequences_per_object)

    def collect(ids: set[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for object_uid in sorted(ids):
            group = sorted(by_object[object_uid], key=lambda row: str(row["uid"]))
            if max_sequences > 0:
                group = group[:max_sequences]
            rows.extend(group)
        return rows

    splits = {
        "train": collect(train_ids),
        "val": collect(val_ids),
        "holdout": collect(holdout_ids),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, samples in splits.items():
        hashes[name] = write_split(
            output_dir / f"{name}.json", metadata, samples, name, group_key=str(args.group_key)
        )
        if args.write_one_sequence_per_object:
            selected = []
            seen_objects = set()
            for row in sorted(samples, key=lambda item: str(item["uid"])):
                object_uid = str(row["object_uid"])
                if object_uid not in seen_objects:
                    seen_objects.add(object_uid)
                    selected.append(row)
            write_split(
                output_dir / f"{name}_one_per_object.json",
                metadata,
                selected,
                name,
                group_key=str(args.group_key),
            )

    object_sets = {name: {str(row["object_uid"]) for row in rows} for name, rows in splits.items()}
    asset_sets = {name: {str(row["source_glb"]) for row in rows} for name, rows in splits.items()}
    asset_overlap = {
        "train_val": len(asset_sets["train"] & asset_sets["val"]),
        "train_holdout": len(asset_sets["train"] & asset_sets["holdout"]),
        "val_holdout": len(asset_sets["val"] & asset_sets["holdout"]),
    }
    if any(asset_overlap.values()):
        raise RuntimeError(f"source asset leakage detected: {asset_overlap}")

    report = {
        "seed": int(args.seed),
        "input_manifests": [str(Path(path)) for path in args.input_manifests],
        "group_key": str(args.group_key),
        "unique_input_samples": len(seen_uids),
        "unique_objects": len(object_ids),
        "splits": {
            name: {
                "samples": len(samples),
                "objects": len({str(row["object_uid"]) for row in samples}),
                "assets": len({str(row["source_glb"]) for row in samples}),
                "sequence_count_per_object_histogram": dict(
                    sorted(
                        Counter(
                            sum(str(row["object_uid"]) == object_uid for row in samples)
                            for object_uid in {str(row["object_uid"]) for row in samples}
                        ).items()
                    )
                ),
                "manifest_hash": hashes[name],
            }
            for name, samples in splits.items()
        },
        "object_overlap": {
            "train_val": len(object_sets["train"] & object_sets["val"]),
            "train_holdout": len(object_sets["train"] & object_sets["holdout"]),
            "val_holdout": len(object_sets["val"] & object_sets["holdout"]),
        },
        "source_asset_overlap": asset_overlap,
    }
    (output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
