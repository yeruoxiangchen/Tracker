#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic object-balanced manifest subset.")
    parser.add_argument("--source_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--object_count", type=int, default=64)
    parser.add_argument("--sequences_per_object", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_path = Path(args.source_manifest)
    payload: dict[str, Any] = json.loads(source_path.read_text(encoding="utf-8"))
    rows = payload.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"source manifest has no samples: {source_path}")
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_uids: set[str] = set()
    for row in rows:
        uid = str(row.get("uid", ""))
        object_uid = str(row.get("object_uid", ""))
        if not uid or not object_uid or uid in seen_uids:
            raise ValueError(f"invalid or duplicate uid/object_uid: uid={uid} object_uid={object_uid}")
        seen_uids.add(uid)
        by_object[object_uid].append(row)
    object_count = int(args.object_count)
    per_object = int(args.sequences_per_object)
    eligible = sorted(key for key, group in by_object.items() if len(group) >= per_object)
    if object_count <= 0 or per_object <= 0 or len(eligible) < object_count:
        raise ValueError(
            f"insufficient eligible objects: requested={object_count} x {per_object}, eligible={len(eligible)}"
        )
    rng = random.Random(int(args.seed))
    rng.shuffle(eligible)
    selected_objects = eligible[:object_count]
    selected: list[dict[str, Any]] = []
    for object_uid in selected_objects:
        group = sorted(by_object[object_uid], key=lambda row: str(row["uid"]))
        rng.shuffle(group)
        selected.extend(group[:per_object])
    if len({str(row["object_uid"]) for row in selected}) != object_count:
        raise RuntimeError("object uniqueness invariant failed")
    if len(selected) != object_count * per_object:
        raise RuntimeError("sample count invariant failed")

    output = dict(payload)
    output["samples"] = selected
    output["subset"] = {
        "source_manifest": str(source_path),
        "object_count": object_count,
        "sequences_per_object": per_object,
        "sample_count": len(selected),
        "seed": int(args.seed),
        "object_uids": selected_objects,
    }
    canonical = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    output["manifest_hash"] = hashlib.sha256(canonical).hexdigest()
    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output["subset"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
