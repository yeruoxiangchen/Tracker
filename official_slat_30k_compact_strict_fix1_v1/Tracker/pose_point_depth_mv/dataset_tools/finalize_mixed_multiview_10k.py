#!/usr/bin/env python3
"""Freeze strict-QC render shards into an object-disjoint mixed 10k dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "pixal3d_multiview.mixed_sparse_10k.v1"
SOURCES = ("objaverse", "omni")
SPLITS = ("train", "val", "test")
COMPATIBILITY_KEYS = (
    "extrinsics_type",
    "camera_forward_sign",
    "coordinate_frame",
    "canonical_latent_frame",
    "num_views",
    "candidate_views",
    "image_size",
    "voxel_resolution",
    "encoder_pretrained",
    "trajectory_mode",
    "renderer",
    "build_config",
    "code_bindings",
    "images_are_masked",
    "quality_policy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine completed Objaverse and Omni strict-render shards into "
            "one immutable, object-disjoint 10k manifest."
        )
    )
    parser.add_argument("--source_plan", required=True)
    parser.add_argument("--objaverse_render_root", required=True)
    parser.add_argument("--omni_render_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--objaverse_objects", type=int, default=6000)
    parser.add_argument("--omni_objects", type=int, default=4000)
    parser.add_argument("--train_objects", type=int, default=9000)
    parser.add_argument("--val_objects", type=int, default=500)
    parser.add_argument("--test_objects", type=int, default=500)
    parser.add_argument("--sequences_per_object", type=int, default=2)
    parser.add_argument(
        "--max_sequences_per_object",
        type=int,
        default=1,
        help="0 keeps every accepted sequence; default freezes one sample per object.",
    )
    parser.add_argument("--max_low_texture_ratio", type=float, default=0.22)
    parser.add_argument("--required_renderer", default="blender")
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(*parts: Any) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def safe_relative(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path: {value}")
    return path


def resolve(root: str, relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else Path(root) / path


def load_source_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tracker.mixed_mesh10k_sources.v1":
        raise ValueError(f"unsupported source plan schema: {payload.get('schema')}")
    for source in SOURCES:
        if source not in payload or not isinstance(payload[source].get("shards"), list):
            raise ValueError(f"source plan lacks {source} shards")
    return payload


def validate_render_manifest_metadata(
    payload: dict[str, Any],
    *,
    path: Path,
    required_renderer: str,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(payload.get("samples"), list) or not isinstance(
        payload.get("failures"), list
    ):
        raise ValueError(f"{path}: incomplete render manifest")
    if required_renderer and payload.get("renderer") != required_renderer:
        raise ValueError(
            f"{path}: renderer={payload.get('renderer')} "
            f"does not match required {required_renderer}"
        )
    metadata = {key: payload.get(key) for key in COMPATIBILITY_KEYS}
    if reference is not None and metadata != reference:
        differing = [
            key for key in COMPATIBILITY_KEYS if metadata.get(key) != reference.get(key)
        ]
        raise ValueError(f"{path}: incompatible build metadata: {differing}")
    return metadata


def expected_source_rows(
    source_plan_path: Path,
    source_plan: dict[str, Any],
    source: str,
) -> dict[int, tuple[dict[str, str], dict[str, Any]]]:
    result = {}
    for shard in source_plan[source]["shards"]:
        index = int(shard["index"])
        manifest_path = source_plan_path.parent / str(shard["path"])
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing frozen source shard: {manifest_path}")
        if sha256_file(manifest_path) != shard["manifest_sha256"]:
            raise RuntimeError(f"frozen source shard changed: {manifest_path}")
        values = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError(f"{manifest_path}: expected UID-to-mesh object")
        if len(values) != int(shard["object_count"]):
            raise RuntimeError(f"source shard object count changed: {manifest_path}")
        result[index] = ({str(key): str(value) for key, value in values.items()}, shard)
    expected_indices = set(range(int(source_plan[source]["shard_count"])))
    if set(result) != expected_indices:
        raise RuntimeError(
            f"{source} source shard indices differ: "
            f"got={sorted(result)} expected={sorted(expected_indices)}"
        )
    return result


def load_rendered_source(
    *,
    source: str,
    render_root: Path,
    source_plan_path: Path,
    source_plan: dict[str, Any],
    sequences_per_object: int,
    required_renderer: str,
    reference_metadata: dict[str, Any] | None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    expected = expected_source_rows(
        source_plan_path,
        source_plan,
        source,
    )
    by_object: dict[str, list[dict[str, Any]]] = {}
    bindings = []
    metadata = reference_metadata
    for index in sorted(expected):
        expected_objects, source_shard = expected[index]
        render_manifest = render_root / f"shard_{index:03d}" / "manifest.json"
        if not render_manifest.is_file():
            raise FileNotFoundError(f"render shard is not complete: {render_manifest}")
        payload = json.loads(render_manifest.read_text(encoding="utf-8"))
        metadata = validate_render_manifest_metadata(
            payload,
            path=render_manifest,
            required_renderer=required_renderer,
            reference=metadata,
        )

        samples = payload["samples"]
        failures = payload["failures"]
        task_count = len(samples) + len(failures)
        expected_tasks = len(expected_objects) * sequences_per_object
        if task_count != expected_tasks:
            raise RuntimeError(
                f"{render_manifest}: task count {task_count} != {expected_tasks}; "
                "the shard is partial or used a different sequences_per_object"
            )
        observed_objects = {
            str(row.get("object_uid", row.get("uid", ""))) for row in samples
        } | {str(row.get("uid", "")) for row in failures}
        observed_objects.discard("")
        if observed_objects != set(expected_objects):
            raise RuntimeError(
                f"{render_manifest}: attempted object set differs from frozen shard; "
                f"missing={len(set(expected_objects) - observed_objects)} "
                f"extra={len(observed_objects - set(expected_objects))}"
            )

        image_root = str(Path(payload["image_root"]).expanduser().resolve())
        mask_root = str(Path(payload["mask_root"]).expanduser().resolve())
        latent_root = str(Path(payload["latent_root"]).expanduser().resolve())
        for raw_sample in samples:
            sample = copy.deepcopy(raw_sample)
            object_uid = str(sample.get("object_uid", ""))
            if object_uid not in expected_objects:
                raise RuntimeError(
                    f"{render_manifest}: accepted object not in source shard: {object_uid}"
                )
            expected_mesh = Path(expected_objects[object_uid]).expanduser().resolve()
            actual_mesh = Path(str(sample.get("source_glb", ""))).expanduser().resolve()
            if actual_mesh != expected_mesh:
                raise RuntimeError(
                    f"{render_manifest}: source mesh mismatch for {object_uid}: "
                    f"{actual_mesh} != {expected_mesh}"
                )
            sample["_source"] = source
            sample["_render_manifest"] = str(render_manifest)
            sample["_image_root"] = image_root
            sample["_mask_root"] = mask_root
            sample["_latent_root"] = latent_root
            by_object.setdefault(object_uid, []).append(sample)

        bindings.append(
            {
                "index": index,
                "path": str(render_manifest),
                "sha256": sha256_file(render_manifest),
                "attempted_objects": len(expected_objects),
                "accepted_samples": len(samples),
                "accepted_objects": len(
                    {
                        str(row.get("object_uid", row.get("uid", "")))
                        for row in samples
                    }
                ),
                "failed_tasks": len(failures),
                "source_manifest_sha256": source_shard["manifest_sha256"],
            }
        )

    duplicate_samples = [
        uid
        for uid, rows in by_object.items()
        if len({str(row["uid"]) for row in rows}) != len(rows)
    ]
    if duplicate_samples:
        raise RuntimeError(f"{source}: duplicate accepted sample UIDs: {duplicate_samples[:10]}")
    return by_object, bindings, metadata or {}


def choose_object_sample(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    object_uid: str,
) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: stable_key(seed, "sample", object_uid, row["uid"]),
    )


def select_source_objects(
    by_object: dict[str, list[dict[str, Any]]],
    *,
    source: str,
    quota: int,
    seed: int,
    max_low_texture_ratio: float,
) -> list[str]:
    if len(by_object) < quota:
        raise RuntimeError(
            f"{source}: only {len(by_object)} strict-QC objects are available, "
            f"below quota {quota} (shortfall {quota - len(by_object)})"
        )
    max_low = (
        quota
        if max_low_texture_ratio < 0
        else int(math.floor(quota * max_low_texture_ratio + 1e-12))
    )
    ranked = sorted(by_object, key=lambda uid: stable_key(seed, source, "object", uid))
    selected: list[str] = []
    low_count = 0
    for uid in ranked:
        sample = choose_object_sample(by_object[uid], seed=seed, object_uid=uid)
        is_low = bool(sample.get("quality_flags", {}).get("low_texture", False))
        if is_low and low_count >= max_low:
            continue
        selected.append(uid)
        low_count += int(is_low)
        if len(selected) == quota:
            break
    if len(selected) != quota:
        normal_available = sum(
            not bool(
                choose_object_sample(rows, seed=seed, object_uid=uid)
                .get("quality_flags", {})
                .get("low_texture", False)
            )
            for uid, rows in by_object.items()
        )
        raise RuntimeError(
            f"{source}: low-texture cap prevents filling quota; selected={len(selected)} "
            f"quota={quota} normal_available={normal_available} max_low={max_low}"
        )
    return selected


def source_split_quotas(
    source_quotas: dict[str, int],
    split_quotas: dict[str, int],
) -> dict[str, dict[str, int]]:
    total = sum(source_quotas.values())
    if total != sum(split_quotas.values()):
        raise ValueError("source quotas and split quotas must have the same total")
    first = SOURCES[0]
    raw = {
        split: split_quotas[split] * source_quotas[first] / total for split in SPLITS
    }
    first_counts = {split: int(math.floor(raw[split])) for split in SPLITS}
    remainder = source_quotas[first] - sum(first_counts.values())
    order = sorted(
        SPLITS,
        key=lambda split: (-(raw[split] - first_counts[split]), split),
    )
    for split in order[:remainder]:
        first_counts[split] += 1
    second = SOURCES[1]
    result = {
        first: dict(first_counts),
        second: {
            split: split_quotas[split] - first_counts[split] for split in SPLITS
        },
    }
    for source in SOURCES:
        if sum(result[source].values()) != source_quotas[source]:
            raise RuntimeError(f"cannot allocate exact split quota for {source}")
        if any(value < 0 for value in result[source].values()):
            raise RuntimeError(f"negative split quota for {source}: {result[source]}")
    return result


def assign_splits(
    selected: dict[str, list[str]],
    *,
    quotas: dict[str, dict[str, int]],
    seed: int,
) -> dict[str, dict[str, str]]:
    assignments: dict[str, dict[str, str]] = {}
    for source in SOURCES:
        ranked = sorted(
            selected[source],
            key=lambda uid: stable_key(seed, source, "split", uid),
        )
        cursor = 0
        for split in SPLITS:
            count = quotas[source][split]
            for uid in ranked[cursor : cursor + count]:
                assignments[uid] = {"source": source, "split": split}
            cursor += count
        if cursor != len(ranked):
            raise RuntimeError(f"split assignment did not consume {source} objects")
    return assignments


def link_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"missing selected asset: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"duplicate final asset path: {destination}")
    os.symlink(str(source.resolve()), destination)


def link_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"missing selected asset directory: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"duplicate final asset directory: {destination}")
    os.symlink(str(source.resolve()), destination, target_is_directory=True)


def materialize_sample(
    raw_sample: dict[str, Any],
    *,
    destination_root: Path,
    split: str,
    source: str,
) -> dict[str, Any]:
    sample = {
        key: copy.deepcopy(value)
        for key, value in raw_sample.items()
        if not key.startswith("_")
    }
    frames = sample.get("frames", [])
    if not frames:
        raise ValueError(f"selected sample {sample.get('uid')} has no frames")

    image_parents = {
        safe_relative(str(frame["image"]), "frame.image").parent for frame in frames
    }
    mask_parents = {
        safe_relative(str(frame["mask"]), "frame.mask").parent for frame in frames
    }
    if len(image_parents) != 1 or len(mask_parents) != 1:
        raise ValueError(
            f"sample {sample.get('uid')} spans multiple asset directories"
        )
    image_parent = next(iter(image_parents))
    mask_parent = next(iter(mask_parents))
    image_root = Path(raw_sample["_image_root"])
    mask_root = Path(raw_sample["_mask_root"])
    latent_root = Path(raw_sample["_latent_root"])
    latent_relative = safe_relative(str(sample["ss_latent"]), "sample.ss_latent")

    link_directory(
        image_root / image_parent,
        destination_root / "images" / image_parent,
    )
    link_directory(
        mask_root / mask_parent,
        destination_root / "masks" / mask_parent,
    )
    link_file(
        latent_root / latent_relative,
        destination_root / "ss_latents" / latent_relative,
    )
    sample["dataset_source"] = source
    sample["frozen_split"] = split
    return sample


def samples_for_object(
    rows: list[dict[str, Any]],
    *,
    object_uid: str,
    seed: int,
    max_sequences: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: stable_key(seed, "sample", object_uid, row["uid"]),
    )
    if max_sequences > 0:
        ranked = ranked[:max_sequences]
    return ranked


def main() -> None:
    args = parse_args()
    source_plan_path = Path(args.source_plan).expanduser().resolve()
    source_plan = load_source_plan(source_plan_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        report_path = output_dir / "report.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("schema") == SCHEMA and report.get("passed") is True:
                print(
                    json.dumps(
                        {
                            "reused": True,
                            "output_dir": str(output_dir),
                            "object_count": report["object_count"],
                            "sample_count": report["sample_count"],
                        },
                        indent=2,
                    ),
                    flush=True,
                )
                return
        raise RuntimeError(f"output directory is not an immutable reusable freeze: {output_dir}")

    source_quotas = {
        "objaverse": int(args.objaverse_objects),
        "omni": int(args.omni_objects),
    }
    split_quotas = {
        "train": int(args.train_objects),
        "val": int(args.val_objects),
        "test": int(args.test_objects),
    }
    if any(value <= 0 for value in source_quotas.values()):
        raise ValueError("source object quotas must be positive")
    if any(value < 0 for value in split_quotas.values()):
        raise ValueError("split object quotas must be nonnegative")
    if sum(source_quotas.values()) != sum(split_quotas.values()):
        raise ValueError(
            f"source total {sum(source_quotas.values())} != "
            f"split total {sum(split_quotas.values())}"
        )
    if args.sequences_per_object <= 0:
        raise ValueError("--sequences_per_object must be positive")
    if not (args.max_low_texture_ratio < 0 or 0 <= args.max_low_texture_ratio <= 1):
        raise ValueError("--max_low_texture_ratio must be in [0,1], or negative to disable")

    by_source: dict[str, dict[str, list[dict[str, Any]]]] = {}
    bindings: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, Any] | None = None
    render_roots = {
        "objaverse": Path(args.objaverse_render_root).expanduser().resolve(),
        "omni": Path(args.omni_render_root).expanduser().resolve(),
    }
    for source in SOURCES:
        rows, source_bindings, metadata = load_rendered_source(
            source=source,
            render_root=render_roots[source],
            source_plan_path=source_plan_path,
            source_plan=source_plan,
            sequences_per_object=args.sequences_per_object,
            required_renderer=args.required_renderer,
            reference_metadata=metadata,
        )
        by_source[source] = rows
        bindings[source] = source_bindings

    if set(by_source["objaverse"]) & set(by_source["omni"]):
        raise RuntimeError("object UID collision exists between the two sources")

    selected = {
        source: select_source_objects(
            by_source[source],
            source=source,
            quota=source_quotas[source],
            seed=args.seed,
            max_low_texture_ratio=args.max_low_texture_ratio,
        )
        for source in SOURCES
    }
    per_source_split = source_split_quotas(source_quotas, split_quotas)
    assignments = assign_splits(
        selected,
        quotas=per_source_split,
        seed=args.seed,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging.",
            dir=output_dir.parent,
        )
    )
    try:
        split_samples: dict[str, list[dict[str, Any]]] = {
            split: [] for split in SPLITS
        }
        selected_objects = []
        for source in SOURCES:
            for object_uid in selected[source]:
                assignment = assignments[object_uid]
                split = assignment["split"]
                rows = samples_for_object(
                    by_source[source][object_uid],
                    object_uid=object_uid,
                    seed=args.seed,
                    max_sequences=args.max_sequences_per_object,
                )
                if not rows:
                    raise RuntimeError(f"selected object has no accepted samples: {object_uid}")
                for row in rows:
                    split_samples[split].append(
                        materialize_sample(
                            row,
                            destination_root=staging,
                            split=split,
                            source=source,
                        )
                    )
                selected_objects.append(
                    {
                        "object_uid": object_uid,
                        "dataset_source": source,
                        "split": split,
                        "accepted_sequence_count": len(by_source[source][object_uid]),
                        "frozen_sample_uids": [str(row["uid"]) for row in rows],
                        "source_glb": str(rows[0]["source_glb"]),
                        "selection_key": stable_key(
                            args.seed,
                            source,
                            "object",
                            object_uid,
                        ),
                    }
                )

        for split in SPLITS:
            split_samples[split].sort(
                key=lambda row: stable_key(
                    args.seed,
                    split,
                    row["object_uid"],
                    row["uid"],
                )
            )
        all_samples = [
            row for split in SPLITS for row in split_samples[split]
        ]
        object_sets = {
            split: {str(row["object_uid"]) for row in split_samples[split]}
            for split in SPLITS
        }
        overlap = {
            f"{left}_{right}": sorted(object_sets[left] & object_sets[right])
            for index, left in enumerate(SPLITS)
            for right in SPLITS[index + 1 :]
        }
        if any(overlap.values()):
            raise RuntimeError(f"object leakage across splits: {overlap}")

        low_counts = {
            split: sum(
                bool(row.get("quality_flags", {}).get("low_texture", False))
                for row in split_samples[split]
            )
            for split in SPLITS
        }
        common = {
            "format": SCHEMA,
            **(metadata or {}),
            "image_root": str(output_dir / "images"),
            "mask_root": str(output_dir / "masks"),
            "latent_root": str(output_dir / "ss_latents"),
            "source_plan": str(source_plan_path),
            "source_plan_sha256": sha256_file(source_plan_path),
            "selection_policy": {
                "seed": int(args.seed),
                "object_order": "SHA256(seed|source|object|object_uid)",
                "sample_order": "SHA256(seed|sample|object_uid|sample_uid)",
                "source_object_quotas": source_quotas,
                "split_object_quotas": split_quotas,
                "per_source_split_object_quotas": per_source_split,
                "max_sequences_per_object": int(args.max_sequences_per_object),
                "max_low_texture_ratio_per_source": float(
                    args.max_low_texture_ratio
                ),
            },
            "source_bindings": bindings,
            "split_stats": {
                split: {
                    "object_count": len(object_sets[split]),
                    "sample_count": len(split_samples[split]),
                    "source_object_counts": dict(
                        Counter(
                            row["dataset_source"]
                            for row in selected_objects
                            if row["split"] == split
                        )
                    ),
                    "low_texture_sample_count": low_counts[split],
                }
                for split in SPLITS
            },
            "samples": all_samples,
        }
        write_json(staging / "manifest.json", common)
        for split in SPLITS:
            write_json(
                staging / f"{split}.json",
                {**common, "samples": split_samples[split]},
            )
        write_json(
            staging / "selected_objects.json",
            {
                "schema": SCHEMA,
                "objects": sorted(
                    selected_objects,
                    key=lambda row: (
                        row["split"],
                        row["dataset_source"],
                        row["object_uid"],
                    ),
                ),
            },
        )

        report = {
            "schema": SCHEMA,
            "passed": True,
            "output_dir": str(output_dir),
            "code_binding": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "object_count": len(selected_objects),
            "sample_count": len(all_samples),
            "source_object_counts": dict(
                Counter(row["dataset_source"] for row in selected_objects)
            ),
            "split_object_counts": {
                split: len(object_sets[split]) for split in SPLITS
            },
            "split_sample_counts": {
                split: len(split_samples[split]) for split in SPLITS
            },
            "low_texture_sample_counts": low_counts,
            "object_overlap_counts": {
                key: len(value) for key, value in overlap.items()
            },
            "render_bindings": bindings,
            "manifest_sha256": sha256_file(staging / "manifest.json"),
            "train_sha256": sha256_file(staging / "train.json"),
            "val_sha256": sha256_file(staging / "val.json"),
            "test_sha256": sha256_file(staging / "test.json"),
            "selected_objects_sha256": sha256_file(
                staging / "selected_objects.json"
            ),
            "hard_guards": {
                "exact_source_object_quotas": dict(
                    Counter(row["dataset_source"] for row in selected_objects)
                )
                == source_quotas,
                "exact_split_object_quotas": {
                    split: len(object_sets[split]) for split in SPLITS
                }
                == split_quotas,
                "object_disjoint_splits": not any(overlap.values()),
                "all_render_shards_complete_and_bound": True,
                "selected_assets_exist": True,
            },
        }
        if not all(report["hard_guards"].values()):
            raise RuntimeError(f"final freeze hard guard failed: {report['hard_guards']}")
        write_json(staging / "report.json", report)
        os.replace(staging, output_dir)
    except BaseException:
        # Keep staging for diagnosis and never expose a partial final dataset.
        raise

    print(
        json.dumps(
            {
                "reused": False,
                "passed": True,
                "output_dir": str(output_dir),
                "object_count": len(selected_objects),
                "sample_count": len(all_samples),
                "source_object_counts": report["source_object_counts"],
                "split_object_counts": report["split_object_counts"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
