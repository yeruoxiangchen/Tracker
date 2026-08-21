#!/usr/bin/env python3
"""Freeze reviewed Objaverse and completed Omni renders into split manifests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FORMAT = "pose_point_depth_mv.reviewed_mixed1k.v1"
MARKER_FORMAT = "tracker.mixed_multiview_render_shard_complete.v1"
KEEP_LABEL = "keep_single_subject"
SPLITS = ("train", "val", "test")
RENDER_METADATA_KEYS = (
    "format",
    "extrinsics_type",
    "camera_forward_sign",
    "coordinate_frame",
    "canonical_latent_frame",
    "num_views",
    "images_are_masked",
    "image_size",
    "voxel_resolution",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "sha256": file_sha256(resolved),
    }


def stable_key(seed: int, *values: str) -> str:
    text = "|".join((str(seed), *values))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_path(root: str | None, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and root:
        path = Path(root) / path
    return path.resolve()


def object_uid(sample: dict[str, Any]) -> str:
    value = str(sample.get("object_uid", ""))
    if value:
        return value
    uid = str(sample.get("uid", ""))
    return uid.rsplit("_seq", 1)[0] if "_seq" in uid else uid


def normalize_sample(
    sample: dict[str, Any],
    payload: dict[str, Any],
    manifest_path: Path,
    *,
    dataset_source: str,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    row = copy.deepcopy(sample)
    image_root = row.pop("image_root", payload.get("image_root"))
    mask_root = row.pop("mask_root", payload.get("mask_root"))
    latent_root = row.pop("latent_root", payload.get("latent_root"))
    row["object_uid"] = object_uid(row)
    row["source_glb"] = str(
        resolve_path(str(manifest_path.parent), str(row["source_glb"]))
    )
    row["ss_latent"] = str(resolve_path(latent_root, str(row["ss_latent"])))
    for frame in row.get("frames", []):
        frame["image"] = str(resolve_path(image_root, str(frame["image"])))
        if frame.get("mask"):
            frame["mask"] = str(resolve_path(mask_root, str(frame["mask"])))
    row["dataset_source"] = dataset_source
    row["source_manifest"] = str(manifest_path.resolve())
    row["source_manifest_sha256"] = (
        manifest_sha256 if manifest_sha256 is not None else file_sha256(manifest_path)
    )
    return row


def load_manifest(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"manifest must contain a samples list: {path}")
    return payload


def completed_render_manifests(
    render_root: Path,
    *,
    source: str,
    expected_shards: int,
) -> tuple[list[Path], list[Path]]:
    source_root = render_root.resolve() / source
    marker_paths = sorted(source_root.glob("shard_*/_WORKER_COMPLETE.json"))
    actual_indices: set[int] = set()
    manifest_paths: list[Path] = []
    for marker_path in marker_paths:
        marker = load_json(marker_path)
        if marker.get("schema") != MARKER_FORMAT:
            raise ValueError(f"unsupported completion marker: {marker_path}")
        if str(marker.get("source")) != source:
            raise ValueError(f"completion marker source mismatch: {marker_path}")
        index = int(marker["shard_index"])
        if index in actual_indices:
            raise ValueError(f"duplicate completed shard index {index}: {render_root}")
        actual_indices.add(index)
        manifest_path = marker_path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if marker.get("render_manifest_sha256") != file_sha256(manifest_path):
            raise ValueError(f"completed render manifest changed: {manifest_path}")
        manifest_paths.append(manifest_path.resolve())
    if len(actual_indices) != int(expected_shards):
        raise ValueError(
            f"completed shard mismatch for {render_root}: "
            f"expected_count={expected_shards} actual_indices={sorted(actual_indices)}"
        )
    return manifest_paths, [path.resolve() for path in marker_paths]


def load_reviewed_objects(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    source_identities: dict[str, str] = {}
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"reviewed object input must be a list: {path}")
        for raw in payload:
            row = copy.deepcopy(raw)
            uid = str(row.get("object_uid", ""))
            if not uid:
                raise ValueError(f"reviewed object has no object_uid: {path}")
            if uid in output:
                raise ValueError(f"duplicate reviewed object_uid: {uid}")
            if row.get("human_reviewed") is not True:
                raise ValueError(f"Objaverse object is not human-reviewed: {uid}")
            if str(row.get("semantic_subject_label", "")) != KEEP_LABEL:
                raise ValueError(f"Objaverse object is not accepted: {uid}")
            reviewer = str(row.get("semantic_reviewer", row.get("reviewer", "")))
            if not reviewer:
                raise ValueError(f"Objaverse review has no reviewer: {uid}")
            source = str(resolve_path(None, str(row["source_glb"])))
            if source in source_identities:
                raise ValueError(
                    "duplicate reviewed source mesh: "
                    f"{source_identities[source]} and {uid} -> {source}"
                )
            source_identities[source] = uid
            row["object_uid"] = uid
            row["source_glb"] = source
            row["semantic_reviewer"] = reviewer
            row["review_source"] = str(path.resolve())
            output[uid] = row
    if not output:
        raise ValueError("no reviewed Objaverse objects were provided")
    return output


def collect_objaverse(
    reviewed: dict[str, dict[str, Any]],
    manifest_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    samples_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata_values: dict[str, set[str]] = defaultdict(set)
    seen_sample_uids: set[str] = set()
    for manifest_path in manifest_paths:
        payload = load_manifest(manifest_path)
        manifest_sha = file_sha256(manifest_path)
        for key in RENDER_METADATA_KEYS:
            metadata_values[key].add(json.dumps(payload.get(key), sort_keys=True))
        for sample in payload["samples"]:
            uid = object_uid(sample)
            if uid not in reviewed:
                continue
            row = normalize_sample(
                sample,
                payload,
                manifest_path,
                dataset_source="objaverse",
                manifest_sha256=manifest_sha,
            )
            sample_uid = str(row.get("uid", ""))
            if not sample_uid:
                raise ValueError(f"Objaverse sample has no uid: {manifest_path}")
            if sample_uid in seen_sample_uids:
                raise ValueError(f"duplicate Objaverse sample uid: {sample_uid}")
            seen_sample_uids.add(sample_uid)
            samples_by_object[uid].append(row)

    inconsistent = {
        key: sorted(values) for key, values in metadata_values.items() if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"Objaverse manifest metadata mismatch: {inconsistent}")

    samples: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    for uid, review in sorted(reviewed.items()):
        object_samples = samples_by_object.get(uid, [])
        if not object_samples:
            raise ValueError(f"reviewed Objaverse object has no accepted sample: {uid}")
        sample_sources = {str(row["source_glb"]) for row in object_samples}
        if sample_sources != {str(review["source_glb"])}:
            raise ValueError(
                f"review/source manifest mesh mismatch for {uid}: "
                f"review={review['source_glb']} samples={sorted(sample_sources)}"
            )
        source_group = str(review.get("source_group", "reviewed_objaverse"))
        for row in object_samples:
            row["source_group"] = source_group
            row["semantic_subject_label"] = KEEP_LABEL
            row["semantic_reviewer"] = str(review["semantic_reviewer"])
        samples.extend(object_samples)
        objects.append(
            {
                "object_uid": uid,
                "dataset_source": "objaverse",
                "source_group": source_group,
                "source_glb": str(review["source_glb"]),
                "human_reviewed": True,
                "semantic_subject_label": KEEP_LABEL,
                "semantic_reviewer": str(review["semantic_reviewer"]),
                "review_id": str(review.get("review_id", "")),
                "review_source": str(review["review_source"]),
                "sample_count": len(object_samples),
            }
        )
    metadata = {
        key: json.loads(next(iter(values))) for key, values in metadata_values.items()
    }
    return samples, objects, metadata


def collect_omni(
    manifest_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    samples_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metadata_values: dict[str, set[str]] = defaultdict(set)
    seen_sample_uids: set[str] = set()
    for manifest_path in manifest_paths:
        payload = load_manifest(manifest_path)
        manifest_sha = file_sha256(manifest_path)
        for key in RENDER_METADATA_KEYS:
            metadata_values[key].add(json.dumps(payload.get(key), sort_keys=True))
        for sample in payload["samples"]:
            row = normalize_sample(
                sample,
                payload,
                manifest_path,
                dataset_source="omni",
                manifest_sha256=manifest_sha,
            )
            uid = str(row.get("uid", ""))
            if not uid:
                raise ValueError(f"Omni sample has no uid: {manifest_path}")
            if uid in seen_sample_uids:
                raise ValueError(f"duplicate Omni sample uid: {uid}")
            seen_sample_uids.add(uid)
            row["source_group"] = "omni_completed_render"
            samples.append(row)
            samples_by_object[str(row["object_uid"])].append(row)
    objects = []
    for uid, rows in sorted(samples_by_object.items()):
        sources = {str(row["source_glb"]) for row in rows}
        if len(sources) != 1:
            raise ValueError(f"Omni object has multiple source meshes: {uid}")
        objects.append(
            {
                "object_uid": uid,
                "dataset_source": "omni",
                "source_group": "omni_completed_render",
                "source_glb": next(iter(sources)),
                "human_reviewed": False,
                "semantic_subject_label": "source_family_single_object_assumption",
                "sample_count": len(rows),
            }
        )
    if not objects:
        raise ValueError("no accepted Omni objects were found")
    inconsistent = {
        key: sorted(values) for key, values in metadata_values.items() if len(values) != 1
    }
    if inconsistent:
        raise ValueError(f"Omni manifest metadata mismatch: {inconsistent}")
    metadata = {
        key: json.loads(next(iter(values))) for key, values in metadata_values.items()
    }
    return samples, objects, metadata


def referenced_path_failures(samples: Iterable[dict[str, Any]]) -> Counter[str]:
    missing: Counter[str] = Counter()
    for sample in samples:
        if not Path(str(sample["source_glb"])).is_file():
            missing["source_glb"] += 1
        if not Path(str(sample["ss_latent"])).is_file():
            missing["ss_latent"] += 1
        for frame in sample.get("frames", []):
            if not Path(str(frame["image"])).is_file():
                missing["image"] += 1
            if frame.get("mask") and not Path(str(frame["mask"])).is_file():
                missing["mask"] += 1
    return missing


def excluded_source_meshes(paths: Iterable[Path]) -> set[str]:
    output: set[str] = set()
    for path in paths:
        payload = load_manifest(path)
        for sample in payload["samples"]:
            output.add(
                str(resolve_path(str(path.parent), str(sample["source_glb"])))
            )
    return output


def largest_remainder(group_sizes: dict[str, int], target: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    if target < 0 or target > total:
        raise ValueError(f"cannot allocate target={target} over total={total}")
    if target == 0:
        return {key: 0 for key in group_sizes}
    exact = {key: target * size / total for key, size in group_sizes.items()}
    result = {key: min(group_sizes[key], int(exact[key])) for key in group_sizes}
    remaining = target - sum(result.values())
    order = sorted(
        group_sizes,
        key=lambda key: (exact[key] - result[key], group_sizes[key], key),
        reverse=True,
    )
    for key in order:
        if remaining == 0:
            break
        if result[key] < group_sizes[key]:
            result[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("largest-remainder allocation did not reach target")
    return result


def split_objects(
    objects: list[dict[str, Any]],
    *,
    seed: int,
    val_count: int,
    test_count: int,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in objects:
        group = f"{row['dataset_source']}:{row['source_group']}"
        groups[group].append(row)
    sizes = {key: len(rows) for key, rows in groups.items()}
    test_quota = largest_remainder(sizes, int(test_count))
    remaining_sizes = {key: sizes[key] - test_quota[key] for key in sizes}
    val_quota = largest_remainder(remaining_sizes, int(val_count))
    output = {split: [] for split in SPLITS}
    for group, rows in sorted(groups.items()):
        ordered = sorted(
            rows,
            key=lambda row: stable_key(seed, group, str(row["object_uid"])),
        )
        test_end = test_quota[group]
        val_end = test_end + val_quota[group]
        for split, selected in (
            ("test", ordered[:test_end]),
            ("val", ordered[test_end:val_end]),
            ("train", ordered[val_end:]),
        ):
            for row in selected:
                output[split].append({**row, "split": split})
    for split in SPLITS:
        output[split].sort(key=lambda row: str(row["object_uid"]))
    return output


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    existing_report: dict[str, Any] | None = None
    if output_dir.exists():
        report_path = output_dir / "report.json"
        if report_path.is_file() and load_json(report_path).get("passed") is True:
            existing_report = load_json(report_path)
        else:
            raise FileExistsError(f"partial immutable output exists: {output_dir}")

    review_paths = [Path(value).resolve() for value in args.objaverse_review]
    legacy_manifest = Path(args.legacy_objaverse_manifest).resolve()
    exclusion_paths = [Path(value).resolve() for value in args.exclude_manifest]
    for path in [legacy_manifest, *review_paths, *exclusion_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)

    obj_render_manifests: list[Path] = []
    obj_markers: list[Path] = []
    for value in args.objaverse_render_root:
        manifests, markers = completed_render_manifests(
            Path(value),
            source="objaverse",
            expected_shards=int(args.expected_objaverse_shards_per_root),
        )
        obj_render_manifests.extend(manifests)
        obj_markers.extend(markers)
    omni_manifests: list[Path] = []
    omni_markers: list[Path] = []
    for value in args.omni_render_root:
        manifests, markers = completed_render_manifests(
            Path(value),
            source="omni",
            expected_shards=int(args.expected_omni_shards_per_root),
        )
        omni_manifests.extend(manifests)
        omni_markers.extend(markers)

    reviewed = load_reviewed_objects(review_paths)
    obj_samples, obj_objects, metadata = collect_objaverse(
        reviewed, [legacy_manifest, *obj_render_manifests]
    )
    omni_samples, omni_objects, omni_metadata = collect_omni(omni_manifests)
    if metadata != omni_metadata:
        mismatch = {
            key: {"objaverse": metadata.get(key), "omni": omni_metadata.get(key)}
            for key in RENDER_METADATA_KEYS
            if metadata.get(key) != omni_metadata.get(key)
        }
        raise ValueError(f"Objaverse/Omni render metadata mismatch: {mismatch}")
    samples = [*obj_samples, *omni_samples]
    objects = [*obj_objects, *omni_objects]

    sample_uids = [str(row["uid"]) for row in samples]
    if len(sample_uids) != len(set(sample_uids)):
        raise ValueError("sample uid collision across admitted sources")
    object_uids = [str(row["object_uid"]) for row in objects]
    if len(object_uids) != len(set(object_uids)):
        raise ValueError("object uid collision across admitted sources")
    source_to_uid: dict[str, str] = {}
    for row in objects:
        source = str(row["source_glb"])
        if source in source_to_uid:
            raise ValueError(
                f"source mesh collision: {source_to_uid[source]} and "
                f"{row['object_uid']} -> {source}"
            )
        source_to_uid[source] = str(row["object_uid"])
    if not int(args.min_objects) <= len(objects) <= int(args.max_objects):
        raise ValueError(
            f"admitted object count {len(objects)} is outside "
            f"[{args.min_objects}, {args.max_objects}]"
        )

    excluded = excluded_source_meshes(exclusion_paths)
    leaked = sorted(set(source_to_uid) & excluded)
    if leaked:
        raise ValueError(f"old val/holdout source mesh leaked into reviewed pool: {leaked[:10]}")
    missing = referenced_path_failures(samples)
    if missing:
        raise FileNotFoundError(f"admitted samples reference missing files: {dict(missing)}")

    split_records = split_objects(
        objects,
        seed=int(args.seed),
        val_count=int(args.val_objects),
        test_count=int(args.test_objects),
    )
    split_by_object = {
        str(row["object_uid"]): split
        for split, rows in split_records.items()
        for row in rows
    }
    split_samples = {split: [] for split in SPLITS}
    for row in samples:
        split_samples[split_by_object[str(row["object_uid"])]].append(row)
    for split in SPLITS:
        split_samples[split].sort(key=lambda row: str(row["uid"]))

    object_sets = {
        split: {str(row["object_uid"]) for row in rows}
        for split, rows in split_records.items()
    }
    source_sets = {
        split: {str(row["source_glb"]) for row in rows}
        for split, rows in split_records.items()
    }
    object_overlap = {
        f"{left}_{right}": sorted(object_sets[left] & object_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    source_overlap = {
        f"{left}_{right}": sorted(source_sets[left] & source_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if any(object_overlap.values()) or any(source_overlap.values()):
        raise RuntimeError("object or source mesh leakage across frozen splits")

    source_files = [
        legacy_manifest,
        *review_paths,
        *exclusion_paths,
        *obj_render_manifests,
        *obj_markers,
        *omni_manifests,
        *omni_markers,
    ]
    bindings = [bind_file(path) for path in source_files]
    freeze_config = {
        "legacy_objaverse_manifest": str(legacy_manifest),
        "objaverse_reviews": [str(path) for path in review_paths],
        "objaverse_render_roots": [
            str(Path(value).resolve()) for value in args.objaverse_render_root
        ],
        "omni_render_roots": [
            str(Path(value).resolve()) for value in args.omni_render_root
        ],
        "exclude_manifests": [str(path) for path in exclusion_paths],
        "expected_objaverse_shards_per_root": int(
            args.expected_objaverse_shards_per_root
        ),
        "expected_omni_shards_per_root": int(args.expected_omni_shards_per_root),
        "val_objects": int(args.val_objects),
        "test_objects": int(args.test_objects),
        "min_objects": int(args.min_objects),
        "max_objects": int(args.max_objects),
        "seed": int(args.seed),
    }
    code_binding = bind_file(Path(__file__))
    if existing_report is not None:
        if existing_report.get("source_bindings") != bindings:
            raise ValueError("complete output source bindings differ from this invocation")
        if existing_report.get("freeze_config") != freeze_config:
            raise ValueError("complete output freeze config differs from this invocation")
        if existing_report.get("code_binding") != code_binding:
            raise ValueError("complete output code binding differs from current code")
        for name, key in (
            ("manifest.json", "manifest_sha256"),
            ("train.json", "train_sha256"),
            ("val.json", "val_sha256"),
            ("test.json", "test_sha256"),
        ):
            path = output_dir / name
            if not path.is_file() or file_sha256(path) != existing_report.get(key):
                raise ValueError(f"complete output artifact changed: {path}")
        print(
            json.dumps(
                {"reused": True, **existing_report["summary"]},
                indent=2,
                ensure_ascii=False,
            )
        )
        return existing_report
    staging = Path(f"{output_dir}.staging")
    if staging.exists():
        raise FileExistsError(f"stale staging output exists: {staging}")
    staging.mkdir(parents=True)
    try:
        split_stats = {
            split: {
                "object_count": len(split_records[split]),
                "sample_count": len(split_samples[split]),
                "object_counts_by_group": dict(
                    sorted(
                        Counter(
                            f"{row['dataset_source']}:{row['source_group']}"
                            for row in split_records[split]
                        ).items()
                    )
                ),
            }
            for split in SPLITS
        }
        common = {
            "format": FORMAT,
            "status": "complete",
            "training_ready": True,
            "formal_confirmatory": False,
            "selection_policy": (
                "all explicitly reviewed keep_single_subject Objaverse objects plus "
                "all Omni objects with at least one accepted sequence in fully "
                "completed immutable render shards"
            ),
            "split_policy": {
                "seed": int(args.seed),
                "method": "source-group-stratified SHA256 order",
                "val_objects": int(args.val_objects),
                "test_objects": int(args.test_objects),
            },
            **metadata,
            "image_root": "/",
            "mask_root": "/",
            "latent_root": "/",
            "source_bindings": bindings,
            "freeze_config": freeze_config,
            "split_stats": split_stats,
        }
        write_json(
            staging / "manifest.json",
            {
                **common,
                "split": "all",
                "samples": sorted(samples, key=lambda row: str(row["uid"])),
                "object_records": sorted(objects, key=lambda row: str(row["object_uid"])),
            },
        )
        for split in SPLITS:
            write_json(
                staging / f"{split}.json",
                {
                    **common,
                    "split": split,
                    "samples": split_samples[split],
                    "object_records": split_records[split],
                },
            )
        summary = {
            "object_count": len(objects),
            "sample_count": len(samples),
            "object_counts_by_source": dict(
                sorted(Counter(row["dataset_source"] for row in objects).items())
            ),
            "object_counts_by_group": dict(
                sorted(
                    Counter(
                        f"{row['dataset_source']}:{row['source_group']}" for row in objects
                    ).items()
                )
            ),
            "split_object_counts": {
                split: len(split_records[split]) for split in SPLITS
            },
            "split_sample_counts": {
                split: len(split_samples[split]) for split in SPLITS
            },
        }
        hard_guards = {
            "object_count_within_requested_range": True,
            "every_objaverse_object_human_reviewed_keep": all(
                row["human_reviewed"] is True
                and row["semantic_subject_label"] == KEEP_LABEL
                for row in obj_objects
            ),
            "all_render_shards_complete_and_hash_bound": True,
            "sample_uids_unique": len(sample_uids) == len(set(sample_uids)),
            "object_uids_unique": len(object_uids) == len(set(object_uids)),
            "resolved_source_meshes_unique": len(source_to_uid) == len(objects),
            "old_val_holdout_sources_excluded": not leaked,
            "all_referenced_assets_exist": not missing,
            "object_disjoint_splits": not any(object_overlap.values()),
            "source_mesh_disjoint_splits": not any(source_overlap.values()),
            "exact_val_test_object_counts": (
                len(split_records["val"]) == int(args.val_objects)
                and len(split_records["test"]) == int(args.test_objects)
            ),
        }
        if not all(hard_guards.values()):
            raise RuntimeError(f"reviewed Mixed1k hard guard failed: {hard_guards}")
        report = {
            "format": FORMAT,
            "status": "complete",
            "passed": True,
            "output_dir": str(output_dir),
            "summary": summary,
            "hard_guards": hard_guards,
            "source_bindings": bindings,
            "freeze_config": freeze_config,
            "manifest_sha256": file_sha256(staging / "manifest.json"),
            "train_sha256": file_sha256(staging / "train.json"),
            "val_sha256": file_sha256(staging / "val.json"),
            "test_sha256": file_sha256(staging / "test.json"),
            "code_binding": code_binding,
        }
        write_json(staging / "report.json", report)
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()
        raise
    print(json.dumps({"reused": False, **summary}, indent=2, ensure_ascii=False))
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy_objaverse_manifest", required=True)
    parser.add_argument("--objaverse_review", action="append", required=True)
    parser.add_argument("--objaverse_render_root", action="append", required=True)
    parser.add_argument("--omni_render_root", action="append", required=True)
    parser.add_argument("--exclude_manifest", action="append", default=[])
    parser.add_argument("--expected_objaverse_shards_per_root", type=int, default=16)
    parser.add_argument("--expected_omni_shards_per_root", type=int, default=4)
    parser.add_argument("--val_objects", type=int, default=64)
    parser.add_argument("--test_objects", type=int, default=64)
    parser.add_argument("--min_objects", type=int, default=950)
    parser.add_argument("--max_objects", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    freeze(make_parser().parse_args())


if __name__ == "__main__":
    main()
