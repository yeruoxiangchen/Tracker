#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


FORMAT = "pose_point_depth_mv.existing_render_mixed1k.v1"
PREVIEW_FORMAT = "pose_point_depth_mv.mixed1k_object_preview.v1"
REPORT_FORMAT = "pose_point_depth_mv.existing_render_mixed1k_report.v1"
COMPLETE_MARKER_FORMAT = "tracker.mixed_multiview_render_shard_complete.v1"
KEEP_SINGLE_SUBJECT = "keep_single_subject"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(root: str | None, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not root:
        return path
    return Path(root) / path


def bind_file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "sha256": file_sha256(path),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"manifest must contain a samples list: {path}")
    return payload


def normalized_sample(
    sample: dict[str, Any],
    payload: dict[str, Any],
    *,
    dataset_source: str,
    tier: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    row = copy.deepcopy(sample)
    image_root = row.pop("image_root", payload.get("image_root"))
    mask_root = row.pop("mask_root", payload.get("mask_root"))
    latent_root = row.pop("latent_root", payload.get("latent_root"))
    for frame in row.get("frames", []):
        frame["image"] = str(resolve_path(image_root, str(frame["image"])).resolve())
        if frame.get("mask"):
            frame["mask"] = str(
                resolve_path(mask_root, str(frame["mask"])).resolve()
            )
    row["ss_latent"] = str(
        resolve_path(latent_root, str(row["ss_latent"])).resolve()
    )
    row["source_glb"] = str(Path(str(row["source_glb"])).resolve())
    row["dataset_source"] = dataset_source
    row["single_object_tier"] = tier
    row["source_manifest"] = str(manifest_path.resolve())
    row["source_manifest_sha256"] = manifest_sha256
    return row


def require_paths(samples: Iterable[dict[str, Any]]) -> Counter[str]:
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


def load_objaverse(
    manifest_path: Path, audit_objects_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_manifest(manifest_path)
    manifest_sha = file_sha256(manifest_path)
    audited = load_json(audit_objects_path)
    if not isinstance(audited, list):
        raise ValueError(f"Objaverse audit objects must be a list: {audit_objects_path}")
    audited_by_uid = {str(row["object_uid"]): row for row in audited}
    if len(audited_by_uid) != len(audited):
        raise ValueError("Objaverse audit contains duplicate object_uid values")
    invalid = [
        uid
        for uid, row in audited_by_uid.items()
        if row.get("mesh_audit", {}).get("mesh_valid") is not True
    ]
    if invalid:
        raise ValueError(
            f"Objaverse audit contains non-hard-valid objects: {invalid[:10]}"
        )
    samples: list[dict[str, Any]] = []
    for sample in payload["samples"]:
        object_uid = str(sample["object_uid"])
        if object_uid not in audited_by_uid:
            raise KeyError(f"Objaverse object is absent from tier audit: {object_uid}")
        audit_row = audited_by_uid[object_uid]
        if Path(str(sample["source_glb"])).resolve() != Path(
            str(audit_row["source_glb"])
        ).resolve():
            raise ValueError(
                f"Objaverse source mesh differs from audit for {object_uid}"
            )
        samples.append(
            normalized_sample(
                sample,
                payload,
                dataset_source="objaverse",
                tier=str(audit_row["final_tier"]),
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha,
            )
        )
    objects = [
        {
            "object_uid": str(row["object_uid"]),
            "dataset_source": "objaverse",
            "single_object_tier": str(row["final_tier"]),
            "tier_reasons": list(row.get("tier_reasons", [])),
            "human_reviewed": bool(row.get("human_reviewed", False)),
            "semantic_subject_label": str(
                row.get("semantic_subject_label", "")
            ),
            "mesh_valid": True,
        }
        for row in audited
    ]
    return samples, objects


def validate_complete_marker(manifest_path: Path) -> Path:
    marker_path = manifest_path.parent / "_WORKER_COMPLETE.json"
    if not marker_path.is_file():
        raise FileNotFoundError(
            f"Omni manifest lacks completed-shard marker: {marker_path}"
        )
    marker = load_json(marker_path)
    if marker.get("schema") != COMPLETE_MARKER_FORMAT:
        raise ValueError(
            f"unsupported completed-shard marker schema: {marker_path}"
        )
    if marker.get("render_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError(f"completed-shard manifest hash changed: {manifest_path}")
    return marker_path


def load_omni(
    manifest_paths: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    seen_uids: dict[str, str] = {}
    object_manifests: dict[str, set[str]] = defaultdict(set)
    for manifest_path in sorted(manifest_paths):
        payload = load_manifest(manifest_path)
        manifest_sha = file_sha256(manifest_path)
        for sample in payload["samples"]:
            row = normalized_sample(
                sample,
                payload,
                dataset_source="omni",
                tier="OMNI_RENDER_ACCEPTED",
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha,
            )
            uid = str(row["uid"])
            fingerprint = canonical_sha256(row)
            if uid in seen_uids:
                if seen_uids[uid] != fingerprint:
                    raise ValueError(f"inconsistent duplicate Omni sample uid: {uid}")
                continue
            seen_uids[uid] = fingerprint
            samples.append(row)
            object_manifests[str(row["object_uid"])].add(str(manifest_path.resolve()))
    objects = [
        {
            "object_uid": object_uid,
            "dataset_source": "omni",
            "single_object_tier": "OMNI_RENDER_ACCEPTED",
            "tier_reasons": ["at least one sequence passed the render/quality pipeline"],
            "human_reviewed": False,
            "source_manifests": sorted(paths),
        }
        for object_uid, paths in sorted(object_manifests.items())
    ]
    return samples, objects


def unique_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, str] = {}
    output: list[dict[str, Any]] = []
    for sample in samples:
        uid = str(sample["uid"])
        fingerprint = canonical_sha256(sample)
        if uid in seen:
            if seen[uid] != fingerprint:
                raise ValueError(f"inconsistent duplicate sample uid: {uid}")
            continue
        seen[uid] = fingerprint
        output.append(sample)
    return output


def allocation_by_largest_remainder(
    group_sizes: dict[str, int], target: int
) -> dict[str, int]:
    total = sum(group_sizes.values())
    if target <= 0 or target > total:
        raise ValueError(f"invalid preview target={target} for objects={total}")
    exact = {key: target * value / total for key, value in group_sizes.items()}
    allocated = {key: min(value, int(math.floor(exact[key]))) for key, value in group_sizes.items()}
    remaining = target - sum(allocated.values())
    order = sorted(
        group_sizes,
        key=lambda key: (exact[key] - allocated[key], group_sizes[key], key),
        reverse=True,
    )
    for key in order:
        if remaining <= 0:
            break
        if allocated[key] < group_sizes[key]:
            allocated[key] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("failed to allocate the requested preview count")
    return allocated


def inspection_group(row: dict[str, Any]) -> str:
    if row["dataset_source"] == "omni":
        return "omni"
    return f"objaverse_{row['single_object_tier']}"


def select_preview(
    objects: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    samples_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        samples_by_object[str(sample["object_uid"])].append(sample)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in objects:
        object_uid = str(row["object_uid"])
        if object_uid in samples_by_object:
            groups[inspection_group(row)].append(row)
    sizes = {key: len(rows) for key, rows in groups.items()}
    allocation = allocation_by_largest_remainder(sizes, count)
    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    for group in sorted(groups):
        candidates = sorted(groups[group], key=lambda row: str(row["object_uid"]))
        chosen = rng.sample(candidates, allocation[group])
        for object_row in chosen:
            object_uid = str(object_row["object_uid"])
            sample = rng.choice(
                sorted(samples_by_object[object_uid], key=lambda row: str(row["uid"]))
            )
            frame_index = rng.randrange(len(sample["frames"]))
            frame = sample["frames"][frame_index]
            selected.append(
                {
                    **copy.deepcopy(object_row),
                    "inspection_group": group,
                    "sample_uid": str(sample["uid"]),
                    "frame_index": int(frame_index),
                    "source_view_index": frame.get("source_view_index"),
                    "source_image": str(frame["image"]),
                    "source_mask": str(frame.get("mask", "")),
                }
            )
    rng.shuffle(selected)
    if len(selected) != count or len({row["object_uid"] for row in selected}) != count:
        raise RuntimeError("preview selection is not object-unique")
    return selected, allocation


def make_thumbnail(image_path: Path, mask_path: Path | None, size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if mask_path is not None and mask_path.is_file():
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        background = Image.new("RGB", image.size, (128, 128, 128))
        background.paste(image, mask=mask)
        image = background
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def write_preview(
    preview_dir: Path,
    rows: list[dict[str, Any]],
    *,
    seed: int,
    allocation: dict[str, int],
    review_status: str = "pending",
) -> dict[str, Any]:
    if preview_dir.exists():
        raise FileExistsError(f"preview output already exists; preserve it: {preview_dir}")
    images_dir = preview_dir / "images"
    images_dir.mkdir(parents=True)
    copied: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source = Path(row["source_image"])
        suffix = source.suffix.lower() or ".png"
        target = images_dir / (
            f"{index:03d}_{row['inspection_group']}_{row['object_uid']}"
            f"_{row['sample_uid']}_view{row['frame_index']:02d}{suffix}"
        )
        shutil.copy2(source, target)
        copied.append(
            {
                **row,
                "preview_index": index,
                "copied_image": str(target.resolve()),
                "copied_image_sha256": file_sha256(target),
            }
        )

    cell = 192
    label_height = 44
    columns = 10
    rows_n = math.ceil(len(copied) / columns)
    sheet = Image.new("RGB", (columns * cell, rows_n * (cell + label_height)), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row in copied:
        index = int(row["preview_index"])
        x = (index % columns) * cell
        y = (index // columns) * (cell + label_height)
        thumb = make_thumbnail(
            Path(row["source_image"]),
            Path(row["source_mask"]) if row.get("source_mask") else None,
            cell,
        )
        sheet.paste(thumb, (x, y))
        label = f"{index:03d} {row['inspection_group']}\n{row['object_uid'][:22]}"
        draw.text((x + 3, y + cell + 3), label, fill=(245, 245, 245), font=font)
    sheet_path = preview_dir / "contact_sheet.jpg"
    sheet.save(sheet_path, quality=92, subsampling=0)

    csv_path = preview_dir / "preview_manifest.csv"
    fields = [
        "preview_index",
        "inspection_group",
        "dataset_source",
        "single_object_tier",
        "object_uid",
        "sample_uid",
        "frame_index",
        "source_view_index",
        "copied_image",
        "source_image",
        "source_glb",
        "review_decision",
        "review_note",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in copied:
            writer.writerow({**row, "review_decision": "", "review_note": ""})

    cards = []
    for row in copied:
        relative = Path(row["copied_image"]).relative_to(preview_dir)
        cards.append(
            "<article>"
            f"<img src='{html.escape(str(relative))}' loading='lazy'>"
            f"<div><b>{row['preview_index']:03d} · {html.escape(row['inspection_group'])}</b></div>"
            f"<code>{html.escape(row['object_uid'])}</code>"
            f"<div>{html.escape(row['sample_uid'])} / frame {row['frame_index']}</div>"
            "</article>"
        )
    index_path = preview_dir / "index.html"
    index_path.write_text(
        """<!doctype html><meta charset="utf-8"><title>Mixed1k 100-object review</title>
<style>body{background:#181818;color:#eee;font:14px sans-serif;margin:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}
article{background:#282828;padding:10px;border-radius:7px;overflow:hidden}
img{width:100%;aspect-ratio:1;object-fit:contain;background:#888}code{font-size:11px}</style>
"""
        + f"<h1>Mixed1k 对象级随机抽检</h1><p>seed={seed}; allocation={html.escape(str(allocation))}</p>"
        + "<div class='grid'>"
        + "\n".join(cards)
        + "</div>",
        encoding="utf-8",
    )

    manifest_body = {
        "format": PREVIEW_FORMAT,
        "review_status": review_status,
        "selection_unit": "unique object_uid",
        "seed": int(seed),
        "count": len(copied),
        "allocation": allocation,
        "records": copied,
        "contact_sheet": str(sheet_path.resolve()),
        "html": str(index_path.resolve()),
        "review_csv": str(csv_path.resolve()),
    }
    manifest = {**manifest_body, "preview_sha256": canonical_sha256(manifest_body)}
    (preview_dir / "preview_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def write_dataset(
    output_dir: Path,
    *,
    samples: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    preview_manifest: dict[str, Any],
    source_bindings: list[dict[str, Any]],
    seed: int,
    training_mode: str = "draft",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"dataset output already exists; preserve it: {output_dir}")
    initial_finetune = training_mode == "initial_finetune"
    if initial_finetune:
        unreviewed_objaverse = [
            str(row["object_uid"])
            for row in objects
            if row["dataset_source"] == "objaverse"
            and (
                row.get("human_reviewed") is not True
                or row.get("semantic_subject_label") != KEEP_SINGLE_SUBJECT
            )
        ]
        if unreviewed_objaverse:
            raise ValueError(
                "initial_finetune requires every admitted Objaverse object to be "
                "human-reviewed as keep_single_subject; invalid objects: "
                f"{unreviewed_objaverse[:10]} (count={len(unreviewed_objaverse)})"
            )
    output_dir.mkdir(parents=True)
    sample_counts = Counter(row["dataset_source"] for row in samples)
    object_counts = Counter(row["dataset_source"] for row in objects)
    tier_object_counts = Counter(inspection_group(row) for row in objects)
    tier_sample_counts = Counter(inspection_group(row) for row in samples)
    review_status = "objaverse_reviewed" if initial_finetune else "pending"
    scope_guard = (
        "Training-ready only for initial nonformal fine-tuning and engineering "
        "evaluation. Every admitted Objaverse object was explicitly reviewed as "
        "keep_single_subject. Completed-pilot Omni objects are admitted when at "
        "least one sequence passed rendering/quality checks under the single-object "
        "source-family assumption. This is not a confirmatory evaluation freeze."
        if initial_finetune
        else (
            "Draft manifest only. It may be used for cache/runtime smoke tests, "
            "but must not be called a reviewed single-object training freeze "
            "until the preview and C-tier objects receive explicit review."
        )
    )
    body = {
        "format": FORMAT,
        "review_status": review_status,
        "formal": False,
        "training_ready": initial_finetune,
        "training_mode": training_mode,
        "split": "train",
        "selection_policy": (
            "human-reviewed keep_single_subject Objaverse objects plus every unique "
            "Omni object with at least one accepted sequence in completed pilot shards"
        ),
        "seed": int(seed),
        "image_root": "/",
        "mask_root": "/",
        "latent_root": "/",
        "extrinsics_type": "c2w",
        "camera_forward_sign": 1.0,
        "coordinate_frame": "normalized_object",
        "canonical_latent_frame": "TRELLIS_64",
        "images_are_masked": True,
        "samples": sorted(samples, key=lambda row: str(row["uid"])),
        "object_records": sorted(objects, key=lambda row: str(row["object_uid"])),
        "summary": {
            "sample_count": len(samples),
            "object_count": len(objects),
            "sample_counts_by_source": dict(sorted(sample_counts.items())),
            "object_counts_by_source": dict(sorted(object_counts.items())),
            "sample_counts_by_inspection_group": dict(sorted(tier_sample_counts.items())),
            "object_counts_by_inspection_group": dict(sorted(tier_object_counts.items())),
        },
        "preview_binding": {
            "path": str(
                (Path(preview_manifest["html"]).parent / "preview_manifest.json").resolve()
            ),
            "preview_sha256": preview_manifest["preview_sha256"],
            "review_status": preview_manifest["review_status"],
        },
        "source_bindings": source_bindings,
        "admission_policy": {
            "legacy_objaverse": (
                "explicit human review decision keep_single_subject, successful "
                "render history, and hard-valid mesh audit"
            ),
            "omni": (
                "at least one accepted sequence in a completed render shard; "
                "single-object source-family assumption; no manual semantic review"
            ),
            "fresh_objaverse": "not included",
        },
        "scope_guard": scope_guard,
    }
    manifest = {**body, "manifest_sha256": canonical_sha256(body)}
    manifest_path = output_dir / (
        "train.json" if initial_finetune else "train_draft.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report_body = {
        "format": REPORT_FORMAT,
        "passed": True,
        "formal": False,
        "training_ready": initial_finetune,
        "training_mode": training_mode,
        "review_status": review_status,
        "manifest": bind_file(manifest_path),
        "manifest_identity": manifest["manifest_sha256"],
        "summary": body["summary"],
        "path_audit": {
            "missing_counts": {},
            "all_referenced_files_exist": True,
        },
        "preview_binding": body["preview_binding"],
        "source_bindings": source_bindings,
        "scope_guard": body["scope_guard"],
    }
    report = {**report_body, "report_sha256": canonical_sha256(report_body)}
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                "Existing-render Mixed1k draft dataset",
                "=====================================",
                f"objects: {len(objects)}",
                f"samples: {len(samples)}",
                f"objects by group: {dict(sorted(tier_object_counts.items()))}",
                f"samples by group: {dict(sorted(tier_sample_counts.items()))}",
                f"training_ready: {str(initial_finetune).lower()}",
                f"training_mode: {training_mode}",
                "formal: false",
                f"review_status: {review_status}",
                f"manifest: {manifest_path.resolve()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a draft mixed dataset view from existing Objaverse and Omni renders."
    )
    parser.add_argument("--objaverse_manifest", required=True)
    parser.add_argument("--objaverse_audit_objects", required=True)
    parser.add_argument("--omni_manifest", action="append", default=[], required=True)
    parser.add_argument("--preview_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--preview_count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--training_mode",
        choices=("draft", "initial_finetune"),
        default="draft",
        help=(
            "initial_finetune marks the immutable output training-ready only for "
            "nonformal bootstrap fine-tuning and requires every admitted Objaverse "
            "object to be human-reviewed as keep_single_subject"
        ),
    )
    args = parser.parse_args()

    obj_manifest = Path(args.objaverse_manifest).resolve()
    obj_audit = Path(args.objaverse_audit_objects).resolve()
    omni_manifests = [Path(value).resolve() for value in args.omni_manifest]
    omni_markers = [validate_complete_marker(path) for path in omni_manifests]
    source_paths = [
        obj_manifest,
        obj_audit,
        *omni_manifests,
        *omni_markers,
    ]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    obj_samples, obj_objects = load_objaverse(obj_manifest, obj_audit)
    omni_samples, omni_objects = load_omni(omni_manifests)
    samples = unique_samples([*obj_samples, *omni_samples])
    objects_by_uid: dict[str, dict[str, Any]] = {}
    for row in [*obj_objects, *omni_objects]:
        uid = str(row["object_uid"])
        if uid in objects_by_uid:
            raise ValueError(f"duplicate object_uid across sources: {uid}")
        objects_by_uid[uid] = row
    objects = list(objects_by_uid.values())
    sample_objects = {str(row["object_uid"]) for row in samples}
    if sample_objects != set(objects_by_uid):
        raise ValueError(
            f"object/sample mismatch: objects_only={len(set(objects_by_uid)-sample_objects)} "
            f"samples_only={len(sample_objects-set(objects_by_uid))}"
        )
    missing = require_paths(samples)
    if missing:
        raise FileNotFoundError(f"mixed dataset has missing referenced files: {dict(missing)}")

    preview_rows, allocation = select_preview(
        objects, samples, count=int(args.preview_count), seed=int(args.seed)
    )
    preview_manifest = write_preview(
        Path(args.preview_dir).resolve(),
        preview_rows,
        seed=int(args.seed),
        allocation=allocation,
        review_status=(
            "objaverse_reviewed"
            if args.training_mode == "initial_finetune"
            else "pending"
        ),
    )
    source_bindings = [bind_file(path) for path in source_paths]
    report = write_dataset(
        Path(args.output_dir).resolve(),
        samples=samples,
        objects=objects,
        preview_manifest=preview_manifest,
        source_bindings=source_bindings,
        seed=int(args.seed),
        training_mode=str(args.training_mode),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
