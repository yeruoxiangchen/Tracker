#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


FORMAT = "pose_point_depth_mv.objaverse_semantic_review.v1"
COMPLETE_MARKER_FORMAT = "tracker.mixed_multiview_render_shard_complete.v1"
DECISIONS = {
    "keep_single_subject",
    "reject_scene_or_fragment",
    "uncertain_review",
}
DECISION_ALIASES = {
    "1": "keep_single_subject",
    "0": "reject_scene_or_fragment",
    **{value: value for value in DECISIONS},
}


def normalize_decision(value: Any) -> str:
    label = str(value).strip()
    if label not in DECISION_ALIASES:
        raise ValueError(f"missing/invalid semantic decision: {label!r}")
    return DECISION_ALIASES[label]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(root: str | Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(root) / path
    return path.resolve()


def sample_object_uid(row: dict[str, Any]) -> str:
    value = str(row.get("object_uid", ""))
    if value:
        return value
    uid = str(row.get("uid", ""))
    return uid.rsplit("_seq", 1)[0] if "_seq" in uid else uid


def source_identities(paths: Iterable[Path]) -> set[str]:
    identities: set[str] = set()
    for path in paths:
        payload = load_json(path)
        for row in [*payload.get("samples", []), *payload.get("failures", [])]:
            source = str(row.get("source_glb", ""))
            if source:
                identities.add(str(resolve_path(path.parent, source)))
    return identities


def find_review_tile(images_root: Path, uid: str) -> Path:
    matches = sorted(images_root.glob(f"*/{uid}.jpg"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one four-view tile for {uid}, found {len(matches)}"
        )
    return matches[0].resolve()


def legacy_candidates(objects_path: Path, images_root: Path) -> list[dict[str, Any]]:
    payload = load_json(objects_path)
    if not isinstance(payload, list):
        raise ValueError(f"legacy objects must be a list: {objects_path}")
    rows = []
    for row in payload:
        uid = str(row["object_uid"])
        if row.get("mesh_audit", {}).get("mesh_valid") is not True:
            continue
        rows.append(
            {
                "source_group": "legacy897",
                "object_uid": uid,
                "source_glb": str(Path(str(row["source_glb"])).resolve()),
                "auto_tier": str(row.get("auto_tier", "")),
                "review_tile": str(find_review_tile(images_root, uid)),
                "source_object": row,
            }
        )
    if len(rows) != 897:
        raise ValueError(f"expected 897 hard-valid legacy objects, found {len(rows)}")
    return rows


def completed_manifests(render_root: Path) -> list[Path]:
    manifests = []
    for marker_path in sorted(render_root.glob("*/shard_*/_WORKER_COMPLETE.json")):
        manifest_path = marker_path.parent / "manifest.json"
        marker = load_json(marker_path)
        if marker.get("schema") != COMPLETE_MARKER_FORMAT:
            raise ValueError(f"unsupported complete marker: {marker_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if marker.get("render_manifest_sha256") != file_sha256(manifest_path):
            raise ValueError(f"completed manifest hash changed: {manifest_path}")
        manifests.append(manifest_path)
    if not manifests:
        raise ValueError(f"no completed render manifests under {render_root}")
    return manifests


def make_tile(paths: list[Path], title: str, *, cell: int = 160) -> Image.Image:
    if len(paths) > 4:
        indices = [round(index * (len(paths) - 1) / 3) for index in range(4)]
        paths = [paths[index] for index in indices]
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    if not images:
        raise ValueError(f"candidate has no readable preview frames: {title}")
    header = 24
    tile = Image.new("RGB", (cell * len(images), cell + header), "black")
    draw = ImageDraw.Draw(tile)
    draw.text((4, 4), title[:92], fill="white", font=ImageFont.load_default())
    for index, image in enumerate(images):
        image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        x = index * cell + (cell - image.width) // 2
        y = header + (cell - image.height) // 2
        tile.paste(image, (x, y))
    return tile


def pilot_candidates(
    render_root: Path,
    output_dir: Path,
    excluded_sources: set[str],
    *,
    source_group: str = "pilot_objaverse217",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_uid: dict[str, dict[str, Any]] = {}
    excluded = Counter()
    for manifest_path in completed_manifests(render_root):
        payload = load_json(manifest_path)
        image_root = resolve_path(manifest_path.parent, payload.get("image_root", "."))
        for sample in payload.get("samples", []):
            uid = sample_object_uid(sample)
            source_glb = str(resolve_path(manifest_path.parent, sample["source_glb"]))
            if source_glb in excluded_sources:
                excluded["historical_source_mesh"] += 1
                continue
            frames = [
                resolve_path(image_root, frame["image"])
                for frame in sample.get("frames", [])
                if frame.get("image")
            ]
            row = by_uid.setdefault(
                uid,
                {
                    "source_group": source_group,
                    "object_uid": uid,
                    "source_glb": source_glb,
                    "auto_tier": "RENDER_ACCEPTED",
                    "preview_frames": frames,
                    "sample_uids": [],
                    "source_manifests": [],
                },
            )
            if row["source_glb"] != source_glb:
                raise ValueError(f"pilot object has conflicting source meshes: {uid}")
            row["sample_uids"].append(str(sample["uid"]))
            row["source_manifests"].append(str(manifest_path))

    tile_root = output_dir / "tiles" / source_group
    tile_root.mkdir(parents=True)
    rows = []
    seen_sources: set[str] = set()
    for uid, row in sorted(by_uid.items()):
        if row["source_glb"] in seen_sources:
            excluded["duplicate_pilot_source_mesh"] += 1
            continue
        seen_sources.add(row["source_glb"])
        tile_path = tile_root / f"{uid}.jpg"
        tile = make_tile(row.pop("preview_frames"), f"PILOT {uid}")
        tile.save(tile_path, quality=92, subsampling=0)
        row["sample_uids"] = sorted(set(row["sample_uids"]))
        row["source_manifests"] = sorted(set(row["source_manifests"]))
        row["review_tile"] = str(tile_path.resolve())
        rows.append(row)
    return rows, dict(sorted(excluded.items()))


def labeled_tile(row: dict[str, Any], width: int = 640) -> Image.Image:
    with Image.open(row["review_tile"]) as image:
        tile = image.convert("RGB")
    if tile.width != width:
        height = round(tile.height * width / tile.width)
        tile = tile.resize((width, height), Image.Resampling.LANCZOS)
    footer = 26
    output = Image.new("RGB", (width, tile.height + footer), (18, 18, 18))
    output.paste(tile, (0, 0))
    draw = ImageDraw.Draw(output)
    draw.text(
        (5, tile.height + 5),
        f"{row['review_id']}  {row['source_group']}  auto={row['auto_tier']}",
        fill="white",
        font=ImageFont.load_default(),
    )
    return output


def write_batches(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    columns: int,
) -> list[dict[str, Any]]:
    reports = []
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(str(row["source_group"]), []).append(row)
    for group, group_rows in sorted(by_group.items()):
        batch_root = output_dir / "batches" / group
        batch_root.mkdir(parents=True)
        for start in range(0, len(group_rows), batch_size):
            selected = group_rows[start : start + batch_size]
            tiles = [labeled_tile(row) for row in selected]
            cell_width = max(tile.width for tile in tiles)
            cell_height = max(tile.height for tile in tiles)
            rows_n = math.ceil(len(tiles) / columns)
            sheet = Image.new(
                "RGB", (cell_width * columns, cell_height * rows_n), (8, 8, 8)
            )
            for index, tile in enumerate(tiles):
                sheet.paste(
                    tile,
                    ((index % columns) * cell_width, (index // columns) * cell_height),
                )
            batch_index = start // batch_size
            path = batch_root / f"batch_{batch_index:03d}.jpg"
            sheet.save(path, quality=94, subsampling=0)
            reports.append(
                {
                    "source_group": group,
                    "batch_index": batch_index,
                    "path": str(path.resolve()),
                    "review_ids": [row["review_id"] for row in selected],
                }
            )
    return reports


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "review_id",
        "source_group",
        "object_uid",
        "auto_tier",
        "semantic_subject_label",
        "rejection_reason",
        "reviewer",
        "review_tile",
        "source_glb",
    ]
    known = {
        "9197c6a6beac431ba7961dba4dfab4ae": (
            "reject_scene_or_fragment",
            "aerial_scene_or_environment_fragment",
        ),
        "158abf7655284c94801acdae8f3d383e": ("keep_single_subject", ""),
        "b3f0fba05f264d7f893ab81a088b9328": ("keep_single_subject", ""),
        "f34142be694f4bec81c5da23380879d5": ("keep_single_subject", ""),
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            decision, reason = known.get(str(row["object_uid"]), ("", ""))
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fields},
                    "semantic_subject_label": decision,
                    "rejection_reason": reason,
                    "reviewer": "user_confirmed_20260730" if decision else "",
                }
            )


def merge_review_csv(
    base_csv: Path,
    decisions_csv: Path,
    output_csv: Path,
) -> dict[str, int]:
    if output_csv.exists():
        raise FileExistsError(f"immutable merged review CSV exists: {output_csv}")
    with base_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        base_rows = list(reader)
    required_fields = {
        "review_id",
        "semantic_subject_label",
        "rejection_reason",
        "reviewer",
    }
    if not required_fields.issubset(fieldnames):
        raise ValueError(f"base review CSV is missing fields: {base_csv}")

    decisions: dict[str, dict[str, str]] = {}
    with decisions_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            review_id = str(row.get("review_id", "")).strip()
            if not review_id:
                raise ValueError("decision row has no review_id")
            if review_id in decisions:
                raise ValueError(f"duplicate decision: {review_id}")
            try:
                label = normalize_decision(row.get("semantic_subject_label", ""))
            except ValueError as error:
                raise ValueError(f"invalid decision for {review_id}: {error}") from error
            if not str(row.get("reviewer", "")).strip():
                raise ValueError(f"reviewer is required for {review_id}")
            decisions[review_id] = {**row, "semantic_subject_label": label}

    base_ids = [str(row["review_id"]) for row in base_rows]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("base review CSV has duplicate review_id values")
    missing = sorted(set(base_ids) - set(decisions))
    extra = sorted(set(decisions) - set(base_ids))
    if missing or extra:
        raise ValueError(
            f"decision coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    counts = Counter()
    merged_rows = []
    for row in base_rows:
        decision = decisions[str(row["review_id"])]
        merged = dict(row)
        for field in ("semantic_subject_label", "rejection_reason", "reviewer"):
            merged[field] = str(decision.get(field, "")).strip()
        counts[merged["semantic_subject_label"]] += 1
        merged_rows.append(merged)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    return dict(sorted(counts.items()))


def command_prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable review output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    legacy = legacy_candidates(
        Path(args.legacy_objects).resolve(), Path(args.legacy_review_images).resolve()
    )
    excluded = source_identities(Path(path).resolve() for path in args.exclude_manifest)
    pilot, exclusion_counts = pilot_candidates(
        Path(args.pilot_render_root).resolve(), output_dir, excluded
    )
    rows = []
    for prefix, candidates in (("L", legacy), ("P", pilot)):
        for index, row in enumerate(candidates, 1):
            rows.append({**row, "review_id": f"{prefix}{index:04d}"})
    batch_reports = write_batches(
        output_dir, rows, batch_size=int(args.batch_size), columns=int(args.columns)
    )
    write_review_csv(output_dir / "review_decisions.csv", rows)
    payload = {
        "format": FORMAT,
        "status": "pending_review",
        "candidates": rows,
        "summary": {
            "candidate_count": len(rows),
            "legacy897_count": len(legacy),
            "pilot_objaverse_count": len(pilot),
            "pilot_exclusion_counts": exclusion_counts,
            "batch_count": len(batch_reports),
        },
        "batches": batch_reports,
        "review_csv": str((output_dir / "review_decisions.csv").resolve()),
    }
    (output_dir / "candidates.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


def command_merge(args: argparse.Namespace) -> None:
    counts = merge_review_csv(
        Path(args.base_csv).resolve(),
        Path(args.decisions_csv).resolve(),
        Path(args.output_csv).resolve(),
    )
    print(json.dumps({"decision_counts": counts}, indent=2, ensure_ascii=False))


def command_prepare_render(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable review output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    excluded = source_identities(Path(path).resolve() for path in args.exclude_manifest)
    candidates, exclusion_counts = pilot_candidates(
        Path(args.render_root).resolve(),
        output_dir,
        excluded,
        source_group=str(args.source_group),
    )
    rows = [
        {**row, "review_id": f"{args.review_prefix}{index:04d}"}
        for index, row in enumerate(candidates, 1)
    ]
    batch_reports = write_batches(
        output_dir, rows, batch_size=int(args.batch_size), columns=int(args.columns)
    )
    write_review_csv(output_dir / "review_decisions.csv", rows)
    payload = {
        "format": FORMAT,
        "status": "pending_review",
        "candidates": rows,
        "summary": {
            "candidate_count": len(rows),
            "source_group": str(args.source_group),
            "exclusion_counts": exclusion_counts,
            "batch_count": len(batch_reports),
        },
        "batches": batch_reports,
        "review_csv": str((output_dir / "review_decisions.csv").resolve()),
    }
    (output_dir / "candidates.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


def command_finalize(args: argparse.Namespace) -> None:
    candidates_path = Path(args.candidates).resolve()
    review_csv = Path(args.review_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable finalized review exists: {output_dir}")
    candidate_payload = load_json(candidates_path)
    candidates = {
        str(row["review_id"]): row for row in candidate_payload["candidates"]
    }
    decisions = {}
    with review_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            review_id = str(row["review_id"])
            if review_id not in candidates:
                raise ValueError(f"unknown review_id in decisions: {review_id}")
            try:
                decision = normalize_decision(row["semantic_subject_label"])
            except ValueError as error:
                raise ValueError(f"invalid decision for {review_id}: {error}") from error
            reviewer = str(args.default_reviewer).strip() or str(row["reviewer"]).strip()
            if not reviewer:
                raise ValueError(f"reviewer is required for {review_id}")
            if review_id in decisions:
                raise ValueError(f"duplicate decision: {review_id}")
            decisions[review_id] = {
                **row,
                "semantic_subject_label": decision,
                "reviewer": reviewer,
            }
    if set(decisions) != set(candidates):
        raise ValueError("review CSV does not cover every candidate exactly once")

    reviewed = []
    for review_id, candidate in candidates.items():
        decision = decisions[review_id]
        source = dict(candidate.get("source_object", {}))
        source.update(
            {
                "review_id": review_id,
                "object_uid": candidate["object_uid"],
                "source_glb": candidate["source_glb"],
                "source_group": candidate["source_group"],
                "human_reviewed": True,
                "semantic_subject_label": decision["semantic_subject_label"],
                "semantic_rejection_reason": decision["rejection_reason"],
                "semantic_reviewer": decision["reviewer"],
            }
        )
        reviewed.append(source)
    output_dir.mkdir(parents=True)
    accepted = [
        row for row in reviewed if row["semantic_subject_label"] == "keep_single_subject"
    ]
    rejected = [
        row
        for row in reviewed
        if row["semantic_subject_label"] == "reject_scene_or_fragment"
    ]
    uncertain = [
        row for row in reviewed if row["semantic_subject_label"] == "uncertain_review"
    ]
    for name, payload in (
        ("reviewed_objects.json", reviewed),
        ("accepted_objects.json", accepted),
        ("rejected_objects.json", rejected),
        ("uncertain_objects.json", uncertain),
    ):
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    summary = {
        "reviewed_count": len(reviewed),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "uncertain_count": len(uncertain),
        "accepted_by_source": dict(
            sorted(Counter(row["source_group"] for row in accepted).items())
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "status": "complete",
                "source_candidates": str(candidates_path),
                "source_candidates_sha256": file_sha256(candidates_path),
                "review_csv": str(review_csv),
                "review_csv_sha256": file_sha256(review_csv),
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--legacy_objects", required=True)
    prepare.add_argument("--legacy_review_images", required=True)
    prepare.add_argument("--pilot_render_root", required=True)
    prepare.add_argument("--exclude_manifest", action="append", default=[])
    prepare.add_argument("--output_dir", required=True)
    prepare.add_argument("--batch_size", type=int, default=20)
    prepare.add_argument("--columns", type=int, default=2)
    prepare.set_defaults(handler=command_prepare)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--base_csv", required=True)
    merge.add_argument("--decisions_csv", required=True)
    merge.add_argument("--output_csv", required=True)
    merge.set_defaults(handler=command_merge)

    prepare_render = subparsers.add_parser("prepare-render")
    prepare_render.add_argument("--render_root", required=True)
    prepare_render.add_argument("--exclude_manifest", action="append", default=[])
    prepare_render.add_argument("--source_group", required=True)
    prepare_render.add_argument("--review_prefix", default="G")
    prepare_render.add_argument("--output_dir", required=True)
    prepare_render.add_argument("--batch_size", type=int, default=20)
    prepare_render.add_argument("--columns", type=int, default=2)
    prepare_render.set_defaults(handler=command_prepare_render)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--candidates", required=True)
    finalize.add_argument("--review_csv", required=True)
    finalize.add_argument("--output_dir", required=True)
    finalize.add_argument("--default_reviewer", default="")
    finalize.set_defaults(handler=command_finalize)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
