#!/usr/bin/env python3
"""Freeze eligible Omni real-video objects and leakage-safe raw splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_INVENTORY_FORMAT = "pose_point_depth_mv.omni_real_video_inventory.v1"
ELIGIBILITY_FORMAT = "pose_point_depth_mv.omni_real_video_eligibility.v2"
SPLIT_FORMAT = "pose_point_depth_mv.omni_real_novel_raw_split.v2"
PILOT_FREE_EVAL_SPLIT_FORMAT = "pose_point_depth_mv.omni_real_novel_raw_split.v3"
SPLIT_ROWS_FORMAT = "pose_point_depth_mv.omni_real_novel_raw_split_rows.v2"
BENCHMARK_FORMAT = "pose_point_depth_mv.omni_real_dev_benchmark.v1"
BENCHMARK_ROWS_FORMAT = "pose_point_depth_mv.omni_real_dev_benchmark_rows.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_lines(values: list[str]) -> str:
    return stable_hash("".join(f"{value}\n" for value in sorted(values)))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolved_scan(row: dict[str, Any]) -> str:
    return str(Path(str(row["scan_obj"])).expanduser().resolve())


def validate_unique_objects(rows: list[dict[str, Any]], label: str) -> None:
    object_ids = [str(row["object_id"]) for row in rows]
    scans = [resolved_scan(row) for row in rows]
    require(
        len(object_ids) == len(set(object_ids)),
        f"{label} contains duplicate physical object IDs",
    )
    require(
        len(scans) == len(set(scans)),
        f"{label} contains duplicate Scan.obj paths",
    )


def rejection_reasons(row: dict[str, Any]) -> list[str]:
    checks = row.get("checks")
    require(isinstance(checks, dict) and bool(checks), "object checks are missing")
    return sorted(str(name) for name, passed in checks.items() if passed is not True)


def build_eligibility_inventory(
    source_inventory_path: Path,
    output_dir: Path,
    required_eligible_count: int = 628,
) -> dict[str, Any]:
    source_inventory_path = source_inventory_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source = load_json(source_inventory_path)
    require(
        source.get("format") == SOURCE_INVENTORY_FORMAT,
        f"unexpected source inventory format: {source.get('format')!r}",
    )
    categories = list(source.get("categories", []))
    objects = list(source.get("objects", []))
    require(categories and objects, "source inventory is empty")
    require(
        len(categories) == int(source.get("category_count", -1)),
        "source category count is inconsistent",
    )
    require(
        len(objects) == int(source.get("video_object_count", -1)),
        "source object count is inconsistent",
    )
    flattened = [obj for category in categories for obj in category.get("objects", [])]
    require(
        [str(row["object_id"]) for row in flattened]
        == [str(row["object_id"]) for row in objects],
        "top-level and per-category object inventories differ",
    )
    validate_unique_objects(objects, "source inventory")

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_row in objects:
        row = dict(source_row)
        reasons = rejection_reasons(row)
        computed_passed = not reasons
        require(
            row.get("passed") is computed_passed,
            f"object passed/check mismatch: {row.get('object_id')}",
        )
        if computed_passed:
            eligible.append(row)
        else:
            row["rejection_reasons"] = reasons
            rejected.append(row)

    require(
        len(eligible) == int(source.get("passed_object_count", -1)),
        "source passed-object count is inconsistent",
    )
    require(required_eligible_count > 0, "required eligible count must be positive")
    eligible_by_category = Counter(str(row["category"]) for row in eligible)
    rejected_by_category = Counter(str(row["category"]) for row in rejected)
    checks = {
        "source_inventory_structure_complete": True,
        "object_counts_consistent": len(eligible) + len(rejected) == len(objects),
        "unique_physical_object_ids": True,
        "unique_scan_paths": True,
        "eligible_rows_pass_all_checks": all(row.get("passed") is True for row in eligible),
        "rejected_rows_fail_at_least_one_check": all(
            row.get("passed") is False and bool(row.get("rejection_reasons"))
            for row in rejected
        ),
        "at_least_required_eligible_objects": len(eligible) >= required_eligible_count,
    }
    payload = {
        "format": ELIGIBILITY_FORMAT,
        "created_at_utc": utc_now(),
        "source_inventory": str(source_inventory_path),
        "source_inventory_sha256": sha256_file(source_inventory_path),
        "source_inventory_passed": source.get("passed") is True,
        "category_count": len(categories),
        "video_object_count": len(objects),
        "eligible_object_count": len(eligible),
        "rejected_object_count": len(rejected),
        "required_eligible_object_count": required_eligible_count,
        "eligible_object_id_hash": hash_lines(
            [str(row["object_id"]) for row in eligible]
        ),
        "rejected_object_id_hash": hash_lines(
            [str(row["object_id"]) for row in rejected]
        ),
        "eligible_objects_by_category": dict(sorted(eligible_by_category.items())),
        "rejected_objects_by_category": dict(sorted(rejected_by_category.items())),
        "eligible_objects": eligible,
        "rejected_objects": rejected,
        "checks": checks,
        "training_ready": False,
        "scope_guard": (
            "Eligibility projection of the frozen D4 v1 archive inventory. Rejected "
            "objects remain audit evidence and are forbidden from every split. This "
            "is not a distortion-aware or aligned training cache."
        ),
        "passed": all(value is True for value in checks.values()),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "inventory.json", payload)
    (output_dir / "eligible_objects.txt").write_text(
        "".join(
            f"{row['category']}:{row['object_id']}\n"
            for row in sorted(eligible, key=lambda item: str(item["object_id"]))
        ),
        encoding="utf-8",
    )
    write_json(output_dir / "rejected_objects.json", {"objects": rejected})
    return payload


def verify_eligibility_inventory(
    report_path: Path, source_inventory_path: Path
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    source_inventory_path = source_inventory_path.expanduser().resolve()
    report = load_json(report_path)
    require(report.get("format") == ELIGIBILITY_FORMAT, "unexpected eligibility format")
    require(report.get("passed") is True, "eligibility report did not pass")
    require(report.get("training_ready") is False, "raw inventory marked training-ready")
    require(
        report.get("source_inventory") == str(source_inventory_path),
        "eligibility source path changed",
    )
    require(
        report.get("source_inventory_sha256") == sha256_file(source_inventory_path),
        "eligibility source hash changed",
    )
    eligible = list(report.get("eligible_objects", []))
    rejected = list(report.get("rejected_objects", []))
    validate_unique_objects(eligible + rejected, "eligibility inventory")
    require(
        len(eligible) == int(report.get("eligible_object_count", -1)),
        "eligibility count changed",
    )
    require(
        len(rejected) == int(report.get("rejected_object_count", -1)),
        "rejection count changed",
    )
    require(all(row.get("passed") is True for row in eligible), "invalid eligible row")
    require(all(row.get("passed") is False for row in rejected), "invalid rejected row")
    return report


def take_balanced(
    groups: dict[str, list[dict[str, Any]]], count: int, phase: str, seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < count:
        available = [category for category, rows in groups.items() if rows]
        if not available:
            raise RuntimeError(f"ran out of objects while selecting {phase}")
        available.sort(
            key=lambda category: stable_hash(
                f"{seed}:{phase}:{round_index}:{category}"
            )
        )
        progressed = False
        for category in available:
            if len(selected) >= count:
                break
            if groups[category]:
                selected.append(groups[category].pop(0))
                progressed = True
        if not progressed:
            raise RuntimeError(f"category-balanced selector stalled for {phase}")
        round_index += 1
    return selected


def build_raw_split(
    eligibility_path: Path,
    reviewed_path: Path,
    pilot_path: Path,
    output_dir: Path,
    train_count: int = 500,
    dev_count: int = 64,
    holdout_count: int = 64,
    seed: int = 20260804,
    expected_reviewed_count: int = 135,
    expected_pilot_count: int = 29,
    exclude_pilot_from_eval: bool = False,
) -> dict[str, Any]:
    eligibility_path = eligibility_path.expanduser().resolve()
    reviewed_path = reviewed_path.expanduser().resolve()
    pilot_path = pilot_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    eligibility = load_json(eligibility_path)
    reviewed = load_json(reviewed_path)
    pilot = load_json(pilot_path)
    require(
        eligibility.get("format") == ELIGIBILITY_FORMAT
        and eligibility.get("passed") is True,
        "D4 v2 eligibility inventory did not pass",
    )
    require(eligibility.get("training_ready") is False, "D4 v2 is not a raw inventory")
    require(
        reviewed.get("passed") is True
        and int(reviewed.get("excluded_object_count", -1)) == expected_reviewed_count,
        "reviewed1k exclusion contract changed",
    )
    require(
        pilot.get("passed") is True
        and int(pilot.get("pilot_object_count", -1)) == expected_pilot_count,
        "pilot accounting contract changed",
    )
    require(pilot.get("exclusion_applied") is False, "pilot exclusion was applied")
    require(pilot.get("formal_split_allowed") is True, "pilot split is not allowed")

    eligible = list(eligibility.get("eligible_objects", []))
    rejected = list(eligibility.get("rejected_objects", []))
    require(
        len(eligible) == int(eligibility.get("eligible_object_count", -1)),
        "D4 v2 eligible count changed",
    )
    require(
        len(rejected) == int(eligibility.get("rejected_object_count", -1)),
        "D4 v2 rejected count changed",
    )
    validate_unique_objects(eligible + rejected, "D4 v2 inventory")
    require(all(row.get("passed") is True for row in eligible), "D4 v2 eligible row failed")
    require(all(row.get("passed") is False for row in rejected), "D4 v2 rejected row passed")

    reviewed_ids = {str(row["real_object_id"]) for row in reviewed["objects"]}
    reviewed_scans = {resolved_scan(row) for row in reviewed["objects"]}
    pilot_ids = {str(row["object_id"]) for row in pilot["objects"]}
    pilot_scans = {resolved_scan(row) for row in pilot["objects"]}
    require(
        len(reviewed_ids) == expected_reviewed_count
        and len(reviewed_scans) == expected_reviewed_count,
        "reviewed1k exclusion contains duplicate identities",
    )
    require(
        len(pilot_ids) == expected_pilot_count and len(pilot_scans) == expected_pilot_count,
        "pilot accounting contains duplicate identities",
    )

    source = eligible + rejected
    source_pilot_ids = {str(row["object_id"]) for row in source}.intersection(pilot_ids)
    source_pilot_scans = {resolved_scan(row) for row in source}.intersection(pilot_scans)
    require(
        len(source_pilot_ids) == expected_pilot_count
        and len(source_pilot_scans) == expected_pilot_count,
        "D4 v2 does not contain every protocol pilot object",
    )

    reviewed_overlap: list[dict[str, Any]] = []
    pilot_overlap: list[dict[str, Any]] = []
    novel: list[dict[str, Any]] = []
    for row in eligible:
        object_id = str(row["object_id"])
        scan_obj = resolved_scan(row)
        reviewed_id_overlap = object_id in reviewed_ids
        reviewed_scan_overlap = scan_obj in reviewed_scans
        pilot_id_overlap = object_id in pilot_ids
        pilot_scan_overlap = scan_obj in pilot_scans
        require(
            reviewed_id_overlap == reviewed_scan_overlap,
            f"reviewed1k ID/Scan disagreement for {object_id}",
        )
        require(
            pilot_id_overlap == pilot_scan_overlap,
            f"pilot ID/Scan disagreement for {object_id}",
        )
        if reviewed_id_overlap:
            reviewed_overlap.append(row)
        if pilot_id_overlap:
            pilot_overlap.append(row)
        if not reviewed_id_overlap:
            novel.append(row)

    required = train_count + dev_count + holdout_count
    eval_required = dev_count + holdout_count
    eval_eligible = [
        row for row in novel if str(row["object_id"]) not in pilot_ids
    ]
    require(required > 0 and min(train_count, dev_count, holdout_count) >= 0, "invalid split sizes")
    base = {
        "format": (
            PILOT_FREE_EVAL_SPLIT_FORMAT
            if exclude_pilot_from_eval
            else SPLIT_FORMAT
        ),
        "created_at_utc": utc_now(),
        "eligibility_inventory": str(eligibility_path),
        "eligibility_inventory_sha256": sha256_file(eligibility_path),
        "reviewed1k_exclusion": str(reviewed_path),
        "reviewed1k_exclusion_sha256": sha256_file(reviewed_path),
        "protocol_pilot29_accounting": str(pilot_path),
        "protocol_pilot29_accounting_sha256": sha256_file(pilot_path),
        "seed": seed,
        "source_video_object_count": int(eligibility["video_object_count"]),
        "eligible_video_mesh_count": len(eligible),
        "rejected_object_count": len(rejected),
        "rejected_object_ids": sorted(str(row["object_id"]) for row in rejected),
        "declared_reviewed1k_exclusion_count": len(reviewed_ids),
        "excluded_old_omni_count": len(reviewed_overlap),
        "pilot29_in_source_inventory_count": len(source_pilot_ids),
        "pilot29_in_eligible_inventory_count": len(pilot_overlap),
        "pilot29_eligible_count": len(
            {str(row["object_id"]) for row in pilot_overlap}.difference(reviewed_ids)
        ),
        "novel_object_count": len(novel),
        "required_novel_object_count": required,
        "shortfall": max(0, required - len(novel)),
        "exclude_pilot_from_eval": bool(exclude_pilot_from_eval),
        "eval_eligible_nonpilot_object_count": len(eval_eligible),
        "required_eval_nonpilot_object_count": eval_required,
        "eval_shortfall": (
            max(0, eval_required - len(eval_eligible))
            if exclude_pilot_from_eval
            else 0
        ),
        "pilot29_split_counts": None,
        "reviewed1k_overlap_object_ids": sorted(
            str(row["object_id"]) for row in reviewed_overlap
        ),
        "training_ready": False,
        "scope_guard": (
            "Object-disjoint raw video/mesh split only. Distortion-aware canonical "
            "cache, COLMAP-to-Scan alignment, feature materialization and model "
            "training remain separate later stages."
        ),
    }
    if len(novel) < required or (
        exclude_pilot_from_eval and len(eval_eligible) < eval_required
    ):
        payload = {**base, "passed": False, "split_counts": None, "checks": {
            "at_least_required_novel_objects": len(novel) >= required,
            "at_least_required_nonpilot_eval_objects": (
                not exclude_pilot_from_eval or len(eval_eligible) >= eval_required
            ),
        }}
        output_dir.mkdir(parents=True, exist_ok=False)
        write_json(output_dir / "split_report.json", payload)
        return payload

    def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            result[str(row["category"])].append(row)
        return result

    groups = grouped(eval_eligible if exclude_pilot_from_eval else novel)
    for category, rows in groups.items():
        rows.sort(
            key=lambda row: stable_hash(
                f"{seed}:object:{category}:{row['object_id']}"
            )
        )

    holdout = take_balanced(groups, holdout_count, "holdout", seed)
    dev = take_balanced(groups, dev_count, "dev", seed)
    eval_ids = {str(row["object_id"]) for row in holdout + dev}
    remaining = [
        row for row in novel if str(row["object_id"]) not in eval_ids
    ]
    train_groups = grouped(remaining)
    for category, rows in train_groups.items():
        rows.sort(
            key=lambda row: stable_hash(
                f"{seed}:object:{category}:{row['object_id']}"
            )
        )
    train = take_balanced(train_groups, train_count, "train", seed)
    reserve = [
        row for category in sorted(train_groups) for row in train_groups[category]
    ]
    splits = {"train": train, "dev": dev, "holdout": holdout, "reserve": reserve}
    sets = {
        name: {str(row["object_id"]) for row in rows}
        for name, rows in splits.items()
    }
    for index, left in enumerate(sets):
        for right in list(sets)[index + 1 :]:
            require(not sets[left].intersection(sets[right]), f"split leakage: {left}/{right}")

    reviewed_overlap_by_split = {
        name: len(values.intersection(reviewed_ids)) for name, values in sets.items()
    }
    rejected_ids = {str(row["object_id"]) for row in rejected}
    rejected_overlap_by_split = {
        name: len(values.intersection(rejected_ids)) for name, values in sets.items()
    }
    pilot_overlap_by_split = {
        name: len(values.intersection(pilot_ids)) for name, values in sets.items()
    }
    novel_ids = {str(row["object_id"]) for row in novel}
    require(set().union(*sets.values()) == novel_ids, "split coverage differs from novel set")

    output_dir.mkdir(parents=True, exist_ok=False)
    bindings: dict[str, dict[str, Any]] = {}
    for name, rows in splits.items():
        path = output_dir / f"{name}.json"
        body = {
            "format": SPLIT_ROWS_FORMAT,
            "split": name,
            "object_count": len(rows),
            "objects_by_category": dict(
                sorted(Counter(str(row["category"]) for row in rows).items())
            ),
            "objects": rows,
            "eligibility_inventory": str(eligibility_path),
            "eligibility_inventory_sha256": sha256_file(eligibility_path),
            "training_ready": False,
        }
        write_json(path, body)
        bindings[name] = {
            "path": str(path),
            "object_count": len(rows),
            "object_id_hash": hash_lines(list(sets[name])),
            "file_sha256": sha256_file(path),
        }

    checks = {
        "at_least_required_novel_objects": len(novel) >= required,
        "at_least_required_nonpilot_eval_objects": (
            not exclude_pilot_from_eval or len(eval_eligible) >= eval_required
        ),
        "train_count": len(train) == train_count,
        "dev_count": len(dev) == dev_count,
        "holdout_count": len(holdout) == holdout_count,
        "reserve_count": len(reserve) == len(novel) - required,
        "physical_object_disjoint": True,
        "complete_novel_object_coverage": set().union(*sets.values()) == novel_ids,
        "reviewed1k_overlap_all_splits_zero": not any(
            reviewed_overlap_by_split.values()
        ),
        "rejected_overlap_all_splits_zero": not any(rejected_overlap_by_split.values()),
        "pilot29_nonreviewed_preserved": sum(pilot_overlap_by_split.values())
        == len(pilot_ids.difference(reviewed_ids)),
        "pilot_absent_from_dev": (
            not exclude_pilot_from_eval or pilot_overlap_by_split["dev"] == 0
        ),
        "pilot_absent_from_holdout": (
            not exclude_pilot_from_eval or pilot_overlap_by_split["holdout"] == 0
        ),
    }
    payload = {
        **base,
        "passed": all(value is True for value in checks.values()),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "pilot29_split_counts": pilot_overlap_by_split,
        "reviewed1k_overlap_by_split": reviewed_overlap_by_split,
        "rejected_overlap_by_split": rejected_overlap_by_split,
        "splits": bindings,
        "checks": checks,
    }
    write_json(output_dir / "split_report.json", payload)
    return payload


def verify_raw_split(
    report_path: Path,
    eligibility_path: Path,
    reviewed_path: Path,
    pilot_path: Path,
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    report = load_json(report_path)
    require(
        report.get("format") in {SPLIT_FORMAT, PILOT_FREE_EVAL_SPLIT_FORMAT},
        "unexpected split format",
    )
    require(report.get("passed") is True, "raw split did not pass")
    require(report.get("training_ready") is False, "raw split marked training-ready")
    expected = {
        "eligibility_inventory_sha256": sha256_file(eligibility_path.expanduser().resolve()),
        "reviewed1k_exclusion_sha256": sha256_file(reviewed_path.expanduser().resolve()),
        "protocol_pilot29_accounting_sha256": sha256_file(pilot_path.expanduser().resolve()),
    }
    for key, value in expected.items():
        require(report.get(key) == value, f"split input binding changed: {key}")
    bindings = report.get("splits", {})
    require(set(bindings) == {"train", "dev", "holdout", "reserve"}, "split files missing")
    all_ids: set[str] = set()
    for name, binding in bindings.items():
        path = Path(binding["path"]).resolve()
        require(path.is_file(), f"missing split file: {path}")
        require(binding.get("file_sha256") == sha256_file(path), f"split file changed: {name}")
        body = load_json(path)
        require(body.get("format") == SPLIT_ROWS_FORMAT, f"unexpected {name} format")
        require(body.get("training_ready") is False, f"{name} marked training-ready")
        ids = {str(row["object_id"]) for row in body.get("objects", [])}
        require(len(ids) == int(binding["object_count"]), f"duplicate objects in {name}")
        require(not all_ids.intersection(ids), f"split leakage while verifying {name}")
        all_ids.update(ids)
    if report.get("exclude_pilot_from_eval") is True:
        pilot = load_json(pilot_path.expanduser().resolve())
        pilot_ids = {str(row["object_id"]) for row in pilot.get("objects", [])}
        for name in ("dev", "holdout"):
            body = load_json(Path(bindings[name]["path"]).resolve())
            ids = {str(row["object_id"]) for row in body.get("objects", [])}
            require(not ids.intersection(pilot_ids), f"pilot leakage into {name}")
    return report


def _benchmark_rows_body(
    *,
    role: str,
    rows: list[dict[str, Any]],
    source_dev_path: Path,
    source_dev_sha256: str,
) -> dict[str, Any]:
    return {
        "format": BENCHMARK_ROWS_FORMAT,
        "role": role,
        "object_count": len(rows),
        "object_id_hash": hash_lines([str(row["object_id"]) for row in rows]),
        "objects_by_category": dict(
            sorted(Counter(str(row["category"]) for row in rows).items())
        ),
        "objects": rows,
        "source_dev_manifest": str(source_dev_path),
        "source_dev_manifest_sha256": source_dev_sha256,
        "training_ready": False,
    }


def _subset_inventory(
    *,
    rows: list[dict[str, Any]],
    source_dev_path: Path,
    source_dev_sha256: str,
) -> dict[str, Any]:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["category"])].append(row)
    categories = []
    for category in sorted(grouped_rows):
        objects = sorted(
            grouped_rows[category], key=lambda row: str(row["object_id"])
        )
        archives = {str(row["archive"]) for row in objects}
        require(len(archives) == 1, f"multiple archives for category={category}")
        archive = next(iter(archives))
        categories.append(
            {
                "category": category,
                "archive": archive,
                "archive_bytes": Path(archive).stat().st_size,
                "video_object_count": len(objects),
                "passed_object_count": len(objects),
                "objects": objects,
                "passed": True,
            }
        )
    return {
        "format": SOURCE_INVENTORY_FORMAT,
        "created_at_utc": utc_now(),
        "category_count": len(categories),
        "video_object_count": len(rows),
        "passed_object_count": len(rows),
        "categories": categories,
        "objects": rows,
        "source_dev_manifest": str(source_dev_path),
        "source_dev_manifest_sha256": source_dev_sha256,
        "selection_scope": "frozen development benchmark only",
        "training_ready": False,
        "scope_guard": (
            "Subset inventory for extracting the frozen real-development benchmark. "
            "It is not a training split and must never be extended with holdout rows."
        ),
        "passed": True,
    }


def build_split_extraction_inventory(
    source_split_path: Path, output_path: Path
) -> dict[str, Any]:
    source_split_path = source_split_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    source = load_json(source_split_path)
    require(source.get("format") == SPLIT_ROWS_FORMAT, "unexpected split-row format")
    require(source.get("training_ready") is False, "split rows marked training-ready")
    split = str(source.get("split", ""))
    require(split in {"train", "dev", "holdout", "reserve"}, "invalid split role")
    rows = list(source.get("objects", []))
    require(len(rows) == int(source.get("object_count", -1)), "split count changed")
    validate_unique_objects(rows, f"{split} extraction inventory")
    payload = _subset_inventory(
        rows=rows,
        source_dev_path=source_split_path,
        source_dev_sha256=sha256_file(source_split_path),
    )
    payload.pop("source_dev_manifest", None)
    payload.pop("source_dev_manifest_sha256", None)
    payload.update(
        {
            "source_split": str(source_split_path),
            "source_split_sha256": sha256_file(source_split_path),
            "split": split,
            "selection_scope": f"frozen {split} split extraction only",
            "scope_guard": (
                "Extraction inventory preserves the frozen split exactly. It does "
                "not align meshes, build runtime-O, encode targets, or admit training."
            ),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)
    return payload


def verify_split_extraction_inventory(
    source_split_path: Path, output_path: Path
) -> dict[str, Any]:
    source_split_path = source_split_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    payload = load_json(output_path)
    require(
        payload.get("format") == SOURCE_INVENTORY_FORMAT,
        "unexpected extraction inventory format",
    )
    require(payload.get("passed") is True, "extraction inventory did not pass")
    require(
        payload.get("source_split") == str(source_split_path),
        "extraction inventory source path changed",
    )
    require(
        payload.get("source_split_sha256") == sha256_file(source_split_path),
        "extraction inventory source hash changed",
    )
    source = load_json(source_split_path)
    require(
        int(payload.get("video_object_count", -1))
        == int(source.get("object_count", -2)),
        "extraction inventory object count changed",
    )
    return payload


def build_dev_benchmark(
    source_dev_path: Path,
    pilot_path: Path,
    output_dir: Path,
    benchmark_count: int = 32,
    seed: int = 20260805,
    expected_pilot_count: int = 29,
) -> dict[str, Any]:
    source_dev_path = source_dev_path.expanduser().resolve()
    pilot_path = pilot_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source = load_json(source_dev_path)
    pilot = load_json(pilot_path)
    require(source.get("format") == SPLIT_ROWS_FORMAT, "unexpected dev split format")
    require(source.get("split") == "dev", "benchmark source must be the dev split")
    require(source.get("training_ready") is False, "dev split marked training-ready")
    rows = list(source.get("objects", []))
    require(len(rows) == int(source.get("object_count", -1)), "dev count changed")
    validate_unique_objects(rows, "benchmark source dev split")
    require(
        pilot.get("passed") is True
        and int(pilot.get("pilot_object_count", -1)) == expected_pilot_count,
        "pilot accounting contract changed",
    )
    pilot_ids = {str(row["object_id"]) for row in pilot.get("objects", [])}
    pilot_scans = {resolved_scan(row) for row in pilot.get("objects", [])}
    source_pilot_ids = {str(row["object_id"]) for row in rows}.intersection(pilot_ids)
    source_pilot_scans = {resolved_scan(row) for row in rows}.intersection(pilot_scans)
    require(
        not source_pilot_ids and not source_pilot_scans,
        "dev source contains protocol-pilot objects",
    )
    require(0 < benchmark_count <= len(rows), "invalid benchmark count")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)
    for category, category_rows in groups.items():
        category_rows.sort(
            key=lambda row: stable_hash(
                f"{seed}:benchmark-object:{category}:{row['object_id']}"
            )
        )
    benchmark = take_balanced(groups, benchmark_count, "benchmark", seed)
    remainder = [row for category in sorted(groups) for row in groups[category]]
    benchmark_ids = {str(row["object_id"]) for row in benchmark}
    remainder_ids = {str(row["object_id"]) for row in remainder}
    source_ids = {str(row["object_id"]) for row in rows}
    require(not benchmark_ids.intersection(remainder_ids), "benchmark/remainder leakage")
    require(benchmark_ids.union(remainder_ids) == source_ids, "benchmark coverage changed")

    output_dir.mkdir(parents=True, exist_ok=False)
    source_hash = sha256_file(source_dev_path)
    benchmark_path = output_dir / f"benchmark{len(benchmark)}.json"
    remainder_path = output_dir / f"remainder{len(remainder)}.json"
    inventory_path = output_dir / f"benchmark{len(benchmark)}_inventory.json"
    write_json(
        benchmark_path,
        _benchmark_rows_body(
            role="reusable_development_benchmark",
            rows=benchmark,
            source_dev_path=source_dev_path,
            source_dev_sha256=source_hash,
        ),
    )
    write_json(
        remainder_path,
        _benchmark_rows_body(
            role="unused_development_remainder",
            rows=remainder,
            source_dev_path=source_dev_path,
            source_dev_sha256=source_hash,
        ),
    )
    write_json(
        inventory_path,
        _subset_inventory(
            rows=benchmark,
            source_dev_path=source_dev_path,
            source_dev_sha256=source_hash,
        ),
    )
    checks = {
        "source_is_dev": source.get("split") == "dev",
        "benchmark_count": len(benchmark) == benchmark_count,
        "complete_dev_coverage": benchmark_ids.union(remainder_ids) == source_ids,
        "benchmark_remainder_disjoint": not benchmark_ids.intersection(remainder_ids),
        "pilot_absent_from_source_dev": not source_pilot_ids and not source_pilot_scans,
        "pilot_absent_from_benchmark": not benchmark_ids.intersection(pilot_ids),
    }
    payload = {
        "format": BENCHMARK_FORMAT,
        "created_at_utc": utc_now(),
        "source_dev_manifest": str(source_dev_path),
        "source_dev_manifest_sha256": source_hash,
        "pilot_accounting": str(pilot_path),
        "pilot_accounting_sha256": sha256_file(pilot_path),
        "seed": int(seed),
        "source_dev_object_count": len(rows),
        "benchmark_object_count": len(benchmark),
        "remainder_object_count": len(remainder),
        "benchmark_object_id_hash": hash_lines(list(benchmark_ids)),
        "remainder_object_id_hash": hash_lines(list(remainder_ids)),
        "benchmark_objects_by_category": dict(
            sorted(Counter(str(row["category"]) for row in benchmark).items())
        ),
        "pilot_overlap_count": len(benchmark_ids.intersection(pilot_ids)),
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": sha256_file(benchmark_path),
        },
        "remainder": {
            "path": str(remainder_path),
            "sha256": sha256_file(remainder_path),
        },
        "extraction_inventory": {
            "path": str(inventory_path),
            "sha256": sha256_file(inventory_path),
        },
        "checks": checks,
        "training_ready": False,
        "scope_guard": (
            "Reusable development benchmark for pre/post-adaptation comparisons. "
            "The separate holdout64 remains untouched and is not an input here."
        ),
        "passed": all(value is True for value in checks.values()),
    }
    write_json(output_dir / "benchmark_report.json", payload)
    return payload


def verify_dev_benchmark(
    report_path: Path, source_dev_path: Path, pilot_path: Path
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    source_dev_path = source_dev_path.expanduser().resolve()
    pilot_path = pilot_path.expanduser().resolve()
    report = load_json(report_path)
    require(report.get("format") == BENCHMARK_FORMAT, "unexpected benchmark format")
    require(report.get("passed") is True, "benchmark report did not pass")
    require(report.get("training_ready") is False, "benchmark marked training-ready")
    require(
        report.get("source_dev_manifest_sha256") == sha256_file(source_dev_path),
        "benchmark dev binding changed",
    )
    require(
        report.get("pilot_accounting_sha256") == sha256_file(pilot_path),
        "benchmark pilot binding changed",
    )
    all_ids: set[str] = set()
    for key in ("benchmark", "remainder"):
        binding = report[key]
        path = Path(binding["path"]).resolve()
        require(binding.get("sha256") == sha256_file(path), f"{key} file changed")
        body = load_json(path)
        require(body.get("format") == BENCHMARK_ROWS_FORMAT, f"invalid {key} rows")
        ids = {str(row["object_id"]) for row in body.get("objects", [])}
        require(len(ids) == int(body.get("object_count", -1)), f"duplicate {key} rows")
        require(not ids.intersection(all_ids), f"benchmark leakage at {key}")
        all_ids.update(ids)
    inventory_binding = report["extraction_inventory"]
    inventory_path = Path(inventory_binding["path"]).resolve()
    require(
        inventory_binding.get("sha256") == sha256_file(inventory_path),
        "benchmark extraction inventory changed",
    )
    inventory = load_json(inventory_path)
    require(
        inventory.get("format") == SOURCE_INVENTORY_FORMAT
        and inventory.get("passed") is True
        and inventory.get("training_ready") is False,
        "benchmark extraction inventory is invalid",
    )
    pilot = load_json(pilot_path)
    pilot_ids = {str(row["object_id"]) for row in pilot.get("objects", [])}
    benchmark_rows = load_json(Path(report["benchmark"]["path"]))
    benchmark_ids = {
        str(row["object_id"]) for row in benchmark_rows.get("objects", [])
    }
    require(not benchmark_ids.intersection(pilot_ids), "pilot leaked into benchmark")
    return report


def summary(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "passed",
        "source_inventory_passed",
        "video_object_count",
        "eligible_object_count",
        "rejected_object_count",
        "source_video_object_count",
        "eligible_video_mesh_count",
        "excluded_old_omni_count",
        "pilot29_in_source_inventory_count",
        "pilot29_in_eligible_inventory_count",
        "pilot29_eligible_count",
        "novel_object_count",
        "shortfall",
        "exclude_pilot_from_eval",
        "eval_eligible_nonpilot_object_count",
        "eval_shortfall",
        "split_counts",
        "pilot29_split_counts",
        "source_dev_object_count",
        "benchmark_object_count",
        "remainder_object_count",
        "benchmark_objects_by_category",
        "pilot_overlap_count",
        "benchmark",
        "remainder",
        "extraction_inventory",
        "checks",
        "training_ready",
        "scope_guard",
    )
    return {key: payload[key] for key in keys if key in payload}


def command_eligibility(args: argparse.Namespace) -> int:
    source = Path(args.source_inventory)
    output_dir = Path(args.output_dir)
    report_path = output_dir / "inventory.json"
    if report_path.is_file():
        payload = verify_eligibility_inventory(report_path, source)
    elif output_dir.exists():
        raise RuntimeError(f"partial output exists; preserve and inspect: {output_dir}")
    else:
        payload = build_eligibility_inventory(
            source, output_dir, args.required_eligible_count
        )
    print(json.dumps(summary(payload), indent=2, ensure_ascii=False))
    return 0 if payload.get("passed") is True else 2


def command_split(args: argparse.Namespace) -> int:
    eligibility = Path(args.eligibility_inventory)
    reviewed = Path(args.reviewed_exclusion)
    pilot = Path(args.pilot_accounting)
    output_dir = Path(args.output_dir)
    report_path = output_dir / "split_report.json"
    if report_path.is_file():
        payload = verify_raw_split(report_path, eligibility, reviewed, pilot)
        expected_counts = {
            "train": int(args.train_count),
            "dev": int(args.dev_count),
            "holdout": int(args.holdout_count),
        }
        actual_counts = dict(payload.get("split_counts", {}))
        require(
            all(actual_counts.get(name) == count for name, count in expected_counts.items()),
            "reused split counts differ from requested counts",
        )
        require(int(payload.get("seed", -1)) == int(args.seed), "reused split seed changed")
        require(
            bool(payload.get("exclude_pilot_from_eval", False))
            == bool(args.exclude_pilot_from_eval),
            "reused split pilot-eval policy changed",
        )
    elif output_dir.exists():
        raise RuntimeError(f"partial output exists; preserve and inspect: {output_dir}")
    else:
        payload = build_raw_split(
            eligibility,
            reviewed,
            pilot,
            output_dir,
            train_count=args.train_count,
            dev_count=args.dev_count,
            holdout_count=args.holdout_count,
            seed=args.seed,
            expected_reviewed_count=args.expected_reviewed_count,
            expected_pilot_count=args.expected_pilot_count,
            exclude_pilot_from_eval=args.exclude_pilot_from_eval,
        )
    print(json.dumps(summary(payload), indent=2, ensure_ascii=False))
    return 0 if payload.get("passed") is True else 2


def command_benchmark(args: argparse.Namespace) -> int:
    source_dev = Path(args.source_dev_manifest)
    pilot = Path(args.pilot_accounting)
    output_dir = Path(args.output_dir)
    report_path = output_dir / "benchmark_report.json"
    if report_path.is_file():
        payload = verify_dev_benchmark(report_path, source_dev, pilot)
        require(
            int(payload.get("benchmark_object_count", -1))
            == int(args.benchmark_count),
            "reused benchmark count changed",
        )
        require(
            int(payload.get("seed", -1)) == int(args.seed),
            "reused benchmark seed changed",
        )
    elif output_dir.exists():
        raise RuntimeError(f"partial output exists; preserve and inspect: {output_dir}")
    else:
        payload = build_dev_benchmark(
            source_dev,
            pilot,
            output_dir,
            benchmark_count=args.benchmark_count,
            seed=args.seed,
            expected_pilot_count=args.expected_pilot_count,
        )
    print(json.dumps(summary(payload), indent=2, ensure_ascii=False))
    return 0 if payload.get("passed") is True else 2


def command_inventory(args: argparse.Namespace) -> int:
    source = Path(args.source_split)
    output = Path(args.output)
    if output.is_file():
        payload = verify_split_extraction_inventory(source, output)
    elif output.exists():
        raise RuntimeError(f"extraction inventory output is not a file: {output}")
    else:
        payload = build_split_extraction_inventory(source, output)
    print(json.dumps(summary(payload), indent=2, ensure_ascii=False))
    return 0 if payload.get("passed") is True else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    eligibility = subparsers.add_parser("eligibility")
    eligibility.add_argument("--source_inventory", required=True)
    eligibility.add_argument("--output_dir", required=True)
    eligibility.add_argument("--required_eligible_count", type=int, default=628)
    eligibility.set_defaults(handler=command_eligibility)

    split = subparsers.add_parser("split")
    split.add_argument("--eligibility_inventory", required=True)
    split.add_argument("--reviewed_exclusion", required=True)
    split.add_argument("--pilot_accounting", required=True)
    split.add_argument("--output_dir", required=True)
    split.add_argument("--train_count", type=int, default=500)
    split.add_argument("--dev_count", type=int, default=64)
    split.add_argument("--holdout_count", type=int, default=64)
    split.add_argument("--seed", type=int, default=20260804)
    split.add_argument("--expected_reviewed_count", type=int, default=135)
    split.add_argument("--expected_pilot_count", type=int, default=29)
    split.add_argument("--exclude_pilot_from_eval", action="store_true")
    split.set_defaults(handler=command_split)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--source_dev_manifest", required=True)
    benchmark.add_argument("--pilot_accounting", required=True)
    benchmark.add_argument("--output_dir", required=True)
    benchmark.add_argument("--benchmark_count", type=int, default=32)
    benchmark.add_argument("--seed", type=int, default=20260805)
    benchmark.add_argument("--expected_pilot_count", type=int, default=29)
    benchmark.set_defaults(handler=command_benchmark)

    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--source_split", required=True)
    inventory.add_argument("--output", required=True)
    inventory.set_defaults(handler=command_inventory)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
