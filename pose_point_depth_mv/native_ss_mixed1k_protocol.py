#!/usr/bin/env python3
"""Freeze and audit the source-balanced Native SS mixed-1k protocol."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


SPLIT_FORMAT = "pose_point_depth_mv.native_ss_sourcebalanced_split.v1"
SPLIT_IDENTITY = "native_ss.sourcebalanced_split.seed20260801.v1"
SOURCES = (
    "legacy_objaverse",
    "gap_objaverse",
    "pilot_objaverse",
    "omni",
)
SOURCE_RULES = (
    ("objaverse_sparse_mv_artraj_pbr_5000_v9_select8", "legacy_objaverse"),
    ("gap_objaverse288_cyclescuda_v1", "gap_objaverse"),
    ("pilot_objaverse256_cyclescuda_v2", "pilot_objaverse"),
    ("pilot_omni256_cyclescuda_v4", "omni"),
)
VIEW_QUOTAS = {
    "checkpoint": {
        "legacy_objaverse": {2: 2, 4: 1, 8: 1},
        "gap_objaverse": {2: 1, 4: 2, 8: 1},
        "pilot_objaverse": {2: 1, 4: 1, 8: 2},
        "omni": {2: 2, 4: 1, 8: 1},
    },
    "cfg": {
        "legacy_objaverse": {2: 1, 4: 2, 8: 1},
        "gap_objaverse": {2: 1, 4: 1, 8: 2},
        "pilot_objaverse": {2: 2, 4: 1, 8: 1},
        "omni": {2: 1, 4: 2, 8: 1},
    },
    "final": {
        "legacy_objaverse": {2: 3, 4: 3, 8: 2},
        "gap_objaverse": {2: 2, 4: 3, 8: 3},
        "pilot_objaverse": {2: 3, 4: 2, 8: 3},
        "omni": {2: 3, 4: 3, 8: 2},
    },
}
PHASE_INDICES = {"checkpoint": "0-15", "cfg": "16-31", "final": "32-63"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def classify_source(path: str) -> str:
    for marker, source in SOURCE_RULES:
        if marker in str(path):
            return source
    raise ValueError(f"unknown render source: {path}")


def stable_rank(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def freeze_split(args: argparse.Namespace) -> None:
    lifting_path = Path(args.lifting_manifest)
    pointpose_path = Path(args.pointpose_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    lifting = read_json(lifting_path)
    pointpose = read_json(pointpose_path)
    if lifting.get("sample_count") != 1534 or lifting.get("object_count") != 932:
        raise ValueError("freeze-split requires the reviewed 1534-sequence/932-object cache")
    pointpose_by_uid = {str(row["uid"]): row for row in pointpose["samples"]}
    if len(pointpose_by_uid) != len(pointpose["samples"]):
        raise ValueError("pointpose manifest contains duplicate sequence UIDs")

    rows_by_object: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    source_by_object: dict[str, str] = {}
    for row in lifting["samples"]:
        uid = str(row["uid"])
        object_uid = str(row["object_uid"])
        source = classify_source(str(pointpose_by_uid[uid]["image_paths"][0]))
        previous = source_by_object.setdefault(object_uid, source)
        if previous != source:
            raise ValueError(f"object spans sources: {object_uid}: {previous} != {source}")
        rows_by_object[object_uid].append(row)

    original_source_counts = collections.Counter(source_by_object.values())
    expected_original = {
        "legacy_objaverse": 515,
        "gap_objaverse": 149,
        "pilot_objaverse": 133,
        "omni": 135,
    }
    if original_source_counts != expected_original:
        raise ValueError(f"reviewed source counts changed: {original_source_counts}")

    # The failed SS64 run selected the first 64 sorted object UIDs.
    old_ss64_objects = set(sorted(rows_by_object)[:64])
    selected_objects: set[str] = set()
    phase_rows: dict[str, list[dict[str, Any]]] = {}
    phase_records: dict[str, list[dict[str, Any]]] = {}

    for phase in ("checkpoint", "cfg", "final"):
        selected_rows = []
        records = []
        for source in SOURCES:
            for view_count, required in VIEW_QUOTAS[phase][source].items():
                candidates = []
                for object_uid, rows in rows_by_object.items():
                    if object_uid in old_ss64_objects or object_uid in selected_objects:
                        continue
                    if source_by_object[object_uid] != source:
                        continue
                    matching = [
                        row for row in rows if int(row["view_count"]) == int(view_count)
                    ]
                    if not matching:
                        continue
                    matching.sort(
                        key=lambda row: (
                            stable_rank(args.seed, f"row:{row['uid']}"),
                            str(row["uid"]),
                        )
                    )
                    candidates.append(
                        (
                            stable_rank(
                                args.seed,
                                f"{phase}:{source}:{view_count}:{object_uid}",
                            ),
                            object_uid,
                            matching[0],
                        )
                    )
                candidates.sort()
                if len(candidates) < required:
                    raise RuntimeError(
                        f"insufficient candidates for {phase}/{source}/view{view_count}: "
                        f"{len(candidates)} < {required}"
                    )
                for _, object_uid, row in candidates[:required]:
                    if object_uid in selected_objects:
                        raise AssertionError(f"duplicate held-out object: {object_uid}")
                    selected_objects.add(object_uid)
                    selected_rows.append(copy.deepcopy(row))
                    records.append(
                        {
                            "uid": str(row["uid"]),
                            "object_uid": object_uid,
                            "source": source,
                            "view_count": int(view_count),
                        }
                    )
        phase_rows[phase] = selected_rows
        phase_records[phase] = records

    if len(selected_objects) != 64 or selected_objects.intersection(old_ss64_objects):
        raise AssertionError("held-out identity contract failed")
    train_rows = [
        copy.deepcopy(row)
        for row in lifting["samples"]
        if str(row["object_uid"]) not in selected_objects
    ]
    train_objects = {str(row["object_uid"]) for row in train_rows}
    val_rows = phase_rows["checkpoint"] + phase_rows["cfg"] + phase_rows["final"]
    if (len(train_rows), len(train_objects), len(val_rows)) != (1417, 868, 64):
        raise AssertionError(
            f"unexpected split sizes={(len(train_rows), len(train_objects), len(val_rows))}"
        )
    if train_objects.intersection(selected_objects):
        raise AssertionError("train/val object overlap")

    expected_phase_counts = {
        "checkpoint": {
            "objects": 16,
            "source": {source: 4 for source in SOURCES},
            "views": {2: 6, 4: 5, 8: 5},
            "indices": PHASE_INDICES["checkpoint"],
        },
        "cfg": {
            "objects": 16,
            "source": {source: 4 for source in SOURCES},
            "views": {2: 5, 4: 6, 8: 5},
            "indices": PHASE_INDICES["cfg"],
        },
        "final": {
            "objects": 32,
            "source": {source: 8 for source in SOURCES},
            "views": {2: 11, 4: 11, 8: 10},
            "indices": PHASE_INDICES["final"],
        },
    }
    for phase, expected in expected_phase_counts.items():
        records = phase_records[phase]
        if len(records) != expected["objects"]:
            raise AssertionError(f"{phase} object count changed")
        if collections.Counter(row["source"] for row in records) != expected["source"]:
            raise AssertionError(f"{phase} source quota changed")
        if collections.Counter(row["view_count"] for row in records) != expected["views"]:
            raise AssertionError(f"{phase} view quota changed")

    train_source_counts = collections.Counter(
        source_by_object[object_uid] for object_uid in train_objects
    )
    expected_train_sources = {
        "legacy_objaverse": 499,
        "gap_objaverse": 133,
        "pilot_objaverse": 117,
        "omni": 119,
    }
    if train_source_counts != expected_train_sources:
        raise AssertionError(f"train source counts changed: {train_source_counts}")

    train_manifest = copy.deepcopy(lifting)
    train_manifest.update(
        {
            "samples": train_rows,
            "sample_count": len(train_rows),
            "object_count": len(train_objects),
            "depth_calibration_enabled_count": sum(
                bool(row.get("depth_calibration_enabled")) for row in train_rows
            ),
            "selection": {
                "mode": "exclude_source_balanced_object_holdout",
                "seed": int(args.seed),
                "heldout_object_count": 64,
            },
            "split_identity": SPLIT_IDENTITY,
        }
    )
    val_manifest = copy.deepcopy(lifting)
    val_manifest.update(
        {
            "samples": val_rows,
            "sample_count": len(val_rows),
            "object_count": 64,
            "depth_calibration_enabled_count": sum(
                bool(row.get("depth_calibration_enabled")) for row in val_rows
            ),
            "selection": {
                "mode": "source_and_view_balanced_object_holdout",
                "seed": int(args.seed),
                "phase_indices": dict(PHASE_INDICES),
            },
            "split_identity": SPLIT_IDENTITY,
        }
    )
    train_out = output_dir / "train868_manifest.json"
    val_out = output_dir / "val64_sourcebalanced_manifest.json"
    write_json(train_out, train_manifest)
    write_json(val_out, val_manifest)

    audit = {
        "format": SPLIT_FORMAT,
        "passed": True,
        "seed": int(args.seed),
        "source_rules": dict(SOURCE_RULES),
        "input": {
            "lifting_manifest": str(lifting_path.resolve()),
            "lifting_manifest_sha256": sha256_file(lifting_path),
            "pointpose_manifest": str(pointpose_path.resolve()),
            "pointpose_manifest_sha256": sha256_file(pointpose_path),
        },
        "output": {
            "train_manifest": str(train_out.resolve()),
            "train_manifest_sha256": sha256_file(train_out),
            "val_manifest": str(val_out.resolve()),
            "val_manifest_sha256": sha256_file(val_out),
        },
        "counts": {
            "original_objects": len(rows_by_object),
            "original_sequences": len(lifting["samples"]),
            "train_objects": len(train_objects),
            "train_sequences": len(train_rows),
            "heldout_objects": len(selected_objects),
            "heldout_sequences": len(val_rows),
            "old_ss64_excluded_from_holdout": len(old_ss64_objects),
            "train_objects_by_source": dict(train_source_counts),
            "original_objects_by_source": dict(original_source_counts),
        },
        "phase_contract": expected_phase_counts,
        "phases": phase_records,
        "old_ss64_object_uid_hash": hashlib.sha256(
            "\n".join(sorted(old_ss64_objects)).encode("utf-8")
        ).hexdigest(),
        "heldout_object_uid_hash": hashlib.sha256(
            "\n".join(sorted(selected_objects)).encode("utf-8")
        ).hexdigest(),
    }
    write_json(output_dir / "split_audit.json", audit)
    print(json.dumps(audit["counts"], indent=2))
    print(f"freeze-split PASS: {output_dir}")


def audit_split(args: argparse.Namespace) -> None:
    from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset
    from pose_point_depth_mv.native_ss_genrecon import (
        validate_genrecon_cache_contract,
    )

    split_dir = Path(args.split_dir)
    audit = read_json(split_dir / "split_audit.json")
    train_path = split_dir / "train868_manifest.json"
    val_path = split_dir / "val64_sourcebalanced_manifest.json"
    if audit.get("format") != SPLIT_FORMAT or audit.get("passed") is not True:
        raise ValueError("invalid split audit")
    if sha256_file(train_path) != audit["output"]["train_manifest_sha256"]:
        raise ValueError("train manifest hash changed")
    if sha256_file(val_path) != audit["output"]["val_manifest_sha256"]:
        raise ValueError("val manifest hash changed")
    train = PoseLiftingCacheDataset(train_path)
    val = PoseLiftingCacheDataset(val_path)
    train_contract = validate_genrecon_cache_contract(train)
    val_contract = validate_genrecon_cache_contract(
        val, training_config_hash=train.config_hash
    )
    train_objects = {str(row["object_uid"]) for row in train.rows}
    val_objects = {str(row["object_uid"]) for row in val.rows}
    checks = {
        "train_sequences": len(train) == 1417,
        "train_objects": len(train_objects) == 868,
        "val_sequences": len(val) == 64,
        "val_objects": len(val_objects) == 64,
        "object_disjoint": not train_objects.intersection(val_objects),
        "cache_contract_equal": train_contract == val_contract,
        "checkpoint_sources": audit["phase_contract"]["checkpoint"]["source"]
        == {source: 4 for source in SOURCES},
        "cfg_sources": audit["phase_contract"]["cfg"]["source"]
        == {source: 4 for source in SOURCES},
        "final_sources": audit["phase_contract"]["final"]["source"]
        == {source: 8 for source in SOURCES},
    }
    # The 64 held-out rows are small enough to replay the full cache contract.
    for index in range(len(val)):
        val[index]
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "phase_contract": audit["phase_contract"],
        "config_hash": train.config_hash,
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


def select_coarse(args: argparse.Namespace) -> None:
    root = Path(args.input_dir)
    output = Path(args.output)
    rows = []
    reference_objects = None
    for path in sorted(root.glob("step*_*/calibration.json")):
        report = read_json(path)
        if report["candidate_cfg_strengths"] != [1.0]:
            raise ValueError(f"coarse report searched CFG: {path}")
        protocol = report["protocol"]
        if protocol["joint_seeds"] != [42] or len(protocol["object_uids"]) != 16:
            raise ValueError(f"coarse report protocol mismatch: {path}")
        objects = tuple(protocol["object_uids"])
        reference_objects = reference_objects or objects
        if objects != reference_objects:
            raise ValueError("coarse candidates used different objects")
        candidate = report["candidates"][0]
        summary = candidate["summary"]
        count = candidate["count_summary"]["full_stock_count_ratio"]
        strict_checks = {
            "base_candidate_eligible": bool(candidate["eligible"]),
            "iou_object_win": float(summary["iou_gain"]["positive_rate"]) >= 0.5,
            "recall_object_win": float(summary["recall_gain"]["positive_rate"])
            >= 0.5,
            "count_ratio_median_lower": float(count["median"]) >= 0.85,
            "count_ratio_median_upper": float(count["median"]) <= 1.20,
            "stock_nonempty": int(candidate["stock_empty_record_count"]) == 0,
        }
        rows.append(
            {
                "report": str(path.resolve()),
                "checkpoint": protocol["checkpoint"],
                "checkpoint_sha256": protocol["checkpoint_sha256"],
                "step": int(protocol["checkpoint_step"]),
                "weights": protocol["weights"],
                "iou_gain": float(summary["iou_gain"]["mean"]),
                "iou_win": float(summary["iou_gain"]["positive_rate"]),
                "recall_gain": float(summary["recall_gain"]["mean"]),
                "recall_win": float(summary["recall_gain"]["positive_rate"]),
                "latent_mse_gain": float(summary["latent_mse_gain"]["mean"]),
                "count_ratio_mean": float(count["mean"]),
                "count_ratio_median": float(count["median"]),
                "strict_checks": strict_checks,
                "strict_eligible": all(strict_checks.values()),
            }
        )
    if len(rows) != 8:
        raise ValueError(f"expected 8 coarse reports, found {len(rows)}")
    eligible = [row for row in rows if row["strict_eligible"]]
    eligible.sort(
        key=lambda row: (
            row["iou_gain"],
            row["recall_gain"],
            row["latent_mse_gain"],
            -abs(row["count_ratio_median"] - 1.0),
            row["weights"] == "ema",
            -row["step"],
        ),
        reverse=True,
    )
    result = {
        "passed": bool(eligible),
        "selection_rule": (
            "strict gate, then lexicographic IoU/recall/latent/"
            "count-closeness/EMA/lower-step"
        ),
        "development_object_count": 16,
        "development_sources": {source: 4 for source in SOURCES},
        "selected": eligible[0] if eligible else None,
        "candidates": rows,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


def audit_calibration(args: argparse.Namespace) -> None:
    report = read_json(args.calibration)
    selected = report.get("selected")
    candidate = None
    if selected is not None:
        candidate = next(
            row
            for row in report["candidates"]
            if float(row["cfg_strength"]) == float(selected["cfg_strength"])
        )
    checks = {
        "base_calibration_passed": report.get("passed") is True,
        "source_balanced_object_count": len(report["protocol"]["object_uids"]) == 16,
        "three_joint_seeds": report["protocol"]["joint_seeds"] == [42, 43, 44],
    }
    if candidate is not None:
        summary = candidate["summary"]
        count = candidate["count_summary"]["full_stock_count_ratio"]
        checks.update(
            {
                "iou_object_win": float(summary["iou_gain"]["positive_rate"])
                >= 0.5,
                "recall_object_win": float(summary["recall_gain"]["positive_rate"])
                >= 0.5,
                "count_ratio_median_lower": float(count["median"]) >= 0.85,
                "count_ratio_median_upper": float(count["median"]) <= 1.20,
                "stock_nonempty": int(candidate["stock_empty_record_count"]) == 0,
            }
        )
    result = {
        "passed": bool(candidate is not None and all(checks.values())),
        "selected_cfg": None if selected is None else selected["cfg_strength"],
        "checks": checks,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


def summarize_final(args: argparse.Namespace) -> None:
    report = read_json(args.report)
    split = read_json(args.split_audit)
    source_by_object = {
        row["object_uid"]: row["source"] for row in split["phases"]["final"]
    }
    rows = report["correct"]["object_rows"]
    if len(rows) != 32 or set(source_by_object) != {
        row["object_uid"] for row in rows
    }:
        raise ValueError("final report does not match the frozen 32 objects")
    metrics = ("iou_gain", "precision_gain", "recall_gain", "latent_mse_gain")
    by_source = {}
    for source in SOURCES:
        selected = [
            row for row in rows if source_by_object[row["object_uid"]] == source
        ]
        if len(selected) != 8:
            raise ValueError(f"final source {source} has {len(selected)} objects")
        ratios = [
            float(row["full_stock_count_ratio"])
            for row in selected
            if row["full_stock_count_ratio"] is not None
        ]
        by_source[source] = {
            "object_count": len(selected),
            **{
                metric: {
                    "mean": float(
                        statistics.fmean(float(row[metric]) for row in selected)
                    ),
                    "median": float(
                        statistics.median(float(row[metric]) for row in selected)
                    ),
                    "positive_rate": float(
                        sum(float(row[metric]) > 0 for row in selected) / len(selected)
                    ),
                }
                for metric in metrics
            },
            "full_stock_count_ratio": {
                "defined_count": len(ratios),
                "mean": float(statistics.fmean(ratios)) if ratios else None,
                "median": float(statistics.median(ratios)) if ratios else None,
            },
        }
    summary = report["correct"]["summary"]
    count = report["correct"]["count_summary"]["full_stock_count_ratio"]
    checks = {
        "base_evaluator_passed": report["passed"] is True,
        "iou_mean_positive": float(summary["iou_gain"]["mean"]) > 0,
        "iou_median_positive": float(summary["iou_gain"]["median"]) > 0,
        "iou_object_win": float(summary["iou_gain"]["positive_rate"]) >= 0.5,
        "iou_ci_lower_positive": float(
            summary["iou_gain"]["bootstrap_mean_95_ci"][0]
        )
        > 0,
        "recall_mean_nonnegative": float(summary["recall_gain"]["mean"]) >= 0,
        "recall_median_nonnegative": float(summary["recall_gain"]["median"]) >= 0,
        "recall_object_win": float(summary["recall_gain"]["positive_rate"]) >= 0.5,
        "count_ratio_mean": 0.85 <= float(count["mean"]) <= 1.20,
        "count_ratio_median": 0.85 <= float(count["median"]) <= 1.20,
        "all_sources_exactly_eight": all(
            row["object_count"] == 8 for row in by_source.values()
        ),
        "all_source_count_ratios_defined": all(
            row["full_stock_count_ratio"]["defined_count"] == 8
            for row in by_source.values()
        ),
    }
    result = {
        "passed": all(checks.values()),
        "scope": "Native SS source-balanced final; SLAT and Mesh remain separate gates",
        "protocol": {
            "objects": 32,
            "joint_seeds": [42, 43, 44],
            "sources": {source: 8 for source in SOURCES},
            "view_counts": {"2": 11, "4": 11, "8": 10},
            "checkpoint": report["protocol"]["checkpoint"],
            "checkpoint_step": report["protocol"]["checkpoint_step"],
            "weights": report["protocol"]["weights"],
            "cfg_strength": report["calibrated_parameters"]["cfg_strength"],
        },
        "overall": {metric: summary[metric] for metric in metrics},
        "count_summary": report["correct"]["count_summary"],
        "correct_over_pose_control_iou": report["correct_over_pose_control_iou"],
        "by_source": by_source,
        "checks": checks,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-split")
    freeze.add_argument("--lifting_manifest", required=True)
    freeze.add_argument("--pointpose_manifest", required=True)
    freeze.add_argument("--output_dir", required=True)
    freeze.add_argument("--seed", type=int, default=20260801)
    freeze.set_defaults(func=freeze_split)

    audit = subparsers.add_parser("audit-split")
    audit.add_argument("--split_dir", required=True)
    audit.set_defaults(func=audit_split)

    select = subparsers.add_parser("select-coarse")
    select.add_argument("--input_dir", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(func=select_coarse)

    calibration = subparsers.add_parser("audit-calibration")
    calibration.add_argument("--calibration", required=True)
    calibration.add_argument("--output", required=True)
    calibration.set_defaults(func=audit_calibration)

    final = subparsers.add_parser("summarize-final")
    final.add_argument("--report", required=True)
    final.add_argument("--split_audit", required=True)
    final.add_argument("--output", required=True)
    final.set_defaults(func=summarize_final)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
