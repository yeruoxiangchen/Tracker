#!/usr/bin/env python3
"""Pure helpers for the frozen Direct-SLAT end-to-end blind holdout."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL_FORMAT = "pose_point_depth_mv.direct_slat_blind_protocol.v2"
SEALED_REPORT_FORMAT = "pose_point_depth_mv.direct_slat_blind_sealed.v2"
PREFLIGHT_REPORT_FORMAT = "pose_point_depth_mv.direct_slat_blind_preflight.v2"
FINAL_REPORT_FORMAT = "pose_point_depth_mv.direct_slat_blind_final.v2"
PUBLIC_BUNDLE_FORMAT = "pose_point_depth_mv.direct_slat_public_bundle.v2"
PUBLIC_ARCHIVE_FORMAT = "pose_point_depth_mv.direct_slat_public_archive.v2"
RATINGS_FREEZE_FORMAT = "pose_point_depth_mv.direct_slat_ratings_freeze.v2"
HOLDOUT_INTEGRITY_FORMAT = "pose_point_depth_mv.direct_slat_holdout_integrity.v2"
EXECUTION_COMPATIBILITY_FORMAT = (
    "pose_point_depth_mv.direct_slat_execution_compatibility.v2"
)

BRANCHES = ("stock", "full")
SIDES = ("A", "B")
TARGET_FAMILY_FIELDS = (
    "format",
    "source_kind",
    "encoder_config_sha256",
    "encoder_weights_sha256",
    "mesh_decoder_weights_sha256",
    "dinov2_hubconf_sha256",
    "dinov2_checkpoint_sha256",
    "dinov2_model",
    "image_size",
    "min_views",
    "max_views",
    "coordinate_source",
    "feature_fusion",
    "posterior",
)
RATER_COLUMNS = (
    "rater_id",
    "pair_id",
    "main_structure_A",
    "main_structure_B",
    "missing_parts_A",
    "missing_parts_B",
    "floating_fragments_A",
    "floating_fragments_B",
    "thin_spikes_A",
    "thin_spikes_B",
    "holes_open_boundaries_A",
    "holes_open_boundaries_B",
    "overall_score_A",
    "overall_score_B",
    "overall_preference",
    "notes",
)
RATING_BENEFIT_DIMENSIONS = ("main_structure", "overall_score")
RATING_DEFECT_DIMENSIONS = (
    "missing_parts",
    "floating_fragments",
    "thin_spikes",
    "holes_open_boundaries",
)
CONTINUOUS_TOPOLOGY_METRICS = (
    "largest_component_ratio",
    "boundary_edge_count",
    "boundary_total_length",
    "nonmanifold_edge_count",
    "component_count",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def bind_file(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def validate_file_binding(binding: dict[str, Any], label: str) -> None:
    path = Path(str(binding.get("path", "")))
    expected = str(binding.get("sha256", ""))
    if not path.is_file() or not expected or sha256_file(path) != expected:
        raise RuntimeError(f"frozen file binding changed: {label}={path}")


def validate_binding_tree(value: Any, label: str = "bindings") -> None:
    if isinstance(value, dict):
        if set(("path", "sha256")).issubset(value):
            validate_file_binding(value, label)
            return
        for key, child in value.items():
            validate_binding_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_binding_tree(child, f"{label}[{index}]")


def parse_csv(value: str, cast) -> list[Any]:
    result = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("CSV values must be non-empty and unique")
    return result


def pair_identity(
    protocol_name: str,
    uid: str,
    seed: int,
    blind_key: bytes,
) -> tuple[str, dict[str, str]]:
    message = f"{protocol_name}|{uid}|{int(seed)}".encode("utf-8")
    pair_id = blind_pair_id(protocol_name, uid, seed)
    digest = hmac.new(blind_key, message, hashlib.sha256).digest()
    if digest[0] & 1:
        return pair_id, {"A": "full", "B": "stock"}
    return pair_id, {"A": "stock", "B": "full"}


def blind_pair_id(protocol_name: str, uid: str, seed: int) -> str:
    message = f"{protocol_name}|{uid}|{int(seed)}".encode("utf-8")
    return hashlib.sha256(message).hexdigest()[:20]


def execution_compatibility_projection(
    protocol: dict[str, Any],
) -> dict[str, Any]:
    bindings = protocol["bindings"]
    return {
        "format": EXECUTION_COMPATIBILITY_FORMAT,
        "candidate": protocol["candidate"],
        "code_bindings": protocol["code_bindings"],
        "runtime_bindings": protocol["runtime_bindings"],
        "checkpoint_sha256": {
            "ss_flow": bindings["ss_flow_checkpoint"]["sha256"],
            "direct_slat": bindings["direct_slat_checkpoint"]["sha256"],
        },
        "pretrained_id": protocol["pretrained_id"],
        "pretrained": protocol["pretrained"],
        "support_runtime_identity": protocol["support_runtime_identity"],
        "target_family_identity": protocol["target_family_identity"],
        "runtime_calibration_identity": protocol.get(
            "runtime_calibration_identity"
        ),
        "runtime": protocol["runtime"],
        "sampling": protocol["sampling"],
        "mesh": protocol["mesh"],
    }


def execution_compatibility_record(
    protocol: dict[str, Any],
) -> dict[str, Any]:
    projection = execution_compatibility_projection(protocol)
    return {
        "format": EXECUTION_COMPATIBILITY_FORMAT,
        "sha256": canonical_sha256(projection),
        "projection": projection,
    }


def validate_execution_compatibility_record(
    protocol: dict[str, Any],
) -> dict[str, Any]:
    saved = protocol.get("execution_compatibility")
    expected = execution_compatibility_record(protocol)
    if saved != expected:
        raise RuntimeError("frozen execution compatibility record changed")
    return expected


def target_family_identity(run_config: dict[str, Any]) -> dict[str, Any]:
    config = run_config.get("config")
    if not isinstance(config, dict):
        raise ValueError("local lh-slats run config has no config dictionary")
    missing = [name for name in TARGET_FAMILY_FIELDS if name not in config]
    if missing:
        raise ValueError(f"local lh-slats target family lacks fields={missing}")
    fields = {name: config[name] for name in TARGET_FAMILY_FIELDS}
    if fields["format"] != "pose_point_depth_mv.local_lh_slats.v2":
        raise ValueError(f"unsupported local lh-slats format={fields['format']!r}")
    return {"fields": fields, "hash": canonical_sha256(fields)}


def manifest_identity_sets(payload: dict[str, Any]) -> dict[str, set[str]]:
    rows = list(payload.get("samples", []))
    objects = list(payload.get("objects", []))
    object_uids = {
        str(row.get("object_uid", row.get("uid", ""))) for row in (*rows, *objects)
    }
    object_uids.discard("")
    source_paths = {
        str(Path(str(row["source_glb"])).resolve())
        for row in (*rows, *objects)
        if row.get("source_glb")
    }
    source_hashes = {
        str(row["source_glb_sha256"])
        for row in (*rows, *objects)
        if row.get("source_glb_sha256")
    }
    return {
        "object_uids": object_uids,
        "source_glb_paths": source_paths,
        "source_glb_sha256": source_hashes,
    }


def assert_unseen_holdout(
    holdout: dict[str, Any],
    seen_manifests: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    holdout_ids = manifest_identity_sets(holdout)
    if not holdout_ids["object_uids"] or not holdout_ids["source_glb_sha256"]:
        raise ValueError("holdout must bind object UIDs and source GLB hashes")
    seen = {
        "object_uids": set(),
        "source_glb_paths": set(),
        "source_glb_sha256": set(),
    }
    for payload in seen_manifests:
        identity = manifest_identity_sets(payload)
        for key in seen:
            seen[key].update(identity[key])
    overlaps = {
        key: sorted(holdout_ids[key] & seen[key])
        for key in holdout_ids
    }
    if any(overlaps.values()):
        raise RuntimeError(
            "confirmatory holdout is not unseen: "
            f"uid={overlaps['object_uids'][:8]} "
            f"path={overlaps['source_glb_paths'][:8]} "
            f"sha256={overlaps['source_glb_sha256'][:8]}"
        )
    return {
        "passed": True,
        "holdout_object_count": len(holdout_ids["object_uids"]),
        "seen_object_count": len(seen["object_uids"]),
        "object_uid_overlap_count": 0,
        "source_glb_path_overlap_count": 0,
        "source_glb_sha256_overlap_count": 0,
    }


def select_object_rows(
    samples: list[dict[str, Any]],
    *,
    seeds: Iterable[int],
    max_objects: int = 0,
) -> list[dict[str, Any]]:
    required_seeds = tuple(int(seed) for seed in seeds)
    if not required_seeds or len(required_seeds) != len(set(required_seeds)):
        raise ValueError("selection seeds must be non-empty and unique")
    by_object: dict[str, dict[str, dict[int, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for index, row in enumerate(samples):
        object_uid = str(row.get("object_uid", ""))
        uid = str(row.get("uid", ""))
        seed = int(row.get("support_seed", -1))
        if not object_uid or not uid or seed < 0:
            raise ValueError(f"invalid direct-SLAT sample identity at index={index}")
        if seed in by_object[object_uid][uid]:
            raise ValueError(f"duplicate direct-SLAT row uid={uid} seed={seed}")
        by_object[object_uid][uid][seed] = index
    selected = []
    for object_uid in sorted(by_object):
        eligible = [
            (uid, seed_map)
            for uid, seed_map in sorted(by_object[object_uid].items())
            if all(seed in seed_map for seed in required_seeds)
        ]
        if not eligible:
            raise RuntimeError(
                f"object={object_uid} has no sequence covering seeds={required_seeds}"
            )
        uid, seed_map = eligible[0]
        selected.append(
            {
                "object_position": len(selected),
                "object_uid": object_uid,
                "uid": uid,
                "view_count": int(samples[seed_map[required_seeds[0]]].get("view_count", 0)),
                "cache_indices": {
                    str(seed): int(seed_map[seed]) for seed in required_seeds
                },
            }
        )
    if int(max_objects) > 0:
        selected = selected[: int(max_objects)]
    if not selected:
        raise ValueError("Direct-SLAT blind selection is empty")
    return selected


def runtime_selection_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    selection = list(protocol.get("selection", {}).get("rows", []))
    sample_bindings = list(protocol.get("sample_bindings", []))
    if not selection or len(selection) != len(sample_bindings):
        raise RuntimeError("selection rows and sample bindings differ in length")
    by_uid: dict[str, dict[str, Any]] = {}
    for row in sample_bindings:
        uid = str(row.get("uid", ""))
        if not uid or uid in by_uid:
            raise RuntimeError(f"missing or duplicate sample binding UID={uid!r}")
        by_uid[uid] = row
    output = []
    identity_fields = (
        "object_position",
        "object_uid",
        "uid",
        "view_count",
        "cache_indices",
    )
    for frozen in selection:
        uid = str(frozen.get("uid", ""))
        bound = by_uid.get(uid)
        if bound is None or any(
            bound.get(name) != frozen.get(name) for name in identity_fields
        ):
            raise RuntimeError(
                f"selection and sample binding identities differ for UID={uid!r}"
            )
        if "source_lifting_index" not in bound:
            raise RuntimeError(
                f"sample binding lacks source lifting index for UID={uid!r}"
            )
        output.append(bound)
    if {str(row["uid"]) for row in output} != set(by_uid):
        raise RuntimeError("sample bindings contain rows outside frozen selection")
    return output


def summarize_values(
    values: Iterable[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("summary values must be a non-empty finite vector")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(array), size=(int(bootstrap_samples), len(array)))
    means = array[draws].mean(axis=1)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "object_win_rate": float(np.mean(array > 0.0)),
        "nonnegative_rate": float(np.mean(array >= 0.0)),
        "bootstrap_mean_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def repeat_floors(
    rows: Iterable[dict[str, Any]],
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("repeat calibration contains no rows")
    metric_names = (
        "chamfer_l1_abs",
        "fscore_0p02_abs",
        "largest_component_ratio_abs",
        "boundary_edge_count_abs",
        "boundary_total_length_abs",
        "nonmanifold_edge_count_abs",
        "component_count_abs",
    )
    max_abs = {
        name: float(max(float(row["metric_abs_diff"][name]) for row in records))
        for name in metric_names
    }
    p95_abs = {}
    median_abs = {}
    for name in metric_names:
        values = np.asarray(
            [float(row["metric_abs_diff"][name]) for row in records],
            dtype=np.float64,
        )
        try:
            p95 = np.quantile(values, 0.95, method="higher")
        except TypeError:  # NumPy < 1.22
            p95 = np.quantile(values, 0.95, interpolation="higher")
        p95_abs[name] = float(p95)
        median_abs[name] = float(np.median(values))
    topology_changes = {
        name: sum(bool(row["topology_changed"][name]) for row in records)
        for name in (
            "mesh_success",
            "is_watertight",
            "zero_boundary",
            "nonmanifold_free",
        )
    }
    topology_change_rates = {
        name: float(count / len(records))
        for name, count in topology_changes.items()
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row.get("branch", "")), str(row.get("object_uid", "")))].append(
            row
        )

    def grouped_summary(values: list[dict[str, Any]]) -> dict[str, Any]:
        group_p95 = {}
        group_max = {}
        for metric_name in metric_names:
            array = np.asarray(
                [float(row["metric_abs_diff"][metric_name]) for row in values],
                dtype=np.float64,
            )
            try:
                quantile = np.quantile(array, 0.95, method="higher")
            except TypeError:  # NumPy < 1.22
                quantile = np.quantile(array, 0.95, interpolation="higher")
            group_p95[metric_name] = float(quantile)
            group_max[metric_name] = float(array.max())
        rates = {
            name: float(
                np.mean([bool(row["topology_changed"][name]) for row in values])
            )
            for name in topology_changes
        }
        return {
            "record_count": len(values),
            "p95_abs": group_p95,
            "max_abs": group_max,
            "topology_change_rates": rates,
        }

    by_branch_object = {
        f"{branch}|{object_uid}": grouped_summary(values)
        for (branch, object_uid), values in sorted(grouped.items())
    }
    worst_group_p95_abs = {
        name: float(
            max(group["p95_abs"][name] for group in by_branch_object.values())
        )
        for name in metric_names
    }
    worst_group_topology_change_rates = {
        name: float(
            max(
                group["topology_change_rates"][name]
                for group in by_branch_object.values()
            )
        )
        for name in topology_changes
    }
    mode = str((policy or {}).get("mode", "zero_jitter"))
    checks: dict[str, bool]
    if mode == "zero_jitter":
        checks = {
            "finite_metrics": bool(all(np.isfinite(list(max_abs.values())))),
            "zero_topology_category_jitter": not any(topology_changes.values()),
        }
        interpretation = (
            "same-model rerun calibration only; confirmatory deltas must exceed "
            "the frozen surface metric floor and topology categories may not jitter"
        )
    elif mode == "multi_repeat_p95":
        regular = dict(policy.get("regular_p95_max", {}))
        catastrophic = dict(policy.get("catastrophic_max", {}))
        flip_limits = dict(policy.get("topology_flip_rate_max", {}))
        if set(regular) != set(metric_names) or set(catastrophic) != set(
            metric_names
        ):
            raise ValueError("multi-repeat policy does not cover every repeat metric")
        expected_flip_names = {"is_watertight", "zero_boundary", "nonmanifold_free"}
        if set(flip_limits) != expected_flip_names:
            raise ValueError("multi-repeat policy topology limits are incomplete")
        checks = {
            "finite_metrics": bool(all(np.isfinite(list(max_abs.values())))),
            "zero_mesh_success_jitter": topology_changes["mesh_success"] == 0,
            **{
                f"p95_{name}": worst_group_p95_abs[name] <= float(regular[name])
                for name in metric_names
            },
            **{
                f"max_{name}": max_abs[name] <= float(catastrophic[name])
                for name in metric_names
            },
            **{
                f"flip_rate_{name}": worst_group_topology_change_rates[name]
                <= float(flip_limits[name])
                for name in expected_flip_names
            },
        }
        interpretation = str(
            policy.get(
                "interpretation",
                "p95 repeat floors with catastrophic max and topology flip limits",
            )
        )
    else:
        raise ValueError(f"unsupported repeat policy mode={mode!r}")
    return {
        "passed": bool(all(checks.values())),
        "policy_mode": mode,
        "record_count": len(records),
        "median_abs": median_abs,
        "p95_abs": p95_abs,
        "worst_group_p95_abs": worst_group_p95_abs,
        "max_abs": max_abs,
        "topology_change_counts": topology_changes,
        "topology_change_rates": topology_change_rates,
        "worst_group_topology_change_rates": worst_group_topology_change_rates,
        "by_branch_object": by_branch_object,
        "checks": checks,
        "interpretation": interpretation,
    }


def _paired_rows(
    records: list[dict[str, Any]],
    mapping: dict[str, dict[str, str]],
    *,
    expected_pairs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    counts: Counter[tuple[str, str]] = Counter()
    for row in records:
        pair_id = str(row.get("pair_id", ""))
        side = str(row.get("side", ""))
        if pair_id not in mapping or side not in SIDES:
            raise ValueError(f"unexpected sealed metric identity={pair_id}/{side}")
        branch = str(mapping[pair_id].get(side, ""))
        if branch not in BRANCHES:
            raise ValueError(f"invalid unblinding branch={branch!r}")
        counts[(pair_id, branch)] += 1
        by_pair[pair_id][branch] = row
    pair_deltas = []
    invalid = []
    for pair_id, branch_rows in sorted(by_pair.items()):
        bad_counts = {
            branch: counts[(pair_id, branch)]
            for branch in BRANCHES
            if counts[(pair_id, branch)] != 1
        }
        if bad_counts or set(branch_rows) != set(BRANCHES):
            invalid.append(
                {
                    "pair_id": pair_id,
                    "error": "duplicate or missing branch",
                    "counts": bad_counts,
                }
            )
            continue
        stock, full = branch_rows["stock"], branch_rows["full"]
        if (
            str(stock.get("object_uid")) != str(full.get("object_uid"))
            or int(stock.get("seed", -1)) != int(full.get("seed", -1))
        ):
            invalid.append({"pair_id": pair_id, "error": "pair identities differ"})
            continue
        if stock.get("passed") is not True or full.get("passed") is not True:
            invalid.append({"pair_id": pair_id, "error": "branch generation failed"})
            continue
        stock_surface, full_surface = stock["surface"], full["surface"]
        stock_structure, full_structure = stock["structure"], full["structure"]
        pair_deltas.append(
            {
                "pair_id": pair_id,
                "object_uid": str(stock["object_uid"]),
                "seed": int(stock["seed"]),
                "chamfer_l1_improvement": float(stock_surface["chamfer_l1"])
                - float(full_surface["chamfer_l1"]),
                "fscore_0p02_delta": float(full_surface["fscore_0p02"])
                - float(stock_surface["fscore_0p02"]),
                "normal_consistency_delta": float(
                    full_surface["normal_consistency"]
                )
                - float(stock_surface["normal_consistency"]),
                "largest_component_ratio_delta": float(
                    full_structure["largest_component_ratio"]
                )
                - float(stock_structure["largest_component_ratio"]),
                "boundary_edge_count_delta": float(
                    full_structure["boundary_edge_count"]
                )
                - float(stock_structure["boundary_edge_count"]),
                "boundary_total_length_delta": float(
                    full_structure["boundary_total_length"]
                )
                - float(stock_structure["boundary_total_length"]),
                "nonmanifold_edge_count_delta": float(
                    full_structure["nonmanifold_edge_count"]
                )
                - float(stock_structure["nonmanifold_edge_count"]),
                "connected_component_count_delta": float(
                    full_structure["component_count"]
                )
                - float(stock_structure["component_count"]),
                "mesh_success_delta": float(full_structure["mesh_success"])
                - float(stock_structure["mesh_success"]),
                "watertight_rate_delta": float(full_structure["is_watertight"])
                - float(stock_structure["is_watertight"]),
                "zero_boundary_rate_delta": float(
                    int(full_structure["boundary_edge_count"]) == 0
                )
                - float(int(stock_structure["boundary_edge_count"]) == 0),
                "nonmanifold_free_rate_delta": float(
                    int(full_structure["nonmanifold_edge_count"]) == 0
                )
                - float(int(stock_structure["nonmanifold_edge_count"]) == 0),
                "stock": {
                    "mesh_success": bool(stock_structure["mesh_success"]),
                    "vertices_finite": bool(stock_structure["vertices_finite"]),
                    "is_winding_consistent": bool(
                        stock_structure["is_winding_consistent"]
                    ),
                },
                "full": {
                    "mesh_success": bool(full_structure["mesh_success"]),
                    "vertices_finite": bool(full_structure["vertices_finite"]),
                    "is_winding_consistent": bool(
                        full_structure["is_winding_consistent"]
                    ),
                },
            }
        )
    if len(by_pair) != int(expected_pairs):
        invalid.append(
            {
                "error": "unexpected pair count",
                "actual": len(by_pair),
                "expected": int(expected_pairs),
            }
        )
    return pair_deltas, invalid


def aggregate_unblinded(
    records: list[dict[str, Any]],
    mapping: dict[str, dict[str, str]],
    *,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    expected_pairs = int(protocol["selection"]["object_count"]) * len(
        protocol["sampling"]["joint_seeds"]
    )
    pair_rows, invalid = _paired_rows(
        records, mapping, expected_pairs=expected_pairs
    )
    metric_names = (
        "chamfer_l1_improvement",
        "fscore_0p02_delta",
        "normal_consistency_delta",
        "largest_component_ratio_delta",
        "boundary_edge_count_delta",
        "boundary_total_length_delta",
        "nonmanifold_edge_count_delta",
        "connected_component_count_delta",
        "mesh_success_delta",
        "watertight_rate_delta",
        "zero_boundary_rate_delta",
        "nonmanifold_free_rate_delta",
    )
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_object[str(row["object_uid"])].append(row)
    object_rows = [
        {
            "object_uid": object_uid,
            **{
                name: float(np.mean([float(row[name]) for row in rows]))
                for name in metric_names
            },
        }
        for object_uid, rows in sorted(by_object.items())
    ]
    floors = protocol["statistics"]["repeat_floors"]
    for row in object_rows:
        row["topology_worsened"] = bool(
            float(row["largest_component_ratio_delta"])
            < -float(floors["largest_component_ratio_abs"])
            or float(row["boundary_edge_count_delta"])
            > float(floors["boundary_edge_count_abs"])
            or float(row["boundary_total_length_delta"])
            > float(floors["boundary_total_length_abs"])
            or float(row["nonmanifold_edge_count_delta"])
            > float(floors["nonmanifold_edge_count_abs"])
            or float(row["connected_component_count_delta"])
            > float(floors["component_count_abs"])
        )
    bootstrap_samples = int(protocol["statistics"]["bootstrap_samples"])
    summary = (
        {
            name: summarize_values(
                [float(row[name]) for row in object_rows],
                bootstrap_samples=bootstrap_samples,
                seed=2026072500 + position,
            )
            for position, name in enumerate(metric_names)
        }
        if object_rows
        else {}
    )
    seed_summary = {}
    for seed in protocol["sampling"]["joint_seeds"]:
        rows = [row for row in pair_rows if int(row["seed"]) == int(seed)]
        seed_summary[str(seed)] = (
            {
                name: summarize_values(
                    [float(row[name]) for row in rows],
                    bootstrap_samples=bootstrap_samples,
                    seed=2026072600 + int(seed) * 17 + position,
                )
                for position, name in enumerate(metric_names)
            }
            if rows
            else {}
        )

    thresholds = protocol["statistics"]["checks"]
    chamfer = summary.get("chamfer_l1_improvement", {})
    fscore = summary.get("fscore_0p02_delta", {})
    branch_rows = [
        value
        for row in pair_rows
        for value in (row["stock"], row["full"])
    ]
    checks = {
        "expected_record_count": len(records) == 2 * expected_pairs,
        "expected_valid_pair_count": not invalid
        and len(pair_rows) == expected_pairs,
        "expected_object_count": len(object_rows)
        == int(protocol["selection"]["object_count"]),
        "both_branches_mesh_success_1": bool(branch_rows)
        and all(row["mesh_success"] for row in branch_rows),
        "both_branches_vertices_finite": bool(branch_rows)
        and all(row["vertices_finite"] for row in branch_rows),
        "both_branches_winding_consistent": bool(branch_rows)
        and all(row["is_winding_consistent"] for row in branch_rows),
        "chamfer_mean_positive": float(chamfer.get("mean", -np.inf)) > 0.0,
        "chamfer_median_positive": float(chamfer.get("median", -np.inf)) > 0.0,
        "chamfer_object_win": float(chamfer.get("object_win_rate", 0.0))
        >= float(thresholds["chamfer_object_win_rate_min"]),
        "chamfer_ci_lower_exceeds_repeat_floor": float(
            chamfer.get("bootstrap_mean_95_ci", [-np.inf, -np.inf])[0]
        )
        > float(floors["chamfer_l1_abs"]),
        "fscore_mean_exceeds_repeat_floor": float(fscore.get("mean", -np.inf))
        > float(floors["fscore_0p02_abs"]),
        "fscore_ci_lower_exceeds_repeat_floor": float(
            fscore.get("bootstrap_mean_95_ci", [-np.inf, -np.inf])[0]
        )
        > float(floors["fscore_0p02_abs"]),
        "largest_component_non_degrading": float(
            summary.get("largest_component_ratio_delta", {}).get("mean", -np.inf)
        )
        >= float(thresholds["largest_component_ratio_mean_delta_min"]),
        "largest_component_ci_non_degrading": float(
            summary.get("largest_component_ratio_delta", {}).get(
                "bootstrap_mean_95_ci", [-np.inf, -np.inf]
            )[0]
        )
        >= float(thresholds["largest_component_ratio_mean_delta_min"])
        - float(floors["largest_component_ratio_abs"]),
        "boundary_edge_count_mean_bounded": float(
            summary.get("boundary_edge_count_delta", {}).get("mean", np.inf)
        )
        <= float(thresholds["boundary_edge_count_mean_increase_max"])
        + float(floors["boundary_edge_count_abs"]),
        "boundary_edge_count_ci_bounded": float(
            summary.get("boundary_edge_count_delta", {}).get(
                "bootstrap_mean_95_ci", [np.inf, np.inf]
            )[1]
        )
        <= float(thresholds["boundary_edge_count_mean_increase_max"])
        + float(floors["boundary_edge_count_abs"]),
        "boundary_total_length_mean_bounded": float(
            summary.get("boundary_total_length_delta", {}).get("mean", np.inf)
        )
        <= float(thresholds["boundary_total_length_mean_increase_max"])
        + float(floors["boundary_total_length_abs"]),
        "boundary_total_length_ci_bounded": float(
            summary.get("boundary_total_length_delta", {}).get(
                "bootstrap_mean_95_ci", [np.inf, np.inf]
            )[1]
        )
        <= float(thresholds["boundary_total_length_mean_increase_max"])
        + float(floors["boundary_total_length_abs"]),
        "nonmanifold_edge_count_mean_bounded": float(
            summary.get("nonmanifold_edge_count_delta", {}).get("mean", np.inf)
        )
        <= float(thresholds["nonmanifold_edge_count_mean_increase_max"])
        + float(floors["nonmanifold_edge_count_abs"]),
        "nonmanifold_edge_count_ci_bounded": float(
            summary.get("nonmanifold_edge_count_delta", {}).get(
                "bootstrap_mean_95_ci", [np.inf, np.inf]
            )[1]
        )
        <= float(thresholds["nonmanifold_edge_count_mean_increase_max"])
        + float(floors["nonmanifold_edge_count_abs"]),
        "component_count_mean_bounded": float(
            summary.get("connected_component_count_delta", {}).get(
                "mean", np.inf
            )
        )
        <= float(thresholds["component_count_mean_increase_max"])
        + float(floors["component_count_abs"]),
        "component_count_ci_bounded": float(
            summary.get("connected_component_count_delta", {}).get(
                "bootstrap_mean_95_ci", [np.inf, np.inf]
            )[1]
        )
        <= float(thresholds["component_count_mean_increase_max"])
        + float(floors["component_count_abs"]),
        "topology_object_worsening_rate_bounded": (
            float(np.mean([bool(row["topology_worsened"]) for row in object_rows]))
            if object_rows
            else np.inf
        )
        <= float(thresholds["topology_object_worsening_rate_max"]),
        "no_catastrophic_object_topology_regression": bool(object_rows)
        and all(
            float(row["largest_component_ratio_delta"])
            >= float(thresholds["largest_component_ratio_object_delta_min"])
            and float(row["boundary_edge_count_delta"])
            <= float(thresholds["boundary_edge_count_object_increase_max"])
            and float(row["boundary_total_length_delta"])
            <= float(thresholds["boundary_total_length_object_increase_max"])
            and float(row["nonmanifold_edge_count_delta"])
            <= float(thresholds["nonmanifold_edge_count_object_increase_max"])
            and float(row["connected_component_count_delta"])
            <= float(thresholds["component_count_object_increase_max"])
            for row in object_rows
        ),
        "no_catastrophic_pair_topology_regression": bool(pair_rows)
        and all(
            float(row["largest_component_ratio_delta"])
            >= float(thresholds["largest_component_ratio_pair_delta_min"])
            and float(row["boundary_edge_count_delta"])
            <= float(thresholds["boundary_edge_count_pair_increase_max"])
            and float(row["boundary_total_length_delta"])
            <= float(thresholds["boundary_total_length_pair_increase_max"])
            and float(row["nonmanifold_edge_count_delta"])
            <= float(thresholds["nonmanifold_edge_count_pair_increase_max"])
            and float(row["connected_component_count_delta"])
            <= float(thresholds["component_count_pair_increase_max"])
            for row in pair_rows
        ),
        "watertight_rate_non_degrading": float(
            summary.get("watertight_rate_delta", {}).get("mean", -np.inf)
        )
        >= float(thresholds["watertight_rate_delta_min"]),
        "zero_boundary_rate_non_degrading": float(
            summary.get("zero_boundary_rate_delta", {}).get("mean", -np.inf)
        )
        >= float(thresholds["zero_boundary_rate_delta_min"]),
        "nonmanifold_free_rate_non_degrading": float(
            summary.get("nonmanifold_free_rate_delta", {}).get("mean", -np.inf)
        )
        >= float(thresholds["nonmanifold_free_rate_delta_min"]),
    }
    nonnegative_seeds = sum(
        float(row.get("chamfer_l1_improvement", {}).get("mean", -np.inf)) >= 0.0
        for row in seed_summary.values()
    )
    checks["minimum_nonnegative_seed_directions"] = nonnegative_seeds >= int(
        thresholds["minimum_nonnegative_seed_directions"]
    )
    checks["every_seed_above_catastrophic_chamfer_floor"] = bool(seed_summary) and all(
        float(row.get("chamfer_l1_improvement", {}).get("mean", -np.inf))
        >= float(thresholds["per_seed_chamfer_mean_min"])
        for row in seed_summary.values()
    )
    return {
        "automatic_passed": all(checks.values()),
        "checks": checks,
        "expected_pair_count": expected_pairs,
        "valid_pair_count": len(pair_rows),
        "invalid_pairs": invalid,
        "object_rows": object_rows,
        "summary": summary,
        "seed_summary": seed_summary,
        "pair_deltas": pair_rows,
        "weighting": "average paired seeds within object, then bootstrap objects",
    }


def read_and_validate_rater_csv(
    path: str | Path,
    *,
    expected_pair_ids: Iterable[str],
) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RATER_COLUMNS:
            raise ValueError(
                f"rater CSV columns differ: {reader.fieldnames} != {RATER_COLUMNS}"
            )
        rows = list(reader)
    expected = set(str(value) for value in expected_pair_ids)
    pair_ids = [str(row["pair_id"]).strip() for row in rows]
    if len(pair_ids) != len(set(pair_ids)) or set(pair_ids) != expected:
        raise ValueError(f"rater CSV pair coverage differs: {source}")
    rater_ids = {str(row["rater_id"]).strip() for row in rows}
    if len(rater_ids) != 1 or "" in rater_ids:
        raise ValueError(f"rater CSV must contain one non-empty rater_id: {source}")
    normalized = []
    for row in rows:
        values: dict[str, dict[str, int]] = {}
        try:
            for dimension in RATING_BENEFIT_DIMENSIONS:
                values[dimension] = {
                    side: int(row[f"{dimension}_{side}"]) for side in SIDES
                }
            for dimension in RATING_DEFECT_DIMENSIONS:
                values[dimension] = {
                    side: int(row[f"{dimension}_{side}"]) for side in SIDES
                }
        except ValueError as error:
            raise ValueError(f"rater scores must be integers: {source}") from error
        if any(
            score not in range(1, 6)
            for dimension in RATING_BENEFIT_DIMENSIONS
            for score in values[dimension].values()
        ):
            raise ValueError(
                f"benefit scores must lie in [1,5]: {source}"
            )
        if any(
            score not in range(0, 4)
            for dimension in RATING_DEFECT_DIMENSIONS
            for score in values[dimension].values()
        ):
            raise ValueError(
                f"defect severity scores must lie in [0,3]: {source}"
            )
        preference = str(row["overall_preference"]).strip()
        if preference not in {"A", "B", "tie"}:
            raise ValueError(f"invalid rater preference={preference!r}: {source}")
        normalized.append(
            {
                "rater_id": next(iter(rater_ids)),
                "pair_id": str(row["pair_id"]).strip(),
                "scores": values,
                "preference": preference,
            }
        )
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "rater_id": next(iter(rater_ids)),
        "rows": normalized,
    }


def aggregate_ratings(
    rater_files: list[dict[str, Any]],
    *,
    mapping: dict[str, dict[str, str]],
    pair_to_object: dict[str, str],
    bootstrap_samples: int,
    checks_config: dict[str, Any],
) -> dict[str, Any]:
    if len(rater_files) < 3:
        raise ValueError("confirmatory unblinding requires at least three raters")
    rater_ids = [str(row["rater_id"]) for row in rater_files]
    if len(rater_ids) != len(set(rater_ids)):
        raise ValueError("rater IDs must be unique across score files")
    by_object: dict[str, list[dict[str, float]]] = defaultdict(list)
    for file_row in rater_files:
        for row in file_row["rows"]:
            pair_id = str(row["pair_id"])
            side_mapping = mapping[pair_id]
            full_side = next(side for side, branch in side_mapping.items() if branch == "full")
            stock_side = next(side for side, branch in side_mapping.items() if branch == "stock")
            scores = row["scores"]
            preference = str(row["preference"])
            preference_delta = (
                0.0
                if preference == "tie"
                else 1.0
                if side_mapping[preference] == "full"
                else -1.0
            )
            by_object[pair_to_object[pair_id]].append(
                {
                    **{
                        f"{dimension}_delta": float(
                            scores[dimension][full_side]
                            - scores[dimension][stock_side]
                        )
                        for dimension in (
                            *RATING_BENEFIT_DIMENSIONS,
                            *RATING_DEFECT_DIMENSIONS,
                        )
                    },
                    **{
                        f"{dimension}_severe_rate_delta": float(
                            int(scores[dimension][full_side] >= 2)
                            - int(scores[dimension][stock_side] >= 2)
                        )
                        for dimension in RATING_DEFECT_DIMENSIONS
                    },
                    "overall_preference_delta": preference_delta,
                }
            )
    metric_names = (
        *(f"{name}_delta" for name in RATING_BENEFIT_DIMENSIONS),
        *(f"{name}_delta" for name in RATING_DEFECT_DIMENSIONS),
        *(f"{name}_severe_rate_delta" for name in RATING_DEFECT_DIMENSIONS),
        "overall_preference_delta",
    )
    object_rows = [
        {
            "object_uid": object_uid,
            **{
                name: float(np.mean([row[name] for row in rows]))
                for name in metric_names
            },
        }
        for object_uid, rows in sorted(by_object.items())
    ]
    summary = {
        name: summarize_values(
            [float(row[name]) for row in object_rows],
            bootstrap_samples=int(bootstrap_samples),
            seed=2026072700 + position,
        )
        for position, name in enumerate(metric_names)
    }
    checks = {
        "at_least_three_unique_raters": len(rater_files) >= 3,
        "main_structure_mean_non_degrading": summary["main_structure_delta"][
            "mean"
        ]
        >= float(checks_config["main_structure_mean_delta_min"]),
        "main_structure_ci_noninferior": summary["main_structure_delta"][
            "bootstrap_mean_95_ci"
        ][0]
        >= float(checks_config["main_structure_ci_lower_min"]),
        "overall_score_strictly_favors_full": summary["overall_score_delta"][
            "mean"
        ]
        > float(checks_config["overall_score_mean_min_exclusive"]),
        "overall_score_ci_nonnegative": summary["overall_score_delta"][
            "bootstrap_mean_95_ci"
        ][0]
        >= float(checks_config["overall_score_ci_lower_min"]),
        "overall_preference_strictly_favors_full": summary[
            "overall_preference_delta"
        ]["mean"]
        > float(checks_config["overall_preference_mean_min_exclusive"]),
        "overall_preference_ci_nonnegative": summary[
            "overall_preference_delta"
        ]["bootstrap_mean_95_ci"][0]
        >= float(checks_config["overall_preference_ci_lower_min"]),
    }
    for dimension in RATING_DEFECT_DIMENSIONS:
        checks[f"{dimension}_mean_non_degrading"] = summary[
            f"{dimension}_delta"
        ]["mean"] <= float(checks_config["defect_mean_delta_max"])
        severe = summary[f"{dimension}_severe_rate_delta"]
        checks[f"{dimension}_severe_rate_noninferior"] = (
            severe["mean"]
            <= float(checks_config["severe_defect_rate_delta_max"])
            and severe["bootstrap_mean_95_ci"][1]
            <= float(checks_config["severe_defect_rate_ci_upper_max"])
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "rater_count": len(rater_files),
        "rater_files": [
            {
                "path": row["path"],
                "sha256": row["sha256"],
                "rater_id": row["rater_id"],
            }
            for row in rater_files
        ],
        "object_rows": object_rows,
        "summary": summary,
        "weighting": "average seeds and raters within object, then bootstrap objects",
    }
