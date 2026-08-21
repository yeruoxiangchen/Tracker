#!/usr/bin/env python3
"""Aggregate the official SS/SLat ablation against strict ReconViaGen.

This is a read-only, CPU-only aggregation over already completed worker reports.
It deliberately evaluates only the held-out official Dev[16:64) slice and never
runs Train64 or a GT-support SLat experiment.

The four routes are:

R: strict ReconViaGen (VGGT -> Stock SS -> Stock SLat)
A: posed-DINO current interface -> Stock SS -> Stock SLat
B: posed-DINO current interface -> official Native-SS step2000 -> Stock SLat
C: posed-DINO current interface -> official Native-SS step2000 -> Native-SLat

B-A isolates the Native-SS change, C-B isolates the Native-SLat change, and
C-R compares the complete deployed endpoint with strict original ReconViaGen.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat import (
    _aggregate_occupancy,
)
from pose_point_depth_mv.evaluate_proobjaverse_official_reconviagen import (
    CURRENT_WORKER_FORMAT,
    LOWER_IS_BETTER,
    RECON_METHOD,
    _absolute_summary,
    _complete_object_uids,
    _load_contract,
    _load_recon_reports,
    _object_means,
    _verify_internal_hash,
    add_common_paths,
    bootstrap_mean_ci,
    parse_csv,
    summarize,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)


REPORT_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_ss_slat_vs_reconviagen.v1"
)
SEEDS = (42, 43, 44)
CURRENT_BRANCHES = ("stock", "native", "native_trained")

ROUTE_R = RECON_METHOD
ROUTE_A = "posed_dino_stock_ss_stock_slat"
ROUTE_B = "posed_dino_official_native_ss_step2000_stock_slat"


def _route_c(step: int) -> str:
    return f"posed_dino_official_native_ss_step2000_native_slat_step{int(step)}"


def _expected_pairs(object_uids: list[str]) -> set[tuple[str, int]]:
    return {(uid, seed) for uid in object_uids for seed in SEEDS}


def _verify_live_artifact(path_value: Any, expected_sha256: Any) -> dict[str, Any]:
    path = Path(str(path_value)).expanduser().resolve(strict=True)
    expected = str(expected_sha256)
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise RuntimeError(
            f"live artifact SHA256 differs: path={path} "
            f"expected={expected} actual={actual}"
        )
    return {"path": str(path), "sha256": actual, "passed": True}


def _validate_current_failure(row: dict[str, Any]) -> None:
    """Accept only failures deliberately recorded as model decoder outcomes."""

    branch = str(row.get("branch", ""))
    error = row.get("error")
    expected_stage = {
        "stock": "stock_slat_mesh_decode",
        "native": "native_slat_mesh_decode",
        "native_trained": "native_trained_slat_mesh_decode",
    }.get(branch)
    if (
        row.get("passed") is not False
        or expected_stage is None
        or not isinstance(error, dict)
        or str(error.get("type")) != "RuntimeError"
        or str(error.get("stage")) != expected_stage
        or not str(error.get("message", "")).startswith(
            "SLat decoder input exceeds safe active-point limit:"
        )
    ):
        raise RuntimeError(
            "unapproved failed current endpoint record: "
            f"branch={branch} uid={row.get('object_uid')} seed={row.get('seed')}"
        )


def _load_current_branch_reports(
    paths: list[str],
    *,
    all_dev_uids: list[str],
    expected_step: int,
    expected_sha256: str,
    expected_protocol_sha256: str,
) -> dict[str, Any]:
    heldout_uids = all_dev_uids[16:64]
    heldout_pairs = _expected_pairs(heldout_uids)
    branch_records: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        branch: {} for branch in CURRENT_BRANCHES
    }
    ss_records: list[dict[str, Any]] = []
    ss_keys: set[tuple[str, int]] = set()
    observed_uids: list[str] = []
    bindings: list[dict[str, Any]] = []
    shared_identity: dict[str, Any] | None = None
    shared_ss_binding: dict[str, Any] | None = None
    shared_slat_binding: dict[str, Any] | None = None

    for value in paths:
        path = Path(value).expanduser().resolve(strict=True)
        payload = load_json(path)
        _verify_internal_hash(payload, path=path)
        if (
            payload.get("format") != CURRENT_WORKER_FORMAT
            or payload.get("complete") is not True
        ):
            raise RuntimeError(f"invalid/incomplete current worker report: {path}")

        identity = dict(payload.get("run_identity", {}))
        required_identity = {
            "format": CURRENT_WORKER_FORMAT,
            "expected_trained_slat_step": int(expected_step),
            "trained_slat_checkpoint_sha256": str(expected_sha256),
            "trained_slat_weights": "ema",
            "official_protocol_sha256": str(expected_protocol_sha256),
            "joint_seeds": list(SEEDS),
            "weights": "ema",
            "amp_dtype": "bf16",
            "surface_samples": 20000,
            "save_meshes": False,
            "same_ss_initial_noise": True,
            "same_slat_condition": True,
            "same_coordinate_keyed_slat_noise": True,
            "stock_slat_pair_native_ss_only_difference": True,
            "end_to_end_pair_changes_native_ss_and_slat": True,
            "paired_branches": list(CURRENT_BRANCHES),
        }
        mismatch = {
            key: {"observed": identity.get(key), "expected": expected}
            for key, expected in required_identity.items()
            if identity.get(key) != expected
        }
        if mismatch:
            raise RuntimeError(f"current worker identity differs: {mismatch}")

        start = int(identity.get("object_start", -1))
        end = int(identity.get("object_end", -1))
        object_uids = [str(uid) for uid in identity.get("object_uids", [])]
        if (
            start < 16
            or end <= start
            or end > 64
            or object_uids != all_dev_uids[start:end]
            or int(payload.get("object_count", -1)) != end - start
            or int(payload.get("record_count", -1))
            != (end - start) * len(SEEDS)
        ):
            raise RuntimeError(f"current worker object slice differs: {path}")
        observed_uids.extend(object_uids)

        comparable_identity = {
            key: item
            for key, item in identity.items()
            if key not in {"object_start", "object_end", "object_uids"}
        }
        if shared_identity is None:
            shared_identity = comparable_identity
        elif comparable_identity != shared_identity:
            raise RuntimeError("current worker run identities differ")

        native_ss_binding = dict(payload.get("native_ss_binding", {}))
        required_ss_binding = {
            "checkpoint_step": 2000,
            "weights": "ema",
            "amp_dtype": "bf16",
            "false_checks": [],
        }
        ss_mismatch = {
            key: {
                "observed": native_ss_binding.get(key),
                "expected": expected,
            }
            for key, expected in required_ss_binding.items()
            if native_ss_binding.get(key) != expected
        }
        if ss_mismatch:
            raise RuntimeError(f"Native-SS binding differs: {ss_mismatch}")
        if shared_ss_binding is None:
            shared_ss_binding = native_ss_binding
        elif native_ss_binding != shared_ss_binding:
            raise RuntimeError("current worker Native-SS bindings differ")

        trained_slat = dict(payload.get("trained_slat", {}))
        required_slat_binding = {
            "checkpoint_step": int(expected_step),
            "checkpoint_sha256": str(expected_sha256),
            "weights": "ema",
        }
        slat_mismatch = {
            key: {"observed": trained_slat.get(key), "expected": expected}
            for key, expected in required_slat_binding.items()
            if trained_slat.get(key) != expected
        }
        if slat_mismatch:
            raise RuntimeError(f"Native-SLat binding differs: {slat_mismatch}")
        if shared_slat_binding is None:
            shared_slat_binding = trained_slat
        elif trained_slat != shared_slat_binding:
            raise RuntimeError("current worker Native-SLat bindings differ")

        local_expected_pairs = _expected_pairs(object_uids)
        local_ss_keys: set[tuple[str, int]] = set()
        for row in payload.get("ss_records", []):
            key = (str(row.get("object_uid")), int(row.get("seed", -1)))
            if key in ss_keys or key in local_ss_keys:
                raise RuntimeError(f"duplicate current SS record={key}")
            if row.get("passed") is not True or row.get("same_initial_noise") is not True:
                raise RuntimeError(f"invalid current SS record={key}")
            local_ss_keys.add(key)
            ss_keys.add(key)
            ss_records.append(row)
        if local_ss_keys != local_expected_pairs:
            raise RuntimeError(f"current SS shard matrix differs: {path}")

        local_branch_keys: dict[str, set[tuple[str, int]]] = {
            branch: set() for branch in CURRENT_BRANCHES
        }
        local_by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = (
            defaultdict(dict)
        )
        for row in payload.get("mesh_branch_records", []):
            branch = str(row.get("branch", ""))
            if branch not in branch_records:
                raise RuntimeError(f"unknown current mesh branch={branch}")
            key = (str(row.get("object_uid")), int(row.get("seed", -1)))
            if key in branch_records[branch] or key in local_branch_keys[branch]:
                raise RuntimeError(f"duplicate current mesh record={branch}:{key}")
            if not isinstance(row.get("target_structure"), dict):
                raise RuntimeError(f"current mesh target structure missing={branch}:{key}")
            if row.get("passed") is not True:
                _validate_current_failure(row)
            local_branch_keys[branch].add(key)
            branch_records[branch][key] = row
            local_by_pair[key][branch] = row
        for branch, keys in local_branch_keys.items():
            if keys != local_expected_pairs:
                raise RuntimeError(
                    f"current branch shard matrix differs: branch={branch} path={path}"
                )
        for key, rows in local_by_pair.items():
            if set(rows) != set(CURRENT_BRANCHES):
                raise RuntimeError(f"current pair lacks all branches={key}")
            target_structures = [rows[branch]["target_structure"] for branch in CURRENT_BRANCHES]
            if any(value != target_structures[0] for value in target_structures[1:]):
                raise RuntimeError(f"current branches use different targets={key}")

        bindings.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "object_start": start,
                "object_end": end,
                "runtime_passed": payload.get("passed") is True,
            }
        )

    if (
        len(observed_uids) != 48
        or len(set(observed_uids)) != 48
        or set(observed_uids) != set(heldout_uids)
    ):
        raise RuntimeError("current worker reports do not exactly cover Dev[16:64)")
    if ss_keys != heldout_pairs:
        raise RuntimeError("current SS records do not exactly cover Dev48 x three seeds")
    for branch in CURRENT_BRANCHES:
        if set(branch_records[branch]) != heldout_pairs:
            raise RuntimeError(
                f"current branch does not exactly cover Dev48 x three seeds={branch}"
            )
    if shared_identity is None or shared_ss_binding is None or shared_slat_binding is None:
        raise RuntimeError("no current worker reports were loaded")

    # Revalidate the live scientific artifacts once.  This aggregator does not
    # modify them and never trusts a path string without its frozen content hash.
    if shared_identity.get("native_ss_report") != shared_ss_binding.get("report"):
        raise RuntimeError("run identity and Native-SS binding report paths differ")
    if shared_identity.get("native_ss_report_sha256") != shared_ss_binding.get(
        "report_sha256"
    ):
        raise RuntimeError("run identity and Native-SS binding report hashes differ")
    if shared_identity.get("trained_slat_checkpoint") != shared_slat_binding.get(
        "checkpoint"
    ):
        raise RuntimeError("run identity and Native-SLat checkpoint paths differ")

    artifact_checks = {
        "cache_manifest": _verify_live_artifact(
            shared_identity["cache_manifest"],
            shared_identity["cache_manifest_sha256"],
        ),
        "lifting_cache_manifest": _verify_live_artifact(
            shared_identity["lifting_cache_manifest"],
            shared_identity["lifting_cache_manifest_sha256"],
        ),
        "native_ss_report": _verify_live_artifact(
            shared_ss_binding["report"], shared_ss_binding["report_sha256"]
        ),
        "native_ss_checkpoint": _verify_live_artifact(
            shared_ss_binding["checkpoint"],
            shared_ss_binding["checkpoint_sha256"],
        ),
        "stock_slat_freeze": _verify_live_artifact(
            shared_identity["stock_slat_freeze"],
            shared_identity["stock_slat_freeze_sha256"],
        ),
        "native_slat_checkpoint": _verify_live_artifact(
            shared_slat_binding["checkpoint"],
            shared_slat_binding["checkpoint_sha256"],
        ),
    }
    return {
        "worker_reports": sorted(bindings, key=lambda row: row["object_start"]),
        "branch_records": branch_records,
        "ss_records": ss_records,
        "shared_identity": shared_identity,
        "native_ss_binding": shared_ss_binding,
        "native_slat_binding": shared_slat_binding,
        "artifact_checks": artifact_checks,
    }


def paired_route_summary(
    candidate: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    *,
    candidate_name: str,
    baseline_name: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    comparison_kind: str,
    clean_component_isolation: bool,
    caveat: str,
) -> dict[str, Any]:
    """Return object-paired deltas; positive always means candidate is better."""

    if not candidate or set(candidate) != set(baseline):
        raise RuntimeError("paired route object sets differ or are empty")
    reference_metrics: set[str] | None = None
    improvements: dict[str, list[float]] = defaultdict(list)
    per_object: dict[str, dict[str, float]] = {}
    for uid in sorted(candidate):
        candidate_metrics = set(candidate[uid])
        baseline_metrics = set(baseline[uid])
        if candidate_metrics != baseline_metrics:
            raise RuntimeError(f"paired route metric sets differ uid={uid}")
        if reference_metrics is None:
            reference_metrics = candidate_metrics
        elif candidate_metrics != reference_metrics:
            raise RuntimeError(f"route metric schema changes across objects uid={uid}")
        per_object[uid] = {}
        for metric in sorted(candidate_metrics):
            if metric in LOWER_IS_BETTER:
                delta = baseline[uid][metric] - candidate[uid][metric]
            else:
                delta = candidate[uid][metric] - baseline[uid][metric]
            improvements[metric].append(float(delta))
            per_object[uid][metric] = float(delta)

    metric_deltas: dict[str, Any] = {}
    for position, (metric, values) in enumerate(sorted(improvements.items())):
        metric_deltas[metric] = {
            **summarize(values),
            "object_bootstrap_mean_95_ci": bootstrap_mean_ci(
                values,
                samples=int(bootstrap_samples),
                seed=int(bootstrap_seed) + position,
            ),
            "delta_formula": (
                "baseline_minus_candidate"
                if metric in LOWER_IS_BETTER
                else "candidate_minus_baseline"
            ),
        }
    return {
        "candidate": candidate_name,
        "baseline": baseline_name,
        "comparison_kind": comparison_kind,
        "clean_component_isolation": bool(clean_component_isolation),
        "positive_definition": "positive means candidate is better than baseline",
        "unit_of_analysis": (
            "the same complete held-out Dev48 objects; every object value is the "
            "mean of seeds 42/43/44"
        ),
        "object_count": len(candidate),
        "metric_deltas": metric_deltas,
        "per_object_deltas": per_object,
        "caveat": caveat,
    }


def _failed_records(
    records: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    failures = []
    for (uid, seed), row in records.items():
        if row.get("passed") is True:
            continue
        failures.append(
            {
                "object_uid": uid,
                "seed": seed,
                "error": row.get("error"),
            }
        )
    return sorted(failures, key=lambda row: (row["object_uid"], row["seed"]))


def _route_runtime_summary(
    records: dict[tuple[str, int], dict[str, Any]],
    heldout_uids: list[str],
) -> dict[str, Any]:
    complete = _complete_object_uids(records, heldout_uids)
    successful = sum(row.get("passed") is True for row in records.values())
    return {
        "record_count": len(records),
        "successful_record_count": successful,
        "failed_record_count": len(records) - successful,
        "mesh_success_rate": float(successful / len(records)),
        "complete_surface_object_count": len(complete),
        "complete_surface_object_uids": complete,
        "incomplete_surface_object_uids": [
            uid for uid in heldout_uids if uid not in set(complete)
        ],
        "failed_seed_records": _failed_records(records),
    }


def _format_core(comparison: dict[str, Any]) -> list[str]:
    lines = []
    labels = (
        ("chamfer_l1", "Chamfer-L1 improvement"),
        ("fscore_0p02", "F-score@0.02 delta"),
        ("normal_consistency", "normal-consistency delta"),
        ("largest_component_ratio", "largest-component-ratio delta"),
    )
    for key, label in labels:
        row = comparison["metric_deltas"][key]
        lines.append(
            f"  {label}: mean={row['mean']:+.8f} median={row['median']:+.8f} "
            f"win={row['positive_rate']:.6f} "
            f"CI={row['object_bootstrap_mean_95_ci']}"
        )
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = _load_contract(args)
    all_dev_uids = [str(row["uid"]) for row in contract["rows"]]
    heldout_uids = all_dev_uids[16:64]
    heldout_pairs = _expected_pairs(heldout_uids)

    recon_bindings, recon_all_records = _load_recon_reports(
        parse_csv(args.recon_reports, str)
    )
    if set(recon_all_records) != _expected_pairs(all_dev_uids):
        raise RuntimeError("strict ReconViaGen reports do not exactly cover Dev64 x 3")
    recon_records = {
        key: row for key, row in recon_all_records.items() if key in heldout_pairs
    }
    if set(recon_records) != heldout_pairs:
        raise RuntimeError("strict ReconViaGen reports do not cover held-out Dev48 x 3")

    current = _load_current_branch_reports(
        parse_csv(args.current_reports, str),
        all_dev_uids=all_dev_uids,
        expected_step=int(args.expected_current_step),
        expected_sha256=str(args.expected_current_sha256),
        expected_protocol_sha256=str(contract["protocol_sha256"]),
    )
    branch_records = current["branch_records"]

    # All four routes must be scored against the exact same target structure.
    for key in sorted(heldout_pairs):
        target = recon_records[key].get("target_structure")
        if not isinstance(target, dict):
            raise RuntimeError(f"strict ReconViaGen target structure missing={key}")
        for branch in CURRENT_BRANCHES:
            if branch_records[branch][key].get("target_structure") != target:
                raise RuntimeError(
                    f"strict/current target Mesh structure differs={branch}:{key}"
                )

    records_by_route = {
        ROUTE_R: recon_records,
        ROUTE_A: branch_records["stock"],
        ROUTE_B: branch_records["native"],
        _route_c(args.expected_current_step): branch_records["native_trained"],
    }
    runtime = {
        route: _route_runtime_summary(records, heldout_uids)
        for route, records in records_by_route.items()
    }
    complete_sets = {
        route: set(summary["complete_surface_object_uids"])
        for route, summary in runtime.items()
    }
    common_uids = [
        uid
        for uid in heldout_uids
        if all(uid in values for values in complete_sets.values())
    ]
    if not common_uids:
        raise RuntimeError("no common complete objects remain across R/A/B/C")

    means = {
        route: _object_means(records, common_uids)
        for route, records in records_by_route.items()
    }
    route_c = _route_c(args.expected_current_step)
    comparisons = {
        "current_ss_vs_stock_ss_same_current_interface": paired_route_summary(
            means[ROUTE_B],
            means[ROUTE_A],
            candidate_name=ROUTE_B,
            baseline_name=ROUTE_A,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=2026081600,
            comparison_kind="B_minus_A_native_ss_component_ablation",
            clean_component_isolation=True,
            caveat=(
                "The input, noise, Stock SLat, decoder, target, and metric are "
                "fixed; only Stock SS is replaced by official Native-SS step2000."
            ),
        ),
        "current_slat_vs_stock_slat_same_current_ss": paired_route_summary(
            means[route_c],
            means[ROUTE_B],
            candidate_name=route_c,
            baseline_name=ROUTE_B,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=2026081700,
            comparison_kind="C_minus_B_native_slat_component_ablation",
            clean_component_isolation=True,
            caveat=(
                "The official Native-SS support, posed-DINO condition, "
                "coordinate-keyed noise, decoder, target, and metric are fixed; "
                f"only Stock SLat is replaced by Native-SLat step{int(args.expected_current_step)}."
            ),
        ),
        "current_full_vs_strict_reconviagen": paired_route_summary(
            means[route_c],
            means[ROUTE_R],
            candidate_name=route_c,
            baseline_name=ROUTE_R,
            bootstrap_samples=int(args.bootstrap_samples),
            # Preserve the exact deterministic bootstrap used by the existing
            # strict ReconViaGen-vs-current aggregate for this same C-R pair.
            bootstrap_seed=20260816,
            comparison_kind="C_minus_R_complete_endpoint_comparison",
            clean_component_isolation=False,
            caveat=(
                "This is the requested complete endpoint comparison. It changes "
                "the input interface (VGGT to posed-DINO), SS, and SLat together, "
                "so it must not be attributed to either learned component alone."
            ),
        ),
        "current_ss_stock_slat_vs_strict_reconviagen": paired_route_summary(
            means[ROUTE_B],
            means[ROUTE_R],
            candidate_name=ROUTE_B,
            baseline_name=ROUTE_R,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=2026081900,
            comparison_kind="B_minus_R_diagnostic",
            clean_component_isolation=False,
            caveat=(
                "Diagnostic only: both the input interface and SS change, so this "
                "is not a pure Native-SS effect. Use B-A for the SS attribution."
            ),
        ),
        "current_interface_all_stock_vs_strict_reconviagen": paired_route_summary(
            means[ROUTE_A],
            means[ROUTE_R],
            candidate_name=ROUTE_A,
            baseline_name=ROUTE_R,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=2026082000,
            comparison_kind="A_minus_R_input_interface_diagnostic",
            clean_component_isolation=False,
            caveat=(
                "Diagnostic only: both routes use Stock SS and Stock SLat, but R "
                "uses strict ReconViaGen VGGT while A uses the posed-DINO interface."
            ),
        ),
    }

    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "runtime_integrity_passed": True,
        "formal": False,
        "post_selection_development_diagnostic": True,
        "split": {
            "name": "official_proobjaverse_dev",
            "object_start": 16,
            "object_end": 64,
            "requested_object_count": 48,
            "seeds": list(SEEDS),
            "train_evaluated": False,
            "gt_support_slat_evaluated": False,
            "reason": (
                "Dev[0:16) was used for Native-SS CFG calibration; this report "
                "uses only held-out Dev[16:64)."
            ),
        },
        "official_protocol_sha256": contract["protocol_sha256"],
        "common_complete_object_count": len(common_uids),
        "common_complete_object_uids": common_uids,
        "excluded_from_paired_surface_metrics": [
            uid for uid in heldout_uids if uid not in set(common_uids)
        ],
        "route_runtime": runtime,
        "route_absolute_summary_on_common_objects": {
            route: _absolute_summary(values) for route, values in means.items()
        },
        "current_ss_occupancy_vs_stock_ss": _aggregate_occupancy(
            current["ss_records"],
            bootstrap_samples=int(args.bootstrap_samples),
        ),
        "comparisons": comparisons,
        "method_contracts": {
            ROUTE_R: {
                "label": "R",
                "pipeline": "VGGT -> Stock SS -> Stock SLat -> Stock Mesh decoder",
                "explicit_camera_pose_consumed": False,
                "vggt_model_executed": True,
            },
            ROUTE_A: {
                "label": "A",
                "pipeline": (
                    "posed-DINO current interface -> Stock SS -> Stock SLat -> "
                    "Stock Mesh decoder"
                ),
                "explicit_camera_pose_consumed": True,
                "vggt_model_executed": False,
            },
            ROUTE_B: {
                "label": "B",
                "pipeline": (
                    "posed-DINO current interface -> official Native-SS step2000 "
                    "EMA -> Stock SLat -> Stock Mesh decoder"
                ),
            },
            route_c: {
                "label": "C",
                "pipeline": (
                    "posed-DINO current interface -> official Native-SS step2000 "
                    f"EMA -> Native-SLat step{int(args.expected_current_step)} EMA "
                    "-> Stock Mesh decoder"
                ),
            },
        },
        "comparability_guard": {
            "same_official_object_seed_pairs": True,
            "same_target_mesh_structure": True,
            "same_surface_samples": 20000,
            "same_fscore_thresholds": [0.01, 0.02, 0.05],
            "same_common_object_denominator_for_all_comparisons": True,
            "b_minus_a_isolates_native_ss": True,
            "c_minus_b_isolates_native_slat": True,
            "c_minus_r_is_complete_endpoint_not_component_attribution": True,
        },
        "artifact_checks": current["artifact_checks"],
        "reconviagen_worker_reports": recon_bindings,
        "current_worker_reports": current["worker_reports"],
        "native_ss_binding": current["native_ss_binding"],
        "native_slat_binding": current["native_slat_binding"],
        "source_contract": {
            "dev_split": str(contract["split_path"]),
            "dev_split_sha256": contract["split_sha256"],
            "cache_report": str(contract["cache_report_path"]),
            "cache_report_sha256": contract["cache_report_sha256"],
            "target_report": str(contract["target_report_path"]),
            "target_report_sha256": contract["target_report_sha256"],
            "paired_target_cache_roots": contract["paired_target_cache_roots"],
        },
        "scope_guard": (
            "This is a held-out Dev48 quantitative development diagnostic. It "
            "does not evaluate Train64 or GT-support SLat. Surface comparisons "
            "use only the same objects with three valid seeds in every R/A/B/C "
            "route; all decoder failures and per-route success rates remain visible."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)

    lines = [
        "Official ProObjaverse held-out Dev48: SS2k/SLat vs strict ReconViaGen",
        "=" * 74,
        "R = strict ReconViaGen: VGGT -> Stock SS -> Stock SLat",
        "A = posed-DINO current interface -> Stock SS -> Stock SLat",
        "B = posed-DINO -> official Native-SS step2000 -> Stock SLat",
        (
            "C = posed-DINO -> official Native-SS step2000 -> "
            f"Native-SLat step{int(args.expected_current_step)}"
        ),
        "Train64 evaluated: no; GT-support Dev evaluated: no",
        f"common complete objects: {len(common_uids)}/48",
        f"excluded objects: {report['excluded_from_paired_surface_metrics']}",
        "",
        "B-A: current Native-SS benefit with interface and Stock SLat fixed",
    ]
    lines.extend(
        _format_core(comparisons["current_ss_vs_stock_ss_same_current_interface"])
    )
    lines.extend(
        [
            "",
            "C-B: current Native-SLat benefit with current SS support fixed",
        ]
    )
    lines.extend(
        _format_core(comparisons["current_slat_vs_stock_slat_same_current_ss"])
    )
    lines.extend(["", "C-R: complete current endpoint vs strict ReconViaGen"])
    lines.extend(_format_core(comparisons["current_full_vs_strict_reconviagen"]))
    lines.extend(
        [
            "",
            "A-R and B-R are retained in report.json as interface diagnostics; ",
            "they are not pure component attributions.",
            report["scope_guard"],
        ]
    )

    if args.dry_run:
        print("\n".join(lines))
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "passed": True,
                    "report_sha256_if_written_now": report["report_sha256"],
                    "output_dir_not_created": str(
                        Path(args.output_dir).expanduser().resolve()
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return report

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "report.json", report)
    lines.append(f"report: {output / 'report.json'}")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_paths(parser)
    parser.add_argument("--recon_reports", required=True)
    parser.add_argument("--current_reports", required=True)
    parser.add_argument("--expected_current_step", type=int, default=25000)
    parser.add_argument("--expected_current_sha256", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
