#!/usr/bin/env python3
"""Aggregate the official SS/SLat ablation against strict ReconViaGen.

This is a read-only, CPU-only aggregation over completed worker reports.  The
legacy default remains Dev[16:64); registered callers may request an exact
held-out Dev range such as the disjoint official 30K Dev[0:64).

The four routes are:

R: strict ReconViaGen (VGGT -> Stock SS -> Stock SLat)
A: posed-DINO current interface -> Stock SS -> Stock SLat
B: posed-DINO current interface -> official Native-SS -> Stock SLat
C: posed-DINO current interface -> official Native-SS -> Native-SLat

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
    _validate_worker_shard_local_bindings,
    _worker_global_run_identity,
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
def _route_b(ss_step: int) -> str:
    return f"posed_dino_official_native_ss_step{int(ss_step)}_stock_slat"


def _route_c(ss_step: int, slat_step: int) -> str:
    return (
        f"posed_dino_official_native_ss_step{int(ss_step)}_"
        f"native_slat_step{int(slat_step)}"
    )


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
    object_start: int = 16,
    object_end: int = 64,
    expected_ss_step: int = 2000,
    allow_native_ss_science_failed: bool = False,
    require_all_disjoint_membership: bool = False,
    evaluation_membership_scope: str = "heldout",
) -> dict[str, Any]:
    if not 0 <= int(object_start) < int(object_end) <= len(all_dev_uids):
        raise ValueError("invalid current-report evaluation object range")
    heldout_uids = all_dev_uids[int(object_start) : int(object_end)]
    heldout_pairs = _expected_pairs(heldout_uids)
    branch_records: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        branch: {} for branch in CURRENT_BRANCHES
    }
    ss_records: list[dict[str, Any]] = []
    ss_keys: set[tuple[str, int]] = set()
    observed_uids: list[str] = []
    bindings: list[dict[str, Any]] = []
    shard_local_identity_bindings: list[dict[str, Any]] = []
    shared_identity: dict[str, Any] | None = None
    shared_floor_identity: dict[str, Any] | None | object = object()
    shared_ss_binding: dict[str, Any] | None = None
    shared_slat_binding: dict[str, Any] | None = None
    membership_audits: list[dict[str, Any]] = []

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
        if allow_native_ss_science_failed:
            if identity.get("allow_native_ss_science_failed") is not True:
                raise RuntimeError(
                    "current worker lacks registered science-failed SS opt-in"
                )
        if evaluation_membership_scope == "checkpoint_training_overlap":
            registered = {
                "allow_trained_slat_target_protocol_mismatch": True,
                "expected_checkpoint_training_membership": "all_training",
            }
            registration_mismatch = {
                key: {"observed": identity.get(key), "expected": expected}
                for key, expected in registered.items()
                if identity.get(key) != expected
            }
            if registration_mismatch:
                raise RuntimeError(
                    "current worker cross-protocol registration differs: "
                    f"{registration_mismatch}"
                )

        start = int(identity.get("object_start", -1))
        end = int(identity.get("object_end", -1))
        object_uids = [str(uid) for uid in identity.get("object_uids", [])]
        if (
            start < int(object_start)
            or end <= start
            or end > int(object_end)
            or object_uids != all_dev_uids[start:end]
            or int(payload.get("object_count", -1)) != end - start
            or int(payload.get("record_count", -1))
            != (end - start) * len(SEEDS)
        ):
            raise RuntimeError(f"current worker object slice differs: {path}")
        observed_uids.extend(object_uids)

        membership = payload.get("checkpoint_evaluation_membership")
        if evaluation_membership_scope == "checkpoint_training_overlap":
            if (
                not isinstance(membership, dict)
                or membership.get("passed") is not True
                or membership.get("expected_membership") != "all_training"
                or membership.get("protocol_relation") != "different"
                or membership.get(
                    "all_evaluation_objects_in_checkpoint_training"
                )
                is not True
                or int(membership.get("training_overlap_count", -1))
                != len(object_uids)
            ):
                raise RuntimeError(
                    f"current worker training-membership audit differs: {path}"
                )
        if membership is not None:
            membership_audits.append(dict(membership))

        local_binding = _validate_worker_shard_local_bindings(
            payload,
            report_path=path,
            seeds=list(SEEDS),
        )
        shard_local_identity_bindings.append(local_binding)
        floor_identity = local_binding["frozen_stock_floor_global_identity"]
        if not isinstance(shared_floor_identity, (dict, type(None))):
            shared_floor_identity = floor_identity
        elif floor_identity != shared_floor_identity:
            raise RuntimeError(
                "current worker frozen Stock-floor global identities differ"
            )

        comparable_identity = _worker_global_run_identity(identity)
        if shared_identity is None:
            shared_identity = comparable_identity
        elif comparable_identity != shared_identity:
            raise RuntimeError("current worker run identities differ")

        native_ss_binding = dict(payload.get("native_ss_binding", {}))
        required_ss_binding = {
            "checkpoint_step": int(expected_ss_step),
            "weights": "ema",
            "amp_dtype": "bf16",
        }
        false_checks = native_ss_binding.get("false_checks")
        if (
            not isinstance(false_checks, list)
            or len(false_checks) != len(set(str(value) for value in false_checks))
            or (
                not allow_native_ss_science_failed
                and false_checks != []
            )
        ):
            raise RuntimeError("Native-SS failed-check registration differs")
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

    expected_object_count = int(object_end) - int(object_start)
    if (
        len(observed_uids) != expected_object_count
        or len(set(observed_uids)) != expected_object_count
        or set(observed_uids) != set(heldout_uids)
    ):
        raise RuntimeError(
            "current worker reports do not exactly cover registered Dev range"
        )
    if ss_keys != heldout_pairs:
        raise RuntimeError(
            "current SS records do not exactly cover registered Dev range x seeds"
        )
    for branch in CURRENT_BRANCHES:
        if set(branch_records[branch]) != heldout_pairs:
            raise RuntimeError(
                "current branch does not exactly cover registered Dev range x "
                f"seeds={branch}"
            )
    if shared_identity is None or shared_ss_binding is None or shared_slat_binding is None:
        raise RuntimeError("no current worker reports were loaded")
    invariant_keys = (
        "checkpoint_protocol_sha256",
        "evaluation_protocol_sha256",
        "checkpoint_training_object_count",
        "checkpoint_training_uid_sha256",
    )
    if evaluation_membership_scope == "checkpoint_training_overlap":
        first_membership = membership_audits[0]
        if len(membership_audits) != len(paths) or any(
            any(row.get(key) != first_membership.get(key) for key in invariant_keys)
            for row in membership_audits[1:]
        ):
            raise RuntimeError("current worker training-membership identities differ")
    elif require_all_disjoint_membership:
        if len(membership_audits) != len(paths):
            raise RuntimeError("held-out workers lack checkpoint membership audits")
        first_membership = membership_audits[0]
        if any(
            any(row.get(key) != first_membership.get(key) for key in invariant_keys)
            for row in membership_audits[1:]
        ):
            raise RuntimeError("held-out checkpoint membership identities differ")
        for membership in membership_audits:
            if (
                membership.get("passed") is not True
                or membership.get("expected_membership") != "all_disjoint"
                or int(membership.get("training_overlap_count", -1)) != 0
                or membership.get(
                    "all_evaluation_objects_disjoint_from_checkpoint_training"
                )
                is not True
            ):
                raise RuntimeError("held-out checkpoint membership audit differs")

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
    native_ss_report_payload = load_json(shared_ss_binding["report"])
    native_ss_report_checks = native_ss_report_payload.get("checks")
    if not isinstance(native_ss_report_checks, dict):
        raise RuntimeError("live Native-SS report lacks registered checks")
    report_false_checks = sorted(
        str(key)
        for key, value in native_ss_report_checks.items()
        if value is not True
    )
    if (
        report_false_checks != shared_ss_binding.get("false_checks")
        or bool(native_ss_report_payload.get("passed"))
        != (not report_false_checks)
    ):
        raise RuntimeError("Native-SS report/binding science status differs")
    return {
        "worker_reports": sorted(bindings, key=lambda row: row["object_start"]),
        "shard_local_identity_bindings": sorted(
            shard_local_identity_bindings,
            key=lambda row: row["object_start"],
        ),
        "branch_records": branch_records,
        "ss_records": ss_records,
        "shared_identity": shared_identity,
        "native_ss_binding": shared_ss_binding,
        "native_slat_binding": shared_slat_binding,
        "artifact_checks": artifact_checks,
        "checkpoint_evaluation_membership": (
            None
            if not membership_audits
            else {
                "version": "pose_point_depth_mv.slat_checkpoint_evaluation_membership_aggregate.v1",
                **{key: membership_audits[0][key] for key in invariant_keys},
                "expected_membership": str(
                    membership_audits[0]["expected_membership"]
                ),
                "protocol_relation": str(
                    membership_audits[0]["protocol_relation"]
                ),
                "evaluation_object_count": expected_object_count,
                "training_overlap_count": sum(
                    int(row["training_overlap_count"])
                    for row in membership_audits
                ),
                "training_overlap_rate": sum(
                    int(row["training_overlap_count"])
                    for row in membership_audits
                )
                / expected_object_count,
                "all_evaluation_objects_in_checkpoint_training": all(
                    row.get("all_evaluation_objects_in_checkpoint_training") is True
                    for row in membership_audits
                ),
                "all_evaluation_objects_disjoint_from_checkpoint_training": all(
                    row.get("all_evaluation_objects_disjoint_from_checkpoint_training")
                    is True
                    for row in membership_audits
                ),
                "passed": True,
            }
        ),
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
    unit_scope: str = "heldout_development",
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
            "the same complete registered checkpoint-training-overlap objects; "
            "every object value is the mean of seeds 42/43/44"
            if unit_scope == "checkpoint_training_overlap"
            else (
                "the same complete held-out development objects; every object value is "
                "the mean of seeds 42/43/44"
            )
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
    evaluation_membership_scope = str(
        getattr(args, "evaluation_membership_scope", "heldout")
    )
    comparison_unit_scope = (
        "checkpoint_training_overlap"
        if evaluation_membership_scope == "checkpoint_training_overlap"
        else "heldout_development"
    )
    contract = _load_contract(args)
    all_dev_uids = [str(row["uid"]) for row in contract["rows"]]
    object_start = int(getattr(args, "object_start", 16))
    object_end = int(getattr(args, "object_end", 64))
    expected_ss_step = int(getattr(args, "expected_native_ss_step", 2000))
    if not 0 <= object_start < object_end <= len(all_dev_uids):
        raise ValueError("invalid aggregate Dev object range")
    heldout_uids = all_dev_uids[object_start:object_end]
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
        raise RuntimeError(
            "strict ReconViaGen reports do not cover registered Dev range x 3"
        )

    current = _load_current_branch_reports(
        parse_csv(args.current_reports, str),
        all_dev_uids=all_dev_uids,
        expected_step=int(args.expected_current_step),
        expected_sha256=str(args.expected_current_sha256),
        expected_protocol_sha256=str(contract["protocol_sha256"]),
        object_start=object_start,
        object_end=object_end,
        expected_ss_step=expected_ss_step,
        allow_native_ss_science_failed=bool(
            getattr(args, "allow_native_ss_science_failed", False)
        ),
        require_all_disjoint_membership=bool(
            getattr(args, "require_all_disjoint_membership", False)
        ),
        evaluation_membership_scope=evaluation_membership_scope,
    )
    branch_records = current["branch_records"]
    require_strict_origin = bool(
        getattr(args, "require_exact_strict_target_sha256", False)
    )
    require_exact_target_sha = require_strict_origin or bool(
        getattr(args, "require_exact_shared_target_sha256", False)
    )
    if require_strict_origin and current["shared_identity"].get(
        "target_mesh_policy"
    ) != "exact_npz_from_frozen_strict_reconviagen_reports":
        raise RuntimeError(
            "current workers did not use the frozen strict target Mesh policy"
        )

    # All four routes must be scored against the exact same target structure.
    for key in sorted(heldout_pairs):
        target = recon_records[key].get("target_structure")
        if not isinstance(target, dict):
            raise RuntimeError(f"strict ReconViaGen target structure missing={key}")
        for branch in CURRENT_BRANCHES:
            current_row = branch_records[branch][key]
            if current_row.get("target_structure") != target:
                raise RuntimeError(
                    f"strict/current target Mesh structure differs={branch}:{key}"
                )
            if require_exact_target_sha and current_row.get(
                "target_mesh_sha256"
            ) != recon_records[key].get("target_mesh_sha256"):
                raise RuntimeError(
                    f"strict/current target Mesh SHA256 differs={branch}:{key}"
                )

    route_b = _route_b(expected_ss_step)
    route_c = _route_c(expected_ss_step, args.expected_current_step)
    records_by_route = {
        ROUTE_R: recon_records,
        ROUTE_A: branch_records["stock"],
        route_b: branch_records["native"],
        route_c: branch_records["native_trained"],
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
    comparisons = {
        "current_ss_vs_stock_ss_same_current_interface": paired_route_summary(
            means[route_b],
            means[ROUTE_A],
            candidate_name=route_b,
            baseline_name=ROUTE_A,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=2026081600,
            comparison_kind="B_minus_A_native_ss_component_ablation",
            clean_component_isolation=True,
            caveat=(
                "The input, noise, Stock SLat, decoder, target, and metric are "
                "fixed; only Stock SS is replaced by official Native-SS "
                f"step{expected_ss_step}."
            ),
            unit_scope=comparison_unit_scope,
        ),
        "current_slat_vs_stock_slat_same_current_ss": paired_route_summary(
            means[route_c],
            means[route_b],
            candidate_name=route_c,
            baseline_name=route_b,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=2026081700,
            comparison_kind="C_minus_B_native_slat_component_ablation",
            clean_component_isolation=True,
            caveat=(
                "The official Native-SS support, posed-DINO condition, "
                "coordinate-keyed noise, decoder, target, and metric are fixed; "
                f"only Stock SLat is replaced by Native-SLat step{int(args.expected_current_step)}."
            ),
            unit_scope=comparison_unit_scope,
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
            unit_scope=comparison_unit_scope,
        ),
        "current_ss_stock_slat_vs_strict_reconviagen": paired_route_summary(
            means[route_b],
            means[ROUTE_R],
            candidate_name=route_b,
            baseline_name=ROUTE_R,
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=2026081900,
            comparison_kind="B_minus_R_diagnostic",
            clean_component_isolation=False,
            caveat=(
                "Diagnostic only: both the input interface and SS change, so this "
                "is not a pure Native-SS effect. Use B-A for the SS attribution."
            ),
            unit_scope=comparison_unit_scope,
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
            unit_scope=comparison_unit_scope,
        ),
    }

    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "runtime_integrity_passed": True,
        "formal": False,
        "post_selection_development_diagnostic": (
            evaluation_membership_scope == "heldout"
        ),
        "split": {
            "name": "official_proobjaverse_dev",
            "object_start": object_start,
            "object_end": object_end,
            "requested_object_count": len(heldout_uids),
            "seeds": list(SEEDS),
            "train_evaluated": False,
            "gt_support_slat_evaluated": False,
            "reason": (
                "The legacy 2K Dev[16:64) objects are all members of this "
                "checkpoint's 30K training UID set; this is a compatibility and "
                "training-overlap diagnostic, not held-out generalization."
                if evaluation_membership_scope
                == "checkpoint_training_overlap"
                else (
                    "Native-SS CFG was frozen on the disjoint protocol bridge split; "
                    f"this report uses held-out Dev[{object_start}:{object_end})."
                )
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
            route_b: {
                "label": "B",
                "pipeline": (
                    "posed-DINO current interface -> official Native-SS "
                    f"step{expected_ss_step} "
                    "EMA -> Stock SLat -> Stock Mesh decoder"
                ),
            },
            route_c: {
                "label": "C",
                "pipeline": (
                    "posed-DINO current interface -> official Native-SS "
                    f"step{expected_ss_step} "
                    f"EMA -> Native-SLat step{int(args.expected_current_step)} EMA "
                    "-> Stock Mesh decoder"
                ),
            },
        },
        "comparability_guard": {
            "same_official_object_seed_pairs": True,
            "same_target_mesh_structure": True,
            "same_target_mesh_sha256": require_exact_target_sha,
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
        "current_shard_local_identity_bindings": current[
            "shard_local_identity_bindings"
        ],
        "native_ss_binding": current["native_ss_binding"],
        "native_ss_science_passed": not bool(
            current["native_ss_binding"].get("false_checks", [])
        ),
        "native_slat_binding": current["native_slat_binding"],
        "checkpoint_evaluation_membership": current[
            "checkpoint_evaluation_membership"
        ],
        "source_contract": {
            "dev_split": str(contract["split_path"]),
            "dev_split_sha256": contract["split_sha256"],
            "cache_report": str(contract["cache_report_path"]),
            "cache_report_sha256": contract["cache_report_sha256"],
            "target_report": (
                ""
                if contract["target_report_path"] is None
                else str(contract["target_report_path"])
            ),
            "target_report_sha256": contract["target_report_sha256"],
            "paired_target_cache_roots": contract["paired_target_cache_roots"],
        },
        "scope_guard": (
            "This is a legacy Dev48 checkpoint-training-overlap compatibility "
            "diagnostic against strict ReconViaGen, not a held-out/generalization "
            "claim. Surface comparisons use only the same objects with three valid "
            "seeds in every R/A/B/C route; all decoder failures and per-route "
            "success rates remain visible."
            if evaluation_membership_scope == "checkpoint_training_overlap"
            else (
                "This is a held-out Dev64 quantitative development diagnostic. It "
                "does not evaluate Train64 or GT-support SLat. Surface comparisons "
                "use only the same objects with three valid seeds in every R/A/B/C "
                "route; all decoder failures and per-route success rates remain "
                "visible."
            )
        ),
    }
    report["report_sha256"] = canonical_sha256(report)

    lines = [
        (
            "Legacy ProObjaverse Dev48 training-overlap: SS2k/SLat vs strict ReconViaGen"
            if evaluation_membership_scope == "checkpoint_training_overlap"
            else (
                "Official ProObjaverse held-out Dev64: "
                f"SS{expected_ss_step}/SLat vs strict ReconViaGen"
            )
        ),
        "=" * 74,
        "R = strict ReconViaGen: VGGT -> Stock SS -> Stock SLat",
        "A = posed-DINO current interface -> Stock SS -> Stock SLat",
        f"B = posed-DINO -> official Native-SS step{expected_ss_step} -> Stock SLat",
        (
            f"C = posed-DINO -> official Native-SS step{expected_ss_step} -> "
            f"Native-SLat step{int(args.expected_current_step)}"
        ),
        (
            "Native-SS registered science gates: PASS"
            if not current["native_ss_binding"].get("false_checks", [])
            else (
                "Native-SS registered science gates: FAIL "
                f"{current['native_ss_binding']['false_checks']} "
                "(candidate endpoint remains evaluated without rewriting gates)"
            )
        ),
        "Train64 evaluated: no; GT-support Dev evaluated: no",
        f"common complete objects: {len(common_uids)}/{len(heldout_uids)}",
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
    parser.add_argument("--expected_native_ss_step", type=int, default=2000)
    parser.add_argument(
        "--allow_native_ss_science_failed",
        action="store_true",
        help=(
            "aggregate a pre-registered candidate even when its Native-SS "
            "held-out science gates failed; failed checks remain explicit"
        ),
    )
    parser.add_argument("--expected_current_sha256", required=True)
    parser.add_argument("--object_start", type=int, default=16)
    parser.add_argument("--object_end", type=int, default=64)
    parser.add_argument(
        "--require_all_disjoint_membership",
        action="store_true",
        help="require every current worker object to be outside checkpoint training",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument(
        "--require_exact_strict_target_sha256",
        action="store_true",
        help=(
            "require every current branch to use the exact target NPZ SHA256 "
            "recorded by the strict ReconViaGen route"
        ),
    )
    parser.add_argument(
        "--require_exact_shared_target_sha256",
        action="store_true",
        help=(
            "require strict ReconViaGen and A/B/C to consume byte-identical "
            "target NPZ files, regardless of which route materialized them first"
        ),
    )
    parser.add_argument(
        "--evaluation_membership_scope",
        choices=("heldout", "checkpoint_training_overlap"),
        default="heldout",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
