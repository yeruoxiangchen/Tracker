"""Run predicted-support Train64/Dev48 tests for the with-VGGT VSS + V endpoint.

The implementation composes the existing official three-branch evaluator.
There is intentionally no GT-support SLat prediction branch: official GT is
decoded only as the metric target.
"""

from __future__ import annotations

import copy
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any

from pose_point_depth_mv import (
    evaluate_proobjaverse_official_native_ss_stock_slat as _base,
)
from pose_point_depth_mv.native_ss_genrecon import sha256_file
from pose_point_depth_mv.proobjaverse_official_slat_protocol import canonical_sha256

from .artifact_validation import _validate_endpoint_worker_outcomes
from .ss_slat_endpoint import (
    REPORT_FORMAT,
    WORKER_FORMAT,
    WithVGGTSSSLatEndpointDataset,
    activate_ss_cache_manifest,
    build_trained_slat_pipeline,
    endpoint_contract,
    load_ss_runtime,
    official_target_contract,
)


_BASE_WORKER_PARSER = _base._worker_parser
_BASE_AGGREGATE_PARSER = _base._aggregate_parser
_BASE_RUN_WORKER = _base.run_worker
_BASE_RUN_AGGREGATE = _base.run_aggregate
_CONFIGURED = False

_SURFACE_DISTANCE_KEYS = (
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
)
_SURFACE_SCORE_KEYS = (
    "precision_0p01",
    "precision_0p02",
    "precision_0p05",
    "recall_0p01",
    "recall_0p02",
    "recall_0p05",
    "fscore_0p01",
    "fscore_0p02",
    "fscore_0p05",
    "normal_consistency",
)


def _write_hashed_json(path: Path, payload: dict[str, Any]) -> None:
    value = copy.deepcopy(payload)
    value.pop("report_sha256", None)
    value["report_sha256"] = canonical_sha256(value)
    temporary = path.with_name(f".{path.name}.endpoint-{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _verify_report_hash(payload: dict[str, Any], *, path: Path) -> None:
    body = dict(payload)
    saved = str(body.pop("report_sha256", ""))
    if not saved or canonical_sha256(body) != saved:
        raise RuntimeError(f"report hash mismatch before endpoint annotation: {path}")


def _extended_pairwise_metrics(
    records: list[dict[str, Any]],
    *,
    baseline: str,
    candidate: str,
    expected_uids: set[str],
    seeds: list[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Summarize all already-computed surface fields at object level."""

    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicates: list[dict[str, Any]] = []
    for row in records:
        branch = str(row.get("branch", ""))
        if branch not in {baseline, candidate}:
            continue
        key = (str(row.get("object_uid", "")), int(row.get("seed", -1)))
        if branch in by_pair[key]:
            duplicates.append({"object_uid": key[0], "seed": key[1], "branch": branch})
        by_pair[key][branch] = row
    expected_pairs = {(uid, seed) for uid in expected_uids for seed in seeds}
    invalid_pairs: list[dict[str, Any]] = list(duplicates)
    deltas: dict[tuple[str, int], dict[str, float]] = {}
    for key in sorted(expected_pairs):
        branches = by_pair.get(key, {})
        if set(branches) != {baseline, candidate} or any(
            branches[name].get("passed") is not True for name in branches
        ):
            invalid_pairs.append(
                {"object_uid": key[0], "seed": key[1], "error": "missing/failed branch"}
            )
            continue
        base_surface = branches[baseline]["surface"]
        candidate_surface = branches[candidate]["surface"]
        row: dict[str, float] = {}
        for name in _SURFACE_DISTANCE_KEYS:
            row[f"{name}_improvement"] = float(base_surface[name]) - float(
                candidate_surface[name]
            )
        for name in _SURFACE_SCORE_KEYS:
            row[f"{name}_delta"] = float(candidate_surface[name]) - float(
                base_surface[name]
            )
        row["largest_component_ratio_delta"] = float(
            branches[candidate]["structure"]["largest_component_ratio"]
        ) - float(branches[baseline]["structure"]["largest_component_ratio"])
        deltas[key] = row
    by_object: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (uid, _seed), row in deltas.items():
        by_object[uid].append(row)
    metric_names = tuple(
        [f"{name}_improvement" for name in _SURFACE_DISTANCE_KEYS]
        + [f"{name}_delta" for name in _SURFACE_SCORE_KEYS]
        + ["largest_component_ratio_delta"]
    )
    complete_rows: list[dict[str, Any]] = []
    excluded_objects: list[str] = []
    for uid in sorted(expected_uids):
        rows = by_object.get(uid, [])
        if len(rows) != len(seeds):
            excluded_objects.append(uid)
            continue
        complete_rows.append(
            {
                "object_uid": uid,
                **{
                    name: sum(row[name] for row in rows) / len(rows)
                    for name in metric_names
                },
            }
        )
    summary = {
        name: _base.summary_with_ci(
            [float(row[name]) for row in complete_rows],
            samples=int(bootstrap_samples),
            seed=int(bootstrap_seed) + position,
        )
        for position, name in enumerate(metric_names)
    }
    return {
        "baseline_branch": baseline,
        "candidate_branch": candidate,
        "expected_pair_count": len(expected_pairs),
        "valid_pair_count": len(deltas),
        "complete_object_count": len(complete_rows),
        "excluded_incomplete_objects": excluded_objects,
        "invalid_pairs": invalid_pairs,
        "summary": summary,
        "object_rows": complete_rows,
    }


def build_extended_comparisons(
    reports: list[dict[str, Any]],
    *,
    expected_uids: set[str],
    seeds: list[int],
    bootstrap_samples: int,
) -> dict[str, Any]:
    records = [
        row
        for report in reports
        for row in report.get("mesh_branch_records", [])
    ]
    return {
        "B_minus_A__VSS_support_increment_through_V0": _extended_pairwise_metrics(
            records,
            baseline="stock",
            candidate="native",
            expected_uids=expected_uids,
            seeds=seeds,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=20261701,
        ),
        "C_minus_B__V_SLat_increment_on_VSS_support": _extended_pairwise_metrics(
            records,
            baseline="native",
            candidate="native_trained",
            expected_uids=expected_uids,
            seeds=seeds,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=20261801,
        ),
        "C_minus_A__full_VSS_plus_V_endpoint_increment": _extended_pairwise_metrics(
            records,
            baseline="stock",
            candidate="native_trained",
            expected_uids=expected_uids,
            seeds=seeds,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=20261901,
        ),
    }


def _annotate_worker(args: Any) -> None:
    path = Path(args.output_dir).expanduser().resolve() / "report.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    _verify_report_hash(payload, path=path)
    contract = endpoint_contract()
    existing = payload.get("with_vggt_endpoint_contract")
    if existing is not None:
        if existing != contract:
            raise RuntimeError("existing worker endpoint contract differs")
        return
    identity = dict(payload["run_identity"])
    identity.update(
        {
            "ss_cache_manifest": str(
                Path(args.ss_cache_manifest).expanduser().resolve(strict=True)
            ),
            "ss_cache_manifest_sha256": sha256_file(args.ss_cache_manifest),
            "endpoint_version": contract["version"],
            "branch_semantics": copy.deepcopy(contract["branches"]),
            "slat_support_input": "predicted_only",
            "gt_support_used_as_slat_input": False,
        }
    )
    payload["run_identity"] = identity
    payload["with_vggt_endpoint_contract"] = contract
    split = WithVGGTSSSLatEndpointDataset(
        args.cache_manifest,
        args.lifting_cache_manifest,
        ss_cache_manifest=args.ss_cache_manifest,
    ).config["target_source"]["split"]
    payload["evaluation_split"] = split
    payload["scope_guard"] = (
        "Train64 training-overlap fit diagnosis; all SLat predictions use VSS0/VSS "
        "predicted support and official GT is metric target only"
        if split == "train"
        else "held-out Dev48 development diagnosis excluding CFG-calibration Dev[0:16); "
        "all SLat predictions use VSS0/VSS predicted support"
    )
    _write_hashed_json(path, payload)


def _annotate_aggregate(args: Any) -> None:
    root = Path(args.output_dir).expanduser().resolve()
    path = root / "report.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    _verify_report_hash(payload, path=path)
    contract = endpoint_contract()
    existing = payload.get("with_vggt_endpoint_contract")
    if existing is not None:
        if existing != contract:
            raise RuntimeError("existing aggregate endpoint contract differs")
        return
    dataset = WithVGGTSSSLatEndpointDataset(
        args.cache_manifest,
        args.lifting_cache_manifest,
        ss_cache_manifest=args.ss_cache_manifest,
    )
    split = str(dataset.config["target_source"]["split"])
    payload["with_vggt_endpoint_contract"] = contract
    payload["evaluation_split"] = split
    payload["evaluation_role"] = (
        "training_overlap_fit_diagnosis"
        if split == "train"
        else "held_out_development_generalization"
    )
    payload["comparisons"] = {
        "B_minus_A__VSS_support_increment_through_V0": copy.deepcopy(
            payload["stock_slat_mesh_transfer"]
        ),
        "C_minus_B__V_SLat_increment_on_VSS_support": copy.deepcopy(
            payload["trained_slat_increment_on_native_support"]
        ),
        "C_minus_A__full_VSS_plus_V_endpoint_increment": copy.deepcopy(
            payload["trained_slat_end_to_end_transfer"]
        ),
    }
    worker_reports = [
        json.loads(Path(item.strip()).read_text(encoding="utf-8"))
        for item in args.shard_reports.split(",")
        if item.strip()
    ]
    object_end = (
        len(dataset) if int(args.object_end) <= 0 else int(args.object_end)
    )
    expected_uids = {
        str(row["object_uid"])
        for row in dataset.rows[int(args.object_start) : object_end]
    }
    seeds = _base.parse_csv(args.joint_seeds, int)
    worker_outcomes = [
        _validate_endpoint_worker_outcomes(
            report,
            object_uids=[
                str(value) for value in report["run_identity"]["object_uids"]
            ],
            seeds=seeds,
        )
        for report in worker_reports
    ]
    registered_failures = [
        failure
        for outcome in worker_outcomes
        for failure in outcome["registered_model_output_failures"]
    ]
    payload["endpoint_runtime_integrity"] = {
        "passed": True,
        "complete_branch_record_matrix": True,
        "only_registered_model_output_failures": True,
        "all_model_outputs_passed": not registered_failures,
        "registered_model_output_failure_count": len(registered_failures),
        "registered_model_output_failures": registered_failures,
        "interpretation": (
            "Program/report integrity passed. Registered decoder/topology failures "
            "remain scientific model outcomes and are not relabeled successful."
        ),
    }
    payload["extended_pairwise_metrics"] = build_extended_comparisons(
        worker_reports,
        expected_uids=expected_uids,
        seeds=seeds,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    payload["scope_guard"] = (
        "Train64 is a training-overlap fit diagnosis, not a generalization claim; "
        "GT support is never a SLat inference input"
        if split == "train"
        else "Dev48 is object-disjoint and excludes the 16 CFG-calibration objects; "
        "it is a development result, not a final untouched claim"
    )
    _write_hashed_json(path, payload)
    summary = root / "summary.txt"
    previous = summary.read_text(encoding="utf-8") if summary.is_file() else ""
    addition = [
        "",
        "with-VGGT predicted-support endpoint branch map",
        "================================================",
        f"split: {split}",
        "A: VSS0 predicted support -> V0 Stock SLat",
        "B: VSS predicted support -> V0 Stock SLat",
        "C: VSS predicted support -> V trained SLat",
        "B-A: SS increment; C-B: SLat increment; C-A: full endpoint increment",
        "SLat input support: predicted only (official GT is metric target only)",
        payload["scope_guard"],
    ]
    summary_temporary = summary.with_name(
        f".{summary.name}.endpoint-{os.getpid()}.tmp"
    )
    summary_temporary.write_text(
        previous.rstrip() + "\n" + "\n".join(addition) + "\n",
        encoding="utf-8",
    )
    os.replace(summary_temporary, summary)


def _worker_parser(subparsers: Any) -> None:
    _BASE_WORKER_PARSER(subparsers)
    subparsers.choices["worker"].add_argument("--ss_cache_manifest", required=True)


def _aggregate_parser(subparsers: Any) -> None:
    _BASE_AGGREGATE_PARSER(subparsers)
    subparsers.choices["aggregate"].add_argument(
        "--ss_cache_manifest", required=True
    )


def _run_worker(args: Any) -> None:
    activate_ss_cache_manifest(args.ss_cache_manifest)
    # The base worker uses exit code 2 for a *completed* report containing a
    # registered model-output failure.  Annotation is part of this endpoint's
    # immutable identity and must run for both science-positive and
    # science-negative completed reports.  Program/runtime failures still
    # propagate without a report and therefore cannot be mislabeled complete.
    exit_code: int | None = None
    try:
        _BASE_RUN_WORKER(args)
    except SystemExit as error:
        exit_code = int(error.code or 0)
    _annotate_worker(args)
    if exit_code is not None:
        raise SystemExit(exit_code)


def _run_aggregate(args: Any) -> None:
    activate_ss_cache_manifest(args.ss_cache_manifest)
    exit_code: int | None = None
    try:
        _BASE_RUN_AGGREGATE(args)
    except SystemExit as error:
        exit_code = int(error.code or 0)
    _annotate_aggregate(args)
    if exit_code is not None:
        raise SystemExit(exit_code)


def configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _base.NativeConditionSLatDataset = WithVGGTSSSLatEndpointDataset
    _base._official_target_contract = official_target_contract
    _base._load_ss_runtime = load_ss_runtime
    _base._build_trained_slat_pipeline = build_trained_slat_pipeline
    _base.WORKER_FORMAT = WORKER_FORMAT
    _base.END_TO_END_WORKER_FORMAT = WORKER_FORMAT
    _base.REPORT_FORMAT = REPORT_FORMAT
    _base.END_TO_END_REPORT_FORMAT = REPORT_FORMAT
    _base._worker_parser = _worker_parser
    _base._aggregate_parser = _aggregate_parser
    _base.run_worker = _run_worker
    _base.run_aggregate = _run_aggregate
    _CONFIGURED = True


def main() -> None:
    configure()
    _base.main()


if __name__ == "__main__":
    main()


__all__ = ["configure", "main"]
