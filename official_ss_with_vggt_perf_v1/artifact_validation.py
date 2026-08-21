"""Lightweight identity checks for frozen with-VGGT SS/SLat eval artifacts.

This module intentionally imports neither Torch nor TRELLIS.  It is used by
the shell orchestration before reusing an existing report, so a file is never
accepted merely because ``report.json`` exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


VSS_AGGREGATE_FORMAT = "official_ss_with_vggt_perf_v1.eval_aggregate.v1"
ENDPOINT_VERSION = "official_ss_with_vggt_perf_v1.predicted_ss_slat_endpoint.v2"
ENDPOINT_WORKER_FORMAT = f"{ENDPOINT_VERSION}.worker"
ENDPOINT_AGGREGATE_FORMAT = f"{ENDPOINT_VERSION}.aggregate"
TARGET_JOIN_CONTRACT = {
    "version": "official_ss_with_vggt_perf_v1.official_target_join.v2",
    "shared_source_identity": "exact_official_lh_slat_sha256",
    "slat_metric_target": "raw_official_lh_slat_coords_and_features",
    "ss_flow_target": "frozen_ss_decoder_projected_coords",
    "exact_cross_target_coordinate_equality_required": False,
    "observed_cross_target_iou_must_match_frozen_ss_roundtrip_iou": True,
    "slat_runtime_target_preserved": True,
}
ENDPOINT_COMPARISONS = {
    "B_minus_A__VSS_support_increment_through_V0",
    "C_minus_B__V_SLat_increment_on_VSS_support",
    "C_minus_A__full_VSS_plus_V_endpoint_increment",
}
ENDPOINT_BRANCHES = ("stock", "native", "native_trained")
ENDPOINT_FAILURE_STAGES = {
    branch: f"{branch}_slat_mesh_decode" for branch in ENDPOINT_BRANCHES
}
RUNTIME_INTEGRITY_KEYS = (
    "correct_record_matrix_exact",
    "pose_control_record_matrix_exact",
    "stock_baseline_nonempty",
    "disabled_stock_equivalence",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {resolved}")
    return resolved, payload


def load_hashed_report(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved, payload = _load_object(path)
    body = dict(payload)
    saved = str(body.pop("report_sha256", ""))
    if not saved or canonical_sha256(body) != saved:
        raise RuntimeError(f"report identity/hash mismatch: {resolved}")
    return resolved, payload


def _resolved_string(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=True))


def _parse_seeds(values: str | Iterable[int]) -> list[int]:
    if isinstance(values, str):
        seeds = [int(item) for item in values.split(",") if item]
    else:
        seeds = [int(item) for item in values]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("joint seeds must be non-empty and unique")
    return seeds


def _pair_id(object_uid: str, seed: int) -> str:
    return hashlib.sha256(f"{object_uid}|{int(seed)}".encode("utf-8")).hexdigest()[:24]


def _registered_mesh_failure(row: dict[str, Any]) -> bool:
    branch = str(row.get("branch", ""))
    error = row.get("error")
    if not isinstance(error, dict):
        return False
    message = str(error.get("message", ""))
    return bool(
        row.get("passed") is False
        and branch in ENDPOINT_FAILURE_STAGES
        and error.get("type") == "RuntimeError"
        and error.get("stage") == ENDPOINT_FAILURE_STAGES[branch]
        and (
            message.startswith("FlexiCubes topology index is inconsistent:")
            or message.startswith(
                "SLat decoder input exceeds safe active-point limit:"
            )
            or message.startswith("SLat decoder CUDA topology failure:")
            or (message.startswith("decoded ") and " Mesh is invalid:" in message)
        )
    )


def _validate_endpoint_worker_outcomes(
    payload: dict[str, Any],
    *,
    object_uids: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    """Validate a complete outcome matrix without hiding model failures.

    A registered decoder/topology failure is a scientific model outcome, not a
    corrupt report.  Every other missing, duplicate, or failed row remains a
    hard program-integrity error.
    """

    if len(object_uids) != len(set(object_uids)):
        raise RuntimeError("endpoint worker object UIDs are not unique")
    expected_pairs = {(uid, seed) for uid in object_uids for seed in seeds}
    ss_records = payload.get("ss_records")
    if not isinstance(ss_records, list) or len(ss_records) != len(expected_pairs):
        raise RuntimeError("endpoint worker SS record matrix differs")
    observed_ss: set[tuple[str, int]] = set()
    for row in ss_records:
        if not isinstance(row, dict):
            raise RuntimeError("endpoint worker SS record is not an object")
        key = (str(row.get("object_uid", "")), int(row.get("seed", -1)))
        if (
            key not in expected_pairs
            or key in observed_ss
            or row.get("passed") is not True
            or row.get("same_initial_noise") is not True
        ):
            raise RuntimeError(f"endpoint worker SS outcome differs: {key}")
        observed_ss.add(key)
    if observed_ss != expected_pairs:
        raise RuntimeError("endpoint worker SS key coverage differs")

    expected_mesh = {
        (uid, seed, branch)
        for uid, seed in expected_pairs
        for branch in ENDPOINT_BRANCHES
    }
    mesh_records = payload.get("mesh_branch_records")
    if not isinstance(mesh_records, list) or len(mesh_records) != len(expected_mesh):
        raise RuntimeError("endpoint worker Mesh record matrix differs")
    observed_mesh: set[tuple[str, int, str]] = set()
    failures: list[dict[str, Any]] = []
    for row in mesh_records:
        if not isinstance(row, dict):
            raise RuntimeError("endpoint worker Mesh record is not an object")
        key = (
            str(row.get("object_uid", "")),
            int(row.get("seed", -1)),
            str(row.get("branch", "")),
        )
        if (
            key not in expected_mesh
            or key in observed_mesh
            or str(row.get("pair_id", "")) != _pair_id(key[0], key[1])
        ):
            raise RuntimeError(f"endpoint worker Mesh identity differs: {key}")
        if row.get("passed") is True:
            if (
                not isinstance(row.get("surface"), dict)
                or not isinstance(row.get("structure"), dict)
                or row["structure"].get("mesh_success") is not True
            ):
                raise RuntimeError(f"endpoint successful Mesh payload differs: {key}")
        elif _registered_mesh_failure(row):
            failures.append(
                {
                    "object_uid": key[0],
                    "seed": key[1],
                    "branch": key[2],
                    "error": dict(row["error"]),
                }
            )
        else:
            raise RuntimeError(f"endpoint worker has unregistered failure: {key}")
        observed_mesh.add(key)
    if observed_mesh != expected_mesh:
        raise RuntimeError("endpoint worker Mesh key coverage differs")
    if payload.get("passed") is not (len(failures) == 0):
        raise RuntimeError("endpoint worker outcome flag is inconsistent")
    return {
        "program_integrity_passed": True,
        "all_model_outputs_passed": not failures,
        "registered_model_output_failure_count": len(failures),
        "registered_model_output_failures": failures,
    }


def _validate_endpoint_contract(payload: dict[str, Any]) -> None:
    contract = payload.get("with_vggt_endpoint_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("endpoint report lacks its frozen branch contract")
    if (
        contract.get("version") != ENDPOINT_VERSION
        or contract.get("slat_support_input") != "predicted_only"
        or contract.get("gt_support_used_as_slat_input") is not False
        or set(contract.get("branches", {})) != {"A", "B", "C"}
        or set(contract.get("comparisons", {}))
        != {"B_minus_A", "C_minus_B", "C_minus_A"}
        or contract.get("official_target_join_contract") != TARGET_JOIN_CONTRACT
    ):
        raise RuntimeError("endpoint branch/support contract differs")


def validate_endpoint_worker(
    report_path: str | Path,
    *,
    expected_split: str,
    expected_start: int,
    expected_end: int,
    expected_seeds: str | Iterable[int],
    expected_slat_step: int,
    expected_slat_checkpoint: str | Path,
    expected_ss_cache: str | Path,
    expected_vss_report: str | Path,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    path, payload = load_hashed_report(report_path)
    seeds = _parse_seeds(expected_seeds)
    start, end = int(expected_start), int(expected_end)
    if start < 0 or end <= start:
        raise ValueError(f"invalid expected worker range [{start},{end})")
    if (
        payload.get("format") != ENDPOINT_WORKER_FORMAT
        or payload.get("complete") is not True
        or payload.get("formal") is not False
        or payload.get("evaluation_split") != expected_split
        or int(payload.get("object_count", -1)) != end - start
        or int(payload.get("record_count", -1)) != (end - start) * len(seeds)
    ):
        raise RuntimeError(f"endpoint worker scope/format differs: {path}")
    _validate_endpoint_contract(payload)
    identity = payload.get("run_identity")
    if not isinstance(identity, dict):
        raise RuntimeError(f"endpoint worker lacks run identity: {path}")
    slat_checkpoint = _resolved_string(expected_slat_checkpoint)
    ss_cache = _resolved_string(expected_ss_cache)
    vss_report = _resolved_string(expected_vss_report)
    hashes = expected_hashes or {
        "slat_checkpoint": sha256_file(slat_checkpoint),
        "ss_cache": sha256_file(ss_cache),
        "vss_report": sha256_file(vss_report),
    }
    expected_identity = {
        "format": ENDPOINT_WORKER_FORMAT,
        "object_start": start,
        "object_end": end,
        "joint_seeds": seeds,
        "expected_trained_slat_step": int(expected_slat_step),
        "trained_slat_checkpoint": slat_checkpoint,
        "trained_slat_checkpoint_sha256": hashes["slat_checkpoint"],
        "ss_cache_manifest": ss_cache,
        "ss_cache_manifest_sha256": hashes["ss_cache"],
        "native_ss_report": vss_report,
        "native_ss_report_sha256": hashes["vss_report"],
        "slat_support_input": "predicted_only",
        "gt_support_used_as_slat_input": False,
    }
    mismatches = {
        key: {"observed": identity.get(key), "expected": value}
        for key, value in expected_identity.items()
        if identity.get(key) != value
    }
    object_uids = identity.get("object_uids")
    if not isinstance(object_uids, list) or len(object_uids) != end - start:
        mismatches["object_uids"] = {
            "observed_count": len(object_uids) if isinstance(object_uids, list) else None,
            "expected_count": end - start,
        }
    if mismatches:
        raise RuntimeError(f"endpoint worker identity differs: {mismatches}")
    outcome = _validate_endpoint_worker_outcomes(
        payload,
        object_uids=[str(value) for value in object_uids],
        seeds=seeds,
    )
    return {
        "report": str(path),
        "runtime_passed": outcome["program_integrity_passed"],
        "all_model_outputs_passed": outcome["all_model_outputs_passed"],
        "registered_model_output_failure_count": outcome[
            "registered_model_output_failure_count"
        ],
        "registered_model_output_failures": outcome[
            "registered_model_output_failures"
        ],
        "object_start": start,
        "object_end": end,
        "object_uids": [str(value) for value in object_uids],
        "report_sha256": str(payload["report_sha256"]),
    }


def validate_endpoint_aggregate(
    report_path: str | Path,
    *,
    expected_split: str,
    expected_start: int,
    expected_end: int,
    expected_objects: int,
    expected_seeds: str | Iterable[int],
    expected_slat_step: int,
    expected_slat_checkpoint: str | Path,
    expected_ss_cache: str | Path,
    expected_vss_report: str | Path,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    path, payload = load_hashed_report(report_path)
    seeds = _parse_seeds(expected_seeds)
    start, end, objects = (
        int(expected_start),
        int(expected_end),
        int(expected_objects),
    )
    if end - start != objects:
        raise ValueError("expected aggregate range/object count differs")
    if (
        payload.get("format") != ENDPOINT_AGGREGATE_FORMAT
        or not isinstance(payload.get("passed"), bool)
        or payload.get("formal") is not False
        or payload.get("evaluation_split") != expected_split
        or int(payload.get("object_start", -1)) != start
        or int(payload.get("object_end", -1)) != end
        or int(payload.get("object_count", -1)) != objects
        or int(payload.get("record_count", -1)) != objects * len(seeds)
        or payload.get("joint_seeds") != seeds
    ):
        raise RuntimeError(f"endpoint aggregate scope/runtime differs: {path}")
    _validate_endpoint_contract(payload)
    integrity = payload.get("integrity")
    if not isinstance(integrity, dict) or any(
        integrity.get(key) is not True
        for key in ("object_coverage_exact", "ss_records_exact")
    ):
        raise RuntimeError("endpoint aggregate runtime-integrity gate failed")
    endpoint_integrity = payload.get("endpoint_runtime_integrity")
    if (
        not isinstance(endpoint_integrity, dict)
        or endpoint_integrity.get("passed") is not True
        or endpoint_integrity.get("complete_branch_record_matrix") is not True
        or endpoint_integrity.get("only_registered_model_output_failures") is not True
    ):
        raise RuntimeError("endpoint aggregate program-integrity gate failed")
    if set(payload.get("comparisons", {})) != ENDPOINT_COMPARISONS or set(
        payload.get("extended_pairwise_metrics", {})
    ) != ENDPOINT_COMPARISONS:
        raise RuntimeError("endpoint aggregate comparison matrix differs")
    decision = payload.get("decision")
    if not isinstance(decision, dict) or not isinstance(
        decision.get("native_ss_trained_slat_end_to_end_passed"), bool
    ):
        raise RuntimeError("endpoint aggregate lacks the registered C-A science gate")

    hashes = expected_hashes or {
        "slat_checkpoint": sha256_file(expected_slat_checkpoint),
        "ss_cache": sha256_file(expected_ss_cache),
        "vss_report": sha256_file(expected_vss_report),
    }
    worker_bindings = payload.get("worker_reports")
    if not isinstance(worker_bindings, list) or not worker_bindings:
        raise RuntimeError("endpoint aggregate lacks worker report bindings")
    ranges: list[tuple[int, int]] = []
    observed_uids: list[str] = []
    registered_failures: list[dict[str, Any]] = []
    for binding in worker_bindings:
        if not isinstance(binding, dict):
            raise RuntimeError("endpoint aggregate worker binding is not an object")
        worker_path = Path(str(binding.get("path", ""))).expanduser().resolve(strict=True)
        if sha256_file(worker_path) != str(binding.get("sha256", "")):
            raise RuntimeError(f"endpoint worker file hash differs: {worker_path}")
        _, worker_payload = load_hashed_report(worker_path)
        if worker_payload["report_sha256"] != binding.get("report_sha256"):
            raise RuntimeError(f"endpoint worker report hash binding differs: {worker_path}")
        identity = worker_payload.get("run_identity", {})
        worker_start = int(identity.get("object_start", -1))
        worker_end = int(identity.get("object_end", -1))
        row = validate_endpoint_worker(
            worker_path,
            expected_split=expected_split,
            expected_start=worker_start,
            expected_end=worker_end,
            expected_seeds=seeds,
            expected_slat_step=expected_slat_step,
            expected_slat_checkpoint=expected_slat_checkpoint,
            expected_ss_cache=expected_ss_cache,
            expected_vss_report=expected_vss_report,
            expected_hashes=hashes,
        )
        ranges.append((worker_start, worker_end))
        observed_uids.extend(row["object_uids"])
        registered_failures.extend(row["registered_model_output_failures"])
    cursor = start
    for worker_start, worker_end in sorted(ranges):
        if worker_start != cursor:
            raise RuntimeError(f"endpoint worker ranges have a gap/overlap at {cursor}")
        cursor = worker_end
    if (
        cursor != end
        or len(observed_uids) != objects
        or len(set(observed_uids)) != objects
    ):
        raise RuntimeError("endpoint worker object coverage differs")
    failure_count = len(registered_failures)
    if (
        int(endpoint_integrity.get("registered_model_output_failure_count", -1))
        != failure_count
        or endpoint_integrity.get("all_model_outputs_passed")
        is not (failure_count == 0)
        or payload.get("passed") is not (failure_count == 0)
        or integrity.get("mesh_pairs_exact") is not (failure_count == 0)
    ):
        raise RuntimeError("endpoint aggregate model-outcome accounting differs")
    return {
        "report": str(path),
        "runtime_integrity_passed": True,
        "full_endpoint_science_passed": bool(
            decision["native_ss_trained_slat_end_to_end_passed"]
        ),
        "objects": objects,
        "records": objects * len(seeds),
        "all_model_outputs_passed": failure_count == 0,
        "registered_model_output_failure_count": failure_count,
        "registered_model_output_failures": registered_failures,
    }


def validate_vss_aggregate(
    report_path: str | Path,
    *,
    expected_checkpoint: str | Path,
    expected_calibration: str | Path,
    expected_objects: int = 48,
    expected_seeds: str | Iterable[int] = "42,43,44",
    expected_step: int = 2000,
) -> dict[str, Any]:
    path, payload = load_hashed_report(report_path)
    seeds = _parse_seeds(expected_seeds)
    objects = int(expected_objects)
    if (
        payload.get("format") != VSS_AGGREGATE_FORMAT
        or payload.get("formal") is not False
        or int(payload.get("object_count", -1)) != objects
        or int(payload.get("record_count", -1)) != objects * len(seeds)
    ):
        raise RuntimeError(f"VSS aggregate scope/format differs: {path}")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or any(
        checks.get(key) is not True for key in RUNTIME_INTEGRITY_KEYS
    ):
        raise RuntimeError("VSS aggregate runtime-integrity gate failed")
    if (
        payload.get("calibration_sha256") != sha256_file(expected_calibration)
        or payload.get("protocol", {}).get("joint_seeds") != seeds
    ):
        raise RuntimeError("VSS calibration/seed binding differs")
    deployment = payload.get("deployment")
    if not isinstance(deployment, dict):
        raise RuntimeError("VSS aggregate lacks deployment binding")
    checkpoint = _resolved_string(expected_checkpoint)
    if (
        deployment.get("checkpoint") != checkpoint
        or deployment.get("checkpoint_sha256") != sha256_file(checkpoint)
        or int(deployment.get("checkpoint_step", -1)) != int(expected_step)
        or deployment.get("weights") != "ema"
    ):
        raise RuntimeError("VSS checkpoint/weights deployment binding differs")
    return {
        "report": str(path),
        "runtime_integrity_passed": True,
        "science_passed": payload.get("passed") is True,
        "objects": objects,
        "records": objects * len(seeds),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("endpoint-worker")
    aggregate = subparsers.add_parser("endpoint-aggregate")
    for child in (worker, aggregate):
        child.add_argument("--report", required=True)
        child.add_argument("--split", choices=("train", "dev"), required=True)
        child.add_argument("--object-start", type=int, required=True)
        child.add_argument("--object-end", type=int, required=True)
        child.add_argument("--joint-seeds", default="42,43,44")
        child.add_argument("--slat-step", type=int, required=True)
        child.add_argument("--slat-checkpoint", required=True)
        child.add_argument("--ss-cache", required=True)
        child.add_argument("--vss-report", required=True)
        child.add_argument("--slat-checkpoint-sha256", default="")
        child.add_argument("--ss-cache-sha256", default="")
        child.add_argument("--vss-report-sha256", default="")
    aggregate.add_argument("--expected-objects", type=int, required=True)
    vss = subparsers.add_parser("vss-aggregate")
    vss.add_argument("--report", required=True)
    vss.add_argument("--checkpoint", required=True)
    vss.add_argument("--calibration", required=True)
    vss.add_argument("--expected-objects", type=int, default=48)
    vss.add_argument("--joint-seeds", default="42,43,44")
    vss.add_argument("--checkpoint-step", type=int, default=2000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    expected_hashes = None
    if args.command in {"endpoint-worker", "endpoint-aggregate"}:
        supplied = (
            args.slat_checkpoint_sha256,
            args.ss_cache_sha256,
            args.vss_report_sha256,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("endpoint expected hashes must be supplied together")
        if all(supplied):
            expected_hashes = {
                "slat_checkpoint": args.slat_checkpoint_sha256,
                "ss_cache": args.ss_cache_sha256,
                "vss_report": args.vss_report_sha256,
            }
    if args.command == "endpoint-worker":
        result = validate_endpoint_worker(
            args.report,
            expected_split=args.split,
            expected_start=args.object_start,
            expected_end=args.object_end,
            expected_seeds=args.joint_seeds,
            expected_slat_step=args.slat_step,
            expected_slat_checkpoint=args.slat_checkpoint,
            expected_ss_cache=args.ss_cache,
            expected_vss_report=args.vss_report,
            expected_hashes=expected_hashes,
        )
    elif args.command == "endpoint-aggregate":
        result = validate_endpoint_aggregate(
            args.report,
            expected_split=args.split,
            expected_start=args.object_start,
            expected_end=args.object_end,
            expected_objects=args.expected_objects,
            expected_seeds=args.joint_seeds,
            expected_slat_step=args.slat_step,
            expected_slat_checkpoint=args.slat_checkpoint,
            expected_ss_cache=args.ss_cache,
            expected_vss_report=args.vss_report,
            expected_hashes=expected_hashes,
        )
    else:
        result = validate_vss_aggregate(
            args.report,
            expected_checkpoint=args.checkpoint,
            expected_calibration=args.calibration,
            expected_objects=args.expected_objects,
            expected_seeds=args.joint_seeds,
            expected_step=args.checkpoint_step,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
