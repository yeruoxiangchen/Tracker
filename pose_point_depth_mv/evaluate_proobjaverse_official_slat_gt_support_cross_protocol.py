#!/usr/bin/env python3
"""Run a registered cross-protocol GT-support compatibility evaluation.

This adapter keeps the official GT-support evaluator's numerical path intact.
It only permits a checkpoint/cache protocol mismatch after proving the exact
evaluation UIDs have the pre-registered relationship to the checkpoint's
frozen training UID list.  It also supports the already-audited host-path
relocation of the upstream Native-SS evidence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv import evaluate_proobjaverse_official_slat_gt_support as _base
from pose_point_depth_mv.evaluate_proobjaverse_official_slat_gt_support_cross_host import (
    validate_checkpoint_native_ss_binding_relocation,
)
from pose_point_depth_mv.native_3d_condition import NativeConditionSLatDataset
from pose_point_depth_mv.no_vggt_ss_evidence import load_no_vggt_ss_evidence
from pose_point_depth_mv.proobjaverse_official_slat_protocol import canonical_sha256
from pose_point_depth_mv.slat_checkpoint_evaluation_membership import (
    MEMBERSHIP_POLICIES,
    audit_checkpoint_evaluation_membership,
)


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("SLat checkpoint payload must be a dictionary")
    return checkpoint


def _selected_uids(args) -> tuple[str, list[str]]:
    dataset = NativeConditionSLatDataset(
        args.cache_manifest, args.lifting_cache_manifest, indices="all"
    )
    target = dict(dataset.config.get("target_source", {}))
    protocol_sha256 = str(target.get("protocol_sha256", ""))
    if not protocol_sha256:
        raise RuntimeError("evaluation cache target protocol is missing")

    unique_indices: list[int] = []
    seen: set[str] = set()
    for index, row in enumerate(dataset.rows):
        uid = str(row["object_uid"])
        if uid not in seen:
            seen.add(uid)
            unique_indices.append(index)
        if len(unique_indices) == int(args.max_objects):
            break
    start = int(args.object_start)
    end = len(unique_indices) if int(args.object_end) <= 0 else int(args.object_end)
    if start < 0 or end <= start or end > len(unique_indices):
        raise ValueError(
            f"invalid object slice [{start}:{end}] for {len(unique_indices)} objects"
        )
    return protocol_sha256, [
        str(dataset.rows[index]["object_uid"]) for index in unique_indices[start:end]
    ]


def _rewrite_report(output_dir: str | Path, membership: dict[str, Any]) -> None:
    output = Path(output_dir).expanduser().resolve()
    report_path = output / "report.json"
    config_path = output / "run_config.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_config = dict(report["run_config"])
    if list(run_config.get("object_uids", [])) == []:
        raise RuntimeError("GT-support report does not record evaluation UIDs")
    if canonical_sha256(list(run_config["object_uids"])) != membership["evaluation_uid_sha256"]:
        raise RuntimeError("GT-support runtime UIDs differ from preflight membership audit")

    run_config.update(
        {
            "checkpoint_training_protocol_sha256": membership[
                "checkpoint_protocol_sha256"
            ],
            "checkpoint_evaluation_protocol_relation": membership[
                "protocol_relation"
            ],
            "checkpoint_training_membership_policy": membership[
                "expected_membership"
            ],
            "checkpoint_training_overlap_count": membership[
                "training_overlap_count"
            ],
            "checkpoint_training_overlap_rate": membership[
                "training_overlap_rate"
            ],
            "training_overlap": membership[
                "training_overlap_count"
            ]
            > 0,
        }
    )
    report["run_config"] = run_config
    report["checkpoint_evaluation_membership"] = membership
    if membership["all_evaluation_objects_in_checkpoint_training"]:
        report["scope_guard"] = (
            "registered cross-protocol legacy compatibility and checkpoint-training-"
            "overlap diagnosis only; no held-out/generalization claim and no "
            "Native-SS predicted-support claim"
        )
    elif membership["all_evaluation_objects_disjoint_from_checkpoint_training"]:
        report["scope_guard"] = (
            "registered cross-protocol GT-support development diagnosis only; no "
            "final untouched claim and no Native-SS predicted-support claim"
        )
    else:
        report["scope_guard"] = (
            "registered cross-protocol mixed-membership diagnostic only; do not use "
            "as a held-out/generalization claim"
        )
    report.pop("report_sha256", None)
    report["report_sha256"] = canonical_sha256(report)
    config_path.write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    original_make_parser = _base.make_parser
    original_upstream_binding = _base._upstream_binding

    def make_parser():
        parser = original_make_parser()
        parser.add_argument(
            "--allow_checkpoint_data_path_relocation",
            action="store_true",
        )
        parser.add_argument(
            "--allow_checkpoint_target_protocol_mismatch",
            action="store_true",
        )
        parser.add_argument(
            "--expected_checkpoint_training_membership",
            choices=MEMBERSHIP_POLICIES,
            default="any",
        )
        return parser

    args = make_parser().parse_args()
    checkpoint = _load_checkpoint(args.checkpoint)
    evaluation_protocol, evaluation_uids = _selected_uids(args)
    membership = audit_checkpoint_evaluation_membership(
        checkpoint,
        evaluation_protocol_sha256=evaluation_protocol,
        evaluation_object_uids=evaluation_uids,
        expected_membership=str(args.expected_checkpoint_training_membership),
    )
    if (
        membership["protocol_relation"] != "same"
        and not args.allow_checkpoint_target_protocol_mismatch
    ):
        raise RuntimeError(
            "checkpoint/evaluation target protocols differ; the explicit "
            "cross-protocol flag is required"
        )

    saved_binding = checkpoint.get("data_identity", {}).get("native_ss")
    summary_binding = checkpoint.get("model_summary", {}).get("upstream_native_ss")
    if not isinstance(saved_binding, dict) or saved_binding != summary_binding:
        raise RuntimeError("SLat checkpoint Native-SS identity is internally inconsistent")
    _, runtime_deployment = load_no_vggt_ss_evidence(args.native_ss_report)
    runtime_binding = original_upstream_binding(runtime_deployment)
    transition = validate_checkpoint_native_ss_binding_relocation(
        saved_binding,
        runtime_binding,
        allow_path_relocation=bool(args.allow_checkpoint_data_path_relocation),
    )
    print(
        "[official_slat_gt_support:checkpoint_binding] "
        + json.dumps(transition, sort_keys=True, ensure_ascii=False),
        flush=True,
    )
    print(
        "[official_slat_gt_support:membership] "
        + json.dumps(membership, sort_keys=True, ensure_ascii=False),
        flush=True,
    )

    def checkpoint_compatible_upstream_binding(value: dict[str, Any]) -> dict[str, Any]:
        current = original_upstream_binding(value)
        validate_checkpoint_native_ss_binding_relocation(
            saved_binding,
            current,
            allow_path_relocation=bool(args.allow_checkpoint_data_path_relocation),
        )
        return copy.deepcopy(saved_binding)

    _base.make_parser = make_parser
    _base._upstream_binding = checkpoint_compatible_upstream_binding
    try:
        _base.main()
    finally:
        _base.make_parser = original_make_parser
        _base._upstream_binding = original_upstream_binding
    _rewrite_report(args.output_dir, membership)


if __name__ == "__main__":
    main()
