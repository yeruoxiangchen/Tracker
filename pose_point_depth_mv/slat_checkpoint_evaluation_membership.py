#!/usr/bin/env python3
"""Audit evaluation-object membership in a Native-SLat checkpoint dataset.

The checkpoint target protocol and an evaluation cache protocol may differ for
an explicitly registered compatibility benchmark.  A protocol mismatch must
never silently turn a training-overlap object set into a generalization claim,
so callers must pre-register and verify the exact membership relationship.
"""

from __future__ import annotations

from typing import Any, Iterable

from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    canonical_sha256,
)


MEMBERSHIP_POLICIES = ("any", "all_training", "all_disjoint")


def checkpoint_target_protocol_sha256(checkpoint: dict[str, Any]) -> str:
    value = (
        checkpoint.get("data_identity", {})
        .get("target_decoder_audit", {})
        .get("protocol_sha256")
    )
    if not isinstance(value, str) or not value:
        raise RuntimeError("Native-SLat checkpoint target protocol is missing")
    return value


def audit_checkpoint_evaluation_membership(
    checkpoint: dict[str, Any],
    *,
    evaluation_protocol_sha256: str,
    evaluation_object_uids: Iterable[str],
    expected_membership: str = "any",
) -> dict[str, Any]:
    """Return and enforce the checkpoint/evaluation membership relationship."""

    if expected_membership not in MEMBERSHIP_POLICIES:
        raise ValueError(
            f"unexpected membership policy={expected_membership!r}; "
            f"expected one of {MEMBERSHIP_POLICIES}"
        )
    evaluation_protocol = str(evaluation_protocol_sha256)
    if not evaluation_protocol:
        raise RuntimeError("evaluation target protocol is missing")
    checkpoint_protocol = checkpoint_target_protocol_sha256(checkpoint)

    raw_training_uids = checkpoint.get("data_identity", {}).get("object_uids")
    if not isinstance(raw_training_uids, list) or not raw_training_uids:
        raise RuntimeError("Native-SLat checkpoint training object UIDs are missing")
    training_uids = [str(value) for value in raw_training_uids]
    if not all(training_uids) or len(training_uids) != len(set(training_uids)):
        raise RuntimeError("Native-SLat checkpoint training object UIDs are invalid")

    evaluation_uids = [str(value) for value in evaluation_object_uids]
    if not evaluation_uids or not all(evaluation_uids):
        raise RuntimeError("evaluation object UIDs are empty/invalid")
    if len(evaluation_uids) != len(set(evaluation_uids)):
        raise RuntimeError("evaluation object UIDs are duplicated")

    training_set = set(training_uids)
    overlap = [uid for uid in evaluation_uids if uid in training_set]
    overlap_count = len(overlap)
    evaluation_count = len(evaluation_uids)
    if expected_membership == "all_training" and overlap_count != evaluation_count:
        raise RuntimeError(
            "evaluation set is not entirely contained in checkpoint training data: "
            f"overlap={overlap_count}/{evaluation_count}"
        )
    if expected_membership == "all_disjoint" and overlap_count != 0:
        raise RuntimeError(
            "evaluation set is not disjoint from checkpoint training data: "
            f"overlap={overlap_count}/{evaluation_count}"
        )

    return {
        "version": "pose_point_depth_mv.slat_checkpoint_evaluation_membership.v1",
        "checkpoint_protocol_sha256": checkpoint_protocol,
        "evaluation_protocol_sha256": evaluation_protocol,
        "protocol_relation": (
            "same" if checkpoint_protocol == evaluation_protocol else "different"
        ),
        "expected_membership": expected_membership,
        "checkpoint_training_object_count": len(training_uids),
        "checkpoint_training_uid_sha256": canonical_sha256(training_uids),
        "evaluation_object_count": evaluation_count,
        "evaluation_uid_sha256": canonical_sha256(evaluation_uids),
        "training_overlap_count": overlap_count,
        "training_overlap_rate": overlap_count / evaluation_count,
        "all_evaluation_objects_in_checkpoint_training": (
            overlap_count == evaluation_count
        ),
        "all_evaluation_objects_disjoint_from_checkpoint_training": (
            overlap_count == 0
        ),
        "passed": True,
    }


__all__ = [
    "MEMBERSHIP_POLICIES",
    "audit_checkpoint_evaluation_membership",
    "checkpoint_target_protocol_sha256",
]
