#!/usr/bin/env python3
"""Shared contracts for non-formal Objaverse training-overlap diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pose_point_depth_mv.freeze_objaverse16_test import PROTOCOL_FORMAT


TRAINING_OVERLAP_PROTOCOL_FORMAT = (
    "pose_point_depth_mv.objaverse_training_overlap_subset.v1"
)
TRAINING_OVERLAP_SCOPE = "training_overlap_objaverse_subset"
SOURCE_SCOPES = ("objaverse2k_train", "mixed_objaverse_train")
OBJAVERSE2K_DEV64_PROTOCOL_FORMAT = (
    "pose_point_depth_mv.objaverse2k_dev64_reconviagen_selection.v1"
)
OBJAVERSE2K_DEV64_SCOPE = "object_disjoint_objaverse2k_dev64_reconviagen"


@dataclass(frozen=True)
class SelectionContract:
    scope: str
    object_count: int
    selected_uids: tuple[str, ...]
    selected_object_uids: tuple[str, ...]
    training_overlap: bool
    training_object_disjoint: bool
    source_mesh_disjoint: bool
    source_scope: str | None


def validate_selection(payload: dict[str, Any]) -> SelectionContract:
    """Validate either the original test16 or the new overlap-only protocol."""

    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("selection contains no samples")
    uids = tuple(str(row.get("uid", "")) for row in samples)
    objects = tuple(str(row.get("object_uid", "")) for row in samples)
    if (
        not all(uids)
        or not all(objects)
        or len(uids) != len(set(uids))
        or len(objects) != len(set(objects))
    ):
        raise RuntimeError("selection identities are empty or duplicated")

    frozen = payload.get("objaverse16_protocol")
    if isinstance(frozen, dict):
        required = {
            "format": PROTOCOL_FORMAT,
            "scope": "frozen_objaverse_test16",
            "passed": True,
            "object_count": 16,
            "training_object_disjoint": True,
            "source_mesh_disjoint": True,
        }
        mismatch = {
            key: (frozen.get(key), expected)
            for key, expected in required.items()
            if frozen.get(key) != expected
        }
        if mismatch or len(samples) != 16:
            raise RuntimeError(f"frozen Objaverse test16 contract differs: {mismatch}")
        if tuple(map(str, frozen.get("selected_uids", []))) != uids:
            raise RuntimeError("frozen Objaverse selected UID order differs")
        return SelectionContract(
            scope="frozen_objaverse_test16",
            object_count=16,
            selected_uids=uids,
            selected_object_uids=objects,
            training_overlap=False,
            training_object_disjoint=True,
            source_mesh_disjoint=True,
            source_scope=None,
        )

    dev64 = payload.get("objaverse2k_dev64_protocol")
    if isinstance(dev64, dict):
        required = {
            "format": OBJAVERSE2K_DEV64_PROTOCOL_FORMAT,
            "scope": OBJAVERSE2K_DEV64_SCOPE,
            "formal": False,
            "training_overlap": False,
            "training_object_disjoint": True,
            "source_mesh_disjoint": True,
            "passed": True,
            "object_count": 64,
        }
        mismatch = {
            key: (dev64.get(key), expected)
            for key, expected in required.items()
            if dev64.get(key) != expected
        }
        if mismatch or len(samples) != 64:
            raise RuntimeError(f"Objaverse2K dev64 contract differs: {mismatch}")
        if tuple(map(str, dev64.get("selected_uids", []))) != uids:
            raise RuntimeError("Objaverse2K dev64 selected UID order differs")
        if tuple(map(str, dev64.get("selected_object_uids", []))) != objects:
            raise RuntimeError("Objaverse2K dev64 selected object order differs")
        if any(str(row.get("dataset_source", "")) != "objaverse" for row in samples):
            raise RuntimeError("Objaverse2K dev64 contains a non-Objaverse row")
        return SelectionContract(
            scope=OBJAVERSE2K_DEV64_SCOPE,
            object_count=64,
            selected_uids=uids,
            selected_object_uids=objects,
            training_overlap=False,
            training_object_disjoint=True,
            source_mesh_disjoint=True,
            source_scope="objaverse2k_dev64",
        )

    overlap = payload.get("training_overlap_protocol")
    if not isinstance(overlap, dict):
        raise RuntimeError("selection has no supported Objaverse protocol")
    source_scope = str(overlap.get("source_scope", ""))
    required = {
        "format": TRAINING_OVERLAP_PROTOCOL_FORMAT,
        "scope": TRAINING_OVERLAP_SCOPE,
        "formal": False,
        "training_overlap": True,
        "training_object_disjoint": False,
        "source_mesh_disjoint": False,
        "passed": True,
        "object_count": len(samples),
    }
    mismatch = {
        key: (overlap.get(key), expected)
        for key, expected in required.items()
        if overlap.get(key) != expected
    }
    if source_scope not in SOURCE_SCOPES:
        mismatch["source_scope"] = (source_scope, SOURCE_SCOPES)
    if mismatch:
        raise RuntimeError(f"training-overlap protocol differs: {mismatch}")
    if tuple(map(str, overlap.get("selected_uids", []))) != uids:
        raise RuntimeError("training-overlap selected UID order differs")
    if tuple(map(str, overlap.get("selected_object_uids", []))) != objects:
        raise RuntimeError("training-overlap selected object order differs")
    if any(str(row.get("dataset_source", "")) != "objaverse" for row in samples):
        raise RuntimeError("training-overlap selection contains a non-Objaverse row")
    return SelectionContract(
        scope=TRAINING_OVERLAP_SCOPE,
        object_count=len(samples),
        selected_uids=uids,
        selected_object_uids=objects,
        training_overlap=True,
        training_object_disjoint=False,
        source_mesh_disjoint=False,
        source_scope=source_scope,
    )


def expected_view_count(row: dict[str, Any]) -> int:
    frozen = row.get("objaverse16_selection")
    if isinstance(frozen, dict):
        return int(frozen["expected_point_prior_view_count"])
    overlap = row.get("training_overlap_selection")
    if isinstance(overlap, dict):
        return int(overlap["expected_view_count"])
    dev64 = row.get("objaverse2k_dev64_selection")
    if isinstance(dev64, dict):
        return int(dev64["expected_view_count"])
    raise RuntimeError(f"selection row has no view-count contract: {row.get('uid')}")


__all__ = [
    "SelectionContract",
    "OBJAVERSE2K_DEV64_PROTOCOL_FORMAT",
    "OBJAVERSE2K_DEV64_SCOPE",
    "SOURCE_SCOPES",
    "TRAINING_OVERLAP_PROTOCOL_FORMAT",
    "TRAINING_OVERLAP_SCOPE",
    "expected_view_count",
    "validate_selection",
]
