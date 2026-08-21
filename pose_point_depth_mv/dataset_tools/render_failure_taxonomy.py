#!/usr/bin/env python3
"""Canonical failure taxonomy shared by render admission and aggregation."""

from __future__ import annotations

from collections.abc import Mapping


FAILURE_TAXONOMY_SCHEMA = "tracker.mixed_multiview_failure_taxonomy.v1"

INFRASTRUCTURE_FAILURES = frozenset(
    {
        "renderer_failed",
        "encoder_failed",
        "other_failed",
    }
)

_LEGACY_SOURCE_ASSET_ERRORS = (
    "scene has no mesh geometry",
)


def canonical_failure_class(failure: Mapping[str, object]) -> str:
    """Return the admission class while preserving raw manifest evidence."""

    raw_class = str(failure.get("failure_class", "missing"))
    error = str(failure.get("error", "")).lower()
    if raw_class == "other_failed" and any(
        pattern in error for pattern in _LEGACY_SOURCE_ASSET_ERRORS
    ):
        return "source_asset_rejected"
    return raw_class
