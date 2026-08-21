#!/usr/bin/env python3
"""Pure helpers for the matched-coordinate Direct-SLAT Mesh blind gate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_FORMAT = "pose_point_depth_mv.direct_slat_matched_mesh_blind_protocol.v1"
REPORT_FORMAT = "pose_point_depth_mv.direct_slat_matched_mesh_blind_report.v1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_file(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def validate_file_binding(binding: dict[str, Any], label: str) -> None:
    if set(binding) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"{label} has an invalid file-binding schema")
    path = Path(str(binding["path"])).resolve()
    if (
        not path.is_file()
        or int(path.stat().st_size) != int(binding["size_bytes"])
        or sha256_file(path) != str(binding["sha256"])
    ):
        raise RuntimeError(f"frozen binding changed: {label}={path}")


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("matched Mesh blind protocol must be a JSON object")
    body = dict(protocol)
    saved = str(body.pop("protocol_sha256", ""))
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("formal") is not True
        or not saved
        or canonical_sha256(body) != saved
    ):
        raise RuntimeError("matched Mesh blind protocol identity is invalid")
    for label, binding in dict(protocol["bindings"]).items():
        if isinstance(binding, dict) and set(binding) == {
            "path",
            "size_bytes",
            "sha256",
        }:
            validate_file_binding(binding, f"bindings.{label}")
    selected = list(protocol.get("selected", []))
    expected = int(protocol["selection"]["expected_objects"])
    if len(selected) != expected:
        raise RuntimeError("matched Mesh blind selected-object count changed")
    object_uids = [str(row["object_uid"]) for row in selected]
    uids = [str(row["uid"]) for row in selected]
    if (
        len(set(object_uids)) != expected
        or len(set(uids)) != expected
        or any(not value for value in [*object_uids, *uids])
    ):
        raise RuntimeError("matched Mesh blind selection is not object-unique")
    excluded = set(str(value) for value in protocol["selection"]["excluded_object_uids"])
    if excluded.intersection(object_uids):
        raise RuntimeError("matched Mesh blind selection overlaps excluded objects")
    return protocol


def stable_rank(selection_seed: int, object_uid: str, uid: str) -> str:
    return hashlib.sha256(
        f"{int(selection_seed)}|{object_uid}|{uid}".encode("utf-8")
    ).hexdigest()


def select_fresh_rows(
    rows: Iterable[dict[str, Any]],
    *,
    seeds: Iterable[int],
    excluded_object_uids: Iterable[str],
    expected_objects: int,
    selection_seed: int,
) -> list[dict[str, Any]]:
    requested_seeds = tuple(int(value) for value in seeds)
    if not requested_seeds or len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("blind seeds must be non-empty and unique")
    excluded = {str(value) for value in excluded_object_uids}
    by_object: dict[str, dict[str, dict[int, int]]] = {}
    for index, row in enumerate(rows):
        object_uid = str(row.get("object_uid", ""))
        uid = str(row.get("uid", ""))
        seed = int(row.get("support_seed", -1))
        if not object_uid or not uid:
            raise ValueError("cache row lacks object/sequence identity")
        by_object.setdefault(object_uid, {}).setdefault(uid, {})[seed] = index

    eligible: list[dict[str, Any]] = []
    for object_uid, sequences in by_object.items():
        if object_uid in excluded:
            continue
        choices = [
            (uid, seed_map)
            for uid, seed_map in sorted(sequences.items())
            if all(seed in seed_map for seed in requested_seeds)
        ]
        if not choices:
            continue
        uid, seed_map = choices[0]
        eligible.append(
            {
                "rank": stable_rank(selection_seed, object_uid, uid),
                "object_uid": object_uid,
                "uid": uid,
                "cache_indices": {
                    str(seed): int(seed_map[seed]) for seed in requested_seeds
                },
            }
        )
    eligible.sort(key=lambda row: (str(row["rank"]), str(row["object_uid"])))
    if len(eligible) < int(expected_objects):
        raise RuntimeError(
            f"only {len(eligible)} fresh eligible objects; need {expected_objects}"
        )
    selected = eligible[: int(expected_objects)]
    for position, row in enumerate(selected):
        row["object_position"] = position
    return selected


def summarize_seed_directions(
    records: Iterable[dict[str, Any]],
    metric_names: Iterable[str],
) -> dict[str, dict[str, dict[str, float]]]:
    names = tuple(str(value) for value in metric_names)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(int(row["joint_seed"]), []).append(row)
    output: dict[str, dict[str, dict[str, float]]] = {}
    for seed, values in sorted(grouped.items()):
        output[str(seed)] = {}
        for name in names:
            metric_values = [float(row[name]) for row in values]
            output[str(seed)][name] = {
                "count": int(len(metric_values)),
                "mean": float(sum(metric_values) / len(metric_values)),
                "positive_rate": float(
                    sum(value > 0.0 for value in metric_values)
                    / len(metric_values)
                ),
            }
    return output


def formal_decision(
    summary: dict[str, Any],
    by_seed_summary: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    primary = dict(summary["chamfer_l1_improvement"])
    lower = float(primary["bootstrap_mean_95_ci"][0])
    seed_means = [
        float(value["chamfer_l1_improvement"]["mean"])
        for value in by_seed_summary.values()
    ]
    positive_seed_fraction = (
        float(sum(value > 0.0 for value in seed_means) / len(seed_means))
        if seed_means
        else 0.0
    )
    primary_checks = {
        "bootstrap_lower_gt_min": lower
        > float(thresholds["min_chamfer_bootstrap_lower"]),
        "median_gt_min": float(primary["median"])
        > float(thresholds["min_chamfer_median"]),
        "object_win_rate_ge_min": float(primary["positive_rate"])
        >= float(thresholds["min_chamfer_object_win_rate"]),
        "positive_seed_fraction_ge_min": positive_seed_fraction
        >= float(thresholds["min_positive_seed_fraction"]),
    }
    secondary_checks = {
        name: float(summary[name]["mean"])
        >= float(thresholds["secondary_mean_floors"][name])
        for name in (
            "fscore_0p02_delta",
            "normal_consistency_delta",
            "largest_component_ratio_delta",
        )
    }
    values = [
        *[float(value) for value in primary["bootstrap_mean_95_ci"]],
        float(primary["median"]),
        float(primary["positive_rate"]),
        positive_seed_fraction,
        *[
            float(summary[name]["mean"])
            for name in (
                "fscore_0p02_delta",
                "normal_consistency_delta",
                "largest_component_ratio_delta",
            )
        ],
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("formal matched Mesh decision contains non-finite values")
    primary_pass = all(primary_checks.values())
    secondary_pass = all(secondary_checks.values())
    return {
        "primary_metric": "object-averaged chamfer_l1_improvement",
        "primary_checks": primary_checks,
        "secondary_non_degradation_checks": secondary_checks,
        "positive_seed_fraction": positive_seed_fraction,
        "primary_pass": primary_pass,
        "secondary_pass": secondary_pass,
        "formal_pass": primary_pass and secondary_pass,
        "scope": (
            "matched corrected-coordinate Direct-SLAT utility only; "
            "not an end-to-end Direct-SS plus Direct-SLAT claim"
        ),
    }
