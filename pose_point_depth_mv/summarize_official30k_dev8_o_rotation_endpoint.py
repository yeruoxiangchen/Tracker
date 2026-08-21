#!/usr/bin/env python3
"""Aggregate the eight-arm Dev8 object-frame endpoint diagnostic."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from pose_point_depth_mv.eval_direct_flow import bootstrap_mean_ci
from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import write_json
from pose_point_depth_mv.evaluate_official30k_dev8_o_rotation_endpoint import (
    ARMS,
    REPORT_FORMAT,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import canonical_sha256


AGGREGATE_FORMAT = "pose_point_depth_mv.official30k_dev8_o_rotation_endpoint_aggregate.v1"


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "bootstrap_mean_95_ci": bootstrap_mean_ci(values, samples=5000, seed=20260819),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    reports: dict[str, dict[str, Any]] = {}
    keyed: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    stable_identity = None
    for arm in ARMS:
        path = root / arm / "report.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format") != REPORT_FORMAT or value.get("passed") is not True:
            raise RuntimeError(f"arm report did not pass: {path}")
        body = dict(value)
        saved = str(body.pop("report_sha256"))
        if canonical_sha256(body) != saved:
            raise RuntimeError(f"arm report SHA differs: {path}")
        identity = dict(value["run_identity"])
        for field in ("arm",):
            identity.pop(field, None)
        if stable_identity is None:
            stable_identity = identity
        elif identity != stable_identity:
            raise RuntimeError(f"arm run identity differs beyond arm: {path}")
        rows = {(str(r["object_uid"]), int(r["seed"])): r for r in value["records"]}
        if len(rows) != len(value["records"]):
            raise RuntimeError(f"duplicate rows: {path}")
        reports[arm] = value
        keyed[arm] = rows
    keys = set(keyed["official_o"])
    if any(set(rows) != keys for rows in keyed.values()):
        raise RuntimeError("arm object/seed coverage differs")

    absolute: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    metrics = (
        ("chamfer_l1", "surface", "lower"),
        ("fscore_0p02", "surface", "higher"),
        ("normal_consistency", "surface", "higher"),
        ("largest_component_ratio", "structure", "higher"),
    )
    base = keyed["official_o"]
    for arm in ARMS:
        absolute[arm] = {}
        paired[arm] = {}
        for metric, section, direction in metrics:
            values = [float(keyed[arm][key][section][metric]) for key in sorted(keys)]
            absolute[arm][metric] = _summary(values)
            # Signed so positive always means the rotated arm is worse than official O.
            if direction == "lower":
                degradation = [
                    float(keyed[arm][key][section][metric])
                    - float(base[key][section][metric])
                    for key in sorted(keys)
                ]
            else:
                degradation = [
                    float(base[key][section][metric])
                    - float(keyed[arm][key][section][metric])
                    for key in sorted(keys)
                ]
            summary = _summary(degradation)
            summary["positive_degradation_rate"] = float(
                np.mean(np.asarray(degradation) > 0.0)
            )
            summary["sign_convention"] = "positive means worse than official_o"
            paired[arm][f"{metric}_degradation_vs_official_o"] = summary

    phone_chamfer = paired["phone_o"]["chamfer_l1_degradation_vs_official_o"]
    output = {
        "format": AGGREGATE_FORMAT,
        "passed": True,
        "formal": False,
        "objects": len({key[0] for key in keys}),
        "records_per_arm": len(keys),
        "arms": list(ARMS),
        "absolute": absolute,
        "paired_degradation_vs_official_o": paired,
        "primary_readout": {
            "phone_o_chamfer_l1_degradation": phone_chamfer,
            "phone_o_is_rotation_sensitive": bool(
                float(phone_chamfer["mean"]) > 0.0
                and float(phone_chamfer["positive_degradation_rate"]) > 0.5
            ),
        },
        "scope_guard": (
            "Dev8 seed42 endpoint sensitivity diagnostic; center/scale are exact, "
            "phone O changes orientation only, and every Mesh is mapped back to "
            "official O before scoring"
        ),
    }
    output["report_sha256"] = canonical_sha256(output)
    destination = Path(args.output).expanduser().resolve()
    write_json(destination, output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
