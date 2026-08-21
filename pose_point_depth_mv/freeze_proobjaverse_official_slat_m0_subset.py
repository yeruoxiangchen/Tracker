#!/usr/bin/env python3
"""Freeze a hash-ordered Train2000 subset for matched-support G/M0 tests."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv.proobjaverse_official_ss import (
    load_official_native_ss_deployment,
)

from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)


SUBSET_FORMAT = "pose_point_depth_mv.proobjaverse_official_slat_m0_subset.v1"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slat_manifest", required=True)
    parser.add_argument("--lifting_manifest", required=True)
    parser.add_argument(
        "--native_ss_report",
        required=True,
        help="passed official Native-SS deployment used by both G512 and M0-512",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--object_count", type=int, default=512)
    return parser


def _write_or_validate(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        if load_json(path) != payload:
            raise RuntimeError(f"refusing to overwrite changed frozen subset: {path}")
        return
    atomic_json(path, payload)


def main() -> None:
    args = make_parser().parse_args()
    count = int(args.object_count)
    if count <= 0:
        raise ValueError("object_count must be positive")
    slat_path = Path(args.slat_manifest).expanduser().resolve()
    lifting_path = Path(args.lifting_manifest).expanduser().resolve()
    slat = load_json(slat_path)
    lifting = load_json(lifting_path)
    ss_report, ss_binding = load_official_native_ss_deployment(
        args.native_ss_report
    )
    slat_rows = list(slat.get("samples", ()))
    lifting_rows = list(lifting.get("samples", ()))
    if len(slat_rows) < count:
        raise ValueError(f"requested {count} objects from only {len(slat_rows)} rows")
    lifting_by_uid = {str(row["uid"]): row for row in lifting_rows}
    if len(lifting_by_uid) != len(lifting_rows):
        raise ValueError("lifting manifest has duplicate UIDs")
    # The parent official train split is already SHA256(seed:uid)-ordered.  A
    # prefix therefore remains independent of filesystem/shard enumeration.
    selected_slat = [dict(row) for row in slat_rows[:count]]
    selected_uids = [str(row["uid"]) for row in selected_slat]
    if len(selected_uids) != len(set(selected_uids)):
        raise ValueError("selected SLat subset has duplicate UIDs")
    selected_lifting = [dict(lifting_by_uid[uid]) for uid in selected_uids]
    checkpoint = torch.load(
        ss_binding["checkpoint"], map_location="cpu", weights_only=False
    )
    training_uids = checkpoint.get("data_identity", {}).get("object_uids")
    if not isinstance(training_uids, list):
        raise RuntimeError("official Native-SS checkpoint lacks training object_uids")
    missing_from_ss_training = sorted(set(selected_uids) - set(map(str, training_uids)))
    if missing_from_ss_training:
        raise RuntimeError(
            "Train512 is not covered by the frozen Native-SS training split: "
            f"{missing_from_ss_training[:8]}"
        )
    target = dict(slat.get("config", {})).get("target_source", {})
    domain = ss_report.get("official_ss_domain_contract")
    if (
        not isinstance(domain, dict)
        or str(domain.get("official_slat_protocol_sha256", ""))
        != str(target.get("protocol_sha256", ""))
    ):
        raise RuntimeError("official Native-SS and SLat target protocols differ")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "format": SUBSET_FORMAT,
        "passed": True,
        "selection_policy": "parent_manifest_prefix_from_hash_ordered_train_split",
        "object_count": count,
        "object_uids": selected_uids,
        "object_uid_hash": canonical_sha256(selected_uids),
        "parent_slat_manifest": str(slat_path),
        "parent_slat_manifest_sha256": sha256_file(slat_path),
        "parent_lifting_manifest": str(lifting_path),
        "parent_lifting_manifest_sha256": sha256_file(lifting_path),
        "native_ss_report": str(Path(args.native_ss_report).expanduser().resolve()),
        "native_ss_report_sha256": sha256_file(args.native_ss_report),
        "native_ss_deployment": ss_binding,
        "native_ss_training_overlap_verified": True,
    }
    derived_config = copy.deepcopy(dict(slat.get("config", {})))
    derived_config["native_ss_deployment"] = copy.deepcopy(ss_binding)
    derived_config["training_coordinate_policy"] = (
        "official GT SLat coordinates for G512, or exact GT intersection with "
        "the frozen official Native-SS prediction for M0-512; predicted-only "
        "coordinates are never assigned synthetic target features"
    )
    slat_subset = {
        **slat,
        "config": derived_config,
        "config_hash": canonical_sha256(derived_config),
        "samples": selected_slat,
        "sample_count": count,
        "object_count": count,
        "derived_selection": provenance,
    }
    lifting_subset = {
        **lifting,
        "samples": selected_lifting,
        "sample_count": count,
        "object_count": count,
        "selection": {
            "mode": "frozen_m0_parent_manifest_prefix",
            "count": count,
            "object_uid_hash": provenance["object_uid_hash"],
        },
        "derived_selection": provenance,
    }
    slat_out = output / "slat_manifest.json"
    lifting_out = output / "lifting_manifest.json"
    _write_or_validate(slat_out, slat_subset)
    _write_or_validate(lifting_out, lifting_subset)
    report = {
        **provenance,
        "slat_manifest": str(slat_out),
        "slat_manifest_sha256": sha256_file(slat_out),
        "lifting_manifest": str(lifting_out),
        "lifting_manifest_sha256": sha256_file(lifting_out),
    }
    report["report_sha256"] = canonical_sha256(report)
    _write_or_validate(output / "report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
