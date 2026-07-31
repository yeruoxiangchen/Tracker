#!/usr/bin/env python3
"""Freeze a fresh matched-coordinate multi-seed Mesh gate for Direct-SLAT."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import torch

from pose_point_depth_mv.direct_slat_flow import (
    DIRECT_SLAT_FLOW_VERSION,
    DIRECT_SLAT_TRAINING_SEMANTICS_V4,
    DIRECT_SLAT_TRAINING_SEMANTICS_V5,
    SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET,
    assert_disjoint_object_splits,
    support_generator_identity,
)
from pose_point_depth_mv.direct_slat_matched_mesh_blind import (
    PROTOCOL_FORMAT,
    bind_file,
    canonical_sha256,
    select_fresh_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--train_cache_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exclude_report", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--protocol_name", required=True)
    parser.add_argument("--expected_checkpoint_step", type=int, default=200)
    parser.add_argument("--expected_objects", type=int, default=32)
    parser.add_argument("--joint_seeds", default="42,43,44")
    parser.add_argument("--selection_seed", type=int, default=20260729)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg_strength", type=float, default=5.0)
    parser.add_argument("--cfg_interval", default="0.5,1.0")
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    return parser.parse_args()


def parse_csv(value: str, cast) -> list[Any]:
    values = [cast(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("CSV values must be non-empty and unique")
    return values


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def excluded_from_report(report: dict[str, Any]) -> set[str]:
    excluded: set[str] = set()
    protocols = report.get("protocols")
    if isinstance(protocols, dict):
        for value in protocols.values():
            if isinstance(value, dict):
                excluded.update(
                    str(item) for item in value.get("object_ids", []) if str(item)
                )
    for key in ("object_rows", "records", "selected"):
        rows = report.get(key)
        if isinstance(rows, list):
            excluded.update(
                str(row.get("object_uid", ""))
                for row in rows
                if isinstance(row, dict) and str(row.get("object_uid", ""))
            )
    return excluded


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    seeds = parse_csv(args.joint_seeds, int)
    interval = parse_csv(args.cfg_interval, float)
    if len(interval) != 2 or not 0.0 <= interval[0] <= interval[1] <= 1.0:
        raise ValueError("CFG interval must contain two ordered values in [0,1]")
    if min(
        int(args.expected_objects),
        int(args.steps),
        int(args.surface_samples),
        int(args.bootstrap_samples),
    ) <= 0:
        raise ValueError("blind protocol counts must be positive")

    cache_path = Path(args.cache_manifest).resolve()
    train_path = Path(args.train_cache_manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    cache = load_json(cache_path)
    train = load_json(train_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != DIRECT_SLAT_FLOW_VERSION:
        raise RuntimeError("candidate checkpoint is not Direct-SLAT")
    if int(checkpoint.get("step", -1)) != int(args.expected_checkpoint_step):
        raise RuntimeError("candidate checkpoint step differs from frozen request")
    saved_args = dict(checkpoint.get("args", {}))
    training_semantics = saved_args.get("training_semantics")
    if (
        training_semantics
        not in {
            DIRECT_SLAT_TRAINING_SEMANTICS_V4,
            DIRECT_SLAT_TRAINING_SEMANTICS_V5,
        }
        or saved_args.get("slat_guided_delta_policy") != "post_cfg_v2"
        or saved_args.get("slat_delta_bound_mode") != "smooth_rms_v2"
        or saved_args.get("support_interval_policy") != "cfg_active_only_v1"
    ):
        raise RuntimeError("candidate is not a frozen V4/V5 rollout endpoint model")
    if (
        training_semantics == DIRECT_SLAT_TRAINING_SEMANTICS_V5
        and saved_args.get("slat_residual_combination_policy")
        != SLAT_RESIDUAL_COMBINATION_BRANCH_BUDGET
    ):
        raise RuntimeError("V5 candidate lacks the frozen LoRA/support branch budget")
    if float(saved_args.get("train_cfg_strength", -1.0)) != float(args.cfg_strength):
        raise RuntimeError("candidate training CFG strength differs")
    if [float(value) for value in saved_args.get("train_cfg_interval", [])] != [
        float(value) for value in interval
    ]:
        raise RuntimeError("candidate training CFG interval differs")
    if (
        checkpoint.get("model_summary", {}).get("support_generator")
        != support_generator_identity(dict(cache["config"]))
    ):
        raise RuntimeError("candidate and blind cache support generators differ")
    if str(checkpoint.get("model_summary", {}).get("slat_normalization_hash", "")) != str(
        cache.get("slat_normalization_hash", "")
    ):
        raise RuntimeError("candidate and blind cache SLAT normalization differ")

    split_audit = assert_disjoint_object_splits(
        list(train["samples"]),
        list(cache["samples"]),
    )
    excluded: set[str] = set()
    exclusion_bindings = []
    for report_path in args.exclude_report:
        report = load_json(report_path)
        excluded.update(excluded_from_report(report))
        exclusion_bindings.append(bind_file(report_path))
    if not excluded:
        raise RuntimeError("blind freeze requires non-empty prior-object exclusions")

    selected = select_fresh_rows(
        list(cache["samples"]),
        seeds=seeds,
        excluded_object_uids=excluded,
        expected_objects=int(args.expected_objects),
        selection_seed=int(args.selection_seed),
    )
    thresholds = {
        "min_chamfer_bootstrap_lower": 0.0,
        "min_chamfer_median": 0.0,
        "min_chamfer_object_win_rate": 0.5,
        "min_positive_seed_fraction": 2.0 / 3.0,
        "secondary_mean_floors": {
            "fscore_0p02_delta": 0.0,
            "normal_consistency_delta": 0.0,
            "largest_component_ratio_delta": 0.0,
        },
    }
    protocol: dict[str, Any] = {
        "format": PROTOCOL_FORMAT,
        "formal": True,
        "protocol_name": str(args.protocol_name),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "candidate": {
            "checkpoint_step": int(checkpoint["step"]),
            "training_semantics": saved_args["training_semantics"],
            "support_scale": float(saved_args.get("support_scale", 1.0)),
        },
        "bindings": {
            "cache_manifest": bind_file(cache_path),
            "train_cache_manifest": bind_file(train_path),
            "checkpoint": bind_file(checkpoint_path),
        },
        "exclusion_report_bindings": exclusion_bindings,
        "selection": {
            "policy": "sha256_rank_after_prior_object_exclusion.v1",
            "selection_seed": int(args.selection_seed),
            "expected_objects": int(args.expected_objects),
            "eligible_cache_object_count": int(cache.get("object_count", -1)),
            "excluded_object_uids": sorted(excluded),
            "fresh_candidate_mesh_outputs_read": False,
            "prior_reports_read_for_exclusion_only": True,
        },
        "selected": selected,
        "sampling": {
            "joint_seeds": seeds,
            "steps": int(args.steps),
            "cfg_strength": float(args.cfg_strength),
            "cfg_interval": [float(value) for value in interval],
            "rescale_t": float(args.rescale_t),
            "surface_samples": int(args.surface_samples),
            "bootstrap_samples": int(args.bootstrap_samples),
            "amp_dtype": str(args.amp_dtype),
            "noise_identity_mode": "object_uid_v1",
            "same_coordinates": "both branches use frozen corrected-SS coords",
            "same_noise": "coordinate-keyed SLAT initial noise is bit-identical",
            "slat_delta_policy": {
                "scale": float(saved_args.get("slat_delta_scale", 1.0)),
                "rms_ratio_cap": float(
                    saved_args.get("slat_delta_rms_ratio_cap", 0.1)
                ),
                "guided_delta_policy": str(
                    saved_args["slat_guided_delta_policy"]
                ),
                "bound_mode": str(saved_args["slat_delta_bound_mode"]),
                "support_interval_policy": str(
                    saved_args["support_interval_policy"]
                ),
                "residual_combination_policy": str(
                    saved_args.get(
                        "slat_residual_combination_policy",
                        "joint_total_v1",
                    )
                ),
                "lora_delta_scale": float(
                    saved_args.get("slat_lora_delta_scale", 1.0)
                ),
                "lora_delta_rms_ratio_cap": float(
                    saved_args.get("slat_lora_delta_rms_ratio_cap", -1.0)
                ),
                "support_delta_scale": float(
                    saved_args.get("slat_support_delta_scale", 1.0)
                ),
                "support_delta_rms_ratio_cap": float(
                    saved_args.get(
                        "slat_support_delta_rms_ratio_cap",
                        -1.0,
                    )
                ),
            },
        },
        "statistics": {
            "independent_unit": "object_uid",
            "seed_aggregation": "average paired seed deltas within object first",
            "bootstrap_unit": "object_uid",
            "thresholds": thresholds,
        },
        "train_eval_split_audit": split_audit,
        "scope_guard": (
            "confirmatory matched corrected-coordinate Direct-SLAT Full-vs-Stock "
            "utility; this does not test the end-to-end Direct-SS coordinate branch"
        ),
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    output_dir.mkdir(parents=True)
    path = output_dir / "protocol.json"
    path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "protocol": str(path),
                "protocol_sha256": protocol["protocol_sha256"],
                "selected_objects": len(selected),
                "excluded_objects": len(excluded),
                "joint_seeds": seeds,
                "expected_pairs": len(selected) * len(seeds),
                "scope_guard": protocol["scope_guard"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
