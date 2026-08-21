#!/usr/bin/env python3
"""Freeze and verify the formal Holdout64 Pose+Mask blind-addendum contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)


FORMAT = "pose_point_depth_mv.holdout64_pose_mask_blind_protocol.v1"
EXPECTED_OBJECTS = 64
EXPECTED_SEED = 42
FORBIDDEN_RESULT_BASENAME = "M11M_holdout64_fiveway_no_vggt_mixed1244_seed42_v1"


def _binding(path: str | Path) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"protocol binding is missing: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _payload_sha256(payload: dict[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "payload_sha256"}
    )


def validate_protocol_contract(
    path: str | Path, *, require_bound_files: bool = True
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = load_json(resolved)
    if payload.get("format") != FORMAT or payload.get("passed") is not True:
        raise RuntimeError(f"blind protocol did not pass: {resolved}")
    if payload.get("payload_sha256") != _payload_sha256(payload):
        raise RuntimeError("blind protocol payload hash changed")
    if (
        payload.get("blindness", {}).get("result_inspected_before_freeze") is not False
        or payload.get("blindness", {}).get("one_shot_joint_unblinding") is not True
        or payload.get("blindness", {}).get("old_fiveway_report_consumption_forbidden")
        is not True
    ):
        raise RuntimeError("blindness declaration is incomplete")
    evaluation = payload.get("evaluation", {})
    if (
        int(evaluation.get("object_count", -1)) != EXPECTED_OBJECTS
        or evaluation.get("seeds") != [EXPECTED_SEED]
        or int(evaluation.get("surface_samples", -1)) != 20000
        or evaluation.get("weights") != "ema"
        or evaluation.get("point_pose_sampling_must_match") is not True
        or evaluation.get("gt_alignment_fit_allowed") is not False
    ):
        raise RuntimeError("blind evaluation contract changed")
    forbidden = str(payload.get("forbidden_inputs", {}).get("old_fiveway_report", ""))
    if FORBIDDEN_RESULT_BASENAME not in forbidden:
        raise RuntimeError("old five-way result is not explicitly forbidden")
    if require_bound_files:
        for group_name in ("frozen_inputs", "implementation"):
            group = payload.get(group_name, {})
            if not group:
                raise RuntimeError(f"blind protocol has no {group_name} bindings")
            for label, binding in group.items():
                bound = Path(str(binding.get("path", ""))).resolve()
                if not bound.is_file() or sha256_file(bound) != binding.get("sha256"):
                    raise RuntimeError(f"blind protocol binding changed: {label}={bound}")
    return payload


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--split", required=True)
    freeze.add_argument("--raw_cache_report", required=True)
    freeze.add_argument("--reference_runtime_manifest", required=True)
    freeze.add_argument("--label_manifest", required=True)
    freeze.add_argument("--ss_checkpoint", required=True)
    freeze.add_argument("--slat_checkpoint", required=True)
    freeze.add_argument("--ss_contract", required=True)
    freeze.add_argument("--slat_contract", required=True)
    freeze.add_argument("--stock_slat_freeze", required=True)
    freeze.add_argument("--point_manifest", required=True)
    freeze.add_argument("--real_full_manifest", required=True)
    freeze.add_argument("--synthetic_full_manifest", required=True)
    freeze.add_argument("--reconviagen_manifest", required=True)
    freeze.add_argument("--pixal3d_manifest", required=True)
    freeze.add_argument("--pose_runtime_output", required=True)
    freeze.add_argument("--pose_model_output", required=True)
    freeze.add_argument("--pose_inference_output", required=True)
    freeze.add_argument("--pose_rebased_output", required=True)
    freeze.add_argument("--joint_report_output", required=True)
    freeze.add_argument("--old_fiveway_report", required=True)
    freeze.add_argument("--code_file", action="append", required=True)
    freeze.add_argument("--command_file", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--contract", required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "verify":
        payload = validate_protocol_contract(args.contract)
        print(
            {
                "passed": True,
                "payload_sha256": payload["payload_sha256"],
                "contract": str(Path(args.contract).resolve()),
            }
        )
        return

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite frozen blind protocol: {output}")
    result_outputs = {
        "M11N_pose_mask_runtime": str(Path(args.pose_runtime_output).resolve()),
        "M11O_dino_only_model_inputs": str(Path(args.pose_model_output).resolve()),
        "M11P_pose_mask_inference": str(Path(args.pose_inference_output).resolve()),
        "M11Q_reference_o_rebase": str(Path(args.pose_rebased_output).resolve()),
        "M11R_joint_sixway_report": str(Path(args.joint_report_output).resolve()),
    }
    existing = [path for path in result_outputs.values() if Path(path).exists()]
    if existing:
        raise RuntimeError(
            "blind outputs already exist before protocol freeze: " + ", ".join(existing)
        )
    frozen_inputs = {
        "holdout_split": _binding(args.split),
        "raw_cache_report": _binding(args.raw_cache_report),
        "reference_runtime_manifest": _binding(args.reference_runtime_manifest),
        "runtime_o_label_manifest": _binding(args.label_manifest),
        "mixed_no_vggt_ss_ema_checkpoint": _binding(args.ss_checkpoint),
        "mixed_no_vggt_slat_ema_checkpoint": _binding(args.slat_checkpoint),
        "ss_migration_contract": _binding(args.ss_contract),
        "slat_migration_contract": _binding(args.slat_contract),
        "stock_slat_freeze": _binding(args.stock_slat_freeze),
    }
    required_manifests = {
        "point_mask": str(Path(args.point_manifest).resolve()),
        "real_adapted_native_v2_full": str(Path(args.real_full_manifest).resolve()),
        "synthetic_parent_native_v2_full": str(
            Path(args.synthetic_full_manifest).resolve()
        ),
        "reconviagen_original": str(Path(args.reconviagen_manifest).resolve()),
        "pixal3d_official": str(Path(args.pixal3d_manifest).resolve()),
    }
    implementation = {
        f"code_{index:02d}": _binding(path)
        for index, path in enumerate(args.code_file, start=1)
    }
    implementation["commands"] = _binding(args.command_file)
    payload: dict[str, Any] = {
        "format": FORMAT,
        "frozen_at_utc": utc_now(),
        "purpose": "Formal Holdout64 Pose+Mask/no-point blind addendum",
        "blindness": {
            "result_inspected_before_freeze": False,
            "no_holdout_metric_log_mesh_screenshot_or_ranking_read": True,
            "one_shot_joint_unblinding": True,
            "old_fiveway_report_consumption_forbidden": True,
            "intermediate_metric_printing_forbidden": True,
        },
        "evaluation": {
            "object_count": EXPECTED_OBJECTS,
            "object_order": "exact M11C reference-runtime object order",
            "seeds": [EXPECTED_SEED],
            "weights": "ema",
            "amp_dtype": "bf16",
            "ss_sampling": "exact checkpoint SLat-upstream binding",
            "slat_sampling": {
                "steps": 25,
                "cfg_strength": 5.0,
                "cfg_interval": [0.5, 1.0],
                "guidance_rescale": 0.0,
                "rescale_t": 3.0,
            },
            "point_pose_sampling_must_match": True,
            "surface_samples": 20000,
            "surface_seed_policy": "seed*1009 + sorted_pair_position*9173",
            "primary_population": "all64",
            "sensitivity_populations": ["reliable", "low_confidence"],
            "gt_alignment_fit_allowed": False,
            "pose_rebase": "O_posemask -> W -> O_reference",
            "passed_semantics": "protocol and mesh completeness only",
        },
        "frozen_inputs": frozen_inputs,
        "required_existing_inference_manifests": required_manifests,
        "result_outputs": result_outputs,
        "forbidden_inputs": {
            "old_fiveway_report": str(Path(args.old_fiveway_report).resolve()),
            "point_cloud_fields": ["P_W", "P_O", "point_confidence"],
            "selection_inputs": ["GT metrics", "old meshes", "rankings", "screenshots"],
        },
        "implementation": implementation,
        "passed": True,
    }
    payload["payload_sha256"] = _payload_sha256(payload)
    atomic_json(output, payload)
    validate_protocol_contract(output)
    print(
        {
            "passed": True,
            "payload_sha256": payload["payload_sha256"],
            "contract": str(output),
        }
    )


if __name__ == "__main__":
    main()
