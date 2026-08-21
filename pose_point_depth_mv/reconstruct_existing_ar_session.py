#!/usr/bin/env python3
"""Materialize and reconstruct an existing phone AR capture session.

This entry point is intentionally independent from the mutable ``current_session``
owned by the phone server.  It lets an older RGB/mask/pose session enter the
current pose+mask reconstruction without changing or stopping a newer capture.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pose_point_depth_mv.ar_object_capture import (
    ARPointFilterConfig,
    finalize_ar_capture,
)
from pose_point_depth_mv.ar_object_reconstruction import (
    ARReconstructionConfig,
    run_ar_object_frame_selection,
    run_ar_object_reconstruction,
)


TRACKER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = TRACKER_ROOT / "pose_point_depth_mv/outputs/可视AR"


def _existing_session_inputs(
    output_root: Path, session_id: str
) -> tuple[Path, Path, Path, list[str]]:
    runtime = output_root / "runtime"
    data_dir = runtime / "data" / session_id
    mask_dir = runtime / "masks" / session_id
    preview_dir = runtime / "previews" / session_id
    if not data_dir.is_dir():
        raise FileNotFoundError(f"session data directory is missing: {data_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"session mask directory is missing: {mask_dir}")
    frame_names = sorted(path.name for path in data_dir.glob("frame_*.jpg"))
    if len(frame_names) < 2:
        raise RuntimeError(f"session has fewer than two RGB frames: {data_dir}")
    missing_masks = [
        name for name in frame_names if not (mask_dir / f"{Path(name).stem}.png").is_file()
    ]
    if missing_masks:
        raise FileNotFoundError(
            f"session is missing {len(missing_masks)} masks; first={missing_masks[0]}"
        )
    for required in (data_dir / "poses.txt", data_dir / "frame_metadata.jsonl"):
        if not required.is_file():
            raise FileNotFoundError(f"session metadata is missing: {required}")
    return data_dir, mask_dir, preview_dir, frame_names


def _input_qc(
    *,
    frame_names: list[str],
    data_dir: Path,
    preview_dir: Path,
) -> dict[str, Any]:
    all_indices = list(range(len(frame_names)))
    failures = []
    if len(frame_names) < 8:
        failures.append(f"candidate frame count {len(frame_names)} < 8")
    return {
        "profile": "defer_cross_view_geometry_to_segmented_runtime_final8_v1",
        "qc_pass": not failures,
        "fail_reasons": failures,
        "warnings": [],
        "reconstruction_candidate_policy": "all_uploaded_rgb_mask_pose_frames",
        "reconstruction_candidate_count": len(all_indices),
        "reconstruction_candidate_indices": all_indices,
        "authoritative_selection_stage": "runtime_o_segmented_all_candidates_to_final8",
        "deferred_checks": [
            "AR pose continuity segmentation",
            "per-segment mask-ray object center",
            "per-segment azimuth-balanced final-view selection",
            "final selected-view ray residual and azimuth coverage",
        ],
        "recovered_existing_session": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume an existing phone RGB/mask/pose session without AR points"
    )
    parser.add_argument("--session_id", required=True)
    parser.add_argument(
        "--source_session_id",
        help=(
            "read RGB/mask/pose files from this existing phone session while "
            "writing an immutable dataset/reconstruction under --session_id"
        ),
    )
    parser.add_argument(
        "--frame_name",
        action="append",
        default=[],
        help=(
            "freeze one source frame into the reconstruction candidate set; "
            "repeat exactly eight times to reconstruct a manually selected set"
        ),
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu", default=os.environ.get("GPU", "4"))
    parser.add_argument(
        "--native_ss_cfg_strength",
        type=float,
        default=3.0,
        help="Native SS CFG for the selected SLat backend (current mixed value: 3.0)",
    )
    parser.add_argument(
        "--slat_backend",
        choices=("native_v2", "stock"),
        default="native_v2",
        help=(
            "native_v2 keeps the current phone model; stock uses the same Native SS "
            "support with the frozen released Stock SLat and Mesh decoder"
        ),
    )
    parser.add_argument(
        "--capture_only",
        action="store_true",
        help="only create the immutable pose+mask dataset; do not run GPU reconstruction",
    )
    parser.add_argument(
        "--selection_only",
        action="store_true",
        help=(
            "materialize all candidate frames and run the exact pose+mask "
            "runtime-O all-to-8 selector/quality gate, then stop before DINO/SS/SLat"
        ),
    )
    parser.add_argument(
        "--diagnostic_bypass_pose_mask_qc",
        action="store_true",
        help=(
            "continue a manually frozen diagnostic case through capture and "
            "runtime quality failures; reports remain explicitly non-formal"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.capture_only and args.selection_only:
        raise ValueError("--capture_only and --selection_only are mutually exclusive")
    output_root = args.output_root.expanduser().resolve()
    source_session_id = str(args.source_session_id or args.session_id)
    data_dir, mask_dir, preview_dir, available_frame_names = _existing_session_inputs(
        output_root, source_session_id
    )
    if args.frame_name:
        frame_names = [str(name) for name in args.frame_name]
        if len(frame_names) != len(set(frame_names)):
            raise ValueError("--frame_name values must be unique")
        missing = [name for name in frame_names if name not in available_frame_names]
        if missing:
            raise FileNotFoundError(
                f"requested source frames are missing from {source_session_id}: {missing}"
            )
        if len(frame_names) < 8:
            raise ValueError("manual reconstruction requires at least eight --frame_name values")
    else:
        frame_names = available_frame_names
    input_qc = _input_qc(
        frame_names=frame_names,
        data_dir=data_dir,
        preview_dir=preview_dir,
    )
    if (
        input_qc.get("qc_pass") is not True
        and not args.diagnostic_bypass_pose_mask_qc
    ):
        reasons = "; ".join(input_qc.get("fail_reasons") or ["unknown reason"])
        raise RuntimeError(f"existing session failed input QC: {reasons}")

    capture_config = (
        ARPointFilterConfig(
            max_ray_residual_median_over_mask_extent=1.0e6,
            max_ray_residual_p90_over_mask_extent=1.0e6,
            min_orbit_gravity_agreement=0.0,
            min_synchronized_frame_ratio=0.0,
        )
        if args.diagnostic_bypass_pose_mask_qc
        else ARPointFilterConfig()
    )
    capture = finalize_ar_capture(
        session_id=str(args.session_id),
        data_dir=data_dir,
        mask_dir=mask_dir,
        frame_names=frame_names,
        output_root=output_root,
        input_qc=input_qc,
        config=capture_config,
        require_point_cloud=False,
    )
    result: dict[str, Any] = {
        "passed": True,
        "session_id": str(args.session_id),
        "source_session_id": source_session_id,
        "geometry_mode": "pose_mask",
        "slat_backend": str(args.slat_backend),
        "point_cloud_consumed": False,
        "candidate_frame_count": len(frame_names),
        "candidate_frame_names": frame_names,
        "input_qc_passed": bool(input_qc.get("qc_pass")),
        "diagnostic_bypass_pose_mask_qc": bool(
            args.diagnostic_bypass_pose_mask_qc
        ),
        "formal_input_claim_allowed": not bool(
            args.diagnostic_bypass_pose_mask_qc
        ),
        "capture_report": str(
            (Path(capture["dataset_dir"]) / "capture_report.json").resolve()
        ),
    }
    if args.selection_only:
        selection = run_ar_object_frame_selection(
            session_id=str(args.session_id),
            dataset_dir=Path(capture["dataset_dir"]),
            output_root=output_root,
            config=ARReconstructionConfig(
                gpu=str(args.gpu),
                geometry_mode="pose_mask",
                slat_backend=str(args.slat_backend),
                native_ss_cfg_strength=float(args.native_ss_cfg_strength),
                diagnostic_bypass_pose_mask_quality=bool(
                    args.diagnostic_bypass_pose_mask_qc
                ),
            ),
        )
        result["frame_selection_report"] = str(
            (
                output_root
                / "reconstructions"
                / str(args.session_id)
                / "phone_frame_selection_report.json"
            ).resolve()
        )
        result["selected_frame_names"] = selection["selected_frame_names"]
        result["input_quality"] = selection["input_quality"]
        result["stopped_before_model_inference"] = True
    elif not args.capture_only:
        reconstruction = run_ar_object_reconstruction(
            session_id=str(args.session_id),
            dataset_dir=Path(capture["dataset_dir"]),
            output_root=output_root,
            config=ARReconstructionConfig(
                gpu=str(args.gpu),
                geometry_mode="pose_mask",
                slat_backend=str(args.slat_backend),
                native_ss_cfg_strength=float(args.native_ss_cfg_strength),
                diagnostic_bypass_pose_mask_quality=bool(
                    args.diagnostic_bypass_pose_mask_qc
                ),
            ),
        )
        result["reconstruction_report"] = str(
            (
                output_root
                / "reconstructions"
                / str(args.session_id)
                / "reconstruction_report.json"
            ).resolve()
        )
        result["meshes"] = reconstruction["meshes"]
        result["previews"] = reconstruction["previews"]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
