#!/usr/bin/env python3
"""Render textureless GT/Native-Full/Stock/Pixal3D Mesh comparisons."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bunny_review.common import (
    atomic_json,
    atomic_text,
    binding,
    canonical_sha256,
    validate_binding,
)
from .bunny_review.finalize import display_transform_from_mesh, load_mesh
from .compare_pixal3d_singleview_smoke import (
    pixal3d_mesh_path,
    pixal3d_result_path,
)
from .evaluate_direct_slat_pixal3d_utility import (
    case_canonical_transform,
    load_pixal_result,
    validate_transform,
)
from .prepare_native_ss_pixal3d_review import validate_native_protocol
from .render_direct_slat_fourway import (
    LATENT_DECODER_TO_REFERENCE,
    render_review_mode,
)


FORMAT = "pose_point_depth_mv.native_ss_pixal3d_fourway_normal_review.v1"
REVIEW_MODES = ("canonical_pose", "rigid_pose_gt", "shape_aligned")


def parse_review_modes(value: str) -> list[str]:
    requested = [item.strip() for item in str(value).split(",") if item.strip()]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("--review_modes must be non-empty and unique")
    unknown = [item for item in requested if item not in REVIEW_MODES]
    if unknown:
        raise ValueError(f"unsupported review modes: {unknown}")
    return [mode for mode in REVIEW_MODES if mode in set(requested)]


def select_case(protocol: dict[str, Any], uid: str, seed: int) -> dict[str, Any]:
    matches = [
        case
        for case in protocol["cases"]
        if str(case["uid"]) == uid and int(case["current_seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one frozen case for uid={uid!r}, seed={seed}; "
            f"found {len(matches)}"
        )
    return matches[0]


def make_html(
    *,
    output_dir: Path,
    case: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
) -> str:
    def relative(path: str | Path) -> str:
        return os.path.relpath(Path(path), output_dir).replace(os.sep, "/")

    sections = []
    for mode_id, comparison in comparisons.items():
        if mode_id == "canonical_pose":
            explanation = (
                "Primary review. All four Meshes retain their frozen canonical pose "
                "and share the one display transform derived from GT. No per-method "
                "centering, scaling, ICP, or autoframe is used."
            )
        elif mode_id == "rigid_pose_gt":
            explanation = (
                "GT-assisted pose-only inspection. Each prediction receives an "
                "independent proper SE(3) fit to GT; scale and reflection are "
                "forbidden. This view compares geometry after removing rigid pose."
            )
        else:
            explanation = (
                "GT-assisted shape inspection only. Each prediction receives a "
                "proper isotropic Sim(3) fit to GT; reflection is forbidden. This "
                "view cannot support camera-pose or AR-placement claims."
            )
        links = "".join(
            f'<li><strong>{html.escape(row["label"])}</strong>: '
            f'<a href="{html.escape(relative(row["mesh"]["path"]))}">Mesh</a> | '
            f'<a href="{html.escape(relative(row["contact_sheet"]["path"]))}">sheet</a></li>'
            for row in comparison["methods"]
        )
        sections.append(
            "<section>"
            f"<h2>{html.escape(comparison['label'])}</h2>"
            f"<p>{html.escape(explanation)}</p>"
            f'<p><a href="{html.escape(relative(comparison["normal_turntable"]["path"]))}">'
            "Side-by-side turntable</a></p>"
            f'<img src="{html.escape(relative(comparison["normal_contact_sheet"]["path"]))}">'
            f"<ul>{links}</ul>"
            "</section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Native SS four-way Mesh review</title>
<style>
body {{ background:#151515; color:#eee; font-family:sans-serif; margin:24px; }}
a {{ color:#7cc7ff; }} img {{ max-width:100%; border:1px solid #555; }}
section {{ margin:30px 0; padding:16px; background:#222; }}
</style>
</head>
<body>
<h1>GT | Native Full | Stock | Pixal3D</h1>
<p>UID: {html.escape(str(case['uid']))}; source: {html.escape(str(case['source']))};
views: {int(case['view_count'])}; seed: {int(case['current_seed'])}.</p>
<p>Every image is a textureless normal/clay render. Pixal3D is the official final
postprocessed GLB, never the intermediate decoded OBJ.</p>
{''.join(sections)}
</body>
</html>
"""


def validate_positive_arguments(args: argparse.Namespace) -> None:
    for name in (
        "render_frames",
        "render_resolution",
        "contact_frames",
        "fps",
        "candidate_samples",
        "alignment_samples",
        "candidate_iterations",
        "final_iterations",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if not 0.0 < float(args.display_margin) <= 1.0:
        raise ValueError("--display_margin must be in (0, 1]")


def run(args: argparse.Namespace) -> None:
    validate_positive_arguments(args)
    review_modes = parse_review_modes(args.review_modes)
    protocol_path = args.protocol.resolve()
    protocol = validate_native_protocol(protocol_path)
    protocol["_protocol_path"] = str(protocol_path)
    case = select_case(protocol, args.uid, int(args.seed))

    transform_path = args.pixal_transform.resolve()
    transform = validate_transform(transform_path)
    pixal_case_matrix = case_canonical_transform(case, transform)
    pixal_source_to_reference = (
        LATENT_DECODER_TO_REFERENCE
        @ np.asarray(pixal_case_matrix, dtype=np.float64)
    )
    pixal_result = load_pixal_result(protocol, case)
    pixal_path = pixal3d_mesh_path(protocol_path, str(case["case_id"])).resolve()
    if Path(str(pixal_result["mesh"])).resolve() != pixal_path:
        raise RuntimeError("Pixal3D official result path is inconsistent")

    target_path = validate_binding(case["target_mesh"], label="canonical GT Mesh")
    native_path = validate_binding(case["native_full_mesh"], label="Native Full Mesh")
    stock_path = validate_binding(case["stock_mesh"], label="Stock Mesh")
    validate_binding(case["pair_record"], label="T3 pair record")
    identity = np.eye(4, dtype=np.float64)
    method_specs = [
        {
            "selector": "reference",
            "method_position": 1,
            "method_id": "reference",
            "label": "GT Mesh",
            "mesh_path": target_path,
            "source_to_canonical": identity,
            "pose_status": "normalized source-GLB canonical frame; display owner",
        },
        {
            "selector": "native_full",
            "method_position": 2,
            "method_id": "native_full",
            "label": f"Native Full ({int(case['view_count'])} views)",
            "mesh_path": native_path,
            "source_to_canonical": LATENT_DECODER_TO_REFERENCE,
            "pose_status": (
                "T3 native branch exported with transform_pose=False; fixed vendored "
                "decoder-to-reference axis conversion, no GT fit"
            ),
        },
        {
            "selector": "stock",
            "method_position": 3,
            "method_id": "stock",
            "label": f"Stock ({int(case['view_count'])} views)",
            "mesh_path": stock_path,
            "source_to_canonical": LATENT_DECODER_TO_REFERENCE,
            "pose_status": (
                "T3 stock branch exported with transform_pose=False; fixed vendored "
                "decoder-to-reference axis conversion, no GT fit"
            ),
        },
        {
            "selector": "pixal3d_official_final",
            "method_position": 4,
            "method_id": "pixal3d_official_final",
            "label": "Pixal3D official final (1 view)",
            "mesh_path": pixal_path,
            "source_to_canonical": pixal_source_to_reference,
            "pose_status": (
                "official postprocessed GLB; frozen selected-c2w metadata transform "
                "and fixed decoder-to-reference axis conversion; no GT fit"
            ),
            "rigid_alignment_policy": "normalized_symmetric",
        },
    ]

    source_bindings = {
        "protocol": binding(protocol_path),
        "pixal_transform": binding(transform_path),
        "target_mesh": binding(target_path),
        "native_full_mesh": binding(native_path),
        "stock_mesh": binding(stock_path),
        "pixal3d_official_final_mesh": binding(pixal_path),
        "pixal3d_result": binding(
            pixal3d_result_path(protocol_path, str(case["case_id"]))
        ),
        "renderer_code": binding(Path(__file__).resolve()),
    }
    run_config = {
        "uid": str(case["uid"]),
        "object_uid": str(case["object_uid"]),
        "source": str(case["source"]),
        "view_count": int(case["view_count"]),
        "seed": int(args.seed),
        "review_modes": review_modes,
        "method_order": ["reference", "native_full", "stock", "pixal3d_official_final"],
        "render_frames": int(args.render_frames),
        "render_resolution": int(args.render_resolution),
        "contact_frames": int(args.contact_frames),
        "fps": int(args.fps),
        "display_margin": float(args.display_margin),
        "alignment_seed": int(args.alignment_seed),
        "candidate_samples": int(args.candidate_samples),
        "alignment_samples": int(args.alignment_samples),
        "candidate_iterations": int(args.candidate_iterations),
        "final_iterations": int(args.final_iterations),
        "pixal3d_case_canonical_matrix": np.asarray(
            pixal_case_matrix, dtype=np.float64
        ).tolist(),
        "latent_decoder_to_reference_matrix": LATENT_DECODER_TO_REFERENCE.tolist(),
        "sources": source_bindings,
    }
    run_config["config_sha256"] = canonical_sha256(run_config)

    output_dir = args.output_dir.resolve()
    report_path = output_dir / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("format") != FORMAT or report.get("run_config") != run_config:
            raise RuntimeError("existing review was produced with another config")
        validate_binding(report["html"], label="review HTML")
        for mode_id in review_modes:
            validate_binding(
                report["comparisons"][mode_id]["normal_contact_sheet"],
                label=f"{mode_id} contact sheet",
            )
            validate_binding(
                report["comparisons"][mode_id]["normal_turntable"],
                label=f"{mode_id} turntable",
            )
        print(
            json.dumps(
                {
                    "status": "reused",
                    "report": str(report_path),
                    "html": report["html"]["path"],
                    "contact_sheets": {
                        mode: report["comparisons"][mode]["normal_contact_sheet"]["path"]
                        for mode in review_modes
                    },
                },
                indent=2,
            )
        )
        return
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise RuntimeError(
            f"partial review exists; inspect it, then rerun with --resume: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("nvdiffrast normal renderer requires --device cuda")
    reference_mesh = load_mesh(target_path)
    shared_display_transform = display_transform_from_mesh(
        reference_mesh,
        float(args.display_margin),
        owner="reference",
    )
    del reference_mesh
    labels = {
        "canonical_pose": "Canonical pose / shared GT display frame",
        "rigid_pose_gt": "GT-assisted pose-only proper SE(3)",
        "shape_aligned": "GT-assisted shape-only proper Sim(3)",
    }
    comparisons = {
        mode_id: render_review_mode(
            mode_id=mode_id,
            mode_label=labels[mode_id],
            output_dir=output_dir,
            method_specs=method_specs,
            target_path=target_path,
            shared_display_transform=shared_display_transform,
            device=device,
            args=args,
        )
        for mode_id in review_modes
    }
    index_path = output_dir / "index.html"
    atomic_text(
        index_path,
        make_html(output_dir=output_dir, case=case, comparisons=comparisons),
    )
    report = {
        "format": FORMAT,
        "complete": True,
        "formal": False,
        "purpose": "textureless four-way human Mesh inspection; not checkpoint selection",
        "run_config": run_config,
        "comparisons": comparisons,
        "html": binding(index_path),
        "transform_gates": {
            "passed": True,
            "shared_display_matrix_owner": "GT Mesh",
            "per_method_display_normalization": False,
            "decoder_axis_conversion_fixed_and_not_gt_fitted": True,
            "pixal3d_official_final_glb_only": True,
            "pixal3d_intermediate_decoded_obj_forbidden": True,
            "pixal3d_canonical_transform_score_independent": True,
            "shape_alignment_gt_assisted": "shape_aligned" in review_modes,
            "rigid_pose_alignment_gt_assisted": "rigid_pose_gt" in review_modes,
            "rigid_pose_alignment_scale_forbidden": True,
            "shape_alignment_proper_no_reflection": True,
        },
        "render_policy": {
            "material": "textureless normal/clay",
            "camera": "shared fixed-pitch yaw turntable",
            "canonical_pose": (
                "one GT-derived bbox center/isotropic display matrix applied unchanged "
                "to all methods"
            ),
            "shape_aligned": (
                "proper isotropic Sim(3) against GT, separately labeled and excluded "
                "from pose/AR claims"
            ),
            "rigid_pose_gt": (
                "independent proper SE(3) against GT for each prediction; scale and "
                "reflection forbidden; excluded from pose/AR claims"
            ),
        },
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "status": "complete",
                "report": str(report_path),
                "html": str(index_path),
                "contact_sheets": {
                    mode: comparisons[mode]["normal_contact_sheet"]["path"]
                    for mode in review_modes
                },
            },
            indent=2,
        )
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pixal_transform", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--review_modes", default="canonical_pose,shape_aligned"
    )
    parser.add_argument("--render_frames", type=int, default=72)
    parser.add_argument("--render_resolution", type=int, default=320)
    parser.add_argument("--contact_frames", type=int, default=6)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--display_margin", type=float, default=0.9)
    parser.add_argument("--alignment_seed", type=int, default=20260802)
    parser.add_argument("--candidate_samples", type=int, default=1000)
    parser.add_argument("--alignment_samples", type=int, default=4000)
    parser.add_argument("--candidate_iterations", type=int, default=8)
    parser.add_argument("--final_iterations", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
