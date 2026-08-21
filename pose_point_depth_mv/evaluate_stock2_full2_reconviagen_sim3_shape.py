#!/usr/bin/env python3
"""Compare Chapter 93/94 ReconViaGen, Stock2, and Full2 after proper Sim(3).

This is a post-hoc shape-only supplement.  It scores the seed-42 intersection
of the Chapter 93 four-way report and the Chapter 94 Stock2/Full2 report.  Each
prediction receives the same deterministic, GT-assisted proper isotropic
similarity alignment.  Reflection and anisotropic scale are forbidden.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pose_point_depth_mv.compare_pixal3d_singleview_smoke import (
    canonical_sha256,
    load_mesh,
    sha256_file,
    similarity_icp,
    surface_metrics,
)


REPORT_FORMAT = "pose_point_depth_mv.stock2_full2_reconviagen_sim3_shape.v1"
CONFIG_FORMAT = "pose_point_depth_mv.stock2_full2_reconviagen_sim3_config.v1"
RECORD_FORMAT = "pose_point_depth_mv.stock2_full2_reconviagen_sim3_record.v1"
CHAPTER93_FORMAT = "pose_point_depth_mv.native_ss_pixal_genrecon_geometry.v1"
CHAPTER94_FORMAT = "pose_point_depth_mv.native_slat_genrecon_mesh.v2"
METHODS = ("reconviagen_original", "stock2", "full2")
COMPARISONS = (
    ("stock2_vs_reconviagen", "stock2", "reconviagen_original"),
    ("full2_vs_reconviagen", "full2", "reconviagen_original"),
    ("full2_vs_stock2", "full2", "stock2"),
)
TRACKS = ("raw_canonical", "sim3_shape_only")
METRICS = (
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
    "fscore_0p01",
    "fscore_0p02",
    "fscore_0p05",
    "normal_consistency",
)
LOWER_IS_BETTER = {
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def file_binding(path: str | Path, *, expected_sha256: str = "") -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = sha256_file(resolved)
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(f"file SHA256 changed: {resolved}")
    return {"path": str(resolved), "sha256": digest}


def _validate_report(payload: dict[str, Any], *, expected_format: str, label: str) -> None:
    if payload.get("format") != expected_format or payload.get("passed") is not True:
        raise RuntimeError(f"{label} report contract differs")
    if payload.get("formal") is not False:
        raise RuntimeError(f"{label} must retain formal=false")


def _safe_pair_id(value: Any) -> str:
    pair_id = str(value)
    if not pair_id or Path(pair_id).name != pair_id or pair_id in {".", ".."}:
        raise RuntimeError(f"unsafe Chapter 94 pair_id={pair_id!r}")
    return pair_id


def build_cases(
    chapter93_path: Path,
    chapter94_path: Path,
    *,
    seed: int,
    expected_objects: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chapter93 = load_json(chapter93_path)
    chapter94 = load_json(chapter94_path)
    _validate_report(chapter93, expected_format=CHAPTER93_FORMAT, label="Chapter 93")
    _validate_report(chapter94, expected_format=CHAPTER94_FORMAT, label="Chapter 94")
    if set(chapter93.get("methods", [])) != {
        "native_full",
        "stock",
        "pixal3d_official",
        "genrecon_official",
    }:
        raise RuntimeError("Chapter 93 method set changed")
    policy = chapter93.get("coordinate_policy", {})
    forbidden = set(policy.get("forbidden", []))
    if not {"per-method ICP", "per-method scaling", "reflection"}.issubset(forbidden):
        raise RuntimeError("Chapter 93 raw-coordinate policy changed")
    run_config = chapter94.get("run_config", {})
    if int(seed) not in [int(value) for value in run_config.get("seeds", [])]:
        raise RuntimeError(f"Chapter 94 does not contain seed={seed}")

    rows93: dict[str, dict[str, Any]] = {}
    for row in chapter93.get("records", []):
        uid = str(row.get("uid", ""))
        if not uid or uid in rows93:
            raise RuntimeError(f"invalid or duplicate Chapter 93 uid={uid!r}")
        if not str(row.get("case_id", "")).endswith(f"_s{int(seed)}"):
            raise RuntimeError("Chapter 93 report is not the requested seed slice")
        rows93[uid] = row

    rows94: dict[str, dict[str, Any]] = {}
    for row in chapter94.get("records", []):
        if int(row.get("seed", -1)) != int(seed):
            continue
        uid = str(row.get("uid", ""))
        if (
            not uid
            or uid in rows94
            or row.get("same_native_ss_coordinates") is not True
            or row.get("same_initial_noise") is not True
        ):
            raise RuntimeError(f"invalid Chapter 94 seed record uid={uid!r}")
        rows94[uid] = row

    common = sorted(set(rows93) & set(rows94))
    if len(common) != int(expected_objects):
        raise RuntimeError(
            f"Chapter 93/94 common object count changed: "
            f"expected={expected_objects}, actual={len(common)}"
        )

    cases = []
    for uid in common:
        row93 = rows93[uid]
        row94 = rows94[uid]
        target_record = row93.get("target_mesh", {})
        method_bindings = row93.get("method_bindings", {})
        recon_record = method_bindings.get("stock", {}).get("mesh", {})
        stock2_t3_record = method_bindings.get("native_full", {}).get("mesh", {})
        pair_id = _safe_pair_id(row94.get("pair_id"))
        pair_root = chapter94_path.parent / "mesh_pairs" / pair_id
        target = file_binding(
            target_record.get("path", ""),
            expected_sha256=str(target_record.get("sha256", "")),
        )
        recon = file_binding(
            recon_record.get("path", ""),
            expected_sha256=str(recon_record.get("sha256", "")),
        )
        stock2_t3 = file_binding(
            stock2_t3_record.get("path", ""),
            expected_sha256=str(stock2_t3_record.get("sha256", "")),
        )
        stock2 = file_binding(pair_root / "stock" / "mesh_canonical.obj")
        full2 = file_binding(pair_root / "full" / "mesh_canonical.obj")
        target94 = row94.get("target", {})
        if str(target94.get("frame", "")) != (
            "canonical latent frame; no per-branch normalization or ICP"
        ):
            raise RuntimeError(f"Chapter 94 target frame changed for uid={uid}")
        cases.append(
            {
                "uid": uid,
                "object_uid": str(row94.get("object_uid", "")),
                "seed": int(seed),
                "chapter93_case_id": str(row93.get("case_id", "")),
                "chapter94_pair_id": pair_id,
                "target": target,
                "methods": {
                    "reconviagen_original": recon,
                    # Use the Chapter 94 Stock2 replay so Stock2 and Full2 share
                    # Native-SS coordinates and initial SLat noise exactly.
                    "stock2": stock2,
                    "full2": full2,
                },
                "stock2_chapter93_replay": stock2_t3,
                "chapter94_target": {
                    key: target94.get(key)
                    for key in (
                        "source_glb_sha256",
                        "normalize_center",
                        "normalize_scale",
                        "canonical_margin",
                        "frame",
                    )
                },
            }
        )
    audit = {
        "chapter93_object_count": len(rows93),
        "chapter94_seed_object_count": len(rows94),
        "common_object_count": len(common),
        "chapter93_only_count": len(set(rows93) - set(rows94)),
        "chapter94_only_count": len(set(rows94) - set(rows93)),
        "common_uids": common,
    }
    return cases, audit


def audit_proper_sim3(alignment: dict[str, Any]) -> dict[str, Any]:
    matrix = np.asarray(alignment.get("matrix"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise RuntimeError("Sim(3) alignment returned an invalid matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise RuntimeError("Sim(3) alignment returned an invalid homogeneous row")
    linear = matrix[:3, :3]
    singular = np.linalg.svd(linear, compute_uv=False)
    if np.any(singular <= 0.0):
        raise RuntimeError("Sim(3) alignment returned a non-positive scale")
    anisotropy = float(singular.max() / singular.min())
    if anisotropy > 1.00001:
        raise RuntimeError(f"Sim(3) alignment is anisotropic: ratio={anisotropy}")
    scale = float(np.cbrt(np.linalg.det(linear)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("Sim(3) alignment contains reflection or invalid scale")
    rotation = linear / scale
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    rotation_determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1.0e-5 or not math.isclose(
        rotation_determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-5
    ):
        raise RuntimeError("Sim(3) alignment rotation is not proper SO(3)")
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return {
        **alignment,
        "proper_sim3_validated": True,
        "reflection": False,
        "anisotropic_scale": False,
        "isotropic_scale": scale,
        "anisotropy_ratio": anisotropy,
        "rotation_determinant": rotation_determinant,
        "rotation_orthogonality_max_abs": orthogonality_error,
        "rotation_angle_deg": float(np.degrees(np.arccos(cosine))),
        "translation_norm": float(np.linalg.norm(matrix[:3, 3])),
    }


def numeric_summary(
    values: Iterable[float], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("numeric summary requires finite values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(bootstrap_samples), len(array)))
    bootstrap_means = array[indices].mean(axis=1)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_rate": float(np.mean(array > 0.0)),
        "nonnegative_rate": float(np.mean(array >= 0.0)),
        "bootstrap_mean_95_ci": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
    }


def summarize_track(
    records: list[dict[str, Any]],
    *,
    track: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method_position, method in enumerate(METHODS):
        methods[method] = {}
        for metric_position, metric in enumerate(METRICS):
            values = [float(row["methods"][method][track][metric]) for row in records]
            methods[method][metric] = numeric_summary(
                values,
                bootstrap_samples=bootstrap_samples,
                seed=seed + method_position * 1000 + metric_position,
            )

    comparisons: dict[str, Any] = {}
    for comparison_position, (name, candidate, baseline) in enumerate(COMPARISONS):
        metrics = {}
        for metric_position, metric in enumerate(METRICS):
            deltas = []
            per_object = {}
            for row in records:
                candidate_value = float(row["methods"][candidate][track][metric])
                baseline_value = float(row["methods"][baseline][track][metric])
                delta = (
                    baseline_value - candidate_value
                    if metric in LOWER_IS_BETTER
                    else candidate_value - baseline_value
                )
                deltas.append(delta)
                per_object[str(row["uid"])] = delta
            summary = numeric_summary(
                deltas,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 10000 + comparison_position * 1000 + metric_position,
            )
            summary["candidate_win_count"] = int(sum(value > 0.0 for value in deltas))
            summary["tie_count"] = int(sum(value == 0.0 for value in deltas))
            summary["baseline_win_count"] = int(sum(value < 0.0 for value in deltas))
            summary["per_object"] = per_object
            metrics[metric] = summary
        comparisons[name] = {
            "candidate": candidate,
            "baseline": baseline,
            "positive_definition": "positive means candidate is better",
            "metrics": metrics,
        }
    return {"methods": methods, "comparisons": comparisons}


def summarize_alignments(
    records: list[dict[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    fields = (
        "isotropic_scale",
        "rotation_angle_deg",
        "translation_norm",
        "cost",
        "anisotropy_ratio",
    )
    output = {}
    for method_position, method in enumerate(METHODS):
        output[method] = {
            field: numeric_summary(
                [float(row["methods"][method]["alignment"][field]) for row in records],
                bootstrap_samples=bootstrap_samples,
                seed=seed + method_position * 100 + field_position,
            )
            for field_position, field in enumerate(fields)
        }
    return output


def summary_text(report: dict[str, Any]) -> str:
    lines = [
        "Stock2 / Full2 / ReconViaGen proper-Sim(3) shape-only supplement",
        "================================================================",
        f"formal: false  post_hoc: true  objects: {report['object_count']}",
        "positive paired values mean the named candidate is better",
    ]
    for track in TRACKS:
        lines.extend(("", f"[{track}]"))
        summary = report["summary"]["tracks"][track]
        for method in METHODS:
            metrics = summary["methods"][method]
            lines.append(
                f"{method}: L1={metrics['chamfer_l1']['mean']:.8f} "
                f"F02={metrics['fscore_0p02']['mean']:.8f} "
                f"Normal={metrics['normal_consistency']['mean']:.8f}"
            )
        for name, comparison in summary["comparisons"].items():
            metrics = comparison["metrics"]
            lines.append(
                f"{name}: L1={metrics['chamfer_l1']['mean']:+.8f} "
                f"win={metrics['chamfer_l1']['positive_rate']:.4f} "
                f"CI={metrics['chamfer_l1']['bootstrap_mean_95_ci']} "
                f"F02={metrics['fscore_0p02']['mean']:+.8f} "
                f"Normal={metrics['normal_consistency']['mean']:+.8f}"
            )
    lines.extend(
        (
            "",
            "Interpretation: sim3_shape_only removes global proper rotation,",
            "translation, and isotropic scale. It cannot support pose, scale, or AR",
            "placement claims. This common16 report is retrospective and formal=false.",
        )
    )
    return "\n".join(lines) + "\n"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter93_report", required=True)
    parser.add_argument("--chapter94_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected_objects", type=int, default=16)
    parser.add_argument("--candidate_samples", type=int, default=2000)
    parser.add_argument("--alignment_samples", type=int, default=10000)
    parser.add_argument("--candidate_iterations", type=int, default=12)
    parser.add_argument("--final_iterations", type=int, default=50)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--metric_seed", type=int, default=20260810)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    positive_values = (
        args.expected_objects,
        args.candidate_samples,
        args.alignment_samples,
        args.candidate_iterations,
        args.final_iterations,
        args.surface_samples,
        args.bootstrap_samples,
    )
    if any(int(value) <= 0 for value in positive_values):
        raise ValueError("all count and iteration arguments must be positive")

    chapter93_path = Path(args.chapter93_report).expanduser().resolve()
    chapter94_path = Path(args.chapter94_report).expanduser().resolve()
    cases, intersection_audit = build_cases(
        chapter93_path,
        chapter94_path,
        seed=int(args.seed),
        expected_objects=int(args.expected_objects),
    )
    implementation = file_binding(Path(__file__).resolve())
    config = {
        "format": CONFIG_FORMAT,
        "formal": False,
        "post_hoc": True,
        "scope": "chapter93_chapter94_seed42_common_objects",
        "chapter93_report": file_binding(chapter93_path),
        "chapter94_report": file_binding(chapter94_path),
        "intersection_audit": intersection_audit,
        "seed": int(args.seed),
        "object_count": len(cases),
        "methods": list(METHODS),
        "method_sources": {
            "reconviagen_original": "Chapter 93 Stock branch (Stock SS + Stock SLat)",
            "stock2": "Chapter 94 Stock branch (Native-SS v2 + Stock SLat)",
            "full2": "Chapter 94 Full branch (Native-SS v2 + Native-SLAT v2)",
        },
        "alignment": {
            "policy": "GT-assisted proper isotropic Sim(3), independently per method",
            "initializations": "24 proper cube rotations followed by similarity ICP",
            "reflection": False,
            "anisotropic_scale": False,
            "candidate_samples": int(args.candidate_samples),
            "alignment_samples": int(args.alignment_samples),
            "candidate_iterations": int(args.candidate_iterations),
            "final_iterations": int(args.final_iterations),
        },
        "evaluation": {
            "surface_samples": int(args.surface_samples),
            "thresholds": [0.01, 0.02, 0.05],
            "metric_seed": int(args.metric_seed),
            "alignment_and_metric_samples_independent": True,
            "bootstrap_samples": int(args.bootstrap_samples),
            "paired_resampling_unit": "object",
        },
        "cases": cases,
        "implementation": implementation,
    }
    config_sha256 = canonical_sha256(config)
    config["config_sha256"] = config_sha256

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.is_file():
        if load_json(config_path) != config:
            raise RuntimeError(f"existing run config differs: {config_path}")
    elif any(output_dir.iterdir()):
        raise RuntimeError(f"unbound output directory is not empty: {output_dir}")
    else:
        atomic_json(config_path, config)

    report_path = output_dir / "report.json"
    if report_path.is_file():
        report = load_json(report_path)
        if (
            report.get("format") != REPORT_FORMAT
            or report.get("config_sha256") != config_sha256
            or report.get("passed") is not True
        ):
            raise RuntimeError(f"existing report contract differs: {report_path}")
        print(summary_text(report), end="")
        print(json.dumps({"passed": True, "reused": True, "report": str(report_path)}))
        return

    thresholds = (0.01, 0.02, 0.05)
    records = []
    for position, case in enumerate(cases):
        record_path = output_dir / "records" / f"{position:02d}_{case['object_uid']}.json"
        if record_path.is_file():
            if not args.resume:
                raise RuntimeError(f"record exists; pass --resume: {record_path}")
            record = load_json(record_path)
            if (
                record.get("format") != RECORD_FORMAT
                or record.get("config_sha256") != config_sha256
                or record.get("uid") != case["uid"]
                or record.get("passed") is not True
            ):
                raise RuntimeError(f"stale shape record: {record_path}")
            records.append(record)
            print(f"[sim3_shape:reuse] {position + 1}/{len(cases)} uid={case['uid']}", flush=True)
            continue

        target = load_mesh(case["target"]["path"])
        alignment_seed = int(args.metric_seed) + position * 100003
        metric_seed = int(args.metric_seed) + 50_000_000 + position * 100003
        method_rows = {}
        for method in METHODS:
            mesh_binding = case["methods"][method]
            mesh = load_mesh(mesh_binding["path"])
            raw = surface_metrics(
                mesh,
                target,
                count=int(args.surface_samples),
                seed=metric_seed,
                thresholds=thresholds,
            )
            aligned, alignment = similarity_icp(
                mesh,
                target,
                seed=alignment_seed,
                candidate_samples=int(args.candidate_samples),
                final_samples=int(args.alignment_samples),
                candidate_iterations=int(args.candidate_iterations),
                final_iterations=int(args.final_iterations),
            )
            audited_alignment = audit_proper_sim3(alignment)
            shape = surface_metrics(
                aligned,
                target,
                count=int(args.surface_samples),
                seed=metric_seed,
                thresholds=thresholds,
            )
            method_rows[method] = {
                "mesh": mesh_binding,
                "raw_canonical": {key: float(raw[key]) for key in METRICS},
                "alignment": audited_alignment,
                "sim3_shape_only": {key: float(shape[key]) for key in METRICS},
            }
            del mesh, aligned

        record = {
            "format": RECORD_FORMAT,
            "created_at_utc": utc_now(),
            "config_sha256": config_sha256,
            "uid": case["uid"],
            "object_uid": case["object_uid"],
            "seed": int(args.seed),
            "target": case["target"],
            "alignment_seed": alignment_seed,
            "metric_seed": metric_seed,
            "methods": method_rows,
            "passed": True,
        }
        atomic_json(record_path, record)
        records.append(record)
        print(f"[sim3_shape] {position + 1}/{len(cases)} uid={case['uid']}", flush=True)
        del target

    tracks = {
        track: summarize_track(
            records,
            track=track,
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.metric_seed) + track_position * 100000,
        )
        for track_position, track in enumerate(TRACKS)
    }
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": len(records) == len(cases),
        "formal": False,
        "post_hoc": True,
        "interpretation_scope": (
            "sim3_shape_only removes global proper rotation, translation, and "
            "isotropic scale; it cannot support pose, scale, or AR placement claims"
        ),
        "config": str(config_path),
        "config_sha256": config_sha256,
        "object_count": len(cases),
        "record_count": len(records) * len(METHODS),
        "summary": {
            "tracks": tracks,
            "alignments": summarize_alignments(
                records,
                bootstrap_samples=int(args.bootstrap_samples),
                seed=int(args.metric_seed) + 900000,
            ),
        },
        "records": records,
    }
    atomic_json(report_path, report)
    text = summary_text(report)
    atomic_text(output_dir / "summary.txt", text)
    print(text, end="")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "formal": report["formal"],
                "objects": report["object_count"],
                "records": report["record_count"],
                "report": str(report_path),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
