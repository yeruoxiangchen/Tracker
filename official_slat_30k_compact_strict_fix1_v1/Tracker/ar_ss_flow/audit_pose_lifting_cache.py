#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any

import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
if str(TRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKER_ROOT))

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset, volume_from_sample
from ar_ss_flow.pose_lifting import build_projection_geometry, projection_roundtrip_audit


POSE_LIFTING_CACHE_AUDIT_VERSION = "ar_ss_flow.pose_lifting_cache_audit.v2"
CACHED_GEOMETRY_FIELDS = (
    "image_grid",
    "patch_grid",
    "camera_depth",
    "valid",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def rebuild_cached_geometry_on_cpu(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Reproduce the geometry cache on the device used by the cache builder."""

    return build_projection_geometry(
        intrinsics=sample["intrinsics"].detach().to(device="cpu", dtype=torch.float32),
        extrinsics=sample["extrinsics"].detach().to(device="cpu", dtype=torch.float32),
        grid_transform=str(sample["grid_transform"]),
        extrinsics_type=str(sample["extrinsics_type"]),
        camera_forward_sign=float(sample["camera_forward_sign"]),
        image_height=int(sample["predicted_depth"].shape[-2]),
        image_width=int(sample["predicted_depth"].shape[-1]),
        patch_grid_side=int(
            round(sample["visual_patch_features"].shape[1] ** 0.5)
        ),
        volume_side=16,
    )


def compare_cached_geometry(
    cached: Any,
    fresh: dict[str, torch.Tensor],
    *,
    max_abs_diff: float,
) -> dict[str, Any]:
    """Compare cached geometry with a same-device CPU reconstruction.

    Geometry is created on CPU when the cache is built. Reconstructing it on
    CUDA changes FP32 projection coordinates by a few ulps and can cross the
    hard support threshold used by lifting. Comparing the source geometry on
    CPU detects cache corruption without conflating it with device numerics.
    """

    required = set(CACHED_GEOMETRY_FIELDS)
    if not isinstance(cached, dict):
        return {
            "present": False,
            "keys_equal": False,
            "missing_keys": list(CACHED_GEOMETRY_FIELDS),
            "unexpected_keys": [],
            "max_abs_diff": None,
            "valid_mismatch_count": None,
            "fields": {},
            "passed": False,
        }

    cached_keys = set(cached)
    missing = sorted(required - cached_keys)
    unexpected = sorted(cached_keys - required)
    fields: dict[str, dict[str, Any]] = {}
    numeric_differences: list[float] = []
    valid_mismatch_count: int | None = None
    all_fields_pass = not missing and not unexpected

    for name in CACHED_GEOMETRY_FIELDS:
        if name not in cached or name not in fresh:
            all_fields_pass = False
            continue
        cached_value = cached[name]
        fresh_value = fresh[name]
        if not isinstance(cached_value, torch.Tensor):
            fields[name] = {
                "is_tensor": False,
                "passed": False,
            }
            all_fields_pass = False
            continue
        cached_cpu = cached_value.detach().cpu()
        fresh_cpu = fresh_value.detach().cpu()
        shape_equal = tuple(cached_cpu.shape) == tuple(fresh_cpu.shape)
        dtype_equal = cached_cpu.dtype == fresh_cpu.dtype
        metrics: dict[str, Any] = {
            "is_tensor": True,
            "cached_shape": list(cached_cpu.shape),
            "fresh_shape": list(fresh_cpu.shape),
            "cached_dtype": str(cached_cpu.dtype),
            "fresh_dtype": str(fresh_cpu.dtype),
            "shape_equal": shape_equal,
            "dtype_equal": dtype_equal,
        }
        if name == "valid":
            mismatch_count = (
                int(torch.ne(cached_cpu, fresh_cpu).sum().item())
                if shape_equal
                else None
            )
            valid_mismatch_count = mismatch_count
            field_passed = bool(
                shape_equal and dtype_equal and mismatch_count == 0
            )
            metrics.update(
                {
                    "mismatch_count": mismatch_count,
                    "passed": field_passed,
                }
            )
        else:
            finite = bool(
                torch.isfinite(cached_cpu.float()).all().item()
                and torch.isfinite(fresh_cpu.float()).all().item()
            )
            difference = (
                float(
                    (cached_cpu.float() - fresh_cpu.float())
                    .abs()
                    .max()
                    .item()
                )
                if shape_equal and finite
                else None
            )
            if difference is not None:
                numeric_differences.append(difference)
            field_passed = bool(
                shape_equal
                and dtype_equal
                and finite
                and difference is not None
                and difference <= float(max_abs_diff)
            )
            metrics.update(
                {
                    "finite": finite,
                    "max_abs_diff": difference,
                    "passed": field_passed,
                }
            )
        fields[name] = metrics
        all_fields_pass = all_fields_pass and field_passed

    return {
        "present": True,
        "keys_equal": not missing and not unexpected,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "max_abs_diff": max(numeric_differences) if numeric_differences else None,
        "valid_mismatch_count": valid_mismatch_count,
        "fields": fields,
        "passed": bool(all_fields_pass),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit reusable pose-lifting cache and geometry.")
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--indices", default="all")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min_depth_enabled_ratio", type=float, default=0.80)
    parser.add_argument("--max_cached_geometry_diff", type=float, default=2.0e-3)
    parser.add_argument("--max_roundtrip_error", type=float, default=1.0e-4)
    parser.add_argument("--fail_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = PoseLiftingCacheDataset(args.cache_manifest, indices=args.indices)
    manifest_path = Path(args.cache_manifest).resolve()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_rows = list(manifest_payload.get("samples", ()))
    count = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index in range(count):
        sample = dataset[index]
        uid = str(sample["uid"])
        try:
            # The cache builder creates correct_geometry from CPU tensors. Keep
            # the reconstruction on CPU as well, then feed both cached and
            # fresh CPU grids through the requested runtime device. This makes
            # the comparison about cache fidelity, not CPU/CUDA FP32 ulps.
            cached_geometry = sample.get("correct_geometry")
            fresh_geometry = rebuild_cached_geometry_on_cpu(sample)
            geometry_audit = compare_cached_geometry(
                cached_geometry,
                fresh_geometry,
                max_abs_diff=float(args.max_cached_geometry_diff),
            )
            volume_cached, metadata_cached, stats = volume_from_sample(
                sample, device=device, mode="correct"
            )
            roundtrip = projection_roundtrip_audit(
                intrinsics=sample["intrinsics"].to(
                    device="cpu", dtype=torch.float32
                ),
                extrinsics=sample["extrinsics"].to(
                    device="cpu", dtype=torch.float32
                ),
                grid_transform=str(sample["grid_transform"]),
                extrinsics_type=str(sample["extrinsics_type"]),
                camera_forward_sign=float(sample["camera_forward_sign"]),
                image_height=int(sample["predicted_depth"].shape[-2]),
                image_width=int(sample["predicted_depth"].shape[-1]),
                patch_grid_side=int(
                    round(sample["visual_patch_features"].shape[1] ** 0.5)
                ),
            )
            if isinstance(cached_geometry, dict):
                fresh_sample = dict(sample)
                fresh_sample["correct_geometry"] = fresh_geometry
                volume_fresh, metadata_fresh, _ = volume_from_sample(
                    fresh_sample, device=device, mode="correct"
                )
                volume_diff = float(
                    (volume_cached.float() - volume_fresh.float()).abs().max().item()
                )
                metadata_diff = float(
                    (metadata_cached.float() - metadata_fresh.float()).abs().max().item()
                )
            else:
                volume_diff = None
                metadata_diff = None
            sample_checks = {
                "cached_geometry_matches_cpu_rebuild": bool(
                    geometry_audit["passed"]
                ),
                "cached_geometry_volume_replay": bool(
                    volume_diff is not None
                    and volume_diff <= float(args.max_cached_geometry_diff)
                ),
                "cached_geometry_metadata_replay": bool(
                    metadata_diff is not None
                    and metadata_diff <= float(args.max_cached_geometry_diff)
                ),
                "positive_voxel_support": stats["supported_voxel_ratio"] > 0.0,
                "nonzero_visual_volume": stats["visual_rms"] > 0.0,
                "valid_projection": roundtrip["valid_projection_count"] > 0,
                "projection_roundtrip": roundtrip["max_error"]
                <= float(args.max_roundtrip_error),
            }
            failed_checks = [
                name for name, passed in sample_checks.items() if not passed
            ]
            row = {
                "uid": uid,
                "depth_enabled": bool(sample["depth_calibration"].get("enabled", False)),
                "depth_match_count": int(sample["depth_calibration"].get("match_count", 0)),
                "depth_selected_model": sample["depth_calibration"].get("selected_model"),
                "depth_median_abs_residual": sample["depth_calibration"].get(
                    "median_abs_residual"
                ),
                "depth_p90_abs_residual": sample["depth_calibration"].get(
                    "p90_abs_residual"
                ),
                "cached_geometry_volume_max_abs_diff": volume_diff,
                "cached_geometry_metadata_max_abs_diff": metadata_diff,
                "cached_geometry_direct_max_abs_diff": geometry_audit[
                    "max_abs_diff"
                ],
                "cached_geometry_valid_mismatch_count": geometry_audit[
                    "valid_mismatch_count"
                ],
                "cached_geometry_audit": geometry_audit,
                "supported_voxel_ratio": stats["supported_voxel_ratio"],
                "visual_rms": stats["visual_rms"],
                "projection_roundtrip": roundtrip,
                "checks": sample_checks,
                "failed_checks": failed_checks,
                "passed": not failed_checks,
            }
            if not row["passed"]:
                failures.append(
                    {
                        "uid": uid,
                        "error": "sample audit threshold failed",
                        "failed_checks": failed_checks,
                    }
                )
            rows.append(row)
        except Exception as error:
            failures.append(
                {
                    "uid": uid,
                    "error": repr(error),
                    "failed_checks": ["exception"],
                }
            )
    depth_enabled_ratio = (
        mean(float(row["depth_enabled"]) for row in rows) if rows else 0.0
    )
    checks = {
        "samples_loaded": len(rows) == count,
        "sample_audits_passed": not failures,
        "depth_calibration_coverage": depth_enabled_ratio
        >= float(args.min_depth_enabled_ratio),
        "no_nonfinite_or_missing_inputs": not any("non-finite" in row["error"] for row in failures),
    }
    manifest_sha256 = sha256_file(manifest_path)
    report = {
        "audit_version": POSE_LIFTING_CACHE_AUDIT_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "comparison_protocol": {
            "geometry_rebuild_device": "cpu",
            "projection_roundtrip_device": "cpu",
            "runtime_volume_device": str(device),
            "cached_and_fresh_volume_use_cpu_built_geometry": True,
        },
        "thresholds": {
            "min_depth_enabled_ratio": float(args.min_depth_enabled_ratio),
            "max_cached_geometry_diff": float(args.max_cached_geometry_diff),
            "max_roundtrip_error": float(args.max_roundtrip_error),
        },
        "sample_count": count,
        "audited_count": len(rows),
        "depth_calibration_enabled_ratio": depth_enabled_ratio,
        "depth_fallback_count": sum(not row["depth_enabled"] for row in rows),
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": manifest_sha256,
        # Admission historically looks for this shorter spelling. Keep both
        # names so old report readers remain compatible while binding content.
        "manifest_sha256": manifest_sha256,
        "cache_config_hash": str(manifest_payload.get("config_hash", "")),
        "cache_schema_hash": json_hash({
            key: manifest_payload.get(key)
            for key in (
                "format", "stock_condition_source", "lifting_feature_source",
                "visual_feature_dim", "feature_metadata", "metadata_names",
                "metadata_schema_hash", "config", "config_hash",
            )
        }),
        "uid_hash": json_hash(sorted(str(row.get("uid", "")) for row in all_rows)),
        "object_uid_hash": json_hash(sorted({
            str(row.get("object_uid", row.get("uid", ""))) for row in all_rows
        })),
        "failures": failures,
        "samples": rows,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "failed_samples.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )
    markdown = [
        "# Pose Lifting Cache Audit",
        "",
        f"- passed: `{report['passed']}`",
        f"- audited: `{len(rows)}/{count}`",
        f"- depth calibration enabled ratio: `{depth_enabled_ratio:.6f}`",
        f"- explicit mask/visual-hull fallback count: `{report['depth_fallback_count']}`",
        f"- failed samples: `{len(failures)}`",
        f"- audit version: `{POSE_LIFTING_CACHE_AUDIT_VERSION}`",
        "- geometry comparison: cached CPU geometry vs fresh CPU reconstruction",
        f"- runtime volume device: `{device}`",
        "",
        "Depth calibration failure never receives a default scale. Those samples explicitly disable depth consistency.",
    ]
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "passed", "sample_count", "audited_count", "depth_calibration_enabled_ratio", "depth_fallback_count"
    )}, indent=2))
    if args.fail_on_error and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
