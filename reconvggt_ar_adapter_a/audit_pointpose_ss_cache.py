#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch

TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reconvggt_ar_adapter_a.build_pointpose_ss_cache import (  # noqa: E402
    frame_paths,
    grid_centers,
    load_and_validate_target,
    make_physical_grid,
    physical_grid_audit,
    require_unique_uids,
    validate_prior_arrays,
)
from reconvggt_ar_adapter_a.pointpose_ss_condition import (  # noqa: E402
    PHYSICAL_FEATURE_NAMES,
    feature_schema_hash,
)
from trellis_point_prior_mv.common import load_manifest, resolve_path  # noqa: E402


def overlap(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    pred_set = {tuple(map(int, row[-3:])) for row in pred}
    target_set = {tuple(map(int, row[-3:])) for row in target}
    intersection = len(pred_set & target_set)
    union = len(pred_set | target_set)
    return {
        "pred_count": len(pred_set),
        "target_count": len(target_set),
        "iou": float(intersection / union) if union else 1.0,
        "recall": float(intersection / len(target_set)) if target_set else 1.0,
        "precision": float(intersection / len(pred_set)) if pred_set else 1.0,
    }


def mapping_audit() -> dict[str, Any]:
    points, xyz = grid_centers(16)
    checks = []
    for index, expected in (
        (0, (-0.46875, -0.46875, -0.46875)),
        (1, (-0.46875, -0.46875, -0.40625)),
        (16, (-0.46875, -0.40625, -0.46875)),
        (256, (-0.40625, -0.46875, -0.46875)),
        (4095, (0.46875, 0.46875, 0.46875)),
    ):
        actual = tuple(float(value) for value in points[index])
        xyz_actual = tuple(float(xyz[axis].reshape(-1)[index]) for axis in range(3))
        checks.append(
            {
                "index": index,
                "point": actual,
                "expected": expected,
                "xyz_feature": xyz_actual,
                "passed": bool(np.allclose(actual, expected) and np.allclose(xyz_actual, np.asarray(expected) * 2.0)),
            }
        )
    token = np.zeros((16, 16, 16), dtype=np.float32)
    token.reshape(-1)[1] = 1.0
    upsampled = np.repeat(np.repeat(np.repeat(token, 4, axis=0), 4, axis=1), 4, axis=2)
    expected = np.zeros((64, 64, 64), dtype=np.float32)
    expected[0:4, 0:4, 4:8] = 1.0
    downsampled = upsampled.reshape(16, 4, 16, 4, 16, 4).max(axis=(1, 3, 5))
    tensor_roundtrip = bool(
        np.array_equal(upsampled, expected)
        and np.count_nonzero(downsampled) == 1
        and downsampled.reshape(-1)[1] == 1.0
    )
    return {
        "checks": checks,
        "tensor_roundtrip_token_index": 1,
        "tensor_roundtrip_passed": tensor_roundtrip,
        "passed": all(row["passed"] for row in checks) and tensor_roundtrip,
    }


def load_decoder(pretrained: str, device: torch.device):
    from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline

    # Cache auditing only needs the SS decoder. Loading TrellisVGGTTo3DPipeline
    # would also instantiate VGGT and image-side models and can exhaust host RAM.
    pipeline = TrellisImageTo3DPipeline.from_pretrained(pretrained)
    decoder = pipeline.models["sparse_structure_decoder"].to(device).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad = False
    return decoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent PointPose SS cache audit.")
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--check_decoder", action="store_true")
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decoder_min_iou", type=float, default=0.99)
    parser.add_argument("--recompute_max_abs_diff", type=float, default=0.001)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    cache_path = Path(args.cache_manifest)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("format") != "reconvggt.pointpose_ss_cache.v1":
        raise ValueError(f"unexpected cache format: {cache.get('format')}")
    if tuple(cache.get("feature_names", [])) != PHYSICAL_FEATURE_NAMES:
        raise ValueError("cache feature_names do not match the code schema")
    if cache.get("feature_schema_hash") != feature_schema_hash():
        raise ValueError("cache feature_schema_hash does not match the code schema")
    samples = cache.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("cache manifest has no samples")
    cache_by_uid = require_unique_uids(samples, label="cache manifest")
    source_payload, source_samples = load_manifest(cache["source_manifest"])
    prior_payload, prior_samples = load_manifest(cache["prior_manifest"])
    if source_payload.get("format") != "pixal3d_multiview.objaverse_sparse.v1":
        raise ValueError(f"unexpected source manifest format: {source_payload.get('format')}")
    if prior_payload.get("format") != "trellis_point_prior_mv_v1":
        raise ValueError(f"unexpected prior manifest format: {prior_payload.get('format')}")
    source_by_uid = require_unique_uids(source_samples, label="source manifest")
    prior_by_uid = require_unique_uids(prior_samples, label="prior manifest")
    cache_root = Path(cache.get("output_dir", cache_path.parent))
    prior_root = prior_payload.get("prior_root", prior_payload.get("output_dir"))
    latent_root = source_payload.get("latent_root")
    if not prior_root or not latent_root:
        raise ValueError(f"missing prior_root/latent_root: prior={prior_root} latent={latent_root}")
    rows = list(cache_by_uid.values())
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]

    decoder = load_decoder(args.pretrained, torch.device(args.device)) if args.check_decoder else None
    failures: list[dict[str, Any]] = []
    sample_reports: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    decoder_ious: list[float] = []

    for index, row in enumerate(rows):
        uid = str(row["uid"])
        try:
            source = source_by_uid[uid]
            prior_sample = prior_by_uid[uid]
            physical_path = resolve_path(str(cache_root), row["physical_grid"])
            if not physical_path.is_file():
                raise FileNotFoundError(physical_path)
            with np.load(physical_path) as physical_data:
                required = {"physical_grid", "prior_coords", "prior_conf", "view_ids"}
                missing = sorted(required - set(physical_data.files))
                if missing:
                    raise ValueError(f"cache npz missing keys: {missing}")
                grid = np.asarray(physical_data["physical_grid"], dtype=np.float32)
                prior_coords = np.asarray(physical_data["prior_coords"], dtype=np.int32)
                prior_conf = np.asarray(physical_data["prior_conf"], dtype=np.float32)
                view_ids = np.asarray(physical_data["view_ids"], dtype=np.int32)
            prior_path = resolve_path(prior_root, prior_sample["prior_npz"])
            if not prior_path.is_file():
                raise FileNotFoundError(prior_path)
            with np.load(prior_path) as prior_data:
                original_coords = np.asarray(prior_data["prior_coords"])
                original_conf = np.asarray(prior_data["prior_conf"])
                original_view_ids = np.asarray(prior_data["view_ids"])
            validate_prior_arrays(
                uid=uid,
                prior_coords=original_coords,
                prior_conf=original_conf,
                view_ids=original_view_ids,
                frame_count=len(source.get("frames") or []),
            )
            if not (
                np.array_equal(prior_coords, original_coords.astype(np.int32))
                and np.array_equal(view_ids, original_view_ids.astype(np.int32))
                and np.allclose(prior_conf, original_conf, atol=1.0e-3, rtol=0.0)
            ):
                raise ValueError("cache prior arrays differ from the source prior npz")
            if grid.shape != (len(PHYSICAL_FEATURE_NAMES), 16, 16, 16):
                raise ValueError(f"physical grid shape mismatch: {grid.shape}")
            if not np.isfinite(grid).all():
                counters["nonfinite_count"] += 1
                raise ValueError("physical grid contains non-finite values")
            frames = source.get("frames") or []
            validate_prior_arrays(
                uid=uid,
                prior_coords=prior_coords,
                prior_conf=prior_conf,
                view_ids=view_ids,
                frame_count=len(frames),
            )
            image_paths, mask_paths, intrinsics, extrinsics, masks = frame_paths(
                source_payload, source, view_ids, uid=uid
            )
            latent_path = resolve_path(latent_root, source["ss_latent"])
            z, target_coords = load_and_validate_target(latent_path, uid=uid)
            recomputed = make_physical_grid(
                prior_coords=prior_coords,
                prior_conf=prior_conf,
                masks=masks,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                grid_transform=str(prior_sample.get("grid_transform", prior_payload.get("grid_transform", "pixal3d_rotation"))),
                extrinsics_type=str(source_payload.get("extrinsics_type", "c2w")),
                camera_forward_sign=float(source_payload.get("camera_forward_sign", 1.0)),
            )
            recompute_diff = float(np.max(np.abs(recomputed - grid)))
            if recompute_diff > float(args.recompute_max_abs_diff):
                raise ValueError(f"physical grid recompute diff={recompute_diff}")
            grid_stats = physical_grid_audit(grid)
            high_conf_support = grid_stats["high_conf_prior_mask_support_mean"]
            if grid_stats["high_conf_prior_cell_count"] > 0 and (high_conf_support is None or high_conf_support <= 0):
                raise ValueError("high-confidence prior cells have no mask projection support")

            decoder_metrics = None
            if decoder is not None:
                latent = torch.from_numpy(z[None]).to(
                    device=torch.device(args.device), dtype=next(decoder.parameters()).dtype
                )
                with torch.no_grad():
                    logits = decoder(latent).float()[0, 0]
                decoded = torch.nonzero(logits > 0, as_tuple=False).cpu().numpy().astype(np.int32)
                decoder_metrics = overlap(decoded, target_coords)
                decoder_ious.append(float(decoder_metrics["iou"]))
                if float(decoder_metrics["iou"]) < float(args.decoder_min_iou):
                    counters["decoder_roundtrip_failure_count"] += 1
                    raise ValueError(f"decoder round-trip IoU below threshold: {decoder_metrics}")
            sample_reports.append(
                {
                    "index": index,
                    "uid": uid,
                    "object_uid": str(row.get("object_uid", "")),
                    "view_count": int(len(view_ids)),
                    "prior_point_count": int(len(prior_coords)),
                    "recompute_max_abs_diff": recompute_diff,
                    "grid": grid_stats,
                    "decoder_roundtrip": decoder_metrics,
                    "passed": True,
                }
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            lowered = message.lower()
            if "non-finite" in lowered:
                counters["nonfinite_count"] += 1
            if "outside [0,63]" in lowered:
                counters["invalid_coord_count"] += 1
            if "target z must" in lowered or "target_coords must" in lowered:
                counters["latent_shape_mismatch"] += 1
            if isinstance(exc, FileNotFoundError):
                counters["missing_file_count"] += 1
            failures.append({"index": index, "uid": uid, "error": message})
            counters["failed_sample_count"] += 1
            print(f"[pointpose_audit] FAIL uid={uid}: {message}", flush=True)

    mapping = mapping_audit()
    if not mapping["passed"]:
        counters["mapping_failure_count"] += 1
    hard_conditions = {
        "nonfinite_count": int(counters["nonfinite_count"]),
        "invalid_coord_count": int(counters["invalid_coord_count"]),
        "missing_file_count": int(counters["missing_file_count"]),
        "latent_shape_mismatch": int(counters["latent_shape_mismatch"]),
        "decoder_roundtrip_failure_count": int(counters["decoder_roundtrip_failure_count"]),
        "failed_sample_count": int(counters["failed_sample_count"]),
        "mapping_failure_count": int(counters["mapping_failure_count"]),
    }
    passed = all(value == 0 for value in hard_conditions.values())
    channel_aggregate: dict[str, Any] = {}
    for name in PHYSICAL_FEATURE_NAMES:
        values = [row["grid"]["channels"][name] for row in sample_reports]
        total = sum(int(value["numel"]) for value in values)
        channel_aggregate[name] = {
            "min": min((float(value["min"]) for value in values), default=0.0),
            "max": max((float(value["max"]) for value in values), default=0.0),
            "mean": float(
                sum(float(value["mean"]) * int(value["numel"]) for value in values) / total
            ) if total else 0.0,
            "nonzero_ratio": float(
                sum(float(value["nonzero_ratio"]) * int(value["numel"]) for value in values) / total
            ) if total else 0.0,
            "numel": total,
        }
    report = {
        "format": "reconvggt.pointpose_ss_cache.independent_audit.v1",
        "args": vars(args),
        "sample_count": len(rows),
        "passed_sample_count": len(sample_reports),
        "unique_uid_count": len(cache_by_uid),
        "unique_object_count": len({str(row.get("object_uid", "")) for row in rows}),
        "mapping_audit": mapping,
        "feature_names": list(PHYSICAL_FEATURE_NAMES),
        "channel_aggregate": channel_aggregate,
        "view_count_histogram": dict(
            sorted(Counter(int(row["view_count"]) for row in sample_reports).items())
        ),
        "hard_conditions": hard_conditions,
        "decoder_roundtrip": {
            "checked": decoder is not None,
            "min_iou": float(min(decoder_ious)) if decoder_ious else None,
            "mean_iou": float(np.mean(decoder_ious)) if decoder_ious else None,
            "required_min_iou": float(args.decoder_min_iou),
        },
        "passed": passed,
        "samples": sample_reports,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "failed_samples.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# PointPose SS cache audit",
        "",
        f"- samples: {len(rows)}",
        f"- passed samples: {len(sample_reports)}",
        f"- passed: {passed}",
        f"- hard conditions: `{hard_conditions}`",
        f"- mapping passed: {mapping['passed']}",
        f"- decoder round-trip: `{report['decoder_roundtrip']}`",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, **hard_conditions, "decoder": report["decoder_roundtrip"]}, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
