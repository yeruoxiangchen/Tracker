#!/usr/bin/env python3
"""Decode external GT SLAT targets with the frozen native mesh decoder."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

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

from pose_point_depth_mv.direct_slat_flow import (  # noqa: E402
    DIRECT_SLAT_CACHE_VERSION,
    canonical_json_sha256,
    support_generator_identity,
)
from pose_point_depth_mv.direct_slat_data import sha256_file  # noqa: E402
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (  # noqa: E402
    load_canonical_gt,
    mesh_structure_metrics,
    surface_metrics,
)
from pose_point_depth_mv.objaverse2k_slat_pipeline import (  # noqa: E402
    resolve_native_objaverse_normalization_bindings,
)
from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs  # noqa: E402
from trellis.modules import sparse as sp  # noqa: E402
from trellis.pipelines import TrellisVGGTTo3DPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--max_objects", type=int, default=32)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--max_chamfer_l1", type=float, default=0.10)
    parser.add_argument("--min_mesh_success_rate", type=float, default=1.0)
    parser.add_argument(
        "--decision_profile", choices=("report_only", "strict"), default="strict"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    manifest_path = Path(args.cache_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != DIRECT_SLAT_CACHE_VERSION:
        raise ValueError(f"unexpected cache format={manifest.get('format')!r}")
    if manifest.get("config", {}).get("pretrained") != args.pretrained:
        raise RuntimeError("target cache pretrained binding differs from decoder audit")
    cache_config = dict(manifest.get("config", {}))
    cache_config_hash = str(manifest.get("config_hash", ""))
    if not cache_config_hash:
        raise RuntimeError("target cache lacks a frozen config hash")
    objects = list(manifest.get("objects", []))
    if int(args.max_objects) > 0:
        objects = objects[: int(args.max_objects)]
    if not objects:
        raise ValueError("target decoder audit has no objects")
    normalization_bindings = resolve_native_objaverse_normalization_bindings(
        manifest_path, manifest, objects
    )
    if int(args.surface_samples) <= 0:
        raise ValueError("surface_samples must be positive")
    output_dir = Path(args.output_dir).resolve()
    run_config = {
        "format": "pose_point_depth_mv.direct_slat_target_decoder_audit_run.v1",
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "cache_config_hash": cache_config_hash,
        "support_generator": support_generator_identity(cache_config),
        "pretrained": args.pretrained,
        "selected_object_uids": [str(row["object_uid"]) for row in objects],
        "surface_samples": int(args.surface_samples),
        "min_mesh_success_rate": float(args.min_mesh_success_rate),
        "max_chamfer_l1": float(args.max_chamfer_l1),
        "decision_profile": args.decision_profile,
    }
    run_config["config_hash"] = canonical_json_sha256(run_config)
    run_config_path = output_dir / "run_config.json"
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError(output_dir)
        if run_config_path.is_file():
            existing = json.loads(run_config_path.read_text(encoding="utf-8"))
            if existing != run_config:
                raise RuntimeError(
                    "target-decoder audit resume arguments/cache binding changed"
                )
        elif any(output_dir.iterdir()):
            raise RuntimeError(
                "refusing to resume an unbound target-decoder audit directory"
            )
    else:
        output_dir.mkdir(parents=True)
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    root = Path(manifest.get("output_dir", manifest_path.parent)).resolve()
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    install_unused_model_stubs()
    pipeline = TrellisVGGTTo3DPipeline.from_pretrained(args.pretrained)
    pipeline._device = device
    decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    records = []
    for position, row in enumerate(objects):
        target_path = Path(row["target_file"])
        if not target_path.is_absolute():
            target_path = root / target_path
        expected_target_hash = str(row.get("target_file_sha256", ""))
        actual_target_hash = sha256_file(target_path)
        if not expected_target_hash or actual_target_hash != expected_target_hash:
            raise RuntimeError(
                f"target artifact hash mismatch object={row['object_uid']}"
            )
        ss_latent_path = Path(row["ss_latent"]).resolve()
        expected_ss_latent_hash = str(row.get("ss_latent_sha256", ""))
        actual_ss_latent_hash = sha256_file(ss_latent_path)
        if not expected_ss_latent_hash or actual_ss_latent_hash != expected_ss_latent_hash:
            raise RuntimeError(
                f"SS latent hash mismatch object={row['object_uid']}"
            )
        with np.load(target_path) as payload:
            coords3 = np.asarray(payload["coords"], dtype=np.int32)
            feats = np.asarray(payload["feats"], dtype=np.float32)
        coords = torch.cat(
            [
                torch.zeros((len(coords3), 1), dtype=torch.int32),
                torch.from_numpy(coords3),
            ],
            dim=1,
        ).to(device=device)
        latent = sp.SparseTensor(
            feats=torch.from_numpy(feats).to(
                device=device, dtype=next(decoder.parameters()).dtype
            ),
            coords=coords,
        )
        decoded = decoder(latent)[0]
        predicted = decoded.to_trimesh(transform_pose=False)
        target, target_metadata = load_canonical_gt(
            row,
            canonical_margin_binding=normalization_bindings.get(str(ss_latent_path)),
        )
        structure = mesh_structure_metrics(predicted)
        surface = (
            surface_metrics(
                predicted,
                target,
                count=int(args.surface_samples),
                seed=20260722 + position * 1009,
                thresholds=(0.01, 0.02, 0.05),
            )
            if structure["mesh_success"]
            else None
        )
        records.append(
            {
                "object_uid": str(row["object_uid"]),
                "target_file": str(target_path),
                "target_file_sha256": actual_target_hash,
                "source_lh_slat": str(row.get("source_lh_slat", "")),
                "source_lh_slat_sha256": str(
                    row.get("source_lh_slat_sha256", "")
                ),
                "ss_latent": str(ss_latent_path),
                "ss_latent_sha256": actual_ss_latent_hash,
                "target_metadata": target_metadata,
                "structure": structure,
                "surface": surface,
            }
        )
        print(
            f"[slat_target_decoder_audit] {position + 1}/{len(objects)} "
            f"{row['object_uid']}",
            flush=True,
        )
        del latent, decoded, predicted, target
        torch.cuda.empty_cache()
    success_rate = float(
        np.mean([bool(row["structure"]["mesh_success"]) for row in records])
    )
    chamfers = [
        float(row["surface"]["chamfer_l1"])
        for row in records
        if row["surface"] is not None
    ]
    summary = {
        "object_count": len(records),
        "mesh_success_rate": success_rate,
        "chamfer_l1_mean": float(np.mean(chamfers)) if chamfers else float("inf"),
        "chamfer_l1_median": float(np.median(chamfers)) if chamfers else float("inf"),
        "chamfer_l1_max": float(np.max(chamfers)) if chamfers else float("inf"),
    }
    checks = {
        "mesh_success_rate": success_rate >= float(args.min_mesh_success_rate),
        "chamfer_l1_median": summary["chamfer_l1_median"]
        <= float(args.max_chamfer_l1),
    }
    report = {
        "format": "pose_point_depth_mv.direct_slat_target_decoder_audit.v1",
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256_file(manifest_path),
        "cache_config_hash": cache_config_hash,
        "support_generator": run_config["support_generator"],
        "pretrained": args.pretrained,
        "thresholds": {
            "min_mesh_success_rate": float(args.min_mesh_success_rate),
            "max_chamfer_l1_median": float(args.max_chamfer_l1),
        },
        "summary": summary,
        "checks": checks,
        "passed": all(checks.values()),
        "records": records,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"summary": summary, "checks": checks}, indent=2), flush=True)
    decoder.cpu()
    del pipeline, decoder
    gc.collect()
    torch.cuda.empty_cache()
    if args.decision_profile == "strict" and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
