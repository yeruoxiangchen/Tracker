#!/usr/bin/env python3
"""Decode official ProObjaverse SLat labels with the frozen Stock decoder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch

from pose_point_depth_mv.export_direct_flow_mesh_pairs import mesh_structure_metrics
from pose_point_depth_mv.native_slat_genrecon import (
    load_stock_slat_freeze,
    validate_runtime_stock_slat,
)
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    SPLIT_FORMAT,
    atomic_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
from trellis.modules import sparse as sp


REPORT_FORMAT = "pose_point_depth_mv.proobjaverse_official_slat_decoder_audit.v1"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--max_objects", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    split_path = Path(args.split_manifest).expanduser().resolve()
    split = load_json(split_path)
    if split.get("format") != SPLIT_FORMAT or split.get("name") != "decoder_audit":
        raise ValueError("decoder audit requires the frozen decoder_audit split")
    rows = list(split["rows"])
    if int(args.max_objects) > 0:
        rows = rows[: int(args.max_objects)]
    if not rows:
        raise ValueError("decoder audit selection is empty")
    output = Path(args.output_dir).expanduser().resolve()
    run_config = {
        "format": f"{REPORT_FORMAT}.run",
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "protocol_sha256": split["protocol_sha256"],
        "stock_slat_freeze": str(Path(args.stock_slat_freeze).resolve()),
        "stock_slat_freeze_file_sha256": sha256_file(args.stock_slat_freeze),
        "pretrained": str(args.pretrained),
        "object_uids": [str(row["uid"]) for row in rows],
        "target_definition": "official SLat coords/features decoded without normalization",
    }
    run_config["run_config_sha256"] = canonical_sha256(run_config)
    run_config_path = output / "run_config.json"
    if output.exists():
        if not args.resume:
            raise FileExistsError(output)
        if not run_config_path.is_file() or load_json(run_config_path) != run_config:
            raise RuntimeError("decoder-audit resume binding changed")
    else:
        output.mkdir(parents=True)
        atomic_json(run_config_path, run_config)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    install_unused_model_stubs()
    from trellis.pipelines import TrellisImageTo3DPipeline

    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    flow = pipeline.models["slat_flow_model"].eval()
    decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    stock_freeze = load_stock_slat_freeze(args.stock_slat_freeze)
    validate_runtime_stock_slat(
        stock_freeze,
        pretrained=args.pretrained,
        flow=flow,
        decoder=decoder,
        sampler_params=dict(pipeline.slat_sampler_params),
        normalization=dict(pipeline.slat_normalization),
    )

    records: list[dict[str, Any]] = []
    target_root = output / "decoded_official_targets"
    for position, row in enumerate(rows, start=1):
        uid = str(row["uid"])
        slat_path = Path(row["slat_npz"])
        if slat_path.stat().st_size != int(row["slat_size"]):
            raise RuntimeError(f"official SLat size changed: {slat_path}")
        with np.load(slat_path, allow_pickle=False) as value:
            coords3 = np.asarray(value["coords"], dtype=np.int32)
            feats = np.asarray(value["feats"], dtype=np.float32)
        coords = torch.cat(
            (
                torch.zeros((len(coords3), 1), dtype=torch.int32),
                torch.from_numpy(coords3),
            ),
            dim=1,
        ).to(device)
        latent = sp.SparseTensor(
            feats=torch.from_numpy(feats).to(
                device=device, dtype=next(decoder.parameters()).dtype
            ),
            coords=coords,
        )
        decoded = decoder(latent)[0]
        mesh = decoded.to_trimesh(transform_pose=False)
        structure = mesh_structure_metrics(mesh)
        if not structure["mesh_success"]:
            raise RuntimeError(f"official SLat target did not decode: {uid}")
        destination = target_root / uid[:2] / f"{uid}.obj"
        destination.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(destination)
        records.append(
            {
                "uid": uid,
                "slat_npz": str(slat_path.resolve()),
                "slat_npz_sha256": sha256_file(slat_path),
                "coord_count": len(coords3),
                "decoded_target_mesh": str(destination.resolve()),
                "decoded_target_mesh_sha256": sha256_file(destination),
                "structure": structure,
            }
        )
        print(f"[official_slat_decoder_audit] {position}/{len(rows)} {uid}", flush=True)
        del latent, decoded, mesh
        torch.cuda.empty_cache()

    success_rate = float(
        np.mean([bool(row["structure"]["mesh_success"]) for row in records])
    )
    component_ratios = np.asarray(
        [float(row["structure"]["largest_component_ratio"]) for row in records]
    )
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "passed": success_rate == 1.0,
        "formal": False,
        "run_config": run_config,
        "pretrained": str(args.pretrained),
        "protocol_sha256": split["protocol_sha256"],
        "stock_slat_freeze_sha256": stock_freeze["freeze_sha256"],
        "summary": {
            "object_count": len(records),
            "mesh_success_rate": success_rate,
            "largest_component_ratio_median": float(np.median(component_ratios)),
            "largest_component_ratio_min": float(component_ratios.min()),
        },
        "records": records,
        "scope_guard": (
            "official target decoder-space validity only; decoded targets are the "
            "GT-support diagnostic labels and are not source-GLB ground truth"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    atomic_json(output / "report.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "summary", "report_sha256")}, indent=2))


if __name__ == "__main__":
    main()
