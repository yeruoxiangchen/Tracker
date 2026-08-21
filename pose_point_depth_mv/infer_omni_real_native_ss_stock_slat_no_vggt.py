#!/usr/bin/env python3
"""Run mixed no-VGGT Native SS with frozen Stock SLat and Mesh decoder."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


os.environ.setdefault("SPCONV_ALGO", "native")
TRACKER_ROOT = Path(__file__).resolve().parents[1]
for dependency in (TRACKER_ROOT, TRACKER_ROOT / "ReconViaGen"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from pose_point_depth_mv import infer_omni_real_native_v2 as _v2_infer  # noqa: E402
from pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs import (  # noqa: E402
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_INPUT_OBJECT_FORMAT,
)
from pose_point_depth_mv.dino_only_condition import DINO_ONLY_CONTEXT_VERSION  # noqa: E402
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (  # noqa: E402
    canonical_coords,
    sparse_noise_from_master,
)
from pose_point_depth_mv.mesh_benchmark_metrics import mesh_structure_metrics  # noqa: E402
from pose_point_depth_mv.native_slat_genrecon import (  # noqa: E402
    load_stock_slat_freeze,
    validate_runtime_stock_slat,
)
from pose_point_depth_mv.native_ss_genrecon_no_vggt import (  # noqa: E402
    NATIVE_SS_NO_VGGT_VERSION,
    NO_VGGT_MODEL_CONTRACT,
    build_native_ss_no_vggt_components,
    validate_native_ss_no_vggt_checkpoint,
)
from pose_point_depth_mv.omni_real_benchmark_common import (  # noqa: E402
    atomic_json,
    canonical_sha256,
    load_json,
    object_key,
    resolve_torch_device,
    select_rows,
    sha256_file,
    to_device_tree,
)
from pose_point_depth_mv.real_full_no_vggt_migration import (  # noqa: E402
    load_migration_contract,
    migration_summary,
    validate_destination_migration,
)


REPORT_FORMAT = "pose_point_depth_mv.omni_real_native_ss_stock_slat_inference.v1"
MANIFEST_FORMAT = (
    "pose_point_depth_mv.omni_real_native_ss_stock_slat_inference_manifest.v1"
)
METHOD = "native_ss_frozen_stock_slat"
STOCK_SLAT_CONTEXT_CONTRACT = {
    "version": DINO_ONLY_CONTEXT_VERSION,
    "flow": "released_frozen_stock_slat",
    "context": "per_view_raw_dino_patch_tokens",
    "view_fusion": "released_uniform_mean",
    "native_slat_lora": False,
    "native_slat_3d_condition": False,
    "vggt_model_loaded": False,
    "vggt_model_executed": False,
}


def stock_sampling_params(defaults: dict[str, Any]) -> dict[str, Any]:
    params = dict(defaults)
    params.setdefault("guidance_rescale", 0.0)
    expected = {
        "steps": 25,
        "cfg_strength": 5.0,
        "cfg_interval": (0.5, 1.0),
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }
    for name, expected_value in expected.items():
        actual = tuple(params[name]) if name == "cfg_interval" else params[name]
        if actual != expected_value:
            raise RuntimeError(f"frozen Stock SLat sampler changed: {name}={actual}")
    return params


def native_ss_binding(
    checkpoint: dict[str, Any], checkpoint_path: Path, *, cfg_strength: float
) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "weights": "ema",
        "steps": 25,
        "cfg_strength": float(cfg_strength),
        "cfg_interval": [0.5, 1.0],
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }


def _reuse_stock_result(
    result_path: Path,
    mesh_path: Path,
    *,
    row: dict[str, Any],
    seed: int,
    ss_sha256: str,
    stock_freeze_sha256: str,
    sampling_sha256: str,
) -> dict[str, Any] | None:
    if not result_path.is_file() or not mesh_path.is_file():
        return None
    result = load_json(result_path)
    expected = {
        "format": REPORT_FORMAT,
        "method": METHOD,
        "object_key": object_key(row),
        "seed": int(seed),
        "model_input_sha256": row["model_input_sha256"],
        "native_ss_checkpoint_sha256": ss_sha256,
        "native_ss_weights": "ema",
        "stock_slat_freeze_sha256": stock_freeze_sha256,
        "sampling_sha256": sampling_sha256,
        "mesh_sha256": sha256_file(mesh_path),
    }
    mismatch = {
        name: (result.get(name), expected_value)
        for name, expected_value in expected.items()
        if result.get(name) != expected_value
    }
    if mismatch:
        raise RuntimeError(f"stale Stock SLat inference result={mismatch}")
    return result


@torch.no_grad()
def run_stock_slat(
    *,
    rows: list[dict[str, Any]],
    seeds: list[int],
    output_dir: Path,
    ss_binding: dict[str, Any],
    stock_freeze_path: Path,
    pretrained: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> list[dict[str, Any]]:
    from pose_point_depth_mv.train_direct_flow import install_unused_model_stubs
    from trellis.pipelines import TrellisImageTo3DPipeline

    install_unused_model_stubs()
    pipeline = TrellisImageTo3DPipeline.from_pretrained(pretrained)
    pipeline._device = device
    pipeline.low_vram = False
    if hasattr(pipeline, "VGGT_model"):
        raise RuntimeError("Stock DINO-only SLat unexpectedly constructed VGGT")

    model = pipeline.models["slat_flow_model"].to(device).eval()
    decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    for module in (model, decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    stock_freeze = load_stock_slat_freeze(stock_freeze_path)
    validate_runtime_stock_slat(
        stock_freeze,
        pretrained=pretrained,
        flow=model,
        decoder=decoder,
        sampler_params=dict(pipeline.slat_sampler_params),
        normalization=dict(pipeline.slat_normalization),
    )
    params = stock_sampling_params(dict(pipeline.slat_sampler_params))
    sampling_sha256 = canonical_sha256(params)
    stock_freeze_sha256 = sha256_file(stock_freeze_path)
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=device)[None]
    std = torch.tensor(pipeline.slat_normalization["std"], device=device)[None]
    reports: list[dict[str, Any]] = []

    for position, row in enumerate(rows):
        sample = _v2_infer._load_model_sample(row)
        condition = to_device_tree(sample["slat_condition"], device)
        for seed in seeds:
            mesh_path, result_path = _v2_infer._mesh_paths(output_dir, row, seed)
            reused = _reuse_stock_result(
                result_path,
                mesh_path,
                row=row,
                seed=seed,
                ss_sha256=str(ss_binding["checkpoint_sha256"]),
                stock_freeze_sha256=stock_freeze_sha256,
                sampling_sha256=sampling_sha256,
            )
            if reused is not None:
                reports.append(reused)
                continue
            if mesh_path.parent.exists():
                raise RuntimeError(f"partial Stock SLat output: {mesh_path.parent}")

            coord_path, coord_report_path = _v2_infer._coord_paths(
                output_dir, row, seed
            )
            coord_report = load_json(coord_report_path)
            if coord_report.get("coords_sha256") != sha256_file(coord_path):
                raise RuntimeError(f"Native SS coordinate binding changed: {coord_path}")
            with np.load(coord_path, allow_pickle=False) as payload:
                coords_np = canonical_coords(payload["coords"], resolution=64)

            master_seed = int(seed) * 2000003 + int(position) * 2017 + 7919
            generator = torch.Generator(device=device).manual_seed(master_seed)
            master = torch.randn(
                (64, 64, 64, 8), generator=generator, device=device
            )
            initial = sparse_noise_from_master(coords_np, master, device=device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                latent = pipeline.slat_sampler.sample(
                    model,
                    initial,
                    **condition,
                    **params,
                    verbose=False,
                ).samples
            decoded = decoder(latent * std + mean)[0]
            mesh = decoded.to_trimesh(transform_pose=False)
            structure = mesh_structure_metrics(mesh)
            if not structure["mesh_success"]:
                raise RuntimeError(f"Stock SLat decoded empty Mesh: {object_key(row)}")

            mesh_path.parent.mkdir(parents=True, exist_ok=False)
            temporary = mesh_path.with_name(f".{mesh_path.name}.tmp-{os.getpid()}")
            mesh.export(temporary, file_type="obj")
            os.replace(temporary, mesh_path)
            result = {
                "format": REPORT_FORMAT,
                "created_at_utc": _v2_infer.utc_now(),
                "method": METHOD,
                "object_key": object_key(row),
                "category": row["category"],
                "object_id": row["object_id"],
                "seed": int(seed),
                "mesh": str(mesh_path),
                "mesh_sha256": sha256_file(mesh_path),
                "structure": structure,
                "coord_count": int(len(coords_np)),
                "model_input": row["model_input"],
                "model_input_sha256": row["model_input_sha256"],
                "native_ss_checkpoint": str(ss_binding["checkpoint"]),
                "native_ss_checkpoint_sha256": str(
                    ss_binding["checkpoint_sha256"]
                ),
                "native_ss_weights": "ema",
                "native_ss_cfg_strength": float(ss_binding["cfg_strength"]),
                "native_slat_checkpoint": None,
                "native_slat_checkpoint_sha256": None,
                "native_slat_weights": None,
                "stock_slat_freeze": str(stock_freeze_path),
                "stock_slat_freeze_sha256": stock_freeze_sha256,
                "sampling": params,
                "sampling_sha256": sampling_sha256,
                "stock_slat_cfg_strength": float(params["cfg_strength"]),
                "slat_backend": "stock",
                "native_slat_lora_enabled": False,
                "native_slat_3d_condition_enabled": False,
                "stock_slat_context_contract": STOCK_SLAT_CONTEXT_CONTRACT,
                "master_noise_seed": master_seed,
                "output_frame": "runtime-O",
                "vggt_model_loaded": False,
                "vggt_model_executed": False,
                "passed": True,
            }
            atomic_json(result_path, result)
            reports.append(result)
            print(
                f"[native_ss_stock_slat:mesh] {position + 1}/{len(rows)} "
                f"object={object_key(row)} seed={seed}",
                flush=True,
            )
            del master, initial, latent, decoded, mesh
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del sample, condition

    model.cpu()
    decoder.cpu()
    del model, decoder, pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reports


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_input_manifest", required=True)
    parser.add_argument("--native_ss_checkpoint", required=True)
    parser.add_argument("--ss_migration_contract", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--weights", choices=("ema",), default="ema")
    parser.add_argument(
        "--native_ss_cfg_strength",
        type=float,
        default=3.0,
        help="Native SS deployment CFG; current mixed phone contract uses 3.0",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--object", action="append")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    model_manifest_path = Path(args.model_input_manifest).expanduser().resolve()
    model_manifest = load_json(model_manifest_path)
    if (
        model_manifest.get("format") != MODEL_INPUT_MANIFEST_FORMAT
        or model_manifest.get("passed") is not True
    ):
        raise RuntimeError(f"model input manifest did not pass: {model_manifest_path}")
    rows = select_rows(model_manifest.get("objects", []), args.object)
    seeds = _v2_infer.parse_csv_int(args.seeds)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ss_path = Path(args.native_ss_checkpoint).expanduser().resolve()
    stock_path = Path(args.stock_slat_freeze).expanduser().resolve()

    checkpoint = torch.load(ss_path, map_location="cpu")
    if checkpoint.get("format") != NATIVE_SS_NO_VGGT_VERSION:
        raise ValueError("Stock SLat phone test requires mixed no-VGGT Native SS")
    validate_native_ss_no_vggt_checkpoint(
        checkpoint, pretrained=args.pretrained, allow_v2_parent=False
    )
    migration = load_migration_contract(args.ss_migration_contract, stage="ss")
    validate_destination_migration(checkpoint, migration)
    if args.native_ss_cfg_strength <= 0.0:
        raise ValueError("native_ss_cfg_strength must be positive")
    ss_binding = native_ss_binding(
        checkpoint, ss_path, cfg_strength=args.native_ss_cfg_strength
    )
    del checkpoint

    _v2_infer.MODEL_INPUT_MANIFEST_FORMAT = MODEL_INPUT_MANIFEST_FORMAT
    _v2_infer.MODEL_INPUT_OBJECT_FORMAT = MODEL_INPUT_OBJECT_FORMAT
    _v2_infer.validate_native_ss_genrecon_checkpoint = (
        validate_native_ss_no_vggt_checkpoint
    )
    _v2_infer.build_native_ss_genrecon_components = (
        build_native_ss_no_vggt_components
    )
    device = resolve_torch_device(args.device)
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    _v2_infer._run_ss(
        rows=rows,
        seeds=seeds,
        output_dir=output_dir,
        checkpoint_path=ss_path,
        checkpoint_sha256=str(ss_binding["checkpoint_sha256"]),
        pretrained=args.pretrained,
        weights="ema",
        device=device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
        upstream_binding=ss_binding,
    )
    reports = run_stock_slat(
        rows=rows,
        seeds=seeds,
        output_dir=output_dir,
        ss_binding=ss_binding,
        stock_freeze_path=stock_path,
        pretrained=args.pretrained,
        device=device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": _v2_infer.utc_now(),
        "method": METHOD,
        "slat_backend": "stock",
        "model_input_manifest": str(model_manifest_path),
        "model_input_manifest_sha256": sha256_file(model_manifest_path),
        "runtime_input_manifest": model_manifest["runtime_input_manifest"],
        "runtime_input_manifest_sha256": model_manifest[
            "runtime_input_manifest_sha256"
        ],
        "native_ss_checkpoint": str(ss_path),
        "native_ss_checkpoint_sha256": str(ss_binding["checkpoint_sha256"]),
        "native_ss_weights": "ema",
        "native_ss_cfg_strength": float(ss_binding["cfg_strength"]),
        "stock_slat_cfg_strength": 5.0,
        "ss_migration_contract": migration_summary(migration),
        "native_slat_checkpoint": None,
        "native_slat_checkpoint_sha256": None,
        "native_slat_weights": None,
        "stock_slat_freeze": str(stock_path),
        "stock_slat_freeze_sha256": sha256_file(stock_path),
        "stock_slat_context_contract": STOCK_SLAT_CONTEXT_CONTRACT,
        "native_slat_lora_enabled": False,
        "native_slat_3d_condition_enabled": False,
        "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
        "seeds": seeds,
        "object_count": len(rows),
        "record_count": len(reports),
        "objects": reports,
        "output_frame": "runtime-O",
        "target_or_metric_consumed": False,
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "passed": len(reports) == len(rows) * len(seeds),
    }
    manifest_path = output_dir / "inference_manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "method": METHOD,
                "object_count": len(rows),
                "record_count": len(reports),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
