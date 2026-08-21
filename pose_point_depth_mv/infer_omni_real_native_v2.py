#!/usr/bin/env python3
"""Run the frozen Native SS + Native-SLAT v2 Full parent on runtime-O inputs."""

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
for dependency in (
    TRACKER_ROOT,
    TRACKER_ROOT / "ReconViaGen",
    TRACKER_ROOT / "ReconViaGen" / "wheels" / "vggt",
):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from pose_point_depth_mv.dataset_tools.prepare_omni_real_model_inputs import (  # noqa: E402
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_INPUT_OBJECT_FORMAT,
)
from pose_point_depth_mv.dataset_tools.prepare_omni_real_video_cache import utc_now  # noqa: E402
from pose_point_depth_mv.eval_direct_flow import decode_coords  # noqa: E402
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (  # noqa: E402
    canonical_coords,
    sparse_noise_from_master,
)
from pose_point_depth_mv.mesh_benchmark_metrics import mesh_structure_metrics  # noqa: E402
from pose_point_depth_mv.native_slat_genrecon_v2 import (  # noqa: E402
    NativeSLatCalibratedCFGFlow,
    build_native_slat_genrecon_v2_components,
    load_stock_slat_freeze,
    load_trainable_state_dict as load_slat_state,
    validate_native_slat_genrecon_v2_checkpoint,
)
from pose_point_depth_mv.native_ss_genrecon import (  # noqa: E402
    NativeSSCalibratedCFGFlow,
    build_native_ss_genrecon_components,
    load_trainable_state_dict as load_ss_state,
    validate_native_ss_genrecon_checkpoint,
)
from pose_point_depth_mv.omni_real_benchmark_common import (  # noqa: E402
    atomic_json,
    atomic_npz,
    canonical_sha256,
    load_json,
    object_key,
    resolve_torch_device,
    select_rows,
    sha256_file,
    to_device_tree,
    validate_bound_file,
)


REPORT_FORMAT = "pose_point_depth_mv.omni_real_native_v2_inference.v1"
MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_native_v2_inference_manifest.v1"


def _noise_object_position(row: dict[str, Any], fallback: int) -> int:
    """Allow specialized sharded entry points to preserve global noise identity."""

    return int(fallback)


def parse_csv_int(value: str) -> list[int]:
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("seeds must be non-empty and unique")
    return values


def _load_model_sample(row: dict[str, Any]) -> dict[str, Any]:
    path = validate_bound_file(
        row["model_input"], row["model_input_sha256"], label="model input"
    )
    payload = torch.load(path, map_location="cpu")
    if (
        payload.get("format") != MODEL_INPUT_OBJECT_FORMAT
        or payload.get("object_key") != object_key(row)
        or payload.get("condition_sha256") != row.get("condition_sha256")
    ):
        raise RuntimeError(f"Native v2 model input identity differs: {path}")
    views = int(payload["visual_patch_features"].shape[0])
    height, width = map(int, payload["projection_image_size"])
    # Native v2 is frustum-only. Both projectors consume only this tensor's
    # image shape, so a zero-stride view avoids a fake depth prediction.
    payload["predicted_depth"] = torch.zeros((), dtype=torch.float16).expand(
        views, height, width
    )
    return payload


def _ss_params(defaults: dict[str, Any], binding: dict[str, Any]) -> dict[str, Any]:
    params = dict(defaults)
    params.update(
        {
            "steps": int(binding["steps"]),
            "cfg_strength": float(binding["cfg_strength"]),
            "cfg_interval": tuple(float(value) for value in binding["cfg_interval"]),
            "guidance_rescale": float(binding["guidance_rescale"]),
            "rescale_t": float(binding["rescale_t"]),
        }
    )
    return params


def _coord_paths(output_dir: Path, row: dict[str, Any], seed: int) -> tuple[Path, Path]:
    root = output_dir / "ss_coords" / row["category"] / row["object_id"]
    return root / f"seed_{seed}.npz", root / f"seed_{seed}.json"


def _mesh_paths(output_dir: Path, row: dict[str, Any], seed: int) -> tuple[Path, Path]:
    root = output_dir / "meshes" / row["category"] / row["object_id"] / f"seed_{seed}"
    return root / "mesh_o.obj", root / "result.json"


def _reuse_result(
    result_path: Path,
    mesh_path: Path,
    *,
    key: str,
    seed: int,
    model_input_sha256: str,
    ss_sha256: str,
    slat_sha256: str,
    ss_weights: str,
    slat_weights: str,
    stock_freeze_sha256: str,
    sampling_sha256: str,
) -> dict[str, Any] | None:
    if not result_path.is_file() or not mesh_path.is_file():
        return None
    result = load_json(result_path)
    expected = {
        "format": REPORT_FORMAT,
        "object_key": key,
        "seed": int(seed),
        "model_input_sha256": model_input_sha256,
        "native_ss_checkpoint_sha256": ss_sha256,
        "native_slat_checkpoint_sha256": slat_sha256,
        "native_ss_weights": ss_weights,
        "native_slat_weights": slat_weights,
        "stock_slat_freeze_sha256": stock_freeze_sha256,
        "sampling_sha256": sampling_sha256,
        "mesh_sha256": sha256_file(mesh_path),
    }
    mismatch = {
        name: (result.get(name), value)
        for name, value in expected.items()
        if result.get(name) != value
    }
    if mismatch:
        raise RuntimeError(f"stale Native v2 inference result={mismatch}")
    return result


def _run_ss(
    *,
    rows: list[dict[str, Any]],
    seeds: list[int],
    output_dir: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    pretrained: str,
    weights: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    upstream_binding: dict[str, Any],
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_native_ss_genrecon_checkpoint(checkpoint, pretrained=pretrained)
    if int(checkpoint.get("step", -1)) != int(upstream_binding["checkpoint_step"]):
        raise RuntimeError("Native SS checkpoint step differs from SLat upstream binding")
    saved = checkpoint["args"]
    sampler, model, decoder, _, defaults = build_native_ss_genrecon_components(
        pretrained=pretrained,
        lora_rank=int(saved["lora_rank"]),
        lora_alpha=int(saved["lora_alpha"]),
        condition_channels=int(saved["condition_channels"]),
        gradient_checkpointing=False,
        need_decoder=True,
        device=device,
    )
    if decoder is None:
        raise RuntimeError("Native SS decoder is required")
    state_key = "ema_trainable_state" if weights == "ema" else "model_trainable_state"
    load_ss_state(model, checkpoint[state_key])
    model.eval()
    decoder.eval()
    params = _ss_params(defaults, upstream_binding)
    for position, row in enumerate(rows):
        sample = _load_model_sample(row)
        positive = sample["stock_condition"].to(device=device)
        negative = torch.zeros_like(positive)
        for seed in seeds:
            coord_path, audit_path = _coord_paths(output_dir, row, seed)
            if coord_path.is_file() and audit_path.is_file():
                audit = load_json(audit_path)
                if (
                    audit.get("object_key") != object_key(row)
                    or int(audit.get("seed", -1)) != seed
                    or audit.get("checkpoint_sha256") != checkpoint_sha256
                    or audit.get("weights") != weights
                    or audit.get("model_input_sha256") != row["model_input_sha256"]
                    or canonical_sha256(audit.get("sampling"))
                    != canonical_sha256(params)
                    or audit.get("coords_sha256") != sha256_file(coord_path)
                ):
                    raise RuntimeError(f"stale Native SS coordinates: {coord_path}")
                continue
            if coord_path.exists() or audit_path.exists():
                raise RuntimeError(f"partial Native SS coordinate output: {coord_path}")
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 1000003 + _noise_object_position(row, position) * 1009
            )
            initial = torch.randn(
                (1, 8, 16, 16, 16), generator=generator, device=device
            )
            flow = NativeSSCalibratedCFGFlow(
                model, positive, sample, enabled=True, projection_mode="correct"
            )
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                latent = sampler.sample(
                    flow,
                    initial,
                    cond=positive,
                    neg_cond=negative,
                    **params,
                    verbose=False,
                ).samples
            if float(params["cfg_strength"]) != 1.0 and (
                flow.positive_calls <= 0 or flow.negative_calls <= 0
            ):
                raise RuntimeError("Native SS standard CFG missed a branch")
            coords = decode_coords(decoder, latent)
            if len(coords) <= 0:
                raise RuntimeError(f"Native SS decoded no occupied voxels: {object_key(row)}")
            atomic_npz(coord_path, coords=coords)
            atomic_json(
                audit_path,
                {
                    "format": "pose_point_depth_mv.omni_real_native_ss_coords.v1",
                    "created_at_utc": utc_now(),
                    "object_key": object_key(row),
                    "seed": int(seed),
                    "coord_count": int(len(coords)),
                    "coords_sha256": sha256_file(coord_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "weights": weights,
                    "model_input_sha256": row["model_input_sha256"],
                    "sampling": params,
                    "wrapper": flow.summary(),
                    "passed": True,
                },
            )
            print(
                f"[real_native_v2:ss] {position + 1}/{len(rows)} "
                f"object={object_key(row)} seed={seed} coords={len(coords)}",
                flush=True,
            )
            del initial, latent, flow
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del sample, positive, negative
    model.cpu()
    decoder.cpu()
    del sampler, model, decoder, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_slat(
    *,
    rows: list[dict[str, Any]],
    seeds: list[int],
    output_dir: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    stock_freeze_path: Path,
    pretrained: str,
    weights: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    stock_freeze = load_stock_slat_freeze(stock_freeze_path)
    stock_freeze_sha256 = sha256_file(stock_freeze_path)
    upstream = dict(checkpoint["model_summary"]["upstream_native_ss"])
    validate_native_slat_genrecon_v2_checkpoint(
        checkpoint,
        pretrained=pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=upstream,
    )
    saved = checkpoint["args"]
    sampler, model, decoder, _, defaults, normalization = (
        build_native_slat_genrecon_v2_components(
            pretrained=pretrained,
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=upstream,
            lora_rank=int(saved["lora_rank"]),
            lora_alpha=int(saved["lora_alpha"]),
            condition_channels=int(saved["condition_channels"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
    )
    if decoder is None:
        raise RuntimeError("Native-SLAT Mesh decoder is required")
    state_key = "ema_trainable_state" if weights == "ema" else "model_trainable_state"
    load_slat_state(model, checkpoint[state_key])
    model.eval()
    decoder.eval()
    params = dict(defaults)
    params.setdefault("guidance_rescale", 0.0)
    expected_params = {
        "steps": 25,
        "cfg_strength": 5.0,
        "cfg_interval": (0.5, 1.0),
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }
    for name, value in expected_params.items():
        actual = tuple(params[name]) if name == "cfg_interval" else params[name]
        if actual != value:
            raise RuntimeError(f"frozen Native-SLAT sampler changed: {name}={actual}")
    sampling_sha256 = canonical_sha256(params)
    mean = torch.tensor(normalization["mean"], device=device)[None]
    std = torch.tensor(normalization["std"], device=device)[None]
    reports: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        sample = _load_model_sample(row)
        condition = to_device_tree(sample["slat_condition"], device)
        for seed in seeds:
            mesh_path, result_path = _mesh_paths(output_dir, row, seed)
            reused = _reuse_result(
                result_path,
                mesh_path,
                key=object_key(row),
                seed=seed,
                model_input_sha256=row["model_input_sha256"],
                ss_sha256=upstream["checkpoint_sha256"],
                slat_sha256=checkpoint_sha256,
                ss_weights=str(upstream["weights"]),
                slat_weights=weights,
                stock_freeze_sha256=stock_freeze_sha256,
                sampling_sha256=sampling_sha256,
            )
            if reused is not None:
                reports.append(reused)
                continue
            if mesh_path.parent.exists():
                raise RuntimeError(f"partial Native v2 mesh output: {mesh_path.parent}")
            coord_path, coord_report_path = _coord_paths(output_dir, row, seed)
            coord_report = load_json(coord_report_path)
            if coord_report.get("coords_sha256") != sha256_file(coord_path):
                raise RuntimeError(f"Native SS coordinate binding changed: {coord_path}")
            with np.load(coord_path, allow_pickle=False) as payload:
                coords_np = canonical_coords(payload["coords"], resolution=64)
            master_seed = (
                int(seed) * 2000003
                + _noise_object_position(row, position) * 2017
                + 7919
            )
            generator = torch.Generator(device=device).manual_seed(master_seed)
            master = torch.randn(
                (64, 64, 64, 8), generator=generator, device=device
            )
            initial = sparse_noise_from_master(coords_np, master, device=device)
            flow = NativeSLatCalibratedCFGFlow(
                model,
                condition["cond"],
                sample,
                enabled=True,
                projection_mode="correct",
            )
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_enabled
            ):
                latent = sampler.sample(
                    flow,
                    initial,
                    **condition,
                    **params,
                    verbose=False,
                ).samples
            # The Mesh extractor creates fp32 scatter targets internally. Keep
            # decoding outside bf16/fp16 autocast, matching the formal v2 evaluator.
            decoded = decoder(latent * std + mean)[0]
            if flow.positive_calls <= 0 or flow.negative_calls <= 0:
                raise RuntimeError("Native-SLAT standard CFG missed a branch")
            mesh = decoded.to_trimesh(transform_pose=False)
            structure = mesh_structure_metrics(mesh)
            if not structure["mesh_success"]:
                raise RuntimeError(f"Native v2 decoded empty Mesh: {object_key(row)}")
            mesh_path.parent.mkdir(parents=True, exist_ok=False)
            temporary = mesh_path.with_name(f".{mesh_path.name}.tmp-{os.getpid()}")
            mesh.export(temporary, file_type="obj")
            os.replace(temporary, mesh_path)
            result = {
                "format": REPORT_FORMAT,
                "created_at_utc": utc_now(),
                "method": "native_v2_full",
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
                "native_ss_checkpoint": upstream["checkpoint"],
                "native_ss_checkpoint_sha256": upstream["checkpoint_sha256"],
                "native_ss_weights": upstream["weights"],
                "native_slat_checkpoint": str(checkpoint_path),
                "native_slat_checkpoint_sha256": checkpoint_sha256,
                "native_slat_weights": weights,
                "stock_slat_freeze": str(stock_freeze_path),
                "stock_slat_freeze_sha256": stock_freeze_sha256,
                "sampling": params,
                "sampling_sha256": sampling_sha256,
                "wrapper": flow.summary(),
                "output_frame": "runtime-O",
                "post_cfg_cap": False,
                "passed": True,
            }
            atomic_json(result_path, result)
            reports.append(result)
            print(
                f"[real_native_v2:mesh] {position + 1}/{len(rows)} "
                f"object={object_key(row)} seed={seed}",
                flush=True,
            )
            del master, initial, latent, decoded, mesh, flow
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del sample, condition
    model.cpu()
    decoder.cpu()
    del sampler, model, decoder, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reports


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_input_manifest", required=True)
    parser.add_argument("--native_ss_checkpoint", required=True)
    parser.add_argument("--native_slat_checkpoint", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
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
    seeds = parse_csv_int(args.seeds)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ss_path = Path(args.native_ss_checkpoint).expanduser().resolve()
    slat_path = Path(args.native_slat_checkpoint).expanduser().resolve()
    stock_path = Path(args.stock_slat_freeze).expanduser().resolve()
    ss_sha = sha256_file(ss_path)
    slat_sha = sha256_file(slat_path)
    slat_header = torch.load(slat_path, map_location="cpu")
    upstream = dict(slat_header["model_summary"]["upstream_native_ss"])
    if upstream.get("checkpoint_sha256") != ss_sha:
        raise RuntimeError("requested Native SS deployment differs from v2 Full parent")
    del slat_header
    device = resolve_torch_device(args.device)
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    _run_ss(
        rows=rows,
        seeds=seeds,
        output_dir=output_dir,
        checkpoint_path=ss_path,
        checkpoint_sha256=ss_sha,
        pretrained=args.pretrained,
        weights=str(upstream["weights"]),
        device=device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
        upstream_binding=upstream,
    )
    reports = _run_slat(
        rows=rows,
        seeds=seeds,
        output_dir=output_dir,
        checkpoint_path=slat_path,
        checkpoint_sha256=slat_sha,
        stock_freeze_path=stock_path,
        pretrained=args.pretrained,
        weights=args.weights,
        device=device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "method": "native_v2_full",
        "model_input_manifest": str(model_manifest_path),
        "model_input_manifest_sha256": sha256_file(model_manifest_path),
        "runtime_input_manifest": model_manifest["runtime_input_manifest"],
        "runtime_input_manifest_sha256": model_manifest[
            "runtime_input_manifest_sha256"
        ],
        "native_ss_checkpoint": str(ss_path),
        "native_ss_checkpoint_sha256": ss_sha,
        "native_slat_checkpoint": str(slat_path),
        "native_slat_checkpoint_sha256": slat_sha,
        "stock_slat_freeze": str(stock_path),
        "stock_slat_freeze_sha256": sha256_file(stock_path),
        "native_ss_weights": str(upstream["weights"]),
        "native_slat_weights": args.weights,
        "seeds": seeds,
        "object_count": len(rows),
        "record_count": len(reports),
        "objects": reports,
        "output_frame": "runtime-O",
        "target_or_metric_consumed": False,
        "passed": len(reports) == len(rows) * len(seeds),
    }
    manifest_path = output_dir / "inference_manifest.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "passed": manifest["passed"],
        "object_count": len(rows),
        "record_count": len(reports),
        "manifest": str(manifest_path),
    }, indent=2))
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
