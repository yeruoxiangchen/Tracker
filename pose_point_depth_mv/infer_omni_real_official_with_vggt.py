#!/usr/bin/env python3
"""Run official VSS2k + V-SLat15k on frozen runtime-O Omni inputs.

This is a qualitative real-input adapter.  It consumes the unchanged native
ReconViaGen VGGT/DINO encoder cache made by
``prepare_omni_real_model_inputs`` but feeds only the trailing DINO patch
channels to the posed-DINO branch.  Native ``ss_vggt_cond`` and
``slat_vggt_cond`` remain the Stock cross-attention contexts.  VGGT cameras and
depth are never used; known runtime-O K/T remain authoritative.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from official_ss_with_vggt_perf_v1.model import (
    build_components as build_vss_components,
    validate_checkpoint as validate_vss_checkpoint,
)
from official_ss_with_vggt_perf_v1.ss_slat_endpoint import load_vss_deployment
from pose_point_depth_mv.dataset_tools.prepare_omni_real_model_inputs import (
    FEATURE_CONTRACT,
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_INPUT_OBJECT_FORMAT,
)
from pose_point_depth_mv.eval_direct_flow import decode_coords
from pose_point_depth_mv.evaluate_native_ss_genrecon import sampling_params
from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import (
    make_sampling_namespace,
)
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    canonical_coords,
    sparse_noise_from_master,
)
from pose_point_depth_mv.mesh_benchmark_metrics import mesh_structure_metrics
from pose_point_depth_mv.native_slat_genrecon_v2 import (
    NativeSLatCalibratedCFGFlow,
    load_stock_slat_freeze,
    load_trainable_state_dict as load_slat_state,
)
from pose_point_depth_mv.native_slat_genrecon_with_vggt_official import (
    NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION,
    build_native_slat_official_with_vggt_components,
    validate_native_slat_official_with_vggt_checkpoint,
)
from pose_point_depth_mv.native_ss_genrecon import (
    NativeSSCalibratedCFGFlow,
    load_trainable_state_dict as load_ss_state,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    atomic_json,
    atomic_npz,
    canonical_sha256,
    load_json,
    object_key,
    resolve_torch_device,
    select_rows,
    sha256_file,
    to_device_tree,
)
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (
    decoded_mesh_to_sparse_grid_frame,
    mesh_frame_contract_fields,
    validate_runtime_o_mesh_frame_contract,
)


REPORT_FORMAT = "pose_point_depth_mv.omni_real_official_with_vggt_inference.v1"
MANIFEST_FORMAT = "pose_point_depth_mv.omni_real_official_with_vggt_inference_manifest.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_int(value: str) -> list[int]:
    result = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("seeds must be non-empty and unique")
    return result


def _validate_tensor(value: Any, shape: tuple[int | None, ...], *, label: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != len(shape):
        raise RuntimeError(f"{label} tensor rank differs")
    for observed, expected in zip(value.shape, shape):
        if expected is not None and int(observed) != int(expected):
            raise RuntimeError(f"{label} shape differs: {tuple(value.shape)}")
    if not bool(torch.isfinite(value.float()).all().item()):
        raise RuntimeError(f"{label} contains non-finite values")
    return value


def _load_model_sample(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["model_input"]).expanduser().resolve(strict=True)
    if sha256_file(path) != str(row["model_input_sha256"]):
        raise RuntimeError(f"with-VGGT model input hash differs: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format") != MODEL_INPUT_OBJECT_FORMAT
        or payload.get("object_key") != object_key(row)
        or payload.get("condition_sha256") != row.get("condition_sha256")
        or payload.get("feature_contract") != FEATURE_CONTRACT
    ):
        raise RuntimeError(f"with-VGGT model input identity differs: {path}")
    full_visual = _validate_tensor(
        payload.get("visual_patch_features"), (None, 1369, 3072), label="VGGT+DINO visual"
    )
    views = int(full_visual.shape[0])
    # The official 2K posed branch is DINO-only.  The leading 2048 VGGT
    # channels belong to historical Native-v2 spatial lifting and must not be
    # passed into the official posed-DINO projection.
    dino = full_visual[..., -1024:].contiguous()
    _validate_tensor(dino, (views, 1369, 1024), label="official posed-DINO visual")
    stock = _validate_tensor(
        payload.get("stock_condition"), (1, 4096, 1024), label="native ss_vggt_cond"
    )
    slat = payload.get("slat_condition")
    if not isinstance(slat, dict) or set(slat) != {"cond", "neg_cond"}:
        raise RuntimeError("native slat_vggt_cond tree differs")
    for branch in ("cond", "neg_cond"):
        values = slat[branch]
        if not isinstance(values, list) or len(values) != views:
            raise RuntimeError(f"native SLat {branch} view count differs")
        for position, value in enumerate(values):
            _validate_tensor(
                value, (1, 1374, 1024), label=f"native SLat {branch}[{position}]"
            )
    if any(int(torch.count_nonzero(value).item()) != 0 for value in slat["neg_cond"]):
        raise RuntimeError("native SLat negative context is not zero")
    intrinsics = _validate_tensor(payload.get("intrinsics"), (views, 3, 3), label="K")
    extrinsics = _validate_tensor(payload.get("extrinsics"), (views, 4, 4), label="T")
    if (
        payload.get("grid_transform") != "identity"
        or payload.get("extrinsics_type") != "w2c"
        or payload.get("extrinsics_source") != "T_O2C_lifting"
        or float(payload.get("camera_forward_sign", float("nan"))) != 1.0
        or tuple(payload.get("projection_image_size", ())) != (518, 518)
    ):
        raise RuntimeError("runtime-O K/T projection contract differs")
    sample = dict(payload)
    sample["visual_patch_features"] = dino
    sample["stock_condition"] = stock
    # Frustum-only official projection consumes only the image shape.  A
    # zero-stride tensor avoids fabricating a depth prediction.
    sample["predicted_depth"] = torch.zeros((), dtype=torch.float16).expand(
        views, 518, 518
    )
    sample["depth_confidence"] = torch.zeros((), dtype=torch.float16).expand(
        views, 518, 518
    )
    sample["intrinsics"] = intrinsics
    sample["extrinsics"] = extrinsics
    sample["official_with_vggt_runtime_adapter"] = {
        "spatial_features": "trailing 1024 DINO patch channels only",
        "ss_stock_context": "native ss_vggt_cond",
        "slat_stock_context": "native slat_vggt_cond",
        "vggt_camera_consumed": False,
        "vggt_depth_consumed": False,
        "known_runtime_o_K_T_consumed": True,
    }
    return sample


def _coord_paths(output: Path, row: dict[str, Any], seed: int) -> tuple[Path, Path]:
    root = output / "ss_coords" / row["category"] / row["object_id"]
    return root / f"seed_{seed}.npz", root / f"seed_{seed}.json"


def _mesh_paths(output: Path, row: dict[str, Any], seed: int) -> tuple[Path, Path]:
    root = output / "meshes" / row["category"] / row["object_id"] / f"seed_{seed}"
    return root / "mesh_o.obj", root / "result.json"


@torch.no_grad()
def _run_ss(
    *,
    rows: list[dict[str, Any]],
    seeds: list[int],
    output: Path,
    report_path: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> dict[str, Any]:
    evidence, binding = load_vss_deployment(report_path)
    checkpoint_path = Path(binding["checkpoint"]).resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_vss_checkpoint(checkpoint, pretrained="Stable-X/trellis-vggt-v0-2")
    saved = checkpoint["args"]
    sampler, model, decoder, _, defaults = build_vss_components(
        pretrained="Stable-X/trellis-vggt-v0-2",
        lora_rank=int(saved["lora_rank"]),
        lora_alpha=int(saved["lora_alpha"]),
        condition_channels=int(saved["condition_channels"]),
        gradient_checkpointing=False,
        need_decoder=True,
        device=device,
    )
    if decoder is None:
        raise RuntimeError("official VSS inference requires the frozen SS decoder")
    load_ss_state(model, checkpoint["ema_trainable_state"])
    model.eval()
    decoder.eval()
    params = sampling_params(
        defaults, make_sampling_namespace(binding), float(binding["cfg_strength"])
    )
    for position, row in enumerate(rows):
        sample = _load_model_sample(row)
        positive = sample["stock_condition"].to(device=device)
        negative = torch.zeros_like(positive)
        for seed in seeds:
            coord_path, audit_path = _coord_paths(output, row, seed)
            if coord_path.is_file() and audit_path.is_file():
                audit = load_json(audit_path)
                if (
                    audit.get("coords_sha256") != sha256_file(coord_path)
                    or audit.get("checkpoint_sha256") != binding["checkpoint_sha256"]
                    or audit.get("model_input_sha256") != row["model_input_sha256"]
                ):
                    raise RuntimeError(f"stale VSS coordinate cache: {coord_path}")
                continue
            if coord_path.exists() or audit_path.exists():
                raise RuntimeError(f"partial VSS coordinate cache: {coord_path}")
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 1000003 + int(position) * 1009
            )
            initial = torch.randn((1, 8, 16, 16, 16), generator=generator, device=device)
            flow = NativeSSCalibratedCFGFlow(
                model, positive, sample, enabled=True, projection_mode="correct"
            )
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                latent = sampler.sample(
                    flow,
                    initial,
                    cond=positive,
                    neg_cond=negative,
                    **params,
                    verbose=False,
                ).samples
            if float(binding["cfg_strength"]) != 1.0 and (
                flow.positive_calls <= 0 or flow.negative_calls <= 0
            ):
                raise RuntimeError("official VSS CFG missed a branch")
            coords = decode_coords(decoder, latent)
            if len(coords) <= 0:
                raise RuntimeError(f"official VSS decoded empty support: {object_key(row)}")
            atomic_npz(coord_path, coords=coords.astype(np.int32))
            atomic_json(
                audit_path,
                {
                    "format": "pose_point_depth_mv.omni_real_official_vss_coords.v1",
                    "created_at_utc": utc_now(),
                    "passed": True,
                    "object_key": object_key(row),
                    "seed": int(seed),
                    "coord_count": int(len(coords)),
                    "coords_sha256": sha256_file(coord_path),
                    "model_input_sha256": row["model_input_sha256"],
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": binding["checkpoint_sha256"],
                    "weights": binding["weights"],
                    "sampling": params,
                    "sampling_sha256": canonical_sha256(params),
                    "wrapper": flow.summary(),
                },
            )
            print(f"[real_official_vss:ss] object={object_key(row)} seed={seed} coords={len(coords)}", flush=True)
            del initial, latent, flow
            torch.cuda.empty_cache()
        del sample, positive, negative
    model.cpu()
    decoder.cpu()
    del sampler, model, decoder, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return {**binding, "report_passed": evidence.get("passed") is True}


@torch.no_grad()
def _run_slat(
    *,
    rows: list[dict[str, Any]],
    seeds: list[int],
    output: Path,
    checkpoint_path: Path,
    stock_freeze_path: Path,
    evaluation_ss: dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("format") != NATIVE_SLAT_OFFICIAL_WITH_VGGT_VERSION
        or int(checkpoint.get("step", -1)) != 15000
    ):
        raise RuntimeError("requested V-SLat checkpoint is not official with-VGGT step15000")
    stock_freeze = load_stock_slat_freeze(stock_freeze_path)
    saved_upstream = checkpoint.get("model_summary", {}).get("upstream_native_ss")
    data_upstream = checkpoint.get("data_identity", {}).get("native_ss")
    if not isinstance(saved_upstream, dict) or saved_upstream != data_upstream:
        raise RuntimeError("V-SLat training-upstream Native-SS identity differs")
    validate_native_slat_official_with_vggt_checkpoint(
        checkpoint,
        pretrained="Stable-X/trellis-vggt-v0-2",
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=saved_upstream,
    )
    saved = checkpoint["args"]
    sampler, model, decoder, _, defaults, normalization = (
        build_native_slat_official_with_vggt_components(
            pretrained="Stable-X/trellis-vggt-v0-2",
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=saved_upstream,
            lora_rank=int(saved["lora_rank"]),
            lora_alpha=int(saved["lora_alpha"]),
            condition_channels=int(saved["condition_channels"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
    )
    if decoder is None:
        raise RuntimeError("official V-SLat inference requires Stock Mesh decoder")
    load_slat_state(model, checkpoint["ema_trainable_state"])
    model.eval()
    decoder.eval()
    params = dict(defaults)
    params.setdefault("guidance_rescale", 0.0)
    required = {
        "steps": 25,
        "cfg_strength": 5.0,
        "cfg_interval": (0.5, 1.0),
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }
    for name, expected in required.items():
        observed = tuple(params[name]) if name == "cfg_interval" else params[name]
        if observed != expected:
            raise RuntimeError(f"official V-SLat sampler changed: {name}={observed}")
    mean = torch.tensor(normalization["mean"], device=device)[None]
    std = torch.tensor(normalization["std"], device=device)[None]
    slat_sha = sha256_file(checkpoint_path)
    stock_sha = sha256_file(stock_freeze_path)
    frame_fields = mesh_frame_contract_fields(
        export_policy="MeshExtractResult.to_trimesh(transform_pose=False)"
    )
    reports: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        sample = _load_model_sample(row)
        condition = to_device_tree(sample["slat_condition"], device)
        for seed in seeds:
            mesh_path, result_path = _mesh_paths(output, row, seed)
            if mesh_path.is_file() and result_path.is_file():
                result = load_json(result_path)
                if (
                    result.get("format") != REPORT_FORMAT
                    or result.get("mesh_sha256") != sha256_file(mesh_path)
                    or result.get("native_slat_checkpoint_sha256") != slat_sha
                    or result.get("evaluation_native_ss_checkpoint_sha256")
                    != evaluation_ss["checkpoint_sha256"]
                    or result.get("model_input_sha256") != row["model_input_sha256"]
                ):
                    raise RuntimeError(f"stale VSS+V-SLat Mesh: {mesh_path}")
                validate_runtime_o_mesh_frame_contract(result)
                reports.append(result)
                continue
            if mesh_path.parent.exists():
                raise RuntimeError(f"partial VSS+V-SLat Mesh output: {mesh_path.parent}")
            coord_path, coord_audit_path = _coord_paths(output, row, seed)
            coord_audit = load_json(coord_audit_path)
            if (
                coord_audit.get("coords_sha256") != sha256_file(coord_path)
                or coord_audit.get("checkpoint_sha256")
                != evaluation_ss["checkpoint_sha256"]
            ):
                raise RuntimeError("V-SLat support is not bound to evaluation VSS")
            with np.load(coord_path, allow_pickle=False) as payload:
                coords_np = canonical_coords(payload["coords"], resolution=64)
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 2000003 + int(position) * 2017 + 7919
            )
            master = torch.randn((64, 64, 64, 8), generator=generator, device=device)
            initial = sparse_noise_from_master(coords_np, master, device=device)
            flow = NativeSLatCalibratedCFGFlow(
                model, condition["cond"], sample, enabled=True, projection_mode="correct"
            )
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                latent = sampler.sample(
                    flow, initial, **condition, **params, verbose=False
                ).samples
            if flow.positive_calls <= 0 or flow.negative_calls <= 0:
                raise RuntimeError("official V-SLat CFG missed a branch")
            decoded = decoder(latent * std + mean)[0]
            mesh = decoded_mesh_to_sparse_grid_frame(decoded)
            structure = mesh_structure_metrics(mesh)
            if not structure["mesh_success"]:
                raise RuntimeError(f"official V-SLat decoded invalid Mesh: {object_key(row)}")
            mesh_path.parent.mkdir(parents=True, exist_ok=False)
            temporary = mesh_path.with_name(f".{mesh_path.name}.tmp-{os.getpid()}")
            mesh.export(temporary, file_type="obj")
            os.replace(temporary, mesh_path)
            result = {
                "format": REPORT_FORMAT,
                "created_at_utc": utc_now(),
                "passed": True,
                "method": "official_VSS_step2000_plus_V_SLat_step15000",
                "object_key": object_key(row),
                "category": row["category"],
                "object_id": row["object_id"],
                "seed": int(seed),
                "coord_count": int(len(coords_np)),
                "structure": structure,
                "mesh": str(mesh_path),
                "mesh_sha256": sha256_file(mesh_path),
                "result": str(result_path),
                "model_input": row["model_input"],
                "model_input_sha256": row["model_input_sha256"],
                "evaluation_native_ss_report": evaluation_ss["report"],
                "evaluation_native_ss_report_sha256": evaluation_ss["report_sha256"],
                "evaluation_native_ss_checkpoint": evaluation_ss["checkpoint"],
                "evaluation_native_ss_checkpoint_sha256": evaluation_ss["checkpoint_sha256"],
                "evaluation_native_ss_checkpoint_step": evaluation_ss["checkpoint_step"],
                "evaluation_native_ss_weights": evaluation_ss["weights"],
                "native_slat_checkpoint": str(checkpoint_path),
                "native_slat_checkpoint_sha256": slat_sha,
                "native_slat_checkpoint_step": 15000,
                "native_slat_weights": "ema",
                "training_upstream_native_ss": saved_upstream,
                "support_deployment_differs_from_training_upstream": True,
                "stock_slat_freeze": str(stock_freeze_path),
                "stock_slat_freeze_sha256": stock_sha,
                "sampling": params,
                "sampling_sha256": canonical_sha256(params),
                "wrapper": flow.summary(),
                "output_frame": "runtime-O",
                "decoder_mesh_export_transform_pose": False,
                **frame_fields,
                "vggt_camera_consumed": False,
                "vggt_depth_consumed": False,
                "known_runtime_o_K_T_consumed": True,
                "target_or_metric_consumed": False,
            }
            atomic_json(result_path, result)
            reports.append(result)
            print(f"[real_official_vss_vslat:mesh] object={object_key(row)} seed={seed}", flush=True)
            del master, initial, latent, decoded, mesh, flow
            torch.cuda.empty_cache()
        del sample, condition
    model.cpu()
    decoder.cpu()
    del sampler, model, decoder, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return reports


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_input_manifest", required=True)
    parser.add_argument("--native_ss_report", required=True)
    parser.add_argument("--native_slat_checkpoint", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--object", action="append")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    model_path = Path(args.model_input_manifest).expanduser().resolve(strict=True)
    model_manifest = load_json(model_path)
    if (
        model_manifest.get("format") != MODEL_INPUT_MANIFEST_FORMAT
        or model_manifest.get("passed") is not True
    ):
        raise RuntimeError(f"with-VGGT model input manifest did not pass: {model_path}")
    rows = select_rows(model_manifest.get("objects", []), args.object)
    seeds = parse_csv_int(args.seeds)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ss_report = Path(args.native_ss_report).expanduser().resolve(strict=True)
    slat_checkpoint = Path(args.native_slat_checkpoint).expanduser().resolve(strict=True)
    stock_freeze = Path(args.stock_slat_freeze).expanduser().resolve(strict=True)
    device = resolve_torch_device(args.device)
    if device.type != "cuda":
        raise ValueError("official VSS+V-SLat qualitative inference requires CUDA")
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    evaluation_ss = _run_ss(
        rows=rows,
        seeds=seeds,
        output=output,
        report_path=ss_report,
        device=device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    reports = _run_slat(
        rows=rows,
        seeds=seeds,
        output=output,
        checkpoint_path=slat_checkpoint,
        stock_freeze_path=stock_freeze,
        evaluation_ss=evaluation_ss,
        device=device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": utc_now(),
        "passed": len(reports) == len(rows) * len(seeds),
        "method": "official_VSS_step2000_plus_V_SLat_step15000",
        "model_input_manifest": str(model_path),
        "model_input_manifest_sha256": sha256_file(model_path),
        "runtime_input_manifest": model_manifest["runtime_input_manifest"],
        "runtime_input_manifest_sha256": model_manifest["runtime_input_manifest_sha256"],
        "native_ss_deployment": evaluation_ss,
        "native_slat_checkpoint": str(slat_checkpoint),
        "native_slat_checkpoint_sha256": sha256_file(slat_checkpoint),
        "native_slat_checkpoint_step": 15000,
        "native_slat_weights": "ema",
        "stock_slat_freeze": str(stock_freeze),
        "stock_slat_freeze_sha256": sha256_file(stock_freeze),
        "seeds": seeds,
        "object_count": len(rows),
        "record_count": len(reports),
        "objects": reports,
        "output_frame": "runtime-O",
        **mesh_frame_contract_fields(
            export_policy="MeshExtractResult.to_trimesh(transform_pose=False)"
        ),
        "spatial_condition": "posed DINO using frozen runtime-O known K/T",
        "stock_context": "native ReconViaGen ss_vggt_cond / slat_vggt_cond",
        "vggt_camera_consumed": False,
        "vggt_depth_consumed": False,
        "target_or_metric_consumed": False,
        "scope_guard": (
            "Qualitative real-input result. Chapter 158 C-A is positive, while C-B "
            "does not establish an independent V-SLat15k improvement."
        ),
    }
    atomic_json(output / "inference_manifest.json", manifest)
    print(json.dumps({"passed": manifest["passed"], "objects": len(rows), "records": len(reports), "manifest": str(output / 'inference_manifest.json')}, indent=2))
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
