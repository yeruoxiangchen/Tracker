#!/usr/bin/env python3
"""Isolate Stock, synthetic1k, and mixed SLat quality on one fixed support."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import (
    build_slat_condition,
    build_stock_conditioning_pipeline,
    torch_save_atomic,
    write_json,
)
from pose_point_depth_mv.evaluate_objaverse16_no_vggt import (
    aggregate,
    export_obj_atomic,
    parse_float_csv,
    stable_metric_seed,
)
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    canonical_coords,
    load_canonical_gt,
    sample_slat_explicit,
    sparse_noise_from_master,
    tensor_sha256,
)
from pose_point_depth_mv.freeze_objaverse16_test import PROTOCOL_FORMAT, sha256_file
from pose_point_depth_mv.infer_objaverse16_no_vggt_mixed import (
    MANIFEST_FORMAT as MIXED_MANIFEST_FORMAT,
    _load_model_sample,
)
from pose_point_depth_mv.infer_objaverse16_no_vggt_synthetic1k import (
    MANIFEST_FORMAT as SYNTHETIC_MANIFEST_FORMAT,
)
from pose_point_depth_mv.mesh_benchmark_metrics import (
    mesh_structure_metrics,
    surface_metrics,
)
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (
    NativeSLatCalibratedCFGFlow,
    build_native_slat_no_vggt_components,
    load_stock_slat_freeze,
    load_trainable_state_dict,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_point_depth_mv.omni_real_benchmark_common import (
    canonical_sha256,
    load_json,
    resolve_torch_device,
    to_device_tree,
    validate_bound_file,
)
from pose_point_depth_mv.prepare_objaverse16_no_vggt_model_inputs import (
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
)
from pose_point_depth_mv.render_direct_slat_fourway import (
    LATENT_DECODER_TO_REFERENCE,
)


REPORT_FORMAT = "pose_point_depth_mv.objaverse16_slat_oracle_swap.v1"
RECORD_FORMAT = "pose_point_depth_mv.objaverse16_slat_oracle_swap_record.v1"
METHODS = ("reconviagen_stock_slat", "synthetic1k_slat", "mixed_slat")
LOWER_IS_BETTER = {
    "pred_to_gt_mean",
    "gt_to_pred_mean",
    "chamfer_l1",
    "chamfer_l2",
}
MATCHED_SAMPLING = {
    "steps": 25,
    "cfg_strength": 5.0,
    "cfg_interval": (0.5, 1.0),
    "guidance_rescale": 0.0,
    "rescale_t": 3.0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection_manifest", required=True)
    parser.add_argument("--stock_lifting_manifest", required=True)
    parser.add_argument("--model_input_manifest", required=True)
    parser.add_argument("--synthetic_inference_manifest", required=True)
    parser.add_argument("--mixed_inference_manifest", required=True)
    parser.add_argument("--existing_o11_report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument(
        "--fscore_thresholds", type=parse_float_csv, default=[0.01, 0.02, 0.05]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--max_objects", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    return parser


def _index_unique(
    rows: Iterable[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in result:
            raise RuntimeError(f"duplicate {label}={value}")
        result[value] = row
    return result


def _resolve_cache_file(manifest: dict[str, Any], row: dict[str, Any]) -> Path:
    value = Path(str(row["cache_file"])).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (Path(str(manifest["output_dir"])).expanduser().resolve() / value).resolve()


def _oracle_support(row: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    latent_path = Path(str(row["ss_latent"])).expanduser().resolve()
    with np.load(latent_path, allow_pickle=False) as payload:
        if int(payload["voxel_resolution"]) != 64:
            raise RuntimeError(f"oracle support is not 64^3: {latent_path}")
        if str(payload["uid"]) != str(row["uid"]):
            raise RuntimeError(f"oracle support UID differs: {latent_path}")
        coords = canonical_coords(payload["target_coords"], resolution=64)
    digest = hashlib.sha256()
    digest.update(np.asarray(coords.shape, dtype=np.int64).tobytes())
    digest.update(coords.tobytes())
    return coords, {
        "source": "frozen repaired SS target_coords",
        "ss_latent": str(latent_path),
        "ss_latent_sha256": sha256_file(latent_path),
        "coord_count": int(len(coords)),
        "coords_sha256": digest.hexdigest(),
        "resolution": 64,
        "target_consumed": True,
    }


def _view_count(row: dict[str, Any]) -> int:
    selection = dict(row.get("objaverse16_selection", {}))
    return int(selection.get("expected_point_prior_view_count", len(row["frames"])))


def _master_noise(
    *, seed: int, position: int, coords: np.ndarray, device: torch.device
) -> tuple[Any, str, int]:
    master_seed = int(seed) * 2000003 + int(position) * 2017 + 7919
    generator = torch.Generator(device=device).manual_seed(master_seed)
    master = torch.randn(
        (64, 64, 64, 8), generator=generator, device=device, dtype=torch.float32
    )
    noise_hash = tensor_sha256(master)
    return (
        sparse_noise_from_master(coords, master, device=device),
        noise_hash,
        master_seed,
    )


def _amp(args: argparse.Namespace, device: torch.device):
    enabled = args.amp_dtype != "none" and device.type == "cuda"
    dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def _result_paths(
    output_dir: Path, method: str, object_uid: str, seed: int
) -> tuple[Path, Path, Path]:
    root = output_dir / "branches" / method / object_uid / f"seed_{int(seed)}"
    return root / "result.json", root / "mesh_decoder.obj", root / "mesh_source.obj"


def _reuse_result(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    method: str,
    row: dict[str, Any],
    model_sha256: str,
    condition_sha256: str,
    support: dict[str, Any],
    sampling_sha256: str,
) -> dict[str, Any] | None:
    result_path, decoder_path, source_path = _result_paths(
        output_dir, method, str(row["object_uid"]), int(args.seed)
    )
    present = (result_path.is_file(), decoder_path.is_file(), source_path.is_file())
    if not any(present):
        return None
    if not all(present) or not args.resume:
        raise RuntimeError(
            f"partial or non-resumable branch output: {result_path.parent}"
        )
    result = load_json(result_path)
    expected = {
        "format": RECORD_FORMAT,
        "method": method,
        "uid": str(row["uid"]),
        "object_uid": str(row["object_uid"]),
        "seed": int(args.seed),
        "model_sha256": model_sha256,
        "condition_sha256": condition_sha256,
        "oracle_support_sha256": support["coords_sha256"],
        "sampling_sha256": sampling_sha256,
        "mesh_decoder_sha256": sha256_file(decoder_path),
        "mesh_source_sha256": sha256_file(source_path),
        "passed": True,
    }
    mismatch = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"stale SLat swap output={mismatch}")
    return result


def _finish_result(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    method: str,
    row: dict[str, Any],
    position: int,
    decoded: Any,
    model_path: str,
    model_sha256: str,
    condition_path: str,
    condition_sha256: str,
    support: dict[str, Any],
    master_noise_sha256: str,
    master_noise_seed: int,
    wrapper: dict[str, Any] | None,
) -> dict[str, Any]:
    result_path, decoder_path, source_path = _result_paths(
        output_dir, method, str(row["object_uid"]), int(args.seed)
    )
    if result_path.parent.exists() and any(result_path.parent.iterdir()):
        raise RuntimeError(f"branch output already exists: {result_path.parent}")
    decoder_mesh = decoded.to_trimesh(transform_pose=False)
    if not len(decoder_mesh.vertices) or not len(decoder_mesh.faces):
        raise RuntimeError(f"decoded empty mesh: {method}/{row['object_uid']}")
    source_mesh = decoder_mesh.copy()
    source_mesh.apply_transform(LATENT_DECODER_TO_REFERENCE)
    export_obj_atomic(decoder_mesh, decoder_path)
    export_obj_atomic(source_mesh, source_path)
    target_mesh, target_metadata = load_canonical_gt(row)
    structure = mesh_structure_metrics(source_mesh)
    surface = surface_metrics(
        source_mesh,
        target_mesh,
        count=int(args.surface_samples),
        seed=stable_metric_seed(str(row["uid"]), int(args.seed)),
        thresholds=list(args.fscore_thresholds),
    )
    result = {
        "format": RECORD_FORMAT,
        "created_at_utc": utc_now(),
        "method": method,
        "uid": str(row["uid"]),
        "object_uid": str(row["object_uid"]),
        "source_group": str(row.get("source_group", "unknown")),
        "view_count": _view_count(row),
        "position": int(position),
        "seed": int(args.seed),
        "model": model_path,
        "model_sha256": model_sha256,
        "condition": condition_path,
        "condition_sha256": condition_sha256,
        "oracle_support": support,
        "oracle_support_sha256": support["coords_sha256"],
        "master_noise_seed": int(master_noise_seed),
        "master_noise_sha256": master_noise_sha256,
        "sampling": {
            **MATCHED_SAMPLING,
            "cfg_interval": list(MATCHED_SAMPLING["cfg_interval"]),
        },
        "sampling_sha256": canonical_sha256(MATCHED_SAMPLING),
        "same_support_and_coordinate_keyed_noise_across_methods": True,
        "mesh_decoder": str(decoder_path),
        "mesh_decoder_sha256": sha256_file(decoder_path),
        "mesh_source": str(source_path),
        "mesh_source_sha256": sha256_file(source_path),
        "decoder_to_source_axis_transform": LATENT_DECODER_TO_REFERENCE.tolist(),
        "surface": surface,
        "structure": structure,
        "target": target_metadata,
        "wrapper": wrapper,
        "target_or_metric_consumed": True,
        "passed": bool(structure["mesh_success"]),
    }
    write_json(result_path, result)
    return result


def _prepare_stock_conditions(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    full_manifest: dict[str, Any],
    full_by_object: dict[str, dict[str, Any]],
    output_dir: Path,
    device: torch.device,
):
    pipeline = build_stock_conditioning_pipeline(args.pretrained, device)
    condition_root = output_dir / "stock_conditions"
    for position, row in enumerate(rows):
        object_uid = str(row["object_uid"])
        source_path = _resolve_cache_file(full_manifest, full_by_object[object_uid])
        source_sha = sha256_file(source_path)
        condition_path = condition_root / f"{row['uid']}.pt"
        audit_path = condition_root / f"{row['uid']}.json"
        if condition_path.is_file() and audit_path.is_file():
            audit = load_json(audit_path)
            expected = {
                "uid": str(row["uid"]),
                "source_cache_sha256": source_sha,
                "condition_pt_sha256": sha256_file(condition_path),
                "passed": True,
            }
            mismatch = {
                key: (audit.get(key), value)
                for key, value in expected.items()
                if audit.get(key) != value
            }
            if mismatch:
                raise RuntimeError(f"stale Stock condition={mismatch}")
            continue
        if condition_path.exists() or audit_path.exists():
            raise RuntimeError(f"partial Stock condition: {condition_path}")
        sample = torch.load(source_path, map_location="cpu")
        condition, audit = build_slat_condition(
            pipeline, sample, device=device, condition_tolerance=1.0e-2
        )
        torch_save_atomic(condition, condition_path)
        audit.update(
            {
                "source_cache": str(source_path),
                "source_cache_sha256": source_sha,
                "condition_pt_sha256": sha256_file(condition_path),
            }
        )
        write_json(audit_path, audit)
        print(
            f"[slat_oracle:stock_condition] {position + 1}/{len(rows)} uid={row['uid']}",
            flush=True,
        )
        del sample, condition
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pipeline, condition_root


def _run_stock(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    pipeline: Any,
    condition_root: Path,
    stock_freeze_path: Path,
) -> list[dict[str, Any]]:
    for name in ("image_cond_model", "sparse_structure_vggt_cond", "slat_vggt_cond"):
        pipeline.models[name].cpu()
    pipeline.VGGT_model.cpu()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    flow = pipeline.models["slat_flow_model"].to(device).eval()
    decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    for module in (flow, decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    params = dict(pipeline.slat_sampler_params)
    params.update(MATCHED_SAMPLING)
    sampling_sha = canonical_sha256(MATCHED_SAMPLING)
    stock_freeze_sha = sha256_file(stock_freeze_path)
    records = []
    for position, row in enumerate(rows):
        coords, support = _oracle_support(row)
        condition_path = condition_root / f"{row['uid']}.pt"
        condition_sha = sha256_file(condition_path)
        reused = _reuse_result(
            args=args,
            output_dir=output_dir,
            method=METHODS[0],
            row=row,
            model_sha256=stock_freeze_sha,
            condition_sha256=condition_sha,
            support=support,
            sampling_sha256=sampling_sha,
        )
        if reused is not None:
            records.append(reused)
            continue
        condition_cpu = torch.load(condition_path, map_location="cpu")
        condition = to_device_tree(condition_cpu, device)
        initial, noise_sha, noise_seed = _master_noise(
            seed=int(args.seed), position=position, coords=coords, device=device
        )
        with _amp(args, device):
            sampled = sample_slat_explicit(
                pipeline=pipeline,
                slat_condition=condition,
                initial_noise=initial,
                params=params,
            )
        decoded = decoder(sampled)[0]
        records.append(
            _finish_result(
                args=args,
                output_dir=output_dir,
                method=METHODS[0],
                row=row,
                position=position,
                decoded=decoded,
                model_path=str(stock_freeze_path),
                model_sha256=stock_freeze_sha,
                condition_path=str(condition_path),
                condition_sha256=condition_sha,
                support=support,
                master_noise_sha256=noise_sha,
                master_noise_seed=noise_seed,
                wrapper=None,
            )
        )
        print(
            f"[slat_oracle:{METHODS[0]}] {position + 1}/{len(rows)} uid={row['uid']}",
            flush=True,
        )
        del condition_cpu, condition, initial, sampled, decoded
        if device.type == "cuda":
            torch.cuda.empty_cache()
    flow.cpu()
    decoder.cpu()
    del flow, decoder, pipeline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def _run_native(
    *,
    args: argparse.Namespace,
    method: str,
    rows: list[dict[str, Any]],
    model_by_object: dict[str, dict[str, Any]],
    checkpoint_path: Path,
    stock_freeze_path: Path,
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    checkpoint_sha = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    upstream = dict(checkpoint["model_summary"]["upstream_native_ss"])
    stock_freeze = load_stock_slat_freeze(stock_freeze_path)
    validate_native_slat_no_vggt_checkpoint(
        checkpoint,
        pretrained=args.pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=upstream,
    )
    saved = checkpoint["args"]
    sampler, model, decoder, _, defaults, normalization = (
        build_native_slat_no_vggt_components(
            pretrained=args.pretrained,
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
        raise RuntimeError("Native SLat Mesh decoder is unavailable")
    load_trainable_state_dict(model, checkpoint["ema_trainable_state"])
    model.eval()
    decoder.eval()
    params = dict(defaults)
    params.update(MATCHED_SAMPLING)
    sampling_sha = canonical_sha256(MATCHED_SAMPLING)
    mean = torch.tensor(normalization["mean"], device=device)[None]
    std = torch.tensor(normalization["std"], device=device)[None]
    records = []
    for position, row in enumerate(rows):
        model_row = model_by_object[str(row["object_uid"])]
        sample = _load_model_sample(model_row)
        condition_path = Path(str(model_row["model_input"])).resolve()
        condition_sha = str(model_row["condition_sha256"])
        coords, support = _oracle_support(row)
        reused = _reuse_result(
            args=args,
            output_dir=output_dir,
            method=method,
            row=row,
            model_sha256=checkpoint_sha,
            condition_sha256=condition_sha,
            support=support,
            sampling_sha256=sampling_sha,
        )
        if reused is not None:
            records.append(reused)
            continue
        condition = to_device_tree(sample["slat_condition"], device)
        initial, noise_sha, noise_seed = _master_noise(
            seed=int(args.seed), position=position, coords=coords, device=device
        )
        flow = NativeSLatCalibratedCFGFlow(
            model, condition["cond"], sample, enabled=True, projection_mode="correct"
        )
        with _amp(args, device):
            latent = sampler.sample(
                flow, initial, **condition, **params, verbose=False
            ).samples
        decoded = decoder(latent * std + mean)[0]
        if flow.positive_calls <= 0 or flow.negative_calls <= 0:
            raise RuntimeError(f"{method} did not execute both CFG branches")
        records.append(
            _finish_result(
                args=args,
                output_dir=output_dir,
                method=method,
                row=row,
                position=position,
                decoded=decoded,
                model_path=str(checkpoint_path),
                model_sha256=checkpoint_sha,
                condition_path=str(condition_path),
                condition_sha256=condition_sha,
                support=support,
                master_noise_sha256=noise_sha,
                master_noise_seed=noise_seed,
                wrapper=flow.summary(),
            )
        )
        print(
            f"[slat_oracle:{method}] {position + 1}/{len(rows)} uid={row['uid']}",
            flush=True,
        )
        del sample, condition, initial, latent, decoded, flow
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model.cpu()
    decoder.cpu()
    del sampler, model, decoder, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("paired distribution is empty or non-finite")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "positive_rate": float(np.mean(array > 0.0)),
        "nonnegative_rate": float(np.mean(array >= 0.0)),
    }


def paired_comparison(
    records: list[dict[str, Any]], *, left: str, right: str
) -> dict[str, Any]:
    left_rows = _index_unique(
        (row for row in records if row["method"] == left), "object_uid", left
    )
    right_rows = _index_unique(
        (row for row in records if row["method"] == right), "object_uid", right
    )
    if set(left_rows) != set(right_rows) or not left_rows:
        raise RuntimeError(f"paired object sets differ: {left}/{right}")
    metric_names = sorted(left_rows[next(iter(left_rows))]["surface"])
    deltas: dict[str, list[float]] = {name: [] for name in metric_names}
    per_object = {}
    for object_uid in sorted(left_rows):
        row_deltas = {}
        for name in metric_names:
            left_value = float(left_rows[object_uid]["surface"][name])
            right_value = float(right_rows[object_uid]["surface"][name])
            delta = (
                right_value - left_value
                if name in LOWER_IS_BETTER
                else left_value - right_value
            )
            deltas[name].append(delta)
            row_deltas[name] = delta
        per_object[object_uid] = row_deltas
    summary = {
        f"{name}_left_improvement": _distribution(values)
        for name, values in deltas.items()
    }
    chamfer = summary["chamfer_l1_left_improvement"]
    fscore = summary["fscore_0p02_left_improvement"]
    normal = summary["normal_consistency_left_improvement"]
    if (
        chamfer["mean"] > 0.0
        and chamfer["median"] > 0.0
        and chamfer["positive_rate"] >= 0.625
        and fscore["mean"] >= 0.0
        and normal["mean"] >= 0.0
    ):
        decision = "exploratory_advantage_supported"
    elif (
        chamfer["mean"] < 0.0
        and chamfer["median"] < 0.0
        and chamfer["positive_rate"] <= 0.375
        and fscore["mean"] <= 0.0
        and normal["mean"] <= 0.0
    ):
        decision = "exploratory_disadvantage_supported"
    else:
        decision = "advantage_not_established"
    return {
        "left": left,
        "right": right,
        "positive_definition": f"positive means {left} is better",
        "decision": decision,
        "decision_rule": (
            "Advantage requires positive Chamfer mean/median, >=62.5% object wins, "
            "and nonnegative mean F@0.02 and Normal. This is exploratory, not a "
            "significance claim."
        ),
        "metric_deltas": summary,
        "per_object_deltas": per_object,
    }


def _validate_inference(
    path: Path, *, expected_format: str, expected_objects: set[str], seed: int
) -> dict[str, Any]:
    value = load_json(path)
    expected = {
        "format": expected_format,
        "passed": True,
        "object_count": len(expected_objects),
        "record_count": len(expected_objects),
        "seeds": [int(seed)],
    }
    mismatch = {
        key: (value.get(key), item)
        for key, item in expected.items()
        if value.get(key) != item
    }
    actual = {str(row["object_id"]) for row in value.get("objects", [])}
    if actual != expected_objects:
        mismatch["object_set"] = (sorted(actual), sorted(expected_objects))
    if mismatch:
        raise RuntimeError(f"inference manifest differs {path}: {mismatch}")
    return value


@torch.no_grad()
def main() -> None:
    args = make_parser().parse_args()
    if args.surface_samples <= 0 or args.max_objects < 0:
        raise ValueError("surface_samples must be positive and max_objects nonnegative")
    selection_path = Path(args.selection_manifest).expanduser().resolve()
    selection = load_json(selection_path)
    protocol = dict(selection.get("objaverse16_protocol", {}))
    if (
        protocol.get("format") != PROTOCOL_FORMAT
        or protocol.get("passed") is not True
        or protocol.get("object_count") != 16
        or protocol.get("training_object_disjoint") is not True
        or protocol.get("source_mesh_disjoint") is not True
    ):
        raise RuntimeError("selection is not the frozen disjoint Objaverse16 protocol")
    rows = list(selection["samples"])
    if args.max_objects:
        rows = rows[: int(args.max_objects)]
    selected_objects = {str(row["object_uid"]) for row in rows}
    selection_sha = sha256_file(selection_path)

    model_manifest_path = Path(args.model_input_manifest).expanduser().resolve()
    model_manifest = load_json(model_manifest_path)
    if (
        model_manifest.get("format") != MODEL_INPUT_MANIFEST_FORMAT
        or model_manifest.get("passed") is not True
        or model_manifest.get("selection_manifest_sha256") != selection_sha
    ):
        raise RuntimeError("target-free Objaverse model-input manifest differs")
    model_by_object = _index_unique(
        model_manifest["objects"], "object_uid", "model input"
    )
    if not selected_objects.issubset(model_by_object):
        raise RuntimeError("selected objects are missing target-free model inputs")

    full_manifest_path = Path(args.stock_lifting_manifest).expanduser().resolve()
    full_manifest = load_json(full_manifest_path)
    if full_manifest.get("format") != "ar_ss_flow.pose_lifting_cache.v1":
        raise RuntimeError("Stock lifting manifest format differs")
    full_by_object = _index_unique(
        full_manifest["samples"], "object_uid", "Stock lifting"
    )
    if not selected_objects.issubset(full_by_object):
        raise RuntimeError("selected objects are missing Stock lifting inputs")

    synthetic_path = Path(args.synthetic_inference_manifest).expanduser().resolve()
    mixed_path = Path(args.mixed_inference_manifest).expanduser().resolve()
    synthetic = _validate_inference(
        synthetic_path,
        expected_format=SYNTHETIC_MANIFEST_FORMAT,
        expected_objects={str(row["object_uid"]) for row in selection["samples"]},
        seed=int(args.seed),
    )
    mixed = _validate_inference(
        mixed_path,
        expected_format=MIXED_MANIFEST_FORMAT,
        expected_objects={str(row["object_uid"]) for row in selection["samples"]},
        seed=int(args.seed),
    )
    stock_paths = {str(synthetic["stock_slat_freeze"]), str(mixed["stock_slat_freeze"])}
    if len(stock_paths) != 1:
        raise RuntimeError("synthetic1k and mixed SLat do not share one Stock freeze")
    stock_freeze_path = Path(stock_paths.pop()).expanduser().resolve()

    o11_path = Path(args.existing_o11_report).expanduser().resolve()
    o11 = load_json(o11_path)
    if (
        o11.get("passed") is not True
        or o11.get("object_count") != 16
        or o11.get("selection_manifest_sha256") != selection_sha
    ):
        raise RuntimeError("O11 full-pipeline reference differs")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_objects = []
    for row in rows:
        object_uid = str(row["object_uid"])
        _, support = _oracle_support(row)
        full_cache = _resolve_cache_file(full_manifest, full_by_object[object_uid])
        model_row = model_by_object[object_uid]
        model_input = validate_bound_file(
            model_row["model_input"],
            model_row["model_input_sha256"],
            label="Objaverse model input",
        )
        preflight_objects.append(
            {
                "uid": str(row["uid"]),
                "object_uid": object_uid,
                "oracle_support": support,
                "stock_lifting_cache": str(full_cache),
                "stock_lifting_cache_sha256": sha256_file(full_cache),
                "model_input": str(model_input),
                "model_input_sha256": sha256_file(model_input),
            }
        )
    checkpoint_paths = {
        "synthetic1k_slat": Path(synthetic["native_slat_checkpoint"]).resolve(),
        "mixed_slat": Path(mixed["native_slat_checkpoint"]).resolve(),
        "stock_slat_freeze": stock_freeze_path,
    }
    preflight = {
        "format": "pose_point_depth_mv.objaverse16_slat_oracle_swap_preflight.v1",
        "passed": True,
        "object_count": len(rows),
        "selection_manifest_sha256": selection_sha,
        "checkpoints": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in checkpoint_paths.items()
        },
        "objects": preflight_objects,
    }
    write_json(output_dir / "preflight.json", preflight)
    if args.preflight_only:
        print(
            f"Objaverse16 SLat oracle preflight passed: objects={len(rows)} "
            f"output={output_dir / 'preflight.json'}",
            flush=True,
        )
        return
    device = resolve_torch_device(args.device)
    pipeline, condition_root = _prepare_stock_conditions(
        args=args,
        rows=rows,
        full_manifest=full_manifest,
        full_by_object=full_by_object,
        output_dir=output_dir,
        device=device,
    )
    records = _run_stock(
        args=args,
        rows=rows,
        output_dir=output_dir,
        device=device,
        pipeline=pipeline,
        condition_root=condition_root,
        stock_freeze_path=stock_freeze_path,
    )
    records.extend(
        _run_native(
            args=args,
            method=METHODS[1],
            rows=rows,
            model_by_object=model_by_object,
            checkpoint_path=Path(synthetic["native_slat_checkpoint"]).resolve(),
            stock_freeze_path=stock_freeze_path,
            output_dir=output_dir,
            device=device,
        )
    )
    records.extend(
        _run_native(
            args=args,
            method=METHODS[2],
            rows=rows,
            model_by_object=model_by_object,
            checkpoint_path=Path(mixed["native_slat_checkpoint"]).resolve(),
            stock_freeze_path=stock_freeze_path,
            output_dir=output_dir,
            device=device,
        )
    )

    by_method = {
        method: [row for row in records if row["method"] == method]
        for method in METHODS
    }
    expected_records = len(rows) * len(METHODS)
    passed = len(records) == expected_records and all(
        len(by_method[method]) == len(rows)
        and all(row["passed"] for row in by_method[method])
        for method in METHODS
    )
    comparisons = {
        "synthetic1k_vs_reconviagen_stock": paired_comparison(
            records, left=METHODS[1], right=METHODS[0]
        ),
        "mixed_vs_reconviagen_stock": paired_comparison(
            records, left=METHODS[2], right=METHODS[0]
        ),
        "mixed_vs_synthetic1k": paired_comparison(
            records, left=METHODS[2], right=METHODS[1]
        ),
    }
    view_counts = sorted({_view_count(row) for row in rows})
    summary_by_view_count = {
        str(view_count): {
            method: aggregate(
                [
                    row
                    for row in by_method[method]
                    if int(row["view_count"]) == view_count
                ]
            )
            for method in METHODS
        }
        for view_count in view_counts
    }
    report = {
        "format": REPORT_FORMAT,
        "created_at_utc": utc_now(),
        "passed": passed,
        "formal": False,
        "scope": "SLat-only oracle-support diagnostic; SS generation is removed",
        "object_count": len(rows),
        "record_count": len(records),
        "methods": list(METHODS),
        "seed": int(args.seed),
        "surface_samples": int(args.surface_samples),
        "fscore_thresholds": list(args.fscore_thresholds),
        "support_contract": {
            "source": "frozen repaired GT SS target_coords",
            "resolution": 64,
            "identical_across_methods": True,
            "target_consumed": True,
            "deployment_metric": False,
        },
        "sampling_contract": {
            **MATCHED_SAMPLING,
            "cfg_interval": list(MATCHED_SAMPLING["cfg_interval"]),
            "identical_across_methods": True,
            "coordinate_keyed_initial_noise_identical": True,
        },
        "method_contracts": {
            METHODS[
                0
            ]: "frozen ReconViaGen Stock SLat Flow + intended VGGT SLat condition",
            METHODS[
                1
            ]: "synthetic1k No-VGGT Native-SLat EMA + intended posed DINO condition",
            METHODS[
                2
            ]: "mixed real376+synth868 No-VGGT Native-SLat EMA + intended posed DINO condition",
        },
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_sha,
        "stock_lifting_manifest": str(full_manifest_path),
        "stock_lifting_manifest_sha256": sha256_file(full_manifest_path),
        "model_input_manifest": str(model_manifest_path),
        "model_input_manifest_sha256": sha256_file(model_manifest_path),
        "synthetic_inference_manifest": str(synthetic_path),
        "synthetic_inference_manifest_sha256": sha256_file(synthetic_path),
        "mixed_inference_manifest": str(mixed_path),
        "mixed_inference_manifest_sha256": sha256_file(mixed_path),
        "existing_o11_full_pipeline_reference": {
            "path": str(o11_path),
            "sha256": sha256_file(o11_path),
            "paired_comparisons": o11.get("paired_comparisons"),
            "existing_o9_report": o11.get("existing_o9_report"),
        },
        "summary": {method: aggregate(by_method[method]) for method in METHODS},
        "summary_by_view_count": summary_by_view_count,
        "paired_comparisons": comparisons,
        "records": records,
        "limitations": [
            "The oracle support consumes frozen GT occupancy and is not a deployment input.",
            "The test isolates the SLat stage, including each method's intended conditioner; Stock uses VGGT while Native methods use posed DINO.",
            "Objaverse16 is exploratory and was previously inspected; use a fresh holdout for a formal claim.",
        ],
    }
    report_path = output_dir / "report.json"
    write_json(report_path, report)
    lines = [
        f"Objaverse16 SLat oracle-support swap: passed={passed} objects={len(rows)} records={len(records)}",
    ]
    for name, comparison in comparisons.items():
        chamfer = comparison["metric_deltas"]["chamfer_l1_left_improvement"]
        fscore = comparison["metric_deltas"]["fscore_0p02_left_improvement"]
        normal = comparison["metric_deltas"]["normal_consistency_left_improvement"]
        lines.append(
            f"{name}: decision={comparison['decision']} "
            f"L1_mean={chamfer['mean']:.8f} L1_median={chamfer['median']:.8f} "
            f"win={chamfer['positive_rate']:.3f} F02={fscore['mean']:.8f} "
            f"Normal={normal['mean']:.8f}"
        )
    lines.append(f"report: {report_path}")
    summary_text = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text, end="", flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
