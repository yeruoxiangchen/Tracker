#!/usr/bin/env python3
"""Diagnose end-to-end Mesh sensitivity to the runtime object-frame axes.

This is deliberately a development diagnostic rather than a benchmark.  RGB
features, selected views, checkpoints, sampler settings and random seed stay
fixed.  Only the object-frame orientation visible to the pose-conditioned SS
and SLat branches changes.  Every decoded Mesh is transformed back to the
official ProObjaverse frame before it is scored against the same frozen target.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import trimesh

from pose_point_depth_mv import (
    evaluate_proobjaverse_official_native_ss_stock_slat as endpoint,
)
from pose_point_depth_mv.eval_direct_flow import decode_coords
from pose_point_depth_mv.evaluate_native_ss_stock_slat_mesh import write_json
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (
    canonical_coords,
    mesh_structure_metrics,
    sparse_noise_from_master,
    surface_metrics,
)
from pose_point_depth_mv.native_3d_condition import NativeConditionSLatDataset
from pose_point_depth_mv.native_ss_genrecon import NativeSSCalibratedCFGFlow
from pose_point_depth_mv.proobjaverse_official_slat_protocol import (
    canonical_sha256,
    sha256_file,
)
from pose_point_depth_mv.real_object_canonicalization import _estimate_axes
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree


REPORT_FORMAT = "pose_point_depth_mv.official30k_dev8_o_rotation_endpoint.v1"
ARMS = (
    "official_o",
    "official_o_rx90",
    "official_o_ry90",
    "official_o_rz90",
    "phone_o",
    "phone_o_rx90",
    "phone_o_ry90",
    "phone_o_rz90",
)


def _rotation(axis: str, degrees: float) -> np.ndarray:
    angle = math.radians(float(degrees))
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        value = ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    elif axis == "y":
        value = ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    elif axis == "z":
        value = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    else:
        raise ValueError(f"unknown axis={axis!r}")
    result = np.asarray(value, dtype=np.float64)
    if not np.allclose(result.T @ result, np.eye(3), atol=1.0e-12):
        raise RuntimeError("rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(result)), 1.0, abs_tol=1.0e-12):
        raise RuntimeError("rotation is not proper")
    return result


def _w2c_from_sample(sample: dict[str, Any]) -> np.ndarray:
    extrinsics = sample["extrinsics"].detach().cpu().numpy().astype(np.float64)
    kind = str(sample["extrinsics_type"])
    if kind == "w2c":
        return extrinsics
    if kind == "c2w":
        return np.linalg.inv(extrinsics)
    raise ValueError(f"unsupported extrinsics_type={kind!r}")


def phone_o_to_official(sample: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the deployed phone axis rule in the synthetic official world.

    The synthetic world has physical up along official +Z, so that vector is
    supplied as gravity.  Center and scale are held exact at zero and one: this
    experiment isolates orientation and does not test mask-derived translation
    or scale accuracy.
    """

    rotation, stats = _estimate_axes(
        np.zeros(3, dtype=np.float64),
        _w2c_from_sample(sample),
        gravity_up_W=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        reference_view_index=0,
    )
    return rotation, stats


def arm_to_official(
    arm: str, sample: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unsupported arm={arm!r}")
    if arm.startswith("phone_o"):
        base, phone_stats = phone_o_to_official(sample)
        base_name = "phone_pose_mask_axis_rule_orientation_only"
    else:
        base = np.eye(3, dtype=np.float64)
        phone_stats = None
        base_name = "official_proobjaverse_o"
    suffix = arm.removeprefix("phone_o").removeprefix("official_o")
    local = np.eye(3, dtype=np.float64)
    if suffix:
        if suffix not in {"_rx90", "_ry90", "_rz90"}:
            raise ValueError(f"invalid arm suffix={suffix!r}")
        local = _rotation(suffix[2], 90.0)
    rotation = base @ local
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform, {
        "arm": arm,
        "base": base_name,
        "local_rotation": suffix.removeprefix("_") or "identity",
        "T_arm_O_to_official_O": transform.tolist(),
        "determinant": float(np.linalg.det(rotation)),
        "phone_axis_stats": phone_stats,
        "phone_up_axis_in_official": (
            base[:, 1].tolist() if arm.startswith("phone_o") else None
        ),
        "phone_front_axis_in_official": (
            base[:, 2].tolist() if arm.startswith("phone_o") else None
        ),
    }


def rotate_lifting_sample(
    sample: dict[str, Any], transform_arm_to_official: np.ndarray
) -> dict[str, Any]:
    """Re-express unchanged cameras in the selected arm's O coordinates."""

    result = dict(sample)
    transform = torch.as_tensor(
        transform_arm_to_official,
        dtype=sample["extrinsics"].dtype,
        device=sample["extrinsics"].device,
    )
    kind = str(sample["extrinsics_type"])
    if kind == "w2c":
        result["extrinsics"] = torch.matmul(sample["extrinsics"], transform)
    elif kind == "c2w":
        result["extrinsics"] = torch.matmul(
            torch.linalg.inv(transform), sample["extrinsics"]
        )
    else:
        raise ValueError(f"unsupported extrinsics_type={kind!r}")
    result.pop("_native_projection_cache_v1", None)
    return result


def _target_mesh(path: Path) -> trimesh.Trimesh:
    with np.load(path, allow_pickle=False) as payload:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(payload["vertices"]),
            faces=np.asarray(payload["faces"]),
            process=False,
        )
    if mesh_structure_metrics(mesh)["mesh_success"] is not True:
        raise RuntimeError(f"target Mesh is invalid: {path}")
    return mesh


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--lifting_cache_manifest", required=True)
    parser.add_argument("--native_ss_report", required=True)
    parser.add_argument("--stock_slat_freeze", required=True)
    parser.add_argument("--trained_slat_checkpoint", required=True)
    parser.add_argument("--target_mesh_cache_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained", default="Stable-X/trellis-vggt-v0-2")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--expected_trained_slat_step", type=int, default=30000)
    parser.add_argument("--joint_seeds", default="42")
    parser.add_argument("--object_start", type=int, default=0)
    parser.add_argument("--object_end", type=int, default=8)
    parser.add_argument("--surface_samples", type=int, default=20000)
    parser.add_argument("--save_meshes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


@torch.no_grad()
def main() -> None:
    args = _parser().parse_args()
    seeds = endpoint.parse_csv(args.joint_seeds, int)
    dataset = NativeConditionSLatDataset(
        args.cache_manifest, args.lifting_cache_manifest, indices="all"
    )
    start, end = int(args.object_start), int(args.object_end)
    if start < 0 or end <= start or end > len(dataset):
        raise ValueError(f"invalid object slice [{start}:{end}] / {len(dataset)}")
    selected = list(range(start, end))
    selected_uids = [str(dataset.rows[index]["object_uid"]) for index in selected]
    target_contract = endpoint._official_target_contract(dataset)
    output = Path(args.output_dir).expanduser().resolve()
    identity = {
        "format": REPORT_FORMAT,
        "arm": str(args.arm),
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "lifting_cache_manifest": str(Path(args.lifting_cache_manifest).resolve()),
        "lifting_cache_manifest_sha256": sha256_file(args.lifting_cache_manifest),
        "native_ss_report": str(Path(args.native_ss_report).resolve()),
        "native_ss_report_sha256": sha256_file(args.native_ss_report),
        "stock_slat_freeze": str(Path(args.stock_slat_freeze).resolve()),
        "stock_slat_freeze_sha256": sha256_file(args.stock_slat_freeze),
        "trained_slat_checkpoint": str(Path(args.trained_slat_checkpoint).resolve()),
        "trained_slat_checkpoint_sha256": sha256_file(args.trained_slat_checkpoint),
        "target_mesh_cache_root": str(Path(args.target_mesh_cache_root).resolve()),
        "official_protocol_sha256": str(target_contract["protocol_sha256"]),
        "object_start": start,
        "object_end": end,
        "object_uids": selected_uids,
        "joint_seeds": seeds,
        "weights": str(args.weights),
        "amp_dtype": str(args.amp_dtype),
        "surface_samples": int(args.surface_samples),
        "same_images": True,
        "same_selected_views": True,
        "same_checkpoints": True,
        "same_sampler_settings": True,
        "same_seed_and_o_indexed_noise": True,
        "score_frame": "all predictions mapped back to official O before scoring",
        "phone_o_scope": "orientation only; exact synthetic center and scale retained",
    }
    identity_path = output / "run_identity.json"
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("resume identity differs")
    else:
        write_json(identity_path, identity)
    report_path = output / "report.json"
    if report_path.is_file():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        meshes_complete = bool(existing_report.get("records")) and all(
            isinstance(row.get("mesh"), str)
            and Path(str(row["mesh"])).is_file()
            and str(row.get("mesh_sha256", "")) == sha256_file(row["mesh"])
            for row in existing_report.get("records", [])
        )
        if not args.save_meshes or meshes_complete:
            print(json.dumps({"reused": True, "report": str(report_path)}, indent=2))
            return
        print(
            json.dumps(
                {
                    "resume_mesh_export": True,
                    "arm": args.arm,
                    "missing_meshes": sum(
                        not isinstance(row.get("mesh"), str)
                        or not Path(str(row.get("mesh", ""))).is_file()
                        for row in existing_report.get("records", [])
                    ),
                },
                indent=2,
            ),
            flush=True,
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    # The helper enforces the exact frozen SS30K deployment binding.
    args.allow_native_ss_science_failed = False
    (
        ss_binding,
        ss_checkpoint,
        ss_sampler,
        ss_model,
        ss_decoder,
        ss_summary,
        ss_params,
    ) = endpoint._load_ss_runtime(args, dataset, selected, target_contract, device)
    coord_root = output / "predicted_support"
    frame_records: dict[str, dict[str, Any]] = {}
    for position, index in enumerate(selected, start=1):
        sample = dataset[index]
        uid = str(sample["object_uid"])
        transform, frame_record = arm_to_official(args.arm, sample["lifting_sample"])
        frame_records[uid] = frame_record
        lifting = rotate_lifting_sample(sample["lifting_sample"], transform)
        positive = lifting["stock_condition"].to(device=device)
        negative = torch.zeros_like(positive)
        for seed in seeds:
            current = endpoint.pair_id(uid, seed)
            coord_path = coord_root / f"{current}.npz"
            if coord_path.is_file():
                continue
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 1000003 + int(index) * 1009
            )
            initial = torch.randn(
                (1, 8, 16, 16, 16), generator=generator, device=device
            )
            native_flow = NativeSSCalibratedCFGFlow(
                ss_model,
                positive,
                lifting,
                enabled=True,
                projection_mode="correct",
            )
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                latent = ss_sampler.sample(
                    native_flow,
                    initial,
                    cond=positive,
                    neg_cond=negative,
                    **ss_params,
                    verbose=False,
                ).samples
            coords = decode_coords(ss_decoder, latent)
            if not len(coords):
                raise RuntimeError(f"empty predicted support uid={uid} seed={seed}")
            coord_root.mkdir(parents=True, exist_ok=True)
            with coord_path.open("wb") as handle:
                np.savez_compressed(handle, coords=coords.astype(np.int32))
            print(
                f"[o_rotation:ss] arm={args.arm} {position}/{len(selected)} "
                f"seed={seed} uid={uid} points={len(coords)}",
                flush=True,
            )
            del initial, latent, native_flow
            torch.cuda.empty_cache()
        del positive, negative, lifting
    ss_model.cpu()
    ss_decoder.cpu()
    del ss_model, ss_decoder, ss_sampler, ss_checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    stock_freeze = endpoint.load_stock_slat_freeze(args.stock_slat_freeze)
    trained = endpoint._build_trained_slat_pipeline(
        checkpoint_path=args.trained_slat_checkpoint,
        weights=str(args.weights),
        pretrained=str(args.pretrained),
        stock_freeze=stock_freeze,
        dataset=dataset,
        expected_step=int(args.expected_trained_slat_step),
        device=device,
        evaluation_object_uids=selected_uids,
        allow_target_protocol_mismatch=False,
        expected_training_membership="all_disjoint",
    )
    channels = int(trained["model"].flow_core.in_channels)
    records: list[dict[str, Any]] = []
    pair_root = output / "mesh_pairs"
    target_root = Path(args.target_mesh_cache_root).expanduser().resolve()
    for position, index in enumerate(selected, start=1):
        sample = dataset[index]
        uid = str(sample["object_uid"])
        transform = np.asarray(
            frame_records[uid]["T_arm_O_to_official_O"], dtype=np.float64
        )
        lifting = rotate_lifting_sample(sample["lifting_sample"], transform)
        condition = to_device_tree(sample["condition"], device)
        target_path = target_root / f"{uid}.npz"
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        target_mesh = _target_mesh(target_path)
        for seed in seeds:
            current = endpoint.pair_id(uid, seed)
            record_path = pair_root / current / "record.json"
            if record_path.is_file():
                existing = json.loads(record_path.read_text(encoding="utf-8"))
                mesh_binding_ok = (
                    isinstance(existing.get("mesh"), str)
                    and Path(str(existing["mesh"])).is_file()
                    and str(existing.get("mesh_sha256", ""))
                    == sha256_file(existing["mesh"])
                )
                if not args.save_meshes or mesh_binding_ok:
                    records.append(existing)
                    continue
            coord_path = coord_root / f"{current}.npz"
            with np.load(coord_path, allow_pickle=False) as payload:
                coords = canonical_coords(payload["coords"], resolution=64)
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 2000003 + int(index) * 2017 + 7919
            )
            master = torch.randn(
                (64, 64, 64, channels),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            initial = sparse_noise_from_master(coords, master, device=device)
            latent, flow_summary = endpoint._sample_trained_slat(
                runtime=trained,
                initial=initial,
                condition=condition,
                lifting_sample=lifting,
                adapted=True,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            active = int(latent.feats.shape[0])
            if active > endpoint.MAX_SAFE_SLAT_DECODER_INPUT_POINTS:
                raise RuntimeError(
                    f"SLat decoder input exceeds safe active-point limit: {active}"
                )
            decoded = trained["decoder"](latent)[0]
            mesh_arm = decoded.to_trimesh(transform_pose=False)
            mesh_official = mesh_arm.copy()
            mesh_official.apply_transform(transform)
            structure = mesh_structure_metrics(mesh_official)
            if structure["mesh_success"] is not True:
                raise RuntimeError(f"decoded Mesh is invalid uid={uid} seed={seed}")
            surface = surface_metrics(
                mesh_official,
                target_mesh,
                count=int(args.surface_samples),
                seed=int(seed) * 1009 + int(index) * 9173,
                thresholds=(0.01, 0.02, 0.05),
            )
            record = {
                "object_uid": uid,
                "seed": int(seed),
                "arm": str(args.arm),
                "passed": True,
                "predicted_support_count": int(len(coords)),
                "slat_active_point_count": active,
                "surface": surface,
                "structure": structure,
                "frame": frame_records[uid],
                "target_mesh": str(target_path),
                "target_mesh_sha256": sha256_file(target_path),
                "flow_summary": flow_summary,
            }
            if args.save_meshes:
                mesh_path = pair_root / current / "mesh_official_o.obj"
                mesh_path.parent.mkdir(parents=True, exist_ok=True)
                mesh_official.export(mesh_path)
                record["mesh"] = str(mesh_path)
                record["mesh_sha256"] = sha256_file(mesh_path)
            write_json(record_path, record)
            records.append(record)
            print(
                f"[o_rotation:mesh] arm={args.arm} {position}/{len(selected)} "
                f"seed={seed} uid={uid} chamfer={surface['chamfer_l1']:.8f}",
                flush=True,
            )
            del master, initial, latent, decoded, mesh_arm, mesh_official
            torch.cuda.empty_cache()
        del condition, lifting, target_mesh
    trained["model"].cpu()
    trained["decoder"].cpu()
    del trained
    gc.collect()
    torch.cuda.empty_cache()

    report = {
        "format": REPORT_FORMAT,
        "complete": True,
        "passed": len(records) == len(selected) * len(seeds),
        "formal": False,
        "run_identity": identity,
        "native_ss_binding": ss_binding,
        "native_ss_model_summary": ss_summary,
        "frame_records": frame_records,
        "records": records,
        "scope_guard": (
            "Dev8 one-seed object-frame rotation sensitivity diagnostic only; "
            "not a generalization or benchmark claim"
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "arm": args.arm,
                "objects": len(selected),
                "records": len(records),
                "report": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
