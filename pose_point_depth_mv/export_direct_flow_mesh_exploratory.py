#!/usr/bin/env python3
"""Export non-formal exploratory direct-Flow SLAT/Mesh pairs from Stage-B.

This entry point intentionally does not modify or replace the frozen Stage-B
exporter.  It reuses validated ``ss_coords`` and ``slat_conditions`` from a
partial formal run, preserves its joint-seed/shared-noise pairing, skips the
failed deterministic-repeat gate, and writes a separate output that is marked
exploratory/non-formal throughout.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import numpy as np
import torch
import trimesh


TRACKER_ROOT = Path(__file__).resolve().parents[1]
RECONVIAGEN_ROOT = TRACKER_ROOT / "ReconViaGen"
VGGT_ROOT = RECONVIAGEN_ROOT / "wheels" / "vggt"
NINJA_WRAPPER_ROOT = TRACKER_ROOT / "pose_point_depth_mv" / "tools"
if (NINJA_WRAPPER_ROOT / "ninja").is_file():
    os.environ["PATH"] = (
        f"{NINJA_WRAPPER_ROOT}{os.pathsep}{os.environ.get('PATH', '')}"
    )
for import_root in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset  # noqa: E402
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (  # noqa: E402
    aggregate_report,
    canonical_coords,
    load_canonical_gt,
    mesh_structure_metrics,
    pair_identity,
    sample_slat_explicit,
    save_preview,
    shared_noise_audit,
    sparse_from_payload,
    sparse_noise_from_master,
    sparse_payload,
    surface_metrics,
    tensor_sha256,
    tensor_tree_sha256,
    to_device_tree,
    torch_save_atomic,
    validate_completion_manifest,
    validate_condition_artifact,
    validate_pair_record,
    validate_protocol,
    validate_slat_artifact,
    validate_ss_artifact,
    write_json,
    write_text,
)
from pose_point_depth_mv.prepare_direct_flow_mesh_protocol import (  # noqa: E402
    sha256_file,
)
from trellis import models  # noqa: E402
from trellis.pipelines import samplers  # noqa: E402


REPORT_FORMAT = "pose_point_depth_mv.exploratory_slat_mesh_report.v1"
COMPLETION_FORMAT = "pose_point_depth_mv.exploratory_slat_mesh_completion.v1"
NON_FORMAL_NOTICE = (
    "EXPLORATORY/NON-FORMAL: the frozen deterministic-repeat SLAT gate was "
    "bypassed after its recorded failure. These meshes may be inspected and "
    "summarized, but this run cannot replace or pass the formal Stage-B gate."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        required=True,
        help="Frozen Stage-B protocol.json used by the partial source run.",
    )
    parser.add_argument(
        "--blind_key_file",
        required=True,
        help="Private key matching the frozen protocol commitment.",
    )
    parser.add_argument(
        "--source_output",
        "--source_dir",
        dest="source_output",
        required=True,
        help="Partial formal B3 output containing ss_coords/slat_conditions.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Separate exploratory output directory; never use source_dir here.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp_dtype",
        choices=("bf16", "fp16", "none"),
        default="bf16",
        help="Must match the frozen protocol; retained as runtime provenance.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max_objects",
        type=int,
        default=0,
        help="Optional prefix size for a quick visual pilot; 0 uses all 32.",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated subset of frozen joint seeds.",
    )
    parser.add_argument(
        "--surface_samples",
        type=int,
        default=0,
        help="Points per Mesh/GT surface; 0 uses the frozen protocol value.",
    )
    parser.add_argument(
        "--preview_pairs",
        type=int,
        default=0,
        help="Render previews for the first N uid/seed pairs; 0 saves only GLB/OBJ.",
    )
    return parser.parse_args()


def load_minimal_slat_pipeline(pretrained: str | Path) -> SimpleNamespace:
    """Load only the SLAT flow and Mesh decoder required after cached conditions."""
    root = Path(pretrained).resolve()
    config_path = root / "pipeline.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"exploratory offline loader requires a local pretrained snapshot: {root}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))["args"]
    model_config = config["models"]
    loaded_models = {
        name: models.from_pretrained(str(root / model_config[name])).eval()
        for name in ("slat_flow_model", "slat_decoder_mesh")
    }
    sampler_config = config["slat_sampler"]
    slat_sampler = getattr(samplers, sampler_config["name"])(
        **sampler_config.get("args", {})
    )
    return SimpleNamespace(
        models=loaded_models,
        slat_sampler=slat_sampler,
        slat_sampler_params=dict(sampler_config.get("params", {})),
        slat_normalization=config["slat_normalization"],
    )


def validate_source(
    *,
    protocol: dict[str, Any],
    protocol_path: Path,
    blind_key: bytes,
    source_dir: Path,
    selected_rows: list[dict[str, Any]],
    seeds: list[int],
) -> tuple[PoseLiftingCacheDataset, dict[str, Any]]:
    if source_dir.resolve() == protocol_path.parent.resolve():
        raise ValueError("source_dir must be the partial run, not the protocol directory")
    source_protocol = source_dir / "protocol.json"
    if not source_protocol.is_file() or sha256_file(source_protocol) != sha256_file(
        protocol_path
    ):
        raise RuntimeError("partial source is not bound to the requested protocol")
    source_identity_path = source_dir / "run_identity.json"
    if not source_identity_path.is_file():
        raise FileNotFoundError(source_identity_path)
    source_identity = json.loads(source_identity_path.read_text(encoding="utf-8"))
    if source_identity.get("formal") is not True:
        raise RuntimeError("source_dir is not the formal B3 partial run")
    if source_identity.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise RuntimeError("source run/protocol identity mismatch")

    bindings = protocol["bindings"]
    rollout_indices = Path(bindings["rollout_indices"]["path"]).read_text(
        encoding="utf-8"
    ).strip()
    if not rollout_indices:
        raise RuntimeError("bound Stage-A indices are empty")
    dataset = PoseLiftingCacheDataset(
        bindings["cache_manifest"]["path"], indices=rollout_indices
    )
    if len(dataset) != 32:
        raise RuntimeError(f"expected frozen 32-row dataset, got {len(dataset)}")

    condition_count = 0
    ss_count = 0
    for frozen in selected_rows:
        rollout_position = int(frozen["rollout_position"])
        sample = dataset[rollout_position]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        if uid != str(frozen["uid"]) or object_uid != str(frozen["object_uid"]):
            raise RuntimeError("dataset identity/order differs from frozen protocol")
        condition_path = source_dir / "slat_conditions" / f"{uid}.pt"
        condition_audit_path = source_dir / "slat_conditions" / f"{uid}.json"
        validate_condition_artifact(condition_path, condition_audit_path, uid=uid)
        condition_count += 1
        for seed in seeds:
            pair_id, _ = pair_identity(
                protocol["protocol_name"], uid, int(seed), blind_key
            )
            validate_ss_artifact(
                source_dir / "ss_coords" / f"{pair_id}.npz",
                source_dir / "ss_coords" / f"{pair_id}.json",
                pair_id=pair_id,
                uid=uid,
                object_uid=object_uid,
                seed=int(seed),
                rollout_position=rollout_position,
            )
            ss_count += 1

    repeat_audit_path = source_dir / "determinism_slat_audit.json"
    repeat_audit = (
        json.loads(repeat_audit_path.read_text(encoding="utf-8"))
        if repeat_audit_path.is_file()
        else None
    )
    return dataset, {
        "source_dir": str(source_dir),
        "source_protocol": str(source_protocol),
        "source_protocol_sha256": sha256_file(source_protocol),
        "source_run_identity": source_identity,
        "validated_condition_count": condition_count,
        "validated_ss_pair_count": ss_count,
        "recorded_deterministic_repeat_slat_audit": repeat_audit,
        "deterministic_repeat_gate_bypassed": True,
        "bypass_scope": "exploratory visual/numeric Mesh inspection only",
    }


def export_mesh_file(mesh: trimesh.Trimesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    mesh.export(temporary)
    os.replace(temporary, path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = validate_protocol(protocol_path)
    source_dir = Path(args.source_output).resolve()
    output_dir = Path(args.output_dir).resolve()
    if source_dir == output_dir:
        raise ValueError("output_dir must differ from the formal source_dir")

    blind_key_path = Path(args.blind_key_file).resolve()
    blind_key = bytes.fromhex(blind_key_path.read_text(encoding="ascii").strip())
    blind_key_commitment = hashlib.sha256(blind_key).hexdigest()
    if blind_key_commitment != str(
        protocol["blinding"]["blind_key_sha256_commitment"]
    ):
        raise RuntimeError("blind key does not match the frozen protocol commitment")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("SLAT sampling/Mesh decoding requires CUDA")
    torch.cuda.set_device(0 if device.index is None else int(device.index))
    if int(args.max_objects) < 0:
        raise ValueError("max_objects must be non-negative")
    if int(args.surface_samples) < 0:
        raise ValueError("surface_samples must be non-negative")
    if int(args.preview_pairs) < 0:
        raise ValueError("preview_pairs must be non-negative")
    if str(args.amp_dtype) != str(protocol["runtime"]["amp_dtype"]):
        raise ValueError("runtime amp dtype differs from the frozen protocol")

    selected_rows = list(protocol["selection"]["rows"])
    if int(args.max_objects) > 0:
        selected_rows = selected_rows[: int(args.max_objects)]
    frozen_seeds = [int(value) for value in protocol["sampling"]["joint_seeds"]]
    seeds = frozen_seeds
    if args.seeds:
        seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
        if not seeds or len(set(seeds)) != len(seeds) or any(
            value not in frozen_seeds for value in seeds
        ):
            raise ValueError("--seeds must be a unique non-empty frozen-seed subset")
    if not selected_rows:
        raise ValueError("empty exploratory selection")
    surface_samples = int(args.surface_samples) or int(
        protocol["mesh"]["surface_samples"]
    )

    run_identity = {
        "format": "pose_point_depth_mv.exploratory_slat_mesh_run.v1",
        "exploratory": True,
        "formal": False,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "source_dir": str(source_dir),
        "selected_uids": [str(row["uid"]) for row in selected_rows],
        "seeds": seeds,
        "surface_samples": surface_samples,
        "amp_dtype": str(args.amp_dtype),
        "preview_pairs": int(args.preview_pairs),
        "deterministic_repeat_slat_audit": "SKIPPED_BY_EXPLORATORY_ENTRY_POINT",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            f"exploratory output exists; pass --resume to validate/reuse it: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "run_identity.json"
    if identity_path.is_file():
        existing_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing_identity != run_identity:
            raise RuntimeError("resume arguments/script differ from existing exploratory run")
    else:
        write_json(identity_path, run_identity)
        write_text(output_dir / "EXPLORATORY_NON_FORMAL.txt", NON_FORMAL_NOTICE + "\n")
        shutil.copy2(protocol_path, output_dir / "source_frozen_protocol.json")

    completion_path = output_dir / "completion_manifest.json"
    if completion_path.is_file():
        completion = validate_completion_manifest(output_dir, expected_formal=False)
        if completion.get("format") != COMPLETION_FORMAT:
            raise RuntimeError("unexpected exploratory completion format")
        print(
            json.dumps(
                {
                    "reused_complete_run": True,
                    "exploratory": True,
                    "formal": False,
                    "runtime_exit_code": completion["runtime_exit_code"],
                    "report": str(output_dir / "report.json"),
                },
                indent=2,
            ),
            flush=True,
        )
        raise SystemExit(int(completion["runtime_exit_code"]))

    dataset, source_validation = validate_source(
        protocol=protocol,
        protocol_path=protocol_path,
        blind_key=blind_key,
        source_dir=source_dir,
        selected_rows=selected_rows,
        seeds=seeds,
    )
    write_json(output_dir / "source_validation.json", source_validation)
    print(NON_FORMAL_NOTICE, flush=True)
    print(
        "[exploratory:source] "
        f"conditions={source_validation['validated_condition_count']} "
        f"ss_pairs={source_validation['validated_ss_pair_count']}",
        flush=True,
    )

    pipeline = load_minimal_slat_pipeline(protocol["pretrained"])
    slat_flow = pipeline.models["slat_flow_model"].to(device).eval()
    slat_resolution = int(getattr(slat_flow, "resolution", 64))
    slat_channels = int(slat_flow.in_channels)
    if slat_resolution != 64:
        raise RuntimeError(f"unexpected SLAT resolution={slat_resolution}")
    slat_params = dict(pipeline.slat_sampler_params)
    slat_params.update(protocol["sampling"]["slat"])
    slat_params.pop("noise_field", None)
    slat_params.pop("seed_formula", None)
    slat_root = output_dir / "slat"
    slat_root.mkdir(exist_ok=True)

    for frozen in selected_rows:
        rollout_position = int(frozen["rollout_position"])
        sample = dataset[rollout_position]
        uid = str(sample["uid"])
        condition_cpu = torch.load(
            source_dir / "slat_conditions" / f"{uid}.pt", map_location="cpu"
        )
        slat_condition = to_device_tree(condition_cpu, device)
        for seed in seeds:
            pair_id, side_mapping = pair_identity(
                protocol["protocol_name"], uid, int(seed), blind_key
            )
            pair_dir = slat_root / pair_id
            pair_dir.mkdir(exist_ok=True)
            audit_path = pair_dir / "noise_audit.json"
            branch_paths = {
                branch: pair_dir / f"{branch}.pt"
                for branch in ("stock", "correct")
            }
            if audit_path.is_file() and all(
                path.is_file() for path in branch_paths.values()
            ):
                validate_slat_artifact(
                    audit_path,
                    branch_paths,
                    pair_id=pair_id,
                    uid=uid,
                    seed=int(seed),
                )
                continue

            coord_path = source_dir / "ss_coords" / f"{pair_id}.npz"
            with np.load(coord_path, allow_pickle=False) as payload:
                coords = {
                    branch: canonical_coords(
                        payload[branch], resolution=slat_resolution
                    )
                    for branch in ("stock", "correct")
                }
            master_seed = int(seed) * 2000003 + rollout_position * 2017 + 7919
            generator = torch.Generator(device=device).manual_seed(master_seed)
            master = torch.randn(
                (
                    slat_resolution,
                    slat_resolution,
                    slat_resolution,
                    slat_channels,
                ),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            initial_noise = {
                branch: sparse_noise_from_master(
                    coords[branch], master, device=device
                )
                for branch in ("stock", "correct")
            }
            noise_audit = shared_noise_audit(
                initial_noise["stock"].coords,
                initial_noise["stock"].feats,
                initial_noise["correct"].coords,
                initial_noise["correct"].feats,
            )
            expected_noise_diff = float(
                protocol["audits"]["shared_slat_noise_common_coord_max_abs"]
            )
            if noise_audit["common_coord_noise_max_abs"] != expected_noise_diff:
                raise RuntimeError(f"shared SLAT noise audit failed for {pair_id}")

            branch_order = [side_mapping[side] for side in ("A", "B")]
            for branch in branch_order:
                sampled = sample_slat_explicit(
                    pipeline=pipeline,
                    slat_condition=slat_condition,
                    initial_noise=initial_noise[branch],
                    params=slat_params,
                )
                torch_save_atomic(sparse_payload(sampled), branch_paths[branch])
                del sampled
                torch.cuda.empty_cache()
            write_json(
                audit_path,
                {
                    "pair_id": pair_id,
                    "uid": uid,
                    "seed": int(seed),
                    "exploratory": True,
                    "formal": False,
                    "deterministic_repeat_slat_audit": "SKIPPED",
                    "master_noise_seed": master_seed,
                    "master_noise_sha256": tensor_sha256(master),
                    "master_noise_shape": list(master.shape),
                    "condition_sha256": tensor_tree_sha256(condition_cpu),
                    "initial_noise_sha256": {
                        branch: tensor_tree_sha256(
                            sparse_payload(initial_noise[branch])
                        )
                        for branch in ("stock", "correct")
                    },
                    "branch_payload_sha256": {
                        branch: sha256_file(branch_paths[branch])
                        for branch in ("stock", "correct")
                    },
                    "private_branch_execution_order": branch_order,
                    **noise_audit,
                    "passed": noise_audit["common_coord_noise_bit_exact"],
                },
            )
            validate_slat_artifact(
                audit_path,
                branch_paths,
                pair_id=pair_id,
                uid=uid,
                seed=int(seed),
            )
            del master, initial_noise
            torch.cuda.empty_cache()
            print(f"[exploratory:slat] {pair_id}", flush=True)
        del condition_cpu, slat_condition
    slat_flow.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    mesh_decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    mesh_root = output_dir / "blind_pairs"
    private_root = output_dir / "private_records"
    mesh_root.mkdir(exist_ok=True)
    private_root.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    blind_manifest: list[dict[str, Any]] = []
    unblinding: dict[str, Any] = {}
    ordered_pair_ids = [
        pair_identity(
            protocol["protocol_name"], str(frozen["uid"]), int(seed), blind_key
        )[0]
        for frozen in selected_rows
        for seed in seeds
    ]
    preview_pair_ids = set(ordered_pair_ids[: int(args.preview_pairs)])

    for frozen in selected_rows:
        rollout_position = int(frozen["rollout_position"])
        sample = dataset[rollout_position]
        uid = str(sample["uid"])
        object_uid = str(sample.get("object_uid", uid))
        target_mesh, target_metadata = load_canonical_gt(sample)
        for seed in seeds:
            pair_id, side_mapping = pair_identity(
                protocol["protocol_name"], uid, int(seed), blind_key
            )
            pair_dir = mesh_root / pair_id
            private_pair_dir = private_root / pair_id
            pair_dir.mkdir(exist_ok=True)
            private_pair_dir.mkdir(exist_ok=True)
            pair_record_path = private_pair_dir / "pair_record.json"
            if pair_record_path.is_file():
                existing_pair = validate_pair_record(
                    pair_record_path,
                    pair_id=pair_id,
                    uid=uid,
                    object_uid=object_uid,
                    seed=int(seed),
                )
                if existing_pair.get("passed") is True:
                    records.extend(existing_pair["branches"])
                    blind_manifest.append(existing_pair["blind_manifest"])
                    unblinding[pair_id] = existing_pair["unblinding"]
                    continue

            branch_records = []
            for side in ("A", "B"):
                branch = side_mapping[side]
                side_dir = pair_dir / side
                side_dir.mkdir(exist_ok=True)
                row: dict[str, Any] = {
                    "pair_id": pair_id,
                    "side": side,
                    "branch": branch,
                    "uid": uid,
                    "object_uid": object_uid,
                    "views": int(frozen["views"]),
                    "seed": int(seed),
                    "exploratory": True,
                    "formal": False,
                    "passed": False,
                }
                try:
                    slat_payload = torch.load(
                        slat_root / pair_id / f"{branch}.pt", map_location="cpu"
                    )
                    slat = sparse_from_payload(slat_payload, device)
                    decoded = mesh_decoder(slat)[0]
                    canonical_mesh = decoded.to_trimesh(transform_pose=False)
                    view_mesh = decoded.to_trimesh(transform_pose=True)
                    structure = mesh_structure_metrics(canonical_mesh)
                    if not structure["mesh_success"]:
                        raise RuntimeError("decoded mesh is empty or non-finite")

                    obj_path = side_dir / "mesh_canonical.obj"
                    glb_path = side_dir / "mesh_view.glb"
                    export_mesh_file(canonical_mesh, obj_path)
                    export_mesh_file(view_mesh, glb_path)
                    reopened_obj = trimesh.load(obj_path, force="mesh", process=False)
                    reopened_glb = trimesh.load(glb_path, force="scene", process=False)
                    if not len(reopened_obj.vertices) or not len(reopened_obj.faces):
                        raise RuntimeError("canonical OBJ roundtrip is empty")
                    glb_geometry = (
                        list(reopened_glb.geometry.values())
                        if isinstance(reopened_glb, trimesh.Scene)
                        else [reopened_glb]
                    )
                    if not any(
                        len(item.vertices) and len(item.faces)
                        for item in glb_geometry
                    ):
                        raise RuntimeError("view GLB roundtrip is empty")

                    surface = surface_metrics(
                        canonical_mesh,
                        target_mesh,
                        count=surface_samples,
                        seed=int(seed) * 1009 + rollout_position * 9173,
                        thresholds=protocol["mesh"]["fscore_thresholds"],
                    )
                    preview = None
                    if pair_id in preview_pair_ids:
                        preview = save_preview(
                            decoded,
                            side_dir,
                            frames=int(protocol["mesh"]["render_frames"]),
                            resolution=int(protocol["mesh"]["render_resolution"]),
                        )
                    row.update(
                        {
                            "passed": True,
                            "structure": structure,
                            "surface": surface,
                            "canonical_obj": str(obj_path),
                            "canonical_obj_sha256": sha256_file(obj_path),
                            "view_glb": str(glb_path),
                            "view_glb_sha256": sha256_file(glb_path),
                            "preview": preview,
                            "target": target_metadata,
                        }
                    )
                    del slat_payload, slat, decoded, canonical_mesh, view_mesh
                except Exception as error:  # Keep exploratory failures visible.
                    row["error"] = repr(error)
                write_json(private_pair_dir / f"{branch}.json", row)
                branch_records.append(row)
                torch.cuda.empty_cache()

            blind_row = {
                "pair_id": pair_id,
                "uid": uid,
                "object_uid": object_uid,
                "views": int(frozen["views"]),
                "seed": int(seed),
                "A": str(pair_dir / "A" / "mesh_view.glb"),
                "B": str(pair_dir / "B" / "mesh_view.glb"),
                "exploratory_non_formal": True,
            }
            key_row = {"A": side_mapping["A"], "B": side_mapping["B"]}
            pair_record = {
                "pair_id": pair_id,
                "uid": uid,
                "object_uid": object_uid,
                "seed": int(seed),
                "exploratory": True,
                "formal": False,
                "passed": all(row.get("passed") is True for row in branch_records),
                "branches": branch_records,
                "blind_manifest": blind_row,
                "unblinding": key_row,
            }
            write_json(pair_record_path, pair_record)
            validate_pair_record(
                pair_record_path,
                pair_id=pair_id,
                uid=uid,
                object_uid=object_uid,
                seed=int(seed),
            )
            records.extend(branch_records)
            blind_manifest.append(blind_row)
            unblinding[pair_id] = key_row
            print(f"[exploratory:mesh] {pair_id}", flush=True)
        del target_mesh
    mesh_decoder.cpu()
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    manifest_path = output_dir / "blind_manifest.csv"
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.stem}.tmp-{os.getpid()}{manifest_path.suffix}"
    )
    with temporary_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blind_manifest[0].keys()))
        writer.writeheader()
        writer.writerows(blind_manifest)
    os.replace(temporary_manifest, manifest_path)
    leaks = [
        str(path.relative_to(mesh_root))
        for path in mesh_root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".json", ".csv"}
            or "stock" in path.name.lower()
            or "correct" in path.name.lower()
        )
    ]
    blinding_audit = {
        "sanitized_bundle": str(mesh_root),
        "allowed_companion_manifest": str(manifest_path),
        "private_records": str(private_root),
        "leaks": leaks,
        "passed": not leaks,
    }
    if leaks:
        raise RuntimeError(f"exploratory blind bundle leaks private labels: {leaks}")
    unblinding_path = output_dir / "unblinding_key.json"
    write_json(unblinding_path, unblinding)
    unblinding_path.chmod(0o600)

    analysis_protocol = copy.deepcopy(protocol)
    analysis_protocol["sampling"]["joint_seeds"] = seeds
    decision = aggregate_report(
        records,
        protocol=analysis_protocol,
        expected_pairs=len(selected_rows) * len(seeds),
        formal=False,
    )
    all_records_passed = all(row.get("passed") is True for row in records)
    report = {
        "format": REPORT_FORMAT,
        "stage": "Exploratory continuation of Stage-B SLAT-to-Mesh export",
        "exploratory": True,
        "formal": False,
        "formal_gate_passed": False,
        "formal_gate_evaluated": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "notice": NON_FORMAL_NOTICE,
        "deterministic_repeat_slat_audit": {
            "status": "SKIPPED",
            "reason": "user requested direct Mesh inspection after recorded native-SLAT repeat failure",
            "source_failure": source_validation[
                "recorded_deterministic_repeat_slat_audit"
            ],
        },
        "source_validation": source_validation,
        "args": vars(args),
        "record_count": len(records),
        "all_records_passed": all_records_passed,
        "blinding_audit": blinding_audit,
        "exploratory_summary": decision,
        "records": records,
        "claim_limit": (
            "This run is useful for visual inspection and descriptive paired metrics only. "
            "It cannot pass the frozen formal Stage-B protocol, and it does not establish "
            "pose/depth-specific causality."
        ),
    }
    write_json(output_dir / "report.json", report)
    summary_lines = [
        "# Exploratory SLAT-to-Mesh continuation",
        "",
        f"> {NON_FORMAL_NOTICE}",
        "",
        f"- Source: `{source_dir}`",
        f"- Objects: `{len(selected_rows)}`",
        f"- Seeds: `{seeds}`",
        f"- Records passed: `{sum(row.get('passed') is True for row in records)} / {len(records)}`",
        f"- Valid paired comparisons: `{decision['valid_pair_count']}`",
        "- Deterministic-repeat SLAT audit: `SKIPPED`",
        "- Formal conclusion: `NOT AVAILABLE`",
        "",
        "## Descriptive paired summary",
        "",
        "```json",
        json.dumps(decision["summary"], indent=2),
        "```",
        "",
        "Use `blind_pairs/` with `blind_manifest.csv` for inspection; keep "
        "`unblinding_key.json` hidden until ratings are frozen.",
    ]
    write_text(output_dir / "report.md", "\n".join(summary_lines) + "\n")

    artifact_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "completion_manifest.json"
    )
    runtime_exit_code = 0 if all_records_passed else 3
    completion = {
        "format": COMPLETION_FORMAT,
        "complete": True,
        "exploratory": True,
        "formal": False,
        "protocol_sha256": protocol["protocol_sha256"],
        "science_passed": False,
        "formal_gate_passed": False,
        "all_records_passed": all_records_passed,
        "runtime_exit_code": runtime_exit_code,
        "file_count": len(artifact_files),
        "files": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
            }
            for path in artifact_files
        ],
    }
    write_json(completion_path, completion)
    validate_completion_manifest(output_dir, expected_formal=False)
    print(
        json.dumps(
            {
                "exploratory": True,
                "formal": False,
                "all_records_passed": all_records_passed,
                "valid_pair_count": decision["valid_pair_count"],
                "report": str(output_dir / "report.json"),
                "blind_manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )
    raise SystemExit(runtime_exit_code)


if __name__ == "__main__":
    main()
