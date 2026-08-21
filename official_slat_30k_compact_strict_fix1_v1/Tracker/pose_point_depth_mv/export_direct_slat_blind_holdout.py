#!/usr/bin/env python3
"""Export frozen anonymous A/C end-to-end Direct-SLAT Mesh pairs."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import sys
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
    os.environ["PATH"] = f"{NINJA_WRAPPER_ROOT}{os.pathsep}{os.environ.get('PATH', '')}"
for path in (TRACKER_ROOT, RECONVIAGEN_ROOT, VGGT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ar_ss_flow.local_pose_lifting_flow import PoseLiftingCacheDataset  # noqa: E402
from pose_point_depth_mv.direct_flow import (  # noqa: E402
    DIRECT_FLOW_VERSION,
    NativeStockFlow,
)
from pose_point_depth_mv.direct_slat_blind import (  # noqa: E402
    PREFLIGHT_REPORT_FORMAT,
    PROTOCOL_FORMAT,
    RATER_COLUMNS,
    SEALED_REPORT_FORMAT,
    atomic_json,
    canonical_sha256,
    pair_identity,
    repeat_floors,
    runtime_selection_rows,
    sha256_file,
    validate_binding_tree,
    validate_execution_compatibility_record,
)
from pose_point_depth_mv.direct_slat_data import DirectSLatCacheDataset  # noqa: E402
from pose_point_depth_mv.direct_slat_flow import (  # noqa: E402
    DIRECT_SLAT_FLOW_VERSION,
    NativeStockSLATFlow,
    PositiveSupportSLATRolloutFlow,
    build_direct_slat_components,
    canonical_json_sha256,
    load_strict_trainable_state,
)
from pose_point_depth_mv.eval_direct_flow import decode_coords  # noqa: E402
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (  # noqa: E402
    canonical_coords,
    load_canonical_gt,
    mesh_structure_metrics,
    save_preview,
    shared_noise_audit,
    sparse_from_payload,
    sparse_noise_from_master,
    sparse_payload,
    surface_metrics,
    temporary_path,
    tensor_sha256,
)
from pose_point_depth_mv.train_direct_flow import build_direct_components  # noqa: E402
from pose_point_depth_mv.train_direct_slat_flow import to_device_tree  # noqa: E402
from reconvggt_ar_adapter_a.pointpose_ss_condition import load_partial_state  # noqa: E402
from trellis.modules import sparse as sp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--blind_key_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip_renders",
        action="store_true",
        help="Only legal for nonformal preflight engineering runs.",
    )
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError(f"unexpected protocol format={protocol.get('format')!r}")
    body = dict(protocol)
    saved_hash = str(body.pop("protocol_sha256", ""))
    if canonical_sha256(body) != saved_hash:
        raise RuntimeError("protocol canonical SHA-256 mismatch")
    validate_binding_tree(protocol["bindings"], "bindings")
    validate_binding_tree(protocol["sample_bindings"], "sample_bindings")
    validate_binding_tree(protocol["runtime_bindings"], "runtime_bindings")
    validate_binding_tree(protocol["code_bindings"], "code_bindings")
    validate_execution_compatibility_record(protocol)
    return protocol


def configure_frozen_runtime(protocol: dict[str, Any]) -> None:
    runtime = dict(protocol["runtime"])
    if "attention_backend" not in runtime:
        return
    expected = {
        "attention_backend": os.environ.get("ATTN_BACKEND", "flash_attn"),
        "sparse_attention_backend": os.environ.get(
            "SPARSE_ATTN_BACKEND", os.environ.get("ATTN_BACKEND", "flash_attn")
        ),
        "spconv_algo": os.environ.get("SPCONV_ALGO", "native"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }
    for name, actual in expected.items():
        if str(runtime.get(name, "")) != str(actual):
            raise RuntimeError(
                f"runtime environment differs: {name}={actual!r}, "
                f"frozen={runtime.get(name)!r}"
            )
    deterministic = str(runtime.get("deterministic_algorithms", "off"))
    if deterministic == "off":
        torch.use_deterministic_algorithms(False)
    elif deterministic in {"warn", "on"}:
        torch.use_deterministic_algorithms(
            True, warn_only=deterministic == "warn"
        )
    else:
        raise RuntimeError(f"invalid frozen deterministic mode={deterministic!r}")
    allow_tf32 = bool(runtime.get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if runtime.get("torch_version") and str(runtime["torch_version"]) != str(
        torch.__version__
    ):
        raise RuntimeError("frozen torch version differs")
    if runtime.get("cuda_version") and str(runtime["cuda_version"]) != str(
        torch.version.cuda
    ):
        raise RuntimeError("frozen CUDA runtime version differs")


def side_latent_paths(
    pair_latent_dir: Path,
    side: str,
    *,
    repeat_count: int,
    legacy_names: bool,
) -> list[Path]:
    if repeat_count < 1:
        raise ValueError("repeat_count must be positive")
    paths = [pair_latent_dir / f"{side}.pt"]
    if legacy_names and repeat_count == 2:
        paths.append(pair_latent_dir / f"{side}_repeat.pt")
    else:
        paths.extend(
            pair_latent_dir / f"{side}_repeat_{index:03d}.pt"
            for index in range(1, repeat_count)
        )
    return paths


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def export_mesh_atomic(mesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    mesh.export(temporary)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def make_flow(
    branch: str,
    *,
    model,
    condition: dict[str, Any],
    support: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
):
    if branch == "stock":
        return NativeStockSLATFlow(model)
    if branch == "full":
        return PositiveSupportSLATRolloutFlow(
            model,
            condition["cond"],
            support,
            support_scale=1.0,
        )
    raise ValueError(f"unexpected branch={branch!r}")


def sample_slat(
    *,
    branch: str,
    model,
    sampler,
    condition: dict[str, Any],
    support: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    initial: sp.SparseTensor,
    params: dict[str, Any],
    normalization: dict[str, Any],
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> sp.SparseTensor:
    flow = make_flow(
        branch, model=model, condition=condition, support=support
    )
    noise = sp.SparseTensor(
        feats=initial.feats.clone(), coords=initial.coords.clone()
    )
    with torch.autocast(
        device_type="cuda", dtype=amp_dtype, enabled=use_amp
    ):
        sampled = sampler.sample(
            flow,
            noise,
            **condition,
            **params,
            verbose=False,
        ).samples
    if branch == "full" and flow.positive_calls <= 0:
        raise RuntimeError("full Direct-SLAT rollout never used corrected SS support")
    if not torch.equal(sampled.coords, initial.coords):
        raise RuntimeError(
            "SLAT sampler changed or reordered the frozen sparse coordinates"
        )
    mean = torch.tensor(normalization["mean"], device=sampled.feats.device)[None]
    std = torch.tensor(normalization["std"], device=sampled.feats.device)[None]
    return sampled * std + mean


def pair_public_row(
    *,
    pair_id: str,
    object_position: int,
    seed: int,
    rendered: bool,
) -> dict[str, Any]:
    pair_dir = Path("blind_pairs") / pair_id
    row = {
        "pair_id": pair_id,
        "object_id": f"object_{int(object_position):04d}",
        "seed": int(seed),
        "A_mesh": str(pair_dir / "A" / "mesh_view.glb"),
        "B_mesh": str(pair_dir / "B" / "mesh_view.glb"),
        "A_preview": "",
        "B_preview": "",
    }
    if rendered:
        row["A_preview"] = str(pair_dir / "A" / "normal_contact_sheet.png")
        row["B_preview"] = str(pair_dir / "B" / "normal_contact_sheet.png")
    return row


def repeat_diff(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    first_structure = first["structure"]
    second_structure = second["structure"]
    return {
        "metric_abs_diff": {
            "chamfer_l1_abs": abs(
                float(first["surface"]["chamfer_l1"])
                - float(second["surface"]["chamfer_l1"])
            ),
            "fscore_0p02_abs": abs(
                float(first["surface"]["fscore_0p02"])
                - float(second["surface"]["fscore_0p02"])
            ),
            "largest_component_ratio_abs": abs(
                float(first_structure["largest_component_ratio"])
                - float(second_structure["largest_component_ratio"])
            ),
            "boundary_edge_count_abs": abs(
                float(first_structure["boundary_edge_count"])
                - float(second_structure["boundary_edge_count"])
            ),
            "boundary_total_length_abs": abs(
                float(first_structure["boundary_total_length"])
                - float(second_structure["boundary_total_length"])
            ),
            "nonmanifold_edge_count_abs": abs(
                float(first_structure["nonmanifold_edge_count"])
                - float(second_structure["nonmanifold_edge_count"])
            ),
            "component_count_abs": abs(
                float(first_structure["component_count"])
                - float(second_structure["component_count"])
            ),
        },
        "topology_changed": {
            "mesh_success": bool(first_structure["mesh_success"])
            != bool(second_structure["mesh_success"]),
            "is_watertight": bool(first_structure["is_watertight"])
            != bool(second_structure["is_watertight"]),
            "zero_boundary": (
                int(first_structure["boundary_edge_count"]) == 0
            )
            != (int(second_structure["boundary_edge_count"]) == 0),
            "nonmanifold_free": (
                int(first_structure["nonmanifold_edge_count"]) == 0
            )
            != (int(second_structure["nonmanifold_edge_count"]) == 0),
        },
    }


def completion_valid(output_dir: Path, completion: dict[str, Any]) -> None:
    if completion.get("complete") is not True:
        raise RuntimeError("completion manifest is not complete")
    rows = list(completion.get("files", []))
    relative_paths = [str(row.get("path", "")) for row in rows]
    if (
        not relative_paths
        or "" in relative_paths
        or len(relative_paths) != len(set(relative_paths))
    ):
        raise RuntimeError("completion manifest has missing or duplicate files")
    actual_paths = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "completion_manifest.json"
    }
    if set(relative_paths) != actual_paths:
        raise RuntimeError("completion manifest does not exactly cover output files")
    for row in rows:
        path = output_dir / str(row["path"])
        if not path.is_file() or sha256_file(path) != str(row["sha256"]):
            raise RuntimeError(f"completed blind artifact changed: {path}")


def load_coord_pair(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        if set(payload.files) != {"A", "B"}:
            raise RuntimeError(f"unexpected coordinate payload keys: {path}")
        coords = {}
        for side in ("A", "B"):
            raw = np.asarray(payload[side])
            if (
                raw.ndim != 2
                or raw.shape[1] != 4
                or not np.issubdtype(raw.dtype, np.integer)
            ):
                raise RuntimeError(
                    f"coordinate payload is not canonical int [N,4]: {path}/{side}"
                )
            canonical = canonical_coords(raw, resolution=64)
            if not np.array_equal(raw.astype(np.int32), canonical):
                raise RuntimeError(
                    f"coordinate payload is unsorted or duplicated: {path}/{side}"
                )
            coords[side] = canonical
    return coords


def validate_coord_resume(
    coord_path: Path,
    audit_path: Path,
    *,
    pair_id: str,
    frozen: dict[str, Any],
    seed: int,
) -> None:
    coords = load_coord_pair(coord_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_counts = {side: int(len(coords[side])) for side in ("A", "B")}
    if (
        audit.get("pair_id") != pair_id
        or audit.get("object_uid") != str(frozen["object_uid"])
        or audit.get("uid") != str(frozen["uid"])
        or int(audit.get("seed", -1)) != int(seed)
        or audit.get("side_coord_counts") != expected_counts
        or audit.get("coord_npz_sha256") != sha256_file(coord_path)
        or audit.get("shared_ss_noise_by_frozen_seed_formula") is not True
        or audit.get("passed") is not True
    ):
        raise RuntimeError(f"resumed SS coordinate audit differs: {audit_path}")


def validate_sparse_payload_file(
    path: Path,
    *,
    expected_coords: np.ndarray,
) -> None:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"feats", "coords"}:
        raise RuntimeError(f"invalid sparse latent payload: {path}")
    feats = payload["feats"]
    coords = payload["coords"]
    if (
        not torch.is_tensor(feats)
        or not torch.is_tensor(coords)
        or feats.ndim != 2
        or coords.ndim != 2
        or coords.shape[1] != 4
        or coords.dtype not in (torch.int32, torch.int64)
        or len(feats) != len(coords)
        or not bool(torch.isfinite(feats).all())
        or not np.array_equal(
            coords.to(dtype=torch.int32).numpy(),
            np.asarray(expected_coords, dtype=np.int32),
        )
    ):
        raise RuntimeError(f"sparse latent payload failed validation: {path}")


def load_exported_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        pieces = [
            item
            for item in loaded.dump(concatenate=False)
            if isinstance(item, trimesh.Trimesh)
            and len(item.vertices)
            and len(item.faces)
        ]
        if not pieces:
            raise RuntimeError(f"roundtrip Mesh is empty: {path}")
        mesh = trimesh.util.concatenate(pieces)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError(f"unsupported roundtrip Mesh type={type(loaded)}: {path}")
    if not mesh_structure_metrics(mesh)["mesh_success"]:
        raise RuntimeError(f"roundtrip Mesh is empty or non-finite: {path}")
    return mesh


def validate_pair_record(
    pair_record: dict[str, Any],
    *,
    pair_id: str,
    frozen: dict[str, Any],
    seed: int,
    expected_public_row: dict[str, Any],
    preflight: bool,
    repeat_count: int,
) -> None:
    records = list(pair_record.get("records", []))
    sides = [str(row.get("side", "")) for row in records]
    if (
        pair_record.get("pair_id") != pair_id
        or len(records) != 2
        or set(sides) != {"A", "B"}
        or len(sides) != len(set(sides))
        or pair_record.get("public_row") != expected_public_row
    ):
        raise RuntimeError(f"resumed pair record identity differs: {pair_id}")
    for row in records:
        if (
            row.get("pair_id") != pair_id
            or row.get("object_uid") != str(frozen["object_uid"])
            or row.get("uid") != str(frozen["uid"])
            or int(row.get("object_position", -1))
            != int(frozen["object_position"])
            or int(row.get("seed", -1)) != int(seed)
        ):
            raise RuntimeError(f"resumed side record identity differs: {pair_id}")
        if row.get("passed") is True:
            side_dir = Path(expected_public_row[f"{row['side']}_mesh"]).parent
            canonical_path = side_dir / "mesh_canonical.obj"
            view_path = side_dir / "mesh_view.glb"
            if (
                sha256_file(canonical_path)
                != str(row.get("canonical_obj_sha256", ""))
                or sha256_file(view_path)
                != str(row.get("view_glb_sha256", ""))
            ):
                raise RuntimeError(f"resumed Mesh hash differs: {pair_id}")
            preview = row.get("preview")
            if expected_public_row[f"{row['side']}_preview"]:
                if not isinstance(preview, dict):
                    raise RuntimeError(f"resumed preview is missing: {pair_id}")
                for key, hash_key in (
                    ("turntable", "turntable_sha256"),
                    ("contact_sheet", "contact_sheet_sha256"),
                ):
                    if sha256_file(preview[key]) != str(preview.get(hash_key, "")):
                        raise RuntimeError(f"resumed preview hash differs: {pair_id}")
    repeat_rows = list(pair_record.get("repeat_rows", []))
    if not preflight and repeat_rows:
        raise RuntimeError(f"confirmatory pair contains repeat rows: {pair_id}")
    if preflight and all(row.get("passed") is True for row in records):
        repeat_sides = [str(row.get("side", "")) for row in repeat_rows]
        expected_per_side = repeat_count * (repeat_count - 1) // 2
        if (
            len(repeat_rows) != 2 * expected_per_side
            or set(repeat_sides) != {"A", "B"}
            or any(repeat_sides.count(side) != expected_per_side for side in ("A", "B"))
        ):
            raise RuntimeError(f"preflight pair has incomplete repeat rows: {pair_id}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = load_protocol(protocol_path)
    if protocol["formal"] and args.skip_renders:
        raise ValueError("confirmatory blind export cannot skip frozen renders")
    key_path = Path(args.blind_key_file).resolve()
    blind_key = bytes.fromhex(key_path.read_text(encoding="ascii").strip())
    commitment = hashlib.sha256(blind_key).hexdigest()
    if commitment != str(
        protocol["blinding"]["blind_key_sha256_commitment"]
    ):
        raise RuntimeError("blind key does not match protocol commitment")
    output_dir = Path(args.output_dir).resolve()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("end-to-end blind Mesh export requires CUDA")
    torch.cuda.set_device(0 if device.index is None else int(device.index))
    configure_frozen_runtime(protocol)

    run_identity = {
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "mode": protocol["mode"],
        "blind_key_sha256_commitment": commitment,
        "device_type": device.type,
        "skip_renders": bool(args.skip_renders),
    }
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "completion_manifest.json"
    if completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion_valid(output_dir, completion)
        if json.loads(
            (output_dir / "run_identity.json").read_text(encoding="utf-8")
        ) != run_identity:
            raise RuntimeError("completed run identity differs")
        print(json.dumps(completion, indent=2), flush=True)
        raise SystemExit(int(completion["runtime_exit_code"]))
    identity_path = output_dir / "run_identity.json"
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != run_identity:
            raise RuntimeError("resume run identity differs")
    else:
        atomic_json(identity_path, run_identity)
        shutil.copy2(protocol_path, output_dir / "protocol.json")

    bindings = protocol["bindings"]
    cache_manifest = str(bindings["holdout_cache_manifest"]["path"])
    lifting_manifest = str(bindings["source_lifting_manifest"]["path"])
    dataset = DirectSLatCacheDataset(cache_manifest, verify_hashes=False)
    lifting = PoseLiftingCacheDataset(lifting_manifest)
    selected = runtime_selection_rows(protocol)
    seeds = [int(value) for value in protocol["sampling"]["joint_seeds"]]
    expected_pairs = len(selected) * len(seeds)
    for row in selected:
        source_index = int(row["source_lifting_index"])
        if str(lifting.rows[source_index]["uid"]) != str(row["uid"]):
            raise RuntimeError("source lifting order differs from frozen protocol")
        for seed in seeds:
            cache_index = int(row["cache_indices"][str(seed)])
            cache_row = dataset.rows[cache_index]
            if (
                str(cache_row["uid"]) != str(row["uid"])
                or int(cache_row["support_seed"]) != seed
            ):
                raise RuntimeError("Direct-SLAT cache order differs from protocol")

    use_amp = str(protocol["runtime"]["amp_dtype"]) != "none"
    amp_dtype = (
        torch.float16
        if str(protocol["runtime"]["amp_dtype"]) == "fp16"
        else torch.bfloat16
    )
    legacy_repeat_names = "same_process_repeat_count" not in protocol["runtime"]
    repeat_count = (
        int(protocol["runtime"].get("same_process_repeat_count", 2))
        if protocol["mode"] == "preflight"
        else 1
    )
    sealed_root = output_dir / "sealed"
    coord_root = sealed_root / "coords"
    latent_root = sealed_root / "latents"
    pair_record_root = sealed_root / "pair_records"
    public_root = output_dir / "blind_pairs"
    public_root.mkdir(exist_ok=True)

    ss_checkpoint = torch.load(
        bindings["ss_flow_checkpoint"]["path"], map_location="cpu"
    )
    if (
        ss_checkpoint.get("format") != DIRECT_FLOW_VERSION
        or int(ss_checkpoint.get("step", -1)) != 900
    ):
        raise RuntimeError("blind exporter requires frozen Direct-SS step 900")
    ss_args = dict(ss_checkpoint["args"])
    built_ss = build_direct_components(
        pretrained=protocol["pretrained"],
        visual_channels=lifting.visual_feature_dim,
        physical_hidden_dim=int(ss_args["physical_hidden_dim"]),
        lora_rank=int(ss_args["lora_rank"]),
        lora_alpha=int(ss_args["lora_alpha"]),
        gradient_checkpointing=False,
        need_decoder=True,
        device=device,
        retain_pipeline=True,
    )
    ss_sampler, ss_model, ss_decoder, _, ss_defaults, ss_pipeline = built_ss
    load_partial_state(
        ss_model,
        ss_checkpoint["model_trainable_state"],
        require_all_trainable=True,
    )
    ss_model.eval()
    ss_params = dict(ss_defaults)
    ss_params.update(
        {
            key: value
            for key, value in protocol["sampling"]["ss"].items()
            if key in {"steps", "cfg_strength", "guidance_rescale", "rescale_t"}
        }
    )
    for frozen in selected:
        object_position = int(frozen["object_position"])
        source_index = int(frozen["source_lifting_index"])
        lifting_sample = lifting[source_index]
        condition = lifting_sample["stock_condition"].to(device=device)
        negative = torch.zeros_like(condition)
        for seed in seeds:
            pair_id, mapping = pair_identity(
                protocol["protocol_name"], str(frozen["uid"]), seed, blind_key
            )
            coord_path = coord_root / f"{pair_id}.npz"
            audit_path = coord_root / f"{pair_id}.json"
            if coord_path.is_file() and audit_path.is_file():
                validate_coord_resume(
                    coord_path,
                    audit_path,
                    pair_id=pair_id,
                    frozen=frozen,
                    seed=seed,
                )
                continue
            if coord_path.exists() or audit_path.exists():
                raise RuntimeError(
                    f"incomplete resumed SS coordinate pair: {pair_id}"
                )
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 1000003 + source_index * 1009
            )
            ss_noise = torch.randn(
                (1, 8, 16, 16, 16),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_amp
            ):
                stock_ss = ss_sampler.sample(
                    NativeStockFlow(ss_model),
                    ss_noise.clone(),
                    cond=condition,
                    neg_cond=negative,
                    **ss_params,
                    verbose=False,
                ).samples
            stock_coords = canonical_coords(
                decode_coords(ss_decoder, stock_ss), resolution=64
            )
            cache_index = int(frozen["cache_indices"][str(seed)])
            full_sample = dataset[cache_index]
            full_coords = canonical_coords(
                full_sample["corrected_coords64"].numpy(), resolution=64
            )
            branch_coords = {"stock": stock_coords, "full": full_coords}
            atomic_npz(
                coord_path,
                A=branch_coords[mapping["A"]],
                B=branch_coords[mapping["B"]],
            )
            atomic_json(
                audit_path,
                {
                    "pair_id": pair_id,
                    "object_uid": str(frozen["object_uid"]),
                    "uid": str(frozen["uid"]),
                    "seed": int(seed),
                    "source_lifting_index": source_index,
                    "ss_noise_seed": int(seed) * 1000003 + source_index * 1009,
                    "ss_noise_sha256": tensor_sha256(ss_noise),
                    "cached_corrected_support_seed": int(
                        full_sample["support_seed"]
                    ),
                    "side_coord_counts": {
                        side: int(len(branch_coords[mapping[side]]))
                        for side in ("A", "B")
                    },
                    "coord_npz_sha256": sha256_file(coord_path),
                    "shared_ss_noise_by_frozen_seed_formula": True,
                    "passed": True,
                },
            )
            del ss_noise, stock_ss, full_sample
            torch.cuda.empty_cache()
        print(f"[direct_slat_blind:ss] {frozen['uid']}", flush=True)
    ss_model.cpu()
    ss_decoder.cpu()
    for name in ("sparse_structure_flow_model", "sparse_structure_decoder"):
        ss_pipeline.models[name].cpu()
    del ss_model, ss_decoder, ss_sampler, ss_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    slat_checkpoint = torch.load(
        bindings["direct_slat_checkpoint"]["path"], map_location="cpu"
    )
    if (
        slat_checkpoint.get("format") != DIRECT_SLAT_FLOW_VERSION
        or int(slat_checkpoint.get("step", -1)) != 800
    ):
        raise RuntimeError("blind exporter requires frozen Direct-SLAT step 800")
    slat_args = dict(slat_checkpoint["args"])
    built_slat = build_direct_slat_components(
        pretrained=protocol["pretrained"],
        adapter_hidden_dim=int(slat_args["adapter_hidden_dim"]),
        lora_rank=int(slat_args["lora_rank"]),
        lora_alpha=int(slat_args["lora_alpha"]),
        gradient_checkpointing=False,
        device=device,
        retain_pipeline=True,
    )
    slat_sampler, slat_model, slat_defaults, normalization, _, pipeline = built_slat
    runtime_normalization = {
        key: [float(item) for item in value]
        for key, value in normalization.items()
    }
    if canonical_json_sha256(runtime_normalization) != dataset.slat_normalization_hash:
        raise RuntimeError("runtime SLAT normalization differs from holdout cache")
    load_strict_trainable_state(
        slat_model, slat_checkpoint["model_trainable_state"]
    )
    slat_model.eval()
    slat_params = dict(slat_defaults)
    slat_params.update(
        {
            key: value
            for key, value in protocol["sampling"]["slat"].items()
            if key in {"steps", "cfg_strength", "cfg_interval", "rescale_t"}
        }
    )
    preflight = protocol["mode"] == "preflight"
    for frozen in selected:
        object_position = int(frozen["object_position"])
        for seed in seeds:
            pair_id, mapping = pair_identity(
                protocol["protocol_name"], str(frozen["uid"]), seed, blind_key
            )
            pair_latent_dir = latent_root / pair_id
            side_paths = {
                side: side_latent_paths(
                    pair_latent_dir,
                    side,
                    repeat_count=repeat_count,
                    legacy_names=legacy_repeat_names,
                )
                for side in ("A", "B")
            }
            required = [path for side in ("A", "B") for path in side_paths[side]]
            noise_audit_path = pair_latent_dir / "noise_audit.json"
            coords = load_coord_pair(coord_root / f"{pair_id}.npz")
            if all(path.is_file() for path in required) and noise_audit_path.is_file():
                noise_audit = json.loads(
                    noise_audit_path.read_text(encoding="utf-8")
                )
                expected_hashes = {
                    path.name: sha256_file(path) for path in required
                }
                if (
                    noise_audit.get("pair_id") != pair_id
                    or int(noise_audit.get("seed", -1)) != int(seed)
                    or noise_audit.get("latent_sha256") != expected_hashes
                    or noise_audit.get("common_coord_noise_bit_exact") is not True
                    or noise_audit.get("passed") is not True
                ):
                    raise RuntimeError(
                        f"resumed SLAT noise/latent audit differs: {pair_id}"
                    )
                for path in required:
                    side = path.name.split("_", 1)[0].split(".", 1)[0]
                    validate_sparse_payload_file(
                        path, expected_coords=coords[side]
                    )
                continue
            if any(path.exists() for path in required) or noise_audit_path.exists():
                raise RuntimeError(
                    f"incomplete resumed SLAT latent pair: {pair_id}"
                )
            cache_index = int(frozen["cache_indices"][str(seed)])
            sample = dataset[cache_index]
            condition = to_device_tree(sample["condition"], device)
            support = (
                sample["corrected_ss"].to(device=device),
                sample["occupancy_logits64"].to(device=device),
                sample["physical_tokens16"].to(device=device),
            )
            master_seed = int(seed) * 2000003 + object_position * 2017 + 7919
            generator = torch.Generator(device=device).manual_seed(master_seed)
            master = torch.randn(
                (64, 64, 64, 8),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            initial = {
                side: sparse_noise_from_master(
                    coords[side], master, device=device
                )
                for side in ("A", "B")
            }
            noise_audit = shared_noise_audit(
                initial["A"].coords,
                initial["A"].feats,
                initial["B"].coords,
                initial["B"].feats,
            )
            if not noise_audit["common_coord_noise_bit_exact"]:
                raise RuntimeError("coordinate-keyed common SLAT noise is not exact")
            for side in ("A", "B"):
                for latent_path in side_paths[side]:
                    sampled = sample_slat(
                        branch=mapping[side],
                        model=slat_model,
                        sampler=slat_sampler,
                        condition=condition,
                        support=support,
                        initial=initial[side],
                        params=slat_params,
                        normalization=runtime_normalization,
                        amp_dtype=amp_dtype,
                        use_amp=use_amp,
                    )
                    atomic_torch_save(sparse_payload(sampled), latent_path)
                    del sampled
                torch.cuda.empty_cache()
            atomic_json(
                noise_audit_path,
                {
                    "pair_id": pair_id,
                    "seed": int(seed),
                    "master_seed": master_seed,
                    "master_noise_sha256": tensor_sha256(master),
                    "side_initial_noise_sha256": {
                        side: tensor_sha256(initial[side].feats)
                        for side in ("A", "B")
                    },
                    "latent_sha256": {
                        path.name: sha256_file(path) for path in required
                    },
                    **noise_audit,
                    "passed": True,
                },
            )
            del sample, condition, support, master, initial
            torch.cuda.empty_cache()
            print(f"[direct_slat_blind:slat] {pair_id}", flush=True)

    slat_model.cpu()
    del slat_model, slat_sampler
    gc.collect()
    torch.cuda.empty_cache()
    mesh_decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    records: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    for frozen in selected:
        object_position = int(frozen["object_position"])
        first_cache_index = int(frozen["cache_indices"][str(seeds[0])])
        target_sample = dataset[first_cache_index]
        target_mesh, target_metadata = load_canonical_gt(target_sample)
        for seed in seeds:
            pair_id, mapping = pair_identity(
                protocol["protocol_name"], str(frozen["uid"]), seed, blind_key
            )
            pair_latent_dir = latent_root / pair_id
            side_paths = {
                side: side_latent_paths(
                    pair_latent_dir,
                    side,
                    repeat_count=repeat_count,
                    legacy_names=legacy_repeat_names,
                )
                for side in ("A", "B")
            }
            pair_record_path = pair_record_root / f"{pair_id}.json"
            pair_dir = public_root / pair_id
            expected_public_row = pair_public_row(
                pair_id=pair_id,
                object_position=object_position,
                seed=seed,
                rendered=not args.skip_renders,
            )
            if pair_record_path.is_file():
                pair_record = json.loads(
                    pair_record_path.read_text(encoding="utf-8")
                )
                validate_pair_record(
                    pair_record,
                    pair_id=pair_id,
                    frozen=frozen,
                    seed=seed,
                    expected_public_row=expected_public_row,
                    preflight=preflight,
                    repeat_count=repeat_count,
                )
                records.extend(pair_record["records"])
                repeat_rows.extend(pair_record.get("repeat_rows", []))
                public_rows.append(pair_record["public_row"])
                continue
            side_records = []
            side_repeat_rows = []
            for side in ("A", "B"):
                side_dir = pair_dir / side
                row: dict[str, Any] = {
                    "pair_id": pair_id,
                    "side": side,
                    "object_uid": str(frozen["object_uid"]),
                    "uid": str(frozen["uid"]),
                    "object_position": object_position,
                    "seed": int(seed),
                    "passed": False,
                }
                try:
                    payload = torch.load(
                        latent_root / pair_id / f"{side}.pt",
                        map_location="cpu",
                    )
                    decoded = mesh_decoder(
                        sparse_from_payload(payload, device)
                    )[0]
                    canonical_mesh = decoded.to_trimesh(transform_pose=False)
                    view_mesh = decoded.to_trimesh(transform_pose=True)
                    structure = mesh_structure_metrics(canonical_mesh)
                    if not (
                        structure["mesh_success"]
                        and structure["vertices_finite"]
                        and structure["is_winding_consistent"]
                    ):
                        raise RuntimeError(
                            "decoded Mesh failed success/finite/winding hard gate"
                        )
                    canonical_path = side_dir / "mesh_canonical.obj"
                    view_path = side_dir / "mesh_view.glb"
                    export_mesh_atomic(canonical_mesh, canonical_path)
                    export_mesh_atomic(view_mesh, view_path)
                    canonical_roundtrip = load_exported_mesh(canonical_path)
                    view_roundtrip = load_exported_mesh(view_path)
                    metric_seed = int(seed) * 1009 + object_position * 9173
                    surface = surface_metrics(
                        canonical_mesh,
                        target_mesh,
                        count=int(protocol["mesh"]["surface_samples"]),
                        seed=metric_seed,
                        thresholds=protocol["mesh"]["fscore_thresholds"],
                    )
                    preview = None
                    if not args.skip_renders:
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
                            "canonical_obj_sha256": sha256_file(canonical_path),
                            "view_glb_sha256": sha256_file(view_path),
                            "roundtrip": {
                                "canonical_mesh_success": True,
                                "view_mesh_success": True,
                            },
                            "preview": preview,
                            "target": target_metadata,
                        }
                    )
                    if preflight:
                        repeat_results = [
                            {"structure": structure, "surface": surface}
                        ]
                        for repeat_path in side_paths[side][1:]:
                            repeat_payload = torch.load(
                                repeat_path, map_location="cpu"
                            )
                            repeat_decoded = mesh_decoder(
                                sparse_from_payload(repeat_payload, device)
                            )[0]
                            repeat_mesh = repeat_decoded.to_trimesh(
                                transform_pose=False
                            )
                            repeat_structure = mesh_structure_metrics(repeat_mesh)
                            if not (
                                repeat_structure["mesh_success"]
                                and repeat_structure["vertices_finite"]
                                and repeat_structure["is_winding_consistent"]
                            ):
                                raise RuntimeError(
                                    "repeat Mesh failed success/finite/winding hard gate"
                                )
                            repeat_surface = surface_metrics(
                                repeat_mesh,
                                target_mesh,
                                count=int(protocol["mesh"]["surface_samples"]),
                                seed=metric_seed,
                                thresholds=protocol["mesh"]["fscore_thresholds"],
                            )
                            repeat_results.append(
                                {
                                    "structure": repeat_structure,
                                    "surface": repeat_surface,
                                }
                            )
                            del repeat_payload, repeat_decoded, repeat_mesh
                        for (left_index, left), (right_index, right) in combinations(
                            enumerate(repeat_results), 2
                        ):
                            side_repeat_rows.append(
                                {
                                    "pair_id": pair_id,
                                    "side": side,
                                    "branch": mapping[side],
                                    "object_uid": str(frozen["object_uid"]),
                                    "seed": int(seed),
                                    "left_run_index": int(left_index),
                                    "right_run_index": int(right_index),
                                    **repeat_diff(left, right),
                                }
                            )
                    del (
                        payload,
                        decoded,
                        canonical_mesh,
                        view_mesh,
                        canonical_roundtrip,
                        view_roundtrip,
                    )
                except Exception as error:
                    row["passed"] = False
                    row["error"] = repr(error)
                side_records.append(row)
                torch.cuda.empty_cache()
            public_row = expected_public_row
            pair_record = {
                "pair_id": pair_id,
                "records": side_records,
                "repeat_rows": side_repeat_rows,
                "public_row": public_row,
            }
            atomic_json(pair_record_path, pair_record)
            records.extend(side_records)
            repeat_rows.extend(side_repeat_rows)
            public_rows.append(public_row)
            print(f"[direct_slat_blind:mesh] {pair_id}", flush=True)
        del target_sample, target_mesh
    mesh_decoder.cpu()
    del mesh_decoder, pipeline
    gc.collect()
    torch.cuda.empty_cache()

    blind_manifest_path = output_dir / "blind_manifest.csv"
    write_csv(
        blind_manifest_path,
        public_rows,
        [
            "pair_id",
            "object_id",
            "seed",
            "A_mesh",
            "B_mesh",
            "A_preview",
            "B_preview",
        ],
    )
    template_root = output_dir / "score_templates"
    for index in range(1, 4):
        rater_id = f"R{index}"
        rows = [
            {
                "rater_id": rater_id,
                "pair_id": row["pair_id"],
                "main_structure_A": "",
                "main_structure_B": "",
                "missing_parts_A": "",
                "missing_parts_B": "",
                "floating_fragments_A": "",
                "floating_fragments_B": "",
                "thin_spikes_A": "",
                "thin_spikes_B": "",
                "holes_open_boundaries_A": "",
                "holes_open_boundaries_B": "",
                "overall_score_A": "",
                "overall_score_B": "",
                "overall_preference": "",
                "notes": "",
            }
            for row in public_rows
        ]
        write_csv(
            template_root / f"scores_{rater_id}.csv",
            rows,
            list(RATER_COLUMNS),
        )
    (template_root / "rating_instructions.md").write_text(
        "\n".join(
            (
                "# Anonymous Direct-SLAT Mesh rating",
                "",
                "Rate every A/B side independently before choosing a preference.",
                "",
                "- `main_structure` and `overall_score`: integer 1-5; higher is better.",
                "- Defect fields: integer 0-3; 0=none, 1=minor, 2=clear, 3=severe.",
                "- `overall_preference`: exactly `A`, `B`, or `tie`.",
                "- Do not consult sealed outputs, protocol files, keys, or other raters.",
                "- Complete every row; do not remove pairs or change pair IDs.",
                "",
            )
        ),
        encoding="utf-8",
    )

    leak_files = [
        str(path.relative_to(public_root))
        for path in public_root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".json", ".csv", ".pt", ".npz"}
            or any(
                word in path.name.lower()
                for word in ("stock", "full", "correct")
            )
        )
    ]
    if leak_files:
        raise RuntimeError(f"public blind bundle leaks private data: {leak_files}")
    all_records_passed = (
        len(records) == 2 * expected_pairs
        and all(row.get("passed") is True for row in records)
    )
    sealed_report = {
        "format": SEALED_REPORT_FORMAT,
        "complete": True,
        "mode": protocol["mode"],
        "formal": bool(protocol["formal"]),
        "protocol_sha256": protocol["protocol_sha256"],
        "expected_pair_count": expected_pairs,
        "record_count": len(records),
        "all_records_passed": all_records_passed,
        "records": records,
        "blinding_audit": {
            "public_root": str(public_root),
            "leak_files": leak_files,
            "passed": not leak_files,
            "mapping_emitted": False,
        },
        "decision": (
            "withheld until frozen ratings and one-time finalization"
            if protocol["formal"]
            else "nonformal preflight; no science decision"
        ),
    }
    sealed_report_path = sealed_root / "sealed_metrics.json"
    atomic_json(sealed_report_path, sealed_report)

    preflight_passed = True
    if preflight:
        repeat_policy = dict(
            protocol.get("statistics", {}).get(
                "repeat_policy", {"mode": "zero_jitter"}
            )
        )
        calibration = (
            repeat_floors(repeat_rows, policy=repeat_policy)
            if repeat_rows
            else {
                "passed": False,
                "policy_mode": str(repeat_policy.get("mode", "zero_jitter")),
                "record_count": 0,
                "median_abs": {},
                "p95_abs": {},
                "max_abs": {
                    "chamfer_l1_abs": 0.0,
                    "fscore_0p02_abs": 0.0,
                    "largest_component_ratio_abs": 0.0,
                    "boundary_edge_count_abs": 0.0,
                    "boundary_total_length_abs": 0.0,
                    "nonmanifold_edge_count_abs": 0.0,
                    "component_count_abs": 0.0,
                },
                "topology_change_counts": {},
                "topology_change_rates": {},
                "checks": {},
                "interpretation": "repeat calibration could not be computed",
            }
        )
        expected_repeat_rows = (
            2 * expected_pairs * repeat_count * (repeat_count - 1) // 2
        )
        calibration["passed"] = bool(
            calibration["passed"]
            and all_records_passed
            and len(repeat_rows) == expected_repeat_rows
        )
        preflight_report = {
            "format": PREFLIGHT_REPORT_FORMAT,
            "complete": True,
            "formal": False,
            "protocol_sha256": protocol["protocol_sha256"],
            "execution_compatibility_sha256": protocol[
                "execution_compatibility"
            ]["sha256"],
            "expected_pair_count": expected_pairs,
            "all_records_passed": all_records_passed,
            "repeat_calibration": calibration,
            "repeat_rows": repeat_rows,
            "scope_guard": (
                "same-model repeat and blinding engineering calibration only; "
                "objects are already seen and no model advantage is evaluated"
            ),
        }
        preflight_report["report_sha256"] = canonical_sha256(preflight_report)
        atomic_json(output_dir / "repeat_report.json", preflight_report)
        preflight_passed = bool(calibration["passed"])

    artifact_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "completion_manifest.json"
    )
    runtime_exit_code = 0 if all_records_passed and preflight_passed else 3
    completion = {
        "complete": True,
        "mode": protocol["mode"],
        "formal": bool(protocol["formal"]),
        "protocol_sha256": protocol["protocol_sha256"],
        "expected_pair_count": expected_pairs,
        "all_records_passed": all_records_passed,
        "preflight_repeat_passed": preflight_passed if preflight else None,
        "science_decision_emitted": False,
        "runtime_exit_code": runtime_exit_code,
        "sealed_report": str(sealed_report_path.relative_to(output_dir)),
        "sealed_report_sha256": sha256_file(sealed_report_path),
        "files": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
            }
            for path in artifact_files
        ],
    }
    atomic_json(completion_path, completion)
    completion_valid(output_dir, completion)
    print(
        json.dumps(
            {
                "mode": protocol["mode"],
                "pairs": expected_pairs,
                "all_records_passed": all_records_passed,
                "preflight_repeat_passed": (
                    preflight_passed if preflight else None
                ),
                "science_decision_emitted": False,
                "public_bundle": [
                    str(public_root),
                    str(blind_manifest_path),
                    str(template_root),
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    raise SystemExit(runtime_exit_code)


if __name__ == "__main__":
    main()
