#!/usr/bin/env python3
"""Run the frozen Stage-B stock/correct same-noise Mesh comparison."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
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
    PositivePhysicalRolloutFlow,
    load_frozen_correspondence_head,
    make_direct_evidence_bundle,
    validate_n3_checkpoint,
)
from pose_point_depth_mv.eval_direct_flow import (  # noqa: E402
    bootstrap_mean_ci,
    component_metrics,
    decode_coords,
    overlap_metrics,
    positive_rate,
    summarize,
)
from pose_point_depth_mv.prepare_direct_flow_mesh_protocol import (  # noqa: E402
    PROTOCOL_FORMAT,
    canonical_sha256,
    sha256_file,
)
from pose_point_depth_mv.train_direct_flow import build_direct_components  # noqa: E402
from reconvggt_ar_adapter_a.inspect_and_sanity import normalize_image_cond  # noqa: E402
from reconvggt_ar_adapter_a.pointpose_ss_condition import load_partial_state  # noqa: E402
from reconvggt_ar_adapter_a.train_pointpose_ss_lora import rgba_images  # noqa: E402
from trellis.modules import sparse as sp  # noqa: E402
from trellis.utils import render_utils  # noqa: E402


REPORT_FORMAT = "pose_point_depth_mv.direct_flow_mesh_report.v1"


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def tensor_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(tensor_sha256(item).encode("ascii"))
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def to_cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_tree(item) for item in value)
    return value


def to_device_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: to_device_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device_tree(item, device) for item in value)
    return value


def temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def torch_save_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    torch.save(value, temporary)
    os.replace(temporary, path)


def savez_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def validate_bound_file(binding: dict[str, Any], label: str) -> None:
    bound_path = Path(binding["path"])
    if not bound_path.is_file() or sha256_file(bound_path) != str(binding["sha256"]):
        raise RuntimeError(f"protocol file binding changed: {label}={bound_path}")
    if "mode" in binding and oct(bound_path.stat().st_mode & 0o777) != str(
        binding["mode"]
    ):
        raise RuntimeError(f"protocol file mode changed: {label}={bound_path}")


def validate_binding_tree(value: Any, label: str) -> None:
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            validate_bound_file(value, label)
            return
        for key, child in value.items():
            validate_binding_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_binding_tree(child, f"{label}[{index}]")


def pair_identity(
    protocol_name: str,
    uid: str,
    seed: int,
    blind_key: bytes,
) -> tuple[str, dict[str, str]]:
    pair_hash = hashlib.sha256(
        f"{protocol_name}|{uid}|{int(seed)}".encode("utf-8")
    ).hexdigest()
    side_hash = hmac.new(
        blind_key,
        f"{protocol_name}|{uid}|{int(seed)}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    pair_id = f"pair_{pair_hash[:16]}"
    if int(side_hash[:2], 16) % 2 == 0:
        return pair_id, {"A": "stock", "B": "correct"}
    return pair_id, {"A": "correct", "B": "stock"}


def validate_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError(f"unexpected protocol format={protocol.get('format')!r}")
    saved_hash = str(protocol.get("protocol_sha256", ""))
    body = dict(protocol)
    body.pop("protocol_sha256", None)
    actual_hash = canonical_sha256(body)
    if actual_hash != saved_hash:
        raise RuntimeError("protocol canonical SHA-256 mismatch")
    validate_binding_tree(protocol["bindings"], "bindings")
    validate_binding_tree(protocol["sample_bindings"], "sample_bindings")
    validate_binding_tree(protocol["runtime_bindings"], "runtime_bindings")
    validate_binding_tree(protocol["code_bindings"], "code_bindings")
    threshold = float(protocol["mesh"].get("primary_fscore_threshold", math.nan))
    if not math.isclose(threshold, 0.02, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("protocol primary F-score threshold must be 0.02")
    if not any(
        math.isclose(float(value), threshold, rel_tol=0.0, abs_tol=1.0e-12)
        for value in protocol["mesh"]["fscore_thresholds"]
    ):
        raise RuntimeError("primary F-score threshold is absent from metric thresholds")
    return protocol


def require_identity(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{label} identity mismatch: {mismatches}")


def validate_sparse_payload(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected_sha256):
        raise RuntimeError(f"{label} payload hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"feats", "coords"}:
        raise RuntimeError(f"{label} has an invalid SparseTensor payload")
    feats, coords = payload["feats"], payload["coords"]
    if not torch.is_tensor(feats) or not torch.is_tensor(coords):
        raise RuntimeError(f"{label} SparseTensor payload is not tensor-valued")
    if coords.ndim != 2 or coords.shape[1] != 4 or feats.shape[0] != coords.shape[0]:
        raise RuntimeError(f"{label} SparseTensor payload shapes are inconsistent")
    if not feats.numel() or not torch.isfinite(feats).all():
        raise RuntimeError(f"{label} SparseTensor features are empty or non-finite")


def validate_ss_artifact(
    coord_path: Path,
    audit_path: Path,
    *,
    pair_id: str,
    uid: str,
    object_uid: str,
    seed: int,
    rollout_position: int,
) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require_identity(
        audit,
        {
            "pair_id": pair_id,
            "uid": uid,
            "object_uid": object_uid,
            "seed": int(seed),
            "rollout_position": int(rollout_position),
        },
        "SS replay audit",
    )
    if audit.get("passed") is not True:
        raise RuntimeError(f"SS replay audit is not passed: {audit_path}")
    if sha256_file(coord_path) != str(audit.get("coords_npz_sha256", "")):
        raise RuntimeError(f"SS coordinate payload hash mismatch: {coord_path}")
    with np.load(coord_path, allow_pickle=False) as payload:
        if set(payload.files) != {"stock", "correct"}:
            raise RuntimeError(f"SS coordinate branches are invalid: {coord_path}")
        for branch in ("stock", "correct"):
            coords = np.asarray(payload[branch])
            if coords.ndim != 2 or coords.shape[1] not in (3, 4) or not len(coords):
                raise RuntimeError(f"SS coordinates are empty/invalid: {coord_path}:{branch}")
    if set(audit.get("branches", {})) != {"stock", "correct"}:
        raise RuntimeError(f"SS audit branch set is invalid: {audit_path}")


def validate_condition_artifact(
    condition_path: Path,
    audit_path: Path,
    *,
    uid: str,
) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require_identity(audit, {"uid": uid}, "SLAT condition audit")
    if audit.get("passed") is not True:
        raise RuntimeError(f"SLAT condition audit is not passed: {audit_path}")
    if sha256_file(condition_path) != str(audit.get("condition_pt_sha256", "")):
        raise RuntimeError(f"SLAT condition payload hash mismatch: {condition_path}")
    payload = torch.load(condition_path, map_location="cpu")
    if tensor_tree_sha256(payload) != str(audit.get("slat_condition_sha256", "")):
        raise RuntimeError(f"SLAT condition tensor hash mismatch: {condition_path}")


def validate_slat_artifact(
    audit_path: Path,
    branch_paths: dict[str, Path],
    *,
    pair_id: str,
    uid: str,
    seed: int,
) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require_identity(
        audit,
        {"pair_id": pair_id, "uid": uid, "seed": int(seed)},
        "SLAT noise audit",
    )
    if audit.get("passed") is not True or not audit.get(
        "common_coord_noise_bit_exact", False
    ):
        raise RuntimeError(f"SLAT shared-noise audit is not passed: {audit_path}")
    hashes = audit.get("branch_payload_sha256", {})
    if set(hashes) != {"stock", "correct"}:
        raise RuntimeError(f"SLAT payload hashes are incomplete: {audit_path}")
    for branch, path in branch_paths.items():
        validate_sparse_payload(path, str(hashes[branch]), f"{pair_id}:{branch}")


def validate_determinism_slat_artifact(audit_path: Path, payload_path: Path) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise RuntimeError(f"SLAT determinism audit is not passed: {audit_path}")
    validate_sparse_payload(
        payload_path,
        str(audit.get("repeat_payload_sha256", "")),
        "deterministic repeated SLAT",
    )


def validate_determinism_mesh_artifact(audit_path: Path) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise RuntimeError(f"Mesh determinism audit is not passed: {audit_path}")


def validate_pair_record(
    pair_record_path: Path,
    *,
    pair_id: str,
    uid: str,
    object_uid: str,
    seed: int,
) -> dict[str, Any]:
    pair_record = json.loads(pair_record_path.read_text(encoding="utf-8"))
    require_identity(
        pair_record,
        {
            "pair_id": pair_id,
            "uid": uid,
            "object_uid": object_uid,
            "seed": int(seed),
        },
        "Mesh pair record",
    )
    branches = pair_record.get("branches", [])
    if len(branches) != 2 or {row.get("branch") for row in branches} != {
        "stock",
        "correct",
    }:
        raise RuntimeError(f"Mesh pair branch set is invalid: {pair_record_path}")
    if {row.get("side") for row in branches} != {"A", "B"}:
        raise RuntimeError(f"Mesh pair side set is invalid: {pair_record_path}")
    mapping = pair_record.get("unblinding", {})
    if set(mapping) != {"A", "B"} or set(mapping.values()) != {"stock", "correct"}:
        raise RuntimeError(f"Mesh pair unblinding map is invalid: {pair_record_path}")
    for row in branches:
        require_identity(
            row,
            {
                "pair_id": pair_id,
                "uid": uid,
                "object_uid": object_uid,
                "seed": int(seed),
            },
            "Mesh branch record",
        )
        private_row_path = pair_record_path.parent / f"{row['branch']}.json"
        if not private_row_path.is_file() or json.loads(
            private_row_path.read_text(encoding="utf-8")
        ) != row:
            raise RuntimeError(f"private Mesh branch record differs: {private_row_path}")
        if row.get("passed") is True:
            for path_key, hash_key in (
                ("canonical_obj", "canonical_obj_sha256"),
                ("view_glb", "view_glb_sha256"),
            ):
                artifact = Path(row[path_key])
                if not artifact.is_file() or sha256_file(artifact) != row[hash_key]:
                    raise RuntimeError(f"Mesh artifact hash mismatch: {artifact}")
            preview = row.get("preview")
            if preview is not None:
                for path_key, hash_key in (
                    ("turntable", "turntable_sha256"),
                    ("contact_sheet", "contact_sheet_sha256"),
                ):
                    artifact = Path(preview[path_key])
                    if not artifact.is_file() or sha256_file(artifact) != preview[hash_key]:
                        raise RuntimeError(f"Mesh preview hash mismatch: {artifact}")
        elif not row.get("error"):
            raise RuntimeError(f"failed Mesh branch lacks a frozen error: {pair_record_path}")
    return pair_record


def validate_completion_manifest(
    output_dir: str | Path,
    *,
    expected_formal: bool | None = None,
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    path = root / "completion_manifest.json"
    completion = json.loads(path.read_text(encoding="utf-8"))
    if completion.get("complete") is not True:
        raise RuntimeError(f"completion manifest is not complete: {path}")
    if expected_formal is not None and completion.get("formal") != bool(expected_formal):
        raise RuntimeError(f"completion manifest formal identity differs: {path}")
    rows = completion.get("files", [])
    if len(rows) != int(completion.get("file_count", -1)):
        raise RuntimeError(f"completion manifest file count differs: {path}")
    seen: set[str] = set()
    for row in rows:
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise RuntimeError(f"unsafe/duplicate completion path: {relative}")
        seen.add(str(relative))
        artifact = root / relative
        if not artifact.is_file() or sha256_file(artifact) != str(row["sha256"]):
            raise RuntimeError(f"completed artifact hash mismatch: {artifact}")
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    if report.get("formal") is not completion.get("formal"):
        raise RuntimeError("completion/report formal identity differs")
    if report.get("protocol_sha256") != completion.get("protocol_sha256"):
        raise RuntimeError("completion/report protocol identity differs")
    return completion


def canonical_coords(coords: np.ndarray, *, resolution: int) -> np.ndarray:
    value = np.asarray(coords, dtype=np.int32)
    if value.ndim != 2 or value.shape[1] not in (3, 4):
        raise ValueError(f"invalid sparse coords shape={value.shape}")
    xyz = value[:, -3:]
    if xyz.size and (np.any(xyz < 0) or np.any(xyz >= int(resolution))):
        raise ValueError("sparse coords outside canonical resolution")
    xyz = np.unique(xyz, axis=0)
    batch = np.zeros((len(xyz), 1), dtype=np.int32)
    return np.concatenate((batch, xyz), axis=1)


def shared_noise_audit(
    stock_coords: torch.Tensor,
    stock_feats: torch.Tensor,
    correct_coords: torch.Tensor,
    correct_feats: torch.Tensor,
) -> dict[str, Any]:
    if stock_coords.shape[0] != stock_feats.shape[0]:
        raise ValueError("stock initial SLAT coords/features are misaligned")
    if correct_coords.shape[0] != correct_feats.shape[0]:
        raise ValueError("correct initial SLAT coords/features are misaligned")
    stock_xyz = [
        tuple(int(v) for v in row[-3:]) for row in stock_coords.detach().cpu().tolist()
    ]
    correct_xyz = [
        tuple(int(v) for v in row[-3:])
        for row in correct_coords.detach().cpu().tolist()
    ]
    stock = set(stock_xyz)
    correct = set(correct_xyz)
    if len(stock) != len(stock_xyz) or len(correct) != len(correct_xyz):
        raise ValueError("initial SLAT SparseTensor contains duplicate coordinates")
    common = sorted(stock & correct)
    if common:
        stock_position = {coord: index for index, coord in enumerate(stock_xyz)}
        correct_position = {coord: index for index, coord in enumerate(correct_xyz)}
        left = torch.stack([stock_feats[stock_position[coord]] for coord in common])
        right = torch.stack(
            [correct_feats[correct_position[coord]] for coord in common]
        )
        max_abs = float((left - right).abs().max().item())
    else:
        max_abs = 0.0
    return {
        "stock_coord_count": len(stock),
        "correct_coord_count": len(correct),
        "common_coord_count": len(common),
        "common_coord_noise_max_abs": max_abs,
        "common_coord_noise_bit_exact": max_abs == 0.0,
    }


def sparse_noise_from_master(
    coords_np: np.ndarray,
    master: torch.Tensor,
    *,
    device: torch.device,
) -> sp.SparseTensor:
    coords = torch.from_numpy(coords_np).to(device=device, dtype=torch.int32)
    xyz = coords[:, 1:].long()
    feats = master[xyz[:, 0], xyz[:, 1], xyz[:, 2]].clone()
    return sp.SparseTensor(feats=feats, coords=coords)


@torch.no_grad()
def sample_slat_explicit(
    *,
    pipeline,
    slat_condition: dict[str, Any],
    initial_noise: sp.SparseTensor,
    params: dict[str, Any],
) -> sp.SparseTensor:
    flow = pipeline.models["slat_flow_model"]
    # Give each branch an independent SparseTensor object while preserving the
    # audited, coordinate-keyed initial payload exactly.
    noise = sp.SparseTensor(
        feats=initial_noise.feats.clone(),
        coords=initial_noise.coords.clone(),
    )
    sampled = pipeline.slat_sampler.sample(
        flow,
        noise,
        **slat_condition,
        **params,
        verbose=False,
    ).samples
    std = torch.tensor(
        pipeline.slat_normalization["std"], device=sampled.device
    )[None]
    mean = torch.tensor(
        pipeline.slat_normalization["mean"], device=sampled.device
    )[None]
    return sampled * std + mean


def sparse_payload(value: sp.SparseTensor) -> dict[str, torch.Tensor]:
    return {
        "feats": value.feats.detach().cpu(),
        "coords": value.coords.detach().cpu().to(dtype=torch.int32),
    }


def sparse_from_payload(payload: dict[str, torch.Tensor], device: torch.device) -> sp.SparseTensor:
    return sp.SparseTensor(
        feats=payload["feats"].to(device=device),
        coords=payload["coords"].to(device=device, dtype=torch.int32),
    )


def sparse_exact_diff(left: sp.SparseTensor, right: sp.SparseTensor) -> dict[str, Any]:
    coords_equal = torch.equal(left.coords, right.coords)
    feats_equal = torch.equal(left.feats, right.feats)
    max_abs = (
        float((left.feats.float() - right.feats.float()).abs().max().item())
        if left.feats.numel() and left.feats.shape == right.feats.shape
        else math.inf
    )
    return {
        "coords_equal": coords_equal,
        "feats_equal": feats_equal,
        "feats_max_abs": max_abs,
        "bit_exact": coords_equal and feats_equal and max_abs == 0.0,
        "passed": coords_equal and feats_equal and max_abs == 0.0,
    }


def mesh_exact_diff(left, right) -> dict[str, Any]:
    faces_equal = torch.equal(left.faces, right.faces)
    vertex_shape_equal = left.vertices.shape == right.vertices.shape
    max_abs = (
        float((left.vertices.float() - right.vertices.float()).abs().max().item())
        if vertex_shape_equal and left.vertices.numel()
        else math.inf
    )
    return {
        "faces_equal": faces_equal,
        "vertex_shape_equal": vertex_shape_equal,
        "vertices_max_abs": max_abs,
        "bit_exact": faces_equal and vertex_shape_equal and max_abs == 0.0,
        "passed": faces_equal and vertex_shape_equal and max_abs == 0.0,
    }


def vertex_chamfer_metrics(
    left: trimesh.Trimesh, right: trimesh.Trimesh
) -> dict[str, float]:
    left_vertices = np.asarray(left.vertices, dtype=np.float64)
    right_vertices = np.asarray(right.vertices, dtype=np.float64)
    if (
        not len(left_vertices)
        or not len(right_vertices)
        or not np.isfinite(left_vertices).all()
        or not np.isfinite(right_vertices).all()
    ):
        raise ValueError("repeat Mesh vertices are empty or non-finite")
    left_to_right = cKDTree(right_vertices).query(
        left_vertices, k=1, workers=-1
    )[0]
    right_to_left = cKDTree(left_vertices).query(
        right_vertices, k=1, workers=-1
    )[0]
    return {
        "left_to_right_mean": float(np.mean(left_to_right)),
        "right_to_left_mean": float(np.mean(right_to_left)),
        "chamfer_l1": float(
            0.5 * (np.mean(left_to_right) + np.mean(right_to_left))
        ),
        "p99": float(
            max(np.quantile(left_to_right, 0.99), np.quantile(right_to_left, 0.99))
        ),
        "max": float(max(np.max(left_to_right), np.max(right_to_left))),
    }


def deterministic_surface_sample(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise ValueError("cannot sample an empty mesh")
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1.0e-15)
    if not np.any(valid):
        raise ValueError("mesh contains no finite non-degenerate triangles")
    valid_ids = np.flatnonzero(valid)
    probability = double_area[valid] / double_area[valid].sum()
    rng = np.random.default_rng(int(seed))
    face_ids = rng.choice(valid_ids, size=int(count), replace=True, p=probability)
    u = rng.random(int(count))
    v = rng.random(int(count))
    sqrt_u = np.sqrt(u)
    weights = np.stack((1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v), axis=1)
    selected = triangles[face_ids]
    points = np.sum(selected * weights[:, :, None], axis=1)
    normals = cross[face_ids]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-15)
    return points, normals


def surface_metrics(
    predicted: trimesh.Trimesh,
    target: trimesh.Trimesh,
    *,
    count: int,
    seed: int,
    thresholds: Iterable[float],
) -> dict[str, float]:
    pred_points, pred_normals = deterministic_surface_sample(predicted, count, seed)
    # Reusing the same deterministic variates makes the identity case exactly
    # zero while still sampling each mesh according to its own triangle areas.
    target_points, target_normals = deterministic_surface_sample(target, count, seed)
    target_tree = cKDTree(target_points)
    pred_tree = cKDTree(pred_points)
    pred_distance, pred_index = target_tree.query(pred_points, k=1, workers=-1)
    target_distance, target_index = pred_tree.query(target_points, k=1, workers=-1)
    output = {
        "pred_to_gt_mean": float(np.mean(pred_distance)),
        "gt_to_pred_mean": float(np.mean(target_distance)),
        "chamfer_l1": float(0.5 * (np.mean(pred_distance) + np.mean(target_distance))),
        "chamfer_l2": float(
            0.5 * (np.mean(pred_distance**2) + np.mean(target_distance**2))
        ),
        "normal_consistency": float(
            0.5
            * (
                np.mean(np.abs(np.sum(pred_normals * target_normals[pred_index], axis=1)))
                + np.mean(
                    np.abs(np.sum(target_normals * pred_normals[target_index], axis=1))
                )
            )
        ),
    }
    for threshold in thresholds:
        key = str(float(threshold)).replace(".", "p")
        precision = float(np.mean(pred_distance < float(threshold)))
        recall = float(np.mean(target_distance < float(threshold)))
        output[f"precision_{key}"] = precision
        output[f"recall_{key}"] = recall
        output[f"fscore_{key}"] = (
            0.0
            if precision + recall <= 1.0e-12
            else float(2.0 * precision * recall / (precision + recall))
        )
    return output


def mesh_structure_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    finite = bool(np.isfinite(vertices).all())
    result: dict[str, Any] = {
        "mesh_success": bool(len(vertices) > 0 and len(faces) > 0 and finite),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "vertices_finite": finite,
        "is_watertight": bool(mesh.is_watertight) if len(faces) else False,
        "is_winding_consistent": bool(mesh.is_winding_consistent) if len(faces) else False,
    }
    if len(vertices):
        extent = np.ptp(vertices, axis=0)
        result["bbox_extent"] = [float(value) for value in extent]
        result["bbox_diag"] = float(np.linalg.norm(extent))
    if not len(faces):
        result.update(
            {
                "component_count": 0,
                "largest_component_ratio": 0.0,
                "small_component_vertex_ratio_lt100": 0.0,
                "boundary_edge_count": 0,
                "boundary_total_length": 0.0,
                "nonmanifold_edge_count": 0,
            }
        )
        return result

    parent = np.arange(len(vertices), dtype=np.int64)
    size = np.ones(len(vertices), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    def union(left: int, right: int) -> None:
        left_root, right_root = find(int(left)), find(int(right))
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    for face in faces:
        union(face[0], face[1])
        union(face[1], face[2])
        union(face[2], face[0])
    referenced = np.unique(faces.reshape(-1))
    component_sizes = Counter(find(int(index)) for index in referenced)
    sizes = sorted(component_sizes.values(), reverse=True)
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0),
        axis=1,
    )
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[edge_counts == 1]
    boundary_total_length = (
        float(
            np.linalg.norm(
                vertices[boundary_edges[:, 0]] - vertices[boundary_edges[:, 1]],
                axis=1,
            ).sum()
        )
        if len(boundary_edges)
        else 0.0
    )
    result.update(
        {
            "component_count": len(sizes),
            "largest_component_ratio": float(sizes[0] / max(sum(sizes), 1)),
            "small_component_vertex_ratio_lt100": float(
                sum(value for value in sizes if value < 100) / max(sum(sizes), 1)
            ),
            "boundary_edge_count": int(np.sum(edge_counts == 1)),
            "boundary_total_length": boundary_total_length,
            "nonmanifold_edge_count": int(np.sum(edge_counts > 2)),
        }
    )
    return result


def load_canonical_gt(sample: dict[str, Any]) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    latent_path = Path(sample["ss_latent"]).resolve()
    with np.load(latent_path) as payload:
        source_glb = Path(str(payload["source_glb"])).resolve()
        center = np.asarray(payload["normalize_center"], dtype=np.float64)
        scale = float(payload["normalize_scale"])
        margin = float(payload["canonical_margin"])
    loaded = trimesh.load(source_glb, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        pieces = [
            item
            for item in loaded.dump(concatenate=False)
            if isinstance(item, trimesh.Trimesh) and len(item.vertices) and len(item.faces)
        ]
        if not pieces:
            raise ValueError(f"source GLB has no mesh: {source_glb}")
        target = trimesh.util.concatenate(pieces)
    elif isinstance(loaded, trimesh.Trimesh):
        target = loaded.copy()
    else:
        raise TypeError(f"unsupported source mesh type={type(loaded)}")
    target.vertices = (np.asarray(target.vertices) - center[None]) / scale * margin
    return target, {
        "source_glb": str(source_glb),
        "source_glb_sha256": sha256_file(source_glb),
        "latent_path": str(latent_path),
        "latent_sha256": sha256_file(latent_path),
        "normalize_center": center.tolist(),
        "normalize_scale": scale,
        "canonical_margin": margin,
        "frame": "canonical latent frame; no per-branch normalization or ICP",
    }


def save_preview(mesh, output_dir: Path, *, frames: int, resolution: int) -> dict[str, Any]:
    rendered = render_utils.render_video(
        mesh,
        resolution=int(resolution),
        ssaa=1,
        num_frames=int(frames),
    )["normal"]
    arrays = []
    for frame in rendered:
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            maximum = float(np.nanmax(array)) if array.size else 0.0
            array = np.clip(array * (255.0 if maximum <= 1.5 else 1.0), 0, 255).astype(
                np.uint8
            )
        arrays.append(array[..., :3])
    if not arrays:
        raise RuntimeError("renderer returned no frames")
    mp4_path = output_dir / "normal_turntable.mp4"
    mp4_temporary = temporary_path(mp4_path)
    imageio.mimsave(mp4_temporary, arrays, fps=12)
    os.replace(mp4_temporary, mp4_path)
    chosen = np.linspace(0, len(arrays) - 1, min(9, len(arrays)), dtype=int)
    selected = [Image.fromarray(arrays[index]) for index in chosen]
    columns = 3
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * resolution, rows * resolution), (0, 0, 0))
    for index, image in enumerate(selected):
        sheet.paste(image, ((index % columns) * resolution, (index // columns) * resolution))
    sheet_path = output_dir / "normal_contact_sheet.png"
    sheet_temporary = temporary_path(sheet_path)
    sheet.save(sheet_temporary)
    os.replace(sheet_temporary, sheet_path)
    return {
        "turntable": str(mp4_path),
        "turntable_sha256": sha256_file(mp4_path),
        "contact_sheet": str(sheet_path),
        "contact_sheet_sha256": sha256_file(sheet_path),
        "frames": len(arrays),
        "resolution": int(resolution),
    }


def summarize_formal(values: list[float], *, samples: int, seed: int) -> dict[str, Any]:
    result = summarize(values)
    result["bootstrap_mean_95_ci"] = bootstrap_mean_ci(
        values, samples=int(samples), seed=int(seed)
    )
    result["object_win_rate"] = positive_rate(values)
    return result


def aggregate_report(
    records: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    expected_pairs: int,
    formal: bool,
) -> dict[str, Any]:
    by_pair: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    branch_counts: Counter[tuple[str, int, str]] = Counter()
    for row in records:
        pair_key = (str(row["object_uid"]), int(row["seed"]))
        branch = str(row["branch"])
        branch_counts[(pair_key[0], pair_key[1], branch)] += 1
        by_pair[pair_key][branch] = row
    pair_deltas = []
    invalid_pairs = []
    primary_fscore = float(protocol.get("mesh", {}).get("primary_fscore_threshold", 0.02))
    fscore_key = f"fscore_{str(primary_fscore).replace('.', 'p')}"
    for key, branches in sorted(by_pair.items()):
        duplicates = {
            branch: branch_counts[(key[0], key[1], branch)]
            for branch in ("stock", "correct")
            if branch_counts[(key[0], key[1], branch)] != 1
        }
        if duplicates:
            invalid_pairs.append(
                {"key": list(key), "error": "duplicate/missing branch records", "counts": duplicates}
            )
            continue
        if set(branches) != {"stock", "correct"}:
            invalid_pairs.append({"key": list(key), "error": "missing branch"})
            continue
        stock, correct = branches["stock"], branches["correct"]
        if not stock.get("passed") or not correct.get("passed"):
            invalid_pairs.append({"key": list(key), "error": "branch generation failed"})
            continue
        pair_deltas.append(
            {
                "object_uid": key[0],
                "seed": key[1],
                "chamfer_l1_improvement": float(stock["surface"]["chamfer_l1"])
                - float(correct["surface"]["chamfer_l1"]),
                "fscore_0p02_delta": float(correct["surface"][fscore_key])
                - float(stock["surface"][fscore_key]),
                "normal_consistency_delta": float(
                    correct["surface"]["normal_consistency"]
                )
                - float(stock["surface"]["normal_consistency"]),
                "largest_component_ratio_delta": float(
                    correct["structure"]["largest_component_ratio"]
                )
                - float(stock["structure"]["largest_component_ratio"]),
                "mesh_success_delta": float(correct["structure"]["mesh_success"])
                - float(stock["structure"]["mesh_success"]),
            }
        )

    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_deltas:
        by_object[str(row["object_uid"])].append(row)
    metric_names = (
        "chamfer_l1_improvement",
        "fscore_0p02_delta",
        "normal_consistency_delta",
        "largest_component_ratio_delta",
        "mesh_success_delta",
    )
    object_rows = []
    for object_uid, rows in sorted(by_object.items()):
        object_rows.append(
            {
                "object_uid": object_uid,
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in metric_names
                },
            }
        )
    bootstrap_samples = int(protocol["statistics"]["bootstrap_samples"])
    summary = {
        metric: summarize_formal(
            [float(row[metric]) for row in object_rows],
            samples=bootstrap_samples,
            seed=73000 + index,
        )
        for index, metric in enumerate(metric_names)
    }
    seed_summary = {}
    for seed in protocol["sampling"]["joint_seeds"]:
        rows = [row for row in pair_deltas if int(row["seed"]) == int(seed)]
        seed_summary[str(seed)] = {
            metric: summarize([float(row[metric]) for row in rows])
            for metric in metric_names
        }
    checks_config = protocol["statistics"]["checks"]
    chamfer = summary.get("chamfer_l1_improvement", {})
    checks = {
        "expected_record_count": len(records) == 2 * int(expected_pairs),
        "expected_pair_count": len(by_pair) == int(expected_pairs),
        "no_invalid_pairs": not invalid_pairs and len(pair_deltas) == int(expected_pairs),
        "chamfer_mean_positive": float(chamfer.get("mean", 0.0)) > 0.0,
        "chamfer_median_positive": float(chamfer.get("median", 0.0)) > 0.0,
        "chamfer_object_win": float(chamfer.get("object_win_rate", 0.0))
        >= float(checks_config["chamfer_object_win_rate_min"]),
        "chamfer_ci_lower_positive": float(
            chamfer.get("bootstrap_mean_95_ci", [0.0, 0.0])[0]
        )
        > 0.0,
        "fscore_non_degrading": float(summary.get("fscore_0p02_delta", {}).get("mean", -1.0))
        >= float(checks_config["fscore_0p02_mean_delta_min"]),
        "mesh_success_non_degrading": float(
            summary.get("mesh_success_delta", {}).get("mean", -1.0)
        )
        >= float(checks_config["mesh_success_rate_delta_min"]),
        "largest_component_non_degrading": float(
            summary.get("largest_component_ratio_delta", {}).get("mean", -1.0)
        )
        >= float(checks_config["largest_component_ratio_mean_delta_min"]),
    }
    nonnegative_seed_directions = sum(
        float(row["chamfer_l1_improvement"]["mean"]) >= 0.0
        for row in seed_summary.values()
    )
    checks["minimum_nonnegative_seed_directions"] = nonnegative_seed_directions >= int(
        checks_config["minimum_nonnegative_seed_directions"]
    )
    return {
        "formal": bool(formal),
        "passed": bool(formal) and all(checks.values()),
        "checks": checks,
        "expected_pair_count": int(expected_pairs),
        "completed_pair_count": len(by_pair),
        "valid_pair_count": len(pair_deltas),
        "invalid_pairs": invalid_pairs,
        "formal_weighting": "paired seed deltas averaged per object, then object bootstrap",
        "summary": summary,
        "seed_summary": seed_summary,
        "object_rows": object_rows,
        "pair_deltas": pair_deltas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--blind_key_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke_objects",
        type=int,
        default=0,
        help="Run the first N frozen objects as non-formal engineering smoke.",
    )
    parser.add_argument(
        "--smoke_seeds",
        default="",
        help="Optional seed subset; only legal together with --smoke_objects.",
    )
    parser.add_argument("--skip_renders", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = validate_protocol(protocol_path)
    blind_key_path = Path(args.blind_key_file).resolve()
    blind_key = bytes.fromhex(blind_key_path.read_text(encoding="ascii").strip())
    blind_key_commitment = hashlib.sha256(blind_key).hexdigest()
    if blind_key_commitment != str(
        protocol["blinding"]["blind_key_sha256_commitment"]
    ):
        raise RuntimeError("blind key does not match the frozen protocol commitment")
    output_dir = Path(args.output_dir).resolve()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Stage-B Mesh export requires CUDA")
    if str(args.amp_dtype) != str(protocol["runtime"]["amp_dtype"]):
        raise ValueError("runtime amp dtype differs from the frozen protocol")
    torch.cuda.set_device(0 if device.index is None else int(device.index))
    formal = int(args.smoke_objects) <= 0 and not bool(args.smoke_seeds) and not args.skip_renders
    if not formal and int(args.smoke_objects) <= 0:
        raise ValueError("non-formal overrides require --smoke_objects")
    if int(args.smoke_objects) < 0:
        raise ValueError("smoke_objects must be non-negative")

    selected_rows = list(protocol["selection"]["rows"])
    seeds = [int(value) for value in protocol["sampling"]["joint_seeds"]]
    if int(args.smoke_objects) > 0:
        selected_rows = selected_rows[: int(args.smoke_objects)]
        if args.smoke_seeds:
            requested = [
                int(value) for value in args.smoke_seeds.split(",") if value.strip()
            ]
            if not requested or any(value not in seeds for value in requested):
                raise ValueError("smoke seeds must be a non-empty subset of protocol seeds")
            seeds = requested
    expected_pairs = len(selected_rows) * len(seeds)
    if expected_pairs <= 0:
        raise ValueError("empty Mesh evaluation")
    run_identity = {
        "protocol": str(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "formal": formal,
        "selected_uids": [row["uid"] for row in selected_rows],
        "seeds": seeds,
        "amp_dtype": args.amp_dtype,
        "blind_key_sha256_commitment": blind_key_commitment,
        "skip_renders": bool(args.skip_renders),
    }

    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "completion_manifest.json"
    if completion_path.exists():
        existing_identity = json.loads(
            (output_dir / "run_identity.json").read_text(encoding="utf-8")
        )
        if existing_identity != run_identity:
            raise RuntimeError("completed run arguments differ from this invocation")
        completion = validate_completion_manifest(
            output_dir, expected_formal=formal
        )
        print(
            json.dumps(
                {
                    "reused_complete_run": True,
                    "formal": completion["formal"],
                    "science_passed": completion["science_passed"],
                    "runtime_exit_code": completion["runtime_exit_code"],
                },
                indent=2,
            ),
            flush=True,
        )
        raise SystemExit(int(completion["runtime_exit_code"]))
    copied_protocol = output_dir / "protocol.json"
    if copied_protocol.exists():
        if sha256_file(copied_protocol) != sha256_file(protocol_path):
            raise RuntimeError("resume output is bound to another protocol")
    else:
        shutil.copy2(protocol_path, copied_protocol)
    identity_path = output_dir / "run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != run_identity:
            raise RuntimeError("resume arguments differ from the existing run")
    else:
        write_json(identity_path, run_identity)

    bindings = protocol["bindings"]
    rollout_report = json.loads(
        Path(bindings["rollout_report"]["path"]).read_text(encoding="utf-8")
    )
    rollout_indices = Path(bindings["rollout_indices"]["path"]).read_text(
        encoding="utf-8"
    ).strip()
    if not rollout_indices:
        raise RuntimeError("bound Stage-A indices are empty")
    dataset = PoseLiftingCacheDataset(
        bindings["cache_manifest"]["path"], indices=rollout_indices
    )
    if len(dataset) != 32:
        raise RuntimeError(f"expected the full frozen 32-row dataset, got {len(dataset)}")
    for row in selected_rows:
        sample = dataset[int(row["rollout_position"])]
        if str(sample["uid"]) != str(row["uid"]):
            raise RuntimeError("dataset order differs from frozen Mesh protocol")

    checkpoint_path = Path(bindings["flow_checkpoint"]["path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != DIRECT_FLOW_VERSION:
        raise ValueError("unexpected direct Flow checkpoint format")
    saved_args = checkpoint["args"]
    n3_audit = validate_n3_checkpoint(
        bindings["n3_report"]["path"], bindings["correspondence_checkpoint"]["path"]
    )
    correspondence_head, _, correspondence_runtime = load_frozen_correspondence_head(
        bindings["correspondence_checkpoint"]["path"],
        device=device,
        visual_channels=dataset.visual_feature_dim,
    )
    if correspondence_runtime["checkpoint_sha256"] != n3_audit["checkpoint_sha256"]:
        raise RuntimeError("runtime correspondence checkpoint differs from N3")
    built = build_direct_components(
        pretrained=protocol["pretrained"],
        visual_channels=dataset.visual_feature_dim,
        physical_hidden_dim=int(saved_args["physical_hidden_dim"]),
        lora_rank=int(saved_args["lora_rank"]),
        lora_alpha=int(saved_args["lora_alpha"]),
        gradient_checkpointing=False,
        need_decoder=True,
        device=device,
        retain_pipeline=True,
    )
    sampler, model, decoder, model_summary, sampler_defaults, pipeline = built
    load_partial_state(model, checkpoint["model_trainable_state"], require_all_trainable=True)
    model.eval()
    use_amp = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    physical_scale = float(saved_args.get("physical_scale", 1.0))
    ss_params = dict(sampler_defaults)
    ss_params.update(protocol["sampling"]["ss"])
    ss_params.pop("seed_formula", None)

    rollout_by_key = {
        (str(row["uid"]), int(row["seed"]), str(row["branch"])): row
        for row in rollout_report["rollout"]["records"]
    }
    coord_root = output_dir / "ss_coords"
    coord_root.mkdir(exist_ok=True)
    rollout_tolerance = float(protocol["audits"]["rollout_metric_max_abs_diff"])
    for frozen in selected_rows:
        rollout_position = int(frozen["rollout_position"])
        sample = dataset[rollout_position]
        evidence = make_direct_evidence_bundle(
            sample,
            modes=(),
            device=device,
            correspondence_head=correspondence_head,
            correspondence_runtime=correspondence_runtime,
        )
        condition = sample["stock_condition"].to(device=device)
        negative_condition = torch.zeros_like(condition)
        for seed in seeds:
            pair_id, _ = pair_identity(
                protocol["protocol_name"], str(sample["uid"]), seed, blind_key
            )
            coord_path = coord_root / f"{pair_id}.npz"
            audit_path = coord_root / f"{pair_id}.json"
            if coord_path.is_file() and audit_path.is_file():
                validate_ss_artifact(
                    coord_path,
                    audit_path,
                    pair_id=pair_id,
                    uid=str(sample["uid"]),
                    object_uid=str(sample.get("object_uid", sample["uid"])),
                    seed=seed,
                    rollout_position=rollout_position,
                )
                continue
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 1000003 + rollout_position * 1009
            )
            noise = torch.randn(
                (1, 8, 16, 16, 16), generator=generator, device=device
            )
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_amp
            ):
                stock_latent = sampler.sample(
                    NativeStockFlow(model),
                    noise.clone(),
                    cond=condition,
                    neg_cond=negative_condition,
                    **ss_params,
                    verbose=False,
                ).samples
                wrapper = PositivePhysicalRolloutFlow(
                    model,
                    condition,
                    evidence["correct"][:4],
                    physical_scale=physical_scale,
                )
                correct_latent = sampler.sample(
                    wrapper,
                    noise.clone(),
                    cond=condition,
                    neg_cond=negative_condition,
                    **ss_params,
                    verbose=False,
                ).samples
            if wrapper.positive_calls <= 0:
                raise RuntimeError("correct SS rollout never used the physical path")
            coords = {
                "stock": decode_coords(decoder, stock_latent),
                "correct": decode_coords(decoder, correct_latent),
            }
            target_coords = sample["target_coords"].numpy().astype(np.int32)
            branch_audits = {}
            for branch in ("stock", "correct"):
                metrics = {
                    **overlap_metrics(coords[branch], target_coords),
                    **component_metrics(coords[branch]),
                }
                expected = rollout_by_key[(str(sample["uid"]), int(seed), branch)]
                diffs = {
                    name: abs(float(metrics[name]) - float(expected[name]))
                    for name in (
                        "iou",
                        "precision",
                        "recall",
                        "coord_count_ratio",
                        "component_count",
                        "largest_component_ratio",
                    )
                }
                if max(diffs.values(), default=0.0) > rollout_tolerance:
                    raise RuntimeError(
                        f"SS replay differs from Stage-A report: uid={sample['uid']} "
                        f"seed={seed} branch={branch} diffs={diffs}"
                    )
                branch_audits[branch] = {"metrics": metrics, "stage_a_abs_diff": diffs}
            savez_atomic(
                coord_path,
                stock=coords["stock"].astype(np.int32),
                correct=coords["correct"].astype(np.int32),
            )
            write_json(
                audit_path,
                {
                    "pair_id": pair_id,
                    "uid": str(sample["uid"]),
                    "object_uid": str(sample.get("object_uid", sample["uid"])),
                    "seed": int(seed),
                    "rollout_position": rollout_position,
                    "ss_noise_sha256": tensor_sha256(noise),
                    "stock_correct_shared_ss_noise": True,
                    "correct_cfg_calls": {
                        "positive": wrapper.positive_calls,
                        "negative": wrapper.negative_calls,
                    },
                    "coords_npz_sha256": sha256_file(coord_path),
                    "branches": branch_audits,
                    "passed": True,
                },
            )
        print(f"[mesh_B:ss] {rollout_position + 1}/32 {sample['uid']}", flush=True)
        torch.cuda.empty_cache()

    # The direct SS stage is complete.  Move it off GPU before native VGGT/SLAT.
    model.cpu()
    decoder.cpu()
    correspondence_head.cpu()
    for name in ("sparse_structure_flow_model", "sparse_structure_decoder"):
        pipeline.models[name].cpu()
    del model, decoder, correspondence_head
    gc.collect()
    torch.cuda.empty_cache()

    pipeline._device = device
    pipeline.low_vram = True
    condition_root = output_dir / "slat_conditions"
    condition_root.mkdir(exist_ok=True)
    condition_limit = float(
        protocol["audits"]["cached_vs_recomputed_stock_condition_max_abs"]
    )
    for frozen in selected_rows:
        rollout_position = int(frozen["rollout_position"])
        sample = dataset[rollout_position]
        condition_path = condition_root / f"{sample['uid']}.pt"
        audit_path = condition_root / f"{sample['uid']}.json"
        if condition_path.is_file() and audit_path.is_file():
            validate_condition_artifact(
                condition_path, audit_path, uid=str(sample["uid"])
            )
            continue
        images = rgba_images(sample["image_paths"], sample["mask_paths"], pipeline)
        aggregated, image_tensor = pipeline.vggt_feat(images)
        raw_image_cond = pipeline.encode_image(image_tensor)
        batch = int(aggregated[0].shape[0])
        views = int(aggregated[0].shape[1])
        image_cond = normalize_image_cond(raw_image_cond, batch=batch, views=views)
        recomputed_ss = pipeline.get_ss_cond(
            image_cond[:, :, 5:], aggregated, num_samples=1
        )["cond"]
        cached_ss = sample["stock_condition"].to(device=device)
        condition_diff = float(
            (recomputed_ss.float() - cached_ss.float()).abs().max().item()
        )
        if condition_diff > condition_limit:
            raise RuntimeError(
                f"native condition replay differs from cache for uid={sample['uid']}: "
                f"{condition_diff} > {condition_limit}"
            )
        slat_condition = pipeline.get_slat_cond(image_cond, aggregated, num_samples=1)
        cpu_condition = to_cpu_tree(slat_condition)
        torch_save_atomic(cpu_condition, condition_path)
        write_json(
            audit_path,
            {
                "uid": str(sample["uid"]),
                "views": views,
                "cached_vs_recomputed_stock_condition_max_abs": condition_diff,
                "limit": condition_limit,
                "slat_condition_sha256": tensor_tree_sha256(cpu_condition),
                "condition_pt_sha256": sha256_file(condition_path),
                "passed": condition_diff <= condition_limit,
            },
        )
        del images, aggregated, image_tensor, raw_image_cond, image_cond
        del recomputed_ss, cached_ss, slat_condition, cpu_condition
        torch.cuda.empty_cache()
        print(f"[mesh_B:cond] {sample['uid']}", flush=True)

    # Freeze all image-conditioning modules on CPU before SLAT sampling.
    for name in ("image_cond_model", "sparse_structure_vggt_cond", "slat_vggt_cond"):
        pipeline.models[name].cpu()
    pipeline.VGGT_model.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    slat_root = output_dir / "slat"
    slat_root.mkdir(exist_ok=True)
    slat_flow = pipeline.models["slat_flow_model"].to(device).eval()
    slat_resolution = int(getattr(slat_flow, "resolution", 64))
    slat_channels = int(slat_flow.in_channels)
    if slat_resolution != 64:
        raise RuntimeError(f"unexpected SLAT resolution={slat_resolution}")
    slat_params = dict(pipeline.slat_sampler_params)
    slat_params.update(protocol["sampling"]["slat"])
    slat_params.pop("noise_field", None)
    slat_params.pop("seed_formula", None)
    determinism_slat_audit_path = output_dir / "determinism_slat_audit.json"
    determinism_repeat_slat_path = output_dir / "determinism_repeat_slat.pt"
    determinism_slat_complete = (
        determinism_slat_audit_path.is_file()
        and determinism_repeat_slat_path.is_file()
    )
    if determinism_slat_complete:
        validate_determinism_slat_artifact(
            determinism_slat_audit_path, determinism_repeat_slat_path
        )
    for frozen in selected_rows:
        rollout_position = int(frozen["rollout_position"])
        sample = dataset[rollout_position]
        condition_cpu = torch.load(
            condition_root / f"{sample['uid']}.pt", map_location="cpu"
        )
        slat_condition = to_device_tree(condition_cpu, device)
        for seed in seeds:
            pair_id, slat_side_mapping = pair_identity(
                protocol["protocol_name"], str(sample["uid"]), seed, blind_key
            )
            pair_slat_dir = slat_root / pair_id
            pair_slat_dir.mkdir(exist_ok=True)
            audit_path = pair_slat_dir / "noise_audit.json"
            branch_paths = {
                branch: pair_slat_dir / f"{branch}.pt" for branch in ("stock", "correct")
            }
            if audit_path.is_file() and all(path.is_file() for path in branch_paths.values()):
                validate_slat_artifact(
                    audit_path,
                    branch_paths,
                    pair_id=pair_id,
                    uid=str(sample["uid"]),
                    seed=seed,
                )
                if determinism_slat_complete:
                    continue
            with np.load(coord_root / f"{pair_id}.npz") as payload:
                coords = {
                    branch: canonical_coords(payload[branch], resolution=slat_resolution)
                    for branch in ("stock", "correct")
                }
            generator = torch.Generator(device=device).manual_seed(
                int(seed) * 2000003 + rollout_position * 2017 + 7919
            )
            master = torch.randn(
                (slat_resolution, slat_resolution, slat_resolution, slat_channels),
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
            if noise_audit["common_coord_noise_max_abs"] != float(
                protocol["audits"]["shared_slat_noise_common_coord_max_abs"]
            ):
                raise RuntimeError("shared coordinate-keyed SLAT noise audit failed")
            sampled = {}
            slat_branch_order = [
                slat_side_mapping[side] for side in ("A", "B")
            ]
            for branch in slat_branch_order:
                sampled[branch] = sample_slat_explicit(
                    pipeline=pipeline,
                    slat_condition=slat_condition,
                    initial_noise=initial_noise[branch],
                    params=slat_params,
                )
                torch_save_atomic(
                    sparse_payload(sampled[branch]), branch_paths[branch]
                )
            deterministic_slat = None
            if not determinism_slat_complete:
                repeated = sample_slat_explicit(
                    pipeline=pipeline,
                    slat_condition=slat_condition,
                    initial_noise=initial_noise["stock"],
                    params=slat_params,
                )
                deterministic_slat = sparse_exact_diff(sampled["stock"], repeated)
                slat_repeat_limit = float(
                    protocol["audits"]["deterministic_repeat_slat_max_abs"]
                )
                deterministic_slat["limit"] = slat_repeat_limit
                deterministic_slat["passed"] = bool(
                    deterministic_slat["coords_equal"]
                    and deterministic_slat["feats_max_abs"] <= slat_repeat_limit
                )
                torch_save_atomic(
                    sparse_payload(repeated), determinism_repeat_slat_path
                )
                write_json(
                    determinism_slat_audit_path,
                    {
                        "pair_id": pair_id,
                        "uid": str(sample["uid"]),
                        "seed": int(seed),
                        "repeat_payload_sha256": sha256_file(
                            determinism_repeat_slat_path
                        ),
                        **deterministic_slat,
                    },
                )
                if not deterministic_slat["passed"]:
                    raise RuntimeError(
                        "SLAT deterministic repeat audit failed: "
                        f"max_abs={deterministic_slat['feats_max_abs']} "
                        f"limit={slat_repeat_limit}"
                    )
                determinism_slat_complete = True
            write_json(
                audit_path,
                {
                    "pair_id": pair_id,
                    "uid": str(sample["uid"]),
                    "seed": int(seed),
                    "master_noise_seed": int(seed) * 2000003
                    + rollout_position * 2017
                    + 7919,
                    "master_noise_sha256": tensor_sha256(master),
                    "master_noise_shape": list(master.shape),
                    "condition_sha256": tensor_tree_sha256(condition_cpu),
                    "initial_noise_sha256": {
                        branch: tensor_tree_sha256(sparse_payload(initial_noise[branch]))
                        for branch in ("stock", "correct")
                    },
                    "branch_payload_sha256": {
                        branch: sha256_file(branch_paths[branch])
                        for branch in ("stock", "correct")
                    },
                    "private_branch_execution_order": slat_branch_order,
                    **noise_audit,
                    "passed": noise_audit["common_coord_noise_bit_exact"],
                },
            )
            del master, sampled, initial_noise
            torch.cuda.empty_cache()
            print(f"[mesh_B:slat] {pair_id}", flush=True)
        del condition_cpu, slat_condition
    slat_flow.cpu()
    del slat_flow
    gc.collect()
    torch.cuda.empty_cache()

    mesh_decoder = pipeline.models["slat_decoder_mesh"].to(device).eval()
    mesh_root = output_dir / "blind_pairs"
    mesh_root.mkdir(exist_ok=True)
    private_root = output_dir / "private_records"
    private_root.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    blind_manifest = []
    unblinding: dict[str, Any] = {}
    deterministic_mesh_audit_path = output_dir / "determinism_mesh_audit.json"
    deterministic_mesh_complete = deterministic_mesh_audit_path.is_file()
    if deterministic_mesh_complete:
        validate_determinism_mesh_artifact(deterministic_mesh_audit_path)
    for frozen in selected_rows:
        rollout_position = int(frozen["rollout_position"])
        sample = dataset[rollout_position]
        target_mesh, target_metadata = load_canonical_gt(sample)
        for seed in seeds:
            pair_id, side_mapping = pair_identity(
                protocol["protocol_name"], str(sample["uid"]), seed, blind_key
            )
            pair_dir = mesh_root / pair_id
            pair_dir.mkdir(exist_ok=True)
            private_pair_dir = private_root / pair_id
            private_pair_dir.mkdir(exist_ok=True)
            pair_record_path = private_pair_dir / "pair_record.json"
            if pair_record_path.is_file():
                pair_record = validate_pair_record(
                    pair_record_path,
                    pair_id=pair_id,
                    uid=str(sample["uid"]),
                    object_uid=str(sample.get("object_uid", sample["uid"])),
                    seed=seed,
                )
                if deterministic_mesh_complete:
                    records.extend(pair_record["branches"])
                    blind_manifest.append(pair_record["blind_manifest"])
                    unblinding[pair_id] = pair_record["unblinding"]
                    continue
            branch_records = []
            # Decode/export in public side order, not private branch order, so
            # file timestamps cannot reveal which side is stock.
            for side in ("A", "B"):
                branch = side_mapping[side]
                side_dir = pair_dir / side
                side_dir.mkdir(exist_ok=True)
                row: dict[str, Any] = {
                    "pair_id": pair_id,
                    "side": side,
                    "branch": branch,
                    "uid": str(sample["uid"]),
                    "object_uid": str(sample.get("object_uid", sample["uid"])),
                    "views": int(frozen["views"]),
                    "seed": int(seed),
                    "passed": False,
                }
                try:
                    slat_payload = torch.load(
                        slat_root / pair_id / f"{branch}.pt", map_location="cpu"
                    )
                    slat = sparse_from_payload(slat_payload, device)
                    mesh = mesh_decoder(slat)[0]
                    if branch == "stock" and not deterministic_mesh_complete:
                        repeat_payload = torch.load(
                            determinism_repeat_slat_path, map_location="cpu"
                        )
                        repeat_mesh = mesh_decoder(
                            sparse_from_payload(repeat_payload, device)
                        )[0]
                        deterministic_mesh = mesh_exact_diff(mesh, repeat_mesh)
                        first_repeat_mesh = mesh.to_trimesh(transform_pose=False)
                        second_repeat_mesh = repeat_mesh.to_trimesh(
                            transform_pose=False
                        )
                        first_repeat_structure = mesh_structure_metrics(
                            first_repeat_mesh
                        )
                        second_repeat_structure = mesh_structure_metrics(
                            second_repeat_mesh
                        )
                        repeat_vertex_chamfer = vertex_chamfer_metrics(
                            first_repeat_mesh, second_repeat_mesh
                        )
                        mesh_repeat_limit = float(
                            protocol["audits"][
                                "deterministic_repeat_mesh_vertex_chamfer_l1_max"
                            ]
                        )
                        deterministic_mesh.update(
                            {
                                "vertex_chamfer": repeat_vertex_chamfer,
                                "first_structure": first_repeat_structure,
                                "second_structure": second_repeat_structure,
                                "vertex_chamfer_l1_limit": mesh_repeat_limit,
                            }
                        )
                        deterministic_mesh["passed"] = bool(
                            first_repeat_structure["mesh_success"]
                            and second_repeat_structure["mesh_success"]
                            and repeat_vertex_chamfer["chamfer_l1"]
                            <= mesh_repeat_limit
                        )
                        write_json(
                            deterministic_mesh_audit_path,
                            {"pair_id": pair_id, **deterministic_mesh},
                        )
                        if not deterministic_mesh["passed"]:
                            raise RuntimeError(
                                "Mesh deterministic repeat audit failed: "
                                "vertex_chamfer_l1="
                                f"{repeat_vertex_chamfer['chamfer_l1']} "
                                f"limit={mesh_repeat_limit}"
                            )
                        deterministic_mesh_complete = True
                        del repeat_payload, repeat_mesh
                        del first_repeat_mesh, second_repeat_mesh
                    canonical_mesh = mesh.to_trimesh(transform_pose=False)
                    view_mesh = mesh.to_trimesh(transform_pose=True)
                    structure = mesh_structure_metrics(canonical_mesh)
                    if not structure["mesh_success"]:
                        raise RuntimeError("decoded mesh is empty or non-finite")
                    obj_path = side_dir / "mesh_canonical.obj"
                    glb_path = side_dir / "mesh_view.glb"
                    obj_temporary = temporary_path(obj_path)
                    glb_temporary = temporary_path(glb_path)
                    canonical_mesh.export(obj_temporary)
                    view_mesh.export(glb_temporary)
                    os.replace(obj_temporary, obj_path)
                    os.replace(glb_temporary, glb_path)
                    reopened_obj = trimesh.load(obj_path, force="mesh", process=False)
                    reopened_glb = trimesh.load(glb_path, force="scene", process=False)
                    if not len(reopened_obj.vertices) or not len(reopened_obj.faces):
                        raise RuntimeError("canonical OBJ roundtrip is empty")
                    glb_geometry = (
                        list(reopened_glb.geometry.values())
                        if isinstance(reopened_glb, trimesh.Scene)
                        else [reopened_glb]
                    )
                    if not any(len(item.vertices) and len(item.faces) for item in glb_geometry):
                        raise RuntimeError("view GLB roundtrip is empty")
                    metric_seed = int(seed) * 1009 + rollout_position * 9173
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
                            mesh,
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
                    del slat_payload, slat, mesh, canonical_mesh, view_mesh
                except Exception as error:  # Preserve frozen failures; do not exclude.
                    row["error"] = repr(error)
                # Never place branch labels or metrics below blind_pairs/.
                # That directory is the sanitized rater bundle.
                write_json(private_pair_dir / f"{branch}.json", row)
                branch_records.append(row)
                torch.cuda.empty_cache()
            blind_row = {
                "pair_id": pair_id,
                "uid": str(sample["uid"]),
                "object_uid": str(sample.get("object_uid", sample["uid"])),
                "views": int(frozen["views"]),
                "seed": int(seed),
                "A": str(pair_dir / "A" / "mesh_view.glb"),
                "B": str(pair_dir / "B" / "mesh_view.glb"),
            }
            key_row = {"A": side_mapping["A"], "B": side_mapping["B"]}
            pair_record = {
                "pair_id": pair_id,
                "uid": str(sample["uid"]),
                "object_uid": str(sample.get("object_uid", sample["uid"])),
                "seed": int(seed),
                "passed": all(row.get("passed") is True for row in branch_records),
                "branches": branch_records,
                "blind_manifest": blind_row,
                "unblinding": key_row,
            }
            write_json(pair_record_path, pair_record)
            pair_record = validate_pair_record(
                pair_record_path,
                pair_id=pair_id,
                uid=str(sample["uid"]),
                object_uid=str(sample.get("object_uid", sample["uid"])),
                seed=seed,
            )
            records.extend(branch_records)
            blind_manifest.append(blind_row)
            unblinding[pair_id] = key_row
            print(f"[mesh_B:decode] {pair_id}", flush=True)
        del target_mesh
    mesh_decoder.cpu()
    del mesh_decoder, pipeline
    gc.collect()
    torch.cuda.empty_cache()

    blind_manifest_path = output_dir / "blind_manifest.csv"
    blind_manifest_temporary = temporary_path(blind_manifest_path)
    with blind_manifest_temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blind_manifest[0].keys()))
        writer.writeheader()
        writer.writerows(blind_manifest)
    os.replace(blind_manifest_temporary, blind_manifest_path)
    blind_leaks = [
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
        "allowed_companion_manifest": str(output_dir / "blind_manifest.csv"),
        "private_records": str(private_root),
        "leaks": blind_leaks,
        "passed": not blind_leaks,
    }
    if blind_leaks:
        raise RuntimeError(f"blind rater bundle leaks private labels: {blind_leaks}")
    unblinding_path = output_dir / "unblinding_key.json"
    write_json(unblinding_path, unblinding)
    unblinding_path.chmod(0o600)
    decision = aggregate_report(
        records,
        protocol=protocol,
        expected_pairs=expected_pairs,
        formal=formal,
    )
    report = {
        "format": REPORT_FORMAT,
        "stage": "B frozen same-noise native ReconViaGen Mesh comparison",
        "protocol_sha256": protocol["protocol_sha256"],
        "formal": formal,
        "decision": decision,
        "args": vars(args),
        "model_summary": model_summary,
        "n3_audit": n3_audit,
        "record_count": len(records),
        "blinding_audit": blinding_audit,
        "records": records,
        "claim_limit": (
            "This compares stock/base with the trained configuration. It does not establish "
            "pose/depth-specific causality because the Stage-A corruption controls matched correct."
        ),
    }
    write_json(output_dir / "report.json", report)
    lines = [
        "# Stage-B paired Mesh evaluation",
        "",
        f"- Formal: `{formal}`",
        f"- Decision: `{'PASS' if decision['passed'] else 'FAIL'}`",
        f"- Protocol SHA-256: `{protocol['protocol_sha256']}`",
        f"- Completed / expected pairs: `{decision['completed_pair_count']} / {expected_pairs}`",
        f"- Valid pairs: `{decision['valid_pair_count']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in decision["checks"].items())
    lines.extend(
        [
            "",
            "## Primary summary",
            "",
            "```json",
            json.dumps(decision["summary"], indent=2),
            "```",
            "",
            "Keep `unblinding_key.json` hidden until blind ratings are frozen.",
        ]
    )
    write_text(output_dir / "report.md", "\n".join(lines) + "\n")

    artifact_files = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "completion_manifest.json"
    )
    completion = {
        "complete": True,
        "protocol_sha256": protocol["protocol_sha256"],
        "formal": formal,
        "science_passed": decision["passed"],
        "all_records_passed": all(row.get("passed") is True for row in records),
        "runtime_exit_code": (
            2
            if formal and not decision["passed"]
            else 3
            if not all(row.get("passed") is True for row in records)
            else 0
        ),
        "file_count": len(artifact_files),
        "files": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
            }
            for path in artifact_files
        ],
    }
    write_json(output_dir / "completion_manifest.json", completion)
    validate_completion_manifest(output_dir, expected_formal=formal)
    print(json.dumps({
        "formal": formal,
        "passed": decision["passed"],
        "checks": decision["checks"],
        "report": str(output_dir / "report.json"),
    }, indent=2), flush=True)
    raise SystemExit(int(completion["runtime_exit_code"]))


if __name__ == "__main__":
    main()
