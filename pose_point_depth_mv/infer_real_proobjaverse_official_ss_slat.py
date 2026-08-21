#!/usr/bin/env python3
"""Qualitative runtime-O inference with official Native-SS and trained SLat.

The trained SLat checkpoint is constructed and validated against the Native-SS
identity frozen at training time.  A different, official-domain Native-SS may
provide the runtime support only when a completed held-out A/B/C bridge report
cryptographically binds both deployments.  This is an explicit cross-
deployment qualitative test; it never rewrites either checkpoint identity.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


os.environ.setdefault("SPCONV_ALGO", "native")

from pose_point_depth_mv import infer_omni_real_native_v2 as _base_infer  # noqa: E402
from pose_point_depth_mv.dataset_tools.prepare_omni_real_dino_only_model_inputs import (  # noqa: E402
    MANIFEST_FORMAT as MODEL_INPUT_MANIFEST_FORMAT,
    OBJECT_FORMAT as MODEL_INPUT_OBJECT_FORMAT,
)
from pose_point_depth_mv.evaluate_proobjaverse_official_native_ss_stock_slat import (  # noqa: E402
    END_TO_END_REPORT_FORMAT,
    END_TO_END_WORKER_FORMAT,
    MAX_SAFE_SLAT_DECODER_INPUT_POINTS,
)
from pose_point_depth_mv.export_direct_flow_mesh_pairs import (  # noqa: E402
    canonical_coords,
    sparse_noise_from_master,
)
from pose_point_depth_mv.mesh_benchmark_metrics import mesh_structure_metrics  # noqa: E402
from pose_point_depth_mv.native_slat_genrecon_no_vggt import (  # noqa: E402
    NATIVE_SLAT_NO_VGGT_VERSION,
    NO_VGGT_SLAT_CONTRACT,
    build_native_slat_no_vggt_components,
    validate_native_slat_no_vggt_checkpoint,
)
from pose_point_depth_mv.native_slat_genrecon_v2 import (  # noqa: E402
    NativeSLatCalibratedCFGFlow,
    load_stock_slat_freeze,
    load_trainable_state_dict as load_slat_state,
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
from pose_point_depth_mv.proobjaverse_official_ss import (  # noqa: E402
    load_official_native_ss_deployment,
)
from pose_point_depth_mv.trellis_mesh_coordinate_contract import (  # noqa: E402
    decoded_mesh_to_sparse_grid_frame,
    mesh_frame_contract_fields,
    validate_runtime_o_mesh_frame_contract,
)


REPORT_FORMAT = (
    "pose_point_depth_mv.real_proobjaverse_official_ss_slat_inference.v1"
)
MANIFEST_FORMAT = (
    "pose_point_depth_mv.real_proobjaverse_official_ss_slat_inference_manifest.v1"
)
ABC_R_REPORT_FORMAT = (
    "pose_point_depth_mv.proobjaverse_official_ss_slat_vs_reconviagen.v1"
)


def _validate_abc_r_cross_deployment_bridge(
    bridge_path: Path,
    payload: dict[str, Any],
    *,
    native_ss_report: Path,
    native_ss_binding: dict[str, Any],
    trained_slat_checkpoint: Path,
    trained_slat_step: int,
    trained_slat_weights: str,
    stock_slat_freeze: Path,
) -> dict[str, Any]:
    """Validate the frozen held-out A/B/C/R report as deployment evidence.

    The SLat checkpoint intentionally retains its training-time upstream SS
    identity.  This report proves that the requested, different runtime SS was
    paired with the exact SLat checkpoint on a disjoint held-out split.
    """

    embedded = _validate_embedded_report_hash(payload, label="A/B/C/R aggregate")
    if (
        payload.get("passed") is not True
        or payload.get("runtime_integrity_passed") is not True
        or payload.get("native_ss_science_passed") is not True
    ):
        raise RuntimeError("A/B/C/R cross-deployment evidence did not pass")

    expected_ss = dict(native_ss_binding)
    observed_ss = dict(payload.get("native_ss_binding") or {})
    if observed_ss != expected_ss:
        raise RuntimeError("A/B/C/R Native-SS deployment binding differs")

    expected_slat = {
        "checkpoint": str(trained_slat_checkpoint),
        "checkpoint_sha256": sha256_file(trained_slat_checkpoint),
        "checkpoint_step": int(trained_slat_step),
        "weights": str(trained_slat_weights),
    }
    observed_slat = dict(payload.get("native_slat_binding") or {})
    slat_mismatch = {
        name: (observed_slat.get(name), value)
        for name, value in expected_slat.items()
        if observed_slat.get(name) != value
    }
    if slat_mismatch:
        raise RuntimeError(f"A/B/C/R Native-SLat binding differs: {slat_mismatch}")

    expected_artifacts = {
        "native_ss_report": (native_ss_report, expected_ss["report_sha256"]),
        "native_ss_checkpoint": (
            Path(expected_ss["checkpoint"]),
            expected_ss["checkpoint_sha256"],
        ),
        "native_slat_checkpoint": (
            trained_slat_checkpoint,
            expected_slat["checkpoint_sha256"],
        ),
        "stock_slat_freeze": (stock_slat_freeze, sha256_file(stock_slat_freeze)),
    }
    artifact_checks = dict(payload.get("artifact_checks") or {})
    for name, (path, digest) in expected_artifacts.items():
        row = dict(artifact_checks.get(name) or {})
        if (
            row.get("passed") is not True
            or row.get("path") != str(path)
            or row.get("sha256") != str(digest)
            or sha256_file(path) != str(digest)
        ):
            raise RuntimeError(f"A/B/C/R artifact binding differs: {name}")

    references = list(payload.get("current_worker_reports") or [])
    if not references:
        raise RuntimeError("A/B/C/R report has no current endpoint workers")
    observed_uids: list[str] = []
    worker_summaries: list[dict[str, Any]] = []
    for reference in references:
        worker_path = Path(str(reference["path"])).expanduser().resolve(strict=True)
        if sha256_file(worker_path) != str(reference["sha256"]):
            raise RuntimeError(f"A/B/C/R worker file hash differs: {worker_path}")
        worker = load_json(worker_path)
        if (
            worker.get("format") != END_TO_END_WORKER_FORMAT
            or worker.get("complete") is not True
            or worker.get("passed") is not True
        ):
            raise RuntimeError(f"A/B/C/R worker is incomplete: {worker_path}")
        _validate_embedded_report_hash(worker, label=f"A/B/C/R worker {worker_path}")
        identity = dict(worker.get("run_identity") or {})
        worker_expected = {
            "native_ss_report_sha256": expected_ss["report_sha256"],
            "trained_slat_checkpoint_sha256": expected_slat["checkpoint_sha256"],
            "trained_slat_weights": expected_slat["weights"],
            "expected_trained_slat_step": expected_slat["checkpoint_step"],
            "stock_slat_freeze_sha256": sha256_file(stock_slat_freeze),
        }
        mismatches = {
            name: (identity.get(name), value)
            for name, value in worker_expected.items()
            if identity.get(name) != value
        }
        if mismatches:
            raise RuntimeError(f"A/B/C/R worker bindings differ: {mismatches}")
        if dict(worker.get("native_ss_binding") or {}) != expected_ss:
            raise RuntimeError("A/B/C/R worker Native-SS binding differs")
        worker_uids = [str(value) for value in identity.get("object_uids") or []]
        if not worker_uids:
            raise RuntimeError(f"A/B/C/R worker has no object identities: {worker_path}")
        observed_uids.extend(worker_uids)
        worker_summaries.append(
            {
                "path": str(worker_path),
                "sha256": str(reference["sha256"]),
                "object_start": int(identity["object_start"]),
                "object_end": int(identity["object_end"]),
                "object_count": len(worker_uids),
            }
        )

    common_uids = [str(value) for value in payload.get("common_complete_object_uids") or []]
    if (
        not common_uids
        or len(observed_uids) != len(set(observed_uids))
        or set(observed_uids) != set(common_uids)
        or int(payload.get("common_complete_object_count", -1)) != len(common_uids)
    ):
        raise RuntimeError("A/B/C/R worker object coverage differs")
    membership = dict(payload.get("checkpoint_evaluation_membership") or {})
    if (
        membership.get("passed") is not True
        or int(membership.get("training_overlap_count", -1)) != 0
        or membership.get("all_evaluation_objects_disjoint_from_checkpoint_training")
        is not True
    ):
        raise RuntimeError("A/B/C/R held-out membership contract differs")

    route_name = (
        f"posed_dino_official_native_ss_step{expected_ss['checkpoint_step']}"
        f"_native_slat_step{expected_slat['checkpoint_step']}"
    )
    route = dict((payload.get("route_runtime") or {}).get(route_name) or {})
    expected_records = len(common_uids) * 3
    if (
        int(route.get("record_count", -1)) != expected_records
        or int(route.get("successful_record_count", -1)) != expected_records
        or int(route.get("failed_record_count", -1)) != 0
        or float(route.get("mesh_success_rate", -1.0)) != 1.0
    ):
        raise RuntimeError("A/B/C/R route-C runtime coverage differs")

    return {
        "passed": True,
        "purpose": "frozen held-out A/B/C/R cross-deployment artifact binding",
        "path": str(bridge_path),
        "sha256": sha256_file(bridge_path),
        "report_sha256": embedded,
        "formal": bool(payload.get("formal")),
        "runtime_integrity_passed": True,
        "native_ss_science_passed": True,
        "route": route_name,
        "object_count": len(common_uids),
        "record_count": expected_records,
        "worker_count": len(worker_summaries),
        "workers": worker_summaries,
        "scope_guard": (
            "Exact SS30K/SLat30K deployment evidence on frozen held-out Dev; "
            "real-capture outputs remain qualitative and non-formal."
        ),
    }


def _validate_embedded_report_hash(payload: dict[str, Any], *, label: str) -> str:
    body = dict(payload)
    saved = str(body.pop("report_sha256", ""))
    if not saved or canonical_sha256(body) != saved:
        raise RuntimeError(f"{label} embedded report hash differs")
    return saved


def validate_cross_deployment_bridge(
    bridge_report: str | Path,
    *,
    native_ss_report: str | Path,
    native_ss_binding: dict[str, Any],
    trained_slat_checkpoint: str | Path,
    trained_slat_step: int,
    trained_slat_weights: str,
    stock_slat_freeze: str | Path,
) -> dict[str, Any]:
    """Validate completed held-out workers that exercised this SS/SLat pair.

    Scientific gates are deliberately reported rather than required: the real
    reconstruction is qualitative and the existing Dev48 bridge can contain a
    recorded model-output failure.  What is required here is complete,
    hash-valid worker evidence with exact artifact bindings.
    """

    bridge_path = Path(bridge_report).expanduser().resolve(strict=True)
    ss_report_path = Path(native_ss_report).expanduser().resolve(strict=True)
    slat_path = Path(trained_slat_checkpoint).expanduser().resolve(strict=True)
    stock_path = Path(stock_slat_freeze).expanduser().resolve(strict=True)
    payload = load_json(bridge_path)
    if payload.get("format") == ABC_R_REPORT_FORMAT:
        return _validate_abc_r_cross_deployment_bridge(
            bridge_path,
            payload,
            native_ss_report=ss_report_path,
            native_ss_binding=native_ss_binding,
            trained_slat_checkpoint=slat_path,
            trained_slat_step=int(trained_slat_step),
            trained_slat_weights=str(trained_slat_weights),
            stock_slat_freeze=stock_path,
        )
    if payload.get("format") != END_TO_END_REPORT_FORMAT:
        raise RuntimeError("cross-deployment bridge has the wrong aggregate format")
    aggregate_report_sha256 = _validate_embedded_report_hash(
        payload, label="cross-deployment aggregate"
    )
    references = list(payload.get("worker_reports") or [])
    if not references:
        raise RuntimeError("cross-deployment bridge has no worker reports")

    expected = {
        "native_ss_report_sha256": sha256_file(ss_report_path),
        "trained_slat_checkpoint_sha256": sha256_file(slat_path),
        "trained_slat_weights": str(trained_slat_weights),
        "expected_trained_slat_step": int(trained_slat_step),
        "stock_slat_freeze_sha256": sha256_file(stock_path),
    }
    if expected["native_ss_report_sha256"] != str(
        native_ss_binding["report_sha256"]
    ):
        raise RuntimeError("official Native-SS report content hash differs")
    if sha256_file(native_ss_binding["checkpoint"]) != str(
        native_ss_binding["checkpoint_sha256"]
    ):
        raise RuntimeError("official Native-SS checkpoint content hash differs")

    semantic_binding_fields = (
        "report_sha256",
        "checkpoint_sha256",
        "checkpoint_step",
        "weights",
        "cfg_strength",
        "steps",
        "cfg_interval",
        "guidance_rescale",
        "rescale_t",
        "amp_dtype",
        "false_checks",
    )
    observed_uids: list[str] = []
    worker_summaries: list[dict[str, Any]] = []
    for reference in references:
        worker_path = Path(str(reference["path"])).expanduser().resolve(strict=True)
        if sha256_file(worker_path) != str(reference["sha256"]):
            raise RuntimeError(f"cross-deployment worker file hash differs: {worker_path}")
        worker = load_json(worker_path)
        if (
            worker.get("format") != END_TO_END_WORKER_FORMAT
            or worker.get("complete") is not True
        ):
            raise RuntimeError(f"cross-deployment worker is incomplete: {worker_path}")
        embedded = _validate_embedded_report_hash(
            worker, label=f"cross-deployment worker {worker_path}"
        )
        if embedded != str(reference["report_sha256"]):
            raise RuntimeError(f"cross-deployment worker embedded hash differs: {worker_path}")
        identity = dict(worker.get("run_identity") or {})
        mismatches = {
            name: (identity.get(name), value)
            for name, value in expected.items()
            if identity.get(name) != value
        }
        if mismatches:
            raise RuntimeError(
                f"cross-deployment worker artifact bindings differ: {mismatches}"
            )
        binding = dict(worker.get("native_ss_binding") or {})
        semantic_mismatches = {
            name: (binding.get(name), native_ss_binding.get(name))
            for name in semantic_binding_fields
            if binding.get(name) != native_ss_binding.get(name)
        }
        if semantic_mismatches:
            raise RuntimeError(
                "cross-deployment worker Native-SS semantics differ: "
                f"{semantic_mismatches}"
            )
        worker_uids = [str(value) for value in identity.get("object_uids") or []]
        if not worker_uids:
            raise RuntimeError(f"cross-deployment worker has no object identities: {worker_path}")
        observed_uids.extend(worker_uids)
        worker_summaries.append(
            {
                "path": str(worker_path),
                "sha256": str(reference["sha256"]),
                "report_sha256": embedded,
                "object_start": int(identity["object_start"]),
                "object_end": int(identity["object_end"]),
                "object_count": len(worker_uids),
                "runtime_passed": bool(worker.get("passed")),
            }
        )

    expected_objects = int(payload.get("object_count", -1))
    if len(observed_uids) != expected_objects or len(set(observed_uids)) != expected_objects:
        raise RuntimeError("cross-deployment workers do not exactly cover aggregate objects")
    integrity = dict(payload.get("integrity") or {})
    if integrity.get("object_coverage_exact") is not True:
        raise RuntimeError("cross-deployment aggregate object coverage is not exact")
    return {
        "passed": True,
        "purpose": "artifact-binding evidence for qualitative cross-deployment inference",
        "path": str(bridge_path),
        "sha256": sha256_file(bridge_path),
        "report_sha256": aggregate_report_sha256,
        "aggregate_runtime_passed": bool(payload.get("passed")),
        "aggregate_science_decision": payload.get("decision"),
        "object_count": expected_objects,
        "record_count": int(payload.get("record_count", -1)),
        "worker_count": len(worker_summaries),
        "workers": worker_summaries,
        "scope_guard": (
            "The bridge proves exact SS/SLat artifact pairing and completed held-out "
            "execution. Its scientific gates are recorded, not promoted to a pass."
        ),
    }


def _configure_no_vggt_base() -> None:
    _base_infer.MODEL_INPUT_MANIFEST_FORMAT = MODEL_INPUT_MANIFEST_FORMAT
    _base_infer.MODEL_INPUT_OBJECT_FORMAT = MODEL_INPUT_OBJECT_FORMAT
    _base_infer.validate_native_ss_genrecon_checkpoint = (
        validate_native_ss_no_vggt_checkpoint
    )
    _base_infer.build_native_ss_genrecon_components = (
        build_native_ss_no_vggt_components
    )


def _run_official_ss(
    *,
    rows: list[dict[str, Any]],
    seeds: list[int],
    output_dir: Path,
    binding: dict[str, Any],
    pretrained: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> None:
    _configure_no_vggt_base()
    checkpoint_path = Path(binding["checkpoint"]).expanduser().resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != NATIVE_SS_NO_VGGT_VERSION:
        raise RuntimeError("official deployment is not a trained no-VGGT Native-SS")
    validate_native_ss_no_vggt_checkpoint(
        checkpoint, pretrained=pretrained, allow_v2_parent=False
    )
    del checkpoint
    _base_infer._run_ss(
        rows=rows,
        seeds=seeds,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=str(binding["checkpoint_sha256"]),
        pretrained=pretrained,
        weights=str(binding["weights"]),
        device=device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
        upstream_binding=binding,
    )
    for row in rows:
        for seed in seeds:
            _, audit_path = _base_infer._coord_paths(output_dir, row, seed)
            audit = load_json(audit_path)
            audit.update(
                {
                    "official_native_ss_report": str(binding["report"]),
                    "official_native_ss_report_sha256": str(binding["report_sha256"]),
                    "input_context_contract": NO_VGGT_MODEL_CONTRACT,
                    "vggt_model_executed": False,
                }
            )
            atomic_json(audit_path, audit)


def _validate_sampling(params: dict[str, Any]) -> None:
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
            raise RuntimeError(f"frozen trained-SLat sampler changed: {name}={actual}")


def _reuse_mesh_result(
    result_path: Path,
    mesh_path: Path,
    *,
    row: dict[str, Any],
    seed: int,
    native_ss_binding: dict[str, Any],
    slat_sha256: str,
    slat_weights: str,
    stock_sha256: str,
    sampling_sha256: str,
    bridge_sha256: str,
) -> dict[str, Any] | None:
    if not result_path.is_file() or not mesh_path.is_file():
        return None
    result = load_json(result_path)
    expected = {
        "format": REPORT_FORMAT,
        "object_key": object_key(row),
        "seed": int(seed),
        "model_input_sha256": row["model_input_sha256"],
        "native_ss_report_sha256": native_ss_binding["report_sha256"],
        "native_ss_checkpoint_sha256": native_ss_binding["checkpoint_sha256"],
        "native_ss_weights": native_ss_binding["weights"],
        "native_slat_checkpoint_sha256": slat_sha256,
        "native_slat_weights": slat_weights,
        "stock_slat_freeze_sha256": stock_sha256,
        "sampling_sha256": sampling_sha256,
        "cross_deployment_bridge_sha256": bridge_sha256,
        "mesh_sha256": sha256_file(mesh_path),
    }
    mismatch = {
        name: (result.get(name), value)
        for name, value in expected.items()
        if result.get(name) != value
    }
    if mismatch:
        raise RuntimeError(f"stale official real inference result={mismatch}")
    validate_runtime_o_mesh_frame_contract(result)
    return result


@torch.no_grad()
def _run_trained_slat(
    *,
    rows: list[dict[str, Any]],
    seeds: list[int],
    output_dir: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    expected_step: int,
    stock_freeze_path: Path,
    native_ss_binding: dict[str, Any],
    bridge: dict[str, Any],
    pretrained: str,
    weights: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != NATIVE_SLAT_NO_VGGT_VERSION:
        raise RuntimeError("real inference requires a trained no-VGGT Native-SLat")
    if int(checkpoint.get("step", -1)) != int(expected_step):
        raise RuntimeError(
            f"trained SLat checkpoint step differs: {checkpoint.get('step')} != "
            f"{expected_step}"
        )
    stock_freeze = load_stock_slat_freeze(stock_freeze_path)
    stock_sha256 = sha256_file(stock_freeze_path)
    training_upstream = dict(
        checkpoint.get("model_summary", {}).get("upstream_native_ss") or {}
    )
    if not training_upstream:
        raise RuntimeError("trained SLat checkpoint lacks its frozen Native-SS identity")
    if checkpoint.get("data_identity", {}).get("native_ss") != training_upstream:
        raise RuntimeError("trained SLat checkpoint Native-SS identities are inconsistent")
    validate_native_slat_no_vggt_checkpoint(
        checkpoint,
        pretrained=pretrained,
        stock_slat_freeze=stock_freeze,
        upstream_native_ss=training_upstream,
        allow_v2_parent=False,
    )
    saved = dict(checkpoint["args"])
    sampler, model, decoder, model_summary, defaults, normalization = (
        build_native_slat_no_vggt_components(
            pretrained=pretrained,
            stock_slat_freeze=stock_freeze,
            upstream_native_ss=training_upstream,
            lora_rank=int(saved["lora_rank"]),
            lora_alpha=int(saved["lora_alpha"]),
            condition_channels=int(saved["condition_channels"]),
            gradient_checkpointing=False,
            need_decoder=True,
            device=device,
        )
    )
    if decoder is None:
        raise RuntimeError("trained SLat real inference requires the Stock Mesh decoder")
    state_key = "ema_trainable_state" if weights == "ema" else "model_trainable_state"
    load_slat_state(model, checkpoint[state_key])
    model.eval()
    decoder.eval()
    params = dict(defaults)
    params.setdefault("guidance_rescale", 0.0)
    _validate_sampling(params)
    sampling_sha256 = canonical_sha256(params)
    mean = torch.tensor(normalization["mean"], device=device)[None]
    std = torch.tensor(normalization["std"], device=device)[None]
    reports: list[dict[str, Any]] = []

    for position, row in enumerate(rows):
        sample = _base_infer._load_model_sample(row)
        condition = to_device_tree(sample["slat_condition"], device)
        for seed in seeds:
            mesh_path, result_path = _base_infer._mesh_paths(output_dir, row, seed)
            reused = _reuse_mesh_result(
                result_path,
                mesh_path,
                row=row,
                seed=seed,
                native_ss_binding=native_ss_binding,
                slat_sha256=checkpoint_sha256,
                slat_weights=weights,
                stock_sha256=stock_sha256,
                sampling_sha256=sampling_sha256,
                bridge_sha256=str(bridge["sha256"]),
            )
            if reused is not None:
                reports.append(reused)
                continue
            if mesh_path.parent.exists():
                raise RuntimeError(f"partial official real mesh output: {mesh_path.parent}")
            coord_path, coord_report_path = _base_infer._coord_paths(
                output_dir, row, seed
            )
            coord_report = load_json(coord_report_path)
            coord_expected = {
                "object_key": object_key(row),
                "seed": int(seed),
                "coords_sha256": sha256_file(coord_path),
                "checkpoint_sha256": native_ss_binding["checkpoint_sha256"],
                "weights": native_ss_binding["weights"],
                "model_input_sha256": row["model_input_sha256"],
                "official_native_ss_report_sha256": native_ss_binding["report_sha256"],
            }
            coord_mismatch = {
                name: (coord_report.get(name), value)
                for name, value in coord_expected.items()
                if coord_report.get(name) != value
            }
            if coord_mismatch:
                raise RuntimeError(
                    f"official Native-SS coordinate binding changed: {coord_mismatch}"
                )
            with np.load(coord_path, allow_pickle=False) as payload:
                coords_np = canonical_coords(payload["coords"], resolution=64)
            if len(coords_np) > MAX_SAFE_SLAT_DECODER_INPUT_POINTS:
                raise RuntimeError(
                    "SLat decoder input exceeds safe active-point limit: "
                    f"points={len(coords_np)} limit={MAX_SAFE_SLAT_DECODER_INPUT_POINTS}"
                )
            master_seed = int(seed) * 2000003 + int(position) * 2017 + 7919
            generator = torch.Generator(device=device).manual_seed(master_seed)
            master = torch.randn((64, 64, 64, 8), generator=generator, device=device)
            initial = sparse_noise_from_master(coords_np, master, device=device)
            flow = NativeSLatCalibratedCFGFlow(
                model,
                condition["cond"],
                sample,
                enabled=True,
                projection_mode="correct",
            )
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                latent = sampler.sample(
                    flow, initial, **condition, **params, verbose=False
                ).samples
            active_points = int(latent.feats.shape[0])
            if active_points > MAX_SAFE_SLAT_DECODER_INPUT_POINTS:
                raise RuntimeError(
                    "SLat decoder input exceeds safe active-point limit: "
                    f"points={active_points} limit={MAX_SAFE_SLAT_DECODER_INPUT_POINTS}"
                )
            decoded = decoder(latent * std + mean)[0]
            if flow.positive_calls <= 0 or flow.negative_calls <= 0:
                raise RuntimeError("trained Native-SLat CFG missed a branch")
            mesh = decoded_mesh_to_sparse_grid_frame(decoded)
            structure = mesh_structure_metrics(mesh)
            if not structure["mesh_success"]:
                raise RuntimeError(f"trained SLat decoded an empty Mesh: {object_key(row)}")
            mesh_path.parent.mkdir(parents=True, exist_ok=False)
            temporary = mesh_path.with_name(f".{mesh_path.name}.tmp-{os.getpid()}")
            mesh.export(temporary, file_type="obj")
            os.replace(temporary, mesh_path)
            result = {
                "format": REPORT_FORMAT,
                "created_at_utc": _base_infer.utc_now(),
                "passed": True,
                "method": "proobjaverse_official_native_ss_trained_slat",
                "object_key": object_key(row),
                "category": row["category"],
                "object_id": row["object_id"],
                "seed": int(seed),
                "mesh": str(mesh_path),
                "mesh_sha256": sha256_file(mesh_path),
                "result": str(result_path),
                "structure": structure,
                "coord_count": int(len(coords_np)),
                "model_input": row["model_input"],
                "model_input_sha256": row["model_input_sha256"],
                "native_ss_report": str(native_ss_binding["report"]),
                "native_ss_report_sha256": str(native_ss_binding["report_sha256"]),
                "native_ss_checkpoint": str(native_ss_binding["checkpoint"]),
                "native_ss_checkpoint_sha256": str(
                    native_ss_binding["checkpoint_sha256"]
                ),
                "native_ss_weights": str(native_ss_binding["weights"]),
                "native_ss_sampling": {
                    name: native_ss_binding[name]
                    for name in (
                        "steps",
                        "cfg_strength",
                        "cfg_interval",
                        "guidance_rescale",
                        "rescale_t",
                    )
                },
                "native_slat_checkpoint": str(checkpoint_path),
                "native_slat_checkpoint_sha256": checkpoint_sha256,
                "native_slat_checkpoint_step": int(expected_step),
                "native_slat_weights": weights,
                "native_slat_training_upstream": training_upstream,
                "runtime_native_ss_differs_from_training_upstream": bool(
                    str(native_ss_binding["checkpoint_sha256"])
                    != str(training_upstream.get("checkpoint_sha256"))
                ),
                "cross_deployment_bridge": bridge,
                "cross_deployment_bridge_sha256": str(bridge["sha256"]),
                "stock_slat_freeze": str(stock_freeze_path),
                "stock_slat_freeze_sha256": stock_sha256,
                "sampling": params,
                "sampling_sha256": sampling_sha256,
                "wrapper": flow.summary(),
                "ss_input_context_contract": NO_VGGT_MODEL_CONTRACT,
                "slat_input_context_contract": NO_VGGT_SLAT_CONTRACT,
                "master_noise_seed": master_seed,
                "output_frame": "runtime-O",
                **mesh_frame_contract_fields(
                    export_policy="decoded.to_trimesh(transform_pose=False)"
                ),
                "post_cfg_cap": False,
                "target_or_metric_consumed": False,
                "vggt_model_loaded": False,
                "vggt_model_executed": False,
                "formal_claim_allowed": False,
                "passed": True,
            }
            atomic_json(result_path, result)
            reports.append(result)
            print(
                f"[real_official_slat:mesh] {position + 1}/{len(rows)} "
                f"object={object_key(row)} seed={seed} step={expected_step}",
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
    return reports, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(expected_step),
        "weights": weights,
        "training_upstream_native_ss": training_upstream,
        "model_summary": model_summary,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_input_manifest", required=True)
    parser.add_argument("--native_ss_report", required=True)
    parser.add_argument("--native_slat_checkpoint", required=True)
    parser.add_argument("--expected_slat_step", required=True, type=int)
    parser.add_argument("--cross_deployment_bridge_report", required=True)
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
    _configure_no_vggt_base()
    model_manifest_path = Path(args.model_input_manifest).expanduser().resolve(strict=True)
    model_manifest = load_json(model_manifest_path)
    if (
        model_manifest.get("format") != MODEL_INPUT_MANIFEST_FORMAT
        or model_manifest.get("passed") is not True
    ):
        raise RuntimeError(f"DINO-only model input manifest did not pass: {model_manifest_path}")
    rows = select_rows(model_manifest.get("objects", []), args.object)
    seeds = _base_infer.parse_csv_int(args.seeds)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ss_report_path = Path(args.native_ss_report).expanduser().resolve(strict=True)
    ss_payload, ss_binding = load_official_native_ss_deployment(ss_report_path)
    if ss_payload.get("passed") is not True:
        raise RuntimeError("official Native-SS deployment report did not pass")
    slat_path = Path(args.native_slat_checkpoint).expanduser().resolve(strict=True)
    stock_path = Path(args.stock_slat_freeze).expanduser().resolve(strict=True)
    slat_sha256 = sha256_file(slat_path)
    bridge = validate_cross_deployment_bridge(
        args.cross_deployment_bridge_report,
        native_ss_report=ss_report_path,
        native_ss_binding=ss_binding,
        trained_slat_checkpoint=slat_path,
        trained_slat_step=int(args.expected_slat_step),
        trained_slat_weights=str(args.weights),
        stock_slat_freeze=stock_path,
    )
    if str(args.amp_dtype) != str(ss_binding["amp_dtype"]):
        raise RuntimeError("runtime AMP dtype differs from official Native-SS deployment")
    device = resolve_torch_device(args.device)
    amp_enabled = args.amp_dtype != "none"
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    _run_official_ss(
        rows=rows,
        seeds=seeds,
        output_dir=output_dir,
        binding=ss_binding,
        pretrained=args.pretrained,
        device=device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    reports, slat_binding = _run_trained_slat(
        rows=rows,
        seeds=seeds,
        output_dir=output_dir,
        checkpoint_path=slat_path,
        checkpoint_sha256=slat_sha256,
        expected_step=int(args.expected_slat_step),
        stock_freeze_path=stock_path,
        native_ss_binding=ss_binding,
        bridge=bridge,
        pretrained=args.pretrained,
        weights=str(args.weights),
        device=device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    manifest = {
        "format": MANIFEST_FORMAT,
        "created_at_utc": _base_infer.utc_now(),
        "method": "proobjaverse_official_native_ss_trained_slat",
        "model_input_manifest": str(model_manifest_path),
        "model_input_manifest_sha256": sha256_file(model_manifest_path),
        "runtime_input_manifest": model_manifest["runtime_input_manifest"],
        "runtime_input_manifest_sha256": model_manifest[
            "runtime_input_manifest_sha256"
        ],
        "native_ss_deployment": ss_binding,
        "native_slat_deployment": slat_binding,
        "cross_deployment_bridge": bridge,
        "stock_slat_freeze": str(stock_path),
        "stock_slat_freeze_sha256": sha256_file(stock_path),
        "seeds": seeds,
        "object_count": len(rows),
        "record_count": len(reports),
        "objects": reports,
        "output_frame": "runtime-O",
        "target_or_metric_consumed": False,
        "vggt_model_loaded": False,
        "vggt_model_executed": False,
        "formal_claim_allowed": False,
        "scope_guard": (
            "Qualitative real-capture inference. The runtime-O input and output "
            "use no target Mesh; cross-deployment scientific gates are not upgraded."
        ),
        "passed": len(reports) == len(rows) * len(seeds),
    }
    manifest_path = output_dir / "inference_manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "object_count": len(rows),
                "record_count": len(reports),
                "slat_step": int(args.expected_slat_step),
                "manifest": str(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not manifest["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
